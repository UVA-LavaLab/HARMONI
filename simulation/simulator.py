"""
Main simulation orchestrator.
"""
from modeling.core.model_weight import ModelWeightInfo
from modeling.core.dram_info import dram
from modeling.core.memory_system import memsys
from config.model_config import make_model_config
from utils.cache import load_model_dfg, save_model_dfg
from modeling.report.stats_dumper import StatsDumper
from simulation.helpers import (
    check_capacity, setup_simulation_context, run_core_analysis, build_and_map_dfg
)
from utils.logging_util import logger
from modeling.core.tensor import HarmoniTensor
from misc.timing import StageTimer, dump_timing_csv

def start_simulation(args):
    """
    Main simulation entry point. Orchestrates the entire simulation workflow.
    """

    if args.timing:
        logger.info("[TIMING] Timing enabled - initializing StageTimer")
        timer = StageTimer(enabled=True)
        meta = {
                "model": args.model_name,
                "dtype": str(args.dtype),
                "dram": args.dram,
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "batch": args.batch,
            }
        # Save the context manager so we can exit it later
        timing_context = timer.stage("start_simulation")
        timing_context.__enter__()
    else:
        timer = None
        meta = {}
        timing_context = None
    
    try:
        HarmoniTensor.set_global_dram(dram)
        
        # Initialize model configuration
        model_full = make_model_config(args.model_name, args.dtype)
        model_weight = ModelWeightInfo(model_full)
        
        # Setup simulation context (memory system, logging, etc.)
        filename_str = setup_simulation_context(args, dram)
        
        # Check capacity before proceeding
        check_capacity(model_full, model_weight, dram, args.batch)
        
        # Determine optimization flags
        layer_optimization = 'layer' in args.optimization
        fusion_enabled = 1 if (args.fused_attn and args.fused_qkv) else 0
        
        # Try to load from cache
        use_cache = not getattr(args, 'no_cache', False)
       
        model_dfg = None
        task_mappings = None
        temp_tensor_wt = None
        temp_tensor_kv = None
        model_kv = None
        
        if use_cache:
            model_dfg = load_model_dfg(
                args.model_name, args.dram, args.dtype, args.input_tokens,
                args.output_tokens, args.batch, layer_optimization, fusion_enabled, args.buffer
            )
            if model_dfg is not None:
                logger.info(f"[CACHE] Loaded model_dfg from cache (layer_opt={layer_optimization}, fusion_enabled={fusion_enabled}). Skipping graph construction and mapping.")
                # Run core analysis (common to both paths)
                run_core_analysis(model_dfg, dram, memsys, filename_str, args, None)
                
                # Visualization
                if args.visualize:
                    model_dfg.visualize_flowchart_pydot(filename="outputs/Transformer_modified_dfg.png")
                
                # Dump statistics using StatsDumper (task_mappings is None for cached runs)
                stats_dumper = StatsDumper(args, filename_str, model_dfg, task_mappings=None)
                stats_dumper.dump_all()
                
                # Cached path complete - return to skip non-cached path
                return
        
        # Cache miss or cache disabled - build and map DFG
        if model_dfg is None:
            logger.info(f"[CACHE] No cache found or cache disabled. Building model_dfg.")
            seq_len = args.input_tokens + args.output_tokens
            model_dfg, task_mappings, temp_tensor_wt, temp_tensor_kv, model_kv, model_weight = build_and_map_dfg(
                args, model_full, model_weight, dram, seq_len, args.batch, layer_optimization, fusion_enabled, timer
            )

            # Dump pre-mapping statistics
            if 'html_dfg' in getattr(args, 'dump_stats', []):
                stats_dumper_pre = StatsDumper(args, filename_str, model_dfg, task_mappings=None)
                stats_dumper_pre.dump_html_dfg(suffix="_og")
            
            # Save to cache before simulation/analysis
            save_model_dfg(model_dfg, args.model_name, args.dram, args.dtype, args.input_tokens,
                  args.output_tokens, args.batch, layer_optimization, fusion_enabled, args.buffer)
        
        # Run core analysis (common to both paths)
        run_core_analysis(model_dfg, dram, memsys, filename_str, args, timer)
        
        # Visualization
        if args.visualize:
            model_dfg.visualize_flowchart_pydot(filename="outputs/Transformer_modified_dfg.png")
        
        # Dump statistics using StatsDumper
        stats_dumper = StatsDumper(args, filename_str, model_dfg, task_mappings=task_mappings)
        stats_dumper.dump_all()
    
    finally:
        # Dump timing CSV after simulation completes
        if args.timing and timer and timing_context:
            timing_context.__exit__(None, None, None)
            logger.info("[TIMING] Dumping timing CSV")
            print("\n--- Timing Breakdown (ms) ---")
            summary = timer.summary_ms()
            if summary:
                for stage, time_ms in summary.items():
                    print(f"  {stage}: {time_ms:.2f} ms")
            
            # Generate timing CSV filename following the same pattern as other outputs
            import datetime
            t = datetime.datetime.now()
            s = t.strftime("%d%b")
            timing_filename = f"outputs/sim_time_{args.model_name}_{args.dram}_{s}_{args.input_tokens}i_{args.output_tokens}o_B{args.batch}.csv"
            
            dump_timing_csv(
                timer,
                timing_filename,
                meta=meta
            )
            logger.info(f"[TIMING] Timing CSV written to {timing_filename}")
