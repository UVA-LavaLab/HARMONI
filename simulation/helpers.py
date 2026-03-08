"""
Helper functions for simulation workflow.
"""
import sys
import datetime
from contextlib import contextmanager
from utils.logging_util import logger
from misc.type import *
from modeling.core.tensor import HarmoniTensor
from modeling.core.memory_system import memsys
from modeling.core.model_alloc import update_model_weight, update_model_kv
from modeling.core.transformer_dfg import build_model_dfg, get_parallel_execution_levels, print_level_map
from modeling.perf.transformer_sim import get_node_metrics
from modeling.trace.gen_network_trace import generate_network_trace
from modeling.trace.gen_gemm_trace import generate_gemm_trace
from modeling.analysis.timeline_analysis import get_critical_path_analysis
from modeling.core.task_mapping import map_tasks_to_logic_units, add_mapping_to_temp_tensors


@contextmanager
def _noop_context():
    """No-op context manager for when timer is None."""
    yield


def check_capacity(model_full, model_weight, dram, batch):
    """
    Check if DRAM has enough capacity for model weights and KV cache.
    
    Args:
        model_full: Full model configuration dictionary
        model_weight: ModelWeightInfo object
        dram: DRAM configuration object
        batch: Batch size
    
    Raises:
        SystemExit: If capacity is insufficient
    """
    total_weight = model_weight.get_total_weight_size()[3]  # GB
    bytes_per_element = get_bytes_per_element(model_full["dtype"], HarmoniTensorType.KV)
    total_KV = (model_full["nlayers"] * model_full["kv_heads"] * model_full["head_dim"] * 
                model_full["context_len"] * batch * bytes_per_element * 2) / (1024 * 1024 * 1024)  # 2 for K and V
    
    # DRAM capacity
    total_wt_ranks_capacity = dram.calculate_dram_capacity() * dram.num_wt_ranks_per_channel / dram.num_ranks_per_channel
    total_kv_ranks_capacity = dram.calculate_dram_capacity() * dram.num_kv_ranks_per_channel / dram.num_ranks_per_channel
    
    if total_wt_ranks_capacity < total_weight:
        logger.error("Number of weight ranks in the DRAM configuration are not enough for the model")
        raise SystemExit(1)
    if total_kv_ranks_capacity < total_KV:
        logger.error("Number of KV ranks in the DRAM configuration are not enough for the inference requests")
        raise SystemExit(1)


def setup_simulation_context(args, dram):
    """
    Set up the simulation context including memory system and logging.
    
    Args:
        args: Command line arguments
        dram: DRAM configuration object
    
    Returns:
        filename_str: Generated filename string for outputs
    """
    # Generate filename string
    t = datetime.datetime.now()
    s = t.strftime("%d%b")
    filename_str = f"{args.model_name}_{args.dram}_{s}_{args.input_tokens}i_{args.output_tokens}o_B{args.batch}"
    
    # Print memory system info if verbose 
    # CLEANUP Add print_routing_table, and topology dump to StatsDumper.
    if args.verbose:
        print("Memory system: ", memsys)
        logic_units = memsys.get_all_logic_units()
        print("Logic units: ")
        for lu in logic_units:
            print(lu)
        memsys.print_routing_table()
        memsys.build_topology()
        sys.exit(0)
    
    if args.visualize:
        memsys.visualize_networkx_pydot()
    
    return filename_str


def run_core_analysis(model_dfg, dram, memsys, filename_str, args=None, timer=None):
    """
    Run core analysis steps that are common to both cached and non-cached paths.
    
    Args:
        model_dfg: The model dataflow graph
        dram: DRAM configuration object
        memsys: Memory system object
        filename_str: Base filename string for outputs
        args: Command line arguments (optional, for controlling trace dumping)
        timer: StageTimer object (optional, for timing measurements)
    """
    ctx = timer.stage("get_node_metrics") if timer else _noop_context()
    with ctx:
        get_node_metrics(model_dfg, dram, filename_str)
    logger.info("Done getting stats for nodes")
    
    ctx = timer.stage("Misc: Network Trace") if timer else _noop_context()
    with ctx:
        generate_network_trace(model_dfg, memsys, filename_str)
    logger.info("Done calculating communication costs")
    
    # Conditionally dump traces based on --dump_traces flag
    if args is not None:
        dump_traces = getattr(args, 'dump_traces', [])
        
        if 'gemm_trace' in dump_traces:
            ctx = timer.stage("Misc: GEMM Trace") if timer else _noop_context()
            with ctx:
                generate_gemm_trace(model_dfg, filename_str)
            logger.info("Done generating GEMM trace")
    
    ctx = timer.stage("Critical Path Analysis") if timer else _noop_context()
    with ctx:
        get_critical_path_analysis(model_dfg.graph, filename_str, top_k=10, args=args)


def build_and_map_dfg(args, model_full, model_weight, dram, seq_len, batch, layer_optimization, fusion_enabled, timer=None):
    """
    Build the model DFG and perform task mapping.
    
    Args:
        args: Command line arguments
        model_full: Full model configuration dictionary
        model_weight: ModelWeightInfo object
        dram: DRAM configuration object
        seq_len: Sequence length
        batch: Batch size
        layer_optimization: Whether layer optimization is enabled
        fusion_enabled: Whether fusion is enabled
        timer: StageTimer object (optional, for timing measurements)
    
    Returns:
        tuple: (model_dfg, task_mappings, temp_tensor_wt, temp_tensor_kv, model_kv, model_weight)
    """
    ctx = timer.stage("Build and Map DFG") if timer else _noop_context()
    with ctx:
        return _build_and_map_dfg_impl(args, model_full, model_weight, dram, seq_len, batch, layer_optimization, fusion_enabled, timer)


def _build_and_map_dfg_impl(args, model_full, model_weight, dram, seq_len, batch, layer_optimization, fusion_enabled, timer=None):
    """
    Internal implementation of build_and_map_dfg.
    """
    # Update model weights and KV cache
    ctx = timer.stage("nest:Model Weight Alloc") if timer else _noop_context()
    with ctx:
        model_weight, temp_tensor_wt = update_model_weight(model_full, model_weight, dram, args.input_tokens, args.batch)
    model_weight.weights = model_weight.weights_new  
    
    ctx = timer.stage("nest:KV Alloc") if timer else _noop_context()
    with ctx:
        model_kv, temp_tensor_kv = update_model_kv(args, model_full, dram, seq_len, batch)

    # Build DFG
    ctx = timer.stage("nest: Build DFG") if timer else _noop_context()
    with ctx:
        model_dfg = build_model_dfg(
            model_full, args.input_tokens, args.output_tokens, args.batch,
            weights=model_weight.weights, kv_cache=model_kv, dram=dram,
            temp_tensor_wt=temp_tensor_wt, temp_tensor_kv=temp_tensor_kv, args=args
        )
    logger.info("Done building task graph")
    
    # Print initial DFG info if verbose
    if args.verbose:
        print(f"Initial model_dfg:")
        levels = get_parallel_execution_levels(model_dfg)
        print_level_map(levels)
    
    if args.visualize:
        ctx = timer.stage("Misc: DFG Visualization") if timer else _noop_context()
        with ctx:
            model_dfg.visualize_flowchart_pydot(filename="outputs/Transformer_dfg.png")
    
    # Tensor mapping and analysis
    tensor_data = HarmoniTensor.get_other_tensor_sizes()
    if args.verbose:
        for tensor_name, data in tensor_data.items():
            sizes = data['sizes']
            count = data['count']
            if len(sizes) == 1:
                print(f"Tensor: {tensor_name}, Size: {sizes[0]} bytes, Count: {count}")
            else:
                print(f"Tensor: {tensor_name}, Multiple sizes: {sizes}, Total count: {count}")
    
    # Map tasks to logic units
    ctx = timer.stage("nest: Task Mapping") if timer else _noop_context()
    with ctx:
        task_mappings = map_tasks_to_logic_units(model_dfg, memsys, dram, args)
        logger.info("Done mapping tasks")
        
        # Add mappings for temporary tensors
        add_mapping_to_temp_tensors(model_dfg, temp_tensor_wt, temp_tensor_kv, dram, args.buffer)
        logger.info("Done mapping temporary tensors")
    
    if args.verbose:
        print(f"Modified model_dfg before getting stats")
        levels = get_parallel_execution_levels(model_dfg)
        print_level_map(levels)
    
    return model_dfg, task_mappings, temp_tensor_wt, temp_tensor_kv, model_kv, model_weight
