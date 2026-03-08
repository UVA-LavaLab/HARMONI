import networkx as nx
import pandas as pd
from args import get_args
from config.model_config import make_model_config
from utils.logging_util import logger
from collections import defaultdict
from pathlib import Path
import re
from modeling.hardware.power_calculator import get_static_power_aggregate
from config.dram_config import make_dram_config
from modeling.core.dram_info import DRAMConfig
from misc.type import get_bytes_per_element, HarmoniTensorType

def clean_name(x):
    if hasattr(x, 'name'):
        return x.name
    s = str(x)
    # Remove enum prefixes like "NodeTagType." or "NodeType." or any "Something."
    return re.sub(r"^[A-Za-z_]+\.", "", s)

def get_total_static_power(args):
    """
    Calculate total static power consumption for the entire system.
    
    Returns:
        float: Total static power in mW
    """
    # Get DRAM configuration
    dram_config = make_dram_config(args.dram)
    dram = DRAMConfig(dram_config)
    
    bytes_per_element = get_bytes_per_element(args.dtype, HarmoniTensorType.ACT)
    systolic_width = dram.bank_interface // (bytes_per_element * 8)
    
    # Get static power per chiplet in mW
    static_power_mW = get_static_power_aggregate(
        systolic_height=args.systolic_height,
        systolic_width=systolic_width,
        total_banks=dram.total_banks
    )
    
    # Calculate total static power for entire system
    total_static_power_mW = static_power_mW * dram.num_chips_per_rank * dram.num_ranks_per_channel * dram.num_channels
    
    return total_static_power_mW

def print_energy_breakdown(energy_breakdown, filename):
    output_path = f"outputs/energy_breakdown_{filename}.txt"
    Path("outputs").mkdir(exist_ok=True)
    
    # Compute total energy for each phase and tag for sorting
    phase_totals = defaultdict(float)
    tag_totals = defaultdict(lambda: defaultdict(float))
    type_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    # Calculate totals for sorting
    for phase in energy_breakdown:
        for tag in energy_breakdown[phase]:
            for type_ in energy_breakdown[phase][tag]:
                for kernel in energy_breakdown[phase][tag][type_]:
                    exec_energy = energy_breakdown[phase][tag][type_][kernel]['exec']
                    comm_energy = energy_breakdown[phase][tag][type_][kernel]['comm']
                    total_energy = exec_energy + comm_energy
                    
                    phase_totals[phase] += total_energy
                    tag_totals[phase][tag] += total_energy
                    type_totals[phase][tag][type_] += total_energy
    
    total_energy_consumption = sum(phase_totals.values())
    
    with open(output_path, "w") as f:
        f.write("\n=== Energy Breakdown ===\n")
        f.write(f"Dynamic Energy Consumption: {total_energy_consumption:.2f} mJ\n")
        
        # Print breakdown by Phase → Tag → Type → Kernel, all sorted
        f.write("By Phase → NodeTagType → NodeType → KernelType:\n")
        
        # Sort phases by total energy
        for phase in sorted(energy_breakdown.keys(), key=lambda p: -phase_totals[p]):
            clean_phase = clean_name(phase)
            f.write(f"  {clean_phase}:\n")
            
            # Sort tags by total energy within phase
            for tag in sorted(energy_breakdown[phase].keys(), key=lambda t: -tag_totals[phase][t]):
                clean_tag = clean_name(tag)
                f.write(f"    {clean_tag}:\n")
                
                # Sort types by total energy within tag
                for type_ in sorted(energy_breakdown[phase][tag].keys(), key=lambda t: -type_totals[phase][tag][t]):
                    clean_type = clean_name(type_)
                    # Collect and sort kernels by total energy
                    kernel_list = []
                    for kernel in energy_breakdown[phase][tag][type_]:
                        exec_energy = energy_breakdown[phase][tag][type_][kernel]['exec']
                        comm_energy = energy_breakdown[phase][tag][type_][kernel]['comm']
                        total_kernel_energy = exec_energy + comm_energy
                        kernel_list.append((kernel, exec_energy, comm_energy, total_kernel_energy))
                    
                    # Sort kernels by total energy descending
                    for kernel, exec_energy, comm_energy, total_kernel_energy in sorted(kernel_list, key=lambda x: -x[3]):
                        clean_kernel = clean_name(kernel)
                        pct_total = 100 * total_kernel_energy / total_energy_consumption if total_energy_consumption > 0 else 0
                        f.write(f"      {clean_type:11} | {clean_kernel:12}: {total_kernel_energy:8.2f} mJ ({pct_total:6.2f}%) Exec- {exec_energy:8.2f} mJ Comm- {comm_energy:8.2f} mJ\n")
            f.write("\n")
        
        # Energy Attribution (across full system):
        f.write("Energy Attribution (across full system):\n")
        total_exec_energy = 0.0
        total_comm_energy = 0.0
        
        for phase in energy_breakdown:
            for tag in energy_breakdown[phase]:
                for type_ in energy_breakdown[phase][tag]:
                    for kernel in energy_breakdown[phase][tag][type_]:
                        total_exec_energy += energy_breakdown[phase][tag][type_][kernel]['exec']
                        total_comm_energy += energy_breakdown[phase][tag][type_][kernel]['comm']
        
        f.write(f"  Execution Energy  : {total_exec_energy:8.2f} mJ ({100 * total_exec_energy / total_energy_consumption:6.2f}%)\n")
        f.write(f"  Communication Energy: {total_comm_energy:8.2f} mJ ({100 * total_comm_energy / total_energy_consumption:6.2f}%)\n")
        f.write(f"  Total Dynamic Energy: {total_energy_consumption:8.2f} mJ\n\n")
    
    logger.info(f"[FILE] Energy breakdown saved to: {output_path}")
    logger.info(f"Dynamic energy consumption: {total_energy_consumption:.2f} mJ")
    
    return total_energy_consumption #mJ

def compute_energy_start_finish_times_with_parents(dfg: nx.DiGraph, args):
    start_times = {}
    finish_times = {}
    task_profile = {}
    logic_unit_available = defaultdict(lambda: 0)
    critical_parents = {}
    
    # Initialize energy breakdown structure
    energy_breakdown = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))
    
    # Get layer count for multiplier calculation
    model = make_model_config(args.model_name, args.dtype)
    layer_count = model["nlayers"]

    for node in nx.topological_sort(dfg):
        preds = list(dfg.predecessors(node))
        raw_ready_time = max([finish_times[p] for p in preds] or [0])
        max_arrival_time = max([finish_times[p] + dfg[p][node].get('comm_time', 0) for p in preds] or [0])
        
        logic_unit = dfg.nodes[node].get('logic_unit', None)
        exec_time = dfg.nodes[node].get('exec_time', 0)
        exec_energy = dfg.nodes[node].get('exec_energy', 0)
        comm_energy = sum([dfg[p][node].get('comm_energy', 0) for p in preds] or [0]) 

        ready_time = max_arrival_time
        lu_ready_time = logic_unit_available[logic_unit]
        start_time = max(ready_time, lu_ready_time)
        end_time = start_time + exec_time

        if preds:
            critical_pred = max(
                preds,
                key=lambda p: finish_times[p] + dfg[p][node].get('comm_time', 0)
            )
            critical_parents[node] = critical_pred
        else:
            critical_parents[node] = None  # no predecessors (true source node)


        # Update dictionaries
        start_times[node] = start_time
        finish_times[node] = end_time
        logic_unit_available[logic_unit] = end_time

        # Delay from input ready to actual start
        stall_delay = start_time - ready_time
        comm_overhead = max_arrival_time - raw_ready_time

        if lu_ready_time > ready_time:
            stall_cause = "logic"
            logic_wait = stall_delay
        else:
            stall_cause = "none"
            logic_wait = 0

        phase = dfg.nodes[node].get("phase", "")
        type_ = dfg.nodes[node].get("type", "")
        tag = dfg.nodes[node].get("tag", "")
        kernel = dfg.nodes[node].get("kernel", "")
        layer = dfg.nodes[node].get("layer", -1) 
        
        # Calculate multiplier for layer optimization
        multiplier = 1
        if 'layer' in args.optimization:
            if layer != -1:
                multiplier = layer_count

        # Accumulate energy breakdown
        energy_breakdown[phase][tag][type_][kernel]['exec'] += (exec_energy * multiplier) 
        energy_breakdown[phase][tag][type_][kernel]['comm'] += (comm_energy * multiplier) 
        
        task_profile[node] = {
            "task_name": node,
            "phase": phase,
            "type": type_,
            "tag": tag,
            "kernel": kernel,
            "layer": layer,
            "logic_unit": logic_unit,
            "start_time": start_time,
            "end_time": end_time,
            "raw_ready_time": raw_ready_time,
            "ready_time": ready_time,
            "exec_time": exec_time,
            "logic_wait_time": logic_wait,
            "comm_overhead": comm_overhead,
            "stall_cause": stall_cause,
            "stall_delay": stall_delay,
            "num_preds": len(preds),
        }

    return start_times, finish_times, task_profile, critical_parents, energy_breakdown


def trace_critical_path(critical_parents, finish_times):
    node = max(finish_times, key=finish_times.get)
    path = [node]
    while node in critical_parents and critical_parents[node] is not None:
        parent = critical_parents[node]
        path.append(parent)
        node = parent
    return list(reversed(path))


def breakdown_critical_path(task_profile, critical_path, filename, top_k=10, args=None):
 
    if args is None:
        args = get_args()
    from pathlib import Path
    nested_breakdown = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    stall_logic = 0.0
    stall_comm = 0.0

    total_time = 0.0
    path_details = []
    df_rows = []
    model = make_model_config(args.model_name, args.dtype)
    layer_count = model["nlayers"]

    # Aggregate by (phase, tag, type, kernel)
    agg = defaultdict(lambda: {"time_critical": 0.0, "time_active": 0.0})

    
    for node in critical_path:
        info = task_profile[node]
        active_duration = info["end_time"] - info["start_time"]
        critical_duration = info["end_time"] - info["raw_ready_time"]
        phase = clean_name(info["phase"])
        tag = clean_name(info.get("tag", ""))
        type_ = clean_name(info["type"])
        kernel = clean_name(info.get("kernel", ""))
        layer = info.get("layer", -1)

        multiplier = 1
        if 'layer' in args.optimization:
            if layer != -1:
                multiplier = layer_count

        nested_breakdown[phase][tag][type_][kernel] += (critical_duration * multiplier)
        stall_logic += (info["logic_wait_time"] * multiplier)
        stall_comm += (info["comm_overhead"] * multiplier)
        total_time += (critical_duration * multiplier)

        path_details.append((node, phase, tag, type_, kernel, critical_duration * multiplier))
        agg[(phase, tag, type_, kernel)]["time_critical"] += (critical_duration * multiplier)
        agg[(phase, tag, type_, kernel)]["time_active"] += (active_duration * multiplier)

    pure_compute = total_time - stall_logic - stall_comm
    
    #Percentage breakdown
    latency_breakdown = {'Compute': pure_compute*100/total_time, 
                         'Communication': stall_comm*100/total_time, 
                         'Queueing': stall_logic*100/total_time # can also be called structural hazard
                        }

    # Compute total critical time for each phase and tag for sorting
    phase_totals = defaultdict(float)
    tag_totals = defaultdict(lambda: defaultdict(float))
    for (phase, tag, type_, kernel), v in agg.items():
        phase_totals[phase] += v["time_critical"]
        tag_totals[phase][tag] += v["time_critical"]

    # Prepare output file path
    output_path = f"outputs/critical_path_breakdown_{filename}.txt"
    Path("outputs").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n=== Critical Path Breakdown ===\n")
        if 'layer' in args.optimization:
            critical_latency = total_time
        else:
            critical_latency = task_profile[critical_path[-1]]["end_time"]
            if abs(critical_latency - total_time) > 0.01 * total_time:
                logger.error(f"Why does critical_latency {critical_latency} not match total_time {total_time}?")

        logger.info(f"Total Critical Path Latency: {critical_latency:.3f} us\n")
        f.write(f"Total Critical Path Latency: {critical_latency:.3f} us\n\n")
        
        # Print breakdown by Phase → Tag → Type → Kernel, all sorted
        f.write("By Phase → NodeTagType → NodeType → KernelType:\n")
        # Sort phases by total critical time
        for phase in sorted(nested_breakdown.keys(), key=lambda p: -phase_totals[p]):
            f.write(f"  {phase}:\n")
            # Sort tags by total critical time within phase
            for tag in sorted(nested_breakdown[phase].keys(), key=lambda t: -tag_totals[phase][t]):
                f.write(f"    {tag}:\n")
                # Collect and sort (type, kernel) by pct_critical
                type_kernel_list = []
                for type_ in nested_breakdown[phase][tag]:
                    for kernel in nested_breakdown[phase][tag][type_]:
                        time_critical = nested_breakdown[phase][tag][type_][kernel]
                        pct_critical = 100 * time_critical / total_time if total_time > 0 else 0
                        time_active = agg[(phase, tag, type_, kernel)]["time_active"]
                        type_kernel_list.append((type_, kernel, time_critical, pct_critical, time_active))
                # Sort by pct_critical descending
                for type_, kernel, time_critical, pct_critical, time_active in sorted(type_kernel_list, key=lambda x: -x[3]):
                    f.write(f"      {type_:11} | {kernel:12}: {time_critical:8.3f} us ({pct_critical:6.2f}%) Active- {time_active:8.3f} us\n")
            f.write("\n")

        f.write("Stall Attribution (across full critical path):\n")
        f.write(f"  Logic Stall    : {stall_logic:8.3f} us ({100 * stall_logic / total_time:6.2f}%)\n")
        f.write(f"  Comm Overhead  : {stall_comm:8.3f} us ({100 * stall_comm / total_time:6.2f}%)\n")
        f.write(f"  Pure Compute   : {pure_compute:8.3f} us ({100 * pure_compute / total_time:6.2f}%)\n\n")

        f.write(f"Top {top_k} Longest Tasks on Critical Path (with rich info):\n")
        for node, phase, tag, type_, kernel, critical_duration in sorted(path_details, key=lambda x: x[5], reverse=True)[:top_k]:
            f.write(f"  {node:25} | {phase:8} | {tag:10} | {type_:10} | {kernel:12} | dur={critical_duration:7.3f} us\n")

    logger.info(f"[FILE] Critical task breakdown saved to: {output_path}")

    return phase_totals, latency_breakdown

def dump_task_profile_to_csv(task_profile: dict, critical_path: list, filename: str = "with_critical_path.csv"):
    """
    Dumps the full task profile to CSV with a flag marking critical path tasks.

    Args:
        task_profile (dict): Dictionary from compute_energy_start_finish_times_with_parents
        critical_path (list): List of node names on the critical path
        filename (str): Output CSV filename
    """
    df = pd.DataFrame.from_dict(task_profile, orient='index')
    df['on_critical_path'] = df.index.isin(critical_path)
    df['active_duration'] = df['end_time'] - df['start_time']
    # Ensure tag and kernel columns exist and are string type
    if 'tag' not in df.columns:
        df['tag'] = ''
    else:
        df['tag'] = df['tag'].astype(str)
    if 'kernel' not in df.columns:
        df['kernel'] = ''
    else:
        df['kernel'] = df['kernel'].astype(str)

    # Clean up relevant columns
    for col in ['phase', 'type', 'tag', 'kernel', 'logic_unit']:
        if col in df.columns:
            df[col] = df[col].apply(clean_name)

    output_path = f"outputs/task_profile_{filename}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"[FILE] Task profile CSV saved to: {output_path}")

def print_performance_summary(task_profile, critical_path, filename, phase_totals=None, latency_breakdown=None, dynamic_energy_mJ=0, static_power_mW=0, args=None):

    if args is None:
        args = get_args()
    input_tokens = args.input_tokens
    output_tokens = args.output_tokens
    batch_size = args.batch
    
    if 'layer' in args.optimization and phase_totals is not None:
        # Version 1: With layer optimization - use phase_totals from breakdown_critical_path
        prefill_latency_us = phase_totals.get("PREFILL", 0.0)
        decode_latency_us = phase_totals.get("DECODE", 0.0)
        
        TTFT = prefill_latency_us / 1000.0  # microseconds to ms
        decode_latency_ms = decode_latency_us / 1000.0
        E2E_latency_us = prefill_latency_us + decode_latency_us
        E2E_latency_ms = E2E_latency_us / 1000.0
        
        logger.info(f"[LAYER_OPT] Using phase_totals: PREFILL={prefill_latency_us/1000:.3f}ms, DECODE={decode_latency_us/1000:.3f}ms")
        
    else:
        # Version 2: Without layer optimization - use raw end_times
        # Extract end times
        # CLEANUP: check if the version 2 is still needed?
        prefill_end_times = [v["end_time"] for v in task_profile.values() if getattr(v["phase"], 'name', v["phase"]) == "PREFILL"]
        decode_end_times = [v["end_time"] for v in task_profile.values() if getattr(v["phase"], 'name', v["phase"]) == "DECODE"]

        TTFT = max(prefill_end_times or [0]) / 1000.0  # microseconds to ms
        decode_latency_us = max(decode_end_times or [0]) - TTFT * 1000.0
        decode_latency_ms = decode_latency_us / 1000.0
        E2E_latency_us = task_profile[critical_path[-1]]["end_time"]
        E2E_latency_ms = E2E_latency_us / 1000.0
        
        logger.info(f"[NO_LAYER_OPT] Using raw end_times: TTFT={TTFT:.3f}ms, decode={decode_latency_ms:.3f}ms")

    # Compute throughput
    prefill_tokens = batch_size * input_tokens
    decode_tokens = batch_size * output_tokens

    prefill_TPS = prefill_tokens / TTFT if TTFT > 0 else 0
    decode_TPS = decode_tokens / (decode_latency_ms / 1000.0) if decode_latency_ms > 0 else 0
    TPOT = decode_latency_ms / decode_tokens if decode_tokens > 0 else 0
    E2E_TPS = (prefill_tokens + decode_tokens) / (E2E_latency_ms / 1000.0) if E2E_latency_ms > 0 else 0 #NOTE: considering prompt tokens as well similar to CENT
    
    # Calculate static energy using E2E latency
    static_energy_mJ = static_power_mW * E2E_latency_ms/1000.0 if static_power_mW > 0 else 0
    total_energy = static_energy_mJ + dynamic_energy_mJ

    output_path = f"outputs/performance_summary_{filename}.txt"
    Path("outputs").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n=== Inference Performance Summary ===\n")
        if 'layer' in args.optimization:
            f.write("(Using layer optimization - scaled phase totals)\n")
        else:
            f.write("(Using raw timeline - no layer optimization)\n")
        f.write(f"Prefill latency (TTFT)     : {TTFT:7.3f} ms\n")
        f.write(f"Prefill Throughput         : {prefill_TPS:7.2f} tokens/sec\n")
        f.write(f"Decode latency             : {decode_latency_ms:7.3f} ms\n")
        f.write(f"TPOT (ms/token)            : {TPOT:7.3f} ms\n")
        f.write(f"Decode Throughput (TPS)    : {decode_TPS:7.2f} tokens/sec\n")
        f.write(f"E2E Latency                : {E2E_latency_ms:7.3f} ms\n")
        f.write(f"E2E Throughput (TPS)       : {E2E_TPS:7.2f} tokens/sec\n")
        f.write(f"Dynamic Energy (J)         : {dynamic_energy_mJ/1000:7.2f} J\n")
        f.write(f"Static Energy (mJ)         : {static_energy_mJ:7.2f} mJ\n")
        f.write(f"Total Energy (J)           : {(total_energy)/1000:7.2f} J\n")
        f.write(f"Energy per token (mJ)      : {(total_energy/decode_tokens):7.2f} mJ\n")
        f.write(f"Tokens per J               : {((decode_tokens*1000)/total_energy):7.2f} \n")
        
        # Handle latency_breakdown safely
        if latency_breakdown is not None and isinstance(latency_breakdown, dict):
            f.write(f"Compute (%)                : {latency_breakdown.get('Compute', 0.0):7.2f} %\n")
            f.write(f"Communication (%)          : {latency_breakdown.get('Communication', 0.0):7.2f} %\n")
            f.write(f"Queueing (%)               : {latency_breakdown.get('Queueing', 0.0):7.2f} %\n")
        else:
            f.write(f"Compute (%)                : {0.0:7.2f} %\n")
            f.write(f"Communication (%)          : {0.0:7.2f} %\n")
            f.write(f"Queueing (%)               : {0.0:7.2f} %\n")

    logger.info(f"[FILE] Performance summary saved to: {output_path}")

def get_critical_path_analysis(dfg, filename, top_k=10, args=None):

    if args is None:
        args = get_args()
    start_times, finish_times, task_profile, critical_parents, energy_profile = compute_energy_start_finish_times_with_parents(dfg, args)
    critical_path = trace_critical_path(critical_parents, finish_times)
    phase_totals, latency_breakdown = breakdown_critical_path(task_profile, critical_path, filename, top_k=top_k, args=args)
    
    static_power_mW = get_total_static_power(args)
    
    dynamic_energy_mJ = print_energy_breakdown(energy_profile, filename)
    
    print_performance_summary(task_profile, critical_path, filename, phase_totals, latency_breakdown, dynamic_energy_mJ, static_power_mW, args)
    if 'task_profile' in args.dump_stats:
        dump_task_profile_to_csv(task_profile, critical_path, filename)
