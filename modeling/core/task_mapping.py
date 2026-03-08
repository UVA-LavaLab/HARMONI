from misc.type import *
from modeling.core.tensor import *
from modeling.core.logic import *
from modeling.core.memory_system import *
from modeling.core.dram_utils import *
import networkx as nx
from collections import defaultdict
import math
import copy
from tqdm import tqdm
from functools import lru_cache
import csv
from collections import Counter, defaultdict
from args import get_args

_find_matching_logic_units_cache = {}

def find_matching_logic_units(loc_dict, memsys, op_type=None):
    """
    Find logic units that match the given location dictionary.
    Returns a list of matching logic units.
    Uses a cache and pre-computed index to speed up lookups.
    """
    loc_dict_tuple = tuple(sorted(loc_dict.items()))
    cache_key = (loc_dict_tuple, op_type)

    if cache_key in _find_matching_logic_units_cache:

        return _find_matching_logic_units_cache[cache_key]


    # Use the pre-computed partial index from MemorySystem for a direct lookup.
    # This avoids iterating through all logic units.
    matching_units = memsys.logic_unit_partial_index.get(loc_dict_tuple, [])

    # If node_type is specified, filter by supported_ops
    if op_type is not None:
        matching_units = [
            lu for lu in matching_units
            if any(
                op_type == sop or str(op_type) == str(sop)
                for sop in lu.supported_ops
            )
        ]

    _find_matching_logic_units_cache[cache_key] = [lu for lu in matching_units]

    return matching_units

def _get_agg_to_successor_comm_optype(successor_type):
    """Return comm type for agg->successor edges based on destination node type."""
    if successor_type == NodeType.APPEND:
        return CommType.GATHER_SCATTER
    return None


def replace_task_with_aggregate(model_dfg, original_task, successors, agg_node):
    """
    Redirects all outgoing edges from original_task to agg_node and removes original_task.
    """

    # Add edges from agg_node to all successors
    if agg_node is not None:
        for succ in successors:
            succ_type = model_dfg.graph.nodes[succ].get('type')
            comm_optype = _get_agg_to_successor_comm_optype(succ_type)
            model_dfg.add_edge(agg_node, succ, comm_optype=comm_optype)
            model_dfg.graph.remove_edge(original_task, succ)
    
def map_tasks_to_logic_units(model_dfg, memsys, dram, args=None):
    """
    Map tasks to logic units at chip granularity.
    For each tensor partition, create a sub-task and map to the matching logic unit.
    """
    if args is None:
        args = get_args()
    tasks = list(nx.topological_sort(model_dfg.graph))
    task_mappings = defaultdict(list)
    
    for task in tqdm(tasks, desc="Chipwise mapping"):
        
        task_attrs = model_dfg.graph.nodes[task]

        if 'GPU-Emb-Unemb' in args.optimization:        
            kernel_name = task_attrs.get('kernel', '')
            if kernel_name in ['embed', 'lm_head', 'lm_head_a', 'lm_head_b']:
                # Assign the root unit to the task
                lu = memsys.get_logic_unit_by_id({'root': 0})[0]
                logger.info(f"Skipping {task} inside map_tasks_to_logic_units mapped to lu {lu.name}")
                lu.add_task(task)
                task_mappings[task].append(lu)
                # Add logic unit to task attributes
                task_attrs['logic_unit'] = lu 
                continue
        
        input_tensors = [t for t in task_attrs.get('ip', []) if t.tag in [HarmoniTensorType.WEIGHT, HarmoniTensorType.KV]]
        if len(input_tensors) > 1:
            logger.warning(f"For task {task}, look why are there multiple weight/KV input tensors")
        
        if not input_tensors:
            
            if 'static_mapping' in args.optimization:
                kernel_type = task_attrs.get('type', '')
                if kernel_type in [NodeType.RMSNORM, NodeType.SILU, NodeType.ARGMAX, NodeType.EWMUL]:
                    lu = memsys.get_logic_unit_by_id({'channel': 0, 'wt_rank': 0})[0]
                    lu.add_task(task)
                    task_mappings[task].append(lu)
                    task_attrs['logic_unit'] = lu 
                    continue
                if kernel_type in [NodeType.ROTARY]:
                    head_num = task_attrs.get('head', '') 
                    lu = memsys.get_logic_unit_by_id({'channel': 0, 'wt_rank': 0, 'chip': head_num%dram.num_chips_per_rank})[0]
                    lu.add_task(task)
                    task_mappings[task].append(lu)
                    task_attrs['logic_unit'] = lu 
                    continue
            
            # Find the logic unit assigned to the predessor of this task
            # assign the same logic unit to this task and add to the task_mappings 
            # if there are multiple predessor, log warning and get the closest parent of the logic units.
            predecessors = list(model_dfg.graph.predecessors(task))
            
            # If the only predecessor is a SYNC node, use its predecessors instead
            if len(predecessors) == 1:
                pred_node = predecessors[0]
                pred_type = model_dfg.graph.nodes[pred_node].get('type', None)
                if pred_type == NodeType.SYNC:
                    predecessors = list(model_dfg.graph.predecessors(pred_node))
            
            if not predecessors:
                logger.warning(f"Task {task} has no predecessors, cannot assign logic unit") 
                continue

            # Get logic units from predecessors
            pred_logic_units = []
            for pred in predecessors:
                if 'logic_unit' in model_dfg.graph.nodes[pred]:
                    pred_logic_units.append(model_dfg.graph.nodes[pred]['logic_unit'])
            
            if not pred_logic_units:
                logger.warning(f"No logic units found in predecessors of task {task}")
                continue
                
            # If multiple predecessors have different logic units, find closest common parent
            if len(set(pred_logic_units)) > 1:

                lu = find_closest_common_parent(pred_logic_units, memsys)
                if not lu:
                    logger.warning(f"No common parent found for logic units of task {task}") 
                    continue
            else:
                lu = pred_logic_units[0]
                
            # Assign the logic unit to the task
            lu.add_task(task)
            task_mappings[task].append(lu)
            # Add logic unit to task attributes
            task_attrs['logic_unit'] = lu
            continue  # Skip if no relevant tensors
        
        subtask_names = []
        # Getting successors of the task before adding subtasks
        successors = list(model_dfg.graph.successors(task))
        for tensor in input_tensors:
            partitions = tensor.locations

            if len(partitions) == 1:
                # Get the single partition's location and find matching logic unit
                loc_dict, size, start_addr, end_addr, rows, cols = partitions[0]
                matching_units = find_matching_logic_units(loc_dict, memsys, task_attrs['type'])
                if matching_units:
                    lu = matching_units[0]
                    assert len(matching_units) == 1, f"Multiple matching nodes for a tensor location in task {task}"
                    lu.add_task(task)
                    task_mappings[task].append(lu)
                    # Add logic unit to task attributes
                    task_attrs['logic_unit'] = lu
                continue
            
            for loc_dict, size, start_addr, end_addr, rows, cols in partitions:
                # Find logic unit at this chip location
                matching_units = find_matching_logic_units(loc_dict, memsys, task_attrs['type']) #CLEANUP: check if its just one matching unit
                
                for lu in matching_units:
                    if lu.id.get('wt_rank') != None:
                        subtask_name = f"{task}_ch{loc_dict.get('channel')}_wtrank_{loc_dict.get('wt_rank')}_chip{loc_dict.get('chip')}"
                    else:
                        subtask_name = f"{task}_ch{loc_dict.get('channel')}_kvrank_{loc_dict.get('kv_rank')}_chip{loc_dict.get('chip')}"
                    
                    # Modify input tensors for this subtask
                    modified_ip = []
                    for ip_tensor in task_attrs.get('ip', []):
                        if ip_tensor == tensor:
                            # Calculate the ratio of data at this location
                            ratio = size / ip_tensor.size
                            modified_tensor = create_modified_tensor(ip_tensor, loc_dict, size, start_addr, end_addr, rows, cols, ratio, task_attrs)
                            modified_ip.append(modified_tensor)
                        else:
                            # For non-partitioned tensors, keep them as is
                            modified_ip.append(ip_tensor)

                    #If you ever need to mutate a nested object in-place, consider copying that object as well. 
                    subtask_attrs = dict(task_attrs)  # shallow copy
                    subtask_attrs['logic_unit'] = lu
                    subtask_attrs['ip'] = modified_ip 
                    
                    modified_op = []
                    if subtask_attrs['type'] in [NodeType.GEMM,NodeType.FUSED_SCORE]:
                        op_tensor_list = task_attrs.get('op', [])
                        assert len(op_tensor_list) == 1, "GEMM/FUSED_SCORE in node {task} producing multiple output tensors {op_tensor_list}"
                        op_tensor = op_tensor_list[0]
                        new_shape = list(op_tensor_list[0].shape)
                        for idx in range(len(modified_ip[0].shape[:-1])):
                            new_shape[idx] = modified_ip[0].shape[idx]
                        new_shape[-1] = modified_ip[1].shape[-1]

                        # Pass the new shape as a parameter
                        modified_tensor = HarmoniTensor(
                            name=op_tensor.name,
                            tag=op_tensor.tag,
                            precision=op_tensor.precision,
                            shape=new_shape,  # shape is already tuple
                            stride=op_tensor.stride,
                            dram=op_tensor.dram
                        )
                        modified_op.append(modified_tensor)
                    
                    subtask_attrs['op'] = modified_op
                    
                    # Add the node with modified attributes
                    model_dfg.add_node(subtask_name, **subtask_attrs)
                    subtask_names.append(subtask_name)

                    sync_node = f"{task}_sync"
                    sync_attrs = dict(task_attrs)  # shallow copy
                    # Add the sync node with minimal attributes (can be extended as needed)
                    sync_attrs['ip'] = []
                    sync_attrs['op'] = []
                    sync_attrs['type'] = NodeType.SYNC
                    sync_attrs['kernel'] = "sync"

                    model_dfg.add_node(sync_node, **sync_attrs)
                    # Redirect all incoming edges to the sync node
                    predecessors = list(model_dfg.graph.predecessors(task))
                    for pred in predecessors:
                        if model_dfg.graph.has_edge(pred, task):
                            model_dfg.graph.remove_edge(pred, task)
                            model_dfg.add_edge(pred, sync_node)
                    # Connect the sync node to all subtasks
                    model_dfg.add_edge(sync_node, subtask_name)
                    

                    lu.add_task(subtask_name)
                    task_mappings[subtask_name].append(lu) 

            # Add aggregation node after all subtasks for this task are created
            agg_node = add_aggregation_node_for_task(model_dfg, task, task_mappings, memsys, subtask_names)
            replace_task_with_aggregate(model_dfg, task, successors, agg_node)
        
            # Remove the original task node if it exists
            if model_dfg.graph.has_node(task):
                model_dfg.graph.remove_node(task)
    #logger.debug(f"[Updated] Inspecting the model_dfg after update inside map_tasks_to_logic_units function: {model_dfg}")
    return task_mappings

def create_modified_tensor(ip_tensor, loc_dict, size, start_addr, end_addr, rows, cols, ratio, task_attrs):

    """Create a new tensor with modified shape and chip_idx"""

    # For GEMM operations, we need to consider the tensor's role
    # and modify the appropriate dimension
    new_shape = list(ip_tensor.shape)

    # Determine which dimension to split based on tensor's role in GEMM
    if task_attrs.get('type') == NodeType.GEMM: 
        # For GEMM C = A × B:
        # - A has shape [M, K]
        # - B has shape [K, N]
        # - C has shape [M, N]
        
        # If this is the first input tensor (A), split the M dimension
        if ip_tensor == task_attrs.get('ip', [])[0]:
            #M remains same
            #FUTURE-WORK: M could be replicated across the chips (only for GEMMs in decode, for Prefill phase - split this M in multiple ranks, and channels)
            #K remains same
            new_shape = new_shape
             
        else:
            # For second input tensor (B), split the N dimension
            # Split columns (N dimension)
            new_shape[-1] = math.ceil(new_shape[-1] * ratio) 

    else:
        # For non-GEMM operations, use a default strategy
        # (could be customized based on operation type)
        new_shape[-1] = math.ceil(new_shape[-1] * ratio)
    
    # Pass the new shape as a parameter
    modified_tensor = HarmoniTensor(
        name=ip_tensor.name,
        tag=ip_tensor.tag,
        precision=ip_tensor.precision,
        shape=new_shape,  # shape is already tuple
        stride=ip_tensor.stride,
        addr_offset=start_addr, 
        chip_idx=loc_dict.get('chip'),
        mapping = ip_tensor.mapping,
        dram=ip_tensor.dram
    )
    
    modified_tensor.locations = [(loc_dict, size, start_addr, end_addr, rows, cols)]
    modified_tensor.row_accesses = rows #NOTE: rows, cols do not depend on shape as of now
    modified_tensor.col_accesses = cols
    
    if modified_tensor.size != size:
        logger.debug(f"Inside modified tensors the calculated size for {modified_tensor.name} is {modified_tensor.size} and locations size is {size}")
    logger.debug(f"new_shape {new_shape} original shape {ip_tensor.shape}")
    return modified_tensor

def _make_logic_units_hashable(logic_units):
    """
    Convert logic units to a hashable format for caching.
    Returns a tuple of tuples containing the logic unit IDs.
    """
    return tuple(tuple(sorted(lu.id.items())) for lu in logic_units)

@lru_cache(maxsize=1024)
def _find_closest_common_parent_cached(logic_units_tuple, memsys):
    """
    Cached version of find_closest_common_parent that works with hashable logic unit IDs.
    """
    if not logic_units_tuple:
        return None
        
    # Convert back to logic units
    logic_units = []
    for lu_id in logic_units_tuple:
        # Find the logic unit with this ID
        for lu in memsys.get_all_logic_units():
            if tuple(sorted(lu.id.items())) == lu_id:
                logic_units.append(lu)
                break
    
    # Start with the logic units themselves
    current_level = set(logic_units)
    
    if not current_level:
        logger.warning(f"No parents found for logic unit {logic_units[0].name}")
        return None
        
    # Keep track of visited nodes to avoid cycles
    visited = set()
    
    while current_level:
        # Check if any node in current level is common to all logic units
        for node in current_level:
            if node in visited:
                continue
                
            visited.add(node)
            
            # Check if this node is an ancestor of all logic units
            is_common = True
            for lu in logic_units:
                if lu == node:
                    continue
                # Get all ancestors of this logic unit
                lu_ancestors = set()
                current = lu
                while current.parents:
                    lu_ancestors.update([p for p in current.parents if isinstance(p, LogicUnit)])
                    current = current.parents[0] if current.parents else None
                
                if node not in lu_ancestors:
                    is_common = False
                    break
                    
            if is_common:
                return tuple(sorted(node.id.items()))
                
        # Move up one level in hierarchy, only considering logic unit parents
        next_level = set()
        for node in current_level:
            next_level.update([p for p in node.parents if isinstance(p, LogicUnit)])
        current_level = next_level
        
    logger.warning("No common parent found for logic units")    
    return None

def find_closest_common_parent(logic_units, memsys):
    """
    Find the closest common parent of a list of logic units.
    Uses caching for better performance.
    """
    if not logic_units:
        return None
        
    # Convert logic units to hashable format
    logic_units_tuple = _make_logic_units_hashable(logic_units)
    
    # Get result from cached function
    result_id = _find_closest_common_parent_cached(logic_units_tuple, memsys)
    
    if result_id is None:
        logger.warning("No common parent found for logic units")
        return None
        
    # Convert result back to logic unit
    for lu in memsys.get_all_logic_units():
        if tuple(sorted(lu.id.items())) == result_id:
            return lu
            
    return None

def add_aggregation_node_for_task(model_dfg, original_task, task_mappings, memsys, subtask_names, agg_suffix="_agg"):
    """
    Adds an aggregation node that combines all chip-level subtasks for a given original task.
    The aggregation node inherits task attributes from the original task and is mapped to the closest common parent
    of the logic units where subtasks are mapped.
    """
    
    if not subtask_names:
        return None  # No subtasks found

    agg_node_name = f"{original_task}{agg_suffix}"
    
    # Get original task attributes
    original_attrs = model_dfg.graph.nodes[original_task]
    
    # Create aggregation node attributes
    agg_attrs = dict(original_attrs)  # shallow copy
    agg_attrs['type'] = NodeType.AGG
    
    # Find the logic units where subtasks are mapped
    subtask_logic_units = []
    for subtask in subtask_names:
        if 'logic_unit' in model_dfg.graph.nodes[subtask]:
            subtask_logic_units.append(model_dfg.graph.nodes[subtask]['logic_unit'])
    

    if subtask_logic_units:
        # Find the closest common parent using optimized algorithm
        closest_parent = find_closest_common_parent(subtask_logic_units, memsys)
        
        if closest_parent:
            agg_attrs['logic_unit'] = closest_parent
            closest_parent.add_task(agg_node_name)
            # Update task_mappings with the aggregation node's logic unit
            task_mappings[agg_node_name] = [closest_parent]
        else:
            # If no common parent found, use the first subtask's logic unit's parent
            # This ensures we have a valid logic unit for the aggregation node
            first_lu = subtask_logic_units[0]
            if first_lu.parents:
                fallback_parent = next((p for p in first_lu.parents if isinstance(p, LogicUnit)), None)
                if fallback_parent:
                    logger.warning(f"No common parent found for {original_task}, using fallback parent {fallback_parent.name}")
                    agg_attrs['logic_unit'] = fallback_parent
                    fallback_parent.add_task(agg_node_name)
                    task_mappings[agg_node_name] = [fallback_parent]
                else:
                    logger.error(f"No valid parent found for aggregation node {agg_node_name}")
            else:
                logger.error(f"No parents available for fallback in aggregation node {agg_node_name}")
    
    # Add the aggregation node with attributes
    model_dfg.add_node(agg_node_name, **agg_attrs)
    
    # Connect all subtasks to the aggregation node
    for subtask in subtask_names:
        model_dfg.add_edge(subtask, agg_node_name, comm_optype=CommType.REDUCE)
    
    return agg_node_name

def add_mapping_to_temp_tensors(model_dfg, temp_tensor_wt, temp_tensor_kv, dram, buffer):
    """
    Traverse the model_dfg and map temporary tensors to their appropriate locations.
    For tensors with empty mappings, use the logic unit's location to determine:
    1. addr_offset from temp tensors in weight/kv ranks
    2. chip_idx from logic unit id
    """

    # Get all nodes in the DFG
    for node, attrs in model_dfg.graph.nodes(data=True):
        # Check input tensors
        if ('ip' in attrs and 'op' in attrs):
            for attr_name in ['ip', 'op']:
                for tensor in attrs[attr_name]:
                    if tensor.mapping == {} and tensor.tag == HarmoniTensorType.ACT:
                        # Get the logic unit for this node
                        # FUTURE-WORK: fix the analysis for the impact of caching
                        if tensor.size < buffer: #Default 256kB SRAM
                            tensor.size = 1 
                            continue
                        
                        if 'logic_unit' in attrs:
                            lu = attrs['logic_unit']
                            
                            # Debug logging
                            logger.debug(f"Node {node}: Logic unit {lu.name} with ID {lu.id}")
                            
                            # Determine which type of rank this logic unit belongs to
                            is_weight_rank = 'wt_rank' in lu.id
                            is_kv_rank = 'kv_rank' in lu.id
                            
                            logger.debug(f"  - is_weight_rank: {is_weight_rank}, is_kv_rank: {is_kv_rank}")
                            
                            # Get the appropriate temp tensor based on rank type and channel
                            if is_weight_rank:
                                # Use weight rank temp tensor - find by channel and rank
                                rank_idx = lu.id['wt_rank']
                                channel_idx = lu.id.get('channel', 0)  # Default to channel 0 if not specified
                                
                                # Find temp tensor for this specific channel and rank
                                temp_tensor = None
                                for temp_t in temp_tensor_wt:
                                    if f"temp_tensor_wt_ch{channel_idx}_r{rank_idx}" in temp_t.name:
                                        temp_tensor = temp_t
                                        break
                                
                                if temp_tensor is None:
                                    logger.warning(f"No temp tensor found for channel {channel_idx}, wt_rank {rank_idx} for logic unit {lu.name}")
                                    continue
                                
                                tensor.addr_offset = temp_tensor.addr_offset
                                rank_type = "weight"
                                logger.debug(f"  - Using weight rank temp tensor channel {channel_idx} rank {rank_idx}")
                                
                            elif is_kv_rank:
                                # Use kv rank temp tensor - find by channel and rank
                                rank_idx = lu.id['kv_rank']
                                channel_idx = lu.id.get('channel', 0)  # Default to channel 0 if not specified
                                
                                # Find temp tensor for this specific channel and rank
                                temp_tensor = None
                                for temp_t in temp_tensor_kv:
                                    if f"temp_tensor_kv_ch{channel_idx}_r{rank_idx}" in temp_t.name:
                                        temp_tensor = temp_t
                                        break
                                
                                if temp_tensor is None:
                                    logger.warning(f"No temp tensor found for channel {channel_idx}, kv_rank {rank_idx} for logic unit {lu.name}")
                                    continue
                                
                                tensor.addr_offset = temp_tensor.addr_offset
                                rank_type = "kv"
                                logger.debug(f"  - Using KV rank temp tensor channel {channel_idx} rank {rank_idx}")
                                
                            else:
                                tensor.col_accesses = tensor.size 
                                continue
                            
                            # Update tensor properties
                            if 'chip' in lu.id:
                                tensor.chip_idx = lu.id['chip']
                            else:
                                tensor.chip_idx = -1
                            
                            tensor.mapping = get_mapping(tensor.addr_offset, dram, rank_type)
                            tensor.locations, tensor.row_accesses, tensor.col_accesses = HarmoniTensor.get_temp_tensor_locations(channel_idx, rank_idx, tensor.chip_idx, tensor.size, dram, rank_type)


def print_task_mappings(task_mappings, filename_str):
    """Print the task to logic unit mappings"""
    filename = f"outputs/task_mapping_{filename_str}.txt"
    task_mapping_info = []
    
    for task, units in task_mappings.items():
        line = f'{task};'
        for unit in units:
            line += f"{unit.name};{len(unit.instruction_queue)}"
        task_mapping_info.append(line)
    with open(filename, "w") as f:
        f.write("task;logic_unit;queue_size\n")
        for line in task_mapping_info:
            f.write(line + "\n")

def get_logic_units_with_task_types(model_dfg):
    """
    Extract unique logic units from DFG nodes and collect their task types distribution.
    Works with cached DFG (no task_mappings needed).
    
    Returns:
        dict: Mapping from logic unit ID tuple to info dict containing:
            - logic_unit: The LogicUnit object
            - id: Logic unit ID dictionary
            - name: Logic unit name
            - queue_size: Number of tasks in instruction queue
            - task_types: Counter of task types
            - tasks: List of (task_name, task_type) tuples
    """
    unique_lus = {}  # Key: tuple(sorted(lu.id.items()))
    
    for task_name, attrs in model_dfg.graph.nodes(data=True):
        if 'logic_unit' in attrs:
            lu = attrs['logic_unit']
            lu_key = tuple(sorted(lu.id.items()))
            task_type = attrs.get('type')
            
            if lu_key not in unique_lus:
                unique_lus[lu_key] = {
                    'logic_unit': lu,
                    'id': lu.id,
                    'name': lu.name,
                    'queue_size': len(lu.instruction_queue),
                    'task_types': Counter(),
                    'tasks': []
                }
            
            # Add this task's type to the counter
            if task_type:
                unique_lus[lu_key]['task_types'][task_type] += 1
                unique_lus[lu_key]['tasks'].append((task_name, task_type))
    
    return unique_lus


def get_logic_unit_operation_dist(task_mappings, model_dfg, filename_str, args=None):
    """
    Exports coarse- and fine-grained task type distributions to CSV files.
    """
    if args is None:
        args = get_args()
    coarse_file = f'outputs/lu_ops_coarse_dist_{filename_str}.csv'
    fine_file = f'outputs/lu_ops_fine_dist_{filename_str}.csv' 

    coarse_counters = defaultdict(Counter)
    fine_counters = defaultdict(Counter)

    for task, units in task_mappings.items():
        task_attrs = model_dfg.graph.nodes.get(task, {})
        logic_unit = task_attrs.get('logic_unit')
        task_type = task_attrs.get('type')
        if not logic_unit or not task_type:
            continue

        for lu in units:
            id_dict = lu.id

            keys = list(id_dict.keys())
            coarse_key = None
            # Determine the most specific level (chip > wt_rank > channel > root)
            if 'chip' in keys:
                coarse_key = 'chip'
            elif 'wt_rank' in keys:
                coarse_key = 'wt_rank'
            elif 'kv_rank' in keys:
                coarse_key = 'kv_rank'
            elif 'channel' in keys:
                coarse_key = 'channel'
            elif 'root' in keys:
                coarse_key = 'root'
            if coarse_key is None:
                continue  # Skip if logic unit has unknown granularity

            coarse_counters[coarse_key][task_type] += 1

            # Fine-grained full combo (in fixed key order)
            ordered_keys = ['root', 'channel', 'wt_rank', 'kv_rank', 'chip']
            
            if args.verbose:
                fine_key = "|".join(f"{k}-{id_dict[k]}" for k in ordered_keys if k in id_dict)
                fine_counters[fine_key][task_type] += 1

    # Write Coarse-Grained CSV
    with open(coarse_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Logic Unit Level', 'Task Type', 'Count', 'Percentage'])

        for lu_level, counter in coarse_counters.items():
            total = sum(counter.values())
            for task_type, count in counter.items():
                pct = (count / total) * 100 if total else 0
                writer.writerow([lu_level, str(task_type).replace("NodeType.", ""), count, f"{pct:.1f}"])

    if args.verbose:
        # Write Fine-Grained CSV
        with open(fine_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['root', 'channel', 'wt_rank', 'kv_rank', 'chip', 'Task Type', 'Count', 'Percentage'])
            ordered_keys = ['root','channel', 'wt_rank', 'kv_rank', 'chip']
            for lu_id, counter in sorted(fine_counters.items()):
                lu_fields = dict(field.split('-', 1) for field in lu_id.split('|') if '-' in field)
                total = sum(counter.values())
                for task_type, count in counter.items():
                    pct = (count / total) * 100 if total else 0
                    # Extract each field or leave blank if missing
                    row = [lu_fields.get(k, '') for k in ordered_keys]
                    row += [str(task_type).replace("NodeType.", ""), count, f"{pct:.1f}"]
                    writer.writerow(row)
