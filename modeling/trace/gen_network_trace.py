import networkx as nx
import os
import csv
from misc.type import *
from modeling.core.tensor import *
from modeling.core.transformer_dfg import DFG
from modeling.core.memory_system import *
from utils.logging_util import logger
import datetime
from args import get_args

def generate_network_trace(DFG, memsys, filename):
   
   logger.debug("Inside generate network trace")
   trace = []
   detailed_trace = []
   time_stamp = nx.get_node_attributes(DFG.graph, "exec_time")
   total_KV_bytes_moved = 0
   total_ACT_bytes_moved = 0
   total_WT_bytes_moved = 0
   total_IP_bytes_moved = 0
   # Track tensor movements between specific source-destination pairs
   counted_tensor_movements = set()  # Format: (tensor_name, src_lu_id_str, dst_lu_id_str)

   # Get all nodes in topological order to ensure we process dependencies correctly
   for node in nx.topological_sort(DFG.graph):
      node_attrs = DFG.graph.nodes[node]
      
      #FUTURE-WORK: Hack to skip APPEND nodes for communication cost (Could this be combined with no logic unit condition)
      if node_attrs.get('type') == NodeType.APPEND:
         continue
      # Skip if node doesn't have a logic unit assigned
      if 'logic_unit' not in node_attrs: 
         continue
         
      # get src node name
      src_node_name = node
      src_logic_unit = node_attrs['logic_unit']
      src_logic_unit_id = src_logic_unit.id
      src_lu_id_str = '_'.join(f'{k}_{v}' for k, v in src_logic_unit_id.items())
      
      # Get successors of this node (src node)
      successors = list(DFG.graph.successors(node))
      
      total_comm_time = 0
      total_comm_energy = 0
      # For each successor, create a trace entry if it has a logic unit
      for succ in successors:
         succ_attrs = DFG.graph.nodes[succ]
         if 'logic_unit' not in succ_attrs:
            #CLEANUP: What cases are these and mention here along with comments
            continue

         # dst node name
         dst_node_name = succ
         dst_logic_unit = succ_attrs['logic_unit']
         dst_logic_unit_id = dst_logic_unit.id
         dst_lu_id_str = '_'.join(f'{k}_{v}' for k, v in dst_logic_unit_id.items())
         
         # Skip if source and destination logic units are the same
         if src_lu_id_str == dst_lu_id_str:
            continue
          
         # Calculate data size from both input and output tensors of source node
         data_size = 0
         tensors_moved = []
         
         # Check output tensors
         for tensor in node_attrs.get('op', []):
            movement_key = (tensor.name, src_lu_id_str, dst_lu_id_str)
            if movement_key not in counted_tensor_movements:
                  counted_tensor_movements.add(movement_key)

            if tensor.tag == HarmoniTensorType.KV:
               total_KV_bytes_moved += tensor.size
               tensors_moved.append(tensor)
               data_size += tensor.size
            elif tensor.tag == HarmoniTensorType.ACT:
               total_ACT_bytes_moved += tensor.size
               tensors_moved.append(tensor)
               data_size += tensor.size
            elif tensor.tag == HarmoniTensorType.WEIGHT:
               total_WT_bytes_moved += tensor.size
               tensors_moved.append(tensor)
               data_size += tensor.size
            elif tensor.tag == HarmoniTensorType.INPUT:
               total_IP_bytes_moved += tensor.size
               tensors_moved.append(tensor)
               data_size += tensor.size
            
         
         # Only add trace if there is data movement
         if data_size > 0:
            # Get execution time of the source task
            exec_time = time_stamp.get(node, 0)
            
            # Get predecessors of the source task
            pred_predecessors = list(DFG.graph.predecessors(node))
            

            # Use routing table (or operation-aware model if comm_optype is provided)
            edge_comm_optype = DFG.graph[src_node_name][dst_node_name].get('comm_optype')
            comm_info = memsys.get_comm_info(
               src_logic_unit_id,
               dst_logic_unit_id,
               data_size,
               optype=edge_comm_optype,
            )
            comm_time = comm_info['comm_time'] if comm_info is not None else None
            comm_energy = comm_info['comm_energy'] if comm_info is not None else None #pJ
            comm_optype_str = edge_comm_optype.value if isinstance(edge_comm_optype, CommType) else edge_comm_optype

            detailed_trace_entry = {
               'src_node_name': src_node_name,
               'src_logic_unit': src_logic_unit_id,
               'dst_node_name': dst_node_name,
               'dst_logic_unit': dst_logic_unit_id,
               'exec_time': exec_time,
               'data_size': data_size,
               'comm_optype': comm_optype_str,
               'comm_time': comm_time,
               'comm_energy': comm_energy,
               'predecessors': pred_predecessors,
               'tensors_moved': tensors_moved
            }
            detailed_trace.append(detailed_trace_entry)

            # Store comm_time as an edge attribute
            DFG.graph[src_node_name][dst_node_name]['comm_time'] = comm_time
            if comm_time is None:
               logger.error(f"comm time is None for {src_node_name} -> {dst_node_name}")
            total_comm_time += comm_time #FUTURE-WORK: Pessimistic approach, see how to reduce this if it the communication to the destination nodes is parallel

            # Store comm_energy as an edge attribute
            DFG.graph[src_node_name][dst_node_name]['comm_energy'] = comm_energy / 1e9  # Convert pJ to mJ
            if comm_energy is None:
               logger.error(f"comm energy is None for {src_node_name} -> {dst_node_name}")
            total_comm_energy += comm_energy / 1e9  # Convert pJ to mJ for consistency
      
      nx.set_node_attributes(DFG.graph, {node: total_comm_time}, name="comm_time") #us
      nx.set_node_attributes(DFG.graph, {node: total_comm_energy}, name="comm_energy") #mJ (already converted)

   # Write detailed_trace to CSV file
   args = get_args()
   if "network_trace" in args.dump_traces:
      # Create traces directory if it doesn't exist
      os.makedirs("traces", exist_ok=True)
      
      with open(f"traces/detailed_trace_{filename}.csv", "w", newline='') as f:
         writer = csv.writer(f)
         # Write header
         writer.writerow([
            "src_node_name", "src_logic_unit", "dst_node_name", "dst_logic_unit",
            "exec_time", "data_size", "comm_optype", "comm_time", "comm_energy", "predecessors", "tensors_moved"
         ])
         
         # Write data rows
         for entry in detailed_trace:
            # Convert logic unit IDs to string format
            src_id = '_'.join(f'{k}_{v}' for k, v in entry['src_logic_unit'].items())
            dst_id = '_'.join(f'{k}_{v}' for k, v in entry['dst_logic_unit'].items())
            # Convert predecessors list to string; wrap in quotes so CSV viewers
            # that treat ';' as a delimiter keep this as a single field.
            raw_preds_str = ';'.join(entry['predecessors'])
            raw_tensors_str = ';'.join(ten.name for ten in entry['tensors_moved'])
            preds_str = f'"{raw_preds_str}"' if raw_preds_str else ""
            tensors_str = f'"{raw_tensors_str}"' if raw_tensors_str else ""
            
            writer.writerow([
               entry['src_node_name'],
               src_id,
               entry['dst_node_name'],
               dst_id,
               entry['exec_time'],
               entry['data_size'],
               entry['comm_optype'],
               entry['comm_time'],
               entry['comm_energy'],
               preds_str,
               tensors_str
            ])
  
   return 0
