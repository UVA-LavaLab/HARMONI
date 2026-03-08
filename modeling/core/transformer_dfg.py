import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import to_pydot
from collections import defaultdict, deque
from misc.type import *
from modeling.core.tensor import *
from utils.logging_util import logger
import math
import csv
from args import get_args
import copy
import pprint
class DFG: 
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_node(self, name, **attrs):
        self.graph.add_node(name, **attrs)
    
    def add_edge(self, src, dst, comm_optype=None):
        if comm_optype is None:
            self.graph.add_edge(src, dst)
        else:
            self.graph.add_edge(src, dst, comm_optype=comm_optype)

    def print_graph(self):
        print("Nodes and Edges in the DFG:")
        print("\nNodes:")
        for node in self.graph.nodes():
            print(f"  - {node}")
        
        print("\nEdges:")
        for edge in self.graph.edges():
            print(f"  - {edge[0]} -> {edge[1]}")
    
    def pretty_print_node(self, node_name):
        if node_name not in self.graph.nodes:
            print(f"Node '{node_name}' not found in the DFG.")
            return
        attrs = self.graph.nodes[node_name]
        print(f"Node: {node_name}")
        for k, v in attrs.items():
            print(f"  {k}: {v}")
        # Usage: dfg.pretty_print_node("your_node_name")

    def __repr__(self):
        lines = []
        lines.append("DFG Graph:")
        for node in self.graph.nodes:
            node_attrs = pprint.pformat(self.graph.nodes[node])
            lines.append(f"Node: {node} | Attributes: {node_attrs}")
            for succ in self.graph.successors(node):
                edge_attrs = pprint.pformat(self.graph.get_edge_data(node, succ))
                lines.append(f"  -> Edge to {succ} | Attributes: {edge_attrs}")
        return "\n".join(lines)
    
    def visualize_flowchart_pydot(dfg, filename="outputs/dfg_flowchart.png", layout="dot"):
        logger.warning("Do not analyse the model_dfg after this function due to strip_complex_attrs()")
        graph_copy = copy.deepcopy(dfg.graph)
        strip_complex_attrs(graph_copy)
        pydot_graph = to_pydot(graph_copy)
        
        # Improve node labels and layout
        for node in pydot_graph.get_nodes():
            name = node.get_name().strip('"')
            if name in dfg.graph.nodes:
                attrs = dfg.graph.nodes[name]
                label = f"{name}\n[{attrs.get('type', '')}]"
                node.set_label(label)
                node.set_shape("box")
                node.set_style("filled")
                node.set_fillcolor("lightgray")
                node.set_fontsize("10")

        # Set graph layout direction: top to bottom
        pydot_graph.set_rankdir("TB")
        pydot_graph.set_splines("polyline")

        # Render to image
        pydot_graph.write_png(filename)
        logger.info(f"Saved flowchart to: {filename}")


def add_fused_headwise_blocks(dfg, model, dram, weights, kv_cache, layer, token, bs, phase, prefix, node, ip, args=None):
    """
    Add fused headwise blocks to the DFG.
    
    Args:
        args: Command line arguments (optional, will use get_args() if not provided)
    """
    if args is None:
        args = get_args()
    context_outputs = []
    head_dim = model['head_dim']
    dtype = model['dtype']
    hdim = model['hdim']
    num_heads = model['num_heads']
    kv_heads = model['kv_heads']

    t_concat_ip = []
    if phase == PhaseType.PREFILL:
        num_tokens = token
    elif phase == PhaseType.DECODE:
        num_tokens = 1
    else:
        assert 0, "Phase for add_fused_headwise_block not defined"

    #NOTE: order of tensors in add_node is important for extracting m,n,k
    #CLEANUP: Pass datatype as argument
    t_Wqkv = weights[f"layer_{layer}"][f"Wqkv"]
    t_qkv = HarmoniTensor(name=f"t_qkv", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, (num_heads+2*kv_heads)*head_dim), addr_offset = 0)
    dfg.add_node(f"{prefix}_fqkv_gen", kernel="qkv_gen", token=token, layer=layer, phase=phase, tag=NodeTagType.FC, type=NodeType.GEMM, head=-1, ip = [ip, t_Wqkv], op = [t_qkv])
    dfg.add_edge(node, f"{prefix}_fqkv_gen")

    #CLEANUP: GQA separate for loops for Q, KV
    groups = num_heads/kv_heads
    for h in range(kv_heads):
        head_prefix = f"{prefix}_head{h}"
        for i in ['q','k','v']:
            #FUTURE-WORk: should have a way to get the addr_offset of these t_{i}_head tensors from t_qkv addr_offset using shape/size
            if i == 'q':
                t_q_head = HarmoniTensor(name=f"t_{i}_head{h}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs*groups, num_tokens, head_dim), addr_offset = 0)
            elif i == 'k':
                t_k_head = HarmoniTensor(name=f"t_{i}_head{h}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, head_dim), addr_offset = 0)
            elif i == 'v':
                t_v_head = HarmoniTensor(name=f"t_{i}_head{h}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, head_dim), addr_offset = 0)
        t_qrotary_head = HarmoniTensor(name=f"t_qrotary_head{h}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs*groups, num_tokens, head_dim), addr_offset = 0)
        t_krotary_head = HarmoniTensor(name=f"t_krotary_head{h}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, head_dim), addr_offset = 0)

        dfg.add_node(f"{head_prefix}_Qrotary", kernel = "rope", token=token, layer=layer ,phase=phase, tag=NodeTagType.MISC, type=NodeType.ROTARY, head=h, ip = [t_q_head], op = [t_qrotary_head])
        dfg.add_node(f"{head_prefix}_Krotary", kernel = "rope", token=token, layer=layer ,phase=phase, tag=NodeTagType.MISC, type=NodeType.ROTARY, head=h, ip = [t_k_head], op = [t_krotary_head])
        dfg.add_edge(f"{prefix}_fqkv_gen", f"{head_prefix}_Qrotary") 
        dfg.add_edge(f"{prefix}_fqkv_gen", f"{head_prefix}_Krotary")
        
        t_krotary_head_batch = []
        t_qrotary_head_batch = []
        t_v_head_batch = []

        for b in range(bs):
            t_krotary_head_batch.append(HarmoniTensor(name=f"t_krotary_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(num_tokens, head_dim), addr_offset = 0))
            t_qrotary_head_batch.append(HarmoniTensor(name=f"t_qrotary_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, head_dim), addr_offset = 0))
            t_v_head_batch.append(HarmoniTensor(name=f"t_v_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(num_tokens, head_dim), addr_offset = 0))
            
            #FUTURE-WORK: add a node for Kcache transposition or update stride
            t_kcache_head_batch = HarmoniTensor(name=f"t_kcache_head{h}_batch{b}", tag=HarmoniTensorType.KV, precision=dtype, shape=(token, head_dim), addr_offset = 0) # -> transpose t_kcacheT_head_batch
            
            kv_tensor = kv_cache[f"batch_{b}"][f"layer_{layer}"][f"head_{h}"]["v"]
            t_vcache_head_batch = HarmoniTensor(
                name=f"t_vcache_head{h}_batch{b}",
                tag=HarmoniTensorType.KV,
                precision=kv_tensor.precision,
                shape=(token, head_dim),  # Only change the shape
                stride=kv_tensor.stride,
                addr_offset=kv_tensor.addr_offset,
                chip_idx=kv_tensor.chip_idx,
                mapping=kv_tensor.mapping
            )
            t_vcache_head_batch.locations, t_vcache_head_batch.row_accesses, t_vcache_head_batch.col_accesses = HarmoniTensor.get_tensor_locations(t_vcache_head_batch, dram)
            
            kv_tensor = kv_cache[f"batch_{b}"][f"layer_{layer}"][f"head_{h}"]["k"]
            t_kcacheT_head_batch = HarmoniTensor(
                name=f"t_kcacheT_head{h}_batch{b}",
                tag=HarmoniTensorType.KV,
                precision=kv_tensor.precision,
                shape=(head_dim, token),  # Only change the shape
                stride=kv_tensor.stride,
                addr_offset=kv_tensor.addr_offset,
                chip_idx=kv_tensor.chip_idx,
                mapping=kv_tensor.mapping
            )         
            t_kcacheT_head_batch.locations, t_kcacheT_head_batch.row_accesses, t_kcacheT_head_batch.col_accesses = HarmoniTensor.get_tensor_locations(t_kcacheT_head_batch, dram)

            dfg.add_node(f"{head_prefix}_batch{b}_Kappend", kernel = "append", token=token, layer=layer ,phase=phase, tag=NodeTagType.MISC, type=NodeType.APPEND, head=h, ip = [t_krotary_head_batch[b]], op = [t_krotary_head_batch[b]])
            dfg.add_node(f"{head_prefix}_batch{b}_Vappend", kernel = "append", token=token, layer=layer ,phase=phase, tag=NodeTagType.MISC, type=NodeType.APPEND, head=h, ip = [t_v_head_batch[b]], op = [t_v_head_batch[b]]) 

            dfg.add_edge(f"{head_prefix}_Krotary", f"{head_prefix}_batch{b}_Kappend")
            dfg.add_edge(f"{prefix}_fqkv_gen", f"{head_prefix}_batch{b}_Vappend")
            
            if args.fused_attn:
                t_softmax_head_batch = HarmoniTensor(name=f"t_softmax_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, token), addr_offset = 0)
                t_context_head_batch = HarmoniTensor(name=f"t_context_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, head_dim), addr_offset = 0)
                dfg.add_node(f"{head_prefix}_batch{b}_fused_score", kernel = "fused_score", token=token, layer=layer, phase=phase, tag=NodeTagType.ATTN, type=NodeType.FUSED_SCORE, head=h, ip = [t_qrotary_head_batch[b]] + [t_kcacheT_head_batch], op = [t_softmax_head_batch])
                dfg.add_node(f"{head_prefix}_batch{b}_context", kernel = "context", token=token, layer=layer, phase=phase, tag=NodeTagType.ATTN, type=NodeType.GEMM, head=h, ip = [t_softmax_head_batch, t_vcache_head_batch], op = [t_context_head_batch])

                dfg.add_edge(f"{head_prefix}_Qrotary", f"{head_prefix}_batch{b}_fused_score")
                dfg.add_edge(f"{head_prefix}_batch{b}_Kappend", f"{head_prefix}_batch{b}_fused_score")
                dfg.add_edge(f"{head_prefix}_batch{b}_fused_score", f"{head_prefix}_batch{b}_context")
                dfg.add_edge(f"{head_prefix}_batch{b}_Vappend", f"{head_prefix}_batch{b}_context")
            else:
                t_score_head_batch = HarmoniTensor(name=f"t_score_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, token), addr_offset = 0)
                t_scale_head_batch = HarmoniTensor(name=f"t_scale_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, token), addr_offset = 0)
                t_softmax_head_batch = HarmoniTensor(name=f"t_softmax_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, token), addr_offset = 0)
                t_context_head_batch = HarmoniTensor(name=f"t_context_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, head_dim), addr_offset = 0)
                dfg.add_node(f"{head_prefix}_batch{b}_score", kernel = "score", token=token, layer=layer, phase=phase, tag=NodeTagType.ATTN, type=NodeType.GEMM, head=h, ip = [t_qrotary_head_batch[b]] + [t_kcacheT_head_batch], op = [t_score_head_batch])
                dfg.add_node(f"{head_prefix}_batch{b}_scale", kernel = "scale", token=token, layer=layer, phase=phase, tag=NodeTagType.ATTN, type=NodeType.SCALE, head=h, ip = [t_score_head_batch], op = [t_scale_head_batch])
                dfg.add_node(f"{head_prefix}_batch{b}_softmax", kernel = "softmax", token=token, layer=layer, phase=phase, tag=NodeTagType.ATTN, type=NodeType.SOFTMAX, head=h, ip = [t_scale_head_batch], op = [t_softmax_head_batch])
                dfg.add_node(f"{head_prefix}_batch{b}_context", kernel = "context", token=token, layer=layer, phase=phase, tag=NodeTagType.ATTN, type=NodeType.GEMM, head=h, ip = [t_softmax_head_batch, t_vcache_head_batch], op = [t_context_head_batch])

                dfg.add_edge(f"{head_prefix}_Qrotary", f"{head_prefix}_batch{b}_score")
                dfg.add_edge(f"{head_prefix}_batch{b}_Kappend", f"{head_prefix}_batch{b}_score")

                dfg.add_edge(f"{head_prefix}_batch{b}_score", f"{head_prefix}_batch{b}_scale")
                dfg.add_edge(f"{head_prefix}_batch{b}_scale", f"{head_prefix}_batch{b}_softmax")
                dfg.add_edge(f"{head_prefix}_batch{b}_softmax", f"{head_prefix}_batch{b}_context")
                dfg.add_edge(f"{head_prefix}_batch{b}_Vappend", f"{head_prefix}_batch{b}_context")

            context_outputs.append(f"{head_prefix}_batch{b}_context")   
            t_concat_ip.append(t_context_head_batch)

    t_concat = HarmoniTensor(name=f"t_concat", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, hdim), addr_offset = 0)
    concat_node = f"{prefix}_concat"
    dfg.add_node(concat_node, kernel = "concat", token=token, layer=layer ,phase=phase, tag=NodeTagType.ATTN, type=NodeType.CONCAT, head=-1, ip = t_concat_ip, op = [t_concat])
    for ctx in context_outputs:
        dfg.add_edge(ctx, concat_node)

    return t_concat 

def add_headwise_blocks(dfg, model, dram, weights, kv_cache, layer, token, bs, phase, prefix, node, ip):
    context_outputs = []
    head_dim = model['head_dim']
    dtype = model['dtype']
    hdim = model['hdim']
    num_heads = model['num_heads']
    kv_heads = model['kv_heads']

    groups = num_heads/kv_heads

    t_concat_ip = []
    if phase == PhaseType.PREFILL:
        num_tokens = token
    elif phase == PhaseType.DECODE:
        num_tokens = 1
    else:
        assert 0, "Phase for add_headwise_block not defined"

    #NOTE: order of tensors in add_node is important for extracting m,n,k
    for h in range(kv_heads):
        head_prefix = f"{prefix}_head{h}"
        for i in ['q','k','v']:
            t_Wqkv_head = weights[f"layer_{layer}"][f"W{i}"][h] 
            if i == 'q':
                t_q_head = HarmoniTensor(name=f"t_{i}_head{h}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs*groups, num_tokens, head_dim), addr_offset = 0)
                dfg.add_node(f"{head_prefix}_{i}_gen", kernel="qkv_gen", token=token, layer=layer, phase=phase, tag=NodeTagType.FC, type=NodeType.GEMM, head=h, ip = [ip, t_Wqkv_head], op = [t_q_head])
            elif i == 'k':
                t_k_head = HarmoniTensor(name=f"t_{i}_head{h}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, head_dim), addr_offset = 0)
                dfg.add_node(f"{head_prefix}_{i}_gen", kernel="qkv_gen", token=token, layer=layer, phase=phase, tag=NodeTagType.FC, type=NodeType.GEMM, head=h, ip = [ip, t_Wqkv_head], op = [t_k_head])
            elif i == 'v':
                t_v_head = HarmoniTensor(name=f"t_{i}_head{h}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, head_dim), addr_offset = 0)
                dfg.add_node(f"{head_prefix}_{i}_gen", kernel="qkv_gen", token=token, layer=layer, phase=phase, tag=NodeTagType.FC, type=NodeType.GEMM, head=h, ip = [ip, t_Wqkv_head], op = [t_v_head])
            dfg.add_edge(node, f"{head_prefix}_{i}_gen")
        t_qrotary_head = HarmoniTensor(name=f"t_qrotary_head{h}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs*groups, num_tokens, head_dim), addr_offset = 0)
        t_krotary_head = HarmoniTensor(name=f"t_krotary_head{h}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, head_dim), addr_offset = 0)

        dfg.add_node(f"{head_prefix}_Qrotary", kernel = "rope", token=token, layer=layer ,phase=phase, tag=NodeTagType.MISC, type=NodeType.ROTARY, head=h, ip = [t_q_head], op = [t_qrotary_head])
        dfg.add_node(f"{head_prefix}_Krotary", kernel = "rope", token=token, layer=layer ,phase=phase, tag=NodeTagType.MISC, type=NodeType.ROTARY, head=h, ip = [t_k_head], op = [t_krotary_head])
        dfg.add_edge(f"{head_prefix}_q_gen", f"{head_prefix}_Qrotary") 
        dfg.add_edge(f"{head_prefix}_k_gen", f"{head_prefix}_Krotary")
        
        t_krotary_head_batch = []
        t_qrotary_head_batch = []
        t_v_head_batch = []

        for b in range(bs):
            t_krotary_head_batch.append(HarmoniTensor(name=f"t_krotary_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(num_tokens, head_dim), addr_offset = 0))
            t_qrotary_head_batch.append(HarmoniTensor(name=f"t_qrotary_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, head_dim), addr_offset = 0))
            t_v_head_batch.append(HarmoniTensor(name=f"t_v_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(num_tokens, head_dim), addr_offset = 0))
        
        dfg.add_node(f"{head_prefix}_batch_shard", kernel = "shard", token=token, layer=layer, phase=phase, tag=NodeTagType.MISC, type=NodeType.SHARD, head=h, ip = [t_krotary_head, t_qrotary_head, t_v_head], op = t_krotary_head_batch + t_qrotary_head_batch + t_v_head_batch)
        dfg.add_edge(f"{head_prefix}_Qrotary", f"{head_prefix}_batch_shard")
        dfg.add_edge(f"{head_prefix}_Krotary", f"{head_prefix}_batch_shard")
        dfg.add_edge(f"{head_prefix}_v_gen", f"{head_prefix}_batch_shard")
        
        for b in range(bs):
            
            #FUTURE-WORK: add a node for Kcache transposition or update stride
            t_kcache_head_batch = HarmoniTensor(name=f"t_kcache_head{h}_batch{b}", tag=HarmoniTensorType.KV, precision=dtype, shape=(token, head_dim), addr_offset = 0) # -> transpose t_kcacheT_head_batch
            
            kv_tensor = kv_cache[f"batch_{b}"][f"layer_{layer}"][f"head_{h}"]["v"]
            t_vcache_head_batch = HarmoniTensor(
                name=f"t_vcache_head{h}_batch{b}",
                tag=HarmoniTensorType.KV,
                precision=kv_tensor.precision,
                shape=(token, head_dim),  # Only change the shape
                stride=kv_tensor.stride,
                addr_offset=kv_tensor.addr_offset,
                chip_idx=kv_tensor.chip_idx,
                mapping=kv_tensor.mapping
            )
            t_vcache_head_batch.locations, t_vcache_head_batch.row_accesses, t_vcache_head_batch.col_accesses = HarmoniTensor.get_tensor_locations(t_vcache_head_batch, dram)
            
            kv_tensor = kv_cache[f"batch_{b}"][f"layer_{layer}"][f"head_{h}"]["k"]
            t_kcacheT_head_batch = HarmoniTensor(
                name=f"t_kcacheT_head{h}_batch{b}",
                tag=HarmoniTensorType.KV,
                precision=kv_tensor.precision,
                shape=(head_dim, token),  # Only change the shape
                stride=kv_tensor.stride,
                addr_offset=kv_tensor.addr_offset,
                chip_idx=kv_tensor.chip_idx,
                mapping=kv_tensor.mapping
            )         
            t_kcacheT_head_batch.locations, t_kcacheT_head_batch.row_accesses, t_kcacheT_head_batch.col_accesses = HarmoniTensor.get_tensor_locations(t_kcacheT_head_batch, dram)

            dfg.add_node(f"{head_prefix}_batch{b}_Kappend", kernel = "append", token=token, layer=layer ,phase=phase, tag=NodeTagType.MISC, type=NodeType.APPEND, head=h, ip = [t_krotary_head_batch[b]], op = [t_krotary_head_batch[b]])
            dfg.add_node(f"{head_prefix}_batch{b}_Vappend", kernel = "append", token=token, layer=layer ,phase=phase, tag=NodeTagType.MISC, type=NodeType.APPEND, head=h, ip = [t_v_head_batch[b]], op = [t_v_head_batch[b]]) 

            t_score_head_batch = HarmoniTensor(name=f"t_score_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, token), addr_offset = 0)
            t_scale_head_batch = HarmoniTensor(name=f"t_scale_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, token), addr_offset = 0)
            t_softmax_head_batch = HarmoniTensor(name=f"t_softmax_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, token), addr_offset = 0)
            t_context_head_batch = HarmoniTensor(name=f"t_context_head{h}_batch{b}", tag=HarmoniTensorType.ACT, precision=dtype, shape=(groups*num_tokens, head_dim), addr_offset = 0)
            dfg.add_node(f"{head_prefix}_batch{b}_score", kernel = "score", token=token, layer=layer, phase=phase, tag=NodeTagType.ATTN, type=NodeType.GEMM, head=h, ip = [t_qrotary_head_batch[b]] + [t_kcacheT_head_batch], op = [t_score_head_batch])
            dfg.add_node(f"{head_prefix}_batch{b}_scale", kernel = "scale", token=token, layer=layer, phase=phase, tag=NodeTagType.ATTN, type=NodeType.SCALE, head=h, ip = [t_score_head_batch], op = [t_scale_head_batch])
            dfg.add_node(f"{head_prefix}_batch{b}_softmax", kernel = "softmax", token=token, layer=layer, phase=phase, tag=NodeTagType.ATTN, type=NodeType.SOFTMAX, head=h, ip = [t_scale_head_batch], op = [t_softmax_head_batch])
            dfg.add_node(f"{head_prefix}_batch{b}_context", kernel = "context", token=token, layer=layer, phase=phase, tag=NodeTagType.ATTN, type=NodeType.GEMM, head=h, ip = [t_softmax_head_batch, t_vcache_head_batch], op = [t_context_head_batch])

            dfg.add_edge(f"{head_prefix}_batch_shard", f"{head_prefix}_batch{b}_Kappend")
            dfg.add_edge(f"{head_prefix}_batch_shard", f"{head_prefix}_batch{b}_Vappend") 
            dfg.add_edge(f"{head_prefix}_batch_shard", f"{head_prefix}_batch{b}_score") #due to Q
            dfg.add_edge(f"{head_prefix}_batch{b}_Kappend", f"{head_prefix}_batch{b}_score")

            dfg.add_edge(f"{head_prefix}_batch{b}_score", f"{head_prefix}_batch{b}_scale")
            dfg.add_edge(f"{head_prefix}_batch{b}_scale", f"{head_prefix}_batch{b}_softmax")
            dfg.add_edge(f"{head_prefix}_batch{b}_softmax", f"{head_prefix}_batch{b}_context")
            dfg.add_edge(f"{head_prefix}_batch{b}_Vappend", f"{head_prefix}_batch{b}_context")

            context_outputs.append(f"{head_prefix}_batch{b}_context")   
            t_concat_ip.append(t_context_head_batch)

    t_concat = HarmoniTensor(name=f"t_concat", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, hdim), addr_offset = 0)
    concat_node = f"{prefix}_concat"
    dfg.add_node(concat_node, kernel = "concat", token=token, layer=layer ,phase=phase, tag=NodeTagType.ATTN, type=NodeType.CONCAT, head=-1, ip = t_concat_ip, op = [t_concat])
    for ctx in context_outputs:
        dfg.add_edge(ctx, concat_node)

    return t_concat 


def build_decoder_layer(dfg, model, dram, weights, kv_cache, layer, seq_len, bs, layer_id, prev, ip, op, args=None):
    """
    Build a decoder layer in the DFG.
    
    Args:
        args: Command line arguments (optional, will use get_args() if not provided)
    """
    if args is None:
        args = get_args()
    prefix = layer_id
    act_fn = model['act_fn']
    intmdt_dim = model['intmdt_size']
    dtype = model['dtype']
    hdim = model['hdim']
    
    if ("prefill" in layer_id):
        phase = PhaseType.PREFILL
        num_tokens = seq_len
    elif ("decode" in layer_id):
        phase = PhaseType.DECODE
        num_tokens = 1
    else:
        assert 0, "Phase for build_decoder_layer not defined"

    t_lnorm1 = HarmoniTensor(name="t_norm1", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, hdim), addr_offset = 0) 
    
    dfg.add_node(f"{prefix}_lnorm1", kernel = "norm", token=seq_len, layer=layer ,phase=phase, tag=NodeTagType.MISC, type=NodeType.RMSNORM, head=-1, ip = ip, op = [t_lnorm1]) 

    if args.headsplit: #Note: Only implemented for LLAMA
        t_head_emb = []
        t = HarmoniTensor(name=f"t_emb", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, hdim), addr_offset = 0) 
        if args.fused_qkv:
            t_concat = add_fused_headwise_blocks(dfg, model, dram, weights, kv_cache, layer, seq_len, bs, phase, prefix, f"{prefix}_lnorm1", t, args)
        else:
            t_concat = add_headwise_blocks(dfg, model, dram, weights, kv_cache, layer, seq_len, bs, phase, prefix, f"{prefix}_lnorm1", t)
    
    dfg.add_edge(f"{prefix}_concat", f"{prefix}_output_proj")

    t_oproj = HarmoniTensor(name="t_oproj", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, hdim), addr_offset = 0)
    t_Wo = weights[f"layer_{layer}"]["Wo"]
    t_attnadd = HarmoniTensor(name="t_attnadd", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, hdim), addr_offset = 0) 
    dfg.add_node(f"{prefix}_output_proj", kernel = "out_proj", layer=layer, token=seq_len, phase=phase, tag=NodeTagType.ATTN, type=NodeType.GEMM, head=-1, ip = [t_concat, t_Wo], op = [t_oproj])
    dfg.add_node(f"{prefix}_attn_add", kernel = "res_add", layer=layer, token=seq_len, phase=phase, tag=NodeTagType.ATTN, type=NodeType.RESADD, head=-1, ip = ip + [t_oproj], op = [t_attnadd])
    dfg.add_edge(f"{prefix}_output_proj", f"{prefix}_attn_add")
    dfg.add_edge(f"{prev}", f"{prefix}_attn_add")
    
    t_lnorm2 = HarmoniTensor(name="t_norm2", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, hdim), addr_offset = 0)
    
    dfg.add_node(f"{prefix}_lnorm2", kernel = "norm", layer=layer, token=seq_len, phase=phase, tag=NodeTagType.MISC, type=NodeType.RMSNORM, head=-1, ip = [t_attnadd], op = [t_lnorm2])

    if 'LLAMA' in model['name'] or 'MISTRAL' in model['name']: #NOTE: currently only chekcing GQA, so everything else is same
        t_Wup = weights[f"layer_{layer}"]["Wup"] 
        t_up = HarmoniTensor(name="t_up", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, intmdt_dim), addr_offset = 0)
        t_Wgate = weights[f"layer_{layer}"]["Wgate"] 
        t_gate = HarmoniTensor(name="t_gate", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, intmdt_dim), addr_offset = 0)
        t_gate_silu = HarmoniTensor(name="t_gate_silu", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, intmdt_dim), addr_offset = 0) #CLEANUP: do f"t_gate_{act_fn}"?
        t_upgate = HarmoniTensor(name="t_upgate", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, intmdt_dim), addr_offset = 0)
        t_Wdown = weights[f"layer_{layer}"]["Wdown"] 
        t_down = HarmoniTensor(name="t_down", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, hdim), addr_offset = 0) 
        t_mlpadd = HarmoniTensor(name="t_mlpadd", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, num_tokens, hdim), addr_offset = 0) 
        dfg.add_node(f"{prefix}_up_gemm", kernel = "up_gemm", layer=layer, token=seq_len, phase=phase, tag=NodeTagType.FC, type=NodeType.GEMM, head=-1, ip = [t_lnorm2, t_Wup], op = [t_up])
        dfg.add_node(f"{prefix}_gate_gemm", kernel = "gate_gemm", layer=layer, token=seq_len, phase=phase, tag=NodeTagType.FC, type=NodeType.GEMM, head=-1, ip = [t_lnorm2, t_Wgate], op = [t_gate])
        dfg.add_node(f"{prefix}_{act_fn}", kernel = "act_fn", layer=layer, token=seq_len, phase=phase, tag=NodeTagType.FC, type=NodeType.SILU, head=-1, ip = [t_gate], op = [t_gate_silu])
        dfg.add_node(f"{prefix}_upgate_EWmul", kernel = "ew_mul", layer=layer, token=seq_len, phase=phase, tag=NodeTagType.FC, type=NodeType.EWMUL, head=-1, ip = [t_gate_silu, t_up], op = [t_upgate] )
        dfg.add_node(f"{prefix}_down_gemm", kernel = "down_gemm", layer=layer, token=seq_len, phase=phase, tag=NodeTagType.FC, type=NodeType.GEMM, head=-1, ip = [t_upgate, t_Wdown], op = [t_down])
        dfg.add_node(f"{prefix}_mlp_add", kernel = "res_add", layer=layer, token=seq_len, phase=phase, tag=NodeTagType.FC, type=NodeType.RESADD, head=-1, ip = ip + [t_down], op = [t_mlpadd])
        op = [t_mlpadd]

        dfg.add_edge(f"{prefix}_attn_add", f"{prefix}_lnorm2")
        dfg.add_edge(f"{prefix}_lnorm2", f"{prefix}_up_gemm")
        dfg.add_edge(f"{prefix}_lnorm2", f"{prefix}_gate_gemm")
        dfg.add_edge(f"{prefix}_gate_gemm", f"{prefix}_{act_fn}")
        dfg.add_edge(f"{prefix}_{act_fn}", f"{prefix}_upgate_EWmul")
        dfg.add_edge(f"{prefix}_up_gemm", f"{prefix}_upgate_EWmul")
        dfg.add_edge(f"{prefix}_upgate_EWmul", f"{prefix}_down_gemm")
        dfg.add_edge(f"{prefix}_down_gemm", f"{prefix}_mlp_add")
        dfg.add_edge(f"{prefix}_attn_add", f"{prefix}_mlp_add")
    
    return op


# CLEANUP: call different build_decoder_layer for different models
def build_model_dfg(model, lin, lout, bs, weights, kv_cache, dram, temp_tensor_wt=None, temp_tensor_kv=None, args=None):
    """
    Build the model dataflow graph.
    
    Args:
        args: Command line arguments (optional, will use get_args() if not provided)
    """
    if args is None:
        args = get_args()
    logger.info("Building Model DFG")
    if 'layer' in args.optimization:
        logger.info("layer optimization applied")
        num_layers = 1
    else:
        num_layers = model["nlayers"]
    logger.debug(f"nlayers is {num_layers}")
    
    dtype = model['dtype']
    hdim = model["hdim"]
    vocab = model['vocab']

    dfg = DFG()
    #CLEANUP: make the following a part of dram_info
    s = dram.addr_interleaving
    field_bits = {
        'Bt': int(math.ceil(math.log(dram.burst_length, 2))),
        'Co': int(math.ceil(math.log(dram.csl_lines, 2))),
        'Ro': int(math.ceil(math.log(dram.rows_per_bank, 2))),
        'Ba': int(math.ceil(math.log(dram.num_banks_per_bankgroup, 2))),
        'Bg': int(math.ceil(math.log(dram.num_bankgroups_per_chip, 2))),
        'Ra': int(math.ceil(math.log(dram.num_ranks_per_channel, 2))),
        'Ch': int(math.ceil(math.log(dram.num_channels, 2))),
    }
    dram_addr_bits = 0
    for j in range(0, len(s)-1,2):
        dram_addr_bits += field_bits[f"{s[j:j+2]}"]
    #######
    
    t_Wemb = weights["W_embed"]
    t_Wunemb = weights["W_unembed"] 
    t_token = HarmoniTensor(name="t_token", tag=HarmoniTensorType.INPUT, precision=dtype, shape=(bs, lin), addr_offset = 0)
    t_emb = HarmoniTensor(name="t_emb", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, lin, hdim), addr_offset = 0)
    dfg.add_node("Embedding", kernel="embed", layer=-1, token=lin, phase=PhaseType.PREFILL, tag=NodeTagType.EMBED, type=NodeType.LOOKUP, head=-1, ip=[t_token,t_Wemb], op=[t_emb]) 

    # Prefill phase
    prev = "Embedding"
    ip = [t_emb]
    op = []
    for l in range(num_layers):
        ip = build_decoder_layer(dfg, model, dram, weights, kv_cache, l, lin, bs, f"prefill_L{l}", prev, ip, op, args)
        dfg.add_edge(prev, f"prefill_L{l}_lnorm1")
        prev = f"prefill_L{l}_mlp_add"
    
    #CLEANUP: create a modular LMhead function
    t_logits = HarmoniTensor(name="t_logits", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, 1, vocab), addr_offset = 0)
    #argmax = select one tokenid
    t_token_id = HarmoniTensor(name="t_token_id", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, 1), addr_offset = 0) 
    #get the embedding from embedding table
    t_emb_pre = HarmoniTensor(name="t_logits", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, 1, hdim), addr_offset = 0)
    dfg.add_node(f"prefill_LMHead_A", kernel="lm_head_a", layer=-1, token=lin, phase=PhaseType.PREFILL, tag=NodeTagType.EMBED, type=NodeType.GEMM, head=-1, ip=ip + [t_Wunemb], op=[t_logits])
    dfg.add_edge(prev, f"prefill_LMHead_A")
    dfg.add_node(f"prefill_LMHead_B", kernel="lm_head_b", layer=-1, token=lin, phase=PhaseType.PREFILL, tag=NodeTagType.EMBED, type=NodeType.ARGMAX, head=-1, ip=[t_logits], op=[t_token_id])
    dfg.add_edge(f"prefill_LMHead_A", f"prefill_LMHead_B")
    dfg.add_node(f"prefill_LMHead", kernel="lm_head", layer=-1, token=lin, phase=PhaseType.PREFILL, tag=NodeTagType.EMBED, type=NodeType.LOOKUP, head=-1, ip=[t_token_id], op=[t_emb_pre])
    t_lmhead = t_emb_pre
    dfg.add_edge(f"prefill_LMHead_B", f"prefill_LMHead")


    # Decode phase (looped)
    for t in range(1, lout):
        if t == 1:
            prev = "prefill_LMHead"
            ip = [t_lmhead]
        else:
            prev = f"decode_LMHead_T{t-1}"
   
        for l in range(num_layers):
            ip = build_decoder_layer(dfg, model, dram, weights, kv_cache, l, lin+t, bs, f"decode_T{t}_L{l}", prev, ip, op, args)
            dfg.add_edge(prev, f"decode_T{t}_L{l}_lnorm1")
            prev = f"decode_T{t}_L{l}_mlp_add"
        t_logits = HarmoniTensor(name="t_logits", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, 1, vocab), addr_offset = 0)
        #argmax = select one tokenid
        t_token_id = HarmoniTensor(name="t_token_id", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, 1), addr_offset = 0) 
        #get the embedding from embedding table
        t_emb_dec = HarmoniTensor(name="t_logits", tag=HarmoniTensorType.ACT, precision=dtype, shape=(bs, 1, hdim), addr_offset = 0)
        dfg.add_node(f"decode_LMHead_A_T{t}", kernel="lm_head_a", layer=-1, token=lin+t, phase=PhaseType.DECODE, tag=NodeTagType.EMBED, type=NodeType.GEMM, head=-1, ip=ip + [t_Wunemb], op=[t_logits])
        dfg.add_edge(prev, f"decode_LMHead_A_T{t}")
        dfg.add_node(f"decode_LMHead_B_T{t}", kernel="lm_head_b", layer=-1, token=lin+t, phase=PhaseType.DECODE, tag=NodeTagType.EMBED, type=NodeType.ARGMAX, head=-1, ip=[t_logits], op=[t_token_id])
        dfg.add_edge(f"decode_LMHead_A_T{t}", f"decode_LMHead_B_T{t}")
        dfg.add_node(f"decode_LMHead_T{t}", kernel="lm_head", layer=-1, token=lin+t, phase=PhaseType.DECODE, tag=NodeTagType.EMBED, type=NodeType.LOOKUP, head=-1, ip=[t_token_id], op=[t_emb_dec])
        ip = [t_emb_dec]
        dfg.add_edge(f"decode_LMHead_B_T{t}", f"decode_LMHead_T{t}")

    return dfg

def get_parallel_execution_levels(dfg):
    g = dfg.graph
    in_degree = {node: g.in_degree(node) for node in g.nodes()}
    ready_queue = deque()
    level_map = defaultdict(list)
    node_level = dict()

    # Find all nodes with no dependencies
    for node, deg in in_degree.items():
        if deg == 0:
            ready_queue.append((node, 0))
            node_level[node] = 0

    # Traverse the graph in level order
    while ready_queue:
        node, level = ready_queue.popleft()
        level_map[level].append(node)

        for succ in g.successors(node):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                ready_queue.append((succ, level + 1))
                node_level[succ] = level + 1

    return level_map

def print_level_map(level_map):
    print("Nodes grouped by levels:")
    for level, nodes in sorted(level_map.items()):
        print(f"Level {level}: {', '.join(nodes)}")

def strip_complex_attrs(graph):
    for node, attrs in list(graph.nodes(data=True)):
        keys_to_remove = []
        for k, v in attrs.items():
            # Only keep str, int, float, or bool
            if not isinstance(v, (str, int, float, bool)):
                keys_to_remove.append(k)
            # Remove if string contains a colon
            elif isinstance(v, str) and ':' in v:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del attrs[k]
    for u, v, attrs in list(graph.edges(data=True)):
        keys_to_remove = []
        for k, v in attrs.items():
            if not isinstance(v, (str, int, float, bool)):
                keys_to_remove.append(k)
            elif isinstance(v, str) and ':' in v:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del attrs[k]
