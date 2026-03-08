from modeling.core.transformer_dfg import NodeType

def get_logic_unit_config():
    
    logic_unit_config = {
        "root": {
            "supported_ops": [NodeType.RESADD, NodeType.BROADCAST, NodeType.REDUCTION, NodeType.RMSNORM, NodeType.SILU, NodeType.EWMUL, NodeType.SHARD, NodeType.RESADD, NodeType.SPLIT, NodeType.LOOKUP, NodeType.FUSED_SCORE],
            "num_lus": 1,
        },
        "channel": {
            "supported_ops": [NodeType.RESADD, NodeType.BROADCAST, NodeType.REDUCTION, NodeType.RMSNORM, NodeType.SILU, NodeType.EWMUL, NodeType.SHARD, NodeType.RESADD, NodeType.SPLIT, NodeType.LOOKUP, NodeType.FUSED_SCORE],
            "num_lus": 1,
        },
        "wt_rank": {
            "supported_ops": [NodeType.SPLIT, NodeType.CONCAT, NodeType.SHARD, NodeType.RESADD, NodeType.BROADCAST, NodeType.REDUCTION, NodeType.APPEND, NodeType.AGG, NodeType.LOOKUP, NodeType.FUSED_SCORE],
            "num_lus": 1,
        },
        "kv_rank": {
            "supported_ops": [NodeType.SPLIT, NodeType.CONCAT, NodeType.SHARD, NodeType.RESADD, NodeType.BROADCAST, NodeType.REDUCTION, NodeType.APPEND, NodeType.AGG, NodeType.LOOKUP, NodeType.FUSED_SCORE],
            "num_lus": 1,
        },
        "chip": {
            "supported_ops": [NodeType.GEMV, NodeType.GEMM, NodeType.SOFTMAX, NodeType.RMSNORM, NodeType.SILU, NodeType.ROTARY, NodeType.SCALE, NodeType.EWMUL, NodeType.LOOKUP, NodeType.FUSED_SCORE],
            "num_lus": 1,
        },
  
    }
    return logic_unit_config 
