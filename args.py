import argparse
from misc.type import DataType

# Global singleton to store parsed args
_args = None

def parse_args():
    """
    Parse command line arguments. This function will only parse once and cache the result.
    For getting already-parsed args, use get_args() instead.
    """
    global _args
    if _args is not None:
        return _args
    
    parser = argparse.ArgumentParser(description='Generate model configuration.')
    parser.add_argument('--model_name', type=str, required=False, 
                       help='Name of the model',
                       default = 'LLAMA2-7B')
    parser.add_argument('--dtype', type=DataType, required=False, 
                       choices=list(DataType), 
                       help='Data type of the model',
                       default = DataType.BF16)
    parser.add_argument("-b", "--batch", type=int, 
                       help="batch size to group requests", 
                       default=1)
    parser.add_argument("-i", "--input-tokens", type=int, 
                       help="number of tokens in input prompt", 
                       default=32)
    parser.add_argument("-o", "--output-tokens", type=int, 
                       help="number of output tokens", 
                       default=32)
    parser.add_argument('--dram', type=str, required=True,  
                       help='DRAM config')
    parser.add_argument('--fused_qkv', action='store_true',  
                       help='Fused QKV weights and QKV generation')
    parser.add_argument('--fused_attn', action='store_true',  
                       help='Fused attention (score, scale, softmax, context) with parallel reductions')
    parser.add_argument('--headsplit', action='store_true',  
                       help='Head-wise split in model_dfg',
                       default=True)
    parser.add_argument('--no_systolic_array', action='store_true',  
                       help='Performe all the GEMMs without systolic array',
                       default=False)
    parser.add_argument('--systolic_height', type=int,  
                       help='Systolic array height (this is the max batch size supported)',
                       default=8)
    parser.add_argument('--buffer', type=int,  
                       help='SRAM buffer size per center stripe',
                       default=262144)
    parser.add_argument('--simulate', action='store_true', 
                       help='simulate using new modeling style')
    parser.add_argument('--optimization', type=str, 
                       nargs='+', 
                       required=False, 
                       default=[], 
                       choices=['','layer', 'GPU-Emb-Unemb', 'static_mapping'],
                       help='List of optimizations to add (e.g. --optimization layer caching)')
    parser.add_argument('--dump_stats', type=str,
                        nargs='+', 
                       required=False, 
                       default=[], 
                       choices=['', 'logic_unit_ops_dist', 'logic_unit_analysis', 'task_profile', 'task_mapping', 'resource_energy_breakdown', 'html_dfg'], 
                       help='enables stat dumps for the listed categories')
    parser.add_argument('--dump_traces', type=str,
                       nargs='+',
                       required=False,
                       default=[],
                       choices=['gemm_trace', 'network_trace'],
                       help='Enable trace dumping for the listed categories (e.g. --dump_traces gemm_trace network_trace)')
    parser.add_argument('--profile', action='store_true', 
                       help='enables cProfile and print performance stats')
    parser.add_argument('--timing', action='store_true', 
                       help='enables StageTimer and dump_timing_csv')
    parser.add_argument('--verbose', action='store_true', 
                       help='enables all sort of print statements')
    parser.add_argument('--visualize', action='store_true', 
                       help='enables dfg visualization')
    parser.add_argument('--no_cache', action='store_true',
                       help='Force regeneration of model_dfg and ignore cache')

    _args = parser.parse_args()
    return _args

def get_args():
    """
    Get the parsed command line arguments. 
    If not yet parsed, will parse them first (singleton pattern).
    Use this function instead of parse_args() when you just need to access args.
    """
    global _args
    if _args is None:
        return parse_args()
    return _args 
