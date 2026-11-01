"""
Cache management utilities for model DFG caching.
"""
import os
import pickle
from utils.logging_util import logger

# Bump whenever allocation, graph building or task mapping changes, so pickles built by older code never match.
# v2: KV/weight rank allocation fixes (docs/bugfixes/kv_rank_allocation.md).
CACHE_VERSION = 2


def get_graph_flags(args):
    """
    Short tag for the args (besides those already in the filename) that change the cached graph:
    q = --fused_qkv, a = --fused_attn, s = static_mapping, g = GPU-Emb-Unemb.
    Args read only after the cache load (systolic, dump/trace/visualize) are left out.
    """
    flags = [('q', args.fused_qkv), ('a', args.fused_attn),
             ('s', 'static_mapping' in args.optimization), ('g', 'GPU-Emb-Unemb' in args.optimization)]
    return "".join(c for c, on in flags if on)


def is_picklable(obj):
    """
    Check if an object can be pickled.
    
    Args:
        obj: Object to check
    
    Returns:
        bool: True if object is picklable, False otherwise
    """
    try:
        pickle.dumps(obj)
        return True
    except Exception as e:
        logger.error(f"Object not picklable: {e}")
        return False


def get_model_dfg_cache_filename(model_name, dram, dtype, input_tokens, output_tokens, batch, 
                                  layer_optimization=False, graph_flags="", buffer=262144):
    """
    Generate cache filename for model DFG.
    
    Args:
        model_name: Name of the model
        dram: DRAM configuration name
        dtype: Data type
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        batch: Batch size
        layer_optimization: Whether layer optimization is enabled
        graph_flags: Tag from get_graph_flags(args)
        buffer: Buffer size
    
    Returns:
        str: Cache filename
    """
    dtype_str = dtype.name if hasattr(dtype, 'name') else str(dtype)
    layer_opt_suffix = "_layer_opt" if layer_optimization else "_full_layers"
    flags_suffix = f"_{graph_flags}" if graph_flags else ""
    if buffer == 262144:
        buffer_suffix = ""
    else:
        buffer_suffix = "_"+str(buffer)
    return f"graph_cache/{model_name}_{dram}_{dtype_str}_{input_tokens}i_{output_tokens}o_B{batch}{layer_opt_suffix}{buffer_suffix}{flags_suffix}_v{CACHE_VERSION}.pkl"


def save_model_dfg(model_dfg, model_name, dram, dtype, input_tokens, output_tokens, batch, 
                   layer_optimization=False, graph_flags="", buffer=262144):
    """
    Save model DFG to cache.
    
    Args:
        model_dfg: The model dataflow graph to save
        model_name: Name of the model
        dram: DRAM configuration name
        dtype: Data type
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        batch: Batch size
        layer_optimization: Whether layer optimization is enabled
        graph_flags: Tag from get_graph_flags(args)
        buffer: Buffer size
    """
    os.makedirs("graph_cache", exist_ok=True)
    filename = get_model_dfg_cache_filename(model_name, dram, dtype, input_tokens, output_tokens, 
                                            batch, layer_optimization, graph_flags, buffer)
    assert is_picklable(model_dfg), "model_dfg is not picklable! Check node/edge attributes."
    with open(filename, "wb") as f:
        pickle.dump(model_dfg, f)
    logger.info(f"model_dfg saved to {filename}")


def load_model_dfg(model_name, dram, dtype, input_tokens, output_tokens, batch, 
                   layer_optimization=False, graph_flags="", buffer=262144):
    """
    Load model DFG from cache.
    
    Args:
        model_name: Name of the model
        dram: DRAM configuration name
        dtype: Data type
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        batch: Batch size
        layer_optimization: Whether layer optimization is enabled
        graph_flags: Tag from get_graph_flags(args)
        buffer: Buffer size
    
    Returns:
        model_dfg if found, None otherwise
    """
    filename = get_model_dfg_cache_filename(model_name, dram, dtype, input_tokens, output_tokens, 
                                            batch, layer_optimization, graph_flags, buffer)
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

