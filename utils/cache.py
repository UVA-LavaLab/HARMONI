"""
Cache management utilities for model DFG caching.
"""
import os
import pickle
from utils.logging_util import logger


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
                                  layer_optimization=False, fusion_enabled=True, buffer=262144):
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
        fusion_enabled: Whether fusion is enabled
        buffer: Buffer size
    
    Returns:
        str: Cache filename
    """
    dtype_str = dtype.name if hasattr(dtype, 'name') else str(dtype)
    layer_opt_suffix = "_layer_opt" if layer_optimization else "_full_layers"
    fusion_suffix = "" if fusion_enabled else "no_fuse"
    if buffer == 262144:
        buffer_suffix = ""
    else:
        buffer_suffix = "_"+str(buffer)
    return f"graph_cache/{model_name}_{dram}_{dtype_str}_{input_tokens}i_{output_tokens}o_B{batch}{layer_opt_suffix}{buffer_suffix}.pkl"


def save_model_dfg(model_dfg, model_name, dram, dtype, input_tokens, output_tokens, batch, 
                   layer_optimization=False, fusion_enabled=True, buffer=262144):
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
        fusion_enabled: Whether fusion is enabled
        buffer: Buffer size
    """
    os.makedirs("graph_cache", exist_ok=True)
    filename = get_model_dfg_cache_filename(model_name, dram, dtype, input_tokens, output_tokens, 
                                            batch, layer_optimization, fusion_enabled, buffer)
    assert is_picklable(model_dfg), "model_dfg is not picklable! Check node/edge attributes."
    with open(filename, "wb") as f:
        pickle.dump(model_dfg, f)
    logger.info(f"model_dfg saved to {filename}")


def load_model_dfg(model_name, dram, dtype, input_tokens, output_tokens, batch, 
                   layer_optimization=False, fusion_enabled=True, buffer=262144):
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
        fusion_enabled: Whether fusion is enabled
        buffer: Buffer size
    
    Returns:
        model_dfg if found, None otherwise
    """
    filename = get_model_dfg_cache_filename(model_name, dram, dtype, input_tokens, output_tokens, 
                                            batch, layer_optimization, fusion_enabled, buffer)
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

