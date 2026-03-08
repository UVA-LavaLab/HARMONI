import math
from utils.logging_util import logger
from functools import lru_cache

# Use LRU cache with a reasonable size limit for better performance
@lru_cache(maxsize=1024)
def get_mapping_cached(addr_offset, addr_interleaving, rank_type, burst_length, csl_lines, 
                      rows_per_bank, num_banks_per_bankgroup, num_bankgroups_per_chip, 
                      num_ranks_per_channel, num_channels):
    """
    Cached version of get_mapping that takes individual parameters instead of dram object.
    This allows for better cache hit rates.
    """
    # Define bit lengths for each component
    bit_lengths = {
        'Bt': int(math.ceil(math.log(burst_length, 2))),
        'Co': int(math.ceil(math.log(csl_lines, 2))),
        'Ro': int(math.ceil(math.log(rows_per_bank, 2))),
        'Ba': int(math.ceil(math.log(num_banks_per_bankgroup, 2))),
        'Bg': int(math.ceil(math.log(num_bankgroups_per_chip, 2))),
        'Ra': int(math.ceil(math.log(num_ranks_per_channel, 2))),
        'Ch': int(math.ceil(math.log(num_channels, 2))),
    }
    
    # Extract components based on the addr_interleaving scheme
    mapping = {}
    for component in reversed([addr_interleaving[i:i+2] for i in range(0, len(addr_interleaving), 2)]):
        if component not in bit_lengths:
            raise ValueError(f"Unknown component '{component}' in addr_interleaving")
        bits = bit_lengths[component]
        value = addr_offset & (2**bits - 1)
        addr_offset >>= bits
        mapping[component] = value
    
    # Return a copy to prevent sharing between different calls
    return dict(mapping)

def get_mapping(addr_offset, dram, rank_type): #CLEANUP: remove rank_type not getting used
    """
    Returns the address based on address interleaving scheme.
    Uses cached version for better performance.
    """
    return get_mapping_cached(
        int(addr_offset), 
        dram.addr_interleaving, 
        rank_type, 
        dram.burst_length, 
        dram.csl_lines, 
        dram.rows_per_bank, 
        dram.num_banks_per_bankgroup, 
        dram.num_bankgroups_per_chip, 
        dram.num_ranks_per_channel, 
        dram.num_channels
    )

def get_accesses(requests, dram, wmeta):
    """
    Returns the number of row/col accesses made across all chips in all bank mode in parallel. 
    """ 
    #CLEANUP: Update for All bank mode (check why this was added)
    wmeta.col_accesses = requests
    wmeta.row_accesses = math.ceil(requests/dram.csl_lines)

def get_rank_offset(dram):
    """
    Returns rank_offset based on the dram organizations bits value and the interleaving scheme
    """
    s = dram.addr_interleaving
    # Define bit lengths for each component
    
    bit_lengths = {
        'Bt': int(math.ceil(math.log(dram.burst_length, 2))),
        'Co': int(math.ceil(math.log(dram.csl_lines, 2))),
        'Ro': int(math.ceil(math.log(dram.rows_per_bank, 2))),
        'Ba': int(math.ceil(math.log(dram.num_banks_per_bankgroup, 2))),
        'Bg': int(math.ceil(math.log(dram.num_bankgroups_per_chip, 2))),
        'Ra': int(math.ceil(math.log(dram.num_ranks_per_channel, 2))),
        'Ch': int(math.ceil(math.log(dram.num_channels, 2))),
    }

    shift_bits = 0
    for component in reversed([dram.addr_interleaving[i:i+2] for i in range(0, len(dram.addr_interleaving), 2)]):
        if component == 'Ra':
            break
        else:
            shift_bits += bit_lengths[component]
    logger.debug(f"Rank offset calculation shift bits {shift_bits}, rank offset {1 << shift_bits}")
    return 1 << shift_bits

@lru_cache(maxsize=512)
def calculate_interleaving_aware_offset_cached(request_number, dram, rank_type, chip_idx):
    """
    Cached version of calculate_interleaving_aware_offset that takes dram object and rank_type.
    Handles all bank mode and special rank assignment.
    """
    # Define bit lengths for each component
    bit_lengths = {
        'Bt': int(math.ceil(math.log(dram.burst_length, 2))),
        'Co': int(math.ceil(math.log(dram.csl_lines, 2))),
        'Ro': int(math.ceil(math.log(dram.rows_per_bank, 2))),
        'Ba': int(math.ceil(math.log(dram.num_banks_per_bankgroup, 2))),
        'Bg': int(math.ceil(math.log(dram.num_bankgroups_per_chip, 2))),
        'Ra': int(math.ceil(math.log(dram.num_ranks_per_channel, 2))),
        'Ch': int(math.ceil(math.log(dram.num_channels, 2))),
    }
    
    all_bank_mode = dram.mode == "AB"
    interleaving_components = [dram.addr_interleaving[i:i+2] for i in range(0, len(dram.addr_interleaving), 2)]
    components = {}
    remaining_request = request_number

    for component in reversed(interleaving_components):
        if all_bank_mode and component in ['Ba', 'Bg', 'Bt']:
            components[component] = 0
        elif component == 'Ra':
            if rank_type == "weight":
                rank_value = remaining_request % dram.num_wt_ranks_per_channel
                components[component] = rank_value
                remaining_request //= dram.num_wt_ranks_per_channel
            elif rank_type == "kv":
                rank_value = dram.num_wt_ranks_per_channel + (remaining_request % dram.num_kv_ranks_per_channel)
                components[component] = rank_value
                remaining_request //= dram.num_kv_ranks_per_channel
            else:
                raise ValueError("Unknown rank_type")
        else:
            bits = bit_lengths[component]
            max_value = 2**bits - 1
            components[component] = remaining_request & max_value
            remaining_request >>= bits
    
    
    # Reconstruct address from components
    addr_offset = 0
    for i, component in enumerate(interleaving_components):
        shift_bits = 0
        for j in range(i + 1, len(interleaving_components)):
            shift_bits += bit_lengths[interleaving_components[j]]
        addr_offset += components[component] << shift_bits

    return addr_offset

def calculate_interleaving_aware_offset(request_number, dram, rank_type, chip_idx=-1):
    """
    Calculate address offset that respects the memory interleaving pattern.
    This ensures weight tensors never map to KV ranks and vice versa.
    Now passes dram object to the cached function.
    """
    return calculate_interleaving_aware_offset_cached(
        request_number, 
        dram, 
        rank_type, 
        chip_idx
    )

def reconstruct_address_from_mapping(mapping, dram):
    """
    Reconstruct address from mapping components.
    
    Args:
        mapping: Dictionary with address components
        dram: DRAM configuration object
    
    Returns:
        Reconstructed address
    """
    # Define bit lengths for each component
    bit_lengths = {
        'Bt': int(math.ceil(math.log(dram.burst_length, 2))),
        'Co': int(math.ceil(math.log(dram.csl_lines, 2))),
        'Ro': int(math.ceil(math.log(dram.rows_per_bank, 2))),
        'Ba': int(math.ceil(math.log(dram.num_banks_per_bankgroup, 2))),
        'Bg': int(math.ceil(math.log(dram.num_bankgroups_per_chip, 2))),
        'Ra': int(math.ceil(math.log(dram.num_ranks_per_channel, 2))),
        'Ch': int(math.ceil(math.log(dram.num_channels, 2))),
    }
    
    addr_offset = 0
    interleaving_components = [dram.addr_interleaving[i:i+2] for i in range(0, len(dram.addr_interleaving), 2)]
    
    for i, component in enumerate(interleaving_components):
        if component not in mapping:
            raise ValueError(f"Missing component '{component}' in mapping")
        
        shift_bits = 0
        for j in range(i + 1, len(interleaving_components)):
            shift_bits += bit_lengths[interleaving_components[j]]
        
        addr_offset += int(mapping[component]) << shift_bits
    
    return int(addr_offset)


def clear_mapping_cache():
    """Clear the mapping caches to free memory."""
    get_mapping_cached.cache_clear()
    calculate_interleaving_aware_offset_cached.cache_clear()

def calculate_batch_round_robin_rank(batch_idx, dram):
    """
    Calculate which KV rank a batch should be allocated to using round-robin fashion.
    
    Args:
        batch_idx: Batch index (0, 1, 2, ...)
        dram: DRAM configuration object
    
    Returns:
        KV rank index (0 to num_kv_ranks_per_channel - 1)
    """
    return batch_idx % dram.num_kv_ranks_per_channel

def calculate_interleaving_aware_offset_with_batch_round_robin(request_number, dram, rank_type, batch_idx=-1, chip_idx=-1):
    """
    Calculate address offset that respects memory interleaving pattern with batch-based round-robin rank allocation.
    
    Args:
        request_number: Sequential request number (0, 1, 2, ...)
        dram: DRAM configuration object
        rank_type: "weight" or "kv"
        batch_idx: Batch index for round-robin rank allocation (-1 for non-batch allocation)
        chip_idx: Chip index (-1 for all chips, 0 to num_chips-1 for specific chip)
    
    Returns:
        Address offset that respects interleaving and rank boundaries with batch round-robin
    """
    # For KV cache with batch round-robin, we need to override the rank selection
    if rank_type == "kv" and batch_idx >= 0:
        # Calculate the target KV rank using round-robin
        target_kv_rank = calculate_batch_round_robin_rank(batch_idx, dram)
        
        # Define bit lengths for each component
        bit_lengths = {
            'Bt': int(math.ceil(math.log(dram.burst_length, 2))),
            'Co': int(math.ceil(math.log(dram.csl_lines, 2))),
            'Ro': int(math.ceil(math.log(dram.rows_per_bank, 2))),
            'Ba': int(math.ceil(math.log(dram.num_banks_per_bankgroup, 2))),
            'Bg': int(math.ceil(math.log(dram.num_bankgroups_per_chip, 2))),
            'Ra': int(math.ceil(math.log(dram.num_ranks_per_channel, 2))),
            'Ch': int(math.ceil(math.log(dram.num_channels, 2))),
        }

        all_bank_mode = dram.mode == "AB" 
        components = {}
        remaining_request = request_number
        
        # Parse interleaving pattern and calculate components
        interleaving_components = [dram.addr_interleaving[i:i+2] for i in range(0, len(dram.addr_interleaving), 2)]
        
        for component in reversed(interleaving_components):
        
            if component not in bit_lengths:
                raise ValueError(f"Unknown component '{component}' in addr_interleaving")   
                     
            if all_bank_mode and component in ['Ba', 'Bg', 'Bt']:
                components[component] = 0

            elif component == 'Ra':
                # Use the round-robin determined KV rank
                rank_value = dram.num_wt_ranks_per_channel + target_kv_rank
                components[component] = rank_value
                # Adjust remaining request to account for rank selection
                remaining_request //= dram.num_kv_ranks_per_channel
            else:
                bits = bit_lengths[component]
                max_value = 2**bits - 1
                #components[component] = remaining_request % (max_value + 1)
                components[component] = remaining_request & max_value
                #remaining_request //= (max_value + 1)
                remaining_request >>= bits
        
        # Reconstruct address from components
        addr_offset = 0
        for i, component in enumerate(interleaving_components):
            shift_bits = 0
            for j in range(i + 1, len(interleaving_components)):
                shift_bits += bit_lengths[interleaving_components[j]]
            addr_offset += components[component] << shift_bits
        
        return addr_offset
        
    else:
        # Use the original function for non-batch allocation
        return calculate_interleaving_aware_offset(request_number, dram, rank_type, chip_idx)
   

def calculate_interleaving_aware_offset_with_explicit_channel_rank(request_number, dram, rank_type, target_channel, target_rank, chip_idx=-1):
    """
    Calculate address offset with explicit control over channel and rank allocation.
    This ensures temp tensors go to the exact channel and rank specified.
    
    Args:
        request_number: Sequential request number (0, 1, 2, ...)
        dram: DRAM configuration object
        rank_type: "weight" or "kv"
        target_channel: Explicit channel to allocate to (0 to num_channels-1)
        target_rank: Explicit rank within the channel (0 to num_ranks_per_channel-1)
        chip_idx: Chip index (-1 for all chips, 0 to num_chips-1 for specific chip)
    
    Returns:
        Address offset that maps to the exact channel and rank specified
    """
    # Validate inputs
    if target_channel < 0 or target_channel >= dram.num_channels:
        raise ValueError(f"Invalid target_channel {target_channel}. Must be 0 to {dram.num_channels-1}")
    
    if target_rank < 0 or target_rank >= dram.num_ranks_per_channel:
        raise ValueError(f"Invalid target_rank {target_rank}. Must be 0 to {dram.num_ranks_per_channel-1}")
    
    # Define bit lengths for each component
    bit_lengths = {
        'Bt': int(math.ceil(math.log(dram.burst_length, 2))),
        'Co': int(math.ceil(math.log(dram.csl_lines, 2))),
        'Ro': int(math.ceil(math.log(dram.rows_per_bank, 2))),
        'Ba': int(math.ceil(math.log(dram.num_banks_per_bankgroup, 2))),
        'Bg': int(math.ceil(math.log(dram.num_bankgroups_per_chip, 2))),
        'Ra': int(math.ceil(math.log(dram.num_ranks_per_channel, 2))),
        'Ch': int(math.ceil(math.log(dram.num_channels, 2))),
    }
    
    all_bank_mode = dram.mode == "AB"
    interleaving_components = [dram.addr_interleaving[i:i+2] for i in range(0, len(dram.addr_interleaving), 2)]
    components = {}
    remaining_request = request_number

    for component in reversed(interleaving_components):
        if all_bank_mode and component in ['Ba', 'Bg', 'Bt']:
            components[component] = 0
        elif component == 'Ch':
            # Explicitly set the target channel
            components[component] = target_channel
        elif component == 'Ra':
            # Explicitly set the target rank
            components[component] = target_rank
        else:
            # Use remaining request bits for other components
            bits = bit_lengths[component]
            max_value = 2**bits - 1
            components[component] = remaining_request & max_value
            remaining_request >>= bits
    
    # Reconstruct address from components
    addr_offset = 0
    for i, component in enumerate(interleaving_components):
        shift_bits = 0
        for j in range(i + 1, len(interleaving_components)):
            shift_bits += bit_lengths[interleaving_components[j]]
        addr_offset += components[component] << shift_bits

    return addr_offset
