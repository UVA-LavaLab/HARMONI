from misc.type import *
from utils.logging_util import logger
from modeling.core.dram_utils import get_mapping as get_mapping_from_dram, get_accesses as get_accesses_from_dram, add_to_address_skipping_rank, kv_request_address_step
from collections import defaultdict
import math

#CLEANUP: if required - model should be a class which can be imported here

# Define the HarmoniTensor class
class HarmoniTensor:
    # Class variable to store the dram configuration
    _global_dram = None
    
    # NOTE: other_tensor_sizes is only implemented by development purpose (can be removed)
    _other_tensor_sizes = defaultdict(lambda: {'sizes': [], 'count': 0})
    
    @classmethod
    def set_global_dram(cls, dram):
        """Set the global dram configuration for all HarmoniTensor instances"""
        cls._global_dram = dram
        
    @classmethod
    def get_other_tensor_sizes(cls):
        """Get the collected sizes of tensors that are not weight or KV"""
        return dict(cls._other_tensor_sizes)

    def __init__(self, name, tag, precision, shape, stride=[], addr_offset=0, chip_idx=-1, mapping={}, dram=None):
        self.name = name # String
        self.tag = tag  # Enum (Tag)
        self.precision = precision  # Enum (Precision)
        self.shape = tuple(shape)  # Tuple of integers
        
        # Use provided dram or fall back to global dram
        self.dram = dram if dram is not None else self._global_dram

        self.addr_offset = addr_offset  # Hexadecimal value (int or str)
        self.chip_idx = chip_idx
        self.stride = self.get_stride(stride)  # List of integers
        self.numel = self.get_numel()  # Integer
        self.size = self.get_size() # Bytes
        if mapping:
            self.mapping = mapping # Dictionary
        else:
            self.mapping = self.get_mapping()
        self.requests = 0
        
        self.locations = None
        self.row_accesses = 0
        self.col_accesses = 0
        
    def __repr__(self):
        def format_locations(locations):
            if not locations:
                return "N/A"
            lines = []
            for loc_dict, size, start_addr, end_addr, rows, cols in locations:
                loc_str = f"loc={loc_dict}, size={size}, start=0x{start_addr:x}, end=0x{end_addr:x}, rows={rows}, cols={cols}"
                lines.append(loc_str)
            return "[\n  " + ",\n  ".join(lines) + "\n]"
        locations_str = format_locations(self.locations) if self.locations is not None else "N/A"
        return (f"HarmoniTensor(name={self.name}, tag={self.tag}, shape={self.shape}, stride={self.stride}, precision={self.precision}, "
                f"size={self.size}, addr_offset={hex(self.addr_offset) if isinstance(self.addr_offset, int) else self.addr_offset}, "
                f"mapping={self.mapping}, chip_idx={self.chip_idx}, row_accesses={self.row_accesses}, col_accesses={self.col_accesses},\n"
                f"locations={locations_str})")
    
    def get_numel(self):
        """
        Gets the total number of elements in the tensor
        """
        numel = 1
        for i in list(self.shape):
            numel = numel * i
        return numel
    
    def get_size(self):
        """
        Gets the size of tensor in bytes
        """
        size_bytes = get_bytes_per_element(self.precision, self.tag)
        return self.numel*size_bytes
    
    def get_mapping(self):
        """
        Gets DRAM mapping based on addr_offset and size of the tensor 
        """
        if self.dram is None:
            logger.error(f"DRAM configuration not defined for {self.name} tensor")
            # Return empty mapping if dram is not available
            # For WEIGHT and KV tensors, mapping should be set later when dram is available
            return {}
        
        # Determine rank type based on tensor tag
        if (self.tag == HarmoniTensorType.WEIGHT):
            rank_type = "weight"
            mapping = get_mapping_from_dram(self.addr_offset, self.dram, rank_type)
        elif (self.tag == HarmoniTensorType.KV):
            rank_type = "kv"
            mapping = get_mapping_from_dram(self.addr_offset, self.dram, rank_type)
        else:
            # Track tensor sizes and counts
            data = HarmoniTensor._other_tensor_sizes[self.name]
            if self.size not in data['sizes']:
                data['sizes'].append(self.size)
            data['count'] += 1
            mapping = {}

        return mapping
    
    def get_stride(self, stride):
        """Computes default stride for a row-major tensor."""
        if stride == []:
            product = 1
            for dim in reversed(self.shape):
                stride.insert(0, product)
                product *= dim
            return tuple(stride)
        else:
            return stride
        
    @staticmethod
    def get_tensor_cache_key(tensor):
        """
        Create a hashable cache key from tensor and dram objects.
        Only includes the attributes that affect the tensor location calculation.
        """
        tensor_key = (
            tensor.name,
            tensor.size,
            tensor.chip_idx,
            tensor.addr_offset,
            tensor.tag
        )
        return tensor_key
    
    @staticmethod   
    def get_tensor_locations(tensor, dram):
        """
        Calculate all locations where a tensor's data is stored based on its size, address interleaving, and chip_idx information. Returns a tuple: (merged_locations, num_unique_rows, total_col_accesses) where merged_locations is a list of (location, size, start_addr, end_addr, rows, cols), num_unique_rows is the number of unique rows accessed (using the row field from the address mapping), and total_col_accesses is the total number of accesses (i.e., every time you step through your access granularity).
        """
        if tensor.tag == HarmoniTensorType.ACT:
            logger.error(f"why ACT is using get_tensor_locations")
        
        # Use the cache key function
        cache_key = HarmoniTensor.get_tensor_cache_key(tensor)
        if tensor.tag == HarmoniTensorType.WEIGHT:
            if hasattr(HarmoniTensor.get_tensor_locations, 'cache') and cache_key in HarmoniTensor.get_tensor_locations.cache:
                return HarmoniTensor.get_tensor_locations.cache[cache_key]

        tensor_size = int(tensor.size)  # in bytes
        partition_size = int(dram.calculate_partition_size() * 1024)  # in bytes
        num_chips = int(dram.num_chips_per_rank)
        partition_map = {}  # key: (channel, rank, chip), value: (size, start_addr, end_addr)

        access_info = {}  # key: (channel, rank, chip), value: (unique_rows_set, col_accesses)
        # For row/col accesses
        unique_rows = set()  # (channel, rank, chip, row)
        total_col_accesses = 0

        # Calculate number of partitions needed for this tensor
        num_partitions = math.ceil(tensor_size / partition_size)
        chip_partition_size = partition_size // num_chips    
        # Address distance between consecutive chip-partitions. One KV chip-partition is one allocator request
        # (one Co step, 0x100 on DDR5), not chip_partition_size address units (512 on DDR5, 1024 on GDDR6), which
        # skipped columns and ran past the tensor's reservation into the next one. Weights keep chip_partition_size
        # because their allocator orders fields differently (Co -> Ra -> Ch -> Ro); Fix 6 revisits them
        addr_step = kv_request_address_step(dram) if tensor.tag == HarmoniTensorType.KV else chip_partition_size
             
        for i in range(num_partitions*num_chips):
            
            if tensor.chip_idx == -1: # Tensor is spread across all chips in the rank
                chip = i % num_chips
                step = int((i // num_chips) * addr_step)
            else:
                chip = tensor.chip_idx
                step = int(i * addr_step)

            if tensor.tag == HarmoniTensorType.KV:
                # KV rank is pinned by batch round-robin: carries must skip the Ra field, not spill into it
                addr = add_to_address_skipping_rank(int(tensor.addr_offset), step, dram)
            else:
                addr = int(tensor.addr_offset) + step

            
            partition_mapping = get_mapping_from_dram(int(addr), dram, "weight" if tensor.tag == HarmoniTensorType.WEIGHT else "kv") #NOTE: Should also work for ACT tensor since get_mapping is not using rank_type. Also, get_mapping for ACT tensor is done later!

            # Get values with None check
            rank = partition_mapping.get('Ra')
            row = partition_mapping.get('Ro')
            col = partition_mapping.get('Co')
            channel = partition_mapping.get('Ch')

            if any(x is None for x in [rank, row, col, channel]):
                logger.error(f"Missing mapping fields for tensor {tensor.name} at addr 0x{addr:x}: Ra={rank}, Ro={row}, Co={col}, Ch={channel}")
                continue
            
            # Respect the tensor's tag when determining rank type
            if tensor.tag == HarmoniTensorType.WEIGHT:
                rank = rank % dram.num_wt_ranks_per_channel
            elif tensor.tag == HarmoniTensorType.KV:
                # Ra is pinned to the batch's KV rank; never write to partition_mapping (shared lru_cache dict).
                # A weight rank here would give a negative kv_rank that silently matches no logic unit, so fail loudly.
                if rank < dram.num_wt_ranks_per_channel:
                    raise ValueError(f"KV tensor {tensor.name} decoded to weight rank Ra={rank} at addr 0x{addr:x}")
                rank = rank - dram.num_wt_ranks_per_channel

            key = (channel, rank, chip)
            
            # Track unique rows and column accesses:
            if key not in access_info:
                access_info[key] = ({row}, 1)  # (unique_rows_set, col_accesses)
            else:
                prev_unique_rows, prev_col_accesses = access_info[key]
                access_info[key] = (prev_unique_rows | {row}, prev_col_accesses + 1)

            if key not in partition_map:
                # First time this key is seen: set size, start_addr, end_addr
                partition_map[key] = (chip_partition_size, addr, addr + addr_step) #NOTE: Do not get mapping from this partition end_addr 
            else:
                prev_size, start_addr, prev_end_addr = partition_map[key]
                # Update size and end_addr
                partition_map[key] = (prev_size + chip_partition_size, start_addr, addr + addr_step)

        # Convert back to list of (location_dict, size, start_addr, end_addr)
        merged_locations = []
        total_col_accesses = 0
        total_row_accesses = 0
        for key, (size, start_addr, end_addr) in partition_map.items():
            channel = key[0]
            rank = key[1]
            chip = key[2]
            
            # Respect the tensor's tag when determining rank type
            if tensor.tag == HarmoniTensorType.WEIGHT:
                # For weight tensors, always use weight ranks
                loc_dict = {'channel': channel, 'wt_rank': rank, 'chip': chip}
            elif tensor.tag == HarmoniTensorType.KV:
                loc_dict = {'channel': channel, 'kv_rank': rank, 'chip': chip}
            else: 
                loc_dict = {}
            # Add the access counts to loc_dict:
            if key in access_info:
                unique_rows, col_accesses = access_info[key]
                total_col_accesses += col_accesses
                total_row_accesses += len(unique_rows)
            merged_locations.append((loc_dict, size, start_addr, end_addr, len(unique_rows), col_accesses))
            
        # Cache the results
        if tensor.tag == HarmoniTensorType.WEIGHT:
            if not hasattr(HarmoniTensor.get_tensor_locations, 'cache'):
                HarmoniTensor.get_tensor_locations.cache = {}
            HarmoniTensor.get_tensor_locations.cache[cache_key] = (merged_locations, total_row_accesses, total_col_accesses)
        
        return merged_locations, total_row_accesses, total_col_accesses  
    
    @staticmethod
    def get_temp_tensor_locations(channel, rank, chip, tensor_size, dram, rank_type):
        partition_size = dram.calculate_partition_size() * 1024
        num_partitions = math.ceil(tensor_size / partition_size)
        total_row_accesses = math.ceil(num_partitions/dram.csl_lines)
        total_col_accesses = num_partitions
        loc_dict = {'channel': channel, f'{rank_type}_rank': rank, 'chip': chip}       
        return [(loc_dict, tensor_size, 0, 0, total_row_accesses, total_col_accesses)], total_row_accesses, total_col_accesses