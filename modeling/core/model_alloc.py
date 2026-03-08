import math
from collections import OrderedDict
from misc.type import *
from modeling.core.dram_utils import calculate_interleaving_aware_offset, calculate_interleaving_aware_offset_with_explicit_channel_rank, calculate_interleaving_aware_offset_with_batch_round_robin, get_mapping
from utils.logging_util import logger
from modeling.core.tensor import HarmoniTensor

def update_model_weight(model, model_weight, dram, lin, batch):
    """
    Update model weights with interleaving-aware address allocation.
    This ensures weight tensors never map to KV ranks and respects memory interleaving patterns.
    """
    partition_size = dram.calculate_partition_size() * 1024 # in bytes
    num_chips = dram.num_chips_per_rank 
    
    # Track request numbers for sequential allocation
    weight_request_counter = 0

    for layer, data in model_weight.weights_new.items():
        if "layer" in layer:
            for key, value in data.items():
                if key in {"Wq", "Wk", "Wv"}: 
                    # Group heads by chips - each group of num_chips heads shares the same address offset
                    for head, wmeta in value.items():    
                        requests_local = wmeta.size/(partition_size) #num of allbank(mode) requests. Each request is a burst access
                        wmeta.requests = requests_local*num_chips #hack to show the total requests made in each chip
                        chip_idx = head % num_chips
                        # Assign chip index based on head
                        wmeta.chip_idx = chip_idx
                        # Use the same address offset for all heads in the same group
                        wmeta.addr_offset = int(calculate_interleaving_aware_offset(weight_request_counter, dram, "weight", chip_idx))
                        #logger.debug(f"{layer} {key} {wmeta.addr_offset} {weight_request_counter}") 
                        
                        
                        wmeta.mapping = get_mapping(int(wmeta.addr_offset), dram, "weight")
                        wmeta.locations, wmeta.row_accesses, wmeta.col_accesses = HarmoniTensor.get_tensor_locations(wmeta, dram)
                        
                        # Only increment counter when we finish a complete group
                        if (head + 1) % num_chips == 0:
                            weight_request_counter += math.ceil(requests_local * num_chips)
                        
                elif key in {"Wqkv", "Wo", "Wup", "Wgate", "Wdown", "Wff1", "Wff2"}:
                    requests_local = value.size/partition_size
                    value.requests = requests_local
                    
                    # Use interleaving-aware address calculation
                    value.addr_offset = int(calculate_interleaving_aware_offset(weight_request_counter, dram, "weight", -1))
                    value.chip_idx = -1
                    value.mapping = get_mapping(int(value.addr_offset), dram, "weight")
                    value.locations, value.row_accesses, value.col_accesses = HarmoniTensor.get_tensor_locations(value, dram) 
                    weight_request_counter += math.ceil(requests_local)
        else:
            requests_local = data.size/partition_size
            data.requests = requests_local
            
            # Use interleaving-aware address calculation
            data.addr_offset = int(calculate_interleaving_aware_offset(weight_request_counter, dram, "weight", -1))
            data.chip_idx = -1
            data.mapping = get_mapping(int(data.addr_offset), dram, "weight")
            data.locations, data.row_accesses, data.col_accesses = HarmoniTensor.get_tensor_locations(data, dram)
            weight_request_counter += math.ceil(requests_local)
        
    for layer, data in model_weight.weights_new.items():
        if "layer" in layer:
            for key, value in data.items():
                if key in {"rms_norm1", "rms_norm2"}:
                    requests_local = value.size/partition_size
                    value.requests = requests_local
                    
                    # Use interleaving-aware address calculation
                    value.addr_offset = int(calculate_interleaving_aware_offset(weight_request_counter, dram, "weight", -1))
                    value.chip_idx = -1
                    value.mapping = get_mapping(int(value.addr_offset), dram, "weight")
                    value.locations, value.row_accesses, value.col_accesses = HarmoniTensor.get_tensor_locations(value, dram)
                    weight_request_counter += math.ceil(requests_local)
    logger.debug(f"Total AllChip requests made to load model weights in {dram.mode} mode is: {weight_request_counter}") 
    
    hdim = model['hdim']

    #ISSUE: Potential issue, overwriting exisitng tensors :(
    temp_tensor_wt = []
    for channel in range(dram.num_channels):
        for rank in range(dram.num_wt_ranks_per_channel):
            # Use explicit channel and rank allocation for temp tensors
            temp_addr_offset = int(calculate_interleaving_aware_offset_with_explicit_channel_rank(
                weight_request_counter, dram, "weight", channel, rank, -1))
            temp_tensor_wt.append(HarmoniTensor(
                name=f"temp_tensor_wt_ch{channel}_r{rank}", 
                tag=HarmoniTensorType.ACT, 
                #precision=DataType.BF16, 
                precision=model['dtype'], 
                shape=(batch, lin, hdim), 
                addr_offset=temp_addr_offset, 
                mapping=get_mapping(int(temp_addr_offset), dram, "weight")))
            weight_request_counter += math.ceil(temp_tensor_wt[-1].size/partition_size)
                  
    return model_weight, temp_tensor_wt


def update_model_kv(args, model, dram, seq_len, batch=1):
    """
    Update model KV cache with interleaving-aware address allocation.
    This ensures KV tensors map to KV ranks only and respects memory interleaving patterns.
    """
    model_kv = OrderedDict()
    
    k_addr_offset = [[[[] for _ in range(model["kv_heads"])] for _ in range(model["nlayers"])] for _ in range(batch)]
    v_addr_offset = [[[[] for _ in range(model["kv_heads"])] for _ in range(model["nlayers"])] for _ in range(batch)]

    k_chip_idx = [[[[] for _ in range(model["kv_heads"])] for _ in range(model["nlayers"])] for _ in range(batch)]
    v_chip_idx = [[[[] for _ in range(model["kv_heads"])] for _ in range(model["nlayers"])] for _ in range(batch)]


        
    KV_size_per_layer_per_batch = model["kv_heads"] * model["hdim"] / model["num_heads"] * seq_len * 2

    size_bytes = get_bytes_per_element(model["dtype"], HarmoniTensorType.KV)

    KV_bytes_per_layer_per_batch = KV_size_per_layer_per_batch * size_bytes
    KV_bytes_per_head_per_layer_per_batch = KV_bytes_per_layer_per_batch // model["kv_heads"] 
    partition_size = dram.calculate_partition_size() * 1024 # in bytes
    num_chips = dram.num_chips_per_rank

    # Track request numbers for sequential allocation
    kv_request_counter = 0 
    
    temp_tensor_kv = []
    
    requests_local = math.ceil((KV_bytes_per_head_per_layer_per_batch/partition_size)/2)

    # Allocate KV cache tensors with batch round-robin
    for i in range(batch):
        for layers in range(model["nlayers"]):
            # Process "k" cache first
            k_cache_start_counter = kv_request_counter
            for heads in range(model["kv_heads"]):
                
                # Use batch-based round-robin rank allocation for KV cache
                if heads % num_chips == 0:
                    addr_offset = int(calculate_interleaving_aware_offset_with_batch_round_robin(kv_request_counter, dram, "kv", batch_idx=i, chip_idx=-1))
                
                k_addr_offset[i][layers][heads] = addr_offset
                k_chip_idx[i][layers][heads] = heads % num_chips
                
                # Only increment counter when we finish a complete group
                if (heads + 1) % num_chips == 0:
                    kv_request_counter += math.ceil(requests_local * num_chips)
            
            # After processing all "k" cache heads, calculate total "k" space and advance counter
            # This ensures "v" cache starts after all "k" cache space
            # NOTE: KV_bytes_per_head_per_layer_per_batch includes both K and V, so K is half of it
            total_k_bytes = (KV_bytes_per_head_per_layer_per_batch / 2) * model["kv_heads"]
            total_k_requests = math.ceil(total_k_bytes / partition_size)
            kv_request_counter = k_cache_start_counter + total_k_requests
            
            # Now process "v" cache
            for heads in range(model["kv_heads"]):
                # Use batch-based round-robin rank allocation for KV cache
                if heads % num_chips == 0:
                    addr_offset = int(calculate_interleaving_aware_offset_with_batch_round_robin(kv_request_counter, dram, "kv", batch_idx=i, chip_idx=-1))
                
                v_addr_offset[i][layers][heads] = addr_offset
                #logger.debug(f"Addr offset for v batch {i} layer {layers} head {heads} is {hex(addr_offset)}")
                v_chip_idx[i][layers][heads] = heads % num_chips
                
                # Only increment counter when we finish a complete group
                if (heads + 1) % num_chips == 0:
                    kv_request_counter += math.ceil(requests_local * num_chips)
    
    #ISSUE: Potential issue, overwriting existing tensors
    for channel in range(dram.num_channels):
        for rank in range(dram.num_kv_ranks_per_channel):
            # Use explicit channel and rank allocation for KV temp tensors
            temp_addr_offset = int(calculate_interleaving_aware_offset_with_explicit_channel_rank(
                kv_request_counter, dram, "kv", channel, dram.num_wt_ranks_per_channel + rank, -1))
            temp_tensor_kv.append(HarmoniTensor(
                name=f"temp_tensor_kv_ch{channel}_r{rank}", 
                tag=HarmoniTensorType.ACT, 
                #precision=DataType.BF16, 
                precision=model['dtype'], 
                shape=(1, model['hdim']), 
                addr_offset=temp_addr_offset, 
                mapping=get_mapping(int(temp_addr_offset), dram, "kv")))
            #logger.info(f"Temporary tensor allocation in kv ranks channel {channel} rank {rank}: {temp_tensor_kv[-1]}")
            kv_request_counter += math.ceil(temp_tensor_kv[-1].size/partition_size)
    
    logger.info("Creating model_kv")
    for b in range(batch):
        model_kv[f"batch_{b}"] = OrderedDict({
            f"layer_{l}" : OrderedDict({
                f"head_{h}" : {
                    "k" : HarmoniTensor(name=f"t_kcache_b{b}_l{l}_h{h}", tag=HarmoniTensorType.KV, precision=model['dtype'], shape=(1, seq_len, model['head_dim']), addr_offset = k_addr_offset[b][l][h], chip_idx = k_chip_idx[b][l][h], mapping=get_mapping(int(k_addr_offset[b][l][h]), dram, "kv")), 

                    "v" : HarmoniTensor(name=f"t_vcache_b{b}_l{l}_h{h}", tag=HarmoniTensorType.KV, precision=model['dtype'], shape=(1, seq_len, model['head_dim']), addr_offset = v_addr_offset[b][l][h], chip_idx = v_chip_idx[b][l][h], mapping=get_mapping(int(v_addr_offset[b][l][h]), dram, "kv")), 
                }
                for h in range(model["kv_heads"])
            })
            for l in range(model["nlayers"])
        })

    return model_kv, temp_tensor_kv 
