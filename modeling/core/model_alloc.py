import math
from collections import OrderedDict
from misc.type import *
from modeling.core.dram_utils import calculate_interleaving_aware_offset, calculate_interleaving_aware_offset_with_explicit_channel_rank, calculate_interleaving_aware_offset_with_batch_round_robin, get_mapping, calculate_batch_round_robin_rank, requests_before_wrap
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
    # The weight allocator counts Co, then the weight ranks, then Ch and Ro, so it holds num_wt x 2^(Co+Ch+Ro) requests
    weight_capacity = requests_before_wrap(dram, ['Ra']) * dram.num_wt_ranks_per_channel

    def check_weight_capacity(what, start_request, num_requests):
        """Raise instead of letting Ro wrap: a wrapped allocation silently lands on earlier weights."""
        if start_request + num_requests > weight_capacity:
            raise ValueError(f"{what} does not fit in the weight ranks ({model['name']}, {dram.num_channels} channels x {dram.num_wt_ranks_per_channel} weight ranks, "
                             f"input {lin}, batch {batch}): it needs requests {start_request}-{start_request + num_requests - 1}, "
                             f"but the weight ranks hold {weight_capacity} requests. Use a DRAM config with more weight capacity.")

    for layer, data in model_weight.weights_new.items():
        if "layer" in layer:
            for key, value in data.items():
                if key in {"Wq", "Wk", "Wv"}: 
                    # Group heads by chips - each group of num_chips heads shares the same address offset
                    for head, wmeta in value.items():    
                        requests_local = wmeta.size/(partition_size) #num of allbank(mode) requests. Each request is a burst access
                        wmeta.requests = requests_local*num_chips #hack to show the total requests made in each chip
                        chip_idx = head % num_chips
                        if head % num_chips == 0:
                            check_weight_capacity(f"{layer} {key} heads {head}-{min(head + num_chips, len(value)) - 1}", weight_request_counter, math.ceil(requests_local * num_chips))
                        # Assign chip index based on head
                        wmeta.chip_idx = chip_idx
                        # Use the same address offset for all heads in the same group
                        wmeta.addr_offset = int(calculate_interleaving_aware_offset(weight_request_counter, dram, "weight", chip_idx))
                        #logger.debug(f"{layer} {key} {wmeta.addr_offset} {weight_request_counter}") 
                        
                        
                        wmeta.mapping = get_mapping(int(wmeta.addr_offset), dram, "weight")
                        wmeta.locations, wmeta.row_accesses, wmeta.col_accesses = HarmoniTensor.get_tensor_locations(wmeta, dram)
                        
                        # Increment the counter after every group, including a partial last one (heads % num_chips != 0),
                        # so the next tensor does not start on top of it
                        if (head + 1) % num_chips == 0 or head == len(value) - 1:
                            weight_request_counter += math.ceil(requests_local * num_chips)
                        
                elif key in {"Wqkv", "Wo", "Wup", "Wgate", "Wdown", "Wff1", "Wff2"}:
                    requests_local = value.size/partition_size
                    value.requests = requests_local
                    check_weight_capacity(f"{layer} {key}", weight_request_counter, math.ceil(requests_local))
                    
                    # Use interleaving-aware address calculation
                    value.addr_offset = int(calculate_interleaving_aware_offset(weight_request_counter, dram, "weight", -1))
                    value.chip_idx = -1
                    value.mapping = get_mapping(int(value.addr_offset), dram, "weight")
                    value.locations, value.row_accesses, value.col_accesses = HarmoniTensor.get_tensor_locations(value, dram) 
                    weight_request_counter += math.ceil(requests_local)
        else:
            requests_local = data.size/partition_size
            data.requests = requests_local
            check_weight_capacity(f"{layer}", weight_request_counter, math.ceil(requests_local))
            
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
                    check_weight_capacity(f"{layer} {key}", weight_request_counter, math.ceil(requests_local))
                    
                    # Use interleaving-aware address calculation
                    value.addr_offset = int(calculate_interleaving_aware_offset(weight_request_counter, dram, "weight", -1))
                    value.chip_idx = -1
                    value.mapping = get_mapping(int(value.addr_offset), dram, "weight")
                    value.locations, value.row_accesses, value.col_accesses = HarmoniTensor.get_tensor_locations(value, dram)
                    weight_request_counter += math.ceil(requests_local)
    logger.debug(f"Total AllChip requests made to load model weights in {dram.mode} mode is: {weight_request_counter}") 
    
    hdim = model['hdim']

    # Weight temp tensors go in the first Ro row above the weights, so they never overlap them (the weights fill rows
    # 0..last in every channel and weight rank). The explicit channel/rank allocator pins Ch and Ra, so (AB mode, Ro on
    # top) its request number is Ro * requests_per_row + Co, and it holds temp_capacity requests per (channel, rank)
    temp_requests = math.ceil(batch * lin * hdim * get_bytes_per_element(model['dtype'], HarmoniTensorType.ACT) / partition_size)  # (batch, lin, hdim) ACT tensor
    temp_capacity = requests_before_wrap(dram, ['Ch', 'Ra'])
    requests_per_row = requests_before_wrap(dram, ['Ch', 'Ra', 'Ro'])
    first_free_row = 0
    if weight_request_counter > 0:
        last_wt_addr = calculate_interleaving_aware_offset(weight_request_counter - 1, dram, "weight", -1)
        first_free_row = get_mapping(int(last_wt_addr), dram, "weight")['Ro'] + 1
    temp_request = first_free_row * requests_per_row
    temp_tensor_wt = []
    for channel in range(dram.num_channels):
        for rank in range(dram.num_wt_ranks_per_channel):
            if temp_request + temp_requests > temp_capacity:
                raise ValueError(f"temp_tensor_wt_ch{channel}_r{rank} (placed at row {first_free_row}, the first row above the weights) does not fit in channel {channel} of weight rank {rank} "
                                 f"({model['name']}, input {lin}, batch {batch}): it needs requests {temp_request}-{temp_request + temp_requests - 1}, but channel {channel} of weight rank {rank} holds {temp_capacity} requests. "
                                 f"Use a smaller batch or input length, or a DRAM config with more weight capacity.")
            # Use explicit channel and rank allocation for temp tensors
            temp_addr_offset = int(calculate_interleaving_aware_offset_with_explicit_channel_rank(
                temp_request, dram, "weight", channel, rank, -1))
            temp_tensor_wt.append(HarmoniTensor(
                name=f"temp_tensor_wt_ch{channel}_r{rank}", 
                tag=HarmoniTensorType.ACT, 
                #precision=DataType.BF16, 
                precision=model['dtype'], 
                shape=(batch, lin, hdim), 
                addr_offset=temp_addr_offset, 
                mapping=get_mapping(int(temp_addr_offset), dram, "weight")))
            # temp_requests (used in the capacity check above) must match the tensor's real size
            assert math.ceil(temp_tensor_wt[-1].size/partition_size) == temp_requests
                  
    return model_weight, temp_tensor_wt


def check_kv_rank_capacity(args, seq_len, batch, what, where, start_request, num_requests, capacity):
    """Raise instead of letting Ro wrap: a wrapped allocation silently lands on earlier KV data."""
    if start_request + num_requests > capacity:
        raise ValueError(f"{what} does not fit in {where} ({args.model_name} on {args.dram}, seq_len {seq_len}, batch {batch}): "
                         f"it needs requests {start_request}-{start_request + num_requests - 1}, but {where} holds {capacity} requests. "
                         f"Use a smaller batch or sequence length, or a DRAM config with more KV ranks or capacity.")

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

    # Track request numbers for sequential allocation, one counter per KV rank: batches on different KV ranks
    # do not share address space, so each rank is filled densely by its own batches
    kv_request_counter = [0] * dram.num_kv_ranks_per_channel
    kv_rank_capacity = requests_before_wrap(dram, ['Ra'])
    
    temp_tensor_kv = []
    
    requests_local = math.ceil((KV_bytes_per_head_per_layer_per_batch/partition_size)/2)
    # One group = up to num_chips heads, one per chip; each head needs requests_local * num_chips requests on its chip
    requests_per_group = math.ceil(requests_local * num_chips)

    # Allocate KV cache tensors with batch round-robin
    for i in range(batch):
        kv_rank = calculate_batch_round_robin_rank(i, dram)
        for layers in range(model["nlayers"]):
            # Process "k" cache first
            for heads in range(model["kv_heads"]):
                
                # Use batch-based round-robin rank allocation for KV cache
                if heads % num_chips == 0:
                    check_kv_rank_capacity(args, seq_len, batch, f"K cache of batch {i} layer {layers} heads {heads}-{min(heads + num_chips, model['kv_heads']) - 1}",
                                           f"kv_rank {kv_rank} (Ra {dram.num_wt_ranks_per_channel + kv_rank})", kv_request_counter[kv_rank], requests_per_group, kv_rank_capacity)
                    addr_offset = int(calculate_interleaving_aware_offset_with_batch_round_robin(kv_request_counter[kv_rank], dram, "kv", batch_idx=i, chip_idx=-1))
                
                k_addr_offset[i][layers][heads] = addr_offset
                k_chip_idx[i][layers][heads] = heads % num_chips
                
                # Increment the counter after every group, including a partial last one (kv_heads % num_chips != 0).
                # Every group advancing is also what makes "v" start after all "k" space
                if (heads + 1) % num_chips == 0 or heads == model["kv_heads"] - 1:
                    kv_request_counter[kv_rank] += requests_per_group
            
            # Now process "v" cache
            for heads in range(model["kv_heads"]):
                # Use batch-based round-robin rank allocation for KV cache
                if heads % num_chips == 0:
                    check_kv_rank_capacity(args, seq_len, batch, f"V cache of batch {i} layer {layers} heads {heads}-{min(heads + num_chips, model['kv_heads']) - 1}",
                                           f"kv_rank {kv_rank} (Ra {dram.num_wt_ranks_per_channel + kv_rank})", kv_request_counter[kv_rank], requests_per_group, kv_rank_capacity)
                    addr_offset = int(calculate_interleaving_aware_offset_with_batch_round_robin(kv_request_counter[kv_rank], dram, "kv", batch_idx=i, chip_idx=-1))
                
                v_addr_offset[i][layers][heads] = addr_offset
                #logger.debug(f"Addr offset for v batch {i} layer {layers} head {heads} is {hex(addr_offset)}")
                v_chip_idx[i][layers][heads] = heads % num_chips
                
                # Increment the counter after every group, including a partial last one (kv_heads % num_chips != 0)
                if (heads + 1) % num_chips == 0 or heads == model["kv_heads"] - 1:
                    kv_request_counter[kv_rank] += requests_per_group
    
    # KV temp tensors go in the first Ro row above their rank's KV data, so they never overlap it (the KV data of a
    # rank fills rows 0..last in every channel). The explicit channel/rank allocator pins Ch and Ra, so (AB mode, Ro on
    # top) its request number is Ro * requests_per_row + Co, and it holds temp_capacity requests per (channel, rank)
    temp_requests = math.ceil(model['hdim'] * get_bytes_per_element(model['dtype'], HarmoniTensorType.ACT) / partition_size)  # (1, hdim) ACT tensor
    temp_capacity = requests_before_wrap(dram, ['Ch', 'Ra'])
    requests_per_row = requests_before_wrap(dram, ['Ch', 'Ra', 'Ro'])
    first_free_row = [0] * dram.num_kv_ranks_per_channel
    for rank in range(dram.num_kv_ranks_per_channel):
        if kv_request_counter[rank] > 0:
            last_kv_addr = calculate_interleaving_aware_offset_with_batch_round_robin(kv_request_counter[rank] - 1, dram, "kv", batch_idx=rank, chip_idx=-1)
            first_free_row[rank] = get_mapping(int(last_kv_addr), dram, "kv")['Ro'] + 1
    for channel in range(dram.num_channels):
        for rank in range(dram.num_kv_ranks_per_channel):
            temp_request = first_free_row[rank] * requests_per_row
            check_kv_rank_capacity(args, seq_len, batch, f"temp_tensor_kv_ch{channel}_r{rank} (placed at row {first_free_row[rank]}, the first row above the KV data)",
                                   f"channel {channel} of kv_rank {rank} (Ra {dram.num_wt_ranks_per_channel + rank})", temp_request, temp_requests, temp_capacity)
            # Use explicit channel and rank allocation for KV temp tensors
            temp_addr_offset = int(calculate_interleaving_aware_offset_with_explicit_channel_rank(
                temp_request, dram, "kv", channel, dram.num_wt_ranks_per_channel + rank, -1))
            temp_tensor_kv.append(HarmoniTensor(
                name=f"temp_tensor_kv_ch{channel}_r{rank}", 
                tag=HarmoniTensorType.ACT, 
                #precision=DataType.BF16, 
                precision=model['dtype'], 
                shape=(1, model['hdim']), 
                addr_offset=temp_addr_offset, 
                mapping=get_mapping(int(temp_addr_offset), dram, "kv")))
            #logger.info(f"Temporary tensor allocation in kv ranks channel {channel} rank {rank}: {temp_tensor_kv[-1]}")
            # temp_requests (used in the capacity check above) must match the tensor's real size
            assert math.ceil(temp_tensor_kv[-1].size/partition_size) == temp_requests
    
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
