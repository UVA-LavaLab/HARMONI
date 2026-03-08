import networkx as nx
from misc.type import *
from modeling.core.tensor import *
from modeling.core.transformer_dfg import DFG
from utils.logging_util import logger
from args import get_args
from modeling.core.dram_info import *
from modeling.perf.pu_latency_energy import *


def ops_per_element():
    """
    Returns a dictionary mapping NodeType enums to their operations per element.
    These values represent the number of floating point operations needed per element
    for each operation type.
    """
    ops = {
        NodeType.GEMV: 2,        # 2 ops per element (multiply-add)
        NodeType.GEMM: 2,        # 2 ops per element (multiply-add)
        NodeType.SOFTMAX: 5,     # 5 ops per element (exp, sum, div)
        NodeType.RMSNORM: 3,     # 3 ops per element (square, mean, div)
        NodeType.SILU: 2,        # 2 ops per element (multiply, sigmoid)
        NodeType.RESADD: 1,      # 1 op per element (add)
        NodeType.CONCAT: 0,      # 0 ops (just memory movement)
        NodeType.BROADCAST: 0,   # 0 ops (just memory movement)
        NodeType.REDUCTION: 1,   # 1 op per element (add)
        NodeType.ROTARY: 4,      # 4 ops per element (sin, cos, multiply)
        NodeType.SCALE: 1,       # 1 op per element (multiply)
        NodeType.EWMUL: 1,       # 1 op per element (multiply)
        NodeType.SPLIT: 0,       # 0 ops (just memory movement)
        NodeType.APPEND: 0,      # 0 ops (just memory movement)
        NodeType.SHARD: 0,       # 0 ops (just memory movement)
        NodeType.LOOKUP: 1,      
        NodeType.ARGMAX: 1,       # 1 op per element (compare)
        NodeType.AGG: 1,       
        NodeType.SYNC: 0,
        NodeType.FUSED_SCORE: 8, #GEMV + SOFTMAX + SCALE       
    }
    return ops

def peak_throughput(dram):
    """
    Creates a dictionary mapping NodeType enums to their peak throughput in FLOPS/sec.
    These are ideal values.    
    """
    nbanks = dram.total_banks #per chip
    bank_intrfc = dram.bank_interface #bits
    bits_per_elem = 16 #CLEANUP: parameterize with precision types in type.py
    clk_frequency = 1e9 #Hz
    SIMD_throughput = nbanks * (bank_intrfc/bits_per_elem) * clk_frequency
    SA_throughput = math.pow(nbanks * (bank_intrfc/bits_per_elem),2) * clk_frequency #assuming square systolic array (#CLEANUP: use systolic height)
    bin_tree_throughput = math.log2((nbanks * (bank_intrfc/bits_per_elem))) * clk_frequency

    peak_throughput = {
        NodeType.GEMV: SIMD_throughput,      
        NodeType.GEMM: SA_throughput,      
        NodeType.SOFTMAX: SIMD_throughput,   
        NodeType.RMSNORM: SIMD_throughput,   
        NodeType.SILU: SIMD_throughput,      
        NodeType.RESADD: SIMD_throughput,    
        NodeType.CONCAT: SIMD_throughput,    
        NodeType.BROADCAST: 0e9,  
        NodeType.REDUCTION: bin_tree_throughput,  
        NodeType.ROTARY: SIMD_throughput,    
        NodeType.SCALE: SIMD_throughput,     
        NodeType.EWMUL: SIMD_throughput,     
        NodeType.SPLIT: 0e9,     
        NodeType.APPEND: 0e9,    
        NodeType.SHARD: 0e9,      
        NodeType.LOOKUP: SIMD_throughput, #arbitrary      
        NodeType.ARGMAX: SIMD_throughput,      
        NodeType.AGG: SIMD_throughput,      
        NodeType.SYNC: 0e9,
        NodeType.FUSED_SCORE: SIMD_throughput, #temp dummy      
    }

    return peak_throughput #Flops/sec

def get_node_metrics(DFG, dram, filename_str=''):
    """
    Get the FLOPS, byte_accessed, and Operational Intensity, mem_access, comp_time, and exec_time per node
    """
    ops = ops_per_element()

    BW = dram.calculate_center_strip_BW() #[tCCDs, tCCDl, tRC]
    tCCDs = dram.tCCDs
    tCCDl = dram.tCCDl
    tRC = dram.tRC
    avg_BW = (dram.csl_lines * dram.bank_interface)/(tCCDs*dram.csl_lines+tRC)

    args = get_args()

    for node, attributes in DFG.graph.nodes(data=True):
        flops = 0
        bytes_accessed = 0
        OI = 1
        comp_time = 0
        ideal_comp_time_tCCDs = 0
        ideal_comp_time_tavg = 0
        ideal_comm_time = 0
        ideal_mem_time = 0
        ideal_energy = 0
        node_type = DFG.graph.nodes[node]['type']
        
        node_throughput = peak_throughput(dram)[node_type]
        ops_per_elem = ops[node_type]

        PU_latency = 0
        Energy = 0

        if("ip" in attributes and "op" in attributes):
            #based on the type of node and using both ip and op determine dimensions (m,n,k)
            for tensor_list in (attributes['ip'], attributes['op']):
                for tensor in tensor_list:

                    if len(tensor.shape) > 3:
                        logger.error("How to identify M,N,K")
                        exit(0)
                    if node_type == NodeType.AGG: 
                        bytes_accessed +=0
                        ideal_mem_time +=0
                    else:
                        bytes_accessed += tensor.size
                        
                        #NOTE: Hack for row-buffer hits
                        if tensor.row_accesses == 0:
                            ideal_mem_time = 0 #NOTE: assuming SRAM/eDRAM access is free
                        else:
                            rows = max(tensor.row_accesses-1,0)
                            cols = tensor.col_accesses
                            ideal_mem_time += ((rows * tRC) + (cols * tCCDs)) #ns
                            ideal_energy = memory_access_energy(rows, cols, dram)
                        #logger.info(f"Tensor inside get node metrics {tensor} for node {node}")
                        if tensor.locations != None:
                            if len(tensor.locations) != 1:
                                print(f"[WARNING] Tensor {tensor} in node {node} has more than one location")
                        # else:
                        #     print(f"[INFO] Tensor {tensor.name} of size {tensor.size} in node {node} accessed from Buffer in {DFG.graph.nodes[node]['logic_unit']}")

            if node_type == NodeType.GEMM: 
                if(len(attributes['ip'])!=2):
                    logger.error(f"{node} does not have 2 input tensors {attributes['ip']}")
                    exit(0)
                red_dim = attributes['ip'][0].shape[-1] #k common dimension in the input tensors
                assert red_dim == attributes['ip'][1].shape[0], f"K Dimension mismatch for node {node}- Shapes of Input 1 {attributes['ip'][0].shape} and Input 2 {attributes['ip'][1].shape}" 
                m = 1
                for dim in attributes['ip'][0].shape[:-1]:
                    m *= dim #product of all dimension of the tensor shape list but last
                n = attributes['op'][0].shape[-1] 
                assert n == attributes['ip'][1].shape[-1], f"N Dimension mismatch for node {node}- Shapes of Input 2 {attributes['ip'][1].shape} and Output 1 {attributes['op'][0].shape}"  
                
                # ---- NEW: GEMM trace bookkeeping ----
                # Store GEMM dimensions even if we later retag this node as GEMV
                # or tile it over a systolic array.
                DFG.graph.nodes[node]['gemm_m'] = int(m)
                DFG.graph.nodes[node]['gemm_n'] = int(n)
                DFG.graph.nodes[node]['gemm_k'] = int(red_dim)
                
                
                if m == 1:
                    #change node type to GEMV
                    DFG.graph.nodes[node]['type'] = NodeType.GEMV
                    ops_per_elem = ops[NodeType.GEMV]
                    PU_latency, Energy, energy_stats = GEMV_latency_energy(n, red_dim, dram)
                    # Store energy breakdown in node attributes
                    nx.set_node_attributes(DFG.graph, {node: energy_stats}, name="energy_breakdown")

                else:
                    if args.no_systolic_array:
                        #change node type to GEMV
                        DFG.graph.nodes[node]['type'] = NodeType.GEMV
                        ops_per_elem = ops[NodeType.GEMV]
                        
                        latency, energy, energy_stats = GEMV_latency_energy(n, red_dim, dram)
                        PU_latency = m * latency
                        Energy = m * energy
                        # Aggregate energy breakdown for multiple iterations
                        aggregated_energy_stats = {}
                        for resource, value in energy_stats.items():
                            aggregated_energy_stats[resource] = value * m
                        nx.set_node_attributes(DFG.graph, {node: aggregated_energy_stats}, name="energy_breakdown")
                       
                    else:
                        num_iter = math.ceil(m / args.systolic_height)
                        latency, energy, energy_stats = IS_GEMM_latency_energy(n, red_dim, dram, args.systolic_height)
                        PU_latency = num_iter * latency
                        Energy = num_iter * energy
                        # Aggregate energy breakdown for multiple iterations
                        aggregated_energy_stats = {}
                        for resource, value in energy_stats.items():
                            aggregated_energy_stats[resource] = value * num_iter
                        nx.set_node_attributes(DFG.graph, {node: aggregated_energy_stats}, name="energy_breakdown")

                #FLOPS calculation
                flops = ops_per_elem * m * n * red_dim
            
            if node_type == NodeType.FUSED_SCORE:
                m = 1
                for dim in attributes['ip'][0].shape[:-1]:
                    m *= dim
                red_dim = attributes['ip'][0].shape[-1]
                assert red_dim == attributes['ip'][1].shape[0], f"Reduction dimension for node {node} mismatch, Input 1 shape {attributes['ip'][0].shape}, Input 2 shape {attributes['ip'][1].shape}"
                vec_len = attributes['ip'][1].shape[-1] #Seq_len block
                if not attributes['op']:
                    logger.error(f"Node {node} of type {node_type} has empty 'op' list!")
                    exit()
                    #continue  # or handle appropriately
                assert vec_len == attributes['op'][0].shape[-1], f"Block of seq_len expected, input-output shape mismatch, last dims of Input {vec_len}, Output {attributes['op'][0].shape[-1]}"

                if m == 1:
                    gemv_latency, gemv_energy, gemv_energy_stats = GEMV_latency_energy(vec_len, red_dim, dram)
                    softmax_latency, softmax_energy, softmax_energy_stats = SOFTMAX_latency_energy(vec_len, dram)
                    PU_latency = gemv_latency + softmax_latency
                    Energy = gemv_energy + softmax_energy
                    # Aggregate energy breakdown
                    combined_energy_stats = {}
                    for resource in gemv_energy_stats:
                        combined_energy_stats[resource] = gemv_energy_stats[resource] + softmax_energy_stats[resource]
                    nx.set_node_attributes(DFG.graph, {node: combined_energy_stats}, name="energy_breakdown")
                else: 
                    if args.no_systolic_array:
                        gemv_latency, gemv_energy, gemv_energy_stats = GEMV_latency_energy(vec_len, red_dim, dram)
                        softmax_latency, softmax_energy, softmax_energy_stats = SOFTMAX_latency_energy(vec_len, dram)
                        PU_latency = m * (gemv_latency + softmax_latency)
                        Energy = m * (gemv_energy + softmax_energy)
                        # Aggregate energy breakdown for multiple iterations
                        combined_energy_stats = {}
                        for resource in gemv_energy_stats:
                            combined_energy_stats[resource] = m * (gemv_energy_stats[resource] + softmax_energy_stats[resource])
                        nx.set_node_attributes(DFG.graph, {node: combined_energy_stats}, name="energy_breakdown")
                    else:
                        num_iter = math.ceil(m / args.systolic_height)
                        gemm_latency, gemm_energy, gemm_energy_stats = IS_GEMM_latency_energy(vec_len, red_dim, dram, args.systolic_height)
                        softmax_latency, softmax_energy, softmax_energy_stats = SOFTMAX_latency_energy(vec_len, dram)
                        PU_latency = num_iter * gemm_latency + m * softmax_latency
                    
                        Energy = num_iter * gemm_energy + m * softmax_energy
                        # Aggregate energy breakdown for multiple iterations
                        combined_energy_stats = {}
                        for resource in gemm_energy_stats:
                            combined_energy_stats[resource] = num_iter * gemm_energy_stats[resource] + m * softmax_energy_stats[resource]
                        nx.set_node_attributes(DFG.graph, {node: combined_energy_stats}, name="energy_breakdown")
                        
                flops = ops_per_elem * m * vec_len * red_dim

            if node_type in [NodeType.RMSNORM, NodeType.SOFTMAX, NodeType.SCALE, NodeType.SILU, NodeType.ROTARY]:
                assert (len(attributes['ip']) == len(attributes['op']) == 1), f"Expected exactly one input and one output tensor for node {node}, but got {len(attributes['ip'])} inputs and {len(attributes['op'])} outputs"
                assert attributes['ip'][0].shape == attributes['op'][0].shape, f"Input and output tensor shapes do not match for node {node}"
                m = 1
                for dim in attributes['ip'][0].shape[:-1]:
                    m *= dim
                vec_len = attributes['ip'][0].shape[-1]

                #FLOPS calcution
                flops = ops_per_elem * m * vec_len

                if node_type == NodeType.SOFTMAX:
                    latency, energy, energy_stats = SOFTMAX_latency_energy(vec_len, dram)
                    PU_latency = m * latency
                    Energy = m * energy
                    # Aggregate energy breakdown for multiple iterations
                    aggregated_energy_stats = {}
                    for resource, value in energy_stats.items():
                        aggregated_energy_stats[resource] = value * m
                    nx.set_node_attributes(DFG.graph, {node: aggregated_energy_stats}, name="energy_breakdown")
                elif node_type == NodeType.RMSNORM:
                    PU_latency, Energy, energy_stats = RMSNorm_latency_energy(m * vec_len, dram)
                    nx.set_node_attributes(DFG.graph, {node: energy_stats}, name="energy_breakdown")
                elif node_type == NodeType.SILU:
                    PU_latency, Energy, energy_stats = SiLU_latency_energy(m * vec_len, dram)
                    nx.set_node_attributes(DFG.graph, {node: energy_stats}, name="energy_breakdown")
                elif node_type == NodeType.ROTARY:
                    PU_latency, Energy, energy_stats = Rotary_latency_energy(m * vec_len, dram)
                    nx.set_node_attributes(DFG.graph, {node: energy_stats}, name="energy_breakdown")
            
            if node_type == NodeType.ARGMAX:
                m = 1
                for dim in attributes['ip'][0].shape[:-1]:
                    m *= dim
                vec_len = attributes['ip'][0].shape[-1]
                PU_latency, Energy, energy_stats = ARGMAX_latency_energy(m * vec_len, dram)
                nx.set_node_attributes(DFG.graph, {node: energy_stats}, name="energy_breakdown")

            #CLEANUP: Add energy for EWMUL, RESADD, LOOKUP
            if node_type in [NodeType.EWMUL, NodeType.RESADD]:
                assert (len(attributes['ip']) == 2), f"Expected two input tensors for node {node}, but got {len(attributes['ip'])} inputs"
                assert (len(attributes['op']) == 1), f"Expected one output tensors for node {node}, but got {len(attributes['op'])} outputs"
                assert attributes['ip'][0].shape == attributes['op'][0].shape, f"Input and output tensor shapes do not match for node {node}"
                m = 1
                for dim in attributes['ip'][0].shape[:-1]:
                    m *= dim
                vec_len = attributes['ip'][0].shape[-1]
            
                #FLOPS calcution
                flops = ops_per_elem * m * vec_len
                #CLEANUP: redo based on the where the node is getting mapped to.
                if node_type == NodeType.EWMUL:
                    PU_latency, Energy, energy_stats = SIMD_multiplier_latency_energy(m*vec_len, dram)
                    nx.set_node_attributes(DFG.graph, {node: energy_stats}, name="energy_breakdown")
                if node_type == NodeType.RESADD:
                    PU_latency, Energy, energy_stats = SIMD_adder_latency_energy(m*vec_len, dram)
                    nx.set_node_attributes(DFG.graph, {node: energy_stats}, name="energy_breakdown")
            
            if node_type in [NodeType.LOOKUP]:
            #else: #all other node types LOOKUP, REDUCTION, AGG
                total_elements = 0
                for tensor in attributes['ip']:
                    for dim in tensor.shape[:-1]:
                        total_elements *= dim 
                flops = ops_per_elem * total_elements
                PU_latency, Energy, energy_stats = SIMD_adder_latency_energy(total_elements, dram)
                nx.set_node_attributes(DFG.graph, {node: energy_stats}, name="energy_breakdown")

        else:
            logger.warning(f"{node} does not have input and output tensors assigned")

        nx.set_node_attributes(DFG.graph, {node: flops}, name="flops")
        nx.set_node_attributes(DFG.graph, {node: bytes_accessed}, name="bytes_accessed")
        if flops !=0 and bytes_accessed>0:
            OI = flops/bytes_accessed
        nx.set_node_attributes(DFG.graph, {node: OI}, name="op_intensity")

        if OI != 0:
            #ideal_comp_time = 1/(OI*BW)
            if flops:
                time_from_bw = (flops/OI) / BW[0]  # BW is in GB/s, so result is in seconds
                time_from_throughput = flops / node_throughput  # node_throughput is in FLOPS/sec
                
                ideal_comp_time_tCCDs = max(time_from_bw, time_from_throughput) * 1e9  # Convert to ns
                
                time_from_avg_bw = (flops/OI) / avg_BW  # avg_BW is in GB/s
                ideal_comp_time_tavg = max(time_from_avg_bw, time_from_throughput) * 1e9  # Convert to ns

                # This represents the latency of the entire kernel, including the memory
                # access latency and the compute latency
                if PU_latency == 0:
                    comp_time = time_from_throughput * 1e9  # Convert to ns
                else:
                    comp_time = PU_latency # ns

        nx.set_node_attributes(DFG.graph, {node: comp_time/1000}, name="peak_comp_time") #us
        nx.set_node_attributes(DFG.graph, {node: ideal_comp_time_tCCDs/1000}, name="tCCDs_comp_time") #us
        nx.set_node_attributes(DFG.graph, {node: ideal_comp_time_tavg/1000}, name="tavg_comp_time") #us
        nx.set_node_attributes(DFG.graph, {node: ideal_comm_time/1000}, name="comm_time") #us
        nx.set_node_attributes(DFG.graph, {node: ideal_mem_time/1000}, name="mem_time") #us
        if ideal_mem_time < 0:
            logger.error(f"Mem time for node {node} is negative")
            #logger.warning(f"tensors of this node are {DFG.graph.nodes[node]['ip']} and {DFG.graph.nodes[node]['op']}")
        
        exec_time = comp_time/1000 #us

        nx.set_node_attributes(DFG.graph, {node: exec_time}, name="exec_time") #us
        
        if Energy != 0:  # Only convert if energy is not zero
            Energy = Energy / 1e9  # Convert pJ to mJ
        
        nx.set_node_attributes(DFG.graph, {node: Energy}, name="exec_energy") #mJ

def get_gemm_trace(DFG):
    """
    Collect GEMM dimension info for all nodes where get_node_metrics()
    has recorded GEMM dimensions.

    Returns:
        List[Tuple[str, int, int, int, class]]: (node_name, M, N, K, logicUnit)
    """
    trace = []
    for node, attrs in DFG.graph.nodes(data=True):
        if (
            'gemm_m' in attrs
            and 'gemm_n' in attrs
            and 'gemm_k' in attrs
        ):
            trace.append((node, attrs['gemm_m'], attrs['gemm_n'], attrs['gemm_k'], attrs['logic_unit']))
    return trace
