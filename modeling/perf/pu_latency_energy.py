import math
from args import parse_args
from modeling.core.memory_system import memsys
from modeling.hardware.power_calculator import get_power_constants
from utils.logging_util import logger

# systolic array configuration
num_banks_per_systolic_array = 1 

# Number of bits in a data element
Data_width = 16 
#NOTE: use this: #get_bytes_per_element(args.dtype, HarmoniTensorType.ACT), check type.py for datatype enums, dtype is part of args = parse_args()
is_PNM = True

# Compute unit configuration
compute_CC = 1 # Assuming 1GHz for compute units, clock cycle = 1ns
ADD_cycle = 2 # assuming 2 cycles for each addition
MUL_cycle = 3 # assuming 3 cycles for each multiplication

# Energy breakdown tracking functions
def create_energy_stats():
    """Create a dictionary to track energy consumption by resource type"""
    return {
        'systolic_array': 0.0,
        'sram_buffer': 0.0, 
        'adder_tree': 0.0,
        'accumulator': 0.0,
        'dram_background': 0.0,
        'dram_read': 0.0,
        'dram_write': 0.0,
        'dram_activate': 0.0,
        'simd_multiplier': 0.0,
        'simd_adder': 0.0,
        'max_tree': 0.0,
        'SOFTMAX_unit': 0.0,
        'actv_unit': 0.0,
        'RMSNorm_unit': 0.0,
        'center_stripe_logic': 0.0,
        'dram_chiplet': 0.0,
        'comm': 0.0 
    }

def aggregate_energy_stats(energy_stats_list):
    """Aggregate multiple energy stats dictionaries"""
    if not energy_stats_list:
        return create_energy_stats()
    
    aggregated = create_energy_stats()
    for stats in energy_stats_list:
        for resource, energy in stats.items():
            aggregated[resource] += energy
    
    return aggregated

#TODO: use dram config to determine STD
STD = "GDDR6" 

if STD == "DDR5":
    #NOTE: We only have Voltage and current values for DDR5
    # https://github.com/CMU-SAFARI/ramulator2/blob/main/src/dram/impl/DDR5.cpp
    # V * mA * ns = pJ
    VDD = 1.1 # V
    IDD0 = 60 # mA
    IDD2N = 50 # mA
    IDD3N = 55 # mA
    IDD4R = 145 # mA
    IDD4W = 145 # mA
elif STD == "GDDR6":
    #Did not find values for 16Gb GDDR6 (so scaling accordingly with reference to 8Gb GDDR6: https://github.com/umd-memsys/DRAMsim3/blob/master/configs/GDDR6_8Gb_x16.ini)
    VDD = 1.35 # V
    IDD0 = 1196 # mA (Act/Pre: +15% larger row decoders)
    IDD2N = 838 # mA (Pre Standby: +25% higher transistor leakage)
    IDD3N = 1437 # mA (Act Standby: +25% higher transistor leakage)
    IDD4R = 3100 # mA (Burst read: +5% mostly IO bound)
    IDD4W = 3320 # mA (Burst write: +5% mostly IO bound)

# https://github.com/umd-memsys/DRAMsim3/blob/29817593b3389f1337235d63cac515024ab8fd6e/src/configuration.cc#L203
def act_energy_per_bank(dram):
    # Energy per chip per active operation
    act_energy_per_bank = VDD * (IDD0 * dram.tRC - (IDD3N * dram.tRAS + IDD2N * dram.tRP)) #background power doesn't need to be multplied by the number of banks (IDD3N is already accounting for all bank active current). But it also doesn't effect by the 0.34% ratio, so we separate it our from rea/write/act calculation)
    #act_energy_per_bank = VDD * (IDD0 * dram.tRC) #adding background energy
    return act_energy_per_bank

def read_energy_per_bank(dram):
    # Energy per chip per read operation
    # SIO, LIO, and GIO: 31%, column operations: 20%, CSL: 10%, cite 1810_Ha_DRAMEnergy
    #read_energy_per_bank = VDD * (IDD4R - IDD3N) * BL * (0.31 + 0.1 + 0.2)
    read_energy_per_bank = VDD * (IDD4R - IDD3N) * dram.tCCDs * (0.2 + 0.1 + 0.04) 
    #read_energy_per_bank = VDD * IDD4R * dram.tCCDs * (0.2 + 0.1 + 0.04) # adding background energy
    return read_energy_per_bank

def write_energy_per_bank(dram):
    # Energy per chip per write operation
    #write_energy_per_bank = VDD * (IDD4W - IDD3N) * BL * (0.31 + 0.1 + 0.2)
    write_energy_per_bank = VDD * (IDD4W - IDD3N) * dram.tCCDs * (0.2 + 0.1 + 0.04) 
    #write_energy_per_bank = VDD * IDD4W * dram.tCCDs * (0.2 + 0.1 + 0.04) # adding background energy
    return write_energy_per_bank


def memory_access_energy(rows, cols, dram):
    read_row_energy = cols * read_energy_per_bank(dram) + act_energy_per_bank(dram)
    total_read_energy = read_row_energy * rows
    return total_read_energy

def IS_GEMM_latency_energy(N, K, dram, systolic_height):
    systolic_width = dram.bank_interface // Data_width
    N_per_chip = N
    adder_tree_width_per_chip = dram.total_banks 
    page_size = dram.csl_lines * dram.bank_interface

    num_DRAM_rows_per_chunk = math.floor(N_per_chip * Data_width * systolic_width / page_size)
    num_tail_cols = math.ceil((N_per_chip * Data_width * systolic_height - (num_DRAM_rows_per_chunk * page_size)) / dram.bank_interface)

    #NOTE: Pessimistic approach to consider the entire row
    W_latency = (dram.csl_lines * dram.tCCDs + dram.tRC) * num_DRAM_rows_per_chunk + (num_tail_cols * dram.tCCDs + dram.tRC)

    C = (dram.bank_interface/Data_width)
    chunk_latency = C + W_latency + systolic_height
    
    # Number of systolic array per conventional DRAM chip
    NSA_per_DRAMchip = dram.total_banks / num_banks_per_systolic_array 
    num_chunks = math.ceil(K / (NSA_per_DRAMchip * systolic_width))

    GEMM_latency = num_chunks * chunk_latency + math.log2(adder_tree_width_per_chip) * ADD_cycle * compute_CC + ADD_cycle * compute_CC

    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['dram_background'] = num_chunks * chunk_latency * VDD * IDD3N
    energy_stats['dram_read'] = (dram.csl_lines * num_DRAM_rows_per_chunk + num_tail_cols) * num_chunks * dram.total_banks * read_energy_per_bank(dram)
    energy_stats['dram_activate'] = act_energy_per_bank(dram) * math.ceil(N_per_chip * Data_width * systolic_height / page_size) * num_chunks * dram.total_banks
    energy_stats['sram_buffer'] = get_power_constants()['SRAM_power'] * GEMM_latency
    energy_stats['adder_tree'] = get_power_constants()['adder_tree'].get(systolic_width, 0) * dram.total_banks * GEMM_latency
    energy_stats['accumulator'] = get_power_constants()['accumulator_power'] * GEMM_latency

    SA_power = get_power_constants()['systolic'][(C, systolic_height, 100)]
    energy_stats['systolic_array'] = SA_power * GEMM_latency * dram.total_banks

    GEMM_energy = sum(energy_stats.values())

    return GEMM_latency, GEMM_energy, energy_stats


def GEMV_latency_energy(N, K, dram):
    # we partition K/V column-wise into all banks in a DRAM chip
    page_size = dram.csl_lines * dram.bank_interface
    # Assuming we use GEMV unit to do GEMV
    N_per_bank = math.ceil(N / dram.total_banks)
    num_DRAM_rows_per_bank = math.floor(N_per_bank * Data_width * K / page_size)
    num_tail_cols = math.ceil((N_per_bank*Data_width*K - (num_DRAM_rows_per_bank * page_size)) / dram.bank_interface)

    #integer-safe version
    # total_bits_per_bank = N_per_bank * DATA_WIDTH * K
    # num_DRAM_rows_per_bank = total_bits_per_bank // page_size
    # rem_bits = total_bits_per_bank % page_size
    # num_tail_cols = (rem_bits + dram.bank_interface - 1) // dram.bank_interface

    adder_tree_width_per_chip = dram.total_banks

    W_latency = (dram.csl_lines * dram.tCCDs + dram.tRC) * num_DRAM_rows_per_bank + (num_tail_cols * dram.tCCDs + dram.tRC)

    GEMV_latency = W_latency + math.log2(adder_tree_width_per_chip) * ADD_cycle * compute_CC + ADD_cycle * compute_CC

    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    
    energy_stats['dram_background'] = W_latency * VDD * IDD3N
    energy_stats['dram_read'] = (dram.csl_lines * num_DRAM_rows_per_bank + num_tail_cols)* dram.total_banks * read_energy_per_bank(dram)
    energy_stats['dram_activate'] = act_energy_per_bank(dram) * math.ceil(N_per_bank * Data_width * K / page_size) * dram.total_banks
    energy_stats['sram_buffer'] = get_power_constants()['SRAM_power'] * GEMV_latency
    energy_stats['simd_multiplier'] = get_power_constants()['SIMD_multiplier_power'] * dram.total_banks * GEMV_latency
    energy_stats['adder_tree'] = get_power_constants()['adder_tree'].get(dram.total_banks, 0) * GEMV_latency
    energy_stats['simd_adder'] = get_power_constants()['SIMD_adder_power'] * dram.total_banks * GEMV_latency

    GEMV_energy = sum(energy_stats.values())
    
    return GEMV_latency, GEMV_energy, energy_stats


def SOFTMAX_latency_energy(L, dram):
    input_lane = 32 
    exp_cycle = 23
    exp_lane = 32
    max_tree_chip_width = 32
    sum_tree_chip_width = 32
    sub_lane = 32
    reciprocal_cycle = 41
    adder_cycle = 8
    SOFTMAX_cycle = math.ceil (L/input_lane) + math.log2(max_tree_chip_width) + math.ceil(L/sub_lane) + math.ceil (L/exp_lane) + exp_cycle + math.log2(max_tree_chip_width) +  reciprocal_cycle + (L/input_lane) + adder_cycle
    
    SOFTMAX_latency = SOFTMAX_cycle * compute_CC
    
    SOFTMAX_power = get_power_constants()['SOFTMAX_unit_power']
    SRAM_power = get_power_constants()['SRAM_power']

    SOFTMAX_energy  = SOFTMAX_latency * (SOFTMAX_power + SRAM_power) * 2 * math.ceil ( L / input_lane) * compute_CC

    # Create energy stats dictionary
    energy_stats = create_energy_stats()
    energy_stats['SOFTMAX_unit'] = SOFTMAX_power * SOFTMAX_latency
    energy_stats['sram_buffer'] = SRAM_power * SOFTMAX_latency
    
    SOFTMAX_energy = sum(energy_stats.values())
    return SOFTMAX_latency, SOFTMAX_energy, energy_stats


def GeLU_latency_energy(L, dram):

    actv_lane = 16
    actv_cycle = 14
    GeLU_cycle = math.ceil ( L / actv_lane) + actv_cycle
    GeLU_latency = GeLU_cycle * compute_CC
    GeLU_energy =  GeLU_latency * get_power_constants()['actv_unit_power']

    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['actv_unit'] = get_power_constants()['actv_unit_power'] * GeLU_latency
    GeLU_energy = sum(energy_stats.values())

    return GeLU_latency, GeLU_energy, energy_stats


def RMSNorm_latency_energy(L, dram):

    num_DRAMchip = dram.num_chips_per_rank
    chip_L = L / num_DRAMchip
    input_lane = 16
    adder_tree_chip_width = input_lane
    adder_tree_CXL_ctrl_width = 64
    mul_cycle = 3 # +1 +1 is multiply sum by 1/N and reciprocal square root look-up (pipelined)
    RMSnorm_cycle =  math.ceil ( chip_L / input_lane) +  math.log2(adder_tree_chip_width) + math.log2( L / adder_tree_CXL_ctrl_width) + 1 + 1 + math.ceil ( chip_L / input_lane) + mul_cycle
    RMSNorm_latency = RMSnorm_cycle * compute_CC
    RMSNorm_power = get_power_constants()['RMSNorm_unit_power']
    SRAM_power = get_power_constants()['SRAM_power']
    RMSNorm_energy = RMSNorm_latency * (RMSNorm_power + SRAM_power) *  math.ceil ( chip_L / input_lane)
    
    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['RMSNorm_unit'] = RMSNorm_power * RMSNorm_latency
    energy_stats['sram_buffer'] = SRAM_power * RMSNorm_latency

    RMSNorm_energy = sum(energy_stats.values())
    return RMSNorm_latency, RMSNorm_energy, energy_stats

def SiLU_latency_energy(L, dram): 
    num_DRAMchip = dram.num_chips_per_rank 
    actv_lane = 16
    actv_cycle = 14
    chip_L = math.ceil ( L / num_DRAMchip) #hack
    SiLU_cycle = math.ceil ( chip_L / actv_lane) + actv_cycle
    SiLU_latency = SiLU_cycle * compute_CC
    actv_power = get_power_constants()['actv_unit_power']
    SiLU_energy =  SiLU_latency * actv_power
    
    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['actv_unit'] = actv_power * SiLU_latency

    SiLU_energy = sum(energy_stats.values())

    return SiLU_latency, SiLU_energy, energy_stats

def Rotary_latency_energy(L, dram):

    # We pre-compute sin(theta) and cos(theta) and store them in DRAM
     
    simd_mul_cycle = 3
    simd_mul_width = dram.bank_interface/Data_width
    RoPE_cycle = 2 * (math.ceil(L/simd_mul_width)) + simd_mul_cycle
    Rotary_latency = RoPE_cycle * compute_CC
    simd_mul_power = get_power_constants()['SIMD_multiplier_power']
    simd_add_power = get_power_constants()['SIMD_adder_power']
    SRAM_power = get_power_constants()['SRAM_power']
    Rotary_energy =  Rotary_latency * (simd_mul_power + simd_add_power + SRAM_power)
    
    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['simd_multiplier'] = get_power_constants()['SIMD_multiplier_power'] * Rotary_latency
    energy_stats['simd_adder'] = get_power_constants()['SIMD_adder_power'] * Rotary_latency
    energy_stats['sram_buffer'] = SRAM_power * Rotary_latency

    Rotary_energy = sum(energy_stats.values())

    return Rotary_latency, Rotary_energy, energy_stats

def ARGMAX_latency_energy(L, dram):

    num_DRAMchip = dram.num_chips_per_rank #hack
    chip_L = math.ceil(L/num_DRAMchip)
    max_tree_chip_width = 32
    max_tree_CXL_ctrl_width = 64
    argmax_cycle = math.ceil (chip_L / max_tree_chip_width) + math.log2(max_tree_chip_width) + math.log2(max_tree_CXL_ctrl_width)

    ARGMAX_latency = argmax_cycle * compute_CC 
    
    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['max_tree'] = get_power_constants()['max_tree_power'] * num_DRAMchip * ARGMAX_latency

    ARGMAX_energy = sum(energy_stats.values())

    return ARGMAX_latency, ARGMAX_energy, energy_stats

def SIMD_adder_latency_energy(L, dram): 
    cycles = L/dram.total_banks/dram.bank_interface/Data_width * ADD_cycle
    latency = cycles * compute_CC
    energy_stats = create_energy_stats()
    energy_stats['simd_adder'] = (dram.total_banks * get_power_constants()['SIMD_adder_power']) * latency
    return latency, sum(energy_stats.values()), energy_stats

def SIMD_multiplier_latency_energy(L, dram):
    cycles = L/dram.total_banks/dram.bank_interface/Data_width * MUL_cycle
    latency = cycles * compute_CC
    energy_stats = create_energy_stats()
    energy_stats['simd_multiplier'] = (dram.total_banks * get_power_constants()['SIMD_multiplier_power']) * latency
    return latency, sum(energy_stats.values()), energy_stats
