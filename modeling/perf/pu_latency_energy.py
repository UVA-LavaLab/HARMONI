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

# Compute unit configuration
compute_CC = 1 # Assuming 1GHz for compute units, clock cycle = 1ns
ADD_cycle = 1 # assuming 1 cycles for each addition
MUL_cycle = 2 # assuming 2 cycles for each multiplication
EXP_cycle = 12 # assuming 12 cycle for each exponentiation
COMP_cycle = 1 # assuming 1 cycle for each comparison
DIV_cycle = 8 # assuming 8 cycle for each division
SQRT_cycle = 11 # assuming 11 cycle for each square root

# Global config
max_tree_width_per_chip = 64 # assuming a 64to1 softmax unit per chip
adder_tree_width_per_rank = 128 # assuming a 128to1 adder tree unit per rank
exp_width_per_chip = 32 # assuming a 32-lane SIMD exp unit per chip 
adder_width_per_chip = 32 # assuming a 32-lane SIMD adder unit per chip

# Energy breakdown tracking functions
def create_energy_stats():
    """Create a dictionary to track energy consumption by resource type"""
    return {
        'systolic_array': 0.0,
        'sram_buffer': 0.0, 
        'adder_tree': 0.0,
        'dram_read': 0.0,
        'dram_write': 0.0,
        'dram_activate': 0.0,
        'simd_multiplier': 0.0,
        'simd_adder': 0.0,
        'max_tree': 0.0,
        'exp_unit': 0.0,
        'divider_unit': 0.0,
        'sqrt_unit': 0.0,
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

#NOTE: We only have Voltage and current values for DDR5
# https://github.com/CMU-SAFARI/ramulator2/blob/main/src/dram/impl/DDR5.cpp
# V * mA * ns = pJ
VDD = 1.1 # V
IDD0 = 60 # mA
IDD2N = 50 # mA
IDD3N = 55 # mA
IDD4R = 145 # mA
IDD4W = 145 # mA
BL = 8 * 0.625 # ns

# https://github.com/umd-memsys/DRAMsim3/blob/29817593b3389f1337235d63cac515024ab8fd6e/src/configuration.cc#L203
def act_energy_per_bank(dram):
    # Energy per chip per active operation
    #act_energy_per_bank = VDD * (IDD0 * dram.tRC - (IDD3N * dram.tRAS + IDD2N * dram.tRP))
    act_energy_per_bank = VDD * (IDD0 * dram.tRC) #adding background energy
    return act_energy_per_bank

def read_energy_per_bank(dram):
    # Energy per chip per read operation
    # SIO, LIO, and GIO: 31%, column operations: 20%, CSL: 10%, cite 1810_Ha_DRAMEnergy
    #read_energy_per_bank = VDD * (IDD4R - IDD3N) * BL * (0.31 + 0.1 + 0.2)
    #read_energy_per_bank = VDD * (IDD4R - IDD3N) * dram.tCCDs * (0.2 + 0.1 + 0.04) 
    read_energy_per_bank = VDD * IDD4R * dram.tCCDs * (0.2 + 0.1 + 0.04) # adding background energy
    return read_energy_per_bank

def write_energy_per_bank(dram):
    # Energy per chip per write operation
    #write_energy_per_bank = VDD * (IDD4W - IDD3N) * BL * (0.31 + 0.1 + 0.2)
    #write_energy_per_bank = VDD * (IDD4W - IDD3N) * dram.tCCDs * (0.2 + 0.1 + 0.04) 
    write_energy_per_bank = VDD * IDD4W * dram.tCCDs * (0.2 + 0.1 + 0.04) # adding background energy
    return write_energy_per_bank


def memory_access_energy(rows, cols, dram):
    read_row_energy = cols * read_energy_per_bank(dram) + act_energy_per_bank(dram)
    total_read_energy = read_row_energy * rows

def IS_GEMM_latency_energy(N, K, dram, systolic_height):
    N_per_chip = N
    adder_tree_width_per_chip = dram.total_banks 
    page_size = dram.csl_lines * dram.bank_interface

    num_DRAM_rows_per_chunk = math.floor(N_per_chip * Data_width * systolic_height / page_size)
    num_tail_cols = math.ceil((N_per_chip * Data_width * systolic_height - (num_DRAM_rows_per_chunk * page_size)) / dram.bank_interface)

    #NOTE: Pessimistic approach to consider the entire row
    W_latency = (dram.csl_lines * dram.tCCDs + dram.tRC) * num_DRAM_rows_per_chunk + (num_tail_cols * dram.tCCDs + dram.tRC)

    C = (dram.bank_interface/Data_width)
    chunk_latency = C + W_latency + systolic_height
    
    # Number of systolic array per conventional DRAM chip
    NSA_per_DRAMchip = dram.total_banks / num_banks_per_systolic_array 
    num_chunks = math.ceil(K / (NSA_per_DRAMchip * systolic_height))

    GEMM_latency = num_chunks * chunk_latency + math.log2(adder_tree_width_per_chip) * ADD_cycle * compute_CC + ADD_cycle * compute_CC

    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['dram_read'] = (dram.csl_lines * num_DRAM_rows_per_chunk + num_tail_cols) * num_chunks * dram.total_banks * read_energy_per_bank(dram)
    energy_stats['dram_activate'] = act_energy_per_bank(dram) * math.ceil(N_per_chip * Data_width * systolic_height / page_size) * num_chunks * dram.total_banks
    energy_stats['sram_buffer'] = get_power_constants()['SRAM_power'] * GEMM_latency
    energy_stats['adder_tree'] = get_power_constants()['adder_tree'].get(dram.total_banks, 0) * dram.total_banks * GEMM_latency
    energy_stats['simd_adder'] = get_power_constants()['SIMD_adder_power'] * GEMM_latency

    SA_power = get_power_constants()['systolic'][(C, systolic_height, 250)]#NOTE: for now, only DDR5 is supported (16 * 8 SA), but we can explore energy for other configurations
    energy_stats['systolic_array'] = SA_power * GEMM_latency * dram.total_banks

    GEMM_energy = sum(energy_stats.values())

    return GEMM_latency, GEMM_energy, energy_stats


def GEMV_latency_energy(N, K, dram):
    # we partition K/V column-wise into all banks in a DRAM chip
    page_size = dram.csl_lines * dram.bank_interface
    # Assuming we use GEMV unit to do GEMV
    N_per_bank = math.ceil(N / dram.total_banks)
    num_DRAM_rows_per_bank = math.floor(N_per_bank * Data_width * K / page_size)
    num_tail_cols = math.ceil((N_per_bank*Data_width - (num_DRAM_rows_per_bank * page_size)) / dram.bank_interface)

    adder_tree_width_per_chip = dram.total_banks

    W_latency = (dram.csl_lines * dram.tCCDs + dram.tRC) * num_DRAM_rows_per_bank + (num_tail_cols * dram.tCCDs + dram.tRC)

    GEMV_latency = W_latency + math.log2(adder_tree_width_per_chip) * ADD_cycle * compute_CC + ADD_cycle * compute_CC

    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['dram_read'] = (dram.csl_lines * num_DRAM_rows_per_bank + num_tail_cols)* dram.total_banks * read_energy_per_bank(dram)
    energy_stats['dram_activate'] = act_energy_per_bank(dram) * math.ceil(N_per_bank * Data_width * K / page_size) * dram.total_banks
    energy_stats['sram_buffer'] = get_power_constants()['SRAM_power'] * GEMV_latency
    energy_stats['simd_multiplier'] = get_power_constants()['SIMD_multiplier_power'] * dram.total_banks * GEMV_latency
    energy_stats['adder_tree'] = get_power_constants()['adder_tree'].get(dram.total_banks, 0) * GEMV_latency
    energy_stats['simd_adder'] = get_power_constants()['SIMD_adder_power'] * dram.total_banks * GEMV_latency

    GEMV_energy = sum(energy_stats.values())
    
    return GEMV_latency, GEMV_energy, energy_stats


def SOFTMAX_latency_energy(L, dram):
    # For a 128*2048 GEMV, it takes 2818 ns, assuming 1 GHz for softmax unit, we have 2818 cycles to compute softmax

    # to compute max(z)
    max_tree_cycle = (L/max_tree_width_per_chip + math.log2(max_tree_width_per_chip) + 1) * COMP_cycle

    # to compute z-max(z)
    adder_cycle = ADD_cycle

    # to compute exp(z-max(z))
    exponential_cycle = L/exp_width_per_chip * EXP_cycle

    adder_tree_width_per_chip = dram.total_banks
    # to compute sum(exp(z-max(z)))
    adder_tree_cycle = math.log2(adder_tree_width_per_chip) * ADD_cycle
    
    # to compute 1/sum(exp(z-max(z)))
    div_cycle = DIV_cycle

    mul_width_per_bank = dram.bank_interface/Data_width
    # to compute softmax(z) = exp(z-max(z)) / sum(exp(z-max(z)))
    multiply_cycle = L/(mul_width_per_bank * dram.total_banks) * MUL_cycle

    SOFTMAX_latency = (max_tree_cycle + adder_cycle + exponential_cycle + adder_tree_cycle + div_cycle + multiply_cycle) * compute_CC

    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['max_tree'] = get_power_constants()['max_tree_power'] * max_tree_cycle * compute_CC
    energy_stats['exp_unit'] = get_power_constants()['exp_unit_power'] * exponential_cycle * compute_CC
    energy_stats['divider_unit'] = get_power_constants()['divider_unit_power'] * div_cycle * compute_CC
    energy_stats['simd_multiplier'] = get_power_constants()['SIMD_multiplier_power'] * multiply_cycle * compute_CC
    energy_stats['adder_tree'] = get_power_constants()['adder_tree'].get(dram.total_banks, 0) * exponential_cycle * compute_CC
    energy_stats['simd_adder'] = get_power_constants()['SIMD_adder_power'] * SOFTMAX_latency
    
    SOFTMAX_energy = sum(energy_stats.values())
    return SOFTMAX_latency, SOFTMAX_energy, energy_stats


def GeLU_latency_energy(L, dram):
    # L stands for the total number of elements in the matrix
    # to compute 0.044715 * x^3, mul, mul, mul const
    mul_width_per_bank = dram.bank_interface/Data_width
    num_DRAMchip = dram.num_chips_per_rank 
    multiply_cycle1 = L/num_DRAMchip/(mul_width_per_bank * dram.total_banks) * MUL_cycle * 3

    # to compute x + 0.044715 * x^3
    add_cycle1 = ADD_cycle

    # to compute sqrt(2/pi) * (x + 0.044715 * x^3)
    multiply_cycle2 = MUL_cycle

    # to compute tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    # use tanh(x) = x − x^3/3 + 2x^5/15
    multiply_cycle3 = MUL_cycle * (3 + 5)
    add_cycle3 = ADD_cycle * 2

    # to compute 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    multiply_cycle4 = MUL_cycle * 2
    add_cycle4 = ADD_cycle

    GeLU_latency = (multiply_cycle1 + add_cycle1 + multiply_cycle2 + multiply_cycle3 +
                    add_cycle3 + multiply_cycle4 + add_cycle4) * compute_CC

    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['simd_multiplier'] = get_power_constants()['SIMD_multiplier_power'] * (multiply_cycle1 + multiply_cycle2 + multiply_cycle3 + multiply_cycle4) * compute_CC
    energy_stats['simd_adder'] = get_power_constants()['SIMD_adder_power'] * (add_cycle1 + add_cycle3 + add_cycle4) * compute_CC
    GeLU_energy = sum(energy_stats.values())

    return GeLU_latency, GeLU_energy, energy_stats


def layer_norm_latency_energy(L, dram):
    
    mul_width_per_bank = dram.bank_interface/Data_width
    num_DRAMchip = dram.num_chips_per_rank 
    
    # to compute mean(x)
    adder_tree_cycle = (L/adder_tree_width_per_rank + math.log2(adder_tree_width_per_rank) + 1) * ADD_cycle
    divide_cycle = DIV_cycle

    # to compute x - mean(x), we broadcast mean(x) to all chips
    add_cycle1 = ADD_cycle

    # to compute (x - mean(x))^2
    multiply_cycle1 = L/num_DRAMchip/(mul_width_per_bank * dram.total_banks) * MUL_cycle

    # to compute sqrt((x - mean(x))^2)
    sqrt_cycle = SQRT_cycle
    # to compute 1/sqrt((x - mean(x))^2)
    divide_cycle = DIV_cycle

    # to compute (x - mean(x)) / sqrt((x - mean(x))^2)
    multiply_cycle2 = L/num_DRAMchip/(mul_width_per_bank * dram.total_banks) * MUL_cycle

    # to compute ri * (x - mean(x)) / sqrt((x - mean(x))^2)
    multiply_cycle3 = MUL_cycle

    # to compute ri * (x - mean(x)) / sqrt((x - mean(x))^2) + bi
    add_cycle2 = ADD_cycle

    layer_norm_latency = (adder_tree_cycle + divide_cycle + add_cycle1 + multiply_cycle1 +
                          sqrt_cycle + divide_cycle + multiply_cycle2 + multiply_cycle3 +
                          add_cycle2) * compute_CC

    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['simd_multiplier'] = get_power_constants()['SIMD_multiplier_power'] * (multiply_cycle1 + multiply_cycle2 + multiply_cycle3) * compute_CC
    energy_stats['simd_adder'] = get_power_constants()['SIMD_adder_power'] * (adder_tree_cycle + add_cycle1 + add_cycle2) * compute_CC
    energy_stats['divider_unit'] = get_power_constants()['divider_unit_power'] * (divide_cycle + divide_cycle) * compute_CC
    energy_stats['sqrt_unit'] = get_power_constants()['square_root_unit_power'] * sqrt_cycle * compute_CC

    layer_norm_energy = sum(energy_stats.values())

    return layer_norm_latency, layer_norm_energy, energy_stats

def RMSNorm_latency_energy(L, dram):

    mul_width_per_bank = dram.bank_interface/Data_width
    num_DRAMchip = dram.num_chips_per_rank 
    
    # to compute x^2
    multiply_cycle1 = L/num_DRAMchip/(mul_width_per_bank * dram.total_banks) * MUL_cycle
    
    # to compute sum(x^2)
    adder_tree_cycle = math.log2(adder_tree_width_per_rank) * ADD_cycle

    # to computer sum(x^2) * 1/L
    multiply_cycle2 = MUL_cycle

    # to compute e + sum(x^2) * 1/L
    add_cycle1 = ADD_cycle

    # to compute r = sqrt(e + sum(x^2) * 1/L)
    sqrt_cycle = SQRT_cycle
    
    # to computer 1/r
    divide_cycle = DIV_cycle

    # to computer x * 1/r
    multiply_cycle3 = L/num_DRAMchip/(mul_width_per_bank * dram.total_banks) * MUL_cycle

    # to computer x * 1/r * w
    multiply_cycle4 = MUL_cycle

    #print("RMS: ")
    RMSNorm_latency = (multiply_cycle1 + adder_tree_cycle + multiply_cycle2 + add_cycle1 +
                       sqrt_cycle + divide_cycle + multiply_cycle3 + multiply_cycle4) * compute_CC 
     
    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['simd_multiplier'] = get_power_constants()['SIMD_multiplier_power'] * num_DRAMchip * dram.total_banks * (multiply_cycle1 + multiply_cycle2 + multiply_cycle3 + multiply_cycle4) * compute_CC
    energy_stats['adder_tree'] = get_power_constants()['adder_tree'].get(32, 0) * adder_tree_cycle * (adder_tree_width_per_rank / 32) * compute_CC
    energy_stats['divider_unit'] = get_power_constants()['divider_unit_power'] * (divide_cycle) * compute_CC
    energy_stats['simd_adder'] = get_power_constants()['SIMD_adder_power'] * (add_cycle1) * compute_CC
    energy_stats['sqrt_unit'] = get_power_constants()['square_root_unit_power'] * sqrt_cycle * compute_CC

    RMSNorm_energy = sum(energy_stats.values())
    return RMSNorm_latency, RMSNorm_energy, energy_stats

def SiLU_latency_energy(L, dram): 
    
    # to compute sigmoid(x) only requires multiplication and addition

    mul_width_per_bank = dram.bank_interface/Data_width
    num_DRAMchip = dram.num_chips_per_rank 

    sigmoid_cycle = L/num_DRAMchip/(mul_width_per_bank * dram.total_banks) * MUL_cycle + ADD_cycle
    
    # to compute x * sigmoid(x)
    multiply_cycle1 = MUL_cycle

    SiLU_latency = (sigmoid_cycle + multiply_cycle1) * compute_CC 

    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['simd_multiplier'] = get_power_constants()['SIMD_multiplier_power'] * num_DRAMchip * dram.total_banks * SiLU_latency
    energy_stats['simd_adder'] = get_power_constants()['SIMD_adder_power'] * num_DRAMchip * sigmoid_cycle * compute_CC

    SiLU_energy = sum(energy_stats.values())

    return SiLU_latency, SiLU_energy, energy_stats

def Rotary_latency_energy(L, dram):
    # We pre-compute sin(theta) and cos(theta) and store them in DRAM
    
    mul_width_per_bank = dram.bank_interface/Data_width
    num_DRAMchip = dram.num_chips_per_rank 
    
    # to compute x * cos(theta), y * sin(theta), x * sin(theta), and y * cos(theta)
    multiply_cycle1 = L/(mul_width_per_bank * dram.total_banks) * MUL_cycle * 4 

    # to compute x * sin(theta) + y * cos(theta) and x * cos(theta) - y * sin(theta)
    add_cycle1 = ADD_cycle

    Rotary_latency = (multiply_cycle1 + add_cycle1) * compute_CC 
    
    # Create energy stats dictionary
    energy_stats = create_energy_stats()

    energy_stats['simd_multiplier'] = get_power_constants()['SIMD_multiplier_power'] * dram.total_banks * multiply_cycle1 * compute_CC
    energy_stats['simd_adder'] = get_power_constants()['SIMD_adder_power'] * add_cycle1 * compute_CC

    Rotary_energy = sum(energy_stats.values())

    return Rotary_latency, Rotary_energy, energy_stats

def ARGMAX_latency_energy(L, dram):

    num_DRAMchip = dram.num_chips_per_rank 
    # to compute max(x)
    max_tree_cycle1 = (L/num_DRAMchip/max_tree_width_per_chip + math.log2(max_tree_width_per_chip) + 1) * COMP_cycle

    #rank level max-tree width should be equal to num of chips
    max_tree_width_per_rank = dram.num_chips_per_rank
    max_tree_cycle2 = (math.log2(max_tree_width_per_rank) + 1) * COMP_cycle

    ARGMAX_latency = (max_tree_cycle1 + max_tree_cycle2) * compute_CC 
    
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
