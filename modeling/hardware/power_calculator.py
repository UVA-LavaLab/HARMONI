import sys
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from config.dram_config import make_dram_config
from modeling.core.dram_info import DRAMConfig
from modeling.hardware import area_calculator
def get_power_constants(): # mW
    return {
        #'SRAM_power': 14.89936, 
        'SRAM_power': 4.371416, #Single Bank Scratchpad cache with 128 bit datapath, 256kB, 1 read port, 1 write port  
        'systolic': {
            (16, 16, 250): 30.508548,
            (8, 16, 250): 15.8544652,
            (8, 8, 250): 4.39046188,
            (8, 8, 125): 8.739318,
            (8, 4, 250): 2.250225,
            (4, 4, 125): 2.2071434,
            (4, 4, 250): 1.10290404,
            #for DDR4 / GDDR6 config
            (4, 8, 250): 4.39046188/2,
            (16, 8, 250): 4.39046188*2, 
        },
        'SIMD_multiplier_power': 0.6762693, #dyn power per bank
        'max_tree_power': 1.11126736,
        'SIMD_adder_power': 1.8353622,  #TODO: assumed SIMD width = 32
        'exp_unit_power': 8.4631176,
        'adder_tree': {
            16: 1.0865288,
            32: 7.140448 #Updated (Eight 32to1 adder tree)
        },
        'square_root_unit_power': 0,
        'divider_unit_power': 0,
        'Power_center_stripe_logic': 0,
        'Power_dram_banks': 0
    }

def get_static_power_constants(): #mW #scaled from 14nm to 7nm
    """Returns dictionary of static power constants in uW"""
    return {
        'SRAM_power': 2.440726008, # per chip,  Single Bank Scratchpad cache with 128 bit datapath, 256kB, 1 read port, 1 write port 
        'systolic': { #per bank
            (16, 16, 250): 0,
            (8, 16, 250): 0,
            (8, 8, 250): 0.1447768,
            (8, 8, 125): 0,
            (8, 4, 250): 0,
            (4, 4, 125): 0,
            (4, 4, 250): 0,
            (4, 8, 250): 0, 
            (16, 8, 250): 0, 
        },
        'SIMD_multiplier_power': 0.000448878, #128bit, per bank
        'max_tree_power': 0.000619685, #64 to 1 (16bits), per chip
        'SIMD_adder_power': 0.000906823,  #TODO: assumed SIMD width = 32*16 bits, per chip
        'exp_unit_power': 0.004816499, #32 wide SIMD (16 bits), per chip
        'adder_tree': {
            8: 0, #TODO: add value later
            16: 0,
            32: 0.008158665 #8 (systolic_width) 32to1 adder trees (16 bits), per chip
        },
        'square_root_unit_power': 0,
        'divider_unit_power': 0,
        'Power_center_stripe_logic': 0,
        'Power_dram_banks' : 0
    }

def get_static_power_aggregate(systolic_height=8, systolic_width=8, total_banks=32):
    """
    Returns aggregate static power in mW based on configuration.
    
    Args:
        systolic_height: Height of systolic array (default 8)
        bank_interface: Bank interface width in bits (default 128)
        total_banks: Total number of banks (default 16)
    
    Returns:
        float: Total static power in mW
    """
    const = get_static_power_constants()
    
    # Calculate systolic array power based on configuration
    systolic_key = (systolic_height, systolic_width, 250) #CLEANUP: parameterize further to use tCCD values
    systolic_power = const['systolic'].get(systolic_key, 0)
    
    # Calculate total static power
    total_static_power = (
        const['SRAM_power'] +
        (systolic_power + const['SIMD_multiplier_power']) * total_banks +
        const['max_tree_power'] +
        const['adder_tree'].get(total_banks, 0) +
        const['SIMD_adder_power'] +
        const['exp_unit_power'] +
        const['square_root_unit_power'] +
        const['divider_unit_power'] +
        const['Power_center_stripe_logic'] +
        const['Power_dram_banks']
    )
    
    return total_static_power #per chip (mW)

def compute_logic_power(sa_power, banks, SIMD_power, other_power, center_power):
    return (sa_power + SIMD_power) * banks + other_power + center_power

def estimate_power(dram_config_name):
    dram_config = make_dram_config(dram_config_name)
    dram = DRAMConfig(dram_config)
    const = get_power_constants()

    adder_tree_power = const['adder_tree'].get(dram.total_banks, 0) * dram.total_banks
    other_PU_power = (const['max_tree_power'] + adder_tree_power +
                      const['exp_unit_power'] + const['SIMD_adder_power'] +
                      const['square_root_unit_power'] + const['divider_unit_power'] +
                      const['SRAM_power'])

    logic_powers = {}
    total_powers_per_channel = {}
    power_center_stripe_logic = const['Power_center_stripe_logic']
    power_dram = const['Power_dram_banks']

    if dram.bank_interface == 64:
        sa_power = const['systolic'][(4, 4, 250)]
        logic_powers['4by4SA'] = compute_logic_power(sa_power, dram.total_banks, const['SIMD_multiplier_power'], other_PU_power, power_center_stripe_logic)

    elif dram.bank_interface == 128:
        logic_powers['4by4SA'] = compute_logic_power(const['systolic'][(4, 4, 125)], dram.total_banks, const['SIMD_multiplier_power'], other_PU_power, power_center_stripe_logic)
        logic_powers['8by8SA'] = compute_logic_power(const['systolic'][(8, 8, 250)], dram.total_banks, const['SIMD_multiplier_power'], other_PU_power, power_center_stripe_logic)
        logic_powers['16by8SA'] = compute_logic_power(const['systolic'][(8, 16, 250)], dram.total_banks, const['SIMD_multiplier_power'], other_PU_power, power_center_stripe_logic)

    elif dram.bank_interface == 256:
        logic_powers['16by16SA'] = compute_logic_power(const['systolic'][(16, 16, 250)], dram.total_banks, const['SIMD_multiplier_power'], other_PU_power, power_center_stripe_logic)
        logic_powers['8by16SA'] = compute_logic_power(const['systolic'][(8, 16, 250)], dram.total_banks, const['SIMD_multiplier_power'], other_PU_power, power_center_stripe_logic)
        logic_powers['8by8SA'] = compute_logic_power(const['systolic'][(8, 8, 125)], dram.total_banks, const['SIMD_multiplier_power'], other_PU_power, power_center_stripe_logic)

    for name, pwr in logic_powers.items():
        total_powers_per_channel[name] = (pwr + power_dram) * dram.num_chips_per_rank * dram.num_ranks_per_channel

    return {
        'total_banks': dram.total_banks,
        'Power_dram_banks': power_dram,
        'Power_center_stripe_logic': power_center_stripe_logic,
        'Power_logic_per_chip': logic_powers,
        'Total_power_per_channel': total_powers_per_channel,
        'Power_density': {
            name: pwr / area_calculator.estimate_area(dram_config_name)['Area_logic_per_chip'][name] / 1000
            for name, pwr in logic_powers.items()
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate power with different DRAM configurations.")
    parser.add_argument('--dram', type=str, required=True, help='DRAM config')
    args = parser.parse_args()

    results = estimate_power(args.dram)

    print(f"Total number of banks: {results['total_banks']}")
    print(f"Power_dram_banks: {results['Power_dram_banks']} mW")

    for name, val in results['Power_logic_per_chip'].items():
        print(f"Power_logic_per_chip_{name}: {val:.6f} mW")

    for name, val in results['Power_density'].items():
        print(f"Power density of logic chip assuming {name}: {val:.6f} W/mm^2")

    for name, val in results['Total_power_per_channel'].items():
        print(f"Total power per channel assuming {name}: {val / 1000:.6f} W")

    print(f"Power_center_stripe_logic: {results['Power_center_stripe_logic']} mW")