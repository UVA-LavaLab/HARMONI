import sys
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from config.dram_config import make_dram_config
from modeling.core.dram_info import DRAMConfig

def get_area_per_bank_and_center_stripe(dram_name, total_banks):
    if 'GDDR6' in dram_name:
        # based on 'GDDR6 die photo', density: 8Gb/channel, 16 banks/channel, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8588344
        area_per_bank = (306 * 154) / (1225 * 714) * 47.9 # 2.5807 mm^2
        center_stripe_area = 47.9 - 16 * area_per_bank # 6.608256 mm^2
        area_dram = area_per_bank * total_banks
    elif 'LPDDR5' in dram_name:
        # based on 'LPDDR5 die photo', density: 8Gb/channel, 16 banks/channel, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8944216
        area_per_bank = (313 * 179) / (1450 * 705) * 45.9 # 2.51566 mm^2
        center_stripe_area = 45.9 - 16 * area_per_bank # 5.64944 mm^2
        area_dram = area_per_bank * total_banks
    elif 'DDR5' in dram_name:
        # based on 'DDR5 die photo', density: 32Gb/chip, 32 banks/chip, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10454327
        area_16banks = (300 * 679) / (687 * 700) * 9.7 * 7.9 # 2.30566 mm^2
        center_stripe_area = 9.7 * 7.9 - 2 * area_16banks # 2.851888 mm^2
        area_dram = 2 * area_16banks
    else:
        raise ValueError("Unsupported DRAM type")
    return area_dram, center_stripe_area

def compute_logic_area(sa_area, dram_banks, SIMD_multiplier_area, other_pu_area, center_stripe_area):
    return (sa_area + SIMD_multiplier_area) * dram_banks + other_pu_area + center_stripe_area

def estimate_area(dram_config_name):
    dram_config = make_dram_config(dram_config_name)
    dram = DRAMConfig(dram_config)

    area_const = {
        'SRAM_area': 0.06163656,
        'systolic': {
            (16, 16, 250): 0.1993709,
            (8, 16, 250): 0.0878016,
            (8, 8, 250): 0.02573392,
            (8, 8, 125): 0.02654176,
            (4, 4, 125): 0.00539546,
            (4, 4, 250): 0.00535194
        },
        'SIMD_multiplier_area': 0.001288804,
        'max_tree_area': 0.00228208,
        'SIMD_adder_area': 0.003329246,
        'exp_unit_area': 0.0198764,
        'adder_tree': {16: 0.001734, 32: 0.0282064},
        'square_root_unit_area': 0,
        'divider_unit_area': 0
    }

    adder_tree_area = area_const['adder_tree'].get(dram.total_banks, 0)
    other_pu_area = (area_const['max_tree_area'] + adder_tree_area +
                     area_const['exp_unit_area'] + area_const['SIMD_adder_area'] +
                     area_const['square_root_unit_area'] + area_const['divider_unit_area'] +
                     area_const['SRAM_area'])

    Area_dram_banks, Area_center_stripe_dram_logic = get_area_per_bank_and_center_stripe(dram_config_name, dram.total_banks)

    # For a conservative estimation, we treat 10nm dram process as 16 nm logic process, and then scale it to 7nm logic process.
    Area_center_stripe_dram_logic_after_scaling = Area_center_stripe_dram_logic * 0.31

    Logic_per_chip_area = {}

    if dram.bank_interface == 64:
        sa_area = area_const['systolic'][(4, 4, 250)]
        Logic_per_chip_area['4by4SA'] = compute_logic_area(sa_area, dram.total_banks, area_const['SIMD_multiplier_area'], other_pu_area, Area_center_stripe_dram_logic_after_scaling)

    elif dram.bank_interface == 128:
        Logic_per_chip_area['4by4SA'] = compute_logic_area(area_const['systolic'][(4, 4, 125)], dram.total_banks, area_const['SIMD_multiplier_area'], other_pu_area, Area_center_stripe_dram_logic_after_scaling)
        Logic_per_chip_area['8by8SA'] = compute_logic_area(area_const['systolic'][(8, 8, 250)], dram.total_banks, area_const['SIMD_multiplier_area'], other_pu_area, Area_center_stripe_dram_logic_after_scaling)
        Logic_per_chip_area['16by8SA'] = compute_logic_area(area_const['systolic'][(8, 16, 250)], dram.total_banks, area_const['SIMD_multiplier_area'], other_pu_area, Area_center_stripe_dram_logic_after_scaling)

    elif dram.bank_interface == 256:
        Logic_per_chip_area['16by16SA'] = compute_logic_area(area_const['systolic'][(16, 16, 250)], dram.total_banks, area_const['SIMD_multiplier_area'], other_pu_area, Area_center_stripe_dram_logic_after_scaling)
        Logic_per_chip_area['8by16SA'] = compute_logic_area(area_const['systolic'][(8, 16, 250)], dram.total_banks, area_const['SIMD_multiplier_area'], other_pu_area, Area_center_stripe_dram_logic_after_scaling)
        Logic_per_chip_area['8by8SA'] = compute_logic_area(area_const['systolic'][(8, 8, 125)], dram.total_banks, area_const['SIMD_multiplier_area'], other_pu_area, Area_center_stripe_dram_logic_after_scaling)
        
    Area_four_chips = 4 * (Area_dram_banks + Logic_per_chip_area.get('8by8SA'))

    return {
        'total_banks': dram.total_banks,
        'Area_dram_banks': Area_dram_banks,
        'Area_center_stripe_dram_logic': Area_center_stripe_dram_logic,
        'Area_center_stripe_dram_logic_after_scaling': Area_center_stripe_dram_logic_after_scaling,
        'Sangam_functional_logic_area': (Logic_per_chip_area.get('8by8SA') - Area_center_stripe_dram_logic_after_scaling),
        'Area_logic_per_chip': Logic_per_chip_area,
        'Area_four_chips': Area_four_chips
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate area with different DRAM configurations.")
    parser.add_argument('--dram', type=str, required=True, help='DRAM config')
    args = parser.parse_args()

    result = estimate_area(args.dram)

    print(f"Total number of banks/chps: {result['total_banks']}")
    print(f"Area of dram banks per chip: {result['Area_dram_banks']:.6f} mm^2")
    print(f"Area of dram center stripe logic(dram process): {result['Area_center_stripe_dram_logic']:.6f} mm^2")
    print(f"Area of dram center stripe logic(logic process): {result['Area_center_stripe_dram_logic_after_scaling']:.6f} mm^2")
    print(f"Total area of Sangam functional logic (including SRAM): {result['Sangam_functional_logic_area']:.6f} mm^2")
    for key, val in result['Area_logic_per_chip'].items():
        print(f"Total area of logic per_chip ({key}): {val:.6f} mm^2")
    print(f"Total area of four_chips: {result['Area_four_chips']:.6f} mm^2")

# python ./modeling/hardware/area_calculator.py --dram <>