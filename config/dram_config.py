# Move dram_table and related lists to module level
AddressInterleaving = ['RoCoChRaBgBaBt', 'ChRoCoRaBgBaBt', 'ChRaRoCoBgBaBt', 'RoChRaCoBgBaBt', 'ChRoRaCoBgBaBt', 'RoCo']
Mode = ['ABARACh', 'ABAR', 'ABACh', 'AB', ''] #mode for partition size!

dram_table = {}

#ISPASS configs
#Scale out study
dram_table["DDR5-M1-R4-C8-8-A2"] = [1, 2, 2, 8, 8, 4, 8, 64, 8, 8, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M2-R4-C8-8-A2"] = [2, 2, 2, 8, 8, 4, 8, 64, 8, 8, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M4-R4-C8-8-A2"] = [4, 2, 2, 8, 8, 4, 8, 64, 8, 8, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M8-R4-C8-8-A2"] = [8, 2, 2, 8, 8, 4, 8, 64, 8, 8, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]

#Scale up study
dram_table["DDR5-M4-R2-C4-8-A2"] = [4, 1, 1, 4, 8, 4, 8, 64, 8, 16, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M4-R2-C8-8-A2"] = [4, 1, 1, 8, 8, 4, 8, 64, 8, 8, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M4-R2-C16-8-A2"] = [4, 1, 1, 16, 8, 4, 8, 64, 8, 4, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M4-R4-C4-8-A2"] = [4, 2, 2, 4, 8, 4, 8, 64, 8, 16, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
#repeat dram_table["DDR5-M4-R4-C8-8-A2"] = [4, 2, 2, 8, 8, 4, 8, 64, 8, 8, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M4-R4-C16-8-A2"] = [4, 2, 2, 16, 8, 4, 8, 64, 8, 4, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M4-R8-C4-8-A2"] = [4, 4, 4, 4, 8, 4, 8, 64, 8, 16, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M4-R8-C8-8-A2"] = [4, 4, 4, 8, 8, 4, 8, 64, 8, 8, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M4-R8-C16-8-A2"] = [4, 4, 4, 16, 8, 4, 8, 64, 8, 4, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]

#Scale up for 70b
dram_table["DDR5-M16-R2-C4-8-A2"] = [16, 1, 1, 4, 8, 4, 8, 64, 8, 16, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M16-R2-C8-8-A2"] = [16, 1, 1, 8, 8, 4, 8, 64, 8, 8, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M16-R4-C4-8-A2"] = [16, 2, 2, 4, 8, 4, 8, 64, 8, 16, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M16-R4-C8-8-A2"] = [16, 2, 2, 8, 8, 4, 8, 64, 8, 8, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M16-R8-C4-8-A2"] = [16, 4, 4, 4, 8, 4, 8, 64, 8, 16, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["DDR5-M16-R8-C8-8-A2"] = [16, 4, 4, 8, 8, 4, 8, 64, 8, 8, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]

# Memory standard study
dram_table["DDR4-M8-R4-C8-8-A2"] = [8, 2, 2, 8, 4, 4, 8, 128, 8, 8, 64, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
#repeat dram_table["DDR5-M8-R4-C8-8-A2"] = [8, 2, 2, 8, 8, 4, 8, 64, 8, 8, 128, 'RoChRaCoBgBaBt', 'AB', 2.5, 5, 46.6, 32.5, 13.75]
dram_table["GDDR6-M8-R4-C8-8-A2"] = [8, 2, 2, 8, 8, 4, 8, 32, 8, 8, 256, 'RoChRaCoBgBaBt', 'AB', 1, 2, 43, 32.5, 13.75]

def get_all_dram_names():
    """Return a list of all available DRAM configuration names."""
    return list(dram_table.keys())

def make_dram_config(name):
    # Use the now module-level dram_table
    global dram_table

    if name not in dram_table:
        raise ValueError(f"DRAM {name} not found in dram_table")
    
    num_channels, num_wt_ranks_per_channel, num_kv_ranks_per_channel, num_chips_per_rank, num_bankgroups_per_chip, num_banks_per_bankgroup, capacity_per_chip, csl_lines, burst_length, chip_interface, bank_interface, addr_interleaving, mode, tCCDs, tCCDl, tRC, tRAS, tRP = dram_table[name]
    config = {
        'num_channels': num_channels,
        'num_wt_ranks_per_channel': num_wt_ranks_per_channel,
        'num_kv_ranks_per_channel': num_kv_ranks_per_channel,
        'num_chips_per_rank': num_chips_per_rank,
        'num_bankgroups_per_chip': num_bankgroups_per_chip,
        'num_banks_per_bankgroup': num_banks_per_bankgroup,
        'capacity_per_chip': capacity_per_chip, #in Gb
        'csl_lines': csl_lines,
        'burst_length': burst_length, #used in Bt bits calculation
        'chip_interface': chip_interface, #NOTE: used for calculating UCIe BW in network_config.py)
        'bank_interface': bank_interface,
        'addr_interleaving': addr_interleaving,
        'mode': mode,
        'tCCDs': tCCDs, #ns
        'tCCDl': tCCDl, #ns
        'tRC': tRC, #ns
        'tRAS': tRAS, #ns
        'tRP': tRP #ns
    }
    return config 
