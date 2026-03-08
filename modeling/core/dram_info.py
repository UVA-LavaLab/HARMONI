from args import get_args
from config.dram_config import make_dram_config

class DRAMConfig:
    def __init__(self, dram_config):
        """
        Initializes DRAM configuration.
        """
        self.num_channels = dram_config.get('num_channels', 4)
        self.num_wt_ranks_per_channel = dram_config.get('num_wt_ranks_per_channel', 2)
        self.num_kv_ranks_per_channel = dram_config.get('num_kv_ranks_per_channel', 2)
        self.num_ranks_per_channel = self.num_wt_ranks_per_channel + self.num_kv_ranks_per_channel
        self.num_chips_per_rank = dram_config.get('num_chips_per_rank', 8)
        self.num_bankgroups_per_chip = dram_config.get('num_bankgroups_per_chip', 4)
        self.num_banks_per_bankgroup = dram_config.get('num_banks_per_bankgroup', 4)
        self.capacity_per_chip = dram_config.get('capacity_per_chip', 4)  # in Gb
        self.csl_lines = dram_config.get('csl_lines', 128)
        self.chip_interface = dram_config.get('chip_interface', 8) #dram_bus_width/num_of_chips 
        self.bank_interface = dram_config.get('bank_interface', 64)
        self.burst_length = dram_config.get('burst_length', 8) #CLEANUP: recheck: burst length might be a factor of IO path, so for now we can just configure it instead of calculating
        #self.burst_length = self.dram_bus_width/8 (if byte addressable) 
        self.addr_interleaving = dram_config.get('addr_interleaving', 'RoCoChRaBgBaBt')
        self.mode = dram_config.get('mode', 'ABARACh')
        self.total_banks = (self.num_bankgroups_per_chip *
                            self.num_banks_per_bankgroup) #per chip
        self.rows_per_bank = self.capacity_per_chip * 2**30 // (self.total_banks * self.csl_lines * self.bank_interface)
        self.tCCDs = dram_config.get('tCCDs', 2.5) #ns
        self.tCCDl = dram_config.get('tCCDl', 5) #ns
        self.tRC = dram_config.get('tRC', 46.9) #ns
        self.tCK = dram_config.get('tCK', 0.625) #ns
        self.tRAS = dram_config.get('tRAS', 32.5) #ns
        self.tRP = dram_config.get('tRP', 13.75) #ns
        

    def calculate_dram_capacity(self):
        """
        Calculates the total capacity of the DRAM.
        """
        total_capacity = (self.num_channels *
                          self.num_ranks_per_channel *
                          self.num_chips_per_rank *
                          self.capacity_per_chip)/8  # in GB
        return total_capacity

    def calculate_channel_capacity(self):
        """
        Calculates the capacity of a single channel.
        """
        channel_capacity = (self.num_ranks_per_channel *
                            self.num_chips_per_rank *
                            self.capacity_per_chip)/8  # in GB
        return channel_capacity

    def calculate_rank_capacity(self):
        """
        Calculates the capacity of a single rank.
        """
        rank_capacity = (self.num_chips_per_rank *
                         self.capacity_per_chip)/8  # in GB
        return rank_capacity

    def calculate_bank_capacity(self):
        """
        Calculates the capacity of a single bank.
        """

        bank_capacity = (self.capacity_per_chip /
                         self.total_banks)/8  # in GB
        return bank_capacity
    
    def calculate_partition_size(self):
        """
        Calculates the partition size for different levels of parallelism for all chips
        """
        partition_size = self.num_chips_per_rank

        if self.mode == 'ABARACh':
            partition_size *= self.bank_interface * self.total_banks * self.num_channels * self.num_ranks_per_channel
        elif self.mode == 'ABAR':
            partition_size *= self.bank_interface * self.total_banks * self.num_ranks_per_channel
        elif self.mode == 'ABACh':
            partition_size *= self.bank_interface * self.total_banks * self.num_channels
        elif self.mode == 'AB':
            partition_size *= self.bank_interface * self.total_banks
        else:
            partition_size *= self.bank_interface

        return partition_size/(8*1024) # in KB

    def calculate_center_strip_BW(self):
        """
        Calculates the bandwidth at the center strip for the DRAM.
        """
        center_strip_BW = [0, 0, 0]
        bits_per_chip = self.bank_interface*self.num_bankgroups_per_chip*self.num_banks_per_bankgroup
        
        center_strip_BW[0] = bits_per_chip/(8*self.tCCDs)
        center_strip_BW[1] = bits_per_chip/(8*self.tCCDl)
        center_strip_BW[2] = bits_per_chip/(8*self.tRC) 
        
        return center_strip_BW #in GBps

    def calc_avg_center_strip_BW(self):
        tCCDs_BW = self.calculate_center_strip_BW()[0]
        tRC_BW = self.calculate_center_strip_BW()[2]
        avg_BW = tCCDs_BW + tRC_BW/self.csl_lines
        
        return avg_BW #GBps
    
    def calc_peak_BW(self):
        return self.calculate_center_strip_BW()[0]*self.num_chips_per_rank*self.num_ranks_per_channel*self.num_channels #GBps

    def calc_avg_peak_BW(self):
        return self.calc_avg_center_strip_BW() * self.num_chips_per_rank*self.num_ranks_per_channel*self.num_channels #GBps

    def calc_peak_throughput(self):
        return self.calc_peak_BW() * 0.5 #Flops/byte (GFLOPS/s)
        
    def calc_gemm_throughput(self):
        args = get_args()
        return self.calc_peak_throughput() * 2 * args.systolic_height # 2 for MAC, (GFLOPs/s)
    
    def print_dram_info(self):
        """
        Prints the DRAM configuration and information.
        """
        print("------ DRAM INFO ------")
        print("DRAM Configuration: ",self.__dict__)
        print("DRAM capacity: ", self.calculate_dram_capacity(), "GB")
        print("Per Channel capacity: ", self.calculate_channel_capacity(), "GB")
        print("Per Rank capacity: ", self.calculate_rank_capacity(), "GB")
        print("Per Bank capacity: ", self.calculate_bank_capacity()*1024, "MB")
        print("-----------------------")

args = get_args()
dram_config = make_dram_config(args.dram)
dram = DRAMConfig(dram_config)
