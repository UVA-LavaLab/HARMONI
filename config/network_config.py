"""Simple network configuration for the latency estimator."""

from dataclasses import asdict, dataclass, field

# Default network constants (single source of truth)
TOTAL_LANES = 128.0 #144 port switch
BW_PER_LANE_GBPS = 8.0 #GBps
PCIE_BW = 32.0 #GBps PCIE 6.0 x4
UCIE_BW_PER_PIN_GBPS = 8.0 #GBps per pin
DEFAULT_CHIP_INTERFACE = 8.0 #pins per chip (fallback)
UCIE_BW = DEFAULT_CHIP_INTERFACE * UCIE_BW_PER_PIN_GBPS #GBps

CXL_L = 20.0 #ns
PCIE_L = 2.0 #ns

CXL_O = 25.0 #ns
PCIE_O = 5.0 #ns

L0 = 10.0 #ns (arbitrary)
Q = [0.0, 4096.0, 4096.0] #bytes (typical PCIe buffer size)
FLIT_SIZE = 256.0 #bytes
FLIT_HEADER = 4.0 #bytes
INTRANODE_RING = False


@dataclass
class NetworkConfig:
    Channel: int
    Rank: int
    Chip: int
    BW: list[float]
    l: list[float]
    o: list[float]
    L0: float = 11.0
    Q: list[float] = field(default_factory=lambda: [0.0, 4096.0, 4096.0])
    flit_size: float = 256.0
    flit_header: float = 4.0
    intranode_ring: bool = False

    def __post_init__(self) -> None:
        if self.Channel <= 0 or self.Rank <= 0 or self.Chip <= 0:
            raise ValueError("Channel, Rank, and Chip must be > 0.")

        for name, values in (("BW", self.BW), ("l", self.l), ("o", self.o), ("Q", self.Q)):
            if len(values) != 3:
                raise ValueError(f"{name} must have exactly 3 values.")

        if any(v <= 0 for v in self.BW):
            raise ValueError("All BW values must be > 0.")
        if self.flit_size <= 0:
            raise ValueError("flit_size must be > 0.")
        if self.flit_header < 0:
            raise ValueError("flit_header must be >= 0.")

    def to_dict(self) -> dict:
        return asdict(self)


def make_default_network_config(
    num_channels: int,
    num_ranks: int,
    num_chips: int,
    chip_interface: float | None = None,
) -> NetworkConfig:
    lanes_per_channel = TOTAL_LANES / num_channels
    cxl_bw = lanes_per_channel * BW_PER_LANE_GBPS
    ucie_bw = (
        float(chip_interface) * UCIE_BW_PER_PIN_GBPS
        if chip_interface is not None else UCIE_BW
    )
    return NetworkConfig(
        Channel=num_channels,
        Rank=num_ranks,
        Chip=num_chips,
        BW=[cxl_bw, PCIE_BW, ucie_bw],
        l=[CXL_L, PCIE_L, PCIE_L],
        o=[CXL_O, PCIE_O, PCIE_O],
        L0=L0,
        Q=Q.copy(),
        flit_size=FLIT_SIZE,
        flit_header=FLIT_HEADER,
        intranode_ring=INTRANODE_RING,
    )
