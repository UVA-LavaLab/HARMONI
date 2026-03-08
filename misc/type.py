from enum import Enum, IntEnum


class DataType(Enum):
    W16A16 = "W16A16"
    W16A8 = "W16A8"
    W8A16 = "W8A16"
    W8A8 = "W8A8"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    FP32 = "FP32"

class DeviceType(Enum):
    NONE = 0
    GPU = 1
    CPU = 2
    PIM = 3


class InterfaceType(Enum):
    NVLINK4 = 0
    NVLINK3 = 1
    PCIE4 = 2
    PCIE5 = 3


class GPUType(Enum):
    A100a = 0
    H100 = 1

class HarmoniTensorType(Enum):
    WEIGHT = 0
    KV = 1
    ACT = 2
    INPUT = 3
    
class PhaseType(Enum):
    PREFILL = 0
    DECODE = 1

class NodeTagType(Enum):
    ATTN = 0
    FC = 1
    MISC = 2
    EMBED = 3

class NodeType(Enum):
    GEMV = 0
    GEMM = 1
    SOFTMAX = 2
    RMSNORM = 3
    SILU = 4
    RESADD = 5
    CONCAT = 6
    BROADCAST = 7
    REDUCTION = 8
    ROTARY = 9
    SCALE = 10
    EWMUL = 11
    SPLIT = 12
    APPEND = 13
    SHARD = 14
    LOOKUP = 15
    ARGMAX = 16
    AGG = 17
    SYNC = 18
    FUSED_SCORE = 19

class ParallelType(Enum):
    TENSOR = 0
    BANK = 1
    CHIP = 2
    RANK = 3
    CHANNEL = 4


class LinkLevel(IntEnum):
    ROOT_CH = 0
    CH_RANK = 1
    RANK_CHIP = 2

class CommType(Enum):
    GATHER = "gather"
    REDUCE = "reduce"
    BROADCAST = "broadcast"
    SCATTER = "scatter"
    ALL_REDUCE = "all_reduce"
    GATHER_SCATTER = "gather_scatter"
    
def get_bytes_per_element(dtype: DataType, tag: HarmoniTensorType = None) -> int:
    """
    Returns the number of bytes per element for a given DataType.
    For mixed precision types (W<>A<>), uses weight precision for WEIGHT/KV tensors
    and activation precision for ACT tensors.
    For standard precision types (FP16, BF16, INT8, FP32), returns fixed bytes per element.
    
    Args:
        dtype (DataType): The data type to get bytes for
        tag (HarmoniTensorType): The type of tensor (WEIGHT, KV, or ACT)
        
    Returns:
        int: Number of bytes per element
    """
    # Standard precision types - fixed bytes per element
    standard_precision_bytes = {
        DataType.FP16: 2,    # 16-bit floating point
        DataType.BF16: 2,    # 16-bit brain floating point
        DataType.INT8: 1,    # 8-bit integer
        DataType.FP32: 4     # 32-bit floating point
    }
    
    # If it's a standard precision type, return fixed bytes
    if dtype in standard_precision_bytes:
        return standard_precision_bytes[dtype]
    
    # Mixed precision types - bytes depend on tensor tag
    if dtype in [DataType.W16A8, DataType.W8A16, DataType.W8A8, DataType.W16A16]:
        if tag is None:
            raise ValueError(f"Tag must be provided for mixed precision type {dtype}")
            
        if tag in [HarmoniTensorType.WEIGHT, HarmoniTensorType.KV]:
            # Use weight precision
            if dtype == DataType.W16A8 or dtype == DataType.W16A16:
                return 2  # 16-bit weights
            else:  # W8A16 or W8A8
                return 1  # 8-bit weights
        elif tag in [HarmoniTensorType.ACT, HarmoniTensorType.INPUT]:
            # Use activation precision
            if dtype == DataType.W8A16 or dtype == DataType.W16A16:
                return 2  # 16-bit activations
            else:  # W16A8 or W8A8
                return 1  # 8-bit activations
        else:
            raise ValueError(f"Invalid tensor tag {tag} for mixed precision type {dtype}")
    
    raise ValueError(f"Unsupported data type: {dtype}")
