# HARMONI 

Hierarchical ARchitecture MOdeling for LLMs with Near/In Memory Computing

## Code Structure

```text
HARMONI/
├── run.py                      # Main entrypoint (CLI -> simulation)
├── args.py                     # Command-line arguments
├── simulation/                 # Top-level simulation driver
├── modeling/                   # Core modeling stack
│   ├── core/                   # DFG, mapping, memory, tensor, model allocation
│   ├── perf/                   # Performance/latency/energy estimators
│   ├── hardware/               # Area and power models
│   ├── analysis/               # Resource and timeline analysis
│   ├── trace/                  # GEMM/network trace generators
│   ├── report/                 # Stats dumpers
│   └── viz/                    # HTML DFG visualization
├── config/                     # DRAM / network / model / logic configuration
├── misc/                       # Shared enums and timing helpers
├── utils/                      # Logging and cache helpers
├── scripts/                    # Data processing and figure generation scripts
├── params/                     # Experiment parameter presets
├── outputs/, traces/, results/ # Generated artifacts
└── graph_cache/                # Cached DFG/task-mapping artifacts
```

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/UVA-LavaLab/HARMONI.git
cd HARMONI
```

### 2. Environment Setup

#### Create and Activate Conda Environment
```bash
conda create -n harmoni -c conda-forge python=3.10 pip graphviz -y
conda activate harmoni
pip install -r requirements.txt
```

### 3. Initialize Environment
```bash
source env.sh
```

## Usage

### Example Run Command
```bash
python3 run.py \
    --simulate \
    --model_name LLAMA2-7B \
    --dtype BF16 \
    --dram DDR5-M8-R4-C8-8-A2 \
    --batch 1 \
    --input-tokens 32 \
    --output-tokens 32 \
    --fused_qkv \
    --fused_attn \
    --optimization layer static_mapping \
```
## Paper

[Read here](https://github.com/UVA-LavaLab/HARMONI/blob/main/HARMONI_ISPASS2026.pdf)
