#!/bin/bash

#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --job-name=large2
#SBATCH -A LAVAlab-paid
#SBATCH --partition=standard
#SBATCH --output=log_slurm/large2_%A_%a.out
#SBATCH --error=log_slurm/large2_%A_%a.err
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=vyn9mp@virginia.edu
#SBATCH --array=0 # Adjust the range based on the number of lines in params/*.txt

module load miniforge
module load gcc/14.2.0
## module load graphviz

# Load conda for non-interactive shells and activate environment.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate harmoni
PYTHON_BIN="${PYTHON_BIN:-python3}"

export HARMONI_HOME=$PWD
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1


# Read parameters from param.txt based on the array task ID
declare -a PARAMS

while IFS= read -r LINE; do
  [[ -z "$LINE" ]] && continue  # skip empty lines
  PARAMS+=("$LINE")
done < params/ispass_ae_large2.txt

TASK_ID=$SLURM_ARRAY_TASK_ID
IFS=' ' read -r -a ARGS <<< "${PARAMS[$TASK_ID]}"

MODEL_NAME=${ARGS[0]}
DTYPE=${ARGS[1]}
DRAM=${ARGS[2]}
INPUT=${ARGS[3]}
OUTPUT=${ARGS[4]}
BATCH_SIZE=${ARGS[5]}

"$PYTHON_BIN" run.py --model_name "$MODEL_NAME" --dtype "$DTYPE" --dram "$DRAM" -i "$INPUT" -o "$OUTPUT" -b "$BATCH_SIZE" --simulate --optimization layer static_mapping --fused_attn --fused_qkv
