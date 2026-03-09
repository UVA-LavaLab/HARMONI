#!/usr/bin/env bash
set -euo pipefail

if (( $# > 0 )); then
  echo "Usage: ./run_ae.sh"
  echo "This script prepares environment and launches AE jobs."
  exit 1
fi

# Step 1: clone if this is not already the HARMONI repo.
if [[ ! -d ".git" ]]; then
  git clone https://github.com/UVA-LavaLab/HARMONI.git
  cd HARMONI
fi

export HARMONI_HOME="$PWD"

# Step 2: create/activate environment and install requirements.
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH."
  exit 1
fi

# Load conda shell functions for non-interactive shells.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "harmoni"; then
  conda create -n harmoni -c conda-forge python=3.10 pip graphviz -y
else
  echo "Conda environment harmoni already exists. Skipping creation."
fi

conda activate harmoni
pip install -r requirements.txt

# Step 3: clean and create output directories.
source clean.sh

mkdir -p outputs/
mkdir -p traces/
mkdir -p graph_cache/
mkdir -p results/
mkdir -p results/plots/

# Step 4: launch jobs using launch_ae_jobs.sh config.
if [[ ! -x "./launch_ae_jobs.sh" ]]; then
  echo "Error: launch wrapper not found or not executable: ./launch_ae_jobs.sh"
  exit 1
fi

./launch_ae_jobs.sh
echo "Launch stage completed."
echo "When simulation jobs finish, run: ./analyse_ae.sh"
