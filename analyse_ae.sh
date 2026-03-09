#!/usr/bin/env bash
set -euo pipefail

if (( $# > 0 )); then
  echo "Usage: ./analyse_ae.sh"
  echo "This script takes no arguments."
  exit 1
fi

if [[ ! -d "results" || ! -d "scripts" ]]; then
  echo "Error: run this from HARMONI/ root."
  exit 1
fi

# Step 5: aggregate outputs and generate figures.
./group_outputs_by_model_system_pair.sh
./perf_summary_by_model_system_pair.sh

shopt -s nullglob
h100_refs=(reference/*H100*)
if [ "${#h100_refs[@]}" -gt 0 ]; then
  cp "${h100_refs[@]}" results/
else
  echo "No reference/*H100* files found; skipping copy."
fi
shopt -u nullglob

python3 scripts/create_hier_data.py
python3 scripts/diff_hier_data.py

# Plotting
python3 scripts/figures/ispass.py
