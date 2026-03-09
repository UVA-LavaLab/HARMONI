#!/usr/bin/env bash
set -euo pipefail

# Run this from HARMONI/ root.
if [[ ! -d "results" || ! -d "scripts" ]]; then
  echo "Run this script from HARMONI/ (needs ./results and ./scripts)." >&2
  exit 1
fi

model_system_pairs=(
  "LLAMA2-7B_DDR5-M1-R4-C8-8-A2"
  "LLAMA2-7B_DDR5-M2-R4-C8-8-A2"
  "LLAMA2-7B_DDR5-M4-R2-C16-8-A2"
  "LLAMA2-7B_DDR5-M4-R2-C4-8-A2"
  "LLAMA2-7B_DDR5-M4-R2-C8-8-A2"
  "LLAMA2-7B_DDR5-M4-R4-C16-8-A2"
  "LLAMA2-7B_DDR5-M4-R4-C4-8-A2"
  "LLAMA2-7B_DDR5-M4-R4-C8-8-A2"
  "LLAMA2-7B_DDR5-M4-R8-C16-8-A2"
  "LLAMA2-7B_DDR5-M4-R8-C4-8-A2"
  "LLAMA2-7B_DDR5-M4-R8-C8-8-A2"
  "LLAMA2-7B_DDR5-M8-R4-C8-8-A2"
  "LLAMA3-70B_DDR5-M16-R4-C8-8-A2"
  "LLAMA3-70B_DDR5-M16-R8-C8-8-A2"
  "MISTRAL-7B_DDR5-M4-R4-C8-8-A2"
  "MISTRAL-7B_DDR5-M8-R4-C8-8-A2"
)

for model_system_pair in "${model_system_pairs[@]}"; do
  out_dir="results/${model_system_pair}"
  if [[ ! -d "$out_dir" ]]; then
    echo "[SKIP] ${model_system_pair}: directory not found (${out_dir})"
    continue
  fi

  shopt -s nullglob
  perf_files=("${out_dir}"/performance_summary_*.txt)
  shopt -u nullglob
  if (( ${#perf_files[@]} == 0 )); then
    echo "[SKIP] ${model_system_pair}: no performance_summary_*.txt in ${out_dir}"
    continue
  fi

  if ! python3 scripts/summarize_perf.py --output_dir "$out_dir"; then
    echo "[WARN] ${model_system_pair}: summarize_perf.py failed"
    continue
  fi

  csv_path="${out_dir}/${model_system_pair}.csv"
  if [[ -f "$csv_path" ]]; then
    mv "$csv_path" results/
    echo "[OK] ${model_system_pair}: wrote results/${model_system_pair}.csv"
  else
    echo "[WARN] ${model_system_pair}: expected CSV not generated (${csv_path})"
  fi
done
