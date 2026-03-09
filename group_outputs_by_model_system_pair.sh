#!/usr/bin/env bash
set -euo pipefail

# Run this from HARMONI/ root.
if [[ ! -d "outputs" || ! -d "results" ]]; then
  echo "Run this script from HARMONI/ (needs ./outputs and ./results)." >&2
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
  shopt -s nullglob
  matches=(outputs/*"${model_system_pair}"*)
  shopt -u nullglob

  out_count=${#matches[@]}
  if (( out_count == 0 )); then
    echo "[SKIP] ${model_system_pair}: no matching files in outputs/"
    continue
  fi

  mkdir -p "results/${model_system_pair}"
  cp "${matches[@]}" "results/${model_system_pair}/"

  shopt -s nullglob
  result_files=(results/"${model_system_pair}"/*)
  shopt -u nullglob
  res_count=${#result_files[@]}
  echo "[OK] ${model_system_pair}: copied=${out_count}, total_in_results=${res_count}"
done
