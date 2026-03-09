#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-log_local}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"   # Optional extra cap. 0 => no manual cap.
TOTAL_MEM_GB="${TOTAL_MEM_GB:-}"     # Auto-detect from /proc/meminfo if unset.
RESERVE_MEM_GB="${RESERVE_MEM_GB:-8}"

detect_mem_gb() {
  awk '/MemAvailable:/ { print int($2/1024/1024); found=1; exit } END { if (!found) exit 1 }' /proc/meminfo
}

[[ -f run.py ]] || { echo "ERROR: run.py not found in repo root." >&2; exit 1; }
mkdir -p "$LOG_DIR"

if [[ -z "$TOTAL_MEM_GB" ]]; then
  TOTAL_MEM_GB="$(detect_mem_gb)" || { echo "ERROR: could not detect MemAvailable." >&2; exit 1; }
fi

[[ "$TOTAL_MEM_GB" =~ ^[0-9]+$ ]] || { echo "ERROR: TOTAL_MEM_GB must be an integer." >&2; exit 1; }
[[ "$RESERVE_MEM_GB" =~ ^[0-9]+$ ]] || { echo "ERROR: RESERVE_MEM_GB must be an integer." >&2; exit 1; }
[[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || { echo "ERROR: MAX_PARALLEL must be an integer." >&2; exit 1; }

USABLE_MEM_GB=$((TOTAL_MEM_GB - RESERVE_MEM_GB))
(( USABLE_MEM_GB > 0 )) || { echo "ERROR: usable memory must be > 0 GB." >&2; exit 1; }

echo "[INFO] total_mem=${TOTAL_MEM_GB}GB reserve=${RESERVE_MEM_GB}GB usable=${USABLE_MEM_GB}GB"

export HARMONI_HOME="${HARMONI_HOME:-$PWD}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

normalize_size() {
  local size="$1"
  size="${size,,}"
  size="${size//[[:space:]]/}"
  echo "$size"
}

spec_for_size() {
  case "$1" in
    small)  echo "params/ispass_ae_small.txt:16" ;;
    mid)    echo "params/ispass_ae_mid.txt:32" ;;
    large)  echo "params/ispass_ae_large.txt:64" ;;
    large1) echo "params/ispass_ae_large1.txt:128" ;;
    large2) echo "params/ispass_ae_large2.txt:150" ;;
    *) return 1 ;;
  esac
}

declare -a requested_sizes=()
if (( $# > 0 )); then
  requested_sizes=("$@")
elif [[ -n "${SELECT_SIZES:-}" ]]; then
  IFS=',' read -r -a requested_sizes <<< "$SELECT_SIZES"
else
  requested_sizes=(small mid large large1 large2)
fi

declare -a PARAM_SPECS=()
declare -a selected_sizes=()
declare -A seen_sizes=()
for raw_size in "${requested_sizes[@]}"; do
  size="$(normalize_size "$raw_size")"
  [[ -z "$size" ]] && continue

  if [[ -n "${seen_sizes[$size]:-}" ]]; then
    continue
  fi

  if ! spec="$(spec_for_size "$size")"; then
    echo "ERROR: unknown size '$raw_size'. Valid values: small, mid, large, large1, large2." >&2
    exit 1
  fi

  PARAM_SPECS+=("$spec")
  selected_sizes+=("$size")
  seen_sizes["$size"]=1
done

(( ${#PARAM_SPECS[@]} > 0 )) || { echo "ERROR: no valid sizes selected." >&2; exit 1; }
echo "[INFO] selected sizes: ${selected_sizes[*]}"

declare -a PIDS
count=0

for spec in "${PARAM_SPECS[@]}"; do
  file="${spec%%:*}"
  mem_req="${spec##*:}"
  [[ -f "$file" ]] || { echo "ERROR: missing $file" >&2; exit 1; }

  auto_limit=$((USABLE_MEM_GB / mem_req))
  (( auto_limit >= 1 )) || { echo "ERROR: $file needs ${mem_req}GB, but usable memory is ${USABLE_MEM_GB}GB." >&2; exit 1; }

  file_limit="$auto_limit"
  if (( MAX_PARALLEL > 0 && MAX_PARALLEL < file_limit )); then
    file_limit="$MAX_PARALLEL"
  fi

  echo "[INFO] $file mem_per_job=${mem_req}GB parallel_limit=${file_limit}"

  line_no=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    ((line_no += 1))
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    read -r model dtype dram input_tokens output_tokens batch extra <<< "$line"
    [[ -z "${batch:-}" || -n "${extra:-}" ]] && {
      echo "ERROR: bad line ($file:$line_no): $line" >&2
      exit 1
    }

    cmd=(
      "$PYTHON_BIN" run.py
      --model_name "$model" --dtype "$dtype" --dram "$dram"
      -i "$input_tokens" -o "$output_tokens" -b "$batch"
      --simulate --optimization layer static_mapping --fused_attn --fused_qkv
    )

    ((count += 1))

    while (( $(jobs -rp | wc -l) >= file_limit )); do sleep 1; done

    tag="${count}_${model}_${dram}_b${batch}_i${input_tokens}_o${output_tokens}"
    tag="${tag//\//_}"
    log_file="$LOG_DIR/${tag}.log"

    (
      echo "[START] $(date '+%F %T') $file:$line_no"
      echo "[CMD] ${cmd[*]}"
      "${cmd[@]}"
    ) > "$log_file" 2>&1 &

    PIDS+=("$!")
    echo "[LAUNCH] pid=$! $file:$line_no log=$log_file"
  done < "$file"
done

echo "[INFO] total cases: $count"

failures=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || ((failures += 1))
done

echo "[INFO] completed. failures=$failures"
(( failures == 0 ))
