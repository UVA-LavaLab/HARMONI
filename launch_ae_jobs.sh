#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./launch_ae_jobs.sh [slurm|local]

Notes:
  - If no mode is passed, DEFAULT_LAUNCH_MODE is used.
  - Update the configuration section in this script before running run_ae.sh.
EOF
}

# ===============================
# User-editable configuration
# ===============================

# Launch mode used when no CLI mode is passed to this wrapper.
DEFAULT_LAUNCH_MODE="slurm"  # slurm | local

# SLURM mode:
# Required:
SLURM_EMAIL=""               # e.g., your_id@university.edu
SLURM_PARTITION=""           # e.g., standard/cpu
# Optional:
SLURM_ALLOCATION=""          # e.g., your_allocation (SBATCH -A option)
SLURM_ONLY=""                # one of: small, mid, large, large1, large2

# LOCAL mode:
# Optional: empty means run all sizes.
LOCAL_SIZES=""               # comma-separated: small,mid,large,large1,large2
LOCAL_PYTHON_BIN="python3"
LOCAL_LOG_DIR="log_local"
LOCAL_MAX_PARALLEL="0"
LOCAL_TOTAL_MEM_GB=""        # empty => auto-detect
LOCAL_RESERVE_MEM_GB="8"

if (( $# > 1 )); then
  usage >&2
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODE="${1:-$DEFAULT_LAUNCH_MODE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${HARMONI_HOME:-$SCRIPT_DIR}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Error: HARMONI directory not found: $REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"

case "$MODE" in
  slurm)
    [[ -n "$SLURM_EMAIL" ]] || {
      echo "Error: SLURM_EMAIL is required for mode=slurm. Edit launch_ae_jobs.sh."
      exit 1
    }
    [[ -n "$SLURM_PARTITION" ]] || {
      echo "Error: SLURM_PARTITION is required for mode=slurm. Edit launch_ae_jobs.sh."
      exit 1
    }

    if [[ -n "$SLURM_ONLY" ]]; then
      case "$SLURM_ONLY" in
        small|mid|large|large1|large2) ;;
        *)
          echo "Error: SLURM_ONLY must be one of: small, mid, large, large1, large2."
          exit 1
          ;;
      esac
    fi

    cmd=(./launch_all_slurm_ispass_ae.sh --email "$SLURM_EMAIL" --partition "$SLURM_PARTITION")
    [[ -n "$SLURM_ALLOCATION" ]] && cmd+=(--allocation "$SLURM_ALLOCATION")
    [[ -n "$SLURM_ONLY" ]] && cmd+=(--only "$SLURM_ONLY")

    printf '+ '
    printf '%q ' "${cmd[@]}"
    echo
    "${cmd[@]}"
    ;;

  local)
    export PYTHON_BIN="$LOCAL_PYTHON_BIN"
    export LOG_DIR="$LOCAL_LOG_DIR"
    export MAX_PARALLEL="$LOCAL_MAX_PARALLEL"
    export RESERVE_MEM_GB="$LOCAL_RESERVE_MEM_GB"
    if [[ -n "$LOCAL_TOTAL_MEM_GB" ]]; then
      export TOTAL_MEM_GB="$LOCAL_TOTAL_MEM_GB"
    fi

    declare -a size_args=()
    if [[ -n "${LOCAL_SIZES//[[:space:]]/}" ]]; then
      IFS=',' read -r -a raw_sizes <<< "$LOCAL_SIZES"
      for raw_size in "${raw_sizes[@]}"; do
        size="${raw_size,,}"
        size="${size//[[:space:]]/}"
        [[ -z "$size" ]] && continue
        case "$size" in
          small|mid|large|large1|large2) size_args+=("$size") ;;
          *)
            echo "Error: invalid LOCAL_SIZES entry '$raw_size'."
            echo "Allowed values: small, mid, large, large1, large2."
            exit 1
            ;;
        esac
      done
    fi

    cmd=(./run_all_params_memaware.sh)
    if (( ${#size_args[@]} > 0 )); then
      cmd+=("${size_args[@]}")
    fi

    printf '+ '
    printf '%q ' "${cmd[@]}"
    echo
    "${cmd[@]}"
    ;;

  *)
    echo "Error: invalid mode '$MODE'. Use slurm or local."
    usage >&2
    exit 1
    ;;
esac
