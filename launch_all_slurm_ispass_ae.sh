#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./launch_all_slurm_ispass_ae.sh --email <email> --partition <partition> [options]

Required:
  --email <email>          Slurm mail user (equivalent to --mail-user)
  --partition <partition>  Slurm partition

Options:
  --allocation <account>   Slurm account/allocation (equivalent to -A), optional
  --only <size>            Submit only one script: small | mid | large | large1 | large2
  -h, --help               Show this help

Examples:
  # Submit all scripts
  ./launch_all_slurm_ispass_ae.sh --email reviewer@example.edu --partition standard

  # Submit only the "mid" script
  ./launch_all_slurm_ispass_ae.sh --email reviewer@example.edu --partition standard --only mid
EOF
}

EMAIL=""
PARTITION=""
ALLOCATION=""
ONLY=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_FILES=()

cleanup() {
  if ((${#TMP_FILES[@]} > 0)); then
    rm -f "${TMP_FILES[@]}"
  fi
}
trap cleanup EXIT

while (($# > 0)); do
  case "$1" in
    --email)
      EMAIL="${2:-}"
      shift 2
      ;;
    --partition)
      PARTITION="${2:-}"
      shift 2
      ;;
    --allocation|-A)
      ALLOCATION="${2:-}"
      shift 2
      ;;
    --only)
      ONLY="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$EMAIL" || -z "$PARTITION" ]]; then
  echo "Missing required values: --email and --partition" >&2
  usage >&2
  exit 1
fi

shopt -s nullglob
scripts=("$SCRIPT_DIR"/slurm_ispass_ae_*.sh)
shopt -u nullglob

if [[ -n "$ONLY" ]]; then
  case "$ONLY" in
    small|mid|large|large1|large2)
      scripts=("$SCRIPT_DIR/slurm_ispass_ae_${ONLY}.sh")
      if [[ ! -f "${scripts[0]}" ]]; then
        echo "Expected script not found for --only $ONLY: ${scripts[0]}" >&2
        exit 1
      fi
      ;;
    *)
      echo "Invalid --only value: $ONLY" >&2
      echo "Allowed values: small, mid, large, large1, large2" >&2
      exit 1
      ;;
  esac
fi

if ((${#scripts[@]} == 0)); then
  if [[ -n "$ONLY" ]]; then
    echo "No files found for --only $ONLY in: $SCRIPT_DIR" >&2
  else
    echo "No files found matching: $SCRIPT_DIR/slurm_ispass_ae_*.sh" >&2
  fi
  exit 1
fi

IFS=$'\n' scripts=($(printf '%s\n' "${scripts[@]}" | sort))
unset IFS

echo "Found ${#scripts[@]} script(s) in: $SCRIPT_DIR"
echo "Using overrides:"
echo "  email      : $EMAIL"
echo "  partition  : $PARTITION"
if [[ -n "$ALLOCATION" ]]; then
  echo "  allocation : $ALLOCATION"
else
  echo "  allocation : (not set; -A will be omitted)"
fi
if [[ -n "$ONLY" ]]; then
  echo "  only       : $ONLY"
fi
echo

for script in "${scripts[@]}"; do
  submit_script="$script"

  if [[ -z "$ALLOCATION" ]]; then
    submit_script="$(mktemp "${TMPDIR:-/tmp}/slurm_ispass_ae_XXXXXX.sh")"
    TMP_FILES+=("$submit_script")
    sed -E '/^[[:space:]]*#SBATCH[[:space:]]+(-A|--account)([[:space:]=]|$)/d' "$script" > "$submit_script"
  fi

  cmd=(sbatch --mail-user "$EMAIL" --partition "$PARTITION")
  if [[ -n "$ALLOCATION" ]]; then
    cmd+=(-A "$ALLOCATION")
  fi
  cmd+=("$submit_script")
  printf '+ '
  printf '%q ' "${cmd[@]}"
  echo

  "${cmd[@]}"
done
