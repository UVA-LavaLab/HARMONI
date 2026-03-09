#!/usr/bin/env bash
set -euo pipefail

if (( $# > 0 )); then
  echo "Usage: ./run_ae.sh"
  echo "This script launches AE jobs."
  exit 1
fi

confirm_yes_no() {
  local prompt="$1"
  local response

  while true; do
    read -r -p "${prompt} [y/N]: " response || {
      echo "Error: failed to read input."
      exit 1
    }
    case "${response,,}" in
      y|yes) return 0 ;;
      n|no|"") return 1 ;;
      *) echo "Please answer yes or no." ;;
    esac
  done
}

# Prerequisite: follow setup instructions in README.md before running this script.
# Expected: run from HARMONI/ root with dependencies already installed.
if [[ ! -f "run.py" ]]; then
  echo "Error: run.py not found. Run this from HARMONI/ root after completing README setup."
  exit 1
fi

if ! confirm_yes_no "Have you setup HARMONI env as mentioned in the README.md?"; then
  echo "Please complete setup instructions in README.md before running run_ae.sh."
  exit 1
fi

if ! confirm_yes_no "Have you updated launch_ae_jobs.sh?"; then
  echo "Please update launch_ae_jobs.sh before running run_ae.sh."
  exit 1
fi

export HARMONI_HOME=$PWD

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
