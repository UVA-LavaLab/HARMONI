#!/usr/bin/env python3
"""
Generate the ISPASS figure set:
1) scaleout (from this script)
2) scaleup (delegated to scaleup.py)
3) wide_subplots (delegated to wide_subplots.py)
4) mistral_scatter (delegated to throughput_energy_scatter.py)
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from data_loader import load_and_filter_data, save_filtered_data
from plot_stacked import plot_stacked_chart

SCRIPT_DIR = Path(__file__).resolve().parent
HARMONI_HOME = Path(os.environ.get("HARMONI_HOME", SCRIPT_DIR.parents[1]))
CSV_FILE = HARMONI_HOME / "results" / "hierarchical_data.csv"
OUTPUT_DIR = HARMONI_HOME / "results" / "plots"

FILTER_CONFIGS = {
    "scale_out": {
        "models": ["LLAMA2-7B"],
        "systems": [
            "DDR5-M1-R4-C8-8-A2",
            "DDR5-M2-R4-C8-8-A2",
            "DDR5-M4-R4-C8-8-A2",
            "DDR5-M8-R4-C8-8-A2",
        ],
        "batches": [1, 4, 8],
        "in_out_pairs": [(128, 2048), (2048, 128), (2048, 2048)],
        "metrics": [
            "decode_throughput",
            "e2e_latency",
            "total_energy",
            "comp_pct",
            "comm_pct",
            "queue_pct",
            "prefill_latency",
            "decode_latency",
        ],
    },
    "scale_up": {
        "models": ["LLAMA2-7B"],
        "systems": [
            "DDR5-M4-R2-C4-8-A2",
            "DDR5-M4-R2-C8-8-A2",
            "DDR5-M4-R2-C16-8-A2",
            "DDR5-M4-R4-C4-8-A2",
            "DDR5-M4-R4-C8-8-A2",
            "DDR5-M4-R4-C16-8-A2",
            "DDR5-M4-R8-C4-8-A2",
            "DDR5-M4-R8-C8-8-A2",
            "DDR5-M4-R8-C16-8-A2",
        ],
        "batches": [1, 4, 8],
        "in_out_pairs": [(2048, 2048)],
        "metrics": ["e2e_latency", "comp_pct", "comm_pct", "queue_pct"],
    },
    "wide": {
        "models": ["LLAMA2-7B", "MISTRAL-7B", "LLAMA3-70B"],
        "systems": [
            "H100",
            "H100-2",
            "DDR5-M4-R4-C8-8-A2",
            "DDR5-M8-R4-C8-8-A2",
            "DDR5-M16-R4-C8-8-A2",
            "DDR5-M16-R8-C8-8-A2",
        ],
        "batches": [1, 4, 8],
        "in_out_pairs": [(128, 128), (128, 2048), (2048, 128), (2048, 2048)],
        "metrics": [
            "decode_throughput",
            "e2e_latency",
            "total_energy",
            "prefill_latency",
            "decode_latency",
        ],
    },
}


def export_filtered_csvs(csv_file: Path, output_dir: Path):
    scaleup_df = load_and_filter_data(str(csv_file), FILTER_CONFIGS["scale_up"])
    scaleup_csv = output_dir / "filtered_scale_up.csv"
    save_filtered_data(df=scaleup_df, output_path=str(scaleup_csv))

    wide_df = load_and_filter_data(str(csv_file), FILTER_CONFIGS["wide"])
    wide_csv = output_dir / "filtered_wide.csv"
    save_filtered_data(df=wide_df, output_path=str(wide_csv))

    scaleout_df = load_and_filter_data(str(csv_file), FILTER_CONFIGS["scale_out"])
    scaleout_df["e2e_latency_s"] = scaleout_df["e2e_latency"] / 1000.0
    scaleout_csv = output_dir / "filtered_scale_out.csv"
    save_filtered_data(df=scaleout_df, output_path=str(scaleout_csv))

    return scaleup_csv, wide_csv, scaleout_df


def generate_scaleout_plot(scaleout_df, output_dir: Path):
    plot_stacked_chart(
        df=scaleout_df,
        stack_metrics=["queue_pct", "comp_pct", "comm_pct"],
        metric_labels={
            "queue_pct": "Queue",
            "comp_pct": "Computation",
            "comm_pct": "Communication",
        },
        ylabel="E2E Latency (s)",
        output_path=str(output_dir / "scale_out_breakdown.pdf"),
        n_cols=1,
        show_model_labels=True,
        as_percentage=False,
        # queue/comp/comm are percentages; convert them to absolute seconds via e2e_latency_s
        base_metric="e2e_latency_s",
        group_by_inout=True,
    )


def run_script(script_name: str, args: list[str]) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / script_name), *args]
    subprocess.run(cmd, check=True, cwd=str(HARMONI_HOME))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ISPASS figure artifacts for scaleout/scaleup/wide/mistral plots."
    )
    parser.add_argument("--csv", type=Path, default=CSV_FILE, help="Input hierarchical_data.csv path.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for filtered CSVs and generated plots.",
    )
    parser.add_argument(
        "--dump-only",
        action="store_true",
        help="Only export filtered CSVs (for separate scaleup/wide runs).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scaleup_csv, wide_csv, scaleout_df = export_filtered_csvs(args.csv, args.out_dir)
    generate_scaleout_plot(scaleout_df, args.out_dir)

    if args.dump_only:
        print("Dump-only mode enabled: skipped scaleup.py, wide_subplots.py, and mistral scatter generation.")
        return

    run_script(
        "scaleup.py",
        ["--csv", str(scaleup_csv), "--out", str(args.out_dir / "scaleup_heatmap_pies.pdf")],
    )
    run_script(
        "wide_subplots.py",
        ["--csv", str(wide_csv), "--out", str(args.out_dir / "E2E_PD_breakdown_subplots.pdf")],
    )
    run_script(
        "throughput_energy_scatter.py",
        ["--csv", str(args.csv), "--out", str(args.out_dir / "mistral_scatter.pdf")],
    )


if __name__ == "__main__":
    main()
