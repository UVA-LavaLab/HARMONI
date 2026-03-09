#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Wedge
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
HARMONI_HOME = Path(os.environ.get("HARMONI_HOME", SCRIPT_DIR.parents[1]))
DEFAULT_CSV = HARMONI_HOME / "results" / "plots" / "filtered_scale_up.csv"
DEFAULT_OUT = HARMONI_HOME / "results" / "plots" / "scaleup_heatmap_pies.pdf"

BATCHES = [1, 4, 8]
COMPONENTS = ["comp_pct", "comm_pct", "queue_pct"]
COMP_COLORS = {
    "comp_pct": "#1f77b4",
    "comm_pct": "#ff7f0e",
    "queue_pct": "#7f7f7f",
}


def extract_r_c(system):
    r_match = re.search(r"-R(\d+)-", str(system))
    c_match = re.search(r"-C(\d+)-", str(system))
    if r_match is None or c_match is None:
        raise ValueError(f"Unexpected system format (missing R/C fields): {system}")
    return pd.Series({"R": int(r_match.group(1)), "C": int(c_match.group(1))})


def plot_scaleup(df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    required = {"system", "batch", "e2e_latency", "comp_pct", "comm_pct", "queue_pct"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[["R", "C"]] = df["system"].apply(extract_r_c)

    fig = plt.figure(figsize=(3.4, 1.75))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.0, 0.11],
        hspace=0.1,
        wspace=0.06,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    norm = Normalize(vmin=df["e2e_latency"].min(), vmax=df["e2e_latency"].max())
    cmap = plt.cm.Reds
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for i, (ax, batch) in enumerate(zip(axes, BATCHES)):
        d = df[df["batch"] == batch]
        if d.empty:
            continue

        r_vals = sorted(d["R"].unique())
        c_vals = sorted(d["C"].unique())
        r_map = {r: i for i, r in enumerate(r_vals)}
        c_map = {c: i for i, c in enumerate(c_vals)}

        heat = np.full((len(r_vals), len(c_vals)), np.nan)
        for _, row in d.iterrows():
            heat[r_map[row.R], c_map[row.C]] = row.e2e_latency

        ax.imshow(heat, cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
        ax.set_xlim(-0.5, len(c_vals) - 0.5)
        ax.set_ylim(-0.5, len(r_vals) - 0.5)

        ax.set_xticks(np.arange(-0.5, len(c_vals), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(r_vals), 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)

        radius = 0.42
        for _, row in d.iterrows():
            x, y = c_map[row.C], r_map[row.R]
            total = row[COMPONENTS].sum()
            start = 0.0
            for comp in COMPONENTS:
                frac = row[comp] / total
                if frac <= 0:
                    continue
                ax.add_patch(
                    Wedge(
                        (x, y),
                        radius,
                        start * 360,
                        (start + frac) * 360,
                        facecolor=COMP_COLORS[comp],
                        edgecolor="black",
                        linewidth=0.3,
                    )
                )
                start += frac

        ax.set_xticks(range(len(c_vals)))
        ax.set_xticklabels(c_vals, fontsize=7)
        ax.set_xlabel("Chips", fontsize=8, labelpad=1)

        ax.set_yticks(range(len(r_vals)))
        if i == 0:
            ax.set_yticklabels(r_vals, fontsize=7)
            ax.set_ylabel("Ranks", fontsize=8, labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"Batch={batch}", fontsize=8, pad=1.5)

    cax = legend_ax.inset_axes([0.02, 0.35, 0.38, 0.30])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=6, length=2, pad=1)

    legend_ax.text(0.02, 0.50, "E2E(ms)", fontsize=7, ha="right", va="center")
    pie_handles = [
        Patch(facecolor=COMP_COLORS["comp_pct"], edgecolor="black", label="Comp"),
        Patch(facecolor=COMP_COLORS["comm_pct"], edgecolor="black", label="Comm"),
        Patch(facecolor=COMP_COLORS["queue_pct"], edgecolor="black", label="Queue"),
    ]
    legend_ax.legend(
        handles=pie_handles,
        loc="center left",
        bbox_to_anchor=(0.45, 0.50),
        ncol=3,
        frameon=False,
        fontsize=7,
        handlelength=0.9,
        columnspacing=0.65,
    )

    plt.subplots_adjust(left=0.11, right=0.98, top=1, bottom=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the scale-up heatmap pie plot.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input filtered_scale_up CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output figure path.")
    parser.add_argument("--dpi", type=int, default=1500, help="Output DPI.")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.csv}")

    df = pd.read_csv(args.csv)
    plot_scaleup(df, args.out, args.dpi)
    print(f"Saved figure: {args.out}")


if __name__ == "__main__":
    main()
