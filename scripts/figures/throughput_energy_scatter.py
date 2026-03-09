#!/usr/bin/env python3
"""
Compact MISTRAL tokens-efficiency scatter from hierarchical_data.csv.

Filters:
  - model: MISTRAL-7B
  - systems: H100 and DDR5-M4-R4-C8-8-A2
  - batch: 1 and 8
  - in/out pairs: (128,128), (128,2048), (2048,128), (2048,2048)

Metrics are computed directly from hierarchical_data.csv:
  - x: total_Ktokens/second
  - y: total_tokens/Joule
"""

import argparse
from itertools import product
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

from plot_utils import apply_paper_rcparams, get_muted_colors


SCRIPT_DIR = Path(__file__).resolve().parent
HARMONI_HOME = Path(os.environ.get("HARMONI_HOME", SCRIPT_DIR.parents[1]))

MODEL_NAME = "MISTRAL-7B"
SYSTEMS = ["H100", "DDR5-M4-R4-C8-8-A2"]
INPUTS = [128, 2048]
OUTPUTS = [128, 2048]
BATCHES = [1, 8]

INOUT_MARKERS = {
    (128, 128): "*",
    (128, 2048): "s",
    (2048, 128): "^",
    (2048, 2048): "D",
}

# Smaller markers for both batches, with a larger ratio between B=8 and B=1.
BATCH_MARKER_SIZES = {1: 12, 8: 36}


def required_columns() -> list[str]:
    return [
        "model",
        "system",
        "batch",
        "lin",
        "lout",
        "e2e_latency(ms)",
        "total_energy(J)",
    ]


def format_system_label(system: str) -> str:
    return system.replace("-A2", "")


def format_inout_label(lin: int, lout: int) -> str:
    return f"{lin}i/{lout}o"


def parse_label_points(label_specs: list[str]) -> dict[tuple[str, int, int, int], str]:
    """
    Accepts repeated specs:
      SYSTEM|BATCH|LIN|LOUT
      SYSTEM|BATCH|LIN|LOUT|LABEL_TEXT
    """
    label_map: dict[tuple[str, int, int, int], str] = {}
    for raw in label_specs:
        parts = raw.split("|")
        if len(parts) not in (4, 5):
            raise ValueError(
                f"Invalid --label-point '{raw}'. Expected SYSTEM|BATCH|LIN|LOUT[|LABEL_TEXT]."
            )
        system = parts[0].strip()
        batch, lin, lout = int(parts[1]), int(parts[2]), int(parts[3])
        label_text = parts[4].strip() if len(parts) == 5 else f"{lin}i/{lout}o, B={batch}"
        label_map[(system, batch, lin, lout)] = label_text
    return label_map


def load_compute_and_filter(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    out = df.copy()
    for col in ["batch", "lin", "lout", "e2e_latency(ms)", "total_energy(J)"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["batch", "lin", "lout", "e2e_latency(ms)", "total_energy(J)"])
    out = out[(out["e2e_latency(ms)"] > 0) & (out["total_energy(J)"] > 0)]

    # Requested compact subset only.
    out = out[
        (out["model"] == MODEL_NAME)
        & (out["system"].isin(SYSTEMS))
        & (out["batch"].isin(BATCHES))
        & (out["lin"].isin(INPUTS))
        & (out["lout"].isin(OUTPUTS))
    ].copy()

    if out.empty:
        return out

    out["batch"] = out["batch"].astype(int)
    out["lin"] = out["lin"].astype(int)
    out["lout"] = out["lout"].astype(int)

    out["total_tokens"] = (out["lin"] + out["lout"]) * out["batch"]
    total_tokens_per_second = out["total_tokens"] / (out["e2e_latency(ms)"] / 1000.0)
    out["total_ktokens_per_second"] = total_tokens_per_second / 1000.0
    out["total_tokens_per_joule"] = out["total_tokens"] / out["total_energy(J)"]

    out = out.sort_values(["system", "batch", "lin", "lout"]).reset_index(drop=True)
    return out


def print_compact_summary(df: pd.DataFrame) -> None:
    print("Compact subset rows:", len(df))
    if df.empty:
        return

    print(df.groupby(["system", "batch"]).size().rename("rows").to_string())
    expected = set(product(BATCHES, INPUTS, OUTPUTS))
    for system in SYSTEMS:
        sub = df[df["system"] == system]
        have = set(map(tuple, sub[["batch", "lin", "lout"]].drop_duplicates().to_records(index=False)))
        missing = sorted(expected - have)
        if missing:
            print(f"Missing combinations for {system}: {missing}")


def maybe_label_point(
    ax: plt.Axes,
    row: pd.Series,
    label_map: dict[tuple[str, int, int, int], str],
) -> None:
    key = (str(row["system"]), int(row["batch"]), int(row["lin"]), int(row["lout"]))
    if key not in label_map:
        return
    ax.annotate(
        label_map[key],
        (row["total_ktokens_per_second"], row["total_tokens_per_joule"]),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=7,
        alpha=0.9,
    )


def plot_compact(
    df: pd.DataFrame,
    out_path: Path,
    label_map: dict[tuple[str, int, int, int], str],
    width: float,
    height: float,
    save_dpi: int,
) -> None:
    if df.empty:
        raise ValueError("No rows matched compact MISTRAL filters.")

    apply_paper_rcparams(compact=True)
    fig, ax = plt.subplots(1, 1, figsize=(width, height), constrained_layout=True)

    palette = get_muted_colors(len(SYSTEMS))
    color_map = {sys: palette[i] for i, sys in enumerate(SYSTEMS)}

    for _, row in df.iterrows():
        io = (int(row["lin"]), int(row["lout"]))
        marker = INOUT_MARKERS.get(io, "x")
        color = color_map.get(str(row["system"]), "#7f7f7f")
        point_size = BATCH_MARKER_SIZES.get(int(row["batch"]), 12)
        ax.scatter(
            row["total_ktokens_per_second"],
            row["total_tokens_per_joule"],
            s=point_size,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=0.35,
            alpha=0.72,
        )
        maybe_label_point(ax, row, label_map)

    ax.set_xlabel("Ktokens/sec", fontsize=8, labelpad=1)
    ax.set_ylabel("tokens/J", fontsize=8, labelpad=1)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.grid(True, alpha=0.25)

    h100_handle, ddr_handle = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map[s],
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=format_system_label(s),
            markersize=4.6,
        )
        for s in SYSTEMS
    ]
    io_128_128 = Line2D([0], [0], marker=INOUT_MARKERS[(128, 128)], linestyle="None", color="black", label=format_inout_label(128, 128), markersize=4.2)
    io_2048_2048 = Line2D([0], [0], marker=INOUT_MARKERS[(2048, 2048)], linestyle="None", color="black", label=format_inout_label(2048, 2048), markersize=4.2)
    io_128_2048 = Line2D([0], [0], marker=INOUT_MARKERS[(128, 2048)], linestyle="None", color="black", label=format_inout_label(128, 2048), markersize=4.2)
    io_2048_128 = Line2D([0], [0], marker=INOUT_MARKERS[(2048, 128)], linestyle="None", color="black", label=format_inout_label(2048, 128), markersize=4.2)
    b1_handle = plt.scatter([], [], s=BATCH_MARKER_SIZES[1], c="white", edgecolors="black", linewidths=0.35, marker="o", label="B=1")
    b8_handle = plt.scatter([], [], s=BATCH_MARKER_SIZES[8], c="white", edgecolors="black", linewidths=0.35, marker="o", label="B=8")
    spacer = Line2D([0], [0], linestyle="None", marker=None, label="")

    # Displayed rows with ncol=3:
    # Row1: H100, DDR5-M4-R4-C8-8, (blank)
    # Row2: 128i/128o, 2048i/2048o, B=1
    # Row3: 128i/2048o, 2048i/128o, B=8
    handles = [h100_handle, io_128_128, io_128_2048, ddr_handle, io_2048_2048, io_2048_128, spacer, b1_handle, b8_handle]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.997, 0.005),
        ncol=3,
        fontsize=6,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        handlelength=1.0,
        handletextpad=0.42,
        columnspacing=0.72,
        labelspacing=0.374,
        borderpad=0.34,
        borderaxespad=0.12,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
    print(f"Saved figure: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact MISTRAL tokens-efficiency scatter from hierarchical_data.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=HARMONI_HOME / "results" / "hierarchical_data.csv",
        help="Input hierarchical_data CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HARMONI_HOME / "results" / "plots" / "mistral_scatter.pdf",
        help="Output figure path.",
    )
    parser.add_argument("--width", type=float, default=3.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=1.4, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=6000, help="Output DPI.")
    parser.add_argument(
        "--label-point",
        action="append",
        default=[],
        help="Point to label: SYSTEM|BATCH|LIN|LOUT[|LABEL_TEXT]. Can be repeated.",
    )
    args = parser.parse_args()

    label_map = parse_label_points(args.label_point)
    compact_df = load_compute_and_filter(args.csv)
    print_compact_summary(compact_df)

    plot_compact(
        df=compact_df,
        out_path=args.out,
        label_map=label_map,
        width=args.width,
        height=args.height,
        save_dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
