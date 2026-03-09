"""
Two-subplot wide breakdown plot (clipped y-axis):
- Left subplot: 7B models (LLAMA2-7B, MISTRAL-7B)
- Right subplot: 70B model (LLAMA3-70B)

Latency is plotted in seconds (s). Per panel, the y-cap is set to the next
tick above the largest E2E latency among non-H100/H100-2 systems.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import (
    add_hierarchical_xlabels,
    calculate_bar_width,
    get_muted_colors,
    style_axis,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HARMONI_HOME = os.environ.get("HARMONI_HOME", os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
DEFAULT_CSV = os.path.join(HARMONI_HOME, "results", "plots", "filtered_wide.csv")
DEFAULT_OUT = os.path.join(HARMONI_HOME, "results", "plots", "E2E_PD_breakdown_subplots.pdf")

MODEL_ORDER = ["LLAMA2-7B", "MISTRAL-7B", "LLAMA3-70B"]
SYSTEM_ORDER = [
    "H100",
    "H100-2",
    "DDR5-M4-R4-C8-8-A2",
    "DDR5-M8-R4-C8-8-A2",
    "DDR5-M16-R4-C8-8-A2",
    "DDR5-M16-R8-C8-8-A2",
]
ACCEL_SYSTEMS = {"H100", "H100-2"}


def _nice_step(value):
    if value < 1:
        return 0.1
    if value < 5:
        return 0.25
    if value < 20:
        return 1.0
    if value < 100:
        return 5.0
    return 10.0


def _next_tick_cap(value):
    """Return the next tick value strictly above `value`."""
    if value <= 0:
        return 1.0
    step = _nice_step(value)
    k = np.floor(value / step)
    return float((k + 1.0) * step)


def _format_seconds(value):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _build_bar_data(
    df,
    stack_metrics,
    fig_width,
    system_order,
    system_color_map,
    bar_width_override=None,
):
    unique_configs = df[["model", "batch", "lin", "lout"]].drop_duplicates().reset_index(drop=True)

    total_bars = 0
    for _, cfg in unique_configs.iterrows():
        cfg_data = df[
            (df["model"] == cfg["model"])
            & (df["batch"] == cfg["batch"])
            & (df["lin"] == cfg["lin"])
            & (df["lout"] == cfg["lout"])
        ]
        total_bars += len(cfg_data["system"].unique())

    total_bars = max(total_bars, 1)
    bar_width = (
        bar_width_override
        if bar_width_override is not None
        else calculate_bar_width(total_bars, fig_width)
    )
    bar_spacing = bar_width

    available_systems = [s for s in system_order if s in df["system"].values]

    x_positions = []
    system_colors = []
    stack_data = {metric: [] for metric in stack_metrics}
    e2e_values = []
    group_boundaries = []
    current_pos = 0.0

    for _, cfg in unique_configs.iterrows():
        cfg_data = df[
            (df["model"] == cfg["model"])
            & (df["batch"] == cfg["batch"])
            & (df["lin"] == cfg["lin"])
            & (df["lout"] == cfg["lout"])
        ]
        if cfg_data.empty:
            continue

        existing_systems = [sys for sys in available_systems if sys in cfg_data["system"].values]
        if not existing_systems:
            continue

        group_start = current_pos
        group_end = current_pos + (len(existing_systems) - 1) * bar_spacing
        group_boundaries.append(
            {
                "start": group_start,
                "end": group_end,
                "model": cfg["model"],
                "batch": cfg["batch"],
                "lin": cfg["lin"],
                "lout": cfg["lout"],
            }
        )

        for system in existing_systems:
            row = cfg_data[cfg_data["system"] == system].iloc[0]
            x_positions.append(current_pos)
            system_colors.append(system_color_map[system])
            e2e_values.append(float(row["e2e_latency_s"]) if "e2e_latency_s" in row else np.nan)
            for metric in stack_metrics:
                stack_data[metric].append(float(row[metric]) if metric in row else 0.0)
            current_pos += bar_spacing

        current_pos += 0.5

    return {
        "x_positions": np.asarray(x_positions, dtype=float),
        "system_colors": np.asarray(system_colors, dtype=object),
        "stack_data": {k: np.asarray(v, dtype=float) for k, v in stack_data.items()},
        "e2e_values": np.asarray(e2e_values, dtype=float),
        "group_boundaries": group_boundaries,
        "bar_width": bar_width,
    }


def _plot_stacked_on_axis(
    ax,
    df,
    stack_metrics,
    metric_labels,
    ylabel,
    fig_width,
    system_order,
    system_color_map,
    bar_width_override=None,
):
    data = _build_bar_data(
        df=df,
        stack_metrics=stack_metrics,
        fig_width=fig_width,
        system_order=system_order,
        system_color_map=system_color_map,
        bar_width_override=bar_width_override,
    )

    x_positions = data["x_positions"]
    colors = data["system_colors"]
    bar_width = data["bar_width"]
    hatches = ["/////", ".....", "", "*****", "xxxxx", "|||||"]

    totals = np.zeros(len(x_positions))
    for idx, metric in enumerate(stack_metrics):
        values = data["stack_data"][metric]
        label = metric_labels.get(metric, metric.replace("_", " ").title())
        ax.bar(
            x_positions,
            values,
            bar_width,
            bottom=totals,
            color=colors,
            alpha=0.8,
            hatch=hatches[idx % len(hatches)],
            edgecolor="black",
            linewidth=0.3,
            label=label,
        )
        totals += values

    add_hierarchical_xlabels(
        ax,
        data["group_boundaries"],
        show_model_labels=True,
        rotate_labels=False,
        group_by_inout=True,
    )
    style_axis(ax, ylabel)
    data["total_heights"] = totals
    return data


def _remove_left_level_labels(ax):
    for text in list(ax.texts):
        if text.get_text() in {"In", "Out", "Batch", "Model"}:
            text.remove()


def _count_bars(df):
    count = 0
    unique_cfg = df[["model", "batch", "lin", "lout"]].drop_duplicates()
    for _, cfg in unique_cfg.iterrows():
        cfg_data = df[
            (df["model"] == cfg["model"])
            & (df["batch"] == cfg["batch"])
            & (df["lin"] == cfg["lin"])
            & (df["lout"] == cfg["lout"])
        ]
        count += len(cfg_data["system"].unique())
    return max(count, 1)


def _determine_cap_from_non_accel(df_panel):
    non_accel = df_panel[~df_panel["system"].isin(ACCEL_SYSTEMS)]
    if non_accel.empty:
        return None
    non_accel_max = float(non_accel["e2e_latency_s"].max())
    return _next_tick_cap(non_accel_max)


def _apply_clip_with_labels(ax, x_positions, total_heights, clip_cap, cap_padding=0.25):
    upper_limit = clip_cap + cap_padding
    ax.set_ylim(0, upper_limit)
    ax.axhline(clip_cap, color="black", linestyle="--", linewidth=0.6, alpha=0.6)

    clipped = [(x, h) for x, h in zip(x_positions, total_heights) if h > clip_cap]
    if not clipped:
        return 0

    for x_pos, height in clipped:
        ax.text(
            x_pos,
            upper_limit,
            _format_seconds(float(height)),
            ha="center",
            va="top",
            fontsize=5.5,
            fontweight="bold",
            rotation=0,
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "black",
                "linewidth": 0.35,
                "boxstyle": "square,pad=0.12",
            },
        )
    return len(clipped)


def _add_broken_axis_marker(ax, last_tick, tick_step):
    ax.text(
        0.0,
        last_tick + (0.02 * tick_step),
        "~",
        transform=ax.get_yaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )


def _clean_legend_label(system_name):
    return system_name.replace("-A2", "")


def main():
    parser = argparse.ArgumentParser(
        description="Create wide latency breakdown subplots (7B vs 70B) with clipped y-axes."
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Input filtered CSV path.",
    )
    parser.add_argument("--width", type=float, default=7.0, help="Figure width (inches).")
    parser.add_argument("--height", type=float, default=2.0, help="Figure height (inches).")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Missing input CSV: {args.csv}")
    df = pd.read_csv(args.csv)

    if "prefill_latency_s" not in df.columns:
        if "prefill_latency" not in df.columns:
            raise ValueError("Missing prefill latency column.")
        df["prefill_latency_s"] = df["prefill_latency"] / 1000.0
    if "decode_latency_s" not in df.columns:
        if "decode_latency" not in df.columns:
            raise ValueError("Missing decode latency column.")
        df["decode_latency_s"] = df["decode_latency"] / 1000.0
    if "e2e_latency_s" not in df.columns:
        if "e2e_latency" not in df.columns:
            raise ValueError("Missing e2e latency column.")
        df["e2e_latency_s"] = df["e2e_latency"] / 1000.0

    required_cols = {
        "model",
        "system",
        "batch",
        "lin",
        "lout",
        "prefill_latency_s",
        "decode_latency_s",
        "e2e_latency_s",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    model_order = [m for m in MODEL_ORDER if m in df["model"].values]
    if not model_order:
        raise ValueError("No expected models found in input data.")

    system_order = [s for s in SYSTEM_ORDER if s in df["system"].values]
    if not system_order:
        raise ValueError("No expected systems found in input data.")

    colors = get_muted_colors(len(system_order))
    system_color_map = {system: colors[idx] for idx, system in enumerate(system_order)}

    df = df.copy()
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df["system"] = pd.Categorical(df["system"], categories=system_order, ordered=True)
    df = df.sort_values(["model", "batch", "lin", "lout", "system"]).reset_index(drop=True)

    df_7b = df[df["model"].isin(["LLAMA2-7B", "MISTRAL-7B"])].copy()
    df_70b = df[df["model"] == "LLAMA3-70B"].copy()
    if df_7b.empty or df_70b.empty:
        raise ValueError("Need both 7B and 70B data to build split subplots.")

    n_7b = _count_bars(df_7b)
    n_70b = _count_bars(df_70b)
    stack_metrics = ["prefill_latency_s", "decode_latency_s"]
    metric_labels = {"prefill_latency_s": "Prefill", "decode_latency_s": "Decode"}
    ylabel = "E2E Latency (s)"

    fig, (ax_7b, ax_70b) = plt.subplots(
        1,
        2,
        figsize=(args.width, args.height),
        sharey=False,
        gridspec_kw={"width_ratios": [n_7b, n_70b]},
    )

    shared = _build_bar_data(
        df=df,
        stack_metrics=stack_metrics,
        fig_width=args.width,
        system_order=system_order,
        system_color_map=system_color_map,
    )
    data_7b = _plot_stacked_on_axis(
        ax=ax_7b,
        df=df_7b,
        stack_metrics=stack_metrics,
        metric_labels=metric_labels,
        ylabel=ylabel,
        fig_width=args.width * (n_7b / (n_7b + n_70b)),
        system_order=system_order,
        system_color_map=system_color_map,
        bar_width_override=shared["bar_width"],
    )
    data_70b = _plot_stacked_on_axis(
        ax=ax_70b,
        df=df_70b,
        stack_metrics=stack_metrics,
        metric_labels=metric_labels,
        ylabel=ylabel,
        fig_width=args.width * (n_70b / (n_7b + n_70b)),
        system_order=system_order,
        system_color_map=system_color_map,
        bar_width_override=shared["bar_width"],
    )

    cap_7b = _determine_cap_from_non_accel(df_7b)
    cap_70b = _determine_cap_from_non_accel(df_70b)
    clipped_7b = 0
    clipped_70b = 0
    pad_7b = 0.0
    pad_70b = 0.0
    step_7b = 0.0
    step_70b = 0.0
    if cap_7b is not None:
        step_7b = cap_7b / 4.0
        pad_7b = 0.3 * step_7b
    if cap_70b is not None:
        step_70b = cap_70b / 4.0
        pad_70b = 0.3 * step_70b

    if cap_7b is not None:
        clipped_7b = _apply_clip_with_labels(
            ax_7b,
            data_7b["x_positions"],
            data_7b["total_heights"],
            cap_7b,
            cap_padding=pad_7b,
        )
    if cap_70b is not None:
        clipped_70b = _apply_clip_with_labels(
            ax_70b,
            data_70b["x_positions"],
            data_70b["total_heights"],
            cap_70b,
            cap_padding=pad_70b,
        )

    if cap_7b is not None:
        ax_7b.set_yticks(np.linspace(0, cap_7b, 5))
    if cap_70b is not None:
        ax_70b.set_yticks(np.linspace(0, cap_70b, 5))

    ax_70b.set_ylabel("")
    ax_7b.tick_params(axis="y", labelleft=True, left=True, labelright=False, right=False, pad=1)
    ax_70b.tick_params(axis="y", labelleft=True, left=True, labelright=False, right=False, pad=1)
    _remove_left_level_labels(ax_70b)

    legend_elements = []
    for system in system_order:
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=system_color_map[system],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.3,
                label=_clean_legend_label(system),
            )
        )
    hatches = ["/////", ".....", "", "*****", "xxxxx", "|||||"]
    for idx, metric in enumerate(stack_metrics):
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="gray",
                alpha=0.8,
                hatch=hatches[idx % len(hatches)],
                edgecolor="black",
                linewidth=0.3,
                label=metric_labels.get(metric, metric),
            )
        )

    legend = fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.868),
        ncol=len(legend_elements),
        frameon=True,
        fancybox=False,
        shadow=False,
        fontsize=6,
        columnspacing=0.6,
        handlelength=1.0,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.5)

    if cap_7b is not None:
        _add_broken_axis_marker(ax_7b, cap_7b, step_7b)
    if cap_70b is not None:
        _add_broken_axis_marker(ax_70b, cap_70b, step_70b)

    fig.subplots_adjust(left=0.06, right=0.99, wspace=0.08, top=0.855, bottom=0.20)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=6000, bbox_inches="tight")
    plt.close(fig)

    if cap_7b is not None:
        print(f"7B cap from non-H100/H100-2 max: {cap_7b:g}s (clipped bars: {clipped_7b})")
    if cap_70b is not None:
        print(f"70B cap from non-H100/H100-2 max: {cap_70b:g}s (clipped bars: {clipped_70b})")
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()
