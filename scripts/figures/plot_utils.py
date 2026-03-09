"""
Common plotting utilities for ISPASS paper figures.
Provides consistent styling, colors, and layout functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set Times New Roman font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # For math symbols


def apply_paper_rcparams(compact: bool = False):
    """
    Apply matplotlib rcParams tuned for paper-ready small figures.

    Notes:
    - Prefer PDF output (vector) with TrueType fonts for crisp text.
    - Keep linewidths/ticks slightly thicker than default so they survive
      downscaling in the final PDF.
    - `compact=True` is tuned for ~3.5in-wide multi-panel figures.
    """
    base_font = 7 if compact else 8

    plt.rcParams.update(
        {
            # Fonts
            "font.size": base_font,
            "axes.labelsize": base_font,
            "axes.titlesize": base_font,
            "xtick.labelsize": base_font - 1,
            "ytick.labelsize": base_font - 1,
            "legend.fontsize": base_font - 1,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "mathtext.fontset": "stix",
            # Lines / spines / ticks (thicker than defaults for small figs)
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            # Grid
            "grid.linewidth": 0.5,
            "grid.alpha": 0.25,
            # Legend
            "legend.frameon": False,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.2,
            "legend.columnspacing": 0.8,
            # Saving / PDF correctness
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,  # TrueType fonts in PDF
            "ps.fonttype": 42,
        }
    )


def get_muted_colors(n_colors):
    """
    Get muted color palette for systems.
    
    Parameters:
    -----------
    n_colors : int
        Number of colors needed
    
    Returns:
    --------
    list
        List of RGB tuples
    """
    # Use seaborn muted palette
    return sns.color_palette("muted", n_colors)


def get_muted_colors_no_red(n_colors):
    """
    Get muted color palette while avoiding red shades.
    """
    base = sns.color_palette("muted", max(n_colors + 2, 8))
    filtered = [
        c for c in base
        if not (c[0] > 0.6 and c[0] > c[1] + 0.15 and c[0] > c[2] + 0.15)
    ]
    if len(filtered) < n_colors:
        # Fallback to tab10 (skip red at index 3)
        tab = list(sns.color_palette("tab10"))
        filtered = [tab[i] for i in range(len(tab)) if i != 3]
    return filtered[:n_colors]


def get_okabe_ito_colors(n_colors, skip_black=False):
    """
    Get Okabe-Ito colorblind-safe palette.
    """
    okabe_ito = [
        (0.0, 0.0, 0.0),          # Black
        (0.902, 0.624, 0.0),      # Orange
        (0.337, 0.706, 0.914),    # Sky blue
        (0.0, 0.62, 0.451),       # Bluish green
        (0.941, 0.894, 0.259),    # Yellow
        (0.0, 0.447, 0.698),      # Blue
        (0.835, 0.369, 0.0),      # Vermillion
        (0.8, 0.475, 0.655),      # Reddish purple
    ]
    if skip_black:
        okabe_ito = okabe_ito[1:]
    if n_colors <= len(okabe_ito):
        return okabe_ito[:n_colors]
    # Repeat if more colors are requested
    return (okabe_ito * ((n_colors // len(okabe_ito)) + 1))[:n_colors]


def setup_figure(n_cols=1, height=2.0):
    """
    Create figure with appropriate size.
    
    Parameters:
    -----------
    n_cols : int
        Number of columns (1 or 2)
    height : float
        Height in inches
    
    Returns:
    --------
    fig, ax
        Matplotlib figure and axis objects
    """
    # if n_cols == 1:
    #     width = 3.5  # Single column width
    # else:
    #     width = 7.0  # Two column width
    width = n_cols * 3.5 
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def calculate_bar_width(n_bars, figure_width):
    """
    Calculate appropriate bar width based on number of bars and figure width.
    
    Parameters:
    -----------
    n_bars : int
        Total number of bars to plot
    figure_width : float
        Width of figure in inches
    
    Returns:
    --------
    float
        Bar width (0.3 to 1.0)
    """
    # Calculate points per bar (assuming ~70 points per inch)
    points_per_inch = 70
    total_points = figure_width * points_per_inch
    points_per_bar = total_points / n_bars
    
    # Scale bar width: more bars -> narrower bars
    if points_per_bar > 30:
        return 1.0
    elif points_per_bar > 20:
        return 0.8
    elif points_per_bar > 15:
        return 0.6
    elif points_per_bar > 10:
        return 0.4
    else:
        return 0.3


def add_hierarchical_xlabels(ax, group_boundaries, show_model_labels=True, 
                             rotate_labels=False, group_by_inout=False):
    """
    Add multi-level hierarchical labels to x-axis.
    
    Parameters:
    -----------
    ax : matplotlib axis
    group_boundaries : list of dict
        Each dict contains: 'start', 'end', 'model', 'batch', 'lin', 'lout'
    show_model_labels : bool
        Whether to show model labels
    rotate_labels : bool
        Whether to rotate labels (for space-constrained plots)
    group_by_inout : bool
        If True, adjust label positioning for grouped bars
    """
    level_order = ['lin', 'lout', 'batch', 'model']
    
    # Calculate y positions for each level
    y_positions = {}
    y_offset = 0.045
    for i, level in enumerate(level_order):
        y_positions[level] = -0.045 - (i * y_offset)
    
    # Rotation angle
    rotation = 45 if rotate_labels else 0
    ha = 'right' if rotate_labels else 'center'
    
    # Group by model for model labels
    if show_model_labels:
        model_groups = {}
        for group in group_boundaries:
            model = group['model']
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(group)
        
        for model, groups in model_groups.items():
            if groups:
                first_start = groups[0]['start']
                last_end = groups[-1]['end']
                model_center = (first_start + last_end) / 2
                ax.text(model_center, y_positions['model'], model,
                       ha=ha, va='top', fontsize=6, fontweight='bold',
                       rotation=rotation, transform=ax.get_xaxis_transform())
    
    # Group by batch
    batch_groups = {}
    for group in group_boundaries:
        model = group['model']
        batch = group['batch']
        batch_key = f"{model}_{batch}"
        if batch_key not in batch_groups:
            batch_groups[batch_key] = []
        batch_groups[batch_key].append(group)
    
    for batch_key, groups in batch_groups.items():
        if groups:
            first_start = groups[0]['start']
            last_end = groups[-1]['end']
            batch_center = (first_start + last_end) / 2
            batch_num = groups[0]['batch']
            ax.text(batch_center, y_positions['batch'], f'B{batch_num}',
                   ha=ha, va='top', fontsize=6, fontweight='bold',
                   rotation=rotation, transform=ax.get_xaxis_transform())
    
    # Format in/out for display: 2048 -> 2k
    def _fmt(v):
        return '2k' if v == 2048 else str(v)

    # Add lin/lout labels - one per group (since each group is one inout pair)
    for group in group_boundaries:
        # For grouped bars, center label over the entire group
        center_pos = (group['start'] + group['end']) / 2
        lin = group['lin']
        lout = group['lout']
        
        ax.text(center_pos, y_positions['lin'], _fmt(lin),
               ha=ha, va='top', fontsize=5, rotation=rotation,
               transform=ax.get_xaxis_transform())
        ax.text(center_pos, y_positions['lout'], _fmt(lout),
               ha=ha, va='top', fontsize=5, rotation=rotation,
               transform=ax.get_xaxis_transform())
    
    # Add parameter labels on left
    for level in level_order:
        if level == 'model' and not show_model_labels:
            continue
        label_name = level.capitalize()
        if level == 'lin':
            label_name = 'In'
        elif level == 'lout':
            label_name = 'Out'
        ax.text(-0.5, y_positions[level], label_name,
               ha='right', va='top', fontsize=6, fontweight='bold',
               transform=ax.get_xaxis_transform())
    
    # Add vertical separators between batch groups
    for i, (batch_key, groups) in enumerate(batch_groups.items()):
        if i < len(batch_groups) - 1:
            last_end = groups[-1]['end']
            # For grouped bars, separator goes after the last bar plus half the gap
            if group_by_inout:
                separator_pos = last_end + 0.25
            else:
                separator_pos = last_end + 0.25
            ax.plot([separator_pos, separator_pos], [0.0, -0.17],
                   transform=ax.get_xaxis_transform(),
                   color='black', linewidth=0.8, alpha=0.6,
                   solid_capstyle='butt', clip_on=False)


def style_axis(ax, ylabel, show_grid=True):
    """
    Apply consistent styling to axis.
    
    Parameters:
    -----------
    ax : matplotlib axis
    ylabel : str
        Y-axis label text
    show_grid : bool
        Whether to show grid
    """
    ax.set_ylabel(ylabel, fontsize=8)
    if show_grid:
        ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='y', labelsize=7)
    ax.margins(x=0.01, y=0.005)
    
    # Remove x-axis ticks
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', length=0)


def add_baseline_line(ax):
    """Add dashed horizontal line at y=1 for normalized plots."""
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.8, alpha=0.7)


def save_figure(fig, output_path, dpi=4000):
    """Save figure with tight layout."""
    plt.tight_layout(pad=0.3)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
