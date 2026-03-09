"""
Stacked bar chart plotting functions for ISPASS paper.
Supports stacking multiple metrics with patterns/hatches.
"""

import matplotlib.pyplot as plt
import numpy as np
from data_loader import get_unique_configs, get_available_systems, prepare_stacked_data
from plot_utils import (setup_figure, get_muted_colors, calculate_bar_width,
                        add_hierarchical_xlabels, style_axis, save_figure)


def plot_stacked_chart(df, stack_metrics, metric_labels=None, ylabel='Percentage (%)',
                      output_path=None, n_cols=1, show_model_labels=True, 
                      rotate_labels=False, as_percentage=True, base_metric=None,
                      group_by_inout=False):
    """
    Create stacked bar chart with multiple metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Filtered dataframe
    stack_metrics : list of str
        List of metric columns to stack (bottom to top)
    metric_labels : dict, optional
        Mapping of metric names to display labels
    ylabel : str
        Y-axis label
    output_path : str, optional
        Path to save figure
    n_cols : int
        Number of columns (1 or 2)
    show_model_labels : bool
        Whether to show model labels
    rotate_labels : bool
        Whether to rotate x-axis labels
    as_percentage : bool
        If True, normalize stack to 100% and show percentages
    base_metric : str, optional
        If provided, stack absolute values based on this metric (e.g., 'e2e_latency')
    group_by_inout : bool
        If True, group bars by (lin,lout) instead of by system
    
    Returns:
    --------
    fig, ax
        Matplotlib figure and axis objects
    """
    # Prepare stacked data
    if as_percentage:
        df = prepare_stacked_data(df, stack_metrics)
        plot_metrics = [f'{m}_norm' for m in stack_metrics if f'{m}_norm' in df.columns]
    else:
        plot_metrics = stack_metrics
    
    if df.empty:
        print("No data with complete stack metrics found!")
        return None, None
    
    # Setup figure
    fig, ax = setup_figure(n_cols=n_cols, height=2.0)
    
    # Get unique configs and systems
    unique_configs = get_unique_configs(df)
    available_systems = get_available_systems(df)
    
    # Get colors
    colors = get_muted_colors(len(available_systems))
    system_color_map = {sys: colors[i] for i, sys in enumerate(available_systems)}
    
    # Hatching patterns for stacked metrics - denser and more distinct
    hatches = ['/////', '.....', '', '*****', 'xxxxx', '|||||']
    
    # Prepare data for plotting
    x_positions = []
    system_colors = []
    group_boundaries = []
    stack_data = {m: [] for m in plot_metrics}
    base_heights = []
    current_pos = 0
    
    # Calculate adaptive bar spacing if group_by_inout is True
    if group_by_inout:
        # Estimate total number of bars to determine spacing
        total_bars = 0
        for _, config_row in unique_configs.iterrows():
            config_data = df[
                (df['model'] == config_row['model']) &
                (df['batch'] == config_row['batch']) &
                (df['lin'] == config_row['lin']) &
                (df['lout'] == config_row['lout'])
            ]
            total_bars += len(config_data['system'].unique())
        
        # Calculate adaptive bar width based on total bars and figure width
        bar_width_adaptive = calculate_bar_width(total_bars, fig.get_figwidth())
        # Use same value for position increment so bars touch
        bar_spacing = bar_width_adaptive
    else:
        bar_spacing = 1.0  # Default spacing
        bar_width_adaptive = None
    
    for _, config_row in unique_configs.iterrows():
        model = config_row['model']
        batch = config_row['batch']
        lin = config_row['lin']
        lout = config_row['lout']
        
        config_data = df[
            (df['model'] == model) &
            (df['batch'] == batch) &
            (df['lin'] == lin) &
            (df['lout'] == lout)
        ]
        
        if config_data.empty:
            continue
        
        # Get systems in this config, preserving filter order
        existing_systems = [sys for sys in available_systems if sys in config_data['system'].values]
        
        group_start = current_pos
        group_end = current_pos + (len(existing_systems) - 1) * bar_spacing
        group_boundaries.append({
            'start': group_start,
            'end': group_end,
            'model': model,
            'batch': batch,
            'lin': lin,
            'lout': lout
        })
        
        for system in existing_systems:
            row = config_data[config_data['system'] == system].iloc[0]
            
            x_positions.append(current_pos)
            system_colors.append(system_color_map[system])
            
            # Get base height if plotting absolute values
            if base_metric and not as_percentage:
                base_heights.append(row[base_metric] if base_metric in row else 0)
            else:
                base_heights.append(100 if as_percentage else 0)
            
            # Collect stack data
            for metric in plot_metrics:
                if base_metric and not as_percentage:
                    # For absolute value breakdown based on base_metric
                    # The stack metrics contain percentages, need to convert to absolute values
                    base_val = row[base_metric] if base_metric in row else 0
                    pct_val = row[metric] if metric in row else 0
                    # Convert percentage to actual value
                    stack_data[metric].append(base_val * pct_val / 100)
                elif as_percentage:
                    # Regular normalized percentage (should use _norm columns)
                    stack_data[metric].append(row[metric] if metric in row else 0)
                else:
                    # Direct absolute values (no normalization)
                    stack_data[metric].append(row[metric] if metric in row else 0)
            
            # Increment by bar_spacing
            current_pos += bar_spacing
        
        # Add gap between different (lin,lout) groups
        if group_by_inout:
            current_pos += 0.5  # Gap between (lin,lout) groups
    
    if not x_positions:
        print("No data found for stacked chart")
        return None, None
    
    # Calculate bar width
    if group_by_inout:
        # Use pre-calculated adaptive width where bar_width = bar_spacing
        bar_width = bar_width_adaptive
    else:
        # Calculate adaptive bar width for distributed bars
        bar_width = calculate_bar_width(len(x_positions), fig.get_figwidth())
    
    # Create base bars (system colors, lighter for pattern visibility)
    if as_percentage and not base_metric:
        # For normalized percentages, base bar is 100%
        ax.bar(x_positions, [100] * len(x_positions), bar_width,
               color=system_colors, alpha=0.4, edgecolor='black', linewidth=0.3)
    elif base_metric:
        # For absolute value breakdowns, base bar is the base metric
        ax.bar(x_positions, base_heights, bar_width,
               color=system_colors, alpha=0.4, edgecolor='black', linewidth=0.3)
    
    # Add stacked patterns
    bottom = np.zeros(len(x_positions))
    for i, metric in enumerate(plot_metrics):
        values = np.array(stack_data[metric])
        hatch = hatches[i % len(hatches)]
        
        # Use metric label if provided
        if metric_labels and metric in metric_labels:
            label = metric_labels[metric]
        elif metric_labels and metric.replace('_norm', '') in metric_labels:
            label = metric_labels[metric.replace('_norm', '')]
        else:
            label = metric.replace('_norm', '').replace('_', ' ').title()
        
        ax.bar(x_positions, values, bar_width, bottom=bottom,
               color=system_colors, alpha=0.8, hatch=hatch,
               edgecolor='black', linewidth=0.3, label=label)
        
        bottom += values
    
    # Add hierarchical labels
    add_hierarchical_xlabels(ax, group_boundaries, show_model_labels, rotate_labels, group_by_inout)
    
    # Style axis
    style_axis(ax, ylabel)
    
    # Set y-limits
    if as_percentage and not base_metric:
        ax.set_ylim(0, 100)
    
    # Add legend for both systems and patterns
    legend_elements = []
    
    # System colors
    for system in available_systems:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=system_color_map[system],
                         alpha=0.8, edgecolor='black', linewidth=0.3, label=system)
        )
    
    # Stack metric patterns
    for i, metric in enumerate(plot_metrics):
        hatch = hatches[i % len(hatches)]
        if metric_labels and metric in metric_labels:
            label = metric_labels[metric]
        elif metric_labels and metric.replace('_norm', '') in metric_labels:
            label = metric_labels[metric.replace('_norm', '')]
        else:
            label = metric.replace('_norm', '').replace('_', ' ').title()
        
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.8,
                         hatch=hatch, edgecolor='black', linewidth=0.3, label=label)
        )
    
    legend = ax.legend(handles=legend_elements, loc='best', fontsize=5,
                      ncol=2, frameon=True, fancybox=False, shadow=False)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)
    
    # Save if output path provided
    if output_path:
        save_figure(fig, output_path)
        plt.close(fig)
    
    return fig, ax
