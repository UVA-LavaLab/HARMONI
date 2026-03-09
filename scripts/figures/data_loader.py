"""
Common data loading and filtering utilities for ISPASS paper figures.
Provides unified interface for loading, filtering, and preparing data.
"""

import pandas as pd
import numpy as np


def load_and_filter_data(csv_file, filters=None):
    """
    Load CSV data and apply filters.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file
    filters : dict, optional
        Dictionary with filtering criteria:
        - models: list of model names
        - systems: list of system names
        - batches: list of batch sizes
        - in_out_pairs: list of (lin, lout) tuples
        - metrics: list of metric column names to keep (all others dropped)
    
    Returns:
    --------
    pandas.DataFrame
        Filtered and normalized dataframe
    """
    df = pd.read_csv(csv_file)
    
    # Normalize column names (remove units)
    rename_map = {
        'prefill_latency(ms)': 'prefill_latency',
        'decode_latency(ms)': 'decode_latency',
        'e2e_latency(ms)': 'e2e_latency',
        'decode_throughput(tok/s)': 'decode_throughput',
        'total_energy(J)': 'total_energy',
    }
    existing_renames = {src: dst for src, dst in rename_map.items() if src in df.columns}
    if existing_renames:
        df = df.rename(columns=existing_renames)
    
    # Apply filters and preserve order
    if filters:
        if 'models' in filters and filters['models']:
            df = df[df['model'].isin(filters['models'])]
        
        if 'systems' in filters and filters['systems']:
            df = df[df['system'].isin(filters['systems'])]
            # Preserve filter order using categorical
            df['system'] = pd.Categorical(df['system'], categories=filters['systems'], ordered=True)
        
        if 'batches' in filters and filters['batches']:
            df = df[df['batch'].isin(filters['batches'])]
        
        if 'in_out_pairs' in filters and filters['in_out_pairs']:
            mask = pd.Series([False] * len(df), index=df.index)
            for lin, lout in filters['in_out_pairs']:
                mask |= ((df['lin'] == lin) & (df['lout'] == lout))
            df = df[mask]
        
        # Filter columns by metrics if specified
        if 'metrics' in filters and filters['metrics']:
            keep_cols = ['model', 'system', 'batch', 'lin', 'lout'] + filters['metrics']
            df = df[[col for col in keep_cols if col in df.columns]]
    
    # Sort for consistent ordering (system will be sorted by categorical order if set)
    df = df.sort_values(['model', 'batch', 'lin', 'lout', 'system']).reset_index(drop=True)
    
    # Convert categorical back to string for easier handling
    if pd.api.types.is_categorical_dtype(df['system']):
        df['system'] = df['system'].astype(str)
    
    return df


def normalize_to_baseline(df, metric_col, baseline_system_prefix='H100', 
                          higher_is_better=True):
    """
    Normalize metric values to baseline system per configuration.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    metric_col : str
        Column name of metric to normalize
    baseline_system_prefix : str or dict
        Prefix of baseline system name (default: 'H100')
        Can be a string (same baseline for all models) or
        dict mapping model names to baseline system names/prefixes
    higher_is_better : bool
        If True, compute speedup as system/baseline (for throughput)
        If False, compute speedup as baseline/system (for latency)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with additional 'speedup' column
    """
    df = df.copy()
    df['speedup'] = np.nan
    
    # Group by configuration
    for (model, batch, lin, lout), group in df.groupby(['model', 'batch', 'lin', 'lout']):
        # Determine baseline for this model
        if isinstance(baseline_system_prefix, dict):
            baseline_prefix = baseline_system_prefix.get(model, 'H100')
        else:
            baseline_prefix = baseline_system_prefix
        
        # Find baseline
        baseline_rows = group[group['system'].str.startswith(baseline_prefix, na=False)]
        if baseline_rows.empty or baseline_rows.iloc[0][metric_col] == 0:
            continue
        
        baseline_value = baseline_rows.iloc[0][metric_col]
        
        # Compute speedup for all systems in this config
        for idx in group.index:
            system_value = df.loc[idx, metric_col]
            if higher_is_better:
                df.loc[idx, 'speedup'] = system_value / baseline_value
            else:
                df.loc[idx, 'speedup'] = baseline_value / system_value
    
    return df


def prepare_stacked_data(df, stack_metrics):
    """
    Prepare dataframe for stacked bar charts.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    stack_metrics : list of str
        List of metric columns to stack (e.g., ['comp_pct', 'comm_pct', 'queue_pct'])
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with normalized stacked percentages
    """
    df = df.copy()
    
    # Filter rows that have all stack metrics
    mask = pd.Series([True] * len(df), index=df.index)
    for metric in stack_metrics:
        if metric in df.columns:
            mask &= ~df[metric].isna()
    df = df[mask]
    
    # Normalize percentages to sum to 100%
    total = sum(df[m] for m in stack_metrics if m in df.columns)
    for metric in stack_metrics:
        if metric in df.columns:
            df[f'{metric}_norm'] = np.where(total > 0, (df[metric] / total) * 100, 0)
    
    return df


def save_filtered_data(df, output_path):
    """Save filtered dataframe to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Saved filtered data to: {output_path}")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")


def get_unique_configs(df):
    """Get unique configurations (model, batch, lin, lout)."""
    return df[['model', 'batch', 'lin', 'lout']].drop_duplicates().reset_index(drop=True)


def get_available_systems(df, filters=None):
    """Get list of unique systems in order of first appearance (preserves filter order)."""
    # If filters with system order are provided, use that order
    if filters and 'systems' in filters and filters['systems']:
        return [s for s in filters['systems'] if s in df['system'].values]
    # Otherwise, use order of first appearance in dataframe
    return df['system'].unique().tolist()
