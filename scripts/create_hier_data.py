#!/usr/bin/env python3
"""
Data Format Transformation Script

This script transforms CSV data from system-specific files into a hierarchical bar chart format.
All rows have consistent structure: [model, system, batch, lin, lout, prefill_latency(ms), decode_latency(ms), e2e_latency(ms), comp_pct, comm_pct, queue_pct, decode_throughput(tok/s), total_energy(J)]

For systems with breakdown data: comp_pct, comm_pct, queue_pct contain actual percentages
For systems without breakdown data: comp_pct, comm_pct, queue_pct are set to 0

Author: Generated for data transformation
"""

import pandas as pd
import os
import glob
import argparse

def load_system_data(data_dir):
    """
    Load data from system-specific CSV files that contain all metrics in one file.
    Each CSV file contains data for one model-system combination with columns:
    model, system, input, output, batch, prefill_latency(ms), decode_latency(ms), 
    e2e_latency(ms), and optionally comp_pct, comm_pct, queue_pct, decode_throughput(tok/s), total_energy(J).
    
    Args:
        data_dir: Directory containing CSV files
    
    Returns:
        Dictionary with (model, system) tuples as keys and DataFrames as values
    """
    data = {}
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        try:
            df = pd.read_csv(file_path)
            
            # Data must contain the per-pair summary columns.
            required_cols = {'model', 'system', 'input', 'output', 'batch'}
            if not required_cols.issubset(df.columns):
                print(f"Warning: Skipping {filename} (missing required columns: {sorted(required_cols)})")
                continue
            model = df['model'].iloc[0]
            system = df['system'].iloc[0]
            
            key = (model, system)
            data[key] = df
            print(f"Loaded {model} - {system}: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return data

def transform_to_hierarchical_format(data_dir, output_file):
    """
    Transform the data into the hierarchical bar chart format.
    
    Creates a consistent DataFrame where all rows have the same structure:
    [model, system, batch, lin, lout, prefill_latency(ms), decode_latency(ms), e2e_latency(ms), comp_pct, comm_pct, queue_pct, decode_throughput(tok/s), total_energy(J)]
    
    For systems with breakdown data: comp_pct, comm_pct, queue_pct contain actual percentages
    For systems without breakdown data: comp_pct, comm_pct, queue_pct are set to 0
    
    Args:
        data_dir: Directory containing CSV files
        output_file: Output file path for the transformed data
    """
    print(f"Loading data from: {data_dir}")
    
    # Load system data
    system_data = load_system_data(data_dir)
    
    if not system_data:
        print("No data found to transform")
        return
    
    # Get all unique models and systems
    models = sorted(set(key[0] for key in system_data.keys()))
    systems = sorted(set(key[1] for key in system_data.keys()))
    print(f"Found models: {models}")
    print(f"Found systems: {systems}")
    
    # Transform data
    transformed_rows = []
    
    for (model, system), df in system_data.items():
        print(f"Processing {model} - {system}: {len(df)} rows")
        
        # Check if this system has breakdown data (comp_pct, comm_pct, queue_pct columns)
        has_breakdown = all(col in df.columns for col in ['comp_pct', 'comm_pct', 'queue_pct'])
        
        # Process each row in the current system's data
        for idx, row in df.iterrows():
            # Extract basic parameters
            input_val = row['input']
            output_val = row['output']
            batch_val = row['batch']
            
            # Extract latency metrics (handle NaN values)
            prefill_latency = row['prefill_latency(ms)'] if pd.notna(row['prefill_latency(ms)']) else None
            decode_latency = row['decode_latency(ms)'] if pd.notna(row['decode_latency(ms)']) else None
            e2e_latency = row['e2e_latency(ms)'] if pd.notna(row['e2e_latency(ms)']) else None
            
            # Extract throughput and energy metrics (handle NaN values and missing columns)
            decode_throughput = row.get('decode_throughput(tok/s)', None)
            if decode_throughput is not None and pd.notna(decode_throughput):
                decode_throughput = decode_throughput
            else:
                decode_throughput = None
                
            total_energy = row.get('total_energy(J)', None)
            if total_energy is not None and pd.notna(total_energy):
                total_energy = total_energy
            else:
                total_energy = None
            
            # Handle breakdown percentages
            if has_breakdown:
                # System has breakdown data - use actual percentages or 0 if NaN
                comp_pct = row['comp_pct'] if pd.notna(row['comp_pct']) else 0
                comm_pct = row['comm_pct'] if pd.notna(row['comm_pct']) else 0
                queue_pct = row['queue_pct'] if pd.notna(row['queue_pct']) else 0
            else:
                # System doesn't have breakdown data - set all percentages to 0
                comp_pct = 0
                comm_pct = 0
                queue_pct = 0
            
            # Create row with consistent structure for all systems
            row_data = [
                model, system, batch_val, input_val, output_val,
                prefill_latency, decode_latency, e2e_latency, comp_pct, comm_pct, queue_pct,
                decode_throughput, total_energy
            ]
            
            transformed_rows.append(row_data)
    
    # Create DataFrame with consistent structure
    if transformed_rows:
        # All rows have the same structure: [model, system, batch, lin, lout, prefill_latency(ms), decode_latency(ms), e2e_latency(ms), comp_pct, comm_pct, queue_pct, decode_throughput(tok/s), total_energy(J)]
        columns = ['model', 'system', 'batch', 'lin', 'lout', 'prefill_latency(ms)', 'decode_latency(ms)', 'e2e_latency(ms)', 'comp_pct', 'comm_pct', 'queue_pct', 'decode_throughput(tok/s)', 'total_energy(J)']
        df = pd.DataFrame(transformed_rows, columns=columns)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Transformed data saved to: {output_file}")
        
    else:
        print("No data to transform")

def main():
    """Main function to run the transformation script."""
    parser = argparse.ArgumentParser(description='Transform data to hierarchical bar chart format')
    parser.add_argument('--data_dir', default='results', help='Directory containing CSV files with system-specific data')
    parser.add_argument('--output', default='results/hierarchical_data.csv', help='Output CSV file for transformed data')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        return
    
    # Transform data to hierarchical format
    transform_to_hierarchical_format(args.data_dir, args.output)

if __name__ == "__main__":
    main()
