"""
GEMM trace generation utilities.
"""
import os
from collections import defaultdict
from utils.logging_util import logger
from modeling.perf.transformer_sim import get_gemm_trace


def generate_gemm_trace(dfg, filename):
    """
    Generate a TSV trace file for GEMM nodes in the given DFG.

    TSV header: Layer,M,N,K
    Output file: traces/gemm_{filename}.tsv
    
    Args:
        dfg: The dataflow graph
        filename: Base filename string for the output file
    """
    gemm_nodes = get_gemm_trace(dfg)
    if not gemm_nodes:
        logger.info("No GEMM nodes found in DFG for GEMM trace.")
        return

    # Create traces directory if it doesn't exist
    os.makedirs("traces", exist_ok=True)
    
    # Write to TSV file
    output_file = f"traces/gemm_{filename}.csv"
    with open(output_file, 'w', newline='') as f:
        # Write header (tab-separated)
        f.write("Layer,M,N,K,LU\n")
        # Write data rows (tab-separated)
        for name, m, n, k , lu in gemm_nodes:
            f.write(f"{name},{m},{n},{k},{lu.name}\n")
    
    logger.info(f"GEMM trace saved to {output_file}")
    
    # Create directory and separate CSV files for each unique LU
    generate_gemm_trace_by_lu(gemm_nodes, filename)


def generate_gemm_trace_by_lu(gemm_nodes, filename):
    """
    Create a directory and separate CSV files for each unique LU.
    
    Creates directory: gemm_{filename}_dir
    Creates CSV files: one per unique LU, containing only name, m, n, k columns.
    
    Args:
        gemm_nodes: List of tuples (name, m, n, k, lu) from get_gemm_trace()
        filename: Base filename string for the directory name
    """
    if not gemm_nodes:
        logger.info("No GEMM nodes provided for LU-based trace generation.")
        return
    
    # Group gemm_nodes by unique LU
    lu_groups = defaultdict(list)
    for name, m, n, k, lu in gemm_nodes:
        lu_name = lu.name if hasattr(lu, 'name') else str(lu)
        lu_groups[lu_name].append((name, m, n, k))
    
    # Create directory
    dir_name = f"traces/gemm_{filename}_dir"
    os.makedirs(dir_name, exist_ok=True)
    
    # Create CSV file for each unique LU
    for lu_name, nodes in lu_groups.items():
        # Sanitize LU name for filesystem (replace problematic characters)
        safe_lu_name = lu_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        csv_file = os.path.join(dir_name, f"{safe_lu_name}.csv")
        
        with open(csv_file, 'w', newline='') as f:
            # Write header
            f.write("Layer,M,N,K\n")
            # Write data rows
            for name, m, n, k in nodes:
                f.write(f"{name},{m},{n},{k}\n")
        
        logger.info(f"GEMM trace for LU '{lu_name}' saved to {csv_file}")
    
    logger.info(f"Created {len(lu_groups)} CSV files in directory {dir_name}")

