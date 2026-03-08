from pathlib import Path
from utils.logging_util import logger
from modeling.perf.pu_latency_energy import create_energy_stats
from args import get_args
from config.model_config import make_model_config

def collect_resource_energy_breakdown(dfg, filename):
    """
    Collect and report energy breakdown by resource type across all nodes in the DFG.
    
    Args:
        dfg: The Data Flow Graph (NetworkX DiGraph)
        filename: Base filename for output files
    """
    # Get layer count for multiplier calculation
    args = get_args()
    model = make_model_config(args.model_name, args.dtype)
    layer_count = model["nlayers"]
    
    # Initialize aggregated energy stats
    total_energy_stats = create_energy_stats()
    for node, attributes in dfg.graph.nodes(data=True):
        # Calculate multiplier for layer optimization
        multiplier = 1
        if 'layer' in args.optimization:
            layer = attributes.get('layer', -1)
            if layer != -1:
                multiplier = layer_count
        
        if 'energy_breakdown' in attributes:
            energy_breakdown = attributes['energy_breakdown']
            
            # Apply multiplier to energy breakdown for this node
            for resource, energy in energy_breakdown.items():
                total_energy_stats[resource] += (energy * multiplier)
        
        # Collect communication energy from the same node
        if 'comm_energy' in attributes:
            comm_energy = attributes['comm_energy']  # in mJ
            # Convert mJ to pJ for consistency with other energy values
            comm_energy_pJ = comm_energy * 1e9
            total_energy_stats['comm'] += (comm_energy_pJ * multiplier)
    
    # Write detailed breakdown to file
    output_path = f"outputs/resource_energy_breakdown_{filename}.txt"
    Path("outputs").mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("=== Resource Energy Breakdown ===\n")
        f.write(f"Total Energy Consumption: {sum(total_energy_stats.values())/1000000000:.6f} mJ\n")
        f.write("(Includes both execution and communication energy)\n")
        if 'layer' in args.optimization:
            f.write(f"(Layer optimization applied - energy multiplied by {layer_count} layers)\n")
        f.write("\n")
        
        # Overall resource breakdown
        f.write("Overall Resource Breakdown:\n")
        sorted_resources = sorted(total_energy_stats.items(), key=lambda x: x[1], reverse=True)
        for resource, energy in sorted_resources:
            if energy > 0:
                percentage = (energy / sum(total_energy_stats.values())) * 100 if sum(total_energy_stats.values()) > 0 else 0
                f.write(f"  {resource:20}: {energy/1000000000:12.6f} mJ ({percentage:6.2f}%)\n")
        
    logger.info(f"[FILE] Resource energy breakdown saved to: {output_path}")

    return total_energy_stats
