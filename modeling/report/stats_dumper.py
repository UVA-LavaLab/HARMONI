"""
Statistics dumping utilities and StatsDumper class.

This module provides standalone dump functions that can be used anywhere in the codebase,
as well as a StatsDumper class for convenient orchestration of multiple dumps.
"""
import csv
from utils.logging_util import logger
from modeling.core.task_mapping import print_task_mappings, get_logic_unit_operation_dist, get_logic_units_with_task_types
from modeling.analysis.resource_energy_breakdown import collect_resource_energy_breakdown
from modeling.viz.dfg_html_viz import dump_partial_dfg_html


# ============================================================================
# Standalone dump functions - can be used anywhere in the codebase
# ============================================================================

def dump_task_mapping(task_mappings, filename_str):
    """
    Dump task to logic unit mappings.
    
    Args:
        task_mappings: Task to logic unit mappings dictionary
        filename_str: Base filename string for output files
    """
    if task_mappings is None:
        logger.warning("Task mappings not available for dumping (likely from cache)")
        return
    print_task_mappings(task_mappings, filename_str)


def dump_logic_unit_ops_dist(task_mappings, model_dfg, filename_str, args):
    """
    Dump logic unit operation distribution.
    
    Args:
        task_mappings: Task to logic unit mappings dictionary
        model_dfg: The model dataflow graph
        filename_str: Base filename string for output files
        args: Command line arguments object
    """
    get_logic_unit_operation_dist(task_mappings, model_dfg, filename_str, args)


def dump_logic_unit_analysis(model_dfg, filename_str):
    """
    Dump fine-grained logic unit analysis with task types and queue sizes.
    Works with cached DFG (no task_mappings needed).
    
    Args:
        model_dfg: The model dataflow graph
        filename_str: Base filename string for output files
    """
    unique_lus = get_logic_units_with_task_types(model_dfg)
    
    output_file = f'outputs/lu_analysis_{filename_str}.csv'
    ordered_keys = ['root', 'channel', 'wt_rank', 'kv_rank', 'chip']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['root', 'channel', 'wt_rank', 'kv_rank', 'chip', 'Logic Unit Name', 
                         'Queue Size', 'Task Type', 'Count', 'Percentage'])
        
        for lu_key in sorted(unique_lus.keys()):
            lu_info = unique_lus[lu_key]
            lu_id = lu_info['id']
            lu_name = lu_info['name']
            queue_size = lu_info['queue_size']
            task_types = lu_info['task_types']
            
            # Extract ID fields in order
            id_fields = [lu_id.get(k, '') for k in ordered_keys]
            
            total_tasks = sum(task_types.values())
            
            if total_tasks > 0:
                for task_type, count in sorted(task_types.items(), key=lambda x: str(x[0])):
                    pct = (count / total_tasks) * 100 if total_tasks else 0
                    row = id_fields + [lu_name, queue_size, str(task_type).replace("NodeType.", ""), 
                                     count, f"{pct:.1f}"]
                    writer.writerow(row)
            else:
                # Logic unit with no tasks
                row = id_fields + [lu_name, queue_size, '', 0, '0.0']
                writer.writerow(row)


def dump_resource_energy_breakdown(model_dfg, filename_str):
    """
    Dump resource energy breakdown.
    
    Args:
        model_dfg: The model dataflow graph
        filename_str: Base filename string for output files
    """
    logger.info("resource_energy_breakdown enabled")
    collect_resource_energy_breakdown(model_dfg, filename_str)


def dump_html_dfg(model_dfg, filename_str, suffix=""):
    """
    Dump interactive partial DFG visualization.
    
    Args:
        model_dfg: The model dataflow graph
        filename_str: Base filename string for output files
        suffix: Optional suffix to add to the output filename
    """
    output_path = f"outputs/{filename_str}_dfg_partial_token2{suffix}.html"
    dump_partial_dfg_html(model_dfg, output_html_path=output_path)


# ============================================================================
# StatsDumper class - convenience wrapper for orchestration
# ============================================================================

class StatsDumper:
    """
    Convenience wrapper for orchestrating multiple dump operations.
    
    Individual dump functions can be used standalone anywhere in the codebase.
    This class is useful when you want to batch multiple dumps together based on
    command-line arguments.
    """
    
    def __init__(self, args, filename_str, model_dfg, task_mappings=None):
        """
        Initialize the StatsDumper with all necessary data.
        
        Args:
            args: Command line arguments object
            filename_str: Base filename string for output files
            model_dfg: The model dataflow graph
            task_mappings: Task to logic unit mappings (optional, None for cached runs)
        """
        self.args = args
        self.filename_str = filename_str
        self.model_dfg = model_dfg
        self.task_mappings = task_mappings
        self.dump_stats = getattr(args, 'dump_stats', [])
    
    def dump_all(self):
        """
        Main entry point that checks which stats to dump and calls appropriate methods.
        """
        if 'task_mapping' in self.dump_stats:
            self.dump_task_mapping()
        
        if 'logic_unit_ops_dist' in self.dump_stats:
            self.dump_logic_unit_ops_dist()
        
        if 'logic_unit_analysis' in self.dump_stats:
            self.dump_logic_unit_analysis()
        
        if 'resource_energy_breakdown' in self.dump_stats:
            self.dump_resource_energy_breakdown()
        
        if 'html_dfg' in self.dump_stats:
            self.dump_html_dfg()
    
    def dump_task_mapping(self):
        """Dump task to logic unit mappings."""
        dump_task_mapping(self.task_mappings, self.filename_str)
    
    def dump_logic_unit_ops_dist(self):
        """Dump logic unit operation distribution."""
        dump_logic_unit_ops_dist(self.task_mappings, self.model_dfg, self.filename_str, self.args)
    
    def dump_logic_unit_analysis(self):
        """Dump fine-grained logic unit analysis with task types and queue sizes."""
        dump_logic_unit_analysis(self.model_dfg, self.filename_str)
    
    def dump_resource_energy_breakdown(self):
        """Dump resource energy breakdown."""
        dump_resource_energy_breakdown(self.model_dfg, self.filename_str)
    
    def dump_html_dfg(self, suffix=""):
        """Dump interactive partial DFG visualization."""
        dump_html_dfg(self.model_dfg, self.filename_str, suffix)
