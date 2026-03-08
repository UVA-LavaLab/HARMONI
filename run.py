"""
Main entry point for chiplet modeling simulation.
"""
import cProfile
import pstats
from args import parse_args
from utils.logging_util import logger
from simulation.simulator import start_simulation


if __name__ == '__main__':
    args = parse_args()
    logger.info("Parsed args: %s", vars(args))

    if args.simulate:
        if args.profile:
            logger.info("--- Profiling enabled. The simulation will run slower. ---")
            profiler = cProfile.Profile()
            profiler.enable()

            start_simulation(args)

            profiler.disable()

            print("\n--- Profiling Results (Top 30, sorted by cumulative time) ---")
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            stats.print_stats(30)
        else:
            start_simulation(args)
