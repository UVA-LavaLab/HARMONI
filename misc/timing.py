# timing.py
from time import perf_counter_ns
from contextlib import contextmanager
from collections import defaultdict
import csv
                                      
class StageTimer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.times_ns = defaultdict(int)
        self.counts = defaultdict(int)

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return

        t0 = perf_counter_ns()
        yield
        dt = perf_counter_ns() - t0
        self.times_ns[name] += dt
        self.counts[name] += 1

    def summary_ms(self):
        return {k: v / 1e6 for k, v in self.times_ns.items()}

    def total_ms(self):
        return sum(self.times_ns.values()) / 1e6

import csv

def dump_timing_csv(timer, filename, meta=None):
    """
    meta: dict of experiment metadata (model, system, batch, etc.)
    """
    meta = meta or {}
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stage",
            "time_ms",
            "calls",
            *meta.keys()
        ])
        for stage, t_ns in timer.times_ns.items():
            writer.writerow([
                stage,
                t_ns / 1e6,
                timer.counts[stage],
                *meta.values()
            ])
