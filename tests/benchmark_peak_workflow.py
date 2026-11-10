"""Reproducible Peak Analysis timing; run with python -m tests.benchmark_peak_workflow."""
from time import perf_counter
import cProfile, pstats
import numpy as np
from core.loader import DataCube
from core.peak_workspace import create_dataset
from ui_qt.quick.peak_analysis_workflow import analyze_dataset

def make():
    rng = np.random.default_rng(871)
    x = np.linspace(1.6, 1.8, 2001); gate = np.linspace(-3, 3, 120)
    peak = 1 / (1 + (2 * (x - 1.7) / .008) ** 2)
    z = .1 + peak + .008*rng.normal(size=(len(gate), len(x)))
    d = create_dataset(DataCube(x, gate, z, 'Gate', 'benchmark', 'PL'), 'PL', 'PL', 'benchmark', 'benchmark')
    d['settings'].update(detection_method='local', candidate_floor=0., min_snr=0., prominence=.04, neighbor_support=0)
    return d

def timed(label, fn):
    start = perf_counter(); value = fn(); print(label, round(perf_counter()-start, 4), flush=True); return value

def main():
    d = make()
    d = timed('map detect', lambda: analyze_dataset(d, 'detect'))
    print('candidates', len(d['candidates']['points']), 'filtered', len(d['result']['points']), flush=True)
    d = timed('map filter', lambda: analyze_dataset(d, 'filter'))
    d['analysis_scope'] = 'spectrum'; d['row'] = 40; d['candidates'] = None; d['result'] = None; d['branches'] = []
    d = timed('spectrum detect', lambda: analyze_dataset(d, 'detect'))
    d = timed('spectrum filter', lambda: analyze_dataset(d, 'filter'))
    d = timed('spectrum find-fit', lambda: analyze_dataset(d, 'find-fit'))
    timed('spectrum repeat find-fit', lambda: analyze_dataset(d, 'find-fit'))
    profiler = cProfile.Profile(); profiler.enable(); analyze_dataset(d, 'find-fit'); profiler.disable()
    pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(15)

if __name__ == '__main__': main()
