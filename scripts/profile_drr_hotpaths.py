"""Reproducible DRR timings using temporary data and isolated UI settings.

Run with .venv/Scripts/python scripts/profile_drr_hotpaths.py --baseline HEAD
before committing an optimization. The optional Git revision supplies the
reference implementation without changing the checkout. Timings are medians
of separate operations, not an estimate of total application latency.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QT_API', 'pyside6')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Import Qt first for the project's Windows DLL ordering.
from PySide6.QtCore import QSignalBlocker
import numpy as np

from core import drr_sources, processing_run, export
from core.loader import DataCube


def reference(revision, relative, directory):
    data = subprocess.check_output(['git', 'show', f'{revision}:{relative}'], cwd=ROOT)
    path = directory / Path(relative).name
    path.write_bytes(data)
    name = relative.split('/')[0] + '._profile_reference_' + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(function, trials):
    times = []
    for _ in range(trials):
        started = time.perf_counter()
        result = function()
        times.append((time.perf_counter() - started) * 1000)
    return result, {'median_ms': statistics.median(times), 'trials_ms': times}


def scan_peak(function):
    tracemalloc.start()
    try:
        function()
        return tracemalloc.get_traced_memory()[1] / 1024 ** 2
    finally:
        tracemalloc.stop()


def plot_measurements(reuse_versions, trials):
    from scripts.preview_ui import build_preview_window
    from ui_qt import drr_plot_reuse
    from ui_qt.common import LoadedState
    from tests.ui_test_helpers import dispose_owned_window
    window, app, _ = build_preview_window(
        workflow='DRR', theme='light', scale='1', width=1200, height=800,
        sidebar_width=380, demo_data=False)
    errors = []
    window._show_error = errors.append
    window._ensure_loaded_matches_ui_params = lambda _mode: False
    cube = DataCube(np.linspace(1.7, 2.7, 1340), np.linspace(-1, 1, 157),
                    np.random.default_rng(17).normal(size=(157, 1340)),
                    'Gate (V)', 'Synthetic DRR', 'DR/R')
    window.loaded = LoadedState(mode='DRR', folder='', primary_file='synthetic.csv', cube=cube)
    for key, value in dict(xmin=1.7, xmax=2.7, ymin=-1, ymax=1, gate=0, vmin=-3, vmax=3).items():
        with QSignalBlocker(window.drr_spins[key]):
            window.drr_spins[key].setValue(value)
    with QSignalBlocker(window.drr_region_count_combo):
        window.drr_region_count_combo.setCurrentIndex(2)
    with QSignalBlocker(window.drr_split_scale_chk):
        window.drr_split_scale_chk.setChecked(True)
    for key, value in dict(x0=2.0, x1=2.4, left_vmin=-3, left_vmax=3,
                           middle_vmin=-3, middle_vmax=3, right_vmin=-3, right_vmax=3).items():
        with QSignalBlocker(window.drr_split_spins[key]):
            window.drr_split_spins[key].setValue(value)
    window._drr_plot_view = 'raw'
    window._drr_side_by_side = False
    results = {}
    try:
        for name, module in reuse_versions.items():
            context = patch.multiple(drr_plot_reuse, try_update=module.try_update,
                                     remember=module.remember) if name == 'before' else nullcontext()
            with context:
                window._drr_reuse_state = None
                window._last_plot_params_key = None
                window._drr_view_limits = None
                window.last_plotted_mode = None
                window._plot_mode('DRR')
                app.processEvents()
                iterations = 0
                def redraw():
                    nonlocal iterations
                    iterations += 1
                    with QSignalBlocker(window.drr_spins['xmin']):
                        window.drr_spins['xmin'].setValue(1.75 if iterations % 2 else 1.8)
                    window._drr_limits_from_controls = True
                    window._drr_view_limits = None
                    window._plot_mode('DRR')
                    app.processEvents()  # Includes the real Qt canvas idle draw.
                _, results[name] = measure(redraw, trials)
                if errors:
                    raise RuntimeError(errors)
    finally:
        window.thread_pool.waitForDone()
        dispose_owned_window(window)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', help='Git revision to compare against (e.g. HEAD)')
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-ui', action='store_true')
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    results = {'baseline': args.baseline, 'qt_platform': os.environ['QT_QPA_PLATFORM'],
               'numpy': np.__version__, 'python': sys.version.split()[0]}
    if args.baseline:
        results['baseline_commit'] = subprocess.check_output(
            ['git', 'rev-parse', args.baseline], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix='drr-hotpaths-') as temporary:
        root = Path(temporary)
        versions = {'after': (drr_sources, processing_run, export)}
        if args.baseline:
            versions = {'before': tuple(reference(args.baseline, f'core/{name}.py', root)
                                        for name in ('drr_sources', 'processing_run', 'export')), **versions}
        energy = 1240 / np.linspace(780, 700, 1340)
        values = np.random.default_rng(19).normal(size=(301, energy.size))
        csv_path = root / 'synthetic.csv'
        np.savetxt(csv_path, np.column_stack((np.arange(len(values)), values)), delimiter=',',
                   header='Vbg,' + ','.join(map(str, 1240 / energy)), comments='')
        history = root / 'history'
        history.mkdir()
        for index in range(766):
            (history / f'{index:04}.metadata.json').write_text(json.dumps(
                dict(operation='DR/R', analysis_fingerprint=str(index))))
        expected = {}
        for name, (sources, processing, exporting) in versions.items():
            profile, scan = measure(lambda: sources.inspect_csv_gate_profile(csv_path, max_rows=None), args.trials)
            scan['peak_python_mib'] = scan_peak(lambda: sources.inspect_csv_gate_profile(csv_path, max_rows=None))
            derivative, sg = measure(lambda: processing.sg_derivative_origin_rows(values, energy, window_pts=21), args.trials)
            exporting._drr_export_index(history)  # Warm the index, retain fresh validation each time.
            index, export_scan = measure(lambda: exporting._drr_export_index(history), args.trials)
            if expected:
                assert asdict(profile) == expected['profile']
                np.testing.assert_allclose(derivative, expected['derivative'], rtol=1e-10, atol=1e-7)
                assert index == expected['index']
            expected = dict(profile=asdict(profile), derivative=derivative, index=index)
            results[name] = dict(csv_profile=scan, nonuniform_derivative=sg, export_history=export_scan)
            print(name, results[name], flush=True)
        if not args.skip_ui:
            from ui_qt import drr_plot_reuse
            reuse_versions = {'after': drr_plot_reuse}
            if args.baseline:
                reuse_versions = {'before': reference(args.baseline, 'ui_qt/drr_plot_reuse.py', root), **reuse_versions}
            results['three_region_range_redraw'] = plot_measurements(reuse_versions, args.trials)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
