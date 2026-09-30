import unittest
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from core.loader import DataCube
from core.processing import apply_sg_derivative_energy


class MixedDerivativeTests(unittest.TestCase):
    def test_mixed_polynomial_on_uneven_and_descending_coordinates(self):
        for y in [np.linspace(-1, 1, 15), np.linspace(-1, 1, 15)[::-1], np.linspace(0, 1, 15)**1.3]:
            x = 1 + np.linspace(0, 1, 31)**1.2
            z = 3*x[None, :] * y[:, None] + x[None, :]**2 + 2*y[:, None]**2
            cube = DataCube(x, y, z, 'Bias (V)', 'test', 'DR/R')
            result, window = apply_sg_derivative_energy(cube, derivative=('mixed', 7), window_length=9, polyorder=2)
            np.testing.assert_allclose(result.Z, 3., atol=1e-9)
            np.testing.assert_array_equal(result.gate, y)
            self.assertEqual(window, 9)
            self.assertIn('Bias', result.cbar_label)

    def test_constant_or_folded_y_is_rejected(self):
        for y in [np.zeros(9), np.array([0, 1, 2, 3, 4, 3, 2, 1, 0])]:
            cube = DataCube(np.arange(9.), y, np.ones((9, 9)), 'TG-BG', 'test', 'DR/R')
            with self.assertRaisesRegex(ValueError, 'monotonic'):
                apply_sg_derivative_energy(cube, derivative=('mixed', 5), window_length=5, polyorder=2)

    def test_recommendations_are_valid_and_noise_sensitive(self):
        from core.drr_mixed_derivative import recommend_window, window_span
        x = np.linspace(-1, 1, 301)
        clean = np.exp(-(x/.25)**2)
        noisy = clean + np.random.default_rng(42).normal(0, .05, x.size)
        a = recommend_window(x, clean[None, :], 2, maximum=21)
        b = recommend_window(x, noisy[None, :], 2, maximum=21)
        self.assertGreaterEqual(b, a)
        self.assertTrue(5 <= a <= 21 and a % 2 == 1)
        self.assertAlmostEqual(window_span(x, 7), .04)

    def test_export_worker_saves_mixed_dat_and_both_window_settings(self):
        from ui_qt.main_window import MainWindow
        from ui_qt.common import LoadedState, ExportOptions
        from core.plotting import HeatmapParams
        x, y = np.linspace(1, 2, 21), np.linspace(-1, 1, 11)
        cube = DataCube(x, y, 3*y[:, None]*x[None, :], 'Bias (V)', 'mixed test', 'DR/R')
        params = HeatmapParams(cube.title, 'Energy', cube.gate_label, 'DR/R', -5, 5, (1, 2), (-1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'source.csv').write_text('test source')
            loaded = LoadedState('DRR', tmp, primary_file='source.csv', selected_files=['source.csv'], cube=cube)
            options = ExportOptions(mode='DRR', params=params, drr_cube=cube, drr_raw_cube=cube, drr_raw_params=params,
                                    drr_second_sg_window=7, drr_second_sg_polyorder=2, drr_second_y_window=5)
            MainWindow._export_task(object(), loaded, options, progress=SimpleNamespace(emit=lambda *_: None), log=SimpleNamespace(emit=lambda *_: None))
            out = Path(tmp, 'Processed Data/DRR')
            dat = next(out.glob('*dXdY*.dat'))
            np.testing.assert_allclose(np.loadtxt(dat, skiprows=1)[:, 1:], 3., atol=1e-8)
            metadata = json.loads(dat.with_suffix('.metadata.json').read_text())
            self.assertEqual(metadata['processing']['savgol_window_y'], 5)
            self.assertEqual(metadata['processing']['savgol_window'], 7)
            self.assertEqual(metadata['processing']['derivative_axes'], 'XY')
            self.assertIn('Bias', metadata['plot']['cbar_label'])


from tests.test_drr_dual_view_regressions import DrrDualViewRegressionTests


class MixedUiTests(DrrDualViewRegressionTests):
    def test_mixed_mode_invalidates_peak_results_and_blocks_wrong_derivative_tracking(self):
        w = self.window
        tracker = w.drr_peak_analysis
        tracker.result = object()
        tracker.result_key = tracker.key()
        self.assertIsNotNone(tracker.valid_result())
        w.drr_second_kind_combo.setCurrentIndex(1)
        self.assertIsNone(tracker.valid_result())
        tracker.source.setCurrentIndex(tracker.source.findData('second'))
        tracker.analyze()
        self.assertIn('mixed heatmap and linecuts remain available', tracker.summary.text())
        self.assertFalse(tracker.workers)

    def test_constant_y_reports_unavailable_without_starting_derivative(self):
        w = self.window
        w.loaded.cube.gate[:] = 0
        w.drr_second_kind_combo.setCurrentIndex(1)
        self.assertFalse(w._prepare_drr_display_derivative(2))
        self.assertIn('finite samples', w.drr_mixed_recommendation.text())
        raw, *_ = w.drr_controller._drr_cube_with_metadata(None)
        self.assertIs(raw, w.loaded.cube)
        dx, *_ = w.drr_controller._drr_cube_with_metadata(1)
        self.assertEqual(dx.Z.shape, raw.Z.shape)
        from unittest.mock import patch
        w._drr_plot_view = 'raw'
        with patch.object(w, '_schedule_plot_redraw') as redraw:
            w.drr_controller._on_drr_derivative_changed()
            redraw.assert_called_with('DRR')
        w.last_plotted_mode = 'DRR'
        w._ensure_loaded_matches_ui_params = lambda *_: True
        w._start_export('DRR')
        self.errors.assert_called_once()
        self.assertIn('Cannot prepare the DRR export pair', self.errors.call_args.args[0])

    def test_mixed_view_and_manual_y_window_use_distinct_cache_entries(self):
        w = self.window
        x, y = np.linspace(-2, 2, 41), np.linspace(-1, 1, 21)
        w.loaded.cube = DataCube(x, y, 3*y[:, None]*x[None, :], 'Bias (V)', 'mixed', 'DR/R')
        w.drr_second_kind_combo.setCurrentIndex(1)
        w._on_drr_plot_view_changed('second')
        self._wait_derivative()
        self.errors.assert_not_called()
        np.testing.assert_allclose(w._drr_plot_cubes['second'].Z, 3., atol=1e-9)
        self.assertIn('meV', w.drr_mixed_recommendation.text())
        w.drr_sg_auto_chk.setChecked(True)
        self._wait_derivative()
        chosen_x = w.drr_sg_window_spin.value()
        w.drr_controller._drr_cube_with_metadata(None)
        self.assertEqual(w.drr_sg_window_spin.value(), chosen_x)
        original = w._drr_plot_cubes['second']
        w.drr_sg_y_window_spin.setValue(9)
        self._wait_derivative()
        self.assertFalse(w.drr_sg_y_auto_chk.isChecked())
        self.assertIsNot(original, w._drr_plot_cubes['second'])
        w.drr_second_kind_combo.setCurrentIndex(0)
        self._wait_derivative()
        np.testing.assert_allclose(w._drr_plot_cubes['second'].Z, 0., atol=1e-9)
