import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.colors import LogNorm
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from core.loader import DataCube
from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow
from core.export import export_pl_pngs_and_dat
from PIL import Image


class PlDualViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        settings = QSettings(str(Path(self.tmp.name) / 'settings.ini'), QSettings.IniFormat)
        settings_patch = patch('ui_qt.main_window.QSettings', return_value=settings)
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        with patch.object(MainWindow, '_restore_last_folder', lambda s: None), patch.object(
                MainWindow, '_schedule_automatic_update_check', lambda s: None):
            self.w = MainWindow()
        self.addCleanup(self.w.close)
        self.cube = DataCube(np.linspace(1, 2, 6), np.array([-1., 0., 1.]),
            np.array([[-1, 0, 1, 10, 100, 1000]] * 3, dtype=float), 'Gate', 'PL sample', 'Intensity')
        self.w.loaded = LoadedState(mode='PL', folder=self.tmp.name, primary_file='sample.dat', cube=self.cube)
        self.w._ensure_loaded_matches_ui_params = lambda mode: True
        for key, value in dict(xmin=1, xmax=2, ymin=-1, ymax=1, vmin=-1, vmax=1000).items():
            self.w._set_spin_value_silent(self.w.pl_spins[key], value)

    def test_both_maps_share_viewport_and_keep_data(self):
        self.assertTrue(hasattr(self.w, 'pl_view_side_btn'), 'PL needs a side-by-side view control')
        original = self.cube.Z.copy()
        self.w.pl_view_side_btn.click()
        self.w._plot_mode('PL')
        axes = self.w._pl_heatmap_axes
        self.assertEqual(set(axes), {'linear', 'log'})
        self.assertFalse(isinstance(axes['linear'].collections[0].norm, LogNorm))
        self.assertIsInstance(axes['log'].collections[0].norm, LogNorm)
        axes['log'].set_xlim(1.3, 1.7)
        axes['log'].set_ylim(-.5, .5)
        np.testing.assert_allclose(axes['linear'].get_xlim(), [1.3, 1.7])
        np.testing.assert_allclose(axes['linear'].get_ylim(), [-.5, .5])
        np.testing.assert_equal(self.cube.Z, original)

    def test_switch_preserves_independent_manual_color_limits(self):
        self.assertTrue(hasattr(self.w, 'pl_view_log_btn'), 'PL needs independent scale views')
        self.w._plot_mode('PL')
        self.w.pl_spins['vmax'].setValue(600)
        self.w.pl_view_log_btn.click()
        self.w.pl_spins['vmin'].setValue(2)
        self.w.pl_spins['vmax'].setValue(80)
        self.w.pl_view_linear_btn.click()
        self.assertEqual(self.w.pl_spins['vmax'].value(), 600)
        self.w.pl_view_log_btn.click()
        self.assertEqual(self.w.pl_spins['vmin'].value(), 2)
        self.assertEqual(self.w.pl_spins['vmax'].value(), 80)

    def test_optional_pair_png_reuses_both_panels_and_one_dat(self):
        self.assertIn('include_pair', inspect.signature(export_pl_pngs_and_dat).parameters)
        from dataclasses import replace
        params = self.w._make_params('PL', self.cube)
        result = export_pl_pngs_and_dat(self.tmp.name, 'sample.dat',
            cube_linear=self.cube, cube_log=self.cube, params_linear=params,
            params_log=replace(params, log_scale=True, vmin=1, vmax=1000), include_pair=True)
        with Image.open(result['png_linear']) as linear, Image.open(result['png_log']) as log, Image.open(result['png_pair']) as pair:
            self.assertEqual(pair.size, (linear.width + log.width, max(linear.height, log.height)))
            np.testing.assert_equal(np.asarray(pair.crop((0, 0, linear.width, linear.height))), np.asarray(linear.convert('RGB')))
        self.assertEqual(len(list(Path(self.tmp.name).rglob('*.dat'))), 1)

    def test_export_keeps_independent_limits_and_zoom(self):
        self.assertTrue(hasattr(self.w, 'pl_view_side_btn'))
        from ui_qt.pl_views import export_parameters
        self.w._plot_mode('PL')
        self.w.pl_spins['vmax'].setValue(600)
        self.w.pl_view_log_btn.click()
        self.w.pl_spins['vmin'].setValue(2)
        self.w.pl_spins['vmax'].setValue(80)
        self.w._plot_mode('PL')
        self.w._pl_heatmap_ax.set_xlim(1.3, 1.7)
        result = export_parameters(self.w, self.cube)
        self.assertEqual(result['linear'].vmax, 600)
        self.assertEqual(result['log'].vmax, 80)
        self.assertEqual(result['log'].vmin, 2)
        np.testing.assert_allclose(result['linear'].xlim, [1.3, 1.7])
        self.assertEqual(result['linear'].xlim, result['log'].xlim)

    def test_legacy_log_checkbox_switches_from_negative_linear_split(self):
        from ui_qt.pl_views import select_scale, parameters
        w = self.w
        w.pl_split_scale_chk.setChecked(True)
        for key, value in dict(x0=1.5, left_vmin=-1, left_vmax=10, right_vmin=10, right_vmax=1000).items():
            w._set_spin_value_silent(w.pl_split_spins[key], value)
        # The legacy checkbox has already changed when its signal is delivered.
        blocked = w.pl_log_chk.blockSignals(True)
        w.pl_log_chk.setChecked(True)
        w.pl_log_chk.blockSignals(blocked)
        select_scale(w, True)
        result = parameters(w, self.cube)
        self.assertTrue(result['log'].log_scale)
        self.assertGreater(result['log'].split_scale.left_vmin, 0)
        self.assertEqual(result['linear'].split_scale.left_vmin, -1)
        w.pl_split_fix_checks['left_vmax'].setChecked(True)
        select_scale(w, False)
        w.pl_split_scale_chk.setChecked(False)
        select_scale(w, True)
        self.assertTrue(w.pl_split_scale_chk.isChecked())
        self.assertTrue(w.pl_split_fix_checks['left_vmax'].isChecked())

    def test_auto_axes_restore_full_extent_after_toolbar_zoom(self):
        self.w.pl_view_side_btn.click()
        self.w._plot_mode('PL')
        for dimension, zoom, full, action in (
            ('x', (1.3, 1.7), (1., 2.), self.w.pl_controller._auto_pl_xrange),
            ('y', (-.5, .5), (-1., 1.), self.w.pl_controller._auto_pl_yrange),
        ):
            with self.subTest(dimension=dimension):
                axis = self.w._pl_heatmap_axes['log']
                getattr(axis, 'set_' + dimension + 'lim')(zoom)
                action()
                self.w._plot_mode('PL')
                for axis in self.w._pl_heatmap_axes.values():
                    np.testing.assert_allclose(getattr(axis, 'get_' + dimension + 'lim')(), full)

    def test_gate_update_keeps_zoom_on_both_maps(self):
        self.w.pl_view_side_btn.click()
        self.w._plot_mode('PL')
        self.w._pl_heatmap_axes['log'].set_xlim(1.3, 1.7)
        self.w.pl_spins['gate'].setValue(1)
        self.w.pl_controller._update_pl_spectrum_and_gate_line(self.cube)
        for ax in self.w._pl_heatmap_axes.values():
            np.testing.assert_allclose(ax.get_xlim(), [1.3, 1.7])
            np.testing.assert_allclose(ax.lines[0].get_ydata(), [1, 1])

    def test_save_action_writes_three_pngs_and_one_dat_with_independent_scales(self):
        from core.processing_run import save_as_dat
        source = Path(save_as_dat(self.cube.gate, self.cube.energy, self.cube.Z,
            user_folder=self.tmp.name, subfolder='input', basename_override='sample', name_suffix=''))
        self.w.loaded.folder = str(source.parent)
        self.w.loaded.primary_file = source.name
        self.w._plot_mode('PL')
        self.w.pl_spins['vmax'].setValue(600)
        self.w.pl_view_log_btn.click()
        self.w.pl_spins['vmin'].setValue(2)
        self.w.pl_spins['vmax'].setValue(80)
        self.w.pl_export_pair_chk.setChecked(True)
        errors = []
        self.w._on_export_error = errors.append
        self.w._start_export('PL')
        self.assertTrue(self.w.thread_pool.waitForDone(10000))
        self.app.processEvents()
        self.assertFalse(errors)
        output = source.parent / 'Processed Data' / 'PL'
        self.assertEqual(len(list(output.glob('*.png'))), 3)
        self.assertEqual(len(list(output.glob('*.dat'))), 1)
        metadata = json.loads(next(output.glob('*.metadata.json')).read_text(encoding='utf-8'))
        self.assertEqual(metadata['plot']['linear']['vmax'], 600)
        self.assertEqual(metadata['plot']['log']['vmax'], 80)

    def test_zoom_past_split_exports_visible_region_color_limits(self):
        from ui_qt.pl_views import export_parameters
        from core.plotting import plot_pl
        from matplotlib.figure import Figure
        w = self.w
        w.pl_split_scale_chk.setChecked(True)
        for key, value in dict(x0=1.5, left_vmin=-1, left_vmax=10, right_vmin=10, right_vmax=1000).items():
            w._set_spin_value_silent(w.pl_split_spins[key], value)
        w._plot_mode('PL')
        w._pl_heatmap_ax.set_xlim(1.6, 2)
        params = export_parameters(w, self.cube)['linear']
        render = plot_pl(Figure().subplots(), self.cube, params)
        self.assertEqual(render.primary.norm.vmin, 10)
        self.assertEqual(render.primary.norm.vmax, 1000)

    def test_roi_excluding_inactive_split_still_draws_both_maps(self):
        from ui_qt.pl_views import parameters, select_scale
        from core.plotting import plot_pl
        from matplotlib.figure import Figure
        w = self.w
        w.pl_split_scale_chk.setChecked(True)
        w._set_spin_value_silent(w.pl_split_spins['x0'], 1.5)
        select_scale(w, True)
        w.pl_split_scale_chk.setChecked(False)
        w._set_spin_value_silent(w.pl_spins['xmin'], 1.6)
        products = parameters(w, self.cube)
        self.assertEqual(set(products), {'linear', 'log'})
        for params in products.values():
            plot_pl(Figure().subplots(), self.cube, params)
