"""Saved DRR views and wavelength templates must survive automatic refreshes."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
import numpy as np

from core.loader import DataCube
from tests.ui_test_helpers import dispose_owned_window
from ui_qt.common import LoadedState, LoadOptions
from ui_qt.main_window import MainWindow


class DrrDisplayMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.settings = QSettings(str(self.root / 'ui.ini'), QSettings.IniFormat)
        for item in (
            patch('ui_qt.main_window.QSettings', lambda *a, **k: self.settings),
            patch.object(MainWindow, '_restore_last_folder', autospec=True),
            patch.object(MainWindow, '_schedule_automatic_update_check', autospec=True),
            patch.object(MainWindow, '_refresh_drr_saved_history', autospec=True),
            patch.object(MainWindow, '_show_error', autospec=True),
            patch('ui_qt.main_window.QMessageBox.critical', side_effect=AssertionError('Unexpected save error dialog')),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.windows = []
        self.addCleanup(self.dispose_windows)
        self.window = self.new_window()

    def dispose_windows(self):
        for window in self.windows:
            dispose_owned_window(window)

    def new_window(self):
        window = MainWindow()
        window.tabs.setCurrentIndex(next(i for i in range(window.tabs.count())
                                        if window.tabs.tabText(i) == 'DRR'))
        self.windows.append(window)
        return window

    def load(self, name='a_720nmc.csv', window=None, root=None, bounds=(1.5, 1.8), settle=True):
        window = window or self.window
        root = root or self.root
        (root / name).touch()
        energy = np.linspace(*bounds, 31)
        cube = DataCube(energy, np.array([-1., 0., 1.]),
                        np.array([np.sin(energy * 15) * (1 + g / 4) for g in (-1, 0, 1)]),
                        'Gate', name, 'DR/R', gate_unit='V')
        loaded = LoadedState(mode='DRR', folder=str(root), selected_files=[name],
                             primary_file=name, cube=cube, drr_mode_label='DR/R Map')
        window.current_folder = str(root)
        window.drr_selected_files = [name]
        with patch.object(window, '_plot_mode'):
            window._on_loaded(loaded)
        if settle:
            self.settle(window)
        return loaded

    def settle(self, window=None):
        window = window or self.window
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            window.thread_pool.waitForDone(100)
            self.app.processEvents()
            compute = getattr(window, '_drr_display_compute', None)
            if compute is None or compute.worker is None:
                break
        for timer in window._plot_redraw_timers.values():
            timer.stop()
        window._plot_redraw_pending.clear()

    def write_legacy_view(self, name='a_720nmc.csv'):
        folder = self.root / 'Processed Data' / 'DRR'
        folder.mkdir(parents=True, exist_ok=True)
        payload = {'operation': 'DR/R', 'sources': [{'role': 'measurement', 'name': name}],
                   'plot': {'title': name, 'xlabel': 'Energy (eV)', 'ylabel': 'Gate',
                            'cbar_label': 'DR/R', 'xlim': [1.55, 1.74], 'ylim': [-.5, .5],
                            'vmin': -.03, 'vmax': .07, 'cmap': 'coolwarm',
                            'split_scale': {'split_x': 1.63, 'left_vmin': -.02,
                                            'left_vmax': .04, 'right_vmin': -.03,
                                            'right_vmax': .06, 'show_boundary': True}}}
        (folder / 'old.metadata.json').write_text(json.dumps(payload), encoding='utf-8')

    def test_disabled_memory_does_not_query_on_restore_and_retains_manual_fix(self):
        self.window.drr_memory_files_chk.setChecked(False)
        self.window.drr_memory_centers_chk.setChecked(False)
        self.window.drr_fix_checks['ymin'].setChecked(True)
        self.window.drr_spins['ymin'].setValue(-.4)
        with patch('core.drr_display_settings.find_candidates', side_effect=AssertionError('History queried')):
            self.load()
        self.assertEqual(self.window.drr_spins['ymin'].value(), -.4)
        self.assertTrue(self.window.drr_fix_checks['ymin'].isChecked())

    def test_load_worker_respects_captured_memory_preferences(self):
        from types import SimpleNamespace
        options = LoadOptions('DRR', str(self.root), ['a_720nmc.csv'], [], False,
                              'Self', 'last', False, drr_restore_files=False, drr_restore_centers=False)
        loaded = LoadedState(mode='DRR', folder=str(self.root), selected_files=options.selected_files)
        signal = SimpleNamespace(emit=lambda *args: None)
        with patch.object(MainWindow, '_load_task_data', autospec=True, return_value=loaded), \
                patch('core.drr_display_settings.find_candidates', side_effect=AssertionError('History queried')):
            result = self.window._load_task(options, progress=signal, log=signal)
        self.assertEqual(result.drr_display_candidates['requested'], (False, False))

    def test_changed_preferences_discard_incompatible_worker_candidates(self):
        from core.drr_display_settings import find_candidates
        loaded = self.load(settle=False)
        self.write_legacy_view()
        loaded.drr_display_candidates = find_candidates(str(self.root), ['a_720nmc.csv'],
                                                       restore_files=False, restore_centers=True)
        self.window.drr_memory_files_chk.setChecked(True)
        with patch.object(self.window, '_plot_mode'):
            self.window._on_loaded(loaded)
        self.assertEqual(self.window.drr_spins['ymin'].value(), -.5)

    def test_raw_split_load_defers_second_and_view_switch_computes_it(self):
        self.window.drr_region_count_combo.setCurrentIndex(2)
        loaded = self.load()
        self.assertFalse(any(key[1] == 2 for key in self.window._drr_derivative_cache))
        self.assertEqual(self.window.drr_second_split_spins['x0'].value(), self.window.drr_split_spins['x0'].value())
        self.window.last_plotted_mode = 'DRR'
        with patch.object(self.window, '_plot_mode'):
            self.window._on_drr_plot_view_changed('second')
        self.settle()
        self.assertTrue(any(key[0] == id(loaded.cube) and key[1] == 2
                            for key in self.window._drr_derivative_cache))
        self.assertNotEqual(self.window.drr_second_split_spins['middle_vmax'].value(), 1.)

    def test_side_by_side_and_explicit_second_auto_compute_hidden_product(self):
        self.window.drr_region_count_combo.setCurrentIndex(1)
        self.load()
        self.assertFalse(any(key[1] == 2 for key in self.window._drr_derivative_cache))
        self.window._auto_split_vrange('drr_second', 'left')
        self.settle()
        self.assertTrue(any(key[1] == 2 for key in self.window._drr_derivative_cache))
        self.load('b_720nmc.csv')
        self.assertFalse(any(key[1] == 2 for key in self.window._drr_derivative_cache))
        with patch.object(self.window, '_plot_mode'):
            self.window._on_drr_side_by_side_changed(True)
        self.settle()
        self.assertTrue(any(key[1] == 2 for key in self.window._drr_derivative_cache))

    def test_deferred_export_auto_scales_regions_and_preserves_frozen_manual_values(self):
        from core.processing import apply_sg_derivative_energy
        from core.drr_auto_regions import auto_region_params
        from ui_qt.drr_regions import deferred_auto_keys
        w = self.window
        w.drr_region_count_combo.setCurrentIndex(2)
        loaded = self.load()
        w.drr_second_split_fix_checks['left_vmin'].setChecked(True)
        w.drr_second_split_spins['left_vmin'].setValue(-999)
        w.drr_second_split_spins['middle_vmin'].setValue(-777)
        keys = deferred_auto_keys(w)
        self.assertNotIn('left_vmin', keys)
        self.assertNotIn('middle_vmin', keys)
        second, _ = apply_sg_derivative_energy(loaded.cube, derivative=2,
            window_length=w.drr_sg_window_spin.value(), polyorder=w.drr_sg_poly_spin.value())
        expected = auto_region_params(second, w._make_drr_params(second, 2), keys)
        w.last_plotted_mode = 'DRR'
        with patch.object(w, '_ensure_loaded_matches_ui_params', return_value=False):
            w._start_export('DRR')
        # Subsequent edits must not change this export's frozen snapshot.
        w.drr_second_split_spins['right_vmax'].setValue(333)
        deadline = time.monotonic() + 15
        while w._export_in_progress and time.monotonic() < deadline:
            w.thread_pool.waitForDone(100)
            self.app.processEvents()
        self.assertFalse(w._export_in_progress)
        records = [json.loads(path.read_text(encoding='utf-8')) for path in
                   (self.root / 'Processed Data' / 'DRR').glob('*.metadata.json')]
        record = next(item for item in records if item['processing']['derivative_order'] == 2)
        self.assertEqual(record['plot']['split_scale']['left_vmin'], -999)
        self.assertEqual(record['plot']['split_scale']['middle_vmin'], -777)
        for key in ('left_vmax', 'middle_vmax', 'right_vmin', 'right_vmax'):
            self.assertAlmostEqual(record['plot']['split_scale'][key], getattr(expected.split_scale, key))
            self.assertAlmostEqual(record['ui_state']['regions']['drr_second'][key], getattr(expected.split_scale, key))
        self.assertEqual(len(list((self.root / 'Processed Data' / 'DRR').glob('*.png'))), 2)

    def test_raw_toolbar_snapshot_does_not_restore_hidden_auto_placeholders(self):
        from ui_qt.drr_display_memory import capture_visible
        from core.drr_display_settings import save_snapshot
        w = self.window
        w.drr_region_count_combo.setCurrentIndex(1)
        self.load()
        with patch.object(w, '_ensure_loaded_matches_ui_params', return_value=True):
            w._plot_mode('DRR')
        _, state = capture_visible(w)
        save_snapshot(self.root, state)
        restarted = self.new_window()
        self.load(window=restarted)
        self.assertNotIn('drr_second.right_vmax', restarted._drr_memory_protected)
        self.assertNotIn('second_max', restarted._drr_memory_protected)
        with patch.object(restarted, '_plot_mode'):
            restarted._on_drr_plot_view_changed('second')
        self.settle(restarted)
        self.assertNotEqual(restarted.drr_second_split_spins['right_vmax'].value(), 1.)

    def test_malformed_deferred_auto_markers_fall_back_to_auto(self):
        from core.drr_display_settings import find_candidates, save_snapshot
        self.write_legacy_view()
        original = find_candidates(self.root, ['a_720nmc.csv'])['exact']
        for field, value in (('deferred_second_regions', None), ('deferred_second_regions', 'left_vmin'),
                             ('deferred_second_regions', ['invalid']), ('second_auto_pending', 'yes')):
            state = {**original, 'regions': {**original['regions'], 'drr_second': original['regions']['drr']}, field: value}
            save_snapshot(self.root, state)
            self.load()
            self.assertEqual(self.window.drr_spins['xmin'].value(), 1.5)

    def test_hidden_region_auto_remains_automatic_for_future_export_roi(self):
        from ui_qt.drr_regions import deferred_auto_keys
        self.window.drr_region_count_combo.setCurrentIndex(1)
        self.load()
        self.window._auto_split_vrange('drr_second', 'left')
        self.settle()
        self.assertIn('left_vmax', deferred_auto_keys(self.window))

    def test_new_unsplit_load_and_clear_release_old_deferred_cube(self):
        w = self.window
        w.drr_region_count_combo.setCurrentIndex(1)
        self.load()
        self.assertIsNotNone(w._drr_second_split_deferred)
        w.drr_region_count_combo.setCurrentIndex(0)
        self.load('b_720nmc.csv')
        self.assertIsNone(w._drr_second_split_deferred)
        w.drr_region_count_combo.setCurrentIndex(1)
        w._refresh_drr_second_split_ranges()
        self.assertIsNotNone(w._drr_second_split_deferred)
        w._clear_loaded_drr_view()
        self.assertIsNone(w._drr_second_split_deferred)

    def test_legacy_saved_view_survives_load_and_background_auto_ranges(self):
        self.write_legacy_view()
        self.load()
        window = self.window
        window._refresh_automatic_ranges('DRR', refresh_axes=True, center_split=True)
        self.settle()
        self.assertEqual(tuple(window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.55, 1.74))
        self.assertEqual(tuple(window.drr_spins[k].value() for k in ('ymin', 'ymax')), (-.5, .5))
        self.assertEqual(tuple(window.drr_spins[k].value() for k in ('vmin', 'vmax')), (-.03, .07))
        self.assertAlmostEqual(window.drr_split_spins['x0'].value(), 1.63)
        self.assertEqual(window.drr_region_count_combo.currentData(), 2)

    def test_same_center_reuses_geometry_but_auto_scales_new_y_and_colors(self):
        self.write_legacy_view()
        self.load('b_720nmc.csv')
        window = self.window
        self.assertEqual(tuple(window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.55, 1.74))
        self.assertAlmostEqual(window.drr_split_spins['x0'].value(), 1.63)
        self.assertEqual(tuple(window.drr_spins[k].value() for k in ('ymin', 'ymax')), (-1., 1.))
        self.assertNotAlmostEqual(window.drr_spins['vmax'].value(), .07)
        self.assertTrue(window.drr_fix_checks['xmin'].isChecked())
        self.assertTrue(window.drr_split_fix_checks['x0'].isChecked())

    def test_different_center_cannot_inherit_the_previous_template_locks(self):
        self.write_legacy_view()
        self.load('b_720nmc.csv')
        self.load('c_760nmc.csv', bounds=(1.4, 1.7))
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.4, 1.7))

    def test_incompatible_spectral_coverage_falls_back_to_auto(self):
        self.write_legacy_view()
        self.load('b_720nmc.csv', bounds=(1.1, 1.3))
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.1, 1.3))

    def save_view(self, window=None, count=3, x_range=(1.55, 1.74), after_start=None):
        window = window or self.window
        window.drr_region_count_combo.setCurrentIndex(count - 1)
        for key, value in (('xmin', x_range[0]), ('xmax', x_range[1]), ('ymin', -.5), ('ymax', .5),
                           ('vmin', -.03), ('vmax', .07)):
            window._set_spin_value_silent(window.drr_spins[key], value)
        for prefix, factor in (('drr', 1), ('drr_second', 10)):
            for key, value in (('x0', 1.61), ('x1', 1.68), ('left_vmin', -.02),
                               ('left_vmax', .04), ('middle_vmin', -.01),
                               ('middle_vmax', .03), ('right_vmin', -.03), ('right_vmax', .06)):
                if key == 'x1' and count < 3:
                    continue
                window._set_spin_value_silent(getattr(window, prefix + '_split_spins')[key],
                                             value if key.startswith('x') else value * factor)
        window.drr_second_auto_scale = False
        window._set_spin_value_silent(window.drr_second_vmin_spin, -.8)
        window._set_spin_value_silent(window.drr_second_vmax_spin, .9)
        window.last_plotted_mode = 'DRR'
        window._drr_limits_from_controls = True
        with patch.object(window, '_ensure_loaded_matches_ui_params', return_value=False):
            window._start_export('DRR')
        if after_start is not None:
            after_start()
        deadline = time.monotonic() + 15
        while window._export_in_progress and time.monotonic() < deadline:
            window.thread_pool.waitForDone(100)
            self.app.processEvents()
        self.assertFalse(window._export_in_progress, 'Export timed out')
        self.assertTrue(list((self.root / 'Processed Data' / 'DRR').glob('*.png')), window.log.toPlainText() if hasattr(window, 'log') else 'No PNG exported')
        return window

    def test_successful_save_restores_three_regions_after_restart(self):
        self.load()
        self.save_view()
        restarted = self.new_window()
        self.load(window=restarted)
        restarted._refresh_automatic_ranges('DRR', refresh_axes=True, center_split=True)
        self.settle(restarted)
        self.assertEqual(tuple(restarted.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.55, 1.74))
        self.assertEqual(restarted.drr_region_count_combo.currentData(), 3)
        self.assertEqual(tuple(restarted.drr_split_spins[k].value() for k in ('x0', 'x1')), (1.61, 1.68))
        self.assertAlmostEqual(restarted.drr_split_spins['middle_vmax'].value(), .03)
        self.assertAlmostEqual(restarted.drr_second_split_spins['middle_vmax'].value(), .3)
        self.assertAlmostEqual(restarted.drr_second_vmax_spin.value(), .9)

    def test_export_metadata_contains_the_view_snapshot(self):
        self.load()
        self.save_view(count=2)
        metadata = list((self.root / 'Processed Data' / 'DRR').glob('*.metadata.json'))
        self.assertTrue(metadata)
        snapshots = [json.loads(path.read_text(encoding='utf-8')).get('ui_state') for path in metadata]
        self.assertTrue(all(isinstance(state, dict) for state in snapshots))
        self.assertTrue(all(state['ranges']['xmin'] == 1.55 for state in snapshots))

    def test_file_snapshot_wins_over_a_newer_center_template(self):
        self.load()
        self.save_view(count=2)
        self.load('b_720nmc.csv')
        self.save_view(count=2, x_range=(1.56, 1.72))
        self.load()
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.55, 1.74))

    def test_two_region_center_template_does_not_lock_the_unused_second_boundary(self):
        self.load()
        self.save_view(count=2)
        self.load('b_720nmc.csv')
        self.assertFalse(self.window.drr_split_fix_checks['x1'].isChecked())
        self.window.drr_region_count_combo.setCurrentIndex(2)
        xmin, xmax = (self.window.drr_spins[key].value() for key in ('xmin', 'xmax'))
        a, b = (self.window.drr_split_spins[key].value() for key in ('x0', 'x1'))
        self.assertTrue(xmin < a < b < xmax)
        self.window._make_drr_params(self.window.loaded.cube, None)

    def test_color_reuse_is_optional_and_does_not_copy_y_range(self):
        self.write_legacy_view()
        self.window.drr_memory_colors_chk.setChecked(True)
        self.load('b_720nmc.csv')
        self.assertAlmostEqual(self.window.drr_spins['vmax'].value(), .07)
        self.assertAlmostEqual(self.window.drr_split_spins['left_vmax'].value(), .04)
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('ymin', 'ymax')), (-1., 1.))

    def test_clear_center_does_not_remove_file_snapshot_or_revive_legacy_template(self):
        self.write_legacy_view()
        self.load('b_720nmc.csv')
        self.window.drr_memory_clear_btn.click()
        self.load('b_720nmc.csv')
        self.assertEqual(self.window.drr_spins['xmin'].value(), 1.5)
        self.load()
        self.assertEqual(self.window.drr_spins['xmin'].value(), 1.55)

    def test_restore_auto_releases_all_remembered_ranges(self):
        self.write_legacy_view()
        self.load()
        self.window.drr_memory_auto_btn.click()
        self.settle()
        self.window._apply_auto_limits_for_loaded()
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.5, 1.8))
        self.assertNotAlmostEqual(self.window.drr_spins['vmax'].value(), .07)

    def test_corrupt_index_cannot_prevent_loading_a_valid_file(self):
        self.write_legacy_view()
        path = self.root / 'Processed Data' / 'DRR' / '.display_settings.json'
        path.write_text(json.dumps({'version': 1, 'files': [], 'centers': []}), encoding='utf-8')
        self.load()
        self.assertEqual(self.window.drr_spins['xmin'].value(), 1.55)

    def test_malformed_saved_fix_flags_fall_back_to_auto(self):
        from core.drr_display_settings import find_candidates, save_snapshot
        self.write_legacy_view()
        state = find_candidates(str(self.root), ['a_720nmc.csv'])['exact']
        state['fixes'] = []
        save_snapshot(str(self.root), state)
        self.load()
        self.assertEqual(self.window.drr_spins['xmin'].value(), 1.5)

    def test_malformed_legacy_containers_do_not_abort_candidate_scanning(self):
        self.write_legacy_view()
        path = self.root / 'Processed Data' / 'DRR' / 'old.metadata.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['processing'] = ['invalid container']
        path.write_text(json.dumps(payload), encoding='utf-8')
        self.load()
        self.assertEqual(self.window.drr_spins['xmin'].value(), 1.55)
        payload['plot']['split_scale'] = ['invalid container']
        path.write_text(json.dumps(payload), encoding='utf-8')
        self.load()
        self.assertEqual(self.window.drr_spins['xmin'].value(), 1.5)

    def test_exact_view_can_restore_saved_padding_outside_the_data_extent(self):
        self.write_legacy_view()
        path = self.root / 'Processed Data' / 'DRR' / 'old.metadata.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['plot']['xlim'] = [1.49, 1.81]
        path.write_text(json.dumps(payload), encoding='utf-8')
        self.load()
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.49, 1.81))

    def test_saving_with_a_measured_right_axis_restores_its_choice(self):
        self.load()
        window = self.window
        window.loaded.drr_axis_coordinates = {'TG': np.array([-2., 0., 2.])}
        window._update_drr_right_axis_options()
        window.drr_right_yaxis_combo.setCurrentText('TG')
        self.save_view(count=1)
        restarted = self.new_window()
        loaded = self.load(window=restarted)
        loaded.drr_axis_coordinates = {'TG': np.array([-2., 0., 2.])}
        with patch.object(restarted, '_plot_mode'):
            restarted._on_loaded(loaded)
        self.settle(restarted)
        self.assertEqual(restarted.drr_right_yaxis_combo.currentText(), 'TG')

    def test_saved_right_axis_off_clears_a_previous_measured_axis(self):
        from core.drr_display_settings import find_candidates, save_snapshot
        self.write_legacy_view()
        state = find_candidates(str(self.root), ['a_720nmc.csv'])['exact']
        state['right_axis'] = 'Off'
        save_snapshot(str(self.root), state)
        loaded = self.load()
        loaded.drr_axis_coordinates = {'TG': np.array([-2., 0., 2.])}
        self.window._update_drr_right_axis_options()
        self.window.drr_right_yaxis_combo.setCurrentText('TG')
        with patch.object(self.window, '_plot_mode'):
            self.window._on_loaded(loaded)
        self.assertEqual(self.window.drr_right_yaxis_combo.currentText(), 'Off')

    def test_restored_split_panels_and_whole_map_controls_match_the_layout(self):
        self.write_legacy_view()
        self.load()
        self.assertFalse(self.window.drr_split_scale_panel.isHidden())
        self.assertFalse(self.window.drr_second_split_scale_panel.isHidden())
        self.assertFalse(self.window.drr_spins['vmin'].isEnabled())
        self.load('c_760nmc.csv')
        self.assertTrue(self.window.drr_split_scale_panel.isHidden())
        self.assertTrue(self.window.drr_spins['vmin'].isEnabled())

    def test_global_templates_require_center_reuse_to_be_enabled(self):
        self.load()
        self.window.drr_memory_global_chk.setChecked(True)
        self.save_view(count=2)
        another_root = self.root / 'other_experiment'
        another_root.mkdir()
        self.window.drr_memory_centers_chk.setChecked(False)
        self.load('b_720nmc.csv', root=another_root)
        self.assertEqual(self.window.drr_spins['xmin'].value(), 1.5)
        self.window.drr_memory_centers_chk.setChecked(True)
        self.load('b_720nmc.csv', root=another_root)
        self.assertEqual(self.window.drr_spins['xmin'].value(), 1.55)

    def test_custom_colormap_id_is_restored_to_the_matching_menu_item(self):
        self.load()
        window = self.window
        window.drr_cmap.setCurrentIndex(window.drr_cmap.findData('dptk_rdbu_r_p0p45'))
        self.save_view(count=1)
        window.drr_cmap.setCurrentIndex(window.drr_cmap.findData('dptk_rdbu_r_p0p60'))
        self.load()
        self.assertEqual(window.drr_cmap.currentData(), 'dptk_rdbu_r_p0p45')

    def test_verified_working_copy_cleanup_keeps_the_full_file_snapshot(self):
        from core.provenance import verify_initial_data_working_file
        self.load()
        initial = self.root / 'Initial Data'
        initial.mkdir()
        (initial / 'a_720nmc.csv').touch()
        window = self.window
        window.loaded.provenance_records = (verify_initial_data_working_file(
            self.root / 'a_720nmc.csv', self.root, workflow='DRR', role='measurement'),)
        window.clean_verified_sources_chk.setChecked(True)
        self.save_view(count=2)
        self.assertFalse((self.root / 'a_720nmc.csv').exists())
        self.load('Initial Data/a_720nmc.csv')
        self.assertEqual(tuple(window.drr_spins[k].value() for k in ('ymin', 'ymax')), (-.5, .5))
        self.assertAlmostEqual(window.drr_spins['vmax'].value(), .07)

    def test_pending_restore_keeps_a_new_one_region_choice_on_both_products(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QToolButton
        self.write_legacy_view()
        self.load('b_760nmc.csv')
        window = self.window
        window.show()
        spin = window.drr_spins['xmin']
        for head in window.findChildren(QToolButton):
            if head.text() == 'Manual plot ranges' and head.parentWidget().parentWidget().isAncestorOf(spin):
                head.setChecked(True)
        spin.setFocus()
        self.app.processEvents()
        spin.selectAll()
        QTest.keyClicks(spin, '1.57')
        self.load(settle=False)
        QTest.keyClick(spin, Qt.Key_Return)
        window.drr_region_count_combo.setCurrentIndex(2)
        window.drr_region_count_combo.setCurrentIndex(0)
        window._run_scheduled_plot_redraw('DRR')
        self.assertEqual(window.drr_region_count_combo.currentData(), 1)
        self.assertFalse(window.drr_split_scale_chk.isChecked())
        self.assertFalse(window.drr_second_split_scale_chk.isChecked())

    def test_legacy_verified_alias_restores_the_canonical_files_full_view(self):
        from core.provenance import verify_initial_data_working_file
        self.write_legacy_view()
        initial = self.root / 'Initial Data'
        initial.mkdir()
        (initial / 'a_720nmc.csv').touch()
        working = self.root / 'a_720nmc.csv'
        working.touch()
        record = verify_initial_data_working_file(working, self.root, workflow='DRR', role='measurement')
        path = self.root / 'Processed Data' / 'DRR' / 'old.metadata.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['sources'][0]['source_path'] = str(working)
        payload['provenance'] = [record.to_dict()]
        path.write_text(json.dumps(payload), encoding='utf-8')
        working.unlink()
        self.load('Initial Data/a_720nmc.csv')
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('ymin', 'ymax')), (-.5, .5))
        self.assertAlmostEqual(self.window.drr_spins['vmax'].value(), .07)

    def test_pending_restore_preserves_multiple_committed_edits(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QToolButton
        self.write_legacy_view()
        self.load('b_760nmc.csv')
        window = self.window
        window.show()
        xmin, xmax = (window.drr_spins[key] for key in ('xmin', 'xmax'))
        for head in window.findChildren(QToolButton):
            if head.text() == 'Manual plot ranges' and head.parentWidget().parentWidget().isAncestorOf(xmin):
                head.setChecked(True)
        xmin.setFocus()
        self.app.processEvents()
        xmin.selectAll()
        QTest.keyClicks(xmin, '1.57')
        self.load(settle=False)
        QTest.keyClick(xmin, Qt.Key_Return)
        xmax.setFocus()
        xmax.selectAll()
        QTest.keyClicks(xmax, '1.73')
        QTest.keyClick(xmax, Qt.Key_Return)
        window._run_scheduled_plot_redraw('DRR')
        self.assertEqual((xmin.value(), xmax.value()), (1.57, 1.73))
        self.assertAlmostEqual(window.drr_spins['vmax'].value(), .07)

    def test_pending_load_without_a_template_preserves_recommitting_the_same_value(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QToolButton
        self.write_legacy_view()
        self.load()
        window = self.window
        window.show()
        spin = window.drr_spins['xmin']
        for head in window.findChildren(QToolButton):
            if head.text() == 'Manual plot ranges' and head.parentWidget().parentWidget().isAncestorOf(spin):
                head.setChecked(True)
        spin.setFocus()
        self.app.processEvents()
        spin.selectAll()
        QTest.keyClicks(spin, '1.55')
        self.load('b_760nmc.csv', settle=False)
        QTest.keyClick(spin, Qt.Key_Return)
        window._run_scheduled_plot_redraw('DRR')
        self.assertAlmostEqual(spin.value(), 1.55)

    def test_restore_auto_cancels_a_deferred_saved_view(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QToolButton
        self.write_legacy_view()
        self.load('b_760nmc.csv')
        window = self.window
        window.show()
        spin = window.drr_spins['xmin']
        for head in window.findChildren(QToolButton):
            if head.text() == 'Manual plot ranges' and head.parentWidget().parentWidget().isAncestorOf(spin):
                head.setChecked(True)
        spin.setFocus()
        self.app.processEvents()
        spin.selectAll()
        QTest.keyClicks(spin, '1.57')
        self.load(settle=False)
        QTest.keyClick(spin, Qt.Key_Return)
        window.drr_memory_auto_btn.click()
        window._run_scheduled_plot_redraw('DRR')
        self.assertEqual(tuple(window.drr_spins[key].value() for key in ('xmin', 'xmax')), (1.5, 1.8))
        self.assertIsNone(window._drr_memory_pending)

    def test_metadata_path_distinguishes_same_named_files_in_initial_data_and_root(self):
        self.write_legacy_view()
        initial = self.root / 'Initial Data'
        initial.mkdir()
        (initial / 'a_720nmc.csv').touch()
        path = self.root / 'Processed Data' / 'DRR' / 'old.metadata.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['sources'][0]['source_path'] = str(initial / 'a_720nmc.csv')
        path.write_text(json.dumps(payload), encoding='utf-8')
        self.load()
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('ymin', 'ymax')), (-1., 1.))

    def test_unchanged_png_save_updates_fix_flags_in_portable_metadata(self):
        self.load()
        self.save_view(count=1)
        check = self.window.drr_fix_checks['xmin']
        blocked = check.blockSignals(True)
        check.setChecked(True)
        check.blockSignals(blocked)
        self.save_view(count=1)
        metadata = list((self.root / 'Processed Data' / 'DRR').glob('*.metadata.json'))
        self.assertTrue(all(json.loads(path.read_text(encoding='utf-8'))['ui_state']['fixes']['xmin']
                            for path in metadata))

    def test_legacy_derivative_sidecar_restores_second_colors_separately(self):
        self.write_legacy_view()
        folder = self.root / 'Processed Data' / 'DRR'
        raw = json.loads((folder / 'old.metadata.json').read_text(encoding='utf-8'))
        raw['processing'] = {'derivative_order': 2}
        raw['plot']['vmin'], raw['plot']['vmax'] = -4000., 5000.
        raw['plot']['split_scale'] = None
        (folder / 'second.metadata.json').write_text(json.dumps(raw), encoding='utf-8')
        self.load()
        self.assertAlmostEqual(self.window.drr_spins['vmax'].value(), .07)
        self.assertAlmostEqual(self.window.drr_second_vmax_spin.value(), 5000.)

    def test_legacy_y_axis_mismatch_does_not_copy_y_and_colors(self):
        self.write_legacy_view()
        path = self.root / 'Processed Data' / 'DRR' / 'old.metadata.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['processing'] = {'y_axis': 'bias'}
        path.write_text(json.dumps(payload), encoding='utf-8')
        self.load()
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('ymin', 'ymax')), (-1., 1.))

    def test_invalid_optional_template_colors_fall_back_without_breaking_load(self):
        self.write_legacy_view()
        path = self.root / 'Processed Data' / 'DRR' / 'old.metadata.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['plot']['vmin'] = 'invalid color value'
        path.write_text(json.dumps(payload), encoding='utf-8')
        self.window.drr_memory_colors_chk.setChecked(True)
        self.load('b_720nmc.csv')
        self.assertTrue(np.isfinite(self.window.drr_spins['vmin'].value()))

    def test_center_geometry_template_preserves_manual_y_and_color_locks(self):
        self.write_legacy_view()
        for key, value in (('ymin', -.25), ('ymax', .25), ('vmin', -9), ('vmax', 9)):
            self.window._set_spin_value_silent(self.window.drr_spins[key], value)
            self.window.drr_fix_checks[key].setChecked(True)
        self.window.drr_second_auto_scale = False
        self.load('b_720nmc.csv')
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('ymin', 'ymax')), (-.25, .25))
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('vmin', 'vmax')), (-9., 9.))
        self.assertTrue(all(self.window.drr_fix_checks[k].isChecked() for k in ('ymin', 'ymax', 'vmin', 'vmax')))
        self.assertFalse(self.window.drr_second_auto_scale)

    def test_manual_fix_is_identified_and_takes_priority_over_saved_file_values(self):
        self.write_legacy_view()
        for key, value in (('ymin', -.25), ('ymax', .25)):
            self.window._set_spin_value_silent(self.window.drr_spins[key], value)
            self.window.drr_fix_checks[key].setChecked(True)
        self.load()
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('ymin', 'ymax')), (-.25, .25))
        self.assertEqual(self.window.drr_fix_checks['ymin'].property('fixOrigin'), 'manual')
        self.assertEqual(self.window.drr_fix_checks['ymin'].text(), 'Fix M')

    def test_manual_axis_fix_keeps_the_saved_zoom_on_the_other_axis(self):
        self.write_legacy_view()
        path = self.root / 'Processed Data' / 'DRR' / 'old.metadata.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        from core.drr_display_settings import find_candidates
        payload['ui_state'] = find_candidates(str(self.root), ['a_720nmc.csv'])['exact']
        payload['ui_state']['viewport'] = {'xlim': [1.56, 1.60], 'ylim': [-.4, .4]}
        path.write_text(json.dumps(payload), encoding='utf-8')
        for window, key, value, expected in (
            (self.window, 'ymin', -.25, ((1.56, 1.60), (-.25, .5))),
            (self.new_window(), 'xmin', 1.57, ((1.57, 1.74), (-.4, .4))),
        ):
            with self.subTest(manual=key):
                window._set_spin_value_silent(window.drr_spins[key], value)
                window.drr_fix_checks[key].setChecked(True)
                self.load(window=window)
                params = window._drr_params_with_view_limits(window._make_drr_params(window.loaded.cube, None))
                self.assertEqual((params.xlim, params.ylim), expected)

    def set_manual_geometry(self, xmin=1.60, xmax=1.70, boundary=1.65):
        window = self.window
        for key, value in (('xmin', xmin), ('xmax', xmax)):
            window.drr_fix_checks[key].setChecked(True)
            window.drr_spins[key].setValue(value)
        window.drr_region_count_combo.setCurrentIndex(1)
        window.drr_split_fix_checks['x0'].setChecked(True)
        window.drr_split_spins['x0'].setValue(boundary)
        self.settle()

    def test_new_center_replaces_manual_geometry_with_its_template(self):
        self.write_legacy_view()
        self.load('first_760nmc.csv')
        self.set_manual_geometry()
        self.window.drr_fix_checks['ymin'].setChecked(True)
        self.window.drr_spins['ymin'].setValue(-.25)
        self.load('new_720nmc.csv')
        window = self.window
        self.assertEqual(tuple(window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.55, 1.74))
        for prefix in ('drr', 'drr_second'):
            self.assertAlmostEqual(getattr(window, prefix + '_split_spins')['x0'].value(), 1.63)
        self.assertEqual(window.drr_fix_checks['xmin'].text(), 'Fix λ')
        self.assertEqual(window.drr_spins['ymin'].value(), -.25)
        self.assertIn('720', window.drr_memory_status.text())

    def test_new_center_without_template_releases_manual_geometry_to_auto(self):
        self.load()
        self.set_manual_geometry()
        self.load('new_760nmc.csv', bounds=(1.3, 1.5))
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.3, 1.5))
        self.assertFalse(self.window.drr_fix_checks['xmin'].isChecked())
        self.assertFalse(self.window.drr_split_fix_checks['x0'].isChecked())
        self.assertAlmostEqual(self.window.drr_split_spins['x0'].value(), 1.396667)

    def test_same_center_manual_geometry_takes_priority_over_template(self):
        self.write_legacy_view()
        self.load('first_720nmc.csv')
        self.set_manual_geometry()
        self.load('next_720nmc.csv')
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.6, 1.7))
        self.assertAlmostEqual(self.window.drr_split_spins['x0'].value(), 1.65)
        self.assertEqual(self.window.drr_fix_checks['xmin'].text(), 'Fix M')
        self.assertIn('manual', self.window.drr_memory_status.text().lower())

    def test_target_file_snapshot_beats_previous_files_manual_geometry(self):
        self.write_legacy_view()
        self.load('first_720nmc.csv')
        self.set_manual_geometry()
        self.load('a_720nmc.csv')
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.55, 1.74))
        self.assertAlmostEqual(self.window.drr_split_spins['x0'].value(), 1.63)
        # A displaced manual value must not reappear when the snapshot expires.
        self.load('new_760nmc.csv', bounds=(1.3, 1.5))
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.3, 1.5))
        self.assertFalse(self.window.drr_fix_checks['xmin'].isChecked())

    def test_reload_current_file_preserves_unsaved_manual_geometry(self):
        self.write_legacy_view()
        self.load()
        self.set_manual_geometry()
        self.load()
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.6, 1.7))
        self.assertAlmostEqual(self.window.drr_split_spins['x0'].value(), 1.65)

    def test_unknown_center_manual_geometry_is_limited_to_current_file(self):
        self.load('first.csv')
        self.set_manual_geometry()
        self.load('first.csv')
        self.assertAlmostEqual(self.window.drr_spins['xmin'].value(), 1.6)
        self.load('second.csv')
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.5, 1.8))

    def test_explicit_global_x_lock_overrides_file_snapshot_and_new_center(self):
        self.write_legacy_view()
        self.load('first_760nmc.csv')
        self.set_manual_geometry()
        check = getattr(self.window, 'drr_memory_lock_x_chk', None)
        self.assertIsNotNone(check, 'A separate opt-in cross-file X lock is required')
        self.assertFalse(check.isChecked())
        check.setChecked(True)
        self.load()
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.6, 1.7))
        self.assertAlmostEqual(self.window.drr_split_spins['x0'].value(), 1.65)
        self.assertIn('across files', self.window.drr_memory_status.text().lower())
        check.setChecked(False)
        self.load('other_760nmc.csv', bounds=(1.3, 1.5))
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.3, 1.5))

    def test_restart_does_not_promote_scoped_manual_boundary_to_global(self):
        self.load('first_720nmc.csv')
        self.set_manual_geometry()
        self.window.close()
        restarted = self.new_window()
        self.load('next_760nmc.csv', window=restarted)
        self.assertFalse(restarted.drr_split_fix_checks['x0'].isChecked())
        self.assertAlmostEqual(restarted.drr_split_spins['x0'].value(), 1.645)

    def test_global_lock_can_keep_a_restored_preset_without_numeric_edits(self):
        self.write_legacy_view()
        self.load('first_720nmc.csv')
        self.window.drr_memory_lock_x_chk.setChecked(True)
        self.load('next_760nmc.csv')
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.55, 1.74))
        self.assertAlmostEqual(self.window.drr_split_spins['x0'].value(), 1.63)
        self.assertEqual(self.window.drr_fix_checks['xmin'].text(), 'Fix M')
        self.assertIn('across files', self.window.drr_fix_checks['xmin'].toolTip())
        self.window.drr_memory_auto_btn.click()
        self.settle()
        self.assertFalse(self.window.drr_memory_lock_x_chk.isChecked())
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.5, 1.8))

    def test_global_lock_restores_its_values_and_scope_after_restart(self):
        self.write_legacy_view()
        self.load('first_720nmc.csv')
        self.window.drr_memory_lock_x_chk.setChecked(True)
        self.load('next_760nmc.csv')
        self.window.close()
        restarted = self.new_window()
        self.load('last_800nmc.csv', window=restarted)
        self.assertTrue(restarted.drr_memory_lock_x_chk.isChecked())
        self.assertEqual(tuple(restarted.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.55, 1.74))
        self.assertTrue(restarted.drr_fix_checks['xmin'].isChecked())
        self.assertTrue(restarted.drr_fix_checks['xmax'].isChecked())
        self.assertAlmostEqual(restarted.drr_split_spins['x0'].value(), 1.63)
        self.assertIn('across files', restarted.drr_memory_status.text())

    def test_global_x_lock_is_disabled_until_drr_data_is_loaded(self):
        self.assertFalse(self.window.drr_memory_lock_x_chk.isEnabled())
        self.window.drr_memory_lock_x_chk.setChecked(True)
        self.assertFalse(self.window.drr_memory_lock_x_chk.isChecked())
        self.load()
        self.assertTrue(self.window.drr_memory_lock_x_chk.isEnabled())
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.5, 1.8))

    def test_global_lock_preserves_inherited_layout_on_immediate_restart(self):
        self.write_legacy_view()
        self.load('first_720nmc.csv')
        self.window.drr_split_boundary_chk.setChecked(False)
        self.window.drr_memory_lock_x_chk.setChecked(True)
        self.window.close()
        restarted = self.new_window()
        self.load('next_760nmc.csv', window=restarted)
        self.assertEqual(tuple(restarted.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.55, 1.74))
        self.assertEqual(restarted.drr_region_count_combo.currentData(), 2)
        self.assertAlmostEqual(restarted.drr_split_spins['x0'].value(), 1.63)
        self.assertFalse(restarted.drr_split_boundary_chk.isChecked())

    def test_manual_geometry_layout_survives_loading_another_mode(self):
        self.write_legacy_view()
        for global_lock in (False, True):
            with self.subTest(global_lock=global_lock):
                window = self.new_window()
                loaded = self.load('first_720nmc.csv', window=window)
                if global_lock:
                    window.drr_memory_lock_x_chk.setChecked(True)
                else:
                    window.drr_spins['xmin'].setValue(1.56)
                other = LoadedState(mode='PL', folder=str(self.root),
                                    selected_files=['pl.csv'], primary_file='pl.csv', cube=loaded.cube)
                with patch.object(window, '_plot_mode'):
                    window._on_loaded(other)
                self.load('next_760nmc.csv' if global_lock else 'next_720nmc.csv', window=window)
                self.assertEqual(window.drr_region_count_combo.currentData(), 2)
                self.assertAlmostEqual(window.drr_spins['xmin'].value(), 1.55 if global_lock else 1.56)

    def test_missing_global_geometry_does_not_restore_a_nonfunctional_lock(self):
        self.settings.setValue('drr/display_memory/lock_x', True)
        restarted = self.new_window()
        self.assertFalse(restarted.drr_memory_lock_x_chk.isChecked())
        self.load(window=restarted)
        self.assertEqual(tuple(restarted.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.5, 1.8))

    def test_incompatible_same_center_manual_range_falls_back_to_auto(self):
        self.load('first_720nmc.csv')
        self.set_manual_geometry()
        self.load('next_720nmc.csv', bounds=(1.3, 1.5))
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.3, 1.5))
        self.assertFalse(self.window.drr_fix_checks['xmin'].isChecked())

    def test_partial_manual_range_cannot_invert_incoming_template(self):
        self.load('first_720nmc.csv')
        self.window.drr_fix_checks['xmin'].setChecked(True)
        self.window.drr_spins['xmin'].setValue(1.76)
        self.write_legacy_view()
        self.load('next_720nmc.csv')
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.55, 1.74))
        self.assertEqual(self.window.drr_fix_checks['xmin'].text(), 'Fix λ')

    def test_manual_three_region_geometry_survives_same_center_two_region_preset(self):
        self.write_legacy_view()
        self.load('first_720nmc.csv')
        self.set_manual_geometry()
        self.window.drr_region_count_combo.setCurrentIndex(2)
        for key, value in (('x0', 1.62), ('x1', 1.67)):
            self.window.drr_split_fix_checks[key].setChecked(True)
            self.window.drr_split_spins[key].setValue(value)
        self.load('next_720nmc.csv')
        self.assertEqual(self.window.drr_region_count_combo.currentData(), 3)
        self.assertEqual(tuple(self.window.drr_split_spins[k].value() for k in ('x0', 'x1')), (1.62, 1.67))

    def test_wave_fix_has_a_distinct_source_and_editing_it_makes_it_manual(self):
        self.write_legacy_view()
        self.load('b_720nmc.csv')
        window = self.window
        self.assertEqual(window.drr_fix_checks['xmin'].property('fixOrigin'), 'center')
        self.assertEqual(window.drr_fix_checks['xmin'].text(), 'Fix λ')
        self.assertIn('720', window.drr_fix_checks['xmin'].toolTip())
        window.drr_spins['xmin'].setValue(1.56)
        self.assertEqual(window.drr_fix_checks['xmin'].property('fixOrigin'), 'manual')
        self.load('c_720nmc.csv')
        self.assertAlmostEqual(window.drr_spins['xmin'].value(), 1.56)
        self.assertTrue(window.drr_fix_checks['xmin'].isChecked())
        self.assertEqual(window.drr_fix_checks['xmin'].text(), 'Fix M')

    def test_manual_region_fix_survives_a_template_without_color_reuse(self):
        self.write_legacy_view()
        window = self.window
        window.drr_region_count_combo.setCurrentIndex(1)
        for prefix in ('drr', 'drr_second'):
            spins = getattr(window, prefix + '_split_spins')
            fixes = getattr(window, prefix + '_split_fix_checks')
            for key, value in (('left_vmin', -3), ('left_vmax', 5)):
                window._set_spin_value_silent(spins[key], value)
                fixes[key].setChecked(True)
        self.load('b_720nmc.csv')
        for prefix in ('drr', 'drr_second'):
            spins = getattr(window, prefix + '_split_spins')
            fixes = getattr(window, prefix + '_split_fix_checks')
            self.assertEqual(tuple(spins[k].value() for k in ('left_vmin', 'left_vmax')), (-3., 5.))
            self.assertTrue(fixes['left_vmin'].isChecked())
            self.assertEqual(fixes['left_vmin'].property('fixOrigin'), 'manual')

    def test_fixed_boundary_is_recentered_only_when_outside_a_new_x_range(self):
        self.write_legacy_view()
        window = self.window
        window._set_spin_value_silent(window.drr_split_spins['x0'], 1.9)
        window.drr_split_fix_checks['x0'].setChecked(True)
        self.load('b_720nmc.csv')
        self.assertAlmostEqual(window.drr_split_spins['x0'].value(), 1.645)
        for refresh in (
            window._apply_auto_limits_for_loaded,
            lambda: window._refresh_automatic_ranges('DRR', center_split=True),
            lambda: window._refresh_drr_second_split_ranges(center_split=True),
        ):
            for outside in (1.55, 1.74, 1.9):
                window._set_spin_value_silent(window.drr_split_spins['x0'], outside)
                refresh()
                self.settle()
                self.assertAlmostEqual(window.drr_split_spins['x0'].value(), 1.645)
                self.assertAlmostEqual(window.drr_second_split_spins['x0'].value(), 1.645)
                self.assertTrue(window.drr_split_fix_checks['x0'].isChecked())
                self.assertEqual(window.drr_split_fix_checks['x0'].text(), 'Fix M')
            window._set_spin_value_silent(window.drr_split_spins['x0'], 1.66)
            refresh()
            self.settle()
            self.assertAlmostEqual(window.drr_split_spins['x0'].value(), 1.66)
        self.assertEqual(window.drr_split_fix_checks['x0'].text(), 'Fix M')

    def test_recentering_a_wave_boundary_keeps_its_source_and_scope(self):
        self.write_legacy_view()
        self.load('b_720nmc.csv')
        window = self.window
        window.drr_spins['xmin'].setValue(1.68)
        window._refresh_automatic_ranges('DRR', center_split=True)
        self.settle()
        boundary = window.drr_split_spins['x0'].value()
        self.assertTrue(1.68 < boundary < 1.74)
        self.assertEqual(window.drr_split_fix_checks['x0'].text(), 'Fix λ')
        self.load('c_760nmc.csv', bounds=(1.6, 1.8))
        self.assertFalse(window.drr_split_fix_checks['x0'].isChecked())
        self.assertEqual(window.drr_split_fix_checks['x0'].property('fixOrigin'), 'auto')

    def test_rechecking_an_inherited_fix_makes_it_manual_until_auto_is_restored(self):
        self.write_legacy_view()
        self.load('b_720nmc.csv')
        check = self.window.drr_fix_checks['xmin']
        check.setChecked(False)
        check.setChecked(True)
        self.assertEqual(check.text(), 'Fix M')
        self.load('c_720nmc.csv')
        self.assertAlmostEqual(self.window.drr_spins['xmin'].value(), 1.55)
        self.window.drr_memory_auto_btn.click()
        self.settle()
        self.assertFalse(check.isChecked())
        self.assertEqual(check.property('fixOrigin'), 'auto')
        self.assertAlmostEqual(self.window.drr_spins['xmin'].value(), 1.5)

    def test_saved_file_fix_is_scoped_and_has_its_own_visible_source(self):
        self.write_legacy_view()
        path = self.root / 'Processed Data' / 'DRR' / 'old.metadata.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        from core.drr_display_settings import find_candidates
        payload['ui_state'] = find_candidates(str(self.root), ['a_720nmc.csv'])['exact']
        payload['ui_state']['fixes'] = {'ymin': True, 'ymax': True}
        path.write_text(json.dumps(payload), encoding='utf-8')
        self.load()
        self.assertEqual(self.window.drr_fix_checks['ymin'].property('fixOrigin'), 'file')
        self.assertEqual(self.window.drr_fix_checks['ymin'].text(), 'Fix F')
        self.load('b_720nmc.csv')
        self.assertFalse(self.window.drr_fix_checks['ymin'].isChecked())
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('ymin', 'ymax')), (-1., 1.))

    def test_region_auto_does_not_promote_an_unchanged_file_fix_to_manual(self):
        self.write_legacy_view()
        path = self.root / 'Processed Data' / 'DRR' / 'old.metadata.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        from core.drr_display_settings import find_candidates
        payload['ui_state'] = find_candidates(str(self.root), ['a_720nmc.csv'])['exact']
        payload['ui_state']['regions']['drr_fix'] = {'left_vmin': True, 'left_vmax': False}
        path.write_text(json.dumps(payload), encoding='utf-8')
        self.load()
        window = self.window
        check = window.drr_split_fix_checks['left_vmin']
        self.assertEqual(check.text(), 'Fix F')
        self.assertTrue(window.drr_split_auto_left_btn.isEnabled())
        window.drr_split_auto_left_btn.click()
        self.settle()
        self.assertEqual(check.text(), 'Fix F')
        self.assertAlmostEqual(window.drr_split_spins['left_vmin'].value(), -.02)
        self.load('c_760nmc.csv')
        self.assertFalse(check.isChecked())

    def test_axis_auto_respects_inherited_fixed_values_and_their_scope(self):
        self.write_legacy_view()
        self.load('b_720nmc.csv')
        self.window.drr_auto_x_btn.click()
        self.settle()
        self.assertEqual(tuple(self.window.drr_spins[k].value() for k in ('xmin', 'xmax')), (1.55, 1.74))
        self.assertEqual(self.window.drr_fix_checks['xmin'].text(), 'Fix λ')
        self.load('c_760nmc.csv', bounds=(1.4, 1.7))
        self.assertFalse(self.window.drr_fix_checks['xmin'].isChecked())
        self.assertAlmostEqual(self.window.drr_spins['xmin'].value(), 1.4)
        path = self.root / 'Processed Data' / 'DRR' / 'old.metadata.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        from core.drr_display_settings import find_candidates
        state = find_candidates(str(self.root), ['a_720nmc.csv'])['exact']
        for axis in ('y', 'v'):
            with self.subTest(axis=axis):
                window = self.new_window()
                state['regions']['count'] = 1
                state['fixes'] = {axis + 'min': True, axis + 'max': False}
                payload['ui_state'] = state
                path.write_text(json.dumps(payload), encoding='utf-8')
                self.load(window=window)
                check = window.drr_fix_checks[axis + 'min']
                fixed_value = window.drr_spins[axis + 'min'].value()
                old_max = window.drr_spins[axis + 'max'].value()
                getattr(window, 'drr_auto_' + axis + '_btn').click()
                self.settle(window)
                self.assertAlmostEqual(window.drr_spins[axis + 'min'].value(), fixed_value)
                self.assertEqual(check.text(), 'Fix F')
                self.assertNotAlmostEqual(window.drr_spins[axis + 'max'].value(), old_max)
                self.load('c_760nmc.csv', window=window)
                self.assertFalse(check.isChecked())

    def test_restart_does_not_turn_a_center_boundary_fix_into_a_manual_fix(self):
        self.write_legacy_view()
        self.load('b_720nmc.csv')
        self.window.close()
        restarted = self.new_window()
        self.load('c_760nmc.csv', window=restarted)
        self.assertFalse(restarted.drr_split_fix_checks['x0'].isChecked())
        self.assertEqual(restarted.drr_split_fix_checks['x0'].property('fixOrigin'), 'auto')

    def test_incoming_load_preserves_uncommitted_number_and_then_applies_other_saved_fields(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QToolButton
        from ui_qt.common import focused_numeric_input
        self.write_legacy_view()
        self.load('b_760nmc.csv')
        window = self.window
        window.show()
        spin = window.drr_spins['xmin']
        for head in window.findChildren(QToolButton):
            if head.text() == 'Manual plot ranges' and head.parentWidget().parentWidget().isAncestorOf(spin):
                head.setChecked(True)
        spin.setFocus()
        self.app.processEvents()
        self.assertIs(focused_numeric_input(window), spin)
        spin.selectAll()
        QTest.keyClicks(spin, '1.57')
        self.load(settle=False)
        self.assertEqual(spin.lineEdit().text(), '1.57')
        QTest.keyClick(spin, Qt.Key_Return)
        QTest.qWait(250)
        window._run_scheduled_plot_redraw('DRR')
        self.settle()
        self.assertAlmostEqual(spin.value(), 1.57)
        self.assertAlmostEqual(window.drr_spins['xmax'].value(), 1.74)
        self.assertAlmostEqual(window.drr_spins['vmax'].value(), .07)

    def test_clear_center_invalidates_an_already_read_load_snapshot(self):
        from core.drr_display_settings import find_candidates
        self.write_legacy_view()
        loaded = self.load('b_720nmc.csv')
        loaded.drr_display_candidates = find_candidates(str(self.root), ['b_720nmc.csv'])
        self.window.drr_memory_clear_btn.click()
        with patch.object(self.window, '_plot_mode'):
            self.window._on_loaded(loaded)
        self.assertAlmostEqual(self.window.drr_spins['xmin'].value(), 1.5)

    def test_save_uses_the_queued_file_and_view_after_user_switches_files(self):
        from threading import Event
        self.load()
        gate = Event()
        original = MainWindow._export_task
        def delayed(owner, *args, **kwargs):
            self.assertTrue(gate.wait(5))
            return original(owner, *args, **kwargs)
        def switch_file():
            try:
                self.load('b_760nmc.csv', settle=False)
                self.window._set_spin_value_silent(self.window.drr_spins['xmin'], 1.6)
            finally:
                gate.set()
        with patch.object(MainWindow, '_export_task', delayed):
            self.save_view(count=2, after_start=switch_file)
        index = json.loads((self.root / 'Processed Data' / 'DRR' / '.display_settings.json').read_text(encoding='utf-8'))
        saved = list(index['files'].values())
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['files'], ['a_720nmc.csv'])
        self.assertEqual(saved[0]['ranges']['xmin'], 1.55)
        self.assertNotIn('760', index['centers'])

    def test_failed_export_does_not_create_a_display_snapshot(self):
        self.load()
        window = self.window
        window.last_plotted_mode = 'DRR'
        with patch.object(window, '_ensure_loaded_matches_ui_params', return_value=False), patch(
            'ui_qt.main_window.export_drr_pair_pngs_and_dat', side_effect=RuntimeError('export failed')):
            window._start_export('DRR')
            deadline = time.monotonic() + 5
            while window._export_in_progress and time.monotonic() < deadline:
                window.thread_pool.waitForDone(100)
                self.app.processEvents()
        self.assertFalse(window._export_in_progress)
        self.assertFalse((self.root / 'Processed Data' / 'DRR' / '.display_settings.json').exists())
        self.load('b_720nmc.csv')
        self.assertEqual(window.drr_spins['xmin'].value(), 1.5)

    def test_toolbar_png_remembers_the_displayed_view_before_later_edits(self):
        self.check_toolbar_view((1.56, 1.73))

    def test_toolbar_png_remembers_a_zoom_showing_only_one_split_region(self):
        self.check_toolbar_view((1.56, 1.60))

    def test_toolbar_uses_rendered_domain_when_sidebar_has_pending_changes(self):
        self.check_toolbar_view((1.56, 1.73), pending_xmin=1.65)

    def check_toolbar_view(self, bounds, pending_xmin=None):
        self.load()
        window = self.window
        self.write_legacy_view()
        self.load()
        with patch.object(window, '_ensure_loaded_matches_ui_params', return_value=True):
            window._plot_mode('DRR')
        axis = window._drr_heatmap_ax
        self.assertIsNotNone(axis)
        axis.set_xlim(*bounds)
        if pending_xmin is not None:
            window._set_spin_value_silent(window.drr_spins['xmin'], pending_xmin)
        target = self.root / 'toolbar.png'
        with patch('ui_qt.main_window.QFileDialog.getSaveFileName', return_value=(str(target), 'PNG')):
            window.toolbar.save_figure()
        job = window.toolbar._png_save_job
        window._set_spin_value_silent(window.drr_spins['xmin'], 1.6)
        deadline = time.monotonic() + 10
        while not job.done and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertTrue(job.done)
        self.assertIsNone(job.error)
        self.assertTrue(target.exists())
        restarted = self.new_window()
        self.load(window=restarted)
        self.assertEqual(tuple(restarted.drr_spins[key].value() for key in ('xmin', 'xmax')), (1.55, 1.74))
        self.assertEqual(restarted._drr_view_limits[0], bounds)
        self.assertAlmostEqual(restarted.drr_spins['vmax'].value(), .07)
        with patch.object(restarted, '_ensure_loaded_matches_ui_params', return_value=True):
            restarted._plot_mode('DRR')
        self.assertEqual(restarted._drr_heatmap_ax.get_xlim(), bounds)

    def test_partial_pair_failure_does_not_publish_new_display_history(self):
        import core.export as exporter
        self.load()
        original = exporter.export_drr_png_and_dat
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise OSError('second image write failed')
            return original(*args, **kwargs)
        with patch.object(exporter, 'export_drr_png_and_dat', side_effect=fail_second):
            self.save_view(count=2)
        self.assertFalse((self.root / 'Processed Data' / 'DRR' / '.display_settings.json').exists())
        self.load('b_720nmc.csv')
        self.assertEqual(self.window.drr_spins['xmin'].value(), 1.5)

    def test_toolbar_dialog_load_cannot_associate_old_axes_with_the_new_file(self):
        self.load()
        window = self.window
        with patch.object(window, '_ensure_loaded_matches_ui_params', return_value=True):
            window._plot_mode('DRR')
        target = self.root / 'toolbar.png'
        def choose(*args):
            self.load('b_760nmc.csv', settle=False)
            return str(target), 'PNG'
        with patch('ui_qt.main_window.QFileDialog.getSaveFileName', side_effect=choose):
            window.toolbar.save_figure()
        job = window.toolbar._png_save_job
        deadline = time.monotonic() + 10
        while not job.done and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertTrue(job.done)
        self.assertIsNone(job.error)
        self.assertFalse((self.root / 'Processed Data' / 'DRR' / '.display_settings.json').exists())


if __name__ == '__main__':
    unittest.main()
