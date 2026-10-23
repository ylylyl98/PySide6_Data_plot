"""A reference-only reload preserves the complete current DRR comparison view."""
from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from core.loader import DataCube
from tests.ui_test_helpers import dispose_owned_window
from ui_qt.common import LoadedState
from ui_qt.drr_display_memory import (
    _controls, _set, _value, prepare_background_switch, release,
    restore_background_switch, second_is_protected,
)
from ui_qt.main_window import MainWindow


class BnGroupDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.settings = QSettings(str(self.root / 'ui.ini'), QSettings.IniFormat)
        for item in (
            patch('ui_qt.main_window.QSettings', return_value=self.settings),
            patch.object(MainWindow, '_restore_last_folder', lambda self: None),
            patch.object(MainWindow, '_schedule_automatic_update_check', lambda self: None),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.w = MainWindow()
        self.addCleanup(dispose_owned_window, self.w)
        self.w._suspend_drr_autoplot = True
        self.w.current_folder = str(self.root)
        self.w.drr_selected_files = ['first.csv', 'second.csv']
        self.old = self.loaded(1.)
        self.w.loaded = self.old
        self.w._drr_memory_cube = self.old.cube
        self.w._drr_memory_borrowed = {'axis.xmin': (1.8, 1.81)}
        self.w._drr_fix_origins = {'fix.xmin': 'file'}
        controls = _controls(self.w)
        for name, value in (
            ('count', 2), ('split', True), ('second_split', True),
            ('axis.xmin', 1.81), ('axis.xmax', 1.99),
            ('axis.ymin', -.9), ('axis.ymax', .9), ('axis.gate', .6),
            ('axis.vmin', -.017), ('axis.vmax', .029),
            ('second_min', -17.), ('second_max', 29.),
            ('drr.x0', 1.87), ('drr.x1', 1.94),
            ('drr_second.x0', 1.87), ('drr_second.x1', 1.94),
        ):
            _set(controls[name], value)
        for prefix in ('drr', 'drr_second'):
            for index, side in enumerate(('left', 'middle', 'right'), 1):
                factor = index * (.01 if prefix == 'drr' else 10.)
                _set(controls[prefix + '.' + side + '_vmin'], -factor)
                _set(controls[prefix + '.' + side + '_vmax'], 2 * factor)
        for name, widget in controls.items():
            if name.startswith('fix.') or '_fix.' in name:
                _set(widget, False)
        _set(self.w.drr_sg_window_spin, 11)
        _set(self.w.drr_sg_poly_spin, 3)
        _set(self.w.drr_sg_y_window_spin, 7)
        _set(self.w.drr_sg_auto_chk, True)
        _set(self.w.drr_sg_y_auto_chk, True)
        self.w.drr_second_auto_scale = True

    def loaded(self, scale):
        energy = np.linspace(1.8, 2., 61)
        gate = np.linspace(-1., 1., 11)
        cube = DataCube(energy, gate,
            scale * (np.sin(energy * 40)[None, :] + gate[:, None]),
            'Gate', 'same group', 'DR/R', gate_unit='V')
        return LoadedState('DRR', str(self.root), primary_file='first.csv',
                           selected_files=['first.csv', 'second.csv'], cube=cube)

    def test_all_unfixed_bounds_and_latest_toolbar_view_survive_new_bn(self):
        w = self.w
        axis = w.figure.add_subplot(111)
        w._drr_heatmap_ax = axis
        axis.set_xlim(1.855, 1.966)
        axis.set_ylim(-.7, .8)
        # This edit occurs at publication, after any earlier request snapshot.
        _set(w.drr_spins['gate'], .6)
        _set(w.drr_spins['vmax'], .031)
        expected = {name: _value(widget) for name, widget in _controls(w).items()}
        settings_before = {key: self.settings.value(key) for key in self.settings.allKeys()}
        snapshot = prepare_background_switch(w)
        replacement = self.loaded(100.)
        w.loaded = replacement
        self.assertTrue(restore_background_switch(w, replacement, snapshot))
        # Subsequent normal automatic refreshes must preserve unchecked bounds,
        # including hidden d2E and all three independently scaled regions.
        w._refresh_automatic_ranges('DRR', refresh_axes=True)
        w._sync_drr_second_auto_scale(replacement.cube)
        self.assertEqual(expected, {name: _value(widget) for name, widget in _controls(w).items()})
        self.assertEqual(w._drr_view_limits, ((1.855, 1.966), (-.7, .8)))
        self.assertEqual(w.drr_spins['gate'].value(), .6)
        self.assertTrue(w.drr_second_auto_scale)
        self.assertTrue(second_is_protected(w))
        self.assertIs(w._drr_memory_cube, replacement.cube)
        self.assertEqual(w._drr_memory_borrowed, {'axis.xmin': (1.8, 1.81)})
        self.assertEqual(w._drr_fix_origins['fix.xmin'], 'file')
        self.assertEqual(settings_before, {key: self.settings.value(key) for key in self.settings.allKeys()})

    def test_sg_flags_are_retained_and_hold_is_bound_to_the_new_cube(self):
        w = self.w
        _set(w.drr_second_kind_combo, w.drr_second_kind_combo.findData('mixed'))
        snapshot = prepare_background_switch(w)
        replacement = self.loaded(5.)
        w.loaded = replacement
        self.assertTrue(restore_background_switch(w, replacement, snapshot))
        self.assertEqual(w.drr_sg_window_spin.value(), 11)
        self.assertEqual(w.drr_sg_poly_spin.value(), 3)
        self.assertEqual(w.drr_sg_y_window_spin.value(), 7)
        self.assertTrue(w.drr_sg_auto_chk.isChecked())
        self.assertTrue(w.drr_sg_y_auto_chk.isChecked())
        self.assertEqual(w.drr_second_kind_combo.currentData(), 'mixed')
        hold = w._drr_bn_group_processing_hold
        self.assertIs(hold[0], replacement.cube)
        self.assertEqual(hold[1:], (11, 3, 7, True, True))
        from ui_qt import drr_mixed_controls
        with patch.object(drr_mixed_controls, 'recommend_window',
                          side_effect=AssertionError('A BN comparison must keep its current windows')):
            self.assertEqual(drr_mixed_controls.update(w), 11)
        self.assertEqual(w.drr_sg_y_window_spin.value(), 7)
        self.assertIn('BN comparison: unchanged', w.drr_mixed_recommendation.text())
        # A real user edit uses the normal signal wiring, releases the hold,
        # and restores ordinary data-dependent recommendations.
        w.drr_sg_window_spin.setValue(13)
        self.assertIsNone(w._drr_bn_group_processing_hold)
        with patch.object(drr_mixed_controls, 'recommend_window',
                          wraps=drr_mixed_controls.recommend_window) as recommend:
            drr_mixed_controls.update(w)
        self.assertEqual(recommend.call_count, 2)
        self.assertEqual(w.drr_sg_window_spin.value(), 13)
        self.assertFalse(w.drr_sg_auto_chk.isChecked())
        self.assertIn('Recommended start:', w.drr_mixed_recommendation.text())

    def test_accepted_publication_skips_saved_and_automatic_ranges_before_plot(self):
        w = self.w
        axis = w.figure.add_subplot(111)
        w._drr_heatmap_ax = axis
        axis.set_xlim(1.855, 1.966)
        axis.set_ylim(-.7, .8)
        expected = {name: _value(widget) for name, widget in _controls(w).items()}
        w._drr_pending_ranges = (self.old.cube, {'refresh_axes': True})
        w._drr_pending_second_auto = self.old.cube
        w._drr_pending_auto_limits_cube = self.old.cube
        w._drr_pending_split_sides = {('drr', 'left'): self.old.cube}
        replacement = self.loaded(50.)
        with patch('ui_qt.drr_display_memory.prepare_restore',
                   side_effect=AssertionError('Saved display must not override a BN comparison')), \
             patch.object(w, '_apply_auto_limits_for_loaded',
                          side_effect=AssertionError('Unfixed comparison bounds must remain unchanged')), \
             patch.object(w, '_plot_mode') as plot:
            w._on_loaded(replacement, preserve_drr_display=True, drr_derivatives={})
        self.assertIs(w.loaded, replacement)
        self.assertEqual(expected, {name: _value(widget) for name, widget in _controls(w).items()})
        self.assertEqual(w._drr_view_limits, ((1.855, 1.966), (-.7, .8)))
        self.assertIsNone(w._drr_pending_ranges)
        self.assertIsNone(w._drr_pending_second_auto)
        self.assertIsNone(w._drr_pending_auto_limits_cube)
        self.assertEqual(w._drr_pending_split_sides, {})
        plot.assert_called_once_with('DRR', auto=False)

    def test_explicit_auto_releases_only_requested_bounds_without_fixing_them(self):
        w = self.w
        snapshot = prepare_background_switch(w)
        replacement = self.loaded(100.)
        w.loaded = replacement
        self.assertTrue(restore_background_switch(w, replacement, snapshot))
        for name in ('axis.vmin', 'axis.vmax'):
            release(w, name, manual_edit=False)
        self.assertNotIn('axis.vmin', w._drr_memory_protected)
        self.assertIn('drr.middle_vmax', w._drr_memory_protected)
        self.assertIn('second_max', w._drr_memory_protected)
        self.assertFalse(w.drr_fix_checks['vmin'].isChecked())
        for name in ('second_min', 'second_max'):
            release(w, name, manual_edit=False)
        self.assertFalse(second_is_protected(w))

    def test_different_group_or_grid_is_rejected_without_mutating_controls(self):
        w = self.w
        snapshot = prepare_background_switch(w)
        same = self.loaded(2.)
        different_grid = self.loaded(2.)
        different_grid.cube.energy = different_grid.cube.energy + .01
        candidates = (replace(same, folder=str(self.root / 'elsewhere')),
                      replace(same, selected_files=list(reversed(same.selected_files))),
                      replace(same, y_axis_spec='doping'), different_grid)
        for loaded in candidates:
            with self.subTest(folder=loaded.folder, selected=loaded.selected_files, axis=loaded.y_axis_spec):
                w.loaded = loaded
                _set(w.drr_spins['vmax'], .777)
                self.assertFalse(restore_background_switch(w, loaded, snapshot))
                self.assertEqual(w.drr_spins['vmax'].value(), .777)
        w.loaded = same
        w.current_folder = str(self.root / 'new-current-root')
        self.assertFalse(restore_background_switch(w, same, snapshot))
        w.current_folder = str(self.root)
        w.drr_selected_files = ['new-current-selection.csv']
        self.assertFalse(restore_background_switch(w, same, snapshot))
        self.assertIsNone(prepare_background_switch(type('Owner', (), {'loaded': None})()))


if __name__ == '__main__':
    unittest.main()
