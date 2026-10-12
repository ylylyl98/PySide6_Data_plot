"""Background color limits must respect current ROI, edits and saved views."""
import threading
import time
import unittest
import gc
import weakref
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from tests import test_drr_display_memory as memory_tests
from ui_qt.main_window import MainWindow


class DrrColorLimitTests(unittest.TestCase):
    setUpClass = classmethod(memory_tests.DrrDisplayMemoryTests.setUpClass.__func__)
    setUp = memory_tests.DrrDisplayMemoryTests.setUp
    dispose_windows = memory_tests.DrrDisplayMemoryTests.dispose_windows
    new_window = memory_tests.DrrDisplayMemoryTests.new_window
    load = memory_tests.DrrDisplayMemoryTests.load
    settle = memory_tests.DrrDisplayMemoryTests.settle

    def large(self):
        loaded = self.load()
        x = np.linspace(1., 2., 1024)
        y = np.linspace(-1., 1., 256)
        loaded.cube = replace(loaded.cube, energy=x, gate=y, Z=x[None, :] + y[:, None])
        return loaded

    def drain(self):
        deadline = time.monotonic() + 10
        with patch.object(self.window, '_plot_mode'):
            while time.monotonic() < deadline:
                self.window.thread_pool.waitForDone(100)
                self.app.processEvents()
                compute = getattr(self.window, '_drr_color_compute', None)
                if compute is not None and compute.worker is None and not compute.pending:
                    self.app.processEvents()
                    break
            else:
                self.fail('Color calculation did not settle')
        for timer in self.window._plot_redraw_timers.values():
            timer.stop()

    def test_large_auto_percentiles_run_off_gui_and_repeated_roi_is_cached(self):
        loaded = self.large()
        gui_thread = threading.get_ident()
        calls = []
        original = np.nanpercentile
        def percentile(*args, **kwargs):
            calls.append(threading.get_ident())
            if calls[-1] == gui_thread:
                raise AssertionError('Large percentiles ran on the GUI thread')
            return original(*args, **kwargs)
        with patch('numpy.nanpercentile', side_effect=percentile):
            self.window._apply_auto_limits_for_loaded()
            self.drain()
            first = len(calls)
            self.window._refresh_automatic_ranges('DRR')
            self.drain()
            self.assertEqual(len(calls), first)
        self.assertGreater(first, 0)
        expected = original(loaded.cube.Z.ravel(), [.01, 99.99])
        np.testing.assert_allclose([self.window.drr_spins[k].value() for k in ('vmin', 'vmax')], expected, atol=5e-5)

    def test_finished_request_cannot_overwrite_manual_edit_or_new_fix(self):
        self.large()
        started, release = threading.Event(), threading.Event()
        original = np.nanpercentile
        def slow(*args, **kwargs):
            started.set()
            if threading.get_ident() == self.gui_thread:
                raise AssertionError('Blocked percentile on GUI')
            release.wait(5)
            return original(*args, **kwargs)
        self.gui_thread = threading.get_ident()
        with patch('numpy.nanpercentile', side_effect=slow):
            self.window._apply_auto_limits_for_loaded()
            self.assertTrue(started.wait(2))
            self.window.drr_spins['vmin'].setValue(-12.)
            self.window.drr_fix_checks['vmax'].setChecked(True)
            fixed = self.window.drr_spins['vmax'].value()
            release.set()
            self.drain()
        self.assertEqual(self.window.drr_spins['vmin'].value(), -12.)
        self.assertEqual(self.window.drr_spins['vmax'].value(), fixed)

    def test_manual_edit_back_to_original_value_cancels_late_auto(self):
        self.large()
        started, release = threading.Event(), threading.Event()
        original = np.nanpercentile
        def slow(*args, **kwargs):
            started.set()
            release.wait(5)
            return original(*args, **kwargs)
        with patch('numpy.nanpercentile', side_effect=slow):
            try:
                self.window._apply_auto_limits_for_loaded()
                self.assertTrue(started.wait(2))
                value = self.window.drr_spins['vmin'].value()
                self.window.drr_spins['vmin'].setValue(value - 1.)
                self.window.drr_spins['vmin'].setValue(value)
                release.set()
                self.drain()
                self.assertEqual(self.window.drr_spins['vmin'].value(), value)
            finally:
                release.set()

    def test_changed_roi_uses_latest_request_and_changed_source_discards_old_result(self):
        loaded = self.large()
        self.window._apply_auto_limits_for_loaded()
        self.window._set_spin_value_silent(self.window.drr_spins['xmin'], 1.5)
        self.window._refresh_automatic_ranges('DRR')
        self.drain()
        selected = loaded.cube.Z[:, loaded.cube.energy >= 1.5]
        np.testing.assert_allclose([self.window.drr_spins[k].value() for k in ('vmin', 'vmax')],
                                   np.nanpercentile(selected, [.01, 99.99]), atol=5e-5)
        old = loaded.cube
        self.window._set_spin_value_silent(self.window.drr_spins['xmin'], 1.)
        self.window._refresh_automatic_ranges('DRR')
        loaded.cube = replace(old, Z=old.Z + 100)
        self.window._refresh_automatic_ranges('DRR')
        self.drain()
        np.testing.assert_allclose([self.window.drr_spins[k].value() for k in ('vmin', 'vmax')],
                                   np.nanpercentile(loaded.cube.Z, [.01, 99.99]), atol=5e-5)

    def test_fully_fixed_ranges_skip_statistics(self):
        self.large()
        for key in ('vmin', 'vmax'):
            self.window.drr_fix_checks[key].setChecked(True)
        with patch('numpy.nanpercentile', side_effect=AssertionError('Fixed colors requested statistics')):
            self.window._apply_auto_limits_for_loaded()
            self.window._refresh_automatic_ranges('DRR')
            self.window.thread_pool.waitForDone(2000)
            self.app.processEvents()

    def test_pending_export_freezes_roi_and_excludes_manually_edited_fields(self):
        loaded = self.large()
        original = np.nanpercentile
        started, release = threading.Event(), threading.Event()
        def slow(*args, **kwargs):
            started.set()
            release.wait(5)
            return original(*args, **kwargs)
        with patch('numpy.nanpercentile', side_effect=slow):
            try:
                self.window._apply_auto_limits_for_loaded()
                self.assertTrue(started.wait(2))
                self.window.drr_spins['vmin'].setValue(-12.)
                self.window.last_plotted_mode = 'DRR'
                captured = []
                def capture_export(loaded, options, **kwargs):
                    captured.append(options)
                    return {'mode': 'DRR', 'files_to_move': [], 'out_folder': str(self.root)}
                with patch.object(MainWindow, '_export_task', side_effect=capture_export), \
                     patch.object(self.window, '_ensure_loaded_matches_ui_params', return_value=True):
                    self.window._start_export('DRR')
                    self.window._set_spin_value_silent(self.window.drr_spins['xmin'], 1.5)
                    release.set()
                    self.drain()
                self.assertEqual(len(captured), 1)
                pending = captured[0].drr_pending_color_limits
                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0][2].xlim, (1., 2.))
                self.assertEqual(pending[0][3], ('vmax',))
                from core.color_limits import apply_pending_colors
                raw, _ = apply_pending_colors(loaded.cube, loaded.cube,
                    captured[0].drr_raw_params, captured[0].drr_second_params, pending)
                self.assertEqual(raw.vmin, -12.)
                self.assertAlmostEqual(raw.vmax, original(loaded.cube.Z, [.01, 99.99])[1])
            finally:
                release.set()

    def test_three_region_auto_has_no_gui_percentiles_and_preserves_fixed_side(self):
        loaded = self.large()
        w = self.window
        for key, value in (('xmin', 1.), ('xmax', 2.), ('ymin', -1.), ('ymax', 1.)):
            w._set_spin_value_silent(w.drr_spins[key], value)
        w.drr_split_fix_checks['left_vmin'].setChecked(True)
        w.drr_split_fix_checks['left_vmax'].setChecked(True)
        fixed = tuple(w.drr_split_spins[k].value() for k in ('left_vmin', 'left_vmax'))
        gui = threading.get_ident()
        original = np.nanpercentile
        def percentile(*args, **kwargs):
            if threading.get_ident() == gui:
                raise AssertionError('Region percentile on GUI')
            return original(*args, **kwargs)
        with patch('numpy.nanpercentile', side_effect=percentile):
            w.drr_region_count_combo.setCurrentIndex(2)
            self.drain()
        self.assertEqual(tuple(w.drr_split_spins[k].value() for k in ('left_vmin', 'left_vmax')), fixed)
        b1, b2 = (w.drr_split_spins[k].value() for k in ('x0', 'x1'))
        x = loaded.cube.energy
        a = np.argmin(np.abs((x[:-1] + x[1:]) / 2 - b1)) + 1
        b = np.argmin(np.abs((x[:-1] + x[1:]) / 2 - b2)) + 1
        expected = original(loaded.cube.Z[:, a:b], [.01, 99.99])
        np.testing.assert_allclose([w.drr_split_spins[k].value() for k in ('middle_vmin', 'middle_vmax')],
                                   expected, atol=5e-5)

    def test_rapid_roi_requests_are_coalesced_and_apply_only_latest_range(self):
        loaded = self.large()
        w = self.window
        started, release = threading.Event(), threading.Event()
        original = np.nanpercentile
        calls = []
        def slow(*args, **kwargs):
            calls.append(args[0].size)
            started.set()
            release.wait(5)
            return original(*args, **kwargs)
        with patch('numpy.nanpercentile', side_effect=slow):
            try:
                w._apply_auto_limits_for_loaded()
                self.assertTrue(started.wait(2))
                for x in np.linspace(1.1, 1.8, 30):
                    w._set_spin_value_silent(w.drr_spins['xmin'], x)
                    w._refresh_automatic_ranges('DRR')
                self.assertEqual(len(w._drr_color_compute.pending), 1)
                release.set()
                self.drain()
            finally:
                release.set()
        self.assertEqual(len(calls), 2)
        expected = original(loaded.cube.Z[:, loaded.cube.energy >= 1.8], [.01, 99.99])
        np.testing.assert_allclose([w.drr_spins[k].value() for k in ('vmin', 'vmax')], expected, atol=5e-5)

    def test_return_to_inflight_roi_discards_queued_obsolete_range(self):
        self.large()
        w = self.window
        started, release = threading.Event(), threading.Event()
        original = np.nanpercentile
        calls = []
        def slow(*args, **kwargs):
            calls.append(args[0].size)
            started.set()
            release.wait(5)
            return original(*args, **kwargs)
        with patch('numpy.nanpercentile', side_effect=slow):
            try:
                w._apply_auto_limits_for_loaded()
                self.assertTrue(started.wait(2))
                w._set_spin_value_silent(w.drr_spins['xmin'], 1.5)
                w._refresh_automatic_ranges('DRR')
                w._set_spin_value_silent(w.drr_spins['xmin'], 1.)
                w._refresh_automatic_ranges('DRR')
                release.set()
                self.drain()
            finally:
                release.set()
        self.assertEqual(len(calls), 1)

    def test_restored_colors_and_second_manual_scale_survive_pending_results(self):
        loaded = self.large()
        w = self.window
        w._drr_memory_cube = loaded.cube
        w._drr_memory_protected = {'axis.vmin', 'axis.vmax'}
        previous = tuple(w.drr_spins[k].value() for k in ('vmin', 'vmax'))
        with patch('numpy.nanpercentile', side_effect=AssertionError('Restored limits requested statistics')):
            w._apply_auto_limits_for_loaded()
            w._refresh_automatic_ranges('DRR')
        self.assertEqual(tuple(w.drr_spins[k].value() for k in ('vmin', 'vmax')), previous)
        w._drr_memory_protected.clear()
        second = replace(loaded.cube, Z=loaded.cube.Z * 2.)
        w._sync_drr_second_auto_scale(second)
        w.drr_second_vmin_spin.setValue(-50.)
        w.drr_second_vmax_spin.setValue(50.)
        self.drain()
        self.assertEqual((w.drr_second_vmin_spin.value(), w.drr_second_vmax_spin.value()), (-50., 50.))

    def test_clear_and_close_discard_background_controls_and_cache(self):
        self.large()
        w = self.window
        w._apply_auto_limits_for_loaded()
        before = tuple(w.drr_spins[k].value() for k in ('vmin', 'vmax'))
        w._clear_loaded_drr_view()
        w.thread_pool.waitForDone(2000)
        self.app.processEvents()
        self.assertEqual(tuple(w.drr_spins[k].value() for k in ('vmin', 'vmax')), before)
        self.assertFalse(w._drr_color_compute.cache)
        self.assertFalse(w._drr_color_compute.bindings)
        self.assertIsNone(w._drr_color_compute.source)

    def test_completed_color_job_does_not_retain_cleared_large_data(self):
        loaded = self.large()
        array = weakref.ref(loaded.cube.Z)
        w = self.window
        w._apply_auto_limits_for_loaded()
        self.drain()
        w._clear_loaded_drr_view()
        loaded.cube = None
        gc.collect()
        self.assertIsNone(array())

    def test_expired_product_identity_does_not_reuse_cached_colors(self):
        loaded = self.large()
        w = self.window
        # Simulate allocator identity reuse without depending on GC addresses.
        with patch('ui_qt.drr_color_limits._arrays', new=lambda cubes: ((1, 2, 3, 4),)):
            second = replace(loaded.cube, Z=loaded.cube.Z + 10.)
            ref = weakref.ref(second)
            w._sync_drr_second_auto_scale(second)
            self.drain()
            del second
            gc.collect()
            self.assertIsNone(ref())
            second = replace(loaded.cube, Z=loaded.cube.Z + 100.)
            w._sync_drr_second_auto_scale(second)
            self.drain()
        x, y = loaded.cube.energy, loaded.cube.gate
        x0, x1 = (w.drr_spins[k].value() for k in ('xmin', 'xmax'))
        y0, y1 = (w.drr_spins[k].value() for k in ('ymin', 'ymax'))
        roi = second.Z[np.ix_((y >= y0) & (y <= y1), (x >= x0) & (x <= x1))]
        np.testing.assert_allclose((w.drr_second_vmin_spin.value(), w.drr_second_vmax_spin.value()),
                                   np.nanpercentile(roi, [.01, 99.99]), atol=5e-5)

    def test_initial_log_plot_waits_for_valid_background_colors(self):
        self.large()
        w = self.window
        for widget, value in ((w.drr_log_chk, True), (w.drr_split_scale_chk, True)):
            blocked = widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(blocked)
        for key, value in (('left_vmin', -1.), ('left_vmax', 1.), ('right_vmin', -1.), ('right_vmax', 1.)):
            w._set_spin_value_silent(w.drr_split_spins[key], value)
        started, release = threading.Event(), threading.Event()
        original = np.nanpercentile
        def slow(*args, **kwargs):
            started.set()
            release.wait(5)
            return original(*args, **kwargs)
        with patch('numpy.nanpercentile', side_effect=slow), \
             patch.object(w, '_ensure_loaded_matches_ui_params', return_value=True):
            try:
                w._apply_auto_limits_for_loaded()
                self.assertTrue(started.wait(2))
                w._plot_mode('DRR')
                w._show_error.assert_not_called()
                self.assertIsNone(w.last_plotted_mode)
                release.set()
                self.drain()
                w._plot_mode('DRR')
                w._show_error.assert_not_called()
                self.assertEqual(w.last_plotted_mode, 'DRR')
            finally:
                release.set()

    def test_explicit_whole_auto_runs_in_background_and_skips_fixed_colors(self):
        self.large()
        w = self.window
        gui = threading.get_ident()
        original = np.nanpercentile
        def percentile(*args, **kwargs):
            if threading.get_ident() == gui:
                raise AssertionError('Whole Auto V ran on GUI')
            return original(*args, **kwargs)
        with patch('numpy.nanpercentile', side_effect=percentile):
            w.drr_controller._auto_drr_vrange()
            self.drain()
        for key in ('vmin', 'vmax'):
            w.drr_fix_checks[key].setChecked(True)
        with patch('numpy.nanpercentile', side_effect=AssertionError('Fixed Auto requested statistics')):
            w.drr_controller._auto_drr_vrange()

    def test_initial_plot_is_scheduled_when_computed_colors_equal_current_values(self):
        loaded = self.large()
        loaded.cube = replace(loaded.cube, Z=np.ones(loaded.cube.Z.shape))
        w = self.window
        w._set_spin_value_silent(w.drr_spins['vmin'], .99)
        w._set_spin_value_silent(w.drr_spins['vmax'], 1.01)
        w._apply_auto_limits_for_loaded()
        with patch.object(w, '_ensure_loaded_matches_ui_params', return_value=True):
            w._plot_mode('DRR')
        self.drain()
        self.assertIn('DRR', w._plot_redraw_pending)

    def test_initial_plot_resumes_when_background_returns_empty_or_fails(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                self.large()
                w = self.window
                started, release = threading.Event(), threading.Event()
                def finish(*args):
                    started.set()
                    release.wait(5)
                    if fail:
                        raise ValueError('Unable to calculate test limits')
                    return None
                with patch('ui_qt.drr_color_limits.calculate_bounds', side_effect=finish), \
                     patch.object(w, '_ensure_loaded_matches_ui_params', return_value=True):
                    try:
                        w._apply_auto_limits_for_loaded()
                        self.assertTrue(started.wait(2))
                        w._plot_mode('DRR')
                        self.assertIsNone(w.last_plotted_mode)
                        release.set()
                        self.drain()
                    finally:
                        release.set()
                self.assertIn('DRR', w._plot_redraw_pending)

    def test_explicit_auto_waits_for_uncached_derivative_off_gui(self):
        self.large()
        w = self.window
        blocked = w.drr_derivative_combo.blockSignals(True)
        w.drr_derivative_combo.setCurrentText('dE')
        w.drr_derivative_combo.blockSignals(blocked)
        gui = threading.get_ident()
        from core.processing import apply_sg_derivative_energy
        calls = []
        def transform(*args, **kwargs):
            calls.append(threading.get_ident())
            if calls[-1] == gui:
                raise AssertionError('Uncached Auto derivative ran on GUI')
            return apply_sg_derivative_energy(*args, **kwargs)
        with patch('ui_qt.controllers_drr.apply_sg_derivative_energy', side_effect=transform), \
             patch('ui_qt.drr_display_compute.apply_sg_derivative_energy', side_effect=transform), \
             patch.object(w, '_plot_mode'):
            w.drr_controller._auto_drr_vrange()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                w.thread_pool.waitForDone(100)
                self.app.processEvents()
                display = getattr(w, '_drr_display_compute', None)
                color = getattr(w, '_drr_color_compute', None)
                if display and display.worker is None and color and color.worker is None and not color.pending:
                    break
            else:
                self.fail('Derivative Auto did not settle')
        self.assertTrue(calls)
        self.assertAlmostEqual(w.drr_spins['vmin'].value(), 1., places=4)
        self.assertAlmostEqual(w.drr_spins['vmax'].value(), 1., places=4)


if __name__ == '__main__':
    unittest.main()
