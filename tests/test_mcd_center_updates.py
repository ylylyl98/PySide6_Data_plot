import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from ui_qt.main_window import MainWindow, LoadedState
from core.mcd_center_history import merge_center_candidates
from core.mcd import pair_window_trace_by_branch
from tests.test_mcd_unified_workflow import _result


class CenterUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def window(self):
        w = MainWindow()
        self.addCleanup(w.close)
        r = _result()
        r.wavelength_nm = 1239.841984 / r.energy_ev
        w.loaded = LoadedState(mode='MCD', folder='', mcd_result=r)
        w.mcd_window_center_spin.setValue(1.64)
        w._plot_mode('MCD')
        w.mcd_controller._mcd_center_refresh_timer.stop()
        self.app.processEvents()
        return w, w.mcd_unified_view

    def test_release_extracts_once_and_publishes_fits_immediately(self):
        w, v = self.window()
        w.loaded.mcd_result.summary = {'window_center_selection': {'method': 'suggested'}}
        with patch('core.mcd.pair_window_trace_by_branch', wraps=pair_window_trace_by_branch) as extract:
            v._on_map_click(SimpleNamespace(inaxes=v.axes['mcd_map'], xdata=1.65, ydata=0., button=1, key=None))
            v._on_map_release(SimpleNamespace())
            self.assertEqual(w.mcd_window_center_spin.value(), v.state.window_center_ev)
            self.assertFalse(w.mcd_controller._mcd_center_refresh_timer.isActive())
            QTest.qWait(250)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(w.loaded.mcd_result.summary['window_center_selection']['method'], 'manual')

    def test_continuous_spin_changes_are_throttled_not_starved(self):
        w, v = self.window()
        with patch.object(w, '_compute_unified_mcd_slopes', wraps=w._compute_unified_mcd_slopes) as fits:
            for center in np.linspace(1.63, 1.65, 12):
                w.mcd_window_center_spin.setValue(float(center))
                QTest.qWait(12)
            self.assertGreater(fits.call_count, 0)
            self.assertLess(fits.call_count, 12)
            QTest.qWait(80)
            self.assertAlmostEqual(v.state.window_center_ev, 1.65)
            self.assertEqual(w.mcd_controller._mcd_center_refresh_timer.interval(), 40)

    def test_candidate_switch_preserves_artists_and_has_no_full_draw(self):
        w, v = self.window()
        rows = [{'center_ev': e, 'width_mev': 5., 'last_used': '2026-09-14'} for e in np.linspace(1.61, 1.67, 30)]
        v.refresh_catalog(merge_center_candidates([], rows))
        QTest.qWait(100)
        before = tuple(v._artists['feature_selection'])
        with patch.object(v.canvas, 'draw', wraps=v.canvas.draw) as draw:
            for i in (1, 2, 0):
                v.candidate_combo.setCurrentIndex(i)
                QTest.qWait(100)
            self.assertEqual(draw.call_count, 0)
        self.assertEqual(tuple(v._artists['feature_selection']), before)

    def test_small_center_changes_do_not_rescale_trace_each_time(self):
        w, v = self.window()
        v.set_window(1.64, 5.)
        with patch.object(v.axes['mcd_vs_b'], 'draw', wraps=v.axes['mcd_vs_b'].draw) as redraw:
            for x in np.linspace(1.64001, 1.6402, 10):
                v.set_window(x, 5.)
            self.assertLessEqual(redraw.call_count, 2)
        QTest.qWait(250)
        y = np.concatenate([line.get_ydata() for _, line in v._artists['mcd_trace_lines']])
        lo, hi = v.axes['mcd_vs_b'].get_ylim()
        self.assertLessEqual(lo, np.nanmin(y))
        self.assertGreaterEqual(hi, np.nanmax(y))

    def test_full_draw_rebuild_does_not_draw_whole_canvas_twice(self):
        w, v = self.window()
        with patch.object(v.canvas, 'draw', wraps=v.canvas.draw) as draw:
            v.canvas.draw()
            QTest.qWait(100)
            self.assertEqual(draw.call_count, 1)
        self.assertTrue(v._blit_backgrounds)

    def test_all_metrics_nan_branches_and_nearest_sample_are_preserved(self):
        r = SimpleNamespace(wavelength_nm=1239.841984 / np.array([3., 2., 1.]),
                            pair_b=np.array([1., -1.]), pair_labels=np.array(['B increasing', 'B decreasing']),
                            pair_mcd_corrected=np.array([[30., np.nan, 10.], [-60., -40., -20.]]))
        r.pair_mcd_raw = 2 * r.pair_mcd_corrected
        traces = pair_window_trace_by_branch(r, 2., 2000.)
        expected = {'mean': (20., -40.), 'absolute_mean': (20., 40.),
                    'field_signed_absolute_mean': (20., -40.), 'integral': (np.nan, -80.)}
        for metric, values in expected.items():
            for i, branch in enumerate(('B increasing', 'B decreasing')):
                for source, factor in (('raw', 2), ('corrected', 1)):
                    np.testing.assert_allclose(traces[branch][f'{source}_{metric}'][1], [factor * values[i]], equal_nan=True)
        nearest = pair_window_trace_by_branch(r, 2.6, .001)
        np.testing.assert_allclose(nearest['B increasing']['corrected_mean'][1], [30.])
        np.testing.assert_allclose(nearest['B increasing']['corrected_integral'][1], [0.])
        r.wavelength_nm[:] = r.wavelength_nm[::-1]
        nearest = pair_window_trace_by_branch(r, 2.6, .001)
        np.testing.assert_allclose(nearest['B increasing']['corrected_mean'][1], [10.])
