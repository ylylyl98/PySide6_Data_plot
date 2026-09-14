import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from types import SimpleNamespace
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from ui_qt.main_window import MainWindow, LoadedState
from tests.test_mcd_unified_workflow import _result


class FitWindowSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_fit_matches_visible_trace_with_reversed_wavelength_columns_after_drag(self):
        w = MainWindow()
        self.addCleanup(w.close)
        r = _result()
        e = r.energy_ev
        b = np.tile(np.linspace(-2., 2., 17), 2)
        coefficients = np.full(e.size, .001)
        coefficients[20] = .1
        coefficients[60] = .005
        r.pair_b = b
        r.pair_labels = np.repeat(['B increasing', 'B decreasing'], 17)
        r.wavelength_nm = (1239.841984 / e)[::-1]
        r.pair_mcd_corrected = (b[:, None] * coefficients)[..., ::-1]
        r.pair_mcd_raw = r.pair_mcd_corrected
        for name in ('pair_raw_pos', 'pair_raw_neg', 'pair_corrected_pos', 'pair_corrected_neg'):
            setattr(r, name, np.ones_like(r.pair_mcd_corrected))
        w.loaded = LoadedState(mode='MCD', folder='', mcd_result=r)
        w.mcd_window_width_spin.setValue(1.)
        w.mcd_window_center_spin.setValue(1.62)
        controls = w.mcd_unified_controls
        controls.slope_low_spin.setValue(-.5)
        controls.slope_low_end_spin.setValue(.5)
        controls.slope_high_positive_spin.setValue(1.)
        controls.slope_high_positive_end_spin.setValue(2.)
        controls.slope_high_negative_spin.setValue(-2.)
        controls.slope_high_negative_end_spin.setValue(-1.)
        w._plot_mode('MCD')
        view = w.mcd_unified_view
        for center, expected in ((1.62, .1), (1.66, .005)):
            view._on_map_click(SimpleNamespace(inaxes=view.axes['mcd_map'], xdata=center,
                                               ydata=0, button=1, key=None))
            view._on_map_release(SimpleNamespace())
            # Exercise the production release -> spin -> deferred fit timer.
            QTest.qWait(120)
            for branch, line in view._artists['mcd_trace_lines']:
                np.testing.assert_allclose(line.get_ydata(), expected * np.asarray(line.get_xdata()), atol=1e-12)
            fits = w._mcd_unified_slopes.to_dict()['fits']
            valid = [fit for fit in fits if fit['status'] == 'ok']
            self.assertEqual(len(valid), 6)
            for fit in valid:
                self.assertAlmostEqual(fit['slope'], expected, places=10)
                line = view._artists['mcd_fit_lines'][(fit['region'], fit['branch'])]
                np.testing.assert_allclose(line.get_ydata(), expected * np.asarray(line.get_xdata()), atol=1e-12)
