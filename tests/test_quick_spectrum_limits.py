"""Spectrum limits follow visible line segments without losing manual Y views."""
from __future__ import annotations

import unittest

import numpy as np
from matplotlib.figure import Figure

from ui_qt.quick.spectrum_limits import fit_spectrum_y


class SpectrumLimitsTests(unittest.TestCase):
    def setUp(self):
        self.axis = Figure().subplots()

    def test_window_edges_hidden_curves_and_nan_gaps(self):
        axis = self.axis
        line, = axis.plot([0, 1, 2, 3, 4, 5], [1000, 10, 20, np.nan, 50, 1000])
        hidden, = axis.plot([1, 2], [-1000, 3000], visible=False)
        axis.set_xlim(1.25, 1.75)
        self.assertTrue(fit_spectrum_y(axis, (line, hidden)))
        low, high = axis.get_ylim()
        self.assertLess(low, 12.5)
        self.assertGreater(high, 17.5)
        self.assertGreater(low, 10)
        self.assertLess(high, 20)
        self.assertTrue(axis.get_autoscaley_on())
        axis.set_xlim(2.1, 3.9)
        self.assertFalse(fit_spectrum_y(axis, (line,)))
        self.assertEqual(axis.get_ylim(), (low, high))

    def test_current_multiple_curves_shrink_and_constant_values_stay_finite(self):
        axis = self.axis
        first, = axis.plot([0, 1, 2], [-100, 20, 80])
        second, = axis.plot([0, 1, 2], [-30, 40, 120])
        axis.set_xlim(0, 2)
        fit_spectrum_y(axis)
        wide = axis.get_ylim()
        first.set_ydata([5, 5, np.inf])
        second.set_ydata([7, 7, np.nan])
        self.assertTrue(fit_spectrum_y(axis))
        low, high = axis.get_ylim()
        self.assertGreater(low, wide[0])
        self.assertLess(high, wide[1])
        self.assertLess(low, 5)
        self.assertGreater(high, 7)
        second.set_visible(False)
        self.assertTrue(fit_spectrum_y(axis))
        low, high = axis.get_ylim()
        self.assertTrue(np.isfinite([low, high]).all())
        self.assertLess(low, 5)
        self.assertGreater(high, 5)
        self.assertLess(high, 7)

    def test_manual_y_is_preserved_and_empty_curves_do_not_reset_it(self):
        axis = self.axis
        line, = axis.plot([0, 1], [10, 20])
        axis.set_xlim(0, 1)
        axis.set_ylim(11, 12)
        self.assertFalse(fit_spectrum_y(axis))
        self.assertEqual(axis.get_ylim(), (11, 12))
        axis.set_autoscaley_on(True)
        self.assertTrue(fit_spectrum_y(axis))
        previous = axis.get_ylim()
        line.set_ydata([np.nan, np.inf])
        self.assertFalse(fit_spectrum_y(axis))
        self.assertEqual(axis.get_ylim(), previous)


if __name__ == "__main__":
    unittest.main()
