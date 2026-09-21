import unittest
from tests import test_drr_dual_view_regressions as fixture


class DrrInteractionWidgetTests(unittest.TestCase):
    setUpClass = classmethod(fixture.DrrDualViewRegressionTests.setUpClass.__func__)
    tearDownClass = classmethod(fixture.DrrDualViewRegressionTests.tearDownClass.__func__)
    setUp = fixture.DrrDualViewRegressionTests.setUp
    tearDown = fixture.DrrDualViewRegressionTests.tearDown
    _set_silently = staticmethod(fixture.DrrDualViewRegressionTests._set_silently)

    def test_plot_has_one_live_spectrum_per_product_after_gate_changes(self):
        w = self.window
        for both in (False, True, False):
            w.drr_view_side_btn.setChecked(both)
            self.app.processEvents()
            self.assertTrue(all(helper.axes in w.figure.axes
                                for helper in w._drr_region_blitters.values()))
            for axis in w._drr_spectrum_axes.values():
                self.assertEqual(len(axis.lines), 1)
            w.drr_controller._set_drr_gate_spin_value(1.)
            w.drr_controller._update_drr_spectrum_and_gate_line(w._last_plot_cube)
            self.app.processEvents()
            for axis in w._drr_spectrum_axes.values():
                self.assertEqual(len(axis.lines), 1)
        self.errors.assert_not_called()
