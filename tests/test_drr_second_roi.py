import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from core.loader import DataCube
from core.processing import compute_auto_limits
from ui_qt.main_window import MainWindow


class Spin:
    def __init__(self, value): self.number = value
    def value(self): return self.number
    def setValue(self, value): self.number = value
    def blockSignals(self, blocked): return False


class SecondRoiTests(unittest.TestCase):
    def setUp(self):
        self.cube = DataCube(np.array([1., 2., 3.]), np.array([0., 1., 2.]),
            np.array([[1000., 1000., 1000.], [1000., 2., 4.], [1000., 6., 8.]]),
            'Gate', 'ROI', 'd2(DR/R)/dE2')
        self.owner = SimpleNamespace(drr_second_auto_scale=True,
            drr_spins={k: Spin(v) for k,v in dict(xmin=2, xmax=3, ymin=1, ymax=2).items()},
            drr_second_vmin_spin=Spin(-9), drr_second_vmax_spin=Spin(9))

    def test_auto_scale_uses_both_axes_and_keeps_manual_scale(self):
        MainWindow._sync_drr_second_auto_scale(self.owner, self.cube)
        expected = np.percentile([2, 4, 6, 8], [.01, 99.99])
        np.testing.assert_allclose([self.owner.drr_second_vmin_spin.value(), self.owner.drr_second_vmax_spin.value()], expected)
        self.owner.drr_second_auto_scale = False
        self.owner.drr_spins['xmin'].setValue(1)
        MainWindow._sync_drr_second_auto_scale(self.owner, self.cube)
        np.testing.assert_allclose([self.owner.drr_second_vmin_spin.value(), self.owner.drr_second_vmax_spin.value()], expected)

    def test_empty_roi_preserves_previous_scale(self):
        self.owner.drr_spins['xmin'].setValue(4)
        self.owner.drr_spins['xmax'].setValue(5)
        MainWindow._sync_drr_second_auto_scale(self.owner, self.cube)
        self.assertEqual(self.owner.drr_second_vmin_spin.value(), -9)
        self.assertEqual(self.owner.drr_second_vmax_spin.value(), 9)

    def test_auto_button_resumes_roi_scaling(self):
        owner = self.owner
        owner.loaded = SimpleNamespace(mode='DRR')
        owner.drr_second_auto_scale = False
        owner._prepare_drr_display_derivative = Mock(return_value=True)
        owner.drr_controller = SimpleNamespace(_drr_cube_with_metadata=Mock(return_value=(self.cube, 2, 5, 2)))
        owner._status = Mock()
        owner._schedule_plot_redraw = Mock()
        MainWindow._auto_drr_second_vrange(owner)
        self.assertTrue(owner.drr_second_auto_scale)
        self.assertLess(owner.drr_second_vmax_spin.value(), 9)
        owner._schedule_plot_redraw.assert_called_once_with('DRR')

    def test_reversed_limits_and_nonfinite_data(self):
        self.cube.Z[1, 1] = np.nan
        limits = compute_auto_limits(self.cube, xlim=(3, 2), ylim=(2, 1))
        np.testing.assert_allclose([limits.vmin, limits.vmax], np.percentile([4, 6, 8], [.01, 99.99]))
