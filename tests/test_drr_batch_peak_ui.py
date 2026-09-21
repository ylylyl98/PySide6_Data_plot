import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from types import SimpleNamespace
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QDoubleSpinBox, QComboBox
from core.loader import DataCube


class BatchPeakUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui_qt.drr_peak_analysis import DrrPeakAnalysisController
        owner = QWidget()
        owner.loaded = SimpleNamespace(mode='DRR', cube=DataCube(
            np.linspace(1, 2, 31), np.array([3., 2., 1.]), np.ones((3, 31)),
            'Gate (V)', 'Test', 'DR/R'), selected_files=['test.csv'],
            baseline_files=[], drr_baseline_text='Self (last frame)', drr_baseline_which='last',
            y_axis_spec='auto')
        owner.drr_sg_window_spin = QDoubleSpinBox(); owner.drr_sg_window_spin.setValue(21)
        owner.drr_sg_poly_spin = QDoubleSpinBox(); owner.drr_sg_poly_spin.setValue(2)
        owner.drr_selected_files = ['test.csv']; owner.drr_baseline_files_manual = []
        owner.drr_baseline_combo = QComboBox(); owner.drr_baseline_combo.addItem('Self (last frame)')
        owner.drr_baseline_combine_combo = QComboBox(); owner.drr_baseline_combine_combo.addItem('last')
        owner._drr_view_limits = ((1.2, 1.8), (1.5, 2.5))
        controller = DrrPeakAnalysisController(owner)
        controller.build_controls()
        controller.loaded_changed()
        self.owner, self.controller = owner, controller

    def tearDown(self):
        self.owner.deleteLater()
        self.app.processEvents()

    def test_ranges_count_and_view_are_independent(self):
        owner, controller = self.owner, self.controller
        self.assertEqual(controller.selected_count(), 3)
        controller.use_view()
        self.assertEqual(controller.selected_count(), 1)
        self.assertEqual(controller.settings().x_min, 1.2)
        owner._drr_view_limits = ((1., 2.), (1., 3.))
        self.assertEqual(controller.selected_count(), 1)
        controller.full_range()
        self.assertEqual(controller.selected_count(), 3)

    def result(self):
        from core.drr_peak_analysis import analyze_drr_peaks
        return analyze_drr_peaks(self.owner.loaded.cube, self.controller.settings())

    def test_settings_stale_and_late_worker_does_not_replace_result(self):
        controller = self.controller
        key, generation = controller.key(), controller.generation
        result = self.result()
        controller.finished(result, key, generation)
        self.assertIsNotNone(controller.snapshot())
        self.owner.drr_sg_window_spin.setValue(25)
        self.assertIsNone(controller.snapshot())
        controller.clear()
        controller.finished(result, key, generation)
        self.assertIsNone(controller.result)

    def test_view_switch_keeps_result_but_source_switch_invalidates(self):
        c = self.controller
        c.finished(self.result(), c.key(), c.generation)
        self.owner._drr_plot_view = 'second'
        self.assertIsNotNone(c.snapshot())
        self.owner.drr_selected_files = ['other.csv']
        self.assertIsNone(c.snapshot())

    def test_clear_cancels_running_generation(self):
        from threading import Event
        c = self.controller
        c.cancel_event = Event()
        generation = c.generation
        c.clear()
        self.assertTrue(c.cancel_event.is_set())
        self.assertGreater(c.generation, generation)

    def test_full_range_preserves_high_precision_endpoints(self):
        c = self.controller
        self.owner.loaded.cube.gate = np.array([1.12345678912, 1.22345678912, 1.32345678912])
        c.full_range()
        self.assertEqual(c.selected_count(), 3)

    def test_pick_seed_selects_panel_and_invalidates_previous_results(self):
        c=self.controller
        c.finished(self.result(),c.key(),c.generation)
        axis=object();self.owner._drr_heatmap_axes={'second':axis}
        c.seed_button.setChecked(True)
        c.capture_seed(SimpleNamespace(button=1,xdata=1.5,ydata=2.,inaxes=axis))
        self.assertEqual(c.settings().mode,'seed')
        self.assertEqual(c.settings().source,'second')
        self.assertEqual(c.settings().polarity,'both')
        self.assertEqual(c.settings().seed_energy,1.5)
        self.assertIsNone(c.snapshot())
        self.assertFalse(c.seed_button.isChecked())
