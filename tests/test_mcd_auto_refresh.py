from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from core.mcd import McdSettings, _pair_angles
from core.mcd_peak_shift import analyze_peak_shift
from tests.test_mcd_peak_shift import _result
from ui_qt.main_window import LoadedState, MainWindow


class AngleMatchingTests(unittest.TestCase):
    def test_selected_nominal_angle_matches_readback_jitter(self):
        paired = _pair_angles(
            np.array([0., 0., 1., 1.]), np.array([72.999, 10.001, 72.999, 10.001]),
            np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]),
            McdSettings(pos_angle=73., neg_angle=10.),
        )
        np.testing.assert_array_equal(paired[4], [0, 2])
        np.testing.assert_array_equal(paired[5], [1, 3])

    def test_distinct_angle_is_not_silently_reassigned(self):
        with self.assertRaisesRegex(ValueError, "angle"):
            _pair_angles(np.array([0., 0.]), np.array([72.9, 10.]),
                         np.ones((2, 2)), McdSettings(pos_angle=73., neg_angle=10.))


class MCDAutoRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.window = MainWindow()
        self.window.current_folder = self.folder.name
        self.controller = self.window.mcd_controller

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.folder.cleanup()

    def select(self, name, angles=(10., 73.)):
        path = Path(self.folder.name) / name
        path.write_text("B_T,angle_deg,700,710\n" + "".join(
            f"0,{a},100,90\n" for a in angles), encoding="utf-8")
        widget = self.window.mcd_files
        blocked = widget.blockSignals(True)
        widget.clear()
        widget.addItem(name)
        widget.item(0).setSelected(True)
        widget.blockSignals(blocked)
        return path

    def test_missing_replacement_file_cannot_reuse_previous_angles(self):
        self.select("old.csv")
        with patch.object(self.window.thread_pool, "start", side_effect=lambda w: w.run()):
            self.controller._mcd_detect_available_angles()
        self.assertTrue(self.controller._mcd_angles_ready())
        self.select("missing.csv").unlink()
        self.controller._mcd_detect_available_angles()
        self.assertFalse(self.controller._mcd_angles_ready())

    def test_single_angle_replacement_cannot_reuse_previous_angles(self):
        self.select("old.csv")
        with patch.object(self.window.thread_pool, "start", side_effect=lambda w: w.run()):
            self.controller._mcd_detect_available_angles()
            self.select("invalid.csv", (73.,))
            self.controller._mcd_detect_available_angles()
        self.assertFalse(self.controller._mcd_angles_ready())

    def test_repeated_detection_keeps_pending_load_and_one_scan(self):
        self.select("sweep.csv")
        submitted = []
        with patch.object(self.window.thread_pool, "start", side_effect=submitted.append), \
             patch.object(self.window, "_start_load") as load:
            self.controller._on_mcd_source_changed()
            self.controller._mcd_detect_available_angles()
            self.assertEqual(len(submitted), 1)
            submitted[0].run()
            load.assert_called_once_with("MCD")
            self.assertEqual(self.window.mcd_sigma_plus_combo.currentData(), 73.)

    def test_changed_file_during_detection_is_rescanned_before_load(self):
        path = self.select("sweep.csv")
        submitted = []
        with patch.object(self.window.thread_pool, "start", side_effect=submitted.append), \
             patch.object(self.window, "_start_load") as load:
            self.controller._on_mcd_source_changed()
            signature = (path.stat().st_size, path.stat().st_mtime_ns)
            path.write_text("B_T,angle_deg,700,710\n0,20.001,100,90\n0,80.001,100,90\n")
            submitted[0].signals.result.emit((str(path), signature, (10., 73.)))
            load.assert_not_called()
            self.assertFalse(self.controller._mcd_angles_ready())
            self.assertEqual(len(submitted), 2)
            submitted[1].run()
            self.assertEqual(self.window.mcd_sigma_plus_combo.currentData(), 80.001)
            load.assert_called_once_with("MCD")

    def prepare_peak(self, method="Raw spectrum"):
        w = self.window
        w.mcd_peak_tracker_method_combo.setCurrentText(method)
        source = _result([-1., 0., 1.])
        w.loaded = LoadedState(mode="MCD", folder="", mcd_result=source)
        w._update_mcd_peak_shift_source(source)
        return next(i for i in range(w.tabs.count()) if w.tabs.tabText(i) == "MCD Peak Shift")

    def test_tab_entry_analyzes_once_and_reuses_results(self):
        index = self.prepare_peak()
        w = self.window
        with patch("ui_qt.feature_pages.analyze_peak_shift", wraps=analyze_peak_shift) as analyze:
            w.tabs.setCurrentIndex(index)
            self.assertIsNotNone(w.mcd_peak_result)
            result = w.mcd_peak_result
            count = analyze.call_count
            w.tabs.setCurrentIndex(0)
            w.tabs.setCurrentIndex(index)
            self.assertIs(w.mcd_peak_result, result)
            self.assertEqual(analyze.call_count, count)
            w.tabs.setCurrentIndex(0)
            w.mcd_peak_prom_spin.setValue(w.mcd_peak_prom_spin.value() + .01)
            w.tabs.setCurrentIndex(index)
            self.assertGreater(analyze.call_count, count)

    def test_tab_entry_during_local_fit_does_not_restart_worker(self):
        index = self.prepare_peak("Local mixed fit")
        w = self.window
        submitted = []
        with patch.object(w.thread_pool, "start", side_effect=submitted.append):
            w.tabs.setCurrentIndex(index)
            self.assertEqual(len(submitted), 1)
            token = w._mcd_peak_fit_cancel_event
            w.tabs.setCurrentIndex(0)
            w.tabs.setCurrentIndex(index)
            self.assertEqual(len(submitted), 1)
            self.assertFalse(token.is_set())

    def test_load_finishing_on_peak_tab_starts_deferred_analysis(self):
        index = self.prepare_peak()
        w = self.window
        w._load_in_progress = True
        w._active_load_mode = "MCD"
        w._active_load_succeeded = True
        w.tabs.setCurrentIndex(index)
        self.assertIsNone(w.mcd_peak_result)
        w._on_load_finished()
        self.assertIsNotNone(w.mcd_peak_result)

    def test_identical_explicit_local_fit_request_is_not_restarted(self):
        self.prepare_peak("Local mixed fit")
        w = self.window
        submitted = []
        with patch.object(w.thread_pool, "start", side_effect=submitted.append):
            w._request_mcd_local_fit(seed_energy_ev=1.68, feature_kind="peak")
            token = w._mcd_peak_fit_cancel_event
            w._request_mcd_local_fit(seed_energy_ev=1.68, feature_kind="peak")
            self.assertEqual(len(submitted), 1)
            self.assertFalse(token.is_set())

    def test_leaving_local_method_cancels_obsolete_pending_fit(self):
        self.prepare_peak("Local mixed fit")
        w = self.window
        submitted = []
        with patch.object(w.thread_pool, "start", side_effect=submitted.append):
            w._request_mcd_local_fit(seed_energy_ev=1.68, feature_kind="peak")
            token = w._mcd_peak_fit_cancel_event
            w.mcd_peak_tracker_method_combo.setCurrentText("Raw spectrum")
            self.assertTrue(token.is_set())
            w.mcd_peak_tracker_method_combo.setCurrentText("Local mixed fit")
            self.assertEqual(len(submitted), 2)
