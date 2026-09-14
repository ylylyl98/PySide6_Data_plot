import threading
import unittest
import os
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class PeakShiftWorkerTests(unittest.TestCase):
    def test_peak_shift_worker_builds_both_methods_for_both_channels(self):
        from ui_qt.feature_pages import _mcd_peak_shift_worker

        source = SimpleNamespace(
            pair_b=[-1.0, 0.0, 1.0],
            pair_b_pos=[-1.0, 0.0, 1.0],
            pair_b_neg=[-1.0, 0.0, 1.0],
            pair_interpolated_pos=[False, False, False],
            pair_interpolated_neg=[False, False, False],
        )
        calls = []

        def fake_analyze(adapted, **kwargs):
            calls.append((kwargs["tracking_method"], len(adapted.pair_b)))
            return kwargs["tracking_method"]

        with patch("ui_qt.feature_pages.analyze_peak_shift", side_effect=fake_analyze):
            result = _mcd_peak_shift_worker(
                source,
                source_selector="Raw R",
                prominence=0.03,
                distance=5,
                smoothing=7,
                jump=0.04,
                peak_limit=6,
                derivative_window=35,
                cancel_event=threading.Event(),
            )

        self.assertEqual(set(result), {"Raw spectrum", "Second derivative"})
        self.assertEqual(set(result["Raw spectrum"]), {"pos", "neg"})
        self.assertEqual(set(result["Second derivative"]), {"pos", "neg"})
        self.assertEqual(len(calls), 4)

    def test_peak_shift_worker_can_cancel_before_next_analysis(self):
        from ui_qt.feature_pages import _mcd_peak_shift_worker

        source = SimpleNamespace(
            pair_b=[-1.0, 0.0, 1.0],
            pair_b_pos=[-1.0, 0.0, 1.0],
            pair_b_neg=[-1.0, 0.0, 1.0],
            pair_interpolated_pos=[False, False, False],
            pair_interpolated_neg=[False, False, False],
        )
        cancel = threading.Event()
        calls = []

        def fake_analyze(adapted, **kwargs):
            calls.append(kwargs["tracking_method"])
            cancel.set()
            return kwargs["tracking_method"]

        with patch("ui_qt.feature_pages.analyze_peak_shift", side_effect=fake_analyze):
            self.assertIsNone(
                _mcd_peak_shift_worker(
                    source,
                    source_selector="Raw R",
                    prominence=0.03,
                    distance=5,
                    smoothing=7,
                    jump=0.04,
                    peak_limit=6,
                    derivative_window=35,
                    cancel_event=cancel,
                )
            )
        self.assertEqual(len(calls), 1)


class PowerCatalogWorkerTests(unittest.TestCase):
    def test_folder_scan_snapshot_contains_power_rows_and_candidates(self):
        from ui_qt.main_window import _scan_folder_sources_worker

        with TemporaryDirectory() as folder:
            Path(folder, "sweep.csv").write_text(
                "Power_uW,1.5 eV,1.6 eV\n1,2,3\n2,4,5\n",
                encoding="utf-8",
            )
            result = _scan_folder_sources_worker(folder, progress=None, log=None)
            snapshot = next(item for item in result if isinstance(item, dict) and item.get("tag") == "power_catalog")
            self.assertEqual(snapshot["version"], 1)
            self.assertEqual(snapshot["candidates"], ("sweep.csv",))
            source = next(iter(snapshot["sources"].values()))
            self.assertEqual(source.power_values, (1.0, 2.0))


class PeakShiftDispatchTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtCore import QSettings
        import ui_qt.main_window as main_window
        import ui_qt.presentation_widget as presentation_widget

        self._settings_dir = TemporaryDirectory()
        settings_path = str(Path(self._settings_dir.name) / "test-settings.ini")
        settings_factory = lambda *args, **kwargs: QSettings(settings_path, QSettings.IniFormat)
        self._settings_patches = [
            patch.object(main_window, "QSettings", settings_factory),
            patch.object(presentation_widget, "QSettings", settings_factory),
            patch.object(main_window.MainWindow, "_restore_last_folder"),
            patch.object(main_window.MainWindow, "_schedule_automatic_update_check"),
        ]
        for item in self._settings_patches:
            item.start()

    def tearDown(self):
        for item in reversed(self._settings_patches):
            item.stop()
        self._settings_dir.cleanup()

    @staticmethod
    def _wait_until(app, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.001)
        return bool(predicate())

    def test_default_analysis_dispatches_without_waiting_for_worker(self):
        from PySide6.QtWidgets import QApplication
        from ui_qt.main_window import LoadedState, MainWindow

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        source = SimpleNamespace(
            pair_b=[-1.0, 0.0, 1.0], pair_labels=["B sweep"] * 3,
            pair_b_pos=[-1.0, 0.0, 1.0], pair_b_neg=[-1.0, 0.0, 1.0],
            pair_corrected_pos=[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
            pair_corrected_neg=[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
            pair_raw_pos=[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
            pair_raw_neg=[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
            energy_ev=[1.5, 1.6], source_file="synthetic",
        )

        def blocked_worker(*args, **kwargs):
            started.set()
            release.wait(2.0)
            return None

        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=source)
            window._update_mcd_peak_shift_source(source)
            with patch("ui_qt.feature_pages._mcd_peak_shift_worker", side_effect=blocked_worker):
                window._request_mcd_peak_shift_analysis()
                self.assertTrue(self._wait_until(app, started.is_set))
                self.assertFalse(finished.is_set())
                release.set()
                self.assertTrue(self._wait_until(
                    app, lambda: window._mcd_peak_analysis_worker is None
                ))
                finished.set()
        finally:
            release.set()
            window.close()

    def test_tracker_method_switch_reuses_current_numerical_snapshot(self):
        from PySide6.QtWidgets import QApplication
        from ui_qt.main_window import LoadedState, MainWindow

        QApplication.instance() or QApplication([])
        window = MainWindow()
        source = SimpleNamespace(pair_b=[0.0], source_file="synthetic")
        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=source)
            window._update_mcd_peak_shift_source(source)
            key = window._mcd_peak_computation_key()
            window._mcd_peak_analysis_key = key
            window.mcd_peak_result = object()
            window._mcd_peak_selected_result_method = "Raw spectrum"
            window.mcd_peak_method_results = {
                "Raw spectrum": {"pos": object(), "neg": object()},
                "Second derivative": {"pos": object(), "neg": object()},
            }
            blocked = window.mcd_peak_tracker_method_combo.blockSignals(True)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_tracker_method_combo.blockSignals(blocked)
            with patch.object(window, "_apply_mcd_peak_shift_results") as apply, patch.object(window.thread_pool, "start") as start:
                window._request_mcd_peak_shift_analysis()
            apply.assert_called_once()
            start.assert_not_called()
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
