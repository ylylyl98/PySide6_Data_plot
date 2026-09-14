import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialogButtonBox, QListWidget, QPushButton

from core.drr_sources import DrrSourceCache, discover_drr_sources
from ui_qt.controllers_drr import DrrController
from ui_qt.main_window import MainWindow
from tests.ui_test_helpers import wait_for_file_catalog


class DrrMissingSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_missing_sources_preserve_order_and_deduplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.csv").write_text("x", encoding="utf-8")
            controller = DrrController(type("Owner", (), {"current_folder": str(root)})())
            self.assertEqual(controller._drr_missing_sources(["a.csv", "b.csv", "b.csv"]), ["b.csv"])

    @staticmethod
    def _write_csv(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "Vbg,Vtg,740,760,780\n0,0,1,2,3\n1,0,2,3,4\n",
            encoding="utf-8",
        )

    def test_catalog_refresh_retains_deleted_measurement_and_marks_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "Initial Data" / "a_REF.csv"
            second = root / "Initial Data" / "b_REF.csv"
            self._write_csv(first); self._write_csv(second)
            self.window.current_folder = str(root)
            self.window.drr_selected_files = ["Initial Data/a_REF.csv", "Initial Data/b_REF.csv"]
            self.window.drr_available_sources = discover_drr_sources(root)
            second.unlink()
            cache = DrrSourceCache()
            result = (str(root), discover_drr_sources(root, cache=cache), cache)
            self.window._drr_refresh_generation = 1
            self.window._on_drr_catalog_refresh_result(result, 1, True, set())
            self.assertEqual(
                self.window.drr_selected_files,
                ["Initial Data/a_REF.csv", "Initial Data/b_REF.csv"],
            )
            self.assertEqual(
                self.window.drr_controller._drr_missing_sources(self.window.drr_selected_files),
                ["Initial Data/b_REF.csv"],
            )
            self.assertIn("Missing", self.window.drr_measurement_summary.text())

    def test_dialog_blocks_deleted_selection_then_accepts_after_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "Initial Data" / "a_REF.csv"
            second = root / "Initial Data" / "b_REF.csv"
            self._write_csv(first); self._write_csv(second)
            self.window.current_folder = str(root)
            self.window.drr_available_sources = discover_drr_sources(root)
            selected = ["Initial Data/a_REF.csv", "Initial Data/b_REF.csv"]
            observed = {}

            def fake_exec(dialog):
                second.unlink()
                chosen = dialog.findChild(QListWidget, "drr_source_chosen_list")
                buttons = dialog.findChild(QDialogButtonBox)
                buttons.button(QDialogButtonBox.StandardButton.Ok).click()
                observed["blocked_text"] = chosen.item(1).text()
                self.assertEqual(dialog.result(), 0)
                chosen.setCurrentRow(1)
                remove = next(button for button in dialog.findChildren(QPushButton) if button.text() == "Remove")
                remove.click()
                buttons.button(QDialogButtonBox.StandardButton.Ok).click()
                self.assertNotEqual(dialog.result(), 0)
                return dialog.result()

            with patch("ui_qt.controllers_drr.QDialog.exec", fake_exec):
                returned = self.window.drr_controller._open_drr_source_dialog(
                    title="Choose DRR", selected=selected, baseline_mode=False
                )
            self.assertEqual(returned, ["Initial Data/a_REF.csv"])
            self.assertIn("Missing", observed["blocked_text"])

    def test_missing_manual_baseline_blocks_before_background_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement = root / "Initial Data" / "run_REF.csv"
            self._write_csv(measurement)
            self.window.current_folder = str(root)
            self.window.drr_selected_files = ["Initial Data/run_REF.csv"]
            self.window.drr_available_sources = discover_drr_sources(root)
            self.window.drr_baseline_files_manual = ["deleted_REF.csv"]
            self.window.drr_baseline_combo.setCurrentText("External")
            self.window._drr_baseline_user_selected = True
            with (
                patch("ui_qt.main_window.resolve_drr_background_assignments") as resolve,
                patch.object(self.window.thread_pool, "start") as start,
            ):
                self.window._start_load("DRR")
            resolve.assert_not_called()
            start.assert_not_called()
            self.assertIn("Missing DRR source(s)", self.window.statusBar().currentMessage())

    def test_missing_automatic_recipe_and_xlsx_block_before_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement = root / "Initial Data" / "run_REF.csv"
            self._write_csv(measurement)
            self.window.current_folder = str(root)
            self.window.drr_selected_files = ["Initial Data/run_REF.csv"]
            self.window.drr_available_sources = discover_drr_sources(root)
            from core.drr_sources import DrrMeasurementAssignment
            self.window._drr_assignments_automatic = True
            self.window._drr_assignments = (DrrMeasurementAssignment(
                "Initial Data/run_REF.csv", "External", ("deleted_REF.csv",), "last"
            ),)
            with (
                patch("ui_qt.main_window.resolve_drr_background_assignments") as resolve,
                patch.object(self.window.thread_pool, "start") as start,
            ):
                self.window._start_load("DRR")
            resolve.assert_not_called()
            start.assert_not_called()

            self.window.drr_selected_files = ["missing_map.xlsx"]
            self.window._drr_assignments = ()
            self.window._drr_assignments_automatic = False
            with patch.object(self.window.thread_pool, "start") as start:
                self.window._start_load("DRR")
            start.assert_not_called()

    def test_resolved_recipe_missing_baseline_blocks_before_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement = root / "Initial Data" / "run_REF.csv"
            self._write_csv(measurement)
            self.window.current_folder = str(root)
            self.window.drr_selected_files = ["Initial Data/run_REF.csv"]
            self.window.drr_available_sources = discover_drr_sources(root)
            self.window._drr_assignments_automatic = True
            self.window._drr_assignments = ()
            from core.drr_sources import DrrBackgroundResolution, DrrMeasurementAssignment
            resolved = DrrBackgroundResolution(assignments=(DrrMeasurementAssignment(
                "Initial Data/run_REF.csv", "External", ("newly_missing_REF.csv",), "last"
            ),), numerical_path="common")
            with (
                patch("ui_qt.main_window.resolve_drr_background_assignments", return_value=resolved),
                patch.object(self.window.thread_pool, "start") as start,
            ):
                self.window._start_load("DRR")
            start.assert_not_called()
            self.assertIn("Missing DRR source(s)", self.window.statusBar().currentMessage())


if __name__ == "__main__":
    unittest.main()
