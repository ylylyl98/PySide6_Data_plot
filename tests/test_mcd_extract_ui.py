from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import QApplication

from ui_qt.mcd_extract_dialog import McdExtractDialog
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg


class McdExtractDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _wait_for_scan(self, dialog: McdExtractDialog) -> None:
        for _ in range(100):
            QThreadPool.globalInstance().waitForDone(10)
            self.app.processEvents()
            if not dialog._scan_running:
                return
        self.fail("MCD extract scan worker did not finish")

    def test_preview_canvas_uses_theme_aware_qtagg_binding(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            dialog = McdExtractDialog(Path(folder_text))
            try:
                self.assertIsInstance(dialog.canvas, ThemeAwareFigureCanvasQTAgg)
            finally:
                dialog.close()

    def _write_result(
        self, root: Path, *, name: str = "sample", doping: float = 6.3
    ) -> None:
        package = root / "Processed Data" / "MCD" / f"{name}_MCD"
        package.mkdir(parents=True)
        csv_name = f"{name}_MCD_vs_B_E1.650000eV_W5meV.csv"
        settings_name = f"{name}_MCD_settings_E1.650000eV_W5meV.json"
        pd.DataFrame({
            "B_increasing_T": [-0.2, 0.0, 0.2],
            "corrected_signed_mean_increasing": [-0.4, 0.0, 0.4],
            "corrected_field_signed_absolute_mean_increasing": [-0.4, 0.0, 0.4],
            "corrected_integral_increasing": [-0.01, 0.0, 0.01],
            "B_decreasing_T": [0.2, 0.0, -0.2],
            "corrected_signed_mean_decreasing": [0.6, 0.0, -0.6],
            "corrected_field_signed_absolute_mean_decreasing": [0.6, 0.0, -0.6],
            "corrected_integral_decreasing": [0.02, 0.0, -0.02],
        }).to_csv(package / csv_name, index=False)
        (package / settings_name).write_text(json.dumps({
            "workflow": "MCD",
            "source_file": f"{name}.csv",
            "package": f"{name}_MCD",
            "created_utc": "2026-08-25T12:00:00+00:00",
            "outputs": [csv_name, settings_name],
            "mcd_b": {
                "center_ev": 1.65,
                "width_mev": 5.0,
                "primary_metric": "mean",
                "fit_near_zero": True,
                "fit_window_t": 0.2,
                "low_field_mcd_slope_increasing_per_T": 2.0,
                "low_field_mcd_slope_decreasing_per_T": 3.0,
            },
            "acquisition_conditions": {"Doping": [doping, doping], "E-field": [0.2, 0.2]},
        }), encoding="utf-8")

    def test_dialog_exposes_both_sweep_branches_and_excludes_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            self._write_result(root)
            dialog = McdExtractDialog(root)
            try:
                self.assertEqual(dialog.table.rowCount(), 0)
                dialog._scan()
                self._wait_for_scan(dialog)
                self.assertEqual(dialog.table.rowCount(), 1)
                headers = [
                    dialog.table.horizontalHeaderItem(column).text()
                    for column in range(dialog.table.columnCount())
                ]
                self.assertEqual(dialog.table.item(0, headers.index("Increasing slope")).text(), "2")
                self.assertEqual(dialog.table.item(0, headers.index("Decreasing slope")).text(), "3")
                self.assertEqual(dialog.order_combo.currentData(), "Auto")
                self.assertEqual(dialog.palette_combo.currentData(), "viridis")
                self.assertFalse(dialog.export_csv_chk.isChecked())
                self.assertIn("XLSX + PNG", dialog.export_btn.text())
                self.assertEqual(
                    dialog._selected_branches(), ("B increasing", "B decreasing")
                )
                self.assertEqual(len(dialog.figure.axes), 2)
                dialog.decreasing_chk.setChecked(False)
                self.assertEqual(dialog._selected_branches(), ("B increasing",))
                self.assertEqual(len(dialog.figure.axes), 1)
                dialog.table.selectRow(0)
                dialog._exclude_highlighted()
                self.assertEqual(dialog.table.item(0, 0).checkState(), Qt.Unchecked)
                self.assertTrue(next((root / "Processed Data" / "MCD").rglob("*.csv")).is_file())
            finally:
                dialog.close()

    def test_near_equal_doping_values_are_one_tolerant_filter_choice(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            self._write_result(root, name="a", doping=6.299999)
            self._write_result(root, name="b", doping=6.300001)
            dialog = McdExtractDialog(root)
            try:
                self.assertEqual(dialog.table.rowCount(), 0)
                dialog._scan()
                self._wait_for_scan(dialog)
                self.assertEqual(dialog.doping_combo.count(), 2)
                self.assertEqual(dialog.doping_combo.itemData(0), None)
                self.assertAlmostEqual(float(dialog.doping_combo.itemData(1)), 6.3, places=6)
                self.assertIn("2 results", dialog.doping_combo.itemText(1))
                dialog.doping_combo.setCurrentIndex(1)
                doping_column = [
                    dialog.table.horizontalHeaderItem(column).text()
                    for column in range(dialog.table.columnCount())
                ].index("Doping (V)")
                self.assertTrue(all(
                    dialog.table.item(row, doping_column).text() == "6.3"
                    for row in range(dialog.table.rowCount())
                ))
            finally:
                dialog.close()


if __name__ == "__main__":
    unittest.main()
