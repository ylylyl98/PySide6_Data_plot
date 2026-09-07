from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import QApplication

from ui_qt.main_window import MainWindow
from ui_qt.mcd_organizer_window import McdOrganizerWindow
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg


class McdOrganizerWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _wait_for_scan(self, window: McdOrganizerWindow) -> None:
        for _ in range(100):
            QThreadPool.globalInstance().waitForDone(10)
            self.app.processEvents()
            if not window._scan_running:
                return
        self.fail("MCD organizer scan worker did not finish")

    def test_preview_canvases_use_theme_aware_qtagg_binding(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            window = McdOrganizerWindow(Path(folder_text), auto_scan=False)
            try:
                window._init_plot_widgets()
                self.assertIsInstance(window.canvas, ThemeAwareFigureCanvasQTAgg)
                self.assertIsInstance(window.slope_canvas, ThemeAwareFigureCanvasQTAgg)
            finally:
                window.close()

    def _write_result(
        self, root: Path, name: str, *, doping: float, efield: float, energy: float,
        temperature: float = 4.0,
    ) -> None:
        package = root / "Processed Data" / "MCD" / f"{name}_MCD"
        package.mkdir(parents=True)
        tag = f"E{energy:.6f}eV_W5meV"
        csv_name = f"{name}_MCD_vs_B_{tag}.csv"
        settings_name = f"{name}_MCD_settings_{tag}.json"
        pd.DataFrame({
            "B_increasing_T": [-0.2, 0.0, 0.2],
            "corrected_signed_mean_increasing": [-0.1, 0.0, 0.1],
            "corrected_field_signed_absolute_mean_increasing": [-0.1, 0.0, 0.1],
            "corrected_integral_increasing": [-0.01, 0.0, 0.01],
            "B_decreasing_T": [0.2, 0.0, -0.2],
            "corrected_signed_mean_decreasing": [0.1, 0.0, -0.1],
            "corrected_field_signed_absolute_mean_decreasing": [0.1, 0.0, -0.1],
            "corrected_integral_decreasing": [0.01, 0.0, -0.01],
        }).to_csv(package / csv_name, index=False)
        (package / settings_name).write_text(json.dumps({
            "workflow": "MCD",
            "source_file": f"{name}.csv",
            "package": package.name,
            "created_utc": "2026-08-26T12:00:00+00:00",
            "outputs": [csv_name, settings_name],
            "mcd_b": {
                "center_ev": energy,
                "width_mev": 5.0,
                "primary_metric": "mean",
                "fit_near_zero": False,
                "low_field_mcd_slope_increasing_per_T": 0.5,
                "low_field_mcd_slope_decreasing_per_T": -0.5,
            },
            "acquisition_conditions": {
                "Doping": [doping, doping],
                "E-field": [efield, efield],
                "T": [temperature, temperature],
            },
        }), encoding="utf-8")

    def test_series_first_ui_defaults_to_one_unambiguous_export_series(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            self._write_result(root, "d2_low", doping=2.0, efield=-0.2, energy=1.64)
            self._write_result(root, "d2_high", doping=2.0, efield=0.2, energy=1.68)
            self._write_result(root, "d6_low", doping=6.3, efield=-0.2, energy=1.57)
            self._write_result(root, "d6_high", doping=6.3, efield=0.2, energy=1.61)
            window = McdOrganizerWindow(root, auto_scan=False)
            try:
                window._scan()
                self._wait_for_scan(window)
                self.assertEqual(window.series_list.count(), 2)
                self.assertEqual(window.series_list.item(0).checkState(), Qt.Checked)
                self.assertEqual(window.series_list.item(1).checkState(), Qt.Unchecked)
                self.assertIn("1 series / 2 results selected", window.selection_summary.text())
                self.assertIn("Export 1 selected series", window.export_btn.text())
                self.assertEqual(len(window.figure.axes), 2)
                self.assertEqual(len(window.slope_figure.axes), 1)
                self.assertEqual(window.condition_list.count(), 2)
                window.condition_exclude_btn.click()
                self.assertIn("1 series / 1 results selected", window.selection_summary.text())
                window.condition_list.setCurrentRow(1)
                window.condition_exclude_btn.click()
                self.assertIn("1 series / 0 results selected", window.selection_summary.text())
                self.assertFalse(window.export_btn.isEnabled())
                window.condition_restore_btn.click()
                self.assertIn("1 series / 2 results selected", window.selection_summary.text())
                self.assertFalse(window.details_table.isVisible())
                window.details_btn.setChecked(True)
                self.assertEqual(window.details_table.rowCount(), 2)
            finally:
                window.close()

    def test_comparison_axis_switches_from_efield_to_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            self._write_result(
                root, "cold", doping=6.3, efield=0.2, energy=1.58, temperature=4.0
            )
            self._write_result(
                root, "warm", doping=6.3, efield=0.2, energy=1.61, temperature=80.0
            )
            window = McdOrganizerWindow(root, auto_scan=False)
            try:
                window._scan()
                self._wait_for_scan(window)
                self.assertEqual(window.series_list.count(), 0)
                temperature_index = window.compare_combo.findData("Temperature")
                window.compare_combo.setCurrentIndex(temperature_index)
                self.assertEqual(window.series_list.count(), 1)
                self.assertIn("Temperature series", window.series_list.item(0).text())
                self.assertIn("4→80K", window.series_list.item(0).text())
                self.assertIn("1 series / 2 results selected", window.selection_summary.text())
            finally:
                window.close()

    def test_energy_range_filter_removes_unwanted_processed_windows(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            for name, efield, energy in (
                ("low", -0.2, 1.60), ("mid", 0.0, 1.63), ("high", 0.2, 1.66)
            ):
                self._write_result(
                    root, name, doping=6.3, efield=efield, energy=energy, temperature=2.5
                )
            window = McdOrganizerWindow(root, auto_scan=False)
            try:
                window._scan()
                self._wait_for_scan(window)
                self.assertEqual(window.series_list.count(), 1)
                window.energy_filter_chk.setChecked(True)
                window.energy_min_spin.setValue(1.62)
                window.energy_max_spin.setValue(1.67)
                self.assertEqual(window.series_list.count(), 1)
                self.assertEqual(window.condition_list.count(), 2)
                self.assertIn("1 series / 2 results selected", window.selection_summary.text())
            finally:
                window.close()

    def test_main_app_launches_standalone_organizer_with_current_folder(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            window = MainWindow()
            try:
                window.current_folder = str(root)
                with patch(
                    "ui_qt.main_window.QProcess.startDetached", return_value=(True, 1234)
                ) as launch:
                    window._open_mcd_extract_dialog()
                program, arguments, working_directory = launch.call_args.args
                self.assertTrue(program)
                self.assertEqual(arguments[-1], str(root))
                self.assertEqual(working_directory, str(root))
                self.assertIn("run_mcd_organizer.py", " ".join(arguments))
            finally:
                window.close()

    def test_focus_and_exclusion_reuse_existing_plot_artists(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            self._write_result(root, "low", doping=6.3, efield=0.0, energy=1.57)
            self._write_result(root, "high", doping=6.3, efield=20.0, energy=1.64)
            window = McdOrganizerWindow(root, auto_scan=False)
            try:
                window._scan()
                self._wait_for_scan(window)
                window.condition_list.setCurrentRow(0)
                record_id = str(window.condition_list.currentItem().data(Qt.UserRole))
                artists = list(window._plot_artists[record_id])
                with patch.object(window, "_update_preview") as rebuild:
                    window._exclude_focused_condition()
                    rebuild.assert_not_called()
                self.assertTrue(all(not artist.get_visible() for artist in artists))
                window._restore_focused_condition()
                self.assertTrue(all(artist.get_visible() for artist in artists))
            finally:
                window.close()

    def test_palette_can_be_saved_as_the_experiment_default(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            self._write_result(root, "low", doping=6.3, efield=0.0, energy=1.57)
            self._write_result(root, "high", doping=6.3, efield=20.0, energy=1.64)
            first = McdOrganizerWindow(root, auto_scan=False)
            try:
                first._scan()
                self._wait_for_scan(first)
                first.palette_combo.setCurrentIndex(first.palette_combo.findData("YlOrRd"))
                first.palette_default_btn.click()
            finally:
                first.close()

            second = McdOrganizerWindow(root, auto_scan=False)
            try:
                second._scan()
                self._wait_for_scan(second)
                self.assertEqual(second.palette_combo.currentData(), "YlOrRd")
                self.assertEqual(second.palette_default_btn.text(), "Default: YlOrRd")
            finally:
                second.close()


if __name__ == "__main__":
    unittest.main()
