from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.export import build_drr_export_base, _strip_terminal_drr_derivative_suffix
from core.loader import DataCube


class DrrExportBaseTests(unittest.TestCase):
    def test_no_derivative_preserves_existing_filename(self) -> None:
        result = build_drr_export_base(
            "run.csv", 3, "DR/R Self", None, 20, 2, "More correct (regrid)"
        )
        self.assertEqual(result, "run_avg3_DR_R_Self")

    def test_first_derivative_encodes_order_window_poly_grid(self) -> None:
        result = build_drr_export_base(
            "run.csv", 3, "DR/R Self", 1, 21, 2, "More correct (regrid)"
        )
        self.assertEqual(result, "run_avg3_DR_R_Self_dE_W21_O2_Regrid")

    def test_second_derivative_encodes_order_window_poly_grid(self) -> None:
        result = build_drr_export_base(
            "run.csv", 3, "DR/R Self", 2, 21, 2, "More correct (regrid)"
        )
        self.assertEqual(result, "run_avg3_DR_R_Self_d2E_W21_O2_Regrid")

    def test_origin_like_grid_mode_token(self) -> None:
        result = build_drr_export_base(
            "run.csv", 3, "DR/R Self", 1, 21, 2, "Origin-like"
        )
        self.assertEqual(result, "run_avg3_DR_R_Self_dE_W21_O2_OriginLike")

    def test_changing_derivative_settings_changes_filename(self) -> None:
        first = build_drr_export_base("run.csv", 3, "DR/R Self", 1, 21, 2, "More correct (regrid)")
        second = build_drr_export_base("run.csv", 3, "DR/R Self", 1, 23, 2, "More correct (regrid)")
        third = build_drr_export_base("run.csv", 3, "DR/R Self", 2, 21, 3, "More correct (regrid)")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertNotEqual(second, third)

    def test_strips_only_terminal_generated_derivative_suffix(self) -> None:
        self.assertEqual(
            _strip_terminal_drr_derivative_suffix("sample_dE_W21_O2_Regrid"),
            "sample",
        )
        self.assertEqual(
            _strip_terminal_drr_derivative_suffix("sample_d2E_OriginLike"),
            "sample",
        )
        # Ordinary user filename text is left untouched.
        self.assertEqual(
            _strip_terminal_drr_derivative_suffix("sample_dE_scan"),
            "sample_dE_scan",
        )
        self.assertEqual(
            _strip_terminal_drr_derivative_suffix("regrid_test"),
            "regrid_test",
        )

    def test_does_not_duplicate_terminal_generated_derivative_suffix(self) -> None:
        result = build_drr_export_base(
            "sample_dE_W21_O2_Regrid.csv", 4, "DR/R Self", 1, 9, 2, "More correct (regrid)"
        )
        self.assertEqual(result, "sample_avg4_DR_R_Self_dE_W9_O2_Regrid")
        self.assertEqual(result.count("_dE_W"), 1)

    def test_csv_and_xlsx_stems_share_filename_grammar(self) -> None:
        csv_base = build_drr_export_base(
            "dR_R.csv", 2, "DR/R Map", 1, 21, 2, "More correct (regrid)"
        )
        xlsx_base = build_drr_export_base(
            "dR_R.xlsx", 2, "DR/R Map", 1, 21, 2, "More correct (regrid)"
        )
        self.assertEqual(csv_base, xlsx_base)
        self.assertEqual(csv_base, "dR_R_avg2_DR_R_Map_dE_W21_O2_Regrid")


class DrrExportMetadataUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        from ui_qt.main_window import MainWindow

        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def test_export_metadata_records_clamped_window_not_requested_value(self) -> None:
        from ui_qt.main_window import LoadedState

        energy = np.linspace(-2.0, 2.0, 7)
        cube = DataCube(
            energy=energy,
            gate=np.asarray([0.0]),
            Z=np.asarray([energy**2]),
            gate_label="Gate",
            title="quadratic",
            cbar_label="DR/R",
        )
        self.window.loaded = LoadedState(
            mode="DRR",
            folder=".",
            cube=cube,
            primary_file="run.csv",
            selected_files=["run.csv"],
        )
        self.window.drr_derivative_combo.blockSignals(True)
        self.window.drr_sg_poly_spin.blockSignals(True)
        self.window.drr_sg_window_spin.blockSignals(True)
        try:
            self.window.drr_derivative_combo.setCurrentText("dE")
            self.window.drr_sg_poly_spin.setValue(6)
            self.window.drr_sg_window_spin.setValue(20)
        finally:
            self.window.drr_derivative_combo.blockSignals(False)
            self.window.drr_sg_poly_spin.blockSignals(False)
            self.window.drr_sg_window_spin.blockSignals(False)

        _export_cube, deriv, used_win, poly = self.window._drr_cube_with_metadata()

        self.assertEqual(deriv, 1)
        self.assertEqual(poly, 6)
        self.assertEqual(used_win, 7)

        base = build_drr_export_base(
            "run.csv", 1, "DR/R Self", deriv, used_win, poly, "More correct (regrid)"
        )
        self.assertEqual(base, "run_avg1_DR_R_Self_dE_W7_O6_Regrid")


if __name__ == "__main__":
    unittest.main()
