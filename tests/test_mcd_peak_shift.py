from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from core.mcd_peak_shift import analyze_peak_shift, detect_reflection_peaks, valley_quantities


def _result(fields, *, include_zero=True, n_peaks=2):
    energy = np.linspace(1.5, 2.0, 401)
    rows_pos, rows_neg = [], []
    for b in fields:
        centers = [1.68 + 0.01 * b, 1.82 - 0.01 * b]
        if n_peaks > 2:
            centers.append(1.93 + 0.004 * b)
        row = sum(np.exp(-((energy - c) / 0.006) ** 2) for c in centers) + 0.1
        rows_pos.append(row); rows_neg.append(row * 0.98)
    branches = np.array(["B increasing" if i < len(fields) // 2 else "B decreasing" for i in range(len(fields))])
    return SimpleNamespace(energy_ev=energy, pair_b=np.asarray(fields, float), pair_labels=branches,
                           pair_corrected_pos=np.asarray(rows_pos), pair_corrected_neg=np.asarray(rows_neg),
                           pair_raw_pos=np.asarray(rows_pos), pair_raw_neg=np.asarray(rows_neg), source_file="synthetic")


class MCDPeakShiftCoreTests(unittest.TestCase):
    def test_detects_multiple_peaks_and_refines(self):
        result = _result([0.0])
        peaks = detect_reflection_peaks(result.energy_ev, result.pair_corrected_pos[0], min_distance_points=8)
        self.assertEqual(len(peaks), 2)
        self.assertAlmostEqual(peaks[0].energy_ev, 1.68, places=3)

    def test_no_exact_zero_interpolates_and_preserves_branches(self):
        result = analyze_peak_shift(_result([-1.0, 0.5, 1.0, -0.5]), max_jump_ev=0.1)
        self.assertTrue(all(t.reference_method == "interpolated near-zero" for t in result.tracks))
        self.assertEqual(list(result.branches), ["B increasing", "B increasing", "B decreasing", "B decreasing"])
        self.assertEqual(len(result.tracks[0].points), 2)
        self.assertTrue(all(p.delta_energy_ev is not None for p in result.tracks[0].points))

    def test_missing_peak_is_explicit(self):
        result = _result([-1.0, 0.0, 1.0])
        result.pair_corrected_pos[2] = 0.1
        result.pair_corrected_neg[2] = 0.1
        analysis = analyze_peak_shift(result, max_jump_ev=0.05)
        self.assertTrue(any(p.status == "missing" for t in analysis.tracks for p in t.points))

    def test_valley_ordering_for_positive_and_negative_field(self):
        analysis = analyze_peak_shift(_result([-1.0, -0.5, 0.5, 1.0]), max_jump_ev=0.1)
        values = valley_quantities(analysis)
        negative = next(row for row in values if row["B_T"] < 0)
        positive = next(row for row in values if row["B_T"] > 0)
        self.assertGreater(negative["E_K"], negative["E_Kp"])
        self.assertLess(positive["E_K"], positive["E_Kp"])
        self.assertIn("delta_E_K", negative)
        self.assertIn("delta_E_Kp", positive)

    def test_tracks_are_branch_local_and_selected_pair_is_explicit(self):
        analysis = analyze_peak_shift(_result([-1.0, 0.5, 1.0, -0.5], n_peaks=3), max_jump_ev=0.1)
        self.assertEqual({track.branch for track in analysis.tracks}, {"B increasing", "B decreasing"})
        self.assertTrue(all({point.branch for point in track.points} == {track.branch} for track in analysis.tracks))
        self.assertEqual(len(valley_quantities(analysis, (1, 3))), 4)


class MCDPeakShiftUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_page_exists_and_empty_state_is_safe(self):
        from ui_qt.main_window import MainWindow
        window = MainWindow()
        try:
            labels = [window.workflow_tabs.tabText(i) for i in range(window.workflow_tabs.count())]
            self.assertIn("MCD Peak Shift", labels)
            self.assertFalse(window.mcd_peak_analyze_btn.isEnabled())
            self.assertIn("No MCD result", window.mcd_peak_source_summary.text())
        finally:
            window.close()

    def test_repeat_analysis_clears_rows_and_display_uses_shared_canvas(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.5, 1.0, -0.5], n_peaks=3)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_analyze_btn.click()
            first_rows = window.mcd_peak_table.rowCount()
            self.assertGreaterEqual(window.mcd_peak_k_combo.count(), 3)
            window.mcd_peak_analyze_btn.click()
            self.assertEqual(window.mcd_peak_table.rowCount(), first_rows)
            window.mcd_peak_display_combo.setCurrentText("Absolute E")
            self.assertIn("E (eV)", window.figure.axes[0].get_ylabel())
            window.mcd_peak_display_combo.setCurrentText("Delta E")
            self.assertIn("Delta E", window.figure.axes[0].get_ylabel())
            peak_tab = next(
                index for index in range(window.tabs.count())
                if window.tabs.tabText(index) == "MCD Peak Shift"
            )
            window.tabs.setCurrentIndex(peak_tab)
            self.assertIn("Delta E", window.figure.axes[0].get_ylabel())
        finally:
            window.close()
