from __future__ import annotations

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from core.mcd_peak_shift import analyze_peak_shift, detect_reflection_peaks, format_mcd_angle, source_spectra, valley_quantities


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
    def test_mcd_angle_display_snaps_only_near_integer(self):
        self.assertEqual(format_mcd_angle(29.9995), "30")
        self.assertEqual(format_mcd_angle(75.0), "75")
        self.assertEqual(format_mcd_angle(29.98), "29.98")

    def test_second_derivative_tracking_finds_resonance_center(self):
        result = _result([0.0], n_peaks=1)
        analysis = analyze_peak_shift(
            result,
            source="raw pos",
            tracking_method="Second derivative",
            derivative_window_points=35,
            prominence_fraction=0.03,
            max_peaks=4,
        )
        energies = [point.energy_ev for track in analysis.tracks for point in track.points if point.energy_ev is not None]
        self.assertEqual(analysis.tracking_method, "Second derivative")
        self.assertTrue(any(abs(float(energy) - 1.68) < 0.006 for energy in energies))

    def test_source_spectra_aligns_raw_columns_to_ascending_energy(self):
        result = _result([0.0], n_peaks=1)
        result.energy_ev = result.energy_ev[::-1]
        result.pair_raw_pos = result.pair_raw_pos[:, ::-1]
        aligned = source_spectra(result, "raw pos")
        self.assertTrue(np.all(np.diff(result.energy_ev[np.argsort(result.energy_ev)]) > 0))
        self.assertAlmostEqual(float(result.energy_ev[np.argsort(result.energy_ev)][np.argmax(aligned[0])]), 1.68, places=2)

    def test_peak_tracking_uses_ascending_energy_for_descending_input(self):
        ascending = np.linspace(1.5, 2.0, 801)
        spectrum = np.exp(-((ascending - 1.61) / 0.009) ** 2) + 0.1
        descending = SimpleNamespace(
            energy_ev=ascending[::-1].copy(),
            pair_b=np.asarray([0.0]),
            pair_labels=np.asarray(["B sweep"]),
            pair_raw_pos=np.asarray([spectrum[::-1].copy()]),
            pair_raw_neg=np.asarray([spectrum[::-1].copy()]),
            pair_corrected_pos=np.asarray([spectrum[::-1].copy()]),
            pair_corrected_neg=np.asarray([spectrum[::-1].copy()]),
            source_file="synthetic",
        )
        original_energy = descending.energy_ev.copy()
        original_spectrum = descending.pair_raw_pos.copy()
        for method in ("Raw spectrum", "Second derivative"):
            analysis = analyze_peak_shift(descending, source="raw pos", tracking_method=method, derivative_window_points=35, max_peaks=4)
            energies = [point.energy_ev for track in analysis.tracks for point in track.points if point.energy_ev is not None]
            self.assertTrue(energies)
            self.assertAlmostEqual(float(energies[0]), 1.61, places=2)
        np.testing.assert_array_equal(descending.energy_ev, original_energy)
        np.testing.assert_array_equal(descending.pair_raw_pos, original_spectrum)

    def test_peak_tracking_honors_ascending_wavelength_metadata(self):
        ascending = np.linspace(1.5, 2.0, 801)
        spectrum = np.exp(-((ascending - 1.61) / 0.009) ** 2) + 0.1
        descending = SimpleNamespace(
            energy_ev=ascending[::-1].copy(),
            wavelength_nm=(1239.841984 / ascending[::-1]).copy(),
            pair_b=np.asarray([0.0]),
            pair_labels=np.asarray(["B sweep"]),
            pair_raw_pos=np.asarray([spectrum[::-1].copy()]),
            pair_raw_neg=np.asarray([spectrum[::-1].copy()]),
            pair_corrected_pos=np.asarray([spectrum[::-1].copy()]),
            pair_corrected_neg=np.asarray([spectrum[::-1].copy()]),
            source_file="synthetic",
        )
        for method in ("Raw spectrum", "Second derivative"):
            analysis = analyze_peak_shift(descending, source="raw pos", tracking_method=method, derivative_window_points=35, max_peaks=4)
            energies = [point.energy_ev for track in analysis.tracks for point in track.points if point.energy_ev is not None]
            self.assertTrue(energies)
            self.assertAlmostEqual(float(energies[0]), 1.61, places=2)

    def test_second_derivative_tracking_does_not_pair_an_absorption_dip(self):
        result = _result([0.0], n_peaks=1)
        dip = 0.1 - np.exp(-((result.energy_ev - 1.72) / 0.02) ** 2)
        result.pair_raw_pos[0] = dip
        result.pair_raw_neg[0] = dip
        analysis = analyze_peak_shift(result, source="raw pos", tracking_method="Second derivative", derivative_window_points=35, max_peaks=4)
        self.assertTrue(any(track.feature_kind == "dip" for track in analysis.tracks))
        self.assertFalse(any(track.feature_kind == "peak" for track in analysis.tracks))

    def test_raw_peak_and_dip_are_tracked_as_separate_kinds(self):
        result = _result([0.0], n_peaks=1)
        spectrum = 0.1 + np.exp(-((result.energy_ev - 1.72) / 0.008) ** 2) - 0.06 * np.exp(-((result.energy_ev - 1.54) / 0.006) ** 2)
        result.pair_raw_pos[0] = spectrum
        result.pair_raw_neg[0] = spectrum
        analysis = analyze_peak_shift(result, source="raw pos", max_peaks=4)
        centers = {track.feature_kind: float(track.points[0].energy_ev) for track in analysis.tracks}
        self.assertAlmostEqual(centers["peak"], 1.72, places=2)
        self.assertAlmostEqual(centers["dip"], 1.54, places=2)

    def test_second_derivative_keeps_one_candidate_for_one_raw_resonance(self):
        result = _result([0.0], n_peaks=1)
        resonance = np.exp(-((result.energy_ev - 1.72) / 0.005) ** 4) + 0.1
        result.pair_raw_pos[0] = resonance
        result.pair_raw_neg[0] = resonance
        analysis = analyze_peak_shift(result, source="raw pos", tracking_method="Second derivative", derivative_window_points=35, max_peaks=8)
        energies = [point.energy_ev for track in analysis.tracks for point in track.points if point.energy_ev is not None]
        self.assertEqual(len(energies), 1)
        self.assertAlmostEqual(float(energies[0]), 1.72, places=2)

    def test_near_equal_curvature_lobes_are_ambiguous_for_shift_tracking(self):
        energy = np.linspace(1.5, 2.0, 2001)
        fields = np.linspace(-1.0, 1.0, 11)
        rows = [np.exp(-((energy - (1.72 + 0.0017 * field)) / 0.005) ** 4) + 0.1 for field in fields]
        result = SimpleNamespace(
            energy_ev=energy,
            pair_b=fields,
            pair_labels=np.full(fields.size, "B increasing"),
            pair_raw_pos=np.asarray(rows),
            pair_raw_neg=np.asarray(rows),
            pair_corrected_pos=np.asarray(rows),
            pair_corrected_neg=np.asarray(rows),
            source_file="synthetic",
        )
        analysis = analyze_peak_shift(result, source="raw pos", tracking_method="Second derivative", derivative_window_points=35, max_peaks=8)
        self.assertTrue(any(point.status == "ambiguous" for track in analysis.tracks for point in track.points))
        self.assertTrue(all(point.delta_energy_ev is None for track in analysis.tracks for point in track.points))

    def test_boundary_candidate_is_flagged_but_interior_broad_peak_is_reliable(self):
        energy = np.linspace(1.5, 2.0, 401)
        edge = np.exp(-((energy - 1.505) / 0.012) ** 2) + 0.1
        interior = np.exp(-((energy - 1.72) / 0.035) ** 2) + 0.1

        edge_peaks = detect_reflection_peaks(energy, edge, prominence_fraction=0.01)
        interior_peaks = detect_reflection_peaks(energy, interior, prominence_fraction=0.01)

        self.assertEqual(len(edge_peaks), 1)
        self.assertEqual(edge_peaks[0].quality, "Boundary unreliable")
        self.assertEqual(len(interior_peaks), 1)
        self.assertEqual(interior_peaks[0].quality, "OK")

    def test_boundary_point_is_retained_but_excluded_from_valley_quantities(self):
        result = _result([-1.0, 0.0, 1.0])
        energy = result.energy_ev
        edge = np.exp(-((energy - 1.505) / 0.012) ** 2) + 0.1
        result.pair_corrected_pos[0] = edge
        result.pair_corrected_neg[0] = edge
        analysis = analyze_peak_shift(result, max_peaks=2, prominence_fraction=0.01)

        boundary_points = [point for track in analysis.tracks for point in track.points if point.field_t == -1.0]
        self.assertTrue(boundary_points)
        self.assertTrue(any(point.status == "Boundary unreliable" for point in boundary_points))
        rows = valley_quantities(analysis)
        boundary_rows = [row for row in rows if row["B_T"] == -1.0]
        self.assertTrue(boundary_rows)
        self.assertEqual(boundary_rows[0]["status"], "Boundary unreliable")
        self.assertIsNone(boundary_rows[0]["splitting_E_Kp_minus_E_K"])
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

    def test_missing_zero_reference_is_unavailable_instead_of_nearest_nonzero(self):
        # With no zero or opposite-side measurement, a nearest non-zero point
        # must not be labelled or used as E0.
        result = analyze_peak_shift(_result([-1.0]), max_jump_ev=0.1)
        self.assertTrue(result.tracks)
        self.assertTrue(all(track.reference_energy_ev is None for track in result.tracks))
        self.assertTrue(all(track.reference_method == "unavailable" for track in result.tracks))

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

    def test_local_fit_defaults_to_raw_r_and_routes_each_physical_channel(self):
        from ui_qt.feature_pages import _mcd_fit_source_for_channel, _mcd_local_fit_worker

        window = None
        try:
            from ui_qt.main_window import MainWindow
            window = MainWindow()
            self.assertEqual(window.mcd_peak_tracker_method_combo.currentText(), "Local mixed fit")
            self.assertEqual(
                [window.mcd_peak_source_combo.itemText(index) for index in range(window.mcd_peak_source_combo.count())],
                ["Raw R", "MCD-corrected R"],
            )
            self.assertFalse(window.mcd_peak_source_combo.isHidden())
        finally:
            if window is not None:
                window.close()

        energy = np.linspace(1.62, 1.66, 20)
        values = np.tile(np.linspace(0.0, 1.0, 20), (2, 1))
        source = SimpleNamespace(
            pair_b=np.asarray([-0.1, 0.1]), pair_b_pos=np.asarray([-0.1, 0.1]), pair_b_neg=np.asarray([-0.1, 0.1]),
            pair_interpolated_pos=np.zeros(2, dtype=bool), pair_interpolated_neg=np.zeros(2, dtype=bool),
            pair_raw_pos=values, pair_raw_neg=values + 1.0,
            pair_corrected_pos=values + 2.0, pair_corrected_neg=values + 3.0,
            energy_ev=energy, pair_labels=np.asarray(["B increasing", "B decreasing"]), source_file="synthetic",
        )
        self.assertEqual(_mcd_fit_source_for_channel("Raw R", "pos"), "raw pos")
        self.assertEqual(_mcd_fit_source_for_channel("Raw R", "neg"), "raw neg")
        self.assertEqual(_mcd_fit_source_for_channel("MCD-corrected R", "pos"), "corrected pos")
        self.assertEqual(_mcd_fit_source_for_channel("MCD-corrected R", "neg"), "corrected neg")
        with patch("ui_qt.feature_pages.analyze_local_peak_shift", side_effect=lambda _result, **kwargs: kwargs["source"]) as fit:
            output = _mcd_local_fit_worker(
                source, seed_energy_ev=1.64, locator_energy_ev=1.64, feature_kind="peak", peak_id=1,
                spectrum_source="MCD-corrected R", background_model="linear", window_ev=(1.63, 1.65), max_starts=1,
            )
        self.assertEqual(list(output.values()), ["corrected pos", "corrected neg"])
        self.assertEqual([call.kwargs["source"] for call in fit.call_args_list], ["corrected pos", "corrected neg"])

    def test_source_change_invalidates_locators_but_keeps_source_keyed_cache(self):
        from ui_qt.main_window import LoadedState, MainWindow

        window = MainWindow()
        try:
            fake = _result([0.0], n_peaks=1)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window._mcd_peak_locator_results = {"pos": object()}
            cache = {("raw",): object()}
            window._mcd_peak_local_fit_cache = cache
            generation = window._mcd_peak_fit_generation
            with patch.object(window, "_analyze_mcd_peak_shift") as analyze:
                window.mcd_peak_source_combo.setCurrentText("MCD-corrected R")
            self.assertEqual(window._mcd_peak_locator_results, {})
            self.assertIs(window._mcd_peak_local_fit_cache, cache)
            self.assertGreater(window._mcd_peak_fit_generation, generation)
            analyze.assert_called_once_with()
        finally:
            window.close()

    def test_local_export_contains_both_channel_centers_and_valley_splitting(self):
        from core.mcd_peak_shift import PeakPoint, PeakShiftResult, PeakTrack
        from ui_qt.main_window import LoadedState, MainWindow

        window = MainWindow()
        try:
            fields = np.asarray([-1.0, 1.0])
            branches = np.asarray(["B increasing", "B increasing"])
            def analysis(channel, energies, locator):
                points = tuple(
                    PeakPoint(float(field), "B increasing", float(energy), float(energy - energies[0]), "tracked")
                    for field, energy in zip(fields, energies)
                )
                track = PeakTrack(1, "B increasing", points, float(energies[0]), 0.0, "exact 0 T", "OK", "peak", locator, "Local mixed fit (linear background)")
                return PeakShiftResult(fields, branches, ((), ()), (track,), f"raw {channel}", "Local mixed fit", (), locator, (1.63, 1.66), "Local mixed fit (linear background)")
            pos = analysis("pos", (1.68, 1.70), 1.6436)
            neg = analysis("neg", (1.68, 1.67), 1.6436)
            source = _result([-1.0, 1.0], n_peaks=1)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=source)
            window.mcd_peak_result = pos
            window.mcd_peak_method_results = {"Local mixed fit": {"pos": pos, "neg": neg}}
            window.mcd_peak_channel_results = {"pos": pos, "neg": neg}
            window.mcd_peak_selector_combo.addItem("selected", ("pos", 1, "B increasing", "peak"))
            window.mcd_peak_k_combo.addItem("Peak 1", 1)
            window.mcd_peak_kp_combo.addItem("Peak 1", 1)
            with __import__("tempfile").TemporaryDirectory() as temp:
                path = __import__("pathlib").Path(temp) / "local.csv"
                with patch("ui_qt.feature_pages.QFileDialog.getSaveFileName", return_value=(str(path), "CSV files (*.csv)")):
                    window._export_mcd_peak_shift()
                import csv
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            self.assertEqual({row["channel"] for row in rows}, {"pos", "neg"})
            splitting = [row["delta_E_Kp_minus_K_eV"] for row in rows if row["delta_E_Kp_minus_K_eV"]]
            self.assertTrue(splitting)
            self.assertTrue(all(row["fit_source"] == "Raw R" for row in rows))
            self.assertTrue(all(row["fit_window_low_eV"] == "1.63" for row in rows))
            self.assertTrue(all(row["background_model"] == "linear" for row in rows))
        finally:
            window.close()

    def test_stale_local_completion_is_ignored_and_cached_fit_reenables_analyze(self):
        from core.mcd_local_fit import local_fit_cache_key
        from core.mcd_peak_shift import analyze_peak_shift
        from ui_qt.main_window import LoadedState, MainWindow

        window = MainWindow()
        try:
            source = _result([-1.0, 0.0, 1.0], n_peaks=1)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=source)
            window._update_mcd_peak_shift_source(source)
            analysis = analyze_peak_shift(source, source="raw pos", max_peaks=1)
            window._mcd_peak_fit_generation = 2
            window._finish_mcd_local_fit(
                object(), 1, {"pos": analysis, "neg": analysis}, source,
                1.68, 1.68, "peak", ("stale",),
            )
            self.assertIsNone(window.mcd_peak_result)
            self.assertEqual(window._mcd_peak_local_fit_cache, {})

            seed = 1.68
            window_ev = (seed - 0.0136, seed + 0.0164)
            settings = (
                window.mcd_peak_prom_spin.value(), window.mcd_peak_dist_spin.value(),
                window.mcd_peak_smooth_spin.value(), window.mcd_peak_jump_spin.value(),
                window.mcd_peak_deriv_window_spin.value(), "peak", 1, "B increasing",
            )
            key = local_fit_cache_key(
                source, source="Raw R", background_model="linear", window_ev=window_ev,
                seed_energy_ev=seed, settings=settings,
            )
            window._mcd_peak_local_fit_cache = {key: {"pos": analysis, "neg": analysis}}
            window._request_mcd_local_fit(seed_energy_ev=seed, locator_energy_ev=seed, feature_kind="peak")
            self.assertIs(window.mcd_peak_result, analysis)
            self.assertTrue(window.mcd_peak_analyze_btn.isEnabled())
        finally:
            window.close()

    def test_local_cache_miss_keeps_requested_candidate_during_async_fit(self):
        from core.mcd_peak_shift import analyze_peak_shift
        from ui_qt.main_window import LoadedState, MainWindow

        window = MainWindow()
        try:
            source = _result([-1.0, 0.0, 1.0], n_peaks=3)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=source)
            window._update_mcd_peak_shift_source(source)
            locator = analyze_peak_shift(source, source="raw pos", max_peaks=3)
            window._mcd_peak_locator_results = {"pos": locator, "neg": locator}
            window._populate_mcd_peak_preview_controls()
            track = next(track for track in locator.tracks if track.peak_id == 2 and track.reference_energy_ev is not None)
            key = ("pos", int(track.peak_id), str(track.branch), str(track.feature_kind))
            worker = Mock()
            worker.signals.result = Mock()
            worker.signals.error = Mock()
            worker.signals.finished = Mock()
            with patch("ui_qt.feature_pages.Worker", return_value=worker), patch.object(window.thread_pool, "start"), patch.object(window, "_refresh_mcd_peak_plot"):
                window._request_mcd_local_fit(
                    seed_energy_ev=float(track.reference_energy_ev),
                    locator_energy_ev=float(track.reference_energy_ev),
                    feature_kind=str(track.feature_kind),
                    selection_key=key,
                )
            self.assertEqual(tuple(window.mcd_peak_selector_combo.currentData()), key)
            self.assertEqual(tuple(window._mcd_peak_local_requested_key), key)
            self.assertFalse(window.mcd_peak_analyze_btn.isEnabled())
            worker.signals.result.assert_not_called()
        finally:
            window.close()

    def test_legacy_tracker_uses_the_selected_reflection_source(self):
        from core.mcd_peak_shift import analyze_peak_shift
        from ui_qt.main_window import LoadedState, MainWindow

        window = MainWindow()
        try:
            source = _result([-1.0, 0.0, 1.0], n_peaks=2)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=source)
            window._update_mcd_peak_shift_source(source)
            blocked = window.mcd_peak_source_combo.blockSignals(True)
            window.mcd_peak_source_combo.setCurrentText("MCD-corrected R")
            window.mcd_peak_source_combo.blockSignals(blocked)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            with patch("ui_qt.feature_pages.analyze_peak_shift", wraps=analyze_peak_shift) as analyze:
                window._analyze_mcd_peak_shift_legacy()
            self.assertTrue(analyze.call_args_list)
            self.assertEqual(
                {call.kwargs["source"] for call in analyze.call_args_list},
                {"corrected pos", "corrected neg"},
            )
        finally:
            window.close()

    def test_repeat_analysis_clears_rows_and_display_uses_shared_canvas(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.5, 1.0, -0.5], n_peaks=3)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            first_rows = window.mcd_peak_table.rowCount()
            self.assertGreaterEqual(window.mcd_peak_k_combo.count(), 3)
            window.mcd_peak_analyze_btn.click()
            self.assertEqual(window.mcd_peak_table.rowCount(), first_rows)
            window.mcd_peak_display_combo.setCurrentText("Absolute E")
            self.assertIn("R intensity", window.figure.axes[0].get_ylabel())
            window.mcd_peak_display_combo.setCurrentText("Delta E")
            self.assertIn("Energy", window.figure.axes[0].get_xlabel())
            peak_tab = next(
                index for index in range(window.tabs.count())
                if window.tabs.tabText(index) == "MCD Peak Shift"
            )
            window.tabs.setCurrentIndex(peak_tab)
            self.assertIn("R intensity", window.figure.axes[0].get_ylabel())
        finally:
            window.close()

    def test_map_first_preview_has_two_maps_spectrum_and_branch_controls(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, -0.5, 0.5, 1.0], n_peaks=3)
            fake.pos_angle, fake.neg_angle = 75.0, 29.9995
            fake.pair_b_pos = np.asarray(fake.pair_b) + 0.01
            fake.pair_b_neg = np.asarray(fake.pair_b) - 0.01
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            self.assertEqual(len(window.figure.axes), 3)
            self.assertEqual(len(window.mcd_peak_map_axes), 0)
            self.assertTrue(all(axis.get_visible() for axis in window.figure.axes))
            self.assertEqual(window.mcd_peak_spectrum_ax.get_title(), "Selected field spectrum")
            self.assertIn("B increasing", [window.mcd_peak_branch_combo.itemText(i) for i in range(window.mcd_peak_branch_combo.count())])
            self.assertGreater(window.mcd_peak_selector_combo.count(), 0)
            self.assertNotIn("Valley pair", window.mcd_peak_k_combo.parentWidget().accessibleName())
            window.mcd_peak_show_maps_chk.setChecked(True)
            self.assertEqual(len(window.mcd_peak_map_axes), 2)
            self.assertIsNone(window.mcd_peak_spectrum_ax)
            self.assertTrue(all(axis.get_visible() for axis in window.mcd_peak_map_axes))
        finally:
            window.close()

    def test_tracker_method_switch_keeps_nearest_peak_energy(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.5, 1.0, -0.5], n_peaks=3)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            self.assertEqual(set(window.mcd_peak_method_results), {"Raw spectrum", "Second derivative"})
            d2_before = window.mcd_peak_method_results["Second derivative"]["pos"]
            before = window.mcd_peak_selector_combo.currentData()
            old_track = next(track for track in window.mcd_peak_channel_results[before[0]].tracks if track.peak_id == before[1] and track.branch == before[2])
            old_energy = float(np.median([point.energy_ev for point in old_track.points if point.energy_ev is not None]))
            window.mcd_peak_tracker_method_combo.setCurrentText("Raw spectrum")
            after = window.mcd_peak_selector_combo.currentData()
            new_track = next(track for track in window.mcd_peak_channel_results[after[0]].tracks if track.peak_id == after[1] and track.branch == after[2])
            new_energy = float(np.median([point.energy_ev for point in new_track.points if point.energy_ev is not None]))
            self.assertLess(abs(new_energy - old_energy), 0.01)
            window.mcd_peak_method_results["Raw spectrum"]["pos"] = replace(window.mcd_peak_method_results["Raw spectrum"]["pos"], tracks=())
            window._refresh_mcd_peak_plot()
            notes = "\n".join(text.get_text() for text in window.mcd_peak_shift_ax.texts)
            self.assertIn("Raw spectrum · B increasing: unmatched", notes)
            self.assertIn("Raw spectrum · B decreasing: unmatched", notes)
            window.mcd_peak_deriv_window_spin.setValue(37)
            self.assertIsNot(d2_before, window.mcd_peak_method_results["Second derivative"]["pos"])
        finally:
            window.close()

    def test_map_branch_filter_and_click_update_selected_field(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, -0.5, 0.5, 1.0], n_peaks=3)
            fake.pos_angle, fake.neg_angle = 75.0, 29.9995
            fake.pair_b_pos = np.asarray(fake.pair_b) + 0.01
            fake.pair_b_neg = np.asarray(fake.pair_b) - 0.01
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            window.mcd_peak_show_maps_chk.setChecked(True)
            window.mcd_peak_branch_combo.setCurrentText("B increasing")
            self.assertTrue(all(np.isfinite(np.asarray(line.get_xdata(), float)).sum() == 2 for line in window._mcd_peak_track_lines if line.get_visible()))
            window._on_canvas_click(SimpleNamespace(button=1, xdata=1.60, ydata=-0.47, inaxes=window.mcd_peak_map_axes[0]))
            self.assertAlmostEqual(window.mcd_peak_selected_field, -0.5)
            self.assertIn("-0.49", window.mcd_peak_field_combo.currentText())
        finally:
            window.close()

    def test_inspection_axes_are_real_and_toggle_without_hidden_overlays(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.0, 0.0, 1.0], n_peaks=2)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            self.assertEqual(len(window.figure.axes), 3)
            self.assertTrue(all(axis.get_visible() for axis in window.figure.axes))
            window.mcd_peak_show_derivative_chk.setChecked(False)
            self.assertEqual(len(window.figure.axes), 2)
            window.mcd_peak_show_maps_chk.setChecked(True)
            self.assertEqual(len(window.figure.axes), 3)
            self.assertEqual(len(window.mcd_peak_map_axes), 2)
            self.assertTrue(all(axis.get_visible() for axis in window.mcd_peak_map_axes))
        finally:
            window.close()

    def test_branch_switch_matches_unique_true_e0_and_keeps_feature_kind(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.0, 0.0, 1.0], n_peaks=3)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            window.mcd_peak_branch_combo.setCurrentText("B increasing")
            self.assertGreaterEqual(window.mcd_peak_selector_combo.count(), 2)
            second_index = next(
                index for index in range(window.mcd_peak_selector_combo.count())
                if (lambda data: data is not None and next(
                    track for track in window.mcd_peak_channel_results[data[0]].tracks
                    if track.peak_id == data[1] and track.branch == data[2] and track.feature_kind == data[3]
                ).reference_energy_ev > 1.8)(window.mcd_peak_selector_combo.itemData(index))
            )
            window.mcd_peak_selector_combo.setCurrentIndex(second_index)
            selected = window.mcd_peak_selector_combo.currentData()
            selected_track = next(track for track in window.mcd_peak_channel_results[selected[0]].tracks if track.peak_id == selected[1] and track.branch == selected[2] and track.feature_kind == selected[3])
            self.assertIsNotNone(selected_track.reference_energy_ev)
            window.mcd_peak_branch_combo.setCurrentText("B decreasing")
            switched = window.mcd_peak_selector_combo.currentData()
            self.assertIsNotNone(switched)
            switched_track = next(track for track in window.mcd_peak_channel_results[switched[0]].tracks if track.peak_id == switched[1] and track.branch == switched[2] and track.feature_kind == switched[3])
            self.assertEqual(switched[0], selected[0])
            self.assertEqual(switched[3], selected[3])
            self.assertLess(abs(float(switched_track.reference_energy_ev) - float(selected_track.reference_energy_ev)), 0.005)
        finally:
            window.close()

    def test_result_click_uses_nearby_curve_metadata_and_keeps_both_branches(self):
        from types import SimpleNamespace
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.0, 0.0, 1.0], n_peaks=2)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            window.mcd_peak_result_mode_combo.setCurrentText("Single peak shift")
            lines = [line for line in window.mcd_peak_shift_ax.lines if getattr(line, "_mcd_peak_branch", None) == "B decreasing"]
            self.assertGreaterEqual(len(lines), 1)
            line = next(line for line in lines if np.isfinite(np.asarray(line.get_xdata(), float)[-1]) and np.isfinite(np.asarray(line.get_ydata(), float)[-1]))
            xdata = float(np.asarray(line.get_xdata(), float)[-1])
            ydata = float(np.asarray(line.get_ydata(), float)[-1])
            window._on_mcd_peak_result_click(SimpleNamespace(xdata=xdata, ydata=ydata, inaxes=window.mcd_peak_shift_ax))
            self.assertEqual(window.mcd_peak_branch_combo.currentText(), "B decreasing")
            self.assertIn("B decreasing", window.mcd_peak_status.text())
            plotted_branches = {getattr(item, "_mcd_peak_branch", None) for item in window.mcd_peak_shift_ax.lines}
            self.assertIn("B increasing", plotted_branches)
            self.assertIn("B decreasing", plotted_branches)
        finally:
            window.close()

    @staticmethod
    def _plot_event(axis, xdata, ydata, *, button=1, inaxes=None):
        if inaxes is None:
            inaxes = axis
        x, y = axis.transData.transform((float(xdata), float(ydata)))
        return SimpleNamespace(button=button, x=float(x), y=float(y), xdata=xdata, ydata=ydata, inaxes=inaxes, key=None)

    def test_peak_result_cursor_drag_selects_measured_field_across_redraw_and_release_outside(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.0, 0.0, 1.0], n_peaks=2)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            window.mcd_peak_result_mode_combo.setCurrentText("Single peak shift")
            window.canvas.draw()
            branch_before = window.mcd_peak_branch_combo.currentText()
            start_index = int(window.mcd_peak_field_combo.currentData())
            target_index = next(
                int(window.mcd_peak_field_combo.itemData(index))
                for index in range(window.mcd_peak_field_combo.count())
                if int(window.mcd_peak_field_combo.itemData(index)) != start_index
            )
            result_ax = window.mcd_peak_shift_ax
            target_x = float(fake.pair_b[target_index])
            ylim = result_ax.get_ylim()
            press = self._plot_event(result_ax, float(fake.pair_b[start_index]), sum(ylim) / 2.0)
            window._on_canvas_click(press)
            self.assertEqual(window._mcd_peak_drag["kind"], "field")

            # Selection redraws the figure, so motion deliberately uses the new
            # result axis object rather than the axis from button press.
            window.canvas.draw()
            result_ax = window.mcd_peak_shift_ax
            ylim = result_ax.get_ylim()
            motion = self._plot_event(result_ax, target_x, sum(ylim) / 2.0)
            window._on_canvas_motion(motion)
            self.assertEqual(window.mcd_peak_field_combo.currentData(), target_index)
            self.assertEqual(window.mcd_peak_branch_combo.currentText(), branch_before)

            window._on_canvas_release(SimpleNamespace(button=1, x=None, y=None, xdata=None, ydata=None, inaxes=None, key=None))
            self.assertIsNone(window._mcd_peak_drag)
        finally:
            window.close()

    def test_peak_feature_drag_moves_continuous_center_without_mutating_data(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.0, 0.0, 1.0], n_peaks=3)
            original_energy = fake.energy_ev.copy()
            original_pos = fake.pair_raw_pos.copy()
            original_neg = fake.pair_raw_neg.copy()
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            window.canvas.draw()
            selected = window.mcd_peak_selector_combo.currentData()
            self.assertIsNotNone(selected)
            channel, _peak_id, branch, _kind = selected
            tracks = [track for track in window.mcd_peak_channel_results[channel].tracks if track.branch == branch and track.quality != "Boundary unreliable"]
            target_track = next(track for track in tracks if track.peak_id != selected[1])
            current_field = float(getattr(fake, f"pair_b_{channel}", fake.pair_b)[int(window.mcd_peak_field_combo.currentData())])
            target_point = min((point for point in target_track.points if point.status == "tracked"), key=lambda point: abs(float(point.field_t) - current_field))
            raw_ax = window.mcd_peak_spectrum_ax
            line = next(line for line in raw_ax.lines if getattr(line, "_mcd_peak_drag_kind", None) == "feature" and getattr(line, "_mcd_peak_drag_channel", None) == channel)
            selected_energy = float(np.asarray(line.get_xdata(), float)[0])
            ylim = raw_ax.get_ylim()
            window._on_canvas_click(self._plot_event(raw_ax, selected_energy, sum(ylim) / 2.0))
            self.assertEqual(window._mcd_peak_drag["kind"], "feature")
            window.canvas.draw()
            raw_ax = window.mcd_peak_spectrum_ax
            ylim = raw_ax.get_ylim()
            selector_before = window.mcd_peak_selector_combo.currentData()
            axis_before = window.mcd_peak_spectrum_ax
            window._on_canvas_motion(self._plot_event(raw_ax, float(target_point.energy_ev), sum(ylim) / 2.0))
            self.assertEqual(window.mcd_peak_selector_combo.currentData(), selector_before)
            self.assertIs(window.mcd_peak_spectrum_ax, axis_before)
            self.assertAlmostEqual(float(window._mcd_peak_manual_center_ev), float(target_point.energy_ev), places=10)
            np.testing.assert_array_equal(fake.energy_ev, original_energy)
            np.testing.assert_array_equal(fake.pair_raw_pos, original_pos)
            np.testing.assert_array_equal(fake.pair_raw_neg, original_neg)
            window._on_canvas_release(SimpleNamespace(button=1, x=None, y=None, xdata=None, ydata=None, inaxes=None, key=None))
            self.assertIsNotNone(window.mcd_peak_selector_combo.currentData())
            self.assertAlmostEqual(float(window._mcd_peak_manual_center_ev), float(target_point.energy_ev), places=10)
        finally:
            window.close()

    def test_peak_drag_is_ignored_while_toolbar_pan_is_active(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.0, 0.0, 1.0], n_peaks=2)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            window.mcd_peak_result_mode_combo.setCurrentText("Single peak shift")
            window.canvas.draw()
            result_ax = window.mcd_peak_shift_ax
            index = int(window.mcd_peak_field_combo.currentData())
            ylim = result_ax.get_ylim()
            field_before = window.mcd_peak_field_combo.currentData()
            selector_before = window.mcd_peak_selector_combo.currentData()
            window.toolbar.pan()
            try:
                window._on_canvas_click(self._plot_event(result_ax, float(fake.pair_b[index]), sum(ylim) / 2.0))
                self.assertIsNone(window._mcd_peak_drag)
                self.assertEqual(window.mcd_peak_field_combo.currentData(), field_before)
                self.assertEqual(window.mcd_peak_selector_combo.currentData(), selector_before)
            finally:
                window.toolbar.pan()
        finally:
            window.close()

    def test_peak_candidate_bar_and_markers_offer_nearby_features(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.0, 0.0, 1.0], n_peaks=3)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            peak_tab = next(index for index in range(window.tabs.count()) if window.tabs.tabText(index) == "MCD Peak Shift")
            window.tabs.setCurrentIndex(peak_tab)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            window.canvas.draw()
            visible = [button for button in window.mcd_peak_candidate_buttons if not button.isHidden()]
            self.assertGreaterEqual(len(visible), 3)
            self.assertFalse(window.mcd_peak_candidate_bar.isHidden())
            selected = window.mcd_peak_selector_combo.currentData()
            keys = list(window._mcd_peak_candidate_keys)
            self.assertTrue(all(key[0] == selected[0] and key[2] == selected[2] for key in keys))
            target_index = next(index for index, key in enumerate(keys) if key != tuple(selected))
            visible[target_index].click()
            self.assertEqual(tuple(window.mcd_peak_selector_combo.currentData()), keys[target_index])
            self.assertFalse(window._mcd_peak_drag)
            window.canvas.draw()
            marker = next(artist for artist in window._mcd_peak_candidate_artists.values() if tuple(artist._mcd_peak_candidate_key) == tuple(selected))
            bbox = marker.get_window_extent(renderer=window.canvas.get_renderer())
            window._on_canvas_click(SimpleNamespace(
                button=1,
                x=float(bbox.x0 + bbox.width / 2.0),
                y=float(bbox.y0 + bbox.height / 2.0),
                xdata=None,
                ydata=None,
                inaxes=window.mcd_peak_spectrum_ax,
                key=None,
            ))
            self.assertEqual(tuple(window.mcd_peak_selector_combo.currentData()), tuple(selected))
            before_index = window.mcd_peak_selector_combo.currentIndex()
            window.mcd_peak_next_btn.click()
            self.assertEqual(window.mcd_peak_selector_combo.currentIndex(), (before_index + 1) % window.mcd_peak_selector_combo.count())
        finally:
            window.close()

    def test_two_resonance_branch_click_preserves_second_e0_and_field_navigation(self):
        from types import SimpleNamespace
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.0, 0.0, 1.0], n_peaks=2)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            window.mcd_peak_branch_combo.setCurrentText("B increasing")
            self.assertGreaterEqual(window.mcd_peak_selector_combo.count(), 2)
            second_index = next(
                index for index in range(window.mcd_peak_selector_combo.count())
                if (lambda data: data is not None and next(
                    track for track in window.mcd_peak_channel_results[data[0]].tracks
                    if track.peak_id == data[1] and track.branch == data[2] and track.feature_kind == data[3]
                ).reference_energy_ev > 1.8)(window.mcd_peak_selector_combo.itemData(index))
            )
            window.mcd_peak_selector_combo.setCurrentIndex(second_index)
            before = window.mcd_peak_selector_combo.currentData()
            before_track = next(track for track in window.mcd_peak_channel_results[before[0]].tracks if track.peak_id == before[1] and track.branch == before[2] and track.feature_kind == before[3])
            before_e0 = float(before_track.reference_energy_ev)
            window.mcd_peak_branch_combo.setCurrentText("B decreasing")
            after = window.mcd_peak_selector_combo.currentData()
            after_track = next(track for track in window.mcd_peak_channel_results[after[0]].tracks if track.peak_id == after[1] and track.branch == after[2] and track.feature_kind == after[3])
            self.assertEqual(after[0], before[0])
            self.assertEqual(after[3], before[3])
            self.assertAlmostEqual(float(after_track.reference_energy_ev), before_e0, places=3)

            window.mcd_peak_result_mode_combo.setCurrentText("Single peak shift")
            curve_lines = [line for line in window.mcd_peak_shift_ax.lines if getattr(line, "_mcd_peak_method", None) in {"Raw spectrum", "Second derivative"}]
            self.assertEqual({(line._mcd_peak_method, line._mcd_peak_branch) for line in curve_lines}, {
                ("Raw spectrum", "B increasing"), ("Raw spectrum", "B decreasing"),
                ("Second derivative", "B increasing"), ("Second derivative", "B decreasing"),
            })
            self.assertTrue(any(line.get_label() == "_mcd_selected_result_marker" and getattr(line, "_mcd_ignore_click", False) for line in window.mcd_peak_shift_ax.lines))
            decreasing = next(line for line in curve_lines if line._mcd_peak_method == "Second derivative" and line._mcd_peak_branch == "B decreasing")
            xdata = np.asarray(decreasing.get_xdata(), float)
            ydata = np.asarray(decreasing.get_ydata(), float)
            point = int(np.flatnonzero(np.isfinite(xdata) & np.isfinite(ydata))[-1])
            window._on_mcd_peak_result_click(SimpleNamespace(xdata=float(xdata[point]), ydata=float(ydata[point]), inaxes=window.mcd_peak_shift_ax))
            self.assertEqual(window.mcd_peak_branch_combo.currentText(), "B decreasing")
            after_click = window.mcd_peak_selector_combo.currentData()
            clicked_track = next(track for track in window.mcd_peak_channel_results[after_click[0]].tracks if track.peak_id == after_click[1] and track.branch == after_click[2] and track.feature_kind == after_click[3])
            self.assertAlmostEqual(float(clicked_track.reference_energy_ev), before_e0, places=3)
            old_field = window.mcd_peak_field_combo.currentData()
            window.mcd_peak_field_next_btn.click()
            self.assertNotEqual(window.mcd_peak_field_combo.currentData(), old_field)
        finally:
            window.close()

    def test_result_highlight_does_not_jump_across_invalid_nearest_sample(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.0, 0.0, 1.0], n_peaks=2)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            window.mcd_peak_branch_combo.setCurrentText("B decreasing")
            window.mcd_peak_result_mode_combo.setCurrentText("Single peak shift")
            selected = window.mcd_peak_selector_combo.currentData()
            method = "Second derivative"
            analysis = window.mcd_peak_method_results[method][selected[0]]
            tracks = []
            for track in analysis.tracks:
                if track.peak_id != selected[1] or track.branch != selected[2] or track.feature_kind != selected[3]:
                    tracks.append(track)
                    continue
                points = tuple(
                    replace(point, energy_ev=None, delta_energy_ev=None, status="missing")
                    if point.branch == "B decreasing" and abs(float(point.field_t)) < 1e-12 else point
                    for point in track.points
                )
                tracks.append(replace(track, points=points))
            window.mcd_peak_method_results[method][selected[0]] = replace(analysis, tracks=tuple(tracks))
            field_index = window.mcd_peak_field_combo.findData(2)
            window.mcd_peak_field_combo.setCurrentIndex(field_index)
            window._refresh_mcd_peak_plot()
            self.assertFalse(any(line.get_label() == "_mcd_selected_result_marker" for line in window.mcd_peak_shift_ax.lines))
            self.assertTrue(any(np.allclose(np.asarray(line.get_xdata(), float), [0.0, 0.0]) for line in window.mcd_peak_shift_ax.lines))
        finally:
            window.close()

    def test_boundary_quality_is_hidden_from_selected_peak_and_map(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.5, 1.0, -0.5], n_peaks=3)
            edge = np.exp(-((fake.energy_ev - 1.505) / 0.012) ** 2) + 0.1
            fake.pair_corrected_pos[1] = edge
            fake.pair_corrected_neg[1] = edge
            fake.pair_raw_pos[1] = edge
            fake.pair_raw_neg[1] = edge
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            window.mcd_peak_branch_combo.setCurrentText("B increasing")
            labels = [window.mcd_peak_selector_combo.itemText(i) for i in range(window.mcd_peak_selector_combo.count())]
            self.assertFalse(any("Boundary unreliable" in label for label in labels))
            self.assertNotIn("Boundary unreliable", window.mcd_peak_status.text())
            self.assertTrue(all(line.get_visible() for line in window._mcd_peak_track_lines))
        finally:
            window.close()

    def test_all_boundary_tracks_leave_no_table_rows_or_export(self):
        from ui_qt.main_window import LoadedState, MainWindow
        window = MainWindow()
        try:
            fake = _result([-1.0, 0.5, 1.0, -0.5], n_peaks=3)
            edge = np.exp(-((fake.energy_ev - 1.505) / 0.012) ** 2) + 0.1
            for index in (1, 3):
                fake.pair_corrected_pos[index] = edge
                fake.pair_corrected_neg[index] = edge
                fake.pair_raw_pos[index] = edge
                fake.pair_raw_neg[index] = edge
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=fake)
            window._update_mcd_peak_shift_source(fake)
            window.mcd_peak_tracker_method_combo.setCurrentText("Second derivative")
            window.mcd_peak_analyze_btn.click()
            self.assertEqual(window.mcd_peak_selector_combo.count(), 0)
            self.assertEqual(window.mcd_peak_table.rowCount(), 0)
            self.assertFalse(window.mcd_peak_export_btn.isEnabled())
        finally:
            window.close()
