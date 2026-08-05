from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.mcd import McdSettings, background_fit_regions, export_mcd_analysis_bundle, export_mcd_tables, pair_window_trace_by_branch, process_mcd, suggest_mcd_background_ranges, window_trace, window_trace_comparison
from ui_qt.main_window import LoadOptions, LoadedState, MainWindow, QDoubleSpinBox, QSpinBox


class McdProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _write_sweep(self, folder: Path) -> Path:
        # The two angles have a 3x, wavelength-dependent transmission mismatch.
        # Their normalised field response is opposite, so the corrected MCD must
        # be nonzero while its zero-field row remains zero.
        rows = []
        for b in (-1.0, 0.0, 1.0):
            for angle, scale, sign in ((10.0, 1.0, 1.0), (50.0, 3.0, -1.0)):
                rows.append({"B_T": b, "angle_deg": angle, "700": scale * (100 + sign * 4 * b), "710": scale * (60 + sign * 2 * b)})
        path = folder / "sweep.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_mcd_processing_defaults_use_quadratic_spectral_baseline(self) -> None:
        settings = McdSettings()
        self.assertEqual(settings.correction_mode, "pair_spectral")
        self.assertEqual(settings.spectral_order, 2)

    def _write_branch_sweep(self, folder: Path, fields: list[float]) -> Path:
        rows = []
        for b in fields:
            for angle, scale, sign in ((10.0, 1.0, 1.0), (50.0, 3.0, -1.0)):
                rows.append({"B_T": b, "angle_deg": angle, "700": scale * (100 + sign * 4 * b), "710": scale * (60 + sign * 2 * b)})
        path = folder / "branch_sweep.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_spectral_drift_sweep(self, folder: Path, *, quadratic: bool = False, features: bool = False) -> Path:
        wavelength = np.linspace(680.0, 800.0, 81)
        energy = 1239.841984 / wavelength
        center = float(np.median(energy))
        baseline = 25000.0 + 12000.0 * (energy - np.min(energy))
        if features:
            # Three narrow reflection features should be excluded separately,
            # even when the active MCD(B) window protects only the middle one.
            for center, amplitude, width in ((1.570, -0.42, 0.004), (1.635, 0.48, 0.005), (1.708, -0.50, 0.005)):
                baseline *= 1.0 + amplitude * np.exp(-0.5 * ((energy - center) / width) ** 2)
        rows = []
        for field in (-1.0, 0.0, 1.0):
            x = energy - center
            log_drift = field * (0.55 * x + (1.8 * x**2 if quadratic else 0.0))
            for angle, values in ((10.0, baseline * np.exp(log_drift)), (50.0, baseline)):
                row = {"B_T": field, "angle_deg": angle}
                row.update({f"{value:g}": intensity for value, intensity in zip(wavelength, values)})
                rows.append(row)
        path = folder / ("quadratic_spectral_drift.csv" if quadratic else "linear_spectral_drift.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_rising_feature_sweep(
        self,
        folder: Path,
        *,
        points: int = 293,
        include_features: bool = True,
        ripple: bool = False,
    ) -> Path:
        """A realistic sloped reflection trace used to guard auto-protection."""
        energy = np.linspace(1.52, 1.812, points)
        wavelength = 1239.841984 / energy
        reflection = 12000.0 + 48000.0 * (energy - energy.min()) / (energy.max() - energy.min())
        if include_features:
            for center, depth, width in ((1.667, 0.10, 0.006), (1.745, 0.08, 0.006)):
                reflection *= 1.0 - depth * np.exp(-0.5 * ((energy - center) / width) ** 2)
        if ripple:
            # Correlated detector variation that must remain review-only or be
            # ignored; it should not become a third protected resonance.
            reflection *= 1.0 + 0.003 * np.sin(2.0 * np.pi * (energy - energy.min()) / 0.019)
        rows = []
        for field in (-1.0, 0.0, 1.0):
            drift = np.exp(field * 0.15 * (energy - np.median(energy)))
            for angle, values in ((10.0, reflection * drift), (50.0, reflection)):
                row = {"B_T": field, "angle_deg": angle}
                row.update({f"{value:g}": intensity for value, intensity in zip(wavelength, values)})
                rows.append(row)
        path = folder / f"rising_features_{points}_{include_features}_{ripple}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_per_wavelength_reference_removes_angle_throughput(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            result = process_mcd(str(self._write_sweep(Path(folder_text))), McdSettings(zero_window_t=0.05, max_sequence_gap=1, max_delta_b=0.01))
        combo = result.cube("Combo")
        zero_index = int(np.argmin(np.abs(combo.gate)))
        self.assertTrue(np.allclose(combo.Z[zero_index], 0.0, atol=1e-12))
        self.assertTrue(np.all(np.abs(combo.Z[-1]) > 0.01))
        self.assertTrue(np.allclose(combo.Z[-1], -combo.Z[0], atol=1e-12))
        b, trace = window_trace(result, "Combo", 1.75, 25.0)
        self.assertEqual(b.size, trace.size)
        comparison = window_trace_comparison(result, "Combo", 1.75, 25.0)
        self.assertEqual(
            set(comparison),
            {
                "raw_mean", "raw_field_signed_absolute_mean", "raw_absolute_mean", "raw_integral",
                "corrected_mean", "corrected_field_signed_absolute_mean", "corrected_absolute_mean", "corrected_integral",
            },
        )
        self.assertTrue(all(field.size == values.size for field, values in comparison.values()))
        field, signed_magnitude = comparison["corrected_field_signed_absolute_mean"]
        self.assertTrue(np.all(signed_magnitude[field < 0] <= 0.0))
        self.assertTrue(np.all(signed_magnitude[field > 0] >= 0.0))

    def test_window_comparison_export_contains_all_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            result = process_mcd(str(self._write_sweep(folder)), McdSettings(max_sequence_gap=1, max_delta_b=0.01))
            paths = export_mcd_tables(result, str(folder), trace_map="Combo", center_ev=1.75, width_mev=25.0, metric="absolute_mean")
            table = pd.read_csv(paths["window_comparison"])
        self.assertTrue({"B_T", "raw_mean", "raw_field_signed_absolute_mean", "raw_absolute_mean", "raw_integral", "corrected_mean", "corrected_field_signed_absolute_mean", "corrected_absolute_mean", "corrected_integral"}.issubset(table.columns))

    def test_compact_analysis_export_omits_intermediate_maps(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            result = process_mcd(
                str(self._write_branch_sweep(folder, [-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0])),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            export_settings = McdSettings(
                max_sequence_gap=1, max_delta_b=0.01,
                correction_mode="pair_spectral", spectral_order=2,
                background_ranges_ev=((1.5, 1.6), (1.7, 1.8)),
                background_selection="suggested",
                suggestion_protected_ranges_ev=((1.62, 1.68),),
                manual_protected_ranges_ev=((1.66, 1.67), (1.74, 1.75)),
                suggestion_linear_validation_rms=0.004,
                suggestion_quadratic_validation_rms=0.002,
                suggestion_algorithm="full_sweep_feature_review_v1",
            )
            paths = export_mcd_analysis_bundle(
                result,
                str(folder),
                trace_map="Combo",
                center_ev=1.75,
                width_mev=25.0,
                metric="mean",
                settings=export_settings,
            )
            names = {path.name for path in paths.values()}
            self.assertEqual(
                names,
                {
                    "branch_sweep_MCD_vs_B_E1.750000eV_W25meV.png",
                    "branch_sweep_MCD_vs_B_E1.750000eV_W25meV.csv",
                    "branch_sweep_MCD_pair_diagnostics.csv",
                    "branch_sweep_MCD_settings.json",
                },
            )
            self.assertTrue(all(path.exists() for path in paths.values()))
            self.assertFalse(any("_MCD_Raw" in path.name or "_MCD_Normalized" in path.name for path in folder.iterdir()))
            table = pd.read_csv(paths["mcd_vs_b_csv"])
            self.assertEqual(
                list(table.columns),
                [
                    "B_increasing_T", "corrected_signed_mean_increasing",
                    "corrected_field_signed_absolute_mean_increasing", "corrected_integral_increasing",
                    "B_decreasing_T", "corrected_signed_mean_decreasing",
                    "corrected_field_signed_absolute_mean_decreasing", "corrected_integral_decreasing",
                ],
            )
            self.assertEqual(int(table["B_increasing_T"].notna().sum()), 5)
            self.assertEqual(int(table["B_decreasing_T"].notna().sum()), 4)
            diagnostics = pd.read_csv(paths["pair_diagnostics"])
            self.assertTrue({
                "spectral_log_slope_per_eV", "spectral_log_curvature_per_eV2",
                "spectral_correction_min", "spectral_correction_max",
                "background_relative_rms_before", "background_relative_rms",
            }.issubset(diagnostics.columns))
            payload = json.loads(paths["settings"].read_text(encoding="utf-8"))
            self.assertEqual(payload["processing"]["background_selection"], "suggested")
            self.assertEqual(payload["processing"]["suggestion_algorithm"], "full_sweep_feature_review_v1")
            self.assertEqual(payload["processing"]["manual_protected_ranges_ev"], [[1.66, 1.67], [1.74, 1.75]])
            self.assertEqual(payload["mcd_b"]["primary_metric"], "mean")
            self.assertTrue(payload["mcd_b"]["show_signed_mean"])
            self.assertFalse(payload["mcd_b"]["show_field_signed_absolute_mean"])
            self.assertFalse(payload["mcd_b"]["show_integral"])

    def test_pair_branches_follow_the_actual_sweep_direction(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            increasing_first = process_mcd(
                str(self._write_branch_sweep(folder, [-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0])),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            decreasing_first = process_mcd(
                str(self._write_branch_sweep(folder, [1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 1.0])),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            self.assertTrue(np.all(increasing_first.pair_labels[:5] == "B increasing"))
            self.assertTrue(np.all(increasing_first.pair_labels[5:] == "B decreasing"))
            self.assertTrue(np.all(decreasing_first.pair_labels[:5] == "B decreasing"))
            self.assertTrue(np.all(decreasing_first.pair_labels[5:] == "B increasing"))
            branches = pair_window_trace_by_branch(increasing_first, 1.75, 25.0)
            for name in ("raw_mean", "corrected_mean", "corrected_field_signed_absolute_mean", "corrected_integral"):
                self.assertEqual(branches["B increasing"][name][0].size, 5)
                self.assertEqual(branches["B decreasing"][name][0].size, 4)

    def test_auto_drift_fit_uses_both_outer_spectrum_ends(self) -> None:
        regions = background_fit_regions(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]), ())
        self.assertEqual(len(regions), 2)
        self.assertAlmostEqual(regions[0][0], 1.0)
        self.assertAlmostEqual(regions[1][1], 5.0)
        self.assertLess(regions[0][1], regions[1][0])

    def test_linear_spectral_correction_removes_wavelength_dependent_pair_drift(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            path = self._write_spectral_drift_sweep(Path(folder_text))
            scale_only = process_mcd(
                str(path),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01, correction_mode="pair_scale"),
            )
            spectral = process_mcd(
                str(path),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01, correction_mode="pair_spectral", spectral_order=1),
            )
        self.assertGreater(np.nanmean(np.abs(scale_only.pair_mcd_corrected)), 1e-3)
        self.assertLess(np.nanmax(np.abs(spectral.pair_mcd_corrected)), 1e-9)
        self.assertTrue(np.all(spectral.pair_background_rms <= spectral.pair_background_rms_before + 1e-12))
        self.assertAlmostEqual(float(spectral.pair_spectral_slope[-1]), -0.55, places=6)

    def test_quadratic_spectral_correction_is_explicit_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            path = self._write_spectral_drift_sweep(Path(folder_text), quadratic=True)
            linear = process_mcd(
                str(path),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01, correction_mode="pair_spectral", spectral_order=1),
            )
            quadratic = process_mcd(
                str(path),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01, correction_mode="pair_spectral", spectral_order=2),
            )
        self.assertGreater(np.nanmean(np.abs(linear.pair_mcd_corrected)), 1e-5)
        self.assertLess(np.nanmax(np.abs(quadratic.pair_mcd_corrected)), 1e-9)
        self.assertAlmostEqual(float(quadratic.pair_spectral_curvature[-1]), -1.8, places=6)
        self.assertEqual(quadratic.summary["spectral_order"], 2)

    def test_full_sweep_background_suggestion_protects_feature_and_validates_models(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            result = process_mcd(
                str(self._write_spectral_drift_sweep(Path(folder_text), quadratic=True)),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            suggestion = suggest_mcd_background_ranges(
                result,
                protected_ranges_ev=((1.65, 1.70),),
                min_band_width_mev=5.0,
            )
        self.assertGreaterEqual(len(suggestion.ranges), 2)
        self.assertGreater(suggestion.coverage_fraction, 0.1)
        self.assertGreater(suggestion.span_fraction, 0.25)
        self.assertTrue(all(stop < 1.65 or start > 1.70 for start, stop in suggestion.ranges))
        self.assertTrue(np.isfinite(suggestion.linear_validation_rms))
        self.assertTrue(np.isfinite(suggestion.quadratic_validation_rms))
        self.assertEqual(suggestion.suggested_order, 2)

    def test_full_sweep_background_suggestion_detects_multiple_reflection_features(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            result = process_mcd(
                str(self._write_spectral_drift_sweep(Path(folder_text), features=True)),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            suggestion = suggest_mcd_background_ranges(result, protected_ranges_ev=((1.625, 1.645),))
        self.assertTrue(any(start <= 1.570 <= stop for start, stop in suggestion.detected_feature_ranges))
        self.assertTrue(any(start <= 1.708 <= stop for start, stop in suggestion.detected_feature_ranges))
        self.assertTrue(any(start <= 1.635 <= stop for start, stop in suggestion.protected_ranges))
        self.assertIn("peak", suggestion.detected_feature_kinds)
        self.assertIn("dip", suggestion.detected_feature_kinds)
        self.assertTrue(all(
            not (start <= feature <= stop)
            for start, stop in suggestion.ranges
            for feature in (1.570, 1.635, 1.708)
        ))

    def test_feature_detector_isolates_two_dips_on_a_rising_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            result = process_mcd(
                str(self._write_rising_feature_sweep(Path(folder_text))),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            suggestion = suggest_mcd_background_ranges(result)
        recommended = [feature for feature in suggestion.detected_features if feature.recommended]
        self.assertEqual(len(recommended), 2)
        self.assertTrue(all(feature.kind == "dip" for feature in recommended))
        self.assertTrue(any(abs(feature.center_ev - 1.667) <= 0.003 for feature in recommended))
        self.assertTrue(any(abs(feature.center_ev - 1.745) <= 0.003 for feature in recommended))
        self.assertTrue(all(0.006 <= feature.width_ev <= 0.015 for feature in recommended))
        self.assertTrue(all(feature.start_ev <= feature.center_ev <= feature.stop_ev for feature in recommended))
        for feature in suggestion.detected_features:
            is_protected = any(start <= feature.center_ev <= stop for start, stop in suggestion.protected_ranges)
            self.assertEqual(is_protected, feature.recommended)

    def test_feature_detector_does_not_auto_protect_correlated_ripple(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            result = process_mcd(
                str(self._write_rising_feature_sweep(Path(folder_text), ripple=True)),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            suggestion = suggest_mcd_background_ranges(result)
        recommended = [feature for feature in suggestion.detected_features if feature.recommended]
        self.assertEqual(len(recommended), 2)
        self.assertTrue(all(min(abs(feature.center_ev - 1.667), abs(feature.center_ev - 1.745)) <= 0.003 for feature in recommended))

    def test_feature_detector_ignores_featureless_rising_trace(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            result = process_mcd(
                str(self._write_rising_feature_sweep(Path(folder_text), include_features=False, ripple=True)),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            suggestion = suggest_mcd_background_ranges(result)
        self.assertFalse(any(feature.recommended for feature in suggestion.detected_features))

    def test_feature_centers_are_stable_across_sampling_density(self) -> None:
        centers_by_density: list[list[float]] = []
        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            for points in (181, 401):
                result = process_mcd(
                    str(self._write_rising_feature_sweep(folder, points=points)),
                    McdSettings(max_sequence_gap=1, max_delta_b=0.01),
                )
                suggestion = suggest_mcd_background_ranges(result)
                centers_by_density.append(sorted(feature.center_ev for feature in suggestion.detected_features if feature.recommended))
        self.assertEqual([len(centers) for centers in centers_by_density], [2, 2])
        for low_density, high_density in zip(*centers_by_density):
            self.assertAlmostEqual(low_density, high_density, delta=0.003)

    def test_manual_protection_uses_left_middle_and_right_background_bands(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            result = process_mcd(
                str(self._write_rising_feature_sweep(Path(folder_text))),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            suggestion = suggest_mcd_background_ranges(
                result,
                protected_ranges_ev=((1.643, 1.675), (1.737, 1.753)),
                auto_detect_features=False,
                use_all_unprotected_bands=True,
            )
        self.assertEqual(len(suggestion.ranges), 3)
        left, middle, right = suggestion.ranges
        self.assertAlmostEqual(left[0], float(suggestion.energy_ev[0]), delta=0.002)
        self.assertLess(left[1], 1.643)
        self.assertGreater(middle[0], 1.675)
        self.assertLess(middle[1], 1.737)
        self.assertGreater(right[0], 1.753)
        self.assertAlmostEqual(right[1], float(suggestion.energy_ev[-1]), delta=0.002)

    def test_manual_protection_discards_only_subminimum_internal_gap(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            result = process_mcd(
                str(self._write_rising_feature_sweep(Path(folder_text))),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            suggestion = suggest_mcd_background_ranges(
                result,
                protected_ranges_ev=((1.643, 1.700), (1.703, 1.753)),
                min_band_width_mev=5.0,
                auto_detect_features=False,
                use_all_unprotected_bands=True,
            )
        self.assertEqual(len(suggestion.ranges), 2)
        self.assertTrue(all(stop <= 1.643 or start >= 1.753 for start, stop in suggestion.ranges))

    def test_overlapping_manual_protections_merge_before_background_split(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            result = process_mcd(
                str(self._write_rising_feature_sweep(Path(folder_text))),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            suggestion = suggest_mcd_background_ranges(
                result,
                protected_ranges_ev=((1.640, 1.670), (1.660, 1.682), (1.731, 1.763)),
                auto_detect_features=False,
                use_all_unprotected_bands=True,
            )
        self.assertEqual(suggestion.protected_ranges, ((1.64, 1.682), (1.731, 1.763)))
        self.assertEqual(len(suggestion.ranges), 3)
        self.assertTrue(any(start > 1.682 and stop < 1.731 for start, stop in suggestion.ranges))

    def test_mcd_changes_schedule_automatic_recalculation(self) -> None:
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="MCD", folder="")
            window._on_mcd_params_changed()
            self.assertTrue(window._mcd_auto_apply_timer.isActive())
            self.assertIn("Pending", window.mcd_apply_correction_btn.text())
        finally:
            window._mcd_auto_apply_timer.stop()
            window.close()

    def test_spin_boxes_ignore_plain_mouse_wheel_events(self) -> None:
        class PlainWheelEvent:
            ignored = False

            @staticmethod
            def modifiers():
                return Qt.KeyboardModifier.NoModifier

            def ignore(self) -> None:
                self.ignored = True

        for spin in (QDoubleSpinBox(), QSpinBox()):
            spin.setValue(4)
            event = PlainWheelEvent()
            spin.wheelEvent(event)
            self.assertTrue(event.ignored)
            self.assertEqual(spin.value(), 4)

    def test_missing_zero_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            path = self._write_sweep(Path(folder_text))
            frame = pd.read_csv(path)
            frame = frame[frame["B_T"] != 0.0]
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "near-zero reference"):
                process_mcd(
                    str(path),
                    McdSettings(reference_mode="window", zero_window_t=0.001, max_sequence_gap=1, max_delta_b=0.01),
                )

    def test_nearest_reference_accepts_data_without_exact_zero(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            path = self._write_sweep(Path(folder_text))
            frame = pd.read_csv(path)
            frame = frame[frame["B_T"] != 0.0]
            frame.to_csv(path, index=False)
            result = process_mcd(str(path), McdSettings(max_sequence_gap=1, max_delta_b=0.01))
        self.assertAlmostEqual(result.reference_b, -1.0)

    def test_same_b_alignment_interpolates_inside_one_sweep_branch(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            path = Path(folder_text) / "offset_pairs.csv"
            rows = []
            for field, angle in ((0.0, 50.0), (0.0, 10.0)):
                rows.append({"B_T": field, "angle_deg": angle, "700": 100.0 + 10.0 * field, "710": 80.0 + 5.0 * field})
            for index in range(1, 5):
                for field, angle in ((0.2 * index, 50.0), (0.2 * index + 0.1, 10.0)):
                    rows.append({"B_T": field, "angle_deg": angle, "700": 100.0 + 10.0 * field, "710": 80.0 + 5.0 * field})
            pd.DataFrame(rows).to_csv(path, index=False)
            direct = process_mcd(str(path), McdSettings(max_sequence_gap=1, max_delta_b=0.11))
            aligned = process_mcd(str(path), McdSettings(max_sequence_gap=1, max_delta_b=0.11, pair_b_alignment="interpolate"))
        self.assertTrue(np.any(aligned.pair_interpolated_pos[1:-1]))
        self.assertTrue(np.any(aligned.pair_interpolated_neg[1:-1]))
        self.assertLess(np.nanmean(np.abs(aligned.pair_mcd_corrected[1:-1])), np.nanmean(np.abs(direct.pair_mcd_corrected[1:-1])))

    def test_loaded_mcd_renders_its_embedded_plots(self) -> None:
        class Sink:
            def emit(self, *_args) -> None:
                pass

        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            self._write_sweep(folder)
            window = MainWindow()
            try:
                window._set_current_folder(str(folder))
                self.assertEqual(window.mcd_files.count(), 1)
                self.assertEqual(window.mcd_files.item(0).text(), "sweep.csv")
                options = LoadOptions(
                    mode="MCD", folder=str(folder), selected_files=["sweep.csv"], baseline_files=[],
                    pl_log_scale=False, drr_baseline_text="", drr_baseline_which="", compare_log_scale=False,
                    mcd_settings=McdSettings(max_sequence_gap=1, max_delta_b=0.01),
                )
                window._on_loaded(window._load_task(options, progress=Sink(), log=Sink()))
                self.assertEqual(window.last_plotted_mode, "MCD")
                self.assertGreaterEqual(len(window.figure.axes), 4)
                window.canvas.draw()
                heat_bounds = window._mcd_heatmap_ax.get_position()
                spectrum_bounds = window._mcd_spectrum_ax.get_position()
                trace_bounds = window._mcd_trace_ax.get_position()
                colorbar_bounds = window._mcd_colorbar_ax.get_position()
                self.assertGreater(heat_bounds.y0, spectrum_bounds.y1)
                self.assertLess(heat_bounds.x1, trace_bounds.x0)
                self.assertGreater(colorbar_bounds.x0, heat_bounds.x0)
                self.assertLess(colorbar_bounds.x1, heat_bounds.x1)
                self.assertGreater(colorbar_bounds.y0, heat_bounds.y1)
                self.assertLess(colorbar_bounds.y0 - heat_bounds.y1, 0.05)
                self.assertIn("E =", window._mcd_trace_ax.get_title())
                metric_legend = window._mcd_trace_ax.get_legend()
                self.assertIsNotNone(metric_legend)
                self.assertEqual([text.get_text() for text in metric_legend.get_texts()], ["Signed mean"])
                figure_bounds = window.figure.bbox
                for axis in (window._mcd_trace_ax, window._mcd_integral_ax):
                    for text in [axis.yaxis.label, *axis.get_yticklabels()]:
                        bounds = text.get_window_extent(window.canvas.get_renderer())
                        self.assertGreaterEqual(bounds.x0, figure_bounds.x0)
                        self.assertLessEqual(bounds.x1, figure_bounds.x1)
            finally:
                window.close()

    def test_mcd_sidebar_fits_without_horizontal_clipping(self) -> None:
        """The compact MCD controls must fit the normal left sidebar width."""
        window = MainWindow()
        try:
            window.resize(1500, 900)
            window.show()
            self.app.processEvents()
            scroll = window.mcd_tab_scroll
            self.assertLessEqual(
                scroll.widget().minimumSizeHint().width(),
                scroll.viewport().width(),
            )
            for key in ("xmin", "xmax", "ymin", "ymax"):
                self.assertGreater(window.mcd_spins[key].width(), 0)
            self.assertNotIn("gate", window.mcd_spins)
            self.assertEqual(window.mcd_correction_mode_combo.currentText(), "Global gain + per-pair spectral baseline")
            self.assertTrue(window.mcd_spectral_order_combo.isEnabled())
            self.assertEqual(window.mcd_spectral_order_combo.currentText(), "Quadratic (default)")
            self.assertEqual(window.mcd_window_metric_combo.currentText(), "Signed mean")
            self.assertTrue(window.mcd_show_signed_mean_chk.isChecked())
            self.assertFalse(window.mcd_show_absolute_mean_chk.isChecked())
            self.assertFalse(window.mcd_show_unsigned_absolute_mean_chk.isChecked())
            self.assertFalse(window.mcd_show_integral_chk.isChecked())
        finally:
            window.close()

    def test_mcd_reprocess_keeps_the_selected_pair_linecut(self) -> None:
        class Sink:
            def emit(self, *_args) -> None:
                pass

        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            self._write_sweep(folder)
            options = LoadOptions(
                mode="MCD", folder=str(folder), selected_files=["sweep.csv"], baseline_files=[],
                pl_log_scale=False, drr_baseline_text="", drr_baseline_which="", compare_log_scale=False,
                mcd_settings=McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            window = MainWindow()
            try:
                first = window._load_task(options, progress=Sink(), log=Sink())
                window._on_loaded(first)
                selected_index = 2
                window.mcd_pair_b_combo.setCurrentIndex(selected_index)
                window._mcd_pair_selection_to_restore = (
                    float(first.mcd_result.pair_b[selected_index]),
                    str(first.mcd_result.pair_labels[selected_index]),
                )
                window._on_loaded(window._load_task(options, progress=Sink(), log=Sink()))
                restored_index = int(window.mcd_pair_b_combo.currentData())
                self.assertAlmostEqual(
                    float(window.loaded.mcd_result.pair_b[restored_index]),
                    float(first.mcd_result.pair_b[selected_index]),
                )
            finally:
                window.close()

    def test_mcd_heatmap_click_selects_the_nearest_pair(self) -> None:
        class Sink:
            def emit(self, *_args) -> None:
                pass

        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            self._write_sweep(folder)
            options = LoadOptions(
                mode="MCD", folder=str(folder), selected_files=["sweep.csv"], baseline_files=[],
                pl_log_scale=False, drr_baseline_text="", drr_baseline_which="", compare_log_scale=False,
                mcd_settings=McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            window = MainWindow()
            try:
                window._on_loaded(window._load_task(options, progress=Sink(), log=Sink()))
                result = window.loaded.mcd_result
                event = type("McdClick", (), {
                    "button": 1,
                    "inaxes": window._mcd_heatmap_ax,
                    "xdata": 1.75,
                    "ydata": 0.7,
                })()
                window._on_canvas_click(event)
                expected_index = int(np.argmin(np.abs(result.pair_b - 0.7)))
                self.assertEqual(int(window.mcd_pair_b_combo.currentData()), expected_index)
                self.assertIn(f"B = {result.pair_b[expected_index]:.5g} T", window._mcd_spectrum_ax.get_title())
                window.mcd_pair_b_combo.setCurrentIndex(0)
                self.assertIn(f"B = {result.pair_b[0]:.5g} T", window._mcd_spectrum_ax.get_title())
            finally:
                window.close()
