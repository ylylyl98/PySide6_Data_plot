from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.mcd import McdCenterCandidate, McdSettings, background_fit_regions, detect_angles, discover_mcd_processing_status, ensure_mcd_package_dir, export_mcd_analysis_bundle, export_mcd_tables, extract_mcd_acquisition_conditions, format_mcd_acquisition_conditions, format_mcd_energy, load_b_sweep_csv, low_field_mcd_branch_fits, pair_window_trace_by_branch, process_mcd, suggest_mcd_background_ranges, suggest_mcd_window_centers, window_trace, window_trace_comparison
from ui_qt.main_window import LoadOptions, LoadedState, MainWindow, QDoubleSpinBox, QSpinBox
from tests.ui_test_helpers import wait_for_file_catalog


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

    def test_current_acquisition_format_uses_mid_field_and_rotation_angle(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            path = Path(folder_text) / "current_format.csv"
            rows = []
            for index, (field, angle) in enumerate(
                ((-1.0, 29.999), (-1.0, 75.001), (0.0, 30.0), (0.0, 75.0))
            ):
                rows.append(
                    {
                        "timestamp_start_utc": f"2026-08-24T21:26:{index:02d}+00:00",
                        "timestamp_end_utc": f"2026-08-24T21:26:{index + 1:02d}+00:00",
                        "leg": "forward",
                        "direction": "increasing",
                        "rotation_angle_deg": angle,
                        "B0_T": field - 0.01,
                        "B1_T": field + 0.01,
                        "Bmid_T": field,
                        "Vtg_V": 3.15,
                        "Vbg_V": 2.9,
                        "Vbias_V": 0.0,
                        "Doping_V": 6.3,
                        "Efield_V": 25.0,
                        "sample_T0_K": np.nan,
                        "700.0": 100.0 + index,
                        "710.0": 80.0 + index,
                    }
                )
            pd.DataFrame(rows).to_csv(path, index=False)

            field, angle, wavelength, spectra = load_b_sweep_csv(str(path))
            detected = detect_angles(str(path))

        np.testing.assert_allclose(field, [-1.0, -1.0, 0.0, 0.0])
        np.testing.assert_allclose(angle, [29.9995, 75.0005, 29.9995, 75.0005])
        np.testing.assert_allclose(detected, [29.9995, 75.0005])
        np.testing.assert_allclose(wavelength, [700.0, 710.0])
        self.assertEqual(spectra.shape, (4, 2))

    def test_acquisition_conditions_are_formatted_as_a_compact_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            path = Path(folder_text) / "conditions.csv"
            pd.DataFrame(
                {
                    "Vtg_V": [15.65, 15.65],
                    "Vbg_V": [-8.602, -8.602],
                    "Vbias_V": [0.0, 0.0],
                    "Doping_V": [6.3, 6.3],
                    "Efield_V": [24.5, 25.0],
                    "sample_Tmid_K": [np.nan, np.nan],
                }
            ).to_csv(path, index=False)
            conditions = extract_mcd_acquisition_conditions(str(path))

        subtitle = format_mcd_acquisition_conditions(conditions)
        self.assertIn("Vtg = +15.65 V", subtitle)
        self.assertIn("Vbg = -8.602 V", subtitle)
        self.assertIn("Doping = 6.3 V", subtitle)
        self.assertIn("E-field = 24.5 to 25 V", subtitle)
        self.assertNotIn("T =", subtitle)
        self.assertEqual(format_mcd_energy(1.640000), "1.64")
        self.assertEqual(format_mcd_energy(1.640125), "1.640125")

    def test_center_suggestions_rank_multiple_distinct_fixed_width_features(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            result = process_mcd(
                str(self._write_spectral_drift_sweep(Path(folder_text))),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
        energy = 1239.841984 / np.asarray(result.wavelength_nm, float)
        feature = (
            0.035 * np.exp(-0.5 * ((energy - 1.60) / 0.003) ** 2)
            + 0.020 * np.exp(-0.5 * ((energy - 1.72) / 0.004) ** 2)
        )
        result.pair_mcd_corrected = np.sign(result.pair_b)[:, None] * feature[None, :]
        candidates = suggest_mcd_window_centers(
            result, 5.0, metric="mean", energy_range=(1.55, 1.78), max_candidates=5
        )

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(
            [candidate.center_ev for candidate in candidates],
            sorted(candidate.center_ev for candidate in candidates),
        )
        self.assertAlmostEqual(candidates[0].center_ev, 1.60, delta=0.006)
        self.assertTrue(any(abs(candidate.center_ev - 1.72) <= 0.007 for candidate in candidates))
        for left, right in zip(candidates, candidates[1:]):
            self.assertNotEqual(left.center_ev, right.center_ev)

    def test_one_stable_processed_folder_is_used_for_each_raw_mcd_csv(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            parent = Path(folder_text) / "Processed Data" / "MCD"
            legacy_a = parent / "sample_roundtrip_MCD_E1.640000eV_W5meV"
            legacy_b = parent / "sample_roundtrip_MCD_E1.648600eV_W5meV"
            legacy_a.mkdir(parents=True)
            legacy_b.mkdir()
            (legacy_a / "map.dat").write_text("first", encoding="utf-8")
            (legacy_b / "map.dat").write_text("second", encoding="utf-8")
            first = ensure_mcd_package_dir(folder_text, "mcd/sample_roundtrip.csv")
            second = ensure_mcd_package_dir(folder_text, "mcd/sample_roundtrip.csv")

            self.assertEqual(first, second)
            self.assertEqual(first.name, "sample_roundtrip_MCD")
            self.assertFalse(legacy_a.exists())
            self.assertFalse(legacy_b.exists())
            self.assertEqual(
                {path.read_text(encoding="utf-8") for path in first.glob("map*.dat")},
                {"first", "second"},
            )
            self.assertEqual(
                [path.name for path in first.parent.iterdir() if path.is_dir()],
                ["sample_roundtrip_MCD"],
            )

    def test_mcd_processed_status_matches_window_specific_settings(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            package = ensure_mcd_package_dir(root, "mcd/sample.csv")
            settings = package / "sample_MCD_settings_E1.640000eV_W5meV.json"
            settings.write_text(
                json.dumps(
                    {
                        "workflow": "MCD",
                        "source_file": "sample.csv",
                        "created_utc": "2026-08-24T21:30:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            status = discover_mcd_processing_status(root, ["mcd/sample.csv", "mcd/other.csv"])

        self.assertEqual(
            status,
            {"mcd/sample.csv": "2026-08-24T21:30:00+00:00"},
        )

    def _write_branch_sweep(self, folder: Path, fields: list[float]) -> Path:
        rows = []
        for b in fields:
            for angle, scale, sign in ((10.0, 1.0, 1.0), (50.0, 3.0, -1.0)):
                rows.append({
                    "B_T": b, "angle_deg": angle,
                    "Vtg_V": 15.65, "Vbg_V": -8.602,
                    "Doping_V": 6.3, "Efield_V": 25.0,
                    "700": scale * (100 + sign * 4 * b),
                    "710": scale * (60 + sign * 2 * b),
                })
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
                fit_near_zero=True,
                fit_window_t=0.6,
            )
            names = {path.name for path in paths.values()}
            self.assertEqual(
                names,
                {
                    "branch_sweep_MCD_vs_B_E1.750000eV_W25meV.png",
                    "branch_sweep_MCD_vs_B_E1.750000eV_W25meV.csv",
                    "branch_sweep_MCD_pair_diagnostics.csv",
                    "branch_sweep_MCD_settings_E1.750000eV_W25meV.json",
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
                    "low_field_mcd_slope_increasing_per_T",
                    "low_field_mcd_slope_decreasing_per_T",
                    "low_field_fit_half_range_T", "low_field_fit_range_mode",
                ],
            )
            self.assertTrue(np.isfinite(table["low_field_mcd_slope_increasing_per_T"].iloc[0]))
            self.assertTrue(np.isfinite(table["low_field_mcd_slope_decreasing_per_T"].iloc[0]))
            self.assertEqual(table["low_field_fit_half_range_T"].unique().tolist(), [0.6])
            self.assertEqual(table["low_field_fit_range_mode"].unique().tolist(), ["fixed"])
            self.assertIn("B_increasing_T", table.columns)
            self.assertIn("B_decreasing_T", table.columns)
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
            self.assertEqual(payload["mcd_b"]["center_selection"], {"method": "manual"})
            self.assertEqual(payload["acquisition_conditions"]["Doping"], [6.3, 6.3])
            self.assertIn("Vtg = +15.65 V", payload["acquisition_conditions_display"])
            self.assertTrue(payload["mcd_b"]["show_signed_mean"])
            self.assertTrue(payload["mcd_b"]["fit_near_zero"])
            self.assertTrue(np.isfinite(payload["mcd_b"]["low_field_mcd_slope_increasing_per_T"]))
            self.assertTrue(np.isfinite(payload["mcd_b"]["low_field_mcd_slope_decreasing_per_T"]))
            self.assertFalse(payload["mcd_b"]["show_field_signed_absolute_mean"])
            self.assertFalse(payload["mcd_b"]["show_integral"])

            with patch(
                "core.mcd.pair_window_trace_by_branch",
                wraps=pair_window_trace_by_branch,
            ) as calculate_traces:
                second_paths = export_mcd_analysis_bundle(
                    result,
                    str(folder),
                    trace_map="Combo",
                    center_ev=1.70,
                    width_mev=25.0,
                    metric="mean",
                    settings=export_settings,
                )
            self.assertEqual(calculate_traces.call_count, 1)
            self.assertEqual(second_paths["pair_diagnostics"], paths["pair_diagnostics"])
            self.assertEqual(len(list(folder.glob("*_MCD_pair_diagnostics.csv"))), 1)

    def test_condition_box_keeps_the_exported_grid_size_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            result = process_mcd(
                str(self._write_branch_sweep(folder, [-1.0, 0.0, 1.0])),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            with patch.object(Figure, "savefig", autospec=True) as savefig:
                export_mcd_analysis_bundle(
                    result, str(folder / "with_conditions"), trace_map="Combo",
                    center_ev=1.75, width_mev=5.0, metric="mean",
                )
                figure_with_conditions = savefig.call_args.args[0]
                grid_with_conditions = figure_with_conditions.axes[0].get_position().bounds
                condition_artists = [
                    text for text in figure_with_conditions.axes[0].texts
                    if "Vtg" in text.get_text()
                ]
                self.assertEqual(len(condition_artists), 1)
                self.assertEqual(condition_artists[0].get_fontsize(), 16.0)

                result.acquisition_conditions = {}
                export_mcd_analysis_bundle(
                    result, str(folder / "without_conditions"), trace_map="Combo",
                    center_ev=1.75, width_mev=5.0, metric="mean",
                )
                figure_without_conditions = savefig.call_args.args[0]
                grid_without_conditions = figure_without_conditions.axes[0].get_position().bounds

                export_mcd_analysis_bundle(
                    result, str(folder / "with_integral"), trace_map="Combo",
                    center_ev=1.75, width_mev=5.0, metric="integral",
                    show_integral=True,
                )
                figure_with_integral = savefig.call_args.args[0]
                grid_with_integral = figure_with_integral.axes[0].get_position().bounds

        np.testing.assert_allclose(grid_with_conditions, grid_without_conditions, atol=0, rtol=0)
        physical_without_integral = np.asarray(grid_without_conditions) * np.asarray([
            figure_without_conditions.get_figwidth(), figure_without_conditions.get_figheight(),
            figure_without_conditions.get_figwidth(), figure_without_conditions.get_figheight(),
        ])
        physical_with_integral = np.asarray(grid_with_integral) * np.asarray([
            figure_with_integral.get_figwidth(), figure_with_integral.get_figheight(),
            figure_with_integral.get_figwidth(), figure_with_integral.get_figheight(),
        ])
        np.testing.assert_allclose(physical_without_integral, physical_with_integral, atol=1e-12, rtol=0)
        self.assertGreater(figure_with_integral.get_figwidth(), figure_without_conditions.get_figwidth())

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

    def test_pair_trace_can_compute_only_visible_corrected_metric(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            result = process_mcd(
                str(self._write_branch_sweep(folder, [-1.0, 0.0, 1.0, 0.0, -1.0])),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            branches = pair_window_trace_by_branch(
                result, 1.75, 25.0, metrics=("mean",), include_raw=False
            )
        for branch in ("B increasing", "B decreasing"):
            self.assertEqual(set(branches[branch]), {"corrected_mean"})

    def test_processing_reads_the_mcd_table_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            source = self._write_branch_sweep(
                Path(folder_text), [-1.0, 0.0, 1.0, 0.0, -1.0]
            )
            with patch("core.mcd.pd.read_csv", wraps=pd.read_csv) as read_csv:
                process_mcd(
                    str(source), McdSettings(max_sequence_gap=1, max_delta_b=0.01)
                )
        self.assertEqual(read_csv.call_count, 1)

    def test_angle_detection_does_not_parse_the_spectral_table(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            source = self._write_sweep(Path(folder_text))
            with patch("core.mcd.pd.read_csv", wraps=pd.read_csv) as read_csv:
                angles = detect_angles(str(source))
        self.assertEqual(angles, (10.0, 50.0))
        read_csv.assert_not_called()

    def test_recent_mcd_processing_result_is_reused(self) -> None:
        class Sink:
            def emit(self, *_args) -> None:
                pass

        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            source = self._write_sweep(folder)
            settings = McdSettings(max_sequence_gap=1, max_delta_b=0.01)
            options = LoadOptions(
                mode="MCD", folder=str(folder), selected_files=[source.name],
                baseline_files=[], pl_log_scale=False, drr_baseline_text="",
                drr_baseline_which="", compare_log_scale=False,
                mcd_settings=settings,
            )
            window = MainWindow()
            try:
                with patch("ui_qt.main_window.process_mcd", wraps=process_mcd) as process:
                    first = window._load_task(options, progress=Sink(), log=Sink())
                    second = window._load_task(options, progress=Sink(), log=Sink())
                self.assertEqual(process.call_count, 1)
                self.assertFalse(first.mcd_cache_hit)
                self.assertTrue(second.mcd_cache_hit)
                self.assertIs(first.mcd_result, second.mcd_result)
            finally:
                window.close()

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
                wait_for_file_catalog(window)
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
                # A one-item metric legend adds no information; only the
                # increasing/decreasing branch key remains.
                self.assertEqual(
                    [text.get_text() for text in metric_legend.get_texts()],
                    ["B increasing", "B decreasing"],
                )
                figure_bounds = window.figure.bbox
                for axis in (window._mcd_trace_ax, window._mcd_integral_ax):
                    for text in [axis.yaxis.label, *axis.get_yticklabels()]:
                        bounds = text.get_window_extent(window.canvas.get_renderer())
                        self.assertGreaterEqual(bounds.x0, figure_bounds.x0)
                        self.assertLessEqual(bounds.x1, figure_bounds.x1)
                original_trace_artists = {
                    key: id(line) for key, line in window._mcd_trace_lines.items()
                }
                with patch.object(window.canvas, "blit", wraps=window.canvas.blit) as blit:
                    self.assertTrue(window._refresh_mcd_center_trace())
                    self.assertGreater(blit.call_count, 0)
                self.assertEqual(
                    original_trace_artists,
                    {key: id(line) for key, line in window._mcd_trace_lines.items()},
                )
                with (
                    patch.object(window, "_refresh_mcd_center_trace", return_value=True) as refresh,
                    patch.object(window, "_plot_mode") as full_replot,
                ):
                    window.mcd_window_center_spin.setValue(
                        window.mcd_window_center_spin.value() + 1e-6
                    )
                    self.assertTrue(window._mcd_center_refresh_timer.isActive())
                    window._mcd_center_refresh_timer.stop()
                    window._apply_pending_mcd_center_refresh()
                    refresh.assert_called_once_with()
                    full_replot.assert_not_called()
                if window.mcd_pair_b_combo.count() > 1:
                    with (
                        patch.object(window, "_refresh_mcd_pair_panels", return_value=True) as pair_refresh,
                        patch.object(window, "_plot_mode") as full_replot,
                    ):
                        target_index = 0 if window.mcd_pair_b_combo.currentIndex() != 0 else 1
                        window.mcd_pair_b_combo.setCurrentIndex(target_index)
                        pair_refresh.assert_called_once_with()
                        full_replot.assert_not_called()
            finally:
                window.close()

    def test_mcd_subfolder_source_is_discovered_selected_and_loaded(self) -> None:
        class Sink:
            def emit(self, *_args) -> None:
                pass

        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            mcd_folder = folder / "mcd"
            mcd_folder.mkdir()
            self._write_sweep(mcd_folder)
            (folder / "unrelated.csv").write_text("x,y\n1,2\n", encoding="utf-8")

            window = MainWindow()
            try:
                window._set_current_folder(str(folder))
                wait_for_file_catalog(window)

                self.assertEqual(window.mcd_available_files, ["mcd/sweep.csv"])
                self.assertEqual(window.mcd_files.count(), 1)
                self.assertEqual(window.mcd_files.item(0).text(), "mcd/sweep.csv")
                self.assertEqual(window._selected(window.mcd_files), ["mcd/sweep.csv"])
                self.assertIn("Selected:", window.mcd_selection_summary.text())
                self.assertIn("● NEW", window.mcd_selection_summary.text())
                self.assertTrue(
                    any(
                        os.path.samefile(watched, mcd_folder)
                        for watched in window.folder_watcher.directories()
                    )
                )

                options = LoadOptions(
                    mode="MCD", folder=str(folder), selected_files=["mcd/sweep.csv"],
                    baseline_files=[], pl_log_scale=False, drr_baseline_text="",
                    drr_baseline_which="", compare_log_scale=False,
                    mcd_settings=McdSettings(max_sequence_gap=1, max_delta_b=0.01),
                )
                loaded = window._load_task(options, progress=Sink(), log=Sink())
                self.assertEqual(loaded.primary_file, "mcd/sweep.csv")
                self.assertTrue(
                    os.path.samefile(
                        loaded.mcd_result.source_file,
                        mcd_folder / "sweep.csv",
                    )
                )
            finally:
                window.close()

    def test_mcd_source_filter_defaults_to_all_and_reports_counts(self) -> None:
        window = MainWindow()
        try:
            window._mcd_source_filter_preference = "all"
            window.mcd_available_files = ["mcd/a.csv", "mcd/b.csv", "mcd/c.csv"]
            window.mcd_processed_status = {"mcd/b.csv": "2026-08-24T20:00:00+00:00"}

            self.assertEqual(window._mcd_saved_source_filter(), "all")
            self.assertEqual(
                window._mcd_source_filter_counts(),
                {"all": 3, "unprocessed": 2, "processed": 1},
            )
            window._mcd_source_filter_preference = "processed"
            self.assertEqual(window._mcd_saved_source_filter(), "processed")
        finally:
            window.close()

    def test_processed_mcd_selection_has_prominent_status(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            mcd_folder = root / "mcd"
            mcd_folder.mkdir()
            self._write_sweep(mcd_folder)
            package = ensure_mcd_package_dir(root, "mcd/sweep.csv")
            (package / "sweep_MCD_settings_E1.700000eV_W5meV.json").write_text(
                json.dumps(
                    {
                        "workflow": "MCD",
                        "source_file": "sweep.csv",
                        "created_utc": "2026-08-24T21:30:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            window = MainWindow()
            try:
                window._set_current_folder(str(root))
                wait_for_file_catalog(window)
                summary = window.mcd_selection_summary.text()
                self.assertIn("✓ PROCESSED", summary)
                self.assertIn("Last saved: 2026-08-24 21:30", summary)
                self.assertIn("#237A3B", window.mcd_selection_summary.styleSheet())
            finally:
                window.close()

    def test_mcd_file_chooser_updates_the_single_hidden_selection(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            mcd_folder = folder / "mcd"
            mcd_folder.mkdir()
            first = self._write_sweep(mcd_folder)
            first.replace(mcd_folder / "sweep_1.csv")
            self._write_sweep(mcd_folder).replace(mcd_folder / "sweep_2.csv")

            window = MainWindow()
            try:
                window._set_current_folder(str(folder))
                wait_for_file_catalog(window)
                with (
                    patch.object(
                        window, "_open_mcd_source_dialog", return_value="mcd/sweep_2.csv"
                    ),
                    patch.object(window, "_start_load") as start_load,
                ):
                    window._edit_mcd_source()
                self.assertEqual(window._selected(window.mcd_files), ["mcd/sweep_2.csv"])
                self.assertIn("sweep_2.csv", window.mcd_selection_summary.toolTip())
                start_load.assert_called_once_with("MCD")

                window._clear_mcd_source()
                self.assertEqual(window._selected(window.mcd_files), [])
                self.assertEqual(window.mcd_selection_summary.text(), "No MCD CSV selected.")
            finally:
                window.close()

    def test_mcd_chooser_sources_are_sorted_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            mcd_folder = folder / "mcd"
            mcd_folder.mkdir()
            first = self._write_sweep(mcd_folder)
            first = first.replace(mcd_folder / "older.csv")
            second = self._write_sweep(mcd_folder)
            second = second.replace(mcd_folder / "newest.csv")
            os.utime(first, (1000, 1000))
            os.utime(second, (2000, 2000))

            window = MainWindow()
            try:
                window._set_current_folder(str(folder))
                wait_for_file_catalog(window)
                self.assertEqual(
                    window._mcd_sources_newest_first(),
                    ["mcd/newest.csv", "mcd/older.csv"],
                )
            finally:
                window.close()

    def test_multiple_mcd_subfolder_sources_require_user_selection(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            mcd_folder = folder / "mcd"
            mcd_folder.mkdir()
            first = self._write_sweep(mcd_folder)
            first.replace(mcd_folder / "sweep_1.csv")
            self._write_sweep(mcd_folder).replace(mcd_folder / "sweep_2.csv")

            window = MainWindow()
            try:
                window._set_current_folder(str(folder))
                wait_for_file_catalog(window)
                self.assertEqual(
                    window.mcd_available_files,
                    ["mcd/sweep_1.csv", "mcd/sweep_2.csv"],
                )
                self.assertEqual(window._selected(window.mcd_files), [])
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
            self.assertTrue(window.mcd_fit_zero_chk.isChecked())
        finally:
            window.close()

    def test_low_field_slopes_remain_separate_by_branch(self) -> None:
        field = np.array([-0.2, 0.0, 0.2])
        traces = {
            "B increasing": {"corrected_mean": (field, 1.0 + 2.0 * field)},
            "B decreasing": {"corrected_mean": (field, -1.0 + 4.0 * field)},
        }
        fits = low_field_mcd_branch_fits(traces, 0.2)
        self.assertAlmostEqual(fits["B increasing"][0], 2.0)
        self.assertAlmostEqual(fits["B decreasing"][0], 4.0)

    def test_near_zero_fit_does_not_expand_plot_beyond_measured_data(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            result = process_mcd(
                str(self._write_branch_sweep(
                    folder, [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0, -0.2, -0.5, -1.0]
                )),
                McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            with patch.object(Figure, "savefig", autospec=True) as savefig:
                export_mcd_analysis_bundle(
                    result, str(folder), trace_map="Combo", center_ev=1.75,
                    width_mev=25.0, metric="mean", fit_near_zero=True,
                    fit_window_t=0.2,
                )
            figure = savefig.call_args.args[0]
            axis = figure.axes[0]
            fit_lines = [
                line for line in axis.lines
                if line.get_color() in {"#d55e00", "#7a3db8"}
            ]
            self.assertEqual(len(fit_lines), 2)
            for fit_line in fit_lines:
                self.assertAlmostEqual(float(np.min(fit_line.get_xdata())), -1.0)
                self.assertAlmostEqual(float(np.max(fit_line.get_xdata())), 1.0)
            measured = np.concatenate([
                np.asarray(line.get_ydata(), float)
                for line in axis.lines
                if line.get_marker() == "o"
            ])
            y_low, y_high = axis.get_ylim()
            measured_span = float(np.nanmax(measured) - np.nanmin(measured))
            self.assertLessEqual(y_high, float(np.nanmax(measured)) + 0.051 * measured_span)
            self.assertGreaterEqual(y_low, float(np.nanmin(measured)) - 0.051 * measured_span)
            slope_texts = [text for text in axis.texts if "slope" in text.get_text().casefold()]
            self.assertEqual(len(slope_texts), 1)
            self.assertIn("Increasing", slope_texts[0].get_text())
            self.assertIn("Decreasing", slope_texts[0].get_text())
            self.assertEqual(slope_texts[0].get_fontsize(), 16)
            condition_texts = [text for text in axis.texts if "Vtg" in text.get_text()]
            if condition_texts:
                self.assertEqual(condition_texts[0].get_fontsize(), 16)
                self.assertNotEqual(
                    slope_texts[0].get_position(), condition_texts[0].get_position()
                )

    def test_live_preview_slope_box_uses_compact_axes_contained_text(self) -> None:
        window = MainWindow()
        try:
            window.figure.clear()
            axis = window.figure.add_axes([0.72, 0.55, 0.22, 0.34])
            window._add_mcd_preview_slope_box(
                axis,
                {
                    "B increasing": (0.113441, 0.0),
                    "B decreasing": (0.110572, 0.0),
                },
                "lower right",
            )
            window.canvas.draw()
            self.assertEqual(len(axis.texts), 1)
            slope_text = axis.texts[0]
            self.assertLessEqual(slope_text.get_fontsize(), 9.0)
            self.assertGreaterEqual(slope_text.get_fontsize(), 7.0)
            text_bounds = slope_text.get_window_extent(window.canvas.get_renderer())
            axis_bounds = axis.get_window_extent(window.canvas.get_renderer())
            self.assertGreaterEqual(text_bounds.x0, axis_bounds.x0 - 1.0)
            self.assertLessEqual(text_bounds.x1, axis_bounds.x1 + 1.0)
            self.assertGreaterEqual(text_bounds.y0, axis_bounds.y0 - 1.0)
            self.assertLessEqual(text_bounds.y1, axis_bounds.y1 + 1.0)
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
                    "key": "control",
                })()
                window._on_canvas_click(event)
                expected_index = int(np.argmin(np.abs(result.pair_b - 0.7)))
                self.assertEqual(int(window.mcd_pair_b_combo.currentData()), expected_index)
                self.assertIn(f"B = {result.pair_b[expected_index]:.5g} T", window._mcd_spectrum_ax.get_title())
                window.mcd_pair_b_combo.setCurrentIndex(0)
                self.assertIn(f"B = {result.pair_b[0]:.5g} T", window._mcd_spectrum_ax.get_title())
            finally:
                window.close()

    def test_dragging_mcd_window_moves_only_its_center(self) -> None:
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
                original_center = float(window.mcd_window_center_spin.value())
                original_width = float(window.mcd_window_width_spin.value())
                target = original_center + 0.004
                press = type("McdPress", (), {
                    "button": 1, "inaxes": window._mcd_heatmap_ax,
                    "xdata": original_center, "ydata": 0.4,
                })()
                move = type("McdMove", (), {
                    "inaxes": window._mcd_heatmap_ax,
                    "xdata": target, "ydata": 0.4,
                })()
                release = type("McdRelease", (), {"button": 1})()
                window._on_canvas_click(press)
                self.assertTrue(window._mcd_window_dragging)
                window._on_canvas_motion(move)
                window._on_canvas_release(release)

                expected = window._clamp_mcd_window_center(
                    target, window.loaded.mcd_result.energy_ev, original_width
                )
                self.assertAlmostEqual(window.mcd_window_center_spin.value(), expected, places=6)
                self.assertEqual(window.mcd_window_width_spin.value(), original_width)
                self.assertFalse(window._mcd_window_dragging)
            finally:
                window.close()

    def test_candidate_strip_previews_multiple_centers_and_restores_manual_center(self) -> None:
        class Sink:
            def emit(self, *_args) -> None:
                pass

        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            source = self._write_spectral_drift_sweep(folder)
            options = LoadOptions(
                mode="MCD", folder=str(folder), selected_files=[source.name], baseline_files=[],
                pl_log_scale=False, drr_baseline_text="", drr_baseline_which="", compare_log_scale=False,
                mcd_settings=McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            window = MainWindow()
            try:
                window._on_loaded(window._load_task(options, progress=Sink(), log=Sink()))
                result = window.loaded.mcd_result
                energy = 1239.841984 / np.asarray(result.wavelength_nm, float)
                feature = (
                    0.035 * np.exp(-0.5 * ((energy - 1.60) / 0.003) ** 2)
                    + 0.020 * np.exp(-0.5 * ((energy - 1.72) / 0.004) ** 2)
                )
                result.pair_mcd_corrected = np.sign(result.pair_b)[:, None] * feature[None, :]
                window.mcd_spins["xmin"].setValue(1.55)
                window.mcd_spins["xmax"].setValue(1.78)
                manual_center = 1.66
                window.mcd_window_center_spin.setValue(manual_center)
                original_width = window.mcd_window_width_spin.value()

                window._find_mcd_center_candidates()
                self.assertGreaterEqual(len(window._mcd_center_candidates), 2)
                self.assertFalse(window.mcd_candidate_buttons[0].isHidden())
                self.assertIn("1", window.mcd_candidate_buttons[0].text())
                second_center = window._mcd_center_candidates[1].center_ev
                window._use_mcd_center_candidate(1)
                self.assertAlmostEqual(window.mcd_window_center_spin.value(), second_center, places=6)
                self.assertEqual(window.mcd_window_width_spin.value(), original_width)
                self.assertTrue(window.mcd_candidate_buttons[1].isChecked())
                self.assertEqual(
                    result.summary["window_center_selection"]["method"], "suggested"
                )
                self.assertEqual(
                    result.summary["window_center_selection"]["candidate_rank"],
                    window._mcd_center_candidates[1].score_rank,
                )
                self.assertEqual(
                    result.summary["window_center_selection"]["candidate_energy_order"], 2
                )

                ymin, ymax = window._mcd_heatmap_ax.get_ylim()
                marker_index = window._mcd_candidate_marker_at(
                    window._mcd_center_candidates[0].center_ev,
                    ymin + 0.95 * (ymax - ymin),
                )
                self.assertEqual(marker_index, 0)
                window._return_to_manual_mcd_center()
                self.assertEqual(window._mcd_center_candidates, ())
                self.assertAlmostEqual(window.mcd_window_center_spin.value(), manual_center, places=6)
                self.assertEqual(result.summary["window_center_selection"], {"method": "manual"})
            finally:
                window.close()

    def test_center_candidates_refresh_automatically_on_load_and_recalculation(self) -> None:
        class Sink:
            def emit(self, *_args) -> None:
                pass

        first_candidates = (
            McdCenterCandidate(1.751, 8.0, 9.0, 0.90, 0.04, 1),
            McdCenterCandidate(1.762, 6.0, 7.0, 0.85, 0.03, 2),
        )
        refreshed_candidates = (
            McdCenterCandidate(1.756, 10.0, 11.0, 0.92, 0.05, 1),
        )
        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            source = self._write_sweep(folder)
            options = LoadOptions(
                mode="MCD", folder=str(folder), selected_files=[source.name], baseline_files=[],
                pl_log_scale=False, drr_baseline_text="", drr_baseline_which="", compare_log_scale=False,
                mcd_settings=McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            window = MainWindow()
            try:
                manual_center = 1.755
                window.mcd_window_center_spin.setValue(manual_center)
                with patch(
                    "ui_qt.main_window.suggest_mcd_window_centers",
                    side_effect=(first_candidates, refreshed_candidates),
                ) as suggest, patch.object(
                    window, "_plot_mode", wraps=window._plot_mode
                ) as plot_mode:
                    window._on_loaded(window._load_task(options, progress=Sink(), log=Sink()))
                    self.assertEqual(window._mcd_center_candidates, first_candidates)
                    self.assertAlmostEqual(window.mcd_window_center_spin.value(), manual_center)
                    self.assertEqual(
                        window.loaded.mcd_result.summary["window_center_selection"],
                        {"method": "manual"},
                    )
                    self.assertFalse(window.mcd_candidate_buttons[0].isHidden())
                    self.assertEqual(plot_mode.call_count, 1)

                    # A correction recalculation returns through the same load
                    # completion path and must replace, rather than append, suggestions.
                    window._on_loaded(window._load_task(options, progress=Sink(), log=Sink()))
                    self.assertEqual(window._mcd_center_candidates, refreshed_candidates)
                    self.assertAlmostEqual(window.mcd_window_center_spin.value(), manual_center)
                    self.assertEqual(suggest.call_count, 2)
                    self.assertEqual(plot_mode.call_count, 2)
            finally:
                window.close()

    def test_automatic_center_refresh_handles_no_candidates(self) -> None:
        class Sink:
            def emit(self, *_args) -> None:
                pass

        with tempfile.TemporaryDirectory() as folder_text:
            folder = Path(folder_text)
            source = self._write_sweep(folder)
            options = LoadOptions(
                mode="MCD", folder=str(folder), selected_files=[source.name], baseline_files=[],
                pl_log_scale=False, drr_baseline_text="", drr_baseline_which="", compare_log_scale=False,
                mcd_settings=McdSettings(max_sequence_gap=1, max_delta_b=0.01),
            )
            window = MainWindow()
            try:
                with patch("ui_qt.main_window.suggest_mcd_window_centers", return_value=()):
                    window._on_loaded(window._load_task(options, progress=Sink(), log=Sink()))
                self.assertEqual(window._mcd_center_candidates, ())
                self.assertEqual(window.mcd_candidate_label.text(), "Suggested:")
                self.assertNotIn("No reliable", window.statusBar().currentMessage())
            finally:
                window.close()
