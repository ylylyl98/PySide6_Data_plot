from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.shg_history import scan_shg_history
from core.export import export_shg_results
from core.shg import ShgSettings, ShgSweepData, process_shg_sweep
import numpy as np
from core import export as export_module
from core.shg import ShgProcessResult


class ShgHistoryTests(unittest.TestCase):
    def _metadata(self, root: Path, name: str, payload: dict) -> None:
        path = root / "Processed Data" / "SHG" / "package"
        path.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_matches_exact_source_path_and_preserves_roles(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a").mkdir()
            (root / "b").mkdir()
            sources = ["a/sample.csv", "b/sample.csv"]
            self._metadata(root, "result.metadata.json", {
                "workflow": "SHG",
                "created_utc": "2026-09-11T12:00:00+00:00",
                "inputs": [{"role": "reference", "source_relative_path": "a/sample.csv"}],
            })
            snapshot = scan_shg_history(root, sources)
            status, roles = snapshot.processed_at, snapshot.roles
            self.assertEqual(status, {"a/sample.csv": "2026-09-11T12:00:00+00:00"})
            self.assertEqual(roles, {"a/sample.csv": ("reference",)})

    def test_legacy_basename_collision_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = ["a/sample.csv", "b/sample.csv"]
            self._metadata(root, "legacy.metadata.json", {
                "workflow": "SHG",
                "created_utc": "2026-09-11T12:00:00+00:00",
                "inputs": [{"role": "measurement", "name": "sample.csv"}],
            })
            ambiguous: set[str] = set()
            snapshot = scan_shg_history(
                root, sources, ambiguous_sources=ambiguous
            )
            status, roles = snapshot.processed_at, snapshot.roles
            self.assertEqual(status, {})
            self.assertEqual(roles, {})
            self.assertEqual(ambiguous, set(sources))

    def test_real_shg_writer_settings_file_is_discoverable(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "measurement.csv"
            source.write_text("placeholder", encoding="utf-8")
            data = ShgSweepData(
                source_file=source.name,
                wavelength_nm=np.array([508.0, 512.0, 513.0, 513.5, 514.0, 515.0, 516.0, 516.5, 517.0, 518.0, 522.0]),
                spectra=np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
                sweep_axis=("rot1",),
                target_angle_deg=np.array([0.0]),
                measured_angle_deg=np.array([0.0]),
                move_error_deg=np.array([0.0]),
                move_ok=np.array([True]),
                acquisition_ok=np.array([True]),
                source_rows=np.array([2]),
                detected_columns={"measured_angle": "measured_angle"},
            )
            settings = ShgSettings(
                peak_center_nm=515.0, gate_min_nm=514.0, gate_max_nm=516.0,
                left_min_nm=513.0, left_max_nm=513.5,
                right_min_nm=516.5, right_max_nm=517.0, background_method="none",
            )
            result = process_shg_sweep(data, settings)
            paths = export_shg_results(
                str(root), data=data, result=result, settings=settings,
                processed_name="Processed Data/SHG/package",
            )
            self.assertTrue(paths["settings"].exists())
            snapshot = scan_shg_history(root, [source.name])
            self.assertIn(source.name, snapshot.processed_at)
            self.assertIn("measurement", snapshot.roles[source.name])

    def test_shg_csv_bytes_match_legacy_scalar_finite_behavior(self) -> None:
        data = ShgSweepData(
            source_file="special.csv",
            wavelength_nm=np.array([514.0, 515.0, 516.0]),
            spectra=np.zeros((4, 3)),
            sweep_axis=("rot1", "rot1", "rot1", "rot1"),
            target_angle_deg=np.array([1.25, np.nan, np.inf, -0.0]),
            measured_angle_deg=np.array([1.25, np.nan, np.inf, -0.0]),
            move_error_deg=np.array([0.0, np.nan, -np.inf, -0.0]),
            move_ok=np.ones(4, dtype=bool),
            acquisition_ok=np.ones(4, dtype=bool),
            source_rows=np.arange(2, 6),
            detected_columns={"measured_angle": "measured_angle"},
        )
        settings = ShgSettings(
            peak_center_nm=515.0, gate_min_nm=514.0, gate_max_nm=516.0,
            left_min_nm=513.0, left_max_nm=513.5,
            right_min_nm=516.5, right_max_nm=517.0, background_method="none",
        )
        result = ShgProcessResult(
            data=data,
            settings=settings,
            measured_angle_deg=np.array([1.25, np.nan, np.inf, -0.0]),
            cleaned_spectra=data.spectra,
            cosmic_ray_mask=np.zeros((4, 3), dtype=bool),
            cosmic_pixels_removed=np.zeros(4, dtype=int),
            baseline=np.zeros((4, 3)),
            corrected=np.zeros((4, 3)),
            integrated_area=np.array([1.25, np.nan, np.inf, -0.0]),
            area_uncertainty=np.array([np.inf, -np.inf, np.nan, -0.0]),
            peak_height=np.array([1.25, np.nan, np.inf, -0.0]),
            peak_wavelength_nm=np.array([515.0, np.nan, np.inf, -0.0]),
            baseline_slope=np.array([0.0, np.nan, np.inf, -0.0]),
            baseline_at_center=np.array([0.0, np.nan, -np.inf, -0.0]),
            baseline_rms=np.array([0.0, np.nan, np.inf, -0.0]),
            gate_points=np.ones(4, dtype=int),
            included=np.ones(4, dtype=bool),
            quality_flags=("", "NAN", "INF", "NEGZERO"),
        )
        with tempfile.TemporaryDirectory() as current_folder, tempfile.TemporaryDirectory() as legacy_folder:
            current = export_shg_results(
                current_folder, data=data, result=result, settings=settings,
                processed_name="Processed Data/SHG/package",
            )["csv"].read_bytes()
            with patch.object(export_module.math, "isfinite", lambda value: np.isfinite(value)):
                legacy = export_shg_results(
                    legacy_folder, data=data, result=result, settings=settings,
                    processed_name="Processed Data/SHG/package",
                )["csv"].read_bytes()
        self.assertEqual(current, legacy)


if __name__ == "__main__":
    unittest.main()
