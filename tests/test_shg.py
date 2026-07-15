import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.export import export_shg_results
from core.shg import (
    ShgSettings,
    ShgSweepData,
    inspect_shg_csv,
    load_shg_sweep_csv,
    process_shg_sweep,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _synthetic_data() -> tuple[ShgSweepData, np.ndarray]:
    wavelength = np.linspace(507.0, 523.0, 321)
    sigma = 0.35
    amplitudes = np.array([80.0, 140.0, 40.0])
    baseline = 800.0 + 2.2 * (wavelength - 515.0) + 0.06 * (wavelength - 515.0) ** 2
    spectra = np.vstack(
        [baseline + amplitude * np.exp(-0.5 * ((wavelength - 515.0) / sigma) ** 2) for amplitude in amplitudes]
    )
    expected_area = amplitudes * sigma * np.sqrt(2.0 * np.pi)
    data = ShgSweepData(
        source_file="synthetic.csv",
        wavelength_nm=wavelength,
        spectra=spectra,
        sweep_axis=("rot1", "rot1", "rot1"),
        target_angle_deg=np.array([1.0, 0.0, 2.0]),
        measured_angle_deg=np.array([1.01, -0.01, 2.02]),
        move_error_deg=np.array([0.01, 0.01, 0.02]),
        move_ok=np.array([True, True, False]),
        acquisition_ok=np.array([True, True, True]),
        source_rows=np.array([2, 3, 4]),
        detected_columns={"measured_angle": "measured position"},
    )
    return data, expected_area


class ShgLoaderTests(unittest.TestCase):
    def test_wide_table_loader_uses_measured_position_and_numeric_headers(self) -> None:
        file_name = "shg_wide_table.csv"
        self.assertTrue(inspect_shg_csv(str(FIXTURES), file_name))
        data = load_shg_sweep_csv(str(FIXTURES), file_name)
        self.assertEqual(data.source_file, file_name)
        self.assertTrue(np.allclose(data.measured_angle_deg, [0.9919, -0.0025]))
        self.assertTrue(np.allclose(data.target_angle_deg, [1.0, 0.0]))
        self.assertEqual(data.wavelength_nm.tolist(), [508.0, 510.0, 512.0, 514.0, 515.0, 516.0, 518.0, 520.0, 522.0])
        self.assertEqual(data.spectra.shape, (2, 9))
        self.assertEqual(data.move_ok.tolist(), [True, False])
        self.assertEqual(data.acquisition_ok.tolist(), [True, True])
        self.assertEqual(data.source_rows.tolist(), [2, 3])


class ShgProcessingTests(unittest.TestCase):
    def test_quadratic_sideband_fit_recovers_known_gaussian_area(self) -> None:
        data, expected_area = _synthetic_data()
        settings = ShgSettings(background_method="local_quadratic")
        result = process_shg_sweep(data, settings)
        self.assertTrue(np.allclose(result.integrated_area, expected_area, rtol=0.02, atol=0.1))
        self.assertTrue(np.allclose(result.peak_wavelength_nm, 515.0, atol=0.06))
        self.assertEqual(result.included.tolist(), [True, True, False])
        self.assertIn("MOVE_FAILED", result.quality_flags[2])

    def test_angle_calibration_wrap_and_include_failed_rows(self) -> None:
        data, _expected_area = _synthetic_data()
        settings = ShgSettings(
            background_method="local_quadratic",
            angle_scale=2.0,
            angle_offset_deg=1.0,
            angle_wrap_deg=2.0,
            include_failed_rows=True,
        )
        result = process_shg_sweep(data, settings)
        self.assertTrue(np.allclose(result.measured_angle_deg, [1.02, 0.98, 1.04]))
        self.assertEqual(result.included.tolist(), [True, True, True])

    def test_external_background_is_scaled_from_sidebands(self) -> None:
        data, expected_area = _synthetic_data()
        wavelength = data.wavelength_nm
        reference = 600.0 + 1.3 * (wavelength - 515.0) + 0.04 * (wavelength - 515.0) ** 2
        peak_only = data.spectra - (
            800.0 + 2.2 * (wavelength - 515.0) + 0.06 * (wavelength - 515.0) ** 2
        )
        measured = 1.25 * reference + 15.0 + 0.2 * (wavelength - 515.0) + peak_only
        external_data = ShgSweepData(
            **{**data.__dict__, "spectra": measured, "source_file": "measured.csv"}
        )
        background = ShgSweepData(
            **{
                **data.__dict__,
                "spectra": np.vstack([reference, reference]),
                "sweep_axis": ("background", "background"),
                "target_angle_deg": np.array([0.0, 1.0]),
                "measured_angle_deg": np.array([0.0, 1.0]),
                "move_error_deg": np.zeros(2),
                "move_ok": np.ones(2, dtype=bool),
                "acquisition_ok": np.ones(2, dtype=bool),
                "source_rows": np.array([2, 3]),
                "source_file": "background.csv",
            }
        )
        result = process_shg_sweep(
            external_data,
            ShgSettings(background_method="external"),
            background=background,
        )
        self.assertTrue(np.allclose(result.integrated_area, expected_area, rtol=0.02, atol=0.1))
        self.assertEqual(result.background_file, "background.csv")

    def test_export_is_sorted_by_measured_angle_and_includes_settings(self) -> None:
        data, _expected_area = _synthetic_data()
        settings = ShgSettings(background_method="local_quadratic")
        result = process_shg_sweep(data, settings)
        with tempfile.TemporaryDirectory() as folder:
            paths = export_shg_results(folder, data=data, result=result, settings=settings)
            lines = paths["csv"].read_text(encoding="utf-8").splitlines()
            self.assertIn("measured_angle_deg", lines[0])
            self.assertIn("source_row", lines[0])
            self.assertTrue(lines[1].startswith("rot1,0,-0.01"))
            self.assertIn("synthetic.csv", lines[1])
            settings_text = paths["settings"].read_text(encoding="utf-8")
            self.assertIn('"background_method": "local_quadratic"', settings_text)


if __name__ == "__main__":
    unittest.main()
