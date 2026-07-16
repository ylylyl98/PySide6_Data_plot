import unittest
import tempfile

import numpy as np

from core.export import export_shg_twist_comparison
from core.shg import ShgProcessResult, ShgSettings, ShgSweepData, process_shg_sweep
from core.shg_fit import (
    ShgFitSettings,
    evaluate_shg_angular_model,
    fit_shg_angular_result,
    fit_shg_twist_comparison,
    wrap_phase_difference_deg,
)


def _angular_sweep(
    x_center_deg: float,
    *,
    angles: np.ndarray | None = None,
    i0: float = 120.0,
    amplitude: float = 900.0,
) -> tuple[ShgSweepData, ShgProcessResult]:
    angle = np.asarray(angles if angles is not None else np.linspace(0.0, 180.0, 37), float)
    wavelength = np.linspace(507.0, 523.0, 321)
    sigma = 0.35
    normalized_peak = np.exp(-0.5 * ((wavelength - 515.0) / sigma) ** 2) / (
        sigma * np.sqrt(2.0 * np.pi)
    )
    target_area = evaluate_shg_angular_model(angle, i0, amplitude, x_center_deg)
    baseline = 800.0 + 1.8 * (wavelength - 515.0) + 0.04 * (wavelength - 515.0) ** 2
    spectra = baseline[None, :] + target_area[:, None] * normalized_peak[None, :]
    data = ShgSweepData(
        source_file=f"phase_{x_center_deg:g}.csv",
        wavelength_nm=wavelength,
        spectra=spectra,
        sweep_axis=tuple("rot1" for _ in angle),
        target_angle_deg=angle.copy(),
        measured_angle_deg=angle.copy(),
        move_error_deg=np.zeros(angle.size),
        move_ok=np.ones(angle.size, dtype=bool),
        acquisition_ok=np.ones(angle.size, dtype=bool),
        source_rows=np.arange(angle.size, dtype=int) + 2,
        detected_columns={"measured_angle": "measured_value"},
    )
    result = process_shg_sweep(data, ShgSettings(background_method="local_quadratic"))
    return data, result


class ShgAngularFitTests(unittest.TestCase):
    def test_single_file_fit_recovers_center_and_model_parameters(self) -> None:
        _data, result = _angular_sweep(17.5, i0=140.0, amplitude=850.0)
        fit = fit_shg_angular_result(result)
        self.assertAlmostEqual(fit.x_center_deg, 17.5, delta=0.05)
        self.assertAlmostEqual(fit.i0, 140.0, delta=0.5)
        self.assertAlmostEqual(fit.amplitude, 850.0, delta=0.5)
        self.assertGreater(fit.r_squared, 0.9999)
        self.assertEqual(fit.point_count, 37)

    def test_compare_fit_recovers_twist_with_unequal_angle_grids(self) -> None:
        _reference_data, reference = _angular_sweep(8.0, angles=np.linspace(0.0, 180.0, 37))
        _sample_data, sample = _angular_sweep(14.0, angles=np.linspace(0.5, 179.5, 31))
        comparison = fit_shg_twist_comparison(reference, sample)
        self.assertAlmostEqual(comparison.delta_x_center_deg, 6.0, delta=0.05)
        self.assertAlmostEqual(comparison.signed_twist_angle_deg, 4.0, delta=0.05)
        self.assertAlmostEqual(comparison.absolute_twist_angle_deg, 4.0, delta=0.05)

    def test_phase_wrapping_and_branch_selection(self) -> None:
        self.assertAlmostEqual(wrap_phase_difference_deg(2.0 - 88.0), 4.0)
        _reference_data, reference = _angular_sweep(88.0)
        _sample_data, sample = _angular_sweep(2.0)
        nearest = fit_shg_twist_comparison(reference, sample)
        branched = fit_shg_twist_comparison(reference, sample, ShgFitSettings(phase_branch=-1))
        self.assertAlmostEqual(nearest.delta_x_center_deg, 4.0, delta=0.05)
        self.assertAlmostEqual(nearest.signed_twist_angle_deg, 8.0 / 3.0, delta=0.05)
        self.assertAlmostEqual(branched.delta_x_center_deg, -86.0, delta=0.05)

    def test_fit_uses_cosmic_cleaned_area_and_excludes_failed_rows(self) -> None:
        data, _original_result = _angular_sweep(22.0)
        spectra = np.asarray(data.spectra, float).copy()
        spike_index = int(np.argmin(np.abs(data.wavelength_nm - 514.0)))
        spectra[5, spike_index] += 5000.0
        move_ok = np.asarray(data.move_ok, bool).copy()
        move_ok[10] = False
        changed = ShgSweepData(**{**data.__dict__, "spectra": spectra, "move_ok": move_ok})
        result = process_shg_sweep(changed, ShgSettings(background_method="local_quadratic"))
        fit = fit_shg_angular_result(result)
        self.assertGreater(result.cosmic_pixels_removed[5], 0)
        self.assertFalse(fit.fit_mask[10])
        self.assertEqual(fit.point_count, 36)
        self.assertAlmostEqual(fit.x_center_deg, 22.0, delta=0.05)

    def test_fit_rejects_insufficient_angular_span(self) -> None:
        _data, result = _angular_sweep(10.0, angles=np.linspace(0.0, 20.0, 10))
        with self.assertRaisesRegex(ValueError, "angular coverage"):
            fit_shg_angular_result(result)

    def test_comparison_export_keeps_both_processed_csvs_and_twist_summary(self) -> None:
        reference_data, reference = _angular_sweep(5.0)
        sample_data, sample = _angular_sweep(11.0, angles=np.linspace(0.5, 179.5, 31))
        settings = ShgSettings(background_method="local_quadratic")
        twist = fit_shg_twist_comparison(reference, sample)
        with tempfile.TemporaryDirectory() as folder:
            paths = export_shg_twist_comparison(
                folder,
                reference_data=reference_data,
                reference_result=reference,
                sample_data=sample_data,
                sample_result=sample,
                settings=settings,
                twist=twist,
            )
            self.assertTrue(paths["reference_csv"].is_file())
            self.assertTrue(paths["sample_csv"].is_file())
            reference_header = paths["reference_csv"].read_text(encoding="utf-8").splitlines()[0]
            combined = paths["combined_csv"].read_text(encoding="utf-8")
            summary = paths["twist_summary_csv"].read_text(encoding="utf-8")

        self.assertIn("Reference A", combined)
        self.assertIn("Sample B", combined)
        self.assertIn("fit_intensity_counts_nm", reference_header)
        self.assertIn("fit_residual_counts_nm", combined.splitlines()[0])
        self.assertIn("signed_twist_angle_deg", summary.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
