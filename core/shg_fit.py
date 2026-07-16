from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from core.shg import ShgProcessResult


@dataclass(frozen=True)
class ShgFitSettings:
    enabled: bool = True
    angle_min_deg: float | None = None
    angle_max_deg: float | None = None
    use_uncertainty_weights: bool = True
    include_excluded_rows: bool = False
    minimum_points: int = 6
    minimum_span_deg: float = 45.0
    phase_branch: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ShgAngularFitResult:
    settings: ShgFitSettings
    fit_mask: np.ndarray
    fitted_intensity: np.ndarray
    residual: np.ndarray
    i0: float
    amplitude: float
    x_center_deg: float
    i0_uncertainty: float
    amplitude_uncertainty: float
    x_center_uncertainty_deg: float
    r_squared: float
    rmse: float
    reduced_chi_squared: float
    point_count: int
    angle_span_deg: float

    def to_dict(self) -> dict[str, object]:
        return {
            "model": "I(theta) = I0 + A*cos^2(2*(theta-xc))",
            "i0": self.i0,
            "amplitude": self.amplitude,
            "x_center_deg": self.x_center_deg,
            "i0_uncertainty": self.i0_uncertainty,
            "amplitude_uncertainty": self.amplitude_uncertainty,
            "x_center_uncertainty_deg": self.x_center_uncertainty_deg,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "reduced_chi_squared": self.reduced_chi_squared,
            "point_count": self.point_count,
            "angle_span_deg": self.angle_span_deg,
            "settings": self.settings.to_dict(),
        }


@dataclass(frozen=True)
class ShgTwistFitResult:
    reference_fit: ShgAngularFitResult
    sample_fit: ShgAngularFitResult
    nearest_delta_x_center_deg: float
    phase_branch: int
    delta_x_center_deg: float
    delta_x_center_uncertainty_deg: float
    signed_twist_angle_deg: float
    absolute_twist_angle_deg: float
    twist_uncertainty_deg: float

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_period_deg": 90.0,
            "twist_period_deg": 60.0,
            "nearest_delta_x_center_deg": self.nearest_delta_x_center_deg,
            "phase_branch": self.phase_branch,
            "delta_x_center_deg": self.delta_x_center_deg,
            "delta_x_center_uncertainty_deg": self.delta_x_center_uncertainty_deg,
            "signed_twist_angle_deg": self.signed_twist_angle_deg,
            "absolute_twist_angle_deg": self.absolute_twist_angle_deg,
            "twist_uncertainty_deg": self.twist_uncertainty_deg,
            "reference_fit": self.reference_fit.to_dict(),
            "sample_fit": self.sample_fit.to_dict(),
        }


def evaluate_shg_angular_model(
    angle_deg: np.ndarray | float,
    i0: float,
    amplitude: float,
    x_center_deg: float,
) -> np.ndarray:
    angle = np.asarray(angle_deg, float)
    phase = np.deg2rad(2.0 * (angle - x_center_deg))
    return i0 + amplitude * np.cos(phase) ** 2


def wrap_phase_difference_deg(value: float) -> float:
    wrapped = (float(value) + 45.0) % 90.0 - 45.0
    return 45.0 if np.isclose(wrapped, -45.0) and value > 0 else wrapped


def _parameter_uncertainties(
    coefficients: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, float, float]:
    cosine_coefficient = float(coefficients[1])
    sine_coefficient = float(coefficients[2])
    harmonic = float(np.hypot(cosine_coefficient, sine_coefficient))
    if harmonic <= np.finfo(float).eps:
        return np.nan, np.nan, np.nan

    grad_harmonic = np.array([0.0, cosine_coefficient / harmonic, sine_coefficient / harmonic])
    grad_i0 = np.array([1.0, -cosine_coefficient / harmonic, -sine_coefficient / harmonic])
    grad_amplitude = 2.0 * grad_harmonic
    phase_scale = 180.0 / np.pi / 4.0
    grad_phase = np.array(
        [0.0, -phase_scale * sine_coefficient / harmonic**2, phase_scale * cosine_coefficient / harmonic**2]
    )

    def uncertainty(gradient: np.ndarray) -> float:
        variance = float(gradient @ covariance @ gradient)
        return float(np.sqrt(max(0.0, variance))) if np.isfinite(variance) else np.nan

    return uncertainty(grad_i0), uncertainty(grad_amplitude), uncertainty(grad_phase)


def fit_shg_angular_result(
    result: ShgProcessResult,
    settings: ShgFitSettings | None = None,
) -> ShgAngularFitResult:
    fit_settings = settings or ShgFitSettings()
    if not fit_settings.enabled:
        raise ValueError("SHG angular fitting is disabled.")

    angle = np.asarray(result.measured_angle_deg, float)
    intensity = np.asarray(result.integrated_area, float)
    uncertainty = np.asarray(result.area_uncertainty, float)
    mask = np.isfinite(angle) & np.isfinite(intensity)
    if not fit_settings.include_excluded_rows:
        mask &= np.asarray(result.included, bool)
    if fit_settings.angle_min_deg is not None:
        mask &= angle >= float(fit_settings.angle_min_deg)
    if fit_settings.angle_max_deg is not None:
        mask &= angle <= float(fit_settings.angle_max_deg)

    point_count = int(np.count_nonzero(mask))
    if point_count < max(3, int(fit_settings.minimum_points)):
        raise ValueError(
            f"SHG angular fit needs at least {max(3, int(fit_settings.minimum_points))} valid points; "
            f"found {point_count}."
        )
    fit_angle = angle[mask]
    fit_intensity = intensity[mask]
    angle_span = float(np.nanmax(fit_angle) - np.nanmin(fit_angle))
    if angle_span < float(fit_settings.minimum_span_deg):
        raise ValueError(
            f"SHG angular fit needs at least {fit_settings.minimum_span_deg:g}° angular coverage; "
            f"found {angle_span:.6g}°."
        )

    phase = np.deg2rad(4.0 * fit_angle)
    design = np.column_stack((np.ones(point_count), np.cos(phase), np.sin(phase)))
    weights = np.ones(point_count, dtype=float)
    if fit_settings.use_uncertainty_weights:
        sigma = uncertainty[mask]
        positive = sigma[np.isfinite(sigma) & (sigma > 0)]
        if positive.size:
            floor = max(float(np.nanmedian(positive)) * 0.1, np.finfo(float).eps)
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), np.maximum(sigma, floor), np.nanmedian(positive))
            weights = 1.0 / sigma**2

    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, None]
    weighted_values = fit_intensity * root_weight
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        weighted_design,
        weighted_values,
        rcond=None,
    )
    if rank < 3:
        raise ValueError("SHG angular fit is rank deficient; increase the angular range or number of points.")

    offset, cosine_coefficient, sine_coefficient = (float(value) for value in coefficients)
    harmonic = float(np.hypot(cosine_coefficient, sine_coefficient))
    if harmonic <= np.finfo(float).eps:
        raise ValueError("SHG angular modulation amplitude is too small to determine x_c.")
    i0 = offset - harmonic
    amplitude = 2.0 * harmonic
    x_center = (np.degrees(np.arctan2(sine_coefficient, cosine_coefficient)) / 4.0) % 90.0

    fitted_all = evaluate_shg_angular_model(angle, i0, amplitude, x_center)
    residual_all = intensity - fitted_all
    fit_residual = residual_all[mask]
    weighted_rss = float(np.sum(weights * fit_residual**2))
    degrees_of_freedom = max(1, point_count - 3)
    reduced_chi_squared = weighted_rss / degrees_of_freedom
    normal_inverse = np.linalg.pinv(design.T @ (weights[:, None] * design))
    covariance = normal_inverse * reduced_chi_squared
    i0_uncertainty, amplitude_uncertainty, phase_uncertainty = _parameter_uncertainties(
        coefficients,
        covariance,
    )

    total_sum_squares = float(np.sum((fit_intensity - np.mean(fit_intensity)) ** 2))
    residual_sum_squares = float(np.sum(fit_residual**2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else np.nan
    rmse = float(np.sqrt(np.mean(fit_residual**2)))

    return ShgAngularFitResult(
        settings=fit_settings,
        fit_mask=mask,
        fitted_intensity=fitted_all,
        residual=residual_all,
        i0=i0,
        amplitude=amplitude,
        x_center_deg=float(x_center),
        i0_uncertainty=i0_uncertainty,
        amplitude_uncertainty=amplitude_uncertainty,
        x_center_uncertainty_deg=phase_uncertainty,
        r_squared=float(r_squared),
        rmse=rmse,
        reduced_chi_squared=float(reduced_chi_squared),
        point_count=point_count,
        angle_span_deg=angle_span,
    )


def fit_shg_twist_comparison(
    reference: ShgProcessResult,
    sample: ShgProcessResult,
    settings: ShgFitSettings | None = None,
) -> ShgTwistFitResult:
    fit_settings = settings or ShgFitSettings()
    reference_fit = fit_shg_angular_result(reference, fit_settings)
    sample_fit = fit_shg_angular_result(sample, fit_settings)
    nearest_delta = wrap_phase_difference_deg(sample_fit.x_center_deg - reference_fit.x_center_deg)
    delta_x_center = nearest_delta + 90.0 * int(fit_settings.phase_branch)
    delta_uncertainty = float(
        np.hypot(reference_fit.x_center_uncertainty_deg, sample_fit.x_center_uncertainty_deg)
    )
    signed_twist = (2.0 / 3.0) * delta_x_center
    twist_uncertainty = (2.0 / 3.0) * delta_uncertainty
    return ShgTwistFitResult(
        reference_fit=reference_fit,
        sample_fit=sample_fit,
        nearest_delta_x_center_deg=float(nearest_delta),
        phase_branch=int(fit_settings.phase_branch),
        delta_x_center_deg=float(delta_x_center),
        delta_x_center_uncertainty_deg=delta_uncertainty,
        signed_twist_angle_deg=float(signed_twist),
        absolute_twist_angle_deg=float(abs(signed_twist)),
        twist_uncertainty_deg=float(twist_uncertainty),
    )
