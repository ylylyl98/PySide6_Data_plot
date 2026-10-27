"""Configuration for BN comparison, without loading numerical algorithms."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BnCalibrationSettings:
    core_region: tuple[float, float] = (1.900, 1.970)
    train_region: tuple[float, float] = (1.900, 1.935)
    validation_region: tuple[float, float] = (1.945, 1.970)
    taper_width: float = .010
    gate_range: tuple[float, float] | None = None
    gate_count: int = 12
    template_window: int = 101
    template_polyorder: int = 3
    max_shift_pixels: float = .75
    gain_bounds: tuple[float, float] = (.75, 1.25)
    min_repeats: int = 4
    min_band_points: int = 12
    # Operational acceptance thresholds, not statistical confidence levels.
    min_improvement_fraction: float = .10
    min_absolute_improvement: float = 1e-5  # dimensionless DR/R
    drr_noninferiority_fraction: float = .02  # operational allowance, not a noise estimate
    drr_noninferiority_absolute: float = 1e-5
    derivative_window: int = 21
    derivative_polyorder: int = 2
    min_absolute_d2_improvement: float = .1  # (DR/R) / eV^2; operational, not a noise estimate
    max_local_degradation_fraction: float = .05
    max_absolute_drr_degradation: float = 1e-5
    max_absolute_d2_degradation: float = .1  # (DR/R) / eV^2
    local_validation_window: int = 31
    protected_drr_absolute_tolerance: float = 1e-12
    protected_d2_absolute_tolerance: float = 1e-7  # (DR/R) / eV^2; no amplitude normalization
    max_temporal_shift_difference: float = .15
    max_temporal_gain_difference: float = .10
    max_bn_shift_difference: float = .05
    max_bn_gain_difference: float = .05
    trusted_regions: tuple[tuple[float, float], ...] = ()
    uncertain_regions: tuple[tuple[float, float], ...] = ((1.940, 2.020),)
    protected_regions: tuple[tuple[float, float], ...] = ((1.580, 1.860),)
    protection_guard_pixels: int = 12
    train_regions: tuple[tuple[float, float], ...] = ()
