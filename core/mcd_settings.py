"""MCD processing settings shared by the UI and scientific pipeline.

Keep this module independent of numerical and plotting libraries so opening
an empty MCD page does not initialize the processing stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class McdSettings:
    pos_angle: float | None = None
    neg_angle: float | None = None
    max_sequence_gap: int = 3
    max_delta_b: float = 0.1
    pair_b_alignment: Literal["direct", "interpolate"] = "direct"
    zero_window_t: float = 0.02
    reference_mode: Literal["nearest", "window"] = "nearest"
    bin_decimals: int = 3
    gain_mode: Literal["per_wavelength", "smoothed", "scalar"] = "per_wavelength"
    gain_smooth_window: int = 15
    gain_smooth_poly: int = 2
    correction_mode: Literal["global", "pair_scale", "pair_affine", "pair_spectral"] = "pair_spectral"
    spectral_order: Literal[1, 2] = 2
    background_ranges_ev: tuple[tuple[float, float], ...] = ()
    background_selection: Literal["auto", "manual", "suggested"] = "auto"
    suggestion_protected_ranges_ev: tuple[tuple[float, float], ...] = ()
    manual_protected_ranges_ev: tuple[tuple[float, float], ...] = ()
    suggestion_linear_validation_rms: float | None = None
    suggestion_quadratic_validation_rms: float | None = None
    suggestion_algorithm: str | None = None
    dark_pos_file: str | None = None
    dark_neg_file: str | None = None
