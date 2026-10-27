"""SHG request settings shared by controls and the scientific workers.

Keep these value objects independent of numerical libraries so editing controls
or capturing a load request never initializes NumPy or Pandas on the GUI thread.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ShgSettings:
    peak_center_nm: float = 515.0
    gate_min_nm: float = 513.0
    gate_max_nm: float = 517.0
    left_min_nm: float = 508.0
    left_max_nm: float = 512.0
    right_min_nm: float = 518.0
    right_max_nm: float = 522.0
    background_method: str = "local_linear"
    sigma_clip: float = 3.0
    remove_cosmic_rays: bool = True
    cosmic_threshold_mad: float = 8.0
    cosmic_window_points: int = 7
    cosmic_max_width_points: int = 3
    angle_scale: float = 1.0
    angle_offset_deg: float = 0.0
    angle_wrap_deg: float | None = None
    include_failed_rows: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "integration_wavelength_nm": self.peak_center_nm,
                "integration_half_range_nm": 0.5 * (self.gate_max_nm - self.gate_min_nm),
                "integration_interval_nm": [self.gate_min_nm, self.gate_max_nm],
                "integration_source": (
                    "cosmic_cleaned_background_subtracted_spectrum"
                    if self.remove_cosmic_rays
                    else "background_subtracted_spectrum"
                ),
                "left_sideband_gap_nm": self.gate_min_nm - self.left_max_nm,
                "right_sideband_gap_nm": self.right_min_nm - self.gate_max_nm,
                "left_sideband_width_nm": self.left_max_nm - self.left_min_nm,
                "right_sideband_width_nm": self.right_max_nm - self.right_min_nm,
            }
        )
        return payload


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
