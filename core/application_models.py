"""Application state and request records shared by desktop frontends.

These records contain scientific data and plain values only. They can be used
by native Qt Quick or non-UI workers without importing the Widgets UI layer.
The legacy ``ui_qt.common`` module re-exports the same classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.loader import DataCube
from core.data_io import PowerSeriesSource
from core.mcd import McdCenterCandidate, McdResult, McdSettings
from core.plotting import HeatmapParams
from core.provenance import WorkingCopyRecord
from core.shg import ShgProcessResult, ShgSettings, ShgSweepData
from core.shg_fit import ShgAngularFitResult, ShgFitSettings, ShgTwistFitResult


@dataclass
class LoadedState:
    mode: str
    folder: str
    primary_file: str | None = None
    selected_files: List[str] = field(default_factory=list)
    baseline_files: List[str] = field(default_factory=list)
    drr_axis_coordinates: dict = field(default_factory=dict)
    drr_display_candidates: dict | None = None
    cube: DataCube | None = None
    compare_cubes: Dict[str, DataCube] | None = None
    compare_sources: Dict[str, str] = field(default_factory=dict)
    power_records: tuple[Any, ...] = ()
    power_groups: Dict[str, tuple[Any, ...]] = field(default_factory=dict)
    power_results: Dict[str, Any] = field(default_factory=dict)
    power_group_key: str = ""
    shg_data: ShgSweepData | None = None
    shg_background: ShgSweepData | None = None
    shg_result: ShgProcessResult | None = None
    shg_data_b: ShgSweepData | None = None
    shg_background_b: ShgSweepData | None = None
    shg_result_b: ShgProcessResult | None = None
    shg_fit: ShgAngularFitResult | None = None
    shg_fit_b: ShgAngularFitResult | None = None
    shg_twist: ShgTwistFitResult | None = None
    shg_compare: bool = False
    shg_settings: ShgSettings | None = None
    shg_fit_settings: ShgFitSettings | None = None
    mcd_result: McdResult | None = None
    mcd_settings: McdSettings | None = None
    mcd_center_candidates: tuple[McdCenterCandidate, ...] = ()
    mcd_center_history: tuple[dict[str, Any], ...] = ()
    mcd_candidate_search_range: tuple[float, float] | None = None
    mcd_cache_hit: bool = False
    drr_mode_label: str = "DR/R Self"
    drr_derivative_label: str = "None"
    drr_baseline_text: str = "Self (last frame)"
    drr_baseline_which: str = "last"
    drr_background_selection: Dict[str, Any] = field(default_factory=dict)
    drr_assignments: tuple[Any, ...] = ()
    drr_fitted_settings: dict = field(default_factory=dict)
    drr_fitted_references: tuple[Any, ...] = ()
    y_axis_spec: str = "auto"
    provenance_records: tuple[WorkingCopyRecord, ...] = ()
    load_timings: Dict[str, float] = field(default_factory=dict)
    drr_background_dependencies: tuple | None = None


@dataclass(frozen=True)
class LoadOptions:
    mode: str
    folder: str
    selected_files: List[str]
    baseline_files: List[str]
    pl_log_scale: bool
    drr_baseline_text: str
    drr_baseline_which: str
    compare_log_scale: bool
    y_axis_spec: str = "auto"
    compare_sources: Dict[str, str] = field(default_factory=dict)
    power_group_key: str = ""
    shg_settings: ShgSettings | None = None
    shg_fit_settings: ShgFitSettings | None = None
    shg_compare: bool = False
    mcd_settings: McdSettings | None = None
    mcd_candidate_width_mev: float = 5.0
    mcd_candidate_metric: str = "mean"
    mcd_history_roots: tuple[str, ...] = ()
    mcd_candidate_energy_range: tuple[float, float] | None = None
    drr_background_selection: Dict[str, Any] = field(default_factory=dict)
    drr_assignments: tuple[Any, ...] = ()
    drr_fitted_settings: dict = field(default_factory=dict)
    power_role_group_keys: tuple[str, ...] = ()
    drr_restore_files: bool = True
    drr_restore_centers: bool = True
    drr_resolve_background: bool = False
    drr_source_catalog: tuple[Any, ...] = ()
    power_source_catalog: tuple[PowerSeriesSource, ...] = ()
    power_source_signatures: tuple[tuple[str, int | None, int | None], ...] = ()


@dataclass(frozen=True)
class ExportOptions:
    mode: str
    params: HeatmapParams | None
    params_linear: HeatmapParams | None = None
    params_log: HeatmapParams | None = None
    pl_include_pair: bool = False
    params_intensity: HeatmapParams | None = None
    drr_cube: DataCube | None = None
    # Paired DRR export snapshots.  ``drr_cube`` remains the compatibility
    # field used by older callers and non-paired analysis exports.
    drr_raw_cube: DataCube | None = None
    drr_peak_result: dict | None = None
    drr_second_cube: DataCube | None = None
    drr_raw_params: HeatmapParams | None = None
    drr_display_state: dict | None = None
    drr_second_params: HeatmapParams | None = None
    drr_raw_sg_window: int = 0
    drr_raw_sg_polyorder: int = 0
    drr_second_sg_window: int = 0
    drr_second_sg_polyorder: int = 0
    drr_second_y_window: int = 0
    drr_second_auto_scale: bool = True
    drr_second_split_auto_keys: tuple[str, ...] = ()
    drr_pending_color_limits: tuple = ()
    drr_second_scale_xlim: tuple[float, float] | None = None
    drr_second_scale_ylim: tuple[float, float] | None = None
    drr_derivative_order: int | None = None
    drr_sg_window: int = 20
    drr_sg_polyorder: int = 2
    drr_sg_mode_label: str = "More correct (regrid)"
    compare_scale_tag: str = "linear"
    compare_clip: bool = True
    compare_gate: float = 0.0
    compare_background: float = 0.0
    compare_export_vp: bool = True
    compare_vp_vmin: float = -1.0
    compare_vp_vmax: float = 1.0
    power_axis_log: bool = False
    power_view: str = "Intensity"
    power_background: float = 0.0
    power_group_key: str = ""
    power_kk_group_key: str = ""
    power_kkp_group_key: str = ""
    power_kk_cube: DataCube | None = None
    power_kkp_cube: DataCube | None = None
    power_vp_cube: DataCube | None = None
    power_kk_records: tuple[Any, ...] = ()
    power_kkp_records: tuple[Any, ...] = ()
    power_pairing_mode: str = "stage"
    power_stage_pairs: tuple[Any, ...] = ()
    power_peak_payload: dict | None = None
    shg_settings: ShgSettings | None = None
    shg_fit_settings: ShgFitSettings | None = None
    mcd_map_name: str = "Combo"
    mcd_window_center_ev: float = 0.0
    mcd_window_width_mev: float = 5.0
    mcd_window_metric: str = "mean"
    mcd_settings: McdSettings | None = None
    mcd_show_raw: bool = False
    mcd_show_signed_mean: bool = True
    mcd_show_absolute_mean: bool = False
    mcd_show_unsigned_absolute_mean: bool = False
    mcd_show_integral: bool = False
    mcd_fit_near_zero: bool = False
    mcd_fit_window_t: float = 0.2
    cleanup_verified_sources: bool = False


__all__ = ["LoadedState", "LoadOptions", "ExportOptions"]
