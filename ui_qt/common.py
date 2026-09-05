"""Explicit shared-symbol boundary for the DPTK Qt UI layer.

This module owns symbols shared across the application shell, feature pages,
and workflow controllers so they do not depend on ``ui_qt.main_window`` as a
generic symbol provider. Definitions moved here are deliberately kept identical
to their historical definitions; ``ui_qt.main_window`` re-exports them for
compatibility with existing tests and callers.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from PySide6.QtCore import QObject, QRect, QRunnable, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox as _QComboBox,
    QDoubleSpinBox as _QDoubleSpinBox,
    QListWidget,
    QSpinBox as _QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from core.loader import DataCube
from core.mcd import McdCenterCandidate, McdResult, McdSettings
from core.plotting import HeatmapParams
from core.provenance import WorkingCopyRecord
from core.shg import ShgProcessResult, ShgSettings, ShgSweepData
from core.shg_fit import ShgAngularFitResult, ShgFitSettings, ShgTwistFitResult

UI_METRICS = {
    "left_width": 380,
    "sidebar_min_width": 320,
    "sidebar_max_width": 560,
    "main_margin": 8,
    "group_margin": 6,
    "row_spacing": 6,
    "label_col_width": 86,
    "input_h": 28,
    "spin_w": 88,
    "short_combo_w": 145,
    "deriv_combo_w": 90,
    "long_combo_min_w": 200,
    "tool_h": 28,
    "tool_w": 62,
}


class QDoubleSpinBox(_QDoubleSpinBox):
    """A double spin box that cannot be changed by mouse-wheel scrolling."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class QSpinBox(_QSpinBox):
    """An integer spin box that cannot be changed by mouse-wheel scrolling."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class QComboBox(_QComboBox):
    """A combo box that leaves the mouse wheel to its enclosing scroll area."""

    def wheelEvent(self, event) -> None:
        event.ignore()


@dataclass
class LoadedState:
    mode: str
    folder: str
    primary_file: str | None = None
    selected_files: List[str] = field(default_factory=list)
    baseline_files: List[str] = field(default_factory=list)
    cube: DataCube | None = None
    compare_cubes: Dict[str, DataCube] | None = None
    compare_sources: Dict[str, str] = field(default_factory=dict)
    power_records: tuple[Any, ...] = ()
    power_groups: Dict[str, tuple[Any, ...]] = field(default_factory=dict)
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
    mcd_candidate_search_range: tuple[float, float] | None = None
    mcd_cache_hit: bool = False
    drr_mode_label: str = "DR/R Self"
    drr_derivative_label: str = "None"
    drr_baseline_text: str = "Self (last frame)"
    drr_baseline_which: str = "last"
    drr_background_selection: Dict[str, Any] = field(default_factory=dict)
    y_axis_spec: str = "auto"
    provenance_records: tuple[WorkingCopyRecord, ...] = ()


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
    mcd_candidate_energy_range: tuple[float, float] | None = None
    drr_background_selection: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportOptions:
    mode: str
    params: HeatmapParams | None
    params_linear: HeatmapParams | None = None
    params_log: HeatmapParams | None = None
    params_intensity: HeatmapParams | None = None
    drr_cube: DataCube | None = None
    drr_derivative_order: int | None = None
    drr_sg_window: int = 20
    drr_sg_polyorder: int = 2
    drr_sg_mode_label: str = "More correct (regrid)"
    compare_scale_tag: str = "linear"
    compare_clip: bool = True
    compare_gate: float = 0.0
    compare_background: float = 0.0
    compare_export_vp: bool = True
    power_axis_log: bool = False
    power_view: str = "Intensity"
    power_background: float = 0.0
    power_kk_group_key: str = ""
    power_kkp_group_key: str = ""
    power_kk_cube: DataCube | None = None
    power_kkp_cube: DataCube | None = None
    power_vp_cube: DataCube | None = None
    power_kk_records: tuple[Any, ...] = ()
    power_kkp_records: tuple[Any, ...] = ()
    power_pairing_mode: str = "stage"
    power_stage_pairs: tuple[Any, ...] = ()
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


class WorkerSignals(QObject):
    progress = Signal(int)
    log = Signal(str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class Worker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn(*self.args, progress=self.signals.progress, log=self.signals.log, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(f"{exc}\n\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()


class WrappedFilenameDelegate(QStyledItemDelegate):
    """Paint complete filenames as dynamically sized, wrapped list rows."""

    HORIZONTAL_PADDING = 8
    VERTICAL_PADDING = 6

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Wrapped rows are queried repeatedly while QListView lays out and
        # repaints.  Bounding-rect calculation is surprisingly expensive for
        # long paths, so keep the result for each viewport width/text pair.
        self._size_hint_cache: dict[tuple[int, str], QSize] = {}

    def _text_width(self, option: QStyleOptionViewItem) -> int:
        view = self.parent()
        if isinstance(view, QListWidget):
            width = view.viewport().width()
        else:
            width = option.rect.width()
        return max(80, int(width) - 2 * self.HORIZONTAL_PADDING)

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = str(index.data(Qt.DisplayRole) or "")
        cache_key = (self._text_width(opt), text)
        cached = self._size_hint_cache.get(cache_key)
        if cached is not None:
            return QSize(cached)
        text_rect = opt.fontMetrics.boundingRect(
            QRect(0, 0, self._text_width(opt), 10000),
            Qt.AlignLeft | Qt.TextWrapAnywhere,
            text,
        )
        base = super().sizeHint(opt, index)
        row_width = self._text_width(opt) + 2 * self.HORIZONTAL_PADDING
        result = QSize(
            row_width,
            max(base.height(), text_rect.height() + 2 * self.VERTICAL_PADDING),
        )
        # Keep this bounded in practice; a list normally has only a handful
        # of viewport widths, but a dialog can be resized many times.
        if len(self._size_hint_cache) > 512:
            self._size_hint_cache.clear()
        self._size_hint_cache[cache_key] = QSize(result)
        return result

    def paint(self, painter, option: QStyleOptionViewItem, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = opt.text
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)

        painter.save()
        painter.setPen(
            opt.palette.highlightedText().color()
            if opt.state & QStyle.State_Selected
            else opt.palette.text().color()
        )
        text_rect = option.rect.adjusted(
            self.HORIZONTAL_PADDING,
            self.VERTICAL_PADDING,
            -self.HORIZONTAL_PADDING,
            -self.VERTICAL_PADDING,
        )
        painter.drawText(
            text_rect,
            Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWrapAnywhere,
            text,
        )
        painter.restore()
