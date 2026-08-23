from __future__ import annotations

import tempfile
import traceback
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import QFileSystemWatcher, QMimeData, QObject, QRect, QRunnable, QSettings, QSize, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox as _QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QProgressBar,
    QScrollArea,
    QSpinBox as _QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QPlainTextEdit,
    QToolButton,
    QFrame,
    QVBoxLayout,
    QWidget,
)
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from app_version import __version__
from core import data_io
from core.colormaps import CUSTOM_COLORMAPS, STANDARD_COLORMAPS, register_colormaps, resolve_cmap
from core.update_checker import (
    CheckResult,
    DownloadResult,
    check_for_update,
    download_installer,
    expected_installer_name,
    format_version,
    sha256_file,
)
from core.mcd import (
    McdBackgroundSuggestion, McdResult, McdSettings, background_fit_regions, detect_angles,
    export_mcd_analysis_bundle, pair_window_trace_by_branch, process_mcd, suggest_mcd_background_ranges,
)
from core.export import (
    build_drr_export_base,
    create_unique_package_dir,
    compare_source_title,
    export_compare_panels,
    export_drr_png_and_dat,
    export_pl_pngs_and_dat,
    export_power_series_png_and_dat,
    export_power_vp_pngs_and_dat,
    export_shg_results,
    export_shg_twist_comparison,
    vp_compare_export_base,
    vp_compare_title,
)
from core.loader import DataCube, resolve_dat_y_axis
from core.provenance import WorkingCopyRecord, cleanup_working_copy, verify_initial_data_working_file
from core.plotting import (
    COMPARE_PANEL_ORDER,
    HeatmapParams,
    HeatmapRender,
    SplitColorScale,
    plot_compare_panel,
    plot_drr,
    plot_heatmap,
    plot_pl,
    resolve_split_boundary,
)
from core.processing import (
    apply_sg_derivative_energy,
    background_correct_cube,
    clamp_sg_window,
    classify_angle_state,
    classify_compare_channel,
    coherent_compare_auto_assignment,
    compute_auto_limits,
    estimate_constant_background,
    group_measurement_files,
    infer_compare_angle_references,
    power_group_title,
    power_stage_paired_vp_cubes,
    power_valley_polarization_cube,
    nearest_gate_spectrum,
    parse_compare_gate_condition,
    parse_compare_rotation_angles,
    valley_polarization_cube,
)
from core.shg import ShgProcessResult, ShgSettings, ShgSweepData, process_shg_sweep
from core.shg_fit import (
    ShgAngularFitResult,
    ShgFitSettings,
    ShgTwistFitResult,
    evaluate_shg_angular_model,
    fit_shg_angular_result,
    fit_shg_twist_comparison,
)

UI_METRICS = {
    "left_width": 480,
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
    """A double spin box that cannot be changed by accidental page scrolling."""

    def wheelEvent(self, event) -> None:
        if self.hasFocus() and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
        else:
            event.ignore()


class QSpinBox(_QSpinBox):
    """An integer spin box that requires Ctrl+wheel for wheel adjustment."""

    def wheelEvent(self, event) -> None:
        if self.hasFocus() and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
        else:
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
    mcd_result: McdResult | None = None
    mcd_settings: McdSettings | None = None
    drr_mode_label: str = "DR/R Self"
    drr_derivative_label: str = "None"
    drr_baseline_text: str = "Self (last frame)"
    drr_baseline_which: str = "last"
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


def _vp_short_title(kk_title: str, kkp_title: str) -> str:
    kk_words = kk_title.split()
    kkp_words = kkp_title.split()
    n_pre = 0
    for a, b in zip(kk_words, kkp_words):
        if a == b:
            n_pre += 1
        else:
            break
    kk_mid = kk_words[n_pre:]
    kkp_mid = kkp_words[n_pre:]
    n_suf = 0
    for a, b in zip(reversed(kk_mid), reversed(kkp_mid)):
        if a == b:
            n_suf += 1
        else:
            break
    prefix = " ".join(kk_words[:n_pre])
    suffix = " ".join(kk_words[len(kk_words) - n_suf:]) if n_suf else ""
    if prefix or suffix:
        parts = ([prefix] if prefix else []) + ["KK/KKp"] + ([suffix] if suffix else [])
        return "VP " + " ".join(parts)
    return f"VP {kk_title} / {kkp_title}"


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
        text_rect = opt.fontMetrics.boundingRect(
            QRect(0, 0, self._text_width(opt), 10000),
            Qt.AlignLeft | Qt.TextWrapAnywhere,
            text,
        )
        base = super().sizeHint(opt, index)
        row_width = self._text_width(opt) + 2 * self.HORIZONTAL_PADDING
        return QSize(
            row_width,
            max(base.height(), text_rect.height() + 2 * self.VERTICAL_PADDING),
        )

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


class MainWindow(QMainWindow):
    SETTINGS_ORG = "DPTK"
    SETTINGS_APP = "PySide6_Data_Plot"
    SETTINGS_LAST_DATA_FOLDER = "data/last_folder"
    SETTINGS_LAST_PARENT_FOLDER = "data/last_parent_folder"
    SETTINGS_RECENT_FOLDERS = "data/recent_folders"
    SETTINGS_AUTO_UPDATE_CHECK = "updates/check_automatically"
    MAX_RECENT_FOLDERS = 8

    def __init__(self):
        super().__init__()
        register_colormaps()
        self.setWindowTitle("DPTK Desktop (PySide6)")
        self.setMinimumSize(1180, 700)
        self.thread_pool = QThreadPool.globalInstance()
        self.settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        self._pending_update: CheckResult | None = None
        self._download_in_progress = False
        self.current_folder = ""
        self._watched_folder = ""
        self.folder_watcher = QFileSystemWatcher(self)
        self.folder_refresh_timer = QTimer(self)
        self.folder_refresh_timer.setSingleShot(True)
        self.folder_refresh_timer.setInterval(800)
        self.folder_watcher.directoryChanged.connect(self._on_watched_folder_changed)
        self.folder_refresh_timer.timeout.connect(self._refresh_watched_folder)
        self.recent_folders: List[str] = self._load_recent_folders()
        self.available_files: List[str] = []
        self.available_map_files: List[str] = []
        self.drr_selected_files: List[str] = []
        self.drr_baseline_files_manual: List[str] = []
        self.drr_baseline_files_found: List[str] = []
        self.loaded: LoadedState | None = None
        self.last_plotted_mode: str | None = None
        self._last_export_move_folder = ""
        self._last_export_move_sources: list[str] = []
        self.log_lines: deque[str] = deque(maxlen=300)
        self._last_plot_params_key: tuple[Any, ...] | None = None
        self._last_plot_cube: DataCube | None = None
        self._drr_heatmap_ax = None
        self._drr_spectrum_ax = None
        self._pl_heatmap_ax = None
        self._pl_spectrum_ax = None
        self._cmp_heatmap_axes: dict[str, Any] = {}
        self._cmp_gate_lines: dict[str, Any] = {}
        self._cmp_linecut_ax = None
        self._cmp_active_cubes: dict[str, DataCube] = {}
        self._power_heatmap_ax = None
        self._power_heatmap_axes: dict[str, Any] = {}
        self._power_spectrum_ax = None
        self._power_last_plot_cube: DataCube | None = None
        self._power_gate_line = None
        self._power_gate_lines: dict[str, Any] = {}
        self._power_active_cubes: dict[str, DataCube] = {}
        self._power_active_export_cube: DataCube | None = None
        self._power_active_records: tuple[Any, ...] = ()
        self._power_selected_row_index: int | None = None
        self._shg_raw_ax = None
        self._shg_corrected_ax = None
        self._shg_angle_ax = None
        self._mcd_heatmap_ax = None
        self._mcd_spectrum_ax = None
        self._mcd_trace_ax = None
        self._mcd_integral_ax = None
        self._mcd_background_suggestion: McdBackgroundSuggestion | None = None
        self._shg_selected_index: int | None = None
        self._pl_last_plot_cube: DataCube | None = None
        self._gate_line = None
        self._pl_gate_line = None
        self._gate_motion_cid: int | None = None
        self._gate_click_cid: int | None = None
        self._suspend_drr_autoplot = False
        self._automatic_range_update = False
        self._pl_peak_gate: float | None = None
        self._pl_peak_indices: np.ndarray | None = None
        self._pl_fit_gate: float | None = None
        self._pl_fit_x: np.ndarray | None = None
        self._pl_fit_y: np.ndarray | None = None
        self._pl_fit_centers: np.ndarray | None = None
        self._pl_heatmap_peak_artist = None
        self._pl_heatmap_fit_artist = None
        self._drr_peak_gate: float | None = None
        self._drr_peak_indices: np.ndarray | None = None
        self._drr_fit_gate: float | None = None
        self._drr_fit_x: np.ndarray | None = None
        self._drr_fit_y: np.ndarray | None = None
        self._drr_fit_centers: np.ndarray | None = None
        self._drr_heatmap_peak_artist = None
        self._drr_heatmap_fit_artist = None
        self._folder_placeholder_text = ""
        self._load_in_progress = False
        self._active_load_mode: str | None = None
        self._active_load_succeeded = False
        self._mcd_reapply_pending = False
        self._mcd_auto_apply_timer = QTimer(self)
        self._mcd_auto_apply_timer.setSingleShot(True)
        self._mcd_auto_apply_timer.setInterval(650)
        self._mcd_auto_apply_timer.timeout.connect(self._apply_pending_mcd_settings)

        self._build_ui()
        self._folder_placeholder_text = self.folder_edit.placeholderText()
        self.apply_ui_metrics()
        self._wire_actions()
        self._cmp_update_background_mode()
        self._apply_initial_geometry()
        self._set_stage("No data")
        self._update_action_states()
        self._restore_last_folder()
        self.setAcceptDrops(True)
        self._schedule_automatic_update_check()

    def _load_recent_folders(self) -> List[str]:
        raw = self.settings.value(self.SETTINGS_RECENT_FOLDERS, [])
        if raw is None:
            values: list[object] = []
        elif isinstance(raw, str):
            values = [raw]
        else:
            values = list(raw)
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            folder = str(value)
            path = Path(folder)
            key = str(path).lower()
            if key in seen or not path.exists() or not path.is_dir():
                continue
            out.append(str(path))
            seen.add(key)
            if len(out) >= self.MAX_RECENT_FOLDERS:
                break
        return out

    def _settings_folder_value(self, key: str) -> str:
        value = self.settings.value(key, "")
        return str(value) if value is not None else ""

    def _remember_data_folder(self, folder: str) -> None:
        path = Path(folder)
        if not path.exists() or not path.is_dir():
            return
        folder_text = str(path)
        parent_text = str(path.parent)
        self.settings.setValue(self.SETTINGS_LAST_DATA_FOLDER, folder_text)
        self.settings.setValue(self.SETTINGS_LAST_PARENT_FOLDER, parent_text)
        recent: list[str] = [folder_text]
        seen = {folder_text.lower()}
        for existing in self.recent_folders:
            if existing.lower() in seen:
                continue
            if Path(existing).exists() and Path(existing).is_dir():
                recent.append(existing)
                seen.add(existing.lower())
            if len(recent) >= self.MAX_RECENT_FOLDERS:
                break
        self.recent_folders = recent
        self.settings.setValue(self.SETTINGS_RECENT_FOLDERS, self.recent_folders)
        self._populate_recent_folder_combo()

    def _browse_start_folder(self) -> str:
        candidates: list[str] = []
        if self.current_folder:
            candidates.append(str(Path(self.current_folder).parent))
        candidates.extend(
            [
                self._settings_folder_value(self.SETTINGS_LAST_PARENT_FOLDER),
                self.current_folder,
                self._settings_folder_value(self.SETTINGS_LAST_DATA_FOLDER),
                str(Path.home()),
            ]
        )
        for candidate in candidates:
            if candidate and Path(candidate).exists() and Path(candidate).is_dir():
                return candidate
        return str(Path.home())

    def _populate_recent_folder_combo(self) -> None:
        if not hasattr(self, "recent_folder_combo"):
            return
        old = self.recent_folder_combo.blockSignals(True)
        try:
            self.recent_folder_combo.clear()
            self.recent_folder_combo.addItem("Recent folders", "")
            current_idx = 0
            for folder in self.recent_folders:
                path = Path(folder)
                label = path.name or folder
                if path.parent.name:
                    label = f"{label}  ({path.parent.name})"
                self.recent_folder_combo.addItem(label, folder)
                idx = self.recent_folder_combo.count() - 1
                self.recent_folder_combo.setItemData(idx, folder, Qt.ToolTipRole)
                if self.current_folder and folder.lower() == self.current_folder.lower():
                    current_idx = idx
            self.recent_folder_combo.setCurrentIndex(current_idx)
            self.recent_folder_combo.setEnabled(bool(self.recent_folders))
        finally:
            self.recent_folder_combo.blockSignals(old)

    def _set_current_folder(self, folder: str, *, remember: bool = True, refresh: bool = True) -> bool:
        path = Path(folder)
        if not path.exists() or not path.is_dir():
            self._show_error(f"Folder does not exist: {folder}")
            return False
        if str(path).lower() != self.current_folder.lower():
            self._invalidate_export_move_sources()
        self.current_folder = str(path)
        self.folder_edit.setText(self.current_folder)
        self._watch_current_folder()
        if remember:
            self._remember_data_folder(self.current_folder)
        else:
            self._populate_recent_folder_combo()
        if refresh:
            self._refresh_file_lists()
        return True

    def _watch_current_folder(self) -> None:
        watched = list(self.folder_watcher.directories())
        if watched:
            self.folder_watcher.removePaths(watched)
        self._watched_folder = ""
        if not self.current_folder:
            return
        path = Path(self.current_folder)
        if not path.exists() or not path.is_dir():
            return
        if self.folder_watcher.addPath(str(path)):
            self._watched_folder = str(path)

    def _on_watched_folder_changed(self, folder: str) -> None:
        if self.current_folder and str(Path(folder)) == str(Path(self.current_folder)):
            self.folder_refresh_timer.start()

    def _refresh_watched_folder(self) -> None:
        if not self.current_folder:
            return
        path = Path(self.current_folder)
        if not path.exists() or not path.is_dir():
            self._status("Data source unavailable; choose a folder.")
            self._watch_current_folder()
            return
        if self._load_in_progress:
            self.folder_refresh_timer.start()
            return
        self._refresh_file_lists(auto=True)
        if self.current_folder and self.current_folder not in self.folder_watcher.directories():
            self._watch_current_folder()

    def _restore_last_folder(self) -> None:
        self._populate_recent_folder_combo()
        folder = self._settings_folder_value(self.SETTINGS_LAST_DATA_FOLDER)
        if folder and Path(folder).exists() and Path(folder).is_dir():
            self._set_current_folder(folder, remember=False, refresh=True)

    def _apply_initial_geometry(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1320, 820)
            return
        available = screen.availableGeometry()
        width = min(available.width(), max(self.minimumWidth(), int(available.width() * 0.88)))
        height = min(available.height(), max(self.minimumHeight(), int(available.height() * 0.88)))
        self.resize(width, height)
        self.move(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
        )

    def _drop_has_csv(self, mime: QMimeData) -> bool:
        if not mime.hasUrls():
            return False
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir() or path.suffix.lower() in {".csv", ".xlsx", ".dat"}:
                return True
        return False

    def _set_drop_highlight(self, on: bool) -> None:
        if on:
            self.folder_edit.setStyleSheet("border: 1px solid #2F80ED; background-color: #EAF3FF;")
            if not self.current_folder:
                self.folder_edit.setPlaceholderText("Drop to set folder")
            return
        self.folder_edit.setStyleSheet("")
        self.folder_edit.setPlaceholderText(self._folder_placeholder_text)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        source = event.source()
        if isinstance(source, QWidget) and (source is self or self.isAncestorOf(source)):
            event.ignore()
            return
        if self._drop_has_csv(event.mimeData()):
            self._set_drop_highlight(True)
            event.acceptProposedAction()
            return
        self._set_drop_highlight(False)
        if event.mimeData().hasUrls():
            self._status("Drop ignored: only .csv, .xlsx, .dat files or folders are supported")
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        source = event.source()
        if isinstance(source, QWidget) and (source is self or self.isAncestorOf(source)):
            event.ignore()
            return
        if self._drop_has_csv(event.mimeData()):
            self._set_drop_highlight(True)
            event.acceptProposedAction()
            return
        self._set_drop_highlight(False)
        event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self._set_drop_highlight(False)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        self._set_drop_highlight(False)
        source = event.source()
        if isinstance(source, QWidget) and (source is self or self.isAncestorOf(source)):
            event.ignore()
            return
        if not self._drop_has_csv(event.mimeData()):
            self._status("Drop ignored: only .csv, .xlsx, .dat files or folders are supported")
            event.ignore()
            return
        paths: list[Path] = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            raw = url.toLocalFile()
            if raw:
                paths.append(Path(raw))
        if self._handle_dropped_files(paths):
            event.acceptProposedAction()
            return
        event.ignore()

    def _handle_dropped_files(self, paths: List[Path]) -> bool:
        self._set_drop_highlight(False)
        if self._load_in_progress:
            self._status("Drop ignored: a load is already in progress")
            return False
        valid_paths = [path for path in paths if path.exists()]
        if not valid_paths:
            self._status("Drop ignored: no valid local files were found")
            return False

        folders = [path for path in valid_paths if path.is_dir()]
        files = [path for path in valid_paths if path.is_file()]
        data_files = [path for path in files if path.suffix.lower() in {".csv", ".xlsx", ".dat"}]
        ignored_count = len(files) - len(data_files)

        if folders and data_files:
            self._status("Drop ignored: drop either one folder or data files from one folder")
            return False
        if len(folders) > 1:
            self._status("Drop ignored: only one folder can be dropped at a time")
            return False
        if folders:
            folder = str(folders[0])
            if not self._set_current_folder(folder):
                return False
            self._status(f"Folder set: {folders[0].name} — {len(self.available_map_files)} data files found")
            return True

        if not data_files:
            if ignored_count > 0:
                self._status("Drop ignored: only .csv, .xlsx, or .dat files are supported")
            else:
                self._status("Drop ignored: no supported files detected")
            return False

        parent_folders = list(dict.fromkeys(str(path.parent) for path in data_files))
        if len(parent_folders) != 1:
            self._status("Drop ignored: all files must be from the same folder")
            return False

        if not self._set_current_folder(parent_folders[0]):
            return False

        dropped_names = list(dict.fromkeys(path.name for path in data_files))
        selected_names = [name for name in dropped_names if name in self.available_map_files]
        if not selected_names:
            self._status("Drop ignored: dropped data files were not found in the selected folder")
            return False

        self.pl_files.clearSelection()
        pl_matches = self.pl_files.findItems(selected_names[0], Qt.MatchExactly)
        if pl_matches:
            pl_matches[0].setSelected(True)

        self.drr_selected_files = list(selected_names)
        self._update_drr_selection_labels()
        self._power_refresh_groups()
        self._mcd_refresh_sources()
        self._shg_refresh_sources()
        self._restore_list_selection(self.shg_files, [selected_names[0]])

        if len(selected_names) == 1:
            extra = " Ignored 1 unsupported file." if ignored_count == 1 else f" Ignored {ignored_count} unsupported files." if ignored_count else ""
            self._status(f"Dropped: {selected_names[0]} — select a tab and press Load.{extra}")
            return True

        extra = ""
        if ignored_count == 1:
            extra = " Ignored 1 unsupported file."
        elif ignored_count > 1:
            extra = f" Ignored {ignored_count} unsupported files."
        self._status(f"Dropped: {len(selected_names)} files from {Path(self.current_folder).name} — select a tab and press Load.{extra}")
        return True

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Horizontal)
        left = self._build_left_panel()
        right = self._build_plot_panel()
        self.left_panel = left
        self.workspace_splitter = splitter
        left.setFixedWidth(UI_METRICS["left_width"])
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([UI_METRICS["left_width"], 980])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(
            UI_METRICS["main_margin"],
            UI_METRICS["main_margin"],
            UI_METRICS["main_margin"],
            UI_METRICS["main_margin"],
        )
        navigation = QFrame()
        navigation.setObjectName("workflowNavigation")
        navigation.setStyleSheet(
            "QFrame#workflowNavigation { background: #f7f7f9; border: 1px solid #e2e2e7; border-radius: 7px; }"
            "QTabBar::tab { min-width: 76px; padding: 7px 13px; border: none; color: #515154; }"
            "QTabBar::tab:selected { color: #0071e3; font-weight: 600; border-bottom: 2px solid #0071e3; }"
            "QTabBar::tab:hover { color: #0071e3; }"
        )
        navigation_layout = QHBoxLayout(navigation)
        navigation_layout.setContentsMargins(6, 2, 6, 2)
        navigation_layout.setSpacing(4)
        self.sidebar_toggle_btn = QToolButton()
        self.sidebar_toggle_btn.setText("☰ Controls")
        self.sidebar_toggle_btn.setCheckable(True)
        self.sidebar_toggle_btn.setChecked(True)
        self.sidebar_toggle_btn.setToolTip("Show or hide the contextual controls sidebar (Ctrl+B)")
        navigation_layout.addWidget(self.sidebar_toggle_btn)
        self.workflow_tabs = QTabBar()
        self.workflow_tabs.setExpanding(True)
        self.workflow_tabs.setUsesScrollButtons(False)
        for index in range(self.tabs.count()):
            tab_index = self.workflow_tabs.addTab(self.tabs.tabText(index))
            self.workflow_tabs.setTabToolTip(tab_index, self.tabs.tabToolTip(index))
        navigation_layout.addWidget(self.workflow_tabs, 1)
        self.workflow_tabs.currentChanged.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self.workflow_tabs.setCurrentIndex)
        self.sidebar_toggle_btn.toggled.connect(self._set_sidebar_visible)
        layout.addWidget(navigation)
        layout.addWidget(splitter)

        sb = QStatusBar()
        self._status_progress = QProgressBar()
        self._status_progress.setRange(0, 0)  # indeterminate spinner
        self._status_progress.setMaximumWidth(120)
        self._status_progress.setVisible(False)
        self._status_progress.setToolTip("Loading data in background…")
        sb.addPermanentWidget(self._status_progress)
        self.setStatusBar(sb)
        self.statusBar().showMessage("Ready")
        self._update_status_button = QPushButton()
        self._update_status_button.setFlat(True)
        self._update_status_button.setCursor(Qt.PointingHandCursor)
        self._update_status_button.setStyleSheet(
            "QPushButton { color: #0071e3; font-weight: 600; }"
        )
        self._update_status_button.setVisible(False)
        self._update_status_button.clicked.connect(self._on_update_status_clicked)
        self.statusBar().addPermanentWidget(self._update_status_button)
        self._build_log_dock()
        self._build_results_dock()
        self._build_menu_and_toolbar()

    def _set_sidebar_visible(self, visible: bool) -> None:
        """Show or hide the contextual inspector without changing the active workflow."""
        if not hasattr(self, "left_panel"):
            return
        self.left_panel.setVisible(bool(visible))
        if hasattr(self, "sidebar_toggle_btn"):
            blocked = self.sidebar_toggle_btn.blockSignals(True)
            self.sidebar_toggle_btn.setChecked(bool(visible))
            self.sidebar_toggle_btn.blockSignals(blocked)
            self.sidebar_toggle_btn.setText("☰ Controls" if visible else "☰ Show controls")

    def _build_left_panel(self) -> QWidget:
        box = QWidget()
        box.setFixedWidth(UI_METRICS["left_width"])
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(UI_METRICS["row_spacing"])

        # Workflow step banner
        steps_label = QLabel("Select  ›  Load  ›  Plot  ›  Export")
        steps_label.setAlignment(Qt.AlignCenter)
        steps_label.setStyleSheet(
            "QLabel { background: #f5f5f7; color: #6e6e73; border-radius: 6px; "
            "padding: 5px 10px; font-size: 11px; font-weight: 400; "
            "border: 1px solid #e5e5ea; letter-spacing: 0.2px; }"
        )
        layout.addWidget(steps_label)

        # Data source section
        folder_box = QGroupBox("")
        folder_grid = QGridLayout(folder_box)
        folder_grid.setContentsMargins(8, 6, 8, 8)
        folder_grid.setHorizontalSpacing(8)
        folder_grid.setVerticalSpacing(8)
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText("No folder selected — click Browse or Open File")
        self.folder_edit.setToolTip("Current working folder for data files")
        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.setToolTip("Select a folder containing CSV data files")
        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.setToolTip("Open a single CSV file and set its folder as the working directory")
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Re-scan the current folder for new or changed CSV files")
        self.recent_folder_combo = QComboBox()
        self.recent_folder_combo.setToolTip("Switch to a recently used data folder")
        self._style_combo_popup(self.recent_folder_combo)
        folder_grid.addWidget(self.folder_edit, 0, 0, 1, 3)
        folder_grid.addWidget(self.browse_btn, 1, 0)
        folder_grid.addWidget(self.open_file_btn, 1, 1)
        folder_grid.addWidget(self.refresh_btn, 1, 2)
        folder_grid.addWidget(QLabel("Recent"), 2, 0)
        folder_grid.addWidget(self.recent_folder_combo, 2, 1, 1, 2)
        layout.addWidget(self._make_expander("Data Source", folder_box, expanded=False))

        self.tabs = QTabWidget()
        self.tabs.tabBar().setExpanding(True)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.tabs.tabBar().setUsesScrollButtons(False)
        for key, label, tooltip, page in (
            ("pl", "PL", "Photoluminescence", self._build_pl_tab()),
            ("drr", "DRR", "Differential reflectance", self._build_drr_tab()),
            ("cmp", "Compare", "Compare measurement channels", self._build_compare_tab()),
            ("power", "Power", "Power Dependent", self._build_power_tab()),
            ("mcd", "MCD", "Magnetic circular dichroism", self._build_mcd_tab()),
            ("shg", "SHG", "SHG Processing", self._build_shg_tab()),
            ("tools", "Tools", "Log / Tools", self._build_tools_tab()),
        ):
            index = self.tabs.addTab(self._make_scrollable_tab(page, key), label)
            self.tabs.setTabToolTip(index, tooltip)
        # Workflow navigation is rendered above the entire workspace.  This
        # QTabWidget remains the page stack so existing tab-specific code and
        # keyboard behavior continue to work.
        self.tabs.tabBar().hide()
        layout.addWidget(self.tabs, 1)
        return box

    def _make_scrollable_tab(self, page: QWidget, key: str) -> QScrollArea:
        """Keep the tab bar visible while allowing tall control pages to scroll."""
        scroll = QScrollArea()
        scroll.setObjectName(f"{key}_tab_scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # MCD has several detailed controls, but it must still fit the fixed
        # left sidebar.  Ignored lets the form/layout elide controls instead
        # of imposing its widest child as a horizontal minimum.
        horizontal_policy = QSizePolicy.Ignored if key == "mcd" else QSizePolicy.Expanding
        page.setSizePolicy(horizontal_policy, QSizePolicy.Maximum)
        scroll.setWidget(page)
        setattr(self, f"{key}_tab_scroll", scroll)
        return scroll

    def _build_common_range_grid(
        self, prefix: str, default_cmap: str
    ) -> tuple[QGridLayout, Dict[str, QDoubleSpinBox], QCheckBox, QCheckBox, QComboBox, Dict[str, QCheckBox]]:
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        def spin() -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setDecimals(4)
            s.setRange(-1e12, 1e12)
            s.setSingleStep(0.1)
            s.setMaximumWidth(130)
            return s

        spins = {
            "vmin": spin(),
            "vmax": spin(),
            "xmin": spin(),
            "xmax": spin(),
            "ymin": spin(),
            "ymax": spin(),
            "gate": spin(),
        }
        fix_checks = {
            "vmin": QCheckBox("F"),
            "vmax": QCheckBox("F"),
            "xmin": QCheckBox("F"),
            "xmax": QCheckBox("F"),
            "ymin": QCheckBox("F"),
            "ymax": QCheckBox("F"),
        }
        spins["vmin"].setToolTip("Lower color scale bound")
        spins["vmax"].setToolTip("Upper color scale bound")
        spins["xmin"].setToolTip("Left energy axis bound")
        spins["xmax"].setToolTip("Right energy axis bound")
        spins["ymin"].setToolTip("Lower gate axis bound")
        spins["ymax"].setToolTip("Upper gate axis bound")
        spins["gate"].setToolTip("Gate value for spectrum extraction")
        fix_checks["vmin"].setToolTip("Fix vmin when loading/updating files")
        fix_checks["vmax"].setToolTip("Fix vmax when loading/updating files")
        fix_checks["xmin"].setToolTip("Fix xmin when loading/updating files")
        fix_checks["xmax"].setToolTip("Fix xmax when loading/updating files")
        fix_checks["ymin"].setToolTip("Fix ymin when loading/updating files")
        fix_checks["ymax"].setToolTip("Fix ymax when loading/updating files")

        def add_pair(row: int, col: int, text: str, widget: QWidget) -> None:
            label = QLabel(text)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setFixedWidth(50)
            base = col * 2
            grid.addWidget(label, row, base)
            grid.addWidget(widget, row, base + 1)

        def spin_with_fix(k: str) -> QWidget:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            h.addWidget(spins[k])
            h.addWidget(fix_checks[k])
            h.addStretch(1)
            return row

        add_pair(0, 0, "vmin", spin_with_fix("vmin"))
        add_pair(0, 1, "vmax", spin_with_fix("vmax"))
        add_pair(1, 0, "xmin", spin_with_fix("xmin"))
        add_pair(1, 1, "xmax", spin_with_fix("xmax"))
        add_pair(2, 0, "ymin", spin_with_fix("ymin"))
        add_pair(2, 1, "ymax", spin_with_fix("ymax"))
        add_pair(3, 0, "gate", spins["gate"])

        log_chk = QCheckBox("Log Scale")
        clip_chk = QCheckBox("Clip Outliers")
        clip_chk.setChecked(True)
        log_chk.setToolTip("Use logarithmic color normalization")
        clip_chk.setToolTip("Clip values above vmax for cleaner contrast")
        cmap = QComboBox()
        cmap.addItem("Default", default_cmap)
        for cmap_id, label in CUSTOM_COLORMAPS:
            cmap.addItem(label, cmap_id)
        for name in STANDARD_COLORMAPS:
            cmap.addItem(name, name)
        cmap.setCurrentIndex(0)
        cmap.setProperty("default_cmap", default_cmap)
        cmap.setToolTip("Colormap for heatmap rendering")
        self._style_combo_popup(cmap)

        flags = QWidget()
        flags_layout = QHBoxLayout(flags)
        flags_layout.setContentsMargins(0, 0, 0, 0)
        flags_layout.setSpacing(10)
        flags_layout.addWidget(log_chk)
        flags_layout.addWidget(clip_chk)
        flags_layout.addStretch(1)
        grid.addWidget(flags, 3, 2, 1, 2)

        setattr(self, f"{prefix}_spins", spins)
        setattr(self, f"{prefix}_log_chk", log_chk)
        setattr(self, f"{prefix}_clip_chk", clip_chk)
        setattr(self, f"{prefix}_cmap", cmap)
        setattr(self, f"{prefix}_fix_checks", fix_checks)
        self._build_split_scale_controls(prefix)
        return grid, spins, log_chk, clip_chk, cmap, fix_checks

    def _build_split_scale_controls(self, prefix: str) -> None:
        toggle = QCheckBox("Use split color scale")
        toggle.setToolTip("Use independent color limits on the two sides of x0.")

        def split_spin() -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e12, 1e12)
            spin.setSingleStep(0.01)
            spin.setFixedWidth(UI_METRICS["spin_w"] + 12)
            return spin

        split_spins = {
            "x0": split_spin(),
            "left_vmin": split_spin(),
            "left_vmax": split_spin(),
            "right_vmin": split_spin(),
            "right_vmax": split_spin(),
        }
        split_spins["x0"].setToolTip("Requested boundary between the two x-dependent color scales.")
        split_spins["left_vmin"].setToolTip("Color minimum from xmin to x0.")
        split_spins["left_vmax"].setToolTip("Color maximum from xmin to x0.")
        split_spins["right_vmin"].setToolTip("Color minimum from x0 to xmax.")
        split_spins["right_vmax"].setToolTip("Color maximum from x0 to xmax.")

        split_fix_checks = {
            "x0": QCheckBox("Fix"),
            "left_vmin": QCheckBox("Fix"),
            "left_vmax": QCheckBox("Fix"),
            "right_vmin": QCheckBox("Fix"),
            "right_vmax": QCheckBox("Fix"),
        }
        for key, check in split_fix_checks.items():
            if key == "x0":
                check.setToolTip("Keep the split boundary when data or visible x limits change.")
                continue
            region = "left" if key.startswith("left") else "right"
            bound = "minimum" if key.endswith("vmin") else "maximum"
            check.setToolTip(
                f"Keep the {region}-region color {bound} when {region.title()} Auto is used."
            )

        panel = QGroupBox("Two X-Region Color Limits")
        grid = QGridLayout(panel)
        grid.setContentsMargins(6, 8, 6, 6)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(5)
        show_boundary = QCheckBox("Show boundary")
        show_boundary.setChecked(True)
        auto_left = QToolButton()
        auto_left.setText("Auto Left")
        auto_left.setToolTip("Automatically set unlocked limits from data between xmin and x0.")
        auto_right = QToolButton()
        auto_right.setText("Auto Right")
        auto_right.setToolTip("Automatically set unlocked limits from data between x0 and xmax.")

        # Region titles get their own rows so the controls remain readable at
        # the sidebar's minimum width instead of clipping its right edge.
        grid.addWidget(QLabel("Split position (x0)"), 0, 0, 1, 2)
        grid.addWidget(split_spins["x0"], 0, 2, 1, 2)
        grid.addWidget(split_fix_checks["x0"], 0, 4)
        grid.addWidget(show_boundary, 0, 5)

        left_title = QLabel("Region 1: xmin → x0")
        left_title.setStyleSheet("font-weight: 600;")
        grid.addWidget(left_title, 1, 0, 1, 4)
        grid.addWidget(auto_left, 1, 4, 1, 2)
        grid.addWidget(QLabel("vmin"), 2, 0)
        grid.addWidget(split_spins["left_vmin"], 2, 1, 1, 2)
        grid.addWidget(split_fix_checks["left_vmin"], 2, 3)
        grid.addWidget(QLabel("vmax"), 3, 0)
        grid.addWidget(split_spins["left_vmax"], 3, 1, 1, 2)
        grid.addWidget(split_fix_checks["left_vmax"], 3, 3)

        right_title = QLabel("Region 2: x0 → xmax")
        right_title.setStyleSheet("font-weight: 600;")
        grid.addWidget(right_title, 4, 0, 1, 4)
        grid.addWidget(auto_right, 4, 4, 1, 2)
        grid.addWidget(QLabel("vmin"), 5, 0)
        grid.addWidget(split_spins["right_vmin"], 5, 1, 1, 2)
        grid.addWidget(split_fix_checks["right_vmin"], 5, 3)
        grid.addWidget(QLabel("vmax"), 6, 0)
        grid.addWidget(split_spins["right_vmax"], 6, 1, 1, 2)
        grid.addWidget(split_fix_checks["right_vmax"], 6, 3)
        grid.setColumnStretch(4, 1)
        panel.setVisible(False)

        setattr(self, f"{prefix}_split_scale_chk", toggle)
        setattr(self, f"{prefix}_split_scale_panel", panel)
        setattr(self, f"{prefix}_split_spins", split_spins)
        setattr(self, f"{prefix}_split_fix_checks", split_fix_checks)
        setattr(self, f"{prefix}_split_boundary_chk", show_boundary)
        setattr(self, f"{prefix}_split_auto_left_btn", auto_left)
        setattr(self, f"{prefix}_split_auto_right_btn", auto_right)

    def _build_y_axis_controls(self, prefix: str) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        combo = QComboBox()
        combo.addItems(["Auto / Default", "TG", "BG", "Bias", "Advanced..."])
        combo.setToolTip("Choose how the plot y-axis is derived from gate variables.")
        self._style_combo_popup(combo)

        advanced = QGroupBox("Advanced Linear Combination")
        advanced_form = QFormLayout(advanced)
        advanced_form.setContentsMargins(6, 6, 6, 6)
        advanced_form.setHorizontalSpacing(6)
        advanced_form.setVerticalSpacing(4)

        def coeff_spin(default_value: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e6, 1e6)
            spin.setSingleStep(0.1)
            spin.setValue(default_value)
            spin.setFixedWidth(100)
            return spin

        a_spin = coeff_spin(1.0)
        b_spin = coeff_spin(-1.0)
        c_spin = coeff_spin(0.0)
        equation = QLabel("")
        label_preview = QLabel("")
        equation.setWordWrap(True)
        label_preview.setWordWrap(True)

        advanced_form.addRow("a (TG)", a_spin)
        advanced_form.addRow("b (BG)", b_spin)
        advanced_form.addRow("c", c_spin)
        advanced_form.addRow("Equation", equation)
        advanced_form.addRow("Label", label_preview)
        advanced.setVisible(False)

        layout.addWidget(combo)
        layout.addWidget(advanced)

        setattr(self, f"{prefix}_yaxis_combo", combo)
        setattr(self, f"{prefix}_yaxis_advanced_box", advanced)
        setattr(self, f"{prefix}_yaxis_a_spin", a_spin)
        setattr(self, f"{prefix}_yaxis_b_spin", b_spin)
        setattr(self, f"{prefix}_yaxis_c_spin", c_spin)
        setattr(self, f"{prefix}_yaxis_equation_lbl", equation)
        setattr(self, f"{prefix}_yaxis_label_lbl", label_preview)
        return host

    def _build_pl_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        files = QGroupBox("File")
        files_layout = QVBoxLayout(files)
        files_layout.setContentsMargins(6, 6, 6, 6)
        self.pl_files = QListWidget()
        self.pl_files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.pl_files.setMinimumHeight(60)
        self.pl_files.setMaximumHeight(120)
        self.pl_files.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.pl_files.setToolTip("Select a PL measurement file to load")
        self.pl_files.itemSelectionChanged.connect(self._on_pl_selection_changed)
        files_layout.addWidget(self.pl_files)
        layout.addWidget(self._make_expander("Measurement File", files, expanded=True))

        params = QGroupBox("Plot Options")
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 4)
        params_layout.setSpacing(4)
        cfg = QFormLayout()
        cfg.setHorizontalSpacing(6)
        cfg.setVerticalSpacing(4)
        _grid, spins, _, _, cmap, fix_checks = self._build_common_range_grid("pl", "turbo")
        self.pl_yaxis_controls = self._build_y_axis_controls("pl")
        _pl_yc_row = QWidget()
        _pl_yc_h = QHBoxLayout(_pl_yc_row)
        _pl_yc_h.setContentsMargins(0, 0, 0, 0)
        _pl_yc_h.setSpacing(6)
        _pl_yc_h.addWidget(self.pl_yaxis_combo, 1)
        _pl_yc_h.addWidget(QLabel("Cmap"))
        _pl_yc_h.addWidget(cmap)
        cfg.addRow("Y-axis", _pl_yc_row)
        cfg.addRow("", self.pl_yaxis_advanced_box)
        self.pl_dat_yaxis_label_edit = QLineEdit()
        self.pl_dat_yaxis_label_edit.setPlaceholderText("Custom Y-axis label")
        self.pl_dat_yaxis_unit_edit = QLineEdit()
        self.pl_dat_yaxis_unit_edit.setPlaceholderText("Optional unit")
        dat_y_row = QWidget()
        dat_y_layout = QHBoxLayout(dat_y_row)
        dat_y_layout.setContentsMargins(0, 0, 0, 0)
        dat_y_layout.setSpacing(6)
        dat_y_layout.addWidget(self.pl_dat_yaxis_label_edit)
        dat_y_layout.addWidget(self.pl_dat_yaxis_unit_edit)
        cfg.addRow("Imported DAT Y", dat_y_row)
        self.pl_dat_yaxis_label_edit.setVisible(False)
        self.pl_dat_yaxis_unit_edit.setVisible(False)
        params_layout.addLayout(cfg)
        for s in spins.values():
            s.setFixedWidth(UI_METRICS["spin_w"])
            s.setFixedHeight(UI_METRICS["input_h"])

        self.pl_auto_v_btn = QToolButton()
        self.pl_auto_x_btn = QToolButton()
        self.pl_auto_y_btn = QToolButton()

        basic = QGroupBox("Axis Ranges")
        basic_form = QFormLayout(basic)
        basic_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        basic_form.setHorizontalSpacing(4)
        basic_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        basic_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        basic_form.addRow(
            "vmin / vmax",
            self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.pl_auto_v_btn, "Auto V"),
        )
        basic_form.addRow("Color scale", self.pl_split_scale_chk)
        basic_form.addRow(self.pl_split_scale_panel)
        basic_form.addRow(
            "xmin / xmax",
            self._make_axis_range_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.pl_auto_x_btn, "Auto X"),
        )
        basic_form.addRow(
            "ymin / ymax",
            self._make_axis_range_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.pl_auto_y_btn, "Auto Y"),
        )
        basic_form.addRow("Cursor Gate", spins["gate"])
        flags = QWidget()
        flags_h = QHBoxLayout(flags)
        flags_h.setContentsMargins(0, 0, 0, 0)
        flags_h.setSpacing(10)
        flags_h.addWidget(self.pl_log_chk)
        flags_h.addWidget(self.pl_clip_chk)
        flags_h.addStretch(1)
        basic_form.addRow("Scale / Clip", flags)
        self._set_form_label_width(basic_form, UI_METRICS["label_col_width"])

        params_layout.addWidget(self._make_expander("Manual plot ranges", basic, expanded=False))

        analysis = QGroupBox("Spectrum Analysis Controls")
        analysis_form = QFormLayout(analysis)
        analysis_form.setContentsMargins(6, 6, 6, 6)
        analysis_form.setHorizontalSpacing(6)
        analysis_form.setVerticalSpacing(4)
        self.pl_peak_find_btn = QPushButton("Find Peaks")
        self.pl_peak_show_chk = QCheckBox("Show Peaks")
        self.pl_peak_show_chk.setChecked(True)
        self.pl_peak_mode_combo = QComboBox()
        self.pl_peak_mode_combo.addItems(["Peaks", "Dips"])
        row1 = QWidget()
        row1h = QHBoxLayout(row1)
        row1h.setContentsMargins(0, 0, 0, 0)
        row1h.setSpacing(6)
        row1h.addWidget(self.pl_peak_find_btn)
        row1h.addWidget(self.pl_peak_mode_combo)
        row1h.addWidget(self.pl_peak_show_chk)
        row1h.addStretch(1)
        analysis_form.addRow("", row1)
        self.pl_peak_prom_spin = QDoubleSpinBox()
        self.pl_peak_prom_spin.setDecimals(3)
        self.pl_peak_prom_spin.setRange(0.0, 1.0)
        self.pl_peak_prom_spin.setSingleStep(0.01)
        self.pl_peak_prom_spin.setValue(0.05)
        self.pl_peak_dist_spin = QSpinBox()
        self.pl_peak_dist_spin.setRange(1, 500)
        self.pl_peak_dist_spin.setValue(5)
        self.pl_peak_max_spin = QSpinBox()
        self.pl_peak_max_spin.setRange(1, 20)
        self.pl_peak_max_spin.setValue(6)
        row2 = QWidget()
        row2h = QHBoxLayout(row2)
        row2h.setContentsMargins(0, 0, 0, 0)
        row2h.setSpacing(6)
        row2h.addWidget(QLabel("Prom"))
        row2h.addWidget(self.pl_peak_prom_spin)
        row2h.addWidget(QLabel("Dist"))
        row2h.addWidget(self.pl_peak_dist_spin)
        row2h.addWidget(QLabel("Top"))
        row2h.addWidget(self.pl_peak_max_spin)
        analysis_form.addRow("", row2)
        self.pl_fit_btn = QPushButton("Fit Multi-Lorentz")
        self.pl_fit_clear_btn = QPushButton("Clear Fit")
        self.pl_fit_show_chk = QCheckBox("Show Fit")
        self.pl_fit_show_chk.setChecked(True)
        self.pl_fit_n_spin = QSpinBox()
        self.pl_fit_n_spin.setRange(1, 8)
        self.pl_fit_n_spin.setValue(3)
        row3 = QWidget()
        row3h = QHBoxLayout(row3)
        row3h.setContentsMargins(0, 0, 0, 0)
        row3h.setSpacing(6)
        row3h.addWidget(self.pl_fit_btn)
        row3h.addWidget(self.pl_fit_clear_btn)
        row3h.addWidget(self.pl_fit_show_chk)
        row3h.addWidget(QLabel("N"))
        row3h.addWidget(self.pl_fit_n_spin)
        row3h.addStretch(1)
        analysis_form.addRow("", row3)
        self.pl_fit_status = QLabel("")
        self.pl_fit_status.setStyleSheet("QLabel { color: #0071e3; font-size: 10px; }")
        analysis_form.addRow("", self.pl_fit_status)
        self.pl_analysis_text = QPlainTextEdit()
        self.pl_analysis_text.setReadOnly(True)
        self.pl_analysis_text.setMinimumHeight(60)
        self.pl_analysis_text.setMaximumHeight(100)
        self.pl_analysis_text.setPlaceholderText("Peak/fit results will appear here after detection.")
        analysis_form.addRow("", self.pl_analysis_text)
        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addWidget(self._make_expander("Spectrum Analysis", analysis, expanded=False))
        layout.addStretch(1)
        return tab

    def _build_drr_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        files = QGroupBox("Files")
        files_layout = QVBoxLayout(files)
        files_layout.setContentsMargins(6, 6, 6, 6)
        files_layout.setSpacing(6)

        meas_row = QWidget()
        meas_grid = QGridLayout(meas_row)
        meas_grid.setContentsMargins(0, 0, 0, 0)
        meas_grid.setHorizontalSpacing(6)
        meas_grid.setVerticalSpacing(4)
        self.drr_measurement_summary = QLabel("Measurement: 0 files")
        self.drr_measurement_summary.setWordWrap(True)
        self.drr_edit_measurements_btn = QPushButton("Select...")
        self.drr_edit_measurements_btn.setFixedHeight(30)
        self.drr_edit_measurements_btn.setMaximumWidth(92)
        self.drr_clear_measurements_btn = QPushButton("Clear")
        self.drr_clear_measurements_btn.setFixedHeight(30)
        self.drr_clear_measurements_btn.setMaximumWidth(72)
        meas_grid.addWidget(self.drr_measurement_summary, 0, 0, 1, 3)
        meas_grid.setColumnStretch(0, 1)
        meas_grid.addWidget(self.drr_edit_measurements_btn, 1, 1)
        meas_grid.addWidget(self.drr_clear_measurements_btn, 1, 2)
        files_layout.addWidget(meas_row)

        base_row = QWidget()
        self.drr_external_baseline_row = base_row
        base_grid = QGridLayout(base_row)
        base_grid.setContentsMargins(0, 0, 0, 0)
        base_grid.setHorizontalSpacing(6)
        base_grid.setVerticalSpacing(4)
        self.drr_baseline_summary = QLabel("Baselines: 0 files")
        self.drr_baseline_summary.setWordWrap(True)
        self.drr_edit_baselines_btn = QPushButton("Select...")
        self.drr_edit_baselines_btn.setFixedHeight(30)
        self.drr_edit_baselines_btn.setMaximumWidth(92)
        self.drr_baseline_autofind_btn = QPushButton("Auto Find...")
        self.drr_baseline_autofind_btn.setFixedHeight(30)
        self.drr_baseline_autofind_btn.setMaximumWidth(104)
        base_grid.addWidget(self.drr_baseline_summary, 0, 0, 1, 3)
        base_grid.setColumnStretch(0, 1)
        base_grid.addWidget(self.drr_edit_baselines_btn, 1, 1)
        base_grid.addWidget(self.drr_baseline_autofind_btn, 1, 2)
        files_layout.addWidget(base_row)

        self.drr_baseline_combine_combo = QComboBox()
        self.drr_baseline_combine_combo.addItems(
            [
                "Last frame of each baseline file",
                "First frame of each baseline file",
                "Average all baseline files",
            ]
        )
        self.drr_baseline_combine_combo.setMaximumWidth(320)
        self._style_combo_popup(self.drr_baseline_combine_combo)
        files_layout.addWidget(self.drr_baseline_combine_combo)
        self.drr_external_baseline_row.setVisible(False)
        self.drr_baseline_combine_combo.setVisible(False)
        layout.addWidget(self._make_expander("Data", files, expanded=True))

        params = QGroupBox("Plot Options")
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 4)
        params_layout.setSpacing(6)

        self.drr_baseline_combo = QComboBox()
        self.drr_baseline_combo.addItems(["Self (last frame)", "Self (first frame)", "External"])
        self.drr_baseline_combo.setToolTip("Background strategy for DRR")
        self._style_combo_popup(self.drr_baseline_combo)
        self.drr_derivative_combo = QComboBox()
        self.drr_derivative_combo.addItems(["None", "dE", "d2E"])
        self.drr_derivative_combo.setToolTip("Apply derivative transform to DRR")
        self._style_combo_popup(self.drr_derivative_combo)
        _grid, spins, log_chk, clip_chk, cmap, fix_checks = self._build_common_range_grid("drr", "RdBu_r")

        for s in spins.values():
            s.setFixedWidth(UI_METRICS["spin_w"])
            s.setFixedHeight(UI_METRICS["input_h"])
        self.drr_baseline_combo.setMinimumWidth(120)
        self.drr_baseline_combo.setMaximumWidth(180)
        self.drr_baseline_combo.setFixedHeight(UI_METRICS["input_h"])
        self.drr_baseline_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.drr_yaxis_controls = self._build_y_axis_controls("drr")
        cmap.setMinimumWidth(90)
        cmap.setMaximumWidth(115)
        cmap.setFixedHeight(UI_METRICS["input_h"])
        cmap.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.drr_derivative_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.drr_derivative_combo.setMinimumContentsLength(3)
        self.drr_derivative_combo.setFixedWidth(UI_METRICS["deriv_combo_w"])
        self.drr_derivative_combo.setFixedHeight(UI_METRICS["input_h"])
        self.drr_sg_window_spin = QSpinBox()
        self.drr_sg_window_spin.setRange(5, 401)
        self.drr_sg_window_spin.setSingleStep(2)
        self.drr_sg_window_spin.setValue(20)
        self.drr_sg_window_spin.setToolTip("Savitzky-Golay window length (odd).")
        self.drr_sg_window_spin.setFixedWidth(UI_METRICS["spin_w"])
        self.drr_sg_window_spin.setFixedHeight(UI_METRICS["input_h"])
        self.drr_sg_poly_spin = QSpinBox()
        self.drr_sg_poly_spin.setRange(1, 6)
        self.drr_sg_poly_spin.setValue(2)
        self.drr_sg_poly_spin.setToolTip("Savitzky-Golay polynomial order.")
        self.drr_sg_poly_spin.setFixedWidth(UI_METRICS["spin_w"])
        self.drr_sg_poly_spin.setFixedHeight(UI_METRICS["input_h"])
        self.drr_sg_window_spin.setVisible(False)
        self.drr_sg_poly_spin.setVisible(False)

        deriv_row = QWidget()
        deriv_h = QHBoxLayout(deriv_row)
        deriv_h.setContentsMargins(0, 0, 0, 0)
        deriv_h.setSpacing(4)
        deriv_h.addWidget(self.drr_derivative_combo)
        self.drr_sg_window_spin.setPrefix("W ")
        deriv_h.addWidget(self.drr_sg_window_spin)
        self.drr_sg_poly_spin.setPrefix("O ")
        deriv_h.addWidget(self.drr_sg_poly_spin)
        deriv_h.addStretch(1)

        self.drr_auto_v_btn = QToolButton()
        self.drr_auto_x_btn = QToolButton()
        self.drr_auto_y_btn = QToolButton()
        self.drr_center_zero_chk = QCheckBox("Center Zero")
        self.drr_center_zero_chk.setToolTip("When enabled, DRR colormap is centered at zero.")
        self.drr_center_zero_chk.setChecked(False)

        # Config rows outside Axis Ranges — mirrors PL tab structure
        cfg = QFormLayout()
        cfg.setHorizontalSpacing(6)
        cfg.setVerticalSpacing(UI_METRICS["row_spacing"])
        cfg.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        baseline_cmap_row = QWidget()
        baseline_cmap_h = QHBoxLayout(baseline_cmap_row)
        baseline_cmap_h.setContentsMargins(0, 0, 0, 0)
        baseline_cmap_h.setSpacing(6)
        baseline_cmap_h.addWidget(self.drr_baseline_combo, 1)
        baseline_cmap_h.addWidget(QLabel("Cmap"))
        baseline_cmap_h.addWidget(cmap)
        _drr_yc_row = QWidget()
        _drr_yc_h = QHBoxLayout(_drr_yc_row)
        _drr_yc_h.setContentsMargins(0, 0, 0, 0)
        _drr_yc_h.setSpacing(6)
        _drr_yc_h.addWidget(self.drr_yaxis_combo, 1)
        cfg.addRow("DRR Baseline", baseline_cmap_row)
        cfg.addRow("Y-axis", _drr_yc_row)
        cfg.addRow("", self.drr_yaxis_advanced_box)
        cfg.addRow("Derivative / SG", deriv_row)
        self._set_form_label_width(cfg, UI_METRICS["label_col_width"])
        params_layout.addLayout(cfg)

        basic = QGroupBox("Axis Ranges")
        basic_form = QFormLayout(basic)
        basic_form.setContentsMargins(
            4,
            UI_METRICS["group_margin"],
            4,
            UI_METRICS["group_margin"],
        )
        basic_form.setHorizontalSpacing(4)
        basic_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        basic_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        basic_form.addRow(
            "vmin / vmax",
            self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.drr_auto_v_btn, "Auto V"),
        )
        basic_form.addRow("Color scale", self.drr_split_scale_chk)
        basic_form.addRow(self.drr_split_scale_panel)
        basic_form.addRow(
            "xmin / xmax",
            self._make_axis_range_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.drr_auto_x_btn, "Auto X"),
        )
        basic_form.addRow(
            "ymin / ymax",
            self._make_axis_range_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.drr_auto_y_btn, "Auto Y"),
        )
        basic_form.addRow("Cursor Gate", spins["gate"])
        flags = QWidget()
        flags_h = QHBoxLayout(flags)
        flags_h.setContentsMargins(0, 0, 0, 0)
        flags_h.setSpacing(10)
        flags_h.addWidget(log_chk)
        flags_h.addWidget(clip_chk)
        flags_h.addWidget(self.drr_center_zero_chk)
        flags_h.addStretch(1)
        basic_form.addRow("Scale / Clip", flags)
        self._set_form_label_width(basic_form, UI_METRICS["label_col_width"])

        analysis_box = QGroupBox("")
        analysis_form = QFormLayout(analysis_box)
        analysis_form.setContentsMargins(6, 6, 6, 6)
        analysis_form.setHorizontalSpacing(6)
        analysis_form.setVerticalSpacing(4)

        self.drr_peak_show_chk = QCheckBox("Show Peaks")
        self.drr_peak_show_chk.setChecked(True)
        self.drr_peak_find_btn = QPushButton("Find Peaks")
        self.drr_peak_mode_combo = QComboBox()
        self.drr_peak_mode_combo.addItems(["Peaks", "Dips"])
        self.drr_peak_find_btn.setToolTip("Find peaks in current bottom spectrum and overlay on both plots.")
        self.drr_peak_prom_spin = QDoubleSpinBox()
        self.drr_peak_prom_spin.setDecimals(3)
        self.drr_peak_prom_spin.setRange(0.0, 1.0)
        self.drr_peak_prom_spin.setSingleStep(0.01)
        self.drr_peak_prom_spin.setValue(0.05)
        self.drr_peak_prom_spin.setToolTip("Prominence as fraction of visible Y-range.")
        self.drr_peak_dist_spin = QSpinBox()
        self.drr_peak_dist_spin.setRange(1, 500)
        self.drr_peak_dist_spin.setValue(5)
        self.drr_peak_dist_spin.setToolTip("Minimum peak spacing in points.")
        self.drr_peak_max_spin = QSpinBox()
        self.drr_peak_max_spin.setRange(1, 20)
        self.drr_peak_max_spin.setValue(6)
        self.drr_peak_max_spin.setToolTip("Maximum number of strongest peaks to keep.")

        self.drr_fit_show_chk = QCheckBox("Show Fit")
        self.drr_fit_show_chk.setChecked(True)
        self.drr_fit_btn = QPushButton("Fit Multi-Lorentz")
        self.drr_fit_btn.setToolTip("Fit multiple Lorentz peaks to current bottom spectrum.")
        self.drr_fit_clear_btn = QPushButton("Clear Fit")
        self.drr_fit_n_spin = QSpinBox()
        self.drr_fit_n_spin.setRange(1, 8)
        self.drr_fit_n_spin.setValue(3)
        self.drr_fit_n_spin.setToolTip("Number of Lorentz peaks in fit.")
        self.drr_fit_status = QLabel("")

        drr_row1 = QWidget()
        drr_row1h = QHBoxLayout(drr_row1)
        drr_row1h.setContentsMargins(0, 0, 0, 0)
        drr_row1h.setSpacing(6)
        drr_row1h.addWidget(self.drr_peak_find_btn)
        drr_row1h.addWidget(self.drr_peak_mode_combo)
        drr_row1h.addWidget(self.drr_peak_show_chk)
        drr_row1h.addStretch(1)
        analysis_form.addRow("", drr_row1)

        drr_row2 = QWidget()
        drr_row2h = QHBoxLayout(drr_row2)
        drr_row2h.setContentsMargins(0, 0, 0, 0)
        drr_row2h.setSpacing(6)
        drr_row2h.addWidget(QLabel("Prom"))
        drr_row2h.addWidget(self.drr_peak_prom_spin)
        drr_row2h.addWidget(QLabel("Dist"))
        drr_row2h.addWidget(self.drr_peak_dist_spin)
        drr_row2h.addWidget(QLabel("Top"))
        drr_row2h.addWidget(self.drr_peak_max_spin)
        analysis_form.addRow("", drr_row2)

        drr_row3 = QWidget()
        drr_row3h = QHBoxLayout(drr_row3)
        drr_row3h.setContentsMargins(0, 0, 0, 0)
        drr_row3h.setSpacing(6)
        drr_row3h.addWidget(self.drr_fit_btn)
        drr_row3h.addWidget(self.drr_fit_clear_btn)
        drr_row3h.addWidget(self.drr_fit_show_chk)
        drr_row3h.addWidget(QLabel("N"))
        drr_row3h.addWidget(self.drr_fit_n_spin)
        drr_row3h.addStretch(1)
        analysis_form.addRow("", drr_row3)
        analysis_form.addRow("", self.drr_fit_status)
        self.drr_analysis_text = QPlainTextEdit()
        self.drr_analysis_text.setReadOnly(True)
        self.drr_analysis_text.setMinimumHeight(60)
        self.drr_analysis_text.setMaximumHeight(100)
        self.drr_analysis_text.setPlaceholderText("Peak/fit results will appear here after detection.")
        analysis_form.addRow("", self.drr_analysis_text)
        self.drr_fit_status.setStyleSheet("QLabel { color: #0071e3; font-size: 10px; }")
        params_layout.addWidget(self._make_expander("Manual plot ranges", basic, expanded=False))
        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addWidget(self._make_expander("Spectrum Analysis", analysis_box, expanded=False))
        layout.addStretch(1)
        return tab

    def _set_form_label_width(self, form: QFormLayout, width: int) -> None:
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.LabelRole)
            if item and isinstance(item.widget(), QLabel):
                lbl = item.widget()
                lbl.setFixedWidth(width)
                lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def _make_axis_range_row(
        self,
        a: QDoubleSpinBox,
        b: QDoubleSpinBox,
        fa: QCheckBox,
        fb: QCheckBox,
        auto_btn: QToolButton,
        auto_text: str,
    ) -> QWidget:
        """Build a min/max spin pair with Fix checkboxes and an Auto button."""
        auto_btn.setText(auto_text)
        auto_btn.setAutoRaise(True)
        auto_btn.setFixedWidth(UI_METRICS["tool_w"])
        auto_btn.setFixedHeight(UI_METRICS["tool_h"])
        auto_btn.setToolTip(f"Set {auto_text.replace('Auto ', '')} bounds automatically from loaded data")
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(a)
        h.addWidget(fa)
        h.addWidget(b)
        h.addWidget(fb)
        h.addWidget(auto_btn)
        h.addStretch(1)
        return row

    def _style_combo_popup(self, combo: QComboBox) -> None:
        view = combo.view()
        view.setStyleSheet(
            "QListView::item:selected { background-color: #0071e3; color: #ffffff; }"
            "QListView { selection-background-color: #0071e3; selection-color: #ffffff; }"
        )

    def _format_axis_coeff(self, value: float) -> str:
        if not np.isfinite(value):
            raise ValueError("Coefficient must be finite.")
        rounded = round(float(value), 12)
        if abs(rounded - round(rounded)) < 1e-12:
            return str(int(round(rounded)))
        return format(rounded, ".12g")

    def _manual_y_axis_label(self, a: float, b: float, c: float) -> str:
        if not all(np.isfinite(v) for v in (a, b, c)):
            raise ValueError("Coefficients must be finite.")
        if abs(c) < 1e-12:
            if abs(a - 1.0) < 1e-12 and abs(b) < 1e-12:
                return "TG (V)"
            if abs(b - 1.0) < 1e-12 and abs(a) < 1e-12:
                return "BG (V)"
            if abs(b + 1.0) < 1e-12 and abs(a - 1.0) < 1e-12:
                return "TG-BG (V)"
            if abs(b + 1.0) < 1e-12:
                ratio = self._format_axis_coeff(a)
                return f"{ratio}TG-BG (V)" if ratio != "1" else "TG-BG (V)"
        terms: list[str] = []
        for coeff, symbol in ((a, "TG"), (b, "BG")):
            if abs(coeff) < 1e-12:
                continue
            sign = "-" if coeff < 0 else "+"
            mag = abs(float(coeff))
            piece = symbol if abs(mag - 1.0) < 1e-12 else f"{self._format_axis_coeff(mag)}*{symbol}"
            if not terms:
                terms.append(piece if sign == "+" else f"-{piece}")
            else:
                terms.append(f"{sign}{piece}")
        if abs(c) >= 1e-12:
            c_piece = self._format_axis_coeff(abs(float(c)))
            sign = "-" if c < 0 else "+"
            if not terms:
                terms.append(c_piece if sign == "+" else f"-{c_piece}")
            else:
                terms.append(f"{sign}{c_piece}")
        return f"y = {''.join(terms)} (V)"

    def _update_y_axis_controls(self, prefix: str) -> None:
        combo: QComboBox = getattr(self, f"{prefix}_yaxis_combo")
        advanced_box: QGroupBox = getattr(self, f"{prefix}_yaxis_advanced_box")
        a_spin: QDoubleSpinBox = getattr(self, f"{prefix}_yaxis_a_spin")
        b_spin: QDoubleSpinBox = getattr(self, f"{prefix}_yaxis_b_spin")
        c_spin: QDoubleSpinBox = getattr(self, f"{prefix}_yaxis_c_spin")
        equation_lbl: QLabel = getattr(self, f"{prefix}_yaxis_equation_lbl")
        label_lbl: QLabel = getattr(self, f"{prefix}_yaxis_label_lbl")
        advanced_on = combo.currentText() == "Advanced..."
        advanced_box.setVisible(advanced_on)
        a = float(a_spin.value())
        b = float(b_spin.value())
        c = float(c_spin.value())
        equation_lbl.setText(
            f"y = {self._format_axis_coeff(a)}*TG + {self._format_axis_coeff(b)}*BG + {self._format_axis_coeff(c)}"
        )
        label_lbl.setText(self._manual_y_axis_label(a, b, c))

    def _selected_y_axis_spec(self, prefix: str) -> str:
        combo: QComboBox = getattr(self, f"{prefix}_yaxis_combo")
        text = combo.currentText()
        if text == "Auto / Default":
            return "auto"
        if text == "TG":
            return "tg"
        if text == "BG":
            return "bg"
        if text == "Bias":
            return "bias"
        if text == "Doping (V)":
            return "doping"
        if text == "Efield (V)":
            return "efield"
        if prefix == "pl" and text in {"Y", "Doping", "Electric field", "Gate voltage", "Custom"}:
            label = self.pl_dat_yaxis_label_edit.text() if text == "Custom" else ""
            unit = self.pl_dat_yaxis_unit_edit.text()
            return f"dat:{text}:{label}:{unit}"
        if text == "Advanced...":
            a = float(getattr(self, f"{prefix}_yaxis_a_spin").value())
            b = float(getattr(self, f"{prefix}_yaxis_b_spin").value())
            c = float(getattr(self, f"{prefix}_yaxis_c_spin").value())
            if not all(np.isfinite(v) for v in (a, b, c)):
                raise ValueError("Manual linear-combination coefficients must be finite.")
            return f"linear:{self._format_axis_coeff(a)},{self._format_axis_coeff(b)},{self._format_axis_coeff(c)}"
        raise ValueError(f"Unknown y-axis selection: {text}")

    def _csv_yaxis_items(self) -> list[str]:
        return ["Auto / Default", "TG", "BG", "Bias", "Advanced..."]

    def _xlsx_yaxis_items(self) -> list[str]:
        return ["Auto / Default", *data_io.XLSX_Y_LABEL_OPTIONS]

    def _repopulate_yaxis_combo(self, prefix: str, *, xlsx: bool) -> None:
        combo: QComboBox = getattr(self, f"{prefix}_yaxis_combo", None)
        if combo is None:
            return
        file_name = self._selected(self.pl_files)[0] if prefix == "pl" and self._selected(self.pl_files) else ""
        if prefix == "pl" and Path(file_name).suffix.lower() == ".dat":
            items = ["Y", "Doping", "Electric field", "Gate voltage", "Custom"]
        else:
            items = self._xlsx_yaxis_items() if xlsx else self._csv_yaxis_items()
        current = combo.currentText()
        blocked = combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(items)
            combo.setCurrentText(current if current in items else "Auto / Default")
        finally:
            combo.blockSignals(blocked)
        self._update_y_axis_controls(prefix)
        if prefix == "pl":
            is_dat = Path(file_name).suffix.lower() == ".dat"
            self.pl_dat_yaxis_label_edit.setVisible(is_dat and combo.currentText() == "Custom")
            self.pl_dat_yaxis_unit_edit.setVisible(is_dat)

    def _on_pl_selection_changed(self) -> None:
        selected = self._selected(self.pl_files)
        file_name = selected[0] if selected else ""
        self._repopulate_yaxis_combo("pl", xlsx=data_io.is_xlsx_map_file(file_name))

    def _repopulate_drr_yaxis(self) -> None:
        if not hasattr(self, "drr_yaxis_combo"):
            return
        first = self.drr_selected_files[0] if self.drr_selected_files else ""
        self._repopulate_yaxis_combo("drr", xlsx=data_io.is_xlsx_map_file(first))

    def _make_expander(self, title: str, content: QWidget, *, expanded: bool = True) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 3, 0, 4)
        v.setSpacing(4)
        head = QToolButton()
        head.setCheckable(True)
        head.setChecked(bool(expanded))
        head.setAutoRaise(True)
        head.setToolButtonStyle(Qt.ToolButtonTextOnly)
        head.setStyleSheet(
            "QToolButton { border: none; background: transparent; padding: 2px 4px; font-weight: 600; "
            "color: #1d1d1f; font-size: 11px; text-align: left; }"
            "QToolButton:hover { color: #0071e3; background: transparent; }"
        )
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        line.setStyleSheet("QFrame { color: #e5e5ea; }")
        row = QWidget()
        row_h = QHBoxLayout(row)
        row_h.setContentsMargins(0, 0, 0, 0)
        row_h.setSpacing(6)
        row_h.addWidget(head)
        row_h.addWidget(line, 1)
        v.addWidget(row)
        content.setVisible(bool(expanded))
        v.addWidget(content)

        def _update(on: bool) -> None:
            arrow = "▼" if on else "▶"
            head.setText(f"{arrow}  {title}")
            content.setVisible(bool(on))

        _update(bool(expanded))
        head.toggled.connect(_update)
        return box

    def _build_compare_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        params = QGroupBox("Plot Options")
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 4)
        params_layout.setSpacing(4)

        assignment = QGroupBox("Assignment")
        assignment_layout = QVBoxLayout(assignment)
        assignment_layout.setContentsMargins(6, 8, 6, 6)
        assignment_layout.setSpacing(6)
        assignment_form = QFormLayout()
        assignment_form.setContentsMargins(0, 0, 0, 0)
        assignment_form.setHorizontalSpacing(6)
        assignment_form.setVerticalSpacing(4)
        def _angle_spin(default: float = 0.0) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-360.0, 360.0)
            spin.setValue(default)
            spin.setSuffix(" deg")
            spin.setToolTip("Approximate reference angle; matching uses the tolerance below.")
            return spin

        self.cmp_in_k_angle_spin = _angle_spin()
        self.cmp_in_kp_angle_spin = _angle_spin(45.0)
        self.cmp_out_k_angle_spin = _angle_spin()
        self.cmp_out_kp_angle_spin = _angle_spin(45.0)
        self.cmp_angle_tolerance_spin = QDoubleSpinBox()
        self.cmp_angle_tolerance_spin.setDecimals(2)
        self.cmp_angle_tolerance_spin.setRange(0.1, 180.0)
        self.cmp_angle_tolerance_spin.setValue(15.0)
        self.cmp_angle_tolerance_spin.setSuffix(" deg")
        self.cmp_angle_tolerance_spin.setToolTip(
            "Maximum distance from the nearest K or Kp reference. Equal-distance matches are rejected."
        )
        self.cmp_infer_angles_btn = QPushButton("Infer Angles")
        self.cmp_infer_angles_btn.setToolTip(
            "Suggest editable K/Kp references when exactly two filename-angle clusters are detected."
        )
        self.cmp_auto_assign_btn = QPushButton("Auto Detect")
        angle_box = QWidget()
        angle_grid = QGridLayout(angle_box)
        angle_grid.setContentsMargins(0, 0, 0, 0)
        angle_grid.setHorizontalSpacing(6)
        angle_grid.setVerticalSpacing(4)
        angle_grid.addWidget(QLabel("In K"), 0, 0)
        angle_grid.addWidget(self.cmp_in_k_angle_spin, 0, 1)
        angle_grid.addWidget(QLabel("In Kp"), 0, 2)
        angle_grid.addWidget(self.cmp_in_kp_angle_spin, 0, 3)
        angle_grid.addWidget(QLabel("Out K"), 1, 0)
        angle_grid.addWidget(self.cmp_out_k_angle_spin, 1, 1)
        angle_grid.addWidget(QLabel("Out Kp"), 1, 2)
        angle_grid.addWidget(self.cmp_out_kp_angle_spin, 1, 3)
        angle_grid.addWidget(QLabel("Tolerance"), 2, 0)
        angle_grid.addWidget(self.cmp_angle_tolerance_spin, 2, 1)
        angle_grid.addWidget(self.cmp_infer_angles_btn, 2, 2)
        angle_grid.addWidget(self.cmp_auto_assign_btn, 2, 3)
        angle_grid.setColumnStretch(4, 1)
        assignment_form.addRow("Angle Rules", angle_box)
        assignment_layout.addLayout(assignment_form)
        self.cmp_channel_combos: dict[str, QComboBox] = {}
        channels_box = QWidget()
        channels_grid = QGridLayout(channels_box)
        channels_grid.setContentsMargins(0, 0, 0, 0)
        channels_grid.setHorizontalSpacing(8)
        channels_grid.setVerticalSpacing(4)
        for idx, key in enumerate(COMPARE_PANEL_ORDER):
            combo = QComboBox()
            combo.setEditable(False)
            self._style_combo_popup(combo)
            self.cmp_channel_combos[key] = combo
            row = idx // 2
            col = idx % 2
            label = QLabel(key)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            channels_grid.addWidget(label, row, col * 2)
            channels_grid.addWidget(combo, row, col * 2 + 1)
        assignment_layout.addWidget(channels_box)
        self.cmp_assignment_summary = QPlainTextEdit()
        self.cmp_assignment_summary.setReadOnly(True)
        self.cmp_assignment_summary.setMaximumHeight(88)
        summary_form = QFormLayout()
        summary_form.setContentsMargins(0, 0, 0, 0)
        summary_form.setHorizontalSpacing(6)
        summary_form.setVerticalSpacing(4)
        summary_form.addRow("Summary", self.cmp_assignment_summary)
        assignment_layout.addLayout(summary_form)
        layout.addWidget(self._make_expander("Assignment", assignment, expanded=True))

        display = QGroupBox("Display Preset")
        display_form = QFormLayout(display)
        self.cmp_display_preset_combo = QComboBox()
        self.cmp_display_preset_combo.addItems(["KK + KKp", "KpK + KpKp", "All four", "Custom"])
        self._style_combo_popup(self.cmp_display_preset_combo)
        display_form.addRow("Preset", self.cmp_display_preset_combo)
        checks_row = QWidget()
        checks_h = QHBoxLayout(checks_row)
        checks_h.setContentsMargins(0, 0, 0, 0)
        checks_h.setSpacing(10)
        self.cmp_show_checks: dict[str, QCheckBox] = {}
        for key in COMPARE_PANEL_ORDER:
            chk = QCheckBox(key)
            self.cmp_show_checks[key] = chk
            checks_h.addWidget(chk)
        checks_h.addStretch(1)
        display_form.addRow("Channels", checks_row)
        params_layout.addWidget(self._make_expander("Display", display, expanded=False))

        vp_box = QGroupBox("Valley Polarization")
        vp_form = QFormLayout(vp_box)
        vp_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        vp_form.setHorizontalSpacing(6)
        vp_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        self.cmp_vp_background_spin = QDoubleSpinBox()
        self.cmp_vp_background_spin.setDecimals(6)
        self.cmp_vp_background_spin.setRange(-1.0e12, 1.0e12)
        self.cmp_vp_background_spin.setSingleStep(100.0)
        self.cmp_vp_background_spin.setFixedWidth(UI_METRICS["spin_w"] + 18)
        self.cmp_vp_auto_background_chk = QCheckBox("Auto")
        self.cmp_vp_auto_background_chk.setChecked(True)
        self.cmp_vp_auto_background_chk.setToolTip("Estimate one constant background from KK and KKp.")
        bkg_row = QWidget()
        bkg_h = QHBoxLayout(bkg_row)
        bkg_h.setContentsMargins(0, 0, 0, 0)
        bkg_h.setSpacing(8)
        bkg_h.addWidget(self.cmp_vp_background_spin)
        bkg_h.addWidget(self.cmp_vp_auto_background_chk)
        bkg_h.addStretch(1)
        self.cmp_vp_filename_preview = QLineEdit()
        self.cmp_vp_filename_preview.setReadOnly(True)
        self.cmp_vp_filename_preview.setMinimumWidth(200)
        self.cmp_kk_title_preview = QLineEdit()
        self.cmp_kk_title_preview.setReadOnly(True)
        self.cmp_kkp_title_preview = QLineEdit()
        self.cmp_kkp_title_preview.setReadOnly(True)
        self.cmp_vp_title_preview = QLineEdit()
        self.cmp_vp_title_preview.setReadOnly(True)
        vp_form.addRow("Background", bkg_row)
        vp_form.addRow("VP filename", self.cmp_vp_filename_preview)
        vp_form.addRow("KK title", self.cmp_kk_title_preview)
        vp_form.addRow("KKp title", self.cmp_kkp_title_preview)
        vp_form.addRow("VP title", self.cmp_vp_title_preview)
        params_layout.addWidget(self._make_expander("VP", vp_box, expanded=False))

        cfg = QFormLayout()
        cfg.setHorizontalSpacing(6)
        cfg.setVerticalSpacing(4)
        _grid, spins, _, _, cmap, fix_checks = self._build_common_range_grid("cmp", "turbo")
        self.cmp_yaxis_controls = self._build_y_axis_controls("cmp")
        _cmp_yc_row = QWidget()
        _cmp_yc_h = QHBoxLayout(_cmp_yc_row)
        _cmp_yc_h.setContentsMargins(0, 0, 0, 0)
        _cmp_yc_h.setSpacing(6)
        _cmp_yc_h.addWidget(self.cmp_yaxis_combo, 1)
        _cmp_yc_h.addWidget(QLabel("Cmap"))
        _cmp_yc_h.addWidget(cmap)
        cfg.addRow("Y-axis / Cmap", _cmp_yc_row)
        cfg.addRow("", self.cmp_yaxis_advanced_box)
        params_layout.addLayout(cfg)
        for s in spins.values():
            s.setFixedWidth(UI_METRICS["spin_w"])
            s.setFixedHeight(UI_METRICS["input_h"])

        self.cmp_auto_v_btn = QToolButton()
        self.cmp_auto_x_btn = QToolButton()
        self.cmp_auto_y_btn = QToolButton()
        basic = QGroupBox("Axis Ranges")
        basic_form = QFormLayout(basic)
        basic_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        basic_form.setHorizontalSpacing(4)
        basic_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        basic_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        basic_form.addRow(
            "vmin / vmax",
            self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.cmp_auto_v_btn, "Auto V"),
        )
        basic_form.addRow("Color scale", self.cmp_split_scale_chk)
        basic_form.addRow(self.cmp_split_scale_panel)
        basic_form.addRow(
            "xmin / xmax",
            self._make_axis_range_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.cmp_auto_x_btn, "Auto X"),
        )
        basic_form.addRow(
            "ymin / ymax",
            self._make_axis_range_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.cmp_auto_y_btn, "Auto Y"),
        )
        basic_form.addRow("Cursor Gate", spins["gate"])
        flags = QWidget()
        flags_h = QHBoxLayout(flags)
        flags_h.setContentsMargins(0, 0, 0, 0)
        flags_h.setSpacing(10)
        flags_h.addWidget(self.cmp_log_chk)
        flags_h.addWidget(self.cmp_clip_chk)
        flags_h.addStretch(1)
        basic_form.addRow("Scale / Clip", flags)
        self._set_form_label_width(basic_form, UI_METRICS["label_col_width"])
        params_layout.addWidget(self._make_expander("Manual plot ranges", basic, expanded=False))
        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addStretch(1)
        return tab

    def _build_mcd_tab(self) -> QWidget:
        tab = QWidget()
        tab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        source = QGroupBox("B-sweep source")
        source_form = QFormLayout(source)
        source_form.setContentsMargins(6, 4, 6, 4)
        source_form.setHorizontalSpacing(6)
        source_form.setVerticalSpacing(3)
        self.mcd_files = QListWidget()
        self.mcd_files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.mcd_files.setMinimumHeight(60)
        self.mcd_files.setMaximumHeight(120)
        self.mcd_files.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.mcd_files.setToolTip("Select one B-sweep CSV. Long filenames are clipped here; hover an item to read its full name.")
        source_form.addRow(self.mcd_files)
        self.mcd_source_summary = QLabel("Select a B-sweep CSV.")
        self.mcd_source_summary.setWordWrap(True)
        self.mcd_source_summary.setMinimumWidth(0)
        self.mcd_source_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        source_form.addRow("Format", self.mcd_source_summary)
        layout.addWidget(self._make_expander("Source", source, expanded=True))

        correction = QGroupBox("Angle background correction")
        correction_layout = QVBoxLayout(correction)
        correction_layout.setContentsMargins(6, 4, 6, 4)
        correction_layout.setSpacing(3)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(3)
        correction_layout.addLayout(form)
        self.mcd_auto_angles_chk = QCheckBox("Auto-assign sigma+ / sigma-")
        self.mcd_auto_angles_chk.setChecked(True)
        self.mcd_auto_angles_chk.setToolTip(
            "Detect the available angles in the selected CSV. When enabled, the largest angle is assigned to sigma+ and the smallest to sigma-. "
            "Turn this off to choose the sigma+ and sigma- assignments yourself."
        )
        self.mcd_sigma_plus_combo = QComboBox(); self.mcd_sigma_minus_combo = QComboBox()
        for combo in (self.mcd_sigma_plus_combo, self.mcd_sigma_minus_combo):
            combo.addItem("-- Select source CSV first --", None)
            self._style_combo_popup(combo)
        self.mcd_reference_mode_combo = QComboBox()
        self.mcd_reference_mode_combo.addItems(["Nearest paired B (recommended)", "Median near-zero window"])
        self._style_combo_popup(self.mcd_reference_mode_combo)
        self.mcd_zero_spin = QDoubleSpinBox(); self.mcd_zero_spin.setRange(0.0, 10.0); self.mcd_zero_spin.setDecimals(4); self.mcd_zero_spin.setValue(0.02); self.mcd_zero_spin.setSuffix(" T")
        self.mcd_zero_spin.setEnabled(False)
        self.mcd_gap_spin = QSpinBox(); self.mcd_gap_spin.setRange(1, 50); self.mcd_gap_spin.setValue(3)
        self.mcd_delta_b_spin = QDoubleSpinBox(); self.mcd_delta_b_spin.setRange(0.0001, 10.0); self.mcd_delta_b_spin.setDecimals(4); self.mcd_delta_b_spin.setValue(0.1); self.mcd_delta_b_spin.setSuffix(" T")
        self.mcd_pair_alignment_combo = QComboBox(); self.mcd_pair_alignment_combo.addItems(["Direct measured pair", "Interpolate both angles to Bpair"])
        self.mcd_bin_spin = QSpinBox(); self.mcd_bin_spin.setRange(0, 6); self.mcd_bin_spin.setValue(3)
        self.mcd_gain_combo = QComboBox(); self.mcd_gain_combo.addItems(["Per wavelength", "Smoothed per wavelength", "Scalar (diagnostic only)"])
        self.mcd_correction_mode_combo = QComboBox(); self.mcd_correction_mode_combo.addItems([
            "Global reference gain (current)", "Global gain + per-pair scale", "Global gain + per-pair scale/offset",
            "Global gain + per-pair spectral baseline",
        ])
        self.mcd_correction_mode_combo.setCurrentIndex(3)
        self.mcd_spectral_order_combo = QComboBox(); self.mcd_spectral_order_combo.addItems([
            "Linear", "Quadratic (default)",
        ])
        self.mcd_spectral_order_combo.setCurrentIndex(1)
        self.mcd_spectral_order_combo.setEnabled(True)
        self.mcd_background_ranges_edit = QLineEdit()
        self.mcd_background_ranges_edit.setPlaceholderText("Auto outer 15%, or e.g. 1.50-1.58, 1.73-1.79")
        self.mcd_suggest_background_btn = QPushButton("Select protected regions")
        self.mcd_suggest_background_btn.setToolTip(
            "Draw persistent feature-protection windows on the full-sweep reflection plot. "
            "Every sufficiently wide unprotected interval updates automatically as a background band."
        )
        self.mcd_background_preview = QLabel("Auto outer 15% ranges are shown after loading an MCD sweep.")
        self.mcd_background_preview.setWordWrap(True)
        self.mcd_background_preview.setMinimumWidth(0)
        self.mcd_background_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.mcd_apply_correction_btn = QPushButton("Recalculate now")
        self.mcd_apply_correction_btn.setToolTip(
            "MCD processing updates automatically after settings settle. Click to recalculate immediately or retry after an error."
        )
        self.mcd_dark_pos_combo = QComboBox(); self.mcd_dark_neg_combo = QComboBox()
        for combo in (self.mcd_dark_pos_combo, self.mcd_dark_neg_combo):
            combo.addItem("-- No dark / offset file --", "")
            self._style_combo_popup(combo)
        self.mcd_sigma_plus_combo.setToolTip(
            "Measured waveplate/analyser angle to interpret as sigma+. Changing the sigma+ and sigma- assignments reverses the MCD sign."
        )
        self.mcd_sigma_minus_combo.setToolTip(
            "Measured waveplate/analyser angle to interpret as sigma-. It must be different from the sigma+ choice."
        )
        self.mcd_zero_spin.setToolTip(
            "Only used with Median near-zero window. Paired spectra with |B_pair| at or below this value "
            "are combined into each angle's reference spectrum."
        )
        self.mcd_reference_mode_combo.setToolTip(
            "Nearest paired B uses the one valid sigma+/sigma- pair with the smallest |B_pair|. "
            "This is the normal choice when an exact B = 0 spectrum was not acquired. "
            "Median near-zero window combines all pairs inside the reference window; use it only when several "
            "near-zero pairs are available and their physical MCD is negligible."
        )
        self.mcd_gap_spin.setToolTip(
            "Maximum separation, in CSV rows, allowed between opposite-angle spectra in one pair. "
            "A value of 3 accepts an opposite-angle partner within three acquired frames."
        )
        self.mcd_delta_b_spin.setToolTip(
            "Maximum allowed difference between the two raw B-field values in a pair. "
            "The paired B shown in the app is their average. Reduce this to reject poorly matched field pairs."
        )
        self.mcd_pair_alignment_combo.setToolTip(
            "Direct measured pair uses the two acquired spectra as-is. Interpolate both angle channels to Bpair using neighbours from the same sweep branch, "
            "which reduces artifacts when the two angles were measured at different B. It never interpolates across a detected field reversal."
        )
        self.mcd_bin_spin.setToolTip(
            "Round each paired B field to this many decimal places before averaging repeated points into the colormap. "
            "3 means bins spaced by 0.001 T; use fewer decimals only when the field readings are noisy."
        )
        self.mcd_gain_combo.setToolTip(
            "Angle-throughput correction derived from the two near-zero-field reference spectra. "
            "Per wavelength is the normal choice. Smoothed per wavelength suppresses noisy gain ripples. "
            "Scalar applies one number to the whole spectrum and is intended only as a diagnostic."
        )
        self.mcd_correction_mode_combo.setToolTip(
            "Global reference gain applies one reference-derived wavelength correction to every pair. "
            "Per-pair scale additionally corrects intensity drift using only the background energy ranges. "
            "Scale/offset also removes an additive offset. Spectral baseline robustly fits a smooth energy-dependent "
            "sigma+/sigma- ratio in the selected background regions and is the default correction. Review protected regions because an overly broad fit can remove real MCD structure."
        )
        self.mcd_spectral_order_combo.setToolTip(
            "Polynomial order for the per-pair spectral baseline. Quadratic is the default and corrects a broad curved mismatch. "
            "Linear remains available when only a wavelength-dependent tilt is justified; orders above two are intentionally not offered to avoid unstable overfitting."
        )
        self.mcd_background_ranges_edit.setToolTip(
            "Energy intervals used to fit per-pair scale, scale/offset, or the spectral baseline. Separate intervals with commas, for example "
            "1.50-1.58, 1.73-1.79. Leave blank to use both spectrum ends: the lowest 15% and highest 15% of the measured energy range (30% total). "
            "Exclude exciton peaks and the MCD feature of interest. Spectral correction requires separated regions spanning at least 25% of the energy range. "
            "Use Select protected regions to draw resonances that must be excluded; all unprotected background bands recalculate automatically."
        )
        self.mcd_dark_pos_combo.setToolTip(
            "Optional CSV containing the additive dark/stray-light spectrum measured with the positive angle. "
            "It is subtracted before reference normalization and must have the same wavelength columns."
        )
        self.mcd_dark_neg_combo.setToolTip(
            "Optional CSV containing the additive dark/stray-light spectrum measured with the negative angle. "
            "It is subtracted before reference normalization and must have the same wavelength columns."
        )
        mcd_compact_combos = (
            self.mcd_sigma_plus_combo, self.mcd_sigma_minus_combo, self.mcd_reference_mode_combo,
            self.mcd_pair_alignment_combo, self.mcd_gain_combo, self.mcd_correction_mode_combo, self.mcd_spectral_order_combo,
            self.mcd_dark_pos_combo, self.mcd_dark_neg_combo,
        )
        for combo in mcd_compact_combos:
            # Keep the selected long filename/mode readable by tooltip while
            # allowing the closed combo box to elide in the narrow sidebar.
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        for widget in (*mcd_compact_combos, self.mcd_zero_spin, self.mcd_gap_spin, self.mcd_delta_b_spin,
                       self.mcd_bin_spin, self.mcd_background_ranges_edit):
            widget.setFixedHeight(UI_METRICS["input_h"])
        form.addRow("Angles", self.mcd_auto_angles_chk)
        form.addRow("Sigma+ angle", self.mcd_sigma_plus_combo)
        form.addRow("Sigma- angle", self.mcd_sigma_minus_combo)
        form.addRow("Reference method", self.mcd_reference_mode_combo)
        form.addRow("Reference window", self.mcd_zero_spin)
        form.addRow("Sequence gap", self.mcd_gap_spin)
        form.addRow("Pair dB", self.mcd_delta_b_spin)
        form.addRow("Gain", self.mcd_gain_combo)
        form.addRow("Drift correction", self.mcd_correction_mode_combo)
        form.addRow("", self.mcd_apply_correction_btn)
        advanced_correction = QGroupBox("Advanced correction")
        advanced_form = QFormLayout(advanced_correction)
        advanced_form.setContentsMargins(4, 3, 4, 3)
        advanced_form.setHorizontalSpacing(6)
        advanced_form.setVerticalSpacing(3)
        advanced_form.addRow("Pair B alignment", self.mcd_pair_alignment_combo)
        advanced_form.addRow("B bin decimals", self.mcd_bin_spin)
        advanced_form.addRow("Fit bg E (eV)", self.mcd_background_ranges_edit)
        advanced_form.addRow("", self.mcd_suggest_background_btn)
        advanced_form.addRow("Background", self.mcd_background_preview)
        advanced_form.addRow("Spectral fit", self.mcd_spectral_order_combo)
        advanced_form.addRow("Dark sigma+", self.mcd_dark_pos_combo)
        advanced_form.addRow("Dark sigma-", self.mcd_dark_neg_combo)
        correction_layout.addWidget(self._make_expander("Advanced", advanced_correction, expanded=False))
        layout.addWidget(self._make_expander("Correction", correction, expanded=False))

        diagnostics = QGroupBox("Pair diagnostics")
        diagnostics_layout = QVBoxLayout(diagnostics)
        self.mcd_diagnostics_text = QPlainTextEdit()
        self.mcd_diagnostics_text.setReadOnly(True)
        self.mcd_diagnostics_text.setMinimumHeight(125)
        self.mcd_diagnostics_text.setToolTip(
            "One row per sigma+/sigma- pair. Large |dB|, a large relative RMS residual, or a rapidly changing fitted scale "
            "indicates a pair that may not be corrected reliably. Spectral mode also reports log-gain slope/curvature, "
            "the applied gain range, and background RMS before/after fitting. The full table is exported with MCD results."
        )
        diagnostics_layout.addWidget(self.mcd_diagnostics_text)
        self.mcd_diagnostics_expander = self._make_expander("Diagnostics", diagnostics, expanded=False)
        layout.addWidget(self.mcd_diagnostics_expander)

        display = QGroupBox("Display and MCD(B)")
        display_form = QFormLayout(display)
        display_form.setContentsMargins(6, 4, 6, 4)
        display_form.setHorizontalSpacing(6)
        display_form.setVerticalSpacing(3)
        self.mcd_map_combo = QComboBox(); self.mcd_map_combo.addItem("Combo")
        _grid, self.mcd_spins, _a, _b, self.mcd_cmap, _mcd_unused_fix_checks = self._build_common_range_grid("mcd", "RdBu_r")
        # MCD does not expose fixed-range controls. Keeping only the visible
        # widgets avoids retaining unparented QCheckBoxes after construction.
        self.mcd_fix_checks: Dict[str, QCheckBox] = {}
        unused_mcd_cursor = self.mcd_spins.pop("gate")
        unused_mcd_cursor.deleteLater()
        for spin in self.mcd_spins.values():
            # The shared range helper uses generous numerical spin-box hints
            # for the main plot tabs.  Here they appear in compact pairs, so
            # cap their width while retaining a usable minimum.  They must not
            # be Ignored: the trailing stretch in _pair_row would then shrink
            # the Energy/B-field inputs to zero width.
            spin.setMinimumWidth(72)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mcd_center_zero_chk = QCheckBox("Zero-centered"); self.mcd_center_zero_chk.setChecked(True)
        self.mcd_auto_v_btn = QPushButton("Auto color")
        self.mcd_pair_b_combo = QComboBox(); self._style_combo_popup(self.mcd_pair_b_combo)
        self.mcd_pair_b_combo.setToolTip("Choose the paired measurement used for the spectra and MCD linecut. Clicking a B field on the colormap selects the nearest pair here.")
        self.mcd_window_center_spin = QDoubleSpinBox(); self.mcd_window_center_spin.setRange(0.0, 10.0); self.mcd_window_center_spin.setDecimals(6)
        self.mcd_window_width_spin = QDoubleSpinBox(); self.mcd_window_width_spin.setRange(0.01, 1000); self.mcd_window_width_spin.setDecimals(3); self.mcd_window_width_spin.setValue(5.0); self.mcd_window_width_spin.setSuffix(" meV")
        self.mcd_window_metric_combo = QComboBox(); self.mcd_window_metric_combo.addItems(["Field-signed absolute mean", "Signed mean", "Signed integral", "Unsigned absolute mean (diagnostic)"])
        self.mcd_window_metric_combo.setCurrentText("Signed mean")
        self.mcd_show_raw_chk = QCheckBox("Raw"); self.mcd_show_raw_chk.setChecked(False)
        self.mcd_show_signed_mean_chk = QCheckBox("Signed mean"); self.mcd_show_signed_mean_chk.setChecked(True)
        self.mcd_show_absolute_mean_chk = QCheckBox("B*|MCD|"); self.mcd_show_absolute_mean_chk.setChecked(False)
        self.mcd_show_unsigned_absolute_mean_chk = QCheckBox("|MCD|"); self.mcd_show_unsigned_absolute_mean_chk.setChecked(False)
        self.mcd_show_integral_chk = QCheckBox("Integral"); self.mcd_show_integral_chk.setChecked(False)
        trace_visibility = QWidget(); trace_visibility_layout = QGridLayout(trace_visibility)
        trace_visibility_layout.setContentsMargins(0, 0, 0, 0)
        trace_visibility_layout.setHorizontalSpacing(6); trace_visibility_layout.setVerticalSpacing(2)
        trace_visibility_layout.addWidget(self.mcd_show_raw_chk, 0, 0)
        trace_visibility_layout.addWidget(self.mcd_show_signed_mean_chk, 0, 1)
        trace_visibility_layout.addWidget(self.mcd_show_absolute_mean_chk, 1, 0)
        trace_visibility_layout.addWidget(self.mcd_show_unsigned_absolute_mean_chk, 1, 1)
        trace_visibility_layout.addWidget(self.mcd_show_integral_chk, 2, 0)
        trace_visibility_layout.setColumnStretch(0, 1)
        trace_visibility_layout.setColumnStretch(1, 1)
        self.mcd_window_metric_combo.setToolTip("Selects the primary MCD(B) metric recorded in export settings. The Origin-ready MCD(B) CSV contains corrected signed mean, field-signed absolute mean, and signed integral for both B-sweep directions.")
        self.mcd_show_raw_chk.setToolTip("Add dashed raw-MCD curves for comparison with the corrected curves.")
        self.mcd_show_signed_mean_chk.setToolTip("Average signed MCD inside the selected energy window.")
        self.mcd_show_absolute_mean_chk.setToolTip("MCD magnitude multiplied by the sign of Bpair. It avoids cancellation between opposite spectral lobes while following the positive/negative field sides.")
        self.mcd_show_unsigned_absolute_mean_chk.setToolTip("Pure MCD magnitude, always non-negative. Keep this off unless you need a correction/noise diagnostic.")
        self.mcd_show_integral_chk.setToolTip("Signed energy integral of MCD. It uses the right axis because its unit is MCD eV and changes with the selected window width.")
        self.mcd_fit_zero_chk = QCheckBox("Near-zero fit")
        self.mcd_fit_b_window_spin = QDoubleSpinBox(); self.mcd_fit_b_window_spin.setRange(0.001, 10.0); self.mcd_fit_b_window_spin.setDecimals(3); self.mcd_fit_b_window_spin.setValue(0.2); self.mcd_fit_b_window_spin.setSuffix(" T")
        for combo in (self.mcd_map_combo, self.mcd_pair_b_combo, self.mcd_cmap, self.mcd_window_metric_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        display_form.addRow("Map", self.mcd_map_combo)
        display_form.addRow("Color", self._pair_row(self.mcd_spins["vmin"], self.mcd_spins["vmax"], self.mcd_auto_v_btn))
        display_form.addRow("Energy", self._pair_row(self.mcd_spins["xmin"], self.mcd_spins["xmax"]))
        display_form.addRow("B field", self._pair_row(self.mcd_spins["ymin"], self.mcd_spins["ymax"]))
        display_form.addRow("Selected pair", self.mcd_pair_b_combo)
        display_form.addRow("Color map", self.mcd_cmap)
        display_form.addRow("Scale", self.mcd_center_zero_chk)
        display_form.addRow("MCD(B) E0", self.mcd_window_center_spin)
        display_form.addRow("MCD(B) width", self.mcd_window_width_spin)
        display_form.addRow("MCD(B) traces", trace_visibility)
        display_form.addRow("Primary export metric", self.mcd_window_metric_combo)
        display_form.addRow("Near-zero fit", self._pair_row(self.mcd_fit_zero_chk, self.mcd_fit_b_window_spin))
        layout.addWidget(self._make_expander("Plot", display, expanded=True))
        layout.addStretch(1)
        return tab

    @staticmethod
    def _pair_row(*widgets: QWidget) -> QWidget:
        row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        for widget in widgets: h.addWidget(widget)
        h.addStretch(1)
        return row

    def _build_shg_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.shg_workflow_tabs = QTabWidget()
        single_page = QWidget()
        single_layout = QVBoxLayout(single_page)
        single_layout.setContentsMargins(4, 4, 4, 4)
        single_layout.setSpacing(6)

        data_box = QGroupBox("SHG Sweep Table")
        data_layout = QVBoxLayout(data_box)
        data_layout.setContentsMargins(6, 6, 6, 6)
        data_layout.setSpacing(6)
        self.shg_files = QListWidget()
        self.shg_files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.shg_files.setMinimumHeight(60)
        self.shg_files.setMaximumHeight(120)
        self.shg_files.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.shg_files.setToolTip("Select an SHG sweep CSV to load")
        self.shg_background_combo = QComboBox()
        self._style_combo_popup(self.shg_background_combo)
        self.shg_summary = QPlainTextEdit()
        self.shg_summary.setReadOnly(True)
        self.shg_summary.setMaximumHeight(105)
        background_row = QWidget()
        background_layout = QHBoxLayout(background_row)
        background_layout.setContentsMargins(0, 0, 0, 0)
        background_layout.setSpacing(6)
        background_layout.addWidget(QLabel("External background"))
        background_layout.addWidget(self.shg_background_combo, 1)
        data_layout.addWidget(self.shg_files)
        data_layout.addWidget(background_row)
        data_layout.addWidget(self.shg_summary)
        single_layout.addWidget(self._make_expander("Data", data_box, expanded=True))
        single_layout.addStretch(1)
        self.shg_workflow_tabs.addTab(single_page, "Single File")

        compare_page = QWidget()
        compare_layout = QVBoxLayout(compare_page)
        compare_layout.setContentsMargins(4, 4, 4, 4)
        compare_layout.setSpacing(6)
        compare_box = QGroupBox("SHG Twist Comparison")
        compare_form = QFormLayout(compare_box)
        compare_form.setContentsMargins(6, 6, 6, 6)
        compare_form.setSpacing(6)
        self.shg_compare_reference_combo = QComboBox()
        self.shg_compare_sample_combo = QComboBox()
        self.shg_compare_background_a_combo = QComboBox()
        self.shg_compare_background_b_combo = QComboBox()
        self.shg_compare_display_combo = QComboBox()
        self.shg_compare_display_combo.addItems(["Raw area", "Normalized"])
        for combo in (
            self.shg_compare_reference_combo,
            self.shg_compare_sample_combo,
            self.shg_compare_background_a_combo,
            self.shg_compare_background_b_combo,
            self.shg_compare_display_combo,
        ):
            self._style_combo_popup(combo)
        self.shg_compare_summary = QPlainTextEdit()
        self.shg_compare_summary.setReadOnly(True)
        self.shg_compare_summary.setMaximumHeight(150)
        compare_form.addRow("Reference A", self.shg_compare_reference_combo)
        compare_form.addRow("Sample B", self.shg_compare_sample_combo)
        compare_form.addRow("Background A", self.shg_compare_background_a_combo)
        compare_form.addRow("Background B", self.shg_compare_background_b_combo)
        compare_form.addRow("Plot", self.shg_compare_display_combo)
        compare_form.addRow(self.shg_compare_summary)
        self._set_form_label_width(compare_form, UI_METRICS["label_col_width"])
        compare_layout.addWidget(compare_box)
        compare_layout.addStretch(1)
        self.shg_workflow_tabs.addTab(compare_page, "Compare / Twist Angle")
        layout.addWidget(self.shg_workflow_tabs)

        def wavelength_spin(value: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(0.0, 5000.0)
            spin.setSingleStep(0.1)
            spin.setValue(value)
            spin.setFixedWidth(UI_METRICS["spin_w"] + 12)
            return spin

        integration = QGroupBox("SHG Peak Integration")
        integration_form = QFormLayout(integration)
        integration_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        integration_form.setSpacing(6)
        self.shg_peak_center_spin = wavelength_spin(515.0)
        self.shg_gate_half_range_spin = wavelength_spin(3.0)
        self.shg_gate_half_range_spin.setMinimum(0.0001)
        self.shg_sideband_gap_spin = wavelength_spin(1.0)
        self.shg_sideband_width_spin = wavelength_spin(4.0)
        self.shg_sideband_width_spin.setMinimum(0.0001)
        self.shg_background_method_combo = QComboBox()
        self.shg_background_method_combo.addItems(
            ["Local linear", "Local quadratic", "External + local residual", "None"]
        )
        self._style_combo_popup(self.shg_background_method_combo)
        self.shg_sigma_clip_spin = QDoubleSpinBox()
        self.shg_sigma_clip_spin.setDecimals(1)
        self.shg_sigma_clip_spin.setRange(1.0, 10.0)
        self.shg_sigma_clip_spin.setSingleStep(0.5)
        self.shg_sigma_clip_spin.setValue(3.0)
        self.shg_sigma_clip_spin.setFixedWidth(UI_METRICS["spin_w"])

        def value_row(spin: QDoubleSpinBox, suffix: str = "nm") -> QWidget:
            widget = QWidget()
            row_layout = QHBoxLayout(widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.addWidget(spin)
            row_layout.addWidget(QLabel(suffix))
            row_layout.addStretch(1)
            return widget

        integration_form.addRow("Integration wavelength", value_row(self.shg_peak_center_spin))
        integration_form.addRow("Integration range (±)", value_row(self.shg_gate_half_range_spin, "nm on each side"))
        integration_form.addRow("Sideband gap", value_row(self.shg_sideband_gap_spin, "nm from peak gate"))
        integration_form.addRow("Sideband width", value_row(self.shg_sideband_width_spin, "nm on each side"))
        integration_form.addRow("Background", self.shg_background_method_combo)
        integration_form.addRow("Sigma clip", self.shg_sigma_clip_spin)
        self._set_form_label_width(integration_form, UI_METRICS["label_col_width"])
        layout.addWidget(self._make_expander("Peak Integration", integration, expanded=True))

        cosmic_box = QGroupBox("Cosmic Ray Removal")
        cosmic_form = QFormLayout(cosmic_box)
        cosmic_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        cosmic_form.setSpacing(6)
        self.shg_cosmic_enable_chk = QCheckBox("Remove narrow positive spikes")
        self.shg_cosmic_enable_chk.setChecked(True)
        self.shg_cosmic_threshold_spin = QDoubleSpinBox()
        self.shg_cosmic_threshold_spin.setDecimals(1)
        self.shg_cosmic_threshold_spin.setRange(3.0, 30.0)
        self.shg_cosmic_threshold_spin.setSingleStep(0.5)
        self.shg_cosmic_threshold_spin.setValue(8.0)
        self.shg_cosmic_window_spin = QSpinBox()
        self.shg_cosmic_window_spin.setRange(3, 51)
        self.shg_cosmic_window_spin.setSingleStep(2)
        self.shg_cosmic_window_spin.setValue(7)
        self.shg_cosmic_max_width_spin = QSpinBox()
        self.shg_cosmic_max_width_spin.setRange(1, 15)
        self.shg_cosmic_max_width_spin.setValue(3)
        self.shg_spectrum_view_combo = QComboBox()
        self.shg_spectrum_view_combo.addItems(["Raw + cleaned", "Raw", "Cosmic-cleaned"])
        self._style_combo_popup(self.shg_spectrum_view_combo)
        cosmic_form.addRow(self.shg_cosmic_enable_chk)
        cosmic_form.addRow("Threshold (MAD)", self.shg_cosmic_threshold_spin)
        cosmic_form.addRow("Detection window", self.shg_cosmic_window_spin)
        cosmic_form.addRow("Maximum width", self.shg_cosmic_max_width_spin)
        cosmic_form.addRow("Spectrum view", self.shg_spectrum_view_combo)
        self._set_form_label_width(cosmic_form, UI_METRICS["label_col_width"])
        layout.addWidget(self._make_expander("Cosmic Rays", cosmic_box, expanded=False))

        angle_box = QGroupBox("Measured Angle")
        angle_form = QFormLayout(angle_box)
        angle_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        angle_form.setSpacing(6)
        self.shg_angle_scale_spin = QDoubleSpinBox()
        self.shg_angle_scale_spin.setDecimals(6)
        self.shg_angle_scale_spin.setRange(-1.0e6, 1.0e6)
        self.shg_angle_scale_spin.setValue(1.0)
        self.shg_angle_offset_spin = QDoubleSpinBox()
        self.shg_angle_offset_spin.setDecimals(6)
        self.shg_angle_offset_spin.setRange(-1.0e6, 1.0e6)
        self.shg_angle_offset_spin.setValue(0.0)
        self.shg_angle_wrap_combo = QComboBox()
        self.shg_angle_wrap_combo.addItems(["None", "0-180°", "0-360°"])
        self._style_combo_popup(self.shg_angle_wrap_combo)
        self.shg_include_failed_chk = QCheckBox("Include move/acquisition failures")
        self.shg_angle_cursor_spin = QDoubleSpinBox()
        self.shg_angle_cursor_spin.setDecimals(6)
        self.shg_angle_cursor_spin.setRange(-1.0e9, 1.0e9)
        self.shg_angle_cursor_spin.setSingleStep(1.0)
        angle_form.addRow("Scale", self.shg_angle_scale_spin)
        angle_form.addRow("Offset (deg)", self.shg_angle_offset_spin)
        angle_form.addRow("Wrap", self.shg_angle_wrap_combo)
        angle_form.addRow("Rows", self.shg_include_failed_chk)
        angle_form.addRow("Selected angle", self.shg_angle_cursor_spin)
        self._set_form_label_width(angle_form, UI_METRICS["label_col_width"])
        layout.addWidget(self._make_expander("Angle", angle_box, expanded=False))

        fit_box = QGroupBox("Angular Fit")
        fit_form = QFormLayout(fit_box)
        fit_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        fit_form.setSpacing(6)
        self.shg_fit_enable_chk = QCheckBox("Fit I(θ) = I₀ + A cos²[2(θ-xc)]")
        self.shg_fit_enable_chk.setChecked(True)
        self.shg_fit_min_spin = QDoubleSpinBox()
        self.shg_fit_max_spin = QDoubleSpinBox()
        for spin, value in ((self.shg_fit_min_spin, 0.0), (self.shg_fit_max_spin, 180.0)):
            spin.setDecimals(4)
            spin.setRange(-1.0e6, 1.0e6)
            spin.setValue(value)
        fit_range_row = QWidget()
        fit_range_layout = QHBoxLayout(fit_range_row)
        fit_range_layout.setContentsMargins(0, 0, 0, 0)
        fit_range_layout.setSpacing(6)
        fit_range_layout.addWidget(self.shg_fit_min_spin)
        fit_range_layout.addWidget(QLabel("to"))
        fit_range_layout.addWidget(self.shg_fit_max_spin)
        fit_range_layout.addWidget(QLabel("deg"))
        self.shg_fit_weighted_chk = QCheckBox("Use area uncertainty weights")
        self.shg_fit_weighted_chk.setChecked(True)
        self.shg_fit_include_excluded_chk = QCheckBox("Include excluded acquisition rows")
        self.shg_fit_branch_spin = QSpinBox()
        self.shg_fit_branch_spin.setRange(-3, 3)
        self.shg_fit_branch_spin.setValue(0)
        self.shg_fit_branch_spin.setEnabled(False)
        self.shg_fit_branch_spin.setToolTip("Adds 90° per branch to Δxc, equivalent to 60° per twist branch")
        self.shg_fit_summary = QPlainTextEdit()
        self.shg_fit_summary.setReadOnly(True)
        self.shg_fit_summary.setMaximumHeight(150)
        fit_form.addRow(self.shg_fit_enable_chk)
        fit_form.addRow("Fit angle range", fit_range_row)
        fit_form.addRow(self.shg_fit_weighted_chk)
        fit_form.addRow(self.shg_fit_include_excluded_chk)
        fit_form.addRow("Phase branch", self.shg_fit_branch_spin)
        fit_form.addRow(self.shg_fit_summary)
        self._set_form_label_width(fit_form, UI_METRICS["label_col_width"])
        layout.addWidget(self._make_expander("Angular Fit", fit_box, expanded=True))
        self._shg_update_cosmic_controls()
        layout.addStretch(1)
        return tab

    def _build_power_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        grouping = QGroupBox("Power Sweep Files")
        grouping_form = QFormLayout(grouping)
        grouping_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        grouping_form.setHorizontalSpacing(6)
        grouping_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        self.power_group_combo = QComboBox()
        self._style_combo_popup(self.power_group_combo)
        self.power_group_combo.setToolTip("Select a full-sweep CSV with Power_uW, or a legacy filename-based series.")
        self.power_refresh_groups_btn = QPushButton("Detect")
        self.power_refresh_groups_btn.setToolTip("Detect Power_uW tables and legacy filename-based power series.")
        group_row = QWidget()
        group_h = QHBoxLayout(group_row)
        group_h.setContentsMargins(0, 0, 0, 0)
        group_h.setSpacing(6)
        group_h.addWidget(self.power_group_combo, 1)
        group_h.addWidget(self.power_refresh_groups_btn)
        self.power_group_summary = QPlainTextEdit()
        self.power_group_summary.setReadOnly(True)
        self.power_group_summary.setMaximumHeight(88)
        self.power_kk_group_combo = QComboBox()
        self.power_kkp_group_combo = QComboBox()
        self._style_combo_popup(self.power_kk_group_combo)
        self._style_combo_popup(self.power_kkp_group_combo)
        self.power_kk_group_combo.setToolTip("Power-sweep source to treat as KK in VP view.")
        self.power_kkp_group_combo.setToolTip("Power-sweep source to treat as KKp in VP view.")
        grouping_form.addRow("Intensity", group_row)
        grouping_form.addRow("Summary", self.power_group_summary)
        grouping_form.addRow("KK sweep", self.power_kk_group_combo)
        grouping_form.addRow("KKp sweep", self.power_kkp_group_combo)
        layout.addWidget(self._make_expander("Power Sweep Files", grouping, expanded=True))

        params = QWidget()
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(4)

        _grid, spins, _, _, cmap, fix_checks = self._build_common_range_grid("power", "turbo")
        self.power_axis_scale_combo = QComboBox()
        self.power_axis_scale_combo.addItems(["Linear", "Log"])
        self._style_combo_popup(self.power_axis_scale_combo)
        self.power_axis_scale_combo.setToolTip("Set the power y-axis scale.")
        self.power_pair_mode_combo = QComboBox()
        self.power_pair_mode_combo.addItems(["Stage", "Power Interpolation"])
        self._style_combo_popup(self.power_pair_mode_combo)
        self.power_pair_mode_combo.setToolTip("Choose how KK and KKp spectra are paired for VP.")
        self.power_background_spin = QDoubleSpinBox()
        self.power_background_spin.setDecimals(6)
        self.power_background_spin.setRange(-1.0e12, 1.0e12)
        self.power_background_spin.setSingleStep(100.0)
        self.power_background_spin.setFixedWidth(UI_METRICS["spin_w"] + 18)
        self.power_background_auto_chk = QCheckBox("Auto")
        self.power_background_auto_chk.setChecked(True)
        self.power_background_auto_chk.setToolTip("Estimate one constant background from low-percentile intensity.")
        bkg_row = QWidget()
        bkg_h = QHBoxLayout(bkg_row)
        bkg_h.setContentsMargins(0, 0, 0, 0)
        bkg_h.setSpacing(8)
        bkg_h.addWidget(self.power_background_spin)
        bkg_h.addWidget(self.power_background_auto_chk)
        bkg_h.addStretch(1)

        setup = QGroupBox("Plot Setup")
        setup_grid = QGridLayout(setup)
        setup_grid.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        setup_grid.setHorizontalSpacing(8)
        setup_grid.setVerticalSpacing(UI_METRICS["row_spacing"])
        setup_grid.addWidget(QLabel("Power Axis"), 0, 0)
        setup_grid.addWidget(self.power_axis_scale_combo, 0, 1)
        setup_grid.addWidget(QLabel("Cmap"), 0, 2)
        setup_grid.addWidget(cmap, 0, 3)
        setup_grid.addWidget(QLabel("VP Pair By"), 1, 0)
        setup_grid.addWidget(self.power_pair_mode_combo, 1, 1)
        setup_grid.addWidget(QLabel("Background"), 1, 2)
        setup_grid.addWidget(bkg_row, 1, 3)
        setup_grid.setColumnStretch(1, 1)
        setup_grid.setColumnStretch(3, 1)
        params_layout.addWidget(self._make_expander("Plot Setup", setup, expanded=True))

        for s in spins.values():
            s.setFixedWidth(UI_METRICS["spin_w"])
            s.setFixedHeight(UI_METRICS["input_h"])

        self.power_auto_v_btn = QToolButton()
        self.power_auto_x_btn = QToolButton()
        self.power_auto_y_btn = QToolButton()
        basic = QGroupBox("Axis Ranges")
        basic_form = QFormLayout(basic)
        basic_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        basic_form.setHorizontalSpacing(4)
        basic_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        basic_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        basic_form.addRow(
            "vmin / vmax",
            self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.power_auto_v_btn, "Auto V"),
        )
        basic_form.addRow("Color scale", self.power_split_scale_chk)
        basic_form.addRow(self.power_split_scale_panel)
        basic_form.addRow(
            "xmin / xmax",
            self._make_axis_range_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.power_auto_x_btn, "Auto X"),
        )
        basic_form.addRow(
            "pmin / pmax",
            self._make_axis_range_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.power_auto_y_btn, "Auto Y"),
        )
        basic_form.addRow("Cursor Power", spins["gate"])
        flags = QWidget()
        flags_h = QHBoxLayout(flags)
        flags_h.setContentsMargins(0, 0, 0, 0)
        flags_h.setSpacing(10)
        self.power_log_chk.setText("Color Log")
        self.power_log_chk.setToolTip("Use logarithmic color normalization.")
        flags_h.addWidget(self.power_log_chk)
        flags_h.addWidget(self.power_clip_chk)
        flags_h.addStretch(1)
        basic_form.addRow("Color / Clip", flags)
        self._set_form_label_width(basic_form, UI_METRICS["label_col_width"])
        params_layout.addWidget(self._make_expander("Manual plot ranges", basic, expanded=False))

        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addStretch(1)
        return tab

    def _cmp_assign_candidate_files(self) -> list[str]:
        return list(self.available_files)

    def _shg_selected_file(self) -> str:
        if not hasattr(self, "shg_files"):
            return ""
        selected = self._selected(self.shg_files)
        return selected[0] if selected else ""

    def _shg_background_file(self) -> str:
        return str(self.shg_background_combo.currentData() or "") if hasattr(self, "shg_background_combo") else ""

    def _shg_compare_mode(self) -> bool:
        return hasattr(self, "shg_workflow_tabs") and self.shg_workflow_tabs.currentIndex() == 1

    @staticmethod
    def _combo_data_text(combo: QComboBox) -> str:
        return str(combo.currentData() or "")

    def _shg_compare_files(self) -> tuple[str, str]:
        return (
            self._combo_data_text(self.shg_compare_reference_combo),
            self._combo_data_text(self.shg_compare_sample_combo),
        )

    def _shg_compare_background_files(self) -> tuple[str, str]:
        return (
            self._combo_data_text(self.shg_compare_background_a_combo),
            self._combo_data_text(self.shg_compare_background_b_combo),
        )

    def _shg_refresh_sources(self) -> None:
        if not hasattr(self, "shg_files"):
            return
        old_source = self._shg_selected_file()
        old_background = self._shg_background_file()
        old_compare = self._shg_compare_files()
        old_compare_backgrounds = self._shg_compare_background_files()
        source_blocked = self.shg_files.blockSignals(True)
        background_blocked = self.shg_background_combo.blockSignals(True)
        compare_combos = (
            self.shg_compare_reference_combo,
            self.shg_compare_sample_combo,
            self.shg_compare_background_a_combo,
            self.shg_compare_background_b_combo,
        )
        compare_blocked = [combo.blockSignals(True) for combo in compare_combos]
        try:
            self.shg_files.clear()
            self.shg_files.addItems(self.available_files)
            if old_source in self.available_files:
                self._restore_list_selection(self.shg_files, [old_source])
            self.shg_background_combo.clear()
            self.shg_background_combo.addItem("— None —", "")
            for file_name in self.available_files:
                self.shg_background_combo.addItem(file_name, file_name)
            background_index = self.shg_background_combo.findData(old_background)
            self.shg_background_combo.setCurrentIndex(background_index if background_index >= 0 else 0)
            for combo in (self.shg_compare_reference_combo, self.shg_compare_sample_combo):
                combo.clear()
                combo.addItem("— Select —", "")
                for file_name in self.available_files:
                    combo.addItem(file_name, file_name)
            for combo in (self.shg_compare_background_a_combo, self.shg_compare_background_b_combo):
                combo.clear()
                combo.addItem("— None —", "")
                for file_name in self.available_files:
                    combo.addItem(file_name, file_name)
            for combo, previous in zip(
                (self.shg_compare_reference_combo, self.shg_compare_sample_combo),
                old_compare,
            ):
                index = combo.findData(previous)
                combo.setCurrentIndex(index if index >= 0 else 0)
            for combo, previous in zip(
                (self.shg_compare_background_a_combo, self.shg_compare_background_b_combo),
                old_compare_backgrounds,
            ):
                index = combo.findData(previous)
                combo.setCurrentIndex(index if index >= 0 else 0)
            if old_compare[0] not in self.available_files and self.available_files:
                self.shg_compare_reference_combo.setCurrentIndex(1)
            if old_compare[1] not in self.available_files and len(self.available_files) > 1:
                self.shg_compare_sample_combo.setCurrentIndex(2)
        finally:
            self.shg_files.blockSignals(source_blocked)
            self.shg_background_combo.blockSignals(background_blocked)
            for combo, blocked in zip(compare_combos, compare_blocked):
                combo.blockSignals(blocked)
        self._shg_update_background_controls()
        self._shg_update_summary()

    def _mcd_refresh_sources(self) -> None:
        if not hasattr(self, "mcd_files"):
            return
        selected = self._selected(self.mcd_files)
        dark_pos = str(self.mcd_dark_pos_combo.currentData() or "")
        dark_neg = str(self.mcd_dark_neg_combo.currentData() or "")
        widgets = (self.mcd_files, self.mcd_dark_pos_combo, self.mcd_dark_neg_combo)
        blocked = [widget.blockSignals(True) for widget in widgets]
        try:
            self.mcd_files.clear()
            for file_name in self.available_files:
                item = QListWidgetItem(file_name)
                item.setToolTip(file_name)
                self.mcd_files.addItem(item)
            self._restore_list_selection(self.mcd_files, [file_name for file_name in selected if file_name in self.available_files])
            for combo, old in ((self.mcd_dark_pos_combo, dark_pos), (self.mcd_dark_neg_combo, dark_neg)):
                combo.clear(); combo.addItem("-- No dark / offset file --", "")
                for file_name in self.available_files:
                    combo.addItem(file_name, file_name)
                combo.setCurrentIndex(max(0, combo.findData(old)))
        finally:
            for widget, state in zip(widgets, blocked):
                widget.blockSignals(state)
        self._mcd_detect_available_angles()

    def _mcd_detect_available_angles(self) -> None:
        if not self.current_folder:
            return
        selected = self._selected(self.mcd_files)
        source = selected[0] if selected else ""
        if not source:
            return
        try:
            angles = detect_angles(str(Path(self.current_folder) / source))
        except Exception as exc:
            self.mcd_source_summary.setText(f"Could not read MCD angles: {str(exc).splitlines()[0]}")
            return
        if len(angles) < 2:
            self.mcd_source_summary.setText("MCD CSV needs at least two distinct angle values.")
            return
        plus_old = self.mcd_sigma_plus_combo.currentData()
        minus_old = self.mcd_sigma_minus_combo.currentData()
        combos = (self.mcd_sigma_plus_combo, self.mcd_sigma_minus_combo)
        blocked = [combo.blockSignals(True) for combo in combos]
        try:
            for combo in combos:
                combo.clear()
                for angle in angles:
                    combo.addItem(f"{angle:g} deg", float(angle))
            if self.mcd_auto_angles_chk.isChecked():
                self.mcd_sigma_plus_combo.setCurrentIndex(len(angles) - 1)
                self.mcd_sigma_minus_combo.setCurrentIndex(0)
            else:
                plus_index = self.mcd_sigma_plus_combo.findData(plus_old)
                minus_index = self.mcd_sigma_minus_combo.findData(minus_old)
                self.mcd_sigma_plus_combo.setCurrentIndex(plus_index if plus_index >= 0 else len(angles) - 1)
                self.mcd_sigma_minus_combo.setCurrentIndex(minus_index if minus_index >= 0 else 0)
        finally:
            for combo, state in zip(combos, blocked):
                combo.blockSignals(state)
        self.mcd_source_summary.setText("Detected angles: " + ", ".join(f"{angle:g} deg" for angle in angles))

    def _mcd_settings_from_ui(self) -> McdSettings:
        gain = {
            "Per wavelength": "per_wavelength",
            "Smoothed per wavelength": "smoothed",
            "Scalar (diagnostic only)": "scalar",
        }.get(self.mcd_gain_combo.currentText(), "per_wavelength")
        sigma_plus = self.mcd_sigma_plus_combo.currentData()
        sigma_minus = self.mcd_sigma_minus_combo.currentData()
        if sigma_plus is None or sigma_minus is None:
            raise ValueError("Select an MCD source CSV so its sigma+ and sigma- angles can be detected.")
        background_ranges: list[tuple[float, float]] = []
        range_text = self.mcd_background_ranges_edit.text().replace(";", ",").replace("–", "-").strip()
        for item in (part.strip() for part in range_text.split(",") if part.strip()):
            values = item.split("-")
            if len(values) != 2:
                raise ValueError("Fit background energies must be comma-separated ranges such as 1.50-1.58, 1.73-1.79.")
            try:
                start, stop = (float(value.strip()) for value in values)
            except ValueError as exc:
                raise ValueError("Fit background energies must be numeric eV ranges.") from exc
            if start >= stop:
                raise ValueError("Each fit background energy range must increase from left to right.")
            background_ranges.append((start, stop))
        correction_mode = {
            "Global reference gain (current)": "global",
            "Global gain + per-pair scale": "pair_scale",
            "Global gain + per-pair scale/offset": "pair_affine",
            "Global gain + per-pair spectral baseline": "pair_spectral",
        }.get(self.mcd_correction_mode_combo.currentText(), "global")
        suggestion = self._mcd_background_suggestion
        suggestion_active = bool(
            suggestion is not None
            and range_text == self._format_mcd_background_ranges(suggestion.ranges)
        )
        selection_mode = "suggested" if suggestion_active else ("manual" if background_ranges else "auto")
        return McdSettings(
            pos_angle=float(sigma_plus),
            neg_angle=float(sigma_minus),
            max_sequence_gap=int(self.mcd_gap_spin.value()), max_delta_b=float(self.mcd_delta_b_spin.value()),
            pair_b_alignment=("interpolate" if self.mcd_pair_alignment_combo.currentIndex() == 1 else "direct"),
            zero_window_t=float(self.mcd_zero_spin.value()),
            reference_mode=("nearest" if self.mcd_reference_mode_combo.currentIndex() == 0 else "window"),
            bin_decimals=int(self.mcd_bin_spin.value()),
            gain_mode=gain, correction_mode=correction_mode,
            spectral_order=(2 if self.mcd_spectral_order_combo.currentIndex() == 1 else 1),
            background_ranges_ev=tuple(background_ranges),
            background_selection=selection_mode,
            suggestion_protected_ranges_ev=(suggestion.protected_ranges if suggestion_active else ()),
            manual_protected_ranges_ev=(suggestion.manual_protected_ranges if suggestion_active else ()),
            suggestion_linear_validation_rms=(suggestion.linear_validation_rms if suggestion_active else None),
            suggestion_quadratic_validation_rms=(suggestion.quadratic_validation_rms if suggestion_active else None),
            suggestion_algorithm=("manual_unprotected_bands_v3" if suggestion_active else None),
            dark_pos_file=str(self.mcd_dark_pos_combo.currentData() or "") or None,
            dark_neg_file=str(self.mcd_dark_neg_combo.currentData() or "") or None,
        )

    @staticmethod
    def _format_mcd_background_ranges(ranges: Sequence[tuple[float, float]]) -> str:
        return ", ".join(f"{float(start):.4f}-{float(stop):.4f}" for start, stop in ranges)

    def _update_mcd_background_preview(self) -> None:
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            self.mcd_background_preview.setText("Auto outer 15% ranges are shown after loading an MCD sweep.")
            return
        text = self.mcd_background_ranges_edit.text().strip()
        if text:
            self.mcd_background_preview.setText("Manual ranges are active. Blue shading shows the ranges used.")
            return
        ranges = background_fit_regions(self.loaded.mcd_result.energy_ev, ())
        self.mcd_background_preview.setText(f"Using: {self._format_mcd_background_ranges(ranges)} eV")

    def _suggest_mcd_background_ranges(self) -> None:
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            self._show_error("Load an MCD sweep before requesting a full-sweep background suggestion.")
            return
        result = self.loaded.mcd_result
        energy = np.asarray(result.energy_ev, float)
        center = float(self.mcd_window_center_spin.value())
        if not np.isfinite(center) or center <= float(np.nanmin(energy)) or center >= float(np.nanmax(energy)):
            center = float(np.nanmedian(energy))
        # Protect the active MCD(B) window plus a guard.  Feature protection
        # is deliberately manual: the review plot opens ready for the user to
        # drag exactly across each resonance they want excluded.
        guard_ev = max(0.010, float(self.mcd_window_width_spin.value()) * 1e-3)
        protected = ((center - guard_ev, center + guard_ev),)
        prior_manual: tuple[tuple[float, float], ...] = ()
        if self._mcd_background_suggestion is not None:
            prior_manual = tuple(self._mcd_background_suggestion.manual_protected_ranges)
        elif self.loaded.mcd_settings is not None:
            prior_manual = tuple(self.loaded.mcd_settings.manual_protected_ranges_ev)
        try:
            suggestion = suggest_mcd_background_ranges(
                result,
                protected_ranges_ev=protected + prior_manual,
                auto_detect_features=False,
                use_all_unprotected_bands=True,
            )
            suggestion = replace(
                suggestion,
                requested_protected_ranges=protected,
                manual_protected_ranges=prior_manual,
            )
        except Exception as exc:
            self._show_error(f"Could not suggest MCD background ranges: {exc}")
            return
        self._show_mcd_background_suggestion_dialog(result, suggestion)

    def _show_mcd_background_suggestion_dialog_legacy(self, suggestion: McdBackgroundSuggestion) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Review MCD background suggestion")
        if not self.windowIcon().isNull():
            dlg.setWindowIcon(self.windowIcon())
        dlg.setMinimumSize(760, 560)
        dlg.resize(930, 680)
        layout = QVBoxLayout(dlg)
        figure = Figure(figsize=(8.2, 4.8), dpi=100)
        canvas = FigureCanvasQTAgg(figure)
        spectrum_ax, score_ax = figure.subplots(2, 1, sharex=True, height_ratios=[2.2, 1.0])
        energy = np.asarray(suggestion.energy_ev, float)
        spectrum_ax.plot(energy, suggestion.median_reflectance, color="#303030", lw=1.1, label="full-sweep median reflectance")
        for index, (start, stop) in enumerate(suggestion.ranges):
            spectrum_ax.axvspan(start, stop, color="#5790b7", alpha=0.24, label="suggested fit band" if index == 0 else "_nolegend_")
        for index, (start, stop) in enumerate(suggestion.detected_feature_ranges):
            spectrum_ax.axvspan(start, stop, color="#e28743", alpha=0.20, label="auto-detected feature" if index == 0 else "_nolegend_")
        for index, (start, stop) in enumerate(suggestion.requested_protected_ranges):
            spectrum_ax.axvspan(start, stop, color="#c94c00", alpha=0.22, label="active protected window" if index == 0 else "_nolegend_")
        spectrum_ax.set_ylabel("Reflection (a.u.)")
        spectrum_ax.grid(alpha=0.22)
        spectrum_ax.legend(fontsize=8, loc="best")
        score_ax.plot(energy, suggestion.suitability, color="#6a3d9a", lw=0.9)
        for start, stop in suggestion.ranges:
            score_ax.axvspan(start, stop, color="#5790b7", alpha=0.24)
        for start, stop in suggestion.detected_feature_ranges:
            score_ax.axvspan(start, stop, color="#e28743", alpha=0.20)
        for start, stop in suggestion.requested_protected_ranges:
            score_ax.axvspan(start, stop, color="#c94c00", alpha=0.22)
        score_ax.set_xlabel("Energy (eV)")
        score_ax.set_ylabel("feature / noise score")
        score_ax.grid(alpha=0.22)
        figure.tight_layout(pad=1.0)
        layout.addWidget(canvas, 1)
        linear = suggestion.linear_validation_rms
        quadratic = suggestion.quadratic_validation_rms
        summary = QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(132)
        summary.setPlainText(
            "Suggested manual background ranges (eV):\n"
            f"{self._format_mcd_background_ranges(suggestion.ranges)}\n\n"
            f"Active protected window: {self._format_mcd_background_ranges(suggestion.requested_protected_ranges)} eV\n"
            f"Auto-detected feature windows: {self._format_mcd_background_ranges(suggestion.detected_feature_ranges) or 'none'} eV\n"
            f"All excluded windows: {self._format_mcd_background_ranges(suggestion.protected_ranges)} eV\n"
            f"Coverage: {100.0 * suggestion.coverage_fraction:.1f}% of valid spectral points; "
            f"span: {100.0 * suggestion.span_fraction:.1f}% of the energy axis\n"
            f"Held-out RMS — linear: {linear:.4g}; quadratic: {quadratic:.4g}\n"
            f"Held-out diagnostic preference: {'Quadratic' if suggestion.suggested_order == 2 else 'Linear'} (does not change your selected order)\n\n"
            + "\n".join(f"- {note}" for note in suggestion.notes)
        )
        layout.addWidget(summary)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        accept = buttons.addButton("Use suggestion", QDialogButtonBox.AcceptRole)
        accept.setToolTip("Copy the suggested ranges and model into the MCD controls. It will not reprocess until Apply is clicked.")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.Accepted:
            return
        self._mcd_background_suggestion = suggestion
        self.mcd_background_ranges_edit.setText(self._format_mcd_background_ranges(suggestion.ranges))
        self.mcd_correction_mode_combo.setCurrentText("Global gain + per-pair spectral baseline")
        self._update_mcd_background_preview()
        self._on_mcd_params_changed()
        self._mcd_auto_apply_timer.start(0)
        self._status("Selected protection regions saved; MCD recalculation is starting automatically.")

    def _show_mcd_background_suggestion_dialog(self, result: McdResult, suggestion: McdBackgroundSuggestion) -> None:
        """Let the user draw exact feature-protection windows before fitting."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Select MCD feature-protection regions")
        if not self.windowIcon().isNull():
            dlg.setWindowIcon(self.windowIcon())
        dlg.setMinimumSize(800, 660)
        dlg.resize(980, 820)
        layout = QVBoxLayout(dlg)
        figure = Figure(figsize=(8.2, 4.7), dpi=100)
        canvas = FigureCanvasQTAgg(figure)
        spectrum_ax, score_ax = figure.subplots(2, 1, sharex=True, height_ratios=[2.2, 1.0])
        layout.addWidget(canvas, 1)
        energy = np.asarray(suggestion.energy_ev, float)
        energy_min, energy_max = float(np.nanmin(energy)), float(np.nanmax(energy))

        feature_box = QGroupBox("Feature-protection windows")
        feature_layout = QVBoxLayout(feature_box)
        note = QLabel(
            "Drag horizontally on the reflection plot to add a protection window. "
            "Only the active MCD(B) window and the regions you select are excluded from the blue background-fit bands."
        )
        note.setWordWrap(True)
        feature_layout.addWidget(note)
        table = QTableWidget(0, 8)
        table.setHorizontalHeaderLabels(["Use", "Type", "Center (eV)", "Start (eV)", "Stop (eV)", "Width (meV)", "SNR", "Status"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setMaximumHeight(180)
        feature_layout.addWidget(table)
        controls = QHBoxLayout()
        remove = QPushButton("Remove selected")
        select_range = QPushButton("Draw protection region")
        select_range.setToolTip("Drag horizontally on the upper reflection plot to protect exactly that energy region.")
        controls.addWidget(remove)
        controls.addStretch(1)
        controls.addWidget(select_range)
        feature_layout.addLayout(controls)
        layout.addWidget(feature_box)

        summary = QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(125)
        layout.addWidget(summary)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        accept = buttons.addButton("Use selected regions", QDialogButtonBox.AcceptRole)
        accept.setToolTip("Copy fit ranges recalculated from your selected protection regions. Processing starts only after Apply.")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        row_sources: list[str] = []
        row_features: list[object | None] = []
        current: dict[str, McdBackgroundSuggestion] = {"value": suggestion}
        is_refreshing = {"value": False}
        selection_state: dict[str, object] = {"active": True, "selector": None}

        def on_manual_span(left: float, right: float) -> None:
            if abs(float(right) - float(left)) < 5e-4:
                return
            selection_state["active"] = True
            add_row("Manual", min(left, right), max(left, right), source="manual")
            select_range.setText("Drawing: drag another region")
            refresh()

        def configure_span_selector() -> None:
            old_selector = selection_state.get("selector")
            if old_selector is not None and hasattr(old_selector, "disconnect_events"):
                old_selector.disconnect_events()
            selector = SpanSelector(
                spectrum_ax, on_manual_span, "horizontal", useblit=False,
                props={"facecolor": "#8e44ad", "alpha": 0.25}, interactive=False,
            )
            selector.set_active(bool(selection_state["active"]))
            selection_state["selector"] = selector

        def selected_windows() -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...], tuple[tuple[float, float], ...], tuple[str, ...]]:
            active: list[tuple[float, float]] = []
            automatic: list[tuple[float, float]] = []
            manual: list[tuple[float, float]] = []
            automatic_kinds: list[str] = []
            for row in range(table.rowCount()):
                checkbox = table.cellWidget(row, 0)
                left = table.cellWidget(row, 3)
                right = table.cellWidget(row, 4)
                if not isinstance(checkbox, QCheckBox) or not checkbox.isChecked():
                    continue
                if not isinstance(left, QDoubleSpinBox) or not isinstance(right, QDoubleSpinBox):
                    continue
                window = tuple(sorted((float(left.value()), float(right.value()))))
                source = row_sources[row]
                if source == "active":
                    active.append(window)
                elif source == "auto":
                    automatic.append(window)
                    label = table.item(row, 1).text() if table.item(row, 1) else "feature"
                    automatic_kinds.append(label.removeprefix("Auto "))
                else:
                    manual.append(window)
            return tuple(active), tuple(automatic), tuple(manual), tuple(automatic_kinds)

        def merged_enabled_windows() -> tuple[tuple[float, float], ...]:
            active, automatic, manual, _kinds = selected_windows()
            windows = sorted(tuple(active) + tuple(automatic) + tuple(manual))
            merged: list[tuple[float, float]] = []
            for left, right in windows:
                if not merged or left > merged[-1][1] + 1e-12:
                    merged.append((left, right))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            return tuple(merged)

        def unchecked_auto_windows() -> tuple[tuple[float, float], ...]:
            windows: list[tuple[float, float]] = []
            for row, source in enumerate(row_sources):
                if source != "auto":
                    continue
                checkbox = table.cellWidget(row, 0)
                left = table.cellWidget(row, 3)
                right = table.cellWidget(row, 4)
                if isinstance(checkbox, QCheckBox) and not checkbox.isChecked() and isinstance(left, QDoubleSpinBox) and isinstance(right, QDoubleSpinBox):
                    windows.append(tuple(sorted((float(left.value()), float(right.value())))))
            return tuple(windows)

        def redraw() -> None:
            displayed = current["value"]
            active, automatic, manual, _kinds = selected_windows()
            unchecked_auto = unchecked_auto_windows()
            spectrum_ax.clear()
            score_ax.clear()
            spectrum_ax.plot(energy, displayed.median_reflectance, color="#303030", lw=1.1, label="full-sweep median reflection")
            if np.any(np.isfinite(displayed.feature_baseline)):
                spectrum_ax.plot(energy, np.exp(displayed.feature_baseline), color="#8c8c8c", lw=0.9, ls="--", label="broad reflection baseline")
            for index, (left, right) in enumerate(displayed.ranges):
                spectrum_ax.axvspan(left, right, color="#5790b7", alpha=0.24, label="suggested fit band" if index == 0 else "_nolegend_")
            for index, (left, right) in enumerate(active):
                spectrum_ax.axvspan(left, right, color="#c94c00", alpha=0.22, label="active MCD(B) window" if index == 0 else "_nolegend_")
            for index, (left, right) in enumerate(automatic):
                spectrum_ax.axvspan(left, right, color="#e28743", alpha=0.22, label="enabled auto feature" if index == 0 else "_nolegend_")
            for index, (left, right) in enumerate(manual):
                spectrum_ax.axvspan(left, right, color="#8e44ad", alpha=0.20, label="manual protection" if index == 0 else "_nolegend_")
            for index, (left, right) in enumerate(unchecked_auto):
                spectrum_ax.axvspan(left, right, facecolor="none", edgecolor="#e28743", lw=0.9, ls="--", alpha=0.8, label="review-only candidate" if index == 0 else "_nolegend_")
            for feature in suggestion.detected_features:
                marker = "^" if feature.kind == "peak" else "v"
                color = "#e28743" if feature.recommended else "#b27b4d"
                y = float(np.interp(feature.center_ev, energy, displayed.median_reflectance))
                spectrum_ax.plot(feature.center_ev, y, marker=marker, color=color, ms=5, zorder=4)
                if feature.recommended:
                    spectrum_ax.annotate(
                        f"{feature.center_ev:.4f} eV\nSNR {feature.snr:.1f}",
                        (feature.center_ev, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=color,
                    )
            spectrum_ax.set_ylabel("Reflection (a.u.)")
            spectrum_ax.grid(alpha=0.22)
            spectrum_ax.legend(fontsize=8, loc="best")
            score_ax.plot(energy, displayed.suitability, color="#6a3d9a", lw=0.9, label="background unsuitability")
            if np.any(np.isfinite(displayed.feature_detection_score)):
                score_ax.plot(energy, displayed.feature_detection_score, color="#e28743", lw=0.8, alpha=0.85, label="feature SNR score")
            for left, right in displayed.ranges:
                score_ax.axvspan(left, right, color="#5790b7", alpha=0.24)
            for left, right in active:
                score_ax.axvspan(left, right, color="#c94c00", alpha=0.18)
            for left, right in automatic:
                score_ax.axvspan(left, right, color="#e28743", alpha=0.18)
            for left, right in manual:
                score_ax.axvspan(left, right, color="#8e44ad", alpha=0.16)
            score_ax.set_xlabel("Energy (eV)")
            score_ax.set_ylabel("background / feature score")
            score_ax.grid(alpha=0.22)
            score_ax.legend(fontsize=8, loc="best")
            figure.tight_layout(pad=1.0)
            candidate_lines = [
                f"{feature.center_ev:.4f} eV  {feature.kind}  width {1000.0 * feature.width_ev:.1f} meV  "
                f"prominence {100.0 * np.expm1(feature.prominence_log):.1f}%  SNR {feature.snr:.1f}  "
                f"{'recommended' if feature.recommended else 'review only'}"
                for feature in suggestion.detected_features
            ]
            summary.setPlainText(
                "Suggested manual background ranges (eV):\n"
                f"{self._format_mcd_background_ranges(displayed.ranges)}\n\n"
                f"Enabled protection windows: {self._format_mcd_background_ranges(displayed.protected_ranges) or 'none'} eV\n"
                f"Coverage: {100.0 * displayed.coverage_fraction:.1f}% of valid spectral points; "
                f"span: {100.0 * displayed.span_fraction:.1f}% of the energy axis\n"
                f"Held-out RMS - linear: {displayed.linear_validation_rms:.4g}; quadratic: {displayed.quadratic_validation_rms:.4g}\n"
                f"Held-out diagnostic preference: {'Quadratic' if displayed.suggested_order == 2 else 'Linear'} (does not change your selected order)\n\n"
                + "\n".join(f"- {item}" for item in displayed.notes)
                + ("\n\nDetected reflection candidates:\n" + "\n".join(candidate_lines) if candidate_lines else "")
            )
            configure_span_selector()
            canvas.draw_idle()

        def refresh() -> None:
            if is_refreshing["value"]:
                return
            is_refreshing["value"] = True
            try:
                active, automatic, manual, automatic_kinds = selected_windows()
                selected = merged_enabled_windows()
                recalculated = suggest_mcd_background_ranges(
                    result,
                    protected_ranges_ev=selected,
                    auto_detect_features=False,
                    use_all_unprotected_bands=True,
                )
                current["value"] = replace(
                    recalculated,
                    requested_protected_ranges=active,
                    manual_protected_ranges=manual,
                    detected_feature_ranges=automatic,
                    detected_feature_kinds=automatic_kinds,
                    feature_baseline=suggestion.feature_baseline,
                    feature_residual=suggestion.feature_residual,
                    feature_detection_score=suggestion.feature_detection_score,
                    detected_features=suggestion.detected_features,
                )
                redraw()
            except Exception as exc:
                summary.setPlainText(f"No usable fit bands remain for these protection windows:\n{exc}")
            finally:
                is_refreshing["value"] = False

        def connect_row(row: int) -> None:
            checkbox = table.cellWidget(row, 0)
            left = table.cellWidget(row, 3)
            right = table.cellWidget(row, 4)
            if isinstance(checkbox, QCheckBox):
                checkbox.toggled.connect(refresh)
            if isinstance(left, QDoubleSpinBox):
                left.valueChanged.connect(refresh)
            if isinstance(right, QDoubleSpinBox):
                right.valueChanged.connect(refresh)
            def update_width() -> None:
                left_control = table.cellWidget(row, 3)
                right_control = table.cellWidget(row, 4)
                width_item = table.item(row, 5)
                if isinstance(left_control, QDoubleSpinBox) and isinstance(right_control, QDoubleSpinBox) and width_item is not None:
                    width_item.setText(f"{1000.0 * abs(right_control.value() - left_control.value()):.2f}")
            if isinstance(left, QDoubleSpinBox):
                left.valueChanged.connect(update_width)
            if isinstance(right, QDoubleSpinBox):
                right.valueChanged.connect(update_width)

        def add_row(kind: str, left: float, right: float, *, source: str, enabled: bool = True, feature: object | None = None) -> None:
            row = table.rowCount()
            table.insertRow(row)
            checkbox = QCheckBox()
            checkbox.setChecked(enabled)
            table.setCellWidget(row, 0, checkbox)
            label = QTableWidgetItem(kind)
            label.setFlags(label.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 1, label)
            center = float(getattr(feature, "center_ev", 0.5 * (left + right)))
            center_item = QTableWidgetItem(f"{center:.6f}")
            center_item.setFlags(center_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 2, center_item)
            for column, value in ((3, left), (4, right)):
                spin = QDoubleSpinBox()
                spin.setRange(energy_min, energy_max)
                spin.setDecimals(6)
                spin.setSingleStep(0.001)
                spin.setValue(float(value))
                table.setCellWidget(row, column, spin)
            width_item = QTableWidgetItem(f"{1000.0 * (right - left):.2f}")
            width_item.setFlags(width_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 5, width_item)
            snr = getattr(feature, "snr", None)
            snr_item = QTableWidgetItem(f"{float(snr):.1f}" if snr is not None else "-")
            snr_item.setFlags(snr_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 6, snr_item)
            status = "recommended" if bool(getattr(feature, "recommended", False)) else ("manual" if source == "manual" else "active" if source == "active" else "review only")
            status_item = QTableWidgetItem(status)
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 7, status_item)
            row_sources.append(source)
            row_features.append(feature)
            connect_row(row)

        for left, right in suggestion.requested_protected_ranges:
            add_row("Active MCD(B)", left, right, source="active")
        for left, right in suggestion.manual_protected_ranges:
            add_row("Manual", left, right, source="manual")
        def begin_manual_selection() -> None:
            selection_state["active"] = True
            selector = selection_state.get("selector")
            if selector is not None:
                selector.set_active(True)
            select_range.setText("Drag on reflection plot…")
            select_range.setToolTip("Drag across the reflection plot to protect exactly that energy region.")

        def remove_selected() -> None:
            row = table.currentRow()
            if row < 0:
                return
            table.removeRow(row)
            row_sources.pop(row)
            row_features.pop(row)
            refresh()

        select_range.clicked.connect(begin_manual_selection)
        remove.clicked.connect(remove_selected)
        redraw()
        if dlg.exec() != QDialog.Accepted:
            return
        accepted = current["value"]
        self._mcd_background_suggestion = accepted
        self.mcd_background_ranges_edit.setText(self._format_mcd_background_ranges(accepted.ranges))
        self.mcd_correction_mode_combo.setCurrentText("Global gain + per-pair spectral baseline")
        self._update_mcd_background_preview()
        self._on_mcd_params_changed()
        self._mcd_auto_apply_timer.start(0)
        self._status("Selected protection regions saved; MCD recalculation is starting automatically.")

    def _shg_fit_settings_from_ui(self) -> ShgFitSettings:
        angle_min = float(self.shg_fit_min_spin.value())
        angle_max = float(self.shg_fit_max_spin.value())
        if angle_min >= angle_max:
            raise ValueError("SHG fit minimum angle must be smaller than the maximum angle.")
        return ShgFitSettings(
            enabled=bool(self.shg_fit_enable_chk.isChecked()),
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            use_uncertainty_weights=bool(self.shg_fit_weighted_chk.isChecked()),
            include_excluded_rows=bool(self.shg_fit_include_excluded_chk.isChecked()),
            phase_branch=int(self.shg_fit_branch_spin.value()),
        )

    def _shg_settings_from_ui(self) -> ShgSettings:
        method_map = {
            "Local linear": "local_linear",
            "Local quadratic": "local_quadratic",
            "External + local residual": "external",
            "None": "none",
        }
        wrap_map = {"None": None, "0-180°": 180.0, "0-360°": 360.0}
        integration_wavelength = float(self.shg_peak_center_spin.value())
        integration_half_range = float(self.shg_gate_half_range_spin.value())
        sideband_gap = float(self.shg_sideband_gap_spin.value())
        sideband_width = float(self.shg_sideband_width_spin.value())
        gate_min = integration_wavelength - integration_half_range
        gate_max = integration_wavelength + integration_half_range
        return ShgSettings(
            peak_center_nm=integration_wavelength,
            gate_min_nm=gate_min,
            gate_max_nm=gate_max,
            left_min_nm=gate_min - sideband_gap - sideband_width,
            left_max_nm=gate_min - sideband_gap,
            right_min_nm=gate_max + sideband_gap,
            right_max_nm=gate_max + sideband_gap + sideband_width,
            background_method=method_map.get(self.shg_background_method_combo.currentText(), "local_linear"),
            sigma_clip=float(self.shg_sigma_clip_spin.value()),
            remove_cosmic_rays=bool(self.shg_cosmic_enable_chk.isChecked()),
            cosmic_threshold_mad=float(self.shg_cosmic_threshold_spin.value()),
            cosmic_window_points=int(self.shg_cosmic_window_spin.value()),
            cosmic_max_width_points=int(self.shg_cosmic_max_width_spin.value()),
            angle_scale=float(self.shg_angle_scale_spin.value()),
            angle_offset_deg=float(self.shg_angle_offset_spin.value()),
            angle_wrap_deg=wrap_map.get(self.shg_angle_wrap_combo.currentText()),
            include_failed_rows=bool(self.shg_include_failed_chk.isChecked()),
        )

    def _shg_update_background_controls(self) -> None:
        if hasattr(self, "shg_background_combo"):
            external = self.shg_background_method_combo.currentText() == "External + local residual"
            self.shg_background_combo.setEnabled(external)
            self.shg_compare_background_a_combo.setEnabled(external)
            self.shg_compare_background_b_combo.setEnabled(external)

    def _shg_update_cosmic_controls(self) -> None:
        if not hasattr(self, "shg_cosmic_enable_chk"):
            return
        window = int(self.shg_cosmic_window_spin.value())
        if window % 2 == 0:
            window = min(self.shg_cosmic_window_spin.maximum(), window + 1)
            self.shg_cosmic_window_spin.setValue(window)
        self.shg_cosmic_max_width_spin.setMaximum(max(1, window - 1))
        enabled = bool(self.shg_cosmic_enable_chk.isChecked())
        for widget in (
            self.shg_cosmic_threshold_spin,
            self.shg_cosmic_window_spin,
            self.shg_cosmic_max_width_spin,
        ):
            widget.setEnabled(enabled)

    def _shg_update_summary(self) -> None:
        if not hasattr(self, "shg_summary"):
            return
        if self.loaded and self.loaded.mode == "SHG Processing" and self.loaded.shg_result is not None:
            result = self.loaded.shg_result
            data = result.data
            included = int(np.count_nonzero(result.included))
            failures = len(result.quality_flags) - included
            measured_column = data.detected_columns.get("measured_angle", "measured position")
            lines = [
                data.source_file,
                f"{data.spectra.shape[0]} acquisitions, {data.wavelength_nm.size} wavelengths",
                f"Wavelength: {float(np.nanmin(data.wavelength_nm)):.6g}-{float(np.nanmax(data.wavelength_nm)):.6g} nm",
                f"Integrated corrected area: {result.settings.peak_center_nm:g} ± "
                f"{0.5 * (result.settings.gate_max_nm - result.settings.gate_min_nm):g} nm",
                f"Background sidebands: {result.settings.left_min_nm:g}-{result.settings.left_max_nm:g} and "
                f"{result.settings.right_min_nm:g}-{result.settings.right_max_nm:g} nm",
                f"Cosmic rays: {int(np.sum(result.cosmic_pixels_removed))} pixel(s) removed from "
                f"{int(np.count_nonzero(result.cosmic_pixels_removed))} acquisition(s)",
                f"Angle column: {measured_column}; included: {included}; excluded/warned: {failures}",
            ]
            self.shg_summary.setPlainText("\n".join(lines))
        elif self.available_files:
            selected = self._shg_selected_file()
            selected_line = f"Selected: {selected}\n" if selected else "Select one CSV file from the list.\n"
            self.shg_summary.setPlainText(
                selected_line
                + f"{len(self.available_files)} CSV file(s) available; press Load to validate and process.\n"
                "The selected file will be validated as an SHG table during loading."
            )
        else:
            self.shg_summary.setPlainText(
                "No CSV files are available in the selected folder."
            )
        if hasattr(self, "shg_compare_summary"):
            if self.loaded and self.loaded.mode == "SHG Processing" and self.loaded.shg_compare:
                reference_name = self.loaded.shg_data.source_file if self.loaded.shg_data is not None else ""
                sample_name = self.loaded.shg_data_b.source_file if self.loaded.shg_data_b is not None else ""
                self.shg_compare_summary.setPlainText(
                    f"Reference A: {reference_name}\nSample B: {sample_name}\n"
                    "Both curves use the shared processing settings. Twist sign is Sample B minus Reference A."
                )
            else:
                reference_name, sample_name = self._shg_compare_files()
                self.shg_compare_summary.setPlainText(
                    f"Reference A: {reference_name or 'select a file'}\n"
                    f"Sample B: {sample_name or 'select a different file'}\n"
                    "Press Load to process and fit both files. Twist sign is Sample B minus Reference A."
                )
        self._shg_update_fit_summary()

    def _shg_update_fit_summary(self) -> None:
        if not hasattr(self, "shg_fit_summary"):
            return
        if not self.loaded or self.loaded.mode != "SHG Processing":
            self.shg_fit_summary.setPlainText("Load SHG data to calculate the angular fit.")
            return
        if self.loaded.shg_twist is not None:
            twist = self.loaded.shg_twist
            self.shg_fit_summary.setPlainText(
                f"A xc = {twist.reference_fit.x_center_deg:.6g} ± {twist.reference_fit.x_center_uncertainty_deg:.3g}°\n"
                f"B xc = {twist.sample_fit.x_center_deg:.6g} ± {twist.sample_fit.x_center_uncertainty_deg:.3g}°\n"
                f"Δxc = {twist.delta_x_center_deg:.6g} ± {twist.delta_x_center_uncertainty_deg:.3g}°\n"
                f"Twist = {twist.signed_twist_angle_deg:.6g} ± {twist.twist_uncertainty_deg:.3g}° "
                f"(|twist| = {twist.absolute_twist_angle_deg:.6g}°)"
            )
        elif self.loaded.shg_fit is not None:
            fit = self.loaded.shg_fit
            self.shg_fit_summary.setPlainText(
                f"xc = {fit.x_center_deg:.6g} ± {fit.x_center_uncertainty_deg:.3g}°\n"
                f"I₀ = {fit.i0:.6g}; A = {fit.amplitude:.6g}\n"
                f"R² = {fit.r_squared:.6g}; RMSE = {fit.rmse:.6g}; n = {fit.point_count}"
            )
        else:
            self.shg_fit_summary.setPlainText("Angular fit is disabled or unavailable.")

    def _on_shg_source_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._shg_update_summary()
        if (
            self.loaded
            and self.loaded.mode == "SHG Processing"
            and self.loaded.shg_compare == self._shg_compare_mode()
        ):
            self._start_load("SHG Processing")

    def _on_shg_workflow_changed(self) -> None:
        compare = self._shg_compare_mode()
        self.shg_fit_branch_spin.setEnabled(compare)
        self._shg_update_summary()
        self._status("SHG Compare / Twist Angle selected." if compare else "SHG Single File selected.")

    def _on_shg_param_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._shg_update_background_controls()
        if (
            self.loaded
            and self.loaded.mode == "SHG Processing"
            and self.loaded.shg_compare == self._shg_compare_mode()
        ):
            self._plot_mode("SHG Processing")

    def _on_shg_cosmic_param_changed(self) -> None:
        self._shg_update_cosmic_controls()
        self._on_shg_param_changed()

    def _on_shg_fit_param_changed(self) -> None:
        enabled = bool(self.shg_fit_enable_chk.isChecked())
        for widget in (
            self.shg_fit_min_spin,
            self.shg_fit_max_spin,
            self.shg_fit_weighted_chk,
            self.shg_fit_include_excluded_chk,
        ):
            widget.setEnabled(enabled)
        self.shg_fit_branch_spin.setEnabled(enabled and self._shg_compare_mode())
        self._on_shg_param_changed()

    def _on_shg_spectrum_view_changed(self) -> None:
        if (
            self.loaded
            and self.loaded.mode == "SHG Processing"
            and self.loaded.shg_compare == self._shg_compare_mode()
        ):
            self._plot_mode("SHG Processing")

    def _on_shg_background_method_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._shg_update_background_controls()
        if not self.loaded or self.loaded.mode != "SHG Processing":
            return
        if self.shg_background_method_combo.currentText() == "External + local residual":
            if self._shg_compare_mode():
                backgrounds = self._shg_compare_background_files()
                if not all(backgrounds):
                    self._status("Select external SHG background CSVs for both comparison files.")
                    return
            elif not self._shg_background_file():
                self._status("Select an SHG background CSV for external background mode.")
                return
            self._start_load("SHG Processing")
            return
        self._plot_mode("SHG Processing")

    def _power_candidate_files(self) -> list[str]:
        return list(self.available_files)

    def _power_axis_log(self) -> bool:
        return hasattr(self, "power_axis_scale_combo") and self.power_axis_scale_combo.currentText() == "Log"

    def _power_view(self) -> str:
        if hasattr(self, "power_view_vp_btn") and self.power_view_vp_btn.isChecked():
            return "VP"
        return "Intensity"

    def _power_set_view_mode(self, mode: str) -> None:
        vp_mode = mode == "VP"
        if hasattr(self, "power_view_intensity_btn"):
            self.power_view_intensity_btn.setChecked(not vp_mode)
        if hasattr(self, "power_view_vp_btn"):
            self.power_view_vp_btn.setChecked(vp_mode)
        self._power_update_view_mode()

    def _power_update_view_mode(self) -> None:
        vp_mode = self._power_view() == "VP"
        if hasattr(self, "power_group_combo"):
            self.power_group_combo.setEnabled(not vp_mode)
        if hasattr(self, "power_pair_mode_combo"):
            self.power_pair_mode_combo.setEnabled(vp_mode)
        self._update_plot_view_bar_visibility()

    def _power_pairing_mode(self) -> str:
        if not hasattr(self, "power_pair_mode_combo"):
            return "stage"
        return "power" if self.power_pair_mode_combo.currentText() == "Power Interpolation" else "stage"

    def _power_selected_group_key(self) -> str:
        if not hasattr(self, "power_group_combo"):
            return ""
        data = self.power_group_combo.currentData()
        return str(data) if data else str(self.power_group_combo.currentText()).strip()

    def _power_role_group_key(self, role: str) -> str:
        combo = self.power_kk_group_combo if role == "KK" else self.power_kkp_group_combo
        data = combo.currentData()
        return str(data or "")

    def _power_current_sources(self) -> Dict[str, data_io.PowerSeriesSource]:
        return data_io.get_power_series_sources(self.current_folder, self._power_candidate_files())

    def _power_refresh_groups(self) -> None:
        if not hasattr(self, "power_group_combo"):
            return
        old_key = self._power_selected_group_key()
        old_kk = self._power_role_group_key("KK") if hasattr(self, "power_kk_group_combo") else ""
        old_kkp = self._power_role_group_key("KKp") if hasattr(self, "power_kkp_group_combo") else ""
        sources = self._power_current_sources()
        old = self.power_group_combo.blockSignals(True)
        old_role_blocks: list[tuple[QComboBox, bool]] = []
        if hasattr(self, "power_kk_group_combo"):
            old_role_blocks = [
                (self.power_kk_group_combo, self.power_kk_group_combo.blockSignals(True)),
                (self.power_kkp_group_combo, self.power_kkp_group_combo.blockSignals(True)),
            ]
        try:
            self.power_group_combo.clear()
            if hasattr(self, "power_kk_group_combo"):
                self.power_kk_group_combo.clear()
                self.power_kkp_group_combo.clear()
                self.power_kk_group_combo.addItem("— Not assigned —", "")
                self.power_kkp_group_combo.addItem("— Not assigned —", "")
            ordered_sources = sorted(
                sources.items(),
                key=lambda item: (0 if item[1].source_format == "table" else 1, item[1].title.lower()),
            )
            for key, source in ordered_sources:
                if source.source_format == "table":
                    label = f"{source.file_name}  (Power_uW table)"
                else:
                    powers = [float(record.power_uW) for record in source.records]
                    if powers:
                        label = f"{source.title}  ({len(source.records)} files, {min(powers):.4g}-{max(powers):.4g} uW)"
                    else:
                        label = f"{source.title}  ({len(source.records)} files)"
                self.power_group_combo.addItem(label, key)
                if hasattr(self, "power_kk_group_combo"):
                    self.power_kk_group_combo.addItem(label, key)
                    self.power_kkp_group_combo.addItem(label, key)
            if old_key:
                idx = self.power_group_combo.findData(old_key)
                if idx >= 0:
                    self.power_group_combo.setCurrentIndex(idx)
            if hasattr(self, "power_kk_group_combo"):
                kk_idx = self.power_kk_group_combo.findData(old_kk)
                if kk_idx < 0:
                    kk_idx = 0
                kkp_idx = self.power_kkp_group_combo.findData(old_kkp)
                if kkp_idx < 0:
                    kkp_idx = 0
                if kk_idx >= 0:
                    self.power_kk_group_combo.setCurrentIndex(kk_idx)
                if kkp_idx >= 0:
                    self.power_kkp_group_combo.setCurrentIndex(kkp_idx)
        finally:
            self.power_group_combo.blockSignals(old)
            for combo, blocked in old_role_blocks:
                combo.blockSignals(blocked)
        self._power_update_group_summary()
        self._power_update_vp_availability()

    def _power_update_group_summary(self) -> None:
        if not hasattr(self, "power_group_summary"):
            return
        sources = self._power_current_sources()
        key = self._power_selected_group_key()
        source = sources.get(key)
        if source is None and sources:
            source = next(iter(sources.values()))
            key = source.key
        if source is None:
            self.power_group_summary.setPlainText(
                "No power sweeps found. Expected a Power_uW table or filenames containing values such as 37.96uW."
            )
            return
        if source.source_format == "table":
            try:
                result = data_io.load_power_series_cube(
                    self.current_folder,
                    self._power_candidate_files(),
                    group_key=key,
                    y_axis="auto",
                )
            except Exception as exc:
                self.power_group_summary.setPlainText(f"{source.file_name}\nInvalid power table: {exc}")
                return
            records = result.records
            cube = result.cube
            stages = [record.stage for record in records if getattr(record, "stage", None) is not None]
            lines = [
                f"{source.file_name}",
                f"{len(records)} power rows: {float(np.nanmin(cube.gate)):.6g}-{float(np.nanmax(cube.gate)):.6g} uW",
                f"{len(cube.energy)} spectral points: {float(np.nanmin(cube.energy)):.6g}-{float(np.nanmax(cube.energy)):.6g} eV",
                f"stage_pos: {'available' if stages else 'not available'}",
            ]
        else:
            records = source.records
            lines = [f"{source.title}", f"{len(records)} legacy files sorted by power:"]
        for record in records[:8]:
            stage = getattr(record, "stage", None)
            stage_text = "" if stage is None else f", Stage {stage:g}"
            row = getattr(record, "row_index", None)
            row_text = "" if row is None else f", row {row}"
            lines.append(f"{getattr(record, 'power_uW', 0.0):.6g} uW{stage_text}{row_text}")
        if len(records) > 8:
            lines.append(f"+{len(records) - 8} more")
        if hasattr(self, "power_kk_group_combo") and self._power_has_distinct_role_groups():
            lines.append(f"KK -> {power_group_title(self._power_role_group_key('KK')) or 'none'}")
            lines.append(f"KKp -> {power_group_title(self._power_role_group_key('KKp')) or 'none'}")
        self.power_group_summary.setPlainText("\n".join(lines))

    def _power_background_auto_enabled(self) -> bool:
        return bool(hasattr(self, "power_background_auto_chk") and self.power_background_auto_chk.isChecked())

    def _power_set_background_spin_silent(self, value: float) -> None:
        if not hasattr(self, "power_background_spin"):
            return
        old = self.power_background_spin.blockSignals(True)
        try:
            self.power_background_spin.setValue(float(value))
        finally:
            self.power_background_spin.blockSignals(old)

    def _power_background_value(self, cubes: Sequence[DataCube] | Dict[str, DataCube] | None = None) -> float:
        if not hasattr(self, "power_background_spin"):
            return 0.0
        if self._power_background_auto_enabled() and cubes:
            value = estimate_constant_background(cubes, percentile=1.0)
            self._power_set_background_spin_silent(value)
            return value
        return float(self.power_background_spin.value())

    def _power_load_group_result(self, group_key: str) -> data_io.PowerSeriesResult:
        return data_io.load_power_series_cube(
            self.current_folder,
            self._power_candidate_files(),
            group_key=group_key,
            y_axis="auto",
        )

    def _power_corrected_cube(self, cube: DataCube, background: float | None = None) -> DataCube:
        if background is None:
            background = self._power_background_value([cube])
        return background_correct_cube(cube, background, title=cube.title)

    def _power_role_payload(self) -> tuple[data_io.PowerSeriesResult, data_io.PowerSeriesResult, str, str]:
        kk_key = self._power_role_group_key("KK")
        kkp_key = self._power_role_group_key("KKp")
        if not kk_key or not kkp_key:
            raise ValueError("Assign KK and KKp power-sweep sources.")
        if kk_key == kkp_key:
            raise ValueError("KK and KKp must use different power-sweep sources.")
        kk_result = self._power_load_group_result(kk_key)
        kkp_result = self._power_load_group_result(kkp_key)
        return kk_result, kkp_result, kk_key, kkp_key

    def _power_has_distinct_role_groups(self) -> bool:
        kk_key = self._power_role_group_key("KK")
        kkp_key = self._power_role_group_key("KKp")
        return bool(kk_key and kkp_key and kk_key != kkp_key)

    def _power_update_vp_availability(self) -> None:
        has_distinct = self._power_has_distinct_role_groups()
        if hasattr(self, "power_view_vp_btn"):
            self.power_view_vp_btn.setEnabled(has_distinct)
        if not has_distinct and self._power_view() == "VP":
            self._power_set_view_mode("Intensity")
        if hasattr(self, "power_pair_mode_combo"):
            stage_available = False
            if has_distinct:
                try:
                    kk_result, kkp_result, _kk_key, _kkp_key = self._power_role_payload()
                    kk_stages = {float(record.stage) for record in kk_result.records if record.stage is not None}
                    kkp_stages = {float(record.stage) for record in kkp_result.records if record.stage is not None}
                    stage_available = bool(kk_stages & kkp_stages)
                except Exception:
                    stage_available = False
            item = self.power_pair_mode_combo.model().item(0)
            if item is not None:
                item.setEnabled(stage_available)
            if not stage_available and self.power_pair_mode_combo.currentIndex() == 0:
                blocked = self.power_pair_mode_combo.blockSignals(True)
                try:
                    self.power_pair_mode_combo.setCurrentIndex(1)
                finally:
                    self.power_pair_mode_combo.blockSignals(blocked)
            self.power_pair_mode_combo.setToolTip(
                "Pair matching stage_pos rows, or interpolate by Power_uW."
                if stage_available
                else "Stage pairing needs shared stage_pos values; Power Interpolation is selected."
            )

    def _power_vp_payload(self) -> tuple[DataCube, DataCube, DataCube, tuple[Any, ...], tuple[Any, ...], str, str, float, str, tuple[Any, ...]]:
        kk_result, kkp_result, kk_key, kkp_key = self._power_role_payload()
        background = self._power_background_value([kk_result.cube, kkp_result.cube])
        vp_title = _vp_short_title(power_group_title(kk_key), power_group_title(kkp_key))
        pairing_mode = self._power_pairing_mode()
        if pairing_mode == "stage":
            kk_cube, kkp_cube, vp_cube, stage_pairs = power_stage_paired_vp_cubes(
                kk_result.cube,
                kk_result.records,
                kkp_result.cube,
                kkp_result.records,
                background=background,
                title=vp_title,
            )
        else:
            kk_cube, kkp_cube, vp_cube = power_valley_polarization_cube(
                kk_result.cube,
                kkp_result.cube,
                background=background,
                title=vp_title,
            )
            stage_pairs = ()
        return (
            kk_cube,
            kkp_cube,
            vp_cube,
            kk_result.records,
            kkp_result.records,
            kk_key,
            kkp_key,
            background,
            pairing_mode,
            tuple(stage_pairs),
        )

    def _cmp_set_channel_combo_items(self) -> None:
        files = [""] + list(self.available_files)
        for combo in self.cmp_channel_combos.values():
            current = combo.currentText()
            old = combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(files)
                if current in files:
                    combo.setCurrentText(current)
            finally:
                combo.blockSignals(old)

    def _cmp_view_mode(self) -> str:
        if hasattr(self, "cmp_view_vp_btn") and self.cmp_view_vp_btn.isChecked():
            return "Valley Polarization"
        return "Intensity Compare"

    def _cmp_set_view_mode(self, mode: str) -> None:
        vp_mode = mode == "Valley Polarization"
        if hasattr(self, "cmp_view_intensity_btn"):
            self.cmp_view_intensity_btn.setChecked(not vp_mode)
        if hasattr(self, "cmp_view_vp_btn"):
            self.cmp_view_vp_btn.setChecked(vp_mode)

    def _cmp_is_vp_view(self) -> bool:
        return self._cmp_view_mode() == "Valley Polarization"

    def _cmp_background_auto_enabled(self) -> bool:
        return bool(
            hasattr(self, "cmp_vp_auto_background_chk")
            and self.cmp_vp_auto_background_chk.isChecked()
        )

    def _cmp_update_background_mode(self) -> None:
        if hasattr(self, "cmp_vp_background_spin"):
            self.cmp_vp_background_spin.setEnabled(not self._cmp_background_auto_enabled())

    @staticmethod
    def _cmp_background_source_cubes(cubes: Dict[str, DataCube]) -> Dict[str, DataCube]:
        kk_pair = {key: cubes[key] for key in ("KK", "KKp") if key in cubes}
        return kk_pair if kk_pair else dict(cubes)

    def _cmp_set_background_spin_silent(self, value: float) -> None:
        if not hasattr(self, "cmp_vp_background_spin"):
            return
        old = self.cmp_vp_background_spin.blockSignals(True)
        try:
            self.cmp_vp_background_spin.setValue(float(value))
        finally:
            self.cmp_vp_background_spin.blockSignals(old)

    def _cmp_background_value(
        self,
        cubes: Dict[str, DataCube] | None = None,
        *,
        update_spin: bool = True,
    ) -> float:
        if not hasattr(self, "cmp_vp_background_spin"):
            return 0.0
        if self._cmp_background_auto_enabled() and cubes:
            value = estimate_constant_background(
                self._cmp_background_source_cubes(cubes),
                percentile=1.0,
            )
            if update_spin:
                self._cmp_set_background_spin_silent(value)
            return value
        return float(self.cmp_vp_background_spin.value())

    def _cmp_scale_tag(self) -> str:
        return "log" if bool(self.cmp_log_chk.isChecked()) else "linear"

    def _cmp_source_mapping(self) -> dict[str, str]:
        if self.loaded and self.loaded.mode == "Compare" and self.loaded.compare_sources:
            return dict(self.loaded.compare_sources)
        return self._cmp_current_mapping()

    def _cmp_corrected_cubes(
        self,
        cubes: Dict[str, DataCube],
        source_files: dict[str, str] | None = None,
        background: float | None = None,
    ) -> Dict[str, DataCube]:
        if background is None:
            background = self._cmp_background_value(cubes)
        source_files = source_files or self._cmp_source_mapping()
        corrected: dict[str, DataCube] = {}
        for key, cube in cubes.items():
            title = compare_source_title(source_files.get(key, cube.title))
            corrected[key] = background_correct_cube(cube, background, title=title)
        return corrected

    def _cmp_vp_cube(
        self,
        cubes: Dict[str, DataCube],
        source_files: dict[str, str] | None = None,
        background: float | None = None,
    ) -> DataCube:
        if "KK" not in cubes or "KKp" not in cubes:
            raise ValueError("VP needs assigned KK and KKp channels.")
        if background is None:
            background = self._cmp_background_value(cubes)
        source_files = source_files or self._cmp_source_mapping()
        return valley_polarization_cube(
            cubes["KK"],
            cubes["KKp"],
            background=background,
            title=vp_compare_title(source_files, background, self._cmp_scale_tag()),
        )

    def _cmp_update_title_previews(self) -> None:
        if not hasattr(self, "cmp_vp_filename_preview"):
            return
        mapping = self._cmp_current_mapping()
        loaded_cubes = self.loaded.compare_cubes if self.loaded and self.loaded.mode == "Compare" else None
        background = self._cmp_background_value(loaded_cubes, update_spin=loaded_cubes is not None)
        kk_title = compare_source_title(mapping["KK"]) if "KK" in mapping else "Assign KK"
        kkp_title = compare_source_title(mapping["KKp"]) if "KKp" in mapping else "Assign KKp"
        self.cmp_kk_title_preview.setText(kk_title)
        self.cmp_kkp_title_preview.setText(kkp_title)
        if "KK" in mapping and "KKp" in mapping:
            base = vp_compare_export_base(mapping, background, self._cmp_scale_tag())
            title = vp_compare_title(mapping, background, self._cmp_scale_tag())
            self.cmp_vp_filename_preview.setText(f"{base}.png / .dat")
            self.cmp_vp_title_preview.setText(title)
        else:
            self.cmp_vp_filename_preview.setText("Assign KK and KKp")
            self.cmp_vp_title_preview.setText("Assign KK and KKp")

    def _cmp_current_mapping(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        used: set[str] = set()
        for key in COMPARE_PANEL_ORDER:
            name = self.cmp_channel_combos[key].currentText().strip()
            if not name or name in used:
                continue
            mapping[key] = name
            used.add(name)
        return mapping

    def _cmp_visible_channels(self, mapping: dict[str, str] | None = None) -> list[str]:
        mapping = mapping or self._cmp_current_mapping()
        preset = self.cmp_display_preset_combo.currentText()
        if preset == "KK + KKp":
            order = ["KK", "KKp"]
        elif preset == "KpK + KpKp":
            order = ["KpK", "KpKp"]
        elif preset == "All four":
            order = list(COMPARE_PANEL_ORDER)
        else:
            order = [key for key in COMPARE_PANEL_ORDER if self.cmp_show_checks[key].isChecked()]
        return [key for key in order if key in mapping]

    def _cmp_apply_display_preset(self) -> None:
        preset = self.cmp_display_preset_combo.currentText()
        enabled = preset == "Custom" and not self._cmp_is_vp_view()
        desired = {
            "KK + KKp": {"KK", "KKp"},
            "KpK + KpKp": {"KpK", "KpKp"},
            "All four": set(COMPARE_PANEL_ORDER),
            "Custom": None,
        }[preset]
        for key, chk in self.cmp_show_checks.items():
            chk.setEnabled(enabled)
            if desired is not None:
                old = chk.blockSignals(True)
                try:
                    chk.setChecked(key in desired)
                finally:
                    chk.blockSignals(old)

    def _cmp_update_assignment_summary(self) -> None:
        mapping = self._cmp_current_mapping()
        visible = self._cmp_visible_channels(mapping)
        lines: list[str] = []
        for key in COMPARE_PANEL_ORDER:
            source = mapping.get(key, "missing")
            lines.append(f"{key} -> {source}")
        if self._cmp_is_vp_view():
            if "KK" in mapping and "KKp" in mapping:
                lines.append("Visible -> VP from KK, KKp")
            else:
                lines.append("Visible -> VP needs KK and KKp")
        elif visible:
            lines.append("Visible -> " + ", ".join(visible))
        else:
            lines.append("Visible -> none")
        self.cmp_assignment_summary.setPlainText("\n".join(lines))
        self._cmp_update_title_previews()

    def _cmp_update_view_mode(self) -> None:
        vp_mode = self._cmp_is_vp_view()
        self.cmp_display_preset_combo.setEnabled(not vp_mode)
        for chk in self.cmp_show_checks.values():
            chk.setEnabled((not vp_mode) and self.cmp_display_preset_combo.currentText() == "Custom")
        self._update_plot_view_bar_visibility()

    def _update_plot_view_bar_visibility(self) -> None:
        if hasattr(self, "cmp_plot_view_bar"):
            self.cmp_plot_view_bar.setVisible(self._active_mode() == "Compare")
        if hasattr(self, "power_plot_view_bar"):
            self.power_plot_view_bar.setVisible(self._active_mode() == "Power Dependent")

    def _cmp_selection_from_ui(self) -> data_io.CompareSelection:
        mapping = self._cmp_current_mapping()
        if self._cmp_is_vp_view():
            missing = [key for key in ("KK", "KKp") if key not in mapping]
            if missing:
                raise ValueError("VP needs assigned KK and KKp channels.")
            return data_io.CompareSelection.from_mapping(
                mapping,
                visible_order=("KK", "KKp"),
            )
        visible = self._cmp_visible_channels(mapping)
        if len(visible) < 1:
            raise ValueError("Assign at least one compare channel.")
        if len(visible) < 2:
            raise ValueError("Select at least two visible compare channels.")
        return data_io.CompareSelection.from_mapping(
            mapping,
            visible_order=visible,
        )

    def _cmp_infer_angle_references(self) -> None:
        inference = infer_compare_angle_references(
            self._cmp_assign_candidate_files(),
            in_k_anchor=float(self.cmp_in_k_angle_spin.value()),
            out_k_anchor=float(self.cmp_out_k_angle_spin.value()),
            cluster_tolerance=float(self.cmp_angle_tolerance_spin.value()),
        )
        for spin, value in (
            (self.cmp_in_k_angle_spin, inference.in_k),
            (self.cmp_in_kp_angle_spin, inference.in_kp),
            (self.cmp_out_k_angle_spin, inference.out_k),
            (self.cmp_out_kp_angle_spin, inference.out_kp),
        ):
            if value is None:
                continue
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(float(value))
            finally:
                spin.blockSignals(blocked)

        def _clusters_text(values: tuple[float, ...]) -> str:
            return ", ".join(f"{value:.3g}" for value in values) if values else "none"

        self._append_log(
            "Angle inference: "
            f"Rot1 clusters=[{_clusters_text(inference.rot1_clusters)}], "
            f"Rot2 clusters=[{_clusters_text(inference.rot2_clusters)}]"
        )
        suggestions: list[str] = []
        if inference.in_k is not None and inference.in_kp is not None:
            suggestions.append(f"In K={inference.in_k:.3g}, In Kp={inference.in_kp:.3g}")
        if inference.out_k is not None and inference.out_kp is not None:
            suggestions.append(f"Out K={inference.out_k:.3g}, Out Kp={inference.out_kp:.3g}")
        if suggestions:
            self._append_log("  suggested: " + "; ".join(suggestions))
        else:
            self._append_log(
                "  no references changed: inference requires exactly two clusters for an arm"
            )

    def _cmp_auto_assign_channels(self) -> None:
        candidates = self._cmp_assign_candidate_files()
        in_k = float(self.cmp_in_k_angle_spin.value())
        in_kp = float(self.cmp_in_kp_angle_spin.value())
        out_k = float(self.cmp_out_k_angle_spin.value())
        out_kp = float(self.cmp_out_kp_angle_spin.value())
        tolerance = float(self.cmp_angle_tolerance_spin.value())
        found, duplicates, gate_group, gate_groups = coherent_compare_auto_assignment(
            candidates,
            in_k_angle=in_k,
            in_kp_angle=in_kp,
            out_k_angle=out_k,
            out_kp_angle=out_kp,
            tolerance=tolerance,
        )
        for key, combo in self.cmp_channel_combos.items():
            old = combo.blockSignals(True)
            try:
                combo.setCurrentText(found.get(key, ""))
            finally:
                combo.blockSignals(old)
        self._cmp_update_assignment_summary()
        # --- diagnostic logging ---
        classified_counts: dict[str, int] = {}
        group_keys: dict[str, set[str]] = {}
        rejected: list[str] = []
        for fname in candidates:
            ch = classify_compare_channel(
                fname,
                in_k_angle=in_k,
                in_kp_angle=in_kp,
                out_k_angle=out_k,
                out_kp_angle=out_kp,
                tolerance=tolerance,
            )
            if ch:
                classified_counts[ch] = classified_counts.get(ch, 0) + 1
                gk = parse_compare_gate_condition(fname) or "__ungrouped__"
                group_keys.setdefault(gk, set()).add(ch)
                continue
            angles = parse_compare_rotation_angles(fname)
            reasons: list[str] = []
            for arm, angle, k_angle, kp_angle in (
                ("Rot1", angles.rot1, in_k, in_kp),
                ("Rot2", angles.rot2, out_k, out_kp),
            ):
                if angle is None:
                    continue
                match = classify_angle_state(
                    angle,
                    k_angle=k_angle,
                    kp_angle=kp_angle,
                    tolerance=tolerance,
                )
                if match.state is None:
                    reasons.append(
                        f"{arm}={angle:g}: {match.reason} "
                        f"(dK={match.distance_k:.3g}, dKp={match.distance_kp:.3g})"
                    )
            if reasons:
                rejected.append(f"{Path(fname).name}: " + "; ".join(reasons))
        assigned = [k for k in ("KK", "KKp", "KpK", "KpKp") if k in found]
        missing = [k for k in ("KK", "KKp", "KpK", "KpKp") if k not in found]
        self._append_log(
            f"Auto-assign (InK={in_k:.1f}, InKp={in_kp:.1f}, "
            f"OutK={out_k:.1f}, OutKp={out_kp:.1f}, tol={tolerance:.1f} deg): "
            + f"classified {classified_counts} across {len(group_keys)} group(s)"
        )
        for detail in rejected[:8]:
            self._append_log(f"  unassigned angle: {detail}")
        if len(rejected) > 8:
            self._append_log(f"  +{len(rejected) - 8} more unassigned angle match(es)")
        for gk, keys in sorted(group_keys.items()):
            marker = " <-- selected" if gk == (gate_group or "__ungrouped__") else ""
            self._append_log(f"  group [{gk}]: keys={sorted(keys)}{marker}")
        if assigned:
            self._append_log(f"  assigned: {', '.join(assigned)}")
            for fname in dict.fromkeys(found[key] for key in assigned):
                angles = parse_compare_rotation_angles(fname)
                if (angles.rot1 is None) != (angles.rot2 is None):
                    detected = (
                        f"Rot1={angles.rot1:g} deg"
                        if angles.rot1 is not None
                        else f"Rot2={angles.rot2:g} deg"
                    )
                    fixed = "output" if angles.rot1 is not None else "input"
                    self._append_log(
                        f"  partial rotation: {detected}; missing fixed {fixed} arm treated as K"
                    )
        if missing:
            reason_parts: list[str] = []
            for mk in missing:
                if mk not in classified_counts:
                    reason_parts.append(f"{mk}=no file classified as {mk}")
                else:
                    in_selected = mk in group_keys.get(gate_group or "__ungrouped__", set())
                    if not in_selected:
                        reason_parts.append(f"{mk}=only in other gate group(s)")
                    else:
                        reason_parts.append(f"{mk}=duplicate (already assigned)")
            self._append_log(f"  MISSING: {'; '.join(reason_parts)}")
        # --- end diagnostic ---
        if gate_group and len(set(gate_groups)) > 1:
            self._append_log(
                "Compare auto-detect found multiple gate groups: "
                + ", ".join(sorted(set(gate_groups)))
                + f". Using {gate_group}."
            )
        if duplicates:
            dup_text = "; ".join(f"{k}: {', '.join(v)}" for k, v in duplicates.items())
            self._append_log(f"Compare auto-detect found duplicate matches -> {dup_text}")
        self._on_cmp_plot_param_changed()

    def _build_tools_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        log_box = QGroupBox("Log Panel")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)
        hint = QLabel("The log panel records all load, plot, and export events.")
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #6e6e73; font-size: 10px; }")
        log_layout.addWidget(hint)
        log_btn_row = QHBoxLayout()
        log_btn_row.setSpacing(8)
        self.show_log_btn = QPushButton("Show / Hide Log Panel")
        self.show_log_btn.setToolTip("Toggle the bottom log dock panel")
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.setToolTip("Clear all messages from the log")
        log_btn_row.addWidget(self.show_log_btn)
        log_btn_row.addWidget(self.clear_log_btn)
        log_layout.addLayout(log_btn_row)
        layout.addWidget(log_box)

        file_box = QGroupBox("File Management")
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(8, 8, 8, 8)
        file_layout.setSpacing(6)
        file_hint = QLabel(
            "After export, source files remain in place. Use 'Move Exported Sources' "
            "only when you intentionally want to archive them."
        )
        file_hint.setWordWrap(True)
        file_hint.setStyleSheet("QLabel { color: #6e6e73; font-size: 10px; }")
        file_layout.addWidget(file_hint)
        layout.addWidget(file_box)

        layout.addStretch(1)
        return tab

    def apply_ui_metrics(self) -> None:
        h = UI_METRICS["input_h"]
        for w in self.findChildren(QLineEdit):
            if w.minimumHeight() < h:
                w.setMinimumHeight(h)
        for w in self.findChildren(QComboBox):
            if w.minimumHeight() < h:
                w.setMinimumHeight(h)
        for w in self.findChildren(QDoubleSpinBox):
            if w.minimumHeight() < h:
                w.setMinimumHeight(h)
        for w in self.findChildren(QSpinBox):
            if w.minimumHeight() < h:
                w.setMinimumHeight(h)
        for btn in self.findChildren(QToolButton):
            if btn.minimumHeight() < UI_METRICS["tool_h"]:
                btn.setMinimumHeight(UI_METRICS["tool_h"])
        for combo in (self.pl_cmap, self.cmp_cmap):
            combo.setMinimumWidth(UI_METRICS["short_combo_w"])

    def _build_plot_panel(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, box)
        layout.addWidget(self.toolbar)
        self.cmp_plot_view_bar = QFrame()
        self.cmp_plot_view_bar.setFrameShape(QFrame.NoFrame)
        self.cmp_plot_view_bar.setVisible(False)
        self.cmp_plot_view_bar.setStyleSheet(
            "QFrame { background: #f7f7f9; border-top: 1px solid #ececf0; border-bottom: 1px solid #ececf0; }"
            "QToolButton { border: 1px solid #d0d0d5; border-radius: 4px; padding: 3px 10px; "
            "background: #ffffff; font-size: 11px; }"
            "QToolButton:checked { background: #0078d4; color: white; border-color: #0078d4; }"
        )
        view_layout = QHBoxLayout(self.cmp_plot_view_bar)
        view_layout.setContentsMargins(8, 4, 8, 4)
        view_layout.setSpacing(0)
        self.cmp_view_intensity_btn = QToolButton()
        self.cmp_view_intensity_btn.setText("Intensity")
        self.cmp_view_intensity_btn.setToolTip("Show corrected KK/KKp intensity maps and spectra")
        self.cmp_view_intensity_btn.setCheckable(True)
        self.cmp_view_intensity_btn.setChecked(True)
        self.cmp_view_vp_btn = QToolButton()
        self.cmp_view_vp_btn.setText("VP")
        self.cmp_view_vp_btn.setToolTip("Show valley polarization map and linecut")
        self.cmp_view_vp_btn.setCheckable(True)
        view_layout.addWidget(self.cmp_view_intensity_btn)
        view_layout.addWidget(self.cmp_view_vp_btn)
        view_layout.addStretch(1)
        layout.addWidget(self.cmp_plot_view_bar)
        self.power_plot_view_bar = QFrame()
        self.power_plot_view_bar.setFrameShape(QFrame.NoFrame)
        self.power_plot_view_bar.setVisible(False)
        self.power_plot_view_bar.setStyleSheet(self.cmp_plot_view_bar.styleSheet())
        power_view_layout = QHBoxLayout(self.power_plot_view_bar)
        power_view_layout.setContentsMargins(8, 4, 8, 4)
        power_view_layout.setSpacing(0)
        self.power_view_intensity_btn = QToolButton()
        self.power_view_intensity_btn.setText("Intensity")
        self.power_view_intensity_btn.setToolTip("Show corrected power-dependent intensity map and spectrum")
        self.power_view_intensity_btn.setCheckable(True)
        self.power_view_intensity_btn.setChecked(True)
        self.power_view_vp_btn = QToolButton()
        self.power_view_vp_btn.setText("VP")
        self.power_view_vp_btn.setToolTip("Show power-dependent valley polarization from assigned KK and KKp groups")
        self.power_view_vp_btn.setCheckable(True)
        power_view_layout.addWidget(self.power_view_intensity_btn)
        power_view_layout.addWidget(self.power_view_vp_btn)
        power_view_layout.addStretch(1)
        layout.addWidget(self.power_plot_view_bar)
        layout.addWidget(self.canvas, 1)
        return box

    def _build_log_dock(self) -> None:
        self.log_dock = QDockWidget("Run Log", self)
        self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        container = QWidget()
        layout = QVBoxLayout(container)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        self.log_dock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()

    def _build_results_dock(self) -> None:
        """Host read-only analysis output outside the contextual controls."""
        self.results_dock = QDockWidget("Analysis Results", self)
        self.results_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.results_stack = QStackedWidget()
        self._results_page_indices: dict[str, int] = {}

        for mode, title, widget in (
            ("PL", "PL peak and fit results", self.pl_analysis_text),
            ("DRR", "DRR peak and fit results", self.drr_analysis_text),
            ("MCD", "MCD pair diagnostics", self.mcd_diagnostics_text),
        ):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(8, 6, 8, 6)
            label = QLabel(title)
            label.setStyleSheet("QLabel { font-weight: 600; color: #3a3a3c; }")
            page_layout.addWidget(label)
            widget.setMaximumHeight(16777215)
            page_layout.addWidget(widget, 1)
            self._results_page_indices[mode] = self.results_stack.addWidget(page)

        empty_page = QLabel("This workflow has no separate text results. Use the plot and export controls.")
        empty_page.setAlignment(Qt.AlignCenter)
        empty_page.setWordWrap(True)
        self._results_empty_index = self.results_stack.addWidget(empty_page)
        self.results_dock.setWidget(self.results_stack)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.results_dock)
        self.resizeDocks([self.results_dock], [190], Qt.Vertical)
        self.results_dock.hide()
        self.mcd_diagnostics_expander.hide()
        self._update_results_dock_page()

    def _update_results_dock_page(self) -> None:
        if not hasattr(self, "results_stack"):
            return
        mode = self._active_mode()
        index = self._results_page_indices.get(mode or "", self._results_empty_index)
        self.results_stack.setCurrentIndex(index)

    def _build_menu_and_toolbar(self) -> None:
        menu = self.menuBar().addMenu("View")
        self.show_log_action = self.log_dock.toggleViewAction()
        self.show_log_action.setText("Show Log")
        menu.addAction(self.show_log_action)
        self.show_results_action = self.results_dock.toggleViewAction()
        self.show_results_action.setText("Show Analysis Results")
        menu.addAction(self.show_results_action)
        self.show_sidebar_action = QAction("Show Controls Sidebar", self)
        self.show_sidebar_action.setCheckable(True)
        self.show_sidebar_action.setChecked(True)
        self.show_sidebar_action.setShortcut("Ctrl+B")
        self.show_sidebar_action.toggled.connect(self._set_sidebar_visible)
        self.sidebar_toggle_btn.toggled.connect(self.show_sidebar_action.setChecked)
        menu.addAction(self.show_sidebar_action)

        help_menu = self.menuBar().addMenu("Help")
        self.check_updates_action = QAction("Check for Updates...", self)
        self.auto_update_check_action = QAction("Check for Updates Automatically", self)
        self.auto_update_check_action.setCheckable(True)
        self.auto_update_check_action.setChecked(self._auto_update_check_enabled())
        self.about_action = QAction("About", self)
        help_menu.addAction(self.check_updates_action)
        help_menu.addAction(self.auto_update_check_action)
        help_menu.addAction(self.about_action)
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        self.load_action = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), "Load", self)
        self.load_action.setToolTip("Load data for the active tab")
        self.plot_action = QAction(self.style().standardIcon(QStyle.SP_BrowserReload), "Plot / Update", self)
        self.plot_action.setToolTip("Plot/update current state")
        self.save_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), "Save PNG + DAT", self)
        self.save_action.setToolTip("Export for the active tab")
        self.move_now_btn = QPushButton("Move Exported Sources")
        self.move_now_btn.setEnabled(False)
        self.move_now_btn.setToolTip("Save first to enable moving exported source files.")
        self.clean_verified_sources_chk = QCheckBox("Clean verified source copies after successful export")
        self.clean_verified_sources_chk.setChecked(False)
        self.clean_verified_sources_chk.setToolTip(
            "Only verified root-level files matching Initial Data by SHA-256 can be deleted."
        )
        toolbar.addAction(self.load_action)
        toolbar.addAction(self.plot_action)
        toolbar.addAction(self.save_action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addWidget(self.clean_verified_sources_chk)
        toolbar.addWidget(self.move_now_btn)

    def _wire_actions(self) -> None:
        self.browse_btn.clicked.connect(self._browse_folder)
        self.open_file_btn.clicked.connect(self._open_file)
        self.refresh_btn.clicked.connect(lambda: self._refresh_file_lists())
        self.recent_folder_combo.currentIndexChanged.connect(self._on_recent_folder_selected)
        self.load_action.triggered.connect(self._toolbar_load)
        self.plot_action.triggered.connect(self._toolbar_plot)
        self.save_action.triggered.connect(self._toolbar_save)
        self.move_now_btn.clicked.connect(self._manual_move_sources)
        self.check_updates_action.triggered.connect(self._manual_check_updates)
        self.about_action.triggered.connect(self._show_about)
        self.auto_update_check_action.toggled.connect(self._on_auto_update_check_toggled)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        for prefix in ("pl", "drr", "cmp"):
            combo: QComboBox = getattr(self, f"{prefix}_yaxis_combo")
            combo.currentTextChanged.connect(lambda _text, p=prefix: self._update_y_axis_controls(p))
            for key in ("a", "b", "c"):
                spin: QDoubleSpinBox = getattr(self, f"{prefix}_yaxis_{key}_spin")
                spin.valueChanged.connect(lambda _value, p=prefix: self._update_y_axis_controls(p))
        for prefix in ("pl", "drr", "cmp", "power"):
            toggle: QCheckBox = getattr(self, f"{prefix}_split_scale_chk")
            toggle.toggled.connect(
                lambda checked, p=prefix: self._on_split_scale_toggled(p, checked)
            )
            split_spins: Dict[str, QDoubleSpinBox] = getattr(self, f"{prefix}_split_spins")
            for spin in split_spins.values():
                spin.editingFinished.connect(
                    lambda p=prefix: self._on_split_scale_param_changed(p)
                )
            split_fix_checks: Dict[str, QCheckBox] = getattr(
                self, f"{prefix}_split_fix_checks"
            )
            for check in split_fix_checks.values():
                check.toggled.connect(self._update_action_states)
            boundary_chk: QCheckBox = getattr(self, f"{prefix}_split_boundary_chk")
            boundary_chk.toggled.connect(
                lambda _checked, p=prefix: self._on_split_scale_param_changed(p)
            )
            getattr(self, f"{prefix}_split_auto_left_btn").clicked.connect(
                lambda _checked=False, p=prefix: self._auto_split_vrange(p, "left")
            )
            getattr(self, f"{prefix}_split_auto_right_btn").clicked.connect(
                lambda _checked=False, p=prefix: self._auto_split_vrange(p, "right")
            )
        self.pl_auto_v_btn.clicked.connect(self._auto_pl_vrange)
        self.pl_auto_x_btn.clicked.connect(self._auto_pl_xrange)
        self.pl_auto_y_btn.clicked.connect(self._auto_pl_yrange)
        self.pl_yaxis_combo.currentTextChanged.connect(self._on_pl_plot_param_changed)
        self.pl_yaxis_combo.currentTextChanged.connect(self._on_pl_dat_y_axis_changed)
        self.pl_dat_yaxis_label_edit.textChanged.connect(lambda _text: self._on_pl_plot_param_changed())
        self.pl_dat_yaxis_unit_edit.textChanged.connect(lambda _text: self._on_pl_plot_param_changed())
        self.pl_yaxis_a_spin.valueChanged.connect(self._on_pl_plot_param_changed)
        self.pl_yaxis_b_spin.valueChanged.connect(self._on_pl_plot_param_changed)
        self.pl_yaxis_c_spin.valueChanged.connect(self._on_pl_plot_param_changed)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax"):
            self.pl_spins[key].valueChanged.connect(self._on_pl_plot_param_changed)
        self.pl_spins["gate"].valueChanged.connect(self._on_pl_gate_changed)
        self.pl_cmap.currentTextChanged.connect(self._on_pl_plot_param_changed)
        self.pl_log_chk.toggled.connect(self._on_pl_plot_param_changed)
        self.pl_clip_chk.toggled.connect(self._on_pl_plot_param_changed)
        self.cmp_in_k_angle_spin.valueChanged.connect(self._on_cmp_auto_assign_requested)
        self.cmp_in_kp_angle_spin.valueChanged.connect(self._on_cmp_auto_assign_requested)
        self.cmp_out_k_angle_spin.valueChanged.connect(self._on_cmp_auto_assign_requested)
        self.cmp_out_kp_angle_spin.valueChanged.connect(self._on_cmp_auto_assign_requested)
        self.cmp_angle_tolerance_spin.valueChanged.connect(self._on_cmp_auto_assign_requested)
        self.cmp_infer_angles_btn.clicked.connect(self._on_cmp_infer_angles_requested)
        self.cmp_auto_assign_btn.clicked.connect(self._on_cmp_auto_assign_requested)
        self.cmp_view_intensity_btn.clicked.connect(lambda: self._on_cmp_plot_view_button_clicked("Intensity Compare"))
        self.cmp_view_vp_btn.clicked.connect(lambda: self._on_cmp_plot_view_button_clicked("Valley Polarization"))
        self.cmp_vp_background_spin.valueChanged.connect(self._on_cmp_plot_param_changed)
        self.cmp_vp_auto_background_chk.toggled.connect(self._on_cmp_background_mode_changed)
        self.cmp_display_preset_combo.currentTextChanged.connect(self._on_cmp_display_preset_changed)
        for combo in self.cmp_channel_combos.values():
            combo.currentTextChanged.connect(self._on_cmp_plot_param_changed)
        for chk in self.cmp_show_checks.values():
            chk.toggled.connect(self._on_cmp_plot_param_changed)
        self.cmp_yaxis_combo.currentTextChanged.connect(self._on_cmp_plot_param_changed)
        self.cmp_yaxis_a_spin.valueChanged.connect(self._on_cmp_plot_param_changed)
        self.cmp_yaxis_b_spin.valueChanged.connect(self._on_cmp_plot_param_changed)
        self.cmp_yaxis_c_spin.valueChanged.connect(self._on_cmp_plot_param_changed)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax", "gate"):
            self.cmp_spins[key].valueChanged.connect(self._on_cmp_plot_param_changed)
        self.cmp_cmap.currentTextChanged.connect(self._on_cmp_plot_param_changed)
        self.cmp_log_chk.toggled.connect(self._on_cmp_plot_param_changed)
        self.cmp_clip_chk.toggled.connect(self._on_cmp_plot_param_changed)
        self.cmp_auto_v_btn.clicked.connect(self._auto_cmp_vrange)
        self.cmp_auto_x_btn.clicked.connect(self._auto_cmp_xrange)
        self.cmp_auto_y_btn.clicked.connect(self._auto_cmp_yrange)
        self.power_refresh_groups_btn.clicked.connect(self._power_refresh_groups)
        self.power_group_combo.currentIndexChanged.connect(lambda _idx: self._on_power_plot_param_changed())
        self.power_kk_group_combo.currentIndexChanged.connect(lambda _idx: self._on_power_source_assignment_changed())
        self.power_kkp_group_combo.currentIndexChanged.connect(lambda _idx: self._on_power_source_assignment_changed())
        self.power_view_intensity_btn.clicked.connect(lambda: self._on_power_plot_view_button_clicked("Intensity"))
        self.power_view_vp_btn.clicked.connect(lambda: self._on_power_plot_view_button_clicked("VP"))
        self.power_axis_scale_combo.currentTextChanged.connect(self._on_power_axis_scale_changed)
        self.power_pair_mode_combo.currentTextChanged.connect(self._on_power_plot_param_changed)
        self.power_background_spin.valueChanged.connect(self._on_power_plot_param_changed)
        self.power_background_auto_chk.toggled.connect(self._on_power_background_mode_changed)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax", "gate"):
            self.power_spins[key].valueChanged.connect(self._on_power_plot_param_changed)
        self.power_cmap.currentTextChanged.connect(self._on_power_plot_param_changed)
        self.power_log_chk.toggled.connect(self._on_power_plot_param_changed)
        self.power_clip_chk.toggled.connect(self._on_power_plot_param_changed)
        self.power_auto_v_btn.clicked.connect(self._auto_power_vrange)
        self.power_auto_x_btn.clicked.connect(self._auto_power_xrange)
        self.power_auto_y_btn.clicked.connect(self._auto_power_yrange)
        self.shg_files.itemSelectionChanged.connect(self._on_shg_source_changed)
        self.shg_background_combo.currentIndexChanged.connect(lambda _idx: self._on_shg_source_changed())
        self.shg_workflow_tabs.currentChanged.connect(lambda _idx: self._on_shg_workflow_changed())
        for combo in (
            self.shg_compare_reference_combo,
            self.shg_compare_sample_combo,
            self.shg_compare_background_a_combo,
            self.shg_compare_background_b_combo,
        ):
            combo.currentIndexChanged.connect(lambda _idx: self._on_shg_source_changed())
        self.shg_background_method_combo.currentTextChanged.connect(lambda _text: self._on_shg_background_method_changed())
        self.shg_cosmic_enable_chk.toggled.connect(lambda _checked: self._on_shg_cosmic_param_changed())
        self.shg_spectrum_view_combo.currentTextChanged.connect(lambda _text: self._on_shg_spectrum_view_changed())
        self.shg_compare_display_combo.currentTextChanged.connect(lambda _text: self._on_shg_spectrum_view_changed())
        self.shg_angle_wrap_combo.currentTextChanged.connect(lambda _text: self._on_shg_param_changed())
        self.shg_include_failed_chk.toggled.connect(lambda _checked: self._on_shg_param_changed())
        self.shg_angle_cursor_spin.valueChanged.connect(lambda _value: self._on_shg_param_changed())
        for spin in (
            self.shg_peak_center_spin,
            self.shg_gate_half_range_spin,
            self.shg_sideband_gap_spin,
            self.shg_sideband_width_spin,
            self.shg_sigma_clip_spin,
            self.shg_angle_scale_spin,
            self.shg_angle_offset_spin,
        ):
            spin.editingFinished.connect(self._on_shg_param_changed)
        for spin in (
            self.shg_cosmic_threshold_spin,
            self.shg_cosmic_window_spin,
            self.shg_cosmic_max_width_spin,
        ):
            spin.editingFinished.connect(self._on_shg_cosmic_param_changed)
        self.shg_fit_enable_chk.toggled.connect(lambda _checked: self._on_shg_fit_param_changed())
        self.shg_fit_weighted_chk.toggled.connect(lambda _checked: self._on_shg_fit_param_changed())
        self.shg_fit_include_excluded_chk.toggled.connect(lambda _checked: self._on_shg_fit_param_changed())
        self.shg_fit_min_spin.editingFinished.connect(self._on_shg_fit_param_changed)
        self.shg_fit_max_spin.editingFinished.connect(self._on_shg_fit_param_changed)
        self.shg_fit_branch_spin.valueChanged.connect(lambda _value: self._on_shg_fit_param_changed())
        self.drr_yaxis_combo.currentTextChanged.connect(self._on_drr_plot_param_changed)
        self.drr_yaxis_a_spin.valueChanged.connect(self._on_drr_plot_param_changed)
        self.drr_yaxis_b_spin.valueChanged.connect(self._on_drr_plot_param_changed)
        self.drr_yaxis_c_spin.valueChanged.connect(self._on_drr_plot_param_changed)
        self.drr_baseline_combo.currentTextChanged.connect(self._on_drr_plot_param_changed)
        self.drr_derivative_combo.currentTextChanged.connect(self._on_drr_derivative_changed)
        self.drr_sg_window_spin.valueChanged.connect(self._on_drr_derivative_changed)
        self.drr_sg_poly_spin.valueChanged.connect(self._on_drr_derivative_changed)
        self.drr_edit_measurements_btn.clicked.connect(self._edit_drr_measurements)
        self.drr_clear_measurements_btn.clicked.connect(self._clear_drr_measurements)
        self.drr_edit_baselines_btn.clicked.connect(self._edit_drr_baselines_dialog)
        self.drr_baseline_autofind_btn.clicked.connect(self._auto_find_back_baselines)
        self.drr_baseline_combine_combo.currentTextChanged.connect(self._on_drr_baseline_mode_changed)
        self.drr_auto_v_btn.clicked.connect(self._auto_drr_vrange)
        self.drr_auto_x_btn.clicked.connect(self._auto_drr_xrange)
        self.drr_auto_y_btn.clicked.connect(self._auto_drr_yrange)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax", "gate"):
            self.drr_spins[key].valueChanged.connect(self._on_drr_plot_param_changed)
        self.drr_cmap.currentTextChanged.connect(self._on_drr_plot_param_changed)
        self.drr_log_chk.toggled.connect(self._on_drr_plot_param_changed)
        self.drr_clip_chk.toggled.connect(self._on_drr_plot_param_changed)
        self.drr_center_zero_chk.toggled.connect(self._on_drr_plot_param_changed)
        self.drr_peak_find_btn.clicked.connect(self._on_drr_find_peaks)
        self.drr_peak_show_chk.toggled.connect(self._on_drr_analysis_view_changed)
        self.drr_peak_mode_combo.currentTextChanged.connect(self._on_drr_analysis_view_changed)
        self.drr_fit_btn.clicked.connect(self._on_drr_fit_lorentz)
        self.drr_fit_clear_btn.clicked.connect(self._on_drr_clear_fit)
        self.drr_fit_show_chk.toggled.connect(self._on_drr_analysis_view_changed)
        self.mcd_files.itemSelectionChanged.connect(self._on_mcd_source_changed)
        self.mcd_auto_angles_chk.toggled.connect(self._on_mcd_angle_assignment_changed)
        self.mcd_sigma_plus_combo.currentIndexChanged.connect(self._on_mcd_params_changed)
        self.mcd_sigma_minus_combo.currentIndexChanged.connect(self._on_mcd_params_changed)
        self.mcd_reference_mode_combo.currentTextChanged.connect(self._on_mcd_reference_mode_changed)
        self.mcd_zero_spin.valueChanged.connect(self._on_mcd_params_changed)
        self.mcd_gap_spin.valueChanged.connect(self._on_mcd_params_changed)
        self.mcd_delta_b_spin.valueChanged.connect(self._on_mcd_params_changed)
        self.mcd_pair_alignment_combo.currentTextChanged.connect(self._on_mcd_params_changed)
        self.mcd_bin_spin.valueChanged.connect(self._on_mcd_params_changed)
        self.mcd_gain_combo.currentTextChanged.connect(self._on_mcd_params_changed)
        self.mcd_correction_mode_combo.currentTextChanged.connect(self._on_mcd_correction_mode_changed)
        self.mcd_spectral_order_combo.currentTextChanged.connect(self._on_mcd_params_changed)
        self.mcd_background_ranges_edit.editingFinished.connect(self._on_mcd_background_ranges_changed)
        self.mcd_suggest_background_btn.clicked.connect(self._suggest_mcd_background_ranges)
        self.mcd_apply_correction_btn.clicked.connect(self._apply_mcd_now)
        self.mcd_dark_pos_combo.currentIndexChanged.connect(self._on_mcd_params_changed)
        self.mcd_dark_neg_combo.currentIndexChanged.connect(self._on_mcd_params_changed)
        self.mcd_map_combo.currentTextChanged.connect(self._on_mcd_plot_changed)
        self.mcd_auto_v_btn.clicked.connect(self._auto_mcd_vrange)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax"):
            self.mcd_spins[key].valueChanged.connect(self._on_mcd_plot_changed)
        self.mcd_cmap.currentTextChanged.connect(self._on_mcd_plot_changed)
        self.mcd_center_zero_chk.toggled.connect(self._on_mcd_plot_changed)
        self.mcd_window_center_spin.valueChanged.connect(self._on_mcd_plot_changed)
        self.mcd_window_width_spin.valueChanged.connect(self._on_mcd_plot_changed)
        self.mcd_pair_b_combo.currentIndexChanged.connect(self._on_mcd_pair_selection_changed)
        self.mcd_window_metric_combo.currentTextChanged.connect(self._on_mcd_plot_changed)
        self.mcd_show_raw_chk.toggled.connect(self._on_mcd_plot_changed)
        self.mcd_show_signed_mean_chk.toggled.connect(self._on_mcd_plot_changed)
        self.mcd_show_absolute_mean_chk.toggled.connect(self._on_mcd_plot_changed)
        self.mcd_show_unsigned_absolute_mean_chk.toggled.connect(self._on_mcd_plot_changed)
        self.mcd_show_integral_chk.toggled.connect(self._on_mcd_plot_changed)
        self.mcd_fit_zero_chk.toggled.connect(self._on_mcd_plot_changed)
        self.mcd_fit_b_window_spin.valueChanged.connect(self._on_mcd_plot_changed)
        self.pl_peak_find_btn.clicked.connect(self._on_pl_find_peaks)
        self.pl_peak_show_chk.toggled.connect(self._on_pl_analysis_view_changed)
        self.pl_peak_mode_combo.currentTextChanged.connect(self._on_pl_analysis_view_changed)
        self.pl_fit_btn.clicked.connect(self._on_pl_fit_lorentz)
        self.pl_fit_clear_btn.clicked.connect(self._on_pl_clear_fit)
        self.pl_fit_show_chk.toggled.connect(self._on_pl_analysis_view_changed)

        self.show_log_btn.clicked.connect(self._toggle_log)
        self.clear_log_btn.clicked.connect(self._clear_log)
        self._gate_motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self._gate_click_cid = self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        for prefix in ("pl", "drr", "cmp"):
            self._update_y_axis_controls(prefix)
        self._cmp_apply_display_preset()
        self._cmp_update_view_mode()
        self._cmp_set_channel_combo_items()
        self._cmp_update_assignment_summary()
        self._power_refresh_groups()
        self._power_update_view_mode()
        self._mcd_refresh_sources()
        self._shg_refresh_sources()
        if hasattr(self, "power_background_spin"):
            self.power_background_spin.setEnabled(not self._power_background_auto_enabled())
        self._update_plot_view_bar_visibility()

    def _toggle_log(self) -> None:
        self.log_dock.setVisible(not self.log_dock.isVisible())

    def _clear_log(self) -> None:
        self.log_lines.clear()
        self.log_text.clear()

    def _selected(self, widget: QListWidget) -> List[str]:
        return [i.text() for i in widget.selectedItems()]

    def _update_move_exported_sources_state(self) -> None:
        if not hasattr(self, "move_now_btn"):
            return
        enabled = bool(self._last_export_move_sources and self._last_export_move_folder)
        self.move_now_btn.setEnabled(enabled)
        if enabled:
            count = len(self._last_export_move_sources)
            self.move_now_btn.setToolTip(
                f"Move {count} source file{'s' if count != 1 else ''} used by the most recent successful save."
            )
        else:
            self.move_now_btn.setToolTip("Save first to enable moving exported source files.")

    def _set_export_move_sources(self, folder: str, sources: Sequence[str]) -> None:
        self._last_export_move_folder = str(folder)
        self._last_export_move_sources = [name for name in dict.fromkeys(str(s) for s in sources if s)]
        self._update_move_exported_sources_state()

    def _invalidate_export_move_sources(self) -> None:
        self._last_export_move_folder = ""
        self._last_export_move_sources = []
        self._update_move_exported_sources_state()

    def _manual_move_sources(self) -> None:
        folder = self._last_export_move_folder
        names = list(self._last_export_move_sources)
        self._invalidate_export_move_sources()
        if not folder or not names:
            self._show_error("Save first to enable moving exported source files.")
            return
        moved = int(data_io.move_selected_to_archive(folder, names))
        if self.current_folder and str(Path(folder)).lower() == self.current_folder.lower():
            self._refresh_file_lists()
        self._append_log(f"Moved {moved} source CSV file(s) to 'Initial data after processing'.")
        self._status(f"Moved {moved} file(s).")

    def _status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def _append_log(self, text: str) -> None:
        self.log_lines.append(text)
        self.log_text.setPlainText("\n".join(self.log_lines))
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _set_stage(self, stage: str) -> None:
        self._status(f"State: {stage}")
        self._status_progress.setVisible(stage == "Loading...")

    def _active_mode(self) -> str | None:
        text = self.tabs.tabText(self.tabs.currentIndex())
        return {
            "PL": "PL",
            "DRR": "DRR",
            "Compare": "Compare",
            "Power": "Power Dependent",
            "MCD": "MCD",
            "SHG": "SHG Processing",
        }.get(text)

    def _toolbar_load(self) -> None:
        mode = self._active_mode()
        if mode:
            self._start_load(mode)

    def _toolbar_plot(self) -> None:
        mode = self._active_mode()
        if mode:
            self._plot_mode(mode)

    def _toolbar_save(self) -> None:
        mode = self._active_mode()
        if mode:
            self._start_export(mode)

    def _on_drr_derivative_changed(self) -> None:
        self._invalidate_export_move_sources()
        derivative_active = self._drr_derivative_value() is not None
        self.drr_sg_window_spin.setVisible(derivative_active)
        self.drr_sg_poly_spin.setVisible(derivative_active)
        self._enforce_drr_sg_constraints(show_status=True)
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            self._refresh_automatic_ranges("DRR", refresh_split=True)
            self._plot_mode("DRR")

    def _on_drr_plot_param_changed(self) -> None:
        self._invalidate_export_move_sources()
        external_baseline = self.drr_baseline_combo.currentText() == "External"
        self.drr_external_baseline_row.setVisible(external_baseline)
        self.drr_baseline_combine_combo.setVisible(external_baseline)
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            sender = self.sender()
            if sender in (
                self.drr_spins["xmin"], self.drr_spins["xmax"],
                self.drr_spins["ymin"], self.drr_spins["ymax"],
                self.drr_log_chk, self.drr_center_zero_chk,
            ):
                self._refresh_automatic_ranges(
                    "DRR",
                    refresh_split=True,
                    center_split=sender in (self.drr_spins["xmin"], self.drr_spins["xmax"]),
                )
            self._plot_mode("DRR")

    def _on_pl_plot_param_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._apply_dat_y_axis_selection()
        if self.loaded and self.loaded.mode == "PL":
            sender = self.sender()
            if sender in (
                self.pl_spins["xmin"], self.pl_spins["xmax"],
                self.pl_spins["ymin"], self.pl_spins["ymax"], self.pl_log_chk,
            ):
                self._refresh_automatic_ranges(
                    "PL",
                    refresh_split=True,
                    center_split=sender in (self.pl_spins["xmin"], self.pl_spins["xmax"]),
                )
            self._plot_mode("PL")

    def _apply_dat_y_axis_selection(self) -> None:
        if not self.loaded or self.loaded.mode != "PL" or not self.loaded.cube:
            return
        if Path(self.loaded.primary_file or "").suffix.lower() != ".dat":
            return
        label, unit, semantic = resolve_dat_y_axis(
            self.pl_yaxis_combo.currentText(),
            custom_label=self.pl_dat_yaxis_label_edit.text(),
            custom_unit=self.pl_dat_yaxis_unit_edit.text(),
        )
        self.loaded.cube.gate_label = label
        self.loaded.cube.gate_unit = unit
        self.loaded.cube.y_axis_semantic = semantic

    def _on_pl_dat_y_axis_changed(self, _text: str) -> None:
        is_dat = Path((self.loaded.primary_file if self.loaded else "") or "").suffix.lower() == ".dat"
        custom = self.pl_yaxis_combo.currentText() == "Custom"
        self.pl_dat_yaxis_label_edit.setVisible(is_dat and custom)
        self.pl_dat_yaxis_unit_edit.setVisible(is_dat)
        self._on_pl_plot_param_changed()

    def _on_mcd_params_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._update_mcd_background_preview()
        if self.loaded and self.loaded.mode == "MCD":
            self.mcd_apply_correction_btn.setText("Pending update...")
            self._status("MCD processing settings changed; recalculation is scheduled automatically.")
            self._mcd_auto_apply_timer.start()

    def _apply_mcd_now(self) -> None:
        self._mcd_auto_apply_timer.stop()
        self._apply_pending_mcd_settings()

    def _apply_pending_mcd_settings(self) -> None:
        """Apply the newest MCD settings once after a burst of UI changes."""
        if not self.loaded or self.loaded.mode != "MCD":
            return
        if self._load_in_progress:
            self._mcd_reapply_pending = True
            self.mcd_apply_correction_btn.setText("Pending update...")
            return
        if not self._selected(self.mcd_files):
            self.mcd_apply_correction_btn.setText("Recalculate now")
            return
        try:
            self._mcd_settings_from_ui()
        except Exception as exc:
            self.mcd_apply_correction_btn.setText("Fix settings")
            self._status(f"MCD automatic recalculation paused: {exc}")
            return
        self._mcd_reapply_pending = False
        self.mcd_apply_correction_btn.setText("Applying...")
        self._status("Applying updated MCD processing settings...")
        self._start_load("MCD")

    def _on_mcd_correction_mode_changed(self, text: str) -> None:
        self.mcd_spectral_order_combo.setEnabled("spectral baseline" in text.lower())
        self._on_mcd_params_changed()

    def _on_mcd_background_ranges_changed(self) -> None:
        self._mcd_background_suggestion = None
        self._on_mcd_params_changed()

    def _on_mcd_source_changed(self) -> None:
        self._mcd_background_suggestion = None
        self._mcd_detect_available_angles()
        self._on_mcd_params_changed()

    def _on_mcd_angle_assignment_changed(self, automatic: bool) -> None:
        self.mcd_sigma_plus_combo.setEnabled(not automatic)
        self.mcd_sigma_minus_combo.setEnabled(not automatic)
        if automatic:
            self._mcd_detect_available_angles()
        self._on_mcd_params_changed()

    def _on_mcd_reference_mode_changed(self, _text: str) -> None:
        self.mcd_zero_spin.setEnabled(self.mcd_reference_mode_combo.currentIndex() == 1)
        self._on_mcd_params_changed()

    def _on_mcd_plot_changed(self) -> None:
        self._invalidate_export_move_sources()
        if self.loaded and self.loaded.mode == "MCD":
            if self.sender() in (self.mcd_map_combo, self.mcd_center_zero_chk):
                self._refresh_automatic_ranges("MCD", refresh_split=False)
            self._plot_mode("MCD")

    def _select_mcd_pair_at_b(self, requested_b: float, *, clicked: bool = False) -> None:
        """Select the nearest real pair for a B value requested on the map."""
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            return
        result = self.loaded.mcd_result
        if result.pair_b.size == 0:
            return
        distance = np.abs(result.pair_b - float(requested_b))
        pair_index = int(np.nanargmin(distance))
        # If two sweep branches have the same nearest B, retaining the active
        # branch makes repeated clicking at that field deterministic.
        nearest = float(distance[pair_index])
        candidates = np.flatnonzero(np.isclose(distance, nearest, atol=1e-10, rtol=0.0))
        current_index = self.mcd_pair_b_combo.currentData()
        if current_index is not None and int(current_index) in candidates:
            pair_index = int(current_index)
        selected_b = float(result.pair_b[pair_index])
        combo_blocked = self.mcd_pair_b_combo.blockSignals(True)
        self.mcd_pair_b_combo.setCurrentIndex(pair_index)
        self.mcd_pair_b_combo.blockSignals(combo_blocked)
        if clicked:
            self._status(f"Clicked {float(requested_b):.4g} T; showing nearest paired measurement {selected_b:.4g} T.")
        self._on_mcd_plot_changed()

    def _on_mcd_pair_selection_changed(self, _index: int) -> None:
        """Redraw all MCD pair displays after a manual pair selection."""
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            return
        self._on_mcd_plot_changed()

    def _on_power_axis_scale_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._power_update_group_summary()
        self._on_power_plot_param_changed()

    def _on_power_background_mode_changed(self, _checked: bool) -> None:
        self._invalidate_export_move_sources()
        self.power_background_spin.setEnabled(not self._power_background_auto_enabled())
        self._on_power_plot_param_changed()

    def _on_power_plot_param_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._power_update_group_summary()
        if self.loaded and self.loaded.mode == "Power Dependent":
            sender = self.sender()
            if sender in (
                self.power_spins["xmin"], self.power_spins["xmax"],
                self.power_spins["ymin"], self.power_spins["ymax"],
                self.power_log_chk, self.power_background_spin,
                self.power_background_auto_chk,
            ):
                self._refresh_automatic_ranges(
                    "Power Dependent",
                    refresh_split=True,
                    center_split=sender in (self.power_spins["xmin"], self.power_spins["xmax"]),
                )
            self._plot_mode("Power Dependent")

    def _on_power_source_assignment_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._power_update_group_summary()
        self._power_update_vp_availability()
        if self.loaded and self.loaded.mode == "Power Dependent":
            self._refresh_automatic_ranges("Power Dependent", refresh_split=True)
            self._plot_mode("Power Dependent")

    def _on_cmp_auto_assign_requested(self) -> None:
        self._invalidate_export_move_sources()
        self._cmp_auto_assign_channels()
        self._on_cmp_plot_param_changed()

    def _on_cmp_infer_angles_requested(self) -> None:
        self._invalidate_export_move_sources()
        self._cmp_infer_angle_references()
        self._cmp_auto_assign_channels()
        self._on_cmp_plot_param_changed()

    def _on_cmp_display_preset_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._cmp_apply_display_preset()
        self._cmp_update_assignment_summary()
        if self.loaded and self.loaded.mode == "Compare":
            self._refresh_automatic_ranges("Compare", refresh_split=True)
        self._on_cmp_plot_param_changed()

    def _on_cmp_plot_view_button_clicked(self, mode: str) -> None:
        self._cmp_set_view_mode(mode)
        self._on_cmp_view_changed()

    def _on_power_plot_view_button_clicked(self, mode: str) -> None:
        self._invalidate_export_move_sources()
        self._power_selected_row_index = None
        self._power_set_view_mode(mode)
        self._update_action_states()
        if self.loaded and self.loaded.mode == "Power Dependent" and mode == "Intensity":
            self._refresh_automatic_ranges("Power Dependent", refresh_split=True)
        self._on_power_plot_param_changed()

    def _on_cmp_view_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._cmp_update_view_mode()
        self._cmp_update_assignment_summary()
        self._update_action_states()
        if self.loaded and self.loaded.mode == "Compare" and not self._cmp_is_vp_view():
            self._refresh_automatic_ranges("Compare", refresh_split=True)
        self._on_cmp_plot_param_changed()

    def _on_cmp_background_mode_changed(self, _checked: bool) -> None:
        self._invalidate_export_move_sources()
        self._cmp_update_background_mode()
        if self.loaded and self.loaded.mode == "Compare" and self.loaded.compare_cubes:
            self._cmp_background_value(self.loaded.compare_cubes)
        self._on_cmp_plot_param_changed()

    def _on_tab_changed(self, _index: int) -> None:
        self._invalidate_export_move_sources()
        self._update_action_states()
        self._update_plot_view_bar_visibility()
        self._update_results_dock_page()

    def _on_cmp_plot_param_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._cmp_update_assignment_summary()
        if self.loaded and self.loaded.mode == "Compare":
            sender = self.sender()
            if sender in (
                self.cmp_spins["xmin"], self.cmp_spins["xmax"],
                self.cmp_spins["ymin"], self.cmp_spins["ymax"],
                self.cmp_log_chk, self.cmp_vp_background_spin,
                self.cmp_vp_auto_background_chk,
            ) or sender in tuple(self.cmp_channel_combos.values()) or sender in tuple(self.cmp_show_checks.values()):
                self._refresh_automatic_ranges(
                    "Compare",
                    refresh_split=True,
                    center_split=sender in (self.cmp_spins["xmin"], self.cmp_spins["xmax"]),
                )
            self._plot_mode("Compare")

    def _show_error(self, message: str) -> None:
        first = message.splitlines()[0] if message else "Unknown error"
        self._append_log(f"ERROR: {first}")
        self._status(f"Error: {first}")
        QMessageBox.critical(self, "Error", message)

    def _on_recent_folder_selected(self, index: int) -> None:
        folder = self.recent_folder_combo.itemData(index)
        if not folder:
            return
        folder_text = str(folder)
        if self.current_folder and folder_text.lower() == self.current_folder.lower():
            return
        if self._set_current_folder(folder_text):
            self._status(f"Folder set: {Path(folder_text).name} - {len(self.available_files)} CSV files found")

    def _browse_folder(self) -> None:
        start = self._browse_start_folder()
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder", start)
        if not folder:
            return
        if self._set_current_folder(folder):
            self._status(f"Folder set: {Path(folder).name} - {len(self.available_files)} CSV files found")

    def _open_file(self) -> None:
        start = self.current_folder or self._browse_start_folder()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", start, "Data files (*.csv *.xlsx *.dat)"
        )
        if not file_path:
            return
        path = Path(file_path)
        if not self._set_current_folder(str(path.parent)):
            return
        matches = self.pl_files.findItems(path.name, Qt.MatchExactly)
        if matches:
            self.pl_files.clearSelection()
            matches[0].setSelected(True)
        self.drr_selected_files = [path.name]
        self._update_drr_selection_labels()
        self._power_refresh_groups()
        self._mcd_refresh_sources()
        self._shg_refresh_sources()
        self._restore_list_selection(self.shg_files, [path.name])
        self._status(f"Selected {path.name}")

    def _restore_list_selection(self, widget: QListWidget, names: List[str]) -> None:
        widget.clearSelection()
        for name in names:
            for match in widget.findItems(name, Qt.MatchExactly):
                match.setSelected(True)

    def _refresh_file_lists(self, *, auto: bool = False) -> None:
        old_files = set(self.available_files)
        pl_selected = self._selected(self.pl_files)
        self.pl_files.clear()
        if not self.current_folder:
            self.available_files = []
            self.available_map_files = []
            return
        self.available_files = data_io.list_csv_files(self.current_folder)
        self.available_map_files = data_io.list_map_input_files(self.current_folder)
        self.pl_files.addItems(self.available_map_files)
        self._restore_list_selection(self.pl_files, [f for f in pl_selected if f in self.available_map_files])
        self._cmp_set_channel_combo_items()
        self._power_refresh_groups()
        self._mcd_refresh_sources()
        self._shg_refresh_sources()
        self.drr_selected_files = [f for f in self.drr_selected_files if f in self.available_map_files]
        self.drr_baseline_files_manual = [f for f in self.drr_baseline_files_manual if f in self.available_files]
        self.drr_baseline_files_found = [f for f in self.drr_baseline_files_found if f in self.available_files]
        self._cmp_auto_assign_channels()
        self._update_drr_selection_labels()
        new_files = set(self.available_files)
        added = len(new_files - old_files)
        removed = len(old_files - new_files)
        if added or removed:
            self._invalidate_export_move_sources()
        if auto and (added or removed):
            parts = []
            if added:
                parts.append(f"{added} new")
            if removed:
                parts.append(f"{removed} removed")
            self._status(f"Data source updated: {', '.join(parts)} ({len(self.available_files)} CSV files).")
        elif not auto:
            self._status(f"Loaded file list: {len(self.available_files)}")

    def _update_drr_selection_labels(self) -> None:
        def _brief(names: List[str]) -> str:
            if not names:
                return "none"
            # Zero-width break opportunities preserve the complete first name
            # while allowing underscore-heavy measurement names to wrap.
            first = names[0].replace("_", "_\u200b").replace("-", "-\u200b")
            return first if len(names) == 1 else f"{first} (+{len(names) - 1} more)"

        mode_map = {
            "Last frame of each baseline file": "last",
            "First frame of each baseline file": "first",
            "Average all baseline files": "avg",
        }
        mode_short = mode_map.get(self.drr_baseline_combine_combo.currentText(), "last")
        self.drr_measurement_summary.setText(f"Measurement: {len(self.drr_selected_files)} files ({_brief(self.drr_selected_files)})")
        self.drr_measurement_summary.setToolTip(
            "Selected measurement files:\n" + "\n".join(self.drr_selected_files)
            if self.drr_selected_files
            else "No measurement files selected."
        )
        self.drr_baseline_summary.setText(
            f"Baselines: {len(self.drr_baseline_files_manual)} files (mode: {mode_short})"
        )
        self.drr_baseline_summary.setToolTip(
            "Selected baseline files:\n" + "\n".join(self.drr_baseline_files_manual)
            if self.drr_baseline_files_manual
            else "No external baseline files selected."
        )
        self._repopulate_drr_yaxis()

    def _reject_mixed_xlsx_selection(self, selected: List[str]) -> None:
        if not selected:
            return
        xlsx = [name for name in selected if data_io.is_xlsx_map_file(name)]
        if not xlsx:
            return
        if len(xlsx) != 1 or len(selected) != 1:
            raise ValueError("Select exactly one XLSX map for direct DR/R display (XLSX maps cannot be mixed with CSV measurements).")

    def _edit_drr_measurements(self) -> None:
        selected = self._open_dual_list_dialog(
            title="Measurement Files",
            selected=self.drr_selected_files,
            enable_group_auto=True,
            enable_back_auto=False,
            candidates=self.available_map_files,
        )
        self._reject_mixed_xlsx_selection(selected)
        self.drr_selected_files = selected
        self._update_drr_selection_labels()
        if self.loaded and self.loaded.mode == "DRR":
            self._start_load("DRR")

    def _clear_drr_measurements(self) -> None:
        self.drr_selected_files = []
        self._update_drr_selection_labels()

    def _edit_drr_baselines_dialog(self) -> None:
        self.drr_baseline_files_manual = self._open_dual_list_dialog(
            title="Baseline Files",
            selected=self.drr_baseline_files_manual,
            enable_group_auto=False,
            enable_back_auto=True,
        )
        self._update_drr_selection_labels()
        if self.loaded and self.loaded.mode == "DRR":
            self._start_load("DRR")

    def _open_dual_list_dialog(
        self,
        *,
        title: str,
        selected: List[str],
        enable_group_auto: bool,
        enable_back_auto: bool,
        candidates: List[str] | None = None,
    ) -> List[str]:
        if candidates is None:
            candidates = self.available_files
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        if not self.windowIcon().isNull():
            dlg.setWindowIcon(self.windowIcon())
        dlg.setMinimumSize(760, 520)
        dlg.resize(1000, 650)
        dlg.setSizeGripEnabled(True)
        v = QVBoxLayout(dlg)
        top_row = QHBoxLayout()
        filter_edit = QLineEdit()
        filter_edit.setPlaceholderText("Filter filenames...")
        top_row.addWidget(QLabel("Filter"))
        top_row.addWidget(filter_edit, 1)
        if enable_back_auto:
            auto_back_btn = QPushButton("Find 'back'")
            auto_back_btn.setToolTip("Auto-select files where filename contains 'back'.")
            top_row.addWidget(auto_back_btn)
        if enable_group_auto:
            auto_group_btn = QPushButton("Use Largest Group")
            auto_group_btn.setToolTip("Auto-select the largest measurement group in this folder.")
            top_row.addWidget(auto_group_btn)
        v.addLayout(top_row)

        lists_splitter = QSplitter(Qt.Vertical)
        available = QListWidget()
        available.setSelectionMode(QAbstractItemView.ExtendedSelection)
        current = QListWidget()
        current.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for file_list in (available, current):
            file_list.setWordWrap(True)
            file_list.setTextElideMode(Qt.ElideNone)
            file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            file_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
            file_list.setUniformItemSizes(False)
            file_list.setResizeMode(QListView.Adjust)
            file_list.setSpacing(2)
            file_list.setItemDelegate(WrappedFilenameDelegate(file_list))

        selected_set = set(selected)
        for f in candidates:
            if f not in selected_set:
                available.addItem(f)
        current.addItems(selected)

        available_panel = QWidget()
        available_layout = QVBoxLayout(available_panel)
        available_layout.setContentsMargins(0, 0, 0, 0)
        available_layout.setSpacing(4)
        available_label = QLabel()
        available_layout.addWidget(available_label)
        available_layout.addWidget(available, 1)

        current_panel = QWidget()
        current_layout = QVBoxLayout(current_panel)
        current_layout.setContentsMargins(0, 0, 0, 0)
        current_layout.setSpacing(4)
        current_label = QLabel()
        current_layout.addWidget(current_label)
        current_layout.addWidget(current, 1)

        transfer_panel = QWidget()
        transfer_panel.setFixedHeight(44)
        transfer_row = QHBoxLayout(transfer_panel)
        transfer_row.setContentsMargins(0, 4, 0, 4)
        transfer_row.addStretch(1)
        add_btn = QPushButton("Add Selected ↓")
        remove_btn = QPushButton("Remove Selected ↑")
        clear_btn = QPushButton("Clear Selected")
        transfer_row.addWidget(add_btn)
        transfer_row.addWidget(remove_btn)
        transfer_row.addWidget(clear_btn)
        transfer_row.addStretch(1)

        lists_splitter.addWidget(available_panel)
        lists_splitter.addWidget(transfer_panel)
        lists_splitter.addWidget(current_panel)
        lists_splitter.setChildrenCollapsible(False)
        lists_splitter.setStretchFactor(0, 1)
        lists_splitter.setStretchFactor(1, 0)
        lists_splitter.setStretchFactor(2, 1)
        lists_splitter.setSizes([260, 44, 220])
        v.addWidget(lists_splitter, 1)

        def _update_counts() -> None:
            available_label.setText(f"Available files ({available.count()})")
            current_label.setText(f"Selected files ({current.count()})")

        def _move(src: QListWidget, dst: QListWidget) -> None:
            items = [i.text() for i in src.selectedItems()]
            if not items:
                return
            existing = {dst.item(i).text() for i in range(dst.count())}
            for name in items:
                if name not in existing:
                    dst.addItem(name)
                    existing.add(name)
            for it in src.selectedItems():
                src.takeItem(src.row(it))
            _update_counts()

        def _filter() -> None:
            needle = filter_edit.text().strip().lower()
            selected_now = [current.item(i).text() for i in range(current.count())]
            available.clear()
            for f in candidates:
                if f in selected_now:
                    continue
                if not needle or needle in f.lower():
                    available.addItem(f)
            _update_counts()

        def _clear_selected() -> None:
            current.clear()
            _filter()

        def _remove_selected() -> None:
            _move(current, available)
            _filter()

        add_btn.clicked.connect(lambda: _move(available, current))
        remove_btn.clicked.connect(_remove_selected)
        clear_btn.clicked.connect(_clear_selected)
        filter_edit.textChanged.connect(lambda _t: _filter())
        available.itemDoubleClicked.connect(lambda _i: _move(available, current))
        current.itemDoubleClicked.connect(lambda _i: _remove_selected())

        if enable_back_auto:
            def _auto_back() -> None:
                matches = [f for f in candidates if "back" in f.lower()]
                current.clear()
                current.addItems(matches)
                _filter()
                self._status(f"State: Found {len(matches)} baseline files containing 'back'.")
            auto_back_btn.clicked.connect(_auto_back)

        if enable_group_auto:
            def _auto_group() -> None:
                groups = group_measurement_files(candidates)
                if not groups:
                    return
                largest_key = max(groups.keys(), key=lambda k: len(groups[k]))
                matches = [f for f in groups[largest_key] if "back" not in f.lower()]
                current.clear()
                current.addItems(matches)
                _filter()
                self._status(f"State: Selected group '{largest_key}' ({len(matches)} files).")
            auto_group_btn.clicked.connect(_auto_group)

        _update_counts()

        b = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        b.accepted.connect(dlg.accept)
        b.rejected.connect(dlg.reject)
        v.addWidget(b)
        if dlg.exec() != QDialog.Accepted:
            return selected
        return [current.item(i).text() for i in range(current.count())]

    def _on_drr_baseline_mode_changed(self) -> None:
        self._status(f"State: Baseline mode set: {self.drr_baseline_combine_combo.currentText()}.")
        self._update_drr_selection_labels()
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            self._plot_mode("DRR")

    def _auto_find_back_baselines(self) -> None:
        matches = [f for f in self.available_files if "back" in f.lower()]
        self.drr_baseline_files_manual = matches
        if matches:
            self.drr_baseline_combo.setCurrentText("External")
        self._update_drr_selection_labels()
        self._status(f"State: Auto-found {len(matches)} baseline files containing 'back'.")
        if self.loaded and self.loaded.mode == "DRR":
            self._start_load("DRR")

    def _update_action_states(self) -> None:
        active_mode = self._active_mode()
        loaded_mode = self.loaded.mode if self.loaded else None
        plotted_mode = self.last_plotted_mode
        self.load_action.setEnabled(active_mode is not None)
        self.plot_action.setEnabled(active_mode is not None and loaded_mode == active_mode)
        self.save_action.setEnabled(active_mode is not None and plotted_mode == active_mode)
        self._update_move_exported_sources_state()
        pl_loaded = loaded_mode == "PL"
        drr_loaded = loaded_mode == "DRR"
        cmp_loaded = loaded_mode == "Compare"
        power_loaded = loaded_mode == "Power Dependent"
        mcd_loaded = loaded_mode == "MCD"
        self.pl_auto_v_btn.setEnabled(pl_loaded and not self.pl_split_scale_chk.isChecked())
        self.pl_auto_x_btn.setEnabled(pl_loaded)
        self.pl_auto_y_btn.setEnabled(pl_loaded)
        self.drr_auto_v_btn.setEnabled(drr_loaded and not self.drr_split_scale_chk.isChecked())
        self.drr_auto_x_btn.setEnabled(drr_loaded)
        self.drr_auto_y_btn.setEnabled(drr_loaded)
        self.cmp_auto_v_btn.setEnabled(cmp_loaded and not self.cmp_split_scale_chk.isChecked())
        self.cmp_auto_x_btn.setEnabled(cmp_loaded)
        self.cmp_auto_y_btn.setEnabled(cmp_loaded)
        self.power_auto_v_btn.setEnabled(power_loaded and not self.power_split_scale_chk.isChecked())
        self.power_auto_x_btn.setEnabled(power_loaded)
        self.power_auto_y_btn.setEnabled(power_loaded)
        self.mcd_auto_v_btn.setEnabled(mcd_loaded)
        for prefix, enabled in (
            ("pl", pl_loaded),
            ("drr", drr_loaded),
            ("cmp", cmp_loaded and not self._cmp_is_vp_view()),
            ("power", power_loaded and self._power_view() != "VP"),
        ):
            split_enabled = bool(getattr(self, f"{prefix}_split_scale_chk").isChecked())
            split_fix_checks: Dict[str, QCheckBox] = getattr(
                self, f"{prefix}_split_fix_checks"
            )
            left_fully_fixed = all(
                split_fix_checks[key].isChecked()
                for key in ("left_vmin", "left_vmax")
            )
            right_fully_fixed = all(
                split_fix_checks[key].isChecked()
                for key in ("right_vmin", "right_vmax")
            )
            getattr(self, f"{prefix}_split_auto_left_btn").setEnabled(
                enabled and split_enabled and not left_fully_fixed
            )
            getattr(self, f"{prefix}_split_auto_right_btn").setEnabled(
                enabled and split_enabled and not right_fully_fixed
            )
        cmp_split_available = not self._cmp_is_vp_view()
        power_split_available = self._power_view() != "VP"
        self.cmp_split_scale_chk.setEnabled(cmp_split_available)
        self.cmp_split_scale_panel.setEnabled(cmp_split_available)
        self.power_split_scale_chk.setEnabled(power_split_available)
        self.power_split_scale_panel.setEnabled(power_split_available)

    def _auto_drr_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        x = np.asarray(cube.energy, float).ravel()
        y = np.asarray(cube.gate, float).ravel()
        z = np.asarray(cube.Z, float)
        x0, x1 = sorted((float(self.drr_spins["xmin"].value()), float(self.drr_spins["xmax"].value())))
        y0, y1 = sorted((float(self.drr_spins["ymin"].value()), float(self.drr_spins["ymax"].value())))
        x_mask = (x >= x0) & (x <= x1)
        y_mask = (y >= y0) & (y <= y1)
        if np.any(y_mask) and np.any(x_mask):
            z_roi = z[np.ix_(y_mask, x_mask)]
        else:
            z_roi = z
        finite = z_roi[np.isfinite(z_roi)]
        if finite.size == 0:
            self._status("State: Auto vmin/vmax skipped (no finite values in selected x/y range).")
            return
        if self._mode_log("DRR"):
            pos = z_roi[np.isfinite(z_roi) & (z_roi > 0)]
            if pos.size:
                vmin, vmax = np.nanpercentile(pos, [0.01, 99.99])
                vmin = float(max(vmin, 1e-12))
                vmax = float(max(vmax, vmin * 1.01))
            else:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = np.nanpercentile(finite, [0.01, 99.99])
            vmin, vmax = float(vmin), float(vmax)
        spins = self.drr_spins
        spins["vmin"].setValue(vmin)
        spins["vmax"].setValue(vmax)
        self._status(f"State: Auto vmin/vmax (ROI) = {vmin:.4g}, {vmax:.4g}")
        self._plot_mode("DRR")

    def _auto_mcd_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            return
        cube = self.loaded.mcd_result.cube(self.mcd_map_combo.currentText())
        finite = np.asarray(cube.Z, float)[np.isfinite(cube.Z)]
        if not finite.size:
            return
        bound = float(np.nanpercentile(np.abs(finite), 99.5))
        if self.mcd_center_zero_chk.isChecked():
            self.mcd_spins["vmin"].setValue(-bound); self.mcd_spins["vmax"].setValue(bound)
        else:
            lo, hi = np.nanpercentile(finite, [0.5, 99.5])
            self.mcd_spins["vmin"].setValue(float(lo)); self.mcd_spins["vmax"].setValue(float(hi))
        self._plot_mode("MCD")

    def _auto_pl_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "PL" or self.loaded.cube is None:
            return
        cube = self.loaded.cube
        x = np.asarray(cube.energy, float).ravel()
        y = np.asarray(cube.gate, float).ravel()
        z = np.asarray(cube.Z, float)
        x0, x1 = sorted((float(self.pl_spins["xmin"].value()), float(self.pl_spins["xmax"].value())))
        y0, y1 = sorted((float(self.pl_spins["ymin"].value()), float(self.pl_spins["ymax"].value())))
        x_mask = (x >= x0) & (x <= x1)
        y_mask = (y >= y0) & (y <= y1)
        z_roi = z[np.ix_(y_mask, x_mask)] if np.any(y_mask) and np.any(x_mask) else z
        finite = z_roi[np.isfinite(z_roi)]
        if finite.size == 0:
            self._status("State: Auto vmin/vmax skipped (no finite values in selected x/y range).")
            return
        if self._mode_log("PL"):
            pos = z_roi[np.isfinite(z_roi) & (z_roi > 0)]
            if pos.size:
                vmin, vmax = np.nanpercentile(pos, [0.01, 99.99])
                vmin = float(max(vmin, 1e-12))
                vmax = float(max(vmax, vmin * 1.01))
            else:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = np.nanpercentile(finite, [0.01, 99.99])
            vmin, vmax = float(vmin), float(vmax)
        self.pl_spins["vmin"].setValue(vmin)
        self.pl_spins["vmax"].setValue(vmax)
        self._status(f"State: Auto vmin/vmax (ROI) = {vmin:.4g}, {vmax:.4g}")
        self._plot_mode("PL")

    def _auto_cmp_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Compare" or not self.loaded.compare_cubes:
            return
        x0, x1 = sorted((float(self.cmp_spins["xmin"].value()), float(self.cmp_spins["xmax"].value())))
        y0, y1 = sorted((float(self.cmp_spins["ymin"].value()), float(self.cmp_spins["ymax"].value())))
        vals: list[np.ndarray] = []
        background = self._cmp_background_value(self.loaded.compare_cubes)
        for cube in self._cmp_corrected_cubes(self.loaded.compare_cubes, background=background).values():
            x = np.asarray(cube.energy, float).ravel()
            y = np.asarray(cube.gate, float).ravel()
            z = np.asarray(cube.Z, float)
            x_mask = (x >= x0) & (x <= x1)
            y_mask = (y >= y0) & (y <= y1)
            z_roi = z[np.ix_(y_mask, x_mask)] if np.any(y_mask) and np.any(x_mask) else z
            finite = z_roi[np.isfinite(z_roi)]
            if finite.size:
                vals.append(finite)
        if not vals:
            return
        finite = np.concatenate(vals)
        if self._mode_log("Compare"):
            pos = finite[finite > 0]
            if pos.size:
                vmin, vmax = np.nanpercentile(pos, [0.01, 99.99])
                vmin = float(max(vmin, 1e-12))
                vmax = float(max(vmax, vmin * 1.01))
            else:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = np.nanpercentile(finite, [0.01, 99.99])
            vmin, vmax = float(vmin), float(vmax)
        self.cmp_spins["vmin"].setValue(vmin)
        self.cmp_spins["vmax"].setValue(vmax)
        self._plot_mode("Compare")

    def _auto_power_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Power Dependent" or self.loaded.cube is None:
            return
        x0, x1 = sorted((float(self.power_spins["xmin"].value()), float(self.power_spins["xmax"].value())))
        y0, y1 = sorted((float(self.power_spins["ymin"].value()), float(self.power_spins["ymax"].value())))

        def _roi_finite(cube) -> np.ndarray:
            x = np.asarray(cube.energy, float).ravel()
            y = np.asarray(cube.gate, float).ravel()
            z = np.asarray(cube.Z, float)
            x_mask = (x >= x0) & (x <= x1)
            y_mask = (y >= y0) & (y <= y1)
            z_roi = z[np.ix_(y_mask, x_mask)] if np.any(y_mask) and np.any(x_mask) else z
            return z_roi[np.isfinite(z_roi)]

        if self._power_has_distinct_role_groups():
            try:
                kk_result, kkp_result, _kk_key, _kkp_key = self._power_role_payload()
                bg = self._power_background_value([kk_result.cube, kkp_result.cube])
                kk_c = self._power_corrected_cube(kk_result.cube, background=bg)
                kkp_c = self._power_corrected_cube(kkp_result.cube, background=bg)
                parts = [_roi_finite(kk_c), _roi_finite(kkp_c)]
                vals_all = np.concatenate([p for p in parts if p.size > 0]) if any(p.size for p in parts) else np.array([])
            except Exception:
                vals_all = np.array([])
        else:
            vals_all = np.array([])

        if vals_all.size == 0:
            vals_all = _roi_finite(self.loaded.cube)
        if vals_all.size == 0:
            return

        if self._mode_log("Power Dependent"):
            pos = vals_all[vals_all > 0]
            vals = pos if pos.size else vals_all
        else:
            vals = vals_all
        vmin, vmax = np.nanpercentile(vals, [0.01, 99.99])
        vmin, vmax = float(vmin), float(vmax)
        if self._mode_log("Power Dependent"):
            vmin = max(vmin, 1e-12)
            vmax = max(vmax, vmin * 1.01)
        self.power_spins["vmin"].setValue(vmin)
        self.power_spins["vmax"].setValue(vmax)
        self._plot_mode("Power Dependent")

    def _auto_drr_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        self.drr_spins["xmin"].setValue(float(np.nanmin(cube.energy)))
        self.drr_spins["xmax"].setValue(float(np.nanmax(cube.energy)))
        self._status("State: Auto xmin/xmax set from energy axis.")
        self._plot_mode("DRR")

    def _auto_pl_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "PL" or self.loaded.cube is None:
            return
        self.pl_spins["xmin"].setValue(float(np.nanmin(self.loaded.cube.energy)))
        self.pl_spins["xmax"].setValue(float(np.nanmax(self.loaded.cube.energy)))
        self._status("State: Auto xmin/xmax set from energy axis.")
        self._plot_mode("PL")

    def _auto_cmp_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Compare" or not self.loaded.compare_cubes:
            return
        mins = [float(np.nanmin(c.energy)) for c in self.loaded.compare_cubes.values()]
        maxs = [float(np.nanmax(c.energy)) for c in self.loaded.compare_cubes.values()]
        self.cmp_spins["xmin"].setValue(min(mins))
        self.cmp_spins["xmax"].setValue(max(maxs))
        self._plot_mode("Compare")

    def _auto_power_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Power Dependent" or self.loaded.cube is None:
            return
        self.power_spins["xmin"].setValue(float(np.nanmin(self.loaded.cube.energy)))
        self.power_spins["xmax"].setValue(float(np.nanmax(self.loaded.cube.energy)))
        self._plot_mode("Power Dependent")

    def _auto_drr_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        self.drr_spins["ymin"].setValue(float(np.nanmin(cube.gate)))
        self.drr_spins["ymax"].setValue(float(np.nanmax(cube.gate)))
        self._status("State: Auto ymin/ymax set from gate axis.")
        self._plot_mode("DRR")

    def _auto_pl_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "PL" or self.loaded.cube is None:
            return
        self.pl_spins["ymin"].setValue(float(np.nanmin(self.loaded.cube.gate)))
        self.pl_spins["ymax"].setValue(float(np.nanmax(self.loaded.cube.gate)))
        self._status("State: Auto ymin/ymax set from gate axis.")
        self._plot_mode("PL")

    def _auto_cmp_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Compare" or not self.loaded.compare_cubes:
            return
        mins = [float(np.nanmin(c.gate)) for c in self.loaded.compare_cubes.values()]
        maxs = [float(np.nanmax(c.gate)) for c in self.loaded.compare_cubes.values()]
        self.cmp_spins["ymin"].setValue(min(mins))
        self.cmp_spins["ymax"].setValue(max(maxs))
        self._plot_mode("Compare")

    def _auto_power_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Power Dependent" or self.loaded.cube is None:
            return
        powers = np.asarray(self.loaded.cube.gate, float)
        positive = powers[np.isfinite(powers) & (powers > 0)]
        vals = positive if self._power_axis_log() and positive.size else powers[np.isfinite(powers)]
        if vals.size == 0:
            return
        self.power_spins["ymin"].setValue(float(np.nanmin(vals)))
        self.power_spins["ymax"].setValue(float(np.nanmax(vals)))
        self._plot_mode("Power Dependent")

    def _drr_derivative_value(self) -> int | None:
        text = self.drr_derivative_combo.currentText()
        return None if text == "None" else (1 if text == "dE" else 2)

    def _enforce_drr_sg_constraints(self, *, show_status: bool) -> int:
        poly = int(self.drr_sg_poly_spin.value())
        req_win = int(self.drr_sg_window_spin.value())
        n_energy = (
            int(np.asarray(self.loaded.cube.energy).size)
            if self.loaded and self.loaded.mode == "DRR" and self.loaded.cube is not None
            else 401
        )
        used_win = clamp_sg_window(req_win, n_energy=n_energy, polyorder=poly)
        if used_win != req_win:
            self.drr_sg_window_spin.blockSignals(True)
            self.drr_sg_window_spin.setValue(used_win)
            self.drr_sg_window_spin.blockSignals(False)
            if show_status:
                self._status(f"State: SG window clamped to {used_win} (odd, valid for order={poly}).")
        return used_win

    def _drr_cube_with_metadata(self) -> tuple[DataCube, int | None, int, int]:
        """Return the DRR cube plus the derivative parameters actually applied.

        The window is the clamped/validated value returned by the calculation,
        not the raw UI request.
        """
        if not self.loaded or self.loaded.mode != "DRR" or self.loaded.cube is None:
            raise ValueError("No DRR data loaded.")
        deriv = self._drr_derivative_value()
        poly = int(self.drr_sg_poly_spin.value())
        req_win = self._enforce_drr_sg_constraints(show_status=True)
        cube, used_win = apply_sg_derivative_energy(
            self.loaded.cube,
            derivative=deriv,
            window_length=req_win,
            polyorder=poly,
        )
        if deriv is not None and used_win != req_win:
            self._status(f"State: SG window adjusted to {used_win}.")
        return cube, deriv, used_win, poly

    def _drr_cube_for_display(self) -> DataCube:
        cube, _deriv, _used_win, _poly = self._drr_cube_with_metadata()
        return cube

    def _start_load(self, mode: str) -> None:
        if self._load_in_progress:
            self._status("State: Load already in progress.")
            return
        if not self.current_folder:
            self._show_error("Choose a folder first.")
            return
        self._invalidate_export_move_sources()
        compare_sources: Dict[str, str] = {}
        power_group_key = ""
        shg_settings: ShgSettings | None = None
        shg_fit_settings: ShgFitSettings | None = None
        shg_compare = False
        mcd_settings: McdSettings | None = None

        if mode == "PL":
            selected = self._selected(self.pl_files)
            baselines: list[str] = []
            pl_log = bool(self.pl_log_chk.isChecked())
            cmp_log = False
            drr_baseline = "Self (last frame)"
            drr_baseline_which = "last"
            y_axis_spec = self._selected_y_axis_spec("pl")
        elif mode == "DRR":
            selected = list(self.drr_selected_files)
            baselines = list(self.drr_baseline_files_manual)
            pl_log = False
            cmp_log = False
            drr_baseline = self.drr_baseline_combo.currentText()
            which_map = {
                "Last frame of each baseline file": "last",
                "First frame of each baseline file": "first",
                "Average all baseline files": "all",
            }
            drr_baseline_which = which_map.get(self.drr_baseline_combine_combo.currentText(), "last")
            y_axis_spec = self._selected_y_axis_spec("drr")
            power_group_key = ""
        elif mode == "Compare":
            selection = self._cmp_selection_from_ui()
            selected = list(selection.as_pairs().values())
            compare_sources = selection.as_pairs()
            baselines = []
            pl_log = False
            cmp_log = bool(self.cmp_log_chk.isChecked())
            drr_baseline = "Self (last frame)"
            drr_baseline_which = "last"
            y_axis_spec = self._selected_y_axis_spec("cmp")
            power_group_key = ""
        elif mode == "MCD":
            # Reprocessing rebuilds the pair list.  Remember the physical
            # linecut (field and sweep branch), rather than its temporary
            # combo-box index, so Apply does not jump back to the 0-T pair.
            self._mcd_pair_selection_to_restore = None
            if self.loaded and self.loaded.mode == "MCD" and self.loaded.mcd_result is not None:
                pair_index = self.mcd_pair_b_combo.currentData()
                if pair_index is not None:
                    pair_index = int(pair_index)
                    previous = self.loaded.mcd_result
                    if 0 <= pair_index < previous.pair_b.size:
                        self._mcd_pair_selection_to_restore = (
                            float(previous.pair_b[pair_index]),
                            str(previous.pair_labels[pair_index]),
                        )
            selected = self._selected(self.mcd_files)
            baselines = []
            pl_log = False; cmp_log = False
            drr_baseline = "Self (last frame)"; drr_baseline_which = "last"; y_axis_spec = "auto"; power_group_key = ""
            mcd_settings = self._mcd_settings_from_ui()
        elif mode == "SHG Processing":
            shg_settings = self._shg_settings_from_ui()
            shg_fit_settings = self._shg_fit_settings_from_ui()
            shg_compare = self._shg_compare_mode()
            if shg_compare:
                reference_file, sample_file = self._shg_compare_files()
                if reference_file and sample_file and reference_file == sample_file:
                    self._show_error("Select two different SHG files for twist-angle comparison.")
                    return
                selected = [name for name in (reference_file, sample_file) if name]
                if shg_settings.background_method == "external":
                    backgrounds = self._shg_compare_background_files()
                    if not all(backgrounds):
                        self._show_error("External SHG comparison requires a background file for A and B.")
                        return
                    baselines = list(backgrounds)
                else:
                    baselines = []
            else:
                source_file = self._shg_selected_file()
                background_file = (
                    self._shg_background_file()
                    if shg_settings.background_method == "external"
                    else ""
                )
                selected = [source_file] if source_file else []
                baselines = [background_file] if background_file else []
            pl_log = False
            cmp_log = False
            drr_baseline = "Self (last frame)"
            drr_baseline_which = "last"
            y_axis_spec = "auto"
        else:
            selected = self._power_candidate_files()
            compare_sources = {}
            baselines = []
            pl_log = False
            cmp_log = False
            drr_baseline = "Self (last frame)"
            drr_baseline_which = "last"
            y_axis_spec = "auto"
            power_group_key = self._power_selected_group_key()
        if mode == "PL":
            power_group_key = ""
        if mode != "Compare":
            compare_sources = {}

        options = LoadOptions(
            mode=mode,
            folder=self.current_folder,
            selected_files=selected,
            baseline_files=baselines,
            pl_log_scale=pl_log,
            drr_baseline_text=drr_baseline,
            drr_baseline_which=drr_baseline_which,
            compare_log_scale=cmp_log,
            y_axis_spec=y_axis_spec,
            compare_sources=compare_sources,
            power_group_key=power_group_key,
            shg_settings=shg_settings,
            shg_fit_settings=shg_fit_settings,
            shg_compare=shg_compare,
            mcd_settings=mcd_settings,
        )

        self._set_stage("Loading...")
        self._load_in_progress = True
        self._active_load_mode = mode
        self._active_load_succeeded = False
        worker = Worker(self._load_task, options)
        worker.signals.log.connect(self._append_log)
        worker.signals.result.connect(self._on_loaded)
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(self._on_load_finished)
        self.thread_pool.start(worker)

    def _provenance_records_for_load(self, options: LoadOptions) -> tuple[WorkingCopyRecord, ...]:
        if options.mode not in {"PL", "DRR", "Compare"}:
            return ()
        pairs: list[tuple[str, str]] = []
        if options.mode == "PL":
            pairs = [("measurement", options.selected_files[0])] if options.selected_files else []
        elif options.mode == "DRR":
            pairs = [("measurement", name) for name in options.selected_files]
            pairs.extend(("background", name) for name in options.baseline_files)
        else:
            pairs = [(f"source_{key}", name) for key, name in options.compare_sources.items()]
        records: list[WorkingCopyRecord] = []
        for role, name in pairs:
            path = Path(name)
            if not path.is_absolute():
                path = Path(options.folder) / path
            records.append(
                verify_initial_data_working_file(
                    path,
                    options.folder,
                    workflow=options.mode,
                    role=role,
                )
            )
        return tuple(records)

    def _load_task(self, options: LoadOptions, *, progress: Signal, log: Signal) -> LoadedState:
        mode = options.mode
        folder = options.folder
        if not options.selected_files:
            raise ValueError("Select required files before loading.")
        log.emit(f"Loading {mode} ...")
        provenance_records = self._provenance_records_for_load(options)

        if mode == "PL":
            cube = data_io.load_pl_cube(
                folder, options.selected_files[0], log_scale=options.pl_log_scale, y_axis=options.y_axis_spec
            )
            return LoadedState(
                mode="PL",
                folder=folder,
                primary_file=options.selected_files[0],
                selected_files=options.selected_files,
                cube=cube,
                y_axis_spec=options.y_axis_spec,
                provenance_records=provenance_records,
            )

        if mode == "DRR":
            self._reject_mixed_xlsx_selection(options.selected_files)
            if options.selected_files and data_io.is_xlsx_map_file(options.selected_files[0]):
                cube = data_io.load_drr_map_cube(
                    folder, options.selected_files[0], y_axis=options.y_axis_spec
                )
                return LoadedState(
                    mode="DRR",
                    folder=folder,
                    primary_file=options.selected_files[0],
                    selected_files=options.selected_files,
                    baseline_files=[],
                    cube=cube,
                    drr_mode_label="DR/R Map",
                    drr_derivative_label="None",
                    drr_baseline_text="None",
                    drr_baseline_which="last",
                    y_axis_spec=options.y_axis_spec,
                    provenance_records=provenance_records,
                )
            baseline = options.drr_baseline_text
            if baseline == "External":
                if not options.baseline_files:
                    raise ValueError("External DRR mode requires baseline files.")
                cube = data_io.load_drr_external_cube(
                    folder,
                    options.selected_files,
                    options.baseline_files,
                    baseline_which=options.drr_baseline_which,
                    y_axis=options.y_axis_spec,
                    derivative=None,
                )
                drr_mode_label = "DR/R External"
            else:
                cube = data_io.load_drr_self_cube(
                    folder,
                    options.selected_files,
                    use_first_frame=(baseline == "Self (first frame)"),
                    y_axis=options.y_axis_spec,
                    derivative=None,
                )
                drr_mode_label = "DR/R Self"

            return LoadedState(
                mode="DRR",
                folder=folder,
                primary_file=options.selected_files[0],
                selected_files=options.selected_files,
                baseline_files=options.baseline_files,
                cube=cube,
                drr_mode_label=drr_mode_label,
                drr_derivative_label="None",
                drr_baseline_text=baseline,
                drr_baseline_which=options.drr_baseline_which,
                y_axis_spec=options.y_axis_spec,
                provenance_records=provenance_records,
            )

        if mode == "MCD":
            source_path = Path(folder) / options.selected_files[0]
            settings = options.mcd_settings or McdSettings()
            if settings.dark_pos_file:
                settings = McdSettings(**{**settings.__dict__, "dark_pos_file": str(Path(folder) / settings.dark_pos_file)})
            if settings.dark_neg_file:
                settings = McdSettings(**{**settings.__dict__, "dark_neg_file": str(Path(folder) / settings.dark_neg_file)})
            result = process_mcd(str(source_path), settings)
            return LoadedState(
                mode="MCD", folder=folder, primary_file=options.selected_files[0],
                selected_files=options.selected_files, mcd_result=result, mcd_settings=settings,
            )

        if mode == "Power Dependent":
            result = data_io.load_power_series_cube(
                folder,
                options.selected_files,
                group_key=options.power_group_key,
                y_axis="auto",
            )
            return LoadedState(
                mode="Power Dependent",
                folder=folder,
                primary_file=(result.records[0].file_name if result.records else None),
                selected_files=list(dict.fromkeys(record.file_name for record in result.records)),
                cube=result.cube,
                power_records=result.records,
                power_groups=result.groups,
                power_group_key=result.group_key,
                y_axis_spec="auto",
            )

        if mode == "SHG Processing":
            settings = options.shg_settings or ShgSettings()
            fit_settings = options.shg_fit_settings or ShgFitSettings(enabled=False)
            if options.shg_compare and len(options.selected_files) != 2:
                raise ValueError("SHG Compare / Twist Angle requires exactly two files.")
            data = data_io.load_shg_sweep(folder, options.selected_files[0])
            background = (
                data_io.load_shg_sweep(folder, options.baseline_files[0])
                if options.baseline_files
                else None
            )
            result = process_shg_sweep(data, settings, background=background)
            data_b: ShgSweepData | None = None
            background_b: ShgSweepData | None = None
            result_b: ShgProcessResult | None = None
            fit: ShgAngularFitResult | None = None
            fit_b: ShgAngularFitResult | None = None
            twist: ShgTwistFitResult | None = None
            if options.shg_compare:
                data_b = data_io.load_shg_sweep(folder, options.selected_files[1])
                background_b = (
                    data_io.load_shg_sweep(folder, options.baseline_files[1])
                    if len(options.baseline_files) > 1
                    else None
                )
                result_b = process_shg_sweep(data_b, settings, background=background_b)
                if fit_settings.enabled:
                    try:
                        twist = fit_shg_twist_comparison(result, result_b, fit_settings)
                        fit = twist.reference_fit
                        fit_b = twist.sample_fit
                    except ValueError as exc:
                        log.emit(f"SHG twist fit unavailable: {exc}")
            elif fit_settings.enabled:
                try:
                    fit = fit_shg_angular_result(result, fit_settings)
                except ValueError as exc:
                    log.emit(f"SHG angular fit unavailable: {exc}")
            return LoadedState(
                mode="SHG Processing",
                folder=folder,
                primary_file=data.source_file,
                selected_files=[item.source_file for item in (data, data_b) if item is not None],
                baseline_files=[item.source_file for item in (background, background_b) if item is not None],
                shg_data=data,
                shg_background=background,
                shg_result=result,
                shg_data_b=data_b,
                shg_background_b=background_b,
                shg_result_b=result_b,
                shg_fit=fit,
                shg_fit_b=fit_b,
                shg_twist=twist,
                shg_compare=options.shg_compare,
                y_axis_spec="auto",
            )

        selection = data_io.CompareSelection.from_mapping(options.compare_sources)
        cubes = data_io.load_compare_cubes(folder, selection, log_scale=options.compare_log_scale, y_axis=options.y_axis_spec)
        return LoadedState(
            mode="Compare",
            folder=folder,
            selected_files=list(selection.as_pairs().values()),
            compare_cubes=cubes,
            compare_sources=selection.as_pairs(),
            y_axis_spec=options.y_axis_spec,
            provenance_records=provenance_records,
        )

    def _on_loaded(self, loaded: LoadedState) -> None:
        if loaded.mode == self._active_load_mode:
            self._active_load_succeeded = True
        self.loaded = loaded
        self.last_plotted_mode = None
        self._last_plot_params_key = None
        self._last_plot_cube = None
        if loaded.mode == "PL" and Path(loaded.primary_file or "").suffix.lower() == ".dat":
            semantic_to_choice = {
                "doping": "Doping",
                "electric_field": "Electric field",
                "gate_voltage": "Gate voltage",
                "custom": "Custom",
            }
            choice = semantic_to_choice.get(getattr(loaded.cube, "y_axis_semantic", ""), "Y")
            blocked = self.pl_yaxis_combo.blockSignals(True)
            self.pl_yaxis_combo.setCurrentText(choice)
            self.pl_yaxis_combo.blockSignals(blocked)
            self.pl_dat_yaxis_label_edit.setText(
                "" if choice != "Custom" else str(getattr(loaded.cube, "gate_label", ""))
            )
            self.pl_dat_yaxis_unit_edit.setText(str(getattr(loaded.cube, "gate_unit", "")))
            self._on_pl_dat_y_axis_changed(choice)
        if loaded.mode == "Power Dependent":
            self._power_refresh_groups()
            idx = self.power_group_combo.findData(loaded.power_group_key)
            if idx >= 0:
                self.power_group_combo.setCurrentIndex(idx)
        if loaded.mode == "MCD" and loaded.mcd_result is not None:
            blocked = self.mcd_map_combo.blockSignals(True)
            self.mcd_map_combo.clear()
            self.mcd_map_combo.addItems(list(loaded.mcd_result.maps))
            self.mcd_map_combo.setCurrentText("Combo")
            self.mcd_map_combo.blockSignals(blocked)
            pair_blocked = self.mcd_pair_b_combo.blockSignals(True)
            self.mcd_pair_b_combo.clear()
            for index, (field, direction) in enumerate(zip(loaded.mcd_result.pair_b, loaded.mcd_result.pair_labels)):
                self.mcd_pair_b_combo.addItem(f"{float(field):.6g} T  ({direction}, pair {index + 1})", int(index))
            restored_pair = getattr(self, "_mcd_pair_selection_to_restore", None)
            if restored_pair is None:
                pair_index = int(np.argmin(np.abs(loaded.mcd_result.pair_b - np.nanmedian(loaded.mcd_result.pair_b))))
            else:
                target_b, target_branch = restored_pair
                # Prefer the same field-sweep branch.  If the updated pairing
                # no longer has that branch, fall back to its nearest B pair.
                same_branch = np.asarray(loaded.mcd_result.pair_labels, dtype=str) == target_branch
                candidates = np.flatnonzero(same_branch)
                if candidates.size == 0:
                    candidates = np.arange(loaded.mcd_result.pair_b.size)
                pair_index = int(candidates[np.argmin(np.abs(loaded.mcd_result.pair_b[candidates] - target_b))])
            self.mcd_pair_b_combo.setCurrentIndex(pair_index)
            self._mcd_pair_selection_to_restore = None
            self.mcd_pair_b_combo.blockSignals(pair_blocked)
            reference_mode = str(loaded.mcd_result.summary["reference_mode"])
            reference_b = float(loaded.mcd_result.summary["reference_b_t"])
            if reference_mode == "nearest":
                reference_text = f"nearest reference pair: Bpair = {reference_b:.4g} T"
            else:
                reference_text = f"median reference: {loaded.mcd_result.summary['zero_pairs']} near-zero pairs (median Bpair = {reference_b:.4g} T)"
            self.mcd_source_summary.setText(
                f"{loaded.mcd_result.summary['pairs']} angle pairs; sigma+ {loaded.mcd_result.pos_angle:g} deg; "
                f"sigma- {loaded.mcd_result.neg_angle:g} deg; {reference_text}."
            )
            self._update_mcd_background_preview()
            result = loaded.mcd_result
            warning_delta_b = 0.5 * float(result.summary["max_delta_b_t"])
            spectral_mode = str(result.summary.get("correction_mode", "global")) == "pair_spectral"
            if spectral_mode:
                diagnostic_lines = [
                    "#    Bpair       B+       B-       dB  gap gain@Ec slope/eV curve/eV2 gain min:max  RMS before->after  flags"
                ]
            else:
                diagnostic_lines = ["#    Bpair       B+       B-       dB  gap  scale    offset  RMS before->after  flags"]
            for index in range(result.pair_b.size):
                rms = float(result.pair_background_rms[index])
                rms_before = float(result.pair_background_rms_before[index])
                flags = []
                if abs(float(result.pair_delta_b[index])) > warning_delta_b:
                    flags.append("large dB")
                if np.isfinite(rms) and rms > 0.02:
                    flags.append("high residual")
                if result.pair_interpolated_pos[index] or result.pair_interpolated_neg[index]:
                    flags.append("B-aligned")
                if spectral_mode:
                    if not np.isfinite(rms):
                        flags.append("spectral fit unavailable")
                    elif np.isfinite(rms_before) and rms >= 0.98 * rms_before:
                        flags.append("little fit improvement")
                    diagnostic_lines.append(
                        f"{index + 1:>2} {result.pair_b[index]:>8.4f} {result.pair_b_pos[index]:>8.4f} "
                        f"{result.pair_b_neg[index]:>8.4f} {result.pair_delta_b[index]:>+8.4f} "
                        f"{result.pair_sequence_gap[index]:>3} {result.pair_scale[index]:>7.4f} "
                        f"{result.pair_spectral_slope[index]:>+8.3g} {result.pair_spectral_curvature[index]:>+9.3g} "
                        f"{result.pair_correction_min[index]:>5.3f}:{result.pair_correction_max[index]:<5.3f} "
                        f"{rms_before:>7.3g}->{rms:<7.3g}  {', '.join(flags)}"
                    )
                else:
                    diagnostic_lines.append(
                        f"{index + 1:>2} {result.pair_b[index]:>8.4f} {result.pair_b_pos[index]:>8.4f} "
                        f"{result.pair_b_neg[index]:>8.4f} {result.pair_delta_b[index]:>+8.4f} "
                        f"{result.pair_sequence_gap[index]:>3} {result.pair_scale[index]:>6.3f} "
                        f"{result.pair_offset[index]:>9.2g} {rms_before:>7.3g}->{rms:<7.3g}  {', '.join(flags)}"
                    )
            self.mcd_diagnostics_text.setPlainText("\n".join(diagnostic_lines))
        if loaded.mode == "SHG Processing" and loaded.shg_result is not None:
            self._shg_refresh_sources()
            tab_blocked = self.shg_workflow_tabs.blockSignals(True)
            self.shg_workflow_tabs.setCurrentIndex(1 if loaded.shg_compare else 0)
            self.shg_workflow_tabs.blockSignals(tab_blocked)
            self.shg_fit_branch_spin.setEnabled(loaded.shg_compare and self.shg_fit_enable_chk.isChecked())
            if loaded.shg_compare:
                for combo, file_name in zip(
                    (self.shg_compare_reference_combo, self.shg_compare_sample_combo),
                    loaded.selected_files,
                ):
                    blocked = combo.blockSignals(True)
                    index = combo.findData(file_name)
                    if index >= 0:
                        combo.setCurrentIndex(index)
                    combo.blockSignals(blocked)
            else:
                blocked = self.shg_files.blockSignals(True)
                self._restore_list_selection(self.shg_files, [loaded.primary_file or ""])
                self.shg_files.blockSignals(blocked)
            finite = np.flatnonzero(np.isfinite(loaded.shg_result.measured_angle_deg))
            if finite.size:
                included = finite[loaded.shg_result.included[finite]]
                selected = int(included[0] if included.size else finite[0])
                blocked = self.shg_angle_cursor_spin.blockSignals(True)
                self.shg_angle_cursor_spin.setValue(float(loaded.shg_result.measured_angle_deg[selected]))
                self.shg_angle_cursor_spin.blockSignals(blocked)
            self._shg_update_summary()
        self._apply_auto_limits_for_loaded()
        self._set_stage("Loaded")
        self._update_action_states()
        self._status(f"Loaded {loaded.mode}.")
        self._plot_mode(loaded.mode, auto=True)

    def _on_load_finished(self) -> None:
        finished_mode = self._active_load_mode
        succeeded = self._active_load_succeeded
        self._load_in_progress = False
        self._active_load_mode = None
        self._active_load_succeeded = False
        self._status_progress.setVisible(False)
        if finished_mode == "MCD":
            if self._mcd_reapply_pending:
                self._mcd_reapply_pending = False
                self.mcd_apply_correction_btn.setText("Pending update...")
                self._mcd_auto_apply_timer.start(0)
            elif self._mcd_auto_apply_timer.isActive():
                self.mcd_apply_correction_btn.setText("Pending update...")
            elif succeeded:
                self.mcd_apply_correction_btn.setText("Up to date")
            else:
                self.mcd_apply_correction_btn.setText("Recalculate now")

    def _split_prefix_mode(self, prefix: str) -> str:
        return {
            "pl": "PL",
            "drr": "DRR",
            "cmp": "Compare",
            "power": "Power Dependent",
        }[prefix]

    def _on_split_scale_toggled(self, prefix: str, checked: bool) -> None:
        panel: QWidget = getattr(self, f"{prefix}_split_scale_panel")
        panel.setVisible(bool(checked))
        spins = self._mode_spins(self._split_prefix_mode(prefix))
        checks: Dict[str, QCheckBox] = getattr(self, f"{prefix}_fix_checks")
        spins["vmin"].setEnabled(not checked)
        spins["vmax"].setEnabled(not checked)
        checks["vmin"].setEnabled(not checked)
        checks["vmax"].setEnabled(not checked)

        split_spins: Dict[str, QDoubleSpinBox] = getattr(self, f"{prefix}_split_spins")
        xlo, xhi = sorted((float(spins["xmin"].value()), float(spins["xmax"].value())))
        split_fix_checks: Dict[str, QCheckBox] = getattr(
            self, f"{prefix}_split_fix_checks"
        )
        if checked:
            values: Dict[str, float] = {}
            if not xlo < float(split_spins["x0"].value()) < xhi:
                values["x0"] = 0.5 * (xlo + xhi)
            for side in ("left", "right"):
                vmin_key = f"{side}_vmin"
                vmax_key = f"{side}_vmax"
                if split_spins[vmax_key].value() <= split_spins[vmin_key].value():
                    if not split_fix_checks[vmin_key].isChecked():
                        values[vmin_key] = float(spins["vmin"].value())
                    if not split_fix_checks[vmax_key].isChecked():
                        values[vmax_key] = float(spins["vmax"].value())
            for key, value in values.items():
                spin = split_spins[key]
                blocked = spin.blockSignals(True)
                try:
                    spin.setValue(value)
                finally:
                    spin.blockSignals(blocked)
            self._refresh_automatic_ranges(
                self._split_prefix_mode(prefix),
                refresh_split=True,
                center_split=True,
            )
        self._on_split_scale_param_changed(prefix)
        self._update_action_states()

    def _on_split_scale_param_changed(self, prefix: str) -> None:
        toggle: QCheckBox = getattr(self, f"{prefix}_split_scale_chk")
        if not toggle.isChecked():
            return
        self._refresh_automatic_ranges(
            self._split_prefix_mode(prefix), refresh_split=True
        )
        if prefix == "pl":
            self._on_pl_plot_param_changed()
        elif prefix == "drr":
            self._on_drr_plot_param_changed()
        elif prefix == "cmp":
            self._on_cmp_plot_param_changed()
        else:
            self._on_power_plot_param_changed()

    def _split_scale_for_mode(self, mode: str) -> SplitColorScale | None:
        if mode == "MCD":
            return None
        # VP views use a fixed diverging [-1, 1] scale, so a remembered
        # intensity split must not be validated or applied while they are active.
        if mode == "Compare" and self._cmp_is_vp_view():
            return None
        if mode == "Power Dependent" and self._power_view() == "VP":
            return None
        prefix = "pl" if mode == "PL" else "drr" if mode == "DRR" else "power" if mode == "Power Dependent" else "cmp"
        toggle: QCheckBox = getattr(self, f"{prefix}_split_scale_chk")
        if not toggle.isChecked():
            return None
        spins = self._mode_spins(mode)
        split_spins: Dict[str, QDoubleSpinBox] = getattr(self, f"{prefix}_split_spins")
        xmin, xmax = sorted((float(spins["xmin"].value()), float(spins["xmax"].value())))
        x0 = float(split_spins["x0"].value())
        left_vmin = float(split_spins["left_vmin"].value())
        left_vmax = float(split_spins["left_vmax"].value())
        right_vmin = float(split_spins["right_vmin"].value())
        right_vmax = float(split_spins["right_vmax"].value())
        if not xmin < x0 < xmax:
            raise ValueError("Split color scale requires xmin < x0 < xmax.")
        if left_vmax <= left_vmin or right_vmax <= right_vmin:
            raise ValueError("Both split color regions require vmin < vmax.")
        if self._mode_log(mode) and (left_vmin <= 0.0 or right_vmin <= 0.0):
            raise ValueError("Log split color scale requires positive vmin values in both regions.")
        return SplitColorScale(
            split_x=x0,
            left_vmin=left_vmin,
            left_vmax=left_vmax,
            right_vmin=right_vmin,
            right_vmax=right_vmax,
            show_boundary=bool(getattr(self, f"{prefix}_split_boundary_chk").isChecked()),
        )

    def _split_scale_key(self, prefix: str) -> tuple[object, ...]:
        enabled = bool(getattr(self, f"{prefix}_split_scale_chk").isChecked())
        if not enabled:
            return (False,)
        spins: Dict[str, QDoubleSpinBox] = getattr(self, f"{prefix}_split_spins")
        return (
            True,
            float(spins["x0"].value()),
            float(spins["left_vmin"].value()),
            float(spins["left_vmax"].value()),
            float(spins["right_vmin"].value()),
            float(spins["right_vmax"].value()),
            bool(getattr(self, f"{prefix}_split_boundary_chk").isChecked()),
        )

    def _split_auto_cubes(self, mode: str) -> list[DataCube]:
        if not self.loaded or self.loaded.mode != mode:
            return []
        if mode == "PL" and self.loaded.cube is not None:
            return [self.loaded.cube]
        if mode == "DRR" and self.loaded.cube is not None:
            return [self._drr_cube_for_display()]
        if mode == "Compare" and self.loaded.compare_cubes:
            if self._cmp_is_vp_view():
                return []
            raw = {
                key: self.loaded.compare_cubes[key]
                for key in self._cmp_visible_channels()
                if key in self.loaded.compare_cubes
            }
            return list(
                self._cmp_corrected_cubes(
                    raw,
                    self._cmp_source_mapping(),
                    background=self._cmp_background_value(raw),
                ).values()
            )
        if mode == "Power Dependent" and self.loaded.cube is not None:
            if self._power_view() == "VP":
                return []
            if self._power_has_distinct_role_groups():
                kk_result, kkp_result, _kk_key, _kkp_key = self._power_role_payload()
                background = self._power_background_value([kk_result.cube, kkp_result.cube])
                return [
                    self._power_corrected_cube(kk_result.cube, background=background),
                    self._power_corrected_cube(kkp_result.cube, background=background),
                ]
            return [self._power_corrected_cube(self.loaded.cube)]
        return []

    @staticmethod
    def _set_spin_value_silent(spin: QDoubleSpinBox, value: float) -> None:
        blocked = spin.blockSignals(True)
        try:
            spin.setValue(float(value))
        finally:
            spin.blockSignals(blocked)

    def _automatic_cubes_for_mode(self, mode: str) -> list[DataCube]:
        if mode == "MCD" and self.loaded and self.loaded.mode == "MCD" and self.loaded.mcd_result is not None:
            return [self.loaded.mcd_result.cube(self.mcd_map_combo.currentText())]
        return self._split_auto_cubes(mode)

    def _color_bounds_for_cubes(
        self,
        mode: str,
        cubes: Sequence[DataCube],
        *,
        side: str | None = None,
        split_x: float | None = None,
    ) -> tuple[float, float] | None:
        """Calculate robust limits from the data currently visible to the user."""
        spins = self._mode_spins(mode)
        xmin, xmax = sorted((float(spins["xmin"].value()), float(spins["xmax"].value())))
        ymin, ymax = sorted((float(spins["ymin"].value()), float(spins["ymax"].value())))
        values: list[np.ndarray] = []
        for cube in cubes:
            x = np.asarray(cube.energy, float).ravel()
            y = np.asarray(cube.gate, float).ravel()
            z = np.asarray(cube.Z, float)
            xmask = (x >= xmin) & (x <= xmax)
            if side is not None:
                if split_x is None:
                    continue
                try:
                    split_index, _boundary = resolve_split_boundary(x, split_x)
                except ValueError:
                    continue
                columns = np.arange(x.size)
                xmask &= columns < split_index if side == "left" else columns >= split_index
            ymask = (y >= ymin) & (y <= ymax)
            if not np.any(xmask) or not np.any(ymask):
                continue
            finite = z[np.ix_(ymask, xmask)]
            finite = finite[np.isfinite(finite)]
            if self._mode_log(mode):
                finite = finite[finite > 0.0]
            if finite.size:
                values.append(finite)
        if not values:
            return None
        combined = np.concatenate(values)
        vmin, vmax = (float(value) for value in np.nanpercentile(combined, [0.01, 99.99]))
        center_zero = (
            mode == "DRR" and bool(self.drr_center_zero_chk.isChecked())
        ) or (
            mode == "MCD" and bool(self.mcd_center_zero_chk.isChecked())
        )
        if center_zero and not self._mode_log(mode):
            bound = max(abs(vmin), abs(vmax), 1e-12)
            return -bound, bound
        if vmax <= vmin:
            pad = max(1e-12, abs(vmin) * 0.01)
            vmin -= pad
            vmax += pad
        if self._mode_log(mode):
            vmin = max(vmin, 1e-12)
            vmax = max(vmax, vmin * 1.01)
        return vmin, vmax

    def _refresh_automatic_ranges(
        self,
        mode: str,
        *,
        refresh_axes: bool = False,
        refresh_split: bool = True,
        center_split: bool = False,
    ) -> bool:
        """Refresh all unlocked ranges as one UI transaction.

        Numeric fields remain editable, but an unchecked Fix box means the
        value follows the displayed data whenever that data or its ROI changes.
        """
        if self._automatic_range_update:
            return False
        cubes = self._automatic_cubes_for_mode(mode)
        if not cubes:
            return False
        self._automatic_range_update = True
        changed = False
        try:
            spins = self._mode_spins(mode)
            if refresh_axes:
                axis_values = {
                    "xmin": min(float(np.nanmin(cube.energy)) for cube in cubes),
                    "xmax": max(float(np.nanmax(cube.energy)) for cube in cubes),
                    "ymin": min(float(np.nanmin(cube.gate)) for cube in cubes),
                    "ymax": max(float(np.nanmax(cube.gate)) for cube in cubes),
                }
                for key, value in axis_values.items():
                    if not self._mode_fix_value(mode, key):
                        self._set_spin_value_silent(spins[key], value)
                        changed = True

            bounds = self._color_bounds_for_cubes(mode, cubes)
            if bounds is not None:
                for key, value in zip(("vmin", "vmax"), bounds):
                    if not self._mode_fix_value(mode, key):
                        self._set_spin_value_silent(spins[key], value)
                        changed = True

            if mode == "MCD" or not refresh_split:
                return changed
            prefix = "pl" if mode == "PL" else "drr" if mode == "DRR" else "power" if mode == "Power Dependent" else "cmp"
            if not bool(getattr(self, f"{prefix}_split_scale_chk").isChecked()):
                return changed
            split_spins: Dict[str, QDoubleSpinBox] = getattr(self, f"{prefix}_split_spins")
            split_fixes: Dict[str, QCheckBox] = getattr(self, f"{prefix}_split_fix_checks")
            xmin, xmax = sorted((float(spins["xmin"].value()), float(spins["xmax"].value())))
            x0 = float(split_spins["x0"].value())
            if (center_split and not split_fixes["x0"].isChecked()) or not xmin < x0 < xmax:
                requested = 0.5 * (xmin + xmax)
                try:
                    _index, x0 = resolve_split_boundary(np.asarray(cubes[0].energy, float), requested)
                except ValueError:
                    x0 = requested
                self._set_spin_value_silent(split_spins["x0"], x0)
                changed = True

            for side in ("left", "right"):
                side_bounds = self._color_bounds_for_cubes(mode, cubes, side=side, split_x=x0)
                if side_bounds is None:
                    continue
                vmin_key, vmax_key = f"{side}_vmin", f"{side}_vmax"
                candidate_min = float(split_spins[vmin_key].value()) if split_fixes[vmin_key].isChecked() else side_bounds[0]
                candidate_max = float(split_spins[vmax_key].value()) if split_fixes[vmax_key].isChecked() else side_bounds[1]
                if candidate_max <= candidate_min or (self._mode_log(mode) and candidate_min <= 0.0):
                    self._status(f"Automatic {side} color range conflicts with a fixed limit.")
                    continue
                for key, value in ((vmin_key, candidate_min), (vmax_key, candidate_max)):
                    if not split_fixes[key].isChecked():
                        self._set_spin_value_silent(split_spins[key], value)
                        changed = True
            return changed
        finally:
            self._automatic_range_update = False

    def _auto_split_vrange(self, prefix: str, side: str) -> None:
        mode = self._split_prefix_mode(prefix)
        cubes = self._split_auto_cubes(mode)
        if not cubes:
            self._status("Split auto scale is available after loading an intensity heatmap.")
            return
        spins = self._mode_spins(mode)
        split_spins: Dict[str, QDoubleSpinBox] = getattr(self, f"{prefix}_split_spins")
        xmin, xmax = sorted((float(spins["xmin"].value()), float(spins["xmax"].value())))
        ymin, ymax = sorted((float(spins["ymin"].value()), float(spins["ymax"].value())))
        x0 = float(split_spins["x0"].value())
        if not xmin < x0 < xmax:
            self._status("Set x0 strictly between xmin and xmax before auto scaling.")
            return

        values: list[np.ndarray] = []
        for cube in cubes:
            x = np.asarray(cube.energy, float).ravel()
            y = np.asarray(cube.gate, float).ravel()
            z = np.asarray(cube.Z, float)
            try:
                split_index, _applied_boundary = resolve_split_boundary(x, x0)
            except ValueError:
                continue
            column_index = np.arange(x.size)
            side_mask = column_index < split_index if side == "left" else column_index >= split_index
            xmask = side_mask & (x >= xmin) & (x <= xmax)
            ymask = (y >= ymin) & (y <= ymax)
            if not np.any(xmask) or not np.any(ymask):
                continue
            region = z[np.ix_(ymask, xmask)]
            finite = region[np.isfinite(region)]
            if self._mode_log(mode):
                finite = finite[finite > 0.0]
            if finite.size:
                values.append(finite)
        if not values:
            self._status(f"Auto {side} scale skipped: no finite values in that x region.")
            return
        combined = np.concatenate(values)
        vmin, vmax = (float(v) for v in np.nanpercentile(combined, [0.01, 99.99]))
        if vmax <= vmin:
            pad = max(1e-12, abs(vmin) * 0.01, 1e-12)
            vmin -= pad
            vmax += pad
            if self._mode_log(mode):
                vmin = max(vmin, 1e-12)
        split_fix_checks: Dict[str, QCheckBox] = getattr(
            self, f"{prefix}_split_fix_checks"
        )
        vmin_key = f"{side}_vmin"
        vmax_key = f"{side}_vmax"
        candidate_vmin = (
            float(split_spins[vmin_key].value())
            if split_fix_checks[vmin_key].isChecked()
            else vmin
        )
        candidate_vmax = (
            float(split_spins[vmax_key].value())
            if split_fix_checks[vmax_key].isChecked()
            else vmax
        )
        if candidate_vmax <= candidate_vmin:
            self._status(
                f"Auto {side} scale conflicts with a fixed limit; "
                "adjust or unfix that value first."
            )
            return
        if self._mode_log(mode) and candidate_vmin <= 0.0:
            self._status(
                f"Auto {side} scale conflicts with a non-positive fixed vmin in log mode."
            )
            return
        updated: list[str] = []
        for key, value in ((vmin_key, candidate_vmin), (vmax_key, candidate_vmax)):
            if split_fix_checks[key].isChecked():
                continue
            blocked = split_spins[key].blockSignals(True)
            try:
                split_spins[key].setValue(value)
            finally:
                split_spins[key].blockSignals(blocked)
            updated.append("vmin" if key.endswith("vmin") else "vmax")
        if not updated:
            self._status(f"Auto {side} scale skipped: vmin and vmax are fixed.")
            return
        locked = [name for name in ("vmin", "vmax") if name not in updated]
        detail = f"; kept fixed {', '.join(locked)}" if locked else ""
        self._status(
            f"Auto {side} color scale = {candidate_vmin:.4g}, {candidate_vmax:.4g}{detail}."
        )
        self._on_split_scale_param_changed(prefix)

    def _mode_spins(self, mode: str) -> Dict[str, QDoubleSpinBox]:
        if mode == "PL":
            return self.pl_spins
        if mode == "DRR":
            return self.drr_spins
        if mode == "Power Dependent":
            return self.power_spins
        if mode == "MCD":
            return self.mcd_spins
        return self.cmp_spins

    def _mode_cmap(self, mode: str) -> QComboBox:
        return self.pl_cmap if mode == "PL" else self.drr_cmap if mode == "DRR" else self.power_cmap if mode == "Power Dependent" else self.mcd_cmap if mode == "MCD" else self.cmp_cmap

    def _resolved_cmap(self, combo: QComboBox) -> str:
        value = combo.currentData()
        if value is None or value == "":
            value = combo.currentText()
        default = combo.property("default_cmap") or ""
        return resolve_cmap(str(value), str(default))

    def _mode_log(self, mode: str) -> bool:
        return False if mode == "MCD" else bool(self.pl_log_chk.isChecked()) if mode == "PL" else bool(self.drr_log_chk.isChecked()) if mode == "DRR" else bool(self.power_log_chk.isChecked()) if mode == "Power Dependent" else bool(self.cmp_log_chk.isChecked())

    def _mode_clip(self, mode: str) -> bool:
        return False if mode == "MCD" else bool(self.pl_clip_chk.isChecked()) if mode == "PL" else bool(self.drr_clip_chk.isChecked()) if mode == "DRR" else bool(self.power_clip_chk.isChecked()) if mode == "Power Dependent" else bool(self.cmp_clip_chk.isChecked())

    def _mode_fix_value(self, mode: str, key: str) -> bool:
        if mode == "PL":
            checks = self.pl_fix_checks
        elif mode == "DRR":
            checks = self.drr_fix_checks
        elif mode == "Power Dependent":
            checks = self.power_fix_checks
        elif mode == "MCD":
            checks = self.mcd_fix_checks
        else:
            checks = self.cmp_fix_checks
        chk = checks.get(key)
        return bool(chk.isChecked()) if chk is not None else False

    def _mode_y_axis_prefix(self, mode: str) -> str:
        return "pl" if mode == "PL" else "drr" if mode == "DRR" else "cmp"

    def _current_y_axis_spec_for_mode(self, mode: str) -> str:
        if mode in {"Power Dependent", "SHG Processing", "MCD"}:
            return "auto"
        return self._selected_y_axis_spec(self._mode_y_axis_prefix(mode))

    def _ensure_loaded_matches_ui_params(self, mode: str) -> bool:
        if not self.loaded or self.loaded.mode != mode:
            return False
        if mode == "SHG Processing":
            if self.loaded.shg_data is None:
                raise ValueError("No SHG sweep table is loaded.")
            if self.loaded.shg_compare != self._shg_compare_mode():
                raise ValueError("Press Load after changing the SHG Single/Compare workflow.")
            settings = self._shg_settings_from_ui()
            fit_settings = self._shg_fit_settings_from_ui()
            self.loaded.shg_result = process_shg_sweep(
                self.loaded.shg_data,
                settings,
                background=self.loaded.shg_background,
            )
        if mode == "MCD":
            return (
                mode, self.mcd_map_combo.currentText(), float(self.mcd_spins["vmin"].value()), float(self.mcd_spins["vmax"].value()),
                float(self.mcd_spins["xmin"].value()), float(self.mcd_spins["xmax"].value()), float(self.mcd_spins["ymin"].value()),
                float(self.mcd_spins["ymax"].value()), self._resolved_cmap(self.mcd_cmap),
                bool(self.mcd_center_zero_chk.isChecked()), int(self.mcd_pair_b_combo.currentData() or 0),
                float(self.mcd_window_center_spin.value()), float(self.mcd_window_width_spin.value()), self.mcd_window_metric_combo.currentText(),
                bool(self.mcd_show_raw_chk.isChecked()), bool(self.mcd_show_signed_mean_chk.isChecked()),
                bool(self.mcd_show_absolute_mean_chk.isChecked()), bool(self.mcd_show_unsigned_absolute_mean_chk.isChecked()),
                bool(self.mcd_show_integral_chk.isChecked()),
                bool(self.mcd_fit_zero_chk.isChecked()), float(self.mcd_fit_b_window_spin.value()),
            )
            self.loaded.shg_fit = None
            self.loaded.shg_fit_b = None
            self.loaded.shg_twist = None
            if self.loaded.shg_compare:
                if self.loaded.shg_data_b is None:
                    raise ValueError("The SHG comparison sample file is not loaded.")
                self.loaded.shg_result_b = process_shg_sweep(
                    self.loaded.shg_data_b,
                    settings,
                    background=self.loaded.shg_background_b,
                )
                if fit_settings.enabled:
                    try:
                        self.loaded.shg_twist = fit_shg_twist_comparison(
                            self.loaded.shg_result,
                            self.loaded.shg_result_b,
                            fit_settings,
                        )
                        self.loaded.shg_fit = self.loaded.shg_twist.reference_fit
                        self.loaded.shg_fit_b = self.loaded.shg_twist.sample_fit
                    except ValueError as exc:
                        self._status(f"SHG twist fit unavailable: {exc}")
            elif fit_settings.enabled:
                try:
                    self.loaded.shg_fit = fit_shg_angular_result(self.loaded.shg_result, fit_settings)
                except ValueError as exc:
                    self._status(f"SHG angular fit unavailable: {exc}")
            self._shg_update_summary()
            return True
        current_spec = self._current_y_axis_spec_for_mode(mode)
        if mode == "DRR" and current_spec == getattr(self.loaded, "y_axis_spec", "auto"):
            return self._ensure_loaded_matches_drr_params()
        if mode == "Power Dependent":
            desired_key = self._power_selected_group_key()
            desired_files = self._power_candidate_files()
            if desired_key == self.loaded.power_group_key and list(self.loaded.selected_files):
                return False
            result = data_io.load_power_series_cube(
                self.current_folder,
                desired_files,
                group_key=desired_key,
                y_axis="auto",
            )
            self.loaded = LoadedState(
                mode="Power Dependent",
                folder=self.current_folder,
                primary_file=(result.records[0].file_name if result.records else None),
                selected_files=list(dict.fromkeys(record.file_name for record in result.records)),
                cube=result.cube,
                power_records=result.records,
                power_groups=result.groups,
                power_group_key=result.group_key,
                y_axis_spec="auto",
            )
            self._last_plot_cube = None
            self._last_plot_params_key = None
            self._apply_auto_limits_for_loaded()
            return True
        if mode == "Compare":
            selection = self._cmp_selection_from_ui()
            desired_sources = selection.as_pairs()
            if (
                current_spec == getattr(self.loaded, "y_axis_spec", "auto")
                and desired_sources == (self.loaded.compare_sources or {})
            ):
                return False
        elif current_spec == getattr(self.loaded, "y_axis_spec", "auto"):
            return False

        if mode == "PL":
            if not self.loaded.primary_file:
                raise ValueError("No PL file loaded.")
            cube = data_io.load_pl_cube(
                self.current_folder,
                self.loaded.primary_file,
                log_scale=bool(self.pl_log_chk.isChecked()),
                y_axis=current_spec,
            )
            self.loaded = LoadedState(
                mode="PL",
                folder=self.current_folder,
                primary_file=self.loaded.primary_file,
                selected_files=list(self.loaded.selected_files),
                cube=cube,
                y_axis_spec=current_spec,
            )
        elif mode == "Compare":
            selection = self._cmp_selection_from_ui()
            cubes = data_io.load_compare_cubes(
                self.current_folder,
                selection,
                log_scale=bool(self.cmp_log_chk.isChecked()),
                y_axis=current_spec,
            )
            self.loaded = LoadedState(
                mode="Compare",
                folder=self.current_folder,
                selected_files=list(selection.as_pairs().values()),
                compare_cubes=cubes,
                compare_sources=selection.as_pairs(),
                y_axis_spec=current_spec,
            )
        else:
            return self._ensure_loaded_matches_drr_params()

        self._last_plot_cube = None
        self._last_plot_params_key = None
        self._apply_auto_limits_for_loaded()
        return True

    def _apply_auto_limits_for_loaded(self) -> None:
        if not self.loaded:
            return
        mode = self.loaded.mode
        if mode == "SHG Processing":
            self._shg_update_summary()
            return
        if mode == "PL" and self.loaded.cube is not None:
            cube = self.loaded.cube
        elif mode == "DRR" and self.loaded.cube is not None:
            cube = self._drr_cube_for_display()
        elif mode == "Compare" and self.loaded.compare_cubes:
            background = self._cmp_background_value(self.loaded.compare_cubes)
            corrected = self._cmp_corrected_cubes(self.loaded.compare_cubes, background=background)
            cube = next(iter(corrected.values()))
        elif mode == "Power Dependent" and self.loaded.cube is not None:
            if self._power_view() == "VP":
                try:
                    _kk_cube, _kkp_cube, vp_cube, *_rest = self._power_vp_payload()
                    cube = vp_cube
                except Exception:
                    cube = self._power_corrected_cube(self.loaded.cube)
            else:
                cube = self._power_corrected_cube(self.loaded.cube)
        elif mode == "MCD" and self.loaded.mcd_result is not None:
            cube = self.loaded.mcd_result.cube(self.mcd_map_combo.currentText())
        else:
            return

        limits = compute_auto_limits(cube, log_scale=self._mode_log(mode))
        spins = self._mode_spins(mode)
        if not self._mode_fix_value(mode, "vmin"):
            self._set_spin_value_silent(spins["vmin"], limits.vmin)
        if not self._mode_fix_value(mode, "vmax"):
            self._set_spin_value_silent(spins["vmax"], limits.vmax)
        if not self._mode_fix_value(mode, "xmin"):
            self._set_spin_value_silent(spins["xmin"], limits.xmin)
        if not self._mode_fix_value(mode, "xmax"):
            self._set_spin_value_silent(spins["xmax"], limits.xmax)
        if not self._mode_fix_value(mode, "ymin"):
            self._set_spin_value_silent(spins["ymin"], limits.ymin)
        if not self._mode_fix_value(mode, "ymax"):
            self._set_spin_value_silent(spins["ymax"], limits.ymax)
        if mode == "Power Dependent" and self._power_axis_log():
            positive = np.asarray(cube.gate, float)
            positive = positive[np.isfinite(positive) & (positive > 0)]
            if positive.size:
                if not self._mode_fix_value(mode, "ymin"):
                    self._set_spin_value_silent(spins["ymin"], float(np.nanmin(positive)))
                if not self._mode_fix_value(mode, "ymax"):
                    self._set_spin_value_silent(spins["ymax"], float(np.nanmax(positive)))
        if mode == "MCD":
            # MCD cursor selection is tied to the selected raw measurement
            # pair.  Do not overwrite it with the binned-colormap median.
            self._refresh_automatic_ranges(mode, refresh_split=False)
            return
        prefix = "pl" if mode == "PL" else "drr" if mode == "DRR" else "power" if mode == "Power Dependent" else "cmp"
        if bool(getattr(self, f"{prefix}_split_scale_chk").isChecked()):
            split_spins: Dict[str, QDoubleSpinBox] = getattr(self, f"{prefix}_split_spins")
            split_fix_checks: Dict[str, QCheckBox] = getattr(
                self, f"{prefix}_split_fix_checks"
            )
            xmin, xmax = sorted((float(spins["xmin"].value()), float(spins["xmax"].value())))
            if not xmin < float(split_spins["x0"].value()) < xmax:
                blocked = split_spins["x0"].blockSignals(True)
                try:
                    split_spins["x0"].setValue(0.5 * (xmin + xmax))
                finally:
                    split_spins["x0"].blockSignals(blocked)
            # Initialize only invalid regional pairs. Valid manual/fixed values
            # survive file changes and x-range updates unchanged.
            for side in ("left", "right"):
                vmin_key = f"{side}_vmin"
                vmax_key = f"{side}_vmax"
                if split_spins[vmax_key].value() > split_spins[vmin_key].value():
                    continue
                for key, value in (
                    (vmin_key, float(spins["vmin"].value())),
                    (vmax_key, float(spins["vmax"].value())),
                ):
                    if split_fix_checks[key].isChecked():
                        continue
                    blocked = split_spins[key].blockSignals(True)
                    try:
                        split_spins[key].setValue(value)
                    finally:
                        split_spins[key].blockSignals(blocked)
        self._refresh_automatic_ranges(
            mode,
            refresh_split=True,
            center_split=True,
        )
        self._set_spin_value_silent(spins["gate"], float(np.nanmedian(cube.gate)))

    def _make_params(self, mode: str, cube: DataCube) -> HeatmapParams:
        spins = self._mode_spins(mode)
        y_label = cube.gate_label
        if getattr(cube, "gate_unit", ""):
            y_label = f"{y_label} ({cube.gate_unit})"
        return HeatmapParams(
            title=cube.title,
            xlabel="Photon Energy (eV)",
            ylabel=y_label,
            cbar_label=cube.cbar_label,
            vmin=float(spins["vmin"].value()),
            vmax=float(spins["vmax"].value()),
            xlim=(float(spins["xmin"].value()), float(spins["xmax"].value())),
            ylim=(float(spins["ymin"].value()), float(spins["ymax"].value())),
            cmap=self._resolved_cmap(self._mode_cmap(mode)),
            log_scale=self._mode_log(mode),
            y_axis_log=(mode == "Power Dependent" and self._power_axis_log()),
            center_zero=(bool(self.mcd_center_zero_chk.isChecked()) if mode == "MCD" else mode == "DRR" and bool(self.drr_center_zero_chk.isChecked())),
            clip_outliers=self._mode_clip(mode),
            split_scale=self._split_scale_for_mode(mode),
        )

    def _auto_scale_spectrum_y(self, ax, x: np.ndarray, y: np.ndarray, xlim: tuple[float, float]) -> None:
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        lo, hi = sorted((float(xlim[0]), float(xlim[1])))
        mask = (x >= lo) & (x <= hi) & np.isfinite(y)
        region = y[mask] if np.any(mask) else y[np.isfinite(y)]
        if region.size == 0:
            return
        ymin = float(np.nanmin(region))
        ymax = float(np.nanmax(region))
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return
        if ymin == ymax:
            pad = max(1e-12, abs(ymin) * 0.05, 1.0)
        else:
            pad = max(1e-12, (ymax - ymin) * 0.08)
        ax.set_ylim(ymin - pad, ymax + pad)

    def _safe_spectrum_xlim(self, x: np.ndarray, xlim: tuple[float, float]) -> tuple[float, float]:
        x = np.asarray(x, float).ravel()
        lo, hi = float(xlim[0]), float(xlim[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = float(np.nanmin(x))
            hi = float(np.nanmax(x))
        if lo == hi:
            pad = max(1e-9, abs(lo) * 1e-6, (float(np.nanmax(x)) - float(np.nanmin(x))) * 1e-3)
            lo -= pad
            hi += pad
        return (lo, hi)

    def _plot_spectrum_with_roi(
        self,
        ax,
        cube: DataCube,
        gate_value: float,
        *,
        ylabel: str,
        xlim: tuple[float, float],
    ) -> float:
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        ax.plot(x, np.asarray(y, float), linewidth=1.3)
        ax.set_title(f"Spectrum @ {gate_used:.6g} V")
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        lo, hi = float(xlim[0]), float(xlim[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = float(np.nanmin(x))
            hi = float(np.nanmax(x))
        if lo == hi:
            pad = max(1e-9, abs(lo) * 1e-6, (float(np.nanmax(x)) - float(np.nanmin(x))) * 1e-3)
            lo -= pad
            hi += pad
        safe_xlim = (lo, hi)
        ax.set_xlim(safe_xlim)
        self._auto_scale_spectrum_y(ax, x, y, safe_xlim)
        return gate_used

    def _set_drr_gate_spin_value(self, gate_value: float) -> None:
        spin = self.drr_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)

    def _set_pl_gate_spin_value(self, gate_value: float) -> None:
        spin = self.pl_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)

    def _set_cmp_gate_spin_value(self, gate_value: float) -> None:
        spin = self.cmp_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)

    def _set_power_gate_spin_value(self, gate_value: float) -> None:
        spin = self.power_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)

    def _display_power_cube(self, cube: DataCube) -> tuple[DataCube, np.ndarray, np.ndarray]:
        true_power = np.asarray(cube.gate, float).ravel()
        display_power = true_power.astype(float, copy=True)
        if display_power.size > 1:
            finite = display_power[np.isfinite(display_power)]
            span = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else 1.0
            scale = max(span, float(np.nanmax(np.abs(finite))) if finite.size else 1.0, 1.0)
            eps = scale * 1e-6
            for idx in range(1, display_power.size):
                if not np.isfinite(display_power[idx]) or not np.isfinite(display_power[idx - 1]):
                    continue
                if display_power[idx] <= display_power[idx - 1]:
                    display_power[idx] = display_power[idx - 1] + eps
        display_cube = DataCube(
            np.asarray(cube.energy, float).copy(),
            display_power,
            np.asarray(cube.Z, float).copy(),
            cube.gate_label,
            cube.title,
            cube.cbar_label,
        )
        return display_cube, true_power, display_power

    def _apply_power_tick_labels(self, ax, true_power: np.ndarray, display_power: np.ndarray) -> None:
        if true_power.size == 0 or true_power.size > 14:
            return
        if np.allclose(true_power, display_power, rtol=1e-10, atol=1e-12):
            return
        ax.set_yticks(display_power)
        ax.set_yticklabels([f"{float(value):.6g}" for value in true_power])

    def _update_power_compare_spectrum_and_lines(self, cubes: Dict[str, DataCube]) -> None:
        if self._power_spectrum_ax is None or not cubes:
            return
        power_value = float(self.power_spins["gate"].value())
        self._power_spectrum_ax.clear()
        first_cube = next(iter(cubes.values()))
        power_grid = np.asarray(first_cube.gate, float).ravel()
        if self._power_selected_row_index is not None and 0 <= self._power_selected_row_index < power_grid.size:
            idx = int(self._power_selected_row_index)
        else:
            idx = int(np.argmin(np.abs(power_grid - power_value)))
        power_used = float(power_grid[idx])
        x_ref = np.asarray(first_cube.energy, float).ravel()
        for label, cube in cubes.items():
            z = np.asarray(cube.Z, float)
            y = z[idx, :] if idx < z.shape[0] else nearest_gate_spectrum(cube, power_used)[1]
            x = np.asarray(cube.energy, float).ravel()
            self._power_spectrum_ax.plot(x, np.asarray(y, float), linewidth=1.3, label=label)
        if len(cubes) > 1:
            self._power_spectrum_ax.legend(loc="best", fontsize=9)
        self._power_spectrum_ax.set_title(f"Spectrum @ {power_used:.6g} uW")
        self._power_spectrum_ax.set_xlabel("Photon Energy (eV)")
        self._power_spectrum_ax.set_ylabel(first_cube.cbar_label)
        self._power_spectrum_ax.grid(alpha=0.25)
        xlim = self._safe_spectrum_xlim(
            x_ref,
            (float(self.power_spins["xmin"].value()), float(self.power_spins["xmax"].value())),
        )
        self._power_spectrum_ax.set_xlim(xlim)
        ys = []
        xs = []
        for cube in cubes.values():
            z = np.asarray(cube.Z, float)
            y = z[idx, :] if idx < z.shape[0] else nearest_gate_spectrum(cube, power_used)[1]
            ys.append(np.asarray(y, float))
            xs.append(np.asarray(cube.energy, float))
        if ys:
            merged_y = np.concatenate([y[np.isfinite(y)] for y in ys if np.any(np.isfinite(y))])
            if merged_y.size:
                fake_x = np.linspace(xlim[0], xlim[1], merged_y.size)
                self._auto_scale_spectrum_y(self._power_spectrum_ax, fake_x, merged_y, xlim)
        self._set_power_gate_spin_value(power_used)
        for key, ax in self._power_heatmap_axes.items():
            cube = self._power_active_cubes.get(key)
            if cube is None:
                continue
            display_cube, true_power, display_power = self._display_power_cube(cube)
            idx = int(np.argmin(np.abs(true_power - power_used)))
            y_line = float(display_power[idx]) if idx < display_power.size else power_used
            line = self._power_gate_lines.get(key)
            if line is None or getattr(line, "axes", None) is not ax:
                self._power_gate_lines[key] = ax.axhline(
                    y=y_line,
                    lw=1.2,
                    alpha=0.9,
                    color="#222",
                    linestyle="--",
                    zorder=20,
                )
            else:
                line.set_ydata([y_line, y_line])
        self.canvas.draw_idle()

    def _current_drr_spectrum(self, cube: DataCube) -> tuple[float, np.ndarray, np.ndarray]:
        gate_value = float(self.drr_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        return gate_used, x, np.asarray(y, float).ravel()

    def _visible_x_mask(self, x: np.ndarray, spins: Dict[str, QDoubleSpinBox]) -> np.ndarray:
        x0, x1 = sorted((float(spins["xmin"].value()), float(spins["xmax"].value())))
        return (x >= x0) & (x <= x1)

    def _compute_peak_indices(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        spins: Dict[str, QDoubleSpinBox],
        prom_spin: QDoubleSpinBox,
        dist_spin: QSpinBox,
        max_spin: QSpinBox,
        mode: str,
    ) -> np.ndarray:
        mask = self._visible_x_mask(x, spins) & np.isfinite(y)
        idx_all = np.where(mask)[0]
        if idx_all.size < 3:
            return np.asarray([], dtype=int)
        y_sel = y[idx_all]
        rng = float(np.nanmax(y_sel) - np.nanmin(y_sel))
        prom_frac = float(prom_spin.value())
        prom = max(0.0, prom_frac * rng)
        distance = int(dist_spin.value())
        target = -y_sel if str(mode).lower().startswith("dip") else y_sel
        peaks_local, props = find_peaks(target, prominence=prom if prom > 0 else None, distance=max(1, distance))
        if peaks_local.size == 0:
            return np.asarray([], dtype=int)
        peaks = idx_all[peaks_local]
        max_keep = int(max_spin.value())
        if max_keep > 0 and peaks.size > max_keep:
            prom_vals = np.asarray(props.get("prominences", np.zeros(peaks_local.size)), float)
            order = np.argsort(prom_vals)[::-1][:max_keep]
            peaks = peaks[order]
        return np.asarray(np.sort(peaks), dtype=int)

    def _set_fit_n_from_found(self, spin: QSpinBox, count: int) -> None:
        n = max(int(spin.minimum()), min(int(spin.maximum()), int(count)))
        if n <= 0:
            return
        old = spin.blockSignals(True)
        try:
            spin.setValue(n)
        finally:
            spin.blockSignals(old)

    @staticmethod
    def _multi_lorentz_model(x: np.ndarray, *p: float) -> np.ndarray:
        x_arr = np.asarray(x, float)
        out = p[0] + p[1] * x_arr
        n = (len(p) - 2) // 3
        for i in range(n):
            amp = p[2 + 3 * i]
            cen = p[3 + 3 * i]
            gam = max(1e-12, p[4 + 3 * i])
            out = out + amp * (gam * gam) / ((x_arr - cen) * (x_arr - cen) + gam * gam)
        return out

    def _draw_drr_analysis_overlays(self, cube: DataCube, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        if self._drr_spectrum_ax is None or self._drr_heatmap_ax is None:
            return
        if self._drr_heatmap_peak_artist is not None:
            try:
                self._drr_heatmap_peak_artist.remove()
            except Exception:
                pass
            self._drr_heatmap_peak_artist = None
        if self._drr_heatmap_fit_artist is not None:
            try:
                self._drr_heatmap_fit_artist.remove()
            except Exception:
                pass
            self._drr_heatmap_fit_artist = None
        if self._drr_peak_gate is not None and abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            self._drr_peak_gate = None
            self._drr_peak_indices = None
        if self._drr_fit_gate is not None and abs(float(gate_used) - float(self._drr_fit_gate)) > 1e-9:
            self._drr_fit_gate = None
            self._drr_fit_x = None
            self._drr_fit_y = None
            self._drr_fit_centers = None
            self.drr_fit_status.setText("")

        if (
            self.drr_peak_show_chk.isChecked()
            and self._drr_peak_indices is not None
            and self._drr_peak_gate is not None
            and abs(float(gate_used) - float(self._drr_peak_gate)) <= 1e-9
            and self._drr_peak_indices.size > 0
        ):
            pidx = np.asarray(self._drr_peak_indices, dtype=int)
            self._drr_spectrum_ax.scatter(x[pidx], y[pidx], s=26, marker="o", facecolor="#ffd84d", edgecolor="#222", zorder=30)
            self._drr_heatmap_peak_artist = self._drr_heatmap_ax.scatter(
                x[pidx],
                np.full(pidx.size, float(gate_used)),
                s=28,
                marker="o",
                facecolor="#ffd84d",
                edgecolor="#111",
                zorder=35,
            )

        if (
            self.drr_fit_show_chk.isChecked()
            and self._drr_fit_gate is not None
            and self._drr_fit_x is not None
            and self._drr_fit_y is not None
            and abs(float(gate_used) - float(self._drr_fit_gate)) <= 1e-9
        ):
            self._drr_spectrum_ax.plot(self._drr_fit_x, self._drr_fit_y, color="#f28e2b", linewidth=1.6, zorder=28)
            if self._drr_fit_centers is not None and self._drr_fit_centers.size:
                self._drr_heatmap_fit_artist = self._drr_heatmap_ax.scatter(
                    np.asarray(self._drr_fit_centers, float),
                    np.full(int(self._drr_fit_centers.size), float(gate_used)),
                    s=34,
                    marker="x",
                    color="#f28e2b",
                    linewidths=1.5,
                    zorder=36,
                )

    def _on_drr_find_peaks(self) -> None:
        self.results_dock.show()
        self._update_results_dock_page()
        if self.last_plotted_mode != "DRR" or self._last_plot_cube is None:
            return
        gate_used, x, y = self._current_drr_spectrum(self._last_plot_cube)
        peaks = self._compute_peak_indices(
            x,
            y,
            spins=self.drr_spins,
            prom_spin=self.drr_peak_prom_spin,
            dist_spin=self.drr_peak_dist_spin,
            max_spin=self.drr_peak_max_spin,
            mode=self.drr_peak_mode_combo.currentText(),
        )
        self._drr_peak_gate = float(gate_used)
        self._drr_peak_indices = peaks
        if peaks.size > 0:
            self._set_fit_n_from_found(self.drr_fit_n_spin, int(peaks.size))
        self.drr_fit_status.setText(f"Peaks: {int(peaks.size)}")
        self._update_drr_spectrum_and_gate_line(self._last_plot_cube)

    def _on_drr_fit_lorentz(self) -> None:
        self.results_dock.show()
        self._update_results_dock_page()
        if self.last_plotted_mode != "DRR" or self._last_plot_cube is None:
            return
        gate_used, x, y = self._current_drr_spectrum(self._last_plot_cube)
        mask = self._visible_x_mask(x, self.drr_spins) & np.isfinite(y)
        if np.count_nonzero(mask) < 8:
            self.drr_fit_status.setText("Fit failed: not enough points in x-range.")
            return
        x_sel = x[mask]
        y_sel = y[mask]
        n_peaks = int(self.drr_fit_n_spin.value())
        if self._drr_peak_indices is None or self._drr_peak_gate is None or abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            peak_idx = self._compute_peak_indices(
                x,
                y,
                spins=self.drr_spins,
                prom_spin=self.drr_peak_prom_spin,
                dist_spin=self.drr_peak_dist_spin,
                max_spin=self.drr_peak_max_spin,
                mode=self.drr_peak_mode_combo.currentText(),
            )
        else:
            peak_idx = np.asarray(self._drr_peak_indices, dtype=int)
        peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < x.size)]
        if peak_idx.size < n_peaks:
            order = np.argsort(np.abs(y_sel - np.nanmedian(y_sel)))[::-1]
            idx_sel = np.where(mask)[0]
            extra = idx_sel[order[: max(1, n_peaks - peak_idx.size)]]
            peak_idx = np.unique(np.concatenate([peak_idx, extra]))
        if peak_idx.size == 0:
            self.drr_fit_status.setText("Fit failed: no peak candidates.")
            return
        centers0 = np.asarray(np.sort(x[peak_idx])[:n_peaks], float)
        while centers0.size < n_peaks:
            centers0 = np.append(centers0, float(np.nanmean(x_sel)))
        base0 = float(np.nanmedian(y_sel))
        slope0 = 0.0
        y_amp = float(np.nanmax(np.abs(y_sel - base0))) if np.isfinite(np.nanmax(np.abs(y_sel - base0))) else 1.0
        x_rng = max(1e-9, float(np.nanmax(x_sel) - np.nanmin(x_sel)))
        dx = float(np.nanmedian(np.diff(x_sel))) if x_sel.size > 2 else x_rng / 100.0
        g0 = max(abs(dx) * 2.0, x_rng / 80.0, 1e-6)

        p0: list[float] = [base0, slope0]
        lo: list[float] = [float(np.nanmin(y_sel) - 3 * y_amp), -1e9]
        hi: list[float] = [float(np.nanmax(y_sel) + 3 * y_amp), 1e9]
        for c0 in centers0:
            p0.extend([y_amp * 0.7, float(c0), g0])
            lo.extend([-5 * y_amp, float(np.nanmin(x_sel)), max(abs(dx) * 0.25, 1e-8)])
            hi.extend([5 * y_amp, float(np.nanmax(x_sel)), x_rng])
        try:
            popt, _ = curve_fit(
                self._multi_lorentz_model,
                x_sel,
                y_sel,
                p0=np.asarray(p0, float),
                bounds=(np.asarray(lo, float), np.asarray(hi, float)),
                maxfev=50000,
            )
        except Exception as exc:
            self.drr_fit_status.setText(f"Fit failed: {exc}")
            return
        y_fit = self._multi_lorentz_model(x, *popt)
        centers_fit = np.asarray([popt[3 + 3 * i] for i in range(n_peaks)], float)
        self._drr_fit_gate = float(gate_used)
        self._drr_fit_x = x.copy()
        self._drr_fit_y = np.asarray(y_fit, float)
        self._drr_fit_centers = np.asarray(np.sort(centers_fit), float)
        self.drr_fit_status.setText(
            "Fit centers: " + ", ".join(f"{c:.4f}" for c in np.asarray(self._drr_fit_centers, float)[:4])
        )
        self._update_drr_spectrum_and_gate_line(self._last_plot_cube)

    def _on_drr_clear_fit(self) -> None:
        self._drr_fit_gate = None
        self._drr_fit_x = None
        self._drr_fit_y = None
        self._drr_fit_centers = None
        self.drr_fit_status.setText("")
        if self.last_plotted_mode == "DRR" and self._last_plot_cube is not None:
            self._update_drr_spectrum_and_gate_line(self._last_plot_cube)

    def _on_drr_analysis_view_changed(self) -> None:
        if self.last_plotted_mode == "DRR" and self._last_plot_cube is not None:
            self._update_drr_spectrum_and_gate_line(self._last_plot_cube)

    def _update_drr_analysis_text(self, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        lines: list[str] = []
        if (
            self._drr_peak_indices is not None
            and self._drr_peak_gate is not None
            and abs(float(gate_used) - float(self._drr_peak_gate)) <= 1e-9
            and self._drr_peak_indices.size > 0
        ):
            pidx = np.asarray(self._drr_peak_indices, dtype=int)
            pairs = [f"({float(x[i]):.5f}, {float(y[i]):.4g})" for i in pidx]
            lines.append("Found: " + "; ".join(pairs))
        else:
            lines.append("Found: none")
        if (
            self._drr_fit_centers is not None
            and self._drr_fit_centers.size > 0
            and self._drr_fit_gate is not None
            and abs(float(gate_used) - float(self._drr_fit_gate)) <= 1e-9
            and self._drr_fit_x is not None
            and self._drr_fit_y is not None
        ):
            yc = np.interp(np.asarray(self._drr_fit_centers, float), np.asarray(self._drr_fit_x, float), np.asarray(self._drr_fit_y, float))
            fit_pairs = [f"({float(cx):.5f}, {float(cy):.4g})" for cx, cy in zip(np.asarray(self._drr_fit_centers, float), np.asarray(yc, float))]
            lines.append("Fit: " + "; ".join(fit_pairs))
        else:
            lines.append("Fit: none")
        self.drr_analysis_text.setPlainText("\n".join(lines))

    def _current_pl_spectrum(self, cube: DataCube) -> tuple[float, np.ndarray, np.ndarray]:
        gate_value = float(self.pl_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        return gate_used, x, np.asarray(y, float).ravel()

    def _draw_pl_analysis_overlays(self, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        if self._pl_spectrum_ax is None or self._pl_heatmap_ax is None:
            return
        if self._pl_heatmap_peak_artist is not None:
            try:
                self._pl_heatmap_peak_artist.remove()
            except Exception:
                pass
            self._pl_heatmap_peak_artist = None
        if self._pl_heatmap_fit_artist is not None:
            try:
                self._pl_heatmap_fit_artist.remove()
            except Exception:
                pass
            self._pl_heatmap_fit_artist = None
        if self._pl_peak_gate is not None and abs(float(gate_used) - float(self._pl_peak_gate)) > 1e-9:
            self._pl_peak_gate = None
            self._pl_peak_indices = None
        if self._pl_fit_gate is not None and abs(float(gate_used) - float(self._pl_fit_gate)) > 1e-9:
            self._pl_fit_gate = None
            self._pl_fit_x = None
            self._pl_fit_y = None
            self._pl_fit_centers = None
            self.pl_fit_status.setText("")
        if (
            self.pl_peak_show_chk.isChecked()
            and self._pl_peak_indices is not None
            and self._pl_peak_gate is not None
            and abs(float(gate_used) - float(self._pl_peak_gate)) <= 1e-9
            and self._pl_peak_indices.size > 0
        ):
            pidx = np.asarray(self._pl_peak_indices, dtype=int)
            self._pl_spectrum_ax.scatter(x[pidx], y[pidx], s=26, marker="o", facecolor="#ffd84d", edgecolor="#222", zorder=30)
            self._pl_heatmap_peak_artist = self._pl_heatmap_ax.scatter(
                x[pidx],
                np.full(pidx.size, float(gate_used)),
                s=28,
                marker="o",
                facecolor="#ffd84d",
                edgecolor="#111",
                zorder=35,
            )
        if (
            self.pl_fit_show_chk.isChecked()
            and self._pl_fit_gate is not None
            and self._pl_fit_x is not None
            and self._pl_fit_y is not None
            and abs(float(gate_used) - float(self._pl_fit_gate)) <= 1e-9
        ):
            self._pl_spectrum_ax.plot(self._pl_fit_x, self._pl_fit_y, color="#f28e2b", linewidth=1.6, zorder=28)
            if self._pl_fit_centers is not None and self._pl_fit_centers.size:
                self._pl_heatmap_fit_artist = self._pl_heatmap_ax.scatter(
                    np.asarray(self._pl_fit_centers, float),
                    np.full(int(self._pl_fit_centers.size), float(gate_used)),
                    s=34,
                    marker="x",
                    color="#f28e2b",
                    linewidths=1.5,
                    zorder=36,
                )

    def _update_pl_spectrum_with_analysis(self, cube: DataCube) -> None:
        if self._pl_spectrum_ax is None:
            return
        gate_value = float(self.pl_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        self._pl_spectrum_ax.clear()
        self._pl_spectrum_ax.plot(x, np.asarray(y, float), linewidth=1.3)
        self._pl_spectrum_ax.set_title(f"Spectrum @ {gate_used:.6g} V")
        self._pl_spectrum_ax.set_xlabel("Photon Energy (eV)")
        self._pl_spectrum_ax.set_ylabel(cube.cbar_label)
        self._pl_spectrum_ax.grid(alpha=0.25)
        xlim = (float(self.pl_spins["xmin"].value()), float(self.pl_spins["xmax"].value()))
        self._pl_spectrum_ax.set_xlim(xlim)
        self._auto_scale_spectrum_y(self._pl_spectrum_ax, x, y, xlim)
        self._draw_pl_analysis_overlays(gate_used, x, np.asarray(y, float))
        self._update_pl_analysis_text(gate_used, x, np.asarray(y, float))
        self.canvas.draw_idle()

    def _ensure_pl_gate_line(self, cube: DataCube, gate_value: float) -> None:
        if self._pl_heatmap_ax is None:
            return
        gate = np.asarray(cube.gate, float).ravel()
        gate_clamped = float(np.clip(gate_value, float(np.nanmin(gate)), float(np.nanmax(gate))))
        if self._pl_gate_line is None or getattr(self._pl_gate_line, "axes", None) is not self._pl_heatmap_ax:
            self._pl_gate_line = self._pl_heatmap_ax.axhline(
                y=gate_clamped,
                lw=1.2,
                alpha=0.9,
                color="#222",
                linestyle="--",
                zorder=20,
            )
        else:
            self._pl_gate_line.set_ydata([gate_clamped, gate_clamped])
            self._pl_gate_line.set_linestyle("--")

    def _update_pl_spectrum_and_gate_line(self, cube: DataCube) -> None:
        if self._pl_spectrum_ax is None:
            return
        gate_value = float(self.pl_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        self._pl_spectrum_ax.clear()
        self._pl_spectrum_ax.plot(x, np.asarray(y, float), linewidth=1.3)
        self._pl_spectrum_ax.set_title(f"Spectrum @ {gate_used:.6g} V")
        self._pl_spectrum_ax.set_xlabel("Photon Energy (eV)")
        self._pl_spectrum_ax.set_ylabel(cube.cbar_label)
        self._pl_spectrum_ax.grid(alpha=0.25)
        xlim = self._safe_spectrum_xlim(
            x,
            (float(self.pl_spins["xmin"].value()), float(self.pl_spins["xmax"].value())),
        )
        self._pl_spectrum_ax.set_xlim(xlim)
        self._auto_scale_spectrum_y(self._pl_spectrum_ax, x, y, xlim)
        self._set_pl_gate_spin_value(gate_used)
        self._ensure_pl_gate_line(cube, gate_used)
        self._draw_pl_analysis_overlays(gate_used, x, np.asarray(y, float))
        self._update_pl_analysis_text(gate_used, x, np.asarray(y, float))
        self.canvas.draw_idle()

    def _on_pl_find_peaks(self) -> None:
        self.results_dock.show()
        self._update_results_dock_page()
        if self.last_plotted_mode != "PL" or self._pl_last_plot_cube is None:
            return
        gate_used, x, y = self._current_pl_spectrum(self._pl_last_plot_cube)
        peaks = self._compute_peak_indices(
            x,
            y,
            spins=self.pl_spins,
            prom_spin=self.pl_peak_prom_spin,
            dist_spin=self.pl_peak_dist_spin,
            max_spin=self.pl_peak_max_spin,
            mode=self.pl_peak_mode_combo.currentText(),
        )
        self._pl_peak_gate = float(gate_used)
        self._pl_peak_indices = peaks
        if peaks.size > 0:
            self._set_fit_n_from_found(self.pl_fit_n_spin, int(peaks.size))
        self.pl_fit_status.setText(f"Peaks: {int(peaks.size)}")
        self._update_pl_spectrum_with_analysis(self._pl_last_plot_cube)

    def _on_pl_fit_lorentz(self) -> None:
        self.results_dock.show()
        self._update_results_dock_page()
        if self.last_plotted_mode != "PL" or self._pl_last_plot_cube is None:
            return
        gate_used, x, y = self._current_pl_spectrum(self._pl_last_plot_cube)
        mask = self._visible_x_mask(x, self.pl_spins) & np.isfinite(y)
        if np.count_nonzero(mask) < 8:
            self.pl_fit_status.setText("Fit failed: not enough points in x-range.")
            return
        x_sel = x[mask]
        y_sel = y[mask]
        n_peaks = int(self.pl_fit_n_spin.value())
        if self._pl_peak_indices is None or self._pl_peak_gate is None or abs(float(gate_used) - float(self._pl_peak_gate)) > 1e-9:
            peak_idx = self._compute_peak_indices(
                x,
                y,
                spins=self.pl_spins,
                prom_spin=self.pl_peak_prom_spin,
                dist_spin=self.pl_peak_dist_spin,
                max_spin=self.pl_peak_max_spin,
                mode=self.pl_peak_mode_combo.currentText(),
            )
        else:
            peak_idx = np.asarray(self._pl_peak_indices, dtype=int)
        peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < x.size)]
        if peak_idx.size < n_peaks:
            order = np.argsort(np.abs(y_sel - np.nanmedian(y_sel)))[::-1]
            idx_sel = np.where(mask)[0]
            extra = idx_sel[order[: max(1, n_peaks - peak_idx.size)]]
            peak_idx = np.unique(np.concatenate([peak_idx, extra]))
        if peak_idx.size == 0:
            self.pl_fit_status.setText("Fit failed: no peak candidates.")
            return
        centers0 = np.asarray(np.sort(x[peak_idx])[:n_peaks], float)
        while centers0.size < n_peaks:
            centers0 = np.append(centers0, float(np.nanmean(x_sel)))
        base0 = float(np.nanmedian(y_sel))
        slope0 = 0.0
        y_amp = float(np.nanmax(np.abs(y_sel - base0))) if np.isfinite(np.nanmax(np.abs(y_sel - base0))) else 1.0
        x_rng = max(1e-9, float(np.nanmax(x_sel) - np.nanmin(x_sel)))
        dx = float(np.nanmedian(np.diff(x_sel))) if x_sel.size > 2 else x_rng / 100.0
        g0 = max(abs(dx) * 2.0, x_rng / 80.0, 1e-6)
        p0: list[float] = [base0, slope0]
        lo: list[float] = [float(np.nanmin(y_sel) - 3 * y_amp), -1e9]
        hi: list[float] = [float(np.nanmax(y_sel) + 3 * y_amp), 1e9]
        for c0 in centers0:
            p0.extend([y_amp * 0.7, float(c0), g0])
            lo.extend([-5 * y_amp, float(np.nanmin(x_sel)), max(abs(dx) * 0.25, 1e-8)])
            hi.extend([5 * y_amp, float(np.nanmax(x_sel)), x_rng])
        try:
            popt, _ = curve_fit(
                self._multi_lorentz_model,
                x_sel,
                y_sel,
                p0=np.asarray(p0, float),
                bounds=(np.asarray(lo, float), np.asarray(hi, float)),
                maxfev=50000,
            )
        except Exception as exc:
            self.pl_fit_status.setText(f"Fit failed: {exc}")
            return
        y_fit = self._multi_lorentz_model(x, *popt)
        centers_fit = np.asarray([popt[3 + 3 * i] for i in range(n_peaks)], float)
        self._pl_fit_gate = float(gate_used)
        self._pl_fit_x = x.copy()
        self._pl_fit_y = np.asarray(y_fit, float)
        self._pl_fit_centers = np.asarray(np.sort(centers_fit), float)
        self.pl_fit_status.setText("Fit centers: " + ", ".join(f"{c:.4f}" for c in np.asarray(self._pl_fit_centers, float)[:4]))
        self._update_pl_spectrum_with_analysis(self._pl_last_plot_cube)

    def _on_pl_clear_fit(self) -> None:
        self._pl_fit_gate = None
        self._pl_fit_x = None
        self._pl_fit_y = None
        self._pl_fit_centers = None
        self.pl_fit_status.setText("")
        if self.last_plotted_mode == "PL" and self._pl_last_plot_cube is not None:
            self._update_pl_spectrum_with_analysis(self._pl_last_plot_cube)

    def _on_pl_analysis_view_changed(self) -> None:
        if self.last_plotted_mode == "PL" and self._pl_last_plot_cube is not None:
            self._update_pl_spectrum_and_gate_line(self._pl_last_plot_cube)

    def _on_pl_gate_changed(self) -> None:
        if self.last_plotted_mode == "PL" and self._pl_last_plot_cube is not None:
            self._update_pl_spectrum_and_gate_line(self._pl_last_plot_cube)

    def _plot_compare_linecut(
        self,
        ax: Any,
        cubes: Dict[str, DataCube],
        *,
        gate_value: float,
        xlim: tuple[float, float],
    ) -> float:
        gate_used_values: list[float] = []
        for key in [label for label in COMPARE_PANEL_ORDER if label in cubes]:
            cube = cubes[key]
            gate_used, y = nearest_gate_spectrum(cube, gate_value)
            x = np.asarray(cube.energy, float).ravel()
            ax.plot(x, np.asarray(y, float), linewidth=1.3, label=key)
            gate_used_values.append(float(gate_used))
        gate_used = float(np.median(gate_used_values)) if gate_used_values else float(gate_value)
        ax.set_title(f"Compare Spectra @ {gate_used:.6g} V")
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_ylabel("PL corr. (a.u.)")
        ax.grid(alpha=0.25)
        safe_xlim = self._safe_spectrum_xlim(np.asarray(next(iter(cubes.values())).energy, float), xlim)
        ax.set_xlim(safe_xlim)
        finite_lines = [np.asarray(line.get_ydata(), float) for line in ax.lines if len(line.get_ydata())]
        if finite_lines:
            y_all = np.concatenate([line[np.isfinite(line)] for line in finite_lines if np.any(np.isfinite(line))])
            if y_all.size:
                ymin = float(np.nanmin(y_all))
                ymax = float(np.nanmax(y_all))
                pad = max(1e-12, (ymax - ymin) * 0.08) if ymax != ymin else max(1e-12, abs(ymin) * 0.05, 1.0)
                ax.set_ylim(ymin - pad, ymax + pad)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        return gate_used

    def _update_pl_analysis_text(self, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        lines: list[str] = []
        if (
            self._pl_peak_indices is not None
            and self._pl_peak_gate is not None
            and abs(float(gate_used) - float(self._pl_peak_gate)) <= 1e-9
            and self._pl_peak_indices.size > 0
        ):
            pidx = np.asarray(self._pl_peak_indices, dtype=int)
            pairs = [f"({float(x[i]):.5f}, {float(y[i]):.4g})" for i in pidx]
            lines.append("Found: " + "; ".join(pairs))
        else:
            lines.append("Found: none")
        if (
            self._pl_fit_centers is not None
            and self._pl_fit_centers.size > 0
            and self._pl_fit_gate is not None
            and abs(float(gate_used) - float(self._pl_fit_gate)) <= 1e-9
            and self._pl_fit_x is not None
            and self._pl_fit_y is not None
        ):
            yc = np.interp(np.asarray(self._pl_fit_centers, float), np.asarray(self._pl_fit_x, float), np.asarray(self._pl_fit_y, float))
            fit_pairs = [f"({float(cx):.5f}, {float(cy):.4g})" for cx, cy in zip(np.asarray(self._pl_fit_centers, float), np.asarray(yc, float))]
            lines.append("Fit: " + "; ".join(fit_pairs))
        else:
            lines.append("Fit: none")
        self.pl_analysis_text.setPlainText("\n".join(lines))

    def _remove_nearest_drr_peak(self, x_click: float) -> bool:
        if self._last_plot_cube is None or self._drr_peak_indices is None or self._drr_peak_indices.size == 0:
            return False
        gate_used, _y = nearest_gate_spectrum(self._last_plot_cube, float(self.drr_spins["gate"].value()))
        if self._drr_peak_gate is None or abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            return False
        x = np.asarray(self._last_plot_cube.energy, float).ravel()
        pidx = np.asarray(self._drr_peak_indices, dtype=int)
        j = int(np.argmin(np.abs(x[pidx] - float(x_click))))
        self._drr_peak_indices = np.delete(pidx, j)
        self.drr_fit_status.setText(f"Peaks: {int(self._drr_peak_indices.size)}")
        self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
        return True

    def _remove_nearest_pl_peak(self, x_click: float) -> bool:
        if self._pl_last_plot_cube is None or self._pl_peak_indices is None or self._pl_peak_indices.size == 0:
            return False
        gate_used, _y = nearest_gate_spectrum(self._pl_last_plot_cube, float(self.pl_spins["gate"].value()))
        if self._pl_peak_gate is None or abs(float(gate_used) - float(self._pl_peak_gate)) > 1e-9:
            return False
        x = np.asarray(self._pl_last_plot_cube.energy, float).ravel()
        pidx = np.asarray(self._pl_peak_indices, dtype=int)
        j = int(np.argmin(np.abs(x[pidx] - float(x_click))))
        self._pl_peak_indices = np.delete(pidx, j)
        self.pl_fit_status.setText(f"Peaks: {int(self._pl_peak_indices.size)}")
        self._update_pl_spectrum_with_analysis(self._pl_last_plot_cube)
        return True

    def _remove_peak_from_drr_heatmap_click(self, x_click: float, y_click: float) -> bool:
        if (
            self._last_plot_cube is None
            or self._drr_peak_indices is None
            or self._drr_peak_indices.size == 0
            or self._drr_peak_gate is None
        ):
            return False
        gate_axis = np.asarray(self._last_plot_cube.gate, float).ravel()
        if gate_axis.size == 0:
            return False
        gmin, gmax = float(np.nanmin(gate_axis)), float(np.nanmax(gate_axis))
        grng = max(1e-12, gmax - gmin)
        dg = float(np.nanmedian(np.abs(np.diff(gate_axis)))) if gate_axis.size > 1 else grng * 0.02
        tol = max(0.02 * grng, 0.75 * max(dg, 1e-9))
        if abs(float(y_click) - float(self._drr_peak_gate)) > tol:
            return False
        return self._remove_nearest_drr_peak(float(x_click))

    def _remove_peak_from_pl_heatmap_click(self, x_click: float, y_click: float) -> bool:
        if (
            self._pl_last_plot_cube is None
            or self._pl_peak_indices is None
            or self._pl_peak_indices.size == 0
            or self._pl_peak_gate is None
        ):
            return False
        gate_axis = np.asarray(self._pl_last_plot_cube.gate, float).ravel()
        if gate_axis.size == 0:
            return False
        gmin, gmax = float(np.nanmin(gate_axis)), float(np.nanmax(gate_axis))
        grng = max(1e-12, gmax - gmin)
        dg = float(np.nanmedian(np.abs(np.diff(gate_axis)))) if gate_axis.size > 1 else grng * 0.02
        tol = max(0.02 * grng, 0.75 * max(dg, 1e-9))
        if abs(float(y_click) - float(self._pl_peak_gate)) > tol:
            return False
        return self._remove_nearest_pl_peak(float(x_click))

    def _plot_shg_result(self, result: ShgProcessResult) -> None:
        data = result.data
        settings = result.settings
        angle = np.asarray(result.measured_angle_deg, float)
        finite_angles = np.flatnonzero(np.isfinite(angle))
        if finite_angles.size == 0:
            raise ValueError("SHG data contains no finite measured angles.")
        requested = float(self.shg_angle_cursor_spin.value())
        selected = int(finite_angles[np.argmin(np.abs(angle[finite_angles] - requested))])
        self._shg_selected_index = selected
        blocked = self.shg_angle_cursor_spin.blockSignals(True)
        try:
            self.shg_angle_cursor_spin.setValue(float(angle[selected]))
        finally:
            self.shg_angle_cursor_spin.blockSignals(blocked)

        grid = self.figure.add_gridspec(
            nrows=3,
            ncols=2,
            height_ratios=[1.0, 0.9, 0.32],
            wspace=0.25,
            hspace=0.38,
        )
        raw_ax = self.figure.add_subplot(grid[0, 0])
        corrected_ax = self.figure.add_subplot(grid[0, 1], sharex=raw_ax)
        angle_ax = self.figure.add_subplot(grid[1, :])
        residual_ax = self.figure.add_subplot(grid[2, :], sharex=angle_ax)
        wavelength = np.asarray(data.wavelength_nm, float)
        raw = np.asarray(data.spectra[selected], float)
        cleaned = np.asarray(result.cleaned_spectra[selected], float)
        cosmic_mask = np.asarray(result.cosmic_ray_mask[selected], bool)
        baseline = np.asarray(result.baseline[selected], float)
        corrected = np.asarray(result.corrected[selected], float)

        spectrum_view = self.shg_spectrum_view_combo.currentText()
        if spectrum_view in {"Raw + cleaned", "Raw"}:
            raw_ax.plot(wavelength, raw, color="#1769c2", lw=1.0, label="Raw measured")
        if spectrum_view in {"Raw + cleaned", "Cosmic-cleaned"}:
            raw_ax.plot(
                wavelength,
                cleaned,
                color="#7b2cbf",
                lw=1.1,
                alpha=0.9,
                label="Cosmic-cleaned",
            )
        if np.any(cosmic_mask):
            marker_values = raw if spectrum_view != "Cosmic-cleaned" else cleaned
            raw_ax.scatter(
                wavelength[cosmic_mask],
                marker_values[cosmic_mask],
                color="#d62728",
                marker="x",
                s=38,
                linewidths=1.4,
                label=f"Removed ({int(np.count_nonzero(cosmic_mask))})",
                zorder=6,
            )
        raw_ax.plot(wavelength, baseline, color="#ef7f1a", lw=1.5, label="Background")
        raw_ax.set_title(
            f"{data.source_file} row {int(data.source_rows[selected])} — {angle[selected]:.6g}°"
        )
        raw_ax.set_xlabel("Wavelength (nm)")
        raw_ax.set_ylabel("Intensity (counts)")
        raw_ax.grid(alpha=0.25)
        raw_ax.legend(loc="best", fontsize=8)

        corrected_ax.plot(wavelength, corrected, color="#11823b", lw=1.0)
        corrected_ax.axhline(0.0, color="#555", lw=0.8, alpha=0.7)
        corrected_ax.set_title(
            f"Background-subtracted area ({settings.gate_min_nm:g}-{settings.gate_max_nm:g} nm) "
            f"= {result.integrated_area[selected]:.6g} counts·nm"
        )
        corrected_ax.set_xlabel("Wavelength (nm)")
        corrected_ax.set_ylabel("Corrected intensity")
        corrected_ax.grid(alpha=0.25)

        for axis in (raw_ax, corrected_ax):
            axis.axvspan(settings.left_min_nm, settings.left_max_nm, color="#2f80ed", alpha=0.10)
            axis.axvspan(settings.gate_min_nm, settings.gate_max_nm, color="#eb5757", alpha=0.13)
            axis.axvspan(settings.right_min_nm, settings.right_max_nm, color="#2f80ed", alpha=0.10)
            margin = max(0.5, 0.08 * (settings.right_max_nm - settings.left_min_nm))
            axis.set_xlim(settings.left_min_nm - margin, settings.right_max_nm + margin)

        finite_area = np.isfinite(angle) & np.isfinite(result.integrated_area)
        valid = finite_area & np.asarray(result.included, bool)
        invalid = finite_area & ~np.asarray(result.included, bool)
        if np.any(valid):
            valid_indices = np.flatnonzero(valid)
            valid_order = valid_indices[np.argsort(angle[valid_indices], kind="stable")]
            angle_ax.errorbar(
                angle[valid_order],
                result.integrated_area[valid_order],
                yerr=result.area_uncertainty[valid_order],
                color="#1769c2",
                marker="o",
                markersize=3.5,
                lw=1.0,
                capsize=2,
                label="Included",
            )
        if np.any(invalid):
            angle_ax.scatter(
                angle[invalid],
                result.integrated_area[invalid],
                color="#888",
                marker="x",
                s=30,
                label="Excluded",
                zorder=4,
            )
        if np.isfinite(result.integrated_area[selected]):
            angle_ax.scatter(
                [angle[selected]],
                [result.integrated_area[selected]],
                color="#d62728",
                edgecolor="white",
                linewidth=0.8,
                s=55,
                zorder=6,
                label="Selected",
            )
        fit = self.loaded.shg_fit if self.loaded is not None else None
        if fit is not None:
            dense_angle = np.linspace(
                float(np.nanmin(angle[fit.fit_mask])),
                float(np.nanmax(angle[fit.fit_mask])),
                721,
            )
            angle_ax.plot(
                dense_angle,
                evaluate_shg_angular_model(dense_angle, fit.i0, fit.amplitude, fit.x_center_deg),
                color="#d62728",
                lw=1.6,
                label=f"Fit xc={fit.x_center_deg:.4g}°",
            )
            residual_ax.scatter(
                angle[fit.fit_mask],
                fit.residual[fit.fit_mask],
                color="#1769c2",
                s=16,
            )
            residual_ax.axhline(0.0, color="#555", lw=0.8)
            residual_ax.set_ylabel("Residual")
            residual_ax.set_xlabel("Measured angle (deg)")
            residual_ax.grid(alpha=0.25)
            angle_ax.set_xlabel("")
        else:
            residual_ax.set_visible(False)
        angle_ax.set_title("Background-subtracted SHG area versus measured angle")
        angle_ax.set_xlabel("Measured angle (deg)")
        angle_ax.set_ylabel("Background-subtracted area (counts·nm)")
        angle_ax.grid(alpha=0.25)
        angle_ax.legend(loc="best", fontsize=8)
        self._shg_raw_ax = raw_ax
        self._shg_corrected_ax = corrected_ax
        self._shg_angle_ax = angle_ax

    def _plot_shg_comparison(
        self,
        reference: ShgProcessResult,
        sample: ShgProcessResult,
    ) -> None:
        grid = self.figure.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[1.0, 0.32],
            hspace=0.12,
        )
        angle_ax = self.figure.add_subplot(grid[0, 0])
        residual_ax = self.figure.add_subplot(grid[1, 0], sharex=angle_ax)
        normalized = self.shg_compare_display_combo.currentText() == "Normalized"
        plotted_residual = False

        for label, result, fit, color, marker in (
            ("Reference A", reference, self.loaded.shg_fit if self.loaded else None, "#1769c2", "o"),
            ("Sample B", sample, self.loaded.shg_fit_b if self.loaded else None, "#d62728", "s"),
        ):
            angle = np.asarray(result.measured_angle_deg, float)
            area = np.asarray(result.integrated_area, float)
            uncertainty = np.asarray(result.area_uncertainty, float)
            finite = np.isfinite(angle) & np.isfinite(area)
            included = finite & np.asarray(result.included, bool)
            if normalized and fit is not None and fit.amplitude != 0:
                scale = fit.amplitude
                offset = fit.i0
            elif normalized and np.any(finite):
                offset = float(np.nanmin(area[finite]))
                scale = max(float(np.nanmax(area[finite]) - offset), np.finfo(float).eps)
            else:
                scale = 1.0
                offset = 0.0
            plotted_area = (area - offset) / scale
            plotted_uncertainty = uncertainty / abs(scale)
            indices = np.flatnonzero(included)
            order = indices[np.argsort(angle[indices], kind="stable")]
            if order.size:
                angle_ax.errorbar(
                    angle[order],
                    plotted_area[order],
                    yerr=plotted_uncertainty[order],
                    color=color,
                    marker=marker,
                    markersize=4.0,
                    lw=0.9,
                    capsize=2,
                    label=f"{label}: {result.data.source_file}",
                )
            excluded = finite & ~np.asarray(result.included, bool)
            if np.any(excluded):
                angle_ax.scatter(
                    angle[excluded],
                    plotted_area[excluded],
                    color=color,
                    marker="x",
                    s=28,
                    alpha=0.55,
                )
            if fit is not None:
                dense_angle = np.linspace(
                    float(np.nanmin(angle[fit.fit_mask])),
                    float(np.nanmax(angle[fit.fit_mask])),
                    721,
                )
                dense_fit = evaluate_shg_angular_model(
                    dense_angle,
                    fit.i0,
                    fit.amplitude,
                    fit.x_center_deg,
                )
                angle_ax.plot(
                    dense_angle,
                    (dense_fit - offset) / scale,
                    color=color,
                    lw=1.8,
                    alpha=0.9,
                    label=f"{label} fit: xc={fit.x_center_deg:.5g}°",
                )
                residual_ax.scatter(
                    angle[fit.fit_mask],
                    fit.residual[fit.fit_mask] / abs(scale),
                    color=color,
                    marker=marker,
                    s=18,
                    label=label,
                )
                plotted_residual = True

        twist = self.loaded.shg_twist if self.loaded is not None else None
        if twist is not None:
            title = (
                f"SHG twist comparison: Δxc={twist.delta_x_center_deg:.6g}°; "
                f"twist={twist.signed_twist_angle_deg:.6g} ± {twist.twist_uncertainty_deg:.3g}°"
            )
        else:
            title = "SHG comparison versus measured angle (fit unavailable)"
        angle_ax.set_title(title)
        angle_ax.set_ylabel("Normalized area" if normalized else "Background-subtracted area (counts·nm)")
        angle_ax.grid(alpha=0.25)
        angle_ax.legend(loc="best", fontsize=8)
        if plotted_residual:
            residual_ax.axhline(0.0, color="#555", lw=0.8)
            residual_ax.set_ylabel("Residual")
            residual_ax.set_xlabel("Measured angle (deg)")
            residual_ax.grid(alpha=0.25)
            residual_ax.legend(loc="best", fontsize=8, ncol=2)
            angle_ax.tick_params(labelbottom=False)
        else:
            residual_ax.set_visible(False)
            angle_ax.set_xlabel("Measured angle (deg)")
        self._shg_raw_ax = None
        self._shg_corrected_ax = None
        self._shg_angle_ax = angle_ax

    def _add_heatmap_colorbar(
        self,
        render: HeatmapRender,
        cax,
        *,
        label: str,
        ticks_on_left: bool = False,
        ticks_on_top: bool = False,
        orientation: str = "vertical",
    ) -> None:
        if not render.is_split:
            colorbar = self.figure.colorbar(render.primary, cax=cax, label=label, orientation=orientation)
            if ticks_on_left and orientation == "vertical":
                colorbar.ax.yaxis.set_ticks_position("left")
                colorbar.ax.yaxis.set_label_position("left")
            if ticks_on_top and orientation == "horizontal":
                colorbar.ax.xaxis.set_ticks_position("top")
                colorbar.ax.xaxis.set_label_position("top")
            return
        cax.set_axis_off()
        if orientation == "horizontal":
            left_cax = cax.inset_axes([0.00, 0.0, 0.46, 1.0])
            right_cax = cax.inset_axes([0.54, 0.0, 0.46, 1.0])
        else:
            left_cax = cax.inset_axes([0.0, 0.55, 1.0, 0.43])
            right_cax = cax.inset_axes([0.0, 0.02, 1.0, 0.43])
        left_cb = self.figure.colorbar(render.primary, cax=left_cax, orientation=orientation)
        right_cb = self.figure.colorbar(render.secondary, cax=right_cax, orientation=orientation)
        split_text = f"{float(render.split_x):.6g}"
        left_cb.ax.set_title(f"x ≤ {split_text}", fontsize=8, pad=2)
        right_cb.ax.set_title(f"x ≥ {split_text}", fontsize=8, pad=2)
        left_cb.set_label(label, fontsize=8)
        right_cb.set_label(label, fontsize=8)
        left_cb.ax.tick_params(labelsize=7)
        right_cb.ax.tick_params(labelsize=7)
        if ticks_on_left and orientation == "vertical":
            for colorbar in (left_cb, right_cb):
                colorbar.ax.yaxis.set_ticks_position("left")
                colorbar.ax.yaxis.set_label_position("left")
        if ticks_on_top and orientation == "horizontal":
            for colorbar in (left_cb, right_cb):
                colorbar.ax.xaxis.set_ticks_position("top")
                colorbar.ax.xaxis.set_label_position("top")

    def _plot_mode(self, mode: str, *, auto: bool = False) -> None:
        try:
            if not self.loaded or self.loaded.mode != mode:
                self._show_error("Load data for this tab before plotting.")
                return

            self._ensure_loaded_matches_ui_params(mode)
            plot_key = self._current_plot_params_key(mode)
            gate_only_update = mode == "DRR" and self._is_drr_gate_only_change(plot_key)
            if gate_only_update and self._last_plot_cube is not None:
                self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
                self._last_plot_params_key = plot_key
                self.last_plotted_mode = mode
                self._set_stage("Plotted")
                self._update_action_states()
                self._status(f"Plotted DRR (gate update).")
                return

            self.figure.clear()
            self._cmp_heatmap_axes = {}
            self._cmp_gate_lines = {}
            self._cmp_linecut_ax = None
            self._cmp_active_cubes = {}
            self._power_heatmap_ax = None
            self._power_heatmap_axes = {}
            self._power_spectrum_ax = None
            self._power_last_plot_cube = None
            self._power_gate_line = None
            self._power_gate_lines = {}
            self._power_active_cubes = {}
            self._power_active_export_cube = None
            self._power_active_records = ()
            self._shg_raw_ax = None
            self._shg_corrected_ax = None
            self._shg_angle_ax = None
            plot_cube = None
            if mode == "PL" and self.loaded.cube is not None:
                plot_cube = self.loaded.cube
                params = self._make_params(mode, plot_cube)
                gs = self.figure.add_gridspec(
                    nrows=2,
                    ncols=2,
                    width_ratios=[1.0, 0.035],
                    height_ratios=[1.0, 1.0],
                    wspace=0.12,
                    hspace=0.28,
                )
                ax1 = self.figure.add_subplot(gs[0, 0])
                cax = self.figure.add_subplot(gs[0, 1])
                ax2 = self.figure.add_subplot(gs[1, 0], sharex=ax1)
                render = plot_pl(ax1, plot_cube, params)
                self._add_heatmap_colorbar(render, cax, label=params.cbar_label)
                self._pl_heatmap_ax = ax1
                self._pl_spectrum_ax = ax2
                self._pl_last_plot_cube = plot_cube
                self._pl_gate_line = None
                self._pl_heatmap_peak_artist = None
                self._pl_heatmap_fit_artist = None
                self._update_pl_spectrum_and_gate_line(plot_cube)
            elif mode == "DRR" and self.loaded.cube is not None:
                self._pl_heatmap_ax = None
                self._pl_spectrum_ax = None
                self._pl_last_plot_cube = None
                self._pl_gate_line = None
                plot_cube = self._drr_cube_for_display()
                self.loaded.drr_derivative_label = self.drr_derivative_combo.currentText()
                params = self._make_params(mode, plot_cube)
                gs = self.figure.add_gridspec(
                    nrows=2,
                    ncols=2,
                    width_ratios=[1.0, 0.035],
                    height_ratios=[1.0, 1.0],
                    wspace=0.12,
                    hspace=0.28,
                )
                ax1 = self.figure.add_subplot(gs[0, 0])
                cax = self.figure.add_subplot(gs[0, 1])
                ax2 = self.figure.add_subplot(gs[1, 0], sharex=ax1)
                render = plot_drr(ax1, plot_cube, params)
                self._add_heatmap_colorbar(render, cax, label=params.cbar_label)
                gate_val = float(self._mode_spins(mode)["gate"].value())
                gate_used = self._plot_spectrum_with_roi(
                    ax2, plot_cube, gate_val, ylabel=params.cbar_label, xlim=params.xlim
                )
                self._drr_heatmap_ax = ax1
                self._drr_spectrum_ax = ax2
                self._drr_heatmap_peak_artist = None
                self._drr_heatmap_fit_artist = None
                self._set_drr_gate_spin_value(gate_used)
                self._update_drr_spectrum_and_gate_line(plot_cube)
            elif mode == "MCD" and self.loaded.mcd_result is not None:
                result = self.loaded.mcd_result
                plot_cube = result.cube(self.mcd_map_combo.currentText())
                params = self._make_params(mode, plot_cube)
                # Match the PL/DRR hierarchy: the heatmap occupies the
                # upper-left, the selected spectrum sits below, and the
                # derived MCD traces live in a separate right column.  A
                # dedicated header above the map holds its title and colorbar
                # without covering any measured data.
                gs = self.figure.add_gridspec(
                    nrows=2, ncols=3,
                    width_ratios=[1.0, 1.0, 1.0],
                    height_ratios=[1.05, 0.95],
                    left=0.075, right=0.910, bottom=0.085, top=0.935,
                    wspace=0.50, hspace=0.34,
                )
                map_grid = gs[0, :2].subgridspec(2, 1, height_ratios=[0.10, 1.0], hspace=0.06)
                header_ax = self.figure.add_subplot(map_grid[0, 0])
                header_ax.set_axis_off()
                header_ax.text(0.0, 0.23, plot_cube.title, transform=header_ax.transAxes, ha="left", va="center", fontsize=12)
                cax = header_ax.inset_axes([0.66, 0.04, 0.30, 0.38])
                heat_ax = self.figure.add_subplot(map_grid[1, 0])
                trace_ax = self.figure.add_subplot(gs[0, 2])
                pair_ax = self.figure.add_subplot(gs[1, :2], sharex=heat_ax)
                linecut_ax = self.figure.add_subplot(gs[1, 2], sharex=heat_ax)
                pair_index = int(self.mcd_pair_b_combo.currentData() or 0)
                pair_index = int(np.clip(pair_index, 0, result.pair_b.size - 1))
                pair_b = float(result.pair_b[pair_index])
                energy = np.asarray(result.energy_ev, float)
                energy_order = np.argsort(1239.841984 / result.wavelength_nm)
                correction_mode = str(result.summary.get("correction_mode", "global"))
                drift_fit_used = correction_mode in {"pair_scale", "pair_affine", "pair_spectral"}
                if drift_fit_used:
                    settings = self.loaded.mcd_settings or McdSettings()
                    for start, stop in background_fit_regions(energy, settings.background_ranges_ev):
                        pair_ax.axvspan(start, stop, color="#5790b7", alpha=0.13, zorder=0)
                pair_ax.plot(energy, result.pair_raw_pos[pair_index, energy_order], label=f"raw {result.pos_angle:g} deg", lw=1.0, ls="--", alpha=0.65)
                pair_ax.plot(energy, result.pair_raw_neg[pair_index, energy_order], label=f"raw {result.neg_angle:g} deg", lw=1.0, ls="--", alpha=0.65)
                pair_ax.plot(energy, result.pair_corrected_pos[pair_index, energy_order], label=f"corrected {result.pos_angle:g} deg", lw=1.35)
                pair_ax.plot(energy, result.pair_corrected_neg[pair_index, energy_order], label=f"final corrected {result.neg_angle:g} deg", lw=1.35)
                linecut_ax.plot(energy, result.pair_mcd_raw[pair_index, energy_order], label="raw MCD", lw=1.1, alpha=0.8)
                linecut_ax.plot(energy, result.pair_mcd_corrected[pair_index, energy_order], label="corrected MCD", lw=1.4)
                e0 = float(self.mcd_window_center_spin.value()); width = float(self.mcd_window_width_spin.value())
                if e0 <= 0:
                    e0 = float(np.nanmedian(energy)); self.mcd_window_center_spin.blockSignals(True); self.mcd_window_center_spin.setValue(e0); self.mcd_window_center_spin.blockSignals(False)
                half = width * 5e-4
                for axis in (pair_ax, linecut_ax):
                    axis.axvline(e0, color="#333", ls=":", lw=0.9); axis.axvspan(e0 - half, e0 + half, color="#333", alpha=0.10)
                    axis.set_xlim(self._safe_spectrum_xlim(energy, params.xlim)); axis.grid(alpha=0.22); axis.legend(fontsize=7, frameon=False)
                    axis.set_xlabel("Energy (eV)")
                residual = float(result.pair_background_rms[pair_index])
                residual_before = float(result.pair_background_rms_before[pair_index])
                residual_text = f"{residual:.3g}" if np.isfinite(residual) else "--"
                residual_before_text = f"{residual_before:.3g}" if np.isfinite(residual_before) else "--"
                pair_ax.set_title(f"Paired spectra: B = {pair_b:.5g} T")
                pair_ax.set_ylabel("Intensity")
                # Keep correction diagnostics readable without covering the
                # high-energy spectrum.  The upper-left corner is the quiet
                # region for the usual rising PL background, and putting the
                # values in the legend makes them visually consistent with
                # the trace labels rather than a floating annotation.
                pair_handles, pair_labels = pair_ax.get_legend_handles_labels()
                correction_handle = Line2D([], [], linestyle="None", marker=None)
                if correction_mode == "pair_spectral":
                    fit_name = "quadratic" if int(result.summary.get("spectral_order", 1)) == 2 else "linear"
                    correction_label = (
                        f"spectral {fit_name}: gain@Ec={result.pair_scale[pair_index]:.4g}; "
                        f"slope={result.pair_spectral_slope[pair_index]:+.3g}/eV\n"
                        f"RMS {residual_before_text}->{residual_text}; gain range "
                        f"{result.pair_correction_min[pair_index]:.3g}-{result.pair_correction_max[pair_index]:.3g}"
                    )
                else:
                    correction_label = (
                        f"correction: scale={result.pair_scale[pair_index]:.4g}; "
                        f"offset={result.pair_offset[pair_index]:.3g}; RMS={residual_text}"
                    )
                if drift_fit_used:
                    correction_label += " (blue regions used)"
                pair_ax.legend(
                    [*pair_handles, correction_handle], [*pair_labels, correction_label],
                    loc="upper left", fontsize=7, frameon=True, framealpha=0.88,
                    facecolor="white", edgecolor="#b7b7b7", handlelength=2.0,
                    borderpad=0.45, labelspacing=0.35,
                )
                delta_b = float(result.pair_delta_b[pair_index])
                alignment = "; aligned" if (result.pair_interpolated_pos[pair_index] or result.pair_interpolated_neg[pair_index]) else ""
                linecut_ax.set_title(f"MCD linecut: B = {pair_b:.5g} T")
                linecut_ax.set_ylabel("MCD"); linecut_ax.axhline(0, color="#555", lw=0.7)
                linecut_ax.text(
                    0.98, 0.04, f"dB = {delta_b:+.4g} T{alignment}",
                    transform=linecut_ax.transAxes, ha="right", va="bottom", fontsize=7,
                )
                render = plot_heatmap(heat_ax, plot_cube, params)
                heat_ax.set_title("")
                # The paired spectrum below shares the energy axis, so it
                # owns the x label.  Suppressing the redundant heatmap label
                # leaves clear space for the lower-panel title.
                heat_ax.set_xlabel("")
                heat_ax.tick_params(labelbottom=False)
                self._add_heatmap_colorbar(render, cax, label="MCD", ticks_on_top=True, orientation="horizontal")
                # The cursor always represents the selected measured pair,
                # rather than a separate (and visually confusing) map-only B.
                heat_ax.axhline(pair_b, color="#222", ls="--", lw=1.0)
                heat_ax.axvline(e0, color="#222", ls=":", lw=1.0); heat_ax.axvspan(e0 - half, e0 + half, color="#333", alpha=0.12)
                traces = pair_window_trace_by_branch(result, e0, width)
                integral_ax = trace_ax.twinx()
                trace_specs = (
                    ("mean", "Signed mean", "#1666b0", self.mcd_show_signed_mean_chk.isChecked()),
                    ("field_signed_absolute_mean", "Field-signed |MCD|", "#c94c00", self.mcd_show_absolute_mean_chk.isChecked()),
                    ("absolute_mean", "Unsigned |MCD|", "#777777", self.mcd_show_unsigned_absolute_mean_chk.isChecked()),
                    ("integral", "Signed integral", "#6a3d9a", self.mcd_show_integral_chk.isChecked()),
                )
                show_raw = self.mcd_show_raw_chk.isChecked()
                for metric_name, label, color, visible in trace_specs:
                    if not visible:
                        continue
                    axis = integral_ax if metric_name == "integral" else trace_ax
                    for branch, line_style, marker_fill in (("B increasing", "-", color), ("B decreasing", "--", "white")):
                        for source, alpha in (("corrected", 1.0), ("raw", 0.72)):
                            if source == "raw" and not show_raw:
                                continue
                            b_trace, values = traces[branch][f"{source}_{metric_name}"]
                            axis.plot(
                                b_trace, values, f"o{line_style}", ms=3.1, lw=1.25,
                                color=color, alpha=alpha, markerfacecolor=marker_fill,
                                markeredgecolor=color, markeredgewidth=0.9, label="_nolegend_",
                            )
                    if self.mcd_fit_zero_chk.isChecked() and metric_name == "mean":
                        for branch in ("B increasing", "B decreasing"):
                            b_trace, values = traces[branch][f"corrected_{metric_name}"]
                            fit_mask = np.isfinite(b_trace) & np.isfinite(values) & (np.abs(b_trace) <= float(self.mcd_fit_b_window_spin.value()))
                            if np.count_nonzero(fit_mask) >= 2:
                                slope, intercept = np.polyfit(b_trace[fit_mask], values[fit_mask], 1)
                                trace_ax.plot(b_trace, slope * b_trace + intercept, ":", color=color, lw=1.1, label="_nolegend_")
                trace_ax.axhline(0, color="#555", lw=0.7)
                trace_ax.set_title(f"MCD(B): E = {e0:.6g} eV", pad=3)
                trace_ax.set_xlabel("B field (T)")
                trace_ax.set_ylabel("MCD (mean / absolute mean)", labelpad=10)
                trace_ax.grid(alpha=0.25)
                if self.mcd_show_integral_chk.isChecked():
                    integral_ax.set_ylabel("Integrated MCD (eV)", labelpad=2)
                else:
                    integral_ax.set_yticks([]); integral_ax.spines["right"].set_visible(False)
                branch_legend = trace_ax.legend(
                    [
                        Line2D([0], [0], color="#333", marker="o", markerfacecolor="#333", lw=1.15),
                        Line2D([0], [0], color="#333", marker="o", markerfacecolor="white", lw=1.15, ls="--"),
                    ],
                    ["B increasing", "B decreasing"],
                    title="Branch", fontsize=5.8, title_fontsize=6.0, frameon=True, framealpha=0.88, loc="upper left",
                )
                trace_ax.add_artist(branch_legend)
                metric_handles: list[Line2D] = []
                metric_labels: list[str] = []
                for metric_name, label, color, visible in trace_specs:
                    if not visible:
                        continue
                    metric_handles.append(Line2D([0], [0], color=color, lw=1.5))
                    metric_labels.append(f"{label} (right axis)" if metric_name == "integral" else label)
                if metric_handles:
                    trace_ax.legend(metric_handles, metric_labels, title="Metric", fontsize=5.8, title_fontsize=6.0, frameon=True, framealpha=0.88, loc="lower right")
                self._mcd_heatmap_ax, self._mcd_spectrum_ax, self._mcd_trace_ax = heat_ax, linecut_ax, trace_ax
                self._mcd_integral_ax = integral_ax
                self._mcd_colorbar_ax = cax
            elif mode == "SHG Processing" and self.loaded.shg_result is not None:
                self._pl_heatmap_ax = None
                self._pl_spectrum_ax = None
                self._pl_last_plot_cube = None
                if self.loaded.shg_compare and self.loaded.shg_result_b is not None:
                    self._plot_shg_comparison(self.loaded.shg_result, self.loaded.shg_result_b)
                else:
                    self._plot_shg_result(self.loaded.shg_result)
            elif mode == "Power Dependent" and self.loaded.cube is not None:
                self._pl_heatmap_ax = None
                self._pl_spectrum_ax = None
                self._pl_last_plot_cube = None
                self._pl_gate_line = None
                if self._power_view() == "VP":
                    _kk_cube, _kkp_cube, vp_cube, _kk_records, _kkp_records, _kk_key, _kkp_key, background, _pairing, _pairs = self._power_vp_payload()
                    plot_cube = vp_cube
                    params_base = self._make_params(mode, plot_cube)
                    params = HeatmapParams(
                        title=vp_cube.title,
                        xlabel=params_base.xlabel,
                        ylabel=vp_cube.gate_label,
                        cbar_label="VP",
                        vmin=-1.0,
                        vmax=1.0,
                        xlim=params_base.xlim,
                        ylim=params_base.ylim,
                        cmap="RdBu_r",
                        log_scale=False,
                        y_axis_log=params_base.y_axis_log,
                        center_zero=True,
                        clip_outliers=False,
                    )
                    self._power_set_background_spin_silent(background)
                    display_cube, true_power, display_power = self._display_power_cube(plot_cube)
                    gs = self.figure.add_gridspec(
                        nrows=2,
                        ncols=2,
                        width_ratios=[1.0, 0.035],
                        height_ratios=[1.0, 0.95],
                        wspace=0.12,
                        hspace=0.30,
                    )
                    ax1 = self.figure.add_subplot(gs[0, 0])
                    cax = self.figure.add_subplot(gs[0, 1])
                    ax2 = self.figure.add_subplot(gs[1, 0], sharex=ax1)
                    render = plot_heatmap(ax1, display_cube, params)
                    self._apply_power_tick_labels(ax1, true_power, display_power)
                    self._add_heatmap_colorbar(render, cax, label=params.cbar_label)
                    self._power_heatmap_ax = ax1
                    self._power_heatmap_axes = {"VP": ax1}
                    self._power_spectrum_ax = ax2
                    self._power_last_plot_cube = plot_cube
                    self._power_active_cubes = {"VP": plot_cube}
                    self._power_active_export_cube = plot_cube
                    self._power_active_records = ()
                    self._update_power_compare_spectrum_and_lines({"VP": plot_cube})
                else:
                    role_compare = self._power_has_distinct_role_groups()
                    role_titles: dict[str, str] = {}
                    if role_compare:
                        kk_result, kkp_result, kk_key, kkp_key = self._power_role_payload()
                        background = self._power_background_value([kk_result.cube, kkp_result.cube])
                        cubes = {
                            "KK": self._power_corrected_cube(kk_result.cube, background=background),
                            "KKp": self._power_corrected_cube(kkp_result.cube, background=background),
                        }
                        plot_cube = cubes["KK"]
                        role_titles = {"KK": power_group_title(kk_key), "KKp": power_group_title(kkp_key)}
                    else:
                        background = self._power_background_value([self.loaded.cube])
                        plot_cube = self._power_corrected_cube(self.loaded.cube, background=background)
                        cubes = {"Power": plot_cube}
                    params = self._make_params(mode, plot_cube)
                    if role_compare:
                        kk_z = np.asarray(cubes["KK"].Z, float)
                        kkp_z = np.asarray(cubes["KKp"].Z, float)
                        combined_vmin = float(min(np.nanmin(kk_z), np.nanmin(kkp_z)))
                        combined_vmax = float(max(np.nanmax(kk_z), np.nanmax(kkp_z)))
                        new_vmin = params.vmin if self._mode_fix_value(mode, "vmin") else combined_vmin
                        new_vmax = params.vmax if self._mode_fix_value(mode, "vmax") else combined_vmax
                        params = HeatmapParams(**{**params.__dict__, "vmin": new_vmin, "vmax": new_vmax})
                    self._power_set_background_spin_silent(background)
                    n = len(cubes)
                    gs = self.figure.add_gridspec(
                        nrows=2,
                        ncols=n + 1,
                        width_ratios=([1.0] * n) + [0.035],
                        height_ratios=[1.0, 0.95],
                        wspace=0.12,
                        hspace=0.30,
                    )
                    heat_axes = [self.figure.add_subplot(gs[0, idx]) for idx in range(n)]
                    cax = self.figure.add_subplot(gs[0, n])
                    ax2 = self.figure.add_subplot(gs[1, :n], sharex=heat_axes[0])
                    renders: list[HeatmapRender] = []
                    for ax, (key, cube) in zip(heat_axes, cubes.items()):
                        display_cube, true_power, display_power = self._display_power_cube(cube)
                        panel_params = HeatmapParams(**{**params.__dict__, "title": role_titles.get(key, key) if role_compare else cube.title})
                        render = plot_heatmap(ax, display_cube, panel_params)
                        self._apply_power_tick_labels(ax, true_power, display_power)
                        renders.append(render)
                        self._power_heatmap_axes[key] = ax
                    if renders:
                        self._add_heatmap_colorbar(renders[0], cax, label=params.cbar_label)
                    self._power_heatmap_ax = heat_axes[0]
                    self._power_spectrum_ax = ax2
                    self._power_last_plot_cube = plot_cube
                    self._power_active_cubes = cubes
                    self._power_active_export_cube = plot_cube
                    self._power_active_records = tuple(self.loaded.power_records)
                    self._update_power_compare_spectrum_and_lines(cubes)
            elif mode == "Compare" and self.loaded.compare_cubes:
                self._pl_heatmap_ax = None
                self._pl_spectrum_ax = None
                self._pl_last_plot_cube = None
                raw_cubes = {
                    key: self.loaded.compare_cubes[key]
                    for key in COMPARE_PANEL_ORDER
                    if key in self.loaded.compare_cubes
                }
                if len(raw_cubes) < 2:
                    raise ValueError("Compare mode needs at least two visible channels.")
                first = next(iter(raw_cubes.values()))
                params = self._make_params(mode, first)
                source_files = self._cmp_source_mapping()
                background = self._cmp_background_value(raw_cubes)
                self._cmp_update_title_previews()
                if self._cmp_is_vp_view():
                    vp_cube = self._cmp_vp_cube(raw_cubes, source_files, background=background)
                    vp_params = HeatmapParams(
                        title=vp_cube.title,
                        xlabel=params.xlabel,
                        ylabel=vp_cube.gate_label,
                        cbar_label="VP",
                        vmin=-1.0,
                        vmax=1.0,
                        xlim=params.xlim,
                        ylim=params.ylim,
                        cmap="RdBu_r",
                        log_scale=False,
                        center_zero=True,
                        clip_outliers=False,
                    )
                    gs = self.figure.add_gridspec(
                        nrows=2,
                        ncols=2,
                        width_ratios=[1.0, 0.035],
                        height_ratios=[1.0, 0.95],
                        wspace=0.12,
                        hspace=0.30,
                    )
                    heat_ax = self.figure.add_subplot(gs[0, 0])
                    cax = self.figure.add_subplot(gs[0, 1])
                    line_ax = self.figure.add_subplot(gs[1, 0], sharex=heat_ax)
                    render = plot_heatmap(heat_ax, vp_cube, vp_params)
                    heat_ax.text(
                        0.98,
                        0.98,
                        "VP",
                        transform=heat_ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=10,
                        fontweight="bold",
                        color="#111",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.78),
                        zorder=40,
                    )
                    self._add_heatmap_colorbar(render, cax, label="VP")
                    gate_used, y = nearest_gate_spectrum(vp_cube, float(self.cmp_spins["gate"].value()))
                    x = np.asarray(vp_cube.energy, float).ravel()
                    line_ax.plot(x, np.asarray(y, float), linewidth=1.3, color="#1f77b4", label="VP")
                    line_ax.axhline(0.0, color="#333", linewidth=0.9, alpha=0.55)
                    line_ax.set_title(f"VP Linecut @ {gate_used:.6g} V")
                    line_ax.set_xlabel("Photon Energy (eV)")
                    line_ax.set_ylabel("VP (%)")
                    line_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
                    line_ax.set_ylim(-1.05, 1.05)
                    line_ax.set_xlim(self._safe_spectrum_xlim(x, params.xlim))
                    line_ax.grid(alpha=0.25)
                    self._cmp_heatmap_axes["VP"] = heat_ax
                    self._cmp_active_cubes = {"VP": vp_cube}
                    self._cmp_linecut_ax = line_ax
                    self._set_cmp_gate_spin_value(gate_used)
                    self._ensure_cmp_gate_lines({"VP": vp_cube}, gate_used)
                else:
                    cubes = self._cmp_corrected_cubes(raw_cubes, source_files, background=background)
                    n = len(cubes)
                    if n <= 2:
                        gs = self.figure.add_gridspec(
                            nrows=2,
                            ncols=n + 1,
                            width_ratios=([1.0] * n) + [0.035],
                            height_ratios=[1.0, 0.95],
                            wspace=0.12,
                            hspace=0.30,
                        )
                        heat_axes = [self.figure.add_subplot(gs[0, idx]) for idx in range(n)]
                        line_ax = self.figure.add_subplot(gs[1, :n], sharex=heat_axes[0])
                        cax = self.figure.add_subplot(gs[0, n])
                    else:
                        gs = self.figure.add_gridspec(
                            nrows=3,
                            ncols=3,
                            width_ratios=[1.0, 1.0, 0.035],
                            height_ratios=[1.0, 1.0, 0.95],
                            wspace=0.12,
                            hspace=0.30,
                        )
                        heat_axes = [
                            self.figure.add_subplot(gs[0, 0]),
                            self.figure.add_subplot(gs[0, 1]),
                            self.figure.add_subplot(gs[1, 0]),
                            self.figure.add_subplot(gs[1, 1]),
                        ]
                        line_ax = self.figure.add_subplot(gs[2, :2], sharex=heat_axes[0])
                        cax = self.figure.add_subplot(gs[0:2, 2])
                    renders: list[HeatmapRender] = []
                    for ax, key in zip(heat_axes, cubes.keys()):
                        render = plot_compare_panel(ax, key, cubes[key], params)
                        renders.append(render)
                        self._cmp_heatmap_axes[key] = ax
                    if renders:
                        self._add_heatmap_colorbar(renders[0], cax, label="PL corr. (a.u.)")
                    gate_used = self._plot_compare_linecut(
                        line_ax,
                        cubes,
                        gate_value=float(self.cmp_spins["gate"].value()),
                        xlim=(float(self.cmp_spins["xmin"].value()), float(self.cmp_spins["xmax"].value())),
                    )
                    self._cmp_active_cubes = cubes
                    self._cmp_linecut_ax = line_ax
                    self._set_cmp_gate_spin_value(gate_used)
                    self._ensure_cmp_gate_lines(cubes, gate_used)
            else:
                raise ValueError("No loaded data to plot.")

            if mode == "Compare":
                self.figure.tight_layout()
            self.canvas.draw_idle()
            self._last_plot_params_key = plot_key
            self._last_plot_cube = plot_cube if mode == "DRR" else None
            self.last_plotted_mode = mode
            self._set_stage("Plotted")
            self._update_action_states()
            if mode == "DRR":
                self._status(
                    f"Plot: baseline={self._drr_baseline_key()}, deriv={self.drr_derivative_combo.currentText()}, "
                    f"SG(win={int(self.drr_sg_window_spin.value())}, order={int(self.drr_sg_poly_spin.value())})"
                )
            else:
                self._status(f"Plotted {mode}.")
            if not auto:
                self._append_log(f"Plotted {mode}.")
        except Exception as exc:
            self._show_error(str(exc))

    def _drr_baseline_key(self) -> str:
        text = self.drr_baseline_combo.currentText()
        if text == "Self (first frame)":
            return "self_first"
        if text == "External":
            return f"external_{self.drr_baseline_combine_combo.currentText()}"
        return "self_last"

    def _read_drr_params(self) -> Dict[str, Any]:
        s = self.drr_spins
        return {
            "baseline_mode": self.drr_baseline_combo.currentText(),
            "baseline_which": self.drr_baseline_combine_combo.currentText(),
            "baseline_files": tuple(self.drr_baseline_files_manual),
            "selected_files": tuple(self.drr_selected_files),
            "y_axis_spec": self._selected_y_axis_spec("drr"),
            "derivative": self.drr_derivative_combo.currentText(),
            "sg_window": int(self.drr_sg_window_spin.value()),
            "sg_poly": int(self.drr_sg_poly_spin.value()),
            "cmap": self._resolved_cmap(self.drr_cmap),
            "vmin": float(s["vmin"].value()),
            "vmax": float(s["vmax"].value()),
            "xmin": float(s["xmin"].value()),
            "xmax": float(s["xmax"].value()),
            "ymin": float(s["ymin"].value()),
            "ymax": float(s["ymax"].value()),
            "gate": float(s["gate"].value()),
            "log": bool(self.drr_log_chk.isChecked()),
            "clip": bool(self.drr_clip_chk.isChecked()),
            "center_zero": bool(self.drr_center_zero_chk.isChecked()),
        }

    def _current_plot_params_key(self, mode: str) -> tuple[Any, ...]:
        if mode == "SHG Processing":
            settings = self._shg_settings_from_ui()
            fit_settings = self._shg_fit_settings_from_ui()
            return (
                mode,
                self._shg_compare_mode(),
                self._shg_compare_files() if self._shg_compare_mode() else self._shg_selected_file(),
                self._shg_compare_background_files() if self._shg_compare_mode() else self._shg_background_file(),
                *tuple(settings.to_dict().values()),
                *tuple(fit_settings.to_dict().values()),
                self.shg_compare_display_combo.currentText(),
                self.shg_spectrum_view_combo.currentText(),
                float(self.shg_angle_cursor_spin.value()),
            )
        if mode == "Power Dependent":
            return (
                mode,
                self._power_selected_group_key(),
                self._power_view(),
                self._power_role_group_key("KK"),
                self._power_role_group_key("KKp"),
                self._power_pairing_mode(),
                self.power_axis_scale_combo.currentText(),
                bool(self._power_background_auto_enabled()),
                float(self._power_background_value(
                    [self.loaded.cube] if self.loaded and self.loaded.mode == "Power Dependent" and self.loaded.cube is not None else None
                )),
                self._resolved_cmap(self.power_cmap),
                float(self.power_spins["vmin"].value()),
                float(self.power_spins["vmax"].value()),
                float(self.power_spins["xmin"].value()),
                float(self.power_spins["xmax"].value()),
                float(self.power_spins["ymin"].value()),
                float(self.power_spins["ymax"].value()),
                float(self.power_spins["gate"].value()),
                bool(self.power_log_chk.isChecked()),
                bool(self.power_clip_chk.isChecked()),
                self._split_scale_key("power"),
            )
        if mode == "Compare":
            return (
                mode,
                tuple(self._cmp_current_mapping().items()),
                tuple(self._cmp_visible_channels()),
                self._cmp_view_mode(),
                bool(self._cmp_background_auto_enabled()),
                float(
                    self._cmp_background_value(
                        self.loaded.compare_cubes
                        if self.loaded and self.loaded.mode == "Compare" and self.loaded.compare_cubes
                        else None
                    )
                ),
                self._current_y_axis_spec_for_mode(mode),
                self._resolved_cmap(self.cmp_cmap),
                float(self.cmp_spins["vmin"].value()),
                float(self.cmp_spins["vmax"].value()),
                float(self.cmp_spins["xmin"].value()),
                float(self.cmp_spins["xmax"].value()),
                float(self.cmp_spins["ymin"].value()),
                float(self.cmp_spins["ymax"].value()),
                float(self.cmp_spins["gate"].value()),
                bool(self.cmp_log_chk.isChecked()),
                bool(self.cmp_clip_chk.isChecked()),
                self._split_scale_key("cmp"),
            )
        if mode != "DRR":
            return (mode, int(self.tabs.currentIndex()), self.last_plotted_mode, self._current_y_axis_spec_for_mode(mode))
        p = self._read_drr_params()
        return (
            "DRR", p["baseline_mode"], p["baseline_which"], p["baseline_files"], p["selected_files"],
            p["y_axis_spec"], p["derivative"], p["sg_window"], p["sg_poly"], p["cmap"], p["vmin"], p["vmax"],
            p["xmin"], p["xmax"], p["ymin"], p["ymax"], p["gate"], p["log"], p["clip"], p["center_zero"],
            self._split_scale_key("drr"),
        )

    def _is_drr_gate_only_change(self, new_key: tuple[Any, ...]) -> bool:
        if self._last_plot_params_key is None or self._last_plot_cube is None:
            return False
        if len(new_key) != len(self._last_plot_params_key):
            return False
        gate_idx = 16
        return (
            new_key[:gate_idx] == self._last_plot_params_key[:gate_idx]
            and new_key[gate_idx + 1 :] == self._last_plot_params_key[gate_idx + 1 :]
            and new_key[gate_idx] != self._last_plot_params_key[gate_idx]
        )

    def _ensure_loaded_matches_drr_params(self) -> bool:
        if not self.loaded or self.loaded.mode != "DRR":
            return False
        p = self._read_drr_params()
        selected = list(p["selected_files"])
        baselines = list(p["baseline_files"])
        baseline_text = p["baseline_mode"]
        y_axis_spec = str(p["y_axis_spec"])
        which_map = {
            "Last frame of each baseline file": "last",
            "First frame of each baseline file": "first",
            "Average all baseline files": "all",
        }
        baseline_which = which_map.get(str(p["baseline_which"]), "last")
        needs_reload = (
            selected != list(self.loaded.selected_files)
            or baseline_text != self.loaded.drr_baseline_text
            or baseline_which != self.loaded.drr_baseline_which
            or baselines != list(self.loaded.baseline_files)
            or y_axis_spec != getattr(self.loaded, "y_axis_spec", "auto")
        )
        if needs_reload:
            self._reject_mixed_xlsx_selection(selected)
            if selected and data_io.is_xlsx_map_file(selected[0]):
                cube = data_io.load_drr_map_cube(self.current_folder, selected[0], y_axis=y_axis_spec)
                mode_label = "DR/R Map"
                baselines = []
            elif baseline_text == "External":
                if not baselines:
                    raise ValueError("External DRR mode requires baseline files.")
                cube = data_io.load_drr_external_cube(
                    self.current_folder,
                    selected,
                    baselines,
                    baseline_which=baseline_which,
                    y_axis=y_axis_spec,
                    derivative=None,
                )
                mode_label = "DR/R External"
            else:
                cube = data_io.load_drr_self_cube(
                    self.current_folder,
                    selected,
                    use_first_frame=(baseline_text == "Self (first frame)"),
                    y_axis=y_axis_spec,
                    derivative=None,
                )
                mode_label = "DR/R Self"
            self.loaded = LoadedState(
                mode="DRR",
                folder=self.current_folder,
                primary_file=(selected[0] if selected else None),
                selected_files=selected,
                baseline_files=baselines,
                cube=cube,
                drr_mode_label=mode_label,
                drr_derivative_label=self.drr_derivative_combo.currentText(),
                drr_baseline_text=baseline_text,
                drr_baseline_which=baseline_which,
                y_axis_spec=y_axis_spec,
            )
            self._last_plot_cube = None
            self._last_plot_params_key = None
            self._apply_auto_limits_for_loaded()
            return True
        return False

    def _ensure_gate_line(self, cube: DataCube, gate_value: float) -> None:
        if self._drr_heatmap_ax is None:
            return
        gate = np.asarray(cube.gate, float).ravel()
        gate_clamped = float(np.clip(gate_value, float(np.nanmin(gate)), float(np.nanmax(gate))))
        if self._gate_line is None or getattr(self._gate_line, "axes", None) is not self._drr_heatmap_ax:
            self._gate_line = self._drr_heatmap_ax.axhline(
                y=gate_clamped,
                lw=1.2,
                alpha=0.9,
                color="#222",
                linestyle="--",
                zorder=20,
            )
        else:
            self._gate_line.set_ydata([gate_clamped, gate_clamped])
            self._gate_line.set_linestyle("--")

    def _update_drr_spectrum_and_gate_line(self, cube: DataCube) -> None:
        if self._drr_spectrum_ax is None:
            return
        gate_value = float(self.drr_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        self._drr_spectrum_ax.clear()
        self._drr_spectrum_ax.plot(x, np.asarray(y, float), linewidth=1.3)
        self._drr_spectrum_ax.set_title(f"Spectrum @ {gate_used:.6g} V")
        self._drr_spectrum_ax.set_xlabel("Photon Energy (eV)")
        self._drr_spectrum_ax.set_ylabel(cube.cbar_label)
        self._drr_spectrum_ax.grid(alpha=0.25)
        xlim = self._safe_spectrum_xlim(
            x,
            (float(self.drr_spins["xmin"].value()), float(self.drr_spins["xmax"].value())),
        )
        self._drr_spectrum_ax.set_xlim(xlim)
        self._auto_scale_spectrum_y(self._drr_spectrum_ax, x, y, xlim)
        self._set_drr_gate_spin_value(gate_used)
        self._ensure_gate_line(cube, gate_used)
        self._draw_drr_analysis_overlays(cube, gate_used, x, np.asarray(y, float))
        self._update_drr_analysis_text(gate_used, x, np.asarray(y, float))
        self.canvas.draw_idle()

    def _ensure_cmp_gate_lines(self, cubes: Dict[str, DataCube], gate_value: float) -> None:
        active_keys = set(cubes.keys())
        for key in list(self._cmp_gate_lines.keys()):
            if key in active_keys and key in self._cmp_heatmap_axes:
                continue
            line = self._cmp_gate_lines.pop(key, None)
            if line is not None:
                try:
                    line.remove()
                except Exception:
                    pass
        for key, cube in cubes.items():
            ax = self._cmp_heatmap_axes.get(key)
            if ax is None:
                continue
            gate = np.asarray(cube.gate, float).ravel()
            gate_clamped = float(np.clip(gate_value, float(np.nanmin(gate)), float(np.nanmax(gate))))
            line = self._cmp_gate_lines.get(key)
            if line is None or getattr(line, "axes", None) is not ax:
                self._cmp_gate_lines[key] = ax.axhline(
                    y=gate_clamped,
                    lw=1.2,
                    alpha=0.95,
                    color="#222",
                    linestyle="--",
                    zorder=50,
                )
            else:
                line.set_ydata([gate_clamped, gate_clamped])
                line.set_linestyle("--")

    def _on_canvas_motion(self, event: Any) -> None:
        if self.last_plotted_mode == "SHG Processing":
            if event.inaxes is self._shg_angle_ax and event.xdata is not None:
                self.statusBar().showMessage(f"Hover measured angle: {float(event.xdata):.6g} deg")
            return
        if self.last_plotted_mode == "DRR":
            heatmap_ax = self._drr_heatmap_ax
            cube = self._last_plot_cube
        elif self.last_plotted_mode == "PL":
            heatmap_ax = self._pl_heatmap_ax
            cube = self._pl_last_plot_cube
        elif self.last_plotted_mode == "Compare":
            heatmap_ax = event.inaxes if event.inaxes in set(self._cmp_heatmap_axes.values()) else None
            cube = None
            for key, ax in self._cmp_heatmap_axes.items():
                if ax is heatmap_ax:
                    cube = self._cmp_active_cubes.get(key)
                    break
        elif self.last_plotted_mode == "Power Dependent":
            heatmap_ax = event.inaxes if event.inaxes in set(self._power_heatmap_axes.values()) else self._power_heatmap_ax
            cube = None
            for key, ax in self._power_heatmap_axes.items():
                if ax is heatmap_ax:
                    cube = self._power_active_cubes.get(key)
                    break
            if cube is None:
                cube = self._power_last_plot_cube
        else:
            return
        if heatmap_ax is None or cube is None or event.inaxes is not heatmap_ax or event.ydata is None:
            return
        ygrid = np.asarray(cube.gate, float).ravel()
        if self.last_plotted_mode == "Power Dependent":
            _display_cube, true_power, display_power = self._display_power_cube(cube)
            idx = int(np.argmin(np.abs(display_power - float(event.ydata))))
            y = float(true_power[idx])
        else:
            y = float(np.clip(float(event.ydata), float(np.nanmin(ygrid)), float(np.nanmax(ygrid))))
        unit = "uW" if self.last_plotted_mode == "Power Dependent" else "V"
        label = "power" if self.last_plotted_mode == "Power Dependent" else "gate"
        self.statusBar().showMessage(f"Hover {label}: {y:.3f} {unit}")

    def _on_canvas_click(self, event: Any) -> None:
        if event.button != 1:
            return
        if (
            self.last_plotted_mode == "SHG Processing"
            and event.inaxes is self._shg_angle_ax
            and event.xdata is not None
            and self.loaded is not None
            and self.loaded.shg_result is not None
        ):
            angle = np.asarray(self.loaded.shg_result.measured_angle_deg, float)
            finite = np.flatnonzero(np.isfinite(angle))
            if finite.size:
                index = int(finite[np.argmin(np.abs(angle[finite] - float(event.xdata)))])
                blocked = self.shg_angle_cursor_spin.blockSignals(True)
                try:
                    self.shg_angle_cursor_spin.setValue(float(angle[index]))
                finally:
                    self.shg_angle_cursor_spin.blockSignals(blocked)
                self._plot_mode("SHG Processing")
            return
        # Manual peak delete by clicking near a marker on the bottom spectrum.
        if event.xdata is not None and self.last_plotted_mode == "DRR" and event.inaxes is self._drr_spectrum_ax:
            if self._remove_nearest_drr_peak(float(event.xdata)):
                return
        if (
            event.xdata is not None
            and event.ydata is not None
            and self.last_plotted_mode == "DRR"
            and event.inaxes is self._drr_heatmap_ax
        ):
            if self._remove_peak_from_drr_heatmap_click(float(event.xdata), float(event.ydata)):
                return
        if event.xdata is not None and self.last_plotted_mode == "PL" and event.inaxes is self._pl_spectrum_ax:
            if self._remove_nearest_pl_peak(float(event.xdata)):
                return
        if event.xdata is not None and self.last_plotted_mode == "Power Dependent" and event.inaxes is self._power_spectrum_ax:
            return
        if (
            event.xdata is not None
            and event.ydata is not None
            and self.last_plotted_mode == "PL"
            and event.inaxes is self._pl_heatmap_ax
        ):
            if self._remove_peak_from_pl_heatmap_click(float(event.xdata), float(event.ydata)):
                return

        if self.last_plotted_mode == "MCD":
            if event.inaxes is not self._mcd_heatmap_ax or event.ydata is None:
                return
            self._select_mcd_pair_at_b(float(event.ydata), clicked=True)
            return

        if self.last_plotted_mode == "DRR":
            if self._drr_heatmap_ax is None or self._last_plot_cube is None:
                return
            if event.inaxes is not self._drr_heatmap_ax or event.ydata is None:
                return
            ygrid = np.asarray(self._last_plot_cube.gate, float).ravel()
            idx = int(np.argmin(np.abs(ygrid - float(event.ydata))))
            gate = float(ygrid[idx])
            self.drr_spins["gate"].setValue(gate)
            self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
            return

        if self.last_plotted_mode == "Compare":
            if not self._cmp_active_cubes or event.ydata is None:
                return
            clicked_key = None
            for key, ax in self._cmp_heatmap_axes.items():
                if event.inaxes is ax:
                    clicked_key = key
                    break
            if clicked_key is None:
                return
            cube = self._cmp_active_cubes.get(clicked_key)
            if cube is None:
                return
            ygrid = np.asarray(cube.gate, float).ravel()
            idx = int(np.argmin(np.abs(ygrid - float(event.ydata))))
            self._set_cmp_gate_spin_value(float(ygrid[idx]))
            self._plot_mode("Compare")
            return

        if self.last_plotted_mode == "Power Dependent":
            if not self._power_active_cubes or event.ydata is None:
                return
            clicked_key = None
            for key, ax in self._power_heatmap_axes.items():
                if event.inaxes is ax:
                    clicked_key = key
                    break
            if clicked_key is None:
                return
            cube = self._power_active_cubes.get(clicked_key)
            if cube is None:
                return
            _display_cube, true_power, display_power = self._display_power_cube(cube)
            idx = int(np.argmin(np.abs(display_power - float(event.ydata))))
            power = float(true_power[idx])
            self._power_selected_row_index = idx
            self.power_spins["gate"].setValue(power)
            self._update_power_compare_spectrum_and_lines(self._power_active_cubes)
            return

        if self.last_plotted_mode != "PL" or self._pl_heatmap_ax is None or self._pl_last_plot_cube is None:
            return
        if event.inaxes is not self._pl_heatmap_ax or event.ydata is None:
            return
        ygrid = np.asarray(self._pl_last_plot_cube.gate, float).ravel()
        idx = int(np.argmin(np.abs(ygrid - float(event.ydata))))
        gate = float(ygrid[idx])
        self.pl_spins["gate"].setValue(gate)
        self._update_pl_spectrum_and_gate_line(self._pl_last_plot_cube)

    def _start_export(self, mode: str) -> None:
        if not self.loaded or self.loaded.mode != mode:
            self._show_error("Load and plot data before exporting.")
            return
        if self.last_plotted_mode != mode:
            self._show_error("Plot/Update before exporting.")
            return
        self._invalidate_export_move_sources()

        power_vp_payload = None
        params_intensity: HeatmapParams | None = None
        power_records = tuple(self.loaded.power_records) if self.loaded and self.loaded.mode == "Power Dependent" else ()
        drr_deriv: int | None = None
        drr_used_win = 0
        drr_poly = 0
        if mode == "SHG Processing" and self.loaded.shg_result is not None:
            params = None
            export_cube = None
        elif mode == "MCD" and self.loaded.mcd_result is not None:
            export_cube = self.loaded.mcd_result.cube(self.mcd_map_combo.currentText())
            params = self._make_params(mode, export_cube)
        elif mode in {"PL", "DRR", "Power Dependent"} and self.loaded.cube is not None:
            if mode == "DRR":
                export_cube, drr_deriv, drr_used_win, drr_poly = self._drr_cube_with_metadata()
                params = self._make_params(mode, export_cube)
            elif mode == "Power Dependent":
                if self._power_view() == "VP":
                    power_vp_payload = self._power_vp_payload()
                    _kk_cube, _kkp_cube, vp_cube, _kk_records, _kkp_records, _kk_key, _kkp_key, background, _pairing, _pairs = power_vp_payload
                    params_intensity = self._make_params(mode, _kk_cube)
                    _kk_z = np.asarray(_kk_cube.Z, float)
                    _kkp_z = np.asarray(_kkp_cube.Z, float)
                    _comb_vmin = float(min(np.nanmin(_kk_z), np.nanmin(_kkp_z)))
                    _comb_vmax = float(max(np.nanmax(_kk_z), np.nanmax(_kkp_z)))
                    params_intensity = HeatmapParams(**{**params_intensity.__dict__,
                        "vmin": params_intensity.vmin if self._mode_fix_value(mode, "vmin") else _comb_vmin,
                        "vmax": params_intensity.vmax if self._mode_fix_value(mode, "vmax") else _comb_vmax,
                    })
                    export_cube = vp_cube
                    params_base = self._make_params(mode, vp_cube)
                    params = HeatmapParams(
                        title=vp_cube.title,
                        xlabel=params_base.xlabel,
                        ylabel=vp_cube.gate_label,
                        cbar_label="VP",
                        vmin=-1.0,
                        vmax=1.0,
                        xlim=params_base.xlim,
                        ylim=params_base.ylim,
                        cmap="RdBu_r",
                        log_scale=False,
                        y_axis_log=params_base.y_axis_log,
                        center_zero=True,
                        clip_outliers=False,
                    )
                    self._power_set_background_spin_silent(background)
                else:
                    if self._power_has_distinct_role_groups():
                        kk_result, kkp_result, kk_key, kkp_key = self._power_role_payload()
                        background = self._power_background_value([kk_result.cube, kkp_result.cube])
                        kk_cube = self._power_corrected_cube(kk_result.cube, background=background)
                        kkp_cube = self._power_corrected_cube(kkp_result.cube, background=background)
                        export_cube = kk_cube
                        params = self._make_params(mode, kk_cube)
                        _kk_z = np.asarray(kk_cube.Z, float)
                        _kkp_z = np.asarray(kkp_cube.Z, float)
                        _comb_vmin = float(min(np.nanmin(_kk_z), np.nanmin(_kkp_z)))
                        _comb_vmax = float(max(np.nanmax(_kk_z), np.nanmax(_kkp_z)))
                        params = HeatmapParams(**{**params.__dict__,
                            "vmin": params.vmin if self._mode_fix_value(mode, "vmin") else _comb_vmin,
                            "vmax": params.vmax if self._mode_fix_value(mode, "vmax") else _comb_vmax,
                        })
                        power_vp_payload = (
                            kk_cube,
                            kkp_cube,
                            None,
                            kk_result.records,
                            kkp_result.records,
                            kk_key,
                            kkp_key,
                            background,
                            "intensity",
                            (),
                        )
                    else:
                        background = self._power_background_value([self.loaded.cube])
                        export_cube = self._power_corrected_cube(self.loaded.cube, background=background)
                        params = self._make_params(mode, export_cube)
                        power_records = tuple(self.loaded.power_records)
                    self._power_set_background_spin_silent(background)
            else:
                export_cube = self.loaded.cube
                params = self._make_params(mode, export_cube)
        elif mode == "Compare" and self.loaded.compare_cubes:
            first = next(iter(self.loaded.compare_cubes.values()))
            params = self._make_params("Compare", first)
            export_cube = None
        else:
            self._show_error("Nothing to export for this mode.")
            return

        if mode == "PL":
            log_split = params.split_scale
            if log_split is not None and (
                log_split.left_vmin <= 0.0 or log_split.right_vmin <= 0.0
            ):
                log_split = None
            options = ExportOptions(
                mode=mode,
                params=params,
                params_linear=HeatmapParams(**{**params.__dict__, "log_scale": False}),
                params_log=HeatmapParams(
                    **{**params.__dict__, "log_scale": True, "split_scale": log_split}
                ),
                cleanup_verified_sources=bool(self.clean_verified_sources_chk.isChecked()),
            )
        elif mode == "DRR":
            options = ExportOptions(
                mode=mode,
                params=params,
                drr_cube=export_cube,
                drr_derivative_order=drr_deriv,
                drr_sg_window=drr_used_win,
                drr_sg_polyorder=drr_poly,
                drr_sg_mode_label="More correct (regrid)",
                cleanup_verified_sources=bool(self.clean_verified_sources_chk.isChecked()),
            )
        elif mode == "Power Dependent":
            options = ExportOptions(
                mode=mode,
                params=params,
                params_intensity=params_intensity,
                drr_cube=export_cube,
                power_view=("Intensity Compare" if power_vp_payload and power_vp_payload[8] == "intensity" else self._power_view()),
                power_background=float(self.power_background_spin.value()),
                power_axis_log=self._power_axis_log(),
                power_kk_group_key=(power_vp_payload[5] if power_vp_payload else ""),
                power_kkp_group_key=(power_vp_payload[6] if power_vp_payload else ""),
                power_kk_cube=(power_vp_payload[0] if power_vp_payload else None),
                power_kkp_cube=(power_vp_payload[1] if power_vp_payload else None),
                power_vp_cube=(power_vp_payload[2] if power_vp_payload else None),
                power_kk_records=(power_vp_payload[3] if power_vp_payload else ()),
                power_kkp_records=(power_vp_payload[4] if power_vp_payload else ()),
                power_pairing_mode=(power_vp_payload[8] if power_vp_payload else self._power_pairing_mode()),
                power_stage_pairs=(power_vp_payload[9] if power_vp_payload else ()),
            )
        elif mode == "SHG Processing":
            options = ExportOptions(
                mode=mode,
                params=None,
                shg_settings=self._shg_settings_from_ui(),
                shg_fit_settings=self._shg_fit_settings_from_ui(),
            )
        elif mode == "MCD":
            options = ExportOptions(
                mode=mode, params=params, drr_cube=export_cube,
                mcd_map_name=self.mcd_map_combo.currentText(),
                mcd_window_center_ev=float(self.mcd_window_center_spin.value()),
                mcd_window_width_mev=float(self.mcd_window_width_spin.value()),
                mcd_window_metric={
                    "Field-signed absolute mean": "field_signed_absolute_mean",
                    "Signed mean": "mean",
                    "Signed integral": "integral",
                    "Unsigned absolute mean (diagnostic)": "absolute_mean",
                }.get(self.mcd_window_metric_combo.currentText(), "mean"),
                mcd_settings=self.loaded.mcd_settings,
                mcd_show_raw=bool(self.mcd_show_raw_chk.isChecked()),
                mcd_show_signed_mean=bool(self.mcd_show_signed_mean_chk.isChecked()),
                mcd_show_absolute_mean=bool(self.mcd_show_absolute_mean_chk.isChecked()),
                mcd_show_unsigned_absolute_mean=bool(self.mcd_show_unsigned_absolute_mean_chk.isChecked()),
                mcd_show_integral=bool(self.mcd_show_integral_chk.isChecked()),
                mcd_fit_near_zero=bool(self.mcd_fit_zero_chk.isChecked()),
                mcd_fit_window_t=float(self.mcd_fit_b_window_spin.value()),
            )
        else:
            compare_background = self._cmp_background_value(
                self.loaded.compare_cubes
                if self.loaded and self.loaded.mode == "Compare" and self.loaded.compare_cubes
                else None
            )
            options = ExportOptions(
                mode=mode,
                params=params,
                compare_scale_tag=self._cmp_scale_tag(),
                compare_clip=bool(self.cmp_clip_chk.isChecked()),
                compare_gate=float(self.cmp_spins["gate"].value()),
                compare_background=compare_background,
                compare_export_vp=True,
                cleanup_verified_sources=bool(self.clean_verified_sources_chk.isChecked()),
            )

        worker = Worker(self._export_task, self.loaded, options)
        worker.signals.log.connect(self._append_log)
        worker.signals.result.connect(self._on_export_done)
        worker.signals.error.connect(self._show_error)
        self._set_stage("Exporting...")
        self.thread_pool.start(worker)

    def _export_task(self, loaded: LoadedState, options: ExportOptions, *, progress: Signal, log: Signal) -> dict:
        mode = options.mode
        folder = loaded.folder
        out_folder: str | None = None
        files_to_move: list[str] = []
        metadata_extra = {
            "provenance": [record.to_dict() for record in loaded.provenance_records]
        } if mode in {"PL", "DRR", "Compare"} else None
        if mode == "PL" and loaded.primary_file and options.params_linear is not None and options.params_log is not None:
            linear_cube = data_io.load_pl_cube(folder, loaded.primary_file, log_scale=False, y_axis=loaded.y_axis_spec)
            log_cube = data_io.load_pl_cube(folder, loaded.primary_file, log_scale=True, y_axis=loaded.y_axis_spec)
            if Path(loaded.primary_file).suffix.lower() == ".dat" and loaded.cube is not None:
                for cube in (linear_cube, log_cube):
                    cube.gate_label = loaded.cube.gate_label
                    cube.gate_unit = getattr(loaded.cube, "gate_unit", "")
                    cube.y_axis_semantic = getattr(loaded.cube, "y_axis_semantic", "")
            paths = export_pl_pngs_and_dat(
                folder,
                loaded.primary_file,
                cube_linear=linear_cube,
                cube_log=log_cube,
                params_linear=options.params_linear,
                params_log=options.params_log,
                processed_name=str(Path("Processed Data") / "PL"),
                metadata_input_files=(("measurement", loaded.primary_file),),
                metadata_extra=metadata_extra,
            )
            log.emit(f"Exported PNG: {paths['png_linear'].name}, {paths['png_log'].name}")
            log.emit(f"Exported DAT: {paths['dat'].name}")
            out_folder = str(paths["png_linear"].parent)
            files_to_move = [loaded.primary_file]
        elif mode == "DRR" and options.drr_cube is not None and loaded.primary_file:
            base = build_drr_export_base(
                loaded.primary_file,
                len(loaded.selected_files),
                loaded.drr_mode_label,
                options.drr_derivative_order,
                options.drr_sg_window,
                options.drr_sg_polyorder,
                options.drr_sg_mode_label,
            )
            drr_inputs = [("measurement", name) for name in loaded.selected_files]
            drr_inputs.extend(("background", name) for name in loaded.baseline_files)
            paths = export_drr_png_and_dat(
                folder,
                cube=options.drr_cube,
                params=options.params,
                export_base=base,
                processed_name=str(Path("Processed Data") / "DRR"),
                metadata_input_files=drr_inputs,
                metadata_processing={
                    "mode": loaded.drr_mode_label,
                    "baseline_selection": loaded.drr_baseline_text,
                    "baseline_which": loaded.drr_baseline_which,
                    "average_count": len(loaded.selected_files),
                    "average_method": "per-file dR/R, then nanmean",
                    "derivative_order": options.drr_derivative_order,
                    "savgol_window": options.drr_sg_window,
                    "savgol_polyorder": options.drr_sg_polyorder,
                    "derivative_grid": options.drr_sg_mode_label,
                    "y_axis": loaded.y_axis_spec,
                },
                metadata_extra=metadata_extra,
            )
            log.emit(f"Exported PNG: {paths['png'].name}")
            log.emit(f"Exported DAT: {paths['dat'].name}")
            out_folder = str(paths["png"].parent)
            files_to_move = list(loaded.selected_files) + list(loaded.baseline_files)
        elif mode == "MCD" and options.drr_cube is not None and loaded.mcd_result is not None and loaded.primary_file:
            package_dir = create_unique_package_dir(
                Path(folder) / "Processed Data" / "MCD",
                f"{Path(loaded.primary_file).stem}_MCD_"
                f"E{options.mcd_window_center_ev:.6f}eV_W{options.mcd_window_width_mev:.6g}meV",
            )
            package_subfolder = str(Path("Processed Data") / "MCD" / package_dir.name)
            base = f"{Path(loaded.primary_file).stem}_MCD_{options.mcd_map_name.replace(' ', '_')}"
            # Use the same title and colorbar presentation as a PL export:
            # the source-file stem is the title, while the MCD map identity is
            # carried by the export filename (for example, ``_MCD_Combo``).
            mcd_export_params = HeatmapParams(
                **{**options.params.__dict__, "title": Path(loaded.primary_file).stem}
            )
            paths = export_drr_png_and_dat(
                folder,
                cube=options.drr_cube,
                params=mcd_export_params,
                export_base=base,
                processed_name=package_subfolder,
                drr_style=False,
                metadata_input_files=[
                    ("measurement", name) for name in (loaded.selected_files or [loaded.primary_file])
                ] + [("background", name) for name in loaded.baseline_files],
                metadata_processing={
                    "map": options.mcd_map_name,
                    "window_center_ev": options.mcd_window_center_ev,
                    "window_width_mev": options.mcd_window_width_mev,
                    "window_metric": options.mcd_window_metric,
                },
                metadata_extra={"package": package_dir.name},
            )
            analysis_paths = export_mcd_analysis_bundle(
                loaded.mcd_result, str(paths["png"].parent), trace_map=options.mcd_map_name,
                center_ev=options.mcd_window_center_ev, width_mev=options.mcd_window_width_mev,
                metric=options.mcd_window_metric,
                settings=options.mcd_settings,
                show_raw=options.mcd_show_raw,
                show_signed_mean=options.mcd_show_signed_mean,
                show_field_signed_absolute_mean=options.mcd_show_absolute_mean,
                show_unsigned_absolute_mean=options.mcd_show_unsigned_absolute_mean,
                show_integral=options.mcd_show_integral,
                fit_near_zero=options.mcd_fit_near_zero,
                fit_window_t=options.mcd_fit_window_t,
                package_outputs=(
                    paths["png"],
                    paths["dat"],
                    paths["dat"].with_suffix(".metadata.json"),
                ),
            )
            log.emit(f"Exported PNG: {paths['png'].name}")
            log.emit(f"Exported DAT: {paths['dat'].name}")
            log.emit(f"Exported PNG: {analysis_paths['mcd_vs_b_png'].name}")
            log.emit(f"Exported CSV: {analysis_paths['mcd_vs_b_csv'].name}, {analysis_paths['pair_diagnostics'].name}")
            log.emit(f"Exported settings: {analysis_paths['settings'].name}")
            out_folder = str(paths["png"].parent)
            files_to_move = [loaded.primary_file]
        elif mode == "Power Dependent" and options.drr_cube is not None:
            if options.power_view == "VP":
                if options.power_kk_cube is None or options.power_kkp_cube is None or options.power_vp_cube is None:
                    raise ValueError("Power VP export requires KK, KKp, and VP cubes.")
                power_package = create_unique_package_dir(
                    Path(folder) / "Processed Data" / "Power Dependence",
                    f"{options.power_kk_group_key}_vs_{options.power_kkp_group_key}_VP",
                )
                power_subfolder = str(
                    Path("Processed Data") / "Power Dependence" / power_package.name
                )
                power_metadata = {
                    "package": power_package.name,
                    "comparison": {
                        "KK_group": options.power_kk_group_key,
                        "KKp_group": options.power_kkp_group_key,
                    },
                }
                paths = export_power_vp_pngs_and_dat(
                    folder,
                    kk_cube=options.power_kk_cube,
                    kkp_cube=options.power_kkp_cube,
                    vp_cube=options.power_vp_cube,
                    params=options.params,
                    params_intensity=options.params_intensity,
                    kk_group_key=options.power_kk_group_key,
                    kkp_group_key=options.power_kkp_group_key,
                    kk_records=options.power_kk_records,
                    kkp_records=options.power_kkp_records,
                    y_axis_log=options.power_axis_log,
                    background=options.power_background,
                    pairing_mode=options.power_pairing_mode,
                    stage_pairs=options.power_stage_pairs,
                    processed_name=power_subfolder,
                    metadata_extra=power_metadata,
                )
                log.emit(f"Exported {len(paths)} power VP files.")
                out_folder = str(next(iter(paths.values())).parent) if paths else str(Path(folder))
                files_to_move = list(
                    dict.fromkeys(
                        [getattr(record, "file_name", "") for record in options.power_kk_records]
                        + [getattr(record, "file_name", "") for record in options.power_kkp_records]
                    )
                )
            elif options.power_view == "Intensity Compare":
                if options.power_kk_cube is None or options.power_kkp_cube is None:
                    raise ValueError("Power intensity compare export requires KK and KKp cubes.")
                power_package = create_unique_package_dir(
                    Path(folder) / "Processed Data" / "Power Dependence",
                    f"{options.power_kk_group_key}_vs_{options.power_kkp_group_key}_IntensityCompare",
                )
                power_subfolder = str(
                    Path("Processed Data") / "Power Dependence" / power_package.name
                )
                power_metadata = {
                    "package": power_package.name,
                    "comparison": {
                        "KK_group": options.power_kk_group_key,
                        "KKp_group": options.power_kkp_group_key,
                    },
                }
                kk_params = HeatmapParams(**{**options.params.__dict__, "title": options.power_kk_cube.title, "cbar_label": options.power_kk_cube.cbar_label})
                kkp_params = HeatmapParams(**{**options.params.__dict__, "title": options.power_kkp_cube.title, "cbar_label": options.power_kkp_cube.cbar_label})
                kk_paths = export_power_series_png_and_dat(
                    folder,
                    cube=options.power_kk_cube,
                    params=kk_params,
                    records=options.power_kk_records,
                    group_key=f"KK_{options.power_kk_group_key}",
                    y_axis_log=options.power_axis_log,
                    background=options.power_background,
                    processed_name=power_subfolder,
                    metadata_extra=power_metadata,
                )
                kkp_paths = export_power_series_png_and_dat(
                    folder,
                    cube=options.power_kkp_cube,
                    params=kkp_params,
                    records=options.power_kkp_records,
                    group_key=f"KKp_{options.power_kkp_group_key}",
                    y_axis_log=options.power_axis_log,
                    background=options.power_background,
                    processed_name=power_subfolder,
                    metadata_extra=power_metadata,
                )
                log.emit(f"Exported PNG: {kk_paths['png'].name}, {kkp_paths['png'].name}")
                log.emit(f"Exported DAT: {kk_paths['dat'].name}, {kkp_paths['dat'].name}")
                out_folder = str(kk_paths["png"].parent)
                files_to_move = list(
                    dict.fromkeys(
                        [getattr(record, "file_name", "") for record in options.power_kk_records]
                        + [getattr(record, "file_name", "") for record in options.power_kkp_records]
                    )
                )
            else:
                power_package = create_unique_package_dir(
                    Path(folder) / "Processed Data" / "Power Dependence",
                    f"{options.power_group_key}_PowerDep",
                )
                power_subfolder = str(
                    Path("Processed Data") / "Power Dependence" / power_package.name
                )
                paths = export_power_series_png_and_dat(
                    folder,
                    cube=options.drr_cube,
                    params=options.params,
                    records=loaded.power_records,
                    group_key=loaded.power_group_key,
                    y_axis_log=options.power_axis_log,
                    background=options.power_background,
                    processed_name=power_subfolder,
                    metadata_extra={"package": power_package.name},
                )
                log.emit(f"Exported PNG: {paths['png'].name}")
                log.emit(f"Exported DAT: {paths['dat'].name}")
                out_folder = str(paths["png"].parent)
                files_to_move = list(loaded.selected_files)
        elif (
            mode == "SHG Processing"
            and loaded.shg_data is not None
            and loaded.shg_result is not None
            and options.shg_settings is not None
        ):
            if loaded.shg_compare:
                package_base = (
                    f"{Path(loaded.shg_data.source_file).stem}_vs_"
                    f"{Path(loaded.shg_data_b.source_file).stem}_SHG_twist"
                    if loaded.shg_data_b is not None
                    else f"{Path(loaded.shg_data.source_file).stem}_SHG_twist"
                )
            else:
                package_base = (
                    f"{Path(loaded.shg_data.source_file).stem}_SHG_"
                    f"{options.shg_settings.peak_center_nm:g}nm"
                )
            shg_package_dir = create_unique_package_dir(
                Path(folder) / "Processed Data" / "SHG", package_base
            )
            shg_processed_subfolder = str(
                Path("Processed Data") / "SHG" / shg_package_dir.name
            )
            if (
                loaded.shg_compare
                and loaded.shg_data_b is not None
                and loaded.shg_result_b is not None
                and loaded.shg_twist is not None
            ):
                paths = export_shg_twist_comparison(
                    folder,
                    reference_data=loaded.shg_data,
                    reference_result=loaded.shg_result,
                    sample_data=loaded.shg_data_b,
                    sample_result=loaded.shg_result_b,
                    settings=options.shg_settings,
                    twist=loaded.shg_twist,
                    processed_name=shg_processed_subfolder,
                )
                log.emit(f"Exported {len(paths)} SHG comparison and twist files.")
                out_folder = str(paths["combined_csv"].parent)
            elif loaded.shg_compare and loaded.shg_data_b is not None and loaded.shg_result_b is not None:
                reference_paths = export_shg_results(
                    folder,
                    data=loaded.shg_data,
                    result=loaded.shg_result,
                    settings=options.shg_settings,
                    fit=loaded.shg_fit,
                    processed_name=shg_processed_subfolder,
                )
                sample_paths = export_shg_results(
                    folder,
                    data=loaded.shg_data_b,
                    result=loaded.shg_result_b,
                    settings=options.shg_settings,
                    fit=loaded.shg_fit_b,
                    processed_name=shg_processed_subfolder,
                )
                paths = {"reference_csv": reference_paths["csv"], "sample_csv": sample_paths["csv"]}
                log.emit("Exported both SHG processed CSVs; twist summary unavailable because the fit is disabled or failed.")
                out_folder = str(reference_paths["csv"].parent)
            else:
                paths = export_shg_results(
                    folder,
                    data=loaded.shg_data,
                    result=loaded.shg_result,
                    settings=options.shg_settings,
                    fit=loaded.shg_fit,
                    processed_name=shg_processed_subfolder,
                )
                log.emit(f"Exported SHG CSV: {paths['csv'].name}")
                log.emit(f"Exported SHG settings: {paths['settings'].name}")
                out_folder = str(paths["csv"].parent)
            files_to_move = list(loaded.selected_files)
            if loaded.shg_result.background_file or (
                loaded.shg_result_b is not None and loaded.shg_result_b.background_file
            ):
                files_to_move.extend(loaded.baseline_files)
        elif mode == "Compare" and loaded.compare_cubes:
            paths = export_compare_panels(
                folder,
                cubes=loaded.compare_cubes,
                source_files=loaded.compare_sources,
                params=options.params,
                scale_tag=options.compare_scale_tag,
                clip_outliers=options.compare_clip,
                correction_background=options.compare_background,
                export_vp=options.compare_export_vp,
                processed_name=str(Path("Processed Data") / "Compare"),
                metadata_extra=metadata_extra,
            )
            log.emit(f"Exported {len(paths)} compare files.")
            out_folder = str(Path(paths[0]).parent) if paths else str(Path(folder))
            files_to_move = list(loaded.compare_sources.values()) if loaded.compare_sources else list(loaded.selected_files)
        else:
            raise ValueError("Nothing to export for this mode.")

        # Source files remain in the selected working folder after export.
        # Explicit manual archival remains available through the toolbar button.
        cleaned = 0
        if mode in {"PL", "DRR", "Compare"} and options.cleanup_verified_sources:
            for record in loaded.provenance_records:
                if cleanup_working_copy(record, folder):
                    cleaned += 1
            if cleaned:
                log.emit(f"Cleaned {cleaned} verified temporary source copy(s).")
        moved = 0
        return {
            "out_folder": out_folder or str(Path(folder)),
            "moved": moved,
            "source_files": list(dict.fromkeys(name for name in files_to_move if name)),
            "folder": folder,
            "auto_moved": False,
            "cleaned": cleaned,
        }

    def _on_export_done(self, result: object) -> None:
        if isinstance(result, dict):
            out_folder = str(result.get("out_folder", self.current_folder or ""))
            moved = int(result.get("moved", 0))
            source_files = list(result.get("source_files", []))
            folder = str(result.get("folder", self.current_folder or ""))
            auto_moved = bool(result.get("auto_moved", False))
            cleaned = int(result.get("cleaned", 0))
        else:
            out_folder = str(result)
            moved = 0
            source_files = []
            folder = self.current_folder or ""
            auto_moved = False
            cleaned = 0
        if moved > 0:
            self._refresh_file_lists()
        if source_files and not auto_moved:
            self._set_export_move_sources(folder, source_files)
        else:
            self._invalidate_export_move_sources()
        self._set_stage("Exported")
        suffix = f"; cleaned {cleaned} verified source copy(s)" if cleaned else ""
        self._status(f"Export completed: {out_folder}{suffix}")
    def _auto_update_check_enabled(self) -> bool:
        value = self.settings.value(self.SETTINGS_AUTO_UPDATE_CHECK, True)
        if value is None:
            return True
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}
    def _on_auto_update_check_toggled(self, checked: bool) -> None:
        self.settings.setValue(self.SETTINGS_AUTO_UPDATE_CHECK, bool(checked))
    def _schedule_automatic_update_check(self) -> None:
        if not self._auto_update_check_enabled():
            return
        QTimer.singleShot(1500, self._run_automatic_update_check)
    def _run_automatic_update_check(self) -> None:
        self._start_update_check(automatic=True)
    def _manual_check_updates(self) -> None:
        self._start_update_check(automatic=False)
    def _start_update_check(self, *, automatic: bool) -> None:
        if automatic and not self._auto_update_check_enabled():
            return
        worker = Worker(self._update_check_task)
        worker.signals.result.connect(lambda result: self._on_update_check_done(automatic, result))
        worker.signals.error.connect(lambda message: self._on_update_check_error(message, automatic))
        self.thread_pool.start(worker)
        if not automatic:
            self._status('Checking for updates...')
    def _update_check_task(self, *, progress, log) -> CheckResult:
        return check_for_update(__version__)
    def _on_update_check_done(self, automatic: bool, result: object) -> None:
        if not isinstance(result, CheckResult):
            return
        if result.status == 'error':
            self._pending_update = None
            self._hide_update_status_indicator()
            if not automatic:
                self._show_update_error(result.message)
            return
        if result.status == 'update_available':
            self._pending_update = result
            if automatic:
                self._show_update_status_indicator(result)
            else:
                self._show_update_available_dialog(result)
        else:
            self._pending_update = None
            self._hide_update_status_indicator()
            if not automatic:
                self._show_up_to_date(result.message)
    def _on_update_check_error(self, message: str, automatic: bool) -> None:
        self._pending_update = None
        self._hide_update_status_indicator()
        self._status('Update check failed')
        if not automatic:
            QMessageBox.warning(self, 'Check for Updates', self._safe_update_message(message))
    def _safe_update_message(self, message: str) -> str:
        return (message or 'The update check failed.').splitlines()[0]
    def _show_update_available_dialog(self, result: CheckResult) -> None:
        latest = format_version(result.latest_version) if result.latest_version else 'a newer version'
        box = QMessageBox(self)
        box.setWindowTitle('Update Available')
        box.setIcon(QMessageBox.Information)
        box.setText(f'A new version of DPTK Desktop is available: {latest}.')
        box.setInformativeText(f'You are running {__version__}.')
        release_notes_btn = box.addButton('View Release Notes', QMessageBox.ActionRole)
        download_btn = box.addButton('Download Update', QMessageBox.AcceptRole)
        box.addButton('Later', QMessageBox.RejectRole)
        box.setDefaultButton(download_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is release_notes_btn:
            self._view_release_notes(result)
        elif clicked is download_btn:
            self._start_update_download(result)
    def _show_up_to_date(self, message: str) -> None:
        QMessageBox.information(self, 'Check for Updates', message)
    def _show_update_error(self, message: str) -> None:
        QMessageBox.warning(self, 'Check for Updates', self._safe_update_message(message))
    def _show_update_status_indicator(self, result: CheckResult) -> None:
        latest = format_version(result.latest_version) if result.latest_version else 'new version'
        self._update_status_button.setText(f'Update available: v{latest}')
        self._update_status_button.setVisible(True)
    def _hide_update_status_indicator(self) -> None:
        self._update_status_button.setText('')
        self._update_status_button.setVisible(False)
    def _on_update_status_clicked(self) -> None:
        result = self._pending_update
        if result is not None:
            self._show_update_available_dialog(result)
    def _view_release_notes(self, result: CheckResult) -> None:
        if result.release is not None and result.release.html_url:
            QDesktopServices.openUrl(QUrl(result.release.html_url))
    def _download_directory(self) -> Path:
        downloads = Path.home() / 'Downloads'
        if downloads.is_dir():
            return downloads
        return Path(tempfile.gettempdir())
    def _start_update_download(self, result: CheckResult) -> None:
        if self._download_in_progress:
            return
        if not result.installer_url or not result.sums_url or result.latest_version is None:
            self._show_update_error('This update is missing its installer files.')
            return
        filename = expected_installer_name(result.latest_version)
        dest_dir = str(self._download_directory())
        self._download_in_progress = True
        self._status('Downloading update...')
        worker = Worker(self._download_task, result.installer_url, result.sums_url, filename, dest_dir)
        worker.signals.result.connect(self._on_update_download_done)
        worker.signals.error.connect(self._on_update_download_error)
        self.thread_pool.start(worker)
    def _download_task(self, installer_url: str, sums_url: str, filename: str, dest_dir: str, *, progress, log) -> DownloadResult:
        return download_installer(installer_url, sums_url, filename, dest_dir, progress=progress.emit)
    def _on_update_download_done(self, result: object) -> None:
        self._download_in_progress = False
        if not isinstance(result, DownloadResult):
            self._show_update_error('Download failed.')
            return
        if result.status != 'ok':
            self._show_update_error(result.message)
            return
        self._show_download_complete_dialog(result)
    def _on_update_download_error(self, message: str) -> None:
        self._download_in_progress = False
        self._status('Download failed')
        QMessageBox.warning(self, 'Download Update', self._safe_update_message(message))
    def _show_download_complete_dialog(self, result: DownloadResult) -> None:
        path = Path(result.installer_path) if result.installer_path else None
        box = QMessageBox(self)
        box.setWindowTitle('Download Complete')
        box.setIcon(QMessageBox.Information)
        box.setText('The update has been downloaded and verified.')
        if path is not None:
            box.setInformativeText(str(path))
        install_btn = box.addButton('Install Now', QMessageBox.AcceptRole)
        open_folder_btn = box.addButton('Open Folder', QMessageBox.ActionRole)
        box.addButton('Later', QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is install_btn and path is not None:
            self._confirm_and_launch_installer(path, result.expected_sha256)
        elif clicked is open_folder_btn and path is not None:
            self._open_download_folder(path)
    def _confirm_and_launch_installer(self, path: Path, expected_sha256: str | None) -> None:
        if not path.is_file():
            QMessageBox.warning(self, 'Install Update', 'The downloaded installer could not be found.')
            return
        if not expected_sha256:
            QMessageBox.critical(self, 'Install Update', 'The installer checksum could not be verified. It will not be launched.')
            return
        self._status('Verifying installer...')
        worker = Worker(self._verify_installer_task, path)
        worker.signals.result.connect(lambda digest: self._on_installer_verified(path, expected_sha256, digest))
        worker.signals.error.connect(self._on_installer_verify_error)
        self.thread_pool.start(worker)
    def _verify_installer_task(self, path: Path, *, progress, log) -> str:
        return sha256_file(path)
    def _on_installer_verified(self, path: Path, expected_sha256: str, digest: object) -> None:
        if not isinstance(digest, str) or digest.lower() != expected_sha256.lower():
            QMessageBox.critical(self, 'Install Update', 'The installer checksum no longer matches. It will not be launched.')
            return
        confirm = QMessageBox.question(self, 'Install Update', 'Launch the installer now? The installer will guide you through the remaining steps.', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if confirm == QMessageBox.Yes:
            self._launch_installer(path)
    def _on_installer_verify_error(self, message: str) -> None:
        self._status('Installer verification failed')
        QMessageBox.critical(self, 'Install Update', 'The installer could not be verified. It will not be launched.')
    def _launch_installer(self, path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
    def _open_download_folder(self, path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
    def _show_about(self) -> None:
        QMessageBox.about(self, 'About DPTK Desktop', f'DPTK Desktop - Version {__version__}')

