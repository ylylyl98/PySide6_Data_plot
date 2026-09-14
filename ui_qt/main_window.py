from __future__ import annotations

import tempfile
import sys
import threading
from dataclasses import replace
from collections import deque
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

import numpy as np
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from matplotlib.transforms import Bbox
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import QFileSystemWatcher, QMimeData, QProcess, QSettings, Qt, QRunnable, QThreadPool, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
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
    QScrollArea,
    QStyle,
    QSizePolicy,
    QSplitter,
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
from core.drr_sources import (
    DrrSource,
    DrrMeasurementAssignment,
    DrrSourceCache,
    assess_background_gate_files,
    compatible_drr_repeats,
    discover_drr_sources,
    extract_wavelength_center_nm,
    find_saved_drr_recipe,
    guess_drr_background,
    group_drr_sources,
    inspect_csv_wavelength_center,
    resolve_source_path,
    wavelength_centers_match,
    resolve_drr_background_assignments,
)
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
    McdCenterCandidate, McdResult, McdSettings, background_fit_regions,
    discover_mcd_processing_status, ensure_mcd_package_dir, export_mcd_analysis_bundle,
    format_mcd_acquisition_conditions, format_mcd_energy, low_field_mcd_branch_fits,
    mcd_annotation_layout, pair_window_trace_by_branch,
    process_mcd, suggest_mcd_window_centers,
)
from core.export import (
    build_drr_export_base,
    export_drr_pair_pngs_and_dat,
    create_unique_package_dir,
    export_compare_panels,
    export_drr_png_and_dat,
    export_pl_pngs_and_dat,
    export_power_series_png_and_dat,
    export_power_vp_pngs_and_dat,
    export_shg_results,
    export_shg_twist_comparison,
)
from core.loader import DataCube, resolve_dat_y_axis
from core.workflow_request import LatestRequest, RequestToken
from core.provenance import WorkingCopyRecord, cleanup_working_copy, verify_initial_data_working_file
from core.plotting import (
    COMPARE_PANEL_ORDER,
    HeatmapParams,
    HeatmapRender,
    SplitColorScale,
    plot_compare_panel,
    plot_drr,
    downsample_cube_for_display,
    plot_heatmap,
    plot_pl,
    resolve_split_boundary,
)
from core.processing import (
    apply_sg_derivative_energy,
    background_correct_cube,
    clamp_sg_window,
    compute_auto_limits,
    estimate_constant_background,
    group_measurement_files,
    power_group_title,
    power_stage_paired_vp_cubes,
    power_valley_polarization_cube,
    nearest_gate_spectrum,
)
from ui_qt.presentation_widget import PresentationBuilderWidget
from ui_qt.feature_registry import FEATURES
from ui_qt.features_tools import ToolsPageMixin
from ui_qt.fluent_ui.style import apply_accessible_identity, set_fluent_property
from ui_qt.theme import alias as theme_alias, theme_manager
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
from ui_qt.shell.status_bar import StatusBarView
from ui_qt.shell.dock_host import DockHost
from ui_qt.shell.menu_toolbar import MenuToolbarHost
from ui_qt.shell.workspace import WorkspaceShell
from ui_qt.shell.workflow_navigation import WorkflowNavigation
from ui_qt.mcd_unified_page import McdUnifiedView
from ui_qt.mcd_result_retention import McdRetentionController, RetainedResultError


def _plain_retained(value: Any) -> Any:
    """Convert typed retention snapshots to exporter-safe plain containers."""
    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        return _plain_retained(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain_retained(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, (list, tuple)):
        return tuple(_plain_retained(item) for item in value)
    return value
from ui_qt.common import (
    ExportOptions,
    LoadOptions,
    LoadedState,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    UI_METRICS,
    WrappedFilenameDelegate,
    Worker,
    WorkerSignals,
)


class _OwnedRunnable(QRunnable):
    """Execute a MainWindow runnable while retaining it through run return."""

    def __init__(self, pool: "_MainWindowThreadPool", runnable: QRunnable) -> None:
        super().__init__()
        self._pool = pool
        self._runnable = runnable

    def run(self) -> None:
        try:
            self._runnable.run()
        finally:
            self._pool._run_finished(self._runnable)


def _run_unified_mcd_export(snapshot: dict, output_root: str, *, progress=None, log=None) -> dict:
    """Worker adapter adding the shared Worker ``log`` keyword."""
    from core.mcd_unified_export import export_mcd_analysis
    result = export_mcd_analysis(snapshot, output_root, progress=progress)
    if log is not None:
        log.emit(f"Saved unified MCD results: {result.get('export_id', '')}")
    return result


def _read_mcd_history_worker(folder, source, roots, *, progress=None, log=None):
    from core.mcd_center_history import read_center_history
    return read_center_history(folder, source, extra_roots=roots)


def _run_unified_mcd_analysis(result: Any, *, detection_mode: str = "raw", progress=None, log=None) -> Any:
    """Worker adapter for the pure detector.

    ``Worker`` supplies the common ``progress``/``log`` keywords to every
    callable.  The numerical detector intentionally has no Qt-facing worker
    arguments, so keep that boundary explicit and reusable in tests.
    """
    from core.mcd_analysis import detect_analysis_features
    analysis = detect_analysis_features(result, detection_mode=str(detection_mode))
    if log is not None:
        log.emit(f"Detected {len(analysis.features)} unified MCD features.")
    return analysis


def _run_unified_selected_feature_analysis(
    result: Any,
    feature: Any,
    candidates: tuple[Any, ...],
    *,
    method: str,
    source: str,
    half_width_ev: float,
    search_window_ev: tuple[float, float] | None = None,
    prominence_fraction: float = 0.03,
    reference_energy_ev: float | None = None,
    mapping_mode: str = "unknown",
    k_channel: str | None = None,
    reference_id: str | None = None,
    manual_links: tuple[Any, ...] = (),
    progress=None,
    log=None,
) -> dict[str, Any]:
    """Worker adapter for one selected feature and its supported tracks."""
    from core.mcd_feature_analysis import analyze_selected_feature, enrich_valley_analysis
    options = dict(
        candidates=candidates, method=method, source=source,
        half_width_ev=half_width_ev, manual_links=manual_links,
        search_window_ev=search_window_ev, prominence_fraction=prominence_fraction,
        reference_energy_ev=reference_energy_ev,
    )
    try:
        payload = analyze_selected_feature(result, feature, **options)
    except TypeError as exc:
        # Keep compatibility with the first additive helper revision while it
        # is being upgraded with explicit search/reference controls.
        if not any(name in str(exc) for name in ("search_window_ev", "prominence_fraction", "reference_energy_ev")):
            raise
        options.pop("search_window_ev", None); options.pop("prominence_fraction", None); options.pop("reference_energy_ev", None)
        payload = analyze_selected_feature(result, feature, **options)
    # Mapping is an explicit presentation choice.  Enrich the selected result
    # only after the measured tracks are available so splitting is calculated
    # from exact shared fields rather than inferred curves.
    payload = enrich_valley_analysis(
        payload, mapping_mode=mapping_mode, k_channel=k_channel,
        reference_id=reference_id,
    )
    if log is not None:
        log.emit(f"Selected feature analysis: {payload.get('status', 'unknown')}.")
    return payload


class _MainWindowThreadPool(QThreadPool):
    """Per-window pool that owns every runnable until execution has returned."""

    _completed = Signal(object)

    def __init__(self, owner: "MainWindow") -> None:
        super().__init__(owner)
        self._owner = owner
        self._registry_lock = threading.Lock()
        owner._owned_workers: set[QRunnable] = set()
        self._completed.connect(self._retire_worker, Qt.QueuedConnection)

    def start(self, runnable: QRunnable, priority: int = 0) -> None:
        if getattr(self._owner, "_is_closing", False):
            return
        if isinstance(runnable, Worker):
            # These QObjects (and PySide's lambda receivers) belong to the GUI
            # thread. Keep Qt ownership there even if the runnable's final
            # Python reference happens to be released by the pool thread.
            runnable.signals.setParent(self)
        with self._registry_lock:
            self._owner._owned_workers.add(runnable)
        super().start(_OwnedRunnable(self, runnable), priority)

    def _run_finished(self, runnable: QRunnable) -> None:
        # Queue retirement after result/error/finished delivery. Releasing the
        # worker here drops pending lambda callbacks and deletes their Qt
        # receivers in the wrong thread.
        self._completed.emit(runnable)

    @Slot(object)
    def _retire_worker(self, runnable: QRunnable) -> None:
        with self._registry_lock:
            self._owner._owned_workers.discard(runnable)
        if isinstance(runnable, Worker):
            runnable.signals.deleteLater()
from ui_qt.feature_pages import FeatureTabsMixin
from ui_qt.dense_form_layout import DenseFormRowLayout
from ui_qt.controllers_pl import PlController
from ui_qt.controllers_drr import DrrController
from ui_qt.controllers_mcd import McdController
from ui_qt.controllers_compare import CompareController
from ui_qt.controllers_power import PowerController
from ui_qt.controllers_shg import ShgController
from core.shg import ShgProcessResult, ShgSettings, ShgSweepData, process_shg_sweep
from core.shg_fit import (
    ShgAngularFitResult,
    ShgFitSettings,
    ShgTwistFitResult,
    evaluate_shg_angular_model,
    fit_shg_angular_result,
    fit_shg_twist_comparison,
)
from core.shg_history import scan_shg_history
from core.mcd_peak_shift import BOUNDARY_UNRELIABLE, analyze_peak_shift, format_mcd_angle, spectrum_energy_order, valley_quantities
from core.mcd_valley_split import compute_valley_splitting
from core.mcd_peak_display import second_derivative_map

class _PlotToolbar(NavigationToolbar2QT):
    """Toolbar that temporarily disables animated MCD axes for file saving."""

    def save_figure(self, *args: Any) -> None:
        owner = self.window()
        controller = getattr(owner, "mcd_controller", None)
        prepare = getattr(controller, "_prepare_mcd_toolbar_save", None)
        restore = getattr(controller, "_restore_mcd_toolbar_save", None)
        if getattr(owner, "last_plotted_mode", None) == "Compare":
            controller = getattr(owner, "compare_controller", None)
            prepare = getattr(controller, "_prepare_cmp_toolbar_save", None)
            restore = getattr(controller, "_restore_cmp_toolbar_save", None)
        unified = getattr(owner, "mcd_unified_view", None) if getattr(owner, "last_plotted_mode", None) == "MCD" else None
        if unified is not None:
            unified.prepare_full_redraw()
        if callable(prepare):
            prepare()
        try:
            canvas = self.canvas
            publication = getattr(canvas, "publication_context", None)
            if callable(publication):
                with publication():
                    super().save_figure(*args)
            else:
                super().save_figure(*args)
        finally:
            if callable(restore):
                restore()
            if unified is not None:
                unified.restore_interactive_drawing()


def _scan_drr_catalog_worker(
    folder: str,
    cache: DrrSourceCache,
    *,
    publish_cached=None,
    force: bool = False,
    include_all: bool = False,
    selected_sources=(),
    progress,
    log,
) -> tuple[str, List[DrrSource], DrrSourceCache, bool, bool]:
    """Discover DRR sources away from the GUI thread."""
    from core.drr_catalog import load_drr_catalog
    sources = load_drr_catalog(
        folder, cache, force=force, include_all=include_all,
        publish_cached=(lambda sources: publish_cached((folder, sources, cache, True, include_all)))
        if publish_cached is not None else None,
    )
    # Selected files may be absolute paths outside the experiment partition.
    # Inspect those exact files in the worker and merge only successful
    # DrrSource records into the catalog result.
    from core.drr_sources import resolve_source_path
    known = {str(resolve_source_path(folder, source.source).resolve()).casefold()
             for source in sources}
    missing_selected = []
    if selected_sources:
        from core.drr_sources import discover_drr_sources
        for selected in dict.fromkeys(str(value) for value in selected_sources):
            path = resolve_source_path(folder, selected)
            if not path.is_file():
                missing_selected.append(selected)
                continue
            try:
                inspected = discover_drr_sources(path.parent, include_all=True, include_history=False)
            except (OSError, ValueError, TypeError):
                inspected = []
            identity = str(path.resolve()).casefold()
            if identity in known:
                continue
            for source in inspected:
                inspected_identity = str(resolve_source_path(path.parent, source.source).resolve()).casefold()
                if inspected_identity == identity and identity not in known:
                    sources.append(replace(source, source=str(selected), filename=path.name))
                    known.add(identity)
                    break
            else:
                missing_selected.append(selected)
    return folder, sources, cache, False, include_all, tuple(missing_selected)


def _scan_folder_sources_worker(folder: str, *, power_include_legacy: bool = False, mode: str = "all", progress, log) -> tuple:
    """Collect the cross-tab source catalogs without blocking Qt's GUI thread."""
    csv_files = data_io.list_csv_files(folder)
    map_files = data_io.list_map_input_files(folder) if mode in {"all", "Compare"} else []
    pl_files = data_io.list_pl_source_files(folder) if mode in {"all", "PL", "Compare"} else []
    pl_ambiguous: set[str] = set()
    pl_status = data_io.discover_pl_processing_status(folder, pl_files, ambiguous_sources=pl_ambiguous) if mode in {"all", "PL"} else {}
    mcd_files = data_io.list_mcd_csv_files(folder) if mode in {"all", "MCD"} else []
    mcd_ambiguous: set[str] = set()
    mcd_status = discover_mcd_processing_status(folder, mcd_files, ambiguous_sources=mcd_ambiguous) if mode in {"all", "MCD"} else {}
    shg_ambiguous: set[str] = set()
    shg_history = scan_shg_history(folder, csv_files, ambiguous_sources=shg_ambiguous) if mode in {"all", "SHG Processing"} else None
    shg_mtimes: dict[str, float] = {}
    for source in (csv_files if shg_history is not None else ()):
        try:
            shg_mtimes[source] = (Path(folder) / source).stat().st_mtime
        except OSError:
            shg_mtimes[source] = 0.0
    # Publish source mtimes with the worker snapshot so picker sorting never
    # needs to touch the filesystem from the GUI thread.
    mtime_by_source: dict[str, float] = {}
    def _snapshot_mtime(source: str) -> float:
        if source in mtime_by_source:
            return mtime_by_source[source]
        try:
            value = resolve_source_path(folder, source).stat().st_mtime
        except OSError:
            value = 0.0
        mtime_by_source[source] = float(value)
        return float(value)
    pl_mtimes = {source: _snapshot_mtime(source) for source in pl_files}
    mcd_mtimes = {source: _snapshot_mtime(source) for source in mcd_files}
    compare_mtimes = {source: _snapshot_mtime(source) for source in map_files}
    # Power discovery reads headers and, for table sources, the Power_uW
    # column. Keep that work in this catalog worker so GUI group refreshes are
    # a pure application of an accepted snapshot.
    from core.power_workflow import discover_power_files, is_combined, processed_source_names
    power_files = discover_power_files(folder, include_legacy=bool(power_include_legacy)) if mode in {"all", "Power Dependent"} else []
    power_signatures = []
    for name in power_files:
        try:
            stat = (Path(folder) / name).stat()
            power_signatures.append((str(name), int(stat.st_size), int(stat.st_mtime_ns)))
        except OSError:
            power_signatures.append((str(name), None, None))
    power_sources = data_io.get_power_series_sources(folder, power_files) if mode in {"all", "Power Dependent"} else {}
    power_combined = {
        str(source.file_name).replace("\\", "/").casefold()
        for source in power_sources.values()
        if source.file_name and is_combined(folder, source.file_name)
    }
    power_snapshot = {
        "tag": "power_catalog",
        "version": 1,
        "folder": str(folder),
        "include_legacy": bool(power_include_legacy),
        "candidates": tuple(power_files),
        "signatures": tuple(power_signatures),
        "sources": power_sources,
        "processed_names": tuple(sorted(processed_source_names(folder))) if mode in {"all", "Power Dependent"} else (),
        "combined_names": tuple(sorted(power_combined)),
    }
    try:
        from core.power_selection_store import load_power_selections
        power_snapshot["saved_assignments"] = load_power_selections(folder, power_sources)
    except (ImportError, OSError, ValueError, TypeError):
        power_snapshot["saved_assignments"] = {}
    history = []
    if mode in {"all", "Compare"}:
        import json
        for path in (Path(folder) / "Processed Data" / "Compare").rglob("*.metadata.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(record, dict):
                    history.append(dict(record, path=str(path)))
            except (OSError, ValueError, UnicodeError):
                continue
    return (
        folder, csv_files, map_files, pl_files, pl_status, mcd_files, mcd_status,
        pl_ambiguous, mcd_ambiguous, dict(shg_history.processed_at) if shg_history else {},
        shg_ambiguous, shg_mtimes,
        dict(shg_history.roles) if shg_history else {},
        {"tag": "source_metadata", "version": 1, "folder": str(folder),
         "mode": mode, "generation": 0,
         "modes": tuple(name for name in ("PL", "MCD", "Compare", "SHG", "Power", "DRR")
                        if mode == "all" or mode == name
                        or (name == "SHG" and mode == "SHG Processing")
                        or (name == "Power" and mode == "Power Dependent")),
         "pl_mtimes": pl_mtimes, "mcd_mtimes": mcd_mtimes,
         "shg_mtimes": shg_mtimes,
         "compare_mtimes": compare_mtimes},
        power_snapshot,
        {"tag": "catalog_scope", "mode": mode, "compare_history": history},
    )


def _catalog_payload_compatible(payload, folder: str, mode: str) -> bool:
    """Validate scoped metadata before publishing an on-disk preview."""
    if not isinstance(payload, tuple):
        return False
    metadata = next((item for item in payload
                     if isinstance(item, dict) and item.get("tag") == "source_metadata"), None)
    scope = next((item for item in payload
                  if isinstance(item, dict) and item.get("tag") == "catalog_scope"), None)
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        return False
    if str(metadata.get("folder", "")).casefold() != str(folder).casefold():
        return False
    requested = str(mode)
    modes = {str(value) for value in metadata.get("modes", ())}
    covered = requested in modes or (requested == "SHG Processing" and "SHG" in modes) \
        or (requested == "Power Dependent" and "Power" in modes) \
        or (requested == "all" and {"PL", "MCD", "Compare"}.issubset(modes))
    return covered and isinstance(scope, dict) and str(scope.get("mode", "")) == requested


def _cached_folder_sources_worker(folder: str, *, mode: str, power_include_legacy: bool,
                                  force: bool, publish_cached, progress, log) -> tuple:
    from core.source_catalog_cache import SourceCatalogCache
    cache = SourceCatalogCache(folder, f"{mode}-{int(power_include_legacy)}")
    cached = cache.read()
    cache_usable = cached is not None and _catalog_payload_compatible(cached, folder, mode)
    if cache_usable:
        publish_cached(tuple(cached[:-1]) + (dict(cached[-1], preview=True),))
    return cache.refresh(lambda: _scan_folder_sources_worker(
        folder, mode=mode, power_include_legacy=power_include_legacy, progress=progress, log=log),
        force=force or not cache_usable)



def _enumerate_watch_dirs_worker(root: str, *, progress, log) -> tuple[str, list[str]]:
    """Enumerate deep watcher directories without blocking folder changes."""
    path = Path(root)
    try:
        directories = [str(child) for child in path.rglob("*") if child.is_dir()]
    except OSError:
        directories = []
    return root, directories


class MainWindow(FeatureTabsMixin, ToolsPageMixin, QMainWindow):
    file_catalog_refresh_finished = Signal(str, bool)
    drr_catalog_refresh_finished = Signal(str, bool)
    drr_catalog_preview_ready = Signal(str)
    SETTINGS_ORG = "DPTK"
    SETTINGS_APP = "PySide6_Data_Plot"
    SETTINGS_LAST_DATA_FOLDER = "data/last_folder"
    SETTINGS_LAST_PARENT_FOLDER = "data/last_parent_folder"
    SETTINGS_RECENT_FOLDERS = "data/recent_folders"
    SETTINGS_AUTO_UPDATE_CHECK = "updates/check_automatically"
    SETTINGS_MCD_SOURCE_FILTER = "mcd/source_filter"
    SETTINGS_PL_SOURCE_FILTER = "pl/source_filter"
    SETTINGS_PL_AUTO_NEXT = "pl/auto_load_next"
    MAX_RECENT_FOLDERS = 8

    def __init__(self):
        super().__init__()
        from ui_qt.wheel_policy import install_wheel_value_guard
        install_wheel_value_guard()
        register_colormaps()
        self.setWindowTitle("DPTK Desktop (PySide6)")
        self.setMinimumSize(1180, 700)
        self._is_closing = False
        self.thread_pool = _MainWindowThreadPool(self)
        self.settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        saved_mcd_filter = str(
            self.settings.value(self.SETTINGS_MCD_SOURCE_FILTER, "all")
        ).casefold()
        self._mcd_source_filter_preference = (
            saved_mcd_filter
            if saved_mcd_filter in {"all", "unprocessed", "processed", "unknown"}
            else "all"
        )
        saved_pl_filter = str(
            self.settings.value(self.SETTINGS_PL_SOURCE_FILTER, "all")
        ).casefold()
        self._pl_source_filter_preference = (
            saved_pl_filter
            if saved_pl_filter in {"all", "unprocessed", "processed", "unknown"}
            else "all"
        )
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
        self._file_refresh_generation = 0
        self._catalog_ready_modes: set[str] = set()
        self._catalog_displayed_modes: set[str] = set()
        self._catalog_active_scan: str | None = None
        self._catalog_pending_requests: dict[str, bool] = {}
        self._file_refresh_running = False
        self._file_refresh_pending = False
        self._file_refresh_pending_auto = False
        self._file_refresh_workers: list[Worker] = []
        self._pending_open_file: str = ""
        self._watch_generation = 0
        self._watch_workers: list[Worker] = []
        self.recent_folders: List[str] = self._load_recent_folders()
        self.available_files: List[str] = []
        self.available_map_files: List[str] = []
        self.pl_available_files: List[str] = []
        self._pl_source_mtime_cache: dict[str, float] = {}
        self._mcd_source_mtime_cache: dict[str, float] = {}
        self.pl_processed_status: dict[str, str] = {}
        self.pl_processing_ambiguous: set[str] = set()
        self.mcd_available_files: List[str] = []
        self.mcd_processed_status: dict[str, str] = {}
        self.mcd_processing_ambiguous: set[str] = set()
        self.shg_processed_status: dict[str, str] = {}
        self.shg_processing_ambiguous: set[str] = set()
        self.shg_processing_roles: dict[str, tuple[str, ...]] = {}
        self._shg_source_mtime_cache: dict[str, float] = {}
        self.drr_available_sources: List[DrrSource] = []
        self._drr_source_cache = DrrSourceCache(load_on_init=False)
        self._drr_refresh_generation = 0
        self._drr_refresh_running = False
        self._drr_refresh_pending = False
        self._drr_refresh_pending_auto = False
        self._drr_refresh_pending_old_sources: set[str] | None = None
        self._drr_refresh_pending_selected_sources: set[str] | None = None
        self._drr_refresh_workers: list[Worker] = []
        self._drr_derivative_cache: dict[tuple[int, int | None, int, int], tuple[DataCube, int]] = {}
        self.drr_selected_files: List[str] = []
        self.drr_baseline_files_manual: List[str] = []
        self.drr_baseline_files_found: List[str] = []
        self._drr_assignments: tuple[DrrMeasurementAssignment, ...] = ()
        self._drr_assignments_automatic = False
        self._drr_baseline_user_selected = False
        self._drr_background_guess = None
        self.loaded: LoadedState | None = None
        self.last_plotted_mode: str | None = None
        self._last_export_move_folder = ""
        self._last_export_move_sources: list[str] = []
        self._pl_export_source_was_processed = False
        self._pl_last_export_source = ""
        self._pl_auto_next_queue: list[str] = []
        self._pl_auto_next_active = False
        self.log_lines: deque[str] = deque(maxlen=300)
        # Opt-in interaction instrumentation; disabled for normal use.
        self._workflow_metrics_enabled = False
        self._workflow_metrics: deque[dict[str, Any]] = deque(maxlen=2000)
        self._last_plot_params_key: tuple[Any, ...] | None = None
        self._last_plot_cube: DataCube | None = None
        # Plot controls can emit several valueChanged signals during one user
        # edit (and auto-ranging emits another burst).  Keep one coalesced
        # redraw per mode so expensive Matplotlib reconstruction does not run
        # once per intermediate value.
        self._plot_redraw_timers: dict[str, QTimer] = {}
        self._plot_redraw_pending: set[str] = set()
        self._automatic_update_timer: QTimer | None = None
        self._sidebar_last_expanded_width = UI_METRICS["left_width"]
        self._drr_heatmap_ax = None
        self._drr_spectrum_ax = None
        self._drr_heatmap_axes: dict[str, Any] = {}
        self._drr_spectrum_axes: dict[str, Any] = {}
        self._drr_plot_cubes: dict[str, DataCube] = {}
        self._drr_gate_lines: dict[str, Any] = {}
        self._drr_plot_view = "raw"
        self._drr_side_by_side = False
        self._drr_view_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._drr_limits_from_controls = False
        self._pl_heatmap_ax = None
        self._pl_spectrum_ax = None
        self._cmp_heatmap_axes: dict[str, Any] = {}
        self._cmp_gate_lines: dict[str, Any] = {}
        self._cmp_linecut_ax = None
        self._cmp_active_cubes: dict[str, DataCube] = {}
        self._cmp_rendered_loaded_marker = None
        self._cmp_blit_enabled = False
        self._cmp_blit_bbox = None
        self._cmp_blit_background = None
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
        self._power_sources_cache: Dict[str, data_io.PowerSeriesSource] | None = None
        self._power_sources_cache_files: tuple[str, ...] = ()
        self._power_catalog_folder = ""
        self._power_catalog_include_legacy = False
        self._power_catalog_candidates: tuple[str, ...] = ()
        self._power_catalog_signatures: tuple = ()
        self._power_catalog_processed_names: set[str] = set()
        self._power_catalog_combined_names: set[str] = set()
        self._power_catalog_pending = False
        self._power_pending_open_name = ""
        self._power_pending_open_generation = None
        self._power_pending_measurement_selection = None
        self._power_pending_measurement_validation = {}
        self._power_pending_measurement_generation = None
        self._power_result_cache: dict[str, data_io.PowerSeriesResult] = {}
        self._shg_raw_ax = None
        self._shg_corrected_ax = None
        self._shg_angle_ax = None
        self._mcd_heatmap_ax = None
        self._mcd_pair_ax = None
        self._mcd_spectrum_ax = None
        self._mcd_trace_ax = None
        self._mcd_integral_ax = None
        self._mcd_pair_cursor = None
        self._mcd_pair_spectrum_lines: list[Any] = []
        self._mcd_linecut_lines: list[Any] = []
        self._mcd_linecut_diagnostic_text = None
        self._mcd_trace_lines: dict[tuple[str, str, str], Any] = {}
        self._mcd_fit_lines: dict[str, Any] = {}
        self._mcd_slope_text = None
        self._mcd_blit_enabled = False
        self._mcd_blit_in_draw = False
        self._mcd_blit_backgrounds: dict[str, Any] = {}
        self._mcd_blit_bboxes: dict[str, Bbox] = {}
        self._mcd_blit_axes: dict[str, tuple[Any, ...]] = {}
        self._mcd_heat_dynamic_artists: list[Any] = []
        self._mcd_overlay_artists: dict[str, list[Any]] = {}
        self._mcd_toolbar_save_animated_axes: tuple[Any, ...] = ()
        self._mcd_window_artists: list[dict[str, Any]] = []
        self._mcd_window_dragging = False
        self._mcd_window_drag_moved = False
        self._mcd_window_drag_offset = 0.0
        self._mcd_window_drag_center: float | None = None
        # Peak-shift dragging stores the semantic target only.  Plot redraws
        # replace every Matplotlib artist and axis, so retaining artist
        # objects here would make a drag stop as soon as selection changes.
        self._mcd_peak_drag: dict[str, Any] | None = None
        # A manually dragged center is a display cursor, independent of the
        # tracked feature point.  It is kept as a scalar so a redraw cannot
        # quantize it back to the nearest sampled/tracked energy.
        self._mcd_peak_manual_center_ev: float | None = None
        self._mcd_peak_manual_center_channel: str | None = None
        self._mcd_peak_manual_center_field_index: int | None = None
        self._mcd_peak_manual_center_artists: dict[str, Any] = {}
        self._mcd_peak_manual_center_blit_backgrounds: dict[Any, Any] = {}
        self._mcd_center_candidates: tuple[McdCenterCandidate, ...] = ()
        self._mcd_candidate_active_index: int | None = None
        self._mcd_manual_center_before_suggestions: float | None = None
        self._mcd_candidate_applying = False
        self._mcd_candidate_search_range: tuple[float, float] | None = None
        self._mcd_candidate_artists: dict[int, Any] = {}
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
        self._export_in_progress = False
        self._active_export_request_key = ""
        self._last_export_request_key = ""
        self._active_load_mode: str | None = None
        self._active_load_succeeded = False
        self._load_requests: dict[str, LatestRequest[LoadOptions]] = {}
        self._active_load_token: RequestToken | None = None
        self._last_draw_identity: dict[str, tuple[str, ...]] = {}
        self._shown_draw_mode: str | None = None
        self._shown_draw_identity: tuple[str, ...] = ()
        self._shown_selection_identity: tuple[str, ...] = ()
        self._data_state_error: str | None = None
        self._data_state_loading = False
        self._pending_load_mode: str | None = None
        self._pending_load_options: LoadOptions | None = None
        self._capture_pending_load = False
        self._invalidated_load_modes: set[str] = set()
        self._pending_range_refresh: dict[str, bool] = {}
        self._load_lifecycle_generation = 0
        self._mcd_source_generation = 0
        self._mcd_reload_pending = False
        self._mcd_unified_analysis_generation = 0
        self._mcd_unified_analysis_worker = None
        self._mcd_unified_analysis_pending = False
        self._mcd_unified_catalog_requested_key = None
        self._mcd_unified_catalog_completed_key = None
        self._mcd_unified_detector_settings_version = 1
        self._mcd_unified_detection_mode = "raw"
        self._mcd_unified_analysis_source = None
        self._mcd_unified_features: tuple[Any, ...] = ()
        self._mcd_unified_track_generation = 0
        self._mcd_unified_track_worker = None
        self._mcd_unified_track_pending = None
        self._mcd_unified_track_key = None
        self._mcd_unified_track_payload: dict[str, Any] | None = None
        self._mcd_unified_track_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._mcd_unified_manual_links: dict[str, list[str]] = {}
        self._mcd_unified_export_worker = None
        self._mcd_export_output_root: str | None = None
        self.mcd_retention = McdRetentionController(parent=self)
        self.mcd_controller = McdController(self)
        self.pl_controller = PlController(self)
        self.drr_controller = DrrController(self)
        self.compare_controller = CompareController(self)
        self.power_controller = PowerController(self)
        from ui_qt.power_peak_controller import PowerPeakController
        self.power_peak_controller = PowerPeakController(self)
        self.shg_controller = ShgController(self)

        self._build_ui()
        self._folder_placeholder_text = self.folder_edit.placeholderText()
        self.apply_ui_metrics()
        self._wire_actions()
        self.compare_controller._cmp_update_background_mode()
        self._apply_initial_geometry()
        self._set_stage("No data")
        self._update_action_states()
        self._restore_last_folder()
        self.setAcceptDrops(True)
        self._schedule_automatic_update_check()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Invalidate background callbacks and let active file reads finish."""
        self._is_closing = True
        if self.power_peak_controller.worker is not None:
            self.power_peak_controller.worker.cancelled.set()
        for timer in self._plot_redraw_timers.values():
            timer.stop()
        self.folder_refresh_timer.stop()
        self.shg_controller._stop_shg_reprocessing()
        self.mcd_controller._shutdown_mcd_lifecycle()
        peak_cancel = getattr(self, "_mcd_peak_analysis_cancel_event", None)
        if peak_cancel is not None:
            peak_cancel.set()
        if hasattr(self, "_mcd_peak_analysis_generation"):
            self._mcd_peak_analysis_generation += 1
            self._mcd_peak_analysis_pending_request = None
        self._mcd_unified_analysis_generation += 1
        self._mcd_unified_track_generation += 1
        self._mcd_unified_track_pending = None
        if self._automatic_update_timer is not None:
            self._automatic_update_timer.stop()
        self._file_refresh_generation += 1
        self._drr_refresh_generation += 1
        self._watch_generation += 1
        # This pool is private to the window, so waiting cannot stall unrelated
        # global-pool work.  The ownership registry retains each runnable until
        # its run method has actually returned.
        self.thread_pool.waitForDone()
        super().closeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        """A re-shown window is no longer closing; redraws may resume."""
        self._is_closing = False
        super().showEvent(event)

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
            # The combo's display labels are intentionally compact; retain the
            # complete current path in the shared read-only line edit.
            if self.current_folder:
                self.folder_edit.setText(self.current_folder)
            else:
                self.folder_edit.clear()
        finally:
            self.recent_folder_combo.blockSignals(old)

    def _set_current_folder(self, folder: str, *, remember: bool = True, refresh: bool = True) -> bool:
        path = Path(folder)
        if not path.exists() or not path.is_dir():
            self._show_error(f"Folder does not exist: {folder}")
            return False
        folder_changed = str(path).lower() != self.current_folder.lower()
        if folder_changed:
            self._catalog_ready_modes.clear()
            self._drr_include_all_sources = False
            self._catalog_displayed_modes.clear()
            self._catalog_pending_requests.clear()
            self._invalidate_export_move_sources()
            self._reset_workflow_state_for_folder_change()
            if hasattr(self, "drr_pin_baseline_chk"):
                self.drr_baseline_files_manual = []
                self.drr_baseline_files_found = []
                self.drr_pin_baseline_chk.setChecked(False)
        self.current_folder = str(path)
        self.folder_edit.setText(self.current_folder)
        self._watch_current_folder()
        if remember:
            self._remember_data_folder(self.current_folder)
        else:
            self._populate_recent_folder_combo()
        if refresh:
            self._refresh_file_lists(auto=True)
        return True

    def _watch_current_folder(self) -> None:
        if self._is_closing:
            return
        self._watch_generation += 1
        generation = self._watch_generation
        watched = list(self.folder_watcher.directories())
        if watched:
            self.folder_watcher.removePaths(watched)
        self._watched_folder = ""
        if not self.current_folder:
            return
        path = Path(self.current_folder)
        if not path.exists() or not path.is_dir():
            return
        watch_paths = [path]
        initial_data = path / "Initial Data"
        if initial_data.is_dir():
            watch_paths.append(initial_data)
        try:
            mcd_roots = [
                child
                for child in path.iterdir()
                if child.is_dir() and child.name.casefold() == "mcd"
            ]
        except OSError:
            mcd_roots = []
        for mcd_root in mcd_roots:
            watch_paths.append(mcd_root)
        critical_paths = [path]
        if initial_data.is_dir():
            critical_paths.append(initial_data)
        critical_paths.extend(mcd_roots)
        watch_paths = critical_paths + [
            item for item in watch_paths if item not in critical_paths
        ]
        self.folder_watcher.addPaths([str(item) for item in watch_paths[:256]])
        if str(path) in self.folder_watcher.directories():
            self._watched_folder = str(path)
        # Deep trees are common under Initial Data.  Enumerate their child
        # directories in a worker and add the paths only if this folder is
        # still active when the result arrives.
        roots = [initial_data, *mcd_roots]
        for root_dir in roots:
            if not root_dir.is_dir():
                continue
            worker = Worker(_enumerate_watch_dirs_worker, str(root_dir))
            self._watch_workers.append(worker)
            worker.signals.result.connect(
                lambda result, generation=generation, folder=str(path):
                self._on_watch_dirs_result(result, generation, folder)
            )
            worker.signals.finished.connect(lambda worker=worker: self._finish_watch_worker(worker))
            self.thread_pool.start(worker)

    def _on_watch_dirs_result(self, result, generation: int, folder: str) -> None:
        if generation != self._watch_generation or str(folder).casefold() != str(self.current_folder).casefold():
            return
        _root, directories = result
        existing = set(self.folder_watcher.directories())
        additions = [directory for directory in directories if directory not in existing]
        if additions:
            self.folder_watcher.addPaths(additions[: max(0, 256 - len(existing))])

    def _finish_watch_worker(self, worker: Worker) -> None:
        try:
            self._watch_workers.remove(worker)
        except ValueError:
            pass

    def _on_watched_folder_changed(self, folder: str) -> None:
        if not self.current_folder:
            return
        changed = Path(folder).resolve()
        root = Path(self.current_folder).resolve()
        initial_data = (root / "Initial Data").resolve()
        if changed == root:
            self.folder_refresh_timer.start()
            return
        mcd_roots: list[Path] = []
        try:
            mcd_roots = [
                child.resolve()
                for child in root.iterdir()
                if child.is_dir() and child.name.casefold() == "mcd"
            ]
        except OSError:
            pass
        for mcd_root in mcd_roots:
            try:
                changed.relative_to(mcd_root)
            except ValueError:
                continue
            self.folder_refresh_timer.start()
            return
        try:
            changed.relative_to(initial_data)
        except ValueError:
            return
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
        self._catalog_ready_modes.clear()
        self.presentation_widget._plots_catalog_dirty = True
        if self.tabs.tabText(self.tabs.currentIndex()) == "Slides":
            self.presentation_widget.set_experiment_folder(self.current_folder)
        self._refresh_file_lists(auto=True)
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
        set_fluent_property(self.folder_edit, "appRole", "dropTarget" if on else None)
        if on and not self.current_folder:
            self.folder_edit.setPlaceholderText("Drop to set folder")
        else:
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
        self.drr_controller._update_drr_selection_labels()
        self.power_controller._power_refresh_groups()
        self.mcd_controller._mcd_refresh_sources()
        self.shg_controller._shg_refresh_sources()
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
        left = self._build_left_panel()
        right = self._build_plot_panel()
        presentation_widget = PresentationBuilderWidget()
        presentation_widget.status_message.connect(self._status)
        presentation_widget.log_message.connect(self._append_log)
        self._tools_tab_index = next(
            (index for index in range(self.tabs.count()) if self.tabs.tabText(index) == "Tools"),
            -1,
        )
        self._tools_tab_widget = (
            self.tabs.widget(self._tools_tab_index) if self._tools_tab_index >= 0 else None
        )
        self._tools_tab_placeholder = QWidget(self)
        self._tools_tab_placeholder.setObjectName("toolsTabPlaceholder")
        self._tools_tab_placeholder.hide()
        self.tools_workspace = QWidget()
        self.tools_workspace.setObjectName("toolsWorkspace")
        tools_layout = QVBoxLayout(self.tools_workspace)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(0)
        self._tools_workspace_layout = tools_layout
        self._tools_in_workspace = False
        self._tools_transitioning = False
        self.workflow_navigation = WorkflowNavigation(self.tabs, self)
        # Compatibility aliases remain owned by MainWindow for existing callers.
        self.sidebar_toggle_btn = self.workflow_navigation.sidebar_toggle_btn
        self.workflow_tabs = self.workflow_navigation.workflow_tabs
        self.workflow_tabs.currentChanged.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self.workflow_tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self._on_central_tab_changed)
        # Keep the compatibility page in the stack for existing callers and
        # tests, while exposing one user-facing MCD destination.
        hidden_peak_indices = []
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == "MCD Peak Shift":
                self.tabs.setTabVisible(index, False)
                self.workflow_tabs.setTabVisible(index, False)
                hidden_peak_indices.append(index)
        # A prior session or a legacy request may have restored the retired
        # page as the current tab.  Redirect that landing state to the single
        # visible MCD workflow while leaving the compatibility page callable
        # by internal tests/controllers.
        if self.tabs.currentIndex() in hidden_peak_indices:
            mcd_index = next((i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "MCD" and self.tabs.isTabVisible(i)), -1)
            if mcd_index >= 0:
                self.tabs.setCurrentIndex(mcd_index)
        self.sidebar_toggle_btn.toggled.connect(self._set_sidebar_visible)
        self.workspace_shell = WorkspaceShell(
            self,
            navigation=self.workflow_navigation,
            left_panel=left,
            plot_panel=right,
            presentation_widget=presentation_widget,
            tools_widget=self.tools_workspace,
        )
        # Compatibility aliases remain owned by MainWindow for existing callers.
        self.central_widget = self.workspace_shell.central_widget
        self.left_panel = self.workspace_shell.left_panel
        self.workspace_splitter = self.workspace_shell.workspace_splitter
        self.presentation_widget = self.workspace_shell.presentation_widget
        self.workspace_stack = self.workspace_shell.workspace_stack
        self.workspace_splitter.splitterMoved.connect(self._on_sidebar_splitter_moved)

        self.status_bar_view = StatusBarView()
        self.setStatusBar(self.status_bar_view)
        self._status_progress = self.status_bar_view.progress
        self._update_status_button = self.status_bar_view.update_button
        self._update_status_button.clicked.connect(self._on_update_status_clicked)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.dock_host = DockHost(
            self,
            log_content=self.log_text,
            results_pages=(
                ("PL", "PL peak and fit results", self.pl_analysis_text),
                ("DRR", "DRR peak and fit results", self.drr_analysis_text),
                ("MCD", "MCD pair diagnostics", self.mcd_diagnostics_text),
            ),
            results_empty_text="This workflow has no separate text results. Use the plot and export controls.",
        )
        # Compatibility aliases remain owned by MainWindow for existing callers.
        self.log_dock = self.dock_host.log_dock
        self.results_dock = self.dock_host.results_dock
        self.results_stack = self.dock_host.results_stack
        self._results_page_indices = self.dock_host.results_page_indices
        self._results_empty_index = self.dock_host.results_empty_index
        self.mcd_diagnostics_expander.hide()
        self._update_results_dock_page()
        self.menu_toolbar_host = MenuToolbarHost(
            self,
            log_dock=self.log_dock,
            results_dock=self.results_dock,
            data_source_context=self.data_source_context,
        )
        self.menu_toolbar_host.apply_theme(navigation_toolbar=self.toolbar)
        self._theme_manager = theme_manager()
        if self._theme_manager is not None:
            self._theme_manager.themeChanged.connect(
                lambda theme: self.menu_toolbar_host.apply_theme(
                    theme, navigation_toolbar=self.toolbar
                )
            )
            self._theme_manager.themeChanged.connect(
                self.mcd_controller._on_theme_changed
            )
        # Compatibility aliases remain owned by MainWindow for existing callers.
        self.show_log_action = self.menu_toolbar_host.show_log_action
        self.show_results_action = self.menu_toolbar_host.show_results_action
        self.show_sidebar_action = self.menu_toolbar_host.show_sidebar_action
        self.check_updates_action = self.menu_toolbar_host.check_updates_action
        self.auto_update_check_action = self.menu_toolbar_host.auto_update_check_action
        self.about_action = self.menu_toolbar_host.about_action
        self.load_action = self.menu_toolbar_host.load_action
        self.plot_action = self.menu_toolbar_host.plot_action
        self.save_action = self.menu_toolbar_host.save_action
        self.move_now_btn = self.menu_toolbar_host.move_now_btn
        self.clean_verified_sources_chk = self.menu_toolbar_host.clean_verified_sources_chk
        self.show_sidebar_action.setCheckable(True)
        self.show_sidebar_action.setChecked(True)
        self.auto_update_check_action.setCheckable(True)
        self.auto_update_check_action.setChecked(self._auto_update_check_enabled())
        self.move_now_btn.setEnabled(False)
        self.clean_verified_sources_chk.setChecked(False)
        self.show_sidebar_action.setShortcut("Ctrl+B")

    def _set_sidebar_visible(self, visible: bool) -> None:
        """Show or hide the contextual inspector without changing the active workflow."""
        if not hasattr(self, "left_panel"):
            return
        if hasattr(self, "tabs") and self.tabs.tabText(self.tabs.currentIndex()) == "Slides":
            return
        visible = bool(visible)
        if not visible and self.left_panel.isVisible():
            width = self.left_panel.width()
            minimum = UI_METRICS["sidebar_min_width"]
            maximum = UI_METRICS["sidebar_max_width"]
            if minimum <= width <= maximum:
                self._sidebar_last_expanded_width = width
        if visible:
            self.left_panel.setVisible(True)
            sizes = self.workspace_splitter.sizes()
            total = sum(sizes)
            if total <= 0:
                total = self.workspace_splitter.width()
            target = max(
                UI_METRICS["sidebar_min_width"],
                min(UI_METRICS["sidebar_max_width"], int(self._sidebar_last_expanded_width)),
            )
            self.workspace_splitter.setSizes([target, max(1, total - target)])
        else:
            self.left_panel.setVisible(False)
        if hasattr(self, "sidebar_toggle_btn"):
            blocked = self.sidebar_toggle_btn.blockSignals(True)
            self.sidebar_toggle_btn.setChecked(visible)
            self.sidebar_toggle_btn.blockSignals(blocked)
            self.sidebar_toggle_btn.setText("Controls" if visible else "Show controls")

    def _on_sidebar_splitter_moved(self, _position: int, _index: int) -> None:
        """Remember the user-selected expanded sidebar width for restoration."""
        if not self.left_panel.isVisible() or not self.sidebar_toggle_btn.isChecked():
            return
        sizes = self.workspace_splitter.sizes()
        if not sizes:
            return
        minimum = UI_METRICS["sidebar_min_width"]
        maximum = UI_METRICS["sidebar_max_width"]
        self._sidebar_last_expanded_width = max(minimum, min(maximum, int(sizes[0])))

    def _build_left_panel(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(UI_METRICS["row_spacing"])

        # Shared source controls are composed in the workbench Source Bar.
        folder_box = QWidget()
        folder_box.setObjectName("dataSourceContext")
        folder_grid = QHBoxLayout(folder_box)
        # Toolbar presentation is intentionally compact: one row with no
        # group-box framing or extra vertical padding.
        folder_grid.setContentsMargins(0, 0, 0, 0)
        folder_grid.setSpacing(0)
        self.folder_edit = QLineEdit()
        self.folder_edit.setObjectName("folderEdit")
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText("No folder selected — click Browse or Open File")
        self.folder_edit.setToolTip("Current working folder for data files")
        apply_accessible_identity(
            self.folder_edit,
            name="Current data folder",
            description="Current working folder for data files",
            identifier="source.folder",
        )
        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.setObjectName("browseFolderButton")
        self.browse_btn.setToolTip("Select a folder containing CSV data files")
        apply_accessible_identity(self.browse_btn, name="Browse Folder", identifier="source.browse")
        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.setObjectName("openFileButton")
        self.open_file_btn.setToolTip("Open a single CSV file and set its folder as the working directory")
        apply_accessible_identity(self.open_file_btn, name="Open File", identifier="source.open")
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setObjectName("refreshButton")
        self.refresh_btn.setToolTip("Re-scan the current folder for new or changed CSV files")
        apply_accessible_identity(self.refresh_btn, name="Refresh", identifier="source.refresh")
        self.data_state_label = QLabel("Shown: no data")
        self.data_state_label.setObjectName("dataStateLabel")
        self.data_state_label.setToolTip("Shows whether the visible plot matches the current source selection")
        self.data_state_label.hide()
        self.recent_folder_combo = QComboBox()
        self.recent_folder_combo.setObjectName("recentFolderCombo")
        self.recent_folder_combo.setEditable(True)
        self.recent_folder_combo.setLineEdit(self.folder_edit)
        self.recent_folder_combo.setInsertPolicy(QComboBox.NoInsert)
        self.recent_folder_combo.setMinimumWidth(0)
        # Ignored lets this first (stretched) item contract before the fixed
        # readable action buttons when the toolbar reaches its minimum width.
        self.recent_folder_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.recent_folder_combo.setToolTip("Switch to a recently used data folder")
        apply_accessible_identity(
            self.recent_folder_combo,
            name="Recent data folders",
            description="Choose a recently used data folder",
            identifier="source.recent",
        )
        self._style_combo_popup(self.recent_folder_combo)
        source_label = QLabel("Source:")
        source_label.setObjectName("sourceLabel")
        apply_accessible_identity(source_label, name="Data source")
        folder_grid.addWidget(source_label)
        folder_grid.addWidget(self.recent_folder_combo, 1)
        folder_grid.addWidget(self.browse_btn)
        folder_grid.addWidget(self.open_file_btn)
        folder_grid.addWidget(self.refresh_btn)
        folder_grid.addWidget(self.data_state_label)
        QWidget.setTabOrder(self.recent_folder_combo, self.browse_btn)
        QWidget.setTabOrder(self.browse_btn, self.open_file_btn)
        QWidget.setTabOrder(self.open_file_btn, self.refresh_btn)
        self.data_source_context = folder_box

        self.tabs = QTabWidget()
        self.tabs.tabBar().setExpanding(True)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.tabs.tabBar().setUsesScrollButtons(False)
        for feature in FEATURES:
            self._expander_workflow_context = feature.key
            page = feature.build(self)
            if feature.scrollable:
                page = self._make_scrollable_tab(page, feature.key)
            index = self.tabs.addTab(page, feature.label)
            self.tabs.setTabToolTip(index, feature.description)
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
        # Every workflow page lives in the fixed-width controls sidebar.  An
        # ignored horizontal policy lets the page follow the viewport width
        # instead of keeping a wide size hint that would be clipped while the
        # window is at its supported minimum size.
        page.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
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
        # Keep the DRR cursor readable at millivolt precision.
        spins["gate"].setDecimals(3 if prefix == "drr" else 9)
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
            fix_checks[k].setMinimumWidth(0)
            fix_checks[k].setMaximumWidth(34)
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
        # Let the narrow controls sidebar allocate this combo to the grid
        # column instead of allowing its natural text width to overflow.
        cmap.setMinimumWidth(0)
        cmap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        toggle = QCheckBox(
            "Use split color scale", self if prefix == "mcd" else None
        )
        if prefix == "mcd":
            toggle.hide()
        if prefix == "drr_second":
            # The DRR checkbox is the single user-facing switch for both
            # products; this hidden state holder keeps product-specific
            # d2E limits in the same parameter pipeline.
            toggle.hide()
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

        panel = QGroupBox(
            "Two X-Region Color Limits", self if prefix == "mcd" else None
        )
        grid = QGridLayout(panel)
        grid.setContentsMargins(4, 5, 4, 4)
        grid.setHorizontalSpacing(3)
        grid.setVerticalSpacing(3)
        show_boundary = QCheckBox("Show boundary")
        show_boundary.setChecked(True)
        auto_left = QToolButton()
        auto_left.setText("Auto")
        auto_left.setToolTip("Automatically set unlocked limits from data between xmin and x0.")
        auto_right = QToolButton()
        auto_right.setText("Auto")
        auto_right.setToolTip("Automatically set unlocked limits from data between x0 and xmax.")

        if prefix in {"drr", "drr_second"}:
            # DRR uses the compact sidebar form: one control per row keeps
            # the input and Fix checkbox readable at the minimum width.
            grid.addWidget(QLabel("x0"), 0, 0)
            grid.addWidget(split_spins["x0"], 0, 1, 1, 2)
            grid.addWidget(split_fix_checks["x0"], 0, 3)
            grid.addWidget(show_boundary, 1, 0, 1, 4)
            left_title = QLabel("L  xmin → x0")
            set_fluent_property(left_title, "appRole", "regionTitle")
            grid.addWidget(left_title, 2, 0, 1, 3)
            grid.addWidget(auto_left, 2, 3)
            grid.addWidget(QLabel("min"), 3, 0)
            grid.addWidget(split_spins["left_vmin"], 3, 1, 1, 2)
            grid.addWidget(split_fix_checks["left_vmin"], 3, 3)
            grid.addWidget(QLabel("max"), 4, 0)
            grid.addWidget(split_spins["left_vmax"], 4, 1, 1, 2)
            grid.addWidget(split_fix_checks["left_vmax"], 4, 3)
            right_title = QLabel("R  x0 → xmax")
            set_fluent_property(right_title, "appRole", "regionTitle")
            grid.addWidget(right_title, 5, 0, 1, 3)
            grid.addWidget(auto_right, 5, 3)
            grid.addWidget(QLabel("min"), 6, 0)
            grid.addWidget(split_spins["right_vmin"], 6, 1, 1, 2)
            grid.addWidget(split_fix_checks["right_vmin"], 6, 3)
            grid.addWidget(QLabel("max"), 7, 0)
            grid.addWidget(split_spins["right_vmax"], 7, 1, 1, 2)
            grid.addWidget(split_fix_checks["right_vmax"], 7, 3)
        else:
            grid.addWidget(QLabel("Split position (x0)"), 0, 0, 1, 2)
            grid.addWidget(split_spins["x0"], 0, 2, 1, 2)
            grid.addWidget(split_fix_checks["x0"], 0, 4)
            grid.addWidget(show_boundary, 0, 5)
            left_title = QLabel("Region 1: xmin → x0")
            set_fluent_property(left_title, "appRole", "regionTitle")
            grid.addWidget(left_title, 1, 0, 1, 4)
            grid.addWidget(auto_left, 1, 4, 1, 2)
            grid.addWidget(QLabel("vmin"), 2, 0)
            grid.addWidget(split_spins["left_vmin"], 2, 1, 1, 2)
            grid.addWidget(split_fix_checks["left_vmin"], 2, 3)
            grid.addWidget(QLabel("vmax"), 3, 0)
            grid.addWidget(split_spins["left_vmax"], 3, 1, 1, 2)
            grid.addWidget(split_fix_checks["left_vmax"], 3, 3)
            right_title = QLabel("Region 2: x0 → xmax")
            set_fluent_property(right_title, "appRole", "regionTitle")
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
        host = QWidget(self)
        host.hide()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        combo = QComboBox()
        combo.addItems(["Auto / Default", "TG", "BG", "Bias", "Advanced..."])
        combo.setToolTip("Choose how the plot y-axis is derived from gate variables.")
        combo.setMinimumWidth(0)
        combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
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



    def _set_form_label_width(self, form: QFormLayout, width: int) -> None:
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.LabelRole)
            if item and isinstance(item.widget(), QLabel):
                lbl = item.widget()
                # QLabel cannot break a word. Reserve the rendered width of
                # the longest token before enabling wrapping so labels such
                # as ``Integration wavelength`` remain fully visible.
                longest_word = max(
                    (lbl.fontMetrics().horizontalAdvance(word) for word in lbl.text().split()),
                    default=0,
                )
                lbl.setFixedWidth(max(width, longest_word + 4))
                # Long form labels must participate in the form's height-for-
                # width calculation. A fixed-width, single-line QLabel paints
                # past its right edge when QFormLayout wraps the field.
                lbl.setWordWrap(True)
                lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def _make_axis_range_row(
        self,
        a: QDoubleSpinBox,
        b: QDoubleSpinBox,
        fa: QCheckBox,
        fb: QCheckBox,
        auto_btn: QToolButton,
        auto_text: str,
        *,
        dense: bool = False,
        label_text: str | None = None,
    ) -> QWidget:
        """Build a min/max spin pair with Fix checkboxes and an Auto button."""
        auto_btn.setText("Auto" if dense else auto_text)
        if hasattr(auto_btn, "setAutoRaise"):
            auto_btn.setAutoRaise(True)
        # Keep the compact action readable while allowing the row to flex at
        # the supported narrow sidebar width.
        auto_width = max(UI_METRICS["tool_w"], auto_btn.fontMetrics().horizontalAdvance(auto_text) + 12)
        auto_btn.setMinimumWidth(auto_width)
        auto_btn.setMaximumWidth(auto_width)
        auto_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        auto_btn.setMinimumHeight(UI_METRICS["tool_h"])
        auto_btn.setToolTip(f"Set {auto_text.replace('Auto ', '')} bounds automatically from loaded data")
        row = QWidget()
        if dense:
            # Stage 1 integrates the metric-driven layout only for DRR Axis
            # Ranges. Other workflow rows intentionally keep this helper's
            # existing grid behavior until their migration stage.
            for spin in (a, b):
                spin.setMinimumWidth(0)
                # Keep the dense DRR control's proven ordinary-value width
                # stable in compact wrapped rows; expansion remains available
                # to unconstrained text-bearing controls in DenseFormRowLayout.
                spin.setMaximumWidth(130)
                spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            for fix in (fa, fb):
                fix.setMinimumWidth(0)
                fix.setMaximumWidth(row.maximumWidth())
            auto_btn.setMinimumWidth(0)
            auto_btn.setMaximumWidth(row.maximumWidth())
            label = QLabel(label_text) if label_text else None
            if label is not None:
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            dense_layout = DenseFormRowLayout(row, spacing=4, label=label)
            row.setLayout(dense_layout)
            dense_layout.add_group((a, fa), role="range", priority=10, grow_weight=1)
            dense_layout.add_group((b, fb), role="range", priority=10, grow_weight=1)
            dense_layout.add_group((auto_btn,), role="action", priority=1, grow_weight=0)
            return row
        # Keep the normal application font so formatted values remain
        # readable.  The pair controls occupy the first grid row and the
        # Auto action wraps below when the sidebar reaches its minimum width;
        # this avoids compressing either spinbox to an unusable edit field.
        for spin in (a, b):
            # Reserve enough edit-field width for the ordinary ``-12.0000``
            # sentinel while retaining the native stepper controls.
            spin.setMinimumWidth(max(spin.minimumWidth(), 128))
            spin.setMaximumWidth(max(spin.maximumWidth(), 130))
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        fix_width = max(34, fa.minimumSizeHint().width(), fb.minimumSizeHint().width())
        for fix in (fa, fb):
            # Keep both the indicator and the visible ``F`` label within the
            # checkbox hit target at the narrow sidebar width.
            fix.setMinimumWidth(fix_width)
            fix.setMaximumWidth(fix_width)
        grid = QGridLayout(row)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)
        # Each range pair gets its own line; the row widget itself remains a
        # single direct-child region so callers can reason about containment.
        grid.addWidget(a, 0, 0, 1, 3)
        grid.addWidget(fa, 0, 3)
        grid.addWidget(b, 1, 0, 1, 3)
        grid.addWidget(fb, 1, 3)
        grid.addWidget(auto_btn, 2, 0, 1, 4, Qt.AlignLeft)
        for column in range(3):
            grid.setColumnStretch(column, 1)
        return row

    def _style_combo_popup(self, combo: QComboBox) -> None:
        # Combo popup styling is provided by the application QSS
        # (`QComboBox QAbstractItemView`), so no local sheet is needed.
        return None

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














    def _make_expander(self, title: str, content: QWidget, *, expanded: bool = True) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 3, 0, 4)
        v.setSpacing(4)
        head = QToolButton()
        head.setCheckable(True)
        workflow = getattr(self, "_expander_workflow_context", "common")
        state_key = "ui/expanders/" + str(workflow) + "/" + (content.objectName() or title).replace(" ", "_")
        stored = self.settings.value(state_key, None) if hasattr(self, "settings") else None
        if stored is not None:
            expanded = str(stored).strip().casefold() in {"1", "true", "yes", "on"}
        head.setChecked(bool(expanded))
        head.setAutoRaise(True)
        head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        set_fluent_property(head, "appRole", "expanderHead")
        line = QFrame()
        line.setFrameShape(QFrame.NoFrame)
        set_fluent_property(line, "fluentRole", "divider")
        row = QWidget()
        row_h = QHBoxLayout(row)
        row_h.setContentsMargins(0, 0, 0, 0)
        row_h.setSpacing(6)
        row_h.addWidget(head)
        row_h.addWidget(line, 1)
        v.addWidget(row)
        content.setVisible(bool(expanded))
        v.addWidget(content)

        def _update(on: bool, *, persist: bool = True) -> None:
            head.setIcon(
                self.style().standardIcon(QStyle.SP_ArrowDown if on else QStyle.SP_ArrowRight)
            )
            head.setText(title)
            content.setVisible(bool(on))
            if persist and hasattr(self, "settings"):
                self.settings.setValue(state_key, bool(on))

        _update(bool(expanded), persist=False)
        head.toggled.connect(_update)
        return box



    @staticmethod
    def _pair_row(*widgets: QWidget) -> QWidget:
        # Dense MCD rows wrap each control onto its own line at the supported
        # sidebar width so numerical fields retain a readable edit area.
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for widget in widgets:
            layout.addWidget(widget)
        return row


    def _build_power_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        grouping = QGroupBox("Power Sweep Files")
        grouping_form = QFormLayout(grouping)
        grouping_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        grouping_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        grouping_form.setHorizontalSpacing(6)
        grouping_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        self.power_group_combo = QComboBox()
        self.power_group_combo.setMinimumWidth(0)
        self.power_group_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._style_combo_popup(self.power_group_combo)
        self.power_group_combo.setToolTip("Select a full-sweep CSV with Power_uW, or a legacy filename-based series.")
        self.power_refresh_groups_btn = QPushButton("Refresh")
        self.power_refresh_groups_btn.setToolTip("Find power sweeps in this folder and measurement subfolders; old/archive folders and analysis exports are excluded.")
        group_row = QWidget()
        group_h = QVBoxLayout(group_row)
        group_h.setContentsMargins(0, 0, 0, 0)
        group_h.setSpacing(6)
        group_h.addWidget(self.power_group_combo, 1)
        self.power_group_combo.hide()
        self.power_choose_btn = QPushButton("Open sweep…")
        self.power_choose_btn.clicked.connect(lambda _checked=False: self.power_controller._power_choose_dataset())
        group_h.addWidget(self.power_choose_btn)
        self.power_compare_choose_btn = QPushButton("Compare…")
        self.power_compare_choose_btn.clicked.connect(lambda _checked=False: self.power_controller._power_choose_measurement_group(mode='compare'))
        group_h.addWidget(self.power_compare_choose_btn)
        self.power_combine_btn = QPushButton("Combine sweeps…")
        self.power_combine_btn.setToolTip("Combine explicitly selected segments from two sweeps of the same channel.")
        self.power_combine_btn.clicked.connect(self.power_controller._on_power_combine)
        group_h.addWidget(self.power_combine_btn)
        group_h.addWidget(self.power_refresh_groups_btn)
        self.power_group_summary = QPlainTextEdit()
        self.power_group_summary.setReadOnly(True)
        self.power_group_summary.setMaximumHeight(88)
        self.power_kk_group_combo = QComboBox()
        self.power_kkp_group_combo = QComboBox()
        for combo in (self.power_kk_group_combo, self.power_kkp_group_combo):
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._style_combo_popup(self.power_kk_group_combo)
        self._style_combo_popup(self.power_kkp_group_combo)
        self.power_kk_group_combo.setToolTip("KK source for intensity comparison and VP.")
        self.power_kkp_group_combo.setToolTip("KKp source for intensity comparison and VP.")
        grouping_form.addRow(group_row)
        grouping_form.addRow("Summary", self.power_group_summary)
        self.power_compare_chk = QCheckBox("Compare KK / KKp")
        grouping_form.addRow(self.power_compare_chk)
        self.power_roles_widget = QWidget()
        roles_form = QFormLayout(self.power_roles_widget)
        roles_form.setContentsMargins(0, 0, 0, 0)
        roles_form.addRow("KK sweep", self.power_kk_group_combo)
        roles_form.addRow("KKp sweep", self.power_kkp_group_combo)
        for role in ('KK', 'KKp'):
            choose_role = QPushButton(f"Choose {role} sweep…")
            choose_role.setObjectName(f'power_choose_{role.lower()}')
            choose_role.clicked.connect(lambda _checked=False, role=role: self.power_controller._power_choose_dataset(role))
            roles_form.addRow(choose_role)
        role_hint = QLabel("Select two different sweeps for intensity comparison or VP. A single intensity sweep needs no KK/KKp assignment.")
        role_hint.setWordWrap(True)
        roles_form.addRow(role_hint)
        self.power_roles_widget.setVisible(False)
        grouping_form.addRow(self.power_roles_widget)
        layout.addWidget(self._make_expander("Power Sweep Files", grouping, expanded=True))

        params = QWidget()
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(4)

        _grid, spins, _, _, cmap, fix_checks = self._build_common_range_grid("power", "turbo")
        self.power_axis_scale_combo = QComboBox()
        self.power_axis_scale_combo.addItems(["Linear", "Log"])
        self._style_combo_popup(self.power_axis_scale_combo)
        self.power_axis_scale_combo.setToolTip("Set the heatmap power axis and intensity/linewidth trend power axes together.")
        self.power_pair_mode_combo = QComboBox()
        self.power_pair_mode_combo.addItems(["Stage", "Power Interpolation"])
        self._style_combo_popup(self.power_pair_mode_combo)
        self.power_pair_mode_combo.setToolTip("Choose how KK and KKp spectra are paired for VP.")
        for combo in (self.power_axis_scale_combo, self.power_pair_mode_combo):
            combo.setMinimumWidth(0)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.power_background_spin = QDoubleSpinBox()
        self.power_background_spin.setDecimals(6)
        self.power_background_spin.setRange(-1.0e12, 1.0e12)
        self.power_background_spin.setSingleStep(100.0)
        self.power_background_spin.setMinimumWidth(0)
        self.power_background_spin.setMaximumWidth(180)
        self.power_background_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.power_background_auto_chk = QCheckBox("Auto")
        self.power_background_auto_chk.setMinimumWidth(0)
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
        setup_form = QFormLayout(setup)
        # Let wrapped long combo fields use the complete inner width at the
        # minimum sidebar size; the group frame already supplies edge padding.
        setup_form.setContentsMargins(0, UI_METRICS["group_margin"], 0, UI_METRICS["group_margin"])
        setup_form.setHorizontalSpacing(8)
        setup_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        setup_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        setup_form.addRow("Power Axis", self.power_axis_scale_combo)
        setup_form.addRow("Cmap", cmap)
        setup_form.addRow("VP Pair By", self.power_pair_mode_combo)
        setup_form.addRow("Background", bkg_row)
        params_layout.addWidget(self._make_expander("Plot Setup", setup, expanded=True))
        params_layout.addWidget(self._make_expander(
            "Peak Analysis", self.power_peak_controller.build_controls(), expanded=True))

        for s in spins.values():
            s.setMinimumWidth(116)
            s.setMaximumWidth(130)
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            s.setMinimumHeight(s.minimumSizeHint().height())

        self.power_auto_v_btn = QToolButton()
        self.power_auto_x_btn = QToolButton()
        self.power_auto_y_btn = QToolButton()
        basic = QGroupBox("Axis Ranges")
        basic_form = QFormLayout(basic)
        basic_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        basic_form.setHorizontalSpacing(4)
        basic_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        basic_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        power_v_range_row = self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.power_auto_v_btn, "Auto V", dense=True, label_text="vmin / vmax")
        power_x_range_row = self._make_axis_range_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.power_auto_x_btn, "Auto X", dense=True, label_text="xmin / xmax")
        power_y_range_row = self._make_axis_range_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.power_auto_y_btn, "Auto Y", dense=True, label_text="pmin / pmax")
        power_range_rows = (power_v_range_row, power_x_range_row, power_y_range_row)
        basic_form.addRow(power_v_range_row)
        basic_form.addRow("Color scale", self.power_split_scale_chk)
        basic_form.addRow(self.power_split_scale_panel)
        basic_form.addRow(power_x_range_row)
        basic_form.addRow(power_y_range_row)
        basic_form.addRow("Cursor Power", spins["gate"])
        flags = QWidget()
        flags_h = QVBoxLayout(flags)
        flags_h.setContentsMargins(0, 0, 0, 0)
        flags_h.setSpacing(3)
        self.power_log_chk.setText("Color Log")
        self.power_log_chk.setToolTip("Use logarithmic color normalization.")
        flags_h.addWidget(self.power_log_chk)
        flags_h.addWidget(self.power_clip_chk)
        basic_form.addRow("Color / Clip", flags)
        self._set_form_label_width(basic_form, UI_METRICS["label_col_width"])
        manual_ranges = self._make_expander("Manual plot ranges", basic, expanded=False)
        manual_ranges_head = manual_ranges.findChild(QToolButton)

        def _contain_power_range_rows(expanded: bool) -> None:
            if not expanded:
                return
            for range_row in power_range_rows:
                dense_layout = range_row.layout()
                range_row.setMinimumHeight(dense_layout.heightForWidth(range_row.width()))
            basic_form.invalidate()
            basic.setMinimumHeight(basic.minimumSizeHint().height())
            basic_form.activate()

        if manual_ranges_head is not None:
            manual_ranges_head.toggled.connect(_contain_power_range_rows)
        params_layout.addWidget(manual_ranges)

        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addStretch(1)
        return tab

    def _open_mcd_extract_dialog(self) -> None:
        """Launch the focused organizer independently from the plotting app."""
        start = str(self.current_folder or self._browse_start_folder())
        if getattr(sys, "frozen", False):
            program = sys.executable
            arguments = ["--mcd-organizer", start]
        else:
            launcher = Path(__file__).resolve().parents[1] / "run_mcd_organizer.py"
            program = sys.executable
            arguments = [str(launcher), start]
        launched, _process_id = QProcess.startDetached(
            program, arguments, str(Path(start).expanduser())
        )
        if not launched:
            QMessageBox.critical(
                self,
                "MCD Organizer",
                "The standalone MCD Organizer could not be started.",
            )
            return
        self._status("Opened the MCD Organizer in a separate window.")

    def _schedule_plot_redraw(self, mode: str, delay_ms: int = 90) -> None:
        """Queue one redraw for *mode*, coalescing control signal bursts."""
        timer = self._plot_redraw_timers.get(mode)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda m=mode: self._run_scheduled_plot_redraw(m))
            self._plot_redraw_timers[mode] = timer
        self._plot_redraw_pending.add(mode)
        timer.start(max(0, int(delay_ms)))

    def _run_scheduled_plot_redraw(self, mode: str) -> None:
        if getattr(self, "_is_closing", False):
            return
        active_mode = self._active_mode() if callable(getattr(self, "_active_mode", None)) else None
        # The Peak Shift page intentionally has no standalone active-mode
        # controller; ``None`` there still means the shared canvas is owned by
        # that page.  Hidden redraws must remain dirty for every workflow.
        if active_mode is None:
            tabs = getattr(self, "tabs", None)
            tab_text = tabs.tabText(tabs.currentIndex()) if tabs is not None else None
            if tabs is None or tab_text == "MCD Peak Shift":
                active_mode = "MCD Peak Shift"
        if active_mode != mode:
            # Keep the dirty request until its page is visible.  A hidden
            # controller must never draw into a shared canvas owned by the
            # currently visible page.
            return
        self._plot_redraw_pending.discard(mode)
        if self._load_in_progress or not self.loaded or self.loaded.mode != mode:
            return
        if mode == 'Power Dependent' and getattr(self, '_power_pending_range_refresh', False):
            center = bool(getattr(self, '_power_pending_center_split', False))
            self._power_pending_range_refresh = False
            self._power_pending_center_split = False
            self._refresh_automatic_ranges(mode, refresh_split=True, center_split=center)
        elif mode in self._pending_range_refresh:
            center = self._pending_range_refresh.pop(mode)
            self._refresh_automatic_ranges(mode, refresh_split=True, center_split=center)
        self._plot_mode(mode)

    def _on_drr_plot_view_changed(self, view: str, *, redraw: bool = True) -> None:
        view = str(view)
        if view not in {"raw", "second"}:
            return
        self._capture_drr_view_limits()
        if view != self._drr_plot_view:
            self.drr_controller._invalidate_drr_analysis_product()
        self._drr_plot_view = view
        buttons = {"raw": self.drr_view_raw_btn, "second": self.drr_view_second_btn}
        for key, button in buttons.items():
            blocked = button.blockSignals(True)
            button.setChecked(key == view)
            button.blockSignals(blocked)
        if redraw and self.last_plotted_mode == "DRR" and self.loaded and self.loaded.mode == "DRR":
            self._plot_mode("DRR")

    def _drr_gate_input_value(self) -> float:
        return float(self.drr_spins["gate"].value())

    def _set_drr_gate_toolbar_value(self, value: float) -> None:
        toolbar = getattr(self, "drr_gate_toolbar_spin", None)
        if toolbar is None:
            return
        blocked = toolbar.blockSignals(True)
        try:
            toolbar.setValue(float(value))
        finally:
            toolbar.blockSignals(blocked)

    def _on_drr_sidebar_gate_changed(self, spin) -> None:
        self._set_drr_gate_toolbar_value(float(spin.value()))
        self.drr_controller._on_drr_plot_param_changed(spin)

    def _on_drr_toolbar_gate_changed(self, value: float) -> None:
        sidebar = self.drr_spins["gate"]
        blocked = sidebar.blockSignals(True)
        try:
            sidebar.setValue(float(value))
        finally:
            sidebar.blockSignals(blocked)
        self.drr_controller._on_drr_plot_param_changed(self.drr_gate_toolbar_spin)
        self._sync_drr_gate_toolbar_state()

    def _step_drr_gate(self, direction: int) -> None:
        cube = getattr(self, "_last_plot_cube", None)
        if self.loaded is None or self.loaded.mode != "DRR" or cube is None:
            return
        gate = np.asarray(cube.gate, float).ravel()
        finite = np.flatnonzero(np.isfinite(gate))
        if finite.size == 0:
            return
        current = self._drr_gate_input_value()
        nearest = int(finite[np.argmin(np.abs(gate[finite] - current))])
        target = nearest + (1 if int(direction) > 0 else -1)
        if target < 0 or target >= gate.size or not np.isfinite(gate[target]):
            self._sync_drr_gate_toolbar_state()
            return
        target_gate = float(gate[target])
        self._set_drr_gate_toolbar_value(target_gate)
        sidebar = self.drr_spins["gate"]
        blocked = sidebar.blockSignals(True)
        try:
            sidebar.setValue(target_gate)
        finally:
            sidebar.blockSignals(blocked)
        self.drr_controller._on_drr_plot_param_changed(self.drr_gate_toolbar_spin)
        self._sync_drr_gate_toolbar_state()

    def _sync_drr_gate_toolbar_state(self, drr_loaded: bool | None = None) -> None:
        toolbar = getattr(self, "drr_gate_toolbar_spin", None)
        prev = getattr(self, "drr_gate_prev_btn", None)
        nxt = getattr(self, "drr_gate_next_btn", None)
        if toolbar is None or prev is None or nxt is None:
            return
        if drr_loaded is None:
            drr_loaded = bool(self.loaded is not None and self.loaded.mode == "DRR")
        cube = getattr(self, "_last_plot_cube", None)
        gate = np.asarray(cube.gate, float).ravel() if cube is not None else np.empty(0)
        finite = np.flatnonzero(np.isfinite(gate))
        enabled = bool(drr_loaded and finite.size)
        toolbar.setEnabled(enabled)
        if not enabled:
            prev.setEnabled(False)
            nxt.setEnabled(False)
            return
        current = self._drr_gate_input_value()
        index = int(finite[np.argmin(np.abs(gate[finite] - current))])
        prev.setEnabled(index > int(finite[0]))
        nxt.setEnabled(index < int(finite[-1]))

    def _on_drr_side_by_side_changed(self, checked: bool) -> None:
        self._capture_drr_view_limits()
        self._drr_side_by_side = bool(checked)
        # Side-by-side is a layout preference.  Keep the selected product
        # independent so switching it never changes what Save exports.
        if self.last_plotted_mode == "DRR" and self.loaded and self.loaded.mode == "DRR":
            self._plot_mode("DRR")

    def _on_drr_second_scale_changed(self) -> None:
        self.drr_second_auto_scale = False
        self._on_drr_plot_view_changed(self._drr_plot_view, redraw=True)

    def _update_drr_advanced_derivative_label(self) -> None:
        label = getattr(self, "drr_advanced_derivative_label", None)
        if label is None:
            return
        text = self.drr_derivative_combo.currentText()
        label.setText("Advanced: first derivative (dE)" if text == "dE" else f"Advanced: {text}")

    def _auto_drr_second_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        try:
            cube, _derivative, _window, _poly = self.drr_controller._drr_cube_with_metadata(2)
            limits = compute_auto_limits(cube, log_scale=False)
        except (TypeError, ValueError) as exc:
            self._status(f"State: Auto d2E vmin/vmax skipped ({exc}).")
            return
        for spin, value in (
            (self.drr_second_vmin_spin, limits.vmin),
            (self.drr_second_vmax_spin, limits.vmax),
        ):
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(float(value))
            finally:
                spin.blockSignals(blocked)
        self.drr_second_auto_scale = True
        self._status(f"State: Auto d2E vmin/vmax = {limits.vmin:.4g}, {limits.vmax:.4g}")
        self._schedule_plot_redraw("DRR")

    def _sync_drr_second_auto_scale(self, cube: DataCube) -> None:
        if not getattr(self, "drr_second_auto_scale", True):
            return
        try:
            limits = compute_auto_limits(cube, log_scale=False)
        except (TypeError, ValueError):
            return
        for spin, value in (
            (self.drr_second_vmin_spin, limits.vmin),
            (self.drr_second_vmax_spin, limits.vmax),
        ):
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(float(value))
            finally:
                spin.blockSignals(blocked)

    def _capture_drr_view_limits(self) -> None:
        """Snapshot the shared map viewport before a DRR layout rebuild."""
        current_axes = set(getattr(self.figure, "axes", ()))
        axes = getattr(self, "_drr_heatmap_axes", {}) or {}
        axis = getattr(self, "_drr_heatmap_ax", None)
        if axis not in current_axes:
            axis = next((candidate for candidate in axes.values() if candidate in current_axes), None)
        if axis is None:
            return
        self._drr_view_limits = (
            tuple(float(value) for value in axis.get_xlim()),
            tuple(float(value) for value in axis.get_ylim()),
        )

    def _update_plot_view_bar_visibility(self) -> None:
        if hasattr(self, "drr_plot_view_bar"):
            drr_active = self._active_mode() == "DRR"
            self.drr_plot_view_bar.setVisible(drr_active)
            mcd_active = self._active_mode() == "MCD" and hasattr(self, "mcd_unified_view")
            self.save_action.setText(
                "Save both · PNG + DAT" if drr_active else
                "Save unified MCD results" if mcd_active else "Save PNG + DAT"
            )
            self.save_action.setToolTip(
                "Export both ΔR/R and second-derivative PNG + DAT products"
                if drr_active else
                "Export the frozen unified MCD snapshot, spectra, tracks, slopes, and retained windows"
                if mcd_active else "Export for the active tab"
            )
        if hasattr(self, "cmp_plot_view_bar"):
            self.cmp_plot_view_bar.setVisible(self._active_mode() == "Compare")
        if hasattr(self, "power_plot_view_bar"):
            self.power_plot_view_bar.setVisible(self._active_mode() == "Power Dependent")
        unified_active = self._active_mode() == "MCD" and getattr(self, "loaded", None) is not None and getattr(self.loaded, "mode", None) == "MCD"
        if hasattr(self, "mcd_unified_view"):
            self.mcd_unified_view.setVisible(unified_active)
        if hasattr(self, "mcd_candidate_bar"):
            # The unified surface owns the only visible MCD candidate bar.
            # Keep the legacy controls alive for compatibility with old
            # controllers and tests, but never show a second selector row.
            active = self._active_mode() == "MCD" and not hasattr(self, "mcd_unified_view")
            self.mcd_candidate_bar.setVisible(active)
            self.mcd_find_centers_btn.setEnabled(
                self._active_mode() == "MCD"
                and self.loaded is not None
                and self.loaded.mode == "MCD"
                and self.loaded.mcd_result is not None
            )
        if hasattr(self, "mcd_peak_candidate_bar"):
            peak_active = (
                self.tabs.tabText(self.tabs.currentIndex()) == "MCD Peak Shift"
            )
            self.mcd_peak_candidate_bar.setVisible(peak_active)
            if hasattr(self, "_update_mcd_peak_candidate_buttons"):
                self._update_mcd_peak_candidate_buttons()

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
            # These combos sit in rows shared with the y-axis selector.  A
            # hard minimum wider than the sidebar causes clipping at the
            # supported minimum window size, so let the row elide naturally.
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _build_plot_panel(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.canvas = ThemeAwareFigureCanvasQTAgg(self.figure)
        self.toolbar = _PlotToolbar(self.canvas, box)
        layout.addWidget(self.toolbar)
        self.drr_plot_view_bar = QFrame()
        self.drr_plot_view_bar.setFrameShape(QFrame.NoFrame)
        self.drr_plot_view_bar.setVisible(False)
        set_fluent_property(self.drr_plot_view_bar, "fluentRole", "panel")
        drr_view_layout = QHBoxLayout(self.drr_plot_view_bar)
        drr_view_layout.setContentsMargins(8, 4, 8, 4)
        drr_view_layout.setSpacing(0)
        self.drr_view_raw_btn = QToolButton()
        self.drr_view_raw_btn.setText("ΔR/R")
        self.drr_view_raw_btn.setToolTip("Show the baseline-corrected ΔR/R product")
        self.drr_view_raw_btn.setCheckable(True)
        self.drr_view_raw_btn.setChecked(True)
        self.drr_view_second_btn = QToolButton()
        self.drr_view_second_btn.setText("Second derivative")
        self.drr_view_second_btn.setToolTip("Show the energy second derivative of ΔR/R")
        self.drr_view_second_btn.setCheckable(True)
        self.drr_view_side_btn = QToolButton()
        self.drr_view_side_btn.setText("Side by side")
        self.drr_view_side_btn.setToolTip("Arrange the two DRR products side by side")
        self.drr_view_side_btn.setCheckable(True)
        for button in (self.drr_view_raw_btn, self.drr_view_second_btn):
            drr_view_layout.addWidget(button)
        drr_view_layout.addSpacing(8)
        drr_view_layout.addWidget(self.drr_view_side_btn)
        drr_view_layout.addSpacing(12)
        drr_gate_label = QLabel("Gate")
        drr_gate_label.setToolTip("Gate voltage used for the DRR spectrum linecut")
        self.drr_gate_toolbar_spin = QDoubleSpinBox()
        self.drr_gate_toolbar_spin.setObjectName("drr_gate_toolbar_spin")
        self.drr_gate_toolbar_spin.setDecimals(3)
        self.drr_gate_toolbar_spin.setRange(-1e12, 1e12)
        self.drr_gate_toolbar_spin.setSingleStep(0.1)
        self.drr_gate_toolbar_spin.setMaximumWidth(160)
        self.drr_gate_toolbar_spin.setSuffix(" V")
        self.drr_gate_toolbar_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.drr_gate_toolbar_spin.setKeyboardTracking(False)
        self.drr_gate_toolbar_spin.setEnabled(False)
        self.drr_gate_prev_btn = QToolButton()
        self.drr_gate_prev_btn.setObjectName("drr_gate_prev_btn")
        self.drr_gate_prev_btn.setText("Prev")
        self.drr_gate_prev_btn.setToolTip("Select the previous sampled gate")
        self.drr_gate_prev_btn.setEnabled(False)
        self.drr_gate_next_btn = QToolButton()
        self.drr_gate_next_btn.setObjectName("drr_gate_next_btn")
        self.drr_gate_next_btn.setText("Next")
        self.drr_gate_next_btn.setToolTip("Select the next sampled gate")
        self.drr_gate_next_btn.setEnabled(False)
        drr_view_layout.addWidget(drr_gate_label)
        drr_view_layout.addWidget(self.drr_gate_toolbar_spin)
        drr_view_layout.addWidget(self.drr_gate_prev_btn)
        drr_view_layout.addWidget(self.drr_gate_next_btn)
        drr_view_layout.addStretch(1)
        layout.addWidget(self.drr_plot_view_bar)
        self.cmp_plot_view_bar = QFrame()
        self.cmp_plot_view_bar.setFrameShape(QFrame.NoFrame)
        self.cmp_plot_view_bar.setVisible(False)
        set_fluent_property(self.cmp_plot_view_bar, "fluentRole", "panel")
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
        set_fluent_property(self.power_plot_view_bar, "fluentRole", "panel")
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
        self.mcd_candidate_bar = QFrame()
        self.mcd_candidate_bar.setFrameShape(QFrame.NoFrame)
        self.mcd_candidate_bar.setVisible(False)
        set_fluent_property(self.mcd_candidate_bar, "fluentRole", "panel")
        mcd_candidate_layout = QHBoxLayout(self.mcd_candidate_bar)
        mcd_candidate_layout.setContentsMargins(8, 4, 8, 4)
        mcd_candidate_layout.setSpacing(4)
        self.mcd_find_centers_btn = QToolButton()
        self.mcd_find_centers_btn.setText("Find centers")
        self.mcd_find_centers_btn.setToolTip(
            "Find up to five distinct fixed-width energy windows with strong, repeatable field-odd MCD."
        )
        self.mcd_candidate_label = QLabel("Suggested:")
        self.mcd_candidate_buttons: list[QToolButton] = []
        mcd_candidate_layout.addWidget(self.mcd_find_centers_btn)
        mcd_candidate_layout.addWidget(self.mcd_candidate_label)
        for index in range(5):
            button = QToolButton()
            button.setCheckable(True)
            button.setVisible(False)
            button.clicked.connect(
                lambda _checked=False, candidate_index=index: self.mcd_controller._use_mcd_center_candidate(candidate_index)
            )
            self.mcd_candidate_buttons.append(button)
            mcd_candidate_layout.addWidget(button)
        self.mcd_previous_candidate_btn = QToolButton()
        self.mcd_previous_candidate_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        set_fluent_property(self.mcd_previous_candidate_btn, "fluentIconOnly", True)
        self.mcd_previous_candidate_btn.setEnabled(False)
        self.mcd_previous_candidate_btn.setToolTip("Preview the previous suggested center")
        apply_accessible_identity(
            self.mcd_previous_candidate_btn,
            name="Previous suggested center",
            description="Preview the previous suggested center",
        )
        self.mcd_next_candidate_btn = QToolButton()
        self.mcd_next_candidate_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        set_fluent_property(self.mcd_next_candidate_btn, "fluentIconOnly", True)
        self.mcd_next_candidate_btn.setEnabled(False)
        self.mcd_next_candidate_btn.setToolTip("Preview the next suggested center")
        apply_accessible_identity(
            self.mcd_next_candidate_btn,
            name="Next suggested center",
            description="Preview the next suggested center",
        )
        self.mcd_clear_candidates_btn = QToolButton()
        self.mcd_clear_candidates_btn.setText("Return to manual")
        self.mcd_clear_candidates_btn.setEnabled(False)
        self.mcd_clear_candidates_btn.setToolTip(
            "Remove suggestions and restore the center used before the search"
        )
        mcd_candidate_layout.addWidget(self.mcd_previous_candidate_btn)
        mcd_candidate_layout.addWidget(self.mcd_next_candidate_btn)
        mcd_candidate_layout.addWidget(self.mcd_clear_candidates_btn)
        mcd_candidate_layout.addStretch(1)
        layout.addWidget(self.mcd_candidate_bar)
        self.mcd_peak_candidate_bar = QFrame()
        self.mcd_peak_candidate_bar.setFrameShape(QFrame.NoFrame)
        self.mcd_peak_candidate_bar.setVisible(False)
        set_fluent_property(self.mcd_peak_candidate_bar, "fluentRole", "panel")
        peak_candidate_layout = QHBoxLayout(self.mcd_peak_candidate_bar)
        peak_candidate_layout.setContentsMargins(8, 4, 8, 4)
        peak_candidate_layout.setSpacing(4)
        peak_candidate_layout.addWidget(QLabel("Peak center · E0:"))
        for widget in (
            self.mcd_peak_selector_combo,
            self.mcd_peak_prev_btn,
            self.mcd_peak_next_btn,
            *self.mcd_peak_candidate_buttons,
        ):
            widget.setParent(self.mcd_peak_candidate_bar)
        self.mcd_peak_selector_combo.setVisible(True)
        self.mcd_peak_selector_combo.setMinimumWidth(180)
        self.mcd_peak_selector_combo.setToolTip("Select any tracked peak center; nearby E0 candidates are shown as buttons.")
        peak_candidate_layout.addWidget(self.mcd_peak_selector_combo, 1)
        for button in self.mcd_peak_candidate_buttons:
            button.setVisible(False)
            peak_candidate_layout.addWidget(button)
        self.mcd_peak_prev_btn.setVisible(True)
        self.mcd_peak_next_btn.setVisible(True)
        self.mcd_peak_prev_btn.setToolTip("Select previous tracked peak center")
        self.mcd_peak_next_btn.setToolTip("Select next tracked peak center")
        peak_candidate_layout.addWidget(self.mcd_peak_prev_btn)
        peak_candidate_layout.addWidget(self.mcd_peak_next_btn)
        peak_candidate_layout.addStretch(1)
        layout.addWidget(self.mcd_peak_candidate_bar)
        self.mcd_unified_view = McdUnifiedView(
            self, figure=self.figure, canvas=self.canvas, embed_canvas=False
        )
        self.mcd_unified_view.candidate_combo.currentIndexChanged.connect(
            self._on_unified_candidate_selected
        )
        self.mcd_unified_view.candidate_selection_changed.connect(
            self._on_unified_candidate_selection_changed
        )
        self.mcd_unified_view.candidate_filter_combo.currentTextChanged.connect(
            self._on_unified_candidate_filter_changed
        )
        self.mcd_unified_view.detection_mode_changed.connect(self._on_unified_detection_mode_changed)
        self.mcd_unified_view.window_center_changed.connect(self.mcd_window_center_spin.setValue)
        self.mcd_unified_view.manual_candidate_added.connect(
            lambda _item: self._queue_unified_selected_feature_analysis()
        )
        self.mcd_unified_view.setVisible(False)
        layout.addWidget(self.mcd_unified_view)
        canvas_host = QWidget()
        canvas_host.setObjectName("plotCanvasHost")
        canvas_layout = QGridLayout(canvas_host)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        canvas_layout.addWidget(self.canvas, 0, 0)
        self.empty_canvas_overlay = QLabel(
            "Choose a valid source to display it automatically"
        )
        self.empty_canvas_overlay.setObjectName("emptyCanvasOverlay")
        self.empty_canvas_overlay.setAlignment(Qt.AlignCenter)
        self.empty_canvas_overlay.setWordWrap(True)
        self.empty_canvas_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.empty_canvas_overlay.setFocusPolicy(Qt.NoFocus)
        set_fluent_property(self.empty_canvas_overlay, "appRole", "emptyCanvas")
        apply_accessible_identity(
            self.empty_canvas_overlay,
            name="Plot canvas guidance",
            description="Choose a valid source to load and show a scientific plot.",
        )
        canvas_layout.addWidget(self.empty_canvas_overlay, 0, 0)
        self.canvas.mpl_connect("draw_event", self._sync_empty_canvas_overlay)
        self._sync_empty_canvas_overlay()
        self._mcd_unified_canvas_host = canvas_host
        layout.addWidget(canvas_host, 1)
        return box

    def _sync_empty_canvas_overlay(self, *_args: Any) -> None:
        """Show guidance only while the scientific Figure has no axes."""
        overlay = getattr(self, "empty_canvas_overlay", None)
        figure = getattr(self, "figure", None)
        if overlay is not None and figure is not None:
            overlay.setVisible(not bool(figure.axes))

    def _update_results_dock_page(self) -> None:
        if not hasattr(self, "results_stack"):
            return
        mode = self._active_mode()
        index = self._results_page_indices.get(mode or "", self._results_empty_index)
        self.results_stack.setCurrentIndex(index)

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
        self.show_sidebar_action.toggled.connect(self._set_sidebar_visible)
        self.sidebar_toggle_btn.toggled.connect(self.show_sidebar_action.setChecked)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        for prefix in ("pl", "drr", "cmp"):
            combo: QComboBox = getattr(self, f"{prefix}_yaxis_combo")
            combo.currentTextChanged.connect(lambda _text, p=prefix: self._update_y_axis_controls(p))
            for key in ("a", "b", "c"):
                spin: QDoubleSpinBox = getattr(self, f"{prefix}_yaxis_{key}_spin")
                spin.valueChanged.connect(lambda _value, p=prefix: self._update_y_axis_controls(p))
        for prefix in ("pl", "drr", "drr_second", "cmp", "power"):
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
        self.pl_auto_v_btn.clicked.connect(self.pl_controller._auto_pl_vrange)
        self.pl_auto_x_btn.clicked.connect(self.pl_controller._auto_pl_xrange)
        self.pl_auto_y_btn.clicked.connect(self.pl_controller._auto_pl_yrange)
        self.pl_yaxis_combo.currentTextChanged.connect(self.pl_controller._on_pl_plot_param_changed)
        self.pl_yaxis_combo.currentTextChanged.connect(self.pl_controller._on_pl_dat_y_axis_changed)
        self.pl_dat_yaxis_label_edit.textChanged.connect(lambda _text: self.pl_controller._on_pl_plot_param_changed())
        self.pl_dat_yaxis_unit_edit.textChanged.connect(lambda _text: self.pl_controller._on_pl_plot_param_changed())
        self.pl_yaxis_a_spin.valueChanged.connect(self.pl_controller._on_pl_plot_param_changed)
        self.pl_yaxis_b_spin.valueChanged.connect(self.pl_controller._on_pl_plot_param_changed)
        self.pl_yaxis_c_spin.valueChanged.connect(self.pl_controller._on_pl_plot_param_changed)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax"):
            self.pl_spins[key].valueChanged.connect(
                lambda _value, widget=self.pl_spins[key]:
                self.pl_controller._on_pl_plot_param_changed(widget)
            )
        self.pl_spins["gate"].valueChanged.connect(self.pl_controller._on_pl_gate_changed)
        self.pl_cmap.currentTextChanged.connect(self.pl_controller._on_pl_plot_param_changed)
        self.pl_log_chk.toggled.connect(
            lambda _checked: self.pl_controller._on_pl_plot_param_changed(self.pl_log_chk)
        )
        self.pl_clip_chk.toggled.connect(self.pl_controller._on_pl_plot_param_changed)
        self.cmp_in_k_angle_spin.valueChanged.connect(self.compare_controller._on_cmp_angle_reference_changed)
        self.cmp_rotation_mapping_combo.currentIndexChanged.connect(self.compare_controller._on_cmp_rotation_mapping_changed)
        self.cmp_in_kp_angle_spin.valueChanged.connect(self.compare_controller._on_cmp_angle_reference_changed)
        self.cmp_out_k_angle_spin.valueChanged.connect(self.compare_controller._on_cmp_angle_reference_changed)
        self.cmp_out_kp_angle_spin.valueChanged.connect(self.compare_controller._on_cmp_angle_reference_changed)
        self.cmp_angle_tolerance_spin.valueChanged.connect(self.compare_controller._on_cmp_angle_reference_changed)
        self.cmp_infer_angles_btn.clicked.connect(self.compare_controller._on_cmp_infer_angles_requested)
        self.cmp_auto_assign_btn.clicked.connect(self.compare_controller._on_cmp_auto_assign_requested)
        self.cmp_source_filter_combo.currentTextChanged.connect(self.compare_controller._on_cmp_source_filter_changed)
        self.cmp_view_intensity_btn.clicked.connect(lambda: self.compare_controller._on_cmp_plot_view_button_clicked("Intensity Compare"))
        self.cmp_view_vp_btn.clicked.connect(lambda: self.compare_controller._on_cmp_plot_view_button_clicked("Valley Polarization"))
        self.cmp_vp_background_spin.valueChanged.connect(lambda _value: self.compare_controller._on_cmp_plot_param_changed(self.cmp_vp_background_spin))
        self.cmp_vp_auto_background_chk.toggled.connect(self.compare_controller._on_cmp_background_mode_changed)
        for key, spin in self.cmp_vp_spins.items():
            spin.valueChanged.connect(
                lambda _value, widget=spin: self.compare_controller._on_cmp_plot_param_changed(widget)
            )
        for check in self.cmp_vp_fix_checks.values():
            check.toggled.connect(lambda _checked: self._update_action_states())
        self.cmp_vp_auto_v_btn.clicked.connect(self.compare_controller._auto_cmp_vp_range)
        self.cmp_display_preset_combo.currentTextChanged.connect(self.compare_controller._on_cmp_display_preset_changed)
        for combo in self.cmp_channel_combos.values():
            combo.currentTextChanged.connect(lambda _text, widget=combo: self.compare_controller._on_cmp_plot_param_changed(widget))
        for chk in self.cmp_show_checks.values():
            chk.toggled.connect(lambda _checked, widget=chk: self.compare_controller._on_cmp_plot_param_changed(widget))
        self.cmp_yaxis_combo.currentTextChanged.connect(lambda _text: self.compare_controller._on_cmp_plot_param_changed(self.cmp_yaxis_combo))
        self.cmp_yaxis_a_spin.valueChanged.connect(lambda _value: self.compare_controller._on_cmp_plot_param_changed(self.cmp_yaxis_a_spin))
        self.cmp_yaxis_b_spin.valueChanged.connect(lambda _value: self.compare_controller._on_cmp_plot_param_changed(self.cmp_yaxis_b_spin))
        self.cmp_yaxis_c_spin.valueChanged.connect(lambda _value: self.compare_controller._on_cmp_plot_param_changed(self.cmp_yaxis_c_spin))
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax", "gate"):
            self.cmp_spins[key].valueChanged.connect(lambda _value, widget=self.cmp_spins[key]: self.compare_controller._on_cmp_plot_param_changed(widget))
        self.cmp_cmap.currentTextChanged.connect(lambda _text: self.compare_controller._on_cmp_plot_param_changed(self.cmp_cmap))
        self.cmp_log_chk.toggled.connect(lambda _checked: self.compare_controller._on_cmp_plot_param_changed(self.cmp_log_chk))
        self.cmp_clip_chk.toggled.connect(lambda _checked: self.compare_controller._on_cmp_plot_param_changed(self.cmp_clip_chk))
        self.cmp_auto_v_btn.clicked.connect(self.compare_controller._auto_cmp_vrange)
        self.cmp_auto_x_btn.clicked.connect(self.compare_controller._auto_cmp_xrange)
        self.cmp_auto_y_btn.clicked.connect(self.compare_controller._auto_cmp_yrange)
        self.power_refresh_groups_btn.clicked.connect(lambda: self._refresh_file_lists(mode="Power Dependent"))
        self.power_group_combo.currentIndexChanged.connect(self.power_controller._on_power_plot_param_changed)
        self.power_kk_group_combo.currentIndexChanged.connect(self.power_controller._on_power_source_assignment_changed)
        self.power_kkp_group_combo.currentIndexChanged.connect(self.power_controller._on_power_source_assignment_changed)
        self.power_compare_chk.toggled.connect(self.power_controller._on_power_comparison_changed)
        self.power_view_intensity_btn.clicked.connect(lambda: self.power_controller._on_power_plot_view_button_clicked("Intensity"))
        self.power_view_vp_btn.clicked.connect(lambda: self.power_controller._on_power_plot_view_button_clicked("VP"))
        self.power_axis_scale_combo.currentTextChanged.connect(self.power_controller._on_power_axis_scale_changed)
        self.power_pair_mode_combo.currentTextChanged.connect(lambda _text: self.power_controller._on_power_plot_param_changed(self.power_pair_mode_combo))
        self.power_background_spin.valueChanged.connect(lambda _value: self.power_controller._on_power_plot_param_changed(self.power_background_spin))
        self.power_background_auto_chk.toggled.connect(self.power_controller._on_power_background_mode_changed)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax", "gate"):
            self.power_spins[key].valueChanged.connect(lambda _value, widget=self.power_spins[key]: self.power_controller._on_power_plot_param_changed(widget))
        # Commit numeric edits on Return/focus loss.  Intermediate keystrokes
        # are otherwise interpreted as separate expensive redraw requests.
        for spin_map in (self.pl_spins, self.cmp_spins, self.power_spins):
            for spin in spin_map.values():
                if hasattr(spin, "setKeyboardTracking"):
                    spin.setKeyboardTracking(False)
        self.power_cmap.currentTextChanged.connect(lambda _text: self.power_controller._on_power_plot_param_changed(self.power_cmap))
        self.power_log_chk.toggled.connect(lambda _checked: self.power_controller._on_power_plot_param_changed(self.power_log_chk))
        self.power_clip_chk.toggled.connect(lambda _checked: self.power_controller._on_power_plot_param_changed(self.power_clip_chk))
        self.power_auto_v_btn.clicked.connect(self.power_controller._auto_power_vrange)
        self.power_auto_x_btn.clicked.connect(self.power_controller._auto_power_xrange)
        self.power_auto_y_btn.clicked.connect(self.power_controller._auto_power_yrange)
        self.shg_files.itemSelectionChanged.connect(self.shg_controller._on_shg_source_changed)
        self.shg_source_filter_combo.currentIndexChanged.connect(self.shg_controller._on_shg_source_filter_changed)
        self.shg_refresh_sources_btn.clicked.connect(lambda: self._refresh_file_lists(auto=False))
        self.shg_background_combo.currentIndexChanged.connect(lambda _idx: self.shg_controller._on_shg_source_changed())
        self.shg_workflow_tabs.currentChanged.connect(lambda _idx: self.shg_controller._on_shg_workflow_changed())
        for combo in (
            self.shg_compare_reference_combo,
            self.shg_compare_sample_combo,
            self.shg_compare_background_a_combo,
            self.shg_compare_background_b_combo,
        ):
            combo.currentIndexChanged.connect(lambda _idx: self.shg_controller._on_shg_source_changed())
        self.shg_background_method_combo.currentTextChanged.connect(lambda _text: self.shg_controller._on_shg_background_method_changed())
        self.shg_cosmic_enable_chk.toggled.connect(lambda _checked: self.shg_controller._on_shg_cosmic_param_changed())
        self.shg_spectrum_view_combo.currentTextChanged.connect(lambda _text: self.shg_controller._on_shg_spectrum_view_changed())
        self.shg_compare_display_combo.currentTextChanged.connect(lambda _text: self.shg_controller._on_shg_spectrum_view_changed())
        self.shg_angle_wrap_combo.currentTextChanged.connect(lambda _text: self.shg_controller._on_shg_param_changed())
        self.shg_include_failed_chk.toggled.connect(lambda _checked: self.shg_controller._on_shg_param_changed())
        self.shg_angle_cursor_spin.valueChanged.connect(lambda _value: self.shg_controller._on_shg_angle_cursor_changed())
        for spin in (
            self.shg_peak_center_spin,
            self.shg_gate_half_range_spin,
            self.shg_sideband_gap_spin,
            self.shg_sideband_width_spin,
            self.shg_sigma_clip_spin,
            self.shg_angle_scale_spin,
            self.shg_angle_offset_spin,
        ):
            spin.editingFinished.connect(self.shg_controller._on_shg_param_changed)
        for spin in (
            self.shg_cosmic_threshold_spin,
            self.shg_cosmic_window_spin,
            self.shg_cosmic_max_width_spin,
        ):
            spin.editingFinished.connect(self.shg_controller._on_shg_cosmic_param_changed)
        self.shg_fit_enable_chk.toggled.connect(lambda _checked: self.shg_controller._on_shg_fit_param_changed())
        self.shg_fit_weighted_chk.toggled.connect(lambda _checked: self.shg_controller._on_shg_fit_param_changed())
        self.shg_fit_include_excluded_chk.toggled.connect(lambda _checked: self.shg_controller._on_shg_fit_param_changed())
        self.shg_fit_min_spin.editingFinished.connect(self.shg_controller._on_shg_fit_param_changed)
        self.shg_fit_max_spin.editingFinished.connect(self.shg_controller._on_shg_fit_param_changed)
        self.shg_fit_branch_spin.valueChanged.connect(lambda _value: self.shg_controller._on_shg_fit_param_changed())
        self.drr_yaxis_combo.currentTextChanged.connect(
            lambda _value: self.drr_controller._on_drr_plot_param_changed(self.drr_yaxis_combo)
        )
        self.drr_yaxis_a_spin.valueChanged.connect(
            lambda _value: self.drr_controller._on_drr_plot_param_changed(self.drr_yaxis_a_spin)
        )
        self.drr_yaxis_b_spin.valueChanged.connect(
            lambda _value: self.drr_controller._on_drr_plot_param_changed(self.drr_yaxis_b_spin)
        )
        self.drr_yaxis_c_spin.valueChanged.connect(
            lambda _value: self.drr_controller._on_drr_plot_param_changed(self.drr_yaxis_c_spin)
        )
        self.drr_baseline_combo.currentTextChanged.connect(
            lambda _value: self.drr_controller._on_drr_plot_param_changed(self.drr_baseline_combo)
        )
        self.drr_view_raw_btn.clicked.connect(lambda: self._on_drr_plot_view_changed("raw"))
        self.drr_view_second_btn.clicked.connect(lambda: self._on_drr_plot_view_changed("second"))
        self.drr_view_side_btn.toggled.connect(lambda checked: self._on_drr_side_by_side_changed(bool(checked)))
        self.drr_gate_toolbar_spin.valueChanged.connect(self._on_drr_toolbar_gate_changed)
        self.drr_gate_prev_btn.clicked.connect(lambda: self._step_drr_gate(-1))
        self.drr_gate_next_btn.clicked.connect(lambda: self._step_drr_gate(1))
        self.drr_derivative_combo.currentTextChanged.connect(self.drr_controller._on_drr_derivative_changed)
        self.drr_sg_window_spin.valueChanged.connect(self.drr_controller._on_drr_derivative_changed)
        self.drr_sg_poly_spin.valueChanged.connect(self.drr_controller._on_drr_derivative_changed)
        self.drr_edit_measurements_btn.clicked.connect(self.drr_controller._edit_drr_measurements)
        self.drr_clear_measurements_btn.clicked.connect(self.drr_controller._clear_drr_measurements)
        self.drr_edit_baselines_btn.clicked.connect(self.drr_controller._edit_drr_baselines_dialog)
        self.drr_baseline_autofind_btn.clicked.connect(self.drr_controller._clear_drr_baselines)
        self.drr_pin_baseline_chk.toggled.connect(self.drr_controller._on_drr_pin_baseline_toggled)
        self.drr_baseline_combine_combo.currentTextChanged.connect(self.drr_controller._on_drr_baseline_mode_changed)
        self.drr_auto_v_btn.clicked.connect(self.drr_controller._auto_drr_vrange)
        self.drr_second_auto_v_btn.clicked.connect(self._auto_drr_second_vrange)
        self.drr_auto_x_btn.clicked.connect(self.drr_controller._auto_drr_xrange)
        self.drr_auto_y_btn.clicked.connect(self.drr_controller._auto_drr_yrange)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax", "gate"):
            spin = self.drr_spins[key]
            if key == "gate":
                spin.valueChanged.connect(
                    lambda _value, widget=spin: self._on_drr_sidebar_gate_changed(widget)
                )
            else:
                spin.valueChanged.connect(
                    lambda _value, widget=spin: self.drr_controller._on_drr_plot_param_changed(widget)
                )
        self.drr_cmap.currentTextChanged.connect(
            lambda _value: self.drr_controller._on_drr_plot_param_changed(self.drr_cmap)
        )
        self.drr_log_chk.toggled.connect(
            lambda _value: self.drr_controller._on_drr_plot_param_changed(self.drr_log_chk)
        )
        self.drr_clip_chk.toggled.connect(
            lambda _value: self.drr_controller._on_drr_plot_param_changed(self.drr_clip_chk)
        )
        self.drr_center_zero_chk.toggled.connect(
            lambda _value: self.drr_controller._on_drr_plot_param_changed(self.drr_center_zero_chk)
        )
        self.drr_second_vmin_spin.valueChanged.connect(self._on_drr_second_scale_changed)
        self.drr_second_vmax_spin.valueChanged.connect(self._on_drr_second_scale_changed)
        self.drr_second_cmap.currentTextChanged.connect(lambda _text: self._on_drr_plot_view_changed(self._drr_plot_view, redraw=True))
        self.drr_peak_find_btn.clicked.connect(self.drr_controller._on_drr_find_peaks)
        self.drr_peak_show_chk.toggled.connect(self.drr_controller._on_drr_analysis_view_changed)
        self.drr_peak_mode_combo.currentTextChanged.connect(self.drr_controller._on_drr_analysis_view_changed)
        self.drr_fit_btn.clicked.connect(self.drr_controller._on_drr_fit_lorentz)
        self.drr_fit_clear_btn.clicked.connect(self.drr_controller._on_drr_clear_fit)
        self.drr_fit_show_chk.toggled.connect(self.drr_controller._on_drr_analysis_view_changed)
        self.mcd_files.itemSelectionChanged.connect(self.mcd_controller._on_mcd_source_changed)
        self.mcd_select_source_btn.clicked.connect(self.mcd_controller._edit_mcd_source)
        self.mcd_clear_source_btn.clicked.connect(self.mcd_controller._clear_mcd_source)
        self.mcd_extract_btn.clicked.connect(self._open_mcd_extract_dialog)
        self.mcd_auto_angles_chk.toggled.connect(self.mcd_controller._on_mcd_angle_assignment_changed)
        self.mcd_sigma_plus_combo.currentIndexChanged.connect(self.mcd_controller._on_mcd_params_changed)
        self.mcd_sigma_minus_combo.currentIndexChanged.connect(self.mcd_controller._on_mcd_params_changed)
        self.mcd_reference_mode_combo.currentTextChanged.connect(self.mcd_controller._on_mcd_reference_mode_changed)
        self.mcd_zero_spin.valueChanged.connect(self.mcd_controller._on_mcd_params_changed)
        self.mcd_gap_spin.valueChanged.connect(self.mcd_controller._on_mcd_params_changed)
        self.mcd_delta_b_spin.valueChanged.connect(self.mcd_controller._on_mcd_params_changed)
        self.mcd_pair_alignment_combo.currentTextChanged.connect(self.mcd_controller._on_mcd_params_changed)
        self.mcd_bin_spin.valueChanged.connect(self.mcd_controller._on_mcd_params_changed)
        self.mcd_gain_combo.currentTextChanged.connect(self.mcd_controller._on_mcd_params_changed)
        self.mcd_correction_mode_combo.currentTextChanged.connect(self.mcd_controller._on_mcd_correction_mode_changed)
        self.mcd_spectral_order_combo.currentTextChanged.connect(self.mcd_controller._on_mcd_params_changed)
        self.mcd_background_ranges_edit.editingFinished.connect(self.mcd_controller._on_mcd_background_ranges_changed)
        self.mcd_suggest_background_btn.clicked.connect(self.mcd_controller._suggest_mcd_background_ranges)
        self.mcd_apply_correction_btn.clicked.connect(self.mcd_controller._apply_mcd_now)
        self.mcd_dark_pos_combo.currentIndexChanged.connect(self.mcd_controller._on_mcd_params_changed)
        self.mcd_dark_neg_combo.currentIndexChanged.connect(self.mcd_controller._on_mcd_params_changed)
        self.mcd_map_combo.currentTextChanged.connect(lambda _text: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_map_combo))
        self.mcd_auto_v_btn.clicked.connect(self.mcd_controller._auto_mcd_vrange)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax"):
            self.mcd_spins[key].valueChanged.connect(lambda _value, widget=self.mcd_spins[key]: self.mcd_controller._on_mcd_plot_changed(source=widget))
        self.mcd_cmap.currentTextChanged.connect(lambda _text: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_cmap))
        self.mcd_center_zero_chk.toggled.connect(lambda _checked: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_center_zero_chk))
        self.mcd_window_center_spin.valueChanged.connect(lambda _value: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_window_center_spin))
        self.mcd_window_width_spin.valueChanged.connect(lambda _value: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_window_width_spin))
        self.mcd_pair_b_combo.currentIndexChanged.connect(self.mcd_controller._on_mcd_pair_selection_changed)
        self.mcd_window_metric_combo.currentTextChanged.connect(lambda _text: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_window_metric_combo))
        self.mcd_show_raw_chk.toggled.connect(lambda _checked: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_show_raw_chk))
        self.mcd_show_signed_mean_chk.toggled.connect(lambda _checked: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_show_signed_mean_chk))
        self.mcd_show_absolute_mean_chk.toggled.connect(lambda _checked: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_show_absolute_mean_chk))
        self.mcd_show_unsigned_absolute_mean_chk.toggled.connect(lambda _checked: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_show_unsigned_absolute_mean_chk))
        self.mcd_show_integral_chk.toggled.connect(lambda _checked: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_show_integral_chk))
        self.mcd_fit_zero_chk.toggled.connect(lambda _checked: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_fit_zero_chk))
        self.mcd_fit_b_window_spin.valueChanged.connect(lambda _value: self.mcd_controller._on_mcd_plot_changed(source=self.mcd_fit_b_window_spin))
        self.mcd_find_centers_btn.clicked.connect(self.mcd_controller._find_mcd_center_candidates)
        self.mcd_previous_candidate_btn.clicked.connect(lambda: self.mcd_controller._step_mcd_center_candidate(-1))
        self.mcd_next_candidate_btn.clicked.connect(lambda: self.mcd_controller._step_mcd_center_candidate(1))
        self.mcd_clear_candidates_btn.clicked.connect(self.mcd_controller._return_to_manual_mcd_center)
        unified_controls = getattr(self, "mcd_unified_controls", None)
        if unified_controls is not None:
            self.mcd_retention.set_current_provider("window", self._current_unified_window_snapshot)
            self.mcd_retention.set_current_provider("feature", self._current_unified_feature_snapshot)
            self.mcd_retention.bind(
                window_list=unified_controls.retained_window_list,
                feature_list=unified_controls.retained_feature_list,
            )
            self.mcd_retention.inspect_requested.connect(self._inspect_retained_snapshot)
            self.mcd_retention.message_requested.connect(self._status)
            self.mcd_retention.changed.connect(self._update_unified_mcd_save_controls)
            unified_controls.retain_window_btn.clicked.connect(self._retain_unified_mcd_window)
            unified_controls.update_retained_btn.clicked.connect(self._update_unified_mcd_window)
            unified_controls.retain_feature_btn.clicked.connect(self._retain_unified_mcd_feature)
            unified_controls.update_feature_btn.clicked.connect(self._update_unified_mcd_feature)
            unified_controls.save_results_btn.clicked.connect(lambda _checked=False: self._queue_unified_mcd_export(scope="retained"))
            unified_controls.change_save_folder_btn.clicked.connect(self._change_unified_mcd_export_folder)
            unified_controls.feature_method_combo.currentTextChanged.connect(self._queue_unified_selected_feature_analysis)
            unified_controls.feature_source_combo.currentTextChanged.connect(self._queue_unified_selected_feature_analysis)
            unified_controls.feature_search_low_spin.valueChanged.connect(self._queue_unified_selected_feature_analysis)
            unified_controls.feature_search_high_spin.valueChanged.connect(self._queue_unified_selected_feature_analysis)
            unified_controls.reference_mode_combo.currentTextChanged.connect(self._queue_unified_selected_feature_analysis)
            unified_controls.manual_reference_spin.valueChanged.connect(self._queue_unified_selected_feature_analysis)
            unified_controls.manual_link_btn.clicked.connect(self._add_unified_manual_link)
            unified_controls.prominence_spin.valueChanged.connect(self._queue_unified_selected_feature_analysis)
            unified_controls.slope_details_btn.clicked.connect(self._show_unified_slope_details)
            for name in (
                "slope_low_spin", "slope_low_end_spin", "slope_high_positive_spin",
                "slope_high_positive_end_spin", "slope_high_negative_spin",
                "slope_high_negative_end_spin",
            ):
                getattr(unified_controls, name).valueChanged.connect(self._on_unified_feature_display_changed)
        self.pl_peak_find_btn.clicked.connect(self.pl_controller._on_pl_find_peaks)
        self.pl_peak_show_chk.toggled.connect(self.pl_controller._on_pl_analysis_view_changed)
        self.pl_peak_mode_combo.currentTextChanged.connect(self.pl_controller._on_pl_analysis_view_changed)
        self.pl_fit_btn.clicked.connect(self.pl_controller._on_pl_fit_lorentz)
        self.pl_fit_clear_btn.clicked.connect(self.pl_controller._on_pl_clear_fit)
        self.pl_fit_show_chk.toggled.connect(self.pl_controller._on_pl_analysis_view_changed)

        self.show_log_btn.clicked.connect(self._toggle_log)
        self.clear_log_btn.clicked.connect(self._clear_log)
        self._gate_motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self._gate_click_cid = self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self._mcd_window_release_cid = self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._mcd_blit_draw_cid = self.canvas.mpl_connect("draw_event", self.mcd_controller._on_canvas_draw)
        self._cmp_blit_draw_cid = self.canvas.mpl_connect("draw_event", self.compare_controller._on_compare_canvas_draw)
        for prefix in ("pl", "drr", "cmp"):
            self._update_y_axis_controls(prefix)
        self.compare_controller._cmp_apply_display_preset()
        self.compare_controller._cmp_update_view_mode()
        self.compare_controller._cmp_set_channel_combo_items()
        self.compare_controller._cmp_refresh_history_cache(force=True)
        self.compare_controller._cmp_update_assignment_summary()
        self.power_controller._power_refresh_groups()
        self.power_controller._power_update_view_mode()
        self.mcd_controller._mcd_refresh_sources()
        self.shg_controller._shg_refresh_sources()
        if hasattr(self, "power_background_spin"):
            self.power_background_spin.setEnabled(not self.power_controller._power_background_auto_enabled())
        self._update_plot_view_bar_visibility()

    def _toggle_log(self) -> None:
        self.log_dock.setVisible(not self.log_dock.isVisible())

    def _clear_log(self) -> None:
        self.log_lines.clear()
        self.log_text.clear()

    def _unified_mcd_settings_snapshot(self) -> dict[str, Any]:
        return {
            "window_metric": str(self.mcd_window_metric_combo.currentText()),
            "feature_method": str(self.mcd_unified_controls.feature_method_combo.currentText()),
            "feature_source": str(self.mcd_unified_controls.feature_source_combo.currentText()),
            "feature_search_low_ev": float(self.mcd_unified_controls.feature_search_low_spin.value()),
            "feature_search_high_ev": float(self.mcd_unified_controls.feature_search_high_spin.value()),
            "reference_mode": str(self.mcd_unified_controls.reference_mode_combo.currentText()),
            "manual_reference_ev": float(self.mcd_unified_controls.manual_reference_spin.value()),
            "prominence_fraction": float(self.mcd_unified_controls.prominence_spin.value()),
            "channel_mapping": str(self.mcd_unified_view.channel_mapping_combo.currentText()) if hasattr(self, "mcd_unified_view") else "",
            "feature_metric": self.mcd_unified_view.display_metric_label if hasattr(self, "mcd_unified_view") else "Shift (meV)",
            "feature_overlay": bool(self.mcd_unified_view.state.feature_overlay) if hasattr(self, "mcd_unified_view") else False,
            "slope_low_t": float(self.mcd_unified_controls.slope_low_spin.value()),
            "slope_low_end_t": float(self.mcd_unified_controls.slope_low_end_spin.value()),
            "slope_high_positive_t": float(self.mcd_unified_controls.slope_high_positive_spin.value()),
            "slope_high_positive_end_t": float(self.mcd_unified_controls.slope_high_positive_end_spin.value()),
            "slope_high_negative_t": float(self.mcd_unified_controls.slope_high_negative_spin.value()),
            "slope_high_negative_end_t": float(self.mcd_unified_controls.slope_high_negative_end_spin.value()),
        }

    def _current_unified_window_snapshot(self) -> dict[str, Any] | None:
        """Build a complete numerical snapshot for the retention store."""
        if self.loaded is None or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            return None
        result = self.loaded.mcd_result
        energy = np.asarray(result.energy_ev, dtype=float).ravel()
        values = np.asarray(getattr(result, "pair_mcd_corrected", ()), dtype=float)
        try:
            order = np.asarray(spectrum_energy_order(result), dtype=int)
        except (AttributeError, TypeError, ValueError):
            order = np.argsort(energy, kind="stable")
        if values.ndim == 2 and order.size == values.shape[1]:
            values = values[:, order]
        half = float(self.mcd_unified_view.state.window_width_mev) * 0.0005
        mask = (energy >= self.mcd_unified_view.state.window_center_ev - half) & (energy <= self.mcd_unified_view.state.window_center_ev + half)
        selected = values[:, mask] if values.ndim == 2 and np.any(mask) else np.empty((values.shape[0], 0))
        metric_label = str(self.mcd_window_metric_combo.currentText())
        metric = self.mcd_controller._mcd_window_metric()
        # Keep the retained trace identical to the branch-aware renderer and
        # slope fitter.  Rebuilding one joined nanmean here loses the signed
        # metric and can merge increasing/decreasing rows.
        try:
            branch_traces = pair_window_trace_by_branch(
                result,
                float(self.mcd_unified_view.state.window_center_ev),
                float(self.mcd_unified_view.state.window_width_mev),
                metrics=(metric,), include_raw=False,
            )
            labels = np.asarray(getattr(result, "pair_labels", ()), dtype=str).ravel()
            trace = np.full(len(getattr(result, "pair_b", ())), np.nan, dtype=float)
            values_key = f"corrected_{metric}"
            for branch, data in branch_traces.items():
                branch_mask = labels == branch
                branch_values = data.get(values_key, ((), ()))
                values = np.asarray(branch_values[1], dtype=float)
                if values.size == int(np.count_nonzero(branch_mask)):
                    trace[branch_mask] = values
        except (AttributeError, TypeError, ValueError, IndexError):
            if selected.shape[1] == 0:
                trace = np.full(len(getattr(result, "pair_b", ())), np.nan)
            elif metric == "integral":
                trace = np.trapezoid(selected, x=energy[mask], axis=1)
            elif metric in {"absolute_mean", "field_signed_absolute_mean"}:
                trace = np.nanmean(np.abs(selected), axis=1)
                if metric == "field_signed_absolute_mean":
                    trace *= np.sign(np.asarray(getattr(result, "pair_b", ()), dtype=float))
            else:
                trace = np.nanmean(selected, axis=1)
        if metric == "integral" and selected.shape[1] and not np.isfinite(trace).any():
            # Duplicate/degenerate energy columns have zero area, not an
            # unknown metric. Preserve the integral semantic for all branches.
            trace = np.trapezoid(np.nan_to_num(selected, nan=0.0), x=energy[mask], axis=1)
        return {
            "id": f"window-{self.mcd_unified_view.state.source_generation}-{self.mcd_unified_view.state.window_center_ev:.9g}",
            "identity": {"source": str(getattr(result, "source_file", "")), "kind": "mcd-window"},
            "source_generation": int(self.mcd_unified_view.state.source_generation),
            "original_b": np.asarray(getattr(result, "pair_b", ()), dtype=float).copy(),
            "branches": np.asarray(getattr(result, "pair_labels", ()), dtype=str).copy(),
            "trace_values": np.asarray(trace, dtype=float).copy(),
            "metric": metric_label,
            "center_ev": float(self.mcd_unified_view.state.window_center_ev),
            "width_mev": float(self.mcd_unified_view.state.window_width_mev),
            "slopes": self._mcd_unified_slopes.to_dict() if getattr(self, "_mcd_unified_slopes", None) is not None else {},
            "settings": self._unified_mcd_settings_snapshot(),
            "status": "complete",
        }

    def _current_unified_feature_snapshot(self) -> dict[str, Any] | None:
        view = getattr(self, "mcd_unified_view", None)
        result = self.loaded.mcd_result if self.loaded is not None and self.loaded.mode == "MCD" else None
        selected = view.selected_candidate if view is not None else None
        payload = getattr(self, "_mcd_unified_track_payload", None)
        if view is None or result is None or selected is None or not isinstance(payload, Mapping):
            return None
        return {
            "id": str(selected.get("id", "feature")),
            "identity": {"source": str(getattr(result, "source_file", "")), "label": selected.get("label", selected.get("id", "feature"))},
            "source_generation": int(view.state.source_generation),
            "track_payload": dict(payload),
            "original_b": np.asarray(getattr(result, "pair_b", ()), dtype=float).copy(),
            "branches": np.asarray(getattr(result, "pair_labels", ()), dtype=str).copy(),
            "trace_values": np.asarray([point.get("energy_ev") for track in payload.get("tracks", ()) for point in track.get("points", ()) if isinstance(point, Mapping) and point.get("energy_ev") is not None], dtype=float),
            "analysis_results": payload.get("analysis_results", {}),
            "links": payload.get("links", ()),
            "splitting": payload.get("splitting", ()),
            "mapping": payload.get("mapping", {}),
            "settings": self._unified_mcd_settings_snapshot(),
            "status": "complete" if payload.get("status") == "ok" else str(payload.get("status", "uncomputed")),
        }

    def _refresh_unified_retained_list(self) -> None:
        controls = getattr(self, "mcd_unified_controls", None)
        view = getattr(self, "mcd_unified_view", None)
        if controls is None or view is None:
            return
        self.mcd_retention.refresh()
        self._update_unified_mcd_save_controls()

    def _update_unified_mcd_save_controls(self) -> None:
        controls = getattr(self, "mcd_unified_controls", None)
        if controls is None:
            return
        busy = self._mcd_unified_export_worker is not None
        has_selection = any(item.included for kind in ("window", "feature") for item in self.mcd_retention.store.items(kind))
        loaded = self.loaded is not None and self.loaded.mode == "MCD" and self.loaded.mcd_result is not None
        controls.save_results_btn.setEnabled(bool(loaded and has_selection and not busy))
        controls.change_save_folder_btn.setEnabled(not busy)

    def _retain_unified_mcd_window(self) -> None:
        view = getattr(self, "mcd_unified_view", None)
        if view is None:
            return
        snapshot = self._current_unified_window_snapshot()
        if snapshot is None:
            return
        self.mcd_retention.add_window(snapshot, included=True)
        # Keep the legacy state as a read-only compatibility mirror for
        # callers that still inspect UnifiedMcdState directly.
        view.retain_current_window(settings=self._unified_mcd_settings_snapshot())
        self._refresh_unified_retained_list()

    def _retain_unified_mcd_feature(self) -> None:
        view = getattr(self, "mcd_unified_view", None)
        controls = getattr(self, "mcd_unified_controls", None)
        selected = view.selected_candidate if view is not None else None
        if view is None or controls is None or selected is None:
            return
        snapshot = {
            "id": str(selected.get("id", "feature")),
            "identity": {"source": str(getattr(self.loaded.mcd_result, "source_file", "")), "label": selected.get("label", selected.get("id", "feature"))},
            "source_generation": int(view.state.source_generation),
            "track_payload": dict(getattr(self, "_mcd_unified_track_payload", {}) or {}),
            "original_b": np.asarray(getattr(self.loaded.mcd_result, "pair_b", ()), dtype=float).copy(),
            "branches": np.asarray(getattr(self.loaded.mcd_result, "pair_labels", ()), dtype=str).copy(),
            "trace_values": np.asarray([], dtype=float),
            "analysis_results": (getattr(self, "_mcd_unified_track_payload", {}) or {}).get("analysis_results", {}),
            "links": (getattr(self, "_mcd_unified_track_payload", {}) or {}).get("links", ()),
            "splitting": (getattr(self, "_mcd_unified_track_payload", {}) or {}).get("splitting", ()),
            "mapping": (getattr(self, "_mcd_unified_track_payload", {}) or {}).get("mapping", {}),
            "settings": self._unified_mcd_settings_snapshot(),
            "status": "complete" if (getattr(self, "_mcd_unified_track_payload", {}) or {}).get("status") == "ok" else "uncomputed",
        }
        self.mcd_retention.add_feature(snapshot, included=True)
        view.retain_selected_feature(
            selected, settings=self._unified_mcd_settings_snapshot(),
            analysis_payload=getattr(self, "_mcd_unified_track_payload", None),
        )
        self.mcd_retention.refresh()

    def _update_unified_mcd_feature(self) -> None:
        view = getattr(self, "mcd_unified_view", None)
        controls = getattr(self, "mcd_unified_controls", None)
        selected = view.selected_candidate if view is not None else None
        row = controls.retained_feature_list.currentRow() if controls is not None else -1
        if view is None or controls is None or selected is None or row < 0 or row >= len(view.state.retained_features):
            return
        identifier = str(controls.retained_feature_list.currentItem().data(Qt.ItemDataRole.UserRole) or selected.get("id", ""))
        snapshot = self._current_unified_feature_snapshot()
        if snapshot is None:
            return
        try:
            self.mcd_retention.store.update_feature(identifier, snapshot, source_generation=view.state.source_generation)
        except RetainedResultError as exc:
            self._status(str(exc))
            return
        updated = dict(selected)
        updated["source_generation"] = view.state.source_generation
        updated["settings"] = self._unified_mcd_settings_snapshot()
        updated["analysis_payload"] = dict(self._mcd_unified_track_payload or {})
        view.state.retained_features[row] = updated
        self._retain_unified_mcd_feature_refresh()

    def _retain_unified_mcd_feature_refresh(self) -> None:
        view = getattr(self, "mcd_unified_view", None)
        controls = getattr(self, "mcd_unified_controls", None)
        if view is None or controls is None:
            return
        self.mcd_retention.refresh()

    def _inspect_unified_mcd_feature(self, row: int) -> None:
        """Restore one retained feature's measured payload for inspection."""
        view = getattr(self, "mcd_unified_view", None)
        controls = getattr(self, "mcd_unified_controls", None)
        if view is None or controls is None or row < 0 or row >= len(view.state.retained_features):
            return
        item = view.state.retained_features[row]
        generation = int(item.get("source_generation", view.state.source_generation))
        if generation not in {0, int(view.state.source_generation)}:
            self._status("Retained feature belongs to an older MCD source; update it before inspection.")
            return
        selected_id = str(item.get("id", ""))
        for index in range(view.candidate_combo.count()):
            if str(view.candidate_combo.itemData(index, Qt.UserRole) or "") == selected_id:
                view.candidate_combo.setCurrentIndex(index)
                break
        payload = item.get("analysis_payload")
        if isinstance(payload, Mapping):
            self._mcd_unified_track_payload_key = None
            self._mcd_unified_track_payload = dict(payload)
            self._plot_mcd_unified(auto=True)
            self._status(f"Inspecting retained feature {item.get('label', selected_id)}.")

    def _inspect_retained_snapshot(self, kind: str, snapshot: Any) -> None:
        """Display a typed retained snapshot without mutating its arrays."""
        if self.loaded is None or self.loaded.mode != "MCD":
            return
        current_generation = int(getattr(self.mcd_unified_view.state, "source_generation", 0))
        if int(getattr(snapshot, "source_generation", -1)) not in {0, current_generation}:
            self._status("Retained result belongs to an older MCD source generation.")
            return
        if kind == "window":
            center = getattr(snapshot, "center_ev", None)
            width = getattr(snapshot, "width_mev", None)
            if center is not None and width is not None:
                self.mcd_unified_view.set_window(float(center), float(width))
                self.mcd_window_center_spin.setValue(float(center))
                self.mcd_window_width_spin.setValue(float(width))
            self._status(f"Inspecting retained MCD window {getattr(snapshot, 'id', '')}.")
            return
        payload = getattr(snapshot, "track_payload", None)
        if isinstance(payload, Mapping):
            self._mcd_unified_track_payload_key = None
            self._mcd_unified_track_payload = dict(payload)
            selected_id = str(getattr(snapshot, "id", ""))
            view = self.mcd_unified_view
            for index in range(view.candidate_combo.count()):
                if str(view.candidate_combo.itemData(index, Qt.ItemDataRole.UserRole) or "") == selected_id:
                    view.candidate_combo.setCurrentIndex(index)
                    break
            self._plot_mcd_unified(auto=True)
            self._status(f"Inspecting retained feature {selected_id}.")

    def _update_unified_mcd_window(self) -> None:
        view = getattr(self, "mcd_unified_view", None)
        controls = getattr(self, "mcd_unified_controls", None)
        if view is None or controls is None:
            return
        row = controls.retained_window_list.currentRow()
        if row < 0 or row >= len(view.state.retained_windows):
            return
        current = view.state.retained_windows[row]
        item = controls.retained_window_list.currentItem()
        identifier = str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""
        snapshot = self._current_unified_window_snapshot()
        if snapshot is not None and identifier:
            try:
                self.mcd_retention.store.update_window(identifier, snapshot, source_generation=view.state.source_generation)
            except RetainedResultError as exc:
                self._status(str(exc))
                return
        view.state.retained_windows[row] = type(current)(
            current.label, view.state.window_center_ev, view.state.window_width_mev,
            view.state.source_generation, tuple(sorted(self._unified_mcd_settings_snapshot().items())),
        )
        self._refresh_unified_retained_list()

    def _on_unified_feature_display_changed(self, *_args: Any) -> None:
        # Slope bounds are derived from the current window trace, so update
        # their fit metadata when the user edits a bound.  Feature processing
        # controls use the separate selected-feature worker path.
        view = getattr(self, "mcd_unified_view", None)
        if view is not None and view.axes and self.loaded is not None and self.loaded.mode == "MCD":
            self._plot_mcd_unified(auto=True)

    def _show_unified_slope_details(self) -> None:
        view = getattr(self, "mcd_unified_view", None)
        details = str(getattr(view, "_slope_details_text", "No slope fits available."))
        QMessageBox.information(self, "Unified MCD slope fits", details)

    def _unified_window_analysis_key(self) -> tuple[Any, ...]:
        settings = self._unified_mcd_settings_snapshot()
        return (
            id(self.loaded.mcd_result) if self.loaded is not None else None,
            self.mcd_unified_view.state.source_generation,
            float(self.mcd_window_center_spin.value()),
            float(self.mcd_window_width_spin.value()),
            str(self.mcd_window_metric_combo.currentText()),
            tuple((key, value) for key, value in settings.items() if key.startswith("slope_")),
        )

    def _select_unified_mcd_export_items(self, *, scope: str) -> dict[str, Any]:
        if scope == "retained":
            return self.mcd_retention.store.get_included(
                source_generation=int(self.mcd_unified_view.state.source_generation),
            )
        if scope != "current":
            raise ValueError(f"Unknown MCD save scope: {scope}")
        view = self.mcd_unified_view
        if (getattr(self, "_mcd_unified_slopes_key", None) != self._unified_window_analysis_key()
                or view.state.window_center_ev != self.mcd_window_center_spin.value()
                or view.state.window_width_mev != self.mcd_window_width_spin.value()):
            raise ValueError("Current window is updating. Save again after the plot refreshes.")
        window = self._current_unified_window_snapshot()
        if window is None:
            raise ValueError("Current window is unavailable.")
        feature = None
        omission = ""
        if self.mcd_unified_controls.include_feature_chk.isChecked():
            key = self._unified_feature_analysis_key()
            if (self._mcd_unified_track_worker is None and key is not None
                    and getattr(self, "_mcd_unified_track_payload_key", None) == (view.state.source_generation, key)):
                feature = self._current_unified_feature_snapshot()
            if feature is None or feature.get("status") != "complete":
                feature = None
                omission = "Selected feature omitted: analysis is unavailable or updating."
        return {"window": (window,), "feature": (feature,) if feature else (), "omission": omission}

    def _default_unified_mcd_export_folder(self) -> str | None:
        loaded = self.loaded
        result = getattr(loaded, "mcd_result", None)
        source = str(getattr(result, "source_file", "") or "")
        folder = getattr(loaded, "folder", None) or self.current_folder
        if source:
            source_path = Path(source).expanduser()
            if not source_path.is_absolute() and folder:
                source_path = Path(folder) / source_path
            source_folder = source_path.resolve().parent
        elif folder:
            source_folder = Path(folder).expanduser().resolve()
        else:
            return None
        # Acquisition files live in <dataset>/mcd; exports belong beside it.
        if source_folder.name.casefold() == "mcd":
            source_folder = source_folder.parent
        return str(source_folder / "Processed Data" / "MCD")

    def _change_unified_mcd_export_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose MCD results folder", str(getattr(self, "_mcd_export_output_root", None) or self._default_unified_mcd_export_folder() or Path.home())
        )
        if folder and self._unified_mcd_export_folder_writable(folder):
            self._mcd_export_output_root = str(folder)
            self.mcd_unified_controls.save_folder_label.setText(f"Save to: {folder}")
            self.mcd_unified_controls.save_folder_label.setToolTip(str(folder))
        elif folder:
            self._status("Choose a writable MCD results folder.")

    @staticmethod
    def _unified_mcd_export_folder_writable(folder: str | None) -> bool:
        if not folder or not Path(folder).is_dir():
            return False
        try:
            with tempfile.TemporaryFile(dir=folder):
                pass
            return True
        except OSError:
            return False

    def _queue_unified_mcd_export(self, output_root: str | None = None, *, scope: str = "current") -> None:
        """Queue an explicit immutable unified MCD results snapshot."""
        if self.loaded is None or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            self._status("Save Results needs a loaded MCD source.")
            return
        if self._mcd_unified_export_worker is not None:
            self._status("MCD Save Results is already running.")
            return
        try:
            from core.mcd_unified_export import build_mcd_export_snapshot
            view = self.mcd_unified_view
            selected_retained = self._select_unified_mcd_export_items(scope=scope)
            retained_windows = tuple(
                _plain_retained(item)
                for item in selected_retained.get("window", ())
            )
            retained_features = tuple(_plain_retained(item) for item in selected_retained.get("feature", ()))
            result = self.loaded.mcd_result
            try:
                spectrum_order = np.asarray(spectrum_energy_order(result), dtype=int)
            except (AttributeError, TypeError, ValueError):
                spectrum_order = np.argsort(np.asarray(result.energy_ev, dtype=float), kind="stable")
            spectrum_energy = np.asarray(result.energy_ev, dtype=float).ravel()
            spectra_snapshot: dict[str, Any] = {}
            mapping_label = view.channel_mapping_combo.currentText()
            if mapping_label.startswith("K = −"):
                channel_names = (("pos", "K′"), ("neg", "K"))
            elif mapping_label.startswith("K = +"):
                channel_names = (("pos", "K"), ("neg", "K′"))
            else:
                channel_names = (("pos", "+ channel"), ("neg", "− channel"))
            for channel, label in channel_names:
                raw = np.asarray(getattr(result, f"pair_raw_{channel}"), dtype=float)
                corrected = np.asarray(getattr(result, f"pair_corrected_{channel}"), dtype=float)
                fields = np.asarray(getattr(result, f"pair_b_{channel}", result.pair_b), dtype=float).ravel()
                if raw.ndim == 2 and corrected.ndim == 2 and spectrum_order.size == raw.shape[1]:
                    raw, corrected = raw[:, spectrum_order], corrected[:, spectrum_order]
                spectra_snapshot[label] = {
                    "energy_ev": spectrum_energy.copy(), "fields_t": fields.copy(),
                    "values": corrected.copy(), "corrected_values": corrected.copy(),
                    "raw_values": raw.copy(),
                }
            selected_map_snapshot: dict[str, Any] = {}
            map_name = str(self.mcd_map_combo.currentText())
            try:
                map_cube = result.cube(map_name)
                selected_map_snapshot = {
                    "name": map_name,
                    "energy_ev": np.asarray(map_cube.energy, dtype=float).copy(),
                    "fields_t": np.asarray(map_cube.gate, dtype=float).copy(),
                    "values": np.asarray(map_cube.Z, dtype=float).copy(),
                }
            except (AttributeError, KeyError, TypeError, ValueError):
                selected_map_snapshot = {"name": map_name}
            visible_branches = []
            if view.show_inc_chk.isChecked(): visible_branches.append("B increasing")
            if view.show_dec_chk.isChecked(): visible_branches.append("B decreasing")
            # Only checked retained feature snapshots are numerical export
            # inputs.  The current worker payload is a display cache and must
            # not be duplicated into exports when a retained item exists.
            numerical_results = [
                item.get("track_payload") for item in retained_features
                if isinstance(item, Mapping) and isinstance(item.get("track_payload"), Mapping)
            ]
            splitting_rows = []
            link_rows = []
            for payload in numerical_results:
                splitting_rows.extend(payload.get("splitting", ()) or ())
                link_rows.extend(payload.get("links", ()) or ())
            for mcd_id, spectrum_ids in self._mcd_unified_manual_links.items():
                for spectrum_id in spectrum_ids:
                    link_rows.append({"mcd_id": mcd_id, "spectrum_id": spectrum_id, "status": "manual", "kind": "manual_ui"})

            # The XLSX/PNG exporter consumes flat fit records.  Flatten each
            # selected window independently while carrying its identity and
            # requested metric; passing a list of nested SlopeAnalysis
            # dictionaries would otherwise produce blank slope cells.
            slope_rows: list[dict[str, Any]] = []
            for window_index, window in enumerate(retained_windows, start=1):
                slopes = window.get("slopes", {}) if isinstance(window, Mapping) else {}
                metadata = {
                    "window_id": window.get("window_id", window.get("id", f"window-{window_index}")),
                    "center_ev": window.get("center_ev"),
                    "width_mev": window.get("width_mev"),
                    "metric": window.get("metric", "mean"),
                }
                if isinstance(slopes, Mapping) and ("fits" in slopes or "differences" in slopes):
                    for category in ("fits", "differences"):
                        entries = slopes.get(category, ())
                        if isinstance(entries, Mapping):
                            entries = entries.values()
                        for entry_index, entry in enumerate(entries or (), start=1):
                            plain_entry = _plain_retained(entry)
                            record = dict(plain_entry) if isinstance(plain_entry, Mapping) else {"value": plain_entry}
                            record.update({key: value for key, value in metadata.items() if key not in record})
                            record.setdefault("category", category)
                            record.setdefault("fit_id", f"{category}-{entry_index}")
                            slope_rows.append(record)
                elif isinstance(slopes, Mapping) and slopes:
                    record = dict(_plain_retained(slopes))
                    record.update({key: value for key, value in metadata.items() if key not in record})
                    slope_rows.append(record)
            snapshot = build_mcd_export_snapshot(
                self.loaded.mcd_result,
                analysis_results=numerical_results,
                windows=retained_windows,
                slopes=slope_rows,
                links=link_rows,
                features=tuple(retained_features),
                splitting=splitting_rows,
                spectra=spectra_snapshot,
                maps={map_name: selected_map_snapshot} if selected_map_snapshot.get("values") is not None else None,
                selected_map=map_name,
                settings=self._unified_mcd_settings_snapshot(),
                source_descriptor=self.mcd_controller._mcd_peak_source_descriptor(),
                provenance={"source_generation": view.state.source_generation,
                            "save_scope": scope,
                            "selected_window_ids": [item.get("window_id", item.get("id")) for item in retained_windows],
                            "feature_omission": selected_retained.get("omission", "")},
                plot_state={
                    "selected_b_index": view.state.selected_b_index,
                    "selected_field_t": float(result.pair_b[view.state.selected_b_index]) if len(result.pair_b) else None,
                    "spectra_source": view.spectra_source_combo.currentText(),
                    "selected_spectrum": "",
                    "show_corrected": view.spectra_source_combo.currentText().casefold().startswith("correct"),
                    "show_raw": view.spectra_source_combo.currentText().casefold().startswith("raw"),
                    "visible_branches": visible_branches,
                    "window_center_ev": view.state.window_center_ev,
                    "window_width_mev": view.state.window_width_mev,
                    "window_metric": view.state.window_metric,
                    "feature_metric": view.display_metric_label,
                    "feature_overlay": bool(view.state.feature_overlay),
                    "channel_mapping": mapping_label,
                    # Export protocol expects this plot-state value to be a
                    # map name.  The numerical cube is passed through
                    # ``maps`` above; a dict here becomes an unhashable lookup
                    # key in the worker.
                    "selected_map": map_name,
                },
            )
            # Freeze the complete request before a modal dialog can process
            # control changes or background analysis completions.
            output_root = output_root or getattr(self, "_mcd_export_output_root", None)
            custom_output = bool(output_root)
            if not output_root:
                output_root = self._default_unified_mcd_export_folder()
                if output_root:
                    try:
                        Path(output_root).mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass  # Offer another destination if the default is unavailable.
            if not self._unified_mcd_export_folder_writable(output_root):
                output_root = QFileDialog.getExistingDirectory(
                    self, "Choose MCD results folder", str(self.current_folder or Path.home())
                )
                custom_output = True
            if not output_root:
                return
            if not self._unified_mcd_export_folder_writable(output_root):
                raise ValueError("Choose a writable results folder.")
            if custom_output:
                self._mcd_export_output_root = str(output_root)
            self.mcd_unified_controls.save_folder_label.setText(f"Save to: {output_root}")
            self.mcd_unified_controls.save_folder_label.setToolTip(str(output_root))
            worker = Worker(_run_unified_mcd_export, snapshot, str(output_root))
            worker.mcd_history_source = self.loaded
            if scope == "current":
                item = retained_windows[0]
                description = f"current center {item['center_ev']:.6g} eV, width {item['width_mev']:.6g} meV, {item['metric']}"
            else:
                description = f"retained results ({len(retained_windows)} windows, {len(retained_features)} features)"
            worker.mcd_save_description = description
            worker.mcd_feature_omission = selected_retained.get("omission", "")
        except (ImportError, AttributeError, TypeError, ValueError, OSError) as exc:
            self._status(f"MCD Save Results could not be prepared: {exc}")
            return
        self._mcd_unified_export_worker = worker
        self.mcd_unified_controls.save_results_btn.setEnabled(False)
        self.mcd_unified_controls.change_save_folder_btn.setEnabled(False)
        self._update_action_states()
        self._status(f"Saving {description}… {worker.mcd_feature_omission}".strip())
        worker.signals.result.connect(lambda payload, w=worker: self._on_unified_mcd_export_done(w, payload))
        worker.signals.error.connect(lambda message, w=worker: self._on_unified_mcd_export_error(w, message))
        worker.signals.finished.connect(lambda w=worker: self._on_unified_mcd_export_finished(w))
        self.thread_pool.start(worker)

    def _on_unified_mcd_export_done(self, worker: Worker, payload: Any) -> None:
        if worker is self._mcd_unified_export_worker:
            action = "Already saved (unchanged)" if payload.get("reused") else "Saved"
            self._status(f"{action} {worker.mcd_save_description}: {payload.get('export_id', '')} · {payload.get('revision_dir', worker.args[1])}. {worker.mcd_feature_omission}".strip())
            saved = getattr(worker, "mcd_history_source", None)
            if saved is not None:
                roots = self._mcd_history_search_roots(saved.folder)
                roots = tuple(dict.fromkeys((*roots, str(worker.args[1]))))
                self.settings.setValue(self._mcd_history_settings_key(saved.folder), list(roots))
                if self.loaded is saved:
                    self._refresh_saved_mcd_centers(roots)

    @staticmethod
    def _mcd_history_settings_key(folder: str) -> str:
        import hashlib
        key = str(Path(folder).resolve()).casefold().encode("utf-8")
        return "mcd/history_roots/" + hashlib.sha256(key).hexdigest()

    def _mcd_history_search_roots(self, folder: str) -> tuple[str, ...]:
        roots = self.settings.value(self._mcd_history_settings_key(folder), [])
        roots = [roots] if isinstance(roots, str) else list(roots or [])
        current = getattr(self, "_mcd_export_output_root", None)
        if current:
            roots.append(str(current))
        return tuple(dict.fromkeys(roots))

    def _refresh_saved_mcd_centers(self, roots) -> None:
        loaded = self.loaded
        if loaded is None or loaded.mode != "MCD" or not loaded.primary_file:
            return
        worker = Worker(_read_mcd_history_worker, loaded.folder, loaded.primary_file, roots)
        self._mcd_history_worker = worker
        worker.signals.result.connect(lambda rows, w=worker, state=loaded: self._publish_saved_mcd_centers(w, state, rows))
        worker.signals.error.connect(lambda message: self._status(f"Saved MCD center history could not be read: {str(message).splitlines()[0]}"))
        self.thread_pool.start(worker)

    def _publish_saved_mcd_centers(self, worker, loaded, rows) -> None:
        if (getattr(self, "_is_closing", False) or worker is not getattr(self, "_mcd_history_worker", None)
                or loaded is not self.loaded):
            return
        loaded.mcd_center_history = tuple(rows)
        view = self.mcd_unified_view
        if view._owns_current_figure() and view._result is loaded.mcd_result:
            from core.mcd_center_history import merge_center_candidates
            recommendations = [item for item in view._candidates
                               if not item.get("history") or item.get("recommended")
                               or item.get("metadata", {}).get("suggestion")]
            view.refresh_catalog(merge_center_candidates(recommendations, rows), view.analysis_payload)

    def _on_unified_mcd_export_error(self, worker: Worker, message: str) -> None:
        if worker is self._mcd_unified_export_worker:
            self._status(f"Unified MCD export failed: {str(message).splitlines()[0]}")

    def _on_unified_mcd_export_finished(self, worker: Worker) -> None:
        if worker is self._mcd_unified_export_worker:
            self._mcd_unified_export_worker = None
            if not getattr(self, "_is_closing", False):
                self._update_action_states()

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

    def _invalidate_active_load(self, mode: str | None = None) -> None:
        """Prevent completion after its source selection became invalid."""
        self._load_lifecycle_generation = getattr(self, "_load_lifecycle_generation", 0) + 1
        active = getattr(self, "_active_load_mode", None)
        if active and (mode is None or active == mode):
            self._invalidated_load_modes.add(active)
        pending = getattr(self, "_pending_load_mode", None)
        if pending and (mode is None or pending == mode):
            self._invalidated_load_modes.add(pending)
            self._pending_load_mode = None
            self._pending_load_options = None

    def _record_workflow_event(self, *, workflow: str, request_id: int,
                               phase: str, source_key: Any = ()) -> None:
        """Record opt-in workflow events without changing normal behavior."""
        if not getattr(self, "_workflow_metrics_enabled", False):
            return
        self._workflow_metrics.append({
            "workflow": str(workflow),
            "request_id": int(request_id),
            "phase": str(phase),
            "source_key": source_key,
            "elapsed_ms": None,
        })

    def _append_log(self, text: str) -> None:
        self.log_lines.append(text)
        self.log_text.setPlainText("\n".join(self.log_lines))
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _set_stage(self, stage: str) -> None:
        self._status(f"State: {stage}")
        self._status_progress.setVisible(stage == "Loading...")
        self._data_state_loading = stage == "Loading..."
        label = getattr(self, "data_state_label", None)
        if label is not None:
            if stage == "Loading...":
                self._data_state_error = None
                self._refresh_data_state_label()
            elif stage == "No data":
                self._data_state_error = None
                self._refresh_data_state_label()
            elif stage in {"Loaded", "Plotted"}:
                self._data_state_error = None
                self._refresh_data_state_label()
            else:
                self._refresh_data_state_label()

    def _refresh_data_state_label(self) -> None:
        """Refresh source selection against the identity on the shared canvas."""
        label = getattr(self, "data_state_label", None)
        if label is None:
            return
        mode = self._active_mode() if hasattr(self, "tabs") else None
        selected = self._selected_identity_for_status(mode)
        shown = tuple(getattr(self, "_shown_draw_identity", ()))
        selected_lines = "\n".join(str(name) for name in selected) or "none"
        shown_lines = "\n".join(str(name) for name in shown) or "none"
        tooltip = "Selected sources:\n" + selected_lines + "\n\nShown sources:\n" + shown_lines
        label.setToolTip(tooltip)
        for widget_name in ("recent_folder_combo", "folder_edit"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setToolTip("Data source status\n\n" + tooltip)
        if getattr(self, "_data_state_loading", False):
            label.show()
            label.setText("Loading…")
            return
        state = getattr(self, "_data_state_error", None)
        if state:
            label.show()
            label.setText("Load failed")
            return
        if not shown:
            if selected:
                label.show()
                label.setText("Plot not updated")
            else:
                label.hide()
            return
        shown_mode = getattr(self, "_shown_draw_mode", None)
        shown_selection = tuple(getattr(self, "_shown_selection_identity", ()))
        if shown_mode != mode or shown_selection != selected:
            label.show()
            label.setText("Plot not updated")
        else:
            label.hide()

    def _shown_source_identity_for_loaded(self, loaded: Any) -> tuple[str, ...]:
        """Return every source contributing to a successfully rendered state."""
        names: list[str] = []

        def add(values: Any) -> None:
            if isinstance(values, str):
                values = (values,)
            for value in values or ():
                value = getattr(value, "file_name", value)
                text = str(value or "")
                if text and text not in names:
                    names.append(text)

        add(getattr(loaded, "selected_files", ()))
        add(getattr(loaded, "baseline_files", ()))
        compare_sources = getattr(loaded, "compare_sources", {}) or {}
        add(compare_sources.values() if hasattr(compare_sources, "values") else ())
        add(getattr(loaded, "power_records", ()))
        for result in (getattr(loaded, "power_results", {}) or {}).values():
            add(getattr(result, "records", ()))
        return tuple(names)

    def _clear_shown_draw_identity(self, mode: str | None = None) -> None:
        if mode is None or getattr(self, "_shown_draw_mode", None) == mode:
            self._shown_draw_mode = None
            self._shown_draw_identity = ()
            self._shown_selection_identity = ()
            self._refresh_data_state_label()

    def _selected_identity_for_status(self, mode: str | None) -> tuple[str, ...]:
        if mode == "PL":
            return tuple(self._selected(self.pl_files))
        if mode == "DRR":
            return tuple(self.drr_selected_files)
        if mode == "Compare":
            return tuple(self.compare_controller._cmp_current_mapping().values())
        if mode == "Power Dependent":
            keys = [self.power_controller._power_selected_group_key()]
            if self.power_compare_chk.isChecked():
                keys.extend((self.power_controller._power_role_group_key("KK"), self.power_controller._power_role_group_key("KKp")))
            return tuple(key for key in dict.fromkeys(keys) if key)
        if mode == "SHG Processing":
            if self.shg_controller._shg_compare_mode():
                return tuple(key for key in self.shg_controller._shg_compare_files() if key)
            key = self.shg_controller._shg_selected_file()
            return (key,) if key else ()
        if mode == "MCD" and hasattr(self, "mcd_files"):
            return tuple(self._selected(self.mcd_files))
        return ()

    def _active_mode(self) -> str | None:
        text = self.tabs.tabText(self.tabs.currentIndex())
        return {
            "PL": "PL",
            "DRR": "DRR",
            "Compare": "Compare",
            "Power": "Power Dependent",
            "MCD": "MCD",
            "MCD Peak Shift": None,
            "SHG": "SHG Processing",
        }.get(text)

    def _on_central_tab_changed(self, _index: int) -> None:
        mode = self._active_mode()
        if mode in getattr(self, "_plot_redraw_pending", set()):
            timer = getattr(self, "_plot_redraw_timers", {}).get(mode)
            if timer is not None:
                timer.start(0)
        if (self.current_folder and mode is not None and mode not in self._catalog_ready_modes
                and mode != self._catalog_active_scan):
            self._refresh_file_lists(auto=True)
        label = self.tabs.tabText(int(_index))
        if label == "MCD Peak Shift":
            self._plot_mode("MCD Peak Shift")
            self._ensure_mcd_peak_analysis()
        elif (
            label == "MCD"
            and self.loaded is not None
            and self.loaded.mode == "MCD"
            and self.loaded.mcd_result is not None
        ):
            if not self._can_reuse_unified_mcd_view():
                self._plot_mode("MCD")
            if self._unified_catalog_needed():
                self._queue_mcd_unified_analysis()

    def _can_reuse_unified_mcd_view(self) -> bool:
        """Return whether an MCD tab reentry can retain its existing Figure."""
        view = getattr(self, "mcd_unified_view", None)
        loaded = getattr(self, "loaded", None)
        if view is None or loaded is None or loaded.mode != "MCD" or loaded.mcd_result is None:
            return False
        owns = getattr(view, "_owns_current_figure", None)
        if not callable(owns) or not owns() or not getattr(view, "axes", None):
            return False
        signature = (
            id(loaded), id(loaded.mcd_result),
            tuple(self._selected(self.mcd_files)) if hasattr(self, "mcd_files") else (),
            float(self.mcd_window_center_spin.value()),
            float(self.mcd_window_width_spin.value()),
            str(self.mcd_window_metric_combo.currentText()),
            getattr(self, "_mcd_unified_analysis_generation", 0),
        )
        return signature == getattr(self, "_mcd_unified_last_published_signature", None)

    def _toolbar_load(self) -> None:
        mode = self._active_mode()
        if mode:
            if mode == "DRR":
                self._refresh_file_lists(auto=True)
            self._start_load(mode)
        elif self.tabs.tabText(self.tabs.currentIndex()) == "MCD Peak Shift":
            self._start_load("MCD")

    def _toolbar_plot(self) -> None:
        mode = self._active_mode()
        if mode:
            self._plot_mode(mode)
        elif self.tabs.tabText(self.tabs.currentIndex()) == "MCD Peak Shift":
            if self.loaded is not None and self.loaded.mode == "MCD":
                self._analyze_mcd_peak_shift()
            else:
                self._plot_mode("MCD Peak Shift")

    def _toolbar_save(self) -> None:
        mode = self._active_mode()
        if mode:
            if mode == "MCD" and hasattr(self, "mcd_unified_view"):
                self._queue_unified_mcd_export(scope="current")
            else:
                self._start_export(mode)




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





































    def _on_tab_changed(self, _index: int) -> None:
        if self._tools_transitioning:
            return
        self._invalidate_export_move_sources()
        current_label = self.tabs.tabText(self.tabs.currentIndex())
        slides_active = current_label == "Slides"
        tools_active = current_label == "Tools"
        if tools_active and not self._tools_in_workspace:
            self._enter_tools_workspace()
        elif not tools_active and self._tools_in_workspace:
            self._leave_tools_workspace()
            # Reinsert the utility tab without stealing the requested target
            # index (removing the current tab temporarily shifts Qt's index).
            if int(_index) != self.tabs.currentIndex():
                self.tabs.setCurrentIndex(int(_index))
        if hasattr(self, "workspace_stack"):
            self.workspace_stack.setCurrentIndex(1 if slides_active else 2 if tools_active else 0)
        if hasattr(self, "menu_toolbar_host"):
            self.menu_toolbar_host.source_widget_action.setVisible(not slides_active)
            self.menu_toolbar_host.source_separator_action.setVisible(not slides_active)
        if hasattr(self, "sidebar_toggle_btn"):
            self.sidebar_toggle_btn.setEnabled(not slides_active and not tools_active)
        if hasattr(self, "show_sidebar_action"):
            self.show_sidebar_action.setEnabled(not slides_active and not tools_active)
        if slides_active:
            self.left_panel.setVisible(False)
            self.presentation_widget.set_experiment_folder(self.current_folder or None)
        elif tools_active:
            self.left_panel.setVisible(False)
        elif hasattr(self, "left_panel"):
            self._set_sidebar_visible(self.sidebar_toggle_btn.isChecked())
        self._update_action_states()
        self._update_plot_view_bar_visibility()
        self._update_results_dock_page()
        self._refresh_data_state_label()
        if (
            self._active_mode() == "DRR"
            and self.drr_selected_files
            and (
                self.drr_baseline_combo.currentText() != "External"
                or bool(self.drr_baseline_files_manual)
            )
            and not self._load_in_progress
            and (not self.loaded or self.loaded.mode != "DRR")
        ):
            QTimer.singleShot(0, lambda: self._start_load("DRR"))

    def _enter_tools_workspace(self) -> None:
        """Move the existing Tools page into the full-width utility workspace."""
        if self._tools_tab_widget is None or self._tools_in_workspace:
            return
        index = self._tools_tab_index
        if index < 0 or self.tabs.widget(index) is not self._tools_tab_widget:
            return
        self._tools_transitioning = True
        try:
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, self._tools_tab_placeholder, "Tools")
            self.tabs.setTabToolTip(index, "Log / Tools")
            self.tabs.setCurrentIndex(index)
            self._tools_tab_placeholder.show()
            self._tools_workspace_layout.addWidget(self._tools_tab_widget)
            self._tools_tab_widget.show()
            self._tools_in_workspace = True
            self._tools_tab_placeholder.isAncestorOf = self._tools_tab_widget.isAncestorOf
        finally:
            self._tools_transitioning = False

    def _leave_tools_workspace(self) -> None:
        """Restore the Tools page to its original tab without recreating controls."""
        if self._tools_tab_widget is None or not self._tools_in_workspace:
            return
        self._tools_transitioning = True
        try:
            self._tools_workspace_layout.removeWidget(self._tools_tab_widget)
            self._tools_tab_widget.setParent(None)
            index = self._tools_tab_index
            if self.tabs.widget(index) is self._tools_tab_placeholder:
                self.tabs.removeTab(index)
                self._tools_tab_placeholder.hide()
                self.tabs.insertTab(index, self._tools_tab_widget, "Tools")
                self.tabs.setTabToolTip(index, "Log / Tools")
                self.tabs.setCurrentIndex(index)
            self._tools_tab_widget.show()
            self._tools_in_workspace = False
        finally:
            self._tools_transitioning = False


    def _show_error(self, message: str) -> None:
        first = message.splitlines()[0] if message else "Unknown error"
        self._append_log(f"ERROR: {first}")
        self._status(f"Error: {first}")
        label = getattr(self, "data_state_label", None)
        if (
            label is not None
            and (getattr(self, "_load_in_progress", False)
                 or getattr(self, "_active_load_mode", None)
                 or getattr(self, "_data_state_loading", False))
        ):
            self._data_state_loading = False
            self._data_state_error = "failed"
            self._refresh_data_state_label()
        QMessageBox.critical(self, "Error", message)

    def _on_recent_folder_selected(self, index: int) -> None:
        folder = self.recent_folder_combo.itemData(index)
        if not folder:
            return
        folder_text = str(folder)
        if self.current_folder and folder_text.lower() == self.current_folder.lower():
            return
        if self._set_current_folder(folder_text):
            self._status(f"Folder set: {Path(folder_text).name}; loading data source catalog…")

    def _browse_folder(self) -> None:
        start = self._browse_start_folder()
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder", start)
        if not folder:
            return
        if self._set_current_folder(folder):
            self._status(f"Folder set: {Path(folder).name}; loading data source catalog…")

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
        try:
            pending_name = path.relative_to(Path(self.current_folder)).as_posix()
        except ValueError:
            pending_name = path.name
        if self._file_refresh_running:
            self._pending_open_file = pending_name
            self._status(f"Selected {path.name}; waiting for data catalog…")
            return
        matches = self.pl_files.findItems(path.name, Qt.MatchExactly)
        if matches:
            self.pl_files.clearSelection()
            matches[0].setSelected(True)
        self.drr_selected_files = [path.name]
        self.drr_controller._update_drr_selection_labels()
        self.power_controller._power_refresh_groups()
        self.mcd_controller._mcd_refresh_sources()
        self.shg_controller._shg_refresh_sources()
        self._restore_list_selection(self.shg_files, [path.name])
        mcd_name = next(
            (
                candidate
                for candidate in self.mcd_available_files
                if candidate.casefold() == pending_name.casefold()
                or Path(candidate).name.casefold() == path.name.casefold()
            ),
            "",
        )
        if mcd_name:
            self._restore_list_selection(self.mcd_files, [mcd_name])
        self._status(f"Selected {path.name}")

    def _restore_list_selection(self, widget: QListWidget, names: List[str]) -> None:
        widget.clearSelection()
        for name in names:
            matches = widget.findItems(name, Qt.MatchExactly)
            if not matches:
                matches = [
                    widget.item(index)
                    for index in range(widget.count())
                    if str(widget.item(index).data(Qt.UserRole) or "") == str(name)
                ]
            for match in matches:
                match.setSelected(True)

    def _run_pending_catalog_refresh(self) -> None:
        if not self._catalog_pending_requests:
            return
        active = self._active_mode()
        mode = active if active in self._catalog_pending_requests else next(iter(self._catalog_pending_requests))
        force = self._catalog_pending_requests.pop(mode)
        self._refresh_file_lists(auto=not force, mode=mode)

    def _refresh_file_lists(self, *, auto: bool = False, mode: str | None = None) -> None:
        """Queue a folder catalog refresh and apply it only if still current."""
        if self._is_closing:
            return
        old_files = set(self.available_files)
        old_pl_files = set(self.pl_available_files)
        old_mcd_files = set(self.mcd_available_files)
        old_drr_files = {source.source for source in self.drr_available_sources}
        old_source_files = old_files | old_drr_files
        pl_selected = self._selected(self.pl_files)
        if not self.current_folder:
            self._file_refresh_generation += 1
            self.available_files = []
            self.available_map_files = []
            self.pl_available_files = []
            self._pl_source_mtime_cache.clear()
            self.pl_processed_status = {}
            self.pl_processing_ambiguous = set()
            self.mcd_available_files = []
            self.mcd_processed_status = {}
            self.mcd_processing_ambiguous = set()
            self.shg_processed_status = {}
            self.shg_processing_ambiguous = set()
            self.shg_processing_roles = {}
            self._shg_source_mtime_cache.clear()
            self._power_sources_cache = None
            self._power_sources_cache_files = ()
            self._power_catalog_folder = ""
            self._power_catalog_include_legacy = False
            self._power_catalog_candidates = ()
            self._power_catalog_signatures = ()
            self._power_catalog_processed_names = set()
            self._power_catalog_combined_names = set()
            self._power_catalog_pending = False
            self._power_pending_open_name = ""
            self._power_pending_open_generation = None
            self._power_pending_measurement_selection = None
            self._power_pending_measurement_validation = {}
            self._power_pending_measurement_generation = None
            self._power_result_cache.clear()
            self._drr_refresh_pending = False
            self._drr_refresh_pending_auto = False
            self._drr_refresh_pending_old_sources = None
            self._drr_refresh_pending_selected_sources = None
            self.drr_available_sources = []
            self._drr_source_cache.clear()
            return
        mode = mode or self._active_mode()
        if mode == "DRR":
            self._queue_drr_catalog_refresh(auto=auto, old_source_files=old_source_files)
            return
        if mode not in {"PL", "Compare", "Power Dependent", "MCD", "SHG Processing"}:
            return
        if self._file_refresh_running:
            self._catalog_pending_requests[mode] = self._catalog_pending_requests.get(mode, False) or not auto
            self._file_refresh_pending = True
            self._file_refresh_pending_auto = self._file_refresh_pending_auto or auto
            return
        self._status("Loading data source catalog…")
        self._file_refresh_running = True
        self._file_refresh_pending = bool(self._catalog_pending_requests)
        self._catalog_active_scan = mode
        if mode == "Power Dependent":
            self._power_catalog_pending = True
        self._file_refresh_generation += 1
        generation = self._file_refresh_generation
        folder = self.current_folder
        worker = Worker(
            _cached_folder_sources_worker,
            folder, mode=mode, force=not auto, publish_cached=None,
            power_include_legacy=bool(getattr(self, "_power_include_legacy", False)),
        )
        worker.kwargs["publish_cached"] = worker.signals.result.emit
        self._file_refresh_workers.append(worker)
        worker.signals.result.connect(
            lambda result, generation=generation, auto=auto, old_files=set(old_files),
            old_pl_files=set(old_pl_files), old_mcd_files=set(old_mcd_files),
            old_source_files=set(old_source_files), pl_selected=list(pl_selected):
            self._on_file_lists_result(result, generation, auto, old_files, old_pl_files,
                                       old_mcd_files, old_source_files, pl_selected)
        )
        worker.signals.error.connect(
            lambda message, generation=generation, folder=folder:
            self._on_file_lists_error(message, generation, folder)
        )
        worker.signals.finished.connect(lambda worker=worker: self._finish_file_refresh_worker(worker))
        self.thread_pool.start(worker)

    def _on_file_lists_result(self, result, generation: int, auto: bool, old_files: set[str],
                              old_pl_files: set[str], old_mcd_files: set[str],
                              old_source_files: set[str], pl_selected: list[str]) -> None:
        if generation != self._file_refresh_generation:
            return
        folder, available_files, map_files, pl_files, pl_status, mcd_files, mcd_status, *extra = result
        scope = next((item for item in extra if isinstance(item, dict) and item.get("tag") == "catalog_scope"), {})
        mode = scope.get("mode", "all")
        preview = bool(scope.get("preview"))
        if preview and mode in self._catalog_displayed_modes:
            return
        if len(extra) >= 6:
            pl_unknown, mcd_unknown, shg_status, shg_unknown, shg_mtimes, shg_roles = extra[:6]
        elif len(extra) >= 5:
            pl_unknown, mcd_unknown, shg_status, shg_unknown, shg_mtimes = extra[:5]
            shg_roles = {}
        else:
            pl_unknown, mcd_unknown = extra if len(extra) == 2 else (set(), set())
            shg_status, shg_unknown, shg_mtimes, shg_roles = {}, set(), {}, {}
        power_snapshot = next(
            (item for item in extra if isinstance(item, dict) and item.get("tag") == "power_catalog"),
            None,
        )
        source_metadata = next(
            (item for item in extra if isinstance(item, dict) and item.get("tag") == "source_metadata"),
            None,
        )
        if str(folder).casefold() != str(self.current_folder).casefold():
            self._file_refresh_running = False
            if self._file_refresh_pending:
                pending_auto = self._file_refresh_pending_auto
                self._file_refresh_pending = False
                self._file_refresh_pending_auto = False
                QTimer.singleShot(0, self._run_pending_catalog_refresh)
            return
        if not preview:
            self._file_refresh_running = False
            self._catalog_active_scan = None
            self._catalog_ready_modes.add(mode)
            if mode in {"all", "Compare"}:
                self._cmp_history_published_generation = generation
                self._cmp_history_published_folder = str(folder)
        shared_rows_displayed = bool({"PL", "Compare"} & self._catalog_displayed_modes)
        self._catalog_displayed_modes.add(mode)
        if preview and mode in {"PL", "Compare"} and shared_rows_displayed:
            pl_files = self.pl_available_files
        if mode not in {"all", "PL", "Compare"}:
            pl_files = self.pl_available_files
        if mode not in {"all", "PL"}:
            pl_status, pl_unknown = self.pl_processed_status, self.pl_processing_ambiguous
        if mode not in {"all", "MCD"}:
            mcd_files, mcd_status, mcd_unknown = self.mcd_available_files, self.mcd_processed_status, self.mcd_processing_ambiguous
        if mode not in {"all", "SHG Processing"}:
            shg_status, shg_unknown = self.shg_processed_status, self.shg_processing_ambiguous
            shg_mtimes, shg_roles = self._shg_source_mtime_cache, self.shg_processing_roles
        if mode not in {"all", "Compare"}:
            map_files = self.available_map_files
        self.available_files = list(available_files)
        self.available_map_files = list(map_files)
        self.pl_available_files = list(pl_files)
        if source_metadata is not None and str(source_metadata.get("folder", "")).casefold() == str(self.current_folder).casefold():
            published_modes = set(source_metadata.get("modes", ()))
            if "PL" in published_modes:
                self._pl_source_mtime_cache = {
                    str(source): float(value or 0.0)
                    for source, value in dict(source_metadata.get("pl_mtimes", {})).items()
                }
            if "MCD" in published_modes:
                self._mcd_source_mtime_cache = {
                    str(source): float(value or 0.0)
                    for source, value in dict(source_metadata.get("mcd_mtimes", {})).items()
                }
        else:
            self._pl_source_mtime_cache.clear()
            self._mcd_source_mtime_cache.clear()
        if mode in {"all", "Power Dependent"}:
            if power_snapshot is not None and str(power_snapshot.get("folder", "")).casefold() == str(self.current_folder).casefold():
                self._power_sources_cache = dict(power_snapshot.get("sources", {}))
                self._power_sources_cache_files = tuple(power_snapshot.get("signatures", ()))
                self._power_catalog_folder = str(power_snapshot.get("folder", ""))
                self._power_catalog_include_legacy = bool(power_snapshot.get("include_legacy", False))
                self._power_catalog_candidates = tuple(str(value) for value in power_snapshot.get("candidates", ()))
                self._power_catalog_signatures = tuple(power_snapshot.get("signatures", ()))
                self._power_catalog_processed_names = {
                    str(value).replace("\\", "/").casefold()
                    for value in power_snapshot.get("processed_names", ())
                }
                self._power_catalog_combined_names = {
                    str(value).replace("\\", "/").casefold()
                    for value in power_snapshot.get("combined_names", ())
                }
                self._power_saved_assignments_cache = dict(power_snapshot.get("saved_assignments", {}))
                self._power_catalog_pending = False
            else:
                self._power_catalog_pending = False
                self._power_sources_cache = None
                self._power_sources_cache_files = ()
                self._power_catalog_folder = ""
                self._power_catalog_include_legacy = False
                self._power_catalog_candidates = ()
                self._power_catalog_signatures = ()
                self._power_catalog_processed_names = set()
                self._power_catalog_combined_names = set()
                self._power_saved_assignments_cache = {}
            self._power_result_cache.clear()
        self.pl_processed_status = dict(pl_status)
        self.pl_processing_ambiguous = set(pl_unknown)
        self.mcd_available_files = list(mcd_files)
        if mode in {"all", "MCD"}:
            self._mcd_status_from_refresh = dict(mcd_status)
            self._mcd_ambiguous_from_refresh = set(mcd_unknown)
        self.shg_processed_status = dict(shg_status)
        self.shg_processing_ambiguous = set(shg_unknown)
        self._shg_source_mtime_cache = dict(shg_mtimes)
        self.shg_processing_roles = {
            str(source): tuple(roles) for source, roles in dict(shg_roles).items()
        }
        pl_selected = self._selected(self.pl_files)
        if [self.pl_files.item(i).text() for i in range(self.pl_files.count())] != self.pl_available_files:
            self.pl_files.clear()
            self.pl_files.addItems(self.pl_available_files)
            retained_pl = [f for f in pl_selected if f in self.pl_available_files]
            self._restore_list_selection(self.pl_files, retained_pl)
        self.pl_controller._update_pl_selection_summary()
        if mode in {"all", "Compare"}:
            self.compare_controller._cmp_set_channel_combo_items()
            if mode == "Compare":
                self._cmp_history_records = list(scope.get("compare_history", ()))
                self._cmp_history_cache_folder = str(self.current_folder)
                self._cmp_source_mtime_cache = dict((source_metadata or {}).get("compare_mtimes", {}))
                self._cmp_source_mtime_cache_folder = str(self.current_folder)
            else:
                self._cmp_history_records = list(scope.get("compare_history", ()))
                self._cmp_history_cache_folder = str(self.current_folder)
                self._cmp_source_mtime_cache = dict((source_metadata or {}).get("compare_mtimes", {}))
                self._cmp_source_mtime_cache_folder = str(self.current_folder)
        if mode in {"all", "Power Dependent"} and not preview:
            self.power_controller._power_refresh_groups()
            pending_measurement = getattr(self, "_power_pending_measurement_selection", None)
            pending_measurement_generation = getattr(self, "_power_pending_measurement_generation", None)
            if (
                pending_measurement is not None
                and (pending_measurement_generation is None or generation >= int(pending_measurement_generation))
                and bool(pending_measurement.get("legacy", False)) == bool(getattr(self, "_power_catalog_include_legacy", False))
            ):
                required = ([pending_measurement.get("single")]
                            if pending_measurement.get("action") == "Single intensity"
                            else [pending_measurement.get("KK"), pending_measurement.get("KKp")])
                sources = self._power_sources_cache if isinstance(self._power_sources_cache, dict) else {}
                missing = [str(key) for key in required if not key or key not in sources]
                self._power_pending_measurement_selection = None
                validation = getattr(self, "_power_pending_measurement_validation", {})
                self._power_pending_measurement_validation = {}
                self._power_pending_measurement_generation = None
                if missing:
                    self._show_error(
                        "The selected Power source is no longer available after refresh: "
                        + ", ".join(missing)
                    )
                else:
                    self.power_controller._apply_power_measurement_group_from_selection(
                        pending_measurement, validation
                    )
            pending_power_name = str(getattr(self, "_power_pending_open_name", "") or "")
            pending_power_generation = getattr(self, "_power_pending_open_generation", None)
            if pending_power_name and (pending_power_generation is None or generation >= int(pending_power_generation)):
                self._power_pending_open_name = ""
                self._power_pending_open_generation = None
                pending_key = data_io.power_sweep_source_key(pending_power_name)
                index = self.power_group_combo.findData(pending_key)
                if index >= 0:
                    self.power_group_combo.setCurrentIndex(index)
                    self._start_load("Power Dependent")
                else:
                    self._show_error(
                        f"The combined sweep was saved but could not be discovered: {pending_power_name}"
                    )
        elif mode == "Power Dependent":
            self.power_controller._power_refresh_groups()
        if mode in {"all", "MCD"}:
            self.mcd_controller._mcd_refresh_sources()
        if mode in {"all", "SHG Processing"}:
            self.shg_controller._shg_refresh_sources()
        if mode in {"all", "Compare"}:
            self.compare_controller._cmp_auto_assign_channels()
        if preview:
            self._status(f"Showing cached {mode} files; checking for changes…")
            return
        if self._pending_open_file:
            pending = self._pending_open_file
            self._pending_open_file = ""
            pending_basename = Path(pending).name
            pl_name = pending if pending in self.pl_available_files else pending_basename
            self._restore_list_selection(self.pl_files, [pl_name])
            mcd_name = next(
                (
                    candidate
                    for candidate in self.mcd_available_files
                    if candidate.casefold() == pending.casefold()
                    or Path(candidate).name.casefold() == pending_basename.casefold()
                ),
                "",
            )
            if mcd_name:
                self._restore_list_selection(self.mcd_files, [mcd_name])
            if pending in self.available_files:
                self.drr_selected_files = [pending]
                self.drr_controller._update_drr_selection_labels()
            elif pending_basename in self.available_files:
                self.drr_selected_files = [pending_basename]
                self.drr_controller._update_drr_selection_labels()
        old_source_files |= old_pl_files | old_mcd_files
        self._status(f"Data source catalog ready: {len(self.available_files)} CSV files.")
        # A dialog waiting on Refresh must observe the final catalog in a
        # coalesced refresh chain. Intermediate results are already applied
        # to the owner, but emitting them would let a picker rebuild from the
        # stale first scan before the queued scan starts.
        emit_catalog_completion = not self._file_refresh_pending
        if emit_catalog_completion:
            self.file_catalog_refresh_finished.emit(str(self.current_folder), True)
        if self._file_refresh_pending:
            pending_auto = self._file_refresh_pending_auto
            self._file_refresh_pending = False
            self._file_refresh_pending_auto = False
            QTimer.singleShot(0, self._run_pending_catalog_refresh)

    def _on_file_lists_error(self, message: str, generation: int, folder: str) -> None:
        if generation != self._file_refresh_generation:
            return
        self._file_refresh_running = False
        self._catalog_active_scan = None
        if str(folder).casefold() == str(self.current_folder).casefold():
            self._power_catalog_pending = False
            if getattr(self, "_power_pending_open_name", ""):
                self._power_pending_open_name = ""
                self._power_pending_open_generation = None
            self._status(f"Folder refresh failed: {str(message).splitlines()[0]}")
            if not self._file_refresh_pending:
                self.file_catalog_refresh_finished.emit(str(folder), False)
        if self._file_refresh_pending:
            pending_auto = self._file_refresh_pending_auto
            self._file_refresh_pending = False
            self._file_refresh_pending_auto = False
            QTimer.singleShot(0, self._run_pending_catalog_refresh)

    def _finish_file_refresh_worker(self, worker: Worker) -> None:
        try:
            self._file_refresh_workers.remove(worker)
        except ValueError:
            pass

    def _queue_drr_catalog_refresh(self, *, auto: bool, old_source_files: set[str],
                                   selected_sources=()) -> None:
        """Coalesce DRR scans while keeping unrelated file-list refreshes synchronous."""
        if self._is_closing:
            return
        if self._drr_refresh_running:
            if not self._drr_refresh_pending:
                self._drr_refresh_pending_auto = auto
            else:
                self._drr_refresh_pending_auto = self._drr_refresh_pending_auto and auto
            self._drr_refresh_pending = True
            if self._drr_refresh_pending_old_sources is None:
                self._drr_refresh_pending_old_sources = set(old_source_files)
            pending_selected = self._drr_refresh_pending_selected_sources or set()
            pending_selected.update(str(value) for value in selected_sources)
            self._drr_refresh_pending_selected_sources = pending_selected
            return
        kwargs = {"auto": auto, "old_source_files": old_source_files}
        if selected_sources:
            kwargs["selected_sources"] = selected_sources
        self._start_drr_catalog_refresh(**kwargs)

    def _start_drr_catalog_refresh(self, *, auto: bool, old_source_files: set[str],
                                   selected_sources=()) -> None:
        if self._is_closing or not self.current_folder:
            return
        self._drr_refresh_running = True
        self._drr_refresh_generation += 1
        generation = self._drr_refresh_generation
        folder = self.current_folder
        worker = Worker(
            _scan_drr_catalog_worker,
            folder,
            self._drr_source_cache.clone(),
            force=not auto, publish_cached=None,
            include_all=bool(getattr(self, "_drr_include_all_sources", False)),
            selected_sources=tuple(selected_sources),
        )
        worker.kwargs["publish_cached"] = worker.signals.result.emit
        self._drr_refresh_workers.append(worker)
        worker.signals.result.connect(
            lambda result, generation=generation, auto=auto, old_source_files=set(old_source_files),
            selected_sources=tuple(selected_sources):
            self._on_drr_catalog_refresh_result(result, generation, auto, old_source_files, selected_sources)
        )
        worker.signals.error.connect(
            lambda message, generation=generation, folder=folder: self._on_drr_catalog_refresh_error(
                message, generation, folder
            )
        )
        worker.signals.finished.connect(
            lambda worker=worker: self._on_drr_catalog_refresh_finished(worker)
        )
        self.thread_pool.start(worker)

    @staticmethod
    def _drr_missing_result_is_terminal(missing_selected, current_selection,
                                         request_selection, refresh_pending) -> bool:
        """Only the still-current request may invalidate DRR selection."""
        missing = {str(value) for value in missing_selected}
        current = {str(value) for value in current_selection}
        requested = {str(value) for value in request_selection}
        return bool(missing and not refresh_pending and missing & current
                    and (not requested or requested.issuperset(missing)))

    def _on_drr_catalog_refresh_result(
        self,
        result: tuple[str, List[DrrSource], DrrSourceCache],
        generation: int,
        auto: bool,
        old_source_files: set[str],
        selected_sources: tuple[str, ...] = (),
    ) -> None:
        if generation != self._drr_refresh_generation:
            return
        folder, sources, cache = result[:3]
        preview = len(result) > 3 and bool(result[3])
        if len(result) > 4 and bool(result[4]) != bool(getattr(self, "_drr_include_all_sources", False)):
            if not preview:
                self._drr_refresh_running = False
                self._finish_drr_catalog_refresh()
            return
        if preview:
            if str(folder).casefold() == str(self.current_folder).casefold() and "DRR" not in self._catalog_displayed_modes:
                self.drr_available_sources = sources
                self._catalog_displayed_modes.add("DRR")
                self.drr_controller._update_drr_selection_labels()
                self._status("Showing cached DRR files; checking for changes…")
                self.drr_catalog_preview_ready.emit(str(folder))
            return
        missing_selected = tuple(result[5]) if len(result) > 5 else ()
        current_selection = set(str(value) for value in self.drr_selected_files)
        request_selection = set(str(value) for value in selected_sources)
        terminal_missing = self._drr_missing_result_is_terminal(
            missing_selected, current_selection, request_selection,
            bool(self._drr_refresh_pending),
        )
        if terminal_missing and str(folder).casefold() == str(self.current_folder).casefold():
            self._invalidate_drr_for_background_selection(
                "Selected DRR source is unavailable: " + ", ".join(missing_selected)
            )
            self._drr_refresh_running = False
            self.drr_catalog_refresh_finished.emit(str(folder), False)
            self._finish_drr_catalog_refresh()
            return
        self._drr_refresh_running = False
        if str(folder).casefold() != str(self.current_folder).casefold():
            self._finish_drr_catalog_refresh()
            return

        old_drr_groups = group_drr_sources(self.drr_available_sources)
        selected_before = set(self.drr_selected_files)
        selected_missing_before = set(self.drr_controller._drr_missing_sources(self.drr_selected_files))
        selected_complete_group_keys = {
            group.key
            for group in old_drr_groups
            if group.files
            and {source.source for source in group.files}.issubset(selected_before)
            and not selected_missing_before
        }
        self.drr_available_sources = sources
        self._catalog_ready_modes.add("DRR")
        self._catalog_displayed_modes.add("DRR")
        self._drr_source_cache = cache
        drr_candidates = {source.source for source in sources}
        # Preserve chosen identities, including files removed from the
        # filtered catalog.  The dialog and load boundary display/block them.
        self.drr_selected_files = list(dict.fromkeys(self.drr_selected_files))
        if selected_complete_group_keys:
            selected_now = set(self.drr_selected_files)
            for group in group_drr_sources(sources):
                if group.key not in selected_complete_group_keys:
                    continue
                if group.is_background:
                    continue
                selected_members = [
                    source for source in group.files if source.source in selected_now
                ]
                if not selected_members:
                    continue
                compatible = set(
                    compatible_drr_repeats(sources, selected_members[0].source)
                )
                for source in group.files:
                    if source.source not in selected_now and source.source in compatible:
                        self.drr_selected_files.append(source.source)
                        selected_now.add(source.source)
        if not self._drr_assignments_automatic:
            self.drr_baseline_files_manual = list(dict.fromkeys(self.drr_baseline_files_manual))
        self.drr_baseline_files_found = [
            f for f in self.drr_baseline_files_found if f in drr_candidates
        ]
        self.drr_controller._update_drr_selection_labels()
        if self._drr_assignments_automatic and self.drr_selected_files:
            resolved = resolve_drr_background_assignments(
                self.current_folder, self.drr_available_sources, self.drr_selected_files,
            )
            if resolved.resolved:
                self._drr_assignments = tuple(resolved.assignments)
                self.drr_baseline_files_manual = list(dict.fromkeys(
                    source for assignment in resolved.assignments
                    for source in assignment.baseline_files
                ))
            else:
                self._drr_assignments = ()
                # Keep the automatic intent while the assignment set is
                # unresolved.  The display union must never become a manual
                # common-background recipe on the next load.
                self.drr_baseline_files_manual = []
                self.drr_baseline_files_found = []
                self._invalidate_drr_for_background_selection(
                    resolved.reason or "DRR background assignments became unresolved after refresh."
                )

        new_source_files = (
            set(self.available_files)
            | set(self.pl_available_files)
            | drr_candidates
            | set(self.mcd_available_files)
        )
        added = len(new_source_files - old_source_files)
        removed = len(old_source_files - new_source_files)
        if added or removed:
            self._invalidate_export_move_sources()
        if auto and (added or removed):
            parts = []
            if added:
                parts.append(f"{added} new")
            if removed:
                parts.append(f"{removed} removed")
            self._status(
                f"Data source updated: {', '.join(parts)} "
                f"({len(new_source_files)} source files)."
            )
        elif not auto:
            self._status(f"Loaded file list: {len(new_source_files)} source files")
        self.drr_catalog_refresh_finished.emit(str(folder), True)
        self._finish_drr_catalog_refresh()

    def _on_drr_catalog_refresh_error(
        self, message: str, generation: int, folder: str
    ) -> None:
        if generation != self._drr_refresh_generation:
            return
        self._drr_refresh_running = False
        if str(folder).casefold() == str(self.current_folder).casefold():
            self._status(f"DRR catalog refresh failed: {str(message).splitlines()[0]}")
            self.drr_catalog_refresh_finished.emit(str(folder), False)
        self._finish_drr_catalog_refresh()

    def _on_drr_catalog_refresh_finished(self, worker: Worker) -> None:
        if worker in self._drr_refresh_workers:
            self._drr_refresh_workers.remove(worker)

    def _finish_drr_catalog_refresh(self) -> None:
        if self._drr_refresh_running:
            return
        if not self._drr_refresh_pending:
            return
        self._drr_refresh_pending = False
        auto = self._drr_refresh_pending_auto
        old_source_files = self._drr_refresh_pending_old_sources or set()
        self._drr_refresh_pending_auto = False
        self._drr_refresh_pending_old_sources = None
        selected_sources = self._drr_refresh_pending_selected_sources or set()
        self._drr_refresh_pending_selected_sources = None
        kwargs = {"auto": auto, "old_source_files": old_source_files}
        if selected_sources:
            kwargs["selected_sources"] = selected_sources
        self._start_drr_catalog_refresh(**kwargs)

    def _reset_workflow_state_for_folder_change(self) -> None:
        """Prevent selections and plots from one experiment leaking into another."""
        self.available_files = []
        self.available_map_files = []
        self.pl_available_files = []
        self.mcd_available_files = []
        self.drr_available_sources = []
        for name in ("pl_files", "mcd_files", "shg_files"):
            widget = getattr(self, name, None)
            if widget is not None:
                previous = widget.blockSignals(True)
                widget.clear()
                widget.blockSignals(previous)
        self._load_lifecycle_generation += 1
        self._invalidated_load_modes.update(
            mode for mode in (
                getattr(self, "_active_load_mode", None),
                getattr(self, "_pending_load_mode", None),
            ) if mode
        )
        self._pending_load_mode = None
        self._pending_load_options = None
        self.mcd_controller._invalidate_mcd_peak_shift(queue_reload=False)
        self.loaded = None
        self.last_plotted_mode = None
        self._last_draw_identity.clear()
        self._clear_shown_draw_identity()
        self._last_plot_cube = None
        self._last_plot_params_key = None
        self._power_sources_cache = None
        self._power_sources_cache_files = ()
        self._power_catalog_folder = ""
        self._power_catalog_include_legacy = False
        self._power_catalog_candidates = ()
        self._power_catalog_signatures = ()
        self._power_catalog_processed_names = set()
        self._power_catalog_combined_names = set()
        self._power_catalog_pending = False
        self._power_pending_open_name = ""
        self._power_pending_open_generation = None
        self._power_pending_measurement_selection = None
        self._power_pending_measurement_validation = {}
        self._power_pending_measurement_generation = None
        self.drr_selected_files = []
        self.drr_baseline_files_manual = []
        self.drr_baseline_files_found = []
        self._drr_assignments = ()
        self._drr_assignments_automatic = False
        self._drr_baseline_user_selected = False
        self._drr_background_guess = None
        self._drr_source_cache.clear()
        self._pl_auto_next_queue = []
        self._pl_auto_next_active = False
        object.__setattr__(self.shg_controller, "_shg_selected_source", "")
        for name in ("pl_files", "mcd_files", "shg_files"):
            widget = getattr(self, name, None)
            if widget is not None:
                blocked = widget.blockSignals(True)
                widget.clearSelection()
                widget.blockSignals(blocked)
        for name in (
            "shg_background_combo",
            "shg_compare_reference_combo",
            "shg_compare_sample_combo",
            "shg_compare_background_a_combo",
            "shg_compare_background_b_combo",
            "mcd_dark_pos_combo",
            "mcd_dark_neg_combo",
        ):
            combo = getattr(self, name, None)
            if combo is not None and combo.count():
                blocked = combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(blocked)
        self._active_export_request_key = ""
        self._last_export_request_key = ""
        if hasattr(self, "figure"):
            self.figure.clear()
            self.canvas.draw_idle()
        self._drr_heatmap_ax = None
        self._drr_spectrum_ax = None
        self._drr_heatmap_axes = {}
        self._drr_spectrum_axes = {}
        self._drr_plot_cubes = {}
        self._drr_gate_lines = {}
        self._update_action_states()

    def _clear_loaded_drr_view(self) -> None:
        self._invalidate_export_move_sources()
        if self.loaded and self.loaded.mode == "DRR":
            self.loaded = None
        if self.last_plotted_mode == "DRR":
            self.last_plotted_mode = None
            self.figure.clear()
            self.canvas.draw_idle()
            self._clear_shown_draw_identity("DRR")
        self._drr_heatmap_ax = None
        self._drr_spectrum_ax = None
        self._drr_view_limits = None
        self._drr_heatmap_axes = {}
        self._drr_spectrum_axes = {}
        self._drr_plot_cubes = {}
        self._drr_gate_lines = {}
        self._last_plot_cube = None
        self._last_plot_params_key = None

    def _invalidate_drr_for_background_selection(self, message: str) -> None:
        self._invalidate_active_load("DRR")
        self._clear_loaded_drr_view()
        self._set_stage("Background required")
        self._status(message)
        self._update_action_states()



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


    def _update_action_states(self) -> None:
        self._update_unified_mcd_save_controls()
        active_mode = self._active_mode()
        loaded_mode = self.loaded.mode if self.loaded else None
        plotted_mode = self.last_plotted_mode
        peak_shift_active = self.tabs.tabText(self.tabs.currentIndex()) == "MCD Peak Shift"
        self.load_action.setEnabled(active_mode is not None or peak_shift_active)
        self.plot_action.setEnabled(
            (active_mode is not None and loaded_mode == active_mode)
            or (peak_shift_active and loaded_mode == "MCD")
        )
        self.save_action.setEnabled(
            active_mode is not None
            and plotted_mode == active_mode
            and not self._export_in_progress
            and (active_mode != "MCD" or self._mcd_unified_export_worker is None)
        )
        self._update_move_exported_sources_state()
        pl_loaded = loaded_mode == "PL"
        drr_loaded = loaded_mode == "DRR"
        self._sync_drr_gate_toolbar_state(drr_loaded)
        cmp_loaded = loaded_mode == "Compare"
        power_loaded = loaded_mode == "Power Dependent"
        mcd_loaded = loaded_mode == "MCD"
        self.pl_auto_v_btn.setEnabled(pl_loaded and not self.pl_split_scale_chk.isChecked())
        self.pl_auto_x_btn.setEnabled(pl_loaded)
        self.pl_auto_y_btn.setEnabled(pl_loaded)
        self.drr_auto_v_btn.setEnabled(drr_loaded and not self.drr_split_scale_chk.isChecked())
        self.drr_auto_x_btn.setEnabled(drr_loaded)
        self.drr_auto_y_btn.setEnabled(drr_loaded)
        self.cmp_auto_v_btn.setEnabled(
            cmp_loaded
            and not self.compare_controller._cmp_is_vp_view()
            and not self.cmp_split_scale_chk.isChecked()
        )
        self.cmp_auto_x_btn.setEnabled(cmp_loaded)
        self.cmp_auto_y_btn.setEnabled(cmp_loaded)
        vp_active = cmp_loaded and self.compare_controller._cmp_is_vp_view()
        for key in ("vmin", "vmax"):
            self.cmp_spins[key].setEnabled(not vp_active)
            self.cmp_fix_checks[key].setEnabled(not vp_active)
        if hasattr(self, "cmp_vp_auto_v_btn"):
            vp_fixed = all(check.isChecked() for check in self.cmp_vp_fix_checks.values())
            self.cmp_vp_auto_v_btn.setEnabled(vp_active and not vp_fixed)
            for key in ("vmin", "vmax"):
                self.cmp_vp_spins[key].setEnabled(vp_active)
                self.cmp_vp_fix_checks[key].setEnabled(vp_active)
        self.power_auto_v_btn.setEnabled(power_loaded and not self.power_split_scale_chk.isChecked())
        self.power_auto_x_btn.setEnabled(power_loaded)
        self.power_auto_y_btn.setEnabled(power_loaded)
        self.mcd_auto_v_btn.setEnabled(mcd_loaded)
        for prefix, enabled in (
            ("pl", pl_loaded),
            ("drr", drr_loaded),
            ("drr_second", drr_loaded),
            ("cmp", cmp_loaded and not self.compare_controller._cmp_is_vp_view()),
            ("power", power_loaded and self.power_controller._power_view() != "VP"),
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
        cmp_split_available = not self.compare_controller._cmp_is_vp_view()
        power_split_available = self.power_controller._power_view() != "VP"
        self.cmp_split_scale_chk.setEnabled(cmp_split_available)
        self.cmp_split_scale_panel.setEnabled(cmp_split_available)
        self.power_split_scale_chk.setEnabled(power_split_available)
        self.power_split_scale_panel.setEnabled(power_split_available)
        self.drr_second_split_scale_chk.setEnabled(drr_loaded)
        self.drr_second_split_scale_panel.setEnabled(drr_loaded)





    def _launch_load_options(self, options: LoadOptions) -> None:
        """Launch an already captured immutable load request."""
        mode = options.mode
        trackers = getattr(self, "_load_requests", None)
        if trackers is None:
            trackers = {}
            self._load_requests = trackers
        tracker = trackers.setdefault(mode, LatestRequest())
        token, should_start = tracker.submit(mode, str(options.folder).casefold(), options)
        if not should_start:
            self._pending_load_mode = mode
            self._pending_load_options = options
            self._invalidated_load_modes = {mode}
            return
        self._load_lifecycle_generation += 1
        self._set_stage("Loading...")
        self._load_in_progress = True
        self._active_load_mode = mode
        self._active_load_token = token
        self._active_load_succeeded = False
        recorder = getattr(self, "_record_workflow_event", None)
        if callable(recorder):
            recorder(workflow=mode, request_id=token.generation, phase="enqueue",
                     source_key=tuple(options.selected_files))
        worker = Worker(self._load_task, options)
        worker.signals.log.connect(self._append_log)
        load_generation = self._mcd_source_generation if mode == "MCD" else None
        worker.signals.result.connect(
            lambda loaded, token=token, generation=load_generation:
            self._on_loaded(loaded, generation, request_token=token)
        )
        worker.signals.error.connect(lambda message, token=token: self._on_load_error(message, token))
        worker.signals.finished.connect(lambda token=token: self._on_load_finished(request_token=token))
        self.thread_pool.start(worker)

    def _launch_pending_if_current(self, options: LoadOptions, generation: int,
                                   folder: str) -> None:
        """Run a deferred request only while its lifecycle snapshot remains valid."""
        if (
            getattr(self, "_is_closing", False)
            or getattr(self, "_load_in_progress", False)
            or generation != getattr(self, "_load_lifecycle_generation", 0)
            or str(getattr(self, "current_folder", "")) != folder
        ):
            return
        self._launch_load_options(options)

    def _start_load(self, mode: str) -> None:
        if self._is_closing:
            return
        if self._load_in_progress and not getattr(self, "_capture_pending_load", False):
            self._load_lifecycle_generation = getattr(self, "_load_lifecycle_generation", 0) + 1
            active_mode = getattr(self, "_active_load_mode", None)
            if active_mode:
                self._invalidated_load_modes.add(active_mode)
            self._pending_load_mode = mode
            self._invalidated_load_modes.add(mode)
            # Re-enter the option collection path while the worker remains
            # active.  This snapshots the latest controls without starting a
            # second worker or rereading them after completion.
            self._capture_pending_load = True
            try:
                self._start_load(mode)
            finally:
                self._capture_pending_load = False
            self._status("State: Load in progress; latest update queued.")
            return
        capturing_pending = bool(getattr(self, "_capture_pending_load", False))
        has_pending_snapshot = (
            not capturing_pending
            and getattr(self, "_pending_load_mode", None) == mode
            and getattr(self, "_pending_load_options", None) is not None
        )
        if has_pending_snapshot:
            options = self._pending_load_options
            self._pending_load_mode = None
            self._pending_load_options = None
            self._invalidated_load_modes.discard(mode)
            self._launch_load_options(options)
            return
        prepared = getattr(self, '_power_prevalidated_load', None)
        self._power_prevalidated_load = None
        if mode == 'Power Dependent' and prepared is not None and not capturing_pending and not has_pending_snapshot:
            folder, key, result = prepared
            if folder == self.current_folder and key == self.power_controller._power_selected_group_key():
                loaded = LoadedState(mode=mode, folder=folder,
                    primary_file=result.records[0].file_name if result.records else None,
                    selected_files=list(dict.fromkeys(r.file_name for r in result.records)),
                    cube=result.cube, power_records=result.records,
                    power_groups=result.groups, power_group_key=key,
                    power_results={key: result})
                trackers = getattr(self, "_load_requests", None)
                if trackers is None:
                    trackers = {}
                    self._load_requests = trackers
                tracker = trackers.setdefault(mode, LatestRequest())
                token, should_start = tracker.submit(mode, str(folder).casefold(), (folder, key))
                if not should_start:
                    self._pending_load_mode = mode
                    return
                self._load_lifecycle_generation = getattr(self, "_load_lifecycle_generation", 0) + 1
                self._load_in_progress = True
                self._active_load_mode = mode
                self._active_load_token = token
                self._active_load_succeeded = False
                recorder = getattr(self, "_record_workflow_event", None)
                if callable(recorder):
                    recorder(workflow=mode, request_id=token.generation, phase="enqueue", source_key=(key,))
                self._set_stage('Loading...')
                def finish_prevalidated_load():
                    if self._is_closing:
                        return
                    import logging
                    logger = logging.getLogger('dptk')
                    try:
                        if self.current_folder == folder:
                            logger.info('Applying validated Power group on GUI thread: %s', key)
                            self._on_loaded(loaded, request_token=token)
                            logger.info('Validated Power group applied: %s', key)
                    except Exception as exc:
                        import traceback
                        self._show_error(f'{exc}\n\n{traceback.format_exc()}')
                    finally:
                        self._on_load_finished(request_token=token)
                # Leave the modal picker callback before touching plot widgets.
                QTimer.singleShot(0, self, finish_prevalidated_load)
                return
        if not self.current_folder:
            self._show_error("Choose a folder first.")
            return
        if mode == "MCD" and not self.mcd_controller._mcd_angles_ready():
            self.mcd_controller._request_mcd_load()
            return
        self._invalidate_export_move_sources()
        compare_sources: Dict[str, str] = {}
        power_group_key = ""
        shg_settings: ShgSettings | None = None
        shg_fit_settings: ShgFitSettings | None = None
        shg_compare = False
        mcd_settings: McdSettings | None = None
        drr_assignments: tuple[DrrMeasurementAssignment, ...] = ()

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
                "Last frame from each file, then average": "last",
                "First frame from each file, then average": "first",
                "Average all frames in each file, then average files": "all",
            }
            drr_baseline_which = which_map.get(self.drr_baseline_combine_combo.currentText(), "last")
            y_axis_spec = self._selected_y_axis_spec("drr")
            power_group_key = ""
            # Keep the existing automatic-unresolved branch reachable when a
            # measurement itself is stale; its resolver supplies the precise
            # saved-recipe diagnostic. Explicit baselines (and map inputs,
            # which have no background-resolution step) must be rejected now.
            required = list(selected) if (
                not self._drr_assignments_automatic
                or not selected
                or data_io.is_xlsx_map_file(selected[0])
            ) else []
            if not self._drr_assignments_automatic:
                required.extend(baselines)
            for assignment in getattr(self, "_drr_assignments", ()):
                required.extend(getattr(assignment, "baseline_files", ()) or ())
            missing = self.drr_controller._drr_missing_sources(required)
            if missing:
                self._invalidate_drr_for_background_selection(
                    "Missing DRR source(s): " + ", ".join(missing)
                )
                return
            if selected and not data_io.is_xlsx_map_file(selected[0]):
                if self._drr_assignments_automatic and not self.drr_pin_baseline_chk.isChecked():
                    drr_assignments = tuple(self._drr_assignments)
                    # The combo boxes show a readable summary only.  Keep the
                    # actual per-measurement recipes authoritative even when
                    # their display union is non-empty.
                    drr_baseline = "Automatic"
                    if drr_assignments:
                        resolved = None
                    else:
                        resolved = resolve_drr_background_assignments(
                            self.current_folder, self.drr_available_sources, selected,
                        )
                elif not baselines and self._drr_baseline_user_selected and drr_baseline in {"Self (first frame)", "Self (last frame)"}:
                    resolved = resolve_drr_background_assignments(
                        self.current_folder, self.drr_available_sources, selected,
                        explicit_baseline_mode=drr_baseline,
                        explicit_baseline_which=drr_baseline_which,
                    )
                elif baselines:
                    resolved = resolve_drr_background_assignments(
                        self.current_folder, self.drr_available_sources, selected,
                        explicit_baseline_files=baselines,
                        explicit_baseline_mode="External",
                        explicit_baseline_which=drr_baseline_which,
                    )
                else:
                    resolved = resolve_drr_background_assignments(
                        self.current_folder, self.drr_available_sources, selected,
                    )
                    if resolved.self_fallback_allowed and drr_baseline in {"Self (first frame)", "Self (last frame)"}:
                        # Keep saved/validated recipes when available. Otherwise
                        # the visible default Self is a usable recipe, even if
                        # no currentTextChanged signal has fired yet.
                        resolved = resolve_drr_background_assignments(
                            self.current_folder, self.drr_available_sources, selected,
                            explicit_baseline_mode=drr_baseline,
                            explicit_baseline_which=drr_baseline_which,
                        )
                        self._drr_baseline_user_selected = True
                if resolved is None:
                    pass
                elif resolved.resolved:
                    drr_assignments = resolved.assignments
                    self._drr_assignments = drr_assignments
                    self._drr_assignments_automatic = not baselines and not self._drr_baseline_user_selected
                    if self._drr_assignments_automatic:
                        drr_baseline = "Automatic"
                        drr_baseline_which = resolved.assignments[0].baseline_which
                elif resolved is not None and not resolved.resolved and self._drr_assignments_automatic:
                    self.drr_baseline_files_manual = []
                    self.drr_baseline_files_found = []
                    self._invalidate_drr_for_background_selection(
                        resolved.reason or "DRR background assignments are unresolved."
                    )
                    return
                elif baselines or drr_baseline in {"External", "Self (last frame)", "Self (first frame)"}:
                    self._invalidate_drr_for_background_selection(resolved.reason)
                    return
            # Resolution can materialize baseline paths from saved metadata or
            # automatic matching. Validate those resulting identities before
            # constructing a worker; the pre-resolver guard cannot see them.
            resolved_required = list(selected)
            resolved_required.extend(baselines)
            for assignment in drr_assignments:
                resolved_required.extend(getattr(assignment, "baseline_files", ()) or ())
            missing = self.drr_controller._drr_missing_sources(resolved_required)
            if missing:
                self._invalidate_drr_for_background_selection(
                    "Missing DRR source(s): " + ", ".join(missing)
                )
                return
            if drr_baseline == "External" and not baselines:
                self._invalidate_drr_for_background_selection(
                    "Select an external background before processing."
                )
                return
        elif mode == "Compare":
            selection = self.compare_controller._cmp_selection_from_ui()
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
            mcd_settings = self.mcd_controller._mcd_settings_from_ui()
        elif mode == "SHG Processing":
            shg_settings = self.shg_controller._shg_settings_from_ui()
            shg_fit_settings = self.shg_controller._shg_fit_settings_from_ui()
            shg_compare = self.shg_controller._shg_compare_mode()
            if shg_compare:
                reference_file, sample_file = self.shg_controller._shg_compare_files()
                if reference_file and sample_file and reference_file == sample_file:
                    self._show_error("Select two different SHG files for twist-angle comparison.")
                    return
                selected = [name for name in (reference_file, sample_file) if name]
                if shg_settings.background_method == "external":
                    backgrounds = self.shg_controller._shg_compare_background_files()
                    if not all(backgrounds):
                        self._show_error("External SHG comparison requires a background file for A and B.")
                        return
                    baselines = list(backgrounds)
                else:
                    baselines = []
            else:
                source_file = self.shg_controller._shg_selected_file()
                background_file = (
                    self.shg_controller._shg_background_file()
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
            selected = self.power_controller._power_candidate_files()
            compare_sources = {}
            baselines = []
            pl_log = False
            cmp_log = False
            drr_baseline = "Self (last frame)"
            drr_baseline_which = "last"
            y_axis_spec = "auto"
            if self.power_compare_chk.isChecked():
                if not self.power_controller._power_has_distinct_role_groups():
                    self._show_error("Assign distinct KK and KKp power sweeps before loading comparison.")
                    return
                power_group_key = self.power_controller._power_role_group_key("KK")
            else:
                power_group_key = self.power_controller._power_selected_group_key()
                if not power_group_key:
                    self._show_error("Select a power sweep before loading.")
                    return
        if mode == "PL":
            power_group_key = ""
        if mode != "Compare":
            compare_sources = {}
        if mode == "MCD" and selected and self.mcd_controller._mcd_source_needs_stability_wait(selected[0]):
            return

        drr_selection_metadata: dict[str, Any] = {}
        if mode == "DRR" and drr_assignments:
            drr_selection_metadata = {
                "selection_method": "automatic_per_measurement" if self._drr_assignments_automatic else "explicit_per_measurement",
                "numerical_path": "pending",
                "drr_background_assignments": [item.to_dict() for item in drr_assignments],
            }
        pending_options = (
            None
            if capturing_pending or self._pending_load_mode != mode
            else self._pending_load_options
        )
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
            power_role_group_keys=(
                tuple(dict.fromkeys((
                    self.power_controller._power_role_group_key("KK"),
                    self.power_controller._power_role_group_key("KKp"),
                )))
                if mode == "Power Dependent" and self.power_compare_chk.isChecked()
                else ()
            ),
            shg_settings=shg_settings,
            shg_fit_settings=shg_fit_settings,
            shg_compare=shg_compare,
            mcd_settings=mcd_settings,
            mcd_history_roots=self._mcd_history_search_roots(self.current_folder) if mode == "MCD" else (),
            mcd_candidate_width_mev=(
                float(self.mcd_window_width_spin.value()) if mode == "MCD" else 5.0
            ),
            mcd_candidate_metric=(self.mcd_controller._mcd_window_metric() if mode == "MCD" else "mean"),
            mcd_candidate_energy_range=(
                tuple(sorted((
                    float(self.mcd_spins["xmin"].value()),
                    float(self.mcd_spins["xmax"].value()),
                )))
                if mode == "MCD" and self.mcd_spins["xmax"].value() > self.mcd_spins["xmin"].value()
                else None
            ),
            drr_background_selection=(
                {
                    "selection_method": "automatic_raw_spectrum_overlap",
                    "confidence": self._drr_background_guess.confidence,
                    "candidate_group_count": self._drr_background_guess.candidate_group_count,
                    "time_separation_seconds": self._drr_background_guess.time_gap_seconds,
                    "median_intensity_difference_percent": self._drr_background_guess.intensity_difference_percent,
                    "shape_correlation": self._drr_background_guess.shape_correlation,
                    "points_within_5_percent": self._drr_background_guess.points_within_tolerance_percent,
                    "exact_spectral_grid_required": True,
                    "intensity_adjustment": "none",
                    "reason": self._drr_background_guess.reason,
                }
                if mode == "DRR"
                and self._drr_background_guess is not None
                and tuple(baselines) == tuple(self._drr_background_guess.baseline_files)
                else {}
            ),
            drr_assignments=drr_assignments,
        )
        if drr_selection_metadata:
            options = LoadOptions(**{
                **options.__dict__,
                "drr_background_selection": {
                    **dict(options.drr_background_selection),
                    **drr_selection_metadata,
                },
            })

        if pending_options is not None:
            options = pending_options
            self._pending_load_mode = None
            self._pending_load_options = None
            self._invalidated_load_modes.discard(mode)

        if capturing_pending:
            self._pending_load_mode = mode
            self._pending_load_options = options
            return

        self._set_stage("Loading...")
        trackers = getattr(self, "_load_requests", None)
        if trackers is None:
            trackers = {}
            self._load_requests = trackers
        tracker = trackers.setdefault(mode, LatestRequest())
        token, should_start = tracker.submit(mode, str(options.folder).casefold(), options)
        if not should_start:
            self._pending_load_mode = mode
            self._invalidated_load_modes = {mode}
            return
        self._load_lifecycle_generation = getattr(self, "_load_lifecycle_generation", 0) + 1
        self._load_in_progress = True
        self._active_load_mode = mode
        self._active_load_token = token
        self._active_load_succeeded = False
        recorder = getattr(self, "_record_workflow_event", None)
        if callable(recorder):
            recorder(workflow=mode, request_id=token.generation, phase="enqueue", source_key=tuple(options.selected_files))
        worker = Worker(self._load_task, options)
        worker.signals.log.connect(self._append_log)
        load_generation = self._mcd_source_generation if mode == "MCD" else None
        if load_generation is not None:
            worker.signals.setProperty("mcd_source_generation", load_generation)
        worker.signals.result.connect(
            lambda loaded, token=token, generation=load_generation:
            self._on_loaded(loaded, generation, request_token=token)
        )
        worker.signals.error.connect(
            lambda message, token=token: self._on_load_error(message, token)
        )
        worker.signals.finished.connect(
            lambda token=token: self._on_load_finished(request_token=token)
        )
        self.thread_pool.start(worker)

    def _drr_provenance_records(
        self, folder: str, selected: Sequence[str], baselines: Sequence[str]
    ) -> tuple[WorkingCopyRecord, ...]:
        records: list[WorkingCopyRecord] = []
        pairs = [("measurement", name) for name in selected]
        pairs.extend(("background", name) for name in baselines)
        for role, name in pairs:
            records.append(
                verify_initial_data_working_file(name, folder, workflow="DRR", role=role)
            )
        return tuple(records)

    def _provenance_records_for_load(self, options: LoadOptions) -> tuple[WorkingCopyRecord, ...]:
        if options.mode not in {"PL", "DRR", "Compare"}:
            return ()
        pairs: list[tuple[str, str]] = []
        if options.mode == "PL":
            pairs = [("measurement", options.selected_files[0])] if options.selected_files else []
        elif options.mode == "DRR":
            return self._drr_provenance_records(
                options.folder, options.selected_files, options.baseline_files
            )
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
            self.drr_controller._reject_mixed_xlsx_selection(options.selected_files)
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
            assignments = tuple(options.drr_assignments)
            if not assignments and options.selected_files:
                explicit_mode = options.drr_baseline_text
                if explicit_mode == "Automatic":
                    resolved = resolve_drr_background_assignments(
                        folder, discover_drr_sources(folder), options.selected_files,
                    )
                else:
                    resolved = resolve_drr_background_assignments(
                        folder, discover_drr_sources(folder), options.selected_files,
                        explicit_baseline_files=options.baseline_files,
                        explicit_baseline_mode=explicit_mode,
                        explicit_baseline_which=options.drr_baseline_which,
                    )
                if not resolved.resolved:
                    raise ValueError(resolved.reason or "DRR background assignment is unresolved.")
                assignments = tuple(resolved.assignments)
            if assignments:
                cube = data_io.load_drr_resolved_cube(
                    folder, options.selected_files, assignments,
                    y_axis=options.y_axis_spec, derivative=None,
                )
                numerical_path = getattr(cube, "drr_numerical_path", "common")
                mode_label = (
                    "DR/R Automatic" if numerical_path != "common"
                    else "DR/R Self" if assignments[0].baseline_mode.startswith("Self")
                    else "DR/R External"
                )
                return LoadedState(
                    mode="DRR",
                    folder=folder,
                    primary_file=options.selected_files[0],
                    selected_files=options.selected_files,
                    baseline_files=list(dict.fromkeys(
                        source for assignment in assignments
                        for source in assignment.baseline_files
                    )),
                    cube=cube,
                    drr_mode_label=mode_label,
                    drr_derivative_label="None",
                    drr_baseline_text=options.drr_baseline_text,
                    drr_baseline_which=options.drr_baseline_which,
                    drr_background_selection={
                        **dict(options.drr_background_selection),
                        "numerical_path": numerical_path,
                    },
                    drr_assignments=assignments,
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
                drr_background_selection={
                    **dict(options.drr_background_selection),
                    "selection_method": (
                        "automatic_per_measurement"
                        if options.drr_baseline_text == "Automatic"
                        else "explicit_per_measurement"
                    ),
                    "numerical_path": getattr(cube, "drr_numerical_path", "common"),
                    "drr_background_assignments": [item.to_dict() for item in assignments],
                },
                drr_assignments=assignments,
                y_axis_spec=options.y_axis_spec,
                provenance_records=provenance_records,
            )

        if mode == "MCD":
            source_path = resolve_source_path(folder, options.selected_files[0])
            from core.mcd_center_history import read_center_history
            history = read_center_history(folder, options.selected_files[0],
                extra_roots=options.mcd_history_roots)
            settings = options.mcd_settings or McdSettings()
            if settings.dark_pos_file:
                settings = McdSettings(**{**settings.__dict__, "dark_pos_file": str(resolve_source_path(folder, settings.dark_pos_file))})
            if settings.dark_neg_file:
                settings = McdSettings(**{**settings.__dict__, "dark_neg_file": str(resolve_source_path(folder, settings.dark_neg_file))})
            result, cache_key = self.mcd_controller._cached_mcd_result(source_path, settings)
            cache_hit = result is not None
            if result is None:
                result = process_mcd(str(source_path), settings)
                self.mcd_controller._store_cached_mcd_result(cache_key, result)
            else:
                log.emit(f"Reusing cached MCD processing for {source_path.name}.")
            candidate_range = options.mcd_candidate_energy_range
            if candidate_range is not None:
                low, high = sorted((float(candidate_range[0]), float(candidate_range[1])))
                energy_low = float(np.nanmin(result.energy_ev))
                energy_high = float(np.nanmax(result.energy_ev))
                if high <= energy_low or low >= energy_high:
                    candidate_range = None
            candidates = suggest_mcd_window_centers(
                result,
                float(options.mcd_candidate_width_mev),
                metric=options.mcd_candidate_metric,
                energy_range=candidate_range,
                max_candidates=5,
            )
            return LoadedState(
                mode="MCD", folder=folder, primary_file=options.selected_files[0],
                selected_files=options.selected_files, mcd_result=result, mcd_settings=settings,
                mcd_center_candidates=candidates,
                mcd_center_history=tuple(history),
                mcd_candidate_search_range=candidate_range,
                mcd_cache_hit=cache_hit,
            )

        if mode == "Power Dependent":
            result = data_io.load_power_series_cube(
                folder,
                options.selected_files,
                group_key=options.power_group_key,
                y_axis="auto",
            )
            power_results = {result.group_key: result}
            for role_key in options.power_role_group_keys:
                if role_key and role_key not in power_results:
                    power_results[role_key] = data_io.load_power_series_cube(
                        folder, options.selected_files, group_key=role_key, y_axis="auto"
                    )
            return LoadedState(
                mode="Power Dependent",
                folder=folder,
                primary_file=(result.records[0].file_name if result.records else None),
                selected_files=list(dict.fromkeys(record.file_name for record in result.records)),
                cube=result.cube,
                power_records=result.records,
                power_groups=result.groups,
                power_results=power_results,
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
                shg_settings=settings,
                shg_fit_settings=fit_settings,
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

    def _on_load_error(self, message: str, request_token: RequestToken | None = None) -> None:
        if request_token is not None:
            mode = request_token.workflow
            tracker = self._load_requests.get(mode)
            if tracker is None or not tracker.accepts(request_token, str(self.current_folder).casefold()):
                return
            if mode in self._invalidated_load_modes:
                return
        self._show_error(message)

    def _on_loaded(self, loaded: LoadedState, load_generation: int | None = None,
                   *, request_token: RequestToken | None = None) -> None:
        if request_token is not None:
            tracker = self._load_requests.get(request_token.workflow)
            if (
                tracker is None
                or request_token != self._active_load_token
                or not tracker.accepts(request_token, str(self.current_folder).casefold())
                or request_token.workflow in self._invalidated_load_modes
            ):
                return
        if loaded.mode == "MCD" and load_generation is None:
            sender = self.sender()
            if sender is not None:
                sender_generation = sender.property("mcd_source_generation")
                if sender_generation is not None:
                    load_generation = int(sender_generation)
        if (
            loaded.mode == "MCD"
            and load_generation is not None
            and load_generation != self._mcd_source_generation
        ):
            return
        if loaded.mode == self._active_load_mode:
            self._active_load_succeeded = True
        self.loaded = loaded
        if loaded.mode == "DRR":
            self._drr_assignments = tuple(getattr(loaded, "drr_assignments", ()))
            self._drr_assignments_automatic = (
                str(getattr(loaded, "drr_background_selection", {}).get("selection_method", ""))
                == "automatic_per_measurement"
            )
        self._drr_derivative_cache.clear()
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
            self.pl_controller._on_pl_dat_y_axis_changed(choice)
        if loaded.mode == "Power Dependent":
            self._power_result_cache.update(getattr(loaded, "power_results", {}))
            self.power_controller._power_refresh_groups()
            idx = self.power_group_combo.findData(loaded.power_group_key)
            if idx >= 0 and not self.power_compare_chk.isChecked():
                self.power_group_combo.setCurrentIndex(idx)
        if loaded.mode == "MCD" and loaded.mcd_result is not None:
            self.mcd_controller._clear_mcd_center_candidates(restore_manual=False)
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
            source_summary = (
                f"{loaded.mcd_result.summary['pairs']} angle pairs; sigma+ {format_mcd_angle(loaded.mcd_result.pos_angle)} deg; "
                f"sigma- {format_mcd_angle(loaded.mcd_result.neg_angle)} deg; {reference_text}."
            )
            self.mcd_source_summary.setText(source_summary)
            self.mcd_source_summary.setToolTip(source_summary)
            self.mcd_controller._update_mcd_background_preview()
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
            self.shg_controller._shg_refresh_sources()
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
            self.shg_controller._shg_update_summary()
        self._apply_auto_limits_for_loaded()
        if loaded.mode == "MCD" and loaded.mcd_result is not None:
            # Candidate scoring was completed beside CSV processing in the
            # worker. Publishing its small result here avoids blocking the UI
            # between load completion and the first plot.
            self.mcd_controller._prepare_mcd_center_for_loaded_energy()
            self._mcd_manual_center_before_suggestions = float(
                self.mcd_window_center_spin.value()
            )
            self._mcd_center_candidates = tuple(loaded.mcd_center_candidates)
            self._mcd_candidate_active_index = None
            self._mcd_candidate_search_range = loaded.mcd_candidate_search_range
            self.mcd_controller._update_mcd_candidate_bar()
            self._update_mcd_peak_shift_source(loaded.mcd_result)
        elif hasattr(self, "mcd_peak_source_summary"):
            self._update_mcd_peak_shift_source(None)
        self._set_stage("Loaded")
        self._update_action_states()
        self._update_plot_view_bar_visibility()
        self._status(f"Loaded {loaded.mode}.")
        current_tab = self.tabs.tabText(self.tabs.currentIndex())
        plot_mode = "MCD Peak Shift" if loaded.mode == "MCD" and current_tab == "MCD Peak Shift" else loaded.mode
        # A worker may finish after the user switched pages.  Publishing the
        # small LoadedState is useful, but only the active page may redraw the
        # shared canvas.
        active_mode = self._active_mode()
        if request_token is None or active_mode == plot_mode:
            self._plot_mode(plot_mode, auto=True)
        else:
            self._status(f"Loaded {loaded.mode}; switch to its page to display it.")
        if plot_mode == "MCD Peak Shift" and loaded.mcd_result is not None:
            self._ensure_mcd_peak_analysis()
        if (
            loaded.mode == "MCD"
            and loaded.mcd_result is not None
            and self.last_plotted_mode == "MCD"
        ):
            # Give the freshly rendered MCD view a short settling window so
            # the first center trace refresh is not consumed during plotting.
            self.mcd_controller._mcd_center_refresh_timer.start(200)

    def _on_load_finished(self, load_generation: int | None = None,
                          *, request_token: RequestToken | None = None) -> None:
        if request_token is not None and request_token != self._active_load_token:
            return
        finished_mode = self._active_load_mode
        succeeded = self._active_load_succeeded
        self._load_in_progress = False
        self._data_state_loading = False
        self._active_load_mode = None
        self._active_load_token = None
        self._active_load_succeeded = False
        self._status_progress.setVisible(False)
        if not succeeded and (finished_mode or getattr(self, "_data_state_error", None)):
            self._data_state_error = "failed"
            self._refresh_data_state_label()
        elif succeeded:
            self._data_state_error = None
            self._refresh_data_state_label()
        if request_token is not None and finished_mode:
            tracker = self._load_requests.get(finished_mode)
            if tracker is not None:
                tracker.finish(request_token)
            pending_mode = self._pending_load_mode
            pending_options = self._pending_load_options
            if pending_mode is not None and not self._is_closing:
                self._pending_load_mode = None
                self._pending_load_options = None
                self._invalidated_load_modes.clear()
                if pending_options is not None:
                    launch_generation = getattr(self, "_load_lifecycle_generation", 0)
                    launch_folder = str(getattr(pending_options, "folder", getattr(self, "current_folder", "")))
                    QTimer.singleShot(
                        0, self,
                        lambda options=pending_options, generation=launch_generation, folder=launch_folder:
                        MainWindow._launch_pending_if_current(self, options, generation, folder),
                    )
                else:
                    QTimer.singleShot(0, self, lambda mode=pending_mode: self._start_load(mode))
            elif pending_mode is None:
                self._invalidated_load_modes.clear()
        if finished_mode == "PL" and self._pl_auto_next_active:
            if succeeded:
                self._pl_auto_next_active = False
                self._pl_auto_next_queue = []
            elif self._pl_auto_next_queue:
                self.pl_controller._load_next_pl_from_queue()
            else:
                self._pl_auto_next_active = False
                self._status("No valid unprocessed PL measurements remain.")
        if finished_mode == "MCD":
            if self.mcd_controller._mcd_reapply_pending:
                self.mcd_controller._mcd_reapply_pending = False
                self.mcd_apply_correction_btn.setText("Pending update...")
                self.mcd_controller._mcd_auto_apply_timer.start(0)
            elif self.mcd_controller._mcd_auto_apply_timer.isActive():
                self.mcd_apply_correction_btn.setText("Pending update...")
            elif succeeded:
                self.mcd_apply_correction_btn.setText("Up to date")
            else:
                self.mcd_apply_correction_btn.setText("Recalculate now")
            if self._mcd_reload_pending:
                self._mcd_reload_pending = False
                self._status("Loading the newly selected MCD source...")
                self.mcd_controller._request_mcd_load()
            elif succeeded and self.tabs.tabText(self.tabs.currentIndex()) == "MCD Peak Shift":
                self._ensure_mcd_peak_analysis()
            elif succeeded and self.tabs.tabText(self.tabs.currentIndex()) == "MCD":
                self._mcd_unified_track_generation += 1
                self._mcd_unified_track_pending = None
                self._mcd_unified_track_payload = None
                self._mcd_unified_track_cache.clear()
                self._mcd_unified_analysis_pending = False
                if self._unified_catalog_needed():
                    self._queue_mcd_unified_analysis()

    def _split_prefix_mode(self, prefix: str) -> str:
        return {
            "pl": "PL",
            "drr": "DRR",
            "drr_second": "DRR",
            "cmp": "Compare",
            "power": "Power Dependent",
        }[prefix]

    def _on_split_scale_toggled(self, prefix: str, checked: bool) -> None:
        panel: QWidget = getattr(self, f"{prefix}_split_scale_panel")
        panel.setVisible(bool(checked))
        spins = self._mode_spins(self._split_prefix_mode(prefix))
        # The d2E product has its own split ranges.  Toggling them must not
        # disable the raw DRR limits shown in the row above.
        if prefix != "drr_second":
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
        if checked and prefix == "drr_second":
            self._set_spin_value_silent(
                split_spins["x0"], float(self.drr_split_spins["x0"].value())
            )
            self._refresh_drr_second_split_ranges(center_split=True)
        elif checked:
            if prefix == "drr":
                second_toggle = self.drr_second_split_scale_chk
                blocked = second_toggle.blockSignals(True)
                second_toggle.setChecked(True)
                second_toggle.blockSignals(blocked)
                self.drr_second_split_scale_panel.setVisible(True)
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
        elif prefix != "drr_second":
            if prefix == "drr":
                second_toggle = self.drr_second_split_scale_chk
                blocked = second_toggle.blockSignals(True)
                second_toggle.setChecked(False)
                second_toggle.blockSignals(blocked)
                self.drr_second_split_scale_panel.setVisible(False)
            self._refresh_automatic_ranges(
                self._split_prefix_mode(prefix), refresh_split=False
            )
        if prefix == "pl":
            self.pl_controller._on_pl_plot_param_changed()
        elif prefix in {"drr", "drr_second"}:
            self.drr_controller._on_drr_plot_param_changed()
        elif prefix == "cmp":
            self.compare_controller._on_cmp_plot_param_changed()
        else:
            self.power_controller._on_power_plot_param_changed()
        self._update_action_states()

    def _on_split_scale_param_changed(self, prefix: str) -> None:
        toggle: QCheckBox = getattr(self, f"{prefix}_split_scale_chk")
        if not toggle.isChecked():
            return
        if prefix == "drr_second":
            self._set_spin_value_silent(
                self.drr_split_spins["x0"], float(self.drr_second_split_spins["x0"].value())
            )
            self._refresh_drr_second_split_ranges()
        else:
            self._refresh_automatic_ranges(
                self._split_prefix_mode(prefix), refresh_split=True
            )
        if prefix == "pl":
            self.pl_controller._on_pl_plot_param_changed()
        elif prefix in {"drr", "drr_second"}:
            self.drr_controller._on_drr_plot_param_changed()
        elif prefix == "cmp":
            self.compare_controller._on_cmp_plot_param_changed()
        else:
            self.power_controller._on_power_plot_param_changed()

    def _split_scale_for_mode(self, mode: str) -> SplitColorScale | None:
        if mode == "MCD":
            return None
        # VP views use a fixed diverging [-1, 1] scale, so a remembered
        # intensity split must not be validated or applied while they are active.
        if mode == "Compare" and self.compare_controller._cmp_is_vp_view():
            return None
        if mode == "Power Dependent" and self.power_controller._power_view() == "VP":
            return None
        prefix = "pl" if mode == "PL" else "drr" if mode == "DRR" else "power" if mode == "Power Dependent" else "cmp"
        return self._split_scale_for_prefix(prefix, mode)

    def _split_scale_for_prefix(self, prefix: str, mode: str = "DRR") -> SplitColorScale | None:
        """Build a validated split scale for one product's controls."""
        toggle: QCheckBox = getattr(self, f"{prefix}_split_scale_chk")
        enabled = toggle.isChecked() or (
            prefix == "drr_second"
            and getattr(self, "drr_split_scale_chk", toggle).isChecked()
        )
        if not enabled:
            return None
        spins = self._mode_spins(mode)
        split_spins: Dict[str, QDoubleSpinBox] = getattr(self, f"{prefix}_split_spins")
        xmin, xmax = sorted((float(spins["xmin"].value()), float(spins["xmax"].value())))
        shared_boundary = prefix == "drr_second" and hasattr(self, "drr_split_spins")
        x0 = float(self.drr_split_spins["x0"].value()) if shared_boundary else float(split_spins["x0"].value())
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
            show_boundary=bool(
                getattr(
                    self,
                    "drr_split_boundary_chk" if shared_boundary else f"{prefix}_split_boundary_chk",
                ).isChecked()
            ),
        )

    def _split_scale_key(self, prefix: str) -> tuple[object, ...]:
        toggle = getattr(self, f"{prefix}_split_scale_chk")
        enabled = bool(toggle.isChecked()) or (
            prefix == "drr_second"
            and bool(getattr(self, "drr_split_scale_chk", toggle).isChecked())
        )
        if not enabled:
            return (False,)
        spins: Dict[str, QDoubleSpinBox] = getattr(self, f"{prefix}_split_spins")
        shared_boundary = prefix == "drr_second" and hasattr(self, "drr_split_spins")
        shared_spins = self.drr_split_spins if shared_boundary else spins
        return (
            True,
            float(shared_spins["x0"].value()),
            float(spins["left_vmin"].value()),
            float(spins["left_vmax"].value()),
            float(spins["right_vmin"].value()),
            float(spins["right_vmax"].value()),
            bool(
                getattr(
                    self,
                    "drr_split_boundary_chk" if shared_boundary else f"{prefix}_split_boundary_chk",
                ).isChecked()
            ),
        )

    def _split_auto_cubes(self, mode: str, prefix: str | None = None) -> list[DataCube]:
        if not self.loaded or self.loaded.mode != mode:
            return []
        if mode == "PL" and self.loaded.cube is not None:
            return [self.loaded.cube]
        if mode == "DRR" and self.loaded.cube is not None:
            try:
                if prefix == "drr_second":
                    return [self.drr_controller._drr_cube_with_metadata(2)[0]]
                return [self.drr_controller._drr_cube_for_display()]
            except (TypeError, ValueError):
                return []
        if mode == "Compare" and self.loaded.compare_cubes:
            if self.compare_controller._cmp_is_vp_view():
                return []
            raw = {
                key: self.loaded.compare_cubes[key]
                for key in self.compare_controller._cmp_visible_channels()
                if key in self.loaded.compare_cubes
            }
            return list(
                self.compare_controller._cmp_corrected_cubes(
                    raw,
                    self.compare_controller._cmp_source_mapping(),
                    background=self.compare_controller._cmp_background_value(raw),
                ).values()
            )
        if mode == "Power Dependent" and self.loaded.cube is not None:
            if self.power_controller._power_view() == "VP":
                return []
            if self.power_controller._power_has_distinct_role_groups():
                kk_result, kkp_result, _kk_key, _kkp_key = self.power_controller._power_role_payload()
                background = self.power_controller._power_background_value([kk_result.cube, kkp_result.cube])
                return [
                    self.power_controller._power_corrected_cube(kk_result.cube, background=background),
                    self.power_controller._power_corrected_cube(kkp_result.cube, background=background),
                ]
            return [self.power_controller._power_corrected_cube(self.loaded.cube)]
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
        if (
            mode == "Compare"
            and self.loaded
            and self.loaded.mode == "Compare"
            and self.loaded.compare_cubes
            and self.compare_controller._cmp_is_vp_view()
        ):
            raw = {
                key: self.loaded.compare_cubes[key]
                for key in ("KK", "KKp")
                if key in self.loaded.compare_cubes
            }
            if len(raw) == 2:
                try:
                    background = self.compare_controller._cmp_background_value(raw)
                    return [self.compare_controller._cmp_vp_cube(
                        raw,
                        self.compare_controller._cmp_source_mapping(),
                        background=background,
                    )]
                except (KeyError, ValueError):
                    pass
        return self._split_auto_cubes(mode)

    def _refresh_drr_second_split_ranges(self, *, center_split: bool = False) -> bool:
        """Refresh unlocked d2E split limits from the current derivative cube."""
        toggle = getattr(self, "drr_second_split_scale_chk", None)
        if toggle is None or (
            not toggle.isChecked()
            and not getattr(self, "drr_split_scale_chk", toggle).isChecked()
        ):
            return False
        if not self.loaded or self.loaded.mode != "DRR" or self.loaded.cube is None:
            return False
        try:
            second_cube = self.drr_controller._drr_cube_with_metadata(2)[0]
        except (TypeError, ValueError):
            return False
        spins = self.drr_spins
        split_spins: Dict[str, QDoubleSpinBox] = self.drr_second_split_spins
        split_fixes: Dict[str, QCheckBox] = self.drr_second_split_fix_checks
        xmin, xmax = sorted((float(spins["xmin"].value()), float(spins["xmax"].value())))
        x0 = float(self.drr_split_spins["x0"].value())
        self._set_spin_value_silent(split_spins["x0"], x0)
        changed = False
        if (center_split and not self.drr_split_fix_checks["x0"].isChecked()) or not xmin < x0 < xmax:
            requested = 0.5 * (xmin + xmax)
            try:
                _index, x0 = resolve_split_boundary(np.asarray(second_cube.energy, float), requested)
            except ValueError:
                x0 = requested
            self._set_spin_value_silent(split_spins["x0"], x0)
            self._set_spin_value_silent(self.drr_split_spins["x0"], x0)
            changed = True
        # Seed a newly enabled panel from the current whole d2E limits.  This
        # keeps the first toggle valid while the percentile refresh below
        # still supplies independent values for each region.
        for side in ("left", "right"):
            vmin_key, vmax_key = f"{side}_vmin", f"{side}_vmax"
            if split_spins[vmax_key].value() <= split_spins[vmin_key].value():
                if not split_fixes[vmin_key].isChecked():
                    self._set_spin_value_silent(split_spins[vmin_key], float(self.drr_second_vmin_spin.value()))
                if not split_fixes[vmax_key].isChecked():
                    self._set_spin_value_silent(split_spins[vmax_key], float(self.drr_second_vmax_spin.value()))
        for side in ("left", "right"):
            bounds = self._color_bounds_for_cubes(
                "DRR", [second_cube], side=side, split_x=x0
            )
            if bounds is None:
                continue
            vmin_key, vmax_key = f"{side}_vmin", f"{side}_vmax"
            candidate_min = float(split_spins[vmin_key].value()) if split_fixes[vmin_key].isChecked() else bounds[0]
            candidate_max = float(split_spins[vmax_key].value()) if split_fixes[vmax_key].isChecked() else bounds[1]
            if candidate_max <= candidate_min:
                continue
            if not split_fixes[vmin_key].isChecked():
                self._set_spin_value_silent(split_spins[vmin_key], candidate_min)
                changed = True
            if not split_fixes[vmax_key].isChecked():
                self._set_spin_value_silent(split_spins[vmax_key], candidate_max)
                changed = True
        return changed

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
        # Compare VP has an independent color range.  Automatic intensity
        # refreshes must not rewrite the remembered intensity limits while VP
        # is active (background and ROI changes can still request an axis
        # refresh).
        skip_compare_vp_color = (
            mode == "Compare"
            and hasattr(self, "compare_controller")
            and self.compare_controller._cmp_is_vp_view()
        )
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
                if not skip_compare_vp_color:
                    for key, value in zip(("vmin", "vmax"), bounds):
                        if not self._mode_fix_value(mode, key):
                            self._set_spin_value_silent(spins[key], value)
                            changed = True

            if mode == "MCD" or not refresh_split:
                return changed
            if skip_compare_vp_color:
                return changed
            prefix = "pl" if mode == "PL" else "drr" if mode == "DRR" else "power" if mode == "Power Dependent" else "cmp"
            if not bool(getattr(self, f"{prefix}_split_scale_chk").isChecked()):
                if mode == "DRR":
                    return changed or self._refresh_drr_second_split_ranges(center_split=center_split)
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
            if mode == "DRR":
                changed = self._refresh_drr_second_split_ranges(center_split=center_split) or changed
            return changed
        finally:
            self._automatic_range_update = False

    def _auto_split_vrange(self, prefix: str, side: str) -> None:
        mode = self._split_prefix_mode(prefix)
        cubes = self._split_auto_cubes(mode, prefix)
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
            if self.loaded.shg_compare != self.shg_controller._shg_compare_mode():
                raise ValueError("Press Load after changing the SHG Single/Compare workflow.")
            settings = self.shg_controller._shg_settings_from_ui()
            fit_settings = self.shg_controller._shg_fit_settings_from_ui()
            if (
                self.loaded.shg_settings == settings
                and self.loaded.shg_fit_settings == fit_settings
                and self.loaded.shg_result is not None
            ):
                return False
            # Processing is performed by the SHG controller's worker. Keep
            # the last valid result visible while the newest request runs.
            self.shg_controller._request_shg_reprocess()
            return False
        if mode == "MCD":
            # Processing settings are applied asynchronously by the MCD load
            # pipeline. Display-only changes must never re-read the CSV here.
            return False
        current_spec = self._current_y_axis_spec_for_mode(mode)
        if mode == "DRR" and current_spec == getattr(self.loaded, "y_axis_spec", "auto"):
            return self._ensure_loaded_matches_drr_params()
        if mode == "Power Dependent":
            desired_key = self.power_controller._power_selected_group_key()
            if self.power_compare_chk.isChecked() and not self.power_controller._power_has_distinct_role_groups():
                raise ValueError("Assign distinct KK and KKp power sweeps for comparison.")
            if not desired_key:
                raise ValueError("Select a power sweep before plotting.")
            if desired_key == self.loaded.power_group_key and list(self.loaded.selected_files):
                required_result_keys = [desired_key]
                if self.power_compare_chk.isChecked():
                    required_result_keys.append(self.power_controller._power_role_group_key("KKp"))
                missing_result_keys = [
                    key for key in required_result_keys
                    if key and key not in self._power_result_cache
                ]
                if missing_result_keys:
                    self._start_load(mode)
                    self._status("Power source data is updating in the background.")
                return False
            self._start_load(mode)
            self._status("Power source changed; waiting for the current load to finish.")
            return False
        if mode == "Compare":
            selection = self.compare_controller._cmp_selection_from_ui()
            desired_sources = selection.as_pairs()
            if (
                current_spec == getattr(self.loaded, "y_axis_spec", "auto")
                and desired_sources == (self.loaded.compare_sources or {})
            ):
                return False
        elif current_spec == getattr(self.loaded, "y_axis_spec", "auto"):
            return False

        # Source and processing changes are submitted through the worker
        # request path.  Redraws stay display-only and never read a CSV.
        self._start_load(mode)
        self._status(f"{mode} input changed; reloading in the background.")
        return False

        self._last_plot_cube = None
        self._last_plot_params_key = None
        self._apply_auto_limits_for_loaded()
        return True

    def _apply_auto_limits_for_loaded(self) -> None:
        if not self.loaded:
            return
        mode = self.loaded.mode
        if mode == "SHG Processing":
            self.shg_controller._shg_update_summary()
            return
        if mode == "PL" and self.loaded.cube is not None:
            cube = self.loaded.cube
        elif mode == "DRR" and self.loaded.cube is not None:
            cube = self.drr_controller._drr_cube_for_display()
        elif mode == "Compare" and self.loaded.compare_cubes:
            background = self.compare_controller._cmp_background_value(self.loaded.compare_cubes)
            corrected = self.compare_controller._cmp_corrected_cubes(self.loaded.compare_cubes, background=background)
            cube = next(iter(corrected.values()))
        elif mode == "Power Dependent" and self.loaded.cube is not None:
            if self.power_controller._power_view() == "VP":
                try:
                    _kk_cube, _kkp_cube, vp_cube, *_rest = self.power_controller._power_vp_payload()
                    cube = vp_cube
                except Exception:
                    cube = self.power_controller._power_corrected_cube(self.loaded.cube)
            else:
                cube = self.power_controller._power_corrected_cube(self.loaded.cube)
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
        if mode == "Power Dependent" and self.power_controller._power_axis_log():
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
            y_axis_log=(mode == "Power Dependent" and self.power_controller._power_axis_log()),
            center_zero=(bool(self.mcd_center_zero_chk.isChecked()) if mode == "MCD" else mode == "DRR" and bool(self.drr_center_zero_chk.isChecked())),
            clip_outliers=self._mode_clip(mode),
            split_scale=self._split_scale_for_mode(mode),
        )

    def _make_drr_params(self, cube: DataCube, derivative_order: int | None) -> HeatmapParams:
        """Build display parameters with a product-specific color range."""
        params = self._make_params("DRR", cube)
        if derivative_order == 2 and hasattr(self, "drr_second_vmin_spin"):
            params = HeatmapParams(
                **{
                    **params.__dict__,
                    "cmap": self._resolved_cmap(self.drr_second_cmap),
                    "vmin": float(self.drr_second_vmin_spin.value()),
                    "vmax": float(self.drr_second_vmax_spin.value()),
                    "split_scale": self._split_scale_for_prefix("drr_second", "DRR"),
                }
            )
        return params

    def _drr_params_with_view_limits(self, params: HeatmapParams) -> HeatmapParams:
        limits = self._drr_view_limits
        if limits is None:
            return params
        return HeatmapParams(**{**params.__dict__, "xlim": limits[0], "ylim": limits[1]})

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
        from matplotlib.colors import LogNorm
        from matplotlib.ticker import LogFormatter
        from core.plotting import plain_log_ticks

        def numeric_log_labels(colorbar):
            if isinstance(colorbar.norm, LogNorm):
                colorbar.formatter = LogFormatter()
                colorbar.update_ticks()
                plain_log_ticks(colorbar.ax.xaxis if orientation == "horizontal" else colorbar.ax.yaxis)

        if not render.is_split:
            colorbar = self.figure.colorbar(render.primary, cax=cax, label=label, orientation=orientation)
            numeric_log_labels(colorbar)
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
        numeric_log_labels(left_cb)
        numeric_log_labels(right_cb)
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

    def _plot_mcd_peak_shift(self) -> None:
        """Render the compact spectrum/derivative inspection and result view."""
        self.last_plotted_mode = "MCD Peak Shift"
        self.mcd_controller._disable_mcd_blitting()
        self.figure.clear()
        show_maps = bool(self.mcd_peak_show_maps_chk.isChecked())
        show_derivative = bool(self.mcd_peak_show_derivative_chk.isChecked())
        if show_maps:
            gs = self.figure.add_gridspec(2, 2, width_ratios=[1.0, 2.0], height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.22)
            raw_ax = deriv_ax = None
            map_axes = [self.figure.add_subplot(gs[0, 0]), self.figure.add_subplot(gs[1, 0])]
            result_ax = self.figure.add_subplot(gs[:, 1])
        elif show_derivative:
            gs = self.figure.add_gridspec(2, 2, width_ratios=[1.0, 2.0], height_ratios=[2.0, 1.0], hspace=0.30, wspace=0.22)
            raw_ax = self.figure.add_subplot(gs[0, 0])
            deriv_ax = self.figure.add_subplot(gs[1, 0], sharex=raw_ax)
            map_axes = []
            result_ax = self.figure.add_subplot(gs[:, 1])
        else:
            gs = self.figure.add_gridspec(1, 2, width_ratios=[1.0, 2.0], wspace=0.22)
            raw_ax = self.figure.add_subplot(gs[0, 0])
            deriv_ax = None
            map_axes = []
            result_ax = self.figure.add_subplot(gs[0, 1])
        self.mcd_peak_spectrum_ax = raw_ax
        self.mcd_peak_derivative_ax = deriv_ax
        self.mcd_peak_shift_ax = result_ax
        self.mcd_peak_map_axes = map_axes
        self._mcd_peak_track_lines = []
        self._mcd_peak_candidate_artists = {}
        self._mcd_peak_manual_center_artists = {}
        self._mcd_peak_manual_center_blit_backgrounds = {}
        result = getattr(self, "mcd_peak_result", None)
        locator_catalog = getattr(self, "_mcd_peak_locator_results", {})
        if result is None and self.loaded is not None and self.loaded.mcd_result is not None and locator_catalog:
            # Locator results are available immediately, while the selected
            # local fit may still be running.  Keep the measured spectrum and
            # field controls visible instead of leaving the whole canvas blank.
            source = self.loaded.mcd_result
            energy_order = spectrum_energy_order(source)
            energy = np.sort(np.asarray(source.energy_ev, float))
            source_choice = str(self.mcd_peak_source_combo.currentText())
            source_prefix = "pair_corrected" if source_choice.casefold().startswith("mcd-corrected") else "pair_raw"
            spectra = {
                channel: np.asarray(getattr(source, f"{source_prefix}_{channel}"), float)[:, energy_order]
                for channel in ("pos", "neg")
            }
            source_label = "MCD-corrected R" if source_prefix == "pair_corrected" else "Raw R"
            angles = {"pos": float(getattr(source, "pos_angle", np.nan)), "neg": float(getattr(source, "neg_angle", np.nan))}
            fields = {}
            for channel in ("pos", "neg"):
                field = np.asarray(getattr(source, f"pair_b_{channel}", source.pair_b), float)
                interp = np.asarray(getattr(source, f"pair_interpolated_{channel}", np.zeros(field.size, dtype=bool)), bool)
                fields[channel] = np.where(interp, np.asarray(source.pair_b, float), field)
            if raw_ax is not None:
                index = self.mcd_peak_field_combo.currentData()
                index = 0 if index is None else int(index)
                index = max(0, min(index, int(np.asarray(source.pair_b).size) - 1))
                for channel, color in (("pos", "#1769aa"), ("neg", "#d1495b")):
                    raw_ax.plot(energy, spectra[channel][index], color=color, lw=0.9,
                                label=f"{format_mcd_angle(angles[channel])}° · B={fields[channel][index]:.4g} T")
                    if deriv_ax is not None:
                        grid, d2, _ = second_derivative_map(
                            energy, spectra[channel][index][None, :], int(self.mcd_peak_deriv_window_spin.value())
                        )
                        deriv_ax.plot(grid, -d2[0], color=color, lw=0.9)
                raw_ax.set_title(f"Selected field · {source_label}")
                raw_ax.set_xlabel("Energy (eV)"); raw_ax.set_ylabel(f"{source_label} intensity (a.u.)")
                raw_ax.grid(alpha=0.2); raw_ax.legend(fontsize=7, loc="upper left")
                raw_ax.set_xlim(float(np.nanmin(energy)), float(np.nanmax(energy)))
            if deriv_ax is not None:
                deriv_ax.set_title("−d²R/dE² · both channels", fontsize=9, pad=8)
                deriv_ax.set_xlabel("Energy (eV)"); deriv_ax.set_ylabel("−d²R/dE²")
                deriv_ax.grid(alpha=0.2); deriv_ax.set_xlim(float(np.nanmin(energy)), float(np.nanmax(energy)))
            for axis in map_axes:
                axis.axis("off")
            result_ax.axis("off")
            result_ax.text(0.05, 0.95, "Computing local mixed fit…\nLocator candidates are ready.", transform=result_ax.transAxes, va="top")
            self.canvas.draw_idle()
            return
        if result is None or self.loaded is None or self.loaded.mcd_result is None:
            self._mcd_peak_drag = None
            self.canvas.unsetCursor()
            empty_ax = raw_ax if raw_ax is not None else result_ax
            if raw_ax is not None:
                raw_ax.axis("off")
            if deriv_ax is not None:
                deriv_ax.axis("off")
            result_ax.axis("off")
            empty_ax.text(0.5, 0.5, "No peak-shift analysis yet.\nLoad an MCD result and click Analyze.", ha="center", va="center")
            self.canvas.draw_idle(); return
        source = self.loaded.mcd_result
        energy_order = spectrum_energy_order(source)
        energy = np.sort(np.asarray(source.energy_ev, float))
        source_choice = str(self.mcd_peak_source_combo.currentText())
        source_prefix = "pair_corrected" if source_choice.casefold().startswith("mcd-corrected") else "pair_raw"
        spectra = {
            channel: np.asarray(getattr(source, f"{source_prefix}_{channel}"), float)[:, energy_order]
            for channel in ("pos", "neg")
        }
        source_label = "MCD-corrected R" if source_prefix == "pair_corrected" else "Raw R"
        angles = {"pos": float(getattr(source, "pos_angle", np.nan)), "neg": float(getattr(source, "neg_angle", np.nan))}
        fields = {}
        for channel in ("pos", "neg"):
            field = np.asarray(getattr(source, f"pair_b_{channel}", source.pair_b), float)
            interp = np.asarray(getattr(source, f"pair_interpolated_{channel}", np.zeros(field.size, dtype=bool)), bool)
            fields[channel] = np.where(interp, np.asarray(source.pair_b, float), field)
        current_index = self.mcd_peak_field_combo.currentData()
        pair_exact_valid = False
        if current_index is not None:
            current_index = int(current_index)
            pair_exact_valid = all(
                np.isfinite(np.asarray(getattr(source, f"pair_b_{channel}", source.pair_b), float)[current_index])
                and not bool(np.asarray(getattr(source, f"pair_interpolated_{channel}", np.zeros(len(source.pair_b), dtype=bool)), bool)[current_index])
                for channel in ("pos", "neg")
            )
        selected_key = self.mcd_peak_selector_combo.currentData()
        selected_channel = selected_key[0] if selected_key else "pos"
        selected_branch = selected_key[2] if selected_key else self.mcd_peak_branch_combo.currentText()
        selected_kind = selected_key[3] if selected_key and len(selected_key) > 3 else "peak"
        selected_track = None
        if selected_key:
            selected_track = next((track for track in self.mcd_peak_channel_results.get(selected_channel, {}).tracks if track.peak_id == selected_key[1] and track.branch == selected_branch and track.feature_kind == selected_kind), None)
        derivative_lines = {}
        if raw_ax is not None:
            raw_ax.set_title("Selected field spectrum")
            if deriv_ax is not None:
                deriv_ax.set_title("−d²R/dE² · both channels", fontsize=9, pad=8)
            for channel, color in (("pos", "#1769aa"), ("neg", "#d1495b")):
                idx = int(self.mcd_peak_field_combo.currentData() or 0)
                raw_ax.plot(energy, spectra[channel][idx], color=color, lw=0.9, label=f"{format_mcd_angle(angles[channel])}° · B={fields[channel][idx]:.4g} T")
                grid, d2, actual_window = second_derivative_map(energy, spectra[channel][idx][None, :], int(self.mcd_peak_deriv_window_spin.value()))
                derivative_lines[channel] = (grid, -d2[0])
                if deriv_ax is not None:
                    deriv_ax.plot(grid, -d2[0], color=color, lw=0.9, label=f"{format_mcd_angle(angles[channel])}° · B={fields[channel][idx]:.4g} T")
            raw_ax.set_ylabel(f"{source_label} intensity (a.u.)"); raw_ax.set_xlabel("Energy (eV)")
            raw_ax.tick_params(labelbottom=False, bottom=False)
            raw_ax.legend(fontsize=7, loc="upper left")
            raw_ax.grid(alpha=0.2); raw_ax.set_xlim(float(np.nanmin(energy)), float(np.nanmax(energy)))
            if deriv_ax is not None:
                deriv_ax.set_ylabel("−d²R/dE²"); deriv_ax.set_xlabel("Energy (eV)")
                deriv_ax.legend(fontsize=7, loc="best")
                deriv_ax.grid(alpha=0.2); deriv_ax.set_xlim(float(np.nanmin(energy)), float(np.nanmax(energy)))
        selected_index = self.mcd_peak_field_combo.currentData()
        if selected_index is not None and selected_track is not None:
            idx = int(selected_index)
            selected_method = self.mcd_peak_tracker_method_combo.currentText()
            selected_e0 = selected_track.reference_energy_ev
            for channel, color in (("pos", "#1769aa"), ("neg", "#d1495b")):
                analysis = self.mcd_peak_method_results.get(selected_method, {}).get(channel)
                if analysis is None:
                    continue
                if channel == selected_channel:
                    track = selected_track
                else:
                    candidates = [item for item in analysis.tracks if item.branch == selected_branch and item.feature_kind == selected_kind and item.quality != BOUNDARY_UNRELIABLE and selected_e0 is not None and item.reference_energy_ev is not None and abs(float(item.reference_energy_ev) - float(selected_e0)) <= 0.005]
                    track = candidates[0] if len(candidates) == 1 else None
                if track is None:
                    continue
                target_field = float(fields[channel][idx])
                point = next((item for item in track.points if item.status == "tracked" and item.energy_ev is not None and abs(float(item.field_t) - target_field) <= 1e-9), None)
                if point is None:
                    continue
                energy_point = float(point.energy_ev)
                if raw_ax is not None:
                    y_point = float(np.interp(energy_point, energy, spectra[channel][idx]))
                    feature_line = raw_ax.axvline(energy_point, color=color, ls="--", lw=1.1)
                    feature_line._mcd_peak_drag_kind = "feature"
                    feature_line._mcd_peak_drag_channel = channel
                    if pair_exact_valid:
                        raw_ax.plot([energy_point], [y_point], "o", color=color, mec="white", zorder=8, label="_mcd_selection_marker")
                if deriv_ax is not None and channel in derivative_lines:
                    dgrid, dvalues = derivative_lines[channel]
                    feature_line = deriv_ax.axvline(energy_point, color=color, ls="--", lw=1.1)
                    feature_line._mcd_peak_drag_kind = "feature"
                    feature_line._mcd_peak_drag_channel = channel
                    deriv_ax.plot([energy_point], [float(np.interp(energy_point, dgrid, dvalues))], "o", color=color, mec="white", zorder=8, label="_mcd_selection_marker")
        manual_channel = getattr(self, "_mcd_peak_manual_center_channel", None)
        manual_center = getattr(self, "_mcd_peak_manual_center_ev", None)
        manual_index = getattr(self, "_mcd_peak_manual_center_field_index", None)
        if (
            manual_channel in {"pos", "neg"}
            and manual_center is not None
            and np.isfinite(manual_center)
            and manual_index is not None
            and selected_index is not None
            and int(manual_index) == int(selected_index)
        ):
            self._draw_mcd_peak_manual_center(
                raw_ax,
                deriv_ax,
                str(manual_channel),
                float(manual_center),
            )
        if raw_ax is not None:
            self._draw_mcd_peak_candidate_markers(raw_ax, fields, source)
        if show_maps:
            labels = np.asarray(getattr(source, "pair_labels", np.full(len(source.pair_b), "")), str)
            for axis, channel in zip(self.mcd_peak_map_axes, ("pos", "neg")):
                field = fields[channel]
                mask = np.ones(field.size, dtype=bool) if selected_branch in {"", "All sweep directions"} else labels == selected_branch
                order = np.argsort(field[mask])
                if self.mcd_peak_show_tracks_chk.isChecked():
                    analysis = self.mcd_peak_channel_results.get(channel)
                    for track in analysis.tracks if analysis is not None else ():
                        if track.quality == BOUNDARY_UNRELIABLE or (selected_branch not in {"", "All sweep directions"} and track.branch != selected_branch):
                            continue
                        xs = [point.energy_ev if point.status in {"tracked", "ambiguous"} and point.energy_ev is not None else np.nan for point in track.points]
                        ys = [point.field_t for point in track.points]
                        is_selected = selected_key is not None and channel == selected_key[0] and track.peak_id == selected_key[1] and track.branch == selected_key[2] and track.feature_kind == selected_key[3]
                        line = axis.plot(xs, ys, color="#ffffff", lw=1.8 if is_selected else 0.8, alpha=0.95 if is_selected else 0.35, marker=None, label=f"{channel} {track.feature_kind} {track.peak_id}")[0]
                        line._mcd_peak_branch = track.branch
                        line._mcd_peak_channel = channel
                        line._mcd_peak_track = track
                        if is_selected:
                            self._mcd_peak_track_lines.append(line)
                            if pair_exact_valid and current_index is not None:
                                target_field = float(fields[channel][current_index])
                                point = next((item for item in track.points if item.status == "tracked" and item.energy_ev is not None and abs(float(item.field_t) - target_field) <= 1e-9), None)
                                if point is not None:
                                    axis.plot([float(point.energy_ev)], [float(point.field_t)], "o", color="#f4a261", mec="white", ms=4, label="_mcd_selection_marker")
                if self.mcd_peak_selected_field is not None:
                    axis.axhline(float(self.mcd_peak_selected_field), color="#f4a261", lw=1.1, ls="--")
                if self.mcd_peak_map_mode_combo.currentText() == "Raw R":
                    grid = energy; values = spectra[channel][mask][order]
                    finite = np.abs(values[np.isfinite(values)]); limit = float(np.nanpercentile(finite, 98)) if finite.size else 1.0; limit = max(limit, np.finfo(float).eps)
                    axis.pcolormesh(grid, field[mask][order], values, shading="auto", cmap="viridis", vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
                    axis.set_title(f"{channel} · {source_label} · full E")
                    axis.set_xlabel("Energy (eV)"); axis.set_ylabel("B (T)")
                    continue
                grid, values, actual_window = second_derivative_map(energy, spectra[channel][mask][order], int(self.mcd_peak_deriv_window_spin.value()))
                finite = np.abs(values[np.isfinite(values)]); limit = float(np.nanpercentile(finite, 98)) if finite.size else 1.0; limit = max(limit, np.finfo(float).eps)
                axis.pcolormesh(grid, field[mask][order], -values, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
                axis.set_title(f"{channel} · −d²R/dE² · full E"); axis.set_xlabel("Energy (eV)"); axis.set_ylabel("B (T)")
        self._plot_mcd_peak_result(result_ax, source, selected_key, selected_track)
        self.figure.subplots_adjust(left=0.10, right=0.98, bottom=0.16 if not show_maps else 0.13, top=0.95)
        self.canvas.setToolTip(
            "MCD Peak Shift: drag the right B cursor to choose a measured field; "
            "drag a left spectrum/derivative feature line to choose a nearby reliable feature."
        )
        self.canvas.draw_idle()

    def _draw_mcd_peak_manual_center(
        self,
        raw_ax: Any,
        derivative_ax: Any,
        channel: str,
        center_ev: float,
    ) -> None:
        """Draw the free center cursor without changing tracked artists."""
        self._mcd_peak_manual_center_artists = {}
        color = "#1769aa" if channel == "pos" else "#d1495b"
        for name, axis in (("raw", raw_ax), ("derivative", derivative_ax)):
            if axis is None:
                continue
            line = axis.axvline(
                float(center_ev),
                color=color,
                ls="-",
                lw=1.6,
                alpha=0.95,
                zorder=10,
                label="_mcd_manual_center",
            )
            line._mcd_peak_drag_kind = "feature"
            line._mcd_peak_drag_channel = channel
            line._mcd_peak_manual_center = True
            self._mcd_peak_manual_center_artists[name] = line

    def _draw_mcd_peak_candidate_markers(self, axis: Any, fields: dict[str, np.ndarray], source: Any) -> None:
        """Place compact numbered feature markers at the selected field centers."""
        self._mcd_peak_candidate_artists = {}
        keys = list(getattr(self, "_mcd_peak_candidate_keys", ()))
        selected = self.mcd_peak_selector_combo.currentData()
        if axis is None or not keys or selected is None:
            return
        selected = tuple(selected)
        current_index = self.mcd_peak_field_combo.currentData()
        if current_index is None:
            return
        current_index = int(current_index)
        branch = self.mcd_peak_branch_combo.currentText()
        markers: list[tuple[int, tuple, float]] = []
        for index, key in enumerate(keys):
            track = self._mcd_peak_candidate_track(tuple(key))
            if track is None or str(track.branch) != str(branch):
                continue
            channel = str(key[0])
            channel_fields = np.asarray(fields.get(channel, source.pair_b), float)
            if current_index < 0 or current_index >= channel_fields.size or not np.isfinite(channel_fields[current_index]):
                continue
            target_field = float(channel_fields[current_index])
            point = next(
                (item for item in track.points if item.status == "tracked" and item.energy_ev is not None and abs(float(item.field_t) - target_field) <= 1e-9),
                None,
            )
            if point is not None and np.isfinite(point.energy_ev):
                markers.append((index, tuple(key), float(point.energy_ev)))
        if not markers:
            return
        markers.sort(key=lambda item: item[2])
        x_span = max(abs(float(np.diff(axis.get_xlim())[0])), 1e-12)
        last_x_by_row = [-np.inf, -np.inf, -np.inf]
        marker_rows: dict[int, int] = {}
        for index, _key, energy_point in markers:
            row = next((row_index for row_index, last_x in enumerate(last_x_by_row) if energy_point - last_x >= 0.065 * x_span), len(last_x_by_row) - 1)
            marker_rows[index] = row
            last_x_by_row[row] = energy_point
        for index, key, energy_point in markers:
            track = self._mcd_peak_candidate_track(key)
            kind_label = "P" if track is not None and track.feature_kind == "peak" else "D"
            active = tuple(selected) == key
            artist = axis.text(
                energy_point,
                0.965 - 0.068 * marker_rows[index],
                f"{kind_label}{key[1]}",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.6,
                fontweight="bold",
                color="white",
                zorder=28,
                bbox={
                    "boxstyle": "circle,pad=0.22",
                    "facecolor": "#0078d4" if active else "#f0a202",
                    "edgecolor": "white",
                    "linewidth": 0.8,
                    "alpha": 0.96,
                },
            )
            artist._mcd_peak_candidate_key = key
            artist._mcd_peak_candidate_energy = energy_point
            self._mcd_peak_candidate_artists[index] = artist

    def _mcd_peak_candidate_marker_hit(self, event: Any) -> Any | None:
        axis = getattr(self, "mcd_peak_spectrum_ax", None)
        if event.inaxes is not axis or getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return None
        try:
            renderer = self.figure.canvas.get_renderer()
        except Exception:
            return None
        for artist in getattr(self, "_mcd_peak_candidate_artists", {}).values():
            if artist.get_visible() and artist.get_window_extent(renderer=renderer).contains(float(event.x), float(event.y)):
                return artist
        return None

    def _on_mcd_peak_candidate_marker_click(self, event: Any) -> bool:
        artist = self._mcd_peak_candidate_marker_hit(event)
        if artist is None:
            return False
        key = tuple(getattr(artist, "_mcd_peak_candidate_key", ()))
        if len(key) != 4:
            return True
        combo = self.mcd_peak_selector_combo
        for index in range(combo.count()):
            value = combo.itemData(index)
            if value is None or len(value) != 4:
                continue
            normalized = (str(value[0]), int(value[1]), str(value[2]), str(value[3]))
            if normalized == key:
                combo.setCurrentIndex(index)
                return True
        return True

    def _plot_mcd_peak_result(self, axis, source, selected_key, selected_track) -> None:
        axis.clear()
        zero_line = axis.axhline(0.0, color="#777", lw=0.7)
        zero_line._mcd_ignore_click = True
        axis.set_xlabel("B (T)")
        axis.set_ylabel("Kp−K (meV)" if self.mcd_peak_result_mode_combo.currentText() == "Valley splitting" else "ΔE (meV)")
        if selected_key is None:
            axis.set_title("Result"); return
        channel, peak_id, branch, feature_kind = selected_key
        selected_result_method = getattr(self, "_mcd_peak_selected_result_method", self.mcd_peak_tracker_method_combo.currentText())
        available_methods = tuple(self.mcd_peak_method_results.keys())
        methods = ("Local mixed fit",) if selected_result_method == "Local mixed fit" and "Local mixed fit" in available_methods else tuple(method for method in ("Raw spectrum", "Second derivative") if method in available_methods)
        method_colors = {"Local mixed fit": "#0078d4", "Raw spectrum": "#1769aa", "Second derivative": "#d1495b"}
        branches = [str(value) for value in dict.fromkeys(np.asarray(source.pair_labels, str).tolist())]
        selected_result_branch = self.mcd_peak_branch_combo.currentText() or branch
        selected_field_index = self.mcd_peak_field_combo.currentData()
        selected_field = float(source.pair_b[int(selected_field_index)]) if selected_field_index is not None else None

        def highlight_selected_result(line, xs, ys) -> None:
            if getattr(line, "_mcd_peak_method", None) != selected_result_method or getattr(line, "_mcd_peak_branch", None) != selected_result_branch:
                return
            x_values = np.asarray(xs, float)
            y_values = np.asarray(ys, float)
            valid_x = np.isfinite(x_values)
            if not np.any(valid_x):
                return
            target = float(selected_field) if selected_field is not None else float(x_values[np.flatnonzero(valid_x)[0]])
            selected_point = int(np.flatnonzero(valid_x)[np.argmin(np.abs(x_values[valid_x] - target))])
            cursor = axis.axvline(target, color="#111111", lw=0.8, alpha=0.8)
            cursor._mcd_ignore_click = True
            if not np.isfinite(y_values[selected_point]):
                return
            marker = axis.plot(
                [float(x_values[selected_point])], [float(y_values[selected_point])],
                marker="o", ms=7, mfc="none", mec="#111111", mew=1.2,
                linestyle="None", label="_mcd_selected_result_marker",
            )[0]
            marker._mcd_ignore_click = True
        e0 = selected_track.reference_energy_ev if selected_track is not None and selected_track.reference_method != "unavailable" else None
        mode = self.mcd_peak_result_mode_combo.currentText()
        if e0 is None and mode != "Valley splitting":
            axis.set_title(f"Result · E0 unavailable for {channel} {feature_kind} {peak_id}")
            notes = []
            for method in methods:
                analysis = self.mcd_peak_method_results.get(method, {}).get(channel)
                if analysis is None or not analysis.tracks:
                    notes.append(f"{method}: unmatched")
            notes.append("No exact or bracketed E0 reference; shifts unavailable.")
            axis.text(0.03, 0.95, "\n".join(notes), transform=axis.transAxes, va="top", fontsize=8)
            return
        target_energy = float(e0) if e0 is not None else float(np.median([point.energy_ev for point in selected_track.points if point.energy_ev is not None])) if selected_track is not None and any(point.energy_ev is not None for point in selected_track.points) else float("nan")
        if mode == "Valley splitting":
            split_by_method = {}
            fixed_k = None
            valley_notes = []
            directions = branches if e0 is not None else [branch]
            if e0 is None:
                for method in methods:
                    for direction in branches:
                        if direction != branch:
                            valley_notes.append(f"{method} · {direction}: E0 unavailable; branch not matched")
            for method in methods:
                split_by_method[method] = {}
                for direction in directions:
                    split = compute_valley_splitting(self.mcd_peak_method_results, method=method, selected_channel=channel, branch=direction, target_energy_ev=target_energy, tolerance_ev=0.005, feature_kind=feature_kind, allow_energy_fallback=e0 is None)
                    split_by_method[method][direction] = split
                    if split.status != "ok":
                        valley_notes.append(f"{method} · {direction}: {split.status}")
            # Determine K/K' once, preferring the raw-spectrum mapping across
            # every available branch, then apply it to both methods/branches.
            for method in methods:
                if fixed_k is not None:
                    break
                fixed_k = next((split.k_channel for split in split_by_method[method].values() if split.k_channel in {"pos", "neg"}), None)
            for method in split_by_method:
                for direction in directions:
                    split = split_by_method[method][direction]
                    if fixed_k:
                        split = compute_valley_splitting(self.mcd_peak_method_results, method=method, selected_channel=channel, branch=direction, target_energy_ev=target_energy, tolerance_ev=0.005, fixed_k_channel=fixed_k, feature_kind=feature_kind, allow_energy_fallback=e0 is None)
                    xs = [point.field_t for point in split.points]
                    ys = [np.nan if point.splitting_ev is None else 1000.0 * point.splitting_ev for point in split.points]
                    if any(np.isfinite(ys)):
                        line = axis.plot(xs, ys, color=method_colors[method], ls="-" if "increasing" in direction.casefold() else "--", label=f"{method} · {direction}")[0]
                        line._mcd_peak_branch = direction
                        line._mcd_peak_method = method
                        line._mcd_peak_channel = channel
                        line._mcd_peak_track = None
                        marker_x = [x for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
                        marker_y = [y for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
                        if marker_x:
                            marker = axis.plot(marker_x, marker_y, "o", color=method_colors[method], ms=3, label="_mcd_result_marker")[0]
                            marker._mcd_ignore_click = True
                        highlight_selected_result(line, xs, ys)
            if selected_result_method == "Local mixed fit" and selected_track is not None:
                reference_label = "exact" if selected_track.reference_method == "exact 0 T" else "interpolated"
                title_e0 = f"Fit E0={float(e0):.4f} eV ({reference_label})" if e0 is not None else "Fit E0 unavailable; conservative match"
            else:
                title_e0 = f"E0={float(e0):.4f} eV ({'exact' if selected_track is not None and selected_track.reference_method == 'exact 0 T' else 'interpolated'})" if e0 is not None else "E0 unavailable; conservative match"
            axis.set_title(
                f"K′−K · {title_e0}" if selected_result_method == "Local mixed fit"
                else f"Valley splitting · both sweep directions · {title_e0}"
            )
            if valley_notes:
                axis.text(0.03, 0.95, "\n".join(dict.fromkeys(valley_notes)), transform=axis.transAxes, va="top", fontsize=8)
        else:
            for method in methods:
                color = method_colors[method]
                analyses = self.mcd_peak_method_results.get(method, {})
                for direction in branches:
                    analysis = analyses.get(channel)
                    if analysis is None: continue
                    candidates = [track for track in analysis.tracks if track.branch == direction and track.feature_kind == feature_kind and track.quality != BOUNDARY_UNRELIABLE and track.reference_energy_ev is not None and abs(track.reference_energy_ev - float(e0)) <= 0.005]
                    if len(candidates) != 1: continue
                    track = candidates[0]
                    xs = [point.field_t for point in track.points]
                    ys = [np.nan if point.delta_energy_ev is None else 1000.0 * point.delta_energy_ev for point in track.points]
                    line = axis.plot(xs, ys, color=color, ls="-" if "increasing" in direction.casefold() else "--", label=f"{method} · {direction}")[0]
                    line._mcd_peak_branch = direction
                    line._mcd_peak_method = method
                    line._mcd_peak_channel = channel
                    line._mcd_peak_track = track
                    marker_x = [x for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
                    marker_y = [y for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
                    if marker_x:
                        marker = axis.plot(marker_x, marker_y, "o", color=color, ms=3, label="_mcd_result_marker")[0]
                        marker._mcd_ignore_click = True
                    highlight_selected_result(line, xs, ys)
            reference_label = "exact" if selected_track is not None and selected_track.reference_method == "exact 0 T" else "interpolated"
            if selected_result_method == "Local mixed fit" and selected_track is not None and selected_track.locator_energy_ev is not None:
                axis.set_title(
                    f"ΔE(B) · {channel} {feature_kind} {peak_id} · "
                    f"Fit E0={float(e0):.4f} eV ({reference_label})"
                )
            else:
                axis.set_title(f"Selected ΔE(B) · {channel} {feature_kind} {peak_id} · E0={float(e0):.4f} eV ({reference_label})")
        if axis.get_legend_handles_labels()[1]: axis.legend(fontsize=7, loc="best")

    def _on_mcd_peak_map_click(self, event: Any) -> None:
        if event.inaxes not in getattr(self, "mcd_peak_map_axes", ()) or self.loaded is None or self.loaded.mcd_result is None or event.ydata is None:
            return
        source = self.loaded.mcd_result
        channel = "pos" if event.inaxes is self.mcd_peak_map_axes[0] else "neg"
        field = np.asarray(getattr(source, f"pair_b_{channel}", source.pair_b), float)
        interpolated = np.asarray(getattr(source, f"pair_interpolated_{channel}", np.zeros(field.size, dtype=bool)), bool)
        field = np.where(interpolated, np.asarray(source.pair_b, float), field)
        branch = self.mcd_peak_branch_combo.currentText()
        labels = np.asarray(getattr(source, "pair_labels", np.full(field.size, branch)), str)
        candidates = np.flatnonzero(labels == branch) if branch and branch != "All sweep directions" else np.arange(field.size)
        if candidates.size == 0:
            return
        index = int(candidates[np.nanargmin(np.abs(field[candidates] - float(event.ydata)))])
        click_xy = event.inaxes.transData.transform((float(event.xdata), float(event.ydata))) if event.xdata is not None else None
        nearest_feature = None
        nearest_distance = float("inf")
        analysis = self.mcd_peak_channel_results.get(channel)
        for track in analysis.tracks if analysis is not None else ():
            if track.quality == BOUNDARY_UNRELIABLE or (branch and branch != "All sweep directions" and track.branch != branch):
                continue
            points = [point for point in track.points if point.status == "tracked" and point.energy_ev is not None]
            if not points or click_xy is None:
                continue
            pixels = event.inaxes.transData.transform(np.asarray([[point.energy_ev, point.field_t] for point in points], float))
            distance = float(np.min(np.hypot(pixels[:, 0] - click_xy[0], pixels[:, 1] - click_xy[1])))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_feature = track
        if nearest_feature is not None and nearest_distance <= 24.0:
            branch_index = self.mcd_peak_branch_combo.findText(nearest_feature.branch)
            if branch_index >= 0:
                self.mcd_peak_branch_combo.setCurrentIndex(branch_index)
            key = (channel, nearest_feature.peak_id, nearest_feature.branch, nearest_feature.feature_kind)
            selector_index = self.mcd_peak_selector_combo.findData(key)
            if selector_index >= 0:
                self.mcd_peak_selector_combo.setCurrentIndex(selector_index)
        combo_index = self.mcd_peak_field_combo.findData(index)
        if combo_index >= 0:
            self.mcd_peak_field_combo.setCurrentIndex(combo_index)

    def _on_mcd_peak_spectrum_click(self, event: Any) -> None:
        if event.inaxes is not getattr(self, "mcd_peak_spectrum_ax", None) or event.xdata is None or event.ydata is None or self.loaded is None or self.loaded.mcd_result is None:
            return
        source = self.loaded.mcd_result
        index = self.mcd_peak_field_combo.currentData()
        if index is None:
            return
        energy = np.sort(np.asarray(source.energy_ev, float))
        energy_order = spectrum_energy_order(source)
        best = None
        for channel, color, values_name in (("pos", "#1769aa", "pair_raw_pos"), ("neg", "#d1495b", "pair_raw_neg")):
            raw = np.asarray(getattr(source, values_name), float)[int(index), energy_order]
            field = np.asarray(getattr(source, f"pair_b_{channel}", source.pair_b), float)
            interpolated = np.asarray(getattr(source, f"pair_interpolated_{channel}", np.zeros(len(field), dtype=bool)), bool)
            target_field = float(source.pair_b[int(index)]) if interpolated[int(index)] else float(field[int(index)])
            selected_branch = self.mcd_peak_branch_combo.currentText()
            analysis = self.mcd_peak_channel_results.get(channel)
            for track in analysis.tracks if analysis is not None else ():
                if track.quality == BOUNDARY_UNRELIABLE or (selected_branch and selected_branch != "All sweep directions" and track.branch != selected_branch):
                    continue
                points = [point for point in track.points if point.status == "tracked" and point.energy_ev is not None and abs(float(point.field_t) - target_field) <= 1e-9]
                for point in points:
                    y = float(np.interp(float(point.energy_ev), energy, raw))
                    pixel = event.inaxes.transData.transform((float(point.energy_ev), y))
                    click = event.inaxes.transData.transform((float(event.xdata), float(event.ydata)))
                    distance = float(np.hypot(pixel[0] - click[0], pixel[1] - click[1]))
                    if best is None or distance < best[0]:
                        best = (distance, channel, track)
        if best is None or best[0] > 24.0:
            return
        _, channel, track = best
        branch_index = self.mcd_peak_branch_combo.findText(track.branch)
        if branch_index >= 0:
            self.mcd_peak_branch_combo.setCurrentIndex(branch_index)
        selector_index = self.mcd_peak_selector_combo.findData((channel, track.peak_id, track.branch, track.feature_kind))
        if selector_index >= 0:
            self.mcd_peak_selector_combo.setCurrentIndex(selector_index)

    def _on_mcd_peak_result_click(self, event: Any) -> None:
        if event.inaxes is not getattr(self, "mcd_peak_shift_ax", None) or event.xdata is None or event.ydata is None:
            return
        if self.loaded is None or self.loaded.mcd_result is None:
            return
        click_xy = event.inaxes.transData.transform((float(event.xdata), float(event.ydata)))
        best_distance = float("inf")
        selected_line = None
        for line in event.inaxes.lines:
            if getattr(line, "_mcd_ignore_click", False):
                continue
            method = getattr(line, "_mcd_peak_method", None)
            branch = getattr(line, "_mcd_peak_branch", None)
            if method is None or branch is None:
                continue
            xs = np.asarray(line.get_xdata(), float)
            ys = np.asarray(line.get_ydata(), float)
            valid = np.isfinite(xs) & np.isfinite(ys)
            if not np.any(valid):
                continue
            pixels = event.inaxes.transData.transform(np.column_stack((xs[valid], ys[valid])))
            distance = float(np.min(np.hypot(pixels[:, 0] - click_xy[0], pixels[:, 1] - click_xy[1])))
            if distance < best_distance:
                best_distance = distance
                selected_line = line
        if selected_line is None or best_distance > 24.0:
            return
        source = self.loaded.mcd_result
        fields = np.asarray(source.pair_b, float)
        finite = np.flatnonzero(np.isfinite(fields))
        if finite.size == 0:
            return
        selected_branch = str(getattr(selected_line, "_mcd_peak_branch"))
        selected_method = str(getattr(selected_line, "_mcd_peak_method"))
        self._mcd_peak_selected_result_method = selected_method
        self._mcd_peak_selected_result_branch = selected_branch
        if selected_branch and hasattr(self, "mcd_peak_branch_combo"):
            branch_index = self.mcd_peak_branch_combo.findText(selected_branch)
            if branch_index >= 0:
                self.mcd_peak_branch_combo.setCurrentIndex(branch_index)
        labels = np.asarray(getattr(source, "pair_labels", np.full(fields.size, selected_branch)), str)
        branch_fields = finite[labels[finite] == selected_branch]
        if branch_fields.size == 0:
            branch_fields = finite
        nearest = int(branch_fields[np.argmin(np.abs(fields[branch_fields] - float(event.xdata)))])
        combo_index = self.mcd_peak_field_combo.findData(nearest)
        if combo_index >= 0:
            self.mcd_peak_field_combo.setCurrentIndex(combo_index)
            self.mcd_peak_status.setText(f"{selected_method} · {selected_branch}; nearest measured field selected: B = {fields[nearest]:.6g} T")

    def _mcd_peak_toolbar_navigation_active(self) -> bool:
        mode = getattr(self.toolbar, "mode", None)
        return bool(getattr(mode, "_navigate_mode", None))

    def _mcd_peak_line_hit(self, event: Any, axis: Any, *, kind: str | None = None) -> Any | None:
        if event.inaxes is not axis or getattr(event, "x", None) is None:
            return None
        best = None
        best_distance = 8.0
        for line in axis.lines:
            if kind is not None and getattr(line, "_mcd_peak_drag_kind", None) != kind:
                continue
            values = np.asarray(line.get_xdata(), float).ravel()
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            pixel_x = float(axis.transData.transform((float(values[0]), float(axis.get_ylim()[0])))[0])
            distance = abs(pixel_x - float(event.x))
            if distance <= best_distance:
                best = line
                best_distance = distance
        return best

    def _mcd_peak_result_cursor_hit(self, event: Any) -> bool:
        if event.inaxes is not getattr(self, "mcd_peak_shift_ax", None) or getattr(event, "x", None) is None:
            return False
        source = getattr(self.loaded, "mcd_result", None) if self.loaded is not None else None
        index = self.mcd_peak_field_combo.currentData() if hasattr(self, "mcd_peak_field_combo") else None
        if source is None or index is None:
            return False
        fields = np.asarray(source.pair_b, float)
        index = int(index)
        if index < 0 or index >= fields.size or not np.isfinite(fields[index]):
            return False
        axis = self.mcd_peak_shift_ax
        pixel_x = float(axis.transData.transform((float(fields[index]), float(axis.get_ylim()[0])))[0])
        return abs(pixel_x - float(event.x)) <= 8.0

    def _begin_mcd_peak_drag(self, event: Any) -> bool:
        if event.button != 1 or self._mcd_peak_toolbar_navigation_active():
            return False
        if self._mcd_peak_result_cursor_hit(event):
            self._mcd_peak_drag = {"kind": "field"}
            self.canvas.setCursor(Qt.SizeHorCursor)
            self.mcd_peak_status.setText("Drag B cursor to choose a measured field; release anywhere to finish.")
            return True
        for axis in (getattr(self, "mcd_peak_spectrum_ax", None), getattr(self, "mcd_peak_derivative_ax", None)):
            line = self._mcd_peak_line_hit(event, axis, kind="feature") if axis is not None else None
            if line is not None:
                xdata = np.asarray(line.get_xdata(), float).ravel()
                line_center = float(xdata[0]) if xdata.size and np.isfinite(xdata[0]) else None
                channel = str(getattr(line, "_mcd_peak_drag_channel"))
                current_center = getattr(self, "_mcd_peak_manual_center_ev", None)
                if (
                    getattr(self, "_mcd_peak_manual_center_channel", None) == channel
                    and current_center is not None
                    and np.isfinite(current_center)
                ):
                    line_center = float(current_center)
                self._mcd_peak_drag = {
                    "kind": "feature",
                    "channel": channel,
                    "offset": 0.0 if line_center is None or event.xdata is None else float(event.xdata) - line_center,
                    "center": line_center,
                }
                if line_center is not None:
                    self._mcd_peak_manual_center_ev = line_center
                    self._mcd_peak_manual_center_channel = channel
                    self._mcd_peak_manual_center_field_index = (
                        None if self.mcd_peak_field_combo.currentData() is None
                        else int(self.mcd_peak_field_combo.currentData())
                    )
                    self._ensure_mcd_peak_manual_center_artists(channel, line_center)
                    self._start_mcd_peak_manual_center_blit()
                self.canvas.setCursor(Qt.SizeHorCursor)
                self.mcd_peak_status.setText("Drag center freely; release to search and track a nearby reflection.")
                return True
        return False

    def _mcd_peak_field_indices_for_branch(self) -> list[int]:
        combo = getattr(self, "mcd_peak_field_combo", None)
        if combo is None:
            return []
        return [int(combo.itemData(index)) for index in range(combo.count()) if combo.itemData(index) is not None]

    def _update_mcd_peak_field_drag(self, event: Any) -> None:
        if event.xdata is None or self.loaded is None or self.loaded.mcd_result is None:
            return
        source = self.loaded.mcd_result
        candidates = self._mcd_peak_field_indices_for_branch()
        fields = np.asarray(source.pair_b, float)
        candidates = [index for index in candidates if 0 <= index < fields.size and np.isfinite(fields[index])]
        if not candidates:
            return
        index = min(candidates, key=lambda item: abs(float(fields[item]) - float(event.xdata)))
        if self.mcd_peak_field_combo.currentData() != index:
            combo_index = self.mcd_peak_field_combo.findData(index)
            if combo_index >= 0:
                self.mcd_peak_field_combo.setCurrentIndex(combo_index)
        self.status_bar_view.set_cursor_readback(f"Drag B cursor: measured B = {fields[index]:.6g} T")

    def _update_mcd_peak_feature_drag(self, event: Any) -> None:
        if event.xdata is None or self.loaded is None or self.loaded.mcd_result is None:
            return
        channel = str(self._mcd_peak_drag.get("channel"))
        offset = float(self._mcd_peak_drag.get("offset", 0.0))
        energy = np.asarray(self.loaded.mcd_result.energy_ev, float).ravel()
        finite = energy[np.isfinite(energy)]
        if finite.size == 0:
            return
        center = float(np.clip(float(event.xdata) - offset, np.nanmin(finite), np.nanmax(finite)))
        self._mcd_peak_manual_center_ev = center
        self._mcd_peak_manual_center_channel = channel
        index = self.mcd_peak_field_combo.currentData()
        self._mcd_peak_manual_center_field_index = None if index is None else int(index)
        self._ensure_mcd_peak_manual_center_artists(channel, center)
        for artist in getattr(self, "_mcd_peak_manual_center_artists", {}).values():
            artist.set_xdata([center, center])
        self.status_bar_view.set_cursor_readback(
            f"Center E = {center:.6g} eV · release to search ±5 meV"
        )
        if not self._blit_mcd_peak_manual_center():
            self.canvas.draw_idle()

    def _start_mcd_peak_manual_center_blit(self) -> None:
        """Cache the static raw/derivative axes once for a drag gesture."""
        self._mcd_peak_manual_center_blit_backgrounds = {}
        lines = tuple(getattr(self, "_mcd_peak_manual_center_artists", {}).items())
        if not lines:
            return
        try:
            for line in self._mcd_peak_manual_center_artists.values():
                line.set_animated(True)
            self.canvas.draw()
            for name, line in lines:
                axis = getattr(line, "axes", None)
                if axis is not None:
                    self._mcd_peak_manual_center_blit_backgrounds[name] = self.canvas.copy_from_bbox(axis.bbox)
        except Exception:
            for line in self._mcd_peak_manual_center_artists.values():
                line.set_animated(False)
            self._mcd_peak_manual_center_blit_backgrounds = {}

    def _blit_mcd_peak_manual_center(self) -> bool:
        backgrounds = getattr(self, "_mcd_peak_manual_center_blit_backgrounds", {})
        artists = getattr(self, "_mcd_peak_manual_center_artists", {})
        if not backgrounds or not artists:
            return False
        try:
            for name, background in backgrounds.items():
                line = artists.get(name)
                axis = getattr(line, "axes", None)
                if line is None or axis is None:
                    return False
                self.canvas.restore_region(background)
                axis.draw_artist(line)
                self.canvas.blit(axis.bbox)
            return True
        except Exception:
            return False

    def _stop_mcd_peak_manual_center_blit(self) -> None:
        for line in getattr(self, "_mcd_peak_manual_center_artists", {}).values():
            try:
                line.set_animated(False)
            except AttributeError:
                pass
        self._mcd_peak_manual_center_blit_backgrounds = {}

    def _ensure_mcd_peak_manual_center_artists(self, channel: str, center_ev: float) -> None:
        """Create the paired raw/−d² artists once, then move them in place."""
        artists = getattr(self, "_mcd_peak_manual_center_artists", {})
        if artists and all(getattr(line, "axes", None) is not None for line in artists.values()):
            for line in artists.values():
                line.set_xdata([float(center_ev), float(center_ev)])
            return
        self._draw_mcd_peak_manual_center(
            getattr(self, "mcd_peak_spectrum_ax", None),
            getattr(self, "mcd_peak_derivative_ax", None),
            str(channel),
            float(center_ev),
        )

    def _update_mcd_peak_drag(self, event: Any) -> bool:
        if self._mcd_peak_drag is None:
            return False
        if event.inaxes is None:
            return True
        if self._mcd_peak_drag["kind"] == "field":
            if event.inaxes is getattr(self, "mcd_peak_shift_ax", None):
                self._update_mcd_peak_field_drag(event)
        elif event.inaxes in {
            getattr(self, "mcd_peak_spectrum_ax", None),
            getattr(self, "mcd_peak_derivative_ax", None),
        }:
            self._update_mcd_peak_feature_drag(event)
        return True

    def _on_canvas_release(self, event: Any) -> None:
        if self._mcd_peak_drag is not None:
            kind = self._mcd_peak_drag["kind"]
            drag = self._mcd_peak_drag
            self._mcd_peak_drag = None
            self.canvas.unsetCursor()
            if kind == "feature":
                self._stop_mcd_peak_manual_center_blit()
                self._commit_mcd_peak_feature_center(drag)
                return
            self.status_bar_view.set_cursor_readback(
                "MCD Peak Shift selection updated." if kind == "field" else "MCD Peak Shift feature selection updated."
            )
            return
        self.mcd_controller._on_canvas_release(event)


    def _unified_mcd_candidates(self) -> list[dict[str, Any]]:
        """Adapt completed legacy worker results into the unified catalog."""
        output: list[dict[str, Any]] = []
        for feature in getattr(self, "_mcd_unified_features", ()):
            if isinstance(feature, dict):
                item = dict(feature)
            elif hasattr(feature, "to_dict"):
                try:
                    item = dict(feature.to_dict())
                except (TypeError, ValueError):
                    item = {}
            else:
                item = {
                    name: getattr(feature, name)
                    for name in ("id", "domain", "kind", "source", "energy_ev", "field_t", "branch", "prominence", "width_ev", "confidence")
                    if hasattr(feature, name)
                }
            if "energy_ev" in item and "center_ev" not in item:
                item["center_ev"] = item["energy_ev"]
            output.append(item)
        if output:
            return output
        method_results = getattr(self, "_mcd_peak_method_results", {})
        selected = method_results.get("Raw spectrum", {}) if isinstance(method_results, dict) else {}
        for channel, analysis in selected.items() if isinstance(selected, dict) else ():
            for track in getattr(analysis, "tracks", ()):
                kind = str(getattr(track, "feature_kind", "peak"))
                output.append({
                    "id": f"{channel}:{getattr(track, 'peak_id', len(output) + 1)}:{kind}",
                    "label": f"{'P' if kind == 'peak' else 'D'}{getattr(track, 'peak_id', len(output) + 1)}",
                    "domain": "spectrum", "source": "spectrum", "kind": kind,
                    "center_ev": getattr(track, "reference_energy_ev", None),
                })
        return output

    def _unified_window_candidates(self) -> list[dict[str, Any]]:
        """Return advisory fixed-width MCD windows for the Center selector."""
        result = self.loaded.mcd_result if self.loaded is not None and self.loaded.mode == "MCD" else None
        if result is None:
            return []
        energy = np.asarray(result.energy_ev, dtype=float)
        finite = energy[np.isfinite(energy)]
        if finite.size < 2:
            return []
        try:
            width = float(self.mcd_window_width_spin.value())
            search_range = (float(self.mcd_spins["xmin"].value()), float(self.mcd_spins["xmax"].value()))
            if search_range[1] <= search_range[0]:
                search_range = (float(np.min(finite)), float(np.max(finite)))
            metric = self.mcd_controller._mcd_window_metric()
            suggestions = suggest_mcd_window_centers(result, width, metric=metric, energy_range=search_range, max_candidates=None)
        except (AttributeError, TypeError, ValueError, FloatingPointError):
            return []
        suggestions = tuple(candidate for candidate in suggestions if np.isfinite(candidate.snr) and float(candidate.snr) >= 3.0)
        return [{
            "id": f"mcd-window-{index + 1}-{candidate.center_ev:.9f}", "label": str(index + 1),
            "display_id": str(index + 1), "domain": "mcd", "source": "MCD window", "kind": "window",
            "center_ev": float(candidate.center_ev), "energy_ev": float(candidate.center_ev), "width_mev": width,
            "metadata": {"suggestion": True, "score": float(candidate.score), "snr": float(candidate.snr),
                          "branch_agreement": float(candidate.branch_agreement), "quality_gate": "SNR >= 3"},
            "recommended": candidate.score_rank <= 6,
        } for index, candidate in enumerate(sorted(suggestions, key=lambda item: item.center_ev))]

    def _unified_display_candidates(self) -> list[dict[str, Any]]:
        """Use high-field MCD peak windows; retain separate spectrum diagnostics."""
        spectra = [item for item in self._unified_mcd_candidates()
                   if str(item.get("domain", "")).casefold() == "spectrum"]
        from core.mcd_center_history import merge_center_candidates
        history = getattr(self.loaded, "mcd_center_history", ()) if self.loaded is not None else ()
        return merge_center_candidates(self._unified_window_candidates(), history) + spectra

    def _queue_mcd_unified_analysis(self) -> None:
        """Run pure candidate detection in the owned pool with a generation guard."""
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            return
        try:
            from core.mcd_analysis import detect_analysis_features  # noqa: F401 - import check for optional core module
        except ImportError:
            return
        source = self.loaded.mcd_result
        mode = str(getattr(getattr(self, "mcd_unified_view", None), "state", None).detection_mode if getattr(getattr(self, "mcd_unified_view", None), "state", None) is not None else getattr(self, "_mcd_unified_detection_mode", "raw"))
        self._mcd_unified_detection_mode = mode
        request_key = self._unified_catalog_request_key()
        self._mcd_unified_catalog_requested_key = request_key
        active = getattr(self, "_mcd_unified_analysis_worker", None)
        if active is not None:
            if getattr(active, "_mcd_catalog_key", None) != request_key:
                self._mcd_unified_analysis_pending = True
            return
        if self._mcd_unified_catalog_completed_key == request_key:
            return
        self._mcd_unified_analysis_generation += 1
        generation = self._mcd_unified_analysis_generation
        worker = Worker(_run_unified_mcd_analysis, source, detection_mode=mode)
        worker._mcd_source = source
        worker._mcd_catalog_key = request_key
        self._mcd_unified_analysis_worker = worker
        self._mcd_unified_analysis_source = source
        worker.signals.result.connect(
            lambda payload, w=worker, g=generation, s=source, k=request_key: self._on_mcd_unified_analysis_result(w, g, s, payload, k)
        )
        worker.signals.error.connect(
            lambda message, w=worker, g=generation, k=request_key: self._on_mcd_unified_analysis_error(w, g, message, k)
        )
        worker.signals.finished.connect(
            lambda w=worker, g=generation: self._on_mcd_unified_analysis_finished(w, g)
        )
        self.thread_pool.start(worker)

    def _unified_catalog_request_key(self) -> tuple[Any, ...]:
        result = self.loaded.mcd_result if self.loaded is not None and self.loaded.mode == "MCD" else None
        view = getattr(self, "mcd_unified_view", None)
        mode = str(getattr(getattr(view, "state", None), "detection_mode", getattr(self, "_mcd_unified_detection_mode", "raw")))
        return (id(result), int(getattr(self, "_mcd_source_generation", 0)), mode, int(getattr(self, "_mcd_unified_detector_settings_version", 1)))

    def _on_unified_detection_mode_changed(self, mode: str) -> None:
        self._mcd_unified_detection_mode = str(mode).casefold()
        self._mcd_unified_features = ()
        self._mcd_unified_catalog_completed_key = None
        self._mcd_unified_track_payload = None
        self._mcd_unified_track_cache.clear()
        if not getattr(self, "_is_closing", False):
            self._queue_mcd_unified_analysis()

    def _unified_catalog_needed(self) -> bool:
        view = getattr(self, "mcd_unified_view", None)
        if view is None:
            return False
        filter_mode = str(view.candidate_filter_combo.currentText()).casefold()
        detection_mode = str(getattr(view.state, "detection_mode", "raw")).casefold()
        return filter_mode in {"spectrum", "all"} or detection_mode != "raw"

    def _on_unified_candidate_filter_changed(self, value: str) -> None:
        if str(value).casefold() in {"spectrum", "all"} and not getattr(self, "_is_closing", False):
            self._queue_mcd_unified_analysis()

    def _on_unified_candidate_selected(self, _index: int) -> None:
        """Start the existing cancellable tracker for an explicit selection."""
        view = getattr(self, "mcd_unified_view", None)
        if view is None or view.selected_candidate is None:
            return
        selected = view.selected_candidate
        if str(selected.get("kind", "")).casefold() == "window":
            center = selected.get("center_ev", selected.get("energy_ev"))
            try:
                self._mcd_candidate_applying = True
                if selected.get("history"):
                    self.mcd_window_width_spin.setValue(float(selected["width_mev"]))
                self.mcd_window_center_spin.setValue(float(center))
            finally:
                self._mcd_candidate_applying = False
            self._mcd_unified_track_payload = None
            view.candidate_status.setText(f"Window {float(center):.6f} eV · fixed {float(selected.get('width_mev', 0.0)):.4g} meV")
            return
        self._queue_unified_selected_feature_analysis()

    def _on_unified_candidate_selection_changed(self, _identifier: Any) -> None:
        """Queue only spectral feature tracking; window suggestions are cursors."""
        view = getattr(self, "mcd_unified_view", None)
        selected = view.selected_candidate if view is not None else None
        if selected is None or str(selected.get("kind", "")).casefold() == "window":
            return
        QTimer.singleShot(0, self._queue_unified_selected_feature_analysis)

    def _add_unified_manual_link(self) -> None:
        """Create one explicit MCD↔spectrum association for selected tracking."""
        view = getattr(self, "mcd_unified_view", None)
        if view is None:
            return
        catalog = self._unified_mcd_candidates()
        mcd = [item for item in catalog if str(item.get("domain", "")).casefold() == "mcd"]
        spectra = [item for item in catalog if str(item.get("domain", "")).casefold() == "spectrum"]
        if not mcd or not spectra:
            self._status("Manual connection needs both an MCD and spectrum candidate.")
            return
        existing_pairs = [
            (mcd_id, spectrum_id)
            for mcd_id, spectrum_ids in self._mcd_unified_manual_links.items()
            for spectrum_id in spectrum_ids
        ]
        if existing_pairs:
            action, accepted = QInputDialog.getItem(
                self, "MCD ↔ spectrum connections", "Action",
                ["Add connection", "Remove connection"], 0, False,
            )
            if not accepted:
                return
        else:
            action = "Add connection"
        mcd_labels = [f"{item.get('label', item.get('id', 'MCD'))} · {float(item.get('center_ev', 0.0)):.6f} eV" for item in mcd]
        spectrum_labels = [f"{item.get('label', item.get('id', 'spectrum'))} · {float(item.get('center_ev', 0.0)):.6f} eV" for item in spectra]
        if action == "Remove connection":
            labels = [f"{left} ↔ {right}" for left, right in existing_pairs]
            selected_pair, accepted = QInputDialog.getItem(self, "Manual feature connection", "Remove", labels, 0, False)
            if not accepted:
                return
            left_id, right_id = existing_pairs[labels.index(selected_pair)]
            values = [value for value in self._mcd_unified_manual_links.get(left_id, []) if value != right_id]
            if values:
                self._mcd_unified_manual_links[left_id] = values
            else:
                self._mcd_unified_manual_links.pop(left_id, None)
            self._mcd_unified_track_cache.clear()
            self._status(f"Removed connection {left_id} ↔ {right_id}; reprocessing the selected feature.")
            self._queue_unified_selected_feature_analysis()
            return
        left, accepted = QInputDialog.getItem(self, "Manual feature connection", "MCD candidate", mcd_labels, 0, False)
        if not accepted:
            return
        right, accepted = QInputDialog.getItem(self, "Manual feature connection", "Spectrum candidate", spectrum_labels, 0, False)
        if not accepted:
            return
        left_id = str(mcd[mcd_labels.index(left)].get("id", ""))
        right_id = str(spectra[spectrum_labels.index(right)].get("id", ""))
        self._mcd_unified_manual_links.setdefault(left_id, [])
        if right_id not in self._mcd_unified_manual_links[left_id]:
            self._mcd_unified_manual_links[left_id].append(right_id)
        self._mcd_unified_track_cache.clear()
        self._status(f"Connected {left_id} to {right_id}; reprocessing the selected feature.")
        self._queue_unified_selected_feature_analysis()

    def _unified_feature_analysis_key(self) -> tuple[Any, ...] | None:
        view = getattr(self, "mcd_unified_view", None)
        selected = view.selected_candidate if view is not None else None
        if selected is None or selected.get("center_ev") is None or self.loaded is None:
            return None
        controls = self.mcd_unified_controls
        method = str(controls.feature_method_combo.currentText())
        source = str(controls.feature_source_combo.currentText())
        low = float(controls.feature_search_low_spin.value())
        high = float(controls.feature_search_high_spin.value())
        half_width = max(abs(high - low) * 0.5, 0.005)
        return (
            id(self.loaded.mcd_result), str(selected.get("id", "")), method,
            source, round(low, 9), round(high, 9),
            str(controls.reference_mode_combo.currentText()), round(float(controls.manual_reference_spin.value()), 9),
            round(float(controls.prominence_spin.value()), 6),
            str(view.channel_mapping_combo.currentText()),
            tuple((key, tuple(values)) for key, values in sorted(self._mcd_unified_manual_links.items())),
            tuple(str(item.get("id", "")) for item in self._unified_mcd_candidates()),
            self._mcd_unified_catalog_requested_key or self._unified_catalog_request_key(),
        )

    def _queue_unified_selected_feature_analysis(self, *_args: Any) -> None:
        """Queue selected-feature tracking with latest-result/source guards."""
        if self.loaded is None or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            return
        view = getattr(self, "mcd_unified_view", None)
        selected = view.selected_candidate if view is not None else None
        key = self._unified_feature_analysis_key()
        if selected is None or key is None:
            if view is not None:
                view.candidate_status.setText("Select a valid candidate to start measured tracking.")
            return
        active = getattr(self, "_mcd_unified_track_worker", None)
        if active is not None:
            if key != self._mcd_unified_track_key:
                self._mcd_unified_track_pending = key
                self._mcd_unified_track_payload = None
                if view is not None:
                    view.candidate_status.setText("Waiting for the latest selected feature…")
            return
        cached = self._mcd_unified_track_cache.get(key)
        if cached is not None:
            self._mcd_unified_track_payload = dict(cached)
            self._mcd_unified_track_payload_key = (view.state.source_generation, key)
            self._plot_mcd_unified(auto=True)
            return
        try:
            method = str(self.mcd_unified_controls.feature_method_combo.currentText())
            method = "Local fit" if method.casefold().startswith("local") else method
            source = str(self.mcd_unified_controls.feature_source_combo.currentText())
            low = float(self.mcd_unified_controls.feature_search_low_spin.value())
            high = float(self.mcd_unified_controls.feature_search_high_spin.value())
            if high <= low and (low != 0.0 or high != 0.0):
                view.candidate_status.setText("Search bounds must be finite and increasing.")
                self._status("Selected feature analysis was not queued: search bounds must be increasing.")
                return
            half_width = max(abs(high - low) * 0.5, 0.005)
            feature = dict(selected)
            candidates = tuple(self._mcd_unified_features)
            mapping_label = str(view.channel_mapping_combo.currentText())
            if mapping_label.startswith("K = −"):
                mapping_mode, k_channel = "fixed", "neg"
            elif mapping_label.startswith("K = +"):
                mapping_mode, k_channel = "fixed", "pos"
            elif "infer" in mapping_label.casefold() or "energy" in mapping_label.casefold():
                mapping_mode, k_channel = "energy_based", None
            else:
                mapping_mode, k_channel = "unknown", None
            reference_mode = str(self.mcd_unified_controls.reference_mode_combo.currentText())
            reference_energy = (
                float(self.mcd_unified_controls.manual_reference_spin.value())
                if reference_mode.casefold().startswith("manual") else None
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self._status(f"Selected feature analysis could not be queued: {exc}")
            return
        self._mcd_unified_track_generation += 1
        generation = self._mcd_unified_track_generation
        source_result = self.loaded.mcd_result
        worker = Worker(
            _run_unified_selected_feature_analysis, source_result, feature, candidates,
            method=method, source=source, half_width_ev=half_width,
            search_window_ev=(low, high) if high > low else None,
            prominence_fraction=float(self.mcd_unified_controls.prominence_spin.value()),
            reference_energy_ev=reference_energy,
            mapping_mode=mapping_mode,
            k_channel=k_channel,
            reference_id=str(selected.get("id", "")) or None,
            manual_links=dict(self._mcd_unified_manual_links),
        )
        self._mcd_unified_track_payload = {"tracks": [], "status": "processing"}
        self._mcd_unified_track_worker = worker
        self._mcd_unified_track_key = key
        worker.signals.result.connect(
            lambda payload, w=worker, g=generation, s=source_result, k=key: self._on_unified_feature_analysis_result(w, g, s, k, payload)
        )
        worker.signals.error.connect(
            lambda message, w=worker, g=generation, k=key: self._on_unified_feature_analysis_error(w, g, message, k)
        )
        worker.signals.finished.connect(
            lambda w=worker, g=generation: self._on_unified_feature_analysis_finished(w, g)
        )
        # Keep the current combo/catalog stable while the worker runs.  A
        # pre-result full render can rebuild the filtered list during a Qt
        # selection signal and associate the caller's index with another
        # candidate.
        view.candidate_status.setText(f"Analyzing {selected.get('id', 'feature')}…")
        self.thread_pool.start(worker)

    def _on_unified_feature_analysis_result(self, worker: Worker, generation: int, source: Any, key: tuple[Any, ...], payload: Any) -> None:
        if worker is not self._mcd_unified_track_worker or generation != self._mcd_unified_track_generation:
            return
        if self._mcd_unified_track_pending is not None or key != self._mcd_unified_track_key:
            return
        if self.loaded is None or self.loaded.mcd_result is not source:
            return
        value = dict(payload) if isinstance(payload, dict) else {}
        self._mcd_unified_track_cache[key] = value
        self._mcd_unified_track_payload = value
        self._mcd_unified_track_payload_key = (self.mcd_unified_view.state.source_generation, key)
        self._plot_mcd_unified(auto=True)

    def _on_unified_feature_analysis_error(self, worker: Worker, generation: int, message: str, key: Any = None) -> None:
        if worker is self._mcd_unified_track_worker and generation == self._mcd_unified_track_generation:
            if key is not None and key != self._unified_feature_analysis_key():
                return
            self._mcd_unified_track_payload = {"tracks": [], "status": "unsupported", "reason": str(message).splitlines()[0]}
            self._status(f"Selected MCD feature analysis failed: {str(message).splitlines()[0]}")

    def _on_unified_feature_analysis_finished(self, worker: Worker, generation: int) -> None:
        if worker is not self._mcd_unified_track_worker:
            return
        self._mcd_unified_track_worker = None
        pending = self._mcd_unified_track_pending
        self._mcd_unified_track_pending = None
        if pending is not None and not getattr(self, "_is_closing", False):
            QTimer.singleShot(0, self._queue_unified_selected_feature_analysis)

    def _on_mcd_unified_analysis_result(self, worker: Worker, generation: int, source: Any, payload: Any, request_key: Any = None) -> None:
        if worker is not self._mcd_unified_analysis_worker or generation != self._mcd_unified_analysis_generation:
            return
        if self.loaded is None or self.loaded.mcd_result is not source:
            return
        request_key = request_key if request_key is not None else getattr(worker, "_mcd_catalog_key", None)
        if request_key != getattr(self, "_mcd_unified_catalog_requested_key", request_key):
            return
        self._mcd_unified_features = tuple(getattr(payload, "features", payload if isinstance(payload, (list, tuple)) else ()))
        self._mcd_unified_catalog_completed_key = request_key
        if hasattr(self, "mcd_unified_view"):
            self.mcd_unified_view.set_source_generation(generation)
            view = self.mcd_unified_view
            view.refresh_catalog(self._unified_display_candidates(), self._unified_analysis_payload(view))
            # Catalog completion is the normal load path.  Queue the selected
            # candidate after the redraw so the first measured track appears
            # without requiring a second user click.
            QTimer.singleShot(0, self._queue_unified_selected_feature_analysis)

    def _on_mcd_unified_analysis_error(self, worker: Worker, generation: int, message: str = "", request_key: Any = None) -> None:
        if worker is self._mcd_unified_analysis_worker and generation == self._mcd_unified_analysis_generation:
            request_key = request_key if request_key is not None else getattr(worker, "_mcd_catalog_key", None)
            if request_key != getattr(self, "_mcd_unified_catalog_requested_key", request_key):
                return
            self._mcd_unified_features = ()
            detail = str(message).splitlines()[0] if message else "feature detection failed"
            self._status(f"Unified MCD feature detection failed: {detail}")

    def _on_mcd_unified_analysis_finished(self, worker: Worker, generation: int) -> None:
        if worker is self._mcd_unified_analysis_worker:
            self._mcd_unified_analysis_worker = None
            if self._mcd_unified_analysis_pending and not getattr(self, "_is_closing", False):
                self._mcd_unified_analysis_pending = False
                QTimer.singleShot(0, self._queue_mcd_unified_analysis)

    def _compute_unified_mcd_slopes(self, view: McdUnifiedView) -> Any:
        try:
            from core.mcd_analysis import fit_mcd_slopes
            result = self.loaded.mcd_result
            energy = np.asarray(result.energy_ev, float)
            # McdResult.energy_ev is already sorted, whereas paired spectra
            # retain wavelength-column order. Construct the energy axis in
            # that same raw order before applying the spectral permutation.
            wavelength = np.asarray(getattr(result, "wavelength_nm", ()), float)
            column_energy = 1239.841984 / wavelength if wavelength.shape == energy.shape else energy
            values = np.asarray(result.pair_mcd_corrected, float)
            cached_order = getattr(result, "_unified_energy_order", None)
            order_signature = (column_energy.shape, column_energy.tobytes())
            if getattr(result, "_unified_energy_order_signature", None) != order_signature:
                cached_order = None
            ordered_columns = np.asarray(
                spectrum_energy_order(result) if cached_order is None else cached_order,
                dtype=int,
            )
            try:
                result._unified_energy_order = ordered_columns
                result._unified_energy_order_signature = order_signature
            except AttributeError:
                pass
            half = float(view.state.window_width_mev) * 0.0005
            ordered_energy = column_energy[ordered_columns] if ordered_columns.size == energy.size else column_energy
            mask = np.abs(ordered_energy - view.state.window_center_ev) <= half
            # Match pair_window_trace_by_branch for sub-sample windows.
            if ordered_energy.size and not np.any(mask):
                mask[int(np.argmin(np.abs(ordered_energy - view.state.window_center_ev)))] = True
            metric = str(self.mcd_window_metric_combo.currentText()).casefold().replace(" ", "_")
            if values.ndim == 2 and ordered_columns.size == values.shape[1] and np.any(mask):
                # Select the window in energy order before copying columns;
                # center movement no longer duplicates the whole matrix.
                selected = values[:, ordered_columns[mask]]
                selected_energy = ordered_energy[mask]
            else:
                selected = None
                selected_energy = energy[mask]
            if selected is None:
                trace = np.full(np.asarray(result.pair_b).shape, np.nan)
            elif "integral" in metric:
                trace = np.trapezoid(selected, x=selected_energy, axis=1)
            elif "field" in metric and "absolute" in metric:
                trace = np.sign(np.asarray(result.pair_b, dtype=float)) * np.nanmean(np.abs(selected), axis=1)
            elif "absolute" in metric:
                trace = np.nanmean(np.abs(selected), axis=1)
            else:
                trace = np.nanmean(selected, axis=1)
            controls = self.mcd_unified_controls
            ranges = {
                "low": (controls.slope_low_spin.value(), controls.slope_low_end_spin.value()),
                "high_positive": (controls.slope_high_positive_spin.value(), controls.slope_high_positive_end_spin.value()),
                "high_negative": (controls.slope_high_negative_spin.value(), controls.slope_high_negative_end_spin.value()),
            }
            return fit_mcd_slopes(result.pair_b, trace, result.pair_labels, ranges=ranges)
        except (AttributeError, ImportError, TypeError, ValueError, FloatingPointError):
            return None

    def _unified_analysis_payload(self, view: McdUnifiedView) -> dict[str, Any]:
        """Build the result-panel payload without rebuilding the MCD plots."""
        selected_payload = getattr(self, "_mcd_unified_track_payload", None) or {}
        tracks = list(selected_payload.get("tracks", ())) if isinstance(selected_payload, dict) else []
        status = str(selected_payload.get("status", "Select a candidate to show measured tracking points."))
        selected = view.selected_candidate
        if selected is not None and not selected_payload:
            status = f"Select {selected.get('id', 'candidate')} to start measured tracking."
        elif selected is not None and tracks:
            status = f"Showing {len(tracks)} measured track(s) for {selected.get('id', 'candidate')}."
        elif selected is not None and selected_payload:
            status = f"{selected.get('id', 'candidate')} · {status}."
        reason = str(selected_payload.get("reason", "")).strip() if isinstance(selected_payload, Mapping) else ""
        if reason and reason not in status:
            status = f"{status} {reason}"
        self._mcd_unified_slopes = self._compute_unified_mcd_slopes(view)
        self._mcd_unified_slopes_key = self._unified_window_analysis_key()
        slope_payload = self._mcd_unified_slopes.to_dict() if self._mcd_unified_slopes is not None else None
        return {
            "generation": self._mcd_unified_analysis_generation,
            "tracks": tracks,
            "status": status,
            "slopes": slope_payload,
            "mapping": selected_payload.get("mapping", {}) if isinstance(selected_payload, Mapping) else {},
            "splitting": selected_payload.get("splitting", ()) if isinstance(selected_payload, Mapping) else (),
            "analysis_results": selected_payload.get("analysis_results", {}) if isinstance(selected_payload, Mapping) else {},
            "selected_feature": selected_payload.get("selected_feature") if isinstance(selected_payload, Mapping) else None,
            "reason": reason,
        }

    def _plot_mcd_unified(self, *, auto: bool = False) -> None:
        view = getattr(self, "mcd_unified_view", None)
        if view is None:
            return
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            view.setVisible(False)
            self._show_error("Load data for this tab before plotting.")
            return
        # The unified surface is a presentation adapter over the existing
        # loaded result.  It never invokes feature detection synchronously.
        view.setVisible(True)
        output_folder = self._mcd_export_output_root or self._default_unified_mcd_export_folder()
        self.mcd_unified_controls.save_folder_label.setText(f"Save to: {output_folder}")
        self.mcd_unified_controls.save_folder_label.setToolTip(str(output_folder or ""))
        if hasattr(self, "mcd_candidate_bar"):
            self.mcd_candidate_bar.setVisible(False)
        if hasattr(self, "mcd_peak_candidate_bar"):
            self.mcd_peak_candidate_bar.setVisible(False)
        view.state.window_center_ev = float(self.mcd_window_center_spin.value())
        view.state.window_width_mev = float(self.mcd_window_width_spin.value())
        view.state.window_metric = str(self.mcd_window_metric_combo.currentText())
        view.set_source_generation(self._mcd_unified_analysis_generation)
        view.render(
            self.loaded.mcd_result,
            candidates=self._unified_display_candidates(),
            analysis=self._unified_analysis_payload(view),
        )
        self._mcd_unified_last_published_signature = (
            id(self.loaded), id(self.loaded.mcd_result),
            tuple(self._selected(self.mcd_files)) if hasattr(self, "mcd_files") else (),
            float(self.mcd_window_center_spin.value()),
            float(self.mcd_window_width_spin.value()),
            str(self.mcd_window_metric_combo.currentText()),
            getattr(self, "_mcd_unified_analysis_generation", 0),
        )
        self._mcd_heatmap_ax = view.axes.get("mcd_map")
        self._mcd_pair_ax = view.axes.get("spectra")
        self._mcd_spectrum_ax = view.axes.get("mcd_spectra")
        self._mcd_trace_ax = view.axes.get("mcd_vs_b")
        self._mcd_integral_ax = None
        self.last_plotted_mode = "MCD"
        self._set_stage("Plotted")
        self._update_action_states()

    def _plot_mode(self, mode: str, *, auto: bool = False) -> None:
        if mode != "MCD" and hasattr(self, "mcd_unified_view"):
            self.mcd_unified_view.deactivate()
            self.mcd_unified_view.setVisible(False)
        if mode == "MCD" and hasattr(self, "mcd_unified_view"):
            self._plot_mcd_unified(auto=auto)
            return
        if mode == "MCD Peak Shift":
            self._plot_mcd_peak_shift()
            return
        try:
            if not self.loaded or self.loaded.mode != mode:
                self._show_error("Load data for this tab before plotting.")
                return

            request_before = getattr(self, "_active_load_token", None)
            busy_before = bool(getattr(self, "_load_in_progress", False))
            lifecycle_before = getattr(self, "_load_lifecycle_generation", 0)
            self._ensure_loaded_matches_ui_params(mode)
            request_changed = request_before != getattr(self, "_active_load_token", None)
            lifecycle_changed = lifecycle_before != getattr(self, "_load_lifecycle_generation", 0)
            pending_changed = bool(getattr(self, "_pending_load_options", None))
            if self._load_in_progress and (not auto or request_changed or lifecycle_changed or pending_changed or not busy_before):
                return
            plot_key = self._current_plot_params_key(mode)
            gate_only_update = mode == "DRR" and self.drr_controller._is_drr_gate_only_change(plot_key)
            if gate_only_update and self._last_plot_cube is not None:
                self.drr_controller._update_drr_spectrum_and_gate_line(self._last_plot_cube)
                self._last_plot_params_key = plot_key
                self.last_plotted_mode = mode
                self._set_stage("Plotted")
                self._update_action_states()
                self._status(f"Plotted DRR (gate update).")
                return
            if mode == "Compare" and self.compare_controller._update_cmp_gate_only():
                self._last_plot_params_key = plot_key
                self.last_plotted_mode = mode
                self._set_stage("Plotted")
                self._update_action_states()
                self._status("Plotted Compare (gate update).")
                return

            if mode == "DRR" and not self._drr_limits_from_controls:
                self._capture_drr_view_limits()
            self._drr_limits_from_controls = False
            self.compare_controller._disable_cmp_blitting()
            self.figure.clear()
            self._drr_heatmap_ax = None
            self._drr_spectrum_ax = None
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
            self.power_peak_controller.axes = None
            self.power_peak_controller.residual_axis = None
            self._shg_raw_ax = None
            self._shg_corrected_ax = None
            self._shg_angle_ax = None
            self._mcd_pair_ax = None
            self._mcd_pair_cursor = None
            self._mcd_pair_spectrum_lines = []
            self._mcd_linecut_lines = []
            self._mcd_linecut_diagnostic_text = None
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
                render = plot_pl(ax1, downsample_cube_for_display(plot_cube), params)
                self._add_heatmap_colorbar(render, cax, label=params.cbar_label)
                self._pl_heatmap_ax = ax1
                self._pl_spectrum_ax = ax2
                self._pl_last_plot_cube = plot_cube
                self._pl_gate_line = None
                for helper in (getattr(self.pl_controller, "_pl_region_blitters", None) or ()):
                    try:
                        helper.disconnect()
                    except (AttributeError, RuntimeError):
                        pass
                self.pl_controller._pl_region_blitters = None
                self._pl_heatmap_peak_artist = None
                self._pl_heatmap_fit_artist = None
                self.pl_controller._update_pl_spectrum_and_gate_line(plot_cube)
            elif mode == "DRR" and self.loaded.cube is not None:
                self._pl_heatmap_ax = None
                self._pl_spectrum_ax = None
                self._pl_last_plot_cube = None
                self._pl_gate_line = None
                raw_cube, _, raw_win, raw_poly = self.drr_controller._drr_cube_with_metadata(None)
                second_cube = None
                second_win = second_poly = 0
                view = "both" if self._drr_side_by_side else self._drr_plot_view
                products: list[tuple[str, DataCube, HeatmapParams, int | None, int, int]] = []
                raw_params = self._make_drr_params(raw_cube, None)
                advanced_derivative = self.drr_controller._drr_derivative_value()
                if view in {"raw", "both"}:
                    display_raw_cube = raw_cube
                    display_raw_derivative = None
                    if view == "raw" and advanced_derivative == 1:
                        display_raw_cube, _, raw_win, raw_poly = self.drr_controller._drr_cube_with_metadata(1)
                        raw_params = self._make_drr_params(display_raw_cube, 1)
                        raw_params = HeatmapParams(**{
                            **raw_params.__dict__,
                            "title": f"{raw_cube.title} (Advanced first derivative, dE)",
                            "cbar_label": "d(DR/R)/dE",
                        })
                        display_raw_derivative = 1
                    products.append(("raw", display_raw_cube, raw_params, display_raw_derivative, raw_win, raw_poly))
                if view in {"second", "both"}:
                    second_cube, _, second_win, second_poly = self.drr_controller._drr_cube_with_metadata(2)
                    self._sync_drr_second_auto_scale(second_cube)
                    second_params = self._make_drr_params(second_cube, 2)
                    products.append(("second", second_cube, second_params, 2, second_win, second_poly))
                if not products:
                    products.append(("raw", raw_cube, raw_params, None, raw_win, raw_poly))
                self.loaded.drr_derivative_label = (
                    "Advanced first derivative (dE)"
                    if advanced_derivative == 1 and self._drr_plot_view == "raw"
                    else {"raw": "None", "second": "d2E"}.get(self._drr_plot_view, "None")
                )
                self._drr_plot_cubes = {key: cube for key, cube, *_rest in products}
                self._drr_heatmap_axes = {}
                self._drr_spectrum_axes = {}
                both = len(products) == 2
                if both:
                    # Keep one compact filename header for the paired view;
                    # the full source name remains available in the UI.
                    source_label = Path(self.loaded.primary_file or raw_cube.title).name
                    max_chars = 56 if self.figure.get_figwidth() < 11.0 else 82
                    if len(source_label) > max_chars:
                        half = max(12, (max_chars - 1) // 2)
                        source_label = source_label[:half] + "…" + source_label[-half:]
                    self.figure.suptitle(source_label, fontsize=9, y=0.985)
                    figure_width_px = max(
                        1.0, float(self.figure.get_figwidth() * self.figure.dpi)
                    )
                    left_margin_px = float(np.clip(80.0, 70.0, 85.0))
                    right_margin_px = 55.0
                    gs = self.figure.add_gridspec(
                        nrows=1, ncols=2,
                        left=left_margin_px / figure_width_px,
                        right=1.0 - right_margin_px / figure_width_px,
                        bottom=0.105, top=0.90,
                        wspace=0.24 if figure_width_px < 1050.0 else 0.16,
                    )
                else:
                    gs = self.figure.add_gridspec(
                        nrows=2, ncols=2,
                        width_ratios=[1.0, 0.035],
                        height_ratios=[1.0, 1.0], wspace=0.12, hspace=0.28,
                    )
                gate_val = float(self._mode_spins(mode)["gate"].value())
                shared_heat_ax = None
                for index, (key, product_cube, product_params, _deriv, _used_win, _poly) in enumerate(products):
                    if both:
                        product_gs = gs[0, index].subgridspec(
                            nrows=2, ncols=2,
                            width_ratios=[1.0, 0.035],
                            height_ratios=[1.0, 1.0],
                            wspace=0.08, hspace=0.27,
                        )
                        map_slot, cbar_slot, spectrum_slot = product_gs[0, 0], product_gs[0, 1], product_gs[1, 0]
                    else:
                        map_slot, cbar_slot, spectrum_slot = gs[0, 0], gs[0, 1], gs[1, 0]
                    if shared_heat_ax is None:
                        ax1 = self.figure.add_subplot(map_slot)
                    else:
                        ax1 = self.figure.add_subplot(map_slot, sharex=shared_heat_ax, sharey=shared_heat_ax)
                    cax = self.figure.add_subplot(cbar_slot)
                    ax2 = self.figure.add_subplot(spectrum_slot, sharex=ax1)
                    if both:
                        short_title = "ΔR/R" if key == "raw" else "Second derivative"
                        product_params = HeatmapParams(
                            **{
                                **product_params.__dict__,
                                "title": short_title,
                                "xlabel": "",
                            }
                        )
                    render = plot_drr(ax1, downsample_cube_for_display(product_cube), product_params)
                    self._add_heatmap_colorbar(
                        render, cax,
                        label="" if both else product_params.cbar_label,
                    )
                    if both and index > 0:
                        ax1.tick_params(axis="y", labelleft=False)
                        ax1.set_ylabel("")
                    if both:
                        ax1.set_xlabel("")
                        ax1.tick_params(axis="x", labelbottom=False)
                        if key == "second" and isinstance(
                            cax.yaxis.get_major_formatter(), ScalarFormatter
                        ):
                            cax.ticklabel_format(
                                axis="y", style="sci", scilimits=(-2, 2), useMathText=True,
                            )
                    spectrum_ylabel = (
                        "d²(DR/R)/dE²"
                        if both and key == "second"
                        else product_params.cbar_label
                    )
                    self._plot_spectrum_with_roi(
                        ax2, product_cube, gate_val, ylabel=spectrum_ylabel,
                        xlim=product_params.xlim,
                    )
                    self._drr_heatmap_axes[key] = ax1
                    self._drr_spectrum_axes[key] = ax2
                    if shared_heat_ax is None:
                        shared_heat_ax = ax1
                active_key = "second" if self._drr_plot_view == "second" and "second" in self._drr_plot_cubes else "raw"
                plot_cube = self._drr_plot_cubes[active_key]
                self._drr_heatmap_ax = self._drr_heatmap_axes[active_key]
                self._drr_spectrum_ax = self._drr_spectrum_axes[active_key]
                if self._drr_view_limits is not None:
                    old_xlim, old_ylim = self._drr_view_limits
                    for axis in self._drr_heatmap_axes.values():
                        axis.set_xlim(old_xlim)
                        axis.set_ylim(old_ylim)
                self._drr_heatmap_peak_artist = None
                self._drr_heatmap_fit_artist = None
                gate_used, _, _ = self.drr_controller._current_drr_spectrum(plot_cube)
                self.drr_controller._set_drr_gate_spin_value(gate_used)
                self.drr_controller._update_drr_spectrum_and_gate_line(plot_cube)
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
                condition_text = format_mcd_acquisition_conditions(
                    result.acquisition_conditions, include_bias=False
                )
                header_ax.text(
                    0.0, 0.58 if condition_text else 0.23, plot_cube.title,
                    transform=header_ax.transAxes, ha="left", va="center", fontsize=12,
                )
                if condition_text:
                    header_ax.text(
                        0.0, 0.02, condition_text, transform=header_ax.transAxes,
                        ha="left", va="bottom", fontsize=6.6, color="#303030",
                    )
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
                self._mcd_pair_spectrum_lines = [
                    pair_ax.plot(energy, result.pair_raw_pos[pair_index, energy_order], label=f"raw {result.pos_angle:g} deg", lw=1.0, ls="--", alpha=0.65)[0],
                    pair_ax.plot(energy, result.pair_raw_neg[pair_index, energy_order], label=f"raw {result.neg_angle:g} deg", lw=1.0, ls="--", alpha=0.65)[0],
                    pair_ax.plot(energy, result.pair_corrected_pos[pair_index, energy_order], label=f"corrected {result.pos_angle:g} deg", lw=1.35)[0],
                    pair_ax.plot(energy, result.pair_corrected_neg[pair_index, energy_order], label=f"final corrected {result.neg_angle:g} deg", lw=1.35)[0],
                ]
                self._mcd_linecut_lines = [
                    linecut_ax.plot(energy, result.pair_mcd_raw[pair_index, energy_order], label="raw MCD", lw=1.1, alpha=0.8)[0],
                    linecut_ax.plot(energy, result.pair_mcd_corrected[pair_index, energy_order], label="corrected MCD", lw=1.4)[0],
                ]
                e0 = float(self.mcd_window_center_spin.value()); width = float(self.mcd_window_width_spin.value())
                if e0 <= 0:
                    e0 = float(np.nanmedian(energy)); self.mcd_window_center_spin.blockSignals(True); self.mcd_window_center_spin.setValue(e0); self.mcd_window_center_spin.blockSignals(False)
                half = width * 5e-4
                self._mcd_window_artists = []
                for axis in (pair_ax, linecut_ax):
                    self.mcd_controller._add_mcd_window_overlay(axis, e0, half)
                    axis.set_xlim(self._safe_spectrum_xlim(energy, params.xlim)); axis.grid(alpha=0.22); axis.legend(fontsize=7, frameon=False)
                    axis.set_xlabel("Energy (eV)")
                pair_ax.set_title(f"Paired spectra: B = {pair_b:.5g} T")
                pair_ax.set_ylabel("Intensity")
                # Keep correction diagnostics readable without covering the
                # high-energy spectrum.  The upper-left corner is the quiet
                # region for the usual rising PL background, and putting the
                # values in the legend makes them visually consistent with
                # the trace labels rather than a floating annotation.
                pair_handles, pair_labels = pair_ax.get_legend_handles_labels()
                correction_handle = Line2D([], [], linestyle="None", marker=None)
                correction_label = self.mcd_controller._mcd_pair_correction_label(result, pair_index)
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
                self._mcd_linecut_diagnostic_text = linecut_ax.text(
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
                pair_cursor = heat_ax.axhline(pair_b, color="white", lw=0.9, alpha=0.95, zorder=22)
                pair_cursor.set_path_effects([
                    path_effects.Stroke(linewidth=2.5, foreground="#242424", alpha=0.55),
                    path_effects.Normal(),
                ])
                self._mcd_pair_cursor = pair_cursor
                self.mcd_controller._add_mcd_window_overlay(heat_ax, e0, half, draggable=True)
                marker_rows: dict[int, int] = {}
                last_x_by_row = [-np.inf, -np.inf, -np.inf]
                visible_span = max(abs(float(np.diff(heat_ax.get_xlim())[0])), 1e-12)
                for candidate_index in sorted(
                    range(len(self._mcd_center_candidates)),
                    key=lambda item: self._mcd_center_candidates[item].center_ev,
                ):
                    candidate_x = self._mcd_center_candidates[candidate_index].center_ev
                    row = next(
                        (
                            row_index for row_index, last_x in enumerate(last_x_by_row)
                            if candidate_x - last_x >= 0.027 * visible_span
                        ),
                        len(last_x_by_row) - 1,
                    )
                    marker_rows[candidate_index] = row
                    last_x_by_row[row] = candidate_x
                self._mcd_candidate_artists = {}
                for candidate_index, candidate in enumerate(self._mcd_center_candidates):
                    active_candidate = candidate_index == self._mcd_candidate_active_index
                    marker_color = "#0078d4" if active_candidate else "#f0a202"
                    self._mcd_candidate_artists[candidate_index] = heat_ax.text(
                        candidate.center_ev, 0.965 - 0.068 * marker_rows[candidate_index], str(candidate_index + 1),
                        transform=heat_ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=6.6, fontweight="bold", color="white", zorder=28,
                        bbox={
                            "boxstyle": "circle,pad=0.22", "facecolor": marker_color,
                            "edgecolor": "white", "linewidth": 0.8, "alpha": 0.96,
                        },
                        path_effects=[path_effects.withStroke(linewidth=1.4, foreground="#242424", alpha=0.55)],
                    )
                trace_specs = (
                    ("mean", "Signed mean", "#1666b0", self.mcd_show_signed_mean_chk.isChecked()),
                    ("field_signed_absolute_mean", "Field-signed |MCD|", "#c94c00", self.mcd_show_absolute_mean_chk.isChecked()),
                    ("absolute_mean", "Unsigned |MCD|", "#777777", self.mcd_show_unsigned_absolute_mean_chk.isChecked()),
                    ("integral", "Signed integral", "#6a3d9a", self.mcd_show_integral_chk.isChecked()),
                )
                show_raw = self.mcd_show_raw_chk.isChecked()
                requested_metrics = [
                    metric_name for metric_name, _label, _color, visible in trace_specs if visible
                ]
                if self.mcd_fit_zero_chk.isChecked() and "mean" not in requested_metrics:
                    requested_metrics.append("mean")
                traces = pair_window_trace_by_branch(
                    result, e0, width,
                    metrics=requested_metrics,
                    include_raw=show_raw,
                )
                branch_fits = (
                    low_field_mcd_branch_fits(
                        traces, float(self.mcd_fit_b_window_spin.value())
                    )
                    if self.mcd_fit_zero_chk.isChecked() else {}
                )
                integral_ax = trace_ax.twinx()
                self._mcd_trace_lines = {}
                self._mcd_fit_lines = {}
                self._mcd_slope_text = None
                primary_data_values: list[np.ndarray] = []
                integral_data_values: list[np.ndarray] = []
                for metric_name, label, color, visible in trace_specs:
                    if not visible:
                        continue
                    axis = integral_ax if metric_name == "integral" else trace_ax
                    for branch, line_style, marker_fill in (("B increasing", "-", color), ("B decreasing", "--", "white")):
                        for source, alpha in (("corrected", 1.0), ("raw", 0.72)):
                            if source == "raw" and not show_raw:
                                continue
                            b_trace, values = traces[branch][f"{source}_{metric_name}"]
                            finite_values = np.asarray(values, float)
                            if metric_name == "integral":
                                integral_data_values.append(finite_values[np.isfinite(finite_values)])
                            else:
                                primary_data_values.append(finite_values[np.isfinite(finite_values)])
                            line, = axis.plot(
                                b_trace, values, f"o{line_style}", ms=3.1, lw=1.25,
                                color=color, alpha=alpha, markerfacecolor=marker_fill,
                                markeredgecolor=color, markeredgewidth=0.9, label="_nolegend_",
                            )
                            self._mcd_trace_lines[(metric_name, branch, source)] = line
                    if branch_fits and metric_name == "mean":
                        fit_fields = np.concatenate([
                            np.asarray(traces[branch]["corrected_mean"][0], float)
                            for branch in ("B increasing", "B decreasing")
                        ])
                        fit_mask = np.isfinite(fit_fields) & (
                            np.abs(fit_fields) <= float(self.mcd_fit_b_window_spin.value())
                        )
                        if branch_fits and np.count_nonzero(fit_mask) >= 2:
                            finite_fields = fit_fields[np.isfinite(fit_fields)]
                            fit_x = np.array([
                                float(np.min(finite_fields)),
                                float(np.max(finite_fields)),
                            ])
                            fit_styles = {
                                "B increasing": ("#d55e00", "-"),
                                "B decreasing": ("#7a3db8", "--"),
                            }
                            for branch, (slope, intercept) in branch_fits.items():
                                color, line_style = fit_styles[branch]
                                fit_line, = trace_ax.plot(
                                    fit_x, slope * fit_x + intercept,
                                    line_style, color=color, lw=2.2, zorder=26,
                                    label="_nolegend_",
                                )
                                fit_line.set_path_effects([
                                    path_effects.Stroke(
                                        linewidth=3.5, foreground="white", alpha=0.95
                                    ),
                                    path_effects.Normal(),
                                ])
                                self._mcd_fit_lines[branch] = fit_line

                def set_data_ylim(target_axis, arrays: list[np.ndarray]) -> None:
                    finite = [array for array in arrays if array.size]
                    if not finite:
                        return
                    values = np.concatenate(finite)
                    low, high = float(np.min(values)), float(np.max(values))
                    span = high - low
                    padding = 0.05 * (span if span > 0 else max(abs(low), abs(high), 1.0))
                    target_axis.set_ylim(low - padding, high + padding)

                set_data_ylim(trace_ax, primary_data_values)
                if self.mcd_show_integral_chk.isChecked():
                    set_data_ylim(integral_ax, integral_data_values)
                trace_ax.axhline(0, color="#555", lw=0.7)
                trace_title = f"MCD(B): E = {format_mcd_energy(e0)} eV"
                trace_ax.set_title(trace_title, pad=3)
                trace_ax.set_xlabel("B field (T)")
                trace_ax.set_ylabel("MCD (mean / absolute mean)", labelpad=10)
                trace_ax.grid(alpha=0.25)
                if self.mcd_show_integral_chk.isChecked():
                    integral_ax.set_ylabel("Integrated MCD (eV)", labelpad=2)
                else:
                    integral_ax.set_yticks([]); integral_ax.spines["right"].set_visible(False)
                visible_metric_count = sum(
                    1 for _name, _label, _color, visible in trace_specs if visible
                )
                annotation_layout = mcd_annotation_layout(
                    trace_ax, integral_ax,
                    show_conditions=False,
                    show_slopes=bool(branch_fits),
                    show_metric_legend=visible_metric_count > 1,
                )
                if branch_fits:
                    self._mcd_slope_text = self.mcd_controller._add_mcd_preview_slope_box(
                        trace_ax, branch_fits, annotation_layout["slopes"]
                    )
                branch_legend = trace_ax.legend(
                    [
                        Line2D([0], [0], color="#333", marker="o", markerfacecolor="#333", lw=1.15),
                        Line2D([0], [0], color="#333", marker="o", markerfacecolor="white", lw=1.15, ls="--"),
                    ],
                    ["B increasing", "B decreasing"],
                    title="Branch", fontsize=5.8, title_fontsize=6.0, frameon=True, framealpha=0.88,
                    loc=annotation_layout["branch_legend"],
                )
                trace_ax.add_artist(branch_legend)
                metric_handles: list[Line2D] = []
                metric_labels: list[str] = []
                for metric_name, label, color, visible in trace_specs:
                    if not visible:
                        continue
                    metric_handles.append(Line2D([0], [0], color=color, lw=1.5))
                    metric_labels.append(f"{label} (right axis)" if metric_name == "integral" else label)
                if len(metric_handles) > 1:
                    trace_ax.legend(
                        metric_handles, metric_labels, title="Metric", fontsize=5.8,
                        title_fontsize=6.0, frameon=True, framealpha=0.88,
                        loc=annotation_layout["metric_legend"],
                    )
                self._mcd_heatmap_ax, self._mcd_pair_ax = heat_ax, pair_ax
                self._mcd_spectrum_ax, self._mcd_trace_ax = linecut_ax, trace_ax
                self._mcd_integral_ax = integral_ax
                self._mcd_colorbar_ax = cax
                self.mcd_controller._configure_mcd_blitting()
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
                if self.power_controller._power_view() == "VP":
                    _kk_cube, _kkp_cube, vp_cube, _kk_records, _kkp_records, _kk_key, _kkp_key, background, _pairing, _pairs = self.power_controller._power_vp_payload()
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
                    self.power_controller._power_set_background_spin_silent(background)
                    display_cube, true_power, display_power = self.power_controller._display_power_cube(plot_cube)
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
                    self.power_controller._apply_power_tick_labels(ax1, true_power, display_power)
                    self._add_heatmap_colorbar(render, cax, label=params.cbar_label)
                    self._power_heatmap_ax = ax1
                    self._power_heatmap_axes = {"VP": ax1}
                    self._power_spectrum_ax = ax2
                    self._power_last_plot_cube = plot_cube
                    self._power_active_cubes = {"VP": plot_cube}
                    self._power_active_export_cube = plot_cube
                    self._power_active_records = ()
                    self.power_controller._update_power_compare_spectrum_and_lines({"VP": plot_cube})
                else:
                    role_compare = self.power_controller._power_has_distinct_role_groups()
                    role_titles: dict[str, str] = {}
                    if role_compare:
                        kk_result, kkp_result, kk_key, kkp_key = self.power_controller._power_role_payload()
                        background = self.power_controller._power_background_value([kk_result.cube, kkp_result.cube])
                        cubes = {
                            "KK": self.power_controller._power_corrected_cube(kk_result.cube, background=background),
                            "KKp": self.power_controller._power_corrected_cube(kkp_result.cube, background=background),
                        }
                        plot_cube = cubes["KK"]
                        role_titles = {"KK": power_group_title(kk_key), "KKp": power_group_title(kkp_key)}
                    else:
                        background = self.power_controller._power_background_value([self.loaded.cube])
                        plot_cube = self.power_controller._power_corrected_cube(self.loaded.cube, background=background)
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
                    self.power_controller._power_set_background_spin_silent(background)
                    n = len(cubes)
                    if self.power_peak_controller.is_enabled():
                        outer = self.figure.add_gridspec(
                            2, 2, width_ratios=[1.6, 1.0], hspace=.5, wspace=.5)
                        heat_cell, spectrum_cell = outer[0, 0], outer[1, 0]
                        trend_cells = [outer[0, 1], outer[1, 1]]
                        heat = heat_cell.subgridspec(1, n + 1, width_ratios=[1.] * n + [.035], wspace=.12)
                        heat_axes = [self.figure.add_subplot(heat[0, idx]) for idx in range(n)]
                        cax = self.figure.add_subplot(heat[0, n])
                        spectrum_grid = spectrum_cell.subgridspec(2, 1, height_ratios=[3, 1], hspace=.3)
                        ax2 = self.figure.add_subplot(spectrum_grid[0])
                        self.power_peak_controller.residual_axis = self.figure.add_subplot(spectrum_grid[1], sharex=ax2)
                        ax2.tick_params(labelbottom=False)
                        self.power_peak_controller.axes = [self.figure.add_subplot(cell) for cell in trend_cells]
                    else:
                        gs = self.figure.add_gridspec(
                            nrows=2, ncols=n + 1, width_ratios=([1.0] * n) + [0.035],
                            height_ratios=[1.0, 0.95], wspace=0.12, hspace=0.30)
                        heat_axes = [self.figure.add_subplot(gs[0, idx]) for idx in range(n)]
                        cax = self.figure.add_subplot(gs[0, n])
                        ax2 = self.figure.add_subplot(gs[1, :n], sharex=heat_axes[0])
                    renders: list[HeatmapRender] = []
                    for ax, (key, cube) in zip(heat_axes, cubes.items()):
                        display_cube, true_power, display_power = self.power_controller._display_power_cube(cube)
                        panel_params = HeatmapParams(**{**params.__dict__, "title": role_titles.get(key, key) if role_compare else cube.title})
                        render = plot_heatmap(ax, display_cube, panel_params)
                        self.power_controller._apply_power_tick_labels(ax, true_power, display_power)
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
                    self.power_controller._update_power_compare_spectrum_and_lines(cubes)
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
                source_files = self.compare_controller._cmp_source_mapping()
                background = self.compare_controller._cmp_background_value(raw_cubes)
                self.compare_controller._cmp_update_title_previews()
                if self.compare_controller._cmp_is_vp_view():
                    vp_cube = self.compare_controller._cmp_vp_cube(raw_cubes, source_files, background=background)
                    vp_vmin, vp_vmax = self.compare_controller._cmp_vp_color_limits()
                    vp_params = HeatmapParams(
                        title=vp_cube.title,
                        xlabel=params.xlabel,
                        ylabel=vp_cube.gate_label,
                        cbar_label="VP",
                        vmin=vp_vmin,
                        vmax=vp_vmax,
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
                    self.compare_controller._set_cmp_gate_spin_value(gate_used)
                    self.compare_controller._ensure_cmp_gate_lines({"VP": vp_cube}, gate_used)
                else:
                    cubes = self.compare_controller._cmp_corrected_cubes(raw_cubes, source_files, background=background)
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
                    self.compare_controller._set_cmp_gate_spin_value(gate_used)
                    self.compare_controller._ensure_cmp_gate_lines(cubes, gate_used)
            else:
                raise ValueError("No loaded data to plot.")

            if mode == "Compare":
                self.figure.tight_layout()
            self.canvas.draw_idle()
            self._last_plot_params_key = (
                self._current_plot_params_key(mode) if mode == "DRR" else plot_key
            )
            self._last_plot_cube = plot_cube if mode == "DRR" else None
            self.last_plotted_mode = mode
            shown_identity = self._shown_source_identity_for_loaded(self.loaded)
            self._shown_draw_mode = mode
            self._shown_draw_identity = shown_identity
            self._shown_selection_identity = self._selected_identity_for_status(mode)
            self._last_draw_identity[mode] = shown_identity
            if mode == "Compare":
                self._cmp_rendered_loaded_marker = (
                    id(self.loaded),
                    id(getattr(self.loaded, "compare_cubes", None)),
                    str(getattr(self.loaded, "folder", "")),
                    tuple(getattr(self.loaded, "selected_files", ()) or ()),
                )
            token = self._active_load_token
            self._record_workflow_event(
                workflow=mode,
                request_id=token.generation if token is not None and token.workflow == mode else 0,
                phase="first_draw",
                source_key=tuple(self.loaded.selected_files),
            )
            self._set_stage("Plotted")
            self._update_action_states()
            if mode == "DRR":
                self._status(
                    f"Plot: baseline={self.drr_controller._drr_baseline_key()}, deriv={self.drr_derivative_combo.currentText()}, "
                    f"SG(win={int(self.drr_sg_window_spin.value())}, order={int(self.drr_sg_poly_spin.value())})"
                )
            else:
                self._status(f"Plotted {mode}.")
            if not auto:
                self._append_log(f"Plotted {mode}.")
        except Exception as exc:
            self._show_error(str(exc))

    def _current_plot_params_key(self, mode: str) -> tuple[Any, ...]:
        if mode == "SHG Processing":
            settings = self.shg_controller._shg_settings_from_ui()
            fit_settings = self.shg_controller._shg_fit_settings_from_ui()
            return (
                mode,
                self.shg_controller._shg_compare_mode(),
                self.shg_controller._shg_compare_files() if self.shg_controller._shg_compare_mode() else self.shg_controller._shg_selected_file(),
                self.shg_controller._shg_compare_background_files() if self.shg_controller._shg_compare_mode() else self.shg_controller._shg_background_file(),
                *tuple(settings.to_dict().values()),
                *tuple(fit_settings.to_dict().values()),
                self.shg_compare_display_combo.currentText(),
                self.shg_spectrum_view_combo.currentText(),
                float(self.shg_angle_cursor_spin.value()),
            )
        if mode == "Power Dependent":
            return (
                mode,
                self.power_controller._power_selected_group_key(),
                self.power_compare_chk.isChecked(),
                self.power_controller._power_view(),
                self.power_controller._power_role_group_key("KK"),
                self.power_controller._power_role_group_key("KKp"),
                self.power_controller._power_pairing_mode(),
                self.power_axis_scale_combo.currentText(),
                bool(self.power_controller._power_background_auto_enabled()),
                float(self.power_controller._power_background_value(
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
                self.power_peak_controller.enabled.isChecked(),
                self.power_peak_controller.settings(),
                self.power_peak_controller.metric.currentText(),
                self.power_peak_controller.intensity_scale.currentText(),
                self.power_peak_controller.linewidth_scale.currentText(),
                bool(self.power_clip_chk.isChecked()),
                self._split_scale_key("power"),
            )
        if mode == "Compare":
            return (
                mode,
                tuple(self.compare_controller._cmp_current_mapping().items()),
                tuple(self.compare_controller._cmp_visible_channels()),
                self.compare_controller._cmp_view_mode(),
                bool(self.compare_controller._cmp_background_auto_enabled()),
                float(
                    self.compare_controller._cmp_background_value(
                        self.loaded.compare_cubes
                        if self.loaded and self.loaded.mode == "Compare" and self.loaded.compare_cubes
                        else None
                    )
                ),
                self._current_y_axis_spec_for_mode(mode),
                self._resolved_cmap(self.cmp_cmap),
                float(self.cmp_spins["vmin"].value()),
                float(self.cmp_spins["vmax"].value()),
                tuple(self.compare_controller._cmp_vp_color_limits()),
                float(self.cmp_spins["xmin"].value()),
                float(self.cmp_spins["xmax"].value()),
                float(self.cmp_spins["ymin"].value()),
                float(self.cmp_spins["ymax"].value()),
                float(self.cmp_spins["gate"].value()),
                bool(self.cmp_log_chk.isChecked()),
                bool(self.cmp_clip_chk.isChecked()),
                self._split_scale_key("cmp"),
            )
        if mode == "MCD":
            return (
                mode, self.mcd_map_combo.currentText(),
                float(self.mcd_spins["vmin"].value()), float(self.mcd_spins["vmax"].value()),
                float(self.mcd_spins["xmin"].value()), float(self.mcd_spins["xmax"].value()),
                float(self.mcd_spins["ymin"].value()), float(self.mcd_spins["ymax"].value()),
                self._resolved_cmap(self.mcd_cmap), bool(self.mcd_center_zero_chk.isChecked()),
                int(self.mcd_pair_b_combo.currentData() or 0),
                float(self.mcd_window_center_spin.value()), float(self.mcd_window_width_spin.value()),
                self.mcd_window_metric_combo.currentText(), bool(self.mcd_show_raw_chk.isChecked()),
                bool(self.mcd_show_signed_mean_chk.isChecked()),
                bool(self.mcd_show_absolute_mean_chk.isChecked()),
                bool(self.mcd_show_unsigned_absolute_mean_chk.isChecked()),
                bool(self.mcd_show_integral_chk.isChecked()), bool(self.mcd_fit_zero_chk.isChecked()),
                float(self.mcd_fit_b_window_spin.value()),
            )
        if mode != "DRR":
            return (mode, int(self.tabs.currentIndex()), self.last_plotted_mode, self._current_y_axis_spec_for_mode(mode))
        p = self.drr_controller._read_drr_params()
        return (
            "DRR", p["baseline_mode"], p["baseline_which"], p["baseline_files"], p["selected_files"], p.get("drr_assignments", ()),
            p["y_axis_spec"], p["derivative"], p["sg_window"], p["sg_poly"], p["cmap"], p["vmin"], p["vmax"],
            p["xmin"], p["xmax"], p["ymin"], p["ymax"], p["gate"], p["log"], p["clip"], p["center_zero"],
            self._drr_plot_view, bool(self._drr_side_by_side),
            float(self.drr_second_vmin_spin.value()), float(self.drr_second_vmax_spin.value()),
            self._resolved_cmap(self.drr_second_cmap), bool(getattr(self, "drr_second_auto_scale", True)),
            self._split_scale_key("drr"),
            self._split_scale_key("drr_second"),
        )

    def _ensure_loaded_matches_drr_params(self) -> bool:
        if not self.loaded or self.loaded.mode != "DRR":
            return False
        p = self.drr_controller._read_drr_params()
        selected = list(p["selected_files"])
        baselines = list(p["baseline_files"])
        baseline_text = p["baseline_mode"]
        y_axis_spec = str(p["y_axis_spec"])
        try:
            assignments = tuple(
                DrrMeasurementAssignment.from_mapping(item)
                for item in p.get("drr_assignments", ())
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid DRR background assignment: {exc}") from exc
        if not assignments and selected and not data_io.is_xlsx_map_file(selected[0]):
            if baseline_text == "Automatic":
                resolved = resolve_drr_background_assignments(
                    self.current_folder, self.drr_available_sources, selected,
                )
                if not resolved.resolved:
                    raise ValueError(resolved.reason or "DRR background assignment is unresolved.")
                assignments = resolved.assignments
            else:
                explicit_mode = baseline_text if baseline_text in {
                    "Self (first frame)", "Self (last frame)", "External",
                } else "Self (last frame)"
                explicit_which = {
                    "Last frame from each file, then average": "last",
                    "First frame from each file, then average": "first",
                    "Average all frames in each file, then average files": "all",
                }.get(str(p["baseline_which"]), "last")
                # Explicit UI choices already identify the recipe.  Materialize
                # the per-measurement assignments without re-reading the source
                # catalog; the load worker remains responsible for CSV parsing.
                try:
                    template = DrrMeasurementAssignment(
                        measurement_file=selected[0],
                        baseline_mode=explicit_mode,
                        baseline_files=tuple(baselines),
                        baseline_which=explicit_which,
                        selection_reason="explicit user baseline selection",
                    )
                except ValueError as exc:
                    raise ValueError(str(exc)) from exc
                missing = tuple(
                    source for source in template.baseline_files
                    if not resolve_source_path(self.current_folder, source).is_file()
                )
                if missing:
                    raise ValueError(
                        "Explicit DRR baseline is unavailable: " + ", ".join(missing)
                    )
                assignments = tuple(
                    DrrMeasurementAssignment(
                        measurement_file=name,
                        baseline_mode=template.baseline_mode,
                        baseline_files=template.baseline_files,
                        baseline_which=template.baseline_which,
                        selection_reason=template.selection_reason,
                    )
                    for name in selected
                )
        which_map = {
            "Last frame from each file, then average": "last",
            "First frame from each file, then average": "first",
            "Average all frames in each file, then average files": "all",
        }
        baseline_which = which_map.get(str(p["baseline_which"]), "last")
        needs_reload = (
            selected != list(self.loaded.selected_files)
            or baseline_text != self.loaded.drr_baseline_text
            or baseline_which != self.loaded.drr_baseline_which
            or baselines != list(self.loaded.baseline_files)
            or tuple(assignments) != tuple(getattr(self.loaded, "drr_assignments", ()))
            or y_axis_spec != getattr(self.loaded, "y_axis_spec", "auto")
        )
        if needs_reload:
            # A redraw must never turn a source or processing change into a
            # synchronous CSV read.  Reuse the same latest-request worker
            # path as explicit DRR selection changes.
            self._start_load("DRR")
            return False
        if False:  # retained below as a local reference for the worker path
            self.drr_controller._reject_mixed_xlsx_selection(selected)
            if selected and data_io.is_xlsx_map_file(selected[0]):
                cube = data_io.load_drr_map_cube(self.current_folder, selected[0], y_axis=y_axis_spec)
                mode_label = "DR/R Map"
                baselines = []
            elif assignments:
                cube = data_io.load_drr_resolved_cube(
                    self.current_folder, selected, assignments,
                    y_axis=y_axis_spec, derivative=None,
                )
                numerical_path = getattr(cube, "drr_numerical_path", "common")
                mode_label = (
                    "DR/R Automatic" if numerical_path != "common"
                    else "DR/R Self" if assignments[0].baseline_mode.startswith("Self")
                    else "DR/R External"
                )
                baselines = list(dict.fromkeys(
                    source for assignment in assignments for source in assignment.baseline_files
                ))
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
            numerical_path = getattr(cube, "drr_numerical_path", "common")
            selection_metadata = {
                "selection_method": (
                    "automatic_per_measurement" if baseline_text == "Automatic"
                    else "explicit_per_measurement"
                ),
                "numerical_path": numerical_path,
                "drr_background_assignments": [item.to_dict() for item in assignments],
            }
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
                drr_assignments=tuple(assignments),
                drr_background_selection=selection_metadata if assignments else {},
                y_axis_spec=y_axis_spec,
                provenance_records=self._drr_provenance_records(
                    self.current_folder, selected, baselines
                ),
            )
            # Keep the controller state in lockstep with the freshly loaded
            # state so the next parameter key/cache comparison sees the same
            # assignment identity.
            self._drr_assignments = tuple(assignments)
            self._drr_assignments_automatic = baseline_text == "Automatic"
            self._last_plot_cube = None
            self._last_plot_params_key = None
            self._apply_auto_limits_for_loaded()
            return True
        return False

    def _on_canvas_motion(self, event: Any) -> None:
        if self.last_plotted_mode == "MCD Peak Shift":
            self._update_mcd_peak_drag(event)
            return
        if self.last_plotted_mode == "MCD":
            self.mcd_controller._on_mcd_canvas_motion(event)
            return
        if self.last_plotted_mode == "SHG Processing":
            if event.inaxes is self._shg_angle_ax and event.xdata is not None:
                self.status_bar_view.set_cursor_readback(
                    f"Hover measured angle: {float(event.xdata):.6g} deg"
                )
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
            _display_cube, true_power, display_power = self.power_controller._display_power_cube(cube)
            idx = int(np.argmin(np.abs(display_power - float(event.ydata))))
            y = float(true_power[idx])
        else:
            y = float(np.clip(float(event.ydata), float(np.nanmin(ygrid)), float(np.nanmax(ygrid))))
        unit = "uW" if self.last_plotted_mode == "Power Dependent" else "V"
        label = "power" if self.last_plotted_mode == "Power Dependent" else "gate"
        self.status_bar_view.set_cursor_readback(f"Hover {label}: {y:.3f} {unit}")

    def _on_canvas_click(self, event: Any) -> None:
        if event.button != 1:
            return
        if self.last_plotted_mode == "Power Dependent" and self.power_peak_controller.click(event):
            return
        if self.last_plotted_mode == "MCD Peak Shift" and self._mcd_peak_toolbar_navigation_active():
            return
        if self.last_plotted_mode == "MCD Peak Shift" and self._on_mcd_peak_candidate_marker_click(event):
            return
        if self.last_plotted_mode == "MCD Peak Shift" and self._begin_mcd_peak_drag(event):
            return
        if self.last_plotted_mode == "MCD Peak Shift" and event.inaxes in getattr(self, "mcd_peak_map_axes", ()):
            self._on_mcd_peak_map_click(event)
            return
        if self.last_plotted_mode == "MCD Peak Shift" and event.inaxes is getattr(self, "mcd_peak_shift_ax", None):
            self._on_mcd_peak_result_click(event)
            return
        if self.last_plotted_mode == "MCD Peak Shift" and event.inaxes is getattr(self, "mcd_peak_spectrum_ax", None):
            self._on_mcd_peak_spectrum_click(event)
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
            if self.drr_controller._remove_nearest_drr_peak(float(event.xdata)):
                return
        if (
            event.xdata is not None
            and event.ydata is not None
            and self.last_plotted_mode == "DRR"
            and event.inaxes is self._drr_heatmap_ax
        ):
            if self.drr_controller._remove_peak_from_drr_heatmap_click(float(event.xdata), float(event.ydata)):
                return
        if event.xdata is not None and self.last_plotted_mode == "PL" and event.inaxes is self._pl_spectrum_ax:
            if self.pl_controller._remove_nearest_pl_peak(float(event.xdata)):
                return
        if event.xdata is not None and self.last_plotted_mode == "Power Dependent" and event.inaxes is self._power_spectrum_ax:
            return
        if (
            event.xdata is not None
            and event.ydata is not None
            and self.last_plotted_mode == "PL"
            and event.inaxes is self._pl_heatmap_ax
        ):
            if self.pl_controller._remove_peak_from_pl_heatmap_click(float(event.xdata), float(event.ydata)):
                return

        if self.last_plotted_mode == "MCD":
            self.mcd_controller._on_mcd_canvas_click(event)
            return

        if self.last_plotted_mode == "DRR":
            if self._drr_heatmap_ax is None or self._last_plot_cube is None:
                return
            clicked_cube = self._last_plot_cube
            heatmap_axes = getattr(self, "_drr_heatmap_axes", {}) or {}
            for key, axis in heatmap_axes.items():
                if event.inaxes is axis:
                    clicked_cube = getattr(self, "_drr_plot_cubes", {}).get(key, clicked_cube)
                    break
            else:
                if event.inaxes is not self._drr_heatmap_ax:
                    return
            if event.ydata is None:
                return
            ygrid = np.asarray(clicked_cube.gate, float).ravel()
            idx = int(np.argmin(np.abs(ygrid - float(event.ydata))))
            gate = float(ygrid[idx])
            self.drr_spins["gate"].setValue(gate)
            self.drr_controller._update_drr_spectrum_and_gate_line(self._last_plot_cube)
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
            self.compare_controller._set_cmp_gate_spin_value(float(ygrid[idx]))
            if not self.compare_controller._update_cmp_gate_only():
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
            _display_cube, true_power, display_power = self.power_controller._display_power_cube(cube)
            idx = int(np.argmin(np.abs(display_power - float(event.ydata))))
            power = float(true_power[idx])
            raw_indices = (np.flatnonzero(np.asarray(cube.gate) > 0)
                           if self.power_controller._power_axis_log() else np.arange(len(cube.gate)))
            self._power_selected_row_index = int(raw_indices[idx]) if len(self._power_active_cubes) == 1 else None
            blocked = self.power_spins['gate'].blockSignals(True)
            try:
                self.power_spins['gate'].setValue(power)
            finally:
                self.power_spins['gate'].blockSignals(blocked)
            self._invalidate_export_move_sources()
            self.power_controller._update_power_compare_spectrum_and_lines(self._power_active_cubes)
            return

        if self.last_plotted_mode != "PL" or self._pl_heatmap_ax is None or self._pl_last_plot_cube is None:
            return
        if event.inaxes is not self._pl_heatmap_ax or event.ydata is None:
            return
        ygrid = np.asarray(self._pl_last_plot_cube.gate, float).ravel()
        idx = int(np.argmin(np.abs(ygrid - float(event.ydata))))
        gate = float(ygrid[idx])
        self.pl_spins["gate"].setValue(gate)
        self.pl_controller._update_pl_spectrum_and_gate_line(self._pl_last_plot_cube)

    def _start_export(self, mode: str) -> None:
        if self._is_closing:
            return
        if self._export_in_progress:
            self._status("Save already in progress.")
            return
        if not self.loaded or self.loaded.mode != mode:
            self._show_error("Load and plot data before exporting.")
            return
        if self.last_plotted_mode != mode:
            self._show_error("Plot/Update before exporting.")
            return
        try:
            self._ensure_loaded_matches_ui_params(mode)
        except Exception as exc:
            self._show_error(str(exc))
            return
        # Consistency checks may invalidate a stale DRR view when a selected
        # source is missing.  They can therefore clear ``self.loaded`` while
        # returning control here; stop before building an export snapshot.
        if not self.loaded or self.loaded.mode != mode:
            self._status(f"{mode} data is no longer loaded; resolve the current selection before saving.")
            return
        if mode == "SHG Processing" and self.shg_controller._shg_reprocess_pending_for_export():
            self._status("SHG processing is updating; save is available after the update finishes.")
            return
        if (
            self._load_in_progress
            or getattr(self, "_pending_load_mode", None) == mode
        ):
            self._status(f"{mode} is updating; save is available after the current selection is displayed.")
            return
        if mode == "DRR" and not self._drr_limits_from_controls:
            # A user may zoom or pan and save immediately, without causing a
            # tab/layout rebuild.  Freeze that shared viewport in both export
            # parameter snapshots before the worker starts.
            self._capture_drr_view_limits()
        self._invalidate_export_move_sources()
        if mode == "PL":
            self._pl_last_export_source = str(self.loaded.primary_file or "")
            self._pl_export_source_was_processed = self.pl_controller._pl_source_is_processed(
                self._pl_last_export_source
            )

        power_vp_payload = None
        params_intensity: HeatmapParams | None = None
        power_records = tuple(self.loaded.power_records) if self.loaded and self.loaded.mode == "Power Dependent" else ()
        drr_deriv: int | None = None
        drr_used_win = 0
        drr_poly = 0
        drr_raw_cube = None
        drr_second_cube = None
        drr_raw_params = None
        drr_second_params = None
        drr_raw_win = drr_raw_poly = drr_second_win = drr_second_poly = 0
        if mode == "SHG Processing" and self.loaded.shg_result is not None:
            params = None
            export_cube = None
        elif mode == "MCD" and self.loaded.mcd_result is not None:
            export_cube = self.loaded.mcd_result.cube(self.mcd_map_combo.currentText())
            params = self._make_params(mode, export_cube)
        elif mode in {"PL", "DRR", "Power Dependent"} and self.loaded.cube is not None:
            if mode == "DRR":
                try:
                    drr_raw_cube, _raw_deriv, drr_raw_win, drr_raw_poly = self.drr_controller._drr_cube_with_metadata(None)
                except ValueError as exc:
                    self._show_error(f"Cannot prepare the DRR export pair: {exc}")
                    return
                drr_raw_params = self._drr_params_with_view_limits(
                    self._make_drr_params(drr_raw_cube, None)
                )
                # The second derivative is deliberately computed by the
                # owned export worker from this frozen SG snapshot.  Build its
                # presentation snapshot here without touching scientific Z.
                drr_second_win = int(self.drr_sg_window_spin.value())
                drr_second_poly = int(self.drr_sg_poly_spin.value())
                drr_second_params = HeatmapParams(
                    **{
                        **drr_raw_params.__dict__,
                        "title": f"{drr_raw_cube.title} (d2E)",
                        "cbar_label": "d2(DR/R)/dE2",
                        "cmap": self._resolved_cmap(self.drr_second_cmap),
                        "vmin": float(self.drr_second_vmin_spin.value()),
                        "vmax": float(self.drr_second_vmax_spin.value()),
                        "split_scale": self._split_scale_for_prefix("drr_second", "DRR"),
                    }
                )
                drr_second_params = self._drr_params_with_view_limits(drr_second_params)
                # Keep the active visible product available to compatibility
                # paths, while the paired fields below are authoritative.
                export_cube = drr_raw_cube
                params = drr_second_params if self._drr_plot_view == "second" else drr_raw_params
                drr_deriv = 2 if self._drr_plot_view == "second" else None
                drr_used_win = drr_second_win if drr_deriv == 2 else drr_raw_win
                drr_poly = drr_second_poly if drr_deriv == 2 else drr_raw_poly
            elif mode == "Power Dependent":
                if self.power_controller._power_view() == "VP":
                    power_vp_payload = self.power_controller._power_vp_payload()
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
                    self.power_controller._power_set_background_spin_silent(background)
                else:
                    if self.power_controller._power_has_distinct_role_groups():
                        kk_result, kkp_result, kk_key, kkp_key = self.power_controller._power_role_payload()
                        background = self.power_controller._power_background_value([kk_result.cube, kkp_result.cube])
                        kk_cube = self.power_controller._power_corrected_cube(kk_result.cube, background=background)
                        kkp_cube = self.power_controller._power_corrected_cube(kkp_result.cube, background=background)
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
                        background = self.power_controller._power_background_value([self.loaded.cube])
                        export_cube = self.power_controller._power_corrected_cube(self.loaded.cube, background=background)
                        params = self._make_params(mode, export_cube)
                        power_records = tuple(self.loaded.power_records)
                    self.power_controller._power_set_background_spin_silent(background)
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

        power_peak_payload = None
        if mode == "Power Dependent" and self.power_peak_controller.is_enabled():
            if power_vp_payload:
                peak_cubes = {"KK": power_vp_payload[0], "KKp": power_vp_payload[1]}
                peak_records = {"KK": power_vp_payload[3], "KKp": power_vp_payload[4]}
            else:
                peak_cubes = {"Power": export_cube}
                peak_records = {"Power": power_records}
            try:
                power_peak_payload = self.power_peak_controller.export_payload(
                    peak_cubes, peak_records, self.power_background_spin.value())
            except ValueError as exc:
                self._status(str(exc))
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
                drr_raw_cube=drr_raw_cube,
                drr_second_cube=drr_second_cube,
                drr_raw_params=drr_raw_params,
                drr_second_params=drr_second_params,
                drr_raw_sg_window=drr_raw_win,
                drr_raw_sg_polyorder=drr_raw_poly,
                drr_second_sg_window=drr_second_win,
                drr_second_sg_polyorder=drr_second_poly,
                drr_second_auto_scale=bool(getattr(self, "drr_second_auto_scale", True)),
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
                power_view=("Intensity Compare" if power_vp_payload and power_vp_payload[8] == "intensity" else self.power_controller._power_view()),
                power_background=float(self.power_background_spin.value()),
                power_group_key=self.loaded.power_group_key,
                power_axis_log=self.power_controller._power_axis_log(),
                power_kk_group_key=(power_vp_payload[5] if power_vp_payload else ""),
                power_kkp_group_key=(power_vp_payload[6] if power_vp_payload else ""),
                power_kk_cube=(power_vp_payload[0] if power_vp_payload else None),
                power_kkp_cube=(power_vp_payload[1] if power_vp_payload else None),
                power_vp_cube=(power_vp_payload[2] if power_vp_payload else None),
                power_kk_records=(power_vp_payload[3] if power_vp_payload else ()),
                power_kkp_records=(power_vp_payload[4] if power_vp_payload else ()),
                power_pairing_mode=(power_vp_payload[8] if power_vp_payload else self.power_controller._power_pairing_mode()),
                power_stage_pairs=(power_vp_payload[9] if power_vp_payload else ()),
                power_peak_payload=power_peak_payload,
            )
        elif mode == "SHG Processing":
            options = ExportOptions(
                mode=mode,
                params=None,
                shg_settings=self.shg_controller._shg_settings_from_ui(),
                shg_fit_settings=self.shg_controller._shg_fit_settings_from_ui(),
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
            compare_background = self.compare_controller._cmp_background_value(
                self.loaded.compare_cubes
                if self.loaded and self.loaded.mode == "Compare" and self.loaded.compare_cubes
                else None
            )
            options = ExportOptions(
                mode=mode,
                params=params,
                compare_scale_tag=self.compare_controller._cmp_scale_tag(),
                compare_clip=bool(self.cmp_clip_chk.isChecked()),
                compare_gate=float(self.cmp_spins["gate"].value()),
                compare_background=compare_background,
                compare_export_vp=True,
                compare_vp_vmin=float(self.compare_controller._cmp_vp_color_limits()[0]),
                compare_vp_vmax=float(self.compare_controller._cmp_vp_color_limits()[1]),
                cleanup_verified_sources=bool(self.clean_verified_sources_chk.isChecked()),
            )

        source_state: list[tuple[str, int | None, int | None]] = []
        source_names = [*self.loaded.selected_files, *self.loaded.baseline_files]
        source_names.extend((self.loaded.compare_sources or {}).values())
        for source_name in dict.fromkeys(source_names):
            path = resolve_source_path(self.loaded.folder, source_name)
            try:
                stat = path.stat()
                source_state.append((str(path).casefold(), stat.st_size, stat.st_mtime_ns))
            except OSError:
                source_state.append((str(path).casefold(), None, None))
        request_key = repr(
            (
                mode,
                self.loaded.folder,
                tuple(self.loaded.selected_files),
                tuple(self.loaded.baseline_files),
                self.loaded.primary_file,
                tuple(source_state),
                options,
            )
        )
        paired_drr_request = (
            mode == "DRR"
            and getattr(options, "drr_raw_cube", None) is not None
            and getattr(options, "drr_raw_params", None) is not None
        )
        if request_key == self._last_export_request_key and not paired_drr_request:
            self._status("This unchanged result is already saved; no duplicate created.")
            return

        worker = Worker(self._export_task, self.loaded, options)
        worker.signals.log.connect(self._append_log)
        worker.signals.result.connect(self._on_export_done)
        worker.signals.error.connect(self._on_export_error)
        self._export_in_progress = True
        self._active_export_request_key = request_key
        self._update_action_states()
        self._set_stage("Exporting...")
        self.thread_pool.start(worker)

    def _export_task(self, loaded: LoadedState, options: ExportOptions, *, progress: Signal, log: Signal) -> dict:
        mode = options.mode
        folder = loaded.folder
        out_folder: str | None = None
        save_status = "created"
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
            drr_inputs = [("measurement", name) for name in loaded.selected_files]
            drr_inputs.extend(("background", name) for name in loaded.baseline_files)
            # The loaded assignments and numerical path are authoritative.
            # Older metadata may contain a stale mapping from a prior load;
            # never let that dictionary overwrite the current recipe.
            is_precomputed_map = (
                loaded.drr_baseline_text == "None"
                or loaded.drr_mode_label == "DR/R Map"
            )
            drr_assignments = [
                item.to_dict() if hasattr(item, "to_dict") else dict(item)
                for item in getattr(loaded, "drr_assignments", ())
            ]
            if not drr_assignments and loaded.selected_files and not is_precomputed_map:
                if loaded.drr_baseline_text == "Automatic":
                    resolved = resolve_drr_background_assignments(
                        folder, discover_drr_sources(folder), loaded.selected_files,
                    )
                else:
                    resolved = resolve_drr_background_assignments(
                        folder, discover_drr_sources(folder), loaded.selected_files,
                        explicit_baseline_files=loaded.baseline_files,
                        explicit_baseline_mode=loaded.drr_baseline_text,
                        explicit_baseline_which=loaded.drr_baseline_which,
                    )
                if not resolved.resolved:
                    raise ValueError(resolved.reason or "DRR background assignment is unresolved.")
                drr_assignments = [item.to_dict() for item in resolved.assignments]
            current_selection = dict(getattr(loaded, "drr_background_selection", {}) or {})
            for key in (
                "mode", "baseline_selection", "baseline_which", "average_count",
                "average_method", "derivative_order", "savgol_window",
                "savgol_polyorder", "derivative_grid", "y_axis",
                "drr_background_assignments", "selection_method", "numerical_path",
            ):
                current_selection.pop(key, None)
            export_numerical_path = (
                getattr(loaded.cube, "drr_numerical_path", None)
                or current_selection.get("numerical_path", "common")
            )
            drr_processing = {
                "mode": loaded.drr_mode_label,
                "baseline_selection": loaded.drr_baseline_text,
                "baseline_which": loaded.drr_baseline_which,
                "average_count": len(loaded.selected_files),
                "average_method": (
                    "per-file dR/R, align to first measurement grid without extrapolation, then nanmean"
                    if export_numerical_path == "heterogeneous"
                    else "per-file dR/R, then nanmean"
                ),
                "y_axis": loaded.y_axis_spec,
            }
            if not is_precomputed_map:
                drr_processing.update({
                    "drr_background_assignments": drr_assignments,
                    "selection_method": (
                        "automatic_per_measurement"
                        if loaded.drr_baseline_text == "Automatic"
                        else "explicit_per_measurement"
                    ),
                    "numerical_path": export_numerical_path,
                    **current_selection,
                })
            pair_requested = options.drr_raw_cube is not None and options.drr_raw_params is not None
            if pair_requested:
                raw_product = options.drr_raw_cube
                raw_params = options.drr_raw_params
                second_product = options.drr_second_cube
                second_window = int(options.drr_second_sg_window or options.drr_sg_window)
                second_poly = int(options.drr_second_sg_polyorder or options.drr_sg_polyorder)
                if second_product is None:
                    # Export receives immutable SG settings and performs the
                    # expensive transform off the GUI thread.
                    second_product, second_window = apply_sg_derivative_energy(
                        loaded.cube, derivative=2, window_length=second_window,
                        polyorder=second_poly,
                    )
                    second_product.gate_unit = getattr(loaded.cube, "gate_unit", "")
                    second_product.y_axis_semantic = getattr(loaded.cube, "y_axis_semantic", "")
                second_params = options.drr_second_params
                if second_params is None:
                    second_params = HeatmapParams(
                        **{
                            **raw_params.__dict__,
                            "title": f"{second_product.title}",
                            "cbar_label": "d2(DR/R)/dE2",
                            "split_scale": None,
                        }
                    )
                if options.drr_second_auto_scale and second_params.split_scale is None:
                    try:
                        second_limits = compute_auto_limits(second_product, log_scale=False)
                        second_params = HeatmapParams(
                            **{
                                **second_params.__dict__,
                                "vmin": float(second_limits.vmin),
                                "vmax": float(second_limits.vmax),
                            }
                        )
                    except (TypeError, ValueError):
                        pass
                raw_base = build_drr_export_base(
                    loaded.primary_file, len(loaded.selected_files), loaded.drr_mode_label,
                    None, int(options.drr_raw_sg_window or options.drr_sg_window),
                    int(options.drr_raw_sg_polyorder or options.drr_sg_polyorder), options.drr_sg_mode_label,
                )
                second_base = build_drr_export_base(
                    loaded.primary_file, len(loaded.selected_files), loaded.drr_mode_label,
                    2, int(second_window), second_poly, options.drr_sg_mode_label,
                )
                raw_processing = {
                    **drr_processing, "derivative_order": None,
                    "savgol_window": int(options.drr_raw_sg_window or options.drr_sg_window),
                    "savgol_polyorder": int(options.drr_raw_sg_polyorder or options.drr_sg_polyorder),
                    "derivative_grid": options.drr_sg_mode_label,
                }
                second_processing = {
                    **drr_processing, "derivative_order": 2,
                    "savgol_window": int(second_window), "savgol_polyorder": int(second_poly),
                    "derivative_grid": options.drr_sg_mode_label,
                }
                pair_paths = export_drr_pair_pngs_and_dat(
                    folder, raw_cube=raw_product, second_cube=second_product,
                    raw_params=raw_params, second_params=second_params,
                    raw_export_base=raw_base, second_export_base=second_base,
                    processed_name=str(Path("Processed Data") / "DRR"),
                    metadata_input_files=drr_inputs,
                    metadata_processing_raw=raw_processing,
                    metadata_processing_second=second_processing,
                    metadata_extra=metadata_extra,
                )
                save_status = getattr(pair_paths, "save_status", "created")
                log.emit(
                    f"DRR pair: {pair_paths['raw_dat'].name}, {pair_paths['raw_png'].name}; "
                    f"{pair_paths['second_dat'].name}, {pair_paths['second_png'].name}"
                )
                out_folder = str(pair_paths["raw_png"].parent)
                paths = pair_paths
            elif options.drr_cube is not None and options.params is not None:
                # Compatibility for direct callers that still request one
                # DRR product. Normal GUI saves always use the pair fields.
                base = build_drr_export_base(
                    loaded.primary_file, len(loaded.selected_files), loaded.drr_mode_label,
                    options.drr_derivative_order, options.drr_sg_window,
                    options.drr_sg_polyorder, options.drr_sg_mode_label,
                )
                paths = export_drr_png_and_dat(
                    folder, cube=options.drr_cube, params=options.params, export_base=base,
                    processed_name=str(Path("Processed Data") / "DRR"),
                    metadata_input_files=drr_inputs,
                    metadata_processing={
                        **drr_processing,
                        "derivative_order": options.drr_derivative_order,
                        "savgol_window": int(options.drr_sg_window),
                        "savgol_polyorder": int(options.drr_sg_polyorder),
                        "derivative_grid": options.drr_sg_mode_label,
                    },
                    metadata_extra=metadata_extra,
                )
                save_status = getattr(paths, "save_status", "created")
                out_folder = str(paths["png"].parent)
                log.emit(f"DRR: {paths['dat'].name} and {paths['png'].name}")
            # Canonical Initial Data and historical background files are never
            # offered for archival. Only verified legacy root working copies are.
            files_to_move = [
                Path(record.working_copy_path).name
                for record in loaded.provenance_records
                if record.temporary_working_copy
                and Path(record.working_copy_path).parent.resolve() == Path(folder).resolve()
            ]
        elif mode == "MCD" and options.drr_cube is not None and loaded.mcd_result is not None and loaded.primary_file:
            mcd_export_started = perf_counter()
            package_dir = ensure_mcd_package_dir(folder, loaded.primary_file)
            package_subfolder = str(Path("Processed Data") / "MCD" / package_dir.name)
            base = (
                f"{Path(loaded.primary_file).stem}_MCD_"
                f"{options.mcd_map_name.replace(' ', '_')}"
            )
            # Use the same title and colorbar presentation as a PL export:
            # the source-file stem is the title, while the MCD map identity is
            # carried by the export filename (for example, ``_MCD_Combo``).
            mcd_export_params = HeatmapParams(
                **{**options.params.__dict__, "title": Path(loaded.primary_file).stem}
            )
            map_export_started = perf_counter()
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
                    # The 2D Combo map is independent of the integration
                    # center. Processing settings remain in its fingerprint so
                    # a genuinely changed MCD analysis still creates a result.
                    "mcd_settings": options.mcd_settings or McdSettings(),
                },
                metadata_extra={"package": package_dir.name},
                reuse_existing_analysis=True,
            )
            map_elapsed = perf_counter() - map_export_started
            analysis_export_started = perf_counter()
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
                experiment_root=folder,
                package_outputs=(
                    paths["png"],
                    paths["dat"],
                    paths["dat"].with_suffix(".metadata.json"),
                ),
            )
            analysis_elapsed = perf_counter() - analysis_export_started
            if getattr(paths, "save_status", "created") == "reused":
                log.emit(f"Reused center-independent MCD map: {paths['png'].name}, {paths['dat'].name}")
            elif getattr(paths, "save_status", "created") == "updated":
                log.emit(f"Updated MCD map presentation: {paths['png'].name}; reused {paths['dat'].name}")
            else:
                log.emit(f"Exported PNG: {paths['png'].name}")
                log.emit(f"Exported DAT: {paths['dat'].name}")
            log.emit(f"Exported PNG: {analysis_paths['mcd_vs_b_png'].name}")
            log.emit(f"Exported CSV: {analysis_paths['mcd_vs_b_csv'].name}, {analysis_paths['pair_diagnostics'].name}")
            log.emit(f"Exported settings: {analysis_paths['settings'].name}")
            log.emit(
                f"MCD save timing: map/data {map_elapsed:.2f}s; "
                f"center trace {analysis_elapsed:.2f}s; "
                f"total {perf_counter() - mcd_export_started:.2f}s"
            )
            out_folder = str(paths["png"].parent)
            # Dedicated mcd/ acquisition files are canonical sources and must
            # remain in place. Preserve the legacy move option only for old
            # root-level working copies.
            primary_path = Path(loaded.primary_file)
            files_to_move = (
                [loaded.primary_file]
                if not primary_path.is_absolute() and primary_path.name == loaded.primary_file
                else []
            )
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
            if options.power_peak_payload is not None:
                from core.power_peaks import export_peak_analysis
                peak_path = export_peak_analysis(out_folder, options.power_peak_payload)
                log.emit(f"Exported peak analysis: {peak_path.name}, PNG and JSON settings")
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
                vp_vmin=options.compare_vp_vmin,
                vp_vmax=options.compare_vp_vmax,
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
            "mode": mode,
            "save_status": save_status,
        }

    def _on_export_error(self, message: str) -> None:
        self._export_in_progress = False
        self._active_export_request_key = ""
        self._pl_export_source_was_processed = False
        self._pl_last_export_source = ""
        self._update_action_states()
        self._set_stage("Export failed")
        self._show_error(message)

    def _on_export_done(self, result: object) -> None:
        self._export_in_progress = False
        self._last_export_request_key = self._active_export_request_key
        self._active_export_request_key = ""
        self._update_action_states()
        if isinstance(result, dict):
            out_folder = str(result.get("out_folder", self.current_folder or ""))
            moved = int(result.get("moved", 0))
            source_files = list(result.get("source_files", []))
            folder = str(result.get("folder", self.current_folder or ""))
            auto_moved = bool(result.get("auto_moved", False))
            cleaned = int(result.get("cleaned", 0))
            exported_mode = str(result.get("mode", ""))
            save_status = str(result.get("save_status", "created"))
        else:
            out_folder = str(result)
            moved = 0
            source_files = []
            folder = self.current_folder or ""
            auto_moved = False
            cleaned = 0
            exported_mode = ""
            save_status = "created"
        if moved > 0:
            self._refresh_file_lists()
        if source_files and not auto_moved:
            self._set_export_move_sources(folder, source_files)
        else:
            self._invalidate_export_move_sources()
        self._set_stage("Already saved" if save_status == "reused" else "Exported")
        if exported_mode in {"PL", "DRR"} and self.current_folder:
            self._refresh_file_lists(auto=True, mode=exported_mode)
        elif exported_mode == "SHG Processing" and self.current_folder:
            # SHG history lives in export metadata; rescan it after a
            # successful save so the source selector reflects this package.
            self._refresh_file_lists(auto=True, mode=exported_mode)
        elif exported_mode == "MCD" and self.current_folder:
            # Center-only saves do not change raw source discovery. Refreshing
            # just the MCD saved-state avoids rescanning every workflow.
            self.mcd_controller._mcd_refresh_sources()
        elif exported_mode == "Compare" and self.current_folder:
            self.compare_controller._cmp_refresh_history_cache(force=True)
            self.compare_controller._cmp_update_group_badge()
        suffix = f"; cleaned {cleaned} verified source copy(s)" if cleaned else ""
        if save_status == "reused":
            self._status(f"Already saved—no duplicate created: {out_folder}{suffix}")
        elif save_status == "updated":
            self._status(f"Existing DRR result updated: {out_folder}{suffix}")
        else:
            self._status(f"Export completed: {out_folder}{suffix}")
        if (
            exported_mode == "PL"
            and not self._pl_export_source_was_processed
            and self._pl_last_export_source
        ):
            self.pl_controller._auto_load_next_unprocessed_pl(self._pl_last_export_source)
        if exported_mode == "PL":
            self._pl_export_source_was_processed = False
            self._pl_last_export_source = ""
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
        if self._is_closing or not self._auto_update_check_enabled():
            return
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(1500)
        timer.timeout.connect(self._run_automatic_update_check)
        self._automatic_update_timer = timer
        timer.start()
    def _run_automatic_update_check(self) -> None:
        if getattr(self, "_is_closing", False):
            return
        self._start_update_check(automatic=True)
    def _manual_check_updates(self) -> None:
        self._start_update_check(automatic=False)
    def _start_update_check(self, *, automatic: bool) -> None:
        if self._is_closing or (automatic and not self._auto_update_check_enabled()):
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
        if self._is_closing or self._download_in_progress:
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
        if self._is_closing:
            return
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
