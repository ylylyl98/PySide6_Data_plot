"""Focused standalone UI for organizing and exporting processed MCD results."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThreadPool, QTimer, Qt, QLockFile, QSaveFile, QIODevice
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.mcd_extract import (
    BRANCHES,
    SLOPE_METRICS,
    PALETTES,
    McdBranch,
    McdSeries,
    ProcessedMcdRecord,
    assign_plot_colors,
    concise_condition_labels,
    discover_processed_mcd,
    load_branch_traces,
    newest_mcd_versions,
    organize_mcd_series,
    order_mcd_records,
    record_order_value,
)
from ui_qt.mcd_async import McdScanWorker
from ui_qt.common import Worker
from ui_qt.export_workers import OwnedWorkerPool
from core.mcd_extract_export import mcd_extract_export_worker
from ui_qt.fluent_ui.style import set_fluent_property
from ui_qt.theme import alias as theme_alias


def _number(value: float | None) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{float(value):.6g}"


_MISSING_SELECTION = object()


def _merge_selection_changes(base, local, remote):
    """Apply only local changes; untouched values retain other windows' edits."""
    if local == base:
        return remote
    if all(isinstance(value, list) for value in (base, local, remote)):
        # Persisted lists contain record IDs, so independent include/exclude
        # operations can be merged without restoring another window's exclusions.
        return sorted((set(remote) - (set(base) - set(local))) | (set(local) - set(base)))
    if isinstance(local, dict) and (isinstance(base, dict) or base is _MISSING_SELECTION):
        previous = base if isinstance(base, dict) else {}
        result = dict(remote) if isinstance(remote, dict) else {}
        for key in previous.keys() | local.keys():
            value = _merge_selection_changes(previous.get(key, _MISSING_SELECTION),
                                             local.get(key, _MISSING_SELECTION),
                                             result.get(key, _MISSING_SELECTION))
            if value is _MISSING_SELECTION:
                result.pop(key, None)
            else:
                result[key] = value
        return result
    return local


class McdOrganizerWindow(QMainWindow):
    """Series-first MCD extraction window with optional record-level details."""

    DETAIL_COLUMNS = (
        "Source", "Series value", "Energy (eV)", "Width (meV)",
        "Measured T (K)", "Increasing slope", "Decreasing slope",
    )

    def __init__(self, experiment_root: str | Path, *, auto_scan: bool = True):
        super().__init__()
        self.setWindowTitle("MCD Organizer")
        self.resize(1180, 780)
        self.setMinimumSize(880, 620)
        self.experiment_root = Path(experiment_root).expanduser()
        self.all_records: list[ProcessedMcdRecord] = []
        self.records: list[ProcessedMcdRecord] = []
        self.older_records: list[ProcessedMcdRecord] = []
        self._omitted_no_slope = 0
        self.series_groups: list[McdSeries] = []
        self._selected_record_ids: dict[str, set[str]] = {}
        self._default_palette = "tab10"
        self._focused_record_id: str | None = None
        self._group_scopes = {}
        self._scope_series_bindings = {}
        self._legacy_energy_groups = {}
        self._orphan_group_overrides = {}
        self._loaded_selection_path = None
        self._selection_baseline = {}
        self._energy_point_artists = {}
        self._energy_focus_artists = []
        self._energy_hover = None
        self._plot_artists: dict[str, list[object]] = {}
        self._slope_lines: dict[str, object] = {}
        self._trace_array_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self._trace_cache_limit = 128
        self._list_refreshing = False
        self._scan_running = False
        self._scan_pending = False
        self._scan_pending_rebuild = False
        self._scan_generation = 0
        self._scan_workers: list[McdScanWorker] = []
        self._thread_pool = QThreadPool.globalInstance()
        self._export_pool = OwnedWorkerPool(self)
        self._export_worker: Worker | None = None
        self._closing = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(60)
        self._preview_timer.timeout.connect(self._update_preview)
        self._selection_save_timer = QTimer(self)
        self._selection_save_timer.setSingleShot(True)
        self._selection_save_timer.setInterval(300)
        self._selection_save_timer.timeout.connect(self._write_condition_selections)
        self._build_ui()
        self._load_saved_condition_selections()
        self._show_empty_preview("Loading processed MCD catalog…")
        if auto_scan:
            QTimer.singleShot(50, self._scan)
        else:
            self._init_plot_widgets()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        folder_row = QHBoxLayout()
        self.folder_label = QLabel(str(self.experiment_root))
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.folder_label.setToolTip(str(self.experiment_root))
        self.choose_folder_btn = QPushButton("Change folder…")
        self.refresh_btn = QPushButton("Refresh")
        self.rebuild_btn = QPushButton("Rebuild catalog")
        self.rebuild_btn.setToolTip("Inspect every saved result again instead of using cached metadata.")
        folder_row.addWidget(QLabel("Experiment:"))
        folder_row.addWidget(self.folder_label, 1)
        folder_row.addWidget(self.choose_folder_btn)
        folder_row.addWidget(self.refresh_btn)
        folder_row.addWidget(self.rebuild_btn)
        layout.addLayout(folder_row)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_series_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 800])
        layout.addWidget(splitter, 1)

        export_row = QHBoxLayout()
        self.output_label = QLabel()
        self.output_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.output_btn = QPushButton("Change output…")
        self.export_csv_chk = QCheckBox("Also create branch CSV files")
        self.export_btn = QPushButton("Export selected series")
        self.theta_compare_btn = QPushButton('Compare θCW…')
        self.theta_compare_btn.clicked.connect(self._compare_theta)
        self.export_btn.setMinimumWidth(190)
        export_row.addWidget(QLabel("Export to:"))
        export_row.addWidget(self.output_label, 1)
        export_row.addWidget(self.output_btn)
        export_row.addWidget(self.export_csv_chk)
        export_row.addWidget(self.export_btn)
        export_row.addWidget(self.theta_compare_btn)
        layout.addLayout(export_row)

        self.choose_folder_btn.clicked.connect(self._choose_folder)
        self.refresh_btn.clicked.connect(lambda: self._scan(rebuild_catalog=False))
        self.rebuild_btn.clicked.connect(lambda: self._scan(rebuild_catalog=True))
        self.output_btn.clicked.connect(self._choose_output)
        self.export_btn.clicked.connect(self._export)
        self._set_default_output()

    def _build_series_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 4, 0)
        title = QLabel("Processed condition series")
        set_fluent_property(title, "appRole", "pageHeading")
        layout.addWidget(title)
        hint = QLabel(
            "Check the series to export. Click a series to preview it. "
            "Every result keeps its own processed energy."
        )
        hint.setWordWrap(True)
        set_fluent_property(hint, "appRole", "hintText")
        layout.addWidget(hint)
        compare_row = QHBoxLayout()
        compare_row.addWidget(QLabel("Compare different:"))
        self.compare_combo = QComboBox()
        for label, value in (
            ("E-field (same doping and temperature)", "E-field"),
            ("Temperature (same doping and E-field)", "Temperature"),
            ("Doping", "Doping"),
            ("Top-gate voltage", "Vtg"),
            ("Back-gate voltage", "Vbg"),
            ("Bias voltage", "Vbias"),
            ("Processed energy", "Energy"),
            ("Auto-detect", "Auto"),
        ):
            self.compare_combo.addItem(label, value)
        compare_row.addWidget(self.compare_combo, 1)
        layout.addLayout(compare_row)
        self.slope_checks = {}
        for metric, label in SLOPE_METRICS.items():
            check = QCheckBox(label)
            check.setChecked(metric == "near_zero")
            check.setToolTip("Show or hide this slope in the preview. Exports always include all three metrics and both branches.")
            check.toggled.connect(self._slope_visibility_changed)
            self.slope_checks[metric] = check
            layout.addWidget(check)
        group_row = QHBoxLayout()
        group_row.addWidget(QLabel('Energy groups (meV):'))
        self.energy_group_tolerance = QDoubleSpinBox()
        self.energy_group_tolerance.setRange(.1, 100)
        self.energy_group_tolerance.setValue(5.)
        self.energy_group_tolerance.setToolTip('Used for new points and initial grouping. Existing groups keep their names; use Reset groups to regroup them. Saved integration centers, not fitted peaks.')
        group_row.addWidget(self.energy_group_tolerance)
        self.edit_energy_groups_btn = QPushButton('Edit groups…')
        self.edit_energy_groups_btn.clicked.connect(self._edit_energy_groups)
        group_row.addWidget(self.edit_energy_groups_btn)
        layout.addLayout(group_row)
        self.energy_group_tolerance.valueChanged.connect(self._energy_groups_changed)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Find a condition, value, or series…")
        layout.addWidget(self.search_edit)
        energy_row = QHBoxLayout()
        self.energy_filter_chk = QCheckBox("Energy range")
        self.energy_min_spin = QDoubleSpinBox()
        self.energy_max_spin = QDoubleSpinBox()
        for spin, value in ((self.energy_min_spin, 0.0), (self.energy_max_spin, 10.0)):
            spin.setRange(0.0, 10.0)
            spin.setDecimals(6)
            spin.setValue(value)
            spin.setSuffix(" eV")
            spin.setEnabled(False)
        energy_row.addWidget(self.energy_filter_chk)
        energy_row.addWidget(self.energy_min_spin)
        energy_row.addWidget(QLabel("to"))
        energy_row.addWidget(self.energy_max_spin)
        layout.addLayout(energy_row)
        self.series_list = QListWidget()
        self.series_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.series_list.setWordWrap(True)
        self.series_list.setSpacing(3)
        layout.addWidget(self.series_list, 1)
        select_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Check all")
        self.clear_btn = QPushButton("Clear")
        select_row.addWidget(self.select_all_btn)
        select_row.addWidget(self.clear_btn)
        select_row.addStretch(1)
        layout.addLayout(select_row)
        self.newest_chk = QCheckBox("Use newest processing versions only")
        self.newest_chk.setChecked(True)
        self.newest_chk.setToolTip(
            "Only identical source/energy/width reprocessing is hidden. "
            "Different processed energies remain separate results."
        )
        layout.addWidget(self.newest_chk)
        self.selection_summary = QLabel("No processed MCD series found.")
        self.selection_summary.setWordWrap(True)
        layout.addWidget(self.selection_summary)

        self.search_edit.textChanged.connect(self._filter_series_list)
        self.energy_filter_chk.toggled.connect(self._energy_filter_changed)
        self.energy_min_spin.valueChanged.connect(lambda _value: self._regroup())
        self.energy_max_spin.valueChanged.connect(lambda _value: self._regroup())
        self.compare_combo.currentIndexChanged.connect(lambda _index: self._regroup())
        self.series_list.currentItemChanged.connect(self._current_series_changed)
        self.series_list.itemChanged.connect(lambda _item: self._update_export_summary())
        self.select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        self.clear_btn.clicked.connect(lambda: self._set_all_checked(False))
        self.newest_chk.toggled.connect(lambda _checked: self._apply_versions())
        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 0, 0)
        header = QHBoxLayout()
        self.preview_title = QLabel("Select a series to preview")
        set_fluent_property(self.preview_title, "appRole", "pageHeading")
        self.increasing_chk = QCheckBox("Increasing")
        self.decreasing_chk = QCheckBox("Decreasing")
        self.increasing_chk.setChecked(True)
        self.decreasing_chk.setChecked(True)
        self.palette_combo = QComboBox()
        for palette in PALETTES:
            self.palette_combo.addItem(palette, palette)
        self.palette_default_btn = QPushButton("Set as default")
        self.palette_default_btn.setToolTip(
            "Use the current color palette automatically for this experiment next time."
        )
        self.details_btn = QPushButton("Show result details")
        self.details_btn.setCheckable(True)
        header.addWidget(self.preview_title, 1)
        header.addWidget(self.increasing_chk)
        header.addWidget(self.decreasing_chk)
        header.addWidget(QLabel("Colors"))
        header.addWidget(self.palette_combo)
        header.addWidget(self.palette_default_btn)
        header.addWidget(self.details_btn)
        layout.addLayout(header)

        self.details_table = QTableWidget(0, len(self.DETAIL_COLUMNS))
        self.details_table.setHorizontalHeaderLabels(list(self.DETAIL_COLUMNS))
        self.details_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.details_table.setAlternatingRowColors(True)
        self.details_table.verticalHeader().setVisible(False)
        self.details_table.setMaximumHeight(190)
        self.details_table.setVisible(False)
        self.details_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.details_table)

        self.conditions_panel = QWidget()
        conditions_layout = QVBoxLayout(self.conditions_panel)
        conditions_layout.setContentsMargins(4, 0, 0, 0)
        conditions_layout.addWidget(QLabel("Conditions in selected series"))
        self.conditions_btn = QPushButton("Conditions")
        self.conditions_btn.setCheckable(True)
        self.conditions_btn.setChecked(True)
        header.addWidget(self.conditions_btn)
        self.conditions_btn.toggled.connect(self.conditions_panel.setVisible)
        self.condition_list = QListWidget()
        self.condition_list.setMinimumWidth(280)
        self.condition_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.condition_list.setWordWrap(True)
        self.condition_list.setSpacing(1)
        condition_row = QHBoxLayout()
        self.condition_exclude_btn = QPushButton("Exclude")
        self.condition_restore_selected_btn = QPushButton("Restore")
        self.condition_restore_btn = QPushButton("Restore all")
        self.condition_exclude_btn.setEnabled(False)
        self.condition_restore_selected_btn.setEnabled(False)
        condition_row.addWidget(self.condition_exclude_btn)
        condition_row.addWidget(self.condition_restore_selected_btn)
        condition_row.addWidget(self.condition_restore_btn)
        condition_row.addStretch(1)
        conditions_layout.addLayout(condition_row)
        group_row = QHBoxLayout()
        self.assign_group_btn = QPushButton("Assign group…")
        self.reset_group_btn = QPushButton("Reset grouping")
        group_row.addWidget(self.assign_group_btn)
        group_row.addWidget(self.reset_group_btn)
        conditions_layout.addLayout(group_row)
        count_row = QHBoxLayout()
        self.fixed_group_count_chk = QCheckBox('Fixed group count')
        self.group_count_spin = QSpinBox()
        self.group_count_spin.setRange(1, 100)
        self.group_count_spin.setValue(3)
        self.group_count_spin.setKeyboardTracking(False)
        self.group_count_spin.setEnabled(False)
        self.fixed_group_count_chk.setToolTip('Only Group 1–N are allowed in this series. Out-of-range assignments become unassigned; no automatic merging.')
        count_row.addWidget(self.fixed_group_count_chk)
        count_row.addWidget(self.group_count_spin)
        conditions_layout.addLayout(count_row)
        self.fixed_group_count_chk.toggled.connect(self._group_count_changed)
        self.group_count_spin.valueChanged.connect(self._group_count_changed)
        self.assign_group_btn.clicked.connect(self._assign_selected_group)
        self.reset_group_btn.clicked.connect(self._reset_series_groups)
        self.condition_list.setToolTip("Ctrl / Shift: select multiple windows, then Assign group")
        conditions_layout.addWidget(self.condition_list, 1)
        self.condition_summary = QLabel("0 of 0 conditions included")
        set_fluent_property(self.condition_summary, "appRole", "hintText")
        conditions_layout.addWidget(self.condition_summary)

        self.figure = None
        self.canvas = None
        self.slope_figure = None
        self.slope_canvas = None
        self.preview_tabs = QTabWidget()
        mcd_tab = QWidget()
        mcd_layout = QVBoxLayout(mcd_tab)
        mcd_layout.setContentsMargins(0, 0, 0, 0)
        mcd_layout.addWidget(QLabel("Preparing preview…"), 1)
        self.mcd_group_combo = QComboBox()
        self.mcd_group_combo.addItem('All groups', '')
        self.mcd_group_combo.setToolTip('Filter MCD vs B only; export includes all groups')
        self.mcd_group_combo.currentIndexChanged.connect(lambda _i: self._request_preview_update())
        mcd_layout.addWidget(self.mcd_group_combo)

        slope_tab = QWidget()
        slope_layout = QVBoxLayout(slope_tab)
        slope_layout.setContentsMargins(0, 0, 0, 0)
        slope_layout.addWidget(QLabel("Preparing preview…"), 1)
        self._mcd_plot_layout = mcd_layout
        self._slope_plot_layout = slope_layout
        self.preview_tabs.addTab(mcd_tab, "MCD vs B")
        self.preview_tabs.addTab(slope_tab, "Window energy / slopes")
        from ui_qt.curie_weiss_panel import CurieWeissPanel
        self.curie_weiss_panel = CurieWeissPanel()
        self.preview_tabs.addTab(self.curie_weiss_panel, "Curie–Weiss")
        self.preview_splitter = QSplitter(Qt.Horizontal)
        self.preview_splitter.addWidget(self.preview_tabs)
        self.preview_splitter.addWidget(self.conditions_panel)
        self.preview_splitter.setStretchFactor(0, 1)
        self.preview_splitter.setStretchFactor(1, 0)
        self.preview_splitter.setChildrenCollapsible(False)
        self.preview_splitter.setSizes([850, 340])
        layout.addWidget(self.preview_splitter, 1)
        QTimer.singleShot(0, self._init_plot_widgets)

        self.increasing_chk.toggled.connect(lambda _checked: self._request_preview_update())
        self.decreasing_chk.toggled.connect(lambda _checked: self._request_preview_update())
        self.palette_combo.currentIndexChanged.connect(self._energy_palette_changed)
        self.palette_default_btn.clicked.connect(self._set_current_palette_default)
        self.details_btn.toggled.connect(self._toggle_details)
        self.condition_list.currentRowChanged.connect(lambda _row: self._condition_focus_changed())
        self.condition_exclude_btn.clicked.connect(self._exclude_focused_condition)
        self.condition_restore_selected_btn.clicked.connect(self._restore_focused_condition)
        self.condition_restore_btn.clicked.connect(self._restore_all_conditions)
        return panel

    def _init_plot_widgets(self) -> None:
        if self.figure is not None:
            return
        # Import and construct matplotlib only after Qt has had a chance to paint
        # the organizer window, keeping startup responsive on cold launches.
        from matplotlib.figure import Figure
        from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
        self.figure = Figure(figsize=(9, 5), dpi=100, facecolor="white")
        self.canvas = ThemeAwareFigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect('resize_event', lambda event: self._request_preview_update())
        self.slope_figure = Figure(figsize=(9, 5), dpi=100, facecolor="white")
        self.slope_canvas = ThemeAwareFigureCanvasQTAgg(self.slope_figure)
        self.slope_canvas.mpl_connect('motion_notify_event', self._energy_plot_hover)
        self.slope_canvas.mpl_connect('button_press_event', self._energy_plot_click)
        self._mcd_plot_layout.replaceWidget(self._mcd_plot_layout.itemAt(0).widget(), self.canvas)
        self._slope_scroll = QScrollArea()
        self._slope_scroll.setWidgetResizable(True)
        self._slope_scroll.setWidget(self.slope_canvas)
        self._slope_plot_layout.replaceWidget(self._slope_plot_layout.itemAt(0).widget(), self._slope_scroll)
        self._show_empty_preview("Loading processed MCD catalog…")

    def _set_default_output(self) -> None:
        output = self.experiment_root / "Processed Data" / "MCD Extracts"
        self.output_label.setText(str(output))
        self.output_label.setToolTip(str(output))

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose experiment or processed MCD folder", str(self.experiment_root)
        )
        if folder:
            self._selection_save_timer.stop()
            self._write_condition_selections()
            self.experiment_root = Path(folder)
            self.series_groups = []
            self.series_list.clear()
            self._selected_record_ids.clear()
            self._group_scopes.clear()
            self._scope_series_bindings.clear()
            self._legacy_energy_groups.clear()
            self._loaded_selection_path = None
            self.folder_label.setText(str(self.experiment_root))
            self.folder_label.setToolTip(str(self.experiment_root))
            self._set_default_output()
            self._scan()

    def _choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose MCD export folder", self.output_label.text()
        )
        if folder:
            self.output_label.setText(folder)
            self.output_label.setToolTip(folder)

    def _scan(self, rebuild_catalog: bool = False) -> None:
        self._trace_array_cache.clear()
        self._plot_artists.clear()
        self._slope_lines.clear()
        if self._scan_running:
            self._scan_pending = True
            self._scan_pending_rebuild = self._scan_pending_rebuild or bool(rebuild_catalog)
            self.selection_summary.setText("Waiting for the current MCD catalog scan…")
            return
        if self._loaded_selection_path == self._selection_settings_path():
            self._selection_save_timer.stop()
            self._write_condition_selections()
        self._scan_running = True
        self._scan_pending = False
        self._scan_pending_rebuild = False
        self._scan_generation += 1
        generation = self._scan_generation
        self.refresh_btn.setEnabled(False)
        self.rebuild_btn.setEnabled(False)
        self.selection_summary.setText("Loading processed MCD catalog…")
        worker = McdScanWorker(self.experiment_root, rebuild_catalog=bool(rebuild_catalog))
        self._scan_workers.append(worker)
        worker.signals.result.connect(lambda payload, g=generation: self._on_scan_result(g, payload))
        worker.signals.error.connect(lambda message, g=generation: self._on_scan_error(g, message))
        worker.signals.finished.connect(lambda w=worker: self._on_scan_finished(w))
        self._thread_pool.start(worker)

    def _on_scan_result(self, generation: int, payload: tuple[str, list[ProcessedMcdRecord]]) -> None:
        if self._closing:
            return
        root, records = payload
        if generation != self._scan_generation or root.casefold() != str(self.experiment_root.resolve()).casefold():
            return
        self.all_records = list(records)
        try:
            # The organizer is intended for slope-aware comparisons. Older
            # results may still contain traces but lack one or both stored
            # branch fits; omit those from the selectable dataset.
            before_slope_filter = len(self.all_records)
            self.all_records = [
                record for record in self.all_records
                if record.increasing_slope_per_t is not None
                and record.decreasing_slope_per_t is not None
            ]
            self._omitted_no_slope = before_slope_filter - len(self.all_records)
            if self._loaded_selection_path != self._selection_settings_path():
                self._selected_record_ids.clear()
                self._group_scopes.clear()
                self._legacy_energy_groups.clear()
                self._load_saved_condition_selections()
        except OSError as exc:
            self._on_scan_error(generation, str(exc))
            self.all_records = []
        self._apply_versions()

    def _on_scan_error(self, generation: int, message: str) -> None:
        if self._closing:
            return
        if generation == self._scan_generation:
            self.selection_summary.setText(f"MCD catalog scan failed: {message.splitlines()[0]}")

    def _on_scan_finished(self, worker: McdScanWorker) -> None:
        try:
            self._scan_workers.remove(worker)
        except ValueError:
            pass
        if self._closing:
            self._scan_running = False
            return
        self._scan_running = False
        self.refresh_btn.setEnabled(True)
        self.rebuild_btn.setEnabled(True)
        if self._scan_pending:
            pending_rebuild = self._scan_pending_rebuild
            self._scan_pending = False
            self._scan_pending_rebuild = False
            self._scan(rebuild_catalog=pending_rebuild)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._selection_save_timer.stop()
        if self._group_scopes:
            self._write_condition_selections()
        self._closing = True
        # The scan worker may still own the SQLite catalog briefly after its
        # result callback runs.  Wait for it before a temporary experiment
        # directory is removed; Windows otherwise reports the database as
        # locked during test cleanup and occasionally during normal teardown.
        if self._scan_workers:
            self._thread_pool.waitForDone(3000)
        super().closeEvent(event)

    def _apply_versions(self) -> None:
        newest, older = newest_mcd_versions(self.all_records)
        self.older_records = older
        self.records = newest if self.newest_chk.isChecked() else list(self.all_records)
        if self.records:
            self.energy_min_spin.setValue(min(record.center_ev for record in self.records))
            self.energy_max_spin.setValue(max(record.center_ev for record in self.records))
        self._regroup()

    def _energy_filter_changed(self, enabled: bool) -> None:
        self.energy_min_spin.setEnabled(enabled)
        self.energy_max_spin.setEnabled(enabled)
        self._regroup()

    def _records_for_organizing(self) -> list[ProcessedMcdRecord]:
        if not self.energy_filter_chk.isChecked():
            return list(self.records)
        low = min(self.energy_min_spin.value(), self.energy_max_spin.value())
        high = max(self.energy_min_spin.value(), self.energy_max_spin.value())
        return [record for record in self.records if low <= record.center_ev <= high]

    def _regroup(self) -> None:
        for old in self.series_groups:
            state = self._group_scope(old)
            if old.series_id in self._selected_record_ids:
                ids = {r.record_id for r in old.records}
                state['included'] = sorted((set(state.get('included', []))-ids) | self._selected_record_ids[old.series_id])
                state['known'] = sorted(set(state.get('known', [])) | ids)
        self._scope_series_bindings.clear()
        variable = str(self.compare_combo.currentData() or "E-field")
        self.series_groups = organize_mcd_series(
            self._records_for_organizing(), variable, include_singletons=False
        )
        self._populate_series_list()

    def _populate_series_list(self) -> None:
        previous = self._current_series().series_id if self._current_series() else None
        self._list_refreshing = True
        self.series_list.clear()
        for index, series in enumerate(self.series_groups):
            required = {
                "E-field": ("Doping", "Temperature"),
                "Temperature": ("Doping", "E-field"),
            }.get(series.variable, ())
            missing = [
                name for name in required if series.fixed_conditions.get(name) is None
            ]
            prefix = "⚠ " if missing else ""
            item = QListWidgetItem(prefix + series.label)
            item.setData(Qt.UserRole, series.series_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if index == 0 else Qt.Unchecked)
            item.setToolTip(
                f"Matching condition metadata missing: {', '.join(missing)}"
                if missing else series.label
            )
            self.series_list.addItem(item)
            ids = {record.record_id for record in series.records}
            state = self._group_scope(series)
            known = set(state.get('known', []))
            included = set(state.get('included', []))
            if series.series_id not in self._selected_record_ids:
                self._selected_record_ids[series.series_id] = (included & ids) | (ids-known)
            state['known'] = sorted(known | ids)
            state['included'] = sorted((included-ids) | self._selected_record_ids[series.series_id])
        self._list_refreshing = False
        previous_row = next(
            (
                row for row in range(self.series_list.count())
                if self.series_list.item(row).data(Qt.UserRole) == previous
            ),
            -1,
        )
        if self.series_list.count():
            self.series_list.setCurrentRow(previous_row if previous_row >= 0 else 0)
        else:
            variable = str(self.compare_combo.currentData() or "E-field")
            self.preview_title.setText(f"No {variable} comparison available")
            self._show_empty_preview(
                f"No group has two or more {variable} values while the other conditions match."
            )
        self._filter_series_list(self.search_edit.text())
        self._update_export_summary()

    def _series_for_item(self, item: QListWidgetItem | None) -> McdSeries | None:
        if item is None:
            return None
        series_id = item.data(Qt.UserRole)
        return next((series for series in self.series_groups if series.series_id == series_id), None)

    def _current_series(self) -> McdSeries | None:
        return self._series_for_item(self.series_list.currentItem())

    def _checked_series(self) -> list[McdSeries]:
        return [
            series for row in range(self.series_list.count())
            if (item := self.series_list.item(row)).checkState() == Qt.Checked
            and (series := self._series_for_item(item)) is not None
        ]

    def _current_series_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if self._list_refreshing:
            return
        series = self._series_for_item(current)
        self._update_details(series)
        self._populate_condition_list(series)
        self._update_preview()

    def _filter_series_list(self, text: str) -> None:
        needle = text.strip().casefold()
        for row in range(self.series_list.count()):
            item = self.series_list.item(row)
            item.setHidden(bool(needle and needle not in item.text().casefold()))

    def _set_all_checked(self, checked: bool) -> None:
        self._list_refreshing = True
        for row in range(self.series_list.count()):
            item = self.series_list.item(row)
            if not item.isHidden():
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self._list_refreshing = False
        self._update_export_summary()

    def _selected_branches(self) -> tuple[McdBranch, ...]:
        branches: list[McdBranch] = []
        if self.increasing_chk.isChecked():
            branches.append(BRANCHES[0])
        if self.decreasing_chk.isChecked():
            branches.append(BRANCHES[1])
        return tuple(branches)

    def _update_export_summary(self) -> None:
        if self._list_refreshing:
            return
        series = self._checked_series()
        result_count = sum(
            len(self._selected_records_for_series(group)) for group in series
        )
        hidden = (
            f" {len(self.older_records)} older reprocessing version(s) hidden."
            if self.newest_chk.isChecked() and self.older_records else ""
        )
        if self._omitted_no_slope:
            hidden += f" {self._omitted_no_slope} older result(s) hidden: missing branch slope."
        self.selection_summary.setText(
            f"{len(series)} series / {result_count} results selected.{hidden}"
            if self.series_groups else (
                f"No valid {self.compare_combo.currentData() or 'comparison'} series found."
            )
        )
        self.export_btn.setText(
            f"Export {len(series)} selected series" if series else "Export selected series"
        )
        self.export_btn.setEnabled(
            bool(series) and result_count > 0 and bool(self._selected_branches())
        )

    def _toggle_details(self, visible: bool) -> None:
        self.details_table.setVisible(visible)
        self.details_btn.setText("Hide result details" if visible else "Show result details")

    def _selected_slope_metrics(self) -> tuple[str, ...]:
        return tuple(key for key, check in self.slope_checks.items() if check.isChecked())

    def _group_scope(self, series):
        """Stable identity based on comparison conditions, never member IDs."""
        import hashlib
        bound = self._scope_series_bindings.get(series.series_id)
        if bound in self._group_scopes:
            return self._group_scopes[bound]
        fixed = {k: round(v, 8) if v is not None else None
                 for k, v in series.fixed_conditions.items()}
        width = round(series.records[0].width_mev, 6)
        identity = dict(variable=series.variable, fixed=fixed, width=width)
        # Matching tolerances mirror series partitioning. Appending another
        # temperature may slightly change a fixed-condition mean.
        ids = {r.record_id for r in series.records}
        others = {r.record_id for s in self.series_groups if s.series_id != series.series_id
                  and s.variable == series.variable for r in s.records}
        candidates = sorted(self._group_scopes.items(),
                            key=lambda item: -len(set(item[1].get('known', [])) & ids))
        for scope_key, state in candidates:
            if scope_key in self._scope_series_bindings.values():
                continue
            known = set(state.get('known', []))
            if known & others and not known & ids:
                continue
            meta = state.get('identity', {})
            if meta.get('variable') != series.variable or abs(meta.get('width', -1)-width) > .001:
                continue
            previous = meta.get('fixed', {})
            if previous.keys() != fixed.keys():
                continue
            if all((previous[k] is None and v is None) or
                   (previous[k] is not None and v is not None and
                    abs(previous[k]-v) <= (.1 if k == 'Temperature' else .01))
                   for k, v in fixed.items()):
                self._scope_series_bindings[series.series_id] = scope_key
                return state
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key in self._group_scopes:
            key += '-' + hashlib.sha256(series.series_id.encode()).hexdigest()[:10]
        state = dict(identity=identity, assignments={}, manual={}, energies={}, known=[], included=[])
        # Legacy files have no comparison identity. Snapshot their assignments
        # once per new context; subsequent edits are independent.
        state['manual'] = {r.record_id:self._legacy_energy_groups[r.record_id]
                           for r in series.records if r.record_id in self._legacy_energy_groups}
        self._group_scopes[key] = state
        self._scope_series_bindings[series.series_id] = key
        return state

    @property
    def _energy_group_overrides(self):
        series = self._current_series()
        return self._group_scope(series)['manual'] if series else self._orphan_group_overrides

    def _energy_groups(self, records):
        import hashlib
        from core.mcd_energy_groups import initial_energy_groups
        ids = {r.record_id for r in records}
        series = next((s for s in self.series_groups if {r.record_id for r in s.records} == ids), None)
        if series is None:
            return initial_energy_groups(records, self.energy_group_tolerance.value())
        state = self._group_scope(series)
        assignments = state['assignments']
        energies_by_id = state.setdefault('energies', {})
        energies_by_id.update({r.record_id:r.center_ev for r in records})
        if not assignments:
            assignments.update(initial_energy_groups(records, self.energy_group_tolerance.value()))
        assignments.update(state['manual'])
        limit = state.get('group_count')
        allowed = {f'Group {i}' for i in range(1, limit+1)} if limit else None
        if allowed is not None:
            for key, name in list(assignments.items()):
                if name not in allowed and not name.startswith('Unassigned '):
                    assignments[key] = 'Unassigned ' + hashlib.sha256(key.encode()).hexdigest()[:6]
                    if key in state['manual']:
                        state['manual'][key] = assignments[key]
        # Keep all previous names reserved, even for temporarily hidden records.
        number = max([int(name.split()[1]) for name in assignments.values()
                      if name.startswith('Group ') and name[6:].isdigit()] or [0])
        tolerance = self.energy_group_tolerance.value()*.001
        for record in sorted(records, key=lambda r:(r.center_ev, r.record_id)):
            if record.record_id in assignments:
                continue
            candidates = {}
            for key, energy in energies_by_id.items():
                name = assignments.get(key)
                if name and not name.startswith('Unassigned '):
                    candidates.setdefault(name, []).append(energy)
            matches = [name for name, energies in candidates.items()
                       if max(energies+[record.center_ev])-min(energies+[record.center_ev]) <= tolerance+1e-12]
            if len(matches) == 1:
                name = matches[0]
            else:
                if limit:
                    name = 'Unassigned ' + hashlib.sha256(record.record_id.encode()).hexdigest()[:6]
                else:
                    number += 1
                    name = f'Group {number}'
            assignments[record.record_id] = name
        self._queue_condition_selection_save()
        return {key:assignments[key] for key in ids}

    def _energy_groups_changed(self, _value=None):
        # Save the assignment independently of rendering, which can fail.
        self._write_condition_selections()
        try:
            self._populate_condition_list(self._current_series())
        except Exception as exc:
            self.selection_summary.setText(f"Grouping saved; condition refresh failed: {exc}")
            return
        self._queue_condition_selection_save()
        self._request_preview_update()

    def _assignable_group_names(self, series):
        limit = self._group_scope(series).get('group_count')
        if limit:
            return [f'Group {i}' for i in range(1, limit+1)]
        return sorted(set(self._energy_groups(series.records).values()))

    def _group_count_changed(self, *_):
        series = self._current_series()
        if series is None:
            return
        # Materialize existing groups before applying the cap.
        self._energy_groups(series.records)
        self._group_scope(series)['group_count'] = (self.group_count_spin.value()
                                                   if self.fixed_group_count_chk.isChecked() else None)
        self._energy_groups(series.records)
        self._energy_groups_changed()

    def _assign_selected_group(self):
        series = self._current_series()
        ids = [str(item.data(Qt.UserRole)) for item in self.condition_list.selectedItems()]
        if series is None or not ids:
            return
        groups = self._energy_groups(series.records)
        limit = self._group_scope(series).get('group_count')
        name, accepted = QInputDialog.getItem(self, "Assign group", "Choose group:" if limit else "Existing or new group:",
                                             self._assignable_group_names(series), 0, not bool(limit))
        if not accepted or not name.strip():
            return
        name = name.strip()
        # Joining an existing group confirms its existing members too.
        for record_id, group in groups.items():
            if group == name or record_id in ids:
                self._energy_group_overrides[record_id] = name
        self._energy_groups_changed()

    def _reset_series_groups(self):
        series = self._current_series()
        if series is not None:
            state = self._group_scope(series)
            state['assignments'].clear()
            state['manual'].clear()
            state['energies'] = {}
            self._energy_groups_changed()

    def _edit_energy_groups(self):
        series = self._current_series()
        if series is None:
            return
        records = order_mcd_records(series.records,'E-field')[0]
        groups = self._energy_groups(series.records)
        limit = self._group_scope(series).get('group_count')
        dialog = QDialog(self); dialog.setWindowTitle('Edit energy subgroups'); dialog.resize(850,500)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('Saved integration window centers, not fitted peaks. Points in the same group connect across uneven E-field spacing.\nBlank subgroup keeps a point unassigned and unconnected. No values are averaged.'))
        table = QTableWidget(len(records),5,dialog)
        table.setHorizontalHeaderLabels(['Source','E-field (V)','Window center (eV)','Width (meV)','Subgroup'])
        for row,record in enumerate(records):
            for column,value in enumerate((record.source_file,_number(record.condition_value('E-field')),f'{record.center_ev:.12g}',f'{record.width_mev:.12g}',groups[record.record_id])):
                item = QTableWidgetItem(str(value))
                if column < 4:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row,column,item)
            if limit:
                combo = QComboBox()
                combo.addItem('Unassigned', '')
                for name in self._assignable_group_names(series):
                    combo.addItem(name, name)
                combo.setCurrentIndex(max(0, combo.findData(groups[record.record_id])))
                table.setCellWidget(row, 4, combo)
        table.setColumnWidth(0,280);table.setColumnWidth(4,170)
        layout.addWidget(table)
        reset = QPushButton('Reset to automatic groups');layout.addWidget(reset)
        reset_requested = False
        def reset_groups():
            nonlocal reset_requested
            reset_requested = True
            from core.mcd_energy_groups import initial_energy_groups
            automatic = initial_energy_groups(records,self.energy_group_tolerance.value())
            for row,record in enumerate(records):
                if limit:
                    combo = table.cellWidget(row,4)
                    combo.setCurrentIndex(max(0, combo.findData(automatic[record.record_id])))
                else:
                    table.item(row,4).setText(automatic[record.record_id])
        reset.clicked.connect(reset_groups)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel);layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject)
        if dialog.exec() == QDialog.Accepted:
            import hashlib
            from core.mcd_energy_groups import initial_energy_groups
            automatic = initial_energy_groups(records,self.energy_group_tolerance.value())
            if reset_requested:
                self._group_scope(series)['assignments'].clear()
                self._group_scope(series)['manual'].clear()
            for row,record in enumerate(records):
                name = table.cellWidget(row,4).currentData() if limit else table.item(row,4).text().strip()
                if reset_requested and name == automatic[record.record_id]:
                    self._energy_group_overrides.pop(record.record_id,None)
                else:
                    self._energy_group_overrides[record.record_id] = name or 'Unassigned ' + hashlib.sha256(record.record_id.encode()).hexdigest()[:6]
            self._energy_groups_changed()

    def _slope_visibility_changed(self, _checked=False) -> None:
        series = self._current_series()
        self._update_details(series)
        self._populate_condition_list(series)
        self._request_preview_update()

    def _slope_text(self, record, branch, metric) -> str:
        value = record.slope(branch, metric)
        return "N/A" if value is None else _number(value)

    def _update_details(self, series: McdSeries | None) -> None:
        self.details_table.setRowCount(0)
        metrics = self._selected_slope_metrics()
        headers = list(self.DETAIL_COLUMNS[:5]) + [
            f"{SLOPE_METRICS[metric]} · {branch}" for metric in metrics for branch in ("Inc", "Dec")
        ]
        self.details_table.setColumnCount(len(headers))
        self.details_table.setHorizontalHeaderLabels(headers)
        if series is None:
            return
        ordered, _ = order_mcd_records(series.records, series.variable)
        for row, record in enumerate(ordered):
            self.details_table.insertRow(row)
            values = (
                record.source_file,
                _number(record_order_value(record, series.variable)),
                _number(record.center_ev),
                _number(record.width_mev),
                _number(record.temperature_measured_k),
            ) + tuple(self._slope_text(record, branch, metric) for metric in metrics for branch in BRANCHES)
            for column, value in enumerate(values):
                self.details_table.setItem(row, column, QTableWidgetItem(value))
        self.details_table.resizeColumnsToContents()

    def _populate_condition_list(self, series: McdSeries | None) -> None:
        limit = self._group_scope(series).get('group_count') if series else None
        self.fixed_group_count_chk.blockSignals(True)
        self.group_count_spin.blockSignals(True)
        self.fixed_group_count_chk.setChecked(bool(limit))
        self.fixed_group_count_chk.setEnabled(series is not None)
        self.group_count_spin.setValue(limit or 3)
        self.group_count_spin.setEnabled(bool(limit))
        self.fixed_group_count_chk.blockSignals(False)
        self.group_count_spin.blockSignals(False)
        if series is None:
            self.condition_list.clear()
            return
        selected = self._selected_record_ids.setdefault(
            series.series_id, {record.record_id for record in series.records}
        )
        from matplotlib.colors import to_hex
        groups = self._energy_groups(series.records)
        colors = {key:to_hex(color) for key,color in self._preview_energy_colors(series).items()}
        pending_items = []
        selected_rows = {item.data(Qt.UserRole) for item in self.condition_list.selectedItems()}
        current_item = self.condition_list.currentItem()
        current_id = current_item.data(Qt.UserRole) if current_item is not None else None
        for record in order_mcd_records(series.records, series.variable)[0]:
            value = record_order_value(record, series.variable)
            label = (
                f"{groups[record.record_id]} · {series.variable}={_number(value)} · E={record.center_ev:.5g} eV · "
                + " | ".join(f"{SLOPE_METRICS[metric]}: Inc={self._slope_text(record, 'B increasing', metric)} · Dec={self._slope_text(record, 'B decreasing', metric)}"
                             for metric in self._selected_slope_metrics())
            )
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, record.record_id)
            item.setData(Qt.UserRole + 2, label)
            item.setToolTip(record.source_file)
            item.setForeground(QColor(colors[record.record_id]))
            item.setData(Qt.UserRole + 1, colors[record.record_id])
            item.setBackground(
                QColor(theme_alias("selection_subtle_background"))
                if record.record_id in selected
                else QColor(theme_alias("surface_secondary"))
            )
            pending_items.append(item)
        # Build all labels/colors first; a failure must not erase the list.
        previous_refreshing = self._list_refreshing
        signals_blocked = self.condition_list.blockSignals(True)
        self._list_refreshing = True
        try:
            self.condition_list.clear()
            for item in pending_items:
                self.condition_list.addItem(item)
                if item.data(Qt.UserRole) == current_id:
                    self.condition_list.setCurrentItem(item)
            for item in pending_items:
                item.setSelected(item.data(Qt.UserRole) in selected_rows)
        finally:
            self.condition_list.blockSignals(signals_blocked)
            self._list_refreshing = previous_refreshing
        self._style_condition_items()
        self._update_condition_summary()
        if self.condition_list.count() and self.condition_list.currentRow() < 0:
            self.condition_list.setCurrentRow(0)
        self._condition_focus_changed()

    def _selected_records_for_series(self, series: McdSeries | None) -> list[ProcessedMcdRecord]:
        if series is None:
            return []
        selected = self._selected_record_ids.setdefault(
            series.series_id, {record.record_id for record in series.records}
        )
        return [record for record in series.records if record.record_id in selected]

    def _conditions_changed(self) -> None:
        if self._list_refreshing:
            return
        self._update_export_summary()
        self._populate_condition_list(self._current_series())
        self._update_condition_summary()
        self._queue_condition_selection_save()
        self._update_condition_action_buttons()
        self._request_preview_update()

    def _selection_settings_path(self) -> Path:
        root = self.experiment_root
        if root.name.casefold() == "mcd" and root.parent.name.casefold() == "processed data":
            folder = root
        else:
            folder = root / "Processed Data" / "MCD"
        return folder / ".mcd_organizer_selections.json"

    def _load_saved_condition_selections(self) -> None:
        path = self._selection_settings_path()
        self._loaded_selection_path = path
        self._selection_baseline = {}
        self._scope_series_bindings.clear()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self._selection_baseline = copy.deepcopy(payload)
        saved_palette = payload.get("default_palette") if isinstance(payload, dict) else None
        if isinstance(payload, dict):
            overrides = payload.get('energy_groups',{})
            self._legacy_energy_groups = {str(k):str(v) for k,v in overrides.items() if isinstance(v,str) and v} if isinstance(overrides,dict) else {}
            scopes = payload.get('group_scopes', {})
            self._group_scopes = {k:v for k,v in scopes.items() if isinstance(v,dict)
                                  and all(isinstance(v.get(x),dict) for x in ('identity','assignments','manual'))} if isinstance(scopes,dict) else {}
            tolerance = payload.get('energy_group_tolerance_mev',5.)
            if isinstance(tolerance,(int,float)) and np.isfinite(tolerance):
                self.energy_group_tolerance.blockSignals(True)
                self.energy_group_tolerance.setValue(tolerance)
                self.energy_group_tolerance.blockSignals(False)
        if isinstance(saved_palette, str) and saved_palette in PALETTES:
            self._default_palette = saved_palette
            index = self.palette_combo.findData(saved_palette)
            if index >= 0:
                self.palette_combo.setCurrentIndex(index)
            self.palette_default_btn.setText(f"Default: {saved_palette}")
        selections = payload.get("series", {}) if isinstance(payload, dict) else {}
        if not isinstance(selections, dict):
            return
        self._selected_record_ids.update({
            str(series_id): {str(record_id) for record_id in record_ids}
            for series_id, record_ids in selections.items()
            if isinstance(record_ids, list)
        })

    def _queue_condition_selection_save(self) -> None:
        self._selection_save_timer.start()

    def _write_condition_selections(self) -> bool:
        path = self._selection_settings_path()
        for series in self.series_groups:
            state = self._group_scope(series)
            ids = {r.record_id for r in series.records}
            included = self._selected_record_ids.get(series.series_id, ids)
            # Retain hidden/filtered records so returning them does not reset state.
            state['included'] = sorted((set(state.get('included', []))-ids) | set(included))
            state['known'] = sorted(set(state.get('known', [])) | ids)
        payload = {
                "schema_version": 3,
                "default_palette": self._default_palette,
                "energy_groups": self._legacy_energy_groups,
                "group_scopes": self._group_scopes,
                "energy_group_tolerance_mev": self.energy_group_tolerance.value(),
                "series": {
                    series_id: sorted(record_ids)
                    for series_id, record_ids in self._selected_record_ids.items()
                },
            }
        if payload == self._selection_baseline:
            return True
        lock = QLockFile(str(path) + '.lock')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not lock.tryLock(1000):
                raise OSError('Another Organizer is saving this experiment; retry shortly.')
            remote = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            if not isinstance(remote, dict):
                raise ValueError('The saved Organizer settings are not a JSON object.')
            merged = _merge_selection_changes(self._selection_baseline, payload, remote)
            output = QSaveFile(str(path))
            if not output.open(QIODevice.WriteOnly):
                raise OSError(output.errorString())
            data = json.dumps(merged, indent=2, sort_keys=True).encode('utf-8')
            if output.write(data) != len(data):
                output.cancelWriting()
                raise OSError(output.errorString())
            if not output.commit():
                raise OSError(output.errorString())
            # Baseline follows this window's snapshot, not unseen remote edits.
            self._selection_baseline = copy.deepcopy(payload)
            self.statusBar().clearMessage()
            return True
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(f'Group/selection save failed: {exc} | {path}')
            return False
        finally:
            lock.unlock()

    def _set_current_palette_default(self) -> None:
        selected = str(self.palette_combo.currentData() or "tab10")
        self._default_palette = selected if selected in PALETTES else "tab10"
        self._write_condition_selections()
        self.palette_default_btn.setText(f"Default: {self._default_palette}")

    def _condition_focus_changed(self) -> None:
        item = self.condition_list.currentItem()
        self._focused_record_id = (
            str(item.data(Qt.UserRole)) if item is not None else None
        )
        self._update_condition_action_buttons()
        self._apply_focus_style()
        self._highlight_energy_record()

    def _update_condition_action_buttons(self) -> None:
        series = self._current_series()
        item = self.condition_list.currentItem()
        record_id = str(item.data(Qt.UserRole)) if item is not None else None
        included = bool(
            series is not None and record_id is not None
            and record_id in self._selected_record_ids.get(series.series_id, set())
        )
        self.condition_exclude_btn.setEnabled(included)
        self.condition_restore_selected_btn.setEnabled(
            series is not None and record_id is not None and not included
        )

    def _update_condition_summary(self) -> None:
        total = self.condition_list.count()
        series = self._current_series()
        included = len(self._selected_records_for_series(series)) if series is not None else 0
        self.condition_summary.setText(f"{included} of {total} conditions included")

    def _exclude_focused_condition(self) -> None:
        current = self.condition_list.currentRow()
        if current < 0:
            return
        series = self._current_series()
        item = self.condition_list.item(current)
        if series is not None and item is not None:
            self._selected_record_ids.setdefault(series.series_id, set()).discard(
                str(item.data(Qt.UserRole))
            )
        self._conditions_changed()

    def _restore_all_conditions(self) -> None:
        series = self._current_series()
        if series is None:
            return
        self._selected_record_ids[series.series_id] = {
            record.record_id for record in series.records
        }
        self._conditions_changed()

    def _restore_focused_condition(self) -> None:
        series = self._current_series()
        item = self.condition_list.currentItem()
        if series is None or item is None:
            return
        record_id = str(item.data(Qt.UserRole))
        needs_artist = record_id not in self._plot_artists
        self._selected_record_ids.setdefault(series.series_id, set()).add(record_id)
        self._conditions_changed()
        if needs_artist:
            self._request_preview_update()

    def _style_condition_items(self) -> None:
        """Make checked conditions prominent and unchecked ones recede."""
        for row in range(self.condition_list.count()):
            item = self.condition_list.item(row)
            series = self._current_series()
            included_ids = self._selected_record_ids.get(series.series_id, set()) if series else set()
            checked = str(item.data(Qt.UserRole)) in included_ids
            item.setBackground(
                QColor(theme_alias("selection_subtle_background"))
                if checked
                else QColor(theme_alias("surface_secondary"))
            )
            item.setForeground(
                QColor(item.data(Qt.UserRole + 1)) if checked
                else QColor(theme_alias("text_tertiary"))
            )
            font = item.font()
            font.setBold(checked)
            item.setFont(font)
            base_text = str(item.data(Qt.UserRole + 2) or item.text()).lstrip("●○× ")
            item.setData(Qt.UserRole + 2, base_text)
            item.setText(("● " if checked else "× ") + base_text)

    def _request_preview_update(self) -> None:
        """Coalesce rapid checkbox changes before rereading traces/redrawing plots."""
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def _trace_arrays(
        self, record: ProcessedMcdRecord
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        cached = self._trace_array_cache.get(record.record_id)
        if cached is not None:
            return cached
        traces = load_branch_traces(record, BRANCHES)
        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for branch in BRANCHES:
            block = traces[traces["branch"] == branch]
            arrays[branch] = (
                block["B_T"].to_numpy(float, copy=True),
                block["corrected_signed_mean"].to_numpy(float, copy=True),
            )
        self._trace_array_cache[record.record_id] = arrays
        if len(self._trace_array_cache) > self._trace_cache_limit:
            self._trace_array_cache.pop(next(iter(self._trace_array_cache)))
        return arrays

    def _apply_focus_style(self) -> None:
        if self.canvas is None:
            return
        focus_visible = any(
            artist.get_visible()
            for artist in self._plot_artists.get(self._focused_record_id or "", [])
        )
        for record_id, artists in self._plot_artists.items():
            focused = not focus_visible or record_id == self._focused_record_id
            for artist in artists:
                artist.set_linewidth(2.4 if focused else 0.9)
                artist.set_alpha(1.0)
        self.canvas.draw_idle()

    def _apply_inclusion_visibility(self) -> None:
        series = self._current_series()
        if series is None:
            return
        included = self._selected_record_ids.get(series.series_id, set())
        for record_id, artists in self._plot_artists.items():
            for artist in artists:
                artist.set_visible(record_id in included)
        self._update_cached_slope_lines(series, included)
        self._apply_focus_style()
        if self.slope_canvas is not None:
            self.slope_canvas.draw_idle()

    def _update_cached_slope_lines(self, series: McdSeries, included: set[str]) -> None:
        self._update_slope_preview([record for record in series.records if record.record_id in included])

    def _show_empty_preview(self, message: str) -> None:
        self.curie_weiss_panel.clear(message)
        self._energy_point_artists = {}
        self._energy_focus_artists = []
        self._energy_hover = None
        if self.figure is None or self.slope_figure is None:
            return
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center", color="#666")
        axis.set_axis_off()
        self.canvas.draw_idle()
        self.slope_figure.clear()
        slope_axis = self.slope_figure.add_subplot(111)
        slope_axis.text(0.5, 0.5, message, transform=slope_axis.transAxes, ha="center", va="center", color="#666")
        slope_axis.set_axis_off()
        self.slope_canvas.draw_idle()

    def _update_preview(self) -> None:
        if self.figure is None or self.slope_figure is None:
            return
        from matplotlib.lines import Line2D

        series = self._current_series()
        branches = self._selected_branches()
        self._update_export_summary()
        if series is None:
            self.preview_title.setText("Select a series to preview")
            self._show_empty_preview("Select a processed condition series.")
            return
        self.preview_title.setText(series.label)
        records = self._selected_records_for_series(series)
        if not records:
            self._show_empty_preview("No conditions are selected in this series.")
            return
        if not branches:
            self._show_empty_preview("Select at least one field-sweep branch.")
            return
        records, resolved_order = order_mcd_records(records, series.variable)
        slope_records = list(records)
        groups = self._energy_groups(series.records)
        group_names = sorted({groups[r.record_id] for r in records})
        previous = self.mcd_group_combo.currentData()
        self.mcd_group_combo.blockSignals(True)
        self.mcd_group_combo.clear()
        self.mcd_group_combo.addItem('All groups', '')
        for name in group_names:
            self.mcd_group_combo.addItem(name, name)
        self.mcd_group_combo.setCurrentIndex(max(0,self.mcd_group_combo.findData(previous)))
        self.mcd_group_combo.blockSignals(False)
        selected_group = self.mcd_group_combo.currentData()
        if selected_group:
            records = [r for r in records if groups[r.record_id] == selected_group]

        self.figure.clear()
        self._plot_artists.clear()
        axes = np.atleast_1d(
            self.figure.subplots(1, len(branches), sharey=True)
        ).tolist()
        colors = self._preview_energy_colors(series) if series.variable in ('E-field', 'Temperature') else assign_plot_colors(
            records, str(self.palette_combo.currentData() or "tab10"), resolved_order)

        labels, fixed_text = concise_condition_labels(records, resolved_order)
        handles: list[Line2D] = []
        plotted_labels: list[str] = []
        for record, label in zip(records, labels):
            try:
                arrays = self._trace_arrays(record)
            except (OSError, ValueError):
                continue
            color = colors[record.record_id]
            focused = self._focused_record_id is None or record.record_id == self._focused_record_id
            handles.append(Line2D([0], [0], color=color, marker="o", linewidth=1.4))
            plotted_labels.append(label)
            for axis, branch in zip(axes, branches):
                b_values, mcd_values = arrays[branch]
                increasing = branch == "B increasing"
                line, = axis.plot(
                    b_values, mcd_values,
                    linestyle="-" if increasing else "--", marker="o" if increasing else "s", markersize=3.2,
                    markevery=max(1, len(b_values) // 180),
                    linewidth=2.4 if focused else 0.9,
                    alpha=1.0,
                    color=color,
                    markerfacecolor=color if increasing else "white",
                    markeredgecolor=color,
                )
                self._plot_artists.setdefault(record.record_id, []).append(line)
        for index, (axis, branch) in enumerate(zip(axes, branches)):
            axis.axhline(0.0, color="#555", linewidth=0.7)
            axis.grid(alpha=0.22)
            axis.set_xlabel("B field (T)")
            axis.set_title(branch)
            if index == 0:
                axis.set_ylabel("Corrected signed-mean MCD")
        title = f"{selected_group or 'All groups'} · {series.variable} comparison"
        if fixed_text:
            title += f" · {fixed_text}"
        self.figure.suptitle(title, fontsize=12, fontweight="bold")
        if series.variable == 'Temperature':
            from core.mcd_energy_groups import temperature_curve_legend
            temperature_curve_legend(self.figure, records, groups, colors)
        elif handles and len(handles) <= 12:
            self.figure.legend(
                handles, plotted_labels, fontsize=7, title="Processed energy / condition",
                title_fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.13),
                ncol=2, frameon=True,
            )
        elif len(handles) > 12:
            self.figure.text(
                0.5, 0.01,
                "Select a group below; select a condition on the right to highlight a curve.",
                ha="center", va="bottom", fontsize=8, color="#666",
            )
        if series.variable != 'Temperature':
            self.figure.tight_layout(rect=(0.0, 0.16 if len(handles) <= 12 else 0.0, 1.0, 0.93))
        self.canvas.draw_idle()
        self._update_slope_preview(slope_records)

    def _update_slope_preview(self, records: list[ProcessedMcdRecord]) -> None:
        series = self._current_series()
        if series is not None and series.variable == 'Temperature':
            self.curie_weiss_panel.set_series(
                series.series_id, records, self._energy_groups(series.records), self._selected_branches())
        else:
            self.curie_weiss_panel.clear()
        if self.slope_figure is None:
            return
        from core.mcd_energy_groups import draw_energy_slope_panels
        self._energy_focus_artists = []
        self._energy_hover = None
        series = self._current_series()
        self._energy_point_artists = {}
        self._slope_lines.clear()
        if series is None or series.variable not in ('E-field', 'Temperature'):
            self.slope_figure.clear()
            axis = self.slope_figure.add_subplot(111)
            axis.text(.5,.5,'Select an E-field or Temperature comparison series.',ha='center',transform=axis.transAxes)
            axis.set_axis_off();self.slope_canvas.draw_idle()
            return
        if series.variable == 'Temperature':
            from core.mcd_energy_groups import draw_temperature_slope_panels
            self._energy_point_artists = draw_temperature_slope_panels(
                self.slope_figure, records, self._energy_groups(series.records),
                self._selected_slope_metrics(), self._selected_branches(),
                str(self.palette_combo.currentData() or 'tab10'))
        else:
            self._energy_point_artists = draw_energy_slope_panels(
            self.slope_figure, records, self._energy_groups(series.records),
            self._selected_slope_metrics(), self._selected_branches(),
            self.energy_group_tolerance.value(), str(self.palette_combo.currentData() or 'tab10'),
            manual_record_ids=tuple(self._energy_group_overrides),
            record_colors=self._preview_energy_colors(series))
        self.slope_canvas.setMinimumHeight(220 * (1+bool(self._selected_slope_metrics())))
        for artist, (members,metric,branch) in self._energy_point_artists.items():
            if metric != 'window_energy':
                self._slope_lines[(metric,branch, self._energy_groups(series.records).get(members[0].record_id) if members else '')] = artist
        self._highlight_energy_record()
        self.slope_canvas.draw_idle()

    def _energy_hits(self, event):
        hits = []
        if event.inaxes is None:
            return hits
        for artist,(records,metric,branch) in self._energy_point_artists.items():
            if artist.axes is not event.inaxes:
                continue
            contains, info = artist.contains(event)
            if contains:
                for index in info.get('ind',[]):
                    hits.append((records[index],metric,branch))
        return hits

    def _energy_plot_hover(self, event):
        hits = self._energy_hits(event)
        if not hits and self._energy_hover is None:
            return
        if self._energy_hover is not None:
            self._energy_hover.remove();self._energy_hover=None
        if hits:
            from core.mcd_energy_groups import point_description
            series = self._current_series()
            groups = self._energy_groups(series.records) if series else {}
            text = '\n\n'.join(point_description(record,metric,branch,groups.get(record.record_id,''))
                               for record,metric,branch in hits)
            self._energy_hover = event.inaxes.annotate(text,xy=(.01,.99),xycoords='axes fraction',
                va='top',fontsize=8,bbox=dict(boxstyle='round',fc='white',ec='#777',alpha=.96),zorder=20)
        self.slope_canvas.draw_idle()

    def _energy_plot_click(self, event):
        if event.button != 1:
            return
        ids = list(dict.fromkeys(record.record_id for record,_,_ in self._energy_hits(event)))
        if not ids:
            return
        # Repeated clicks cycle coincident windows instead of silently choosing
        # one or combining their measurements.
        index = (ids.index(self._focused_record_id)+1)%len(ids) if self._focused_record_id in ids else 0
        for row in range(self.condition_list.count()):
            if self.condition_list.item(row).data(Qt.UserRole)==ids[index]:
                self.condition_list.setCurrentRow(row)
                self._condition_focus_changed()
                break

    def _highlight_energy_record(self):
        for artist in self._energy_focus_artists:
            artist.remove()
        self._energy_focus_artists = []
        seen = set()
        for artist,(records,metric,branch) in self._energy_point_artists.items():
            for index,record in enumerate(records):
                if record.record_id != self._focused_record_id:
                    continue
                x,y = artist.get_offsets()[index]
                if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                    continue
                key = (id(artist.axes),float(x),float(y))
                if key in seen:
                    continue
                seen.add(key)
                marker, = artist.axes.plot([x],[y],linestyle='none',marker='o',markersize=11,
                    markerfacecolor='none',markeredgecolor='#111111',markeredgewidth=1.5,zorder=10)
                self._energy_focus_artists.append(marker)
        if self.slope_canvas is not None:
            self.slope_canvas.draw_idle()

    def _compare_theta(self):
        series = self._checked_series()
        branches = self._selected_branches()
        if len(series) < 2 or not branches or any(s.variable != 'Temperature' for s in series):
            QMessageBox.information(self, 'Compare θCW',
                'Choose Temperature dependence, check at least two condition series and one sweep branch.')
            return
        from ui_qt.theta_comparison_dialog import ThetaComparisonDialog
        entries = [dict(series_id=s.series_id, label=s.label, fixed_conditions=s.fixed_conditions,
                        records=self._selected_records_for_series(s), groups=self._energy_groups(s.records))
                   for s in series]
        if any(not e['records'] for e in entries):
            QMessageBox.information(self,'Compare θCW','Each checked series needs included records.')
            return
        dialog = ThetaComparisonDialog(entries, branches, self.output_label.text(), self)
        dialog.halfwidth.setValue(self.curie_weiss_panel.b_halfwidth.value())
        dialog.t_min.setValue(self.curie_weiss_panel.t_min.value())
        dialog.t_max.setValue(self.curie_weiss_panel.t_max.value())
        dialog.temperature.setCurrentIndex(dialog.temperature.findData(self.curie_weiss_panel.temperature_combo.currentData()))
        dialog.exec()
        dialog.deleteLater()

    def _export(self) -> None:
        if self._export_worker is not None:
            self.selection_summary.setText("Export already running; wait for it to finish.")
            return
        series = self._checked_series()
        branches = self._selected_branches()
        if not series or not branches:
            QMessageBox.information(
                self, "MCD Organizer", "Check at least one series and one sweep branch."
            )
            return
        records: list[ProcessedMcdRecord] = []
        seen: set[str] = set()
        selected_groups: list[McdSeries] = []
        for group in series:
            members = tuple(self._selected_records_for_series(group))
            if not members:
                continue
            selected_groups.append(
                McdSeries(group.series_id, group.variable, group.label, members, group.fixed_conditions)
            )
            for record in members:
                if record.record_id not in seen:
                    records.append(record)
                    seen.add(record.record_id)
        worker = Worker(
            mcd_extract_export_worker,
            tuple(copy.deepcopy(records)),
            str(self.output_label.text()),
            branches=tuple(branches),
            order_by="Auto",
            palette=str(self.palette_combo.currentData() or "tab10"),
            export_csv=bool(self.export_csv_chk.isChecked()),
            series_groups=tuple(copy.deepcopy(selected_groups)),
            energy_groups=self._export_energy_groups(series),
            energy_group_palette=self._export_energy_colors(series),
            manual_record_ids=tuple({key for group in series for key in self._group_scope(group)['manual']}),
            energy_record_palette=self._export_record_colors(series),
            group_tolerance_mev=float(self.energy_group_tolerance.value()),
        )
        self._export_worker = worker
        self.export_btn.setEnabled(False)
        self.selection_summary.setText("Exporting MCD organization…")
        worker.signals.result.connect(self._on_export_done)
        worker.signals.error.connect(self._on_export_error)
        worker.signals.finished.connect(self._on_export_finished)
        self._export_pool.start_worker(worker)

    def _export_energy_groups(self, series):
        # Group the full preview series before applying inclusion filters.
        output = {}
        for index,group in enumerate(series,start=1):
            prefix = f'Series {index} / ' if len(series)>1 else ''
            output.update({key:prefix+name for key,name in self._energy_groups(group.records).items()})
        return output

    def _energy_palette_changed(self, _index=None):
        series = self._current_series()
        if series is not None:
            from matplotlib.colors import to_hex
            colors = self._preview_energy_colors(series)
            for row in range(self.condition_list.count()):
                item = self.condition_list.item(row)
                color = colors.get(str(item.data(Qt.UserRole)))
                if color is not None:
                    item.setData(Qt.UserRole + 1, to_hex(color))
            self._style_condition_items()
        self._request_preview_update()

    def _preview_energy_colors(self, series):
        from core.mcd_energy_groups import energy_record_colors, energy_group_colors
        groups = self._energy_groups(series.records)
        return energy_record_colors(series.records, groups, energy_group_colors(groups, str(self.palette_combo.currentData() or 'tab10')), variable=series.variable if series.variable == 'Temperature' else 'E-field')

    def _export_record_colors(self, series):
        return {key: color for group in series for key, color in self._preview_energy_colors(group).items()}

    def _export_energy_colors(self, series):
        from core.mcd_energy_groups import energy_group_colors
        colors = {}
        for index, group in enumerate(series, start=1):
            prefix = f'Series {index} / ' if len(series)>1 else ''
            local = energy_group_colors(self._energy_groups(group.records), str(self.palette_combo.currentData() or 'tab10'))
            colors.update({prefix+name: color for name, color in local.items()})
        return colors

    def _on_export_done(self, paths: dict) -> None:
        if self._export_worker is None or self._closing:
            return
        self.selection_summary.setText("Export complete")
        QMessageBox.information(
            self, "MCD export complete",
            "Created:\n" + "\n".join(path.name for path in paths.values()),
        )

    def _on_export_error(self, message: str) -> None:
        if self._export_worker is None or self._closing:
            return
        self.selection_summary.setText(f"MCD export failed: {str(message).splitlines()[0]}")

    def _on_export_finished(self) -> None:
        if self._export_worker is None:
            return
        self._export_worker = None
        if not self._closing:
            self.export_btn.setEnabled(bool(self._checked_series()) and bool(self._selected_branches()))
