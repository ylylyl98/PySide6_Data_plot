"""Focused standalone UI for organizing and exporting processed MCD results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThreadPool, QTimer, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.mcd_extract import (
    BRANCHES,
    PALETTES,
    McdBranch,
    McdSeries,
    ProcessedMcdRecord,
    assign_plot_colors,
    concise_condition_labels,
    discover_processed_mcd,
    export_mcd_extract,
    load_branch_traces,
    newest_mcd_versions,
    organize_mcd_series,
    order_mcd_records,
    record_order_value,
)
from ui_qt.mcd_async import McdScanWorker
from ui_qt.fluent_ui.style import set_fluent_property
from ui_qt.theme import alias as theme_alias


def _number(value: float | None) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{float(value):.6g}"


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
        self._default_palette = "viridis"
        self._focused_record_id: str | None = None
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
        self.export_btn.setMinimumWidth(190)
        export_row.addWidget(QLabel("Export to:"))
        export_row.addWidget(self.output_label, 1)
        export_row.addWidget(self.output_btn)
        export_row.addWidget(self.export_csv_chk)
        export_row.addWidget(self.export_btn)
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

        self.condition_list = QListWidget()
        self.condition_list.setMaximumHeight(150)
        self.condition_list.setWordWrap(True)
        self.condition_list.setSpacing(1)
        condition_row = QHBoxLayout()
        condition_row.addWidget(QLabel("Conditions in selected series:"))
        self.condition_exclude_btn = QPushButton("Exclude highlighted")
        self.condition_restore_selected_btn = QPushButton("Restore highlighted")
        self.condition_restore_btn = QPushButton("Restore all")
        self.condition_exclude_btn.setEnabled(False)
        self.condition_restore_selected_btn.setEnabled(False)
        condition_row.addWidget(self.condition_exclude_btn)
        condition_row.addWidget(self.condition_restore_selected_btn)
        condition_row.addWidget(self.condition_restore_btn)
        condition_row.addStretch(1)
        layout.addLayout(condition_row)
        layout.addWidget(self.condition_list)
        self.condition_summary = QLabel("0 of 0 conditions included")
        set_fluent_property(self.condition_summary, "appRole", "hintText")
        layout.addWidget(self.condition_summary)

        self.figure = None
        self.canvas = None
        self.slope_figure = None
        self.slope_canvas = None
        self.preview_tabs = QTabWidget()
        mcd_tab = QWidget()
        mcd_layout = QVBoxLayout(mcd_tab)
        mcd_layout.setContentsMargins(0, 0, 0, 0)
        mcd_layout.addWidget(QLabel("Preparing preview…"), 1)
        slope_tab = QWidget()
        slope_layout = QVBoxLayout(slope_tab)
        slope_layout.setContentsMargins(0, 0, 0, 0)
        slope_layout.addWidget(QLabel("Preparing preview…"), 1)
        self._mcd_plot_layout = mcd_layout
        self._slope_plot_layout = slope_layout
        self.preview_tabs.addTab(mcd_tab, "MCD vs B")
        self.preview_tabs.addTab(slope_tab, "Slope vs E-field")
        layout.addWidget(self.preview_tabs, 1)
        QTimer.singleShot(0, self._init_plot_widgets)

        self.increasing_chk.toggled.connect(lambda _checked: self._request_preview_update())
        self.decreasing_chk.toggled.connect(lambda _checked: self._request_preview_update())
        self.palette_combo.currentIndexChanged.connect(lambda _index: self._request_preview_update())
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
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(9, 5), dpi=100, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.slope_figure = Figure(figsize=(9, 5), dpi=100, facecolor="white")
        self.slope_canvas = FigureCanvasQTAgg(self.slope_figure)
        self._mcd_plot_layout.replaceWidget(self._mcd_plot_layout.itemAt(0).widget(), self.canvas)
        self._slope_plot_layout.replaceWidget(self._slope_plot_layout.itemAt(0).widget(), self.slope_canvas)
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
            self.experiment_root = Path(folder)
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
            self._selected_record_ids.clear()
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
            self._selected_record_ids.setdefault(
                series.series_id, {record.record_id for record in series.records}
            )
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

    def _update_details(self, series: McdSeries | None) -> None:
        self.details_table.setRowCount(0)
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
                _number(record.increasing_slope_per_t),
                _number(record.decreasing_slope_per_t),
            )
            for column, value in enumerate(values):
                self.details_table.setItem(row, column, QTableWidgetItem(value))
        self.details_table.resizeColumnsToContents()

    def _populate_condition_list(self, series: McdSeries | None) -> None:
        self._list_refreshing = True
        self.condition_list.clear()
        if series is None:
            self._list_refreshing = False
            return
        selected = self._selected_record_ids.setdefault(
            series.series_id, {record.record_id for record in series.records}
        )
        colors = assign_plot_colors(
            series.records, str(self.palette_combo.currentData() or "viridis"), series.variable
        )
        for record in order_mcd_records(series.records, series.variable)[0]:
            value = record_order_value(record, series.variable)
            label = (
                f"{series.variable}={_number(value)} · E={record.center_ev:.5g} eV · "
                f"Inc={_number(record.increasing_slope_per_t)} · Dec={_number(record.decreasing_slope_per_t)}"
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
            self.condition_list.addItem(item)
        self._list_refreshing = False
        self._style_condition_items()
        self._update_condition_summary()
        if self.condition_list.count() and self.condition_list.currentRow() < 0:
            self.condition_list.setCurrentRow(0)

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
        self._style_condition_items()
        self._update_condition_summary()
        self._queue_condition_selection_save()
        self._update_condition_action_buttons()
        self._apply_inclusion_visibility()

    def _selection_settings_path(self) -> Path:
        root = self.experiment_root
        if root.name.casefold() == "mcd" and root.parent.name.casefold() == "processed data":
            folder = root
        else:
            folder = root / "Processed Data" / "MCD"
        return folder / ".mcd_organizer_selections.json"

    def _load_saved_condition_selections(self) -> None:
        path = self._selection_settings_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        saved_palette = payload.get("default_palette") if isinstance(payload, dict) else None
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

    def _write_condition_selections(self) -> None:
        path = self._selection_settings_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "schema_version": 2,
                "default_palette": self._default_palette,
                "series": {
                    series_id: sorted(record_ids)
                    for series_id, record_ids in self._selected_record_ids.items()
                },
            }, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            pass

    def _set_current_palette_default(self) -> None:
        selected = str(self.palette_combo.currentData() or "viridis")
        self._default_palette = selected if selected in PALETTES else "viridis"
        self._write_condition_selections()
        self.palette_default_btn.setText(f"Default: {self._default_palette}")

    def _condition_focus_changed(self) -> None:
        item = self.condition_list.currentItem()
        self._focused_record_id = (
            str(item.data(Qt.UserRole)) if item is not None else None
        )
        self._update_condition_action_buttons()
        self._apply_focus_style()

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
                artist.set_alpha(1.0 if focused else 0.28)
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
        if series.variable != "E-field" or not self._slope_lines:
            return
        ordered, _ = order_mcd_records(series.records, "E-field")
        active = [record for record in ordered if record.record_id in included]
        x = np.asarray([record.condition_value("E-field") for record in active], float)
        for key, attribute in (
            ("increasing", "increasing_slope_per_t"),
            ("decreasing", "decreasing_slope_per_t"),
        ):
            line = self._slope_lines.get(key)
            if line is None:
                continue
            y = np.asarray([
                np.nan if getattr(record, attribute) is None else getattr(record, attribute)
                for record in active
            ], float)
            valid = np.isfinite(x) & np.isfinite(y)
            line.set_data(x[valid], y[valid])
        axis = self.slope_figure.axes[0] if self.slope_figure.axes else None
        if axis is not None:
            axis.relim()
            axis.autoscale_view()

    def _show_empty_preview(self, message: str) -> None:
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
        self.figure.clear()
        self._plot_artists.clear()
        axes = np.atleast_1d(
            self.figure.subplots(1, len(branches), sharey=True)
        ).tolist()
        colors = assign_plot_colors(
            records, str(self.palette_combo.currentData() or "viridis"), resolved_order
        )
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
                    linestyle="-" if increasing else "--", marker="o", markersize=3.2,
                    markevery=max(1, len(b_values) // 180),
                    linewidth=2.4 if focused else 0.9,
                    alpha=1.0 if focused else 0.28,
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
        title = f"{series.variable} comparison"
        if fixed_text:
            title += f" · {fixed_text}"
        self.figure.suptitle(title, fontsize=12, fontweight="bold")
        if handles and len(handles) <= 12:
            self.figure.legend(
                handles, plotted_labels, fontsize=7, title="Processed energy / condition",
                title_fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.13),
                ncol=2, frameon=True,
            )
        elif len(handles) > 12:
            self.figure.text(
                0.5, 0.01,
                "Use the condition checklist at the top to identify or remove curves.",
                ha="center", va="bottom", fontsize=8, color="#666",
            )
        self.figure.tight_layout(rect=(0.0, 0.16 if len(handles) <= 12 else 0.0, 1.0, 0.93))
        self.canvas.draw_idle()
        self._update_slope_preview(records)

    def _update_slope_preview(self, records: list[ProcessedMcdRecord]) -> None:
        from matplotlib.lines import Line2D

        self.slope_figure.clear()
        self._slope_lines.clear()
        axis = self.slope_figure.add_subplot(111)
        series = self._current_series()
        if series is None or series.variable != "E-field":
            axis.text(
                0.5, 0.5,
                "Slope vs E-field is available when Compare different is E-field.",
                transform=axis.transAxes, ha="center", va="center", color="#666",
            )
            axis.set_axis_off()
            self.slope_canvas.draw_idle()
            return
        ordered, _ = order_mcd_records(records, "E-field")
        x = np.asarray([record.condition_value("E-field") for record in ordered], float)
        inc = np.asarray([
            record.increasing_slope_per_t if record.increasing_slope_per_t is not None else np.nan
            for record in ordered
        ], float)
        dec = np.asarray([
            record.decreasing_slope_per_t if record.decreasing_slope_per_t is not None else np.nan
            for record in ordered
        ], float)
        plotted_any = False
        for values, label, linestyle, filled in (
            (inc, "B increasing", "-", True),
            (dec, "B decreasing", "--", False),
        ):
            valid = np.isfinite(x) & np.isfinite(values)
            if not np.any(valid):
                continue
            plotted_any = True
            line, = axis.plot(
                x[valid], values[valid], linestyle=linestyle, marker="o",
                linewidth=2.4 if self._focused_record_id is None else 1.5,
                markersize=5, color="#3568a8",
                markerfacecolor="#3568a8" if filled else "white",
                markeredgecolor="#3568a8", label=label,
            )
            self._slope_lines["increasing" if filled else "decreasing"] = line
        axis.axhline(0.0, color="#555", linewidth=0.7)
        axis.set_xlabel("E-field")
        axis.set_ylabel("Low-field MCD slope (per T)")
        axis.set_title("MCD slope vs E-field")
        axis.grid(alpha=0.25)
        if plotted_any:
            axis.legend(loc="best", frameon=True)
        self.slope_figure.tight_layout()
        self.slope_canvas.draw_idle()

    def _export(self) -> None:
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
        try:
            paths = export_mcd_extract(
                records,
                self.output_label.text(),
                branches=branches,
                order_by="Auto",
                palette=str(self.palette_combo.currentData() or "viridis"),
                export_csv=self.export_csv_chk.isChecked(),
                series_groups=selected_groups,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "MCD export failed", str(exc))
            return
        QMessageBox.information(
            self,
            "MCD export complete",
            "Created:\n" + "\n".join(path.name for path in paths.values()),
        )
