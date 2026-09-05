"""Interactive browser for extracting previously processed MCD(B) results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import QThreadPool, QTimer, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.mcd_extract import (
    BRANCHES,
    ORDER_VARIABLES,
    PALETTES,
    McdBranch,
    McdExtractFilters,
    McdSeries,
    ProcessedMcdRecord,
    assign_plot_colors,
    concise_condition_labels,
    discover_processed_mcd,
    energy_cluster_centers,
    export_mcd_extract,
    filter_processed_mcd,
    load_branch_traces,
    newest_mcd_versions,
    organize_mcd_series,
    order_mcd_records,
    record_order_value,
)
from ui_qt.mcd_async import McdScanWorker
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg


def _number_text(value: float | None, decimals: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.{decimals}g}"


class McdExtractDialog(QDialog):
    """Filter, preview, and export processed MCD(B) traces and slopes."""

    COLUMNS = (
        "Use", "Source", "Doping (V)", "E-field (V)", "T (K)",
        "Vtg (V)", "Vbg (V)", "Vbias (V)", "Energy (eV)",
        "Energy group", "Width (meV)", "Increasing slope", "Decreasing slope", "Processed",
    )

    def __init__(self, experiment_root: str | Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("MCD Extract / Compare")
        self.resize(1320, 920)
        self.setMinimumSize(1000, 760)
        self.experiment_root = Path(experiment_root).expanduser()
        self.records: list[ProcessedMcdRecord] = []
        self.all_records: list[ProcessedMcdRecord] = []
        self.older_records: list[ProcessedMcdRecord] = []
        self.filtered_records: list[ProcessedMcdRecord] = []
        self.series_groups: list[McdSeries] = []
        self._included_ids: set[str] = set()
        self._table_refreshing = False
        self._scan_attempted = False
        self._scan_running = False
        self._scan_pending = False
        self._scan_pending_rebuild = False
        self._scan_generation = 0
        self._scan_workers: list[McdScanWorker] = []
        self._trace_array_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self._trace_cache_limit = 128
        self._thread_pool = QThreadPool.globalInstance()
        self._closing = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(90)
        self._preview_timer.timeout.connect(self._update_preview)
        self._build_ui()
        self.folder_label.setText(str(self.experiment_root))
        # Opening the organizer should be enough to see the catalog.  The
        # single-shot runs after the modal window appears, so construction
        # remains quick and the first paint is not delayed.
        QTimer.singleShot(0, self._scan)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        source_row = QHBoxLayout()
        self.folder_label = QLabel()
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.folder_label.setWordWrap(True)
        self.browse_btn = QPushButton("Choose experiment folder…")
        self.refresh_btn = QPushButton("Refresh")
        self.rebuild_btn = QPushButton("Rebuild catalog")
        self.rebuild_btn.setToolTip("Discard the fast metadata catalog and inspect every saved result again.")
        source_row.addWidget(QLabel("Search:"))
        source_row.addWidget(self.folder_label, 1)
        source_row.addWidget(self.browse_btn)
        source_row.addWidget(self.refresh_btn)
        source_row.addWidget(self.rebuild_btn)
        layout.addLayout(source_row)

        filters_box = QGroupBox("Filters")
        filters = QGridLayout(filters_box)
        self.doping_combo = QComboBox()
        self.efield_combo = QComboBox()
        self.temperature_combo = QComboBox()
        self.vtg_combo = QComboBox()
        self.vbg_combo = QComboBox()
        self.vbias_combo = QComboBox()
        self.width_combo = QComboBox()
        self.doping_tolerance = self._spin(0.01, 0.0, 1000.0, 4, " V")
        self.efield_tolerance = self._spin(0.01, 0.0, 1000.0, 4, " V")
        self.temperature_tolerance = self._spin(0.1, 0.0, 1000.0, 3, " K")
        self.gate_tolerance = self._spin(0.01, 0.0, 1000.0, 4, " V")
        self.energy_min = self._spin(0.0, 0.0, 10.0, 6, " eV")
        self.energy_max = self._spin(10.0, 0.0, 10.0, 6, " eV")
        self.energy_limit_chk = QCheckBox("Limit energy")
        self.energy_group_tolerance = self._spin(5.0, 0.0, 1000.0, 2, " meV")
        self.increasing_chk = QCheckBox("Increasing branch — solid / filled")
        self.decreasing_chk = QCheckBox("Decreasing branch — dashed / open")
        self.increasing_chk.setChecked(True)
        self.decreasing_chk.setChecked(True)
        self.apply_btn = QPushButton("Apply filters")
        filters.addWidget(QLabel("Doping"), 0, 0)
        filters.addWidget(self.doping_combo, 0, 1)
        filters.addWidget(QLabel("Tolerance"), 0, 2)
        filters.addWidget(self.doping_tolerance, 0, 3)
        filters.addWidget(QLabel("E-field"), 0, 4)
        filters.addWidget(self.efield_combo, 0, 5)
        filters.addWidget(QLabel("Tolerance"), 0, 6)
        filters.addWidget(self.efield_tolerance, 0, 7)
        filters.addWidget(QLabel("Integration width"), 1, 0)
        filters.addWidget(self.width_combo, 1, 1)
        filters.addWidget(self.energy_limit_chk, 1, 2)
        filters.addWidget(self.energy_min, 1, 3)
        filters.addWidget(QLabel("to"), 1, 4)
        filters.addWidget(self.energy_max, 1, 5)
        filters.addWidget(QLabel("Energy grouping ±"), 1, 6)
        filters.addWidget(self.energy_group_tolerance, 1, 7)
        filters.addWidget(QLabel("Temperature"), 2, 0)
        filters.addWidget(self.temperature_combo, 2, 1)
        filters.addWidget(QLabel("±"), 2, 2)
        filters.addWidget(self.temperature_tolerance, 2, 3)
        filters.addWidget(QLabel("Vtg / Vbg / Vbias"), 2, 4)
        gate_row = QWidget()
        gate_layout = QHBoxLayout(gate_row)
        gate_layout.setContentsMargins(0, 0, 0, 0)
        gate_layout.setSpacing(4)
        gate_layout.addWidget(self.vtg_combo)
        gate_layout.addWidget(self.vbg_combo)
        gate_layout.addWidget(self.vbias_combo)
        filters.addWidget(gate_row, 2, 5, 1, 2)
        filters.addWidget(self.gate_tolerance, 2, 7)
        filters.addWidget(self.increasing_chk, 3, 0, 1, 3)
        filters.addWidget(self.decreasing_chk, 3, 3, 1, 3)
        filters.addWidget(self.apply_btn, 3, 7)
        filters.setColumnStretch(1, 1)
        filters.setColumnStretch(5, 1)
        layout.addWidget(filters_box)

        organization_box = QGroupBox("Preview and export organization")
        organization = QGridLayout(organization_box)
        self.order_combo = QComboBox()
        self.order_combo.addItem("Auto", "Auto")
        for variable in ORDER_VARIABLES:
            self.order_combo.addItem(variable, variable)
        self.direction_combo = QComboBox()
        self.direction_combo.addItem("Low → High", False)
        self.direction_combo.addItem("High → Low", True)
        self.palette_combo = QComboBox()
        for palette in PALETTES:
            self.palette_combo.addItem(palette, palette)
        self.export_csv_chk = QCheckBox("Also export branch CSV files")
        self.export_csv_chk.setChecked(False)
        self.export_csv_chk.setToolTip(
            "Optional plain-text copies. The XLSX workbook is the primary Origin-ready export."
        )
        self.series_combo = QComboBox()
        self.series_combo.setMinimumContentsLength(32)
        self.select_series_btn = QPushButton("Select this series")
        self.keep_newest_chk = QCheckBox("Newest processing only")
        self.keep_newest_chk.setChecked(True)
        self.keep_newest_chk.setToolTip(
            "Hide older reprocessing versions of the same source, energy, width, and metric. "
            "Different processed energies are always preserved."
        )
        self.grouped_export_chk = QCheckBox("Add automatically grouped series sheets")
        self.grouped_export_chk.setChecked(True)
        organization.addWidget(QLabel("Detected series"), 0, 0)
        organization.addWidget(self.series_combo, 0, 1, 1, 5)
        organization.addWidget(self.select_series_btn, 0, 6)
        organization.addWidget(QLabel("Color/order by"), 1, 0)
        organization.addWidget(self.order_combo, 1, 1)
        organization.addWidget(QLabel("Direction"), 1, 2)
        organization.addWidget(self.direction_combo, 1, 3)
        organization.addWidget(QLabel("Palette"), 1, 4)
        organization.addWidget(self.palette_combo, 1, 5)
        organization.addWidget(self.keep_newest_chk, 2, 0, 1, 2)
        organization.addWidget(self.grouped_export_chk, 2, 2, 1, 3)
        organization.addWidget(self.export_csv_chk, 2, 5, 1, 2)
        organization.setColumnStretch(1, 1)
        layout.addWidget(organization_box)

        splitter = QSplitter(Qt.Vertical)
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        action_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Select all visible")
        self.clear_btn = QPushButton("Clear visible")
        self.exclude_btn = QPushButton("Exclude highlighted")
        self.export_btn = QPushButton("Export XLSX + PNG…")
        self.status_label = QLabel()
        action_row.addWidget(self.select_all_btn)
        action_row.addWidget(self.clear_btn)
        action_row.addWidget(self.exclude_btn)
        action_row.addWidget(self.status_label, 1)
        action_row.addWidget(self.export_btn)
        table_layout.addLayout(action_row)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        table_layout.addWidget(self.table)
        splitter.addWidget(table_container)

        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(QLabel(
            "Preview — increasing and decreasing sweeps remain separate even when they share B values."
        ))
        self.figure = Figure(figsize=(10, 4), dpi=100, facecolor="white")
        self.canvas = ThemeAwareFigureCanvasQTAgg(self.figure)
        preview_layout.addWidget(self.canvas, 1)
        splitter.addWidget(preview_container)
        self.table.setMinimumHeight(120)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 650])
        layout.addWidget(splitter, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self.browse_btn.clicked.connect(self._browse)
        self.refresh_btn.clicked.connect(self._scan)
        self.rebuild_btn.clicked.connect(lambda: self._scan(rebuild_catalog=True))
        self.apply_btn.clicked.connect(self._apply_filters)
        for combo in (
            self.doping_combo, self.efield_combo, self.temperature_combo,
            self.vtg_combo, self.vbg_combo, self.vbias_combo, self.width_combo,
        ):
            combo.currentIndexChanged.connect(lambda _index: self._apply_filters())
        self.energy_group_tolerance.valueChanged.connect(lambda _value: self._rebuild_table())
        self.order_combo.currentIndexChanged.connect(lambda _index: self._rebuild_table())
        self.direction_combo.currentIndexChanged.connect(lambda _index: self._rebuild_table())
        self.palette_combo.currentIndexChanged.connect(lambda _index: self._rebuild_table())
        self.series_combo.currentIndexChanged.connect(lambda _index: self._request_preview_update())
        self.select_series_btn.clicked.connect(self._select_current_series)
        self.keep_newest_chk.toggled.connect(lambda _checked: self._apply_version_preference())
        # Branch visibility changes are infrequent and callers expect the
        # preview to reflect the checkbox immediately; table/filter bursts use
        # the debounced request path below.
        self.increasing_chk.toggled.connect(lambda _checked: self._update_preview())
        self.decreasing_chk.toggled.connect(lambda _checked: self._update_preview())
        self.table.itemChanged.connect(self._on_item_changed)
        self.select_all_btn.clicked.connect(lambda: self._set_visible_checked(True))
        self.clear_btn.clicked.connect(lambda: self._set_visible_checked(False))
        self.exclude_btn.clicked.connect(self._exclude_highlighted)
        self.export_btn.clicked.connect(self._export)
        close_btn.clicked.connect(self.accept)

    @staticmethod
    def _spin(value: float, minimum: float, maximum: float, decimals: int, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose experiment or processed MCD folder", str(self.experiment_root)
        )
        if folder:
            self.experiment_root = Path(folder)
            self._scan()

    def _scan(self, rebuild_catalog: bool = False) -> None:
        self.folder_label.setText(str(self.experiment_root))
        if self._scan_running:
            self._scan_pending = True
            self._scan_pending_rebuild = self._scan_pending_rebuild or bool(rebuild_catalog)
            self.status_label.setText("Waiting for the current MCD catalog scan…")
            return
        self._scan_running = True
        self._scan_pending = False
        self._scan_pending_rebuild = False
        self._scan_generation += 1
        generation = self._scan_generation
        self.refresh_btn.setEnabled(False)
        self.rebuild_btn.setEnabled(False)
        self.status_label.setText("Loading processed MCD catalog…")
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
        self._scan_attempted = True
        self.all_records = list(records)
        self._apply_version_preference(refresh=False)
        self._included_ids = {record.record_id for record in self.records}
        self._populate_filter_choices()
        self._apply_filters()

    def _on_scan_error(self, generation: int, message: str) -> None:
        if self._closing:
            return
        if generation == self._scan_generation:
            self._scan_attempted = True
            self.status_label.setText(f"MCD catalog scan failed: {message.splitlines()[0]}")

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
        super().closeEvent(event)

    def _apply_version_preference(self, *, refresh: bool = True) -> None:
        newest, older = newest_mcd_versions(self.all_records)
        self.older_records = older
        self.records = newest if self.keep_newest_chk.isChecked() else list(self.all_records)
        valid_ids = {record.record_id for record in self.records}
        self._included_ids.intersection_update(valid_ids)
        if refresh and self._scan_attempted:
            self._included_ids.update(valid_ids)
            self._populate_filter_choices()
            self._apply_filters()

    def _populate_filter_choices(self) -> None:
        def populate(
            combo: QComboBox,
            values: list[float],
            suffix: str,
            tolerance: float,
        ) -> None:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All", None)
            groups: dict[str, list[float]] = {}
            for value in sorted(float(item) for item in values if np.isfinite(item)):
                # The displayed six-significant-digit value is the identity the
                # user sees and selects.  Values that display as 6.3 are one
                # filter choice even if their raw floating-point values differ.
                groups.setdefault(f"{value:.6g}", []).append(value)
            for display_value, group in groups.items():
                center = float(np.mean(group))
                count_text = f" ({len(group)} results)" if len(group) > 1 else ""
                combo.addItem(f"{display_value} {suffix}{count_text}", center)
            combo.blockSignals(False)

        populate(
            self.doping_combo,
            [value for record in self.records if (value := record.condition_value("Doping")) is not None],
            "V",
            self.doping_tolerance.value(),
        )
        populate(
            self.efield_combo,
            [value for record in self.records if (value := record.condition_value("E-field")) is not None],
            "V",
            self.efield_tolerance.value(),
        )
        populate(
            self.temperature_combo,
            [value for record in self.records if (value := record.condition_value("T")) is not None],
            "K",
            self.temperature_tolerance.value(),
        )
        for combo, label in (
            (self.vtg_combo, "Vtg"), (self.vbg_combo, "Vbg"), (self.vbias_combo, "Vbias")
        ):
            populate(
                combo,
                [value for record in self.records if (value := record.condition_value(label)) is not None],
                "V",
                self.gate_tolerance.value(),
            )
        populate(
            self.width_combo,
            [record.width_mev for record in self.records],
            "meV",
            1e-3,
        )
        if self.records:
            energies = [record.center_ev for record in self.records]
            self.energy_min.setValue(min(energies))
            self.energy_max.setValue(max(energies))

    def _current_filters(self) -> McdExtractFilters:
        energy_limited = self.energy_limit_chk.isChecked()
        return McdExtractFilters(
            doping_v=self.doping_combo.currentData(),
            doping_tolerance_v=self.doping_tolerance.value(),
            efield_v=self.efield_combo.currentData(),
            efield_tolerance_v=self.efield_tolerance.value(),
            temperature_k=self.temperature_combo.currentData(),
            temperature_tolerance_k=self.temperature_tolerance.value(),
            vtg_v=self.vtg_combo.currentData(),
            vbg_v=self.vbg_combo.currentData(),
            vbias_v=self.vbias_combo.currentData(),
            gate_tolerance_v=self.gate_tolerance.value(),
            energy_min_ev=self.energy_min.value() if energy_limited else None,
            energy_max_ev=self.energy_max.value() if energy_limited else None,
            width_mev=self.width_combo.currentData(),
        )

    def _apply_filters(self) -> None:
        self.filtered_records = filter_processed_mcd(self.records, self._current_filters())
        self._rebuild_table()

    def _refresh_series_choices(self) -> None:
        previous = self.series_combo.currentData()
        requested = str(self.order_combo.currentData() or "Auto")
        self.series_groups = organize_mcd_series(self.filtered_records, requested)
        blocked = self.series_combo.blockSignals(True)
        self.series_combo.clear()
        for series in self.series_groups:
            self.series_combo.addItem(series.label, series.series_id)
            self.series_combo.setItemData(
                self.series_combo.count() - 1,
                "Each result keeps its own processed energy. " + series.label,
                Qt.ToolTipRole,
            )
        index = self.series_combo.findData(previous)
        self.series_combo.setCurrentIndex(index if index >= 0 else (0 if self.series_groups else -1))
        self.series_combo.blockSignals(blocked)

    def _current_series(self) -> McdSeries | None:
        series_id = self.series_combo.currentData()
        return next(
            (series for series in self.series_groups if series.series_id == series_id),
            None,
        )

    def _select_current_series(self) -> None:
        series = self._current_series()
        if series is None:
            return
        self._included_ids = {record.record_id for record in series.records}
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        ordered_records, _resolved_order = self._ordered_records()
        self._refresh_series_choices()
        clusters = energy_cluster_centers(
            ordered_records, self.energy_group_tolerance.value()
        )
        self._table_refreshing = True
        self.table.setRowCount(0)
        for record in ordered_records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            use_item = QTableWidgetItem()
            use_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            use_item.setCheckState(
                Qt.Checked if record.record_id in self._included_ids else Qt.Unchecked
            )
            use_item.setData(Qt.UserRole, record.record_id)
            self.table.setItem(row, 0, use_item)
            values = (
                record.source_file,
                _number_text(record.condition_value("Doping")),
                _number_text(record.condition_value("E-field")),
                _number_text(record.condition_value("T")),
                _number_text(record.condition_value("Vtg")),
                _number_text(record.condition_value("Vbg")),
                _number_text(record.condition_value("Vbias")),
                f"{record.center_ev:.6f}",
                f"{clusters[record.record_id]:.6f}",
                _number_text(record.width_mev),
                _number_text(record.increasing_slope_per_t),
                _number_text(record.decreasing_slope_per_t),
                record.created_utc[:19].replace("T", " ") if record.created_utc else "—",
            )
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, record.record_id)
                if self.COLUMNS[column] == "T (K)":
                    item.setToolTip(
                        f"Temperature source: {record.condition_sources.get('T', 'unresolved')}"
                    )
                self.table.setItem(row, column, item)
        self._table_refreshing = False
        self._update_status()
        self._update_preview()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._table_refreshing or item.column() != 0:
            return
        record_id = str(item.data(Qt.UserRole))
        if item.checkState() == Qt.Checked:
            self._included_ids.add(record_id)
        else:
            self._included_ids.discard(record_id)
        self._update_status()
        self._request_preview_update()

    def _set_visible_checked(self, checked: bool) -> None:
        self._table_refreshing = True
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            record_id = str(item.data(Qt.UserRole))
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            (self._included_ids.add if checked else self._included_ids.discard)(record_id)
        self._table_refreshing = False
        self._update_status()
        self._request_preview_update()

    def _exclude_highlighted(self) -> None:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        self._table_refreshing = True
        for row in rows:
            item = self.table.item(row, 0)
            item.setCheckState(Qt.Unchecked)
            self._included_ids.discard(str(item.data(Qt.UserRole)))
        self._table_refreshing = False
        self._update_status()
        self._request_preview_update()

    def _request_preview_update(self) -> None:
        """Coalesce rapid selection/filter changes before loading traces."""
        self._preview_timer.start()

    def _selected_records(self) -> list[ProcessedMcdRecord]:
        visible_ids = {record.record_id for record in self.filtered_records}
        return [
            record for record in self.filtered_records
            if record.record_id in visible_ids and record.record_id in self._included_ids
        ]

    def _ordered_records(self) -> tuple[list[ProcessedMcdRecord], str]:
        return order_mcd_records(
            self.filtered_records,
            str(self.order_combo.currentData() or "Auto"),
            descending=bool(self.direction_combo.currentData()),
        )

    def _ordered_selected_records(self) -> tuple[list[ProcessedMcdRecord], str]:
        return order_mcd_records(
            self._selected_records(),
            str(self.order_combo.currentData() or "Auto"),
            descending=bool(self.direction_combo.currentData()),
        )

    def _selected_branches(self) -> tuple[McdBranch, ...]:
        branches: list[McdBranch] = []
        if self.increasing_chk.isChecked():
            branches.append("B increasing")
        if self.decreasing_chk.isChecked():
            branches.append("B decreasing")
        return tuple(branches)

    def _update_status(self) -> None:
        selected = len(self._selected_records())
        resolved_order = self._ordered_selected_records()[1] if selected else self._ordered_records()[1]
        if not self._scan_attempted:
            self.status_label.setText("Loading processed MCD catalog…")
            self.export_btn.setEnabled(False)
            return
        if self.records:
            hidden = f"; {len(self.older_records)} older version(s) hidden" if (
                self.keep_newest_chk.isChecked() and self.older_records
            ) else ""
            self.status_label.setText(
                f"{len(self.filtered_records)} result(s) in {len(self.series_groups)} series; "
                f"{selected} selected; ordered by {resolved_order}{hidden}"
            )
        else:
            self.status_label.setText("No processed MCD(B) results found")
        self.export_btn.setEnabled(selected > 0 and bool(self._selected_branches()))

    def _update_preview(self) -> None:
        self.figure.clear()
        if not self.records:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5, 0.5, "Loading processed MCD catalog…",
                transform=axis.transAxes, ha="center", va="center", color="#666",
            )
            axis.set_axis_off()
            self.canvas.draw_idle()
            self._update_status()
            return
        selected_ids = {record.record_id for record in self._selected_records()}
        active_series = self._current_series()
        preview_records = (
            [record for record in active_series.records if record.record_id in selected_ids]
            if active_series is not None else self._selected_records()
        )
        records, resolved_order = order_mcd_records(
            preview_records,
            str(self.order_combo.currentData() or "Auto"),
            descending=bool(self.direction_combo.currentData()),
        )
        branches = list(self._selected_branches())
        if not branches:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5, 0.5, "Select at least one sweep branch",
                transform=axis.transAxes, ha="center", va="center", color="#666",
            )
            axis.set_axis_off()
            self.canvas.draw_idle()
            self._update_status()
            return
        axes = np.atleast_1d(
            self.figure.subplots(1, len(branches), sharey=True)
        ).tolist()
        colors = assign_plot_colors(records, str(self.palette_combo.currentData() or "viridis"))
        plotted = 0
        legend_handles: list[Line2D] = []
        legend_labels: list[str] = []
        concise_labels, fixed_text = concise_condition_labels(records, resolved_order)
        for record, label in zip(records, concise_labels):
            try:
                traces = self._trace_arrays(record)
            except (OSError, ValueError):
                continue
            color = colors[record.record_id]
            legend_handles.append(Line2D([0], [0], color=color, marker="o", linewidth=1.5))
            legend_labels.append(label)
            for axis, branch in zip(axes, branches):
                block = traces.get(branch, (np.empty(0), np.empty(0)))
                increasing = branch == "B increasing"
                axis.plot(
                    block[0], block[1],
                    linestyle="-" if increasing else "--",
                    marker="o", markersize=3.5, linewidth=1.3, color=color,
                    markerfacecolor=color if increasing else "white",
                    markeredgecolor=color,
                    label="_nolegend_",
                )
                plotted += 1
        for index, (axis, branch) in enumerate(zip(axes, branches)):
            axis.axhline(0.0, color="#555", linewidth=0.7)
            axis.set_xlabel("B field (T)")
            if index == 0:
                axis.set_ylabel("Corrected signed-mean MCD")
            axis.grid(alpha=0.25)
            axis.set_title("B increasing" if branch == "B increasing" else "B decreasing")
        title = f"Selected MCD(B) series — ordered by {resolved_order}"
        if fixed_text:
            title += f"\n{fixed_text}"
        self.figure.suptitle(title)
        if plotted:
            self.figure.legend(
                legend_handles, legend_labels,
                fontsize=7, loc="upper left", bbox_to_anchor=(0.78, 0.88),
                title="Selected results", title_fontsize=7,
            )
        else:
            axis = axes[0]
            axis.text(0.5, 0.5, "Select results and at least one branch to preview",
                      transform=axis.transAxes, ha="center", va="center", color="#666")
        self.figure.tight_layout(rect=(0.0, 0.0, 0.77, 0.94))
        self.canvas.draw_idle()
        self._update_status()

    def _trace_arrays(self, record: ProcessedMcdRecord) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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

    def _export(self) -> None:
        records = self._selected_records()
        branches = self._selected_branches()
        if not records or not branches:
            QMessageBox.information(self, "MCD Extract", "Select results and at least one branch first.")
            return
        default = self.experiment_root / "Processed Data" / "MCD Extracts"
        folder = QFileDialog.getExistingDirectory(
            self, "Choose MCD extract output folder", str(default)
        )
        if not folder:
            return
        try:
            paths = export_mcd_extract(
                records,
                folder,
                branches=branches,
                filters=self._current_filters(),
                energy_tolerance_mev=self.energy_group_tolerance.value(),
                order_by=str(self.order_combo.currentData() or "Auto"),
                descending=bool(self.direction_combo.currentData()),
                palette=str(self.palette_combo.currentData() or "viridis"),
                export_csv=self.export_csv_chk.isChecked(),
                series_groups=self.series_groups if self.grouped_export_chk.isChecked() else None,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "MCD Extract failed", str(exc))
            return
        QMessageBox.information(
            self,
            "MCD Extract complete",
            "Created:\n" + "\n".join(path.name for path in paths.values()),
        )
