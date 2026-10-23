"""Edit a private external-background recipe and preview it before applying."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QThreadPool, Qt, Signal, Slot
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHeaderView, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSizePolicy, QStyledItemDelegate,
    QTableView, QVBoxLayout,
)

from core.drr_background_regions import DrrBackgroundRegion, normalize_regions, region_masks
from core.load_cancellation import LoadCancellation, check_load_cancelled
from ui_qt.common import QComboBox, UI_METRICS, Worker
from ui_qt.fluent_ui.style import set_fluent_property
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
from ui_qt.theme import alias, theme_manager


_FRAME_LABELS = {"first": "First frame", "last": "Last frame", "all": "All frames (mean)"}


@dataclass
class _RegionDraft:
    xmin_ev: float
    xmax_ev: float
    baseline_files: tuple[str, ...] = ()
    baseline_which: str = "last"


class _RegionModel(QAbstractTableModel):
    HEADERS = ("From (eV)", "To (eV)", "Background files", "Frames")

    def __init__(self, regions, parent=None):
        super().__init__(parent)
        self.rows = [_RegionDraft(r.xmin_ev, r.xmax_ev, tuple(r.baseline_files), r.baseline_which)
                     for r in normalize_regions(regions)]

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self.HEADERS[section] if orientation == Qt.Horizontal else str(section + 1)
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        value = (row.xmin_ev, row.xmax_ev, row.baseline_files, row.baseline_which)[index.column()]
        if role == Qt.EditRole:
            return value
        if role in (Qt.DisplayRole, Qt.AccessibleTextRole):
            if index.column() < 2:
                return f"{value:.12g}"
            if index.column() == 2:
                return "; ".join(Path(name).name for name in value) or "Choose files…"
            return _FRAME_LABELS[value]
        if role == Qt.ToolTipRole:
            if index.column() == 2:
                return "\n".join(value) or "Select this row, then choose one or more background files."
            if index.column() < 2:
                return f"{value!r} eV; lower limit included, upper limit excluded (except the spectrum endpoint)."
            return "Each file uses this frame selection; selected files are then averaged."
        return None

    def flags(self, index):
        flags = super().flags(index)
        return flags | Qt.ItemIsEditable if index.isValid() and index.column() != 2 else flags

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False
        row = self.rows[index.row()]
        field = ("xmin_ev", "xmax_ev", "baseline_files", "baseline_which")[index.column()]
        try:
            if index.column() < 2:
                value = float(value)
                if not np.isfinite(value):
                    return False
            elif index.column() == 2:
                value = tuple(value)
            elif value not in _FRAME_LABELS:
                return False
        except (TypeError, ValueError):
            return False
        if getattr(row, field) == value:
            return False
        setattr(row, field, value)
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole, Qt.ToolTipRole])
        return True

    def append(self, row):
        position = len(self.rows)
        self.beginInsertRows(QModelIndex(), position, position)
        self.rows.append(row)
        self.endInsertRows()

    def remove(self, position):
        self.beginRemoveRows(QModelIndex(), position, position)
        self.rows.pop(position)
        self.endRemoveRows()

    def regions(self):
        return normalize_regions(tuple(DrrBackgroundRegion(
            row.xmin_ev, row.xmax_ev, row.baseline_files, row.baseline_which) for row in self.rows))


class _RegionDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if index.column() == 3:
            editor = QComboBox(parent)
            for value, label in _FRAME_LABELS.items():
                editor.addItem(label, value)
            editor.setAccessibleName("Background frame selection")
            return editor
        if index.column() < 2:
            editor = QLineEdit(parent)
            validator = QDoubleValidator(editor)
            validator.setLocale(editor.locale().c())
            editor.setValidator(validator)
            editor.setAccessibleName("Lower energy limit (eV)" if index.column() == 0 else "Upper energy limit (eV)")
            return editor
        return None

    def setEditorData(self, editor, index):
        value = index.data(Qt.EditRole)
        if isinstance(editor, QComboBox):
            editor.setCurrentIndex(editor.findData(value))
        else:
            # repr preserves the exact spectral endpoint, unlike a rounded spin box.
            editor.setText(repr(value))
            editor.selectAll()

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData() if isinstance(editor, QComboBox) else editor.text())


def _compute_preview(callback, regions, generation, *, progress, log):
    energy, background = callback(regions)
    check_load_cancelled()
    energy = np.asarray(energy, dtype=float).copy()
    background = np.asarray(background, dtype=float).copy()
    region_masks(energy, regions)
    if background.shape != energy.shape:
        raise ValueError("Background preview must match the measurement energy grid.")
    return generation, regions, energy, background


class DrrBackgroundRegionsDialog(QDialog):
    """The preview callback performs I/O in a worker, returning (energy, B).

    Only an accepted dialog supplies a new recipe to its caller. Draft edits,
    source picking, and previewing never mutate the application's assignment.
    """
    applied = Signal(object)

    def __init__(self, parent=None, *, energy=None, regions=(), default_files=(),
                 default_which="last", select_files: Callable[[list[str]], Sequence[str] | None],
                 preview: Callable[[tuple[DrrBackgroundRegion, ...]], tuple[np.ndarray, np.ndarray]],
                 pool=None):
        super().__init__(parent)
        self.setWindowTitle("DRR · Background by X range")
        self.setObjectName("drr_background_regions_dialog")
        self.resize(960, 730)
        self.setMinimumSize(680, 560)
        self.energy = None if energy is None else np.asarray(energy, dtype=float).copy()
        self.default_files = tuple(default_files)
        self.default_which = default_which
        self._select_files, self._preview = select_files, preview
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._worker = None
        self._generation = 0
        self._validated_regions = None
        self._background = None
        self._closed = False

        layout = QVBoxLayout(self)
        layout.setSpacing(UI_METRICS["row_spacing"])
        self.hint = QLabel("Choose a background for each energy interval. Gaps use the default background. "
                           "Intervals cannot overlap. Limits are stored in eV, independently of plot zoom.")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        names = "; ".join(Path(name).name for name in self.default_files) or "No default files selected"
        self.default_label = QLabel(f"Default background: {names} · {_FRAME_LABELS.get(default_which, default_which)}")
        self.default_label.setWordWrap(True)
        self.default_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.default_label.setToolTip("\n".join(self.default_files))
        self.default_label.setAccessibleName("Default background for all uncovered intervals")
        layout.addWidget(self.default_label)

        self.model = _RegionModel(regions, self)
        self.table = QTableView()
        self.table.setObjectName("drr_background_regions_table")
        self.table.setAccessibleName("External background energy intervals")
        self.table.setAccessibleDescription("Edit From, To and Frames; choose background files with the Choose files button.")
        self.table.setModel(self.model)
        self.table.setItemDelegate(_RegionDelegate(self.table))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        # Arrow keys move among cells; Tab leaves the grid instead of cycling
        # inside it, keeping the file and preview actions keyboard-accessible.
        self.table.setTabKeyNavigation(False)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.table.setMinimumHeight(130)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 118)
        self.table.setColumnWidth(1, 118)
        self.table.setColumnWidth(3, 165)
        layout.addWidget(self.table, 1)
        controls = QHBoxLayout()
        self.add_button = QPushButton("&Add interval")
        self.remove_button = QPushButton("&Remove interval")
        self.files_button = QPushButton("Choose &files…")
        for button in (self.add_button, self.remove_button, self.files_button):
            button.setAutoDefault(False)
            controls.addWidget(button)
        controls.addStretch(1)
        self.preview_button = QPushButton("&Preview background")
        self.preview_button.setDefault(True)
        controls.addWidget(self.preview_button)
        layout.addLayout(controls)
        self.range_label = QLabel()
        self.range_label.setWordWrap(True)
        self._update_range_label()
        layout.addWidget(self.range_label)
        self.figure = Figure(constrained_layout=True)
        self.canvas = ThemeAwareFigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(180)
        self.canvas.setAccessibleName("Effective background preview with interval boundaries")
        layout.addWidget(self.canvas, 2)
        self.status = QLabel("Add intervals, then preview the effective background before applying.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        self.apply_button = self.button_box.button(QDialogButtonBox.Apply)
        self.apply_button.setObjectName("drr_background_regions_apply")
        self.apply_button.setAutoDefault(False)
        self.apply_button.setEnabled(False)
        set_fluent_property(self.apply_button, "fluentAppearance", "primary")
        self.button_box.button(QDialogButtonBox.Cancel).setAutoDefault(False)
        layout.addWidget(self.button_box)
        self.add_button.clicked.connect(self.add_region)
        self.remove_button.clicked.connect(self.remove_region)
        self.files_button.clicked.connect(self.choose_files)
        self.table.doubleClicked.connect(self._cell_activated)
        self.table.selectionModel().selectionChanged.connect(self._update_actions)
        self.model.dataChanged.connect(self._edited)
        self.model.rowsInserted.connect(self._edited)
        self.model.rowsRemoved.connect(self._edited)
        self.preview_button.clicked.connect(self.preview_background)
        self.apply_button.clicked.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.finished.connect(self._stop)
        manager = theme_manager()
        if manager is not None:
            manager.themeChanged.connect(self._draw)
        self._update_actions()
        self._draw()
        self.table.setFocus()

    def regions(self) -> tuple[DrrBackgroundRegion, ...]:
        return self.model.regions()

    def _update_range_label(self):
        if self.energy is None:
            text = "Measurement range will be read during preview. Lower limits are included; upper limits are excluded."
        else:
            text = (f"Measurement: {np.min(self.energy):.12g}–{np.max(self.energy):.12g} eV. "
                    "Lower limits are included; upper limits are excluded, except the final spectral point.")
        self.range_label.setText(text)

    def _update_actions(self, *_):
        busy = self._worker is not None
        selected = self.table.currentIndex().isValid()
        self.table.setEnabled(not busy)
        self.add_button.setEnabled(not busy)
        self.remove_button.setEnabled(selected and not busy)
        self.files_button.setEnabled(selected and not busy)
        self.preview_button.setEnabled(not busy)
        self.apply_button.setEnabled(not busy and self._validated_regions is not None)

    def _edited(self, *_):
        self._generation += 1
        self._validated_regions = None
        self._background = None
        if self._worker is not None:
            self._worker.cancellation.cancel()
        self.status.setText("Changes have not been previewed. Preview the background before applying.")
        set_fluent_property(self.table, "fluentInvalid", False)
        self._update_actions()
        self._draw()

    def add_region(self):
        if self._worker is not None:
            return
        lower, upper = ((float(np.min(self.energy)), float(np.max(self.energy)))
                        if self.energy is not None else (0., 1.))
        # Prefer the first uncovered gap; the user can still edit any endpoint.
        for row in sorted(self.model.rows, key=lambda row: row.xmin_ev):
            if lower < row.xmin_ev:
                upper = min(upper, row.xmin_ev)
                break
            lower = max(lower, row.xmax_ev)
        if lower >= upper:
            lower, upper = ((float(np.min(self.energy)), float(np.max(self.energy)))
                            if self.energy is not None else (0., 1.))
        self.model.append(_RegionDraft(lower, upper, (), self.default_which))
        self.table.setCurrentIndex(self.model.index(self.model.rowCount() - 1, 0))
        self.table.setFocus()
        self.table.edit(self.table.currentIndex())

    def remove_region(self):
        index = self.table.currentIndex()
        if index.isValid() and self._worker is None:
            self.model.remove(index.row())
            self._update_actions()

    def _cell_activated(self, index):
        if index.column() == 2:
            self.choose_files()

    def choose_files(self):
        index = self.table.currentIndex()
        if not index.isValid() or self._worker is not None:
            return
        row = index.row()
        files = self._select_files(list(self.model.rows[row].baseline_files))
        if files is not None:
            self.model.setData(self.model.index(row, 2), tuple(files))
        self.files_button.setFocus()

    @Slot()
    def preview_background(self):
        if self._worker is not None or self._closed:
            return
        # Commit a live delegate editor before validating the recipe.
        self.preview_button.setFocus()
        try:
            regions = self.regions()
            if self.energy is not None:
                region_masks(self.energy, regions)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._validated_regions = None
        self._background = None
        self.status.setText("Loading backgrounds and checking the measurement energy grid…")
        self._worker = Worker(_compute_preview, self._preview, regions, self._generation,
                              cancellation=LoadCancellation())
        self._worker.signals.result.connect(self._preview_ready, Qt.QueuedConnection)
        self._worker.signals.error.connect(self._preview_failed, Qt.QueuedConnection)
        self._worker.signals.finished.connect(self._preview_finished, Qt.QueuedConnection)
        self._update_actions()
        self._pool.start(self._worker)

    def _show_error(self, message):
        self._validated_regions = None
        self._background = None
        self.status.setText(f"Cannot apply: {message}")
        self.table.setAccessibleDescription(message)
        set_fluent_property(self.table, "fluentInvalid", True)
        self._update_actions()
        self._draw()

    @Slot(object)
    def _preview_ready(self, payload):
        generation, regions, energy, background = payload
        if self._closed or generation != self._generation:
            return
        self.energy, self._background = energy, background
        self._validated_regions = regions
        self._update_range_label()
        set_fluent_property(self.table, "fluentInvalid", False)
        self.table.setAccessibleDescription("Background files and energy intervals validated. Ready to apply.")
        self.status.setText(f"Preview ready: {len(regions)} interval(s). Dashed lines mark interval boundaries. "
                            "Apply uses this background recipe; Cancel discards edits.")
        invalid = int(np.count_nonzero(~np.isfinite(background) | (background == 0)))
        if invalid:
            self.status.setText(self.status.text() +
                f" {invalid} zero/non-finite background sample(s) will remain invalid in DRR.")
        self._draw()

    @Slot(str)
    def _preview_failed(self, message):
        if not self._closed and self._worker is not None and not self._worker.cancellation.cancelled:
            self._show_error(message.split("\n\n", 1)[0])

    @Slot()
    def _preview_finished(self):
        self._worker = None
        if not self._closed:
            self._update_actions()

    @Slot(object)
    def _draw(self, *_):
        if self._closed:
            return
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Background intensity")
        if self._background is not None:
            order = np.argsort(self.energy)
            self.ax.plot(self.energy[order], self._background[order], color=alias("brand_background"),
                         label="Effective background", linewidth=1.5)
            boundaries = sorted({value for r in self._validated_regions for value in (r.xmin_ev, r.xmax_ev)})
            for position in boundaries:
                self.ax.axvline(position, color=alias("text_secondary"), linestyle="--", linewidth=1)
            self.ax.legend(loc="best")
        else:
            text = self.ax.text(.5, .5, "Preview the background to inspect interval boundaries",
                                ha="center", va="center", transform=self.ax.transAxes, wrap=True)
            text._use_theme_text = True
            if self.energy is not None:
                self.ax.set_xlim(float(np.min(self.energy)), float(np.max(self.energy)))
        self.canvas.draw_idle()

    def accept(self):
        if self._worker is not None or self._validated_regions is None or self._closed:
            return
        if self.regions() != self._validated_regions:
            self._edited()
            return
        self.applied.emit(self._validated_regions)
        super().accept()

    def _stop(self, *_):
        self._closed = True
        self._generation += 1
        if self._worker is not None:
            self._worker.cancellation.cancel()

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)
