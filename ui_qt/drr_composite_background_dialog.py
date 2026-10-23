"""Build a reusable external background from immutable source snapshots."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QDoubleValidator, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QHeaderView, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSizePolicy, QSplitter, QTableView, QVBoxLayout, QWidget,
)

from core.drr_composite_background import (
    CompositeBackground, CompositeRegion, assign_region, compose_background, composite_metadata_path,
    effective_generation_regions, source_generation,
    is_composite_background, load_composite_source, read_composite_background,
    save_composite_background,
)
from core.drr_sources import resolve_source_path
from core.load_cancellation import LoadCancellation, check_load_cancelled
from ui_qt.common import QComboBox, QDoubleSpinBox, UI_METRICS, Worker
from ui_qt.fluent_ui.style import set_fluent_property
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
from ui_qt.theme import alias, theme_manager


_FRAMES = {"first": "First frame", "last": "Last frame", "all": "All frames (mean)"}


def manual_pn_derived_document(document, parent_version=None):
    """A hand-edited combination cannot retain its parent's algorithm identity."""
    sources = []
    for source in document.sources:
        report = source_generation(source)
        method = str(report.get("algorithm_version") or report.get("algorithm") or "")
        if (parent_version is not None and source.key == document.default_source) or method.startswith("pn_"):
            parent = (dict(version_id=parent_version.version_id,
                           algorithm_version=parent_version.algorithm_version,
                           fingerprint=parent_version.fingerprint) if parent_version is not None else
                      dict(algorithm_version=method, source_key=source.key))
            derived = dict(algorithm_version="manual_pn_derived_v1", algorithm="manual_pn_derived",
                scientifically_validated=False, parent_generation_report=report, derived_from_pn_version=parent)
            for field in ("estimated_regions", "unresolved_regions", "unsupported_regions", "texture_missing_regions"):
                if field in report:
                    derived[field] = report[field]
            source = replace(source, generation_json=json.dumps(derived, ensure_ascii=False, allow_nan=False))
        sources.append(source)
    parent_sources = [s.key for s in sources
                      if source_generation(s).get("algorithm_version") == "manual_pn_derived_v1"]
    if parent_sources:
        sources = [replace(s, generation_json=json.dumps(dict(
            algorithm_version="manual_pn_derived_v1", algorithm="manual_pn_derived",
            scientifically_validated=False, parent_generation_report=source_generation(s),
            derived_from_pn_source_keys=parent_sources), ensure_ascii=False, allow_nan=False))
            if s.key == document.default_source and source_generation(s).get("algorithm_version") != "manual_pn_derived_v1"
            else s for s in sources]
    return replace(document, sources=tuple(sources))


def _run_job(callback, *, progress, log):
    return callback()


def _initial_document(folder, files, which, regions):
    if len(files) == 1 and not regions:
        path = resolve_source_path(folder, files[0])
        if is_composite_background(path):
            return read_composite_background(path)
    sources = []
    by_recipe = {}
    for group, method in [(tuple(files), which),
                          *((tuple(r.baseline_files), r.baseline_which) for r in regions)]:
        check_load_cancelled()
        recipe = (group, method)
        if recipe not in by_recipe:
            source = load_composite_source(folder, group, method)
            sources.append(source)
            by_recipe[recipe] = source.key
    return CompositeBackground(tuple(sources), sources[0].key, tuple(
        CompositeRegion(r.xmin_ev, r.xmax_ev, by_recipe[(tuple(r.baseline_files), r.baseline_which)])
        for r in regions))


def _add_sources(folder, files, which, grouped, document):
    groups = [tuple(files)] if grouped else [(name,) for name in files]
    added = []
    for group in groups:
        check_load_cancelled()
        added.append(load_composite_source(folder, group, which))
    sources = (*document.sources, *added) if document else tuple(added)
    result = CompositeBackground(tuple(sources), document.default_source if document else added[0].key,
                                 document.regions if document else ())
    compose_background(result)
    return result, added[-1].key


def _reload_source(folder, document, key, which):
    old = next(source for source in document.sources if source.key == key)
    if old.generation_json:
        raise ValueError("Generated sources are frozen snapshots. Select an original source to change frames or reload.")
    updated = load_composite_source(folder, old.files, which, key=key, label=old.label)
    result = replace(document, sources=tuple(updated if source.key == key else source
                                             for source in document.sources))
    compose_background(result)
    return result


def _suggest_auto(document, reconstruct_gaps, max_gap_ev, preserve_regions):
    from core.drr_auto_composite import AutoCompositeSettings, suggest_composite_background
    return suggest_composite_background(document,
        settings=AutoCompositeSettings(reconstruct_gaps=reconstruct_gaps, max_gap_ev=max_gap_ev),
        preserve_regions=preserve_regions)


class DrrCompositeBackgroundDialog(QDialog):
    """Source I/O runs in workers; every edit replaces a complete draft document."""
    stateChanged = Signal()

    def __init__(self, parent=None, *, folder="", initial_files=(), initial_which="last",
                 initial_regions=(), initial_document=None, select_files, pool=None,
                 apply_callback=None):
        super().__init__(parent)
        self.setWindowTitle("DRR · Edit / combine backgrounds")
        self.setObjectName("drr_composite_background_dialog")
        self.resize(1080, 830)
        self.setMinimumSize(820, 660)
        self.folder = str(folder)
        self.document = initial_document
        self.saved_path = None
        self._saved_document = None
        self.applied_background = None
        self._apply_callback = apply_callback
        self.protected_paths = set()
        self._initial_files = tuple(initial_files)
        self._initial_which = initial_which
        self._initial_regions = tuple(initial_regions)
        self._select_files = select_files
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._worker = None
        self._job = None
        self._job_document = None
        self._job_selected_key = None
        self._closed = False
        self._refreshing = False
        self._undo = []
        self._selected_region = None
        self._selector = None

        layout = QVBoxLayout(self)
        layout.setSpacing(UI_METRICS["row_spacing"])
        hint = QLabel("Add background files, select a source, then drag an energy interval and choose Use source. "
                      "The bold curve is the combined background. Unassigned ranges use the default source.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        actions = QHBoxLayout()
        self.add_button = QPushButton("&Add sources…")
        self.remove_source_button = QPushButton("Remove source")
        self.group_check = QCheckBox("Average selected files as one source")
        self.group_check.setToolTip("Unchecked: one source per selected file. Checked: average the selected file group.")
        actions.addWidget(self.add_button)
        actions.addWidget(self.remove_source_button)
        actions.addWidget(self.group_check)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.source_model = QStandardItemModel(0, 3, self)
        self.source_model.setHorizontalHeaderLabels(["Source", "Frames", "Files"])
        self.source_table = self._table(self.source_model, "Background source snapshots")
        self.source_table.setObjectName("drr_composite_sources")
        self.source_table.setMinimumHeight(105)
        self.source_table.setMaximumHeight(185)
        self.source_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.source_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.source_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.source_table)
        controls = QHBoxLayout()
        self.frame_combo = QComboBox()
        self.frame_combo.setAccessibleName("Frame selection for the selected source")
        for value, label in _FRAMES.items():
            self.frame_combo.addItem(label, value)
        self.frame_combo.setCurrentIndex(max(0, self.frame_combo.findData(initial_which)))
        self.reload_button = QPushButton("Reload source")
        self.reload_button.setToolTip("Read original files again. A failed read preserves the saved source snapshot.")
        controls.addWidget(QLabel("Selected source frames"))
        controls.addWidget(self.frame_combo)
        controls.addWidget(self.reload_button)
        controls.addStretch(1)
        default_label = QLabel("&Default source")
        self.default_combo = QComboBox()
        self.default_combo.setAccessibleName("Default background source for unassigned energy ranges")
        self.default_combo.setMinimumContentsLength(12)
        self.default_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.default_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        default_label.setBuddy(self.default_combo)
        controls.addWidget(default_label)
        controls.addWidget(self.default_combo, 1)
        layout.addLayout(controls)

        auto_controls = QHBoxLayout()
        self.auto_button = QPushButton("Auto &suggest")
        self.auto_button.setObjectName("drr_composite_auto_suggest")
        self.auto_button.setToolTip("Suggest source choices from at least two original backgrounds. Review the result before saving.")
        self.reconstruct_check = QCheckBox("Reconstruct short gaps")
        self.reconstruct_check.setObjectName("drr_composite_reconstruct_gaps")
        self.reconstruct_check.setToolTip("Estimate short gaps while retaining verifiable interference texture from a source. "
                                         "Without reliable texture, gaps remain unresolved and marked for review.")
        self.max_gap_spin = QDoubleSpinBox()
        self.max_gap_spin.setObjectName("drr_composite_max_gap_mev")
        self.max_gap_spin.setAccessibleName("Maximum reconstructed gap width in millielectronvolts")
        self.max_gap_spin.setRange(1., 1000.)
        self.max_gap_spin.setDecimals(0)
        self.max_gap_spin.setSingleStep(5.)
        self.max_gap_spin.setValue(60.)
        self.max_gap_spin.setSuffix(" meV")
        self.max_gap_spin.setMaximumWidth(115)
        gap_label = QLabel("Max gap")
        gap_label.setBuddy(self.max_gap_spin)
        self.preserve_regions_check = QCheckBox("Keep manual intervals")
        self.preserve_regions_check.setObjectName("drr_composite_preserve_manual")
        self.preserve_regions_check.setChecked(True)
        self.preserve_regions_check.setToolTip("Retain the current interval assignments while suggesting the remaining background.")
        for widget in (self.auto_button, self.reconstruct_check, gap_label,
                       self.max_gap_spin, self.preserve_regions_check):
            auto_controls.addWidget(widget)
        auto_controls.addStretch(1)
        layout.addLayout(auto_controls)

        splitter = QSplitter(Qt.Horizontal)
        self.figure = Figure(constrained_layout=True)
        self.canvas = ThemeAwareFigureCanvasQTAgg(self.figure)
        self.canvas.setAccessibleName("Background sources and combined background; drag an energy interval")
        self.canvas.setMinimumSize(300, 180)
        splitter.addWidget(self.canvas)
        region_panel = QWidget()
        region_layout = QVBoxLayout(region_panel)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.addWidget(QLabel("Assigned intervals"))
        self.region_model = QStandardItemModel(0, 3, self)
        self.region_model.setHorizontalHeaderLabels(["From (eV)", "To (eV)", "Source"])
        self.region_table = self._table(self.region_model, "Assigned background intervals")
        self.region_table.setObjectName("drr_composite_regions")
        self.region_table.setMinimumWidth(240)
        self.region_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.region_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.region_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        region_layout.addWidget(self.region_table, 1)
        region_actions = QHBoxLayout()
        self.new_region_button = QPushButton("New interval")
        self.remove_region_button = QPushButton("Remove interval")
        region_actions.addWidget(self.new_region_button)
        region_actions.addWidget(self.remove_region_button)
        region_layout.addLayout(region_actions)
        splitter.addWidget(region_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([740, 290])
        layout.addWidget(splitter, 1)
        self.splitter = splitter

        interval = QHBoxLayout()
        self.lower_edit = self._energy_edit("Lower interval limit in eV")
        self.upper_edit = self._energy_edit("Upper interval limit in eV")
        lower_label, upper_label = QLabel("&From (eV)"), QLabel("&To (eV)")
        lower_label.setBuddy(self.lower_edit)
        upper_label.setBuddy(self.upper_edit)
        self.use_button = QPushButton("&Use source")
        self.use_button.setToolTip("Assign the selected source to this interval, replacing overlapping assignments.")
        interval.addWidget(lower_label)
        interval.addWidget(self.lower_edit, 1)
        interval.addWidget(upper_label)
        interval.addWidget(self.upper_edit, 1)
        interval.addWidget(self.use_button)
        layout.addLayout(interval)
        self.summary = QLabel("No background sources yet. A measurement is not required.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.auto_review = QLabel()
        self.auto_review.setObjectName("drr_composite_auto_review")
        self.auto_review.setWordWrap(True)
        self.auto_review.setAccessibleName("Review required for automatically suggested background ranges")
        layout.addWidget(self.auto_review)
        self.status = QLabel("Add sources or open a saved composite background.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        bottom = QHBoxLayout()
        self.open_button = QPushButton("&Open composite…")
        self.undo_button = QPushButton("&Undo")
        self.undo_button.setShortcut("Ctrl+Z")
        bottom.addWidget(self.open_button)
        bottom.addWidget(self.undo_button)
        bottom.addStretch(1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.save_button = self.buttons.addButton("Save as…", QDialogButtonBox.AcceptRole)
        self.apply_button = self.buttons.addButton("Use background", QDialogButtonBox.ActionRole)
        self.apply_button.setAccessibleName("Apply this background to the current measurement group")
        self.apply_button.clicked.connect(self.apply_background)
        self.cancel_button = self.buttons.button(QDialogButtonBox.Cancel)
        set_fluent_property(self.apply_button, "fluentAppearance", "primary")
        bottom.addWidget(self.buttons)
        layout.addLayout(bottom)
        for button in (self.add_button, self.remove_source_button, self.reload_button, self.new_region_button,
                       self.remove_region_button, self.use_button, self.open_button, self.undo_button,
                       self.save_button, self.apply_button, self.cancel_button, self.auto_button):
            button.setAutoDefault(False)
        self.use_button.setDefault(True)
        self.add_button.clicked.connect(self.add_sources)
        self.remove_source_button.clicked.connect(self.remove_source)
        self.frame_combo.currentIndexChanged.connect(self.reload_source)
        self.reload_button.clicked.connect(self.reload_source)
        self.auto_button.clicked.connect(self.auto_suggest)
        self.reconstruct_check.toggled.connect(self._update_actions)
        self.default_combo.currentIndexChanged.connect(self.change_default)
        self.source_table.selectionModel().currentRowChanged.connect(self._source_selected)
        self.region_table.selectionModel().currentRowChanged.connect(self._region_selected)
        self.new_region_button.clicked.connect(self.new_region)
        self.remove_region_button.clicked.connect(self.remove_region)
        self.lower_edit.editingFinished.connect(self._numeric_selection)
        self.upper_edit.editingFinished.connect(self._numeric_selection)
        self.use_button.clicked.connect(self.use_source)
        self.undo_button.clicked.connect(self.undo)
        self.open_button.clicked.connect(self.open_composite)
        self.save_button.clicked.connect(self.save_as)
        self.buttons.rejected.connect(self.reject)
        self.finished.connect(self._stop)
        manager = theme_manager()
        if manager is not None:
            manager.themeChanged.connect(self._draw)
        if self.document is not None:
            compose_background(self.document)
        self._refresh()
        if self.document is None and self._initial_files:
            QTimer.singleShot(0, self._load_initial)

    @staticmethod
    def _table(model, name):
        table = QTableView()
        table.setModel(model)
        table.setAccessibleName(name)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setTabKeyNavigation(False)
        table.verticalHeader().hide()
        return table

    @staticmethod
    def _energy_edit(name):
        edit = QLineEdit()
        validator = QDoubleValidator(edit)
        validator.setLocale(edit.locale().c())
        edit.setValidator(validator)
        edit.setAccessibleName(name)
        edit.setPlaceholderText("Select or enter energy")
        return edit

    def selected_source(self):
        index = self.source_table.currentIndex()
        if self.document is None or not index.isValid():
            return None
        return self.document.sources[index.row()]

    def _load_initial(self):
        def work():
            document = _initial_document(self.folder, self._initial_files,
                                         self._initial_which, self._initial_regions)
            path = (resolve_source_path(self.folder, self._initial_files[0])
                    if len(self._initial_files) == 1 and not self._initial_regions else None)
            saved_path = str(path) if path is not None and is_composite_background(path) else None
            return document, saved_path
        self._start_job("initial", work, "Loading background source snapshots…")

    def _start_job(self, kind, callback, message):
        if self._worker is not None or self._closed:
            return
        self._job = kind
        self._job_document = self.document
        selected = self.selected_source()
        self._job_selected_key = selected.key if selected is not None else None
        self.status.setText(message)
        self._worker = Worker(_run_job, callback, cancellation=LoadCancellation())
        self._worker.signals.result.connect(self._job_ready, Qt.QueuedConnection)
        self._worker.signals.error.connect(self._job_failed, Qt.QueuedConnection)
        self._worker.signals.finished.connect(self._job_finished, Qt.QueuedConnection)
        self._update_actions()
        self._pool.start(self._worker)

    @Slot(object)
    def _job_ready(self, payload):
        if self._closed:
            return
        if self._job == "save":
            self.saved_path = str(payload)
            self._saved_document = self._job_document
            self.status.setText(f"Saved {Path(self.saved_path).name}. Use background applies it; saving alone keeps the current background unchanged.")
        elif self._job == "apply":
            self.applied_background = payload
            super().accept()
        elif self._job == "add":
            document, key = payload
            self._commit(document, selected_key=key)
            self.status.setText("Sources added. Select a source and assign an energy interval.")
        elif self._job == "auto":
            if self.document is not self._job_document:
                self.status.setText("The draft changed while Auto suggest was running. The outdated suggestion was discarded.")
                return
            self._commit(payload.document, selected_key=self._job_selected_key)
            self.status.setText("Automatic suggestion added to the draft. Review the source choices and marked ranges before Save as.")
        else:
            saved_path = None
            if self._job in {"open", "initial"}:
                payload, saved_path = payload
            key = self.selected_source().key if self.selected_source() is not None else None
            self._commit(payload, selected_key=key, remember=self._job != "initial",
                         edited=self._job not in {"open", "initial"})
            if saved_path:
                self.saved_path = saved_path
                self._saved_document = self.document
            self.status.setText("Source snapshots ready. Drag a range or select an existing interval to edit.")

    @Slot(str)
    def _job_failed(self, message):
        if self._closed or self._worker is None or self._worker.cancellation.cancelled:
            return
        self.status.setText("Could not complete action: " + message.split("\n\n", 1)[0] +
                            " The current source snapshots and draft are unchanged.")
        self._sync_source_controls()

    @Slot()
    def _job_finished(self):
        self._worker = None
        self._job = None
        self._job_document = None
        self._job_selected_key = None
        if not self._closed:
            self._update_actions()

    def _update_actions(self, *_):
        busy = self._worker is not None
        has_document = self.document is not None
        source = self.selected_source()
        selected = source is not None
        generated = selected and bool(source.generation_json)
        for widget in (self.add_button, self.open_button, self.group_check):
            widget.setEnabled(not busy)
        for widget in (self.source_table, self.default_combo, self.region_table,
                       self.new_region_button, self.lower_edit, self.upper_edit, self.canvas):
            widget.setEnabled(not busy and has_document)
        for widget in (self.frame_combo, self.reload_button, self.remove_source_button, self.use_button):
            widget.setEnabled(not busy and selected)
        self.frame_combo.setEnabled(not busy and selected and not generated)
        self.reload_button.setEnabled(not busy and selected and not generated)
        # The frame selector also controls the method for newly added sources.
        if not has_document:
            self.frame_combo.setEnabled(not busy)
        originals = sum(not item.generation_json for item in self.document.sources) if has_document else 0
        self.auto_button.setEnabled(not busy and originals >= 2)
        self.reconstruct_check.setEnabled(not busy and originals >= 2)
        self.preserve_regions_check.setEnabled(not busy and originals >= 2)
        self.max_gap_spin.setEnabled(not busy and originals >= 2 and self.reconstruct_check.isChecked())
        self.undo_button.setEnabled(not busy and bool(self._undo))
        self.save_button.setEnabled(not busy and has_document)
        self.remove_region_button.setEnabled(not busy and self._selected_region is not None)
        self.cancel_button.setEnabled(self._job != "save")
        active_keys = ({self.document.default_source, *(r.source_key for r in self.document.regions)}
                       if has_document else set())
        needs_snapshot = has_document and any(source.generation_json for source in self.document.sources
                                              if source.key in active_keys)
        saved_current = bool(self.saved_path and self.document is self._saved_document)
        can_apply = saved_current or (self._apply_callback is not None and not needs_snapshot)
        self.apply_button.setEnabled(bool(not busy and has_document and can_apply))
        self.apply_button.setToolTip("Save this generated background before applying it."
                                    if needs_snapshot and not saved_current else
                                    "Apply the current background; unsaved ordinary source intervals remain a recipe.")
        self.stateChanged.emit()

    def apply_background(self):
        if not self.apply_button.isEnabled() or self._closed:
            return
        document = self.document
        path = self.saved_path if document is self._saved_document else None
        def work():
            if self._apply_callback is not None:
                return self._apply_callback(document, path)
            read_composite_background(path)
            return path
        self._start_job("apply", work, "Checking this background before applying…")

    def _commit(self, document, *, selected_key=None, remember=True, edited=True):
        if document is not None:
            if edited:
                document = manual_pn_derived_document(document)
            compose_background(document)
        if remember:
            self._undo.append(self.document)
            self._undo = self._undo[-100:]
        self.document = document
        self._selected_region = None
        self._refresh(selected_key)

    def _refresh(self, selected_key=None):
        self._refreshing = True
        try:
            self.source_model.removeRows(0, self.source_model.rowCount())
            self.region_model.removeRows(0, self.region_model.rowCount())
            self.default_combo.clear()
            if self.document is not None:
                labels = {source.key: source.label for source in self.document.sources}
                for source in self.document.sources:
                    generated = source_generation(source)
                    frames = ("Estimated" if generated.get("estimated_regions") else "Generated") if source.generation_json else _FRAMES[source.which]
                    values = [source.label, frames, "; ".join(Path(name).name for name in source.files)]
                    items = [QStandardItem(value) for value in values]
                    for item in items:
                        item.setToolTip(("Generated snapshot; frame changes and reload are unavailable.\n"
                                         if source.generation_json else "") + "\n".join(source.files))
                    self.source_model.appendRow(items)
                    self.default_combo.addItem(source.label, source.key)
                    self.default_combo.setItemData(self.default_combo.count() - 1,
                        source.label + "\n" + "\n".join(source.files), Qt.ToolTipRole)
                self.default_combo.setCurrentIndex(self.default_combo.findData(self.document.default_source))
                self.default_combo.setToolTip(self.default_combo.currentData(Qt.ToolTipRole) or "")
                for region in self.document.regions:
                    self.region_model.appendRow([QStandardItem(f"{region.xmin_ev:.12g}"),
                        QStandardItem(f"{region.xmax_ev:.12g}"), QStandardItem(labels[region.source_key])])
                row = next((i for i, source in enumerate(self.document.sources) if source.key == selected_key), 0)
                self.source_table.setCurrentIndex(self.source_model.index(row, 0))
                energy, _, _ = compose_background(self.document)
                if not self.lower_edit.text() or not self.upper_edit.text():
                    self.lower_edit.setText(repr(float(np.min(energy))))
                    self.upper_edit.setText(repr(float(np.max(energy))))
        finally:
            self._refreshing = False
        self._sync_source_controls()
        self._update_actions()
        self._draw()

    def _sync_source_controls(self):
        source = self.selected_source()
        if source is not None:
            blocked = self.frame_combo.blockSignals(True)
            self.frame_combo.setPlaceholderText("Generated snapshot" if source.generation_json else "")
            self.frame_combo.setCurrentIndex(-1 if source.generation_json else self.frame_combo.findData(source.which))
            self.frame_combo.blockSignals(blocked)

    def _source_selected(self, *_):
        if self._refreshing:
            return
        self._sync_source_controls()
        self._update_actions()
        self._draw()

    def _region_selected(self, index, *_):
        if self._refreshing or not index.isValid() or self.document is None:
            return
        self._selected_region = index.row()
        region = self.document.regions[index.row()]
        self.lower_edit.setText(repr(region.xmin_ev))
        self.upper_edit.setText(repr(region.xmax_ev))
        row = next(i for i, source in enumerate(self.document.sources) if source.key == region.source_key)
        self.source_table.setCurrentIndex(self.source_model.index(row, 0))
        self.status.setText("Drag the selected interval's handles or edit its limits, then choose Use source to update it.")
        self._update_actions()
        self._draw()

    def add_sources(self):
        if self._worker is not None:
            return
        files = self._select_files([])
        if not files:
            return
        document, which, grouped = self.document, self.frame_combo.currentData() or "last", self.group_check.isChecked()
        self._start_job("add", lambda: _add_sources(self.folder, tuple(files), which, grouped, document),
                        "Reading background files and checking their spectral grids…")

    def reload_source(self, *_):
        if self._refreshing or self._worker is not None or self.document is None:
            return
        source = self.selected_source()
        if source is None:
            return
        if source.generation_json:
            self.status.setText("Generated sources are frozen snapshots. Select an original source to change frames or reload.")
            self._sync_source_controls()
            return
        document, key, which = self.document, source.key, self.frame_combo.currentData()
        self._start_job("reload", lambda: _reload_source(self.folder, document, key, which),
                        "Reading the original files with the selected frame method…")

    def auto_suggest(self):
        document = self.document
        if self._worker is not None or document is None or self._closed:
            return
        if sum(not source.generation_json for source in document.sources) < 2:
            self.status.setText("Add at least two original background sources before requesting an automatic suggestion.")
            return
        reconstruct = self.reconstruct_check.isChecked()
        max_gap_ev = self.max_gap_spin.value() / 1000.
        preserve = self.preserve_regions_check.isChecked()
        self._start_job("auto", lambda: _suggest_auto(document, reconstruct, max_gap_ev, preserve),
                        "Comparing original backgrounds and preparing a suggestion for review…")

    def remove_source(self):
        source = self.selected_source()
        if source is None or self._worker is not None:
            return
        if len(self.document.sources) == 1:
            self._commit(None)
        elif source.key == self.document.default_source or any(r.source_key == source.key for r in self.document.regions):
            self.status.setText("This source is in use. Choose another default and replace its assigned intervals before removing it.")
            return
        else:
            self._commit(replace(self.document, sources=tuple(s for s in self.document.sources if s.key != source.key)))
        self.status.setText("Source removed from the draft. Undo restores it.")

    def change_default(self, *_):
        if self._refreshing or self._worker is not None or self.document is None:
            return
        key = self.default_combo.currentData()
        if key is None or key == self.document.default_source:
            return
        self._commit(replace(self.document, default_source=key,
                             regions=tuple(r for r in self.document.regions if r.source_key != key)),
                     selected_key=key)
        self.status.setText("Default source updated. Assigned intervals keep their selected sources.")

    def new_region(self):
        self._selected_region = None
        self.region_table.clearSelection()
        self.region_table.setCurrentIndex(self.region_model.index(-1, -1))
        self.status.setText("Drag a new interval or enter its limits, then choose Use source.")
        self._update_actions()

    def _selection_limits(self):
        lower, upper = float(self.lower_edit.text()), float(self.upper_edit.text())
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("Enter two finite, increasing energy limits.")
        return lower, upper

    def _span_selected(self, lower, upper):
        if self._worker is not None or lower >= upper:
            return
        self.lower_edit.setText(repr(float(lower)))
        self.upper_edit.setText(repr(float(upper)))
        self.status.setText("Interval selected. Choose Use source to apply it to the combined background.")

    def _numeric_selection(self):
        try:
            limits = self._selection_limits()
        except ValueError:
            return
        if self._selector is not None:
            self._selector.extents = limits
            self.canvas.draw_idle()

    def use_source(self):
        source = self.selected_source()
        if source is None or self._worker is not None:
            return
        try:
            lower, upper = self._selection_limits()
            base = self.document
            if self._selected_region is not None:
                base = replace(base, regions=tuple(r for i, r in enumerate(base.regions)
                                                   if i != self._selected_region))
            updated = assign_region(base, lower, upper, source.key)
            self._commit(updated, selected_key=source.key)
            self.status.setText("Interval assigned. Overlaps were replaced; Undo restores the previous document.")
        except ValueError as exc:
            self.status.setText(f"Cannot assign interval: {exc}")

    def remove_region(self):
        if self.document is None or self._selected_region is None or self._worker is not None:
            return
        self._commit(replace(self.document, regions=tuple(r for i, r in enumerate(self.document.regions)
                                                         if i != self._selected_region)))
        self.status.setText("Interval removed. This range now uses the default source.")

    def undo(self):
        if self._undo and self._worker is None:
            self.document = self._undo.pop()
            self._selected_region = None
            self._refresh()
            self.status.setText("Restored the previous background document.")

    def open_composite(self):
        if self._worker is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open composite background", self.folder,
                                             "Composite backgrounds (*.csv)")
        if path:
            self._start_job("open", lambda: (read_composite_background(path), path), "Opening saved source snapshots…")

    def save_as(self):
        if self.document is None or self._worker is not None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save composite background",
            str(Path(self.folder) / "composite_background.csv"), "CSV background (*.csv)")
        if not path:
            return
        appended_suffix = Path(path).suffix.lower() != ".csv"
        if appended_suffix:
            path += ".csv"
        destination = Path(path)
        from core.drr_pn_store import DrrPnStore
        protected_store = DrrPnStore(self.folder).directory.resolve()
        if destination.resolve() in self.protected_paths or destination.resolve().is_relative_to(protected_store):
            self.status.setText("P/N version history is preserved. Save the edited background outside the P/N version store with a new name.")
            return
        if ((appended_suffix and destination.exists())
                or (not destination.exists() and composite_metadata_path(destination).exists())):
            # The native dialog did not inspect the appended CSV name or an
            # orphaned metadata sidecar, so it could not confirm this overwrite.
            answer = QMessageBox.question(self, "Replace existing composite?",
                f"Background files already exist for {destination.name}. Replace them?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        document = self.document
        self._start_job("save", lambda: save_composite_background(path, document, overwrite=True),
                        "Saving the combined background and editable source snapshots…")

    def _draw_diagnostics(self):
        """Mark only diagnostics belonging to the sources actually in use."""
        styles = (
            ("estimated_regions", "Estimated (model)", "///", "warning_foreground"),
            ("unresolved_regions", "Unresolved (raw)", "xxx", "danger_foreground"),
            ("unsupported_regions", "Unsupported", "...", "text_secondary"),
        )
        self._diagnostic_spans = {field: effective_generation_regions(self.document, field)
                                  for field, _, _, _ in styles}
        self._texture_missing_spans = effective_generation_regions(self.document, "texture_missing_regions")
        handles, labels = [], []
        for field, label, hatch, color in styles:
            for index, (lower, upper) in enumerate(self._diagnostic_spans[field]):
                patch = self.ax.axvspan(lower, upper, facecolor="none", edgecolor=alias(color),
                                       hatch=hatch, linewidth=.6, zorder=.8)
                if index == 0:
                    handles.append(patch)
                    labels.append(label)

        energy = self.document.sources[0].energy
        edges = sorted({float(energy[0]), float(energy[-1]),
                        *(point for region in self.document.regions for point in (region.xmin_ev, region.xmax_ev))})
        active = {next((region.source_key for region in self.document.regions
                        if region.xmin_ev <= lower and region.xmax_ev >= upper), self.document.default_source)
                  for lower, upper in zip(edges[:-1], edges[1:])}
        has_generated = any(source.generation_json and source.key in active for source in self.document.sources)
        self.auto_review.setVisible(has_generated)
        if has_generated:
            merged = []
            for lower, upper in sorted(pair for spans in self._diagnostic_spans.values() for pair in spans):
                if merged and lower <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
                else:
                    merged.append((lower, upper))
            counts = [(name.removesuffix("_regions"), len(spans)) for name, spans in self._diagnostic_spans.items() if spans]
            text = f"Review {len(merged)} marked range(s)"
            if counts:
                text += ": " + ", ".join(f"{name} {count}" for name, count in counts)
            text += ". Automatic source choices need review before saving."
            if self._diagnostic_spans["estimated_regions"]:
                text += " Estimated ranges are model estimates."
            if self._diagnostic_spans["unresolved_regions"]:
                text += " Unresolved ranges retain a raw source."
            if self._diagnostic_spans["unsupported_regions"]:
                text += " Unsupported ranges have insufficient support."
            if self._texture_missing_spans:
                text += f" {len(self._texture_missing_spans)} range(s) lack reliable interference texture and remain unresolved."
            self.auto_review.setText(text)
            self.auto_review.setAccessibleDescription(text)
        else:
            self.auto_review.clear()
        return handles, labels

    @Slot(object)
    def _draw(self, *_):
        if self._closed:
            return
        if self._selector is not None:
            self._selector.disconnect_events()
            self._selector = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Intensity")
        if self.document is None:
            text = self.ax.text(.5, .5, "Add sources or open a composite background",
                ha="center", va="center", transform=self.ax.transAxes, wrap=True)
            text._use_theme_text = True
            self.summary.setText("No background sources yet. A measurement is not required.")
            self._diagnostic_spans = {}
            self._texture_missing_spans = ()
            self.auto_review.clear()
            self.auto_review.hide()
        else:
            energy, background, boundaries = compose_background(self.document)
            order = np.argsort(energy)
            selected = self.selected_source()
            colors = rcParams["axes.prop_cycle"].by_key()["color"]
            for i, source in enumerate(self.document.sources):
                source_order = np.argsort(source.energy)
                name = source.label if len(source.label) <= 32 else source.label[:29] + "…"
                self.ax.plot(source.energy[source_order], source.intensity[source_order],
                    color=colors[i % len(colors)], alpha=1.,
                    linewidth=1.8 if source is selected else 1.2,
                    label=name if not source.generation_json or source is selected else "_nolegend_")
            self.ax.plot(energy[order], background[order], color=alias("text_primary"),
                         linewidth=2.8, label="Combined background", zorder=5)
            for boundary in boundaries:
                self.ax.axvline(boundary, color=alias("text_secondary"), linestyle="--", linewidth=.8)
            handles, labels = self.ax.get_legend_handles_labels()
            if len(handles) > 5:
                handles, labels = handles[:4] + handles[-1:], labels[:4] + labels[-1:]
            diagnostic_handles, diagnostic_labels = self._draw_diagnostics()
            handles += diagnostic_handles
            labels += diagnostic_labels
            self.ax.legend(handles, labels, loc="best", fontsize="small",
                           ncols=2 if diagnostic_handles else 1)
            # SpanSelector creates its patch at zero before setting extents;
            # keep that construction detail out of the spectral view limits.
            self.ax.set_xlim(float(np.min(energy)), float(np.max(energy)))
            self._selector = SpanSelector(self.ax, self._span_selected, "horizontal", useblit=True,
                interactive=True, drag_from_anywhere=False,
                props={"facecolor": alias("brand_background"), "alpha": .15},
                handle_props={"color": alias("brand_foreground"), "linewidth": 1.4})
            try:
                self._selector.extents = self._selection_limits()
            except ValueError:
                pass
            jumps = []
            sorted_energy, sorted_background = energy[order], background[order]
            for boundary in boundaries:
                right = int(np.searchsorted(sorted_energy, boundary))
                if 0 < right < len(sorted_energy):
                    jumps.append(abs(sorted_background[right] - sorted_background[right - 1]))
            finite_jumps = [value for value in jumps if np.isfinite(value)]
            invalid = int(np.count_nonzero(~np.isfinite(background) | (background == 0)))
            message = (f"{len(self.document.sources)} sources · {len(self.document.regions)} assigned intervals · "
                       f"{len(boundaries)} boundaries · {np.min(energy):.8g}–{np.max(energy):.8g} eV")
            if finite_jumps:
                message += f" · Largest adjacent boundary jump: {max(finite_jumps):.5g}"
            if invalid:
                message += f" · {invalid} zero/non-finite samples remain invalid in DRR"
            self.summary.setText(message)
        self.canvas.draw_idle()

    def reject(self):
        if self._job == "save":
            self.status.setText("Save in progress. You can cancel or use the background after it finishes.")
            return
        super().reject()

    def _stop(self, *_):
        self._closed = True
        if self._selector is not None:
            self._selector.disconnect_events()
        if self._worker is not None and self._job != "save":
            self._worker.cancellation.cancel()

    def closeEvent(self, event):
        if self._job == "save":
            event.ignore()
            return
        self._stop()
        super().closeEvent(event)
