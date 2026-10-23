"""Choose a P/N pair and inspect reusable background versions."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThreadPool, QTimer, QElapsedTimer, QModelIndex, Signal, Slot
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QTableView, QVBoxLayout, QHeaderView, QWidget, QScrollArea, QFrame, QSizePolicy,
    QTabWidget, QToolButton, QProgressBar, QLineEdit,
)

from core.drr_composite_background import (
    compose_background, read_composite_background, load_composite_source, composite_metadata_path,
)
from core.drr_pn_generator import ALGORITHM_VERSION, default_settings, generate_pn_background, dependency_paths
from core.drr_pn_pairing import DrrPnPair, describe_manual_pair
from core.drr_pn_selection import build_pn_source_catalog, describe_pn_counterparts, describe_pn_pair
from core.drr_pn_store import DrrPnStore, _file_digest
from core.drr_pn_region_report import describe_pn_regions
from core.drr_sources import inspect_csv_spectral_grid, resolve_source_path
from core.load_cancellation import LoadCancellation, check_load_cancelled
from ui_qt.common import QComboBox, QDoubleSpinBox, Worker, WrappedFilenameDelegate
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
from ui_qt.theme import alias, theme_manager
from ui_qt.time_format import format_local_timestamp
from ui_qt.fluent_ui.style import set_fluent_property
from ui_qt.drr_pn_progress import ReconstructionProgress, TOTAL_STAGES, STAGE_PROGRESS_NOTE
from ui_qt.drr_pn_repeat_selection import choose_training_repeats
from ui_qt.drr_pn_labels import SAVED_COMPOSITE_LABEL, SAVED_COMPOSITE_TOOLTIP, source_frame_summary


# Explicit draft generation is available; saved versions remain the default.
# Availability never marks a generated background scientifically accepted.
RECONSTRUCTION_ENABLED = True


def load_pn_version(folder, version, measurements=()):
    """Load background curves and check headers; sample maps belong to the main view."""
    document = read_composite_background(version.csv_path)
    energy, background, boundaries = compose_background(document)
    payload = dict(version=version, document=document, compatibility_error="", measurement_grid_checked=False)
    references = tuple(source for source in document.sources if source.key != document.default_source)
    if not references and version.fresh:
        references = (
            load_composite_source(folder, version.pair.p_files, version.p_which, key="P", label="P source"),
            load_composite_source(folder, version.pair.n_files, version.n_which, key="N", label="N source"))
    payload["reference_sources"] = references
    try:
        if not np.isfinite(background).all() or np.any(background <= 0):
            raise ValueError("The complete background must be finite and positive.")
        expected = np.asarray(document.sources[0].spectral_columns, dtype=float)
        for name in measurements:
            check_load_cancelled()
            grid = np.asarray(inspect_csv_spectral_grid(resolve_source_path(folder, name)), dtype=float)
            if grid.shape != expected.shape or not np.allclose(grid, expected, rtol=1e-9, atol=1e-10):
                raise ValueError("The complete measurement grid does not match this background.")
        payload["measurement_grid_checked"] = bool(measurements)
    except (ValueError, OSError) as exc:
        payload["compatibility_error"] = str(exc)
    check_load_cancelled()
    return payload


def _run(callback, token, *, progress, log, report_progress=False):
    return token, callback(progress, log) if report_progress else callback()


def _assisted_api():
    # Importing the picker and browsing stored backgrounds must not prepare samples.
    from core import drr_pn_assisted
    return drr_pn_assisted


def _reconstruction_api():
    # Browsing history must not load samples or initialize the fitting chain.
    from core.drr_pn_reconstruction import app
    return app


def _assisted_import_dependencies(folder, path, settings, api):
    """Do not relabel unknown or changed training inputs as a fresh import."""
    training = settings.get("sample_analysis", {}).get("training_files", ())
    metadata = json.loads(composite_metadata_path(path).read_text(encoding="utf-8"))
    records = metadata.get("pn_generation", {}).get("dependencies", ())
    if not training or not records:
        raise ValueError("Sample-assisted import needs its original training-file fingerprints. "
                         "Open an existing historical snapshot or regenerate with the current samples.")
    original = {resolve_source_path(folder, item["path"]): item["sha256"] for item in records}
    training_paths = tuple(resolve_source_path(folder, name) for name in training)
    for target in training_paths:
        if target not in original or not target.is_file() or _file_digest(target) != original[target]:
            raise ValueError(f"Cannot import this assisted version as fresh: training input {target.name} "
                             "is missing, changed, or has no original fingerprint. Existing historical versions remain available.")
    code_paths = tuple(Path(name).resolve() for name in api.dependency_paths())
    code_records = [item for item in records if resolve_source_path(folder, item["path"]) not in training_paths]
    used = set()
    for target in code_paths:
        matches = [(index, item) for index, item in enumerate(code_records)
                   if str(item["path"]).replace("\\", "/").rsplit("/", 1)[-1].casefold() == target.name.casefold()]
        if (len(matches) != 1 or not target.is_file()
                or _file_digest(target) != matches[0][1]["sha256"]):
            raise ValueError(f"Cannot import this assisted version as fresh: original implementation {target.name} "
                             "is missing, ambiguous, or differs from the current implementation. "
                             "Open its existing historical snapshot or generate a new draft.")
        used.add(matches[0][0])
    if len(used) != len(code_records):
        raise ValueError("Assisted import has generation dependencies that cannot be mapped to this implementation.")
    return (*code_paths, *training_paths)


def _reconstruction_import_dependencies(folder, path, settings, api, *, pair, p_which, n_which):
    """Validate store exports or the public native writer's self-contained proof."""
    from core.drr_pn_reconstruction.pipeline import (
        ALGORITHM_VERSION as original_version, GENERALIZED_ALGORITHM_VERSION,
        LOCAL_ALGORITHM_VERSION, CONDITIONED_ALGORITHM_VERSION,
        CONDITIONED_BOUNDED_ALGORITHM_VERSION,
        PROTOCOL, GENERALIZED_PROTOCOL, LOCAL_PROTOCOL, CONDITIONED_PROTOCOL,
        CONDITIONED_BOUNDED_PROTOCOL,
    )
    document = read_composite_background(path)
    source = next(s for s in document.sources if s.key == document.default_source)
    report = json.loads(source.generation_json) if source.generation_json else {}
    method = report.get("algorithm_version")
    protocols = {original_version: PROTOCOL, GENERALIZED_ALGORITHM_VERSION: GENERALIZED_PROTOCOL,
                 LOCAL_ALGORITHM_VERSION: LOCAL_PROTOCOL,
                 CONDITIONED_ALGORITHM_VERSION: CONDITIONED_PROTOCOL,
                 CONDITIONED_BOUNDED_ALGORITHM_VERSION: CONDITIONED_BOUNDED_PROTOCOL}
    if method not in protocols:
        raise ValueError(f"Unsupported reconstruction algorithm: {method!r}. "
                         "It cannot be imported as a fresh reconstruction with this implementation.")
    if (p_which, n_which) != ("all", "all"):
        raise ValueError("Reconstruction import needs All frames for both sources.")
    expected = dict(protocols[method])
    analysis = settings.get("sample_analysis")
    if analysis is not None:
        if len(analysis.get("training_files", [])) != 2:
            raise ValueError("Reconstruction settings need the original two training-file roles.")
        expected["sample_analysis"] = {
            "training_files": analysis.get("training_files", []),
            "subset": "all gates and full native energy", "y_axis": "auto",
            "split_policy": "explicit-two-training-repeats; all other selected files check only"}
    if settings.get("feature_protection"):
        from core.drr_pn_reconstruction.feature_protection import normalize_feature_protection
        if method != CONDITIONED_BOUNDED_ALGORITHM_VERSION:
            raise ValueError("Feature protection requires the conditioned bounded reconstruction method.")
        expected["feature_protection"] = normalize_feature_protection(settings["feature_protection"])
    if settings != expected or report.get("settings") != settings:
        raise ValueError("Reconstruction settings do not match the declared implementation protocol.")
    records = report.get("input_records", [])
    if not records:
        # Store-native exports can carry their verified dependency records instead.
        metadata = json.loads(composite_metadata_path(path).read_text(encoding="utf8"))
        original = metadata.get("pn_generation", {})
        if (original.get("algorithm_version") != method or original.get("settings") != settings
                or original.get("frame_choice") != {"P": "all", "N": "all"}):
            raise ValueError("Reconstruction export has no consistent original generation record.")
        for role, selected in (("P", pair.p_files), ("N", pair.n_files)):
            sources = original.get("pair", {}).get(role, [])
            if (len(sources) != 1 or len(selected) != 1
                    or resolve_source_path(folder, sources[0]["path"]) != resolve_source_path(folder, selected[0])
                    or _file_digest(resolve_source_path(folder, selected[0])) != sources[0].get("sha256")):
                raise ValueError("Reconstruction export does not match the selected original P/N roles.")
        return _assisted_import_dependencies(folder, path, settings, api)
    if [row.get("role") for row in records] != ["P", "N", "training", "training"]:
        raise ValueError("Reconstruction export needs the original P/N and two training-file roles.")
    paths = tuple(resolve_source_path(folder, row["path"]) for row in records)
    selected = tuple(resolve_source_path(folder, name) for name in (*pair.p_files, *pair.n_files))
    if len(set(paths)) != 4 or paths[:2] != selected:
        raise ValueError("Reconstruction export does not match the selected original P/N roles.")
    if analysis is not None and tuple(resolve_source_path(folder, name)
                                     for name in analysis.get("training_files", [])) != paths[2:]:
        raise ValueError("Reconstruction training settings disagree with its original input fingerprints.")
    for target, record in zip(paths, records):
        if not target.is_file() or _file_digest(target) != record.get("sha256"):
            raise ValueError(f"Original reconstruction input {target.name} is missing or changed.")
    code = report.get("implementation_sha256", {})
    current = tuple(Path(name).resolve() for name in api.dependency_paths())
    if not code or not current or len(code) != len(current):
        raise ValueError("Reconstruction export has incomplete implementation fingerprints.")
    used = set()
    for target in current:
        matches = [(name, digest) for name, digest in code.items()
                   if str(name).replace("\\", "/").rsplit("/", 1)[-1].casefold() == target.name.casefold()]
        if len(matches) != 1 or not target.is_file() or _file_digest(target) != matches[0][1]:
            raise ValueError(f"Original reconstruction implementation {target.name} is missing, ambiguous or changed.")
        used.add(matches[0][0])
    if len(used) != len(code):
        raise ValueError("Reconstruction export contains unmatched implementation fingerprints.")
    return (*current, *paths[2:])


class DrrPnBackgroundDialog(QDialog):
    stateChanged = Signal()

    def __init__(self, parent=None, *, folder, catalog=(), measurements=(),
                 select_files, pool=None, sample_scoped=False):
        super().__init__(parent)
        self.setWindowTitle("DRR · P/N backgrounds")
        self.setObjectName("drr_pn_background_dialog")
        self.resize(1120, 850)
        self.setMinimumSize(850, 650)
        self.folder, self.measurements = str(folder), tuple(measurements)
        self.sample_scoped = bool(sample_scoped)
        self._browse_all_history = False
        self._select_files = select_files
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._store = DrrPnStore(folder)
        self._worker = None
        self._generation_active = False
        self._generation_outcome = None
        self._progress_clock = QElapsedTimer()
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(250)
        self._progress_timer.timeout.connect(self._update_elapsed)
        self._token = 0
        self._closed = False
        self._refreshing = False
        self.pair = None
        self._p_files, self._n_files = (), ()
        self._versions = ()
        self._current = None
        self._payload = None
        self.selected_version = None
        self.allow_stale = False
        self._reconstruction_enabled = RECONSTRUCTION_ENABLED
        self._measurement_catalog = tuple(catalog)
        self._automatic_training = choose_training_repeats(
            self.folder, self.measurements, catalog=self._measurement_catalog)
        selected = set(self.measurements)
        self._catalog_entries = build_pn_source_catalog(self._measurement_catalog,
            tuple(source for source in self._measurement_catalog if source.source in selected))
        self._selection_touched = False
        self._pending_role_refresh = False
        self._selection_controls = []

        outer = QVBoxLayout(self)
        controls = QWidget()
        main_layout = QVBoxLayout(controls)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setWidget(controls)
        self.controls_scroll.setMinimumHeight(160)
        self.controls_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.controls_scroll.setAccessibleName("P/N background files and generation settings")
        outer.addWidget(self.controls_scroll, 2)
        hint = QLabel("Choose P and N separately. Compare file modified times; either repeat number can be combined.")
        hint.setWordWrap(True)
        main_layout.addWidget(hint)
        self._build_role_tables(main_layout)
        self.source_labels, self.clear_role_buttons = {}, {}
        for role in ("P", "N"):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Selected {role}"))
            label = QLabel("None")
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            self.source_labels[role] = label
            row.addWidget(label, 1)
            clear = QPushButton("Clear")
            clear.setAccessibleName(f"Clear selected {role} background")
            clear.setAutoDefault(False)
            clear.setEnabled(False)
            clear.clicked.connect(lambda _checked=False, r=role: self._clear_role(r))
            self.clear_role_buttons[role] = clear
            row.addWidget(clear)
            main_layout.addLayout(row)
        self.condition_label = QLabel("Choose P and N backgrounds.")
        self.condition_label.setWordWrap(True)
        main_layout.addWidget(self.condition_label)
        history_row = QHBoxLayout()
        history_row.addWidget(QLabel("Saved version (local time)"))
        self.history = QComboBox()
        self.history.setAccessibleName("Saved P/N baseline versions")
        self.history.setToolTip("Saved version creation times are shown in this computer's local time.")
        self.history.setMinimumContentsLength(16)
        self.history.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        history_row.addWidget(self.history, 1)
        main_layout.addLayout(history_row)
        self.training_summary = QLabel()
        self.training_summary.setObjectName("pn_training_summary")
        self.training_summary.setWordWrap(True)
        main_layout.addWidget(self.training_summary)
        self.feature_protection_preview = QLabel()
        self.feature_protection_preview.setObjectName("pn_feature_protection_preview")
        self.feature_protection_preview.setAccessibleName("Saved feature protection and background scale")
        self.feature_protection_preview.setWordWrap(True)
        self.feature_protection_preview.hide()
        main_layout.addWidget(self.feature_protection_preview)
        self._build_feature_protection(main_layout)
        generate_row = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setAutoDefault(False)
        set_fluent_property(self.generate_btn, "fluentAppearance", "primary")
        self.generate_btn.clicked.connect(lambda: self._generate(False))
        generate_row.addWidget(self.generate_btn)
        generate_row.addStretch(1)
        main_layout.addLayout(generate_row)
        self.advanced_btn = QToolButton()
        self.advanced_btn.setObjectName("pn_advanced_toggle")
        self.advanced_btn.setText("Advanced")
        self.advanced_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setArrowType(Qt.RightArrow)
        main_layout.addWidget(self.advanced_btn)
        self.advanced_panel = QWidget()
        self.advanced_panel.setObjectName("pn_advanced_panel")
        layout = QVBoxLayout(self.advanced_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.show_other_conditions = QCheckBox("Show other conditions")
        self.show_other_conditions.setToolTip("Include other or unverified measurement conditions; each tab remains newest modified time first.")
        self.show_other_conditions.toggled.connect(self._fill_role_tables)
        layout.addWidget(self.show_other_conditions)
        self.frames = {}
        for role in ("P", "N"):
            row = QHBoxLayout()
            button = QPushButton(f"Choose {role}…")
            button.setAccessibleName(f"Choose {role} background files manually")
            button.clicked.connect(lambda _checked=False, r=role: self._choose_role(r))
            row.addWidget(button)
            row.addWidget(QLabel(f"{role} frames"))
            combo = QComboBox()
            for title, key in (("All frames (mean)", "all"), ("First frame", "first"), ("Last frame", "last")):
                combo.addItem(title, key)
            combo.setAccessibleName(f"{role} background frame reduction")
            combo.currentIndexChanged.connect(self._frames_changed)
            self.frames[role] = combo
            row.addWidget(combo, 1)
            self._selection_controls.extend([button, combo])
            layout.addLayout(row)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Background action"))
        self.generation_mode = QComboBox()
        self.generation_mode.setAccessibleName("P/N background generation mode")
        self.generation_mode.addItem("Use saved background", "saved")
        self.generation_mode.addItem("P/N reconstruction draft", "reconstruction")
        self.generation_mode.addItem("Experimental sample-assisted draft", "assisted")
        self.generation_mode.addItem("Experimental P/N only draft", "pn_only")
        initial_mode = "reconstruction" if self.sample_scoped and self.measurements else "saved"
        self.generation_mode.setCurrentIndex(self.generation_mode.findData(initial_mode))
        reconstruction_item = self.generation_mode.model().item(self.generation_mode.findData("reconstruction"))
        reconstruction_item.setEnabled(self._reconstruction_enabled)
        reconstruction_item.setToolTip("Generate a background draft from matched raw P/N and two selected sample repeats; review the intended features before using it."
                                      if self._reconstruction_enabled else "P/N reconstruction is unavailable in this build.")
        self.generation_mode.model().item(self.generation_mode.findData("assisted")).setEnabled(bool(self.measurements))
        self.generation_mode.setToolTip("Saved backgrounds do not require regeneration. New generation creates a separate draft; existing versions remain unchanged.")
        self.generation_mode.currentIndexChanged.connect(self._mode_changed)
        mode_row.addWidget(self.generation_mode, 1)
        layout.addLayout(mode_row)
        self.sample_role_label = QLabel()
        self.sample_role_label.setWordWrap(True)
        layout.addWidget(self.sample_role_label)
        self.training_controls = QWidget()
        training_layout = QVBoxLayout(self.training_controls)
        training_layout.setContentsMargins(0, 0, 0, 0)
        self.manual_training_check = QCheckBox("Choose sample repeats manually")
        self.manual_training_check.setObjectName("pn_manual_training_check")
        self.manual_training_check.setToolTip(
            "Optional override. By default, two repeats from the current measurement group are selected automatically.")
        training_layout.addWidget(self.manual_training_check)
        self._selection_controls.append(self.manual_training_check)
        self.manual_training_controls = QWidget()
        training_row = QHBoxLayout(self.manual_training_controls)
        training_row.setContentsMargins(0, 0, 0, 0)
        training_row.addWidget(QLabel("Training repeats"))
        self.training_repeats = []
        for index in range(2):
            combo = QComboBox()
            combo.setAccessibleName(f"Raw reconstruction training repeat {index+1}")
            combo.setMinimumContentsLength(14)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            for choice, name in enumerate(self.measurements, 1):
                basename = Path(name).name
                label = "…" + basename[-30:] if len(basename) > 30 else basename
                combo.addItem(f"{choice}. {label}", name)
                combo.setItemData(combo.count()-1, str(resolve_source_path(self.folder, name)), Qt.ToolTipRole)
            if self.measurements:
                choice = (combo.findData(self._automatic_training.training_files[index])
                          if self._automatic_training.usable else
                          (0 if index == 0 else min(2, len(self.measurements)-1)))
                combo.setCurrentIndex(choice)
            combo.setToolTip(combo.currentData(Qt.ToolTipRole) or "")
            combo.currentIndexChanged.connect(self._training_changed)
            training_row.addWidget(combo, 1)
            self.training_repeats.append(combo)
            self._selection_controls.append(combo)
        training_layout.addWidget(self.manual_training_controls)
        self.manual_training_check.toggled.connect(self._sample_selection_changed)
        layout.addWidget(self.training_controls)
        self._update_sample_roles()
        settings_row = QHBoxLayout()
        self.transition_label = QLabel("New draft: transition half-width (meV)")
        self.transition_label.setWordWrap(True)
        settings_row.addWidget(self.transition_label)
        self.transition_spin = QDoubleSpinBox()
        self.transition_spin.setRange(.1, 200)
        self.transition_spin.setDecimals(1)
        self.transition_spin.setValue(default_settings()["transition_halfwidth_ev"] * 1000)
        self.transition_spin.setAccessibleName("New draft transition half-width in meV")
        self.transition_spin.setToolTip("Affects new generation only; saved versions keep their original parameters.")
        settings_row.addWidget(self.transition_spin)
        self.regenerate_btn = QPushButton("Regenerate new version")
        self.regenerate_btn.clicked.connect(lambda: self._generate(True))
        settings_row.addWidget(self.regenerate_btn)
        settings_row.addStretch(1)
        layout.addLayout(settings_row)
        history_actions = QHBoxLayout()
        self.all_history_btn = QPushButton("All saved pairs")
        self.all_history_btn.clicked.connect(self._all_history)
        self.import_btn = QPushButton("Import saved…")
        self.import_btn.setToolTip("Register an existing composite CSV and its metadata for the selected P/N pair.")
        self.import_btn.clicked.connect(self._import)
        history_actions.addWidget(self.all_history_btn)
        history_actions.addWidget(self.import_btn)
        history_actions.addStretch(1)
        layout.addLayout(history_actions)
        self._selection_controls.extend([self.history, self.generation_mode, self.transition_spin,
                                         self.all_history_btn, self.import_btn])
        layout.addWidget(QLabel("Region details"))
        self.region_scope_label = QLabel("Load a saved version to inspect its recorded regional support.")
        self.region_scope_label.setWordWrap(True)
        self.region_scope_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.region_scope_label.setAccessibleName("Background file checks and scientific review status")
        layout.addWidget(self.region_scope_label)
        self.region_model = QStandardItemModel(self)
        self.region_model.setHorizontalHeaderLabels(["Energy (eV)", "Source", "Construction / scope", "Support status", "Reason"])
        self.region_table = QTableView()
        self.region_table.setObjectName("pn_region_support_table")
        self.region_table.setModel(self.region_model)
        self.region_table.setAccessibleName("Saved background regional source and conditional support")
        self.region_table.setAccessibleDescription("Read-only saved provenance; support and user selection do not establish scientific acceptance.")
        self.region_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.region_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.region_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.region_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.region_table.verticalHeader().hide()
        self.region_table.setWordWrap(False)
        self.region_table.setMinimumHeight(100)
        self.region_table.setMaximumHeight(175)
        self.region_table.hide()
        layout.addWidget(self.region_table)
        self.advanced_panel.hide()
        self.advanced_btn.toggled.connect(self._advanced_toggled)
        main_layout.addWidget(self.advanced_panel)
        self.method_label = QLabel("No version loaded. Apply a background to inspect ΔR/R and d²E in the main view.")
        self.method_label.setWordWrap(True)
        main_layout.addWidget(self.method_label)
        self.figure = Figure(figsize=(8, 3), constrained_layout=True)
        self.canvas = ThemeAwareFigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(180)
        outer.addWidget(self.canvas, 1)
        self.historical = QCheckBox("Use this saved snapshot although its original inputs changed or are missing")
        self.historical.hide()
        self.historical.toggled.connect(self._controls)
        outer.addWidget(self.historical)
        self.progress_panel = QWidget()
        self.progress_panel.setObjectName("pn_generation_progress_panel")
        progress_layout = QVBoxLayout(self.progress_panel)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_phase_label = QLabel()
        self.progress_phase_label.setObjectName("pn_generation_phase")
        self.progress_phase_label.setWordWrap(True)
        self.progress_phase_label.setAccessibleName("Current background generation stage")
        progress_layout.addWidget(self.progress_phase_label)
        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("pn_generation_progress")
        self.progress_bar.setAccessibleName("P/N background generation stage progress")
        self.progress_bar.setAccessibleDescription(STAGE_PROGRESS_NOTE)
        self.progress_bar.setToolTip(STAGE_PROGRESS_NOTE)
        self.progress_bar.setRange(0, TOTAL_STAGES)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Stage progress: %v / %m")
        progress_row.addWidget(self.progress_bar, 1)
        self.elapsed_label = QLabel("Elapsed 0:00")
        self.elapsed_label.setAccessibleName("Generation elapsed time")
        progress_row.addWidget(self.elapsed_label)
        self.cancel_generation_btn = QPushButton("Cancel generation")
        self.cancel_generation_btn.setObjectName("pn_cancel_generation")
        self.cancel_generation_btn.setAutoDefault(False)
        self.cancel_generation_btn.clicked.connect(self._cancel_generation)
        progress_row.addWidget(self.cancel_generation_btn)
        progress_layout.addLayout(progress_row)
        progress_note = QLabel("Stage progress; durations vary. Remaining time is not estimated.")
        progress_note.setWordWrap(True)
        progress_layout.addWidget(progress_note)
        self.progress_panel.hide()
        outer.addWidget(self.progress_panel)
        self.status = QLabel("Saved CSV and metadata are kept in Processed Data / DRR / PN Backgrounds.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self.status)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.use_btn = QPushButton("Use as Baseline")
        self.use_btn.setDefault(True)
        self.use_btn.clicked.connect(self._use)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        bottom.addWidget(self.use_btn)
        bottom.addWidget(self.close_btn)
        outer.addLayout(bottom)
        self.history.currentIndexChanged.connect(self._history_selected)
        self.finished.connect(self._stop)
        manager = theme_manager()
        if manager is not None:
            manager.themeChanged.connect(self._draw)
        self._fill_role_tables()
        self._draw()
        self._controls()
        QTimer.singleShot(0, self, self._refresh_versions)

    def _build_role_tables(self, layout):
        self.role_tabs = QTabWidget()
        self.role_tabs.setObjectName("pn_role_tabs")
        self.role_tabs.setAccessibleName("P and N background files by modified time")
        self.role_tables, self.role_models, self._role_entries = {}, {}, {}
        for role in ("P", "N"):
            model = QStandardItemModel(0, 5, self)
            model.setHorizontalHeaderLabels(["Modified time", "File", "Frames", "Time difference", "Match"])
            model.horizontalHeaderItem(4).setToolTip("Compatibility with current measurements; this does not mean a counterpart is available or scientifically accepted.")
            table = QTableView()
            table.setObjectName(f"pn_{role.lower()}_files")
            table.setAccessibleName(f"{role} background files, newest modified time first")
            table.setModel(model)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSelectionMode(QAbstractItemView.SingleSelection)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.verticalHeader().hide()
            table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            model.horizontalHeaderItem(3).setToolTip("Absolute file-modified-time difference from the selected opposite background; no automatic pairing.")
            table.setItemDelegateForColumn(1, WrappedFilenameDelegate(table))
            table.setWordWrap(True)
            table.setTextElideMode(Qt.ElideNone)
            table.selectionModel().currentRowChanged.connect(lambda current, previous, r=role: self._role_row_selected(r, current))
            table.clicked.connect(lambda index, r=role: self._role_row_clicked(r, index))
            self.role_models[role], self.role_tables[role] = model, table
            self.role_tabs.addTab(table, f"{role} back")
        self.role_tabs.setMinimumHeight(150)
        self.role_tabs.setMaximumHeight(205)
        layout.addWidget(self.role_tabs)
        self.table, self.model = self.role_tables["P"], self.role_models["P"]

    def _advanced_toggled(self, expanded):
        self.advanced_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.advanced_panel.setVisible(expanded)

    def _build_feature_protection(self, layout):
        self.feature_protection_controls = QWidget()
        self.feature_protection_controls.setObjectName("pn_feature_protection_controls")
        box = QVBoxLayout(self.feature_protection_controls)
        box.setContentsMargins(0, 0, 0, 0)
        self.feature_protection_check = QCheckBox("Protect real features")
        self.feature_protection_check.setObjectName("pn_feature_protection_check")
        self.feature_protection_check.setToolTip(
            "Preserve the selected raw source's local log shape; review the full background. "
            "Background scale can change across the full spectrum. "
            "Only affects a newly generated reconstruction.")
        box.addWidget(self.feature_protection_check)
        self.feature_protection_panel = QWidget()
        self.feature_protection_layout = QVBoxLayout(self.feature_protection_panel)
        self.feature_protection_layout.setContentsMargins(0, 0, 0, 0)
        note = QLabel("Preserve the selected raw source's local log shape; review the full background.")
        note.setWordWrap(True)
        self.feature_protection_layout.addWidget(note)
        self.feature_protection_rows = []
        self.feature_protection_add = QPushButton("Add range")
        self.feature_protection_add.setAutoDefault(False)
        self.feature_protection_add.clicked.connect(lambda: self._add_feature_protection_row(edited=True))
        self.feature_protection_layout.addWidget(self.feature_protection_add, 0, Qt.AlignLeft)
        self.feature_protection_error = QLabel()
        self.feature_protection_error.setWordWrap(True)
        self.feature_protection_error.setAccessibleName("Feature protection input status")
        self.feature_protection_layout.addWidget(self.feature_protection_error)
        box.addWidget(self.feature_protection_panel)
        layout.addWidget(self.feature_protection_controls)
        self._add_feature_protection_row()
        self.feature_protection_panel.hide()
        self.feature_protection_check.toggled.connect(self._feature_protection_changed)

    def _add_feature_protection_row(self, region=None, *, edited=False):
        region = region or {}
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        lower, upper = QLineEdit(), QLineEdit()
        for field, label, key in ((lower, "From (eV)", "xmin_ev"), (upper, "To (eV)", "xmax_ev")):
            field.setPlaceholderText(label)
            field.setAccessibleName(f"Feature protection {label}")
            field.setMinimumWidth(65)
            if key in region:
                field.setText(str(region[key]))
            row.addWidget(field, 1)
        source = QComboBox()
        source.setAccessibleName("Feature protection raw source")
        source.addItem("Source…", None)
        source.addItem("P", "P")
        source.addItem("N", "N")
        source.setCurrentIndex(max(0, source.findData(region.get("source"))))
        row.addWidget(source)
        remove = QPushButton("Remove")
        remove.setAutoDefault(False)
        row.addWidget(remove)
        record = (row_widget, lower, upper, source, remove)
        self.feature_protection_rows.append(record)
        self.feature_protection_layout.insertWidget(self.feature_protection_layout.indexOf(self.feature_protection_add), row_widget)
        lower.textChanged.connect(self._feature_protection_changed)
        upper.textChanged.connect(self._feature_protection_changed)
        source.currentIndexChanged.connect(self._feature_protection_changed)
        remove.clicked.connect(lambda: self._remove_feature_protection_row(record))
        if edited:
            self._feature_protection_changed()

    def _remove_feature_protection_row(self, record):
        self.feature_protection_rows.remove(record)
        self.feature_protection_layout.removeWidget(record[0])
        record[0].deleteLater()
        if not self.feature_protection_rows:
            self._add_feature_protection_row()
        self._feature_protection_changed()

    def _feature_protection_context(self):
        return (self.feature_protection_check.isChecked(), tuple(
            (lower.text(), upper.text(), source.currentData())
            for _, lower, upper, source, _ in self.feature_protection_rows))

    def _feature_protection(self):
        if not self.feature_protection_check.isChecked():
            return []
        from core.drr_pn_reconstruction.feature_protection import normalize_feature_protection
        regions = []
        for index, (_, lower, upper, source, _) in enumerate(self.feature_protection_rows, 1):
            try:
                xmin, xmax = float(lower.text()), float(upper.text())
            except ValueError as exc:
                raise ValueError(f"Range {index}: enter both energy limits in eV.") from exc
            if source.currentData() not in {"P", "N"}:
                raise ValueError(f"Range {index}: choose the raw P or N source.")
            regions.append({"xmin_ev": xmin, "xmax_ev": xmax, "source": source.currentData()})
        return normalize_feature_protection(regions)

    def _restore_feature_protection(self, settings):
        # Restore every saved range without treating history delivery as an edit.
        refreshing = self._refreshing
        self._refreshing = True
        try:
            for record in self.feature_protection_rows:
                self.feature_protection_layout.removeWidget(record[0])
                record[0].deleteLater()
            self.feature_protection_rows.clear()
            regions = settings.get("feature_protection", [])
            for region in regions or (None,):
                self._add_feature_protection_row(region)
            self.feature_protection_check.setChecked(bool(regions))
            self.feature_protection_panel.setVisible(bool(regions))
        finally:
            self._refreshing = refreshing

    def _feature_protection_changed(self, *_):
        if self._refreshing or self._closed:
            return
        if self._worker is not None:
            self._worker.cancellation.cancel()
        self._token += 1
        self._selection_touched = True
        blocked = self.generation_mode.blockSignals(True)
        self.generation_mode.setCurrentIndex(self.generation_mode.findData("reconstruction"))
        self.generation_mode.blockSignals(blocked)
        self.feature_protection_panel.setVisible(self.feature_protection_check.isChecked())
        self._invalidate()

    def _fill_role_tables(self, *_):
        refreshing = self._refreshing
        self._refreshing = True
        try:
            visible = tuple(entry for entry in self._catalog_entries
                            if self.show_other_conditions.isChecked() or entry.matches_measurements)
            for role in ("P", "N"):
                entries = tuple(entry for entry in visible if entry.role == role)
                previous = self._role_entries.get(role, ())
                self._role_entries[role] = entries
                other_files = self._n_files if role == "P" else self._p_files
                opposite = next((entry for entry in self._catalog_entries
                    if entry.role != role and (entry.source.source,) == other_files), None)
                details = {item.entry.source.source: item for item in
                    describe_pn_counterparts(opposite, entries)} if opposite is not None else {}
                model = self.role_models[role]
                rebuild = (tuple(entry.source.source for entry in previous) !=
                           tuple(entry.source.source for entry in entries) or model.rowCount() != len(entries))
                if rebuild:
                    model.removeRows(0, model.rowCount())
                for row, entry in enumerate(entries):
                    stamp = (datetime.fromtimestamp(entry.modified_time).strftime("%Y-%m-%d %H:%M")
                             if entry.modified_time is not None else "Unknown")
                    state = "Match" if entry.matches_measurements and not entry.role_conflict else "Review"
                    detail = details.get(entry.source.source)
                    difference = detail.time_difference_seconds if detail is not None else None
                    values = [stamp, Path(entry.source.filename).name,
                              str(entry.source.frame_count) if entry.source.frame_count is not None else "?",
                              self._time_difference_text(difference) if opposite is not None else "Choose opposite", state]
                    items = [QStandardItem() for _ in values] if rebuild else [model.item(row, col) for col in range(5)]
                    for item, value in zip(items, values):
                        item.setText(value)
                        item.setEditable(False)
                        item.setToolTip(str(resolve_source_path(self.folder, entry.source.source)) +
                            "\nModified time (not acquisition time): " + stamp + "\n" + entry.reason)
                    items[3].setData(difference, Qt.UserRole)
                    items[3].setToolTip((detail.reason + f"\nAbsolute modified-time difference: {difference!r} seconds")
                                        if detail is not None else "Choose a file in the opposite tab to compare modified times.")
                    if rebuild:
                        model.appendRow(items)
                self.role_tabs.setTabText(0 if role == "P" else 1, f"{role} back ({len(entries)})")
            self._sync_role_tables()
        finally:
            self._refreshing = refreshing

    def _sync_role_tables(self):
        if not hasattr(self, "role_tables"):
            return
        for role, files in (("P", self._p_files), ("N", self._n_files)):
            table = self.role_tables[role]
            selection = table.selectionModel()
            blocked = selection.blockSignals(True)
            table.clearSelection()
            row = next((i for i, entry in enumerate(self._role_entries[role])
                        if files == (entry.source.source,)), -1)
            table.setCurrentIndex(self.role_models[role].index(row, 0) if row >= 0 else QModelIndex())
            selection.blockSignals(blocked)

    def _role_row_clicked(self, role, index):
        if self.generation_mode.currentData() == "saved":
            self._role_row_selected(role, index)

    def _role_row_selected(self, role, index):
        if self._refreshing or self._closed or not index.isValid():
            return
        self._select_role(role, (self._role_entries[role][index.row()].source.source,))

    def _clear_role(self, role):
        self._select_role(role, ())

    def _select_role(self, role, files):
        if self._closed:
            return
        if self._worker is not None:
            self._worker.cancellation.cancel()
        self._token += 1
        self._selection_touched = True
        self._browse_all_history = False
        chosen = {"P": self._p_files, "N": self._n_files}
        chosen[role] = tuple(files)
        note = ""
        self._p_files, self._n_files = chosen["P"], chosen["N"]
        try:
            pair = DrrPnPair(self._p_files, self._n_files) if self._p_files and self._n_files else None
        except ValueError as exc:
            pair, note = None, str(exc)
        self._set_pair(pair)
        self._update_pair_summary(note)
        if (self.sample_scoped and self.measurements and self._reconstruction_enabled
                and self.generation_mode.currentData() == "saved"):
            blocked = self.generation_mode.blockSignals(True)
            self.generation_mode.setCurrentIndex(self.generation_mode.findData("reconstruction"))
            self.generation_mode.blockSignals(blocked)
        self._invalidate()
        if self._worker is not None:
            self._pending_role_refresh = True
        else:
            self._refresh_versions()

    def _update_pair_summary(self, note=""):
        if not self._p_files or not self._n_files:
            missing = "P and N" if not self._p_files and not self._n_files else "P" if not self._p_files else "N"
            self.condition_label.setText(f"Choose {missing} background.")
            self.condition_label.setToolTip(note)
            return
        entries = {role: next((entry for entry in self._catalog_entries
                   if entry.role == role and (entry.source.source,) == files), None)
                   for role, files in (("P", self._p_files), ("N", self._n_files))}
        result = describe_pn_pair(entries["P"], entries["N"]) if all(entries.values()) else None
        status = "Selected pair · needs review"
        if result is not None:
            known_match = (result.grid_matches and not result.condition_differences
                           and all(entry.matches_measurements for entry in entries.values()))
            status = "Known conditions match" if known_match else "Conditions need review"
            if result.frame_counts_match is False:
                status += " · frame counts differ"
            if entries["P"].role_conflict or entries["N"].role_conflict:
                status += " · gate role needs review"
            status = "P/N time difference: " + self._time_difference_text(result.time_difference_seconds) + " · " + status
        self.condition_label.setText(status)
        self.condition_label.setToolTip("\n".join(filter(None, (note, result.reason if result else "",
            describe_manual_pair(self.pair, self.measurements) if self.pair else "Invalid P/N combination"))))

    @staticmethod
    def _time_difference_text(seconds):
        if seconds is None:
            return "Unknown"
        if seconds >= 3600:
            return f"{seconds / 3600:.1f} h"
        if seconds >= 60:
            return f"{seconds / 60:.1f} min"
        return f"{seconds:g} s"

    def _update_training_summary(self):
        if not hasattr(self, "training_summary"):
            return
        if self.generation_mode.currentData() == "saved":
            self.training_summary.setText("Saved version: its recorded inputs are used; no training is rerun.")
            self.training_summary.setToolTip("Inspect the selected version's provenance in Advanced. Current training choices affect new generation only.")
            return
        if self.generation_mode.currentData() == "pn_only":
            self.training_summary.setText("P/N only: sample repeats do not enter generation.")
            self.training_summary.setToolTip("")
            return
        if self.generation_mode.currentData() == "assisted":
            self.training_summary.setText("Sample-assisted: selected measurements are analyzed; training/check roles are in Advanced.")
            self.training_summary.setToolTip("")
            return
        names = self._training_files()
        manual = self.manual_training_check.isChecked()
        labels = [("…" + Path(name).name[-30:] if len(Path(name).name) > 30 else Path(name).name)
                  for name in names]
        title = "Sample repeats: manual" if manual else "Sample repeats: automatic"
        detail = " · ".join(labels) if names else (
            "No sample repeats selected" if manual else self._automatic_training.reason)
        self.training_summary.setText(f"{title} · {detail}".replace("_", "_\u200b"))
        policy = "Manual override." if manual else self._automatic_training.reason
        self.training_summary.setToolTip(policy + "\nThese choices apply to the next Generate. A loaded saved version uses its recorded inputs.\n" +
            "\n".join(str(resolve_source_path(self.folder, name)) for name in names))

    def _context(self):
        return (self.folder, self.measurements, self.pair,
                self.frames["P"].currentData(), self.frames["N"].currentData(),
                self.generation_mode.currentData(), self.transition_spin.value(),
                self._training_files(), self.manual_training_check.isChecked(), self._feature_protection_context())

    def _training_files(self):
        if not self.manual_training_check.isChecked():
            return self._automatic_training.training_files
        return tuple(combo.currentData() for combo in self.training_repeats if combo.currentData() is not None)

    def _reconstruction_problem(self):
        if not self._reconstruction_enabled:
            return "P/N reconstruction is not enabled in this build."
        if self.pair is None:
            return "Choose one original P file and one original N file."
        if len(self.pair.p_files) != 1 or len(self.pair.n_files) != 1:
            return "P/N reconstruction requires one original CSV per P/N source."
        if any(self.frames[role].currentData() != "all" for role in ("P", "N")):
            return "P/N reconstruction requires All frames for both P and N."
        if not self.manual_training_check.isChecked() and not self._automatic_training.usable:
            return self._automatic_training.reason
        selected = self._training_files()
        if len(selected) != 2 or len(set(selected)) != 2:
            return "Select two distinct raw training repeats from the current measurement group."
        if any(Path(name).suffix.lower() != ".csv" for name in selected):
            return "Training repeats must be original intensity CSV files."
        try:
            self._feature_protection()
        except ValueError as exc:
            return str(exc)
        return ""

    def _training_changed(self, *_):
        for combo in self.training_repeats:
            combo.setToolTip(combo.currentData(Qt.ToolTipRole) or "")
        if self.manual_training_check.isChecked():
            self._sample_selection_changed()

    def _sample_selection_changed(self, *_):
        self._update_training_summary()
        if self._worker is not None:
            self._worker.cancellation.cancel()
            self.status.setText("Sample repeat selection changed; the pending result will not be selected.")
        self._update_sample_roles()
        if hasattr(self, "generate_btn"):
            self._controls()

    def _update_sample_roles(self, request=None):
        if self.generation_mode.currentData() == "saved":
            text = "Use a saved version to inspect both ΔR/R and d²E in the main view."
        elif self.generation_mode.currentData() == "reconstruction":
            training = request.training_files if request is not None else self._training_files()
            evaluation = request.evaluation_files if request is not None else tuple(
                name for name in self.measurements if name not in training)
            text = (f"Training: {len(training)} raw repeats; {len(evaluation)} other selected file(s) are excluded from generation. "
                    "Review the main features, nearby baseline and repeatability in ΔR/R and d²E. All gates and energies are retained.")
            problem = self._reconstruction_problem()
            if problem:
                text += "\n" + problem
        elif self.generation_mode.currentData() != "assisted":
            text = "P/N only: no samples enter generation. This is a new draft method, not the saved V66 recipe."
        elif request is None:
            text = (f"Selected samples: {len(self.measurements)} file(s). Training and check roles are verified "
                    "on Generate; browsing saved versions reads measurement headers only.")
        else:
            text = (f"Training: {len(request.training_files)} file(s); checks: {len(request.evaluation_files)} file(s). "
                    + ("No withheld files: diagnostics use training data. " if not request.evaluation_files else "")
                    + "Source choice is conditional; diagnostics do not establish acceptance.")
        self.sample_role_label.setText(text)

    def _mode_changed(self, *_):
        # A generation preference never replaces or regenerates the current version.
        if self._worker is not None:
            self._worker.cancellation.cancel()
            self.status.setText("Generation mode changed; the pending result will not be selected.")
        self._update_sample_roles()
        self._controls()

    def _start(self, callback, consume, message, *, report_progress=False, cancellation=None,
               generation=False, progress_total=TOTAL_STAGES):
        if self._closed or self._worker is not None:
            return
        self._token += 1
        token = (self._token, self._context())
        self._active_context = token
        self._consume = consume
        worker = Worker(_run, callback, token, cancellation=cancellation or LoadCancellation(), report_progress=report_progress)
        self._worker = worker
        worker.signals.result.connect(self._result, Qt.QueuedConnection)
        worker.signals.error.connect(self._error, Qt.QueuedConnection)
        worker.signals.finished.connect(self._finished, Qt.QueuedConnection)
        worker.signals.log.connect(self._progress_text, Qt.QueuedConnection)
        worker.signals.progress.connect(self._progress_value, Qt.QueuedConnection)
        self._generation_active = generation
        self._generation_outcome = None
        self._progress_timer.stop()
        self.progress_panel.setVisible(generation)
        if generation:
            self.progress_phase_label.setText(message)
            self.progress_bar.setRange(0, progress_total)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Stage progress: %v / %m")
            self.cancel_generation_btn.setEnabled(True)
            self._progress_clock.start()
            self._update_elapsed()
            self._progress_timer.start()
        self.status.setText(message)
        self._controls()
        self._pool.start(worker)

    def _current_work(self):
        return (not self._closed and self._worker is not None
                and not self._worker.cancellation.cancelled
                and self._active_context == (self._token, self._context()))

    @Slot()
    def _update_elapsed(self):
        if self._progress_clock.isValid():
            seconds = max(0, self._progress_clock.elapsed() // 1000)
            self.elapsed_label.setText(f"Elapsed {seconds // 60}:{seconds % 60:02d}")

    @Slot(int)
    def _progress_value(self, value):
        if self._generation_active and self._generation_outcome is None and self._current_work():
            # Only consuming a valid saved preview may complete the final stage.
            value = max(self.progress_bar.value(), min(self.progress_bar.maximum() - 1, max(0, value)))
            self.progress_bar.setValue(value)

    @Slot()
    def _cancel_generation(self):
        if self._generation_active and self._worker is not None:
            self._worker.cancellation.cancel()
            self.cancel_generation_btn.setEnabled(False)
            self.progress_phase_label.setText("Stopping generation; finishing the current calculation…")
            self.status.setText("Stopping generation; waiting for the current calculation to finish…")

    def _finish_generation(self, outcome):
        self._generation_outcome = outcome
        self._generation_active = False
        self._progress_timer.stop()
        self._update_elapsed()
        self.cancel_generation_btn.setEnabled(False)
        if outcome == "success":
            self.progress_bar.setValue(self.progress_bar.maximum())
            if self._payload and self._payload.get("compatibility_error"):
                self.progress_bar.setFormat("Completed · 100%")
                self.progress_phase_label.setText("Completed — cannot apply this background")
                self.status.setText("Generation complete; compatibility check failed. " + self.status.text())
            else:
                self.progress_bar.setFormat("Ready · 100%")
                self.progress_phase_label.setText("Background ready for review")
                self.status.setText("Generation complete — draft ready for review. " + self.status.text())
        elif outcome == "error":
            self.progress_bar.setFormat("Failed · %v / %m stages")
            self.progress_phase_label.setText("Generation failed — " + self.status.text())
        else:
            self.progress_bar.setFormat("Cancelled · %v / %m stages")
            self.progress_phase_label.setText("Generation cancelled")
            self.status.setText("Generation cancelled. Any draft already saved remains in history; it has not been applied.")

    @Slot(str)
    def _progress_text(self, message):
        if self._current_work() and self._generation_outcome is None:
            self.status.setText(message)
            if self._generation_active:
                self.progress_phase_label.setText(message)

    @Slot(object)
    def _result(self, result):
        token, value = result
        if (self._closed or token != (self._token, self._context())
                or self._worker is None or self._worker.cancellation.cancelled):
            return
        try:
            self._consume(value)
        except Exception as exc:
            # A preview may fail after partially installing history/state.
            # Keep selection usable and prevent applying that partial result.
            self._refreshing = False
            self._current = self._payload = None
            self.feature_protection_preview.clear()
            self.feature_protection_preview.hide()
            self.use_btn.setEnabled(False)
            self._error(str(exc))
            return
        if self._generation_active:
            self._finish_generation("success")

    @Slot(str)
    def _error(self, text):
        if (not self._closed and self._worker and not self._worker.cancellation.cancelled
                and self._active_context == (self._token, self._context())):
            self.status.setText(text.split("\n\n", 1)[0])
            if self._generation_active:
                self._finish_generation("error")

    @Slot()
    def _finished(self):
        if self._generation_active:
            self._finish_generation("cancelled")
        self._worker = None
        if not self._closed:
            if self._pending_role_refresh:
                self._pending_role_refresh = False
                self._refresh_versions()
            else:
                self._controls()

    def _controls(self, *_):
        self._update_training_summary()
        busy = self._worker is not None or self._closed
        for widget in self._selection_controls:
            widget.setEnabled(not busy)
        experimental = self.generation_mode.currentData() in {"assisted", "pn_only"}
        reconstruction = self.generation_mode.currentData() == "reconstruction"
        manual_training = self.manual_training_check.isChecked()
        self.training_controls.setVisible(reconstruction)
        self.manual_training_controls.setVisible(reconstruction and manual_training)
        for combo in self.training_repeats:
            combo.setEnabled(not busy and reconstruction and manual_training)
        self.feature_protection_controls.setVisible(reconstruction)
        self.feature_protection_check.setEnabled(not busy)
        self.feature_protection_panel.setEnabled(not busy)
        try:
            self._feature_protection()
            feature_problem = ""
        except ValueError as exc:
            feature_problem = str(exc)
        self.feature_protection_error.setText(feature_problem)
        self.feature_protection_error.setVisible(bool(feature_problem))
        self.transition_label.setVisible(not reconstruction)
        self.transition_spin.setVisible(not reconstruction)
        reconstruction_problem = self._reconstruction_problem() if reconstruction else ""
        self.import_btn.setEnabled(not busy and self.pair is not None)
        self.transition_spin.setEnabled(not busy and experimental)
        can_generate = (experimental and self.pair is not None) or (reconstruction and not reconstruction_problem)
        self.generate_btn.setEnabled(not busy and can_generate)
        self.generate_btn.setToolTip(reconstruction_problem if reconstruction else "")
        self.regenerate_btn.setEnabled(not busy and can_generate and self._current is not None)
        self.historical.setEnabled(not busy)
        valid = self._current is not None and self._current.status != "invalid"
        fresh = valid and (self._current.fresh or self.historical.isChecked())
        compatible = self._payload and not self._payload["compatibility_error"]
        self.use_btn.setEnabled(bool(not busy and fresh and compatible))
        self.stateChanged.emit()

    def _set_pair(self, pair):
        self.pair = pair
        if pair:
            self._p_files, self._n_files = pair.p_files, pair.n_files
        self._automatic_training = choose_training_repeats(
            self.folder, self.measurements, catalog=self._measurement_catalog,
            excluded_files=(*self._p_files, *self._n_files))
        for role, files in (("P", self._p_files), ("N", self._n_files)):
            names = ", ".join(Path(p).name for p in files) if files else "None"
            # QLabel's word wrapping does not break underscore-separated names.
            # Match the main file summary's invisible wrap opportunities.
            self.source_labels[role].setText(names.replace("_", "_\u200b").replace("-", "-\u200b"))
            self.source_labels[role].setAccessibleName(names)
            self.source_labels[role].setToolTip("\n".join(str(resolve_source_path(self.folder, name)) for name in files))
            if hasattr(self, "clear_role_buttons"):
                self.clear_role_buttons[role].setEnabled(bool(files))
        self._fill_role_tables()
        self._update_pair_summary()

    def _invalidate(self):
        self._current = self._payload = None
        self.feature_protection_preview.clear()
        self.feature_protection_preview.hide()
        self._update_sample_roles()
        self.method_label.setText("No version loaded. Apply a background to inspect ΔR/R and d²E in the main view.")
        self.region_model.removeRows(0, self.region_model.rowCount())
        self.region_table.hide()
        self.region_scope_label.setText("Load a saved version to inspect its recorded regional support.")
        self.region_scope_label.setToolTip("")
        self.historical.setChecked(False)
        self.historical.hide()
        self._draw()
        self._controls()

    def _choose_role(self, role):
        current = self._p_files if role == "P" else self._n_files
        chosen = tuple(self._select_files(current, f"Choose {role} Background Files") or ())
        if chosen == current:
            return
        self._select_role(role, chosen)

    def _frames_changed(self, *_):
        if not self._refreshing and self.pair and self._worker is None:
            self._invalidate()
            self._refresh_versions()

    def _preview(self, version):
        payload = load_pn_version(self.folder, version, self.measurements)
        payload["sample_context_match"] = bool(
            self.measurements and _assisted_api().matching_sample_context(version, self.measurements))
        source = next(s for s in payload["document"].sources if s.key == payload["document"].default_source)
        report = json.loads(source.generation_json) if source.generation_json else {}
        if str(version.algorithm_version).startswith("pn_raw_") or report.get("reconstruction_method"):
            saved_training = version.settings.get("sample_analysis", {}).get("training_files", ()) or tuple(
                row["path"] for row in report.get("input_records", []) if row.get("role") == "training")
            selected = {resolve_source_path(self.folder, name) for name in self.measurements}
            payload["sample_context_match"] = bool(saved_training) and all(
                resolve_source_path(self.folder, name) in selected for name in saved_training)
        return payload

    def _refresh_versions(self):
        pair = self.pair
        p, n = self.frames["P"].currentData(), self.frames["N"].currentData()
        browse_all = self._browse_all_history
        def work():
            if self.sample_scoped and self.measurements and not browse_all:
                problem = ""
                try:
                    preferred = None if self._selection_touched else self._store.find_applied(self.measurements)
                except (ValueError, OSError, KeyError) as exc:
                    preferred, problem = None, str(exc)
                if preferred is not None and pair is not None and (
                        preferred.pair != pair or (preferred.p_which, preferred.n_which) != (p, n)):
                    preferred = None
                selected_pair = preferred.pair if preferred is not None else pair
                versions = self._store.list_versions(selected_pair)
                return versions, self._preview(preferred) if preferred else None, problem
            versions = self._store.list_versions(pair)
            preferred = self._store.preferred_version(pair, p_which=p, n_which=n) if pair else None
            return versions, self._preview(preferred) if preferred else None
        self._start(work, self._loaded_history, "Checking saved P/N versions…")

    def _all_history(self):
        self._browse_all_history = True
        self._p_files = self._n_files = ()
        self._set_pair(None)
        self._invalidate()
        self._refresh_versions()

    def _loaded_history(self, result):
        versions, preview = result[:2]
        association_problem = result[2] if len(result) > 2 else ""
        self._versions = tuple(versions)
        self._refreshing = True
        self.history.clear()
        empty = "No saved version yet"
        self.history.addItem("Select a saved version…" if versions else empty, None)
        for version in versions:
            draft_label = " · Draft" if version.is_draft else ""
            created_at = format_local_timestamp(version.created_at)
            self.history.addItem(f"v{version.number}{draft_label} · {version.algorithm_version} · {version.status} · {created_at}",
                                 version.version_id)
        if preview:
            if self.sample_scoped and self.pair is None:
                self._set_pair(preview["version"].pair)
                for role, which in (("P", preview["version"].p_which), ("N", preview["version"].n_which)):
                    self.frames[role].setCurrentIndex(self.frames[role].findData(which))
            if self.sample_scoped and len(result) > 2:
                blocked = self.generation_mode.blockSignals(True)
                self.generation_mode.setCurrentIndex(self.generation_mode.findData("saved"))
                self.generation_mode.blockSignals(blocked)
            self.history.setCurrentIndex(self.history.findData(preview["version"].version_id))
            self._show_preview(preview)
        else:
            self.status.setText(("No unchanged background is associated with this measurement group. "
                                 "Choose P and N and generate for the current data, or explicitly select a historical version."
                                 + ("\nPrevious association needs review: " + association_problem if association_problem else ""))
                                if self.sample_scoped and self.measurements and not self._browse_all_history else
                                "Choose a historical version." if versions else
                                "No saved background for this pair. Import saved, or explicitly choose an experimental generation mode.")
        self._refreshing = False

    def _history_selected(self, index):
        if self._refreshing or self._worker is not None:
            return
        version = next((v for v in self._versions if v.version_id == self.history.currentData()), None)
        self._invalidate()
        if version is None:
            return
        def work():
            pair = version.pair
            return pair, self._preview(version)
        def consume(result):
            pair, preview = result
            self._refreshing = True
            self._set_pair(pair)
            for role, which in (("P", version.p_which), ("N", version.n_which)):
                self.frames[role].setCurrentIndex(self.frames[role].findData(which))
            self._refreshing = False
            self._show_preview(preview)
        self._start(work, consume, "Loading the saved background and checking measurement grids…")

    def _generate(self, force):
        if not self.pair or self._worker is not None or self._closed:
            return
        if self.generation_mode.currentData() == "saved":
            return
        if self.generation_mode.currentData() == "reconstruction":
            self._generate_reconstruction(force)
            return
        pair = self.pair
        p, n = self.frames["P"].currentData(), self.frames["N"].currentData()
        assisted = self.generation_mode.currentData() == "assisted"
        measurements = tuple(self.measurements)
        if assisted and not measurements:
            self.status.setText("Select a sample group or explicitly choose P/N only generation.")
            return
        settings = default_settings()
        settings["transition_halfwidth_ev"] = self.transition_spin.value() / 1000
        def work(progress, log):
            log.emit("Checking the selected P/N files…")
            request = None
            if assisted:
                api = _assisted_api()
                request = api.prepare_assisted_request(
                    self.folder, pair, p, n, settings=settings, measurements=measurements)
                key = request.key
            else:
                key = self._store.prepare(pair, p_which=p, n_which=n, settings=settings,
                                          algorithm_version=ALGORITHM_VERSION, dependencies=dependency_paths())
            version = None if force else self._store.find_reusable(key)
            assessment_note = "Saved version reused; check diagnostics were not rerun."
            assessment_details = ""
            if version is None:
                progress.emit(1)
                log.emit("Generating a P/N background draft…")
                result = (api.generate_assisted_background(self.folder, pair, p, n, request=request)
                          if assisted else generate_pn_background(self.folder, pair, p, n, settings=settings))
                check_load_cancelled()
                progress.emit(2)
                log.emit("Saving the P/N background draft…")
                version = self._store.save(result.document, key, force_new=force)
                if assisted:
                    progress.emit(3)
                    log.emit("Checking sample repeats for the saved draft…")
                    try:
                        assessment = api.assess_assisted_result(self.folder, result, request)
                    except Exception as exc:
                        check_load_cancelled()
                        assessment = {
                            "status": "error", "error": str(exc),
                            "generation_fingerprint": request.key.fingerprint,
                            "acceptance": {"status": "not assessed", "scientifically_accepted": False},
                        }
                    check_load_cancelled()
                    failed = str(assessment.get("status", "")).lower() in {"error", "failed"} or assessment.get("error")
                    assessment_note = ("Check diagnostics failed: " + str(assessment.get("error", assessment.get("status")))
                                       if failed else "Check diagnostics saved separately; not scientific acceptance.")
                    try:
                        assessment_path = api.save_assessment(self.folder, version, assessment)
                        if assessment_path:
                            assessment_details = f"Assessment: {assessment_path}"
                    except Exception as exc:
                        check_load_cancelled()
                        assessment_note = f"Draft saved, but check diagnostics could not be saved: {exc}"
                else:
                    assessment_note = "P/N-only draft; no sample-assisted check diagnostics were run."
            progress.emit(4)
            log.emit("Loading background preview…")
            payload = self._preview(version)
            payload["assessment_note"] = assessment_note
            payload["assessment_details"] = assessment_details
            payload["assisted_request"] = request
            versions = self._store.list_versions(pair)
            progress.emit(5)
            return versions, payload
        self._start(work, self._loaded_history, "Generating P/N draft and saving a new version…",
                    report_progress=True, generation=True, progress_total=6)

    def _generate_reconstruction(self, force):
        problem = self._reconstruction_problem()
        if problem:
            self.status.setText(problem)
            return
        pair = self.pair
        measurements = tuple(self.measurements)
        training = self._training_files()
        feature_protection = self._feature_protection()
        cancellation = LoadCancellation()

        def work(_progress, log):
            steps = ReconstructionProgress(_progress, log)
            api = _reconstruction_api()
            steps.start("verify_inputs")
            request = api.prepare_reconstruction_request(
                self.folder, pair, measurements, training_files=training, method="conditioned_bounded",
                **({"feature_protection": feature_protection} if feature_protection else {}))
            check_load_cancelled()
            version = None if force else self._store.find_reusable(request.key)
            reused = version is not None
            if version is None:
                steps.start("load_raw")
                def stage_progress(stage, event):
                    check_load_cancelled()
                    # Signal delivery is queued; no QWidget is touched by this worker.
                    steps.event(stage, event)
                result = api.generate_reconstruction_background(
                    self.folder, pair, request=request,
                    cancelled=lambda: cancellation.cancelled, progress=stage_progress,
                    force_source_refit=bool(force))
                check_load_cancelled()
                steps.start("save")
                version = self._store.save(result.document, request.key, force_new=force)
            check_load_cancelled()
            steps.start("preview")
            if reused:
                log.emit("Loading a matching saved background; no new fitting needed…")
            payload = self._preview(version)
            payload["sample_context_match"] = True
            payload["assisted_request"] = request
            payload["assessment_note"] = (
                "Saved reconstruction reused; no fitting or check diagnostics rerun. " if reused else
                "Raw P/N reconstruction saved as a draft. ") + (
                "Other selected repeats did not enter generation. Use as Baseline to inspect the main features and nearby baseline in ΔR/R and d²E; saving does not apply it.")
            versions = self._store.list_versions(pair)
            steps.complete("preview")
            return versions, payload

        self._start(work, self._loaded_history, "Preparing P/N reconstruction draft…",
                    report_progress=True, cancellation=cancellation, generation=True)

    def _import(self):
        if self.pair is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import Saved Composite Background", self.folder, "CSV (*.csv)")
        if not path:
            return
        pair = self.pair
        p, n = self.frames["P"].currentData(), self.frames["N"].currentData()
        def work():
            document = read_composite_background(path)
            source = next(s for s in document.sources if s.key == document.default_source)
            report = json.loads(source.generation_json) if source.generation_json else {}
            method = report.get("algorithm_version") or report.get("algorithm") or "imported_composite"
            settings = report.get("settings", {})
            dependencies = ()
            if method == _assisted_api().ASSISTED_ALGORITHM_VERSION:
                dependencies = _assisted_import_dependencies(self.folder, path, settings, _assisted_api())
            elif str(method).startswith("pn_raw_") or report.get("source_models_refitted_from_raw"):
                # Method identity is authoritative even when an optional report
                # flag is absent/false. Unknown versions in this namespace must
                # fail strict validation, never become a generic fresh import.
                api = _reconstruction_api()
                dependencies = _reconstruction_import_dependencies(
                    self.folder, path, settings, api, pair=pair, p_which=p, n_which=n)
            version = self._store.import_saved(
                path, pair, algorithm_version=method, settings=settings, dependencies=dependencies,
                p_which=p, n_which=n, provenance={"origin": "explicit_user_import"})
            return self._store.list_versions(pair), self._preview(version)
        self._start(work, self._loaded_history, "Importing the saved background with its original metadata…")

    def _show_region_report(self, source, report, payload, version):
        regional = describe_pn_regions(source.energy, report)
        self.region_model.removeRows(0, self.region_model.rowCount())
        for row in regional.rows:
            limits = (f"{row.xmin_ev:.5f}" if row.xmin_ev == row.xmax_ev else
                      f"{row.xmin_ev:.5f}–{row.xmax_ev:.5f}")
            values = (limits, row.source, row.operation, row.support, row.reason)
            tooltip = "\n".join((f"Energy: {limits} eV", f"Source: {row.source}",
                f"Scope: {row.operation}", f"Support: {row.support}", row.reason,
                f"Saved record: {row.record_path}"))
            items = [QStandardItem(value) for value in values]
            for item in items:
                item.setEditable(False)
                item.setToolTip(tooltip)
            self.region_model.appendRow(items)
        self.region_table.show()
        file_check = ("File loaded; numerical compatibility check failed" if payload.get("compatibility_error") else
                      "File readable; native-grid check passed" if payload.get("measurement_grid_checked") else
                      "File readable; no measurement grid checked")
        freshness = "inputs unchanged" if version.fresh else f"input status: {version.status}"
        self.region_scope_label.setText(
            f"{file_check}; {freshness}. {regional.scientific_review}.\n"
            "Regional records may overlap. Use as Baseline records your selection, not scientific acceptance.")
        self.region_scope_label.setToolTip("\n".join(regional.notes))

    def _show_preview(self, payload):
        self._payload = payload
        self._current = payload["version"]
        v = self._current
        self._restore_feature_protection(v.settings)
        document = payload["document"]
        source = next(s for s in document.sources if s.key == document.default_source)
        report = json.loads(source.generation_json) if source.generation_json else {}
        protected_ranges = v.settings.get("feature_protection", [])
        protection_text = ""
        if protected_ranges:
            ranges_text = "; ".join(
                f"{row['source']}: {row['xmin_ev']:.6g}–{row['xmax_ev']:.6g} eV"
                for row in protected_ranges)
            protection_text = (f"Protected local log shape — {ranges_text}.\n"
                "Background scale can change across the full spectrum; compare ΔR/R and d²E.")
            protection_report = report.get("feature_protection", {})
            full_protection = protection_report.get("full", {}) if isinstance(protection_report, dict) else {}
            gain = full_protection.get("whole_axis_gain_minmax") if isinstance(full_protection, dict) else None
            if (isinstance(gain, (list, tuple)) and len(gain) == 2
                    and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and np.isfinite(value) and value > 0 for value in gain)
                    and gain[0] <= gain[1]):
                protection_text += f"\nScale relative to the pre-protection background: {gain[0]:.3g}–{gain[1]:.3g}×."
        self.feature_protection_preview.setText(protection_text)
        self.feature_protection_preview.setVisible(bool(protection_text))
        self._show_region_report(source, report, payload, v)
        review = (" · New method / draft; inspect ΔR/R and d²E in the main view"
                  if v.is_draft or report.get("scientifically_validated") is False or v.settings.get("sample_analysis")
                  else " · Review suitability in the main view")
        unresolved = report.get("unresolved_regions", [])
        method_name = ("P/N reconstruction draft" if str(v.algorithm_version).startswith("pn_raw_") else
                       "Sample-assisted draft" if v.settings.get("sample_analysis")
                       else v.algorithm_version)
        method_text = f"{method_name} · {SAVED_COMPOSITE_LABEL} · {source_frame_summary(v)}{review}"
        summary = []
        if report.get("samples_analyzed"):
            summary.append("Samples analyzed; conditional sample-driven source choice applied"
                           if report.get("assistance_applied") else
                           "Samples analyzed; no supported sample-driven change")
        elif "samples_analyzed" in report:
            summary.append("Samples were not analyzed for this background")
        details = [f"Algorithm: {v.algorithm_version}", SAVED_COMPOSITE_TOOLTIP, source_frame_summary(v)]
        execution = report.get("execution", {})
        preparation = execution.get("source_preparation", {}) if isinstance(execution, dict) else {}
        if isinstance(preparation, dict) and isinstance(preparation.get("reused"), bool):
            summary.append("Matching P/N source fits reused; sample reconstruction recomputed"
                           if preparation["reused"] else "P/N source fits computed for this generation")
        render = report.get("render_diagnostics", {})
        full_render = render.get("full", {}) if isinstance(render, dict) else {}
        full_render = full_render if isinstance(full_render, dict) else {}
        parent_render = report.get("conditioned_parent_diagnostics")
        parent_scope = ("conditioned_parent_diagnostics" in report or
                        full_render.get("parent_diagnostics_path") == "conditioned_parent_diagnostics.full")
        if parent_scope:
            correction_selected = full_render.get("correction_selected")
            if correction_selected is True:
                summary.append("Supported fine correction included before protection" if protected_ranges else
                               "Supported fine correction included")
            elif correction_selected is False:
                summary.append("Before protection: fine correction withheld; base background retained" if protected_ranges else
                               "Fine correction withheld; base background retained")
            else:
                summary.append("Fine correction selection before protection not recorded" if protected_ranges else
                               "Fine correction selection not recorded")
            full_render = parent_render.get("full", {}) if isinstance(parent_render, dict) else {}
            full_render = full_render if isinstance(full_render, dict) else {}
            if not full_render:
                summary.append("Base construction not recorded")
            details.append("Regional repair counts describe the base construction; fine correction is reported separately.")
        if full_render:
            def saved_rows(key):
                value = full_render.get(key)
                return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []
            applied = (saved_rows("applied_repairs") if isinstance(full_render.get("applied_repairs"), list) else
                       saved_rows("regional_applied_repairs") + saved_rows("local_applied_repairs"))
            count = full_render.get("applied_count", len(applied))
            prefix = "Base: " if parent_scope else ""
            summary.append(f"{prefix}{count} regional repair(s); source evidence remains conditional")
            plans = (saved_rows("repair_plan") if isinstance(full_render.get("repair_plan"), list) else
                     saved_rows("regional_plan") + saved_rows("local_plan"))
            deferred = [row for row in plans if row.get("apply") is False and row.get("route") != "compact_only"]
            reasons = sorted({str(reason).replace("_", " ") for row in deferred
                              for reason in (row.get("reasons") if isinstance(row.get("reasons"), list) else [])})
            if full_render.get("partial") or deferred:
                summary.append(f"Partial reconstruction: {len(deferred)} regional repair(s) deferred")
            if full_render.get("no_supported_repairs"):
                summary.append("No regional repair supported; source uncertainty remains")
            opposite = {int(index) for row in applied
                        for index in (row.get("opposite_exclusive_indices")
                                      if isinstance(row.get("opposite_exclusive_indices"), list) else [])
                        if isinstance(index, (int, np.integer)) and not isinstance(index, bool)}
            opposite_count = full_render.get("opposite_exclusive_count", len(opposite))
            if opposite_count:
                summary.append(f"Unresolved opposite-source evidence at {opposite_count} energy point(s)")
            if reasons:
                summary.append("Deferred: " + "; ".join(reasons[:2]))
                details.append("All deferred repair reasons:\n" + "\n".join(reasons))
        reason = report.get("fallback_reason")
        if reason:
            explanation = {"no_consistent_matching_witness_protection_only":
                           "No consistent sample-matching evidence; only supported protection rules could be used."}.get(
                               str(reason), str(reason).replace("_", " "))
            details.append("Fallback: " + explanation)
        if report.get("degradation_reasons"):
            details.append("Limitations: " + "; ".join(str(item).replace("_", " ")
                                                       for item in report["degradation_reasons"]))
        energy = np.asarray(document.sources[0].energy)
        unresolved_mask = np.zeros(energy.shape, dtype=bool)
        for lower, upper in unresolved:
            unresolved_mask |= (energy >= lower) & (energy <= upper)
        if unresolved_mask.size and unresolved_mask.all():
            summary.append("full-range unresolved — review required")
        elif unresolved:
            summary.append(f"{len(unresolved)} unresolved region(s)")
        if summary:
            method_text += "\n" + "; ".join(summary) + "."
        self.method_label.setText(method_text)
        details.append("Settings:\n" + json.dumps(v.settings, ensure_ascii=False, indent=2))
        self.method_label.setToolTip("\n".join(details))
        request = payload.get("assisted_request")
        if request is not None:
            self._update_sample_roles(request)
        else:
            training = v.settings.get("sample_analysis", {}).get("training_files", ()) or tuple(
                row["path"] for row in report.get("input_records", []) if row.get("role") == "training")
            if training:
                self.sample_role_label.setText(
                    f"Saved training: {len(training)} file(s). Check diagnostics have not been rerun "
                    "for this selection; inspect both ΔR/R and d²E in the main view.")
        self.historical.setChecked(False)
        self.historical.setVisible(not v.fresh)
        text = f"Loaded v{v.number}: {v.csv_path.name}"
        status_details = [str(v.csv_path)]
        if payload.get("compatibility_error"):
            text += "\nCannot apply to this measurement: " + payload["compatibility_error"]
        elif v.status_reasons:
            text += "\n" + "; ".join(v.status_reasons)
        if self.measurements and not payload.get("sample_context_match"):
            text += ("\nThis saved background is not linked to the current training-file context. "
                     "Explicit reuse remains available; review its suitability in the main view.")
        if payload.get("assessment_note"):
            text += "\n" + payload["assessment_note"]
        elif v.settings.get("sample_analysis"):
            text += "\nSaved draft; current check diagnostics have not been rerun."
        if payload.get("assessment_details"):
            status_details.append(payload["assessment_details"])
        self.status.setText(text)
        self.status.setToolTip("\n".join(status_details))
        self._draw()

    @Slot()
    def _draw(self, *_):
        if self._closed:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Intensity")
        if not self._payload:
            ax.text(.5, .5, "Choose a P/N pair and load or generate a background",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            document = self._payload["document"]
            energy, background, _ = compose_background(document)
            for source in self._payload["reference_sources"]:
                ax.plot(source.energy, source.intensity, lw=1, label=source.label)
            ax.plot(energy, background, color=alias("text_primary"), lw=1.8, label="Saved background")
            generated = next(s for s in document.sources if s.key == document.default_source)
            report = json.loads(generated.generation_json) if generated.generation_json else {}
            for index, (lower, upper) in enumerate(report.get("unresolved_regions", [])):
                ax.axvspan(lower, upper, alpha=.12, color="orange",
                           label="Unresolved" if index == 0 else "_nolegend_")
            ax.legend(fontsize="small")
        self.canvas.draw_idle()

    def _use(self):
        if not self._current or not self.use_btn.isEnabled():
            return
        version_id = self._current.version_id
        allow_stale = self.historical.isChecked()
        def work():
            current = self._store.get_version(version_id)
            if current is None or current.status == "invalid" or (not current.fresh and not allow_stale):
                raise ValueError("The saved version or its inputs changed. Reload history before applying.")
            # Check headers again; main-view processing starts after application.
            payload = self._preview(current)
            if payload["compatibility_error"]:
                raise ValueError(payload["compatibility_error"])
            return current
        def consume(version):
            self.selected_version, self.allow_stale = version, allow_stale
            self.accept()
        self._start(work, consume, "Verifying the selected version and measurement before applying…")

    def _stop(self, *_):
        self._closed = True
        self._progress_timer.stop()
        self._token += 1
        if self._worker:
            self._worker.cancellation.cancel()

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)
