"""Source-first DRR background settings, kept private until explicit Apply.

The owner supplies validated child workflows. Their result updates this dialog's
draft only; saving in a child workflow never applies a main-view background.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

from PySide6.QtCore import QThreadPool, QTimer, Qt, Slot
from PySide6.QtGui import QStandardItem, QStandardItemModel, QTextOption
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QDialog, QDialogButtonBox,
    QFormLayout, QFrame, QHBoxLayout, QHeaderView, QLabel, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QStackedWidget, QTableView,
    QToolButton, QVBoxLayout, QWidget, QSizePolicy, QMenu, QPlainTextEdit, QListView,
)

from core.load_cancellation import LoadCancellation
from ui_qt.common import QComboBox, Worker, WrappedFilenameDelegate
from ui_qt.fluent_ui.style import set_fluent_property
from ui_qt.drr_pn_labels import SAVED_COMPOSITE_LABEL, SAVED_COMPOSITE_TOOLTIP, source_frame_summary


@dataclass(frozen=True)
class BackgroundDraft:
    kind: str = "self"
    files: tuple[str, ...] = ()
    which: str = "all"
    regions: tuple = ()
    version: Any = None
    allow_stale: bool = False
    bn_plan: Any = None
    fitted_result: Any = None
    fixed: bool = False
    auto_bn_compare: bool = False
    calibration: Any = None
    bn_warnings_acknowledged: bool = False
    bn_candidates: tuple[str, ...] | None = None
    pn_plan: Any = None
    pn_candidates: tuple[str, ...] | None = None
    pn_warnings_acknowledged: bool = False
    saved_assignments: tuple = ()


def _label(text, *, heading=False):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if heading:
        font = label.font()
        font.setBold(True)
        label.setFont(font)
    return label


def describe_background(draft: BackgroundDraft) -> str:
    """Compact summary shared by the dialog and its owning sidebar."""
    if draft.saved_assignments:
        label = "P+N" if draft.kind == "pn" else draft.kind.upper()
        return f"{label} · {len(draft.saved_assignments)} saved measurement assignments"
    if draft.kind in {"p", "n", "pn"} and draft.pn_plan is not None:
        rows = tuple(draft.pn_plan.rows)
        matches = sum(bool(row.baseline_file) for row in rows)
        label = "P+N" if draft.kind == "pn" else draft.kind.upper()
        return f"{label} · {matches}/{len(rows)} measurements matched"
    if draft.kind == "bn" and draft.bn_plan is not None:
        rows = tuple(getattr(draft.bn_plan, "rows", ()))
        matches = sum(bool(getattr(row, "baseline_file", None)) for row in rows)
        return f"BN · {matches}/{len(rows)} measurements matched"
    names = ", ".join(Path(name).name for name in draft.files)
    if draft.kind == "bn":
        return f"BN · common file: {names}" if names else "BN · matching not configured"
    if draft.kind in {"p", "n"}:
        return f"{draft.kind.upper()} only · {names or 'choose source files'}"
    if draft.kind == "pn":
        version = getattr(draft.version, "version_id", "")
        if version or names:
            return f"P+N · {SAVED_COMPOSITE_LABEL} · {version or names}"
        return "P+N composite · configure a background"
    if draft.kind == "self":
        return f"Self · {'first' if draft.which == 'first' else 'last'} frame of each measurement"
    if draft.kind == "fitted":
        return "Fitted · configured fit" if draft.fitted_result is not None else "Fitted · current fit settings"
    return f"Custom background · {names or 'configure a source'}"


def _build_bn_plan(folder, measurements, candidates, stopped, *, confirmed_cache=None, progress, log):
    # Keep source discovery and CSV validation out of the GUI thread.
    from core.drr_bn_matching import build_bn_match_plan
    if confirmed_cache is not None:
        confirmed = confirmed_cache.find(folder, measurements, candidate_files=candidates)
        if confirmed is not None:
            return confirmed, True
    return (build_bn_match_plan(folder, measurements, candidate_files=candidates,
                               cancelled=stopped.is_set), False)


class _SourcePages(QStackedWidget):
    """Use the visible source's preferred size instead of the largest page."""

    def sizeHint(self):
        page = self.currentWidget()
        return page.sizeHint() if page is not None else super().sizeHint()

    def minimumSizeHint(self):
        page = self.currentWidget()
        return page.minimumSizeHint() if page is not None else super().minimumSizeHint()


class DrrBackgroundSettings(QDialog):
    """A transactional front door to the existing DRR background workflows.

    ``select_files(dialog, kind, files, title)`` returns files or None on cancel.
    Other child callbacks receive ``(dialog, draft)`` and return a new
    BackgroundDraft or None. Matching can be injected for tests; the normal
    implementation owns a cancellable worker and ignores results after close.
    """

    def __init__(self, parent=None, *, current: BackgroundDraft, initial_draft=None,
                 measurement_count=None, measurements=(), folder="", pool=None,
                 select_files=None, configure_composite=None, configure_fit=None,
                 import_custom=None, edit_composite=None, configure_calibration=None,
                 request_bn_match=None, confirmed_bn_cache=None, catalog=None,
                 refresh_catalog=None, browse_files=None):
        super().__init__(parent)
        self.setObjectName("drr_background_settings")
        self.setWindowTitle("DRR · Background settings")
        self.setMinimumSize(720, 540)
        available = self.screen().availableGeometry()
        self.resize(min(860, available.width() - 40), min(700, available.height() - 60))
        self.current = current
        self._draft = current if initial_draft is None else initial_draft
        if not isinstance(self._draft, BackgroundDraft):
            raise TypeError("initial_draft must be a BackgroundDraft.")
        if (self._draft.bn_plan is not None and self._draft.bn_candidates is None
                and getattr(self._draft.bn_plan, "automatic_candidates", True) is False):
            self._draft = replace(self._draft, bn_candidates=tuple(self._draft.bn_plan.candidate_files))
        self.selection = None
        self.measurements = tuple(measurements)
        self.measurement_count = len(self.measurements) if measurement_count is None else measurement_count
        self.folder = str(folder)
        self.pool = pool or QThreadPool.globalInstance()
        self._select_files = select_files
        self._catalog = None if catalog is None else tuple(catalog)
        self._refresh_catalog = refresh_catalog
        self._browse_files = browse_files
        self._pool_expanded = False
        self._callbacks = dict(composite=configure_composite, fit=configure_fit,
                               custom=import_custom, edit=edit_composite,
                               calibration=configure_calibration)
        self._request_bn_match = request_bn_match
        self.confirmed_bn_cache = confirmed_bn_cache
        self._closed = False
        self._updating = False
        self._busy = False
        self._worker = None
        self._bn_stop = None
        self._bn_error = ""
        self._drafts = {}
        self._source_widgets = {}
        self._mapped_plan = object()
        self._bn_method_value = "common" if self._draft.kind == "bn" and self._draft.files and self._draft.bn_plan is None else "matched"
        self._bn_method_drafts = {}
        self._pn_files_key = object()
        self._pn_assignment_value = self._pn_assignment_mode(self._draft)
        self._draft = self._restrict_pn_pool(self._draft, self._pn_assignment_value)
        self._pn_assignment_drafts = {}
        self._pn_assignment_modes = {}

        layout = QVBoxLayout(self)
        self.applied_summary = _label("Applied: " + describe_background(current))
        self.applied_summary.setObjectName("drr_background_applied_summary")
        self.applied_summary.setToolTip("\n".join(self.measurements))
        layout.addWidget(self.applied_summary)

        scroll = QScrollArea()
        self._scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        content = QVBoxLayout(body)
        self._body_layout = content
        content.setContentsMargins(0, 0, 0, 0)
        source_row = QHBoxLayout()
        source_row.addWidget(_label("Source"))
        self.source_group = QButtonGroup(self)
        for key, title in (("bn", "&BN"), ("pn", "&P / N"), ("self", "&Self"), ("fitted", "&Fitted")):
            button = QRadioButton(title)
            button.setObjectName("drr_background_source_" + key)
            button.setAccessibleName("Background source " + title.replace("&", ""))
            self.source_group.addButton(button)
            self._source_widgets[key] = button
            source_row.addWidget(button)
            button.toggled.connect(lambda checked, key=key: self._change_source(key) if checked else None)
        source_row.addStretch(1)
        content.addLayout(source_row)
        self.pages = _SourcePages()
        self.pages.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.pages.addWidget(self._bn_page())
        self.pages.addWidget(self._pn_page())
        self.pages.addWidget(self._self_page())
        self.pages.addWidget(self._fitted_page())
        custom_page = QWidget()
        custom_layout = QVBoxLayout(custom_page)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        self.custom_summary = _label("")
        custom_layout.addWidget(self.custom_summary)
        self.pages.addWidget(custom_page)
        content.addWidget(self.pages)
        self._source_page_index = content.indexOf(self.pages)

        self.advanced_button = QToolButton()
        self.advanced_button.setObjectName("drr_background_advanced")
        self.advanced_button.setText("Ad&vanced")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setArrowType(Qt.RightArrow)
        self.advanced_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_button.setAccessibleName("Advanced background settings")
        self.advanced_panel = self._advanced_page()
        self.advanced_panel.hide()
        content.addWidget(self.advanced_panel)
        self.advanced_button.toggled.connect(self._toggle_advanced)
        self._body_spacer_index = content.count()
        content.addStretch(1)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        self.draft_summary = _label("")
        self.draft_summary.setObjectName("drr_background_draft_summary")
        self.draft_summary.setParent(self)
        self.draft_summary.hide()  # Compatibility summary; the footer shows validation once.
        self.status = _label("")
        self.status.setObjectName("drr_background_settings_status")
        self.status.setAccessibleName("Background validation status")
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setAccessibleName("Matching BN backgrounds")
        self.progress.hide()
        layout.addWidget(self.progress)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        self.apply_btn = self.buttons.button(QDialogButtonBox.Apply)
        self.apply_btn.setObjectName("drr_background_settings_apply")
        self.apply_btn.setText("&Apply")
        self.apply_btn.setAutoDefault(False)
        set_fluent_property(self.apply_btn, "fluentAppearance", "primary")
        self.cancel_btn = self.buttons.button(QDialogButtonBox.Cancel)
        self.cancel_btn.setAutoDefault(False)
        self.apply_btn.clicked.connect(self._apply)
        self.buttons.rejected.connect(self.reject)
        footer = QHBoxLayout()
        footer.addWidget(self.advanced_button)
        footer.addStretch(1)
        footer.addWidget(self.buttons)
        layout.addLayout(footer)
        self.finished.connect(self._stop)
        self._sync()
        self._source_widgets[self._source_key(self._draft.kind)].setFocus()

    @property
    def draft(self):
        return self._draft

    @staticmethod
    def _source_key(kind):
        return "pn" if kind in {"p", "n", "pn"} else kind if kind in {"bn", "self", "fitted"} else "bn"

    @staticmethod
    def _pn_assignment_mode(draft):
        if draft.saved_assignments:
            return "saved"
        if draft.pn_plan is not None:
            return "matched"
        return "common" if draft.files or draft.version is not None else "matched"

    @staticmethod
    def _pn_plan_pool(plan):
        if plan is None:
            return ()
        if not getattr(plan, "automatic_candidates", False):
            return tuple(plan.candidate_files)
        assigned = tuple(dict.fromkeys(row.baseline_file for row in getattr(plan, "rows", ())
                                       if row.baseline_file))
        if assigned:
            return assigned
        return tuple(dict.fromkeys(name
            for assignment in getattr(getattr(plan, "resolution", None), "assignments", ())
            for name in assignment.baseline_files))

    @staticmethod
    def _restrict_pn_pool(draft, mode):
        if draft.kind in {"p", "n", "pn"} and mode == "matched" and draft.pn_candidates is None:
            # Reuse the previously assigned backgrounds, never the unselected
            # files discovered by a legacy experiment-wide automatic plan.
            return replace(draft, pn_candidates=DrrBackgroundSettings._pn_plan_pool(draft.pn_plan))
        return draft

    def _button(self, text, name, action):
        button = QPushButton(text)
        button.setObjectName(name)
        button.setAutoDefault(False)
        button.clicked.connect(action)
        return button

    @staticmethod
    def _frames(name):
        combo = QComboBox()
        combo.setObjectName(name)
        combo.setAccessibleName("Frames from each selected background file")
        combo.addItem("All frames from each file, then average", "all")
        combo.addItem("First frame from each file, then average", "first")
        combo.addItem("Last frame from each file, then average", "last")
        return combo

    def _bn_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 8)
        layout.setAlignment(Qt.AlignTop)
        form = QFormLayout()
        self.bn_method = QComboBox()
        self.bn_method.setObjectName("drr_bn_method")
        self.bn_method.addItem("Match BN by batch", "matched")
        self.bn_method.addItem("One BN for all", "common")
        self.bn_method.currentIndexChanged.connect(self._bn_method_changed)
        form.addRow("&Method", self.bn_method)
        layout.addLayout(form)

        # Retain the old controls as hidden adapters for integrations. The
        # visible scope is one compact row and a native menu.
        self.bn_auto = QRadioButton("Discover candidates in the current experiment")
        self.bn_restricted = QRadioButton("Restrict matching to selected candidate files")
        for button in (self.bn_auto, self.bn_restricted):
            button.setParent(self)
            button.hide()
        self.bn_candidate_group = QButtonGroup(self)
        self.bn_candidate_group.addButton(self.bn_auto)
        self.bn_candidate_group.addButton(self.bn_restricted)
        self.bn_auto.toggled.connect(self._candidate_mode_changed)
        self.bn_restricted.toggled.connect(self._candidate_mode_changed)
        self.bn_scope_row = QWidget()
        row = QHBoxLayout(self.bn_scope_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.bn_candidates_label = _label("")
        row.addWidget(self.bn_candidates_label, 1)
        self.bn_scope_btn = QToolButton()
        self.bn_scope_btn.setText("Change…")
        self.bn_scope_btn.setObjectName("drr_bn_scope")
        self.bn_scope_btn.setAccessibleName("Change BN matching candidate scope")
        self.bn_scope_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.bn_scope_btn)
        self.bn_current_scope_action = menu.addAction("Current experiment")
        self.bn_current_scope_action.setCheckable(True)
        self.bn_current_scope_action.triggered.connect(self._use_experiment_candidates)
        self.bn_selected_scope_action = menu.addAction("Choose candidate files…")
        self.bn_selected_scope_action.setCheckable(True)
        self.bn_selected_scope_action.triggered.connect(self._choose_candidates)
        menu.aboutToShow.connect(self._scope_menu_state)
        self.bn_scope_btn.setMenu(menu)
        row.addWidget(self.bn_scope_btn)
        self.bn_candidates_btn = self._button("Choose candidate files…", "drr_bn_candidates", self._choose_candidates)
        self.bn_candidates_btn.setParent(self)
        self.bn_candidates_btn.hide()
        self.bn_match_btn = self._button("Ma&tch BN", "drr_bn_match", self._match_bn)
        row.addWidget(self.bn_match_btn)
        layout.addWidget(self.bn_scope_row)
        self.bn_pool_panel = QWidget()
        pool_layout = QVBoxLayout(self.bn_pool_panel)
        pool_layout.setContentsMargins(0, 0, 0, 0)
        pool_note = _label("Candidate pool only limits matching. It does not select a common background.")
        pool_layout.addWidget(pool_note)
        self.bn_pool_selector = self._inline_selector("BN", single=False, kind="bn_candidates")
        if self.bn_pool_selector is not None:
            pool_layout.addWidget(self.bn_pool_selector)
        self.bn_pool_done = self._button("Hide candidate pool", "drr_bn_hide_pool", self._hide_candidate_pool)
        pool_layout.addWidget(self.bn_pool_done, 0, Qt.AlignLeft)
        if self._catalog is None:
            layout.addWidget(self.bn_pool_panel)
        self.bn_common_btn = self._button("Choose BN…", "drr_bn_common_file", self._choose_common_bn)
        layout.addWidget(self.bn_common_btn, 0, Qt.AlignLeft)
        self.bn_common_selector = self._inline_selector("BN", single=True, kind="bn")
        if self.bn_common_selector is not None:
            layout.addWidget(self.bn_common_selector, 1)
        self.bn_common_summary = _label("")
        if self._catalog is not None:
            layout.addWidget(self.bn_common_summary)
        else:
            self.bn_common_summary.setParent(self)
        self.bn_common_summary.hide()
        self.bn_frames = self._frames("drr_bn_frames")
        self.bn_frames.currentIndexChanged.connect(self._frames_changed)
        layout.addWidget(self.bn_frames)
        self.bn_summary_model = QStandardItemModel(0, 2, self)
        self.bn_summary_model.setHorizontalHeaderLabels(("Coverage", "BN file"))
        self.bn_summary_table = self._filename_table(self.bn_summary_model,
            "drr_bn_batch_summary", "Proposed BN selection and measurement coverage")
        self.bn_summary_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.bn_summary_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.bn_summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.bn_summary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._summary_resize = QTimer(self)
        self._summary_resize.setSingleShot(True)
        self._summary_resize.timeout.connect(self._resize_bn_summary)
        self.bn_summary_table.horizontalHeader().sectionResized.connect(lambda *_: self._summary_resize.start(0))
        layout.addWidget(self.bn_summary_table)
        self.bn_model = QStandardItemModel(0, 2, self)
        self.bn_model.setHorizontalHeaderLabels(("Measurement", "BN file"))
        self.bn_table = self._filename_table(self.bn_model,
            "drr_bn_mapping", "Proposed measurement to BN mapping")
        self.bn_table.setMinimumHeight(180)
        self.bn_table.setMaximumHeight(300)
        header = self.bn_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self._row_resize = QTimer(self)
        self._row_resize.setSingleShot(True)
        self._row_resize.timeout.connect(self.bn_table.resizeRowsToContents)
        header.sectionResized.connect(lambda *_: self._row_resize.start(0))
        self.bn_table.selectionModel().currentRowChanged.connect(self._bn_selected_details)
        self.bn_plan_status = _label("")
        self.bn_details_button = QToolButton()
        self.bn_details_button.setText("Match &details")
        self.bn_details_button.setObjectName("drr_bn_match_details")
        self.bn_details_button.setCheckable(True)
        self.bn_details_button.setArrowType(Qt.RightArrow)
        self.bn_details_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        summary_row = QHBoxLayout()
        summary_row.addWidget(self.bn_plan_status, 1)
        self.bn_manual_btn = self._button("Choose BN manually…", "drr_bn_choose_manually", self._choose_common_bn)
        self.bn_manual_btn.setAccessibleName("Choose BN manually")
        self.bn_manual_btn.setToolTip("Choose one BN for all selected measurements. Review the draft, then Apply.")
        summary_row.addWidget(self.bn_manual_btn)
        summary_row.addWidget(self.bn_details_button)
        QWidget.setTabOrder(self.bn_manual_btn, self.bn_details_button)
        layout.addLayout(summary_row)
        self.bn_review = QCheckBox("I reviewed and accept the flagged matches")
        self.bn_review.setObjectName("drr_bn_warning_review")
        self.bn_review.setAccessibleDescription("I reviewed and accept the timing and condition warnings shown for the suggested BN files.")
        self.bn_review.toggled.connect(self._review_changed)
        layout.addWidget(self.bn_review)
        self.bn_details_panel = QWidget()
        details_layout = QVBoxLayout(self.bn_details_panel)
        details_layout.setContentsMargins(8, 0, 0, 0)
        self.bn_timing_note = _label("")
        details_layout.addWidget(self.bn_timing_note)
        details_layout.addWidget(self.bn_table)
        self.bn_details_text = QPlainTextEdit()
        self.bn_details_text.setReadOnly(True)
        self.bn_details_text.setWordWrapMode(QTextOption.WrapAnywhere)
        self.bn_details_text.setAccessibleName("BN matching evidence and warnings")
        self.bn_details_text.setMinimumHeight(140)
        details_layout.addWidget(self.bn_details_text)
        self.bn_details_panel.hide()
        layout.addWidget(self.bn_details_panel)
        self.bn_details_button.toggled.connect(self._toggle_bn_details)
        return page

    @staticmethod
    def _filename_table(model, name, accessible_name):
        table = QTableView()
        table.setObjectName(name)
        table.setAccessibleName(accessible_name)
        table.setModel(model)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setWordWrap(True)
        table.setTextElideMode(Qt.ElideNone)
        table.setItemDelegate(WrappedFilenameDelegate(table))
        table.setMinimumHeight(100)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().hide()
        return table

    def _resize_bn_summary(self):
        labels = [line for row in range(self.bn_summary_model.rowCount())
                  for line in self.bn_summary_model.item(row, 0).text().splitlines()]
        width = max((self.bn_summary_table.fontMetrics().horizontalAdvance(line) for line in labels), default=100) + 20
        self.bn_summary_table.horizontalHeader().resizeSection(0,
            min(max(120, width), max(120, int(self.bn_summary_table.viewport().width() * .45))))
        self.bn_summary_table.resizeRowsToContents()
        # Keep a two-BN result compact; larger unverified/conflict lists scroll
        # inside the view without pushing Apply off the dialog.
        content_height = (self.bn_summary_table.horizontalHeader().height() +
            sum(self.bn_summary_table.rowHeight(i) for i in range(self.bn_summary_model.rowCount())) + 4)
        self.bn_summary_table.setMaximumHeight(max(100, min(300, content_height)))
        self.pages.updateGeometry()

    def _pn_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 8)
        form = QFormLayout()
        self.pn_method = QComboBox()
        self.pn_method.setObjectName("drr_pn_method")
        self.pn_method.addItem("P only", "p")
        self.pn_method.addItem("N only", "n")
        self.pn_method.addItem("P+N composite", "pn")
        form.addRow("&Method", self.pn_method)
        layout.addLayout(form)
        self.pn_method.currentIndexChanged.connect(self._pn_method_changed)
        self.pn_assignment_method = QComboBox()
        self.pn_assignment_method.setObjectName("drr_pn_assignment_method")
        self.pn_assignment_method.setAccessibleName("P/N background assignment method")
        self.pn_assignment_method.addItem("One background for all", "common")
        self.pn_assignment_method.addItem("Match selected background files", "matched")
        form.addRow("&Assignment", self.pn_assignment_method)
        self.pn_assignment_method.currentIndexChanged.connect(self._pn_assignment_changed)
        from ui_qt.drr_pn_matching_panel import PnMatchingPanel
        self.pn_matching = PnMatchingPanel(self, folder=self.folder,
            measurements=self.measurements, pool=self.pool, select_files=self._select_files)
        self.pn_matching.planChanged.connect(self.set_pn_plan)
        self.pn_matching.candidatesChanged.connect(self._pn_candidates_changed)
        self.pn_matching.reviewChanged.connect(self._pn_review_changed)
        self.pn_matching.busyChanged.connect(self.set_bn_busy)
        layout.addWidget(self.pn_matching)
        self.pn_files_btn = self._button("Choose source files…", "drr_pn_source_files", self._choose_pn_files)
        layout.addWidget(self.pn_files_btn, 0, Qt.AlignLeft)
        self.pn_selector = self._inline_selector("P", single=False, kind="pn")
        if self.pn_selector is not None:
            layout.addWidget(self.pn_selector, 1)
        self.pn_files_summary = _label("")
        layout.addWidget(self.pn_files_summary)
        self.pn_files_model = QStandardItemModel(self)
        self.pn_files_view = QListView()
        self.pn_files_view.setObjectName("drr_pn_selected_sources")
        self.pn_files_view.setAccessibleName("Selected background source files")
        self.pn_files_view.setModel(self.pn_files_model)
        self.pn_files_view.setItemDelegate(WrappedFilenameDelegate(self.pn_files_view))
        self.pn_files_view.setWordWrap(True)
        self.pn_files_view.setTextElideMode(Qt.ElideNone)
        self.pn_files_view.setResizeMode(QListView.Adjust)
        self.pn_files_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pn_files_view.setMinimumHeight(80)
        layout.addWidget(self.pn_files_view)
        self.pn_frames = self._frames("drr_pn_frames")
        self.pn_frames.currentIndexChanged.connect(self._frames_changed)
        layout.addWidget(self.pn_frames)
        self.repeat_note = _label("Selected repeats are averaged equally.")
        layout.addWidget(self.repeat_note)
        self.pn_source_frames_note = _label("")
        self.pn_source_frames_note.setObjectName("drr_pn_source_frames_note")
        self.pn_source_frames_note.setToolTip(SAVED_COMPOSITE_TOOLTIP)
        layout.addWidget(self.pn_source_frames_note)
        self.pn_composite_btn = self._button("Generate P+N / saved history…", "drr_pn_configure", lambda: self._child("composite"))
        layout.addWidget(self.pn_composite_btn, 0, Qt.AlignLeft)
        return page

    def _self_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 8)
        self.self_frames = QComboBox()
        self.self_frames.setObjectName("drr_self_frames")
        self.self_frames.setAccessibleName("Measurement frame used for the self background")
        self.self_frames.addItem("Last frame of each measurement", "last")
        self.self_frames.addItem("First frame of each measurement", "first")
        self.self_frames.currentIndexChanged.connect(self._frames_changed)
        layout.addWidget(self.self_frames)
        return page

    def _fitted_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 8)
        self.fit_summary = _label("")
        layout.addWidget(self.fit_summary)
        self.fit_btn = self._button("Configure fit / preview…", "drr_fit_configure", lambda: self._child("fit"))
        layout.addWidget(self.fit_btn, 0, Qt.AlignLeft)
        return page

    def _advanced_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 0, 0, 4)
        self.bn_scope_advanced = QWidget()
        scope_layout = QVBoxLayout(self.bn_scope_advanced)
        scope_layout.setContentsMargins(0, 0, 0, 0)
        if self._catalog is not None:
            scope_row = QHBoxLayout()
            scope_row.addWidget(_label("BN files to consider for matching"), 1)
            self.bn_scope_btn.setText("Choose scope…")
            scope_row.addWidget(self.bn_scope_btn)
            scope_layout.addLayout(scope_row)
            scope_layout.addWidget(self.bn_pool_panel)
            layout.addWidget(self.bn_scope_advanced)
        self.group_strategy = QComboBox()
        self.group_strategy.setObjectName("drr_background_group_strategy")
        self.group_strategy.setAccessibleName("Background strategy for the next measurement group")
        self.group_strategy.addItem("Follow the current background type", False)
        self.group_strategy.addItem("Keep the current background fixed", True)
        self.group_strategy.currentIndexChanged.connect(self._strategy_changed)
        form = QFormLayout()
        form.addRow("Next group", self.group_strategy)
        layout.addLayout(form)
        self.strategy_note = _label("")
        layout.addWidget(self.strategy_note)
        row = QHBoxLayout()
        self.custom_btn = self._button("Import custom background…", "drr_background_import_custom", lambda: self._child("custom"))
        self.edit_btn = self._button("Edit sources / intervals…", "drr_background_edit", lambda: self._child("edit"))
        self.calibration_btn = self._button("BN calibration…", "drr_background_calibration", lambda: self._child("calibration"))
        row.addWidget(self.custom_btn)
        row.addWidget(self.edit_btn)
        row.addWidget(self.calibration_btn)
        row.addStretch(1)
        layout.addLayout(row)
        self.calibration_summary = _label("")
        layout.addWidget(self.calibration_summary)
        self.auto_bn_compare = QCheckBox("Compare and apply validated BN automatically after loading")
        self.auto_bn_compare.setObjectName("drr_auto_bn_compare")
        self.auto_bn_compare.setToolTip("Runs the validated BN comparison workflow after loading. Ambiguous or failed comparisons keep the current background.")
        self.auto_bn_compare.toggled.connect(self._auto_changed)
        layout.addWidget(self.auto_bn_compare)
        return page

    def _toggle_advanced(self, checked):
        self.advanced_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.advanced_panel.setVisible(checked)
        self.pages.updateGeometry()

    def _toggle_bn_details(self, checked):
        self.bn_details_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.bn_details_panel.setVisible(checked)
        self._row_resize.start(0)
        self.pages.updateGeometry()

    def _scope_menu_state(self):
        automatic = self._draft.bn_candidates is None
        self.bn_current_scope_action.setChecked(automatic)
        self.bn_selected_scope_action.setChecked(not automatic)
        self.bn_selected_scope_action.setEnabled(not self._busy and (self._catalog is not None or self._select_files is not None))

    def _inline_selector(self, role, *, single, kind):
        if self._catalog is None:
            return None
        from ui_qt.drr_background_inline_selection import DrrBackgroundFileSelector
        selector = DrrBackgroundFileSelector(folder=self.folder, catalog=self._catalog,
            measurements=self.measurements, role=role, single=single, active=False)
        selector.filesChanged.connect(lambda files: self._inline_files_changed(kind, files))
        selector.refreshRequested.connect(self._refresh_inline_catalog)
        selector.browseRequested.connect(lambda: self._browse_inline(selector))
        selector.refresh_btn.setEnabled(self._refresh_catalog is not None)
        selector.browse_btn.setEnabled(self._browse_files is not None)
        return selector

    def _inline_files_changed(self, kind, files):
        if self._updating or self._busy:
            return
        if kind == "bn_candidates":
            self._draft = replace(self._draft, kind="bn", bn_candidates=tuple(files),
                files=(), bn_plan=None, calibration=None, regions=(), version=None,
                fixed=False, bn_warnings_acknowledged=False)
        else:
            self._draft = replace(self._draft, files=tuple(files), bn_plan=None,
                calibration=None, regions=(), version=None, bn_warnings_acknowledged=False,
                pn_plan=None, pn_warnings_acknowledged=False, saved_assignments=())
        self._bn_error = ""
        self._sync()

    def set_catalog(self, catalog):
        """Publish cached worker metadata without changing any draft choices."""
        if self._closed:
            return
        self._catalog = tuple(catalog)
        for selector in (self.bn_common_selector, self.bn_pool_selector, self.pn_selector):
            if selector is not None:
                selector.set_catalog(self._catalog)
        self._sync()

    def set_catalog_busy(self, busy, error=""):
        if self._closed:
            return
        for selector in (self.bn_common_selector, self.bn_pool_selector, self.pn_selector):
            if selector is not None:
                selector.set_refreshing(busy, error)

    def _refresh_inline_catalog(self):
        if self._refresh_catalog is not None:
            self.set_catalog_busy(True)
            try:
                self._refresh_catalog(self)
            except Exception as exc:
                self.set_catalog_busy(False, str(exc))

    def _browse_inline(self, selector):
        if self._browse_files is None or self._busy:
            return
        try:
            files = self._browse_files(self, selector.role, selector.single)
            if files:
                selector.add_browsed_files(files)
        except Exception as exc:
            selector.set_refreshing(False, str(exc))

    def _hide_candidate_pool(self):
        self._pool_expanded = False
        self._sync()

    def _use_experiment_candidates(self, *_):
        if self._busy:
            return
        self._pool_expanded = False
        if self._draft.bn_candidates is not None:
            self._draft = replace(self._draft, bn_candidates=None, bn_plan=None,
                                  files=(), regions=(), version=None, calibration=None,
                                  fixed=False, bn_warnings_acknowledged=False)
            self._bn_error = ""
        self._sync()

    def _bn_method_changed(self, *_):
        if self._updating or self._busy or self._draft.kind != "bn":
            return
        method = self.bn_method.currentData()
        if method == self._bn_method_value:
            return
        self._bn_method_drafts[self._bn_method_value] = self._draft
        self._bn_method_value = method
        candidate = self._bn_method_drafts.get(method)
        self._draft = (replace(candidate, auto_bn_compare=self._draft.auto_bn_compare)
                       if candidate is not None else replace(self._draft, files=(), bn_plan=None,
                           calibration=None, regions=(), version=None, bn_warnings_acknowledged=False))
        self._draft = replace(self._draft, calibration=None, regions=(), version=None,
                              fixed=False if method == "matched" else self._draft.fixed)
        self._bn_error = ""
        self._sync()

    def _change_source(self, key):
        if self._updating or self._busy:
            return
        if self._draft.kind in {"p", "n", "pn"}:
            self._pn_assignment_modes[self._draft.kind] = self._pn_assignment_value
        self._drafts[self._draft.kind] = self._draft
        target = key
        if key == "pn":
            target = self.pn_method.currentData() or "p"
        next_draft = self._drafts.get(target, BackgroundDraft(
            kind=target, which="last" if target == "self" else "all",
            fixed=self._draft.fixed, auto_bn_compare=self._draft.auto_bn_compare,
            fitted_result=self._draft.fitted_result))
        self._draft = replace(next_draft, auto_bn_compare=self._draft.auto_bn_compare)
        if target == "bn" and (self._draft.bn_plan is not None or self._draft.files):
            self._bn_method_value = "matched" if self._draft.bn_plan is not None else "common"
        if target in {"p", "n", "pn"}:
            self._pn_assignment_value = ("saved" if self._draft.saved_assignments else
                                         "matched" if self._draft.pn_plan is not None else
                                         self._pn_assignment_modes.get(target, self._pn_assignment_mode(self._draft)))
            self._draft = self._restrict_pn_pool(self._draft, self._pn_assignment_value)
        self.pn_matching.error = ""
        self._bn_error = ""
        self._sync()

    def _pn_method_changed(self, *_):
        if self._updating or self._busy or self._draft.kind not in {"p", "n", "pn"}:
            return
        self._change_source("pn")

    def _pn_assignment_changed(self, *_):
        if self._updating or self._busy:
            return
        value = self.pn_assignment_method.currentData()
        self._pn_assignment_drafts[(self._draft.kind, self._pn_assignment_value)] = self._draft
        self._pn_assignment_value = value
        self._draft = self._pn_assignment_drafts.get((self._draft.kind, value), replace(
            self._draft, files=(), regions=(), version=None, calibration=None,
            bn_plan=None, pn_plan=None, pn_warnings_acknowledged=False, fixed=False, saved_assignments=()))
        self._draft = self._restrict_pn_pool(self._draft, value)
        self.pn_matching.error = ""
        self._sync()

    def _pn_candidates_changed(self, candidates):
        if self._busy or self._closed or self._draft.kind not in {"p", "n", "pn"}:
            return
        self._pn_assignment_value = "matched"
        self._draft = replace(self._draft, pn_candidates=tuple(candidates or ()), pn_plan=None,
            pn_warnings_acknowledged=False, files=(), version=None, regions=(), fixed=False, saved_assignments=())
        self._sync()

    def _pn_review_changed(self, checked):
        self._draft = replace(self._draft, pn_warnings_acknowledged=checked)
        self._sync()

    def set_pn_plan(self, plan):
        if self._closed:
            return
        candidates = self._draft.pn_candidates
        if candidates:
            selected = {str(Path(self.folder, name).resolve()).casefold() for name in candidates}
            if any(str(Path(self.folder, name).resolve()).casefold() not in selected
                   for name in plan.candidate_files):
                self.pn_matching.error = "Matching included a background outside the selected files. Choose backgrounds and match again."
                self._sync()
                return
        else:
            candidates = self._pn_plan_pool(plan)
        self._pn_assignment_value = "matched"
        self._draft = replace(self._draft, kind=plan.kind, which=plan.which, pn_plan=plan,
            pn_candidates=tuple(candidates),
            pn_warnings_acknowledged=False, files=(), version=None, regions=(),
            bn_plan=None, calibration=None, fixed=False, saved_assignments=())
        self.pn_matching.error = ""
        self._sync()

    def _frames_changed(self, *_):
        if self._updating or self._busy:
            return
        combo = self.self_frames if self._draft.kind == "self" else self.bn_frames if self._draft.kind == "bn" else self.pn_frames
        self._draft = replace(self._draft, which=combo.currentData())
        rematch = False
        if self._draft.kind in {"p", "n", "pn"} and self._pn_assignment_value == "matched":
            self._draft = replace(self._draft, pn_plan=None, pn_warnings_acknowledged=False)
            rematch = bool(self._draft.pn_candidates)
        self._sync()
        if rematch:
            expected = (self._draft.kind, self._draft.which, self._draft.pn_candidates)
            QTimer.singleShot(0, self, lambda: self._rematch_pn(expected))

    def _rematch_pn(self, expected):
        if (not self._closed and not self._busy and self._pn_assignment_value == "matched"
                and (self._draft.kind, self._draft.which, self._draft.pn_candidates) == expected):
            self.pn_matching.match()

    def _strategy_changed(self, *_):
        if not self._updating:
            self._draft = replace(self._draft, fixed=bool(self.group_strategy.currentData()))
            self._sync()

    def _auto_changed(self, checked):
        if not self._updating:
            self._draft = replace(self._draft, auto_bn_compare=checked)
            self._sync()

    def _review_changed(self, checked):
        if not self._updating:
            self._draft = replace(self._draft, bn_warnings_acknowledged=checked)
            self._sync()

    def _candidate_mode_changed(self, *_):
        if self._updating or self._busy:
            return
        candidates = None if self.bn_auto.isChecked() else (self._draft.bn_candidates or ())
        if candidates != self._draft.bn_candidates:
            self._draft = replace(self._draft, bn_candidates=candidates, bn_plan=None,
                                  bn_warnings_acknowledged=False, files=(), calibration=None,
                                  regions=(), version=None, fixed=False)
            self._bn_error = ""
            self._sync()

    def _pick(self, kind, files, title):
        if self._select_files is None:
            return None
        try:
            result = self._select_files(self, kind, tuple(files), title)
            return None if result is None else tuple(str(name) for name in result)
        except Exception as exc:
            self.status.setText(str(exc))
            return None

    def _choose_candidates(self):
        if self._catalog is not None:
            self._pool_expanded = True
            self.advanced_button.setChecked(True)
            if self._draft.bn_candidates is None:
                self._inline_files_changed("bn_candidates", ())
            else:
                self._sync()
            self.bn_pool_selector.search.setFocus()
            QTimer.singleShot(0, self, lambda: self._scroll.ensureWidgetVisible(self.bn_pool_selector, 0, 12))
            return
        files = self._pick("bn_candidates", self._draft.bn_candidates or (), "Choose BN matching candidates")
        if files is not None:
            self._bn_method_value = "matched"
            self._draft = replace(self._draft, kind="bn", bn_candidates=files,
                                  files=(), bn_plan=None, bn_warnings_acknowledged=False,
                                  calibration=None, regions=(), version=None, fixed=False)
            self._bn_error = ""
            self._sync()

    def _choose_common_bn(self):
        if self._catalog is not None:
            self.bn_method.setCurrentIndex(self.bn_method.findData("common"))
            self.bn_common_selector.search.setFocus()
            return
        files = self._pick("bn", self._draft.files, "Choose a common BN background")
        if files is not None:
            if len(files) > 1:
                self.status.setText("Choose one common BN file; use Change… to choose matching candidates.")
                return
            if self._bn_method_value == "matched":
                self._bn_method_drafts["matched"] = self._draft
            self._bn_method_value = "common"
            self._draft = replace(self._draft, kind="bn", files=files, bn_plan=None,
                                  regions=(), version=None, calibration=None, bn_warnings_acknowledged=False)
            self._bn_error = ""
            self._sync()

    def _choose_pn_files(self):
        files = self._pick(self._draft.kind, self._draft.files, "Choose " + self._draft.kind.upper() + " source files")
        if files is not None:
            self._draft = replace(self._draft, files=files, regions=(), version=None,
                                  calibration=None, bn_plan=None, pn_plan=None,
                                  pn_warnings_acknowledged=False, saved_assignments=())
            self._sync()

    def _child(self, key):
        callback = self._callbacks[key]
        if callback is None or self._busy:
            return
        try:
            result = callback(self, self._draft)
            if result is not None:
                if not isinstance(result, BackgroundDraft):
                    raise TypeError("Background workflow must return a BackgroundDraft.")
                previous = self._draft
                if previous.kind in {"p", "n", "pn"}:
                    self._drafts[previous.kind] = previous
                    self._pn_assignment_modes[previous.kind] = self._pn_assignment_value
                    self._pn_assignment_drafts[(previous.kind, self._pn_assignment_value)] = previous
                if result.kind in {"p", "n", "pn"} and result.kind != previous.kind:
                    result = replace(result, pn_candidates=None)
                self._draft = result
                if result.kind in {"p", "n", "pn"}:
                    self._draft = replace(result, pn_plan=None, pn_warnings_acknowledged=False, saved_assignments=())
                    self._pn_assignment_value = "common"
                self._bn_error = ""
                self._sync()
        except Exception as exc:
            self.status.setText(str(exc))

    def _match_bn(self):
        if self._busy or self._closed or self._worker is not None:
            return
        self._bn_error = ""
        self._bn_method_value = "matched"
        self._draft = replace(self._draft, files=(), bn_plan=None, bn_warnings_acknowledged=False,
                              calibration=None, regions=(), version=None, fixed=False)
        self.set_bn_busy(True)
        if self._request_bn_match is not None:
            try:
                result = self._request_bn_match(self, self._draft.bn_candidates)
                if result is not None:
                    self.set_bn_plan(result)
            except Exception as exc:
                self.set_bn_error(str(exc))
            return
        self._bn_stop = Event()
        worker = Worker(_build_bn_plan, self.folder, self.measurements,
                        self._draft.bn_candidates, self._bn_stop,
                        confirmed_cache=self.confirmed_bn_cache,
                        cancellation=LoadCancellation())
        self._worker = worker
        worker.signals.result.connect(self._bn_result)
        worker.signals.error.connect(self._bn_failed)
        worker.signals.finished.connect(self._bn_finished)
        self.pool.start(worker)

    @Slot(object)
    def _bn_result(self, result):
        if not self._closed and self._worker is not None:
            plan, confirmed = result if isinstance(result, tuple) and len(result) == 2 else (result, False)
            self.set_bn_plan(plan)
            if confirmed:
                self._draft = replace(self._draft, bn_warnings_acknowledged=True)
                self._sync()

    @Slot(str)
    def _bn_failed(self, error):
        if not self._closed:
            self.set_bn_error(error.split("\n\nTraceback", 1)[0])

    @Slot()
    def _bn_finished(self):
        self._worker = None
        self._bn_stop = None
        if not self._closed:
            self.set_bn_busy(False)

    def set_bn_busy(self, busy):
        if not self._closed:
            self._busy = bool(busy)
            self._sync()

    def set_bn_plan(self, plan):
        if self._closed:
            return
        self._draft = replace(self._draft, kind="bn", files=(), bn_plan=plan,
                              bn_warnings_acknowledged=False, calibration=None, regions=(),
                              version=None, fixed=False)
        self._bn_method_value = "matched"
        self._bn_error = ""
        # Keep controls locked until the worker's finished signal. This prevents
        # a new request from being interleaved with the previous completion.
        self._busy = self._worker is not None
        self._sync()

    def set_bn_error(self, message):
        if self._closed:
            return
        self._bn_error = str(message)
        self._busy = self._worker is not None
        self._sync()

    @staticmethod
    def _warnings(plan):
        if plan is None:
            return ()
        warnings = [str(item) for item in getattr(plan, "warnings", ()) if item]
        for batch in getattr(plan, "batches", ()):
            warnings.extend(str(item) for item in getattr(batch, "warnings", ()) if item)
            if getattr(batch, "status", "") == "conflict" and not getattr(batch, "warnings", ()):
                warnings.append("Per-file BN suggestions need review within this batch.")
        for row in getattr(plan, "rows", ()):
            warnings.extend(str(item) for item in getattr(row, "warnings", ()) if item)
            unknown = getattr(row, "unknown_conditions", ())
            described = " ".join(warnings).casefold()
            if unknown and not all(str(item).casefold() in described for item in unknown):
                warnings.append("Unknown conditions: " + ", ".join(map(str, unknown)))
        return tuple(dict.fromkeys(warnings))

    @staticmethod
    def _batch_records(plan):
        """Legacy plans remain explicit unverified singles, never invented runs."""
        batches = tuple(getattr(plan, "batches", ()))
        if batches:
            return tuple(dict(measurements=tuple(batch.measurement_files),
                baseline=getattr(batch, "baseline_file", "") or "",
                status=getattr(batch, "status", "unverified"),
                verified=bool(getattr(batch, "verified_run", False)),
                run=(getattr(batch, "experiment_id", ""), getattr(batch, "run_id", "")),
                batch=batch) for batch in batches)
        return tuple(dict(measurements=(row.measurement_file,),
            baseline=getattr(row, "baseline_file", "") or "",
            status="unverified" if getattr(row, "baseline_file", "") else "unresolved",
            verified=False, run=None, batch=None) for row in getattr(plan, "rows", ()))

    def _summary_mapping(self):
        self.bn_summary_model.removeRows(0, self.bn_summary_model.rowCount())
        if self._bn_method_value == "common" and self._draft.files:
            groups = [dict(measurements=self.measurements, baseline=self._draft.files[0],
                           status="common", records=())]
        else:
            groups = []
            for record in self._batch_records(self._draft.bn_plan):
                status = record["status"]
                # Combining display rows never changes the underlying batches
                # or chooses a winner for a conflict.
                key = (status, record["baseline"], record["verified"])
                if status in {"conflict", "unresolved"}:
                    key += (record["measurements"],)
                group = next((group for group in groups if group["key"] == key), None)
                if group is None:
                    group = dict(key=key, measurements=(), baseline=record["baseline"],
                                 status=status, records=[])
                    groups.append(group)
                group["measurements"] += record["measurements"]
                group["records"].append(record)
        for group in groups:
            count = self.measurement_count if group["status"] == "common" else len(group["measurements"])
            coverage = f"{count} measurement" + ("" if count == 1 else "s")
            records = group["records"]
            status = group["status"]
            verified_runs = {record["run"] for record in records if record["verified"]}
            if status == "common":
                coverage += "\nOne BN for all"
            elif status == "unverified":
                singles = len(records)
                coverage += f"\n{singles} unverified single" + ("" if singles == 1 else "s")
            elif verified_runs:
                runs = len(verified_runs)
                coverage += f"\n{runs} verified run" + ("" if runs == 1 else "s")
            if status in {"conflict", "unresolved"}:
                coverage += "\n" + status.capitalize()
            baseline = group["baseline"]
            member_names = set(group["measurements"])
            individual_bns = {getattr(row, "baseline_file", "") for row in getattr(self._draft.bn_plan, "rows", ())
                              if row.measurement_file in member_names and getattr(row, "baseline_file", "")}
            name = (Path(baseline).name if baseline else
                    ("Different BNs within this batch — review individual matches" if len(individual_bns) > 1 else
                     "Per-file suggestions need review") if status == "conflict" else
                    "No unambiguous BN — review details")
            first, second = QStandardItem(coverage), QStandardItem(name)
            first.setToolTip("\n".join(str(Path(self.folder, name).resolve()) for name in group["measurements"]))
            second.setToolTip(str(Path(self.folder, baseline).resolve()) if baseline else name)
            self.bn_summary_model.appendRow((first, second))
        self._summary_resize.start(0)

    def _mapping(self):
        draft = self._draft
        key = (id(draft.bn_plan), draft.files, self._bn_method_value)
        if self._mapped_plan == key:
            return
        self._mapped_plan = key
        self._summary_mapping()
        self.bn_model.removeRows(0, self.bn_model.rowCount())
        if draft.kind == "bn" and self._bn_method_value == "common" and draft.files:
            first = QStandardItem(f"All {self.measurement_count} measurements")
            first.setToolTip("\n".join(str(Path(self.folder, name).resolve()) for name in self.measurements))
            second = QStandardItem(Path(draft.files[0]).name)
            second.setToolTip(str(Path(self.folder, draft.files[0]).resolve()))
            self.bn_model.appendRow([first, second])
        else:
            for row in getattr(draft.bn_plan, "rows", ()):
                files = (getattr(row, "measurement_file", ""), getattr(row, "baseline_file", "") or "")
                items = []
                for name in files:
                    item = QStandardItem(Path(name).name if name else "No unambiguous match")
                    item.setToolTip(str(Path(self.folder, name).resolve()) if name else "No unambiguous match")
                    items.append(item)
                self.bn_model.appendRow(items)
        if self.bn_model.rowCount():
            self.bn_table.selectRow(0)
        self._bn_selected_details()
        self._row_resize.start(0)

    def _bn_selected_details(self, *_):
        plan = self._draft.bn_plan
        warnings = self._warnings(plan)
        lines = ["Warnings", *warnings, ""] if warnings else []
        rows = tuple(getattr(plan, "rows", ()))
        index = self.bn_table.currentIndex().row()
        if rows:
            row = rows[max(0, min(index, len(rows) - 1))]
            gap = getattr(row, "time_gap_seconds", None)
            basis = getattr(row, "timing_basis", "")
            basis = {"capture_started_utc": "Capture start (UTC)", "acquisition": "Acquisition time",
                     "filesystem_mtime_proxy": "File modification time",
                     "filesystem_time_mtime": "File modification time", "mtime fallback": "File modification time"}.get(basis, basis or "Unknown")
            lines.extend(("Measurement: " + str(Path(self.folder, row.measurement_file).resolve()),
                          "BN: " + (str(Path(self.folder, row.baseline_file).resolve()) if row.baseline_file else "Unresolved"),
                          "Time basis: " + basis,
                          "Time gap: " + ("Unknown" if gap is None else f"{gap:g} s")))
            runner_up = getattr(row, "runner_up_gap_seconds", None)
            if runner_up is not None:
                lines.append(f"Runner-up gap: {runner_up:g} s")
            evidence = getattr(row, "evidence", "")
            if evidence:
                lines.extend(("", str(evidence)))
            for batch in getattr(plan, "batches", ()):
                if row.measurement_file not in batch.measurement_files:
                    continue
                lines.extend(("", "Batch status: " + batch.status,
                    "Run: " + (batch.run_id if batch.verified_run else "Unverified singleton"),
                    "Batch evidence: " + batch.evidence))
                for label, field in (("First capture start (UTC)", "capture_start"),
                                     ("Last capture start (UTC)", "capture_end")):
                    timestamp = getattr(batch, field, None)
                    if timestamp is not None:
                        lines.append(label + ": " + datetime.fromtimestamp(timestamp, timezone.utc).isoformat())
                for name, gap_sum in getattr(batch, "aggregate_time_gaps", ()):
                    lines.append(f"Candidate gap sum: {Path(name).name} · {gap_sum:g} s")
                break
        elif self._draft.files and self._bn_method_value == "common":
            lines.extend("BN: " + str(Path(self.folder, name).resolve()) for name in self._draft.files)
        self.bn_details_text.setPlainText("\n".join(lines))

    def _can_fix(self):
        draft = self._draft
        if draft.pn_plan is not None or draft.saved_assignments:
            return False
        if draft.kind not in {"bn", "p", "n", "pn", "custom"}:
            return False
        plan = draft.bn_plan
        resolution = getattr(plan, "resolution", None)
        heterogeneous = getattr(resolution, "numerical_path", "") == "heterogeneous"
        calibrated = (draft.calibration is not None or any(
            ".drr_bn_calibration" in str(name).replace("\\", "/").casefold().split("/")
            for name in draft.files))
        configured = bool(draft.files or draft.version) if plan is None else self._bn_ready(plan)
        return configured and not heterogeneous and not calibrated

    @staticmethod
    def _bn_ready(plan):
        rows = tuple(getattr(plan, "rows", ()))
        return (bool(getattr(plan, "ready", False)) and bool(rows)
                and all(getattr(row, "baseline_file", "") for row in rows)
                and not any(getattr(batch, "status", "") == "unresolved"
                            for batch in getattr(plan, "batches", ())))

    def _validation(self):
        if self._busy:
            return False, "Matching P/N…" if self._draft.kind in {"p", "n", "pn"} else "Matching BN…"
        draft = self._draft
        if self.measurement_count <= 0:
            return False, "Select measurements before applying a background."
        if draft.saved_assignments:
            return True, "Ready to apply the saved per-measurement background assignments."
        if draft.kind in {"p", "n", "pn"} and self._pn_assignment_value == "matched":
            if self.pn_matching.error:
                return False, self.pn_matching.error
            if not draft.pn_candidates:
                return False, "Choose background files to match to the selected measurements."
            if draft.pn_plan is None:
                return False, "Match the selected background files to assign backgrounds to measurements."
            if getattr(draft.pn_plan, "automatic_candidates", False):
                return False, "Match the selected background files again to confirm this previous assignment."
            if not draft.pn_plan.ready:
                return False, "Unresolved P/N matches. Change candidate files or assign a compatible background to the unresolved rows."
            if draft.pn_plan.warnings and not draft.pn_warnings_acknowledged:
                return False, "Review the P/N matching evidence and accept the flagged matches."
            return True, "Ready to apply. Every selected measurement keeps its assigned background."
        selector = (self.bn_common_selector if draft.kind == "bn" and self._bn_method_value == "common"
                    else self.pn_selector if draft.kind in {"p", "n"} else None)
        if selector is not None and draft.calibration is None and selector.problem:
            return False, selector.problem
        if draft.kind == "bn":
            if self._bn_error:
                return False, self._bn_error
            if draft.bn_plan is not None:
                if not self._bn_ready(draft.bn_plan):
                    return False, "Unresolved matches. Review details, change candidates or choose one BN for all."
                if self._warnings(draft.bn_plan) and not draft.bn_warnings_acknowledged:
                    return False, "Review Match details and accept the flagged matches."
                if any(getattr(batch, "status", "") == "conflict" for batch in getattr(draft.bn_plan, "batches", ())):
                    return True, "Apply keeps the reviewed per-file BN choices."
                return True, "Ready to apply."
            if not draft.files:
                return False, "Choose a BN file." if self._bn_method_value == "common" else "Match BN to continue."
        elif draft.kind in {"p", "n", "custom"} and not draft.files:
            return False, "Choose background source files before applying."
        elif draft.kind == "pn" and not (draft.files or draft.version):
            return False, "Configure a P+N composite or choose a saved version."
        return True, "Ready to apply."

    def _sync(self):
        if self._closed:
            return
        self._updating = True
        try:
            draft = self._draft
            custom = draft.kind == "custom"
            source = self._source_key(draft.kind)
            self.source_group.setExclusive(False)
            for key, button in self._source_widgets.items():
                button.setChecked(not custom and key == source)
                button.setEnabled(not self._busy)
            self.source_group.setExclusive(True)
            self.pages.setCurrentIndex(4 if custom else {"bn": 0, "pn": 1, "self": 2, "fitted": 3}[source])
            for index in range(self.pages.count()):
                self.pages.widget(index).setSizePolicy(QSizePolicy.Preferred,
                    QSizePolicy.Preferred if index == self.pages.currentIndex() else QSizePolicy.Ignored)
            self.pages.updateGeometry()
            has_mapping = draft.kind == "bn" and (draft.bn_plan is not None or bool(draft.files))
            expanded = has_mapping and self.bn_details_button.isChecked()
            self.pages.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding if expanded else QSizePolicy.Maximum)
            self._body_layout.setStretch(self._source_page_index, 1 if expanded else 0)
            self._body_layout.setStretch(self._body_spacer_index, 0 if expanded else 1)
            matched = self._bn_method_value == "matched"
            inline = self._catalog is not None
            self.bn_method.setCurrentIndex(self.bn_method.findData(self._bn_method_value))
            self.bn_method.setEnabled(not self._busy)
            self.bn_scope_row.setVisible(matched)
            self.bn_scope_btn.setEnabled(not self._busy)
            self._scope_menu_state()
            self.bn_auto.setChecked(draft.bn_candidates is None)
            self.bn_restricted.setChecked(draft.bn_candidates is not None)
            self.bn_auto.setEnabled(not self._busy)
            self.bn_restricted.setEnabled(not self._busy)
            self.bn_candidates_btn.setEnabled(not self._busy and self._select_files is not None)
            self.bn_candidates_label.setText("Current experiment" if draft.bn_candidates is None else f"{len(draft.bn_candidates)} candidate file(s)")
            self.bn_candidates_label.setToolTip("\n".join(draft.bn_candidates or ()))
            max_gap = getattr(draft.bn_plan, "max_gap_seconds", 86400.)
            margin = getattr(draft.bn_plan, "min_margin_seconds", 60.)
            self.bn_timing_note.setText(
                f"Timing checks: maximum gap {max_gap:g} s; minimum nearest-candidate margin {margin:g} s. "
                "These matching heuristics do not measure physical confidence.")
            self.bn_scope_advanced.setVisible(inline and draft.kind == "bn" and matched)
            self.bn_common_btn.setEnabled(not self._busy and self._select_files is not None)
            self.bn_common_btn.setVisible(not matched and not inline)
            self.bn_manual_btn.setEnabled(not self._busy and (inline or self._select_files is not None))
            self.bn_manual_btn.setVisible(matched)
            self.bn_pool_panel.setVisible(inline and matched and self._pool_expanded)
            if inline:
                self.bn_common_selector.setVisible(not matched)
                bn_files = draft.files
                if isinstance(draft.calibration, dict):
                    identity = (draft.calibration.get("payload") or {}).get("source_identity", {})
                    bn_files = tuple(item["path"] for item in identity.get("baseline_files", ()) if "path" in item)
                self.bn_common_selector.set_files(bn_files if draft.kind == "bn" and not matched else ())
                self.bn_pool_selector.set_files(draft.bn_candidates or ())
                self.bn_common_selector.set_active(draft.kind == "bn" and not matched)
                self.bn_pool_selector.set_active(draft.kind == "bn" and matched and self._pool_expanded)
                for selector in (self.bn_common_selector, self.bn_pool_selector, self.pn_selector):
                    selector.setEnabled(not self._busy)
            self.bn_match_btn.setEnabled(not self._busy and self.measurement_count > 0 and
                                        (draft.bn_candidates is None or bool(draft.bn_candidates)) and
                                        not (self.bn_pool_selector is not None and draft.bn_candidates is not None and self.bn_pool_selector.problem))
            common = draft.kind == "bn" and not matched and bool(draft.files)
            self.bn_common_summary.setText("Common BN: " + ", ".join(draft.files) if common else "")
            if inline:
                calibrated = common and draft.calibration is not None
                if calibrated:
                    self.bn_common_summary.setText(f"Calibrated BN from {len(bn_files)} source file" + ("" if len(bn_files) == 1 else "s"))
                    self.bn_common_summary.setToolTip("\n".join(bn_files))
                    self.bn_common_summary.setAccessibleDescription("\n".join(bn_files))
                self.bn_common_summary.setVisible(calibrated)
            self.bn_frames.setVisible(not matched)
            self.bn_frames.setCurrentIndex(max(0, self.bn_frames.findData(draft.which)))
            self.bn_frames.setEnabled(not self._busy)
            self._mapping()
            self.bn_summary_table.setVisible(has_mapping and (matched or not inline))
            self.bn_table.setVisible(has_mapping)
            warnings = self._warnings(draft.bn_plan)
            self.bn_review.setVisible(matched and bool(warnings))
            self.bn_review.setChecked(draft.bn_warnings_acknowledged)
            self.bn_review.setEnabled(not self._busy and self._bn_ready(draft.bn_plan))
            self.bn_review.setToolTip("\n".join(warnings))
            rows = tuple(getattr(draft.bn_plan, "rows", ()))
            records = self._batch_records(draft.bn_plan)
            verified_runs = {record["run"] for record in records if record["verified"]}
            conflicts = sum(record["status"] == "conflict" for record in records)
            unresolved = sum(record["status"] == "unresolved" for record in records)
            warning_text = " ".join(warnings).casefold()
            categories = []
            if any(word in warning_text for word in ("time", "capture", "gap")):
                categories.append("timing")
            if any(word in warning_text for word in ("condition", "temperature", "polarization", "power")):
                categories.append("conditions")
            if any(word in warning_text for word in ("ambiguous", "equally", "same acquisition")):
                categories.append("ambiguity")
            if any(word in warning_text for word in ("unverified", "run provenance")):
                categories.append("unverified runs")
            if "conflict" in warning_text:
                categories.append("conflict")
            warning_summary = (f" · {len(warnings)} warning" + ("" if len(warnings) == 1 else "s") +
                               ": " + ", ".join(categories or ["review"])) if warnings else ""
            overview = f"{len(rows)} measurements retained" if rows else f"{self.measurement_count} measurements"
            if verified_runs:
                overview += f" · {len(verified_runs)} verified run" + ("" if len(verified_runs) == 1 else "s")
            if conflicts:
                overview += f" · {conflicts} batch conflict" + ("" if conflicts == 1 else "s")
            if unresolved:
                overview += f" · {unresolved} unresolved batch" + ("" if unresolved == 1 else "es")
            self.bn_plan_status.setText(overview + warning_summary)
            self.bn_plan_status.setVisible(matched)
            issues = len(warnings) + conflicts + unresolved
            self.bn_details_button.setText(f"Review &details ({issues})" if issues else "Match &details")
            self.bn_details_button.setAccessibleDescription("Show per-file suggestions, run evidence, timing and condition warnings.")
            self.bn_details_button.setVisible(matched and bool(rows or warnings))
            self.bn_details_panel.setVisible(matched and bool(rows or warnings) and self.bn_details_button.isChecked())
            self._bn_selected_details()
            if draft.kind in {"p", "n", "pn"}:
                self.pn_method.setCurrentIndex(self.pn_method.findData(draft.kind))
            composite = draft.kind == "pn"
            pn_matched = self._pn_assignment_value == "matched"
            pn_saved = self._pn_assignment_value == "saved"
            if draft.saved_assignments and self.pn_assignment_method.findData("saved") < 0:
                self.pn_assignment_method.addItem("Saved measurement assignments", "saved")
            elif not draft.saved_assignments and (draft.kind, "saved") not in self._pn_assignment_drafts:
                index = self.pn_assignment_method.findData("saved")
                if index >= 0:
                    self.pn_assignment_method.removeItem(index)
            self.pn_assignment_method.setCurrentIndex(self.pn_assignment_method.findData(self._pn_assignment_value))
            self.pn_assignment_method.setEnabled(not self._busy)
            self.pn_matching.setVisible(pn_matched)
            self.pn_matching.configure(draft, busy=self._busy)
            self.pn_method.setEnabled(not self._busy)
            self.pn_files_btn.setVisible(not composite and not inline and not pn_matched and not pn_saved)
            self.pn_files_btn.setEnabled(not self._busy and self._select_files is not None)
            self.pn_files_summary.setText(describe_background(draft))
            self.pn_files_summary.setToolTip("\n".join(draft.files))
            self.pn_files_summary.setVisible(pn_saved or (not pn_matched and composite and bool(draft.version or draft.files)))
            self.pn_files_view.setVisible(pn_saved or (not pn_matched and not composite and bool(draft.files) and not inline))
            if inline:
                self.pn_selector.setVisible(not composite and not pn_matched and not pn_saved)
                if draft.kind in {"p", "n"}:
                    self.pn_selector.set_role(draft.kind)
                    self.pn_selector.set_files(draft.files)
                self.pn_selector.set_active(draft.kind in {"p", "n"} and not pn_matched and not pn_saved)
            files_key = (tuple(item.measurement_file + " → " + ", ".join(item.baseline_files)
                               for item in draft.saved_assignments) if pn_saved else
                         tuple(draft.files) if draft.kind in {"p", "n"} else ())
            if files_key != self._pn_files_key:
                self._pn_files_key = files_key
                self.pn_files_model.clear()
                for name in files_key:
                    item = QStandardItem(name if pn_saved else Path(name).name)
                    item.setToolTip(name if pn_saved else str(Path(self.folder, name).resolve()))
                    self.pn_files_model.appendRow(item)
            self.pn_frames.setVisible(not composite and not pn_saved)
            for index in range(self.pn_frames.count()):
                method = self.pn_frames.itemData(index)
                labels = ({"all": "Average all frames of each assigned background file",
                           "first": "First frame of each assigned background file",
                           "last": "Last frame of each assigned background file"} if pn_matched else
                          {"all": "All frames from each file, then average",
                           "first": "First frame from each file, then average",
                           "last": "Last frame from each file, then average"})
                self.pn_frames.setItemText(index, labels[method])
            self.repeat_note.setVisible(not composite and not pn_matched and not pn_saved)
            self.pn_source_frames_note.setText(source_frame_summary(draft.version))
            self.pn_source_frames_note.setVisible(not pn_matched and not pn_saved and composite and bool(draft.version or draft.files))
            self.pn_frames.setCurrentIndex(max(0, self.pn_frames.findData(draft.which)))
            self.pn_frames.setEnabled(not self._busy)
            self.pn_composite_btn.setVisible(composite or pn_matched)
            self.pn_composite_btn.setEnabled(not self._busy and self._callbacks["composite"] is not None)
            self.self_frames.setCurrentIndex(max(0, self.self_frames.findData(draft.which)))
            self.self_frames.setEnabled(not self._busy)
            self.fit_summary.setText("Preview configured" if draft.fitted_result is not None else "Current fit settings")
            self.fit_btn.setEnabled(not self._busy and self._callbacks["fit"] is not None and self.measurement_count > 0)
            self.custom_summary.setText(describe_background(draft))
            can_fix = self._can_fix()
            if not can_fix and draft.fixed:
                draft = replace(draft, fixed=False)
                self._draft = draft
            self.group_strategy.setCurrentIndex(1 if draft.fixed else 0)
            self.group_strategy.model().item(1).setEnabled(can_fix)
            self.group_strategy.setEnabled(not self._busy)
            self.strategy_note.setText("Each measurement keeps its own self frame." if draft.kind == "self" else
                "Fitted backgrounds are evaluated for each measurement group." if draft.kind == "fitted" else
                "Choose a common background before keeping it fixed." if not (draft.files or draft.version or draft.bn_plan or draft.pn_plan or draft.saved_assignments) else
                "Per-measurement assignments or group calibration must be reviewed for the next group." if not can_fix else
                "Following the type selects a background for each new group; fixed reuses this exact background.")
            self.group_strategy.setToolTip(self.strategy_note.text())
            self.strategy_note.setVisible(not can_fix)
            for button, callback in ((self.custom_btn, "custom"), (self.edit_btn, "edit")):
                button.setEnabled(not self._busy and self._callbacks[callback] is not None)
            bn_common = (draft.kind == "bn" and self.measurement_count > 0 and
                (bool(draft.files) if draft.bn_plan is None else
                 self._bn_ready(draft.bn_plan) and
                 getattr(getattr(draft.bn_plan, "resolution", None), "numerical_path", "") != "heterogeneous"))
            self.calibration_btn.setEnabled(not self._busy and bn_common and self._callbacks["calibration"] is not None)
            self.calibration_btn.setToolTip("Compare BN calibration for this group; Apply remains separate." if bn_common else
                "Choose one common measured BN before configuring calibration.")
            self.calibration_summary.setText("BN calibration configured for this group." if draft.calibration is not None else "")
            self.calibration_summary.setVisible(draft.calibration is not None)
            self.auto_bn_compare.setChecked(draft.auto_bn_compare)
            self.auto_bn_compare.setEnabled(not self._busy)
            self.progress.setVisible(self._busy)
            self.progress.setAccessibleName("Matching P/N backgrounds" if draft.kind in {"p", "n", "pn"} else "Matching BN backgrounds")
            self.draft_summary.setText("Draft: " + describe_background(draft))
            valid, message = self._validation()
            self.status.setText(message)
            self.status.setAccessibleDescription(message)
            self.apply_btn.setEnabled(valid)
            self.apply_btn.setToolTip(message)
        finally:
            self._updating = False

    def _apply(self):
        valid, message = self._validation()
        if not valid:
            self.status.setText(message)
            return
        self.selection = self._draft
        self.accept()

    def _stop(self, *_):
        self._closed = True
        self.pn_matching.stop()
        if self._bn_stop is not None:
            self._bn_stop.set()
        if self._worker is not None:
            self._worker.cancellation.cancel()
