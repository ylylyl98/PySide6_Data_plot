"""One background workspace with private drafts and explicit application.

The two embedded editors retain their workers and provenance. Only the outer
workflow applies a validated selection to the main view. Saving is independent.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QTabWidget, QVBoxLayout

from core.drr_background_regions import DrrBackgroundRegion
from core.drr_composite_background import compose_background, read_composite_background
from core.drr_pn_store import _file_digest
from core.drr_sources import DrrMeasurementAssignment, portable_source_name, resolve_source_path
from core.load_cancellation import check_load_cancelled
from ui_qt.drr_composite_background_dialog import DrrCompositeBackgroundDialog, manual_pn_derived_document
from ui_qt.drr_pn_background_dialog import DrrPnBackgroundDialog
from ui_qt.fluent_ui.style import set_fluent_property


@dataclass(frozen=True)
class BackgroundSelection:
    files: tuple[str, ...] = ()
    which: str = "last"
    regions: tuple[DrrBackgroundRegion, ...] = ()
    version: object = None
    allow_stale: bool = False


def validate_editor_selection(folder, measurements, document, saved_path=None):
    """Validate in a worker, preserving live recipes versus saved snapshots."""
    from core.data_io import _validate_drr_external_spectral_grids
    from core.loader import build_effective_background
    energy, expected, _boundaries = compose_background(document)
    if saved_path is not None:
        saved = read_composite_background(saved_path)
        saved_energy, saved_intensity, _ = compose_background(saved)
        same_sources = len(saved.sources) == len(document.sources)
        for actual_source, source in zip(saved.sources, document.sources):
            same_sources = same_sources and all(
                getattr(actual_source, field) == getattr(source, field)
                for field in ("key", "label", "files", "which", "spectral_columns", "boundaries_ev", "source_hashes"))
            same_sources = same_sources and (
                np.array_equal(actual_source.energy, source.energy)
                and np.array_equal(actual_source.intensity, source.intensity, equal_nan=True)
                and json.loads(actual_source.generation_json or "{}") == json.loads(source.generation_json or "{}"))
        if (not np.array_equal(saved_energy, energy)
                or not np.array_equal(saved_intensity, expected, equal_nan=True)
                or saved.default_source != document.default_source or saved.regions != document.regions
                or not same_sources):
            raise ValueError("The saved background changed. Save this draft again before using it.")
        files = (portable_source_name(folder, resolve_source_path(folder, saved_path)),)
        if measurements:
            _validate_drr_external_spectral_grids(folder, measurements, files)
        check_load_cancelled()
        return BackgroundSelection(files=files)
    if not measurements:
        raise ValueError("Select measurements to apply a temporary recipe, or save this background first.")
    sources = {source.key: source for source in document.sources}
    keys = {document.default_source, *(region.source_key for region in document.regions)}
    if any(sources[key].generation_json for key in keys):
        raise ValueError("Save the generated source snapshots before applying this background.")
    for key in keys:
        source = sources[key]
        if not source.source_hashes:
            raise ValueError("This snapshot has no original source hashes. Save it before applying.")
        for name, digest in source.source_hashes:
            if _file_digest(resolve_source_path(folder, name)) != digest:
                raise ValueError("An original background changed after preview. Reload its source or save the frozen snapshot.")
    default = sources[document.default_source]
    files = tuple(portable_source_name(folder, resolve_source_path(folder, name)) for name in default.files)
    regions = tuple(DrrBackgroundRegion(region.xmin_ev, region.xmax_ev,
        tuple(portable_source_name(folder, resolve_source_path(folder, name))
              for name in sources[region.source_key].files), sources[region.source_key].which)
        for region in document.regions)
    selection = BackgroundSelection(files=files, which=default.which, regions=regions)
    assignment = DrrMeasurementAssignment(measurements[0], "External", files, default.which, regions=regions)
    _validate_drr_external_spectral_grids(folder, measurements, assignment.baseline_source_files)
    actual = build_effective_background(folder, assignment, energy)
    if not np.array_equal(actual, expected, equal_nan=True):
        raise ValueError("The live source recipe no longer matches its preview. Reload or save the frozen snapshot.")
    check_load_cancelled()
    return selection


class DrrBackgroundWorkspace(QDialog):
    """Generate/history and source/interval editing in one native window."""

    def __init__(self, parent=None, *, folder, catalog=(), measurements=(), initial_files=(),
                 initial_which="last", initial_regions=(), select_files, pool=None,
                 initial_page="current", kept_background=False):
        super().__init__(parent)
        self.setWindowTitle("DRR · Build P/N background")
        self.setObjectName("drr_background_workspace")
        self.setMinimumSize(760, 520)
        available = self.screen().availableGeometry()
        self.resize(min(1160, available.width() - 40), min(920, available.height() - 60))
        self.selection = None
        self._closed = False
        self.measurements = tuple(measurements)
        layout = QVBoxLayout(self)
        self.context_label = QLabel(
            f"Current measurement group: {len(self.measurements)} file(s). "
            "Generate for these data, reuse an explicitly associated version, or edit a background.")
        self.context_label.setWordWrap(True)
        if kept_background and initial_files:
            self.context_label.setText(self.context_label.text() +
                " Keep background is on: the previous recipe is available in Edit background; it is not a verified association for new data.")
        self.context_label.setToolTip("\n".join(self.measurements))
        layout.addWidget(self.context_label)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("drr_background_workspace_pages")
        self.tabs.setAccessibleName("Background generation and editing")
        layout.addWidget(self.tabs, 1)
        self.pn = DrrPnBackgroundDialog(self, folder=folder, catalog=catalog,
            measurements=self.measurements, select_files=select_files, pool=pool, sample_scoped=True)
        self.pn.setWindowFlags(Qt.Widget)
        self.pn.setMinimumSize(0, 0)
        self.pn.close_btn.hide()
        self.pn.use_btn.hide()
        self.pn_page = self._scroll_page(self.pn)
        self.tabs.addTab(self.pn_page, "Current data / P/N")
        self.editor = DrrCompositeBackgroundDialog(self, folder=folder,
            initial_files=initial_files, initial_which=initial_which, initial_regions=initial_regions,
            select_files=lambda files: select_files(files, "Choose background sources"), pool=pool,
            apply_callback=lambda document, path: validate_editor_selection(
                str(folder), self.measurements, document, path))
        self.editor.setWindowFlags(Qt.Widget)
        self.editor.setMinimumSize(0, 0)
        self.editor.cancel_button.hide()
        self.editor.apply_button.hide()
        self.editor_page = self._scroll_page(self.editor)
        self.tabs.addTab(self.editor_page, "Edit background")
        # Keep generation feedback visible even when the P/N page is scrolled.
        layout.addWidget(self.pn.progress_panel)
        row = QHBoxLayout()
        self.edit_selected_btn = QPushButton("Edit selected P/N background")
        self.edit_selected_btn.setToolTip("Copy the selected version into the editor. Its original saved version stays unchanged.")
        row.addWidget(self.edit_selected_btn)
        row.addStretch(1)
        self.apply_btn = QPushButton("Use background")
        self.apply_btn.setObjectName("drr_background_workspace_apply")
        self.apply_btn.setAutoDefault(False)
        set_fluent_property(self.apply_btn, "fluentAppearance", "primary")
        row.addWidget(self.apply_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setAutoDefault(False)
        row.addWidget(self.cancel_btn)
        layout.addLayout(row)
        self.note = QLabel("Saving keeps a reusable copy. Only Use background changes the main view; inspect ΔR/R and d²E there.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.pn.stateChanged.connect(self._controls)
        self.editor.stateChanged.connect(self._controls)
        self.pn.accepted.connect(self._pn_selected)
        self.editor.accepted.connect(self._editor_selected)
        self.pn.rejected.connect(self.reject)
        self.editor.rejected.connect(self.reject)
        self.tabs.currentChanged.connect(self._controls)
        self.edit_selected_btn.clicked.connect(self._edit_selected)
        self.apply_btn.clicked.connect(self._apply)
        self.cancel_btn.clicked.connect(self.reject)
        self.finished.connect(self._stop)
        if initial_page == "edit" or not self.measurements:
            self.tabs.setCurrentWidget(self.editor_page)
        self._controls()

    @staticmethod
    def _scroll_page(editor):
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.NoFrame)
        page.setWidget(editor)
        return page

    def _controls(self, *_):
        if self._closed:
            return
        busy = self.pn._worker is not None or self.editor._worker is not None
        self.tabs.tabBar().setEnabled(not busy)
        current_pn = self.tabs.currentWidget() is self.pn_page
        self.pn.progress_panel.setVisible(current_pn and (
            self.pn._generation_active or self.pn._generation_outcome is not None))
        self.apply_btn.setEnabled(not busy and (self.pn.use_btn.isEnabled()
                                               if current_pn else self.editor.apply_button.isEnabled()))
        self.apply_btn.setToolTip(self.pn.use_btn.toolTip() if current_pn else self.editor.apply_button.toolTip())
        self.edit_selected_btn.setVisible(current_pn)
        self.edit_selected_btn.setEnabled(not busy and self.pn._payload is not None)
        self.cancel_btn.setEnabled(self.editor._job != "save")

    def _edit_selected(self):
        if not self.edit_selected_btn.isEnabled():
            return
        version = self.pn._current
        self.editor.protected_paths.add(Path(version.csv_path).resolve())
        document = self.pn._payload["document"]
        self.editor._commit(manual_pn_derived_document(document, version))
        self.editor.status.setText("Selected P/N version copied into this draft. Save changes as a separate background before using them. Undo restores the previous editor draft.")
        self.tabs.setCurrentWidget(self.editor_page)

    def _apply(self):
        if not self.apply_btn.isEnabled():
            return
        if self.tabs.currentWidget() is self.pn_page:
            self.pn._use()
        else:
            self.editor.apply_background()

    def _pn_selected(self):
        if not self._closed and self.pn.selected_version is not None:
            self.selection = BackgroundSelection(version=self.pn.selected_version, allow_stale=self.pn.allow_stale)
            self.accept()

    def _editor_selected(self):
        if not self._closed and isinstance(self.editor.applied_background, BackgroundSelection):
            self.selection = self.editor.applied_background
            self.accept()

    def reject(self):
        if self.editor._job == "save":
            return
        super().reject()

    def _stop(self, *_):
        self._closed = True
        self.pn._stop()
        self.editor._stop()

    def closeEvent(self, event):
        if self.editor._job == "save":
            event.ignore()
            return
        self._stop()
        super().closeEvent(event)
