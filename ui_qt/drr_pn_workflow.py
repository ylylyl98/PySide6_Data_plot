"""Select a saved P/N result, then apply its one CSV on the GUI thread.

The dialog never records measurement associations. Nested pickers may cancel
after choosing a version; only the ordinary workflow's background worker calls
mark_used. Request identity and the captured UI context guard every delivery.
"""
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Slot
from PySide6.QtWidgets import QDialog

from core.drr_pn_store import DrrPnStore
from core.drr_sources import portable_source_name
from core.load_cancellation import LoadCancellation, check_load_cancelled
from ui_qt.common import Worker


def _context(owner):
    """Cheap UI snapshot: never hash or parse spectra on the GUI thread."""
    return (
        str(owner.current_folder),
        tuple(owner.drr_selected_files),
        owner.drr_baseline_combo.currentText(),
        tuple(owner.drr_baseline_files_manual),
        tuple(owner.drr_baseline_files_found),
        owner.drr_baseline_combine_combo.currentText(),
        tuple(owner._drr_background_regions),
        tuple(owner._drr_assignments),
        bool(owner._drr_assignments_automatic),
        owner.drr_pin_baseline_chk.isChecked(),
    )


def cancel_pn_apply(owner):
    """Invalidate a pending application on selection changes or window close."""
    request = getattr(owner, "_drr_pn_apply_request", None)
    if request is not None:
        request.cancellation.cancel()
    owner._drr_pn_apply_request = None


def _mark_version(folder, version_id, measurements, allow_stale, *, progress, log):
    check_load_cancelled()
    return DrrPnStore(folder).mark_used(
        version_id, measurements=measurements, allow_stale=allow_stale,
    )


def _apply_version(owner, folder, selected, version):
    """Apply one verified generated CSV; P/N inputs never enter averaging."""
    from ui_qt.drr_bn_workflow import forget_bn_choice
    forget_bn_choice(owner)
    name = portable_source_name(folder, version.csv_path)
    owner.drr_controller._cancel_auto_external()
    owner._drr_background_regions = ()
    owner._drr_assignments = ()
    owner._drr_assignments_automatic = False
    owner._drr_baseline_user_selected = True
    owner._drr_external_auto_selected = False
    owner._drr_background_guess = None
    owner.drr_baseline_files_manual = [name]
    owner.drr_baseline_files_found = []
    for widget, value in (
        (owner.drr_baseline_combo, "External"),
        (owner.drr_baseline_combine_combo, "Last frame from each file, then average"),
    ):
        blocked = widget.blockSignals(True)
        widget.setCurrentText(value)
        widget.blockSignals(blocked)
    blocked = owner.drr_pin_baseline_chk.blockSignals(True)
    owner.drr_pin_baseline_chk.setChecked(True)
    owner.drr_pin_baseline_chk.blockSignals(blocked)
    owner.drr_controller._update_drr_selection_labels()
    owner._queue_drr_catalog_refresh(
        auto=True, old_source_files={source.source for source in owner.drr_available_sources},
        selected_sources=(*selected, name), force=True, catalog_only=True,
    )
    historical = " Historical snapshot." if not version.fresh else ""
    owner._status(
        f"P/N background selected: {Path(version.csv_path).name}. "
        f"Keep background is enabled.{historical}"
    )
    if selected and all(Path(source).suffix.lower() == ".csv" for source in selected):
        owner._start_load("DRR")


class _PnApplyRequest(QObject):
    def __init__(self, owner, context, version, allow_stale):
        super().__init__(owner if isinstance(owner, QObject) else None)
        self.owner = owner
        self.context = context
        self.cancellation = LoadCancellation()
        self.worker = Worker(
            _mark_version, context[0], version.version_id, context[1], allow_stale,
            cancellation=self.cancellation,
        )
        self.worker.signals.result.connect(self.complete, Qt.QueuedConnection)
        self.worker.signals.error.connect(self.failed, Qt.QueuedConnection)
        self.worker.signals.finished.connect(self.finished, Qt.QueuedConnection)

    def current(self):
        return (
            not self.cancellation.cancelled
            and not getattr(self.owner, "_is_closing", False)
            and getattr(self.owner, "_drr_pn_apply_request", None) is self
            and _context(self.owner) == self.context
        )

    @Slot(object)
    def complete(self, version):
        if self.current():
            _apply_version(self.owner, self.context[0], self.context[1], version)

    @Slot(str)
    def failed(self, message):
        if self.current():
            self.owner._show_error(
                "Could not apply the P/N background. The current background is unchanged.\n"
                + message.split("\n\n", 1)[0]
            )

    @Slot()
    def finished(self):
        if getattr(self.owner, "_drr_pn_apply_request", None) is self:
            self.owner._drr_pn_apply_request = None
        self.deleteLater()


def apply_pn_version(owner, version, *, allow_stale=False):
    """Commit an accepted outer picker choice through the same guarded worker."""
    if not owner.current_folder or getattr(owner, "_is_closing", False):
        return None
    cancel_pn_apply(owner)
    owner.drr_controller._cancel_auto_external()
    context = _context(owner)
    request = _PnApplyRequest(owner, context, version, bool(allow_stale))
    owner._drr_pn_apply_request = request
    owner._status("Verifying the P/N background and saving its measurement association…")
    owner.thread_pool.start(request.worker)
    return version


def edit_pn_background(owner, *, selection_only=False):
    """Return the selected version, or None; ordinary application is asynchronous.

    selection_only changes no recipe or association; its explicit historical
    consent is exposed as _drr_pn_selection_allow_stale for the outer picker.
    The caller must explicitly commit its accepted outer selection separately.
    """
    from ui_qt.drr_pn_background_dialog import DrrPnBackgroundDialog

    if not owner.current_folder:
        owner._show_error("Open an experiment folder before choosing P/N backgrounds.")
        return None
    # The explicit chooser takes precedence over an in-flight automatic guess.
    # Cancel before the modal event loop can deliver a result into this recipe.
    owner.drr_controller._cancel_auto_external()
    if selection_only:
        owner._drr_pn_selection_allow_stale = False
    context = _context(owner)

    def select_files(current, title):
        return owner.drr_controller._open_drr_source_dialog(
            title=title, selected=list(current), baseline_mode=True,
            independent_background=True,
        )

    dialog = DrrPnBackgroundDialog(
        owner, folder=context[0], catalog=tuple(owner.drr_available_sources),
        measurements=context[1], select_files=select_files, pool=owner.thread_pool,
        sample_scoped=True,
    )
    try:
        if dialog.exec() != QDialog.Accepted or dialog.selected_version is None:
            return None
        version = dialog.selected_version
        if getattr(owner, "_is_closing", False) or _context(owner) != context:
            owner._status("P/N selection was not applied because the current selection changed.")
            return None
        if selection_only:
            owner._drr_pn_selection_allow_stale = bool(dialog.allow_stale)
            return version
        return apply_pn_version(owner, version, allow_stale=dialog.allow_stale)
    finally:
        dialog.deleteLater()
        button = getattr(owner, "drr_pn_background_btn", None)
        if button is not None:
            button.setFocus()
