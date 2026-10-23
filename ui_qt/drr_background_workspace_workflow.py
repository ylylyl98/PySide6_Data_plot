"""Apply the explicit result of the shared background workspace."""
from pathlib import Path

from PySide6.QtWidgets import QDialog

from core.drr_sources import DrrMeasurementAssignment
from ui_qt.drr_pn_workflow import _context, apply_pn_version, cancel_pn_apply
from ui_qt.drr_region_backgrounds import frame_method


def apply_background_selection(owner, selection):
    """No save action calls this function; only an accepted Use action does."""
    if selection.version is not None:
        return apply_pn_version(owner, selection.version, allow_stale=selection.allow_stale)
    from ui_qt.drr_bn_workflow import forget_bn_choice
    forget_bn_choice(owner)
    owner.drr_controller._cancel_auto_external()
    owner._drr_background_regions = tuple(selection.regions)
    owner._drr_assignments = tuple(DrrMeasurementAssignment(
        name, "External", selection.files, selection.which, regions=selection.regions,
        selection_reason="Explicit background workspace selection") for name in owner.drr_selected_files)
    owner._drr_assignments_automatic = False
    owner._drr_baseline_user_selected = True
    owner._drr_external_auto_selected = False
    owner._drr_background_guess = None
    owner.drr_baseline_files_manual = list(selection.files)
    owner.drr_baseline_files_found = []
    frame_label = {"first": "First frame from each file, then average",
                   "last": "Last frame from each file, then average",
                   "all": "Average all frames in each file, then average files"}[selection.which]
    for widget, value in ((owner.drr_baseline_combo, "External"),
                          (owner.drr_baseline_combine_combo, frame_label)):
        previous = widget.blockSignals(True)
        widget.setCurrentText(value)
        widget.blockSignals(previous)
    previous = owner.drr_pin_baseline_chk.blockSignals(True)
    owner.drr_pin_baseline_chk.setChecked(True)
    owner.drr_pin_baseline_chk.blockSignals(previous)
    owner.drr_controller._update_drr_selection_labels()
    sources = tuple(dict.fromkeys((*selection.files,
        *(name for region in selection.regions for name in region.baseline_files))))
    owner._queue_drr_catalog_refresh(auto=True,
        old_source_files={source.source for source in owner.drr_available_sources},
        selected_sources=(*owner.drr_selected_files, *sources), force=True, catalog_only=True)
    owner._status("Background applied to the current group. Keep background is enabled; review ΔR/R and d²E in the main view.")
    if owner.drr_selected_files and all(Path(name).suffix.lower() == ".csv" for name in owner.drr_selected_files):
        owner._start_load("DRR")


def edit_background_workspace(owner, *, initial_page="current"):
    from ui_qt.drr_background_workspace import DrrBackgroundWorkspace
    if not owner.current_folder:
        owner._show_error("Open an experiment folder before editing backgrounds.")
        return
    cancel_pn_apply(owner)
    owner.drr_controller._cancel_auto_external()
    context = _context(owner)
    files, which = tuple(owner.drr_baseline_files_manual), frame_method(owner)
    regions = tuple(owner._drr_background_regions)
    assignments = tuple(owner._drr_assignments)
    if owner._drr_assignments_automatic and assignments:
        recipes = {(item.baseline_mode, item.baseline_files, item.baseline_which, item.regions)
                   for item in assignments}
        if len(recipes) == 1 and assignments[0].baseline_mode == "External":
            files, which, regions = assignments[0].baseline_files, assignments[0].baseline_which, assignments[0].regions
        else:
            files, regions = (), ()

    def select_files(current, title):
        return owner.drr_controller._open_drr_source_dialog(title=title, selected=list(current),
            baseline_mode=True, independent_background=True)

    dialog = DrrBackgroundWorkspace(owner, folder=context[0], catalog=tuple(owner.drr_available_sources),
        measurements=context[1], initial_files=files, initial_which=which, initial_regions=regions,
        select_files=select_files, pool=owner.thread_pool, initial_page=initial_page,
        kept_background=owner.drr_pin_baseline_chk.isChecked())
    try:
        if dialog.exec() != QDialog.Accepted or dialog.selection is None:
            return
        if getattr(owner, "_is_closing", False) or _context(owner) != context:
            owner._status("Background was not applied because the current selection changed. Saved copies remain available.")
            return
        apply_background_selection(owner, dialog.selection)
    finally:
        dialog.deleteLater()
        getattr(owner, "drr_pn_background_btn", owner.drr_edit_baselines_btn).setFocus()
