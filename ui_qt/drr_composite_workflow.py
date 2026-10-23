"""Connect reusable composite backgrounds to the DRR workbench."""
from pathlib import Path

from PySide6.QtWidgets import QDialog

from core.drr_sources import portable_source_name, resolve_source_path
from core.provenance import _digest_signature
from ui_qt.drr_region_backgrounds import frame_method


def background_dependencies(folder, files):
    """Cheap filesystem signatures, with no spectrum or JSON reads on redraw."""
    paths = []
    for name in dict.fromkeys(files):
        path = resolve_source_path(folder, name)
        paths.extend((path, path.with_suffix('.metadata.json')))
    return tuple((str(path), _digest_signature(path)) for path in dict.fromkeys(paths))


def edit_composite_background(owner):
    from ui_qt.drr_composite_background_dialog import DrrCompositeBackgroundDialog

    folder = owner.current_folder
    if not folder:
        owner._show_error('Open an experiment folder before combining backgrounds.')
        return
    selected = tuple(owner.drr_selected_files)
    files = tuple(owner.drr_baseline_files_manual)
    which = frame_method(owner)
    regions = tuple(getattr(owner, '_drr_background_regions', ()))
    assignments = tuple(owner._drr_assignments)
    if owner._drr_assignments_automatic and assignments:
        recipes = {(a.baseline_mode, a.baseline_files, a.baseline_which, a.regions) for a in assignments}
        if len(recipes) == 1 and assignments[0].baseline_mode == 'External':
            files, which, regions = assignments[0].baseline_files, assignments[0].baseline_which, assignments[0].regions
        else:
            files, regions = (), ()

    def select_files(current):
        return owner.drr_controller._open_drr_source_dialog(
            title='Choose Sources to Combine', selected=list(current),
            baseline_mode=True, independent_background=True)

    dialog = DrrCompositeBackgroundDialog(
        owner, folder=folder, initial_files=files, initial_which=which,
        initial_regions=regions, select_files=select_files, pool=owner.thread_pool)
    try:
        if dialog.exec() != QDialog.Accepted or not dialog.saved_path:
            return
        path = Path(dialog.saved_path)
        if owner.current_folder != folder or tuple(owner.drr_selected_files) != selected:
            owner._status(f'Composite background saved: {path}. Select it in External to use it.')
            return
        name = portable_source_name(folder, path)
        owner.drr_controller._cancel_auto_external()
        owner._drr_background_regions = ()
        owner._drr_assignments = ()
        owner._drr_assignments_automatic = False
        owner._drr_baseline_user_selected = True
        owner._drr_external_auto_selected = False
        owner._drr_background_guess = None
        owner.drr_baseline_files_manual = [name]
        owner.drr_baseline_files_found = []
        for widget, value in ((owner.drr_baseline_combo, 'External'),
                              (owner.drr_baseline_combine_combo, 'Last frame from each file, then average')):
            blocked = widget.blockSignals(True)
            widget.setCurrentText(value)
            widget.blockSignals(blocked)
        blocked = owner.drr_pin_baseline_chk.blockSignals(True)
        owner.drr_pin_baseline_chk.setChecked(True)
        owner.drr_pin_baseline_chk.blockSignals(blocked)
        owner.drr_controller._update_drr_selection_labels()
        owner._queue_drr_catalog_refresh(
            auto=True, old_source_files={s.source for s in owner.drr_available_sources},
            selected_sources=(*selected, name), force=True, catalog_only=True)
        owner._status(f'Composite background saved and selected: {path.name}. Keep background is enabled.')
        if selected and all(Path(name).suffix.lower() == '.csv' for name in selected):
            owner._start_load('DRR')
    finally:
        dialog.deleteLater()
        owner.drr_composite_background_btn.setFocus()
