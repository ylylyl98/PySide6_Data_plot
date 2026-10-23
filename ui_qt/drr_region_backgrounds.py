"""Connect the external-background region editor to the DRR workflow."""
from dataclasses import replace

import numpy as np
from PySide6.QtWidgets import QDialog

from core.drr_sources import DrrMeasurementAssignment


def source_files(assignments):
    return list(dict.fromkeys(name for item in assignments for name in item.baseline_source_files))


def restore_regions(owner, assignments):
    """The manual controls describe one common recipe, never an averaged union."""
    recipes = {(item.baseline_mode, item.baseline_files, item.baseline_which, item.regions)
               for item in assignments}
    owner._drr_background_regions = ()
    if len(recipes) == 1 and assignments[0].baseline_mode == 'External':
        owner._drr_background_regions = assignments[0].regions
        owner.drr_baseline_files_manual = list(assignments[0].baseline_files)


def frame_method(owner):
    return {
        'First frame from each file, then average': 'first',
        'Last frame from each file, then average': 'last',
        'Average all frames in each file, then average files': 'all',
    }.get(owner.drr_baseline_combine_combo.currentText(), 'last')


def edit_regions(owner):
    from ui_qt.drr_background_regions_dialog import DrrBackgroundRegionsDialog

    selected = tuple(owner.drr_selected_files)
    folder = owner.current_folder
    if not selected or owner.drr_baseline_combo.currentText() != 'External':
        owner._show_error('Select CSV measurements and an external default background first.')
        return
    assignments = tuple(owner._drr_assignments)
    if owner._drr_assignments_automatic and assignments:
        recipes = {(item.baseline_mode, item.baseline_files, item.baseline_which, item.regions)
                   for item in assignments}
        if len(recipes) != 1 or assignments[0].baseline_mode != 'External':
            owner._show_error('These measurements use different background recipes. '
                              'Select one common external background before editing X ranges.')
            return
        template = assignments[0]
    else:
        try:
            template = DrrMeasurementAssignment(
                selected[0], 'External', tuple(owner.drr_baseline_files_manual), frame_method(owner),
                regions=tuple(getattr(owner, '_drr_background_regions', ())))
        except ValueError as exc:
            owner._show_error(str(exc))
            return
    y_axis = owner._selected_y_axis_spec('drr')
    current = owner.loaded
    energy = (np.asarray(current.cube.energy, float).copy()
              if current is not None and current.mode == 'DRR' and current.cube is not None
              and current.folder == folder and tuple(current.selected_files) == selected else None)

    def preview(regions):
        # Executed by the dialog's worker. Freeze sources and recipe before launch.
        from core import data_io, processing_run
        from core.loader import build_effective_background
        recipe = replace(template, measurement_file=selected[0], regions=regions)
        data_io._validate_drr_external_spectral_grids(folder, selected, recipe.baseline_source_files)
        axis = np.asarray(processing_run._load_canonical(folder, selected[0], y_axis=y_axis)['energy'], float)
        return axis, build_effective_background(folder, recipe, axis, y_axis=y_axis)

    def select_files(files):
        return owner.drr_controller._open_drr_source_dialog(
            title='Choose Background for X Range', selected=list(files), baseline_mode=True)

    dialog = DrrBackgroundRegionsDialog(
        owner, energy=energy, regions=template.regions, default_files=template.baseline_files,
        default_which=template.baseline_which, select_files=select_files,
        preview=preview, pool=owner.thread_pool)
    try:
        if dialog.exec() != QDialog.Accepted:
            return
        if (owner.current_folder != folder or tuple(owner.drr_selected_files) != selected
                or owner.drr_baseline_combo.currentText() != 'External'):
            owner._show_error('Measurement selection changed. Reopen Background by X range to apply the recipe.')
            return
        regions = tuple(dialog.regions())
        owner.drr_controller._cancel_auto_external()
        owner._drr_background_regions = regions
        owner.drr_baseline_files_manual = list(template.baseline_files)
        owner._drr_assignments = tuple(replace(
            template, measurement_file=name, regions=regions, source_recipe='',
            selection_reason='explicit background by X range') for name in selected)
        owner._drr_assignments_automatic = False
        owner._drr_baseline_user_selected = True
        owner._drr_background_guess = None
        owner.drr_controller._update_drr_selection_labels()
        owner._start_load('DRR')
    finally:
        dialog.deleteLater()
