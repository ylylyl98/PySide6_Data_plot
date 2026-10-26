"""Restore the background recorded by an exact DRR group export."""
from pathlib import Path
from typing import Sequence

import numpy as np

from core.drr_composite_background import is_composite_background, read_composite_background
from core.drr_sources import (
    DrrBackgroundResolution, DrrSource, _read_drr_metadata,
    inspect_csv_spectral_grid, resolve_drr_background_assignments, resolve_source_path,
)
from core.load_cancellation import check_load_cancelled


def resolve_saved_group_background(
    folder: str | Path,
    catalog: Sequence[DrrSource],
    selected: Sequence[str],
    *,
    progress=None,
    log=None,
    saved_recipes=None,
) -> DrrBackgroundResolution | None:
    """Return the latest exact-group export recipe, or None without history.

    Run on a load/matching worker. An unusable latest export remains unresolved;
    it must not silently choose an older recipe or a new background candidate.
    Frozen composites are verified from their CSV and editing snapshots, so
    changes to their original P/N inputs or generation code do not reject the
    numerical background actually used by the export.
    """
    progress = getattr(progress, 'emit', progress)
    log = getattr(log, 'emit', log)
    names = tuple(dict.fromkeys(str(name) for name in selected if str(name).strip()))
    if not names or any(Path(name).suffix.casefold() != '.csv' for name in names):
        return None
    check_load_cancelled()
    root = Path(folder).resolve()

    def identity(name):
        return str(resolve_source_path(root, name)).casefold()

    wanted = {identity(name) for name in names}
    if saved_recipes is None:
        _, records = _read_drr_metadata(root, require_drr_operation=True)
    else:
        # A caller restoring other recipe types can share this exact history
        # snapshot, avoiding a second scan and a mid-load change of recipe.
        records = tuple(saved_recipes)
    matches = [record for record in records
               if {identity(name) for name in record.measurement_files} == wanted]
    if not matches:
        return None
    latest_time = max(record.saved_time for record in matches)
    latest = tuple(record for record in matches if record.saved_time == latest_time)
    if log is not None:
        log('Restoring the latest saved DRR background for this measurement group.')
    resolved = resolve_drr_background_assignments(
        root, catalog, names, allow_guess=False, saved_recipes=latest)
    if not resolved.resolved:
        return resolved

    grids = {}

    def grid(name):
        path = resolve_source_path(root, name)
        if path not in grids:
            check_load_cancelled()
            if not path.is_file():
                raise ValueError(f'Saved DRR source is unavailable: {name}')
            grids[path] = np.asarray(inspect_csv_spectral_grid(path), dtype=float)
            if is_composite_background(path):
                read_composite_background(path)
        return grids[path]

    def same_grid(first, second):
        if first.shape != second.shape or not np.allclose(first, second, rtol=1e-9, atol=1e-10):
            raise ValueError('Saved DRR measurements and backgrounds must use one exact native spectral '
                             'grid (same point count, order, and wavelength values).')

    try:
        # Ordinary heterogeneous recipes may use different grids per member;
        # region recipes follow the loader's shared-grid requirement.
        shared_grid = grid(names[0]) if any(item.regions for item in resolved.assignments) else None
        for index, assignment in enumerate(resolved.assignments):
            check_load_cancelled()
            reference = grid(assignment.measurement_file)
            if shared_grid is not None:
                same_grid(shared_grid, reference)
            for name in assignment.baseline_source_files:
                same_grid(reference, grid(name))
            if progress is not None:
                progress(round(100 * (index + 1) / len(resolved.assignments)))
    except (OSError, ValueError, TypeError) as exc:
        return DrrBackgroundResolution(
            reason=f'The latest saved DRR background cannot be restored: {exc}',
            unresolved_measurements=names,
        )
    check_load_cancelled()
    return resolved
