"""External-only background resolution using saved recipes and picker ranking."""
from dataclasses import replace
from pathlib import Path
from threading import Event

from core.drr_baseline_candidates import rank_baseline_candidates, spectral_grids_match
from core.drr_sources import (
    DrrBackgroundResolution, DrrMeasurementAssignment, DrrSource,
    discover_drr_sources, group_drr_sources, inspect_csv_spectral_grid, resolve_source_path,
    resolve_drr_background_assignments,
    _read_drr_metadata,
)


def resolve_auto_external(root, catalog, selected, *, cancelled=None, progress=None, log=None):
    """Run in a worker; never substitute Self for an explicit External request."""
    cancelled = cancelled or Event()
    selected = tuple(dict.fromkeys(selected))
    sources = {item.source: item for item in catalog}
    if not selected or cancelled.is_set():
        return DrrBackgroundResolution(reason='No measurements selected or matching cancelled.')
    if any(name not in sources for name in selected):
        sources.update((item.source, item) for item in discover_drr_sources(root, include_all=True))

    paths = {}
    def source_path(name):
        if name not in paths:
            paths[name] = resolve_source_path(root, name)
        return paths[name]

    def inspected(name):
        path = source_path(name)
        if path.suffix.lower() != '.csv' or not path.is_file():
            return None
        item = sources.get(name)
        if item is None:
            item = DrrSource(name, path.name, '', '', 0, False)
        if len(item.spectral_grid) < 2:
            try:
                item = replace(item, spectral_grid=tuple(inspect_csv_spectral_grid(path)))
            except (OSError, ValueError, TypeError):
                return None
            sources[name] = item
        return item

    selected_paths = {source_path(name) for name in selected}
    candidates = None

    _, saved_recipes = _read_drr_metadata(Path(root), require_drr_operation=True)
    recipes_by_measurement = {}
    for recipe in saved_recipes:
        for name in recipe.measurement_files:
            recipes_by_measurement.setdefault(source_path(name), []).append(recipe)
    assignments = []
    for name in selected:
        if cancelled.is_set():
            return DrrBackgroundResolution(reason='Background matching cancelled.')
        measurement = inspected(name)
        if measurement is None:
            return DrrBackgroundResolution(reason=f'Cannot inspect measurement: {name}',
                                           unresolved_measurements=selected)
        saved = resolve_drr_background_assignments(
            root, tuple(sources.values()), [name], allow_guess=False,
            saved_recipes=recipes_by_measurement.get(source_path(name), ()))
        if saved.resolved:
            assignment = saved.assignments[0]
            backgrounds = [inspected(path) for path in assignment.baseline_files]
            if assignment.baseline_mode == 'External' and backgrounds and all(
                item is not None
                and source_path(item.source) not in selected_paths
                and spectral_grids_match(measurement, item) for item in backgrounds
            ):
                assignments.append(assignment)
                continue
        if candidates is None:
            candidates = []
            for candidate_name in tuple(sources):
                if cancelled.is_set():
                    return DrrBackgroundResolution(reason='Background matching cancelled.')
                if source_path(candidate_name) not in selected_paths:
                    item = inspected(candidate_name)
                    if item is not None:
                        candidates.append(item)
        ranked = rank_baseline_candidates([measurement], candidates)
        if not ranked:
            return DrrBackgroundResolution(
                reason=f'No compatible external background for {Path(name).name}. Choose a background manually.',
                unresolved_measurements=selected,
            )
        best = ranked[0]
        group = next(
            group for group in group_drr_sources([item.source for item in ranked])
            if any(item.source == best.source.source for item in group.files)
        )
        assignments.append(DrrMeasurementAssignment(
            measurement_file=name, baseline_mode='External',
            baseline_files=tuple(item.source for item in group.files),
            baseline_which='all' if all(item.gate_varies is False for item in group.files) else 'last',
            selection_reason=(f'Top compatible recommendation group ({len(group.files)} files): '
                              + best.reason),
        ))
    signatures = {(item.baseline_files, item.baseline_which) for item in assignments}
    return DrrBackgroundResolution(
        assignments=tuple(assignments),
        numerical_path='common' if len(signatures) == 1 else 'heterogeneous',
        reason='Automatic External background selection',
    )
