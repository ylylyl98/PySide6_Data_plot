"""External-only background resolution using saved recipes and picker ranking."""
from dataclasses import replace
from pathlib import Path
from threading import Event

from core.drr_baseline_candidates import rank_baseline_candidates, spectral_grids_match
from core.drr_measured_bn import MeasuredBnMatcher, is_measured_bn_name
from core.drr_pn_store import DrrPnStore
from core.drr_bn_store import DrrBnStore, is_bn_calibration_source
from core.drr_sources import (
    DrrBackgroundResolution, DrrMeasurementAssignment, DrrSource,
    discover_drr_sources, preview_drr_sources, group_drr_sources,
    inspect_csv_spectral_grid, inspect_csv_gate_profile,
    resolve_source_path,
    resolve_drr_background_assignments, portable_source_name,
    _read_drr_metadata,
)


def resolve_auto_external(root, catalog, selected, *, measured_bn_only=False,
                          cancelled=None, progress=None, log=None):
    """Run in a worker; never substitute Self for an explicit External request."""
    cancelled = cancelled or Event()
    selected = tuple(dict.fromkeys(selected))
    sources = {item.source: item for item in catalog}
    if not selected or cancelled.is_set():
        return DrrBackgroundResolution(reason='No measurements selected or matching cancelled.')
    if measured_bn_only:
        return _resolve_measured_bn_only(root, sources, selected, cancelled)
    from core.drr_saved_background import resolve_saved_group_background
    exported = resolve_saved_group_background(root, catalog, selected, progress=progress, log=log)
    if cancelled.is_set():
        return DrrBackgroundResolution(reason='Background matching cancelled.')
    # Reopening an exported group uses the recipe that produced its saved
    # result, ahead of a later applied-but-unsaved P/N or BN choice. Keep this
    # generic External request external; typed group restoration also handles
    # explicitly saved self-reference recipes in its separate workflow.
    bn_store = DrrBnStore(root)
    try:
        if exported is not None and (not exported.resolved or all(
                item.baseline_mode == 'External' for item in exported.assignments)):
            calibrated_export = exported.resolved and any(
                is_bn_calibration_source(root, name)
                for item in exported.assignments for name in item.baseline_source_files)
            # An original-BN choice also revokes calibrated member recipes in
            # an exact-group export. Let the restoration checks below recover
            # its measured reference; ordinary saved backgrounds keep priority.
            if not calibrated_export or bn_store.selection_state(selected) != 'original':
                return exported
        applied_bn = bn_store.find_applied(selected)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return DrrBackgroundResolution(reason=f'Saved group BN calibration needs review: {exc}',
                                       unresolved_measurements=selected)
    if cancelled.is_set():
        return DrrBackgroundResolution(reason='Background matching cancelled.')
    if applied_bn is not None:
        name = portable_source_name(root, applied_bn.csv_path)
        return DrrBackgroundResolution(
            assignments=tuple(DrrMeasurementAssignment(
                measurement_file=measurement, baseline_mode='External', baseline_files=(name,), baseline_which='last',
                selection_reason=f'Validated exact-group BN calibration {applied_bn.version_id}',
            ) for measurement in selected), numerical_path='common',
            reason='Restored validated exact-group BN calibration')
    # Explicit restoration is durable even without an exported DRR recipe or
    # when the superseded generated CSV is unavailable. Merely forgetting a BN
    # calibration for a new manual/PN selection does not set this marker.
    try:
        original = bn_store.original_recipe(selected)
        if original is not None:
            from core.drr_composite_background import is_composite_background
            selected_paths = {resolve_source_path(root, name) for name in selected}
            def original_source(name):
                path = resolve_source_path(root, name)
                if path.suffix.lower() != '.csv' or not path.is_file():
                    raise ValueError(f'Original source is unavailable: {name}')
                return DrrSource(name, path.name, '', '', 0, False,
                                 spectral_grid=tuple(inspect_csv_spectral_grid(path)))
            backgrounds = []
            for name in original['baseline_files']:
                path = resolve_source_path(root, name)
                if (path in selected_paths or is_bn_calibration_source(root, name)
                        or is_composite_background(path)):
                    raise ValueError('Original BN restoration requires independent measured reference files.')
                backgrounds.append(original_source(name))
            for name in selected:
                if cancelled.is_set():
                    return DrrBackgroundResolution(reason='Background matching cancelled.')
                measurement = original_source(name)
                if not all(spectral_grids_match(measurement, background) for background in backgrounds):
                    raise ValueError(f'Original BN spectral grid is incompatible with {Path(name).name}.')
            return DrrBackgroundResolution(
                assignments=tuple(DrrMeasurementAssignment(
                    measurement_file=name, baseline_mode='External',
                    baseline_files=original['baseline_files'], baseline_which=original['baseline_which'],
                    selection_reason='Explicit original-BN restoration for this exact group',
                ) for name in selected), numerical_path='common', reason='Restored explicitly selected original BN')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return DrrBackgroundResolution(reason=f'Saved original BN reference needs review: {exc}',
                                       unresolved_measurements=selected)
    # An explicitly applied P/N version precedes saved recipe guesses. The
    # store checks measurement content, version freshness and native grid.
    # A known but invalid association must stop, not silently select another
    # background through the recommendation fallback.
    pn_store = DrrPnStore(root)
    try:
        applied = pn_store.find_applied(selected)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return DrrBackgroundResolution(
            reason=f'Saved P/N background needs review: {exc}',
            unresolved_measurements=selected,
        )
    if cancelled.is_set():
        return DrrBackgroundResolution(reason='Background matching cancelled.')
    if applied is not None:
        if not applied.fresh:
            return DrrBackgroundResolution(
                reason=f'Saved P/N background is {applied.status}; select a version explicitly.',
                unresolved_measurements=selected,
            )
        name = portable_source_name(root, applied.csv_path)
        return DrrBackgroundResolution(
            assignments=tuple(DrrMeasurementAssignment(
                measurement_file=measurement, baseline_mode='External',
                baseline_files=(name,), baseline_which='last',
                selection_reason=f'Explicitly applied P/N version {applied.version_id}',
            ) for measurement in selected),
            numerical_path='common', reason='Restored explicitly applied P/N background',
        )
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
        # The first inventory shown while opening a folder has filenames but
        # no gate/frame metadata yet. Complete relevant files in this worker
        # so early selections use the same matching policy as the full catalog.
        if ((name in selected or is_measured_bn_name(item.filename))
                and (item.gate_varies is None or item.frame_count is None)):
            try:
                profile = inspect_csv_gate_profile(path, max_rows=None)
                item = replace(item, gate_varies=profile.varies, frame_count=profile.frame_count,
                               gate_grid=profile.gate_grid, spectral_grid=profile.spectral_grid,
                               gate_direction=profile.gate_direction, gate_ranges=profile.gate_ranges,
                               gate_labels=profile.gate_labels, grid_complete=profile.grid_complete)
                sources[name] = item
            except (OSError, ValueError, TypeError):
                return None
        if len(item.spectral_grid) < 2:
            try:
                item = replace(item, spectral_grid=tuple(inspect_csv_spectral_grid(path)))
            except (OSError, ValueError, TypeError):
                return None
            sources[name] = item
        return item

    selected_paths = {source_path(name) for name in selected}
    candidates = None
    measured_bn = None

    _, saved_recipes = _read_drr_metadata(Path(root), require_drr_operation=True)
    recipes_by_measurement = {}
    for recipe in saved_recipes:
        for name in recipe.measurement_files:
            recipes_by_measurement.setdefault(source_path(name), []).append(recipe)
    assignments = []
    pending = []
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
        if not saved.resolved and not saved.self_fallback_allowed:
            return saved
        if saved.resolved:
            assignment = saved.assignments[0]
            # A historical member recipe must not carry a group calibration
            # into a subset, new combination, or an explicitly reverted group.
            # A valid active exact-group version was already restored above.
            if any(is_bn_calibration_source(root, path) for path in assignment.baseline_source_files):
                original_assignment = None
                if bn_store.selection_state(selected) == 'original' and len(assignment.baseline_source_files) == 1:
                    try:
                        previous = bn_store.version_for_csv(assignment.baseline_source_files[0])
                        expected = {source_path(v['path']) for v in previous.source_identity['measurements']}
                        if expected == selected_paths:
                            original_assignment = DrrMeasurementAssignment(
                                measurement_file=name, baseline_mode='External',
                                baseline_files=tuple(v['path'] for v in previous.source_identity['baseline_files']),
                                baseline_which=previous.source_identity['baseline_which'],
                                selection_reason='Explicit original-BN choice; saved calibrated member recipe was not reused')
                    except (OSError, ValueError, KeyError, TypeError):
                        # A damaged generated artifact is never needed to use
                        # a separately chosen ordinary measured reference.
                        pass
                assignment = original_assignment
            if assignment is None:
                backgrounds = []
            else:
                backgrounds = [inspected(path) for path in assignment.baseline_source_files]
            if assignment is not None and assignment.baseline_mode == 'External' and backgrounds and all(
                item is not None
                and source_path(item.source) not in selected_paths
                and spectral_grids_match(measurement, item) for item in backgrounds
            ):
                assignments.append(assignment)
                continue
            if assignment is not None and assignment.regions:
                return DrrBackgroundResolution(
                    reason=f'Saved background by X range is no longer compatible with {Path(name).name}. '
                           'Restore its source files or explicitly select a new background recipe.',
                    unresolved_measurements=selected)
        pending.append((name, measurement))
    for name, measurement in pending:
        if cancelled.is_set():
            return DrrBackgroundResolution(reason='Background matching cancelled.')
        if candidates is None:
            candidates = []
            for candidate_name in tuple(sources):
                if cancelled.is_set():
                    return DrrBackgroundResolution(reason='Background matching cancelled.')
                # A managed P/N result belongs to its explicit sample association.
                # A matching grid or filename is not permission to reuse it for
                # a new group; history remains available for manual selection.
                if source_path(candidate_name).is_relative_to(pn_store.directory):
                    continue
                if is_bn_calibration_source(root, candidate_name):
                    continue
                if source_path(candidate_name) not in selected_paths:
                    item = inspected(candidate_name)
                    if item is not None:
                        candidates.append(item)
            # Saved recipes need no new time inference. Missing capture logs
            # for those members must not downgrade the remaining members.
            measurements = tuple(item for _, item in pending)
            measured_bn = MeasuredBnMatcher(root, measurements, candidates, cancelled=cancelled)
        matched = measured_bn.match(measurement)
        if matched.assignment is not None:
            assignments.append(matched.assignment)
            continue
        if matched.applicable:
            return DrrBackgroundResolution(
                reason=matched.reason, unresolved_measurements=selected)
        # A measured BN that failed strict acquisition matching must not return
        # through the generic picker's intentionally broader ranking policy.
        ranked = rank_baseline_candidates(
            [measurement], [item for item in candidates if not is_measured_bn_name(item.filename)])
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
    by_measurement = {item.measurement_file: item for item in assignments}
    assignments = [by_measurement[name] for name in selected]
    signatures = {(item.baseline_files, item.baseline_which, item.regions) for item in assignments}
    return DrrBackgroundResolution(
        assignments=tuple(assignments),
        numerical_path='common' if len(signatures) == 1 else 'heterogeneous',
        reason='Automatic External background selection',
    )


def _resolve_measured_bn_only(root, sources, selected, cancelled):
    """An explicit BN button never restores P/N, fitted or calibrated recipes.

    Read only selected measurements and named BN references. The ordinary
    External path above still preserves saved associations and their checks.
    This request changes no persistent background association.
    """
    from core.drr_composite_background import is_composite_background

    if any(name not in sources for name in selected):
        sources.update((item.source, item) for item in preview_drr_sources(root, include_all=True))
    paths = {}
    def source_path(name):
        if name not in paths:
            paths[name] = resolve_source_path(root, name)
        return paths[name]

    def inspect(name):
        path = source_path(name)
        if path.suffix.lower() != '.csv' or not path.is_file():
            return None
        item = sources.get(name, DrrSource(name, path.name, '', '', 0, False))
        if item.gate_varies is None or item.frame_count is None or len(item.spectral_grid) < 2:
            profile = inspect_csv_gate_profile(path, max_rows=None)
            item = replace(item, gate_varies=profile.varies, frame_count=profile.frame_count,
                           spectral_grid=profile.spectral_grid)
        return item

    def failure(reason):
        return DrrBackgroundResolution(reason=reason, unresolved_measurements=selected)

    measurements = []
    for name in selected:
        if cancelled.is_set():
            return failure('Measured-BN matching cancelled.')
        item = inspect(name)
        if item is None:
            return failure(f'Cannot inspect measurement: {name}')
        measurements.append(item)
    selected_paths = {source_path(name) for name in selected}
    candidates = []
    for name, item in sources.items():
        if cancelled.is_set():
            return failure('Measured-BN matching cancelled.')
        if not is_measured_bn_name(item.filename):
            continue
        path = source_path(name)
        if (path in selected_paths or is_bn_calibration_source(root, name)
                or is_composite_background(path)):
            continue
        candidate = inspect(name)
        if candidate is not None:
            candidates.append(candidate)
    matcher = MeasuredBnMatcher(root, measurements, candidates, cancelled=cancelled)
    assignments = []
    for measurement in measurements:
        matched = matcher.match(measurement)
        if matched.assignment is None:
            return failure(f'No matching measured BN for {measurement.filename}. {matched.reason} '
                           'Choose BN files manually, or use Build P/N for a P/N background.')
        assignments.append(matched.assignment)
    signatures = {(item.baseline_files, item.baseline_which) for item in assignments}
    return DrrBackgroundResolution(
        assignments=tuple(assignments),
        numerical_path='common' if len(signatures) == 1 else 'heterogeneous',
        reason='Matched measured BN references for every selected measurement')
