"""Reviewable assignments from existing P, N and frozen P/N backgrounds.

No generation, averaging across candidates, or persistent writes happen here.
Composite timing comes only from unchanged raw inputs, never its save time.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path

from core.drr_baseline_candidates import spectral_grids_match
from core.drr_match_conditions import normalize_relaxed_conditions
from core.drr_composite_background import (
    composite_metadata_path, is_composite_background, read_composite_background,
)
from core.drr_pn_selection import build_pn_source_catalog, _compare
from core.drr_pn_pairing import detect_pn_role
from core.drr_sources import (
    DrrSource, DrrBackgroundResolution, DrrMeasurementAssignment, drr_source_paths,
    extract_wavelength_center_nm, inspect_csv_gate_profile, portable_source_name,
    resolve_source_path,
)
from core.load_cancellation import LoadCancelled, check_load_cancelled
from core.provenance import _digest_signature, sha256_file


_PLAN = object()
_CONFIRMATION = object()
_BATCH_CONFLICT_WARNING = ('Verified capture batch: individual suggestions and joint timing lack '
                           'an unambiguous consensus; review each assignment.')

# (candidate name, (relaxed-difference tier, unknown count, time gap or unavailable))
_BatchScore = tuple[str, tuple[int, int, float | None]]
# (member names, per-member candidate scores)
_BatchInput = tuple[tuple[str, ...], tuple[tuple[str, tuple[_BatchScore, ...]], ...]]
# (member names, timing-consensus status, joint candidate gaps)
_BatchResult = tuple[tuple[str, ...], str, tuple[tuple[str, float], ...]]


def _check(cancelled=None):
    check_load_cancelled()
    if cancelled is not None:
        if hasattr(cancelled, 'check'):
            cancelled.check()
        elif cancelled.is_set() if hasattr(cancelled, 'is_set') else cancelled():
            raise LoadCancelled()


@dataclass(frozen=True)
class PnMatchRow:
    measurement_file: str
    assignment: DrrMeasurementAssignment | None = None
    evidence: str = ''
    warnings: tuple[str, ...] = ()
    candidate_files: tuple[str, ...] = ()
    time_gap_seconds: float | None = None
    timing_basis: str = ''
    # Parallel immutable candidate details let a manual override stay I/O-free.
    candidate_details: tuple[tuple[str, str, tuple[str, ...], float | None, str], ...] = ()

    @property
    def baseline_file(self):
        return self.assignment.baseline_files[0] if self.assignment else ''


@dataclass(frozen=True)
class PnMatchPlan:
    root: str
    selected: tuple[str, ...]
    kind: str
    which: str
    candidate_files: tuple[str, ...]
    automatic_candidates: bool
    rows: tuple[PnMatchRow, ...]
    resolution: DrrBackgroundResolution
    warnings: tuple[str, ...] = ()
    snapshot: tuple[tuple[str, tuple[int, ...] | None], ...] = ()
    directory_snapshot: tuple[tuple[str, tuple[int, ...] | None], ...] = ()
    signature: str = ''
    inventory_warnings: tuple[str, ...] = ()
    relaxed_conditions: tuple[str, ...] = ()
    batch_inputs: tuple[_BatchInput, ...] = ()
    batches: tuple[_BatchResult, ...] = ()
    _verified: object = field(default=None, repr=False, compare=False)

    @property
    def ready(self):
        return bool(self._verified is _PLAN and self.selected and self.snapshot
                    and self.kind in ('p', 'n', 'pn') and self.which in ('first', 'last', 'all')
                    and len(set(self.selected)) == len(self.selected)
                    and tuple(row.measurement_file for row in self.rows) == self.selected
                    and all(row.assignment is not None and row.baseline_file in row.candidate_files
                            for row in self.rows)
                    and self.resolution.resolved
                    and tuple(row.assignment for row in self.rows) == self.resolution.assignments)

    @property
    def requires_acknowledgement(self):
        return bool(self.warnings)


@dataclass(frozen=True)
class PreparedPnConfirmation:
    plan: PnMatchPlan
    signature: str
    warnings_acknowledged: bool = False
    _verified: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _Candidate:
    name: str
    source: DrrSource
    origins: tuple[DrrSource, ...]
    timed_origins: tuple[DrrSource, ...]
    warnings: tuple[str, ...] = ()


def _signature(plan):
    payload = dict(root=plan.root, selected=plan.selected, kind=plan.kind, which=plan.which,
        candidate_files=plan.candidate_files, automatic_candidates=plan.automatic_candidates,
        rows=[dict(measurement=row.measurement_file,
                   assignment=row.assignment.to_dict() if row.assignment else None,
                   evidence=row.evidence, warnings=row.warnings, candidates=row.candidate_files,
                   details=row.candidate_details, gap=row.time_gap_seconds, basis=row.timing_basis)
              for row in plan.rows], warnings=plan.warnings, snapshot=plan.snapshot,
        directories=plan.directory_snapshot, inventory_warnings=plan.inventory_warnings,
        relaxed_conditions=plan.relaxed_conditions, batch_inputs=plan.batch_inputs, batches=plan.batches,
        resolution=plan.resolution.to_dict())
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finish(plan, rows):
    rows = [replace(row, warnings=tuple(w for w in row.warnings if w != _BATCH_CONFLICT_WARNING)) for row in rows]
    row_map = {row.measurement_file: row for row in rows}
    batches = []
    for members, scores in plan.batch_inputs:
        score_map = {name: dict(values) for name, values in scores}
        shared = set(score_map[members[0]])
        for name in members[1:]:
            shared.intersection_update(score_map[name])
        # Preserve per-file condition priority before comparing joint timing.
        shared = {candidate for candidate in shared if all(
            score_map[name][candidate][:2] == min(score[:2] for score in score_map[name].values())
            for name in members)}
        gaps = sorted((sum(score_map[name][candidate][2] for name in members), candidate)
                      for candidate in shared if all(score_map[name][candidate][2] is not None for name in members))
        chosen = {row_map[name].baseline_file for name in members}
        ambiguous = len(gaps) > 1 and math.isclose(gaps[0][0], gaps[1][0], abs_tol=1e-6, rel_tol=0)
        conflict = ('' in chosen or len(chosen) != 1 or not gaps or ambiguous or next(iter(chosen)) != gaps[0][1])
        # Consensus describes timing only; existing per-row review warnings remain.
        status = 'conflict' if conflict else 'consensus'
        batches.append((members, status, tuple((candidate, gap) for gap, candidate in gaps)))
        if conflict:
            for name in members:
                row_map[name] = replace(row_map[name], warnings=(*row_map[name].warnings, _BATCH_CONFLICT_WARNING))
    rows = [row_map[row.measurement_file] for row in rows]
    plan = replace(plan, batches=tuple(batches))
    assignments = tuple(row.assignment for row in rows if row.assignment is not None)
    unresolved = tuple(row.measurement_file for row in rows if row.assignment is None)
    recipes = {(item.baseline_files, item.baseline_which) for item in assignments}
    resolution = DrrBackgroundResolution(assignments,
        'unresolved' if unresolved or not assignments else 'common' if len(recipes) == 1 else 'heterogeneous',
        'Review existing P/N background assignments for every measurement.', unresolved)
    warnings = tuple(dict.fromkeys((*plan.inventory_warnings,
        *(warning for row in rows for warning in row.warnings))))
    plan = replace(plan, rows=tuple(rows), resolution=resolution, warnings=warnings)
    return replace(plan, signature=_signature(plan))


def _inventory(root, kind, candidate_files=None, cancelled=None):
    if candidate_files is None:
        paths = list(drr_source_paths(root, include_all=True))
        store = root / 'Processed Data' / 'DRR' / 'PN Backgrounds'
        if kind == 'pn' and store.is_dir():
            paths.extend(store.glob('*.csv'))
    else:
        paths = [resolve_source_path(root, name) for name in candidate_files]
    result = {}
    for path in paths:
        _check(cancelled)
        path = path.resolve()
        if path.suffix.casefold() != '.csv':
            continue
        # For raw roles keep anonymous background names for gate-role inference.
        if candidate_files is None and kind != 'pn':
            from core.drr_measured_bn import is_measured_bn_name
            role = detect_pn_role(path.name)
            if (role is not None and role != kind.upper()) or is_measured_bn_name(path.name):
                continue
            stem = path.stem.casefold()
            if role != kind.upper() and 'back' not in stem and '_bg' not in stem:
                continue
            if is_composite_background(path):
                continue
        if candidate_files is None and kind == 'pn' and not is_composite_background(path):
            continue
        result[str(path).casefold()] = portable_source_name(root, path)
    return tuple(result[key] for key in sorted(result))


def _directories(root, paths, *, include_store=False):
    directories = {root, root / 'Initial Data'}
    if include_store:
        directories.add(root / 'Processed Data' / 'DRR' / 'PN Backgrounds')
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and child.name.casefold() == 'initial data':
                directories.add(child)
                directories.update(path for path in child.rglob('*') if path.is_dir())
    for path in paths:
        directories.add(path.parent)
        if path.parent.parent.is_relative_to(root) and path.parent.name.casefold() != 'pn backgrounds':
            directories.add(path.parent.parent)
    return tuple(sorted(directories, key=str))


def _stats(paths):
    return tuple((str(path), _digest_signature(path)) for path in sorted(set(paths), key=str))


def _unchanged(snapshot):
    return all(_digest_signature(Path(path)) == stamp for path, stamp in snapshot)


def _inspect(root, name):
    path = resolve_source_path(root, name)
    profile = inspect_csv_gate_profile(path)
    source = DrrSource(name, path.name, '', '', path.stat().st_mtime, False,
        gate_varies=profile.varies, frame_count=profile.frame_count,
        spectral_grid=tuple(profile.spectral_grid), gate_grid=profile.gate_grid,
        gate_labels=profile.gate_labels, grid_complete=profile.grid_complete,
        wavelength_center_nm=extract_wavelength_center_nm(path.name))
    if not spectral_grids_match(source, source) or not profile.frame_count:
        raise ValueError('No verifiable native spectral grid or frames.')
    return source


def _composite(root, name, paths, before, inspect, cancelled):
    path = resolve_source_path(root, name)
    metadata = composite_metadata_path(path)
    paths.add(metadata)
    before.setdefault(str(metadata), _digest_signature(metadata))
    document = read_composite_background(path)
    effective = {document.default_source, *(region.source_key for region in document.regions)}
    sources = [source for source in document.sources if source.key in effective]
    bindings = {}
    for source in sources:
        for raw, digest in source.source_hashes:
            resolved = resolve_source_path(root, raw)
            if resolved in bindings and bindings[resolved] != digest:
                raise ValueError('Composite input hashes disagree.')
            bindings[resolved] = digest
    # Unused P/N snapshots still establish roles when a generated source names
    # its inputs indirectly, as in imported historical composites.
    origin_names = {resolve_source_path(root, raw) for source in sources for raw in source.files}
    if not origin_names:
        raise ValueError('Saved composite has no P/N source provenance.')
    roles = {detect_pn_role(raw.name) for raw in origin_names}
    if not {'P', 'N'}.issubset(roles):
        named = {source.key.upper(): source for source in document.sources if source.key.upper() in ('P', 'N')}
        if set(named) != {'P', 'N'}:
            raise ValueError('Saved composite does not identify independent P and N inputs.')
        role_paths = {resolve_source_path(root, raw) for source in named.values() for raw in source.files}
        if not role_paths or not role_paths.issubset(origin_names):
            raise ValueError('Saved P/N snapshots do not describe the composite inputs.')
    origins, timed, warnings = [], [], []
    columns = tuple(float(value) for value in document.sources[0].spectral_columns)
    for raw in sorted(origin_names, key=str):
        _check(cancelled)
        paths.add(raw)
        before.setdefault(str(raw), _digest_signature(raw))
        short = portable_source_name(root, raw)
        # Frozen metadata retains condition evidence even after raw data moves.
        origin = DrrSource(short, raw.name, '', '', 0, True, spectral_grid=columns,
                           wavelength_center_nm=extract_wavelength_center_nm(raw.name))
        origins.append(origin)
        if raw.is_file() and raw in bindings and sha256_file(raw) == bindings[raw]:
            current = inspect(short)
            if spectral_grids_match(origin, current):
                timed.append(current)
        else:
            warnings.append('Original P/N input is missing, changed or unverified; using the frozen background: ' + raw.name)
    source = DrrSource(name, path.name, '', '', 0, True, spectral_grid=columns)
    reports = [json.loads(item.generation_json or '{}') for item in sources]
    if any(report.get('scientifically_validated') is False or
           str(report.get('algorithm_version', '')).startswith('pn_raw_') for report in reports):
        warnings.append('This saved P/N background is an experimental draft; review it before applying.')
    return _Candidate(name, source, tuple(origins), tuple(timed), tuple(dict.fromkeys(warnings)))


def _timing(root, sources, cancelled):
    """Reuse capture-log parsing without changing the measured-BN selector.

    Every raw input is supplied as a measurement, so the BN-specific candidate
    filter plays no part. Its clock consistently uses capture times or mtimes.
    """
    from core.drr_measured_bn import MeasuredBnMatcher
    if not sources:
        return {}, '', '', None
    clock = MeasuredBnMatcher(root, sources, (), cancelled=cancelled, include_capture_contexts=True)
    clock.capture_context(sources[0])
    timing = clock._timing
    return timing.times, timing.basis, timing.error, clock


def _compatibility(measurement, candidate, relaxed_conditions=()):
    if not spectral_grids_match(measurement, candidate.source):
        return None
    unknown, relaxed = set(), []
    for source in candidate.origins:
        differences, absent = _compare(measurement, source)
        if any(item.split(':', 1)[0] not in relaxed_conditions for item in differences):
            return None
        relaxed.extend(f"Measurement {measurement.filename} / background {source.filename}: {item}" for item in differences)
        unknown.update(absent)
    for first in candidate.origins:
        for other in candidate.origins:
            differences, _ = _compare(first, other)
            if any(item.split(':', 1)[0] not in relaxed_conditions for item in differences):
                return None
            relaxed.extend(f"Background {first.filename} / background {other.filename}: {item}" for item in differences)
    return unknown, tuple(dict.fromkeys(relaxed))


def _rank(measurement, candidate, root, times, timing_basis, timing_error, relaxed_conditions=()):
    compatibility = _compatibility(measurement, candidate, relaxed_conditions)
    if compatibility is None:
        return None
    unknown, differences = compatibility
    warnings = list(candidate.warnings)
    if differences:
        warnings.append("Relaxed acquisition conditions; review differences: " + "; ".join(differences))
    if unknown:
        warnings.append('Unknown acquisition conditions: ' + ', '.join(sorted(unknown)) + '.')
    paths = [resolve_source_path(root, source.source) for source in candidate.timed_origins]
    measurement_time = times.get(str(resolve_source_path(root, measurement.source)).casefold())
    stamps = [times.get(str(path).casefold()) for path in paths]
    known_time = (not timing_error and measurement_time is not None and len(stamps) == len(candidate.origins)
                  and all(value is not None for value in stamps))
    gap = max((abs(stamp - measurement_time) for stamp in stamps), default=None) if known_time else None
    basis = timing_basis if gap is not None else 'unavailable'
    if basis == 'filesystem_mtime_proxy':
        warnings.append('Capture logs are incomplete; original source file modification times are a proxy and may change when copied.')
    elif basis == 'unavailable':
        warnings.append('Original acquisition timing is unavailable; the saved background creation time is not used.')
    if timing_error:
        warnings.append(timing_error)
    if gap is not None and gap > 86400:
        warnings.append(f'Original source time gap {gap:g} s exceeds one day; review this assignment.')
    evidence = ('Same ordered native spectral grid; relaxed acquisition-condition differences' if differences
                else 'Same ordered native spectral grid; no known acquisition-condition conflict')
    if differences:
        evidence += '; ' + '; '.join(differences)
    if gap is not None:
        evidence += f'; original source time gap {gap:g} s ({basis})'
    else:
        evidence += '; original acquisition time unavailable'
    # The worst source distance prevents a nearby P from hiding a remote N.
    score = (int(bool(differences)), len(unknown), gap if gap is not None else math.inf)
    return score, evidence, tuple(dict.fromkeys(warnings)), gap, basis


def build_pn_match_plan(root, selected, *, kind, candidate_files=None, which='all', cancelled=None, relaxed_conditions=()):
    """Suggest existing backgrounds, retaining ambiguous rows for manual choice.

    Known condition, role and native-grid conflicts are never eligible. Unknown
    conditions remain warnings. Equal best evidence never selects by filename.
    """
    root = Path(root).resolve()
    relaxed_conditions = normalize_relaxed_conditions(relaxed_conditions)
    if kind not in ('p', 'n', 'pn') or which not in ('first', 'last', 'all'):
        raise ValueError('Choose P only, N only or P+N, and a valid frame method.')
    selected = tuple(selected)
    _check(cancelled)
    candidates = _inventory(root, kind, candidate_files, cancelled)
    paths = {resolve_source_path(root, name) for name in (*selected, *candidates)}
    initial = _stats(paths)
    before = dict(initial)
    directories = _directories(root, paths, include_store=kind == 'pn' and candidate_files is None)
    directory_snapshot = _stats(directories)
    # Capture logs and absent metadata paths are part of the verified snapshot.
    for directory in directories:
        if directory.is_dir():
            paths.update(directory.glob('*.experiment.events.jsonl'))
    before.update(_stats(paths))
    inspected = {}
    def inspect(name):
        _check(cancelled)
        if name not in inspected:
            path = resolve_source_path(root, name)
            before.setdefault(str(path), _digest_signature(path))
            inspected[name] = _inspect(root, name)
        return inspected[name]
    available, warnings, measurement_errors = [], [], {}
    measurements = []
    for name in selected:
        try:
            measurements.append(inspect(name))
        except (OSError, ValueError, TypeError) as exc:
            measurement_errors[name] = str(exc)
    selected_paths = {resolve_source_path(root, name) for name in selected}
    for name in candidates:
        _check(cancelled)
        path = resolve_source_path(root, name)
        if path in selected_paths:
            continue
        try:
            if kind == 'pn':
                if not is_composite_background(path):
                    raise ValueError('Choose a saved P+N composite with its metadata file.')
                available.append(_composite(root, name, paths, before, inspect, cancelled))
            else:
                if is_composite_background(path):
                    raise ValueError('A composite cannot be relabelled as a raw P or N reference.')
                source = inspect(name)
                entries = build_pn_source_catalog((source,))
                entry = entries[0] if entries else None
                if (entry is None or entry.role != kind.upper() or entry.role_conflict
                        or source.gate_varies is not False):
                    raise ValueError('Role or constant-gate evidence is incompatible with this source type.')
                available.append(_Candidate(name, source, (source,), (source,)))
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f'Cannot use {Path(name).name}: {exc}')
    # Snapshot raw provenance before timing; include its log directories too.
    directories = _directories(root, paths, include_store=kind == 'pn' and candidate_files is None)
    for directory in directories:
        if directory.is_dir():
            paths.update(directory.glob('*.experiment.events.jsonl'))
    timing_directories = _stats(directories)
    for path, signature in _stats(paths):
        before.setdefault(path, signature)
    snapshot = _stats(paths)
    available = [candidate for candidate in available
                 if any(_compatibility(measurement, candidate, relaxed_conditions) is not None for measurement in measurements)]
    all_sources = {source.source: source for source in (*measurements,
        *(source for candidate in available for source in candidate.timed_origins))}
    times, timing_basis, timing_error, clock = _timing(root, tuple(all_sources.values()), cancelled)
    rows = []
    score_maps = {}
    by_name = {source.source: source for source in measurements}
    for name in selected:
        _check(cancelled)
        if name in measurement_errors:
            rows.append(PnMatchRow(name, evidence='Cannot inspect measurement: ' + measurement_errors[name]))
            continue
        ranked = []
        for candidate in available:
            result = _rank(by_name[name], candidate, root, times, timing_basis, timing_error, relaxed_conditions)
            if result is not None:
                ranked.append((result[0], candidate.name, *result[1:]))
        ranked.sort(key=lambda value: (value[0], value[1]))
        score_maps[name] = tuple((item[1], (item[0][0], item[0][1], item[0][2] if math.isfinite(item[0][2]) else None)) for item in ranked)
        options = tuple(item[1] for item in ranked)
        details = tuple((item[1], item[2], item[3], item[4], item[5]) for item in ranked)
        if not ranked:
            rows.append(PnMatchRow(name, evidence='No compatible existing background for this measurement.'))
            continue
        best = ranked[0]
        tied = len(ranked) > 1 and (best[0][:2] == ranked[1][0][:2]
            and (best[0][2] == ranked[1][0][2] or math.isclose(best[0][2], ranked[1][0][2], abs_tol=1e-6, rel_tol=0)))
        if tied or timing_error:
            rows.append(PnMatchRow(name, evidence=('Acquisition timing needs review; choose a background explicitly.'
                if timing_error else 'Equally suitable backgrounds; choose one explicitly.'),
                warnings=tuple(dict.fromkeys(warning for item in ranked for warning in item[3])),
                candidate_files=options, candidate_details=details))
            continue
        row_warnings = list(best[3])
        if len(ranked) > 1 and best[0][:2] == ranked[1][0][:2] and ranked[1][0][2] - best[0][2] < 60:
            row_warnings.append('The next candidate is less than 60 s farther away; review the choice.')
        assignment = DrrMeasurementAssignment(name, 'External', (best[1],), which,
            selection_reason=f'P/N match ({kind}): ' + best[2])
        rows.append(PnMatchRow(name, assignment, best[2], tuple(row_warnings), options,
                               best[4], best[5], details))
    groups = {}
    if kind in ('p', 'n') and clock is not None:
        from core.drr_bn_batches import _compatibility_key
        for source in measurements:
            _check(cancelled)
            context = clock.capture_context(source)
            if context.verified_run:
                key = (context.run_id, context.experiment_id, _compatibility_key(source),
                       extract_wavelength_center_nm(source.filename))
                groups.setdefault(key, []).append(source.source)
    batch_inputs = tuple((tuple(members), tuple((name, score_maps.get(name, ())) for name in members))
                         for members in groups.values() if len(members) > 1)
    _check(cancelled)
    changed = (not _unchanged(tuple(before.items())) or not _unchanged(snapshot)
               or not _unchanged(directory_snapshot) or not _unchanged(timing_directories)
               or candidates != _inventory(root, kind, candidate_files, cancelled))
    if changed:
        rows = [PnMatchRow(name, evidence='Inputs changed during matching; refresh and retry.') for name in selected]
    plan = PnMatchPlan(str(root), selected, kind, which, candidates, candidate_files is None, (),
        DrrBackgroundResolution(), snapshot=snapshot, directory_snapshot=_stats(directories),
        inventory_warnings=tuple(dict.fromkeys(warnings)), relaxed_conditions=relaxed_conditions, batch_inputs=batch_inputs, _verified=_PLAN)
    return _finish(plan, rows)


def override_pn_match(plan, measurements, baseline_file):
    """Manually assign one compatible candidate to any selected rows; no I/O."""
    if not isinstance(plan, PnMatchPlan) or plan._verified is not _PLAN or _signature(plan) != plan.signature:
        raise ValueError('Refresh the P/N mapping before editing it.')
    measurements = (measurements,) if isinstance(measurements, str) else tuple(measurements)
    if not measurements or not set(measurements).issubset(plan.selected):
        raise ValueError('Choose measurement rows from this P/N mapping.')
    rows = []
    for row in plan.rows:
        if row.measurement_file in measurements:
            detail = next((item for item in row.candidate_details if item[0] == baseline_file), None)
            if detail is None:
                raise ValueError('This background has incompatible role, conditions or native spectral grid.')
            _, evidence, warnings, gap, basis = detail
            evidence = 'Manual assignment; ' + evidence
            row = replace(row, assignment=DrrMeasurementAssignment(row.measurement_file, 'External',
                (baseline_file,), plan.which, selection_reason=f'P/N match ({plan.kind}): ' + evidence), evidence=evidence,
                warnings=warnings, time_gap_seconds=gap, timing_basis=basis)
        rows.append(row)
    return _finish(plan, rows)


def plan_is_current(plan):
    try:
        if (not isinstance(plan, PnMatchPlan) or not plan.ready or _signature(plan) != plan.signature
                or not _unchanged(plan.directory_snapshot) or not _unchanged(plan.snapshot)):
            return False
        return (not plan.automatic_candidates or _inventory(Path(plan.root), plan.kind) == plan.candidate_files)
    except (OSError, ValueError, TypeError):
        return False


def prepare_pn_confirmation(plan, *, warnings_acknowledged=False):
    """Worker verification before the UI publishes a reviewed assignment."""
    _check()
    if not plan_is_current(plan):
        raise ValueError('P/N mapping is incomplete or inputs changed; match again before applying.')
    if plan.warnings and not warnings_acknowledged:
        raise ValueError('Review and acknowledge the P/N matching warnings before applying.')
    _check()
    return PreparedPnConfirmation(plan, plan.signature, bool(warnings_acknowledged), _CONFIRMATION)


def confirmation_is_current(prepared):
    """Final GUI guard reads only known file/directory stats, never CSV bodies."""
    return bool(isinstance(prepared, PreparedPnConfirmation) and prepared._verified is _CONFIRMATION
        and prepared.plan.ready and prepared.signature == prepared.plan.signature
        and _signature(prepared.plan) == prepared.signature
        and (not prepared.plan.warnings or prepared.warnings_acknowledged)
        and _unchanged(prepared.plan.directory_snapshot) and _unchanged(prepared.plan.snapshot)
        and _unchanged(prepared.plan.directory_snapshot))
