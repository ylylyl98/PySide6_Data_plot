"""Reviewable joint raw P/N assignments and conservative temporal segments.

Planning never generates, averages, applies, or changes source files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path

from core.drr_baseline_candidates import spectral_grids_match
from core.drr_bn_batches import _compatibility_key
from core.drr_composite_background import composite_metadata_path, is_composite_background
from core.drr_match_conditions import normalize_relaxed_conditions
from core.drr_pn_matching import (_Candidate, _check, _compatibility, _directories,
                                  _inspect, _stats, _timing, _unchanged)
from core.drr_pn_selection import build_pn_source_catalog
from core.drr_sources import extract_wavelength_center_nm, measurement_group_key, resolve_source_path
from core.provenance import sha256_file

_VERIFIED = object()


@dataclass(frozen=True)
class PnPairOption:
    p_file: str
    n_file: str
    measurement_p_gap: float | None
    measurement_n_gap: float | None
    pair_gap: float | None
    timing_basis: str
    differences: tuple[str, ...] = ()
    unknown_conditions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    score: tuple[int, int, float | None, float | None] = (0, 0, None, None)


@dataclass(frozen=True)
class PnBatchRow:
    measurement_file: str
    p_file: str = ''
    n_file: str = ''
    options: tuple[PnPairOption, ...] = ()
    evidence: str = ''
    warnings: tuple[str, ...] = ()
    timing_basis: str = ''
    capture_time: float | None = None
    verified_run: bool = False
    run_id: str = ''
    experiment_id: str = ''
    compatibility_key: str = ''
    manual: bool = False


@dataclass(frozen=True)
class PnBatchSegment:
    measurement_files: tuple[str, ...]
    p_file: str
    n_file: str
    training_files: tuple[str, ...]
    method: str


@dataclass(frozen=True)
class PnBatchPlan:
    root: str
    selected: tuple[str, ...]
    p_candidates: tuple[str, ...]
    n_candidates: tuple[str, ...]
    relaxed_conditions: tuple[str, ...]
    rows: tuple[PnBatchRow, ...]
    segments: tuple[PnBatchSegment, ...]
    warnings: tuple[str, ...]
    inventory_warnings: tuple[str, ...]
    boundary_times: tuple[float, ...]
    snapshot: tuple
    content_digests: tuple[tuple[str, str | None], ...]
    directory_snapshot: tuple
    signature: str = ''
    _verified: object = field(default=None, repr=False, compare=False)

    @property
    def ready(self):
        return bool(self._verified is _VERIFIED and self.selected and self.snapshot
                    and len(set(self.selected)) == len(self.selected)
                    and tuple(row.measurement_file for row in self.rows) == self.selected
                    and all(row.p_file and row.n_file for row in self.rows)
                    and sorted(name for seg in self.segments for name in seg.measurement_files) == sorted(self.selected))


def _signature(plan):
    value = {name: getattr(plan, name) for name in ('root', 'selected', 'p_candidates', 'n_candidates',
        'relaxed_conditions', 'warnings', 'inventory_warnings', 'boundary_times', 'snapshot', 'content_digests', 'directory_snapshot')}
    value['rows'] = [asdict(row) for row in plan.rows]
    value['segments'] = [asdict(segment) for segment in plan.segments]
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _log_inventory(root, source_paths):
    directories = _directories(root, set(source_paths))
    return tuple(sorted(str(path) for directory in directories if directory.is_dir()
                        for path in directory.glob('*.experiment.events.jsonl')))


def _content_digests(snapshot, cancelled=None):
    values = []
    for path, signature in snapshot:
        _check(cancelled)
        digest = sha256_file(path) if signature is not None else None
        if signature is not None and digest is None:
            raise ValueError('Cannot read a P/N matching input for content verification: ' + path)
        values.append((path, digest))
    if not _unchanged(snapshot):
        raise ValueError('Inputs changed during P/N content verification; refresh and retry.')
    return tuple(values)


def _snapshot_is_current(plan):
    paths = [resolve_source_path(plan.root, name) for name in
             (*plan.selected, *plan.p_candidates, *plan.n_candidates)]
    return bool(_unchanged(plan.snapshot)
                and _log_inventory(Path(plan.root), paths) == plan.directory_snapshot
                and _content_digests(plan.snapshot) == plan.content_digests
                and _log_inventory(Path(plan.root), paths) == plan.directory_snapshot)


def _segments(rows, boundary_times):
    # No identity or chronology means no inferred cross-file training group.
    ordered = sorted(rows, key=lambda row: (row.capture_time is None, row.capture_time or 0, row.measurement_file))
    groups = []
    for row in ordered:
        if not row.p_file or not row.n_file:
            groups.append([row])
            continue
        previous = groups[-1][-1] if groups else None
        join = bool(previous and row.verified_run and previous.verified_run
                    and row.capture_time is not None and previous.capture_time is not None
                    and row.capture_time > previous.capture_time
                    and row.capture_time - previous.capture_time <= 86400
                    and row.timing_basis == previous.timing_basis == "capture_started_utc"
                    and not any(previous.capture_time <= stamp <= row.capture_time for stamp in boundary_times)
                    and (row.run_id, row.experiment_id, row.compatibility_key, row.p_file, row.n_file)
                    == (previous.run_id, previous.experiment_id, previous.compatibility_key, previous.p_file, previous.n_file))
        if join:
            groups[-1].append(row)
        else:
            groups.append([row])
    result = []
    for members in groups:
        row = members[0]
        if not row.p_file or not row.n_file:
            continue
        names = tuple(member.measurement_file for member in members)
        result.append(PnBatchSegment(names, row.p_file, row.n_file, names[:2],
                                     'conditioned_single' if len(names) == 1 else 'conditioned_bounded'))
    return tuple(result)


def _finish(plan, rows):
    rows = tuple(rows)
    ordered = sorted((row for row in rows if row.capture_time is not None), key=lambda row: row.capture_time)
    split_names = {row.measurement_file for previous, row in zip(ordered, ordered[1:])
                   if row.verified_run and previous.verified_run
                   and (row.run_id, row.experiment_id, row.compatibility_key, row.p_file, row.n_file)
                   == (previous.run_id, previous.experiment_id, previous.compatibility_key, previous.p_file, previous.n_file)
                   and row.capture_time - previous.capture_time > 86400}
    split_warning = ('Training segments were separated after a gap over one day; this is a review rule, not a physical stability certificate.')
    rows = tuple(replace(row, warnings=tuple(dict.fromkeys((*row.warnings, split_warning))))
                 if row.measurement_file in split_names else row for row in rows)
    warnings = tuple(dict.fromkeys((*plan.inventory_warnings, *(w for row in rows for w in row.warnings))))
    plan = replace(plan, rows=rows, segments=_segments(rows, plan.boundary_times), warnings=warnings)
    return replace(plan, signature=_signature(plan))


def build_pn_batch_plan(root, selected, *, p_candidates, n_candidates, relaxed_conditions=(), cancelled=None):
    root = Path(root).resolve()
    selected, p_candidates, n_candidates = tuple(selected), tuple(p_candidates), tuple(n_candidates)
    relaxed = normalize_relaxed_conditions(relaxed_conditions)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select distinct measurements.')
    if len(set(p_candidates)) != len(p_candidates) or len(set(n_candidates)) != len(n_candidates):
        raise ValueError('Select each candidate only once.')
    names = (*selected, *p_candidates, *n_candidates)
    paths = [resolve_source_path(root, name) for name in names]
    if len({str(path).casefold() for path in paths}) != len(paths):
        raise ValueError('Measurement and P/N candidates must be distinct original files.')
    directories = _directories(root, set(paths))
    directory_snapshot = _log_inventory(root, paths)
    logs = [Path(path) for path in directory_snapshot]
    sidecars = [composite_metadata_path(resolve_source_path(root, name)) for name in (*p_candidates, *n_candidates)]
    snapshot = _stats((*paths, *logs, *sidecars))
    content_digests = _content_digests(snapshot, cancelled)
    sources, errors = {}, {}
    for name in names:
        _check(cancelled)
        try:
            source = _inspect(root, name)
            if name in p_candidates or name in n_candidates:
                if is_composite_background(resolve_source_path(root, name)):
                    raise ValueError('Select original raw P/N files, not a composite.')
                entry = build_pn_source_catalog((source,))[0]
                expected = 'P' if name in p_candidates else 'N'
                if entry.role != expected or entry.role_conflict or source.gate_varies is not False:
                    raise ValueError('P/N role or constant gate is incompatible.')
                if source.frame_count < 4 or source.frame_count % 2:
                    raise ValueError('P/N reconstruction requires an even number of at least four frames.')
            sources[name] = source
        except (OSError, ValueError, TypeError, IndexError) as exc:
            errors[name] = str(exc)
    times, basis, timing_error, clock = _timing(root, tuple(sources.values()), cancelled)
    rows = []
    for name in selected:
        _check(cancelled)
        source = sources.get(name)
        if source is None:
            rows.append(PnBatchRow(name, evidence='Cannot inspect measurement: ' + errors.get(name, 'invalid file')))
            continue
        context = clock.capture_context(source) if clock is not None else None
        verified = bool(context and context.verified_run)
        compatibility_key = repr((_compatibility_key(source), extract_wavelength_center_nm(source.filename),
                                  measurement_group_key(source.filename).casefold()))
        options = []
        for p in p_candidates:
            for n in n_candidates:
                _check(cancelled)
                ps, ns = sources.get(p), sources.get(n)
                if (ps is None or ns is None or ps.frame_count != ns.frame_count
                        or not spectral_grids_match(ps, ns)):
                    continue
                compatible = _compatibility(source, _Candidate('', ps, (ps, ns), (ps, ns)), relaxed)
                if compatible is None:
                    continue
                unknown, differences = compatible
                stamp = lambda filename: times.get(str(resolve_source_path(root, filename)).casefold())
                t, tp, tn = stamp(name), stamp(p), stamp(n)
                available = not timing_error and all(value is not None for value in (t, tp, tn))
                pg, ng, pair = (abs(t-tp), abs(t-tn), abs(tp-tn)) if available else (None, None, None)
                warnings = []
                if differences:
                    warnings.append('Relaxed conditions require review: ' + '; '.join(differences))
                if unknown:
                    warnings.append('Unknown acquisition conditions: ' + ', '.join(sorted(unknown)))
                if basis == 'filesystem_mtime_proxy':
                    warnings.append('Capture logs incomplete; filesystem modification times are a proxy and may change on copying.')
                if timing_error or not available:
                    warnings.append(timing_error or 'Acquisition timing unavailable; choose a pair explicitly.')
                if available and max(pg, ng, pair) > 86400:
                    warnings.append('A P/N source or pair time gap exceeds one day; review this assignment.')
                options.append(PnPairOption(p, n, pg, ng, pair, basis if available else 'unavailable',
                    differences, tuple(sorted(unknown)), tuple(warnings),
                    (int(bool(differences)), len(unknown), max(pg, ng) if available else None, pair)))
        options.sort(key=lambda option: (option.score[0], option.score[1],
            option.score[2] if option.score[2] is not None else math.inf,
            option.score[3] if option.score[3] is not None else math.inf, option.p_file, option.n_file))
        chosen = options[0] if options else None
        tied = bool(chosen and len(options) > 1 and chosen.score[:2] == options[1].score[:2]
            and ((chosen.score[2] is None and options[1].score[2] is None) or
                 (chosen.score[2] is not None and options[1].score[2] is not None
                  and math.isclose(chosen.score[2], options[1].score[2], rel_tol=0, abs_tol=1e-6)
                  and math.isclose(chosen.score[3], options[1].score[3], rel_tol=0, abs_tol=1e-6))))
        if chosen and (tied or chosen.score[2] is None):
            chosen = None
        warnings = list(chosen.warnings if chosen else ())
        if not verified or basis != 'capture_started_utc':
            warnings.append('No verified capture batch with complete acquisition timing: this measurement uses a single-measurement draft.')
        else:
            warnings.append('Segments use selected measurement and candidate times; unselected acquisition events are not assessed.')
        if not chosen:
            warnings.append('Equal or unavailable timing requires an explicit P/N pair.' if options else
                            'No compatible P/N pair; retain this measurement for review.')
        elif len(options) > 1 and options[0].score[:2] == options[1].score[:2] and options[1].score[2] is not None and options[1].score[2] - options[0].score[2] < 60:
            warnings.append('The next pair has a worst-source time gap less than 60 seconds farther away; review.')
        rows.append(PnBatchRow(name, chosen.p_file if chosen else '', chosen.n_file if chosen else '',
            tuple(options), 'Joint P/N conditions and time gaps; review before generation.' if chosen else 'Choose an eligible pair explicitly.',
            tuple(warnings), chosen.timing_basis if chosen else basis,
            times.get(str(resolve_source_path(root, name)).casefold()) if not timing_error else None,
            verified, context.run_id if verified else '', context.experiment_id if verified else '', compatibility_key))
    _check(cancelled)
    if (not _unchanged(snapshot) or _content_digests(snapshot, cancelled) != content_digests
            or _log_inventory(root, paths) != directory_snapshot):
        raise ValueError('Inputs changed during P/N pair matching; refresh and retry.')
    inventory_warnings = tuple(f'Cannot use {name}: {errors[name]}' for name in (*p_candidates, *n_candidates) if name in errors)
    boundary_times = tuple(sorted({times[str(resolve_source_path(root, name)).casefold()]
        for name in (*p_candidates, *n_candidates) if str(resolve_source_path(root, name)).casefold() in times}))
    plan = PnBatchPlan(str(root), selected, p_candidates, n_candidates, relaxed, (), (), (), inventory_warnings, boundary_times, snapshot,
                       content_digests, directory_snapshot, _verified=_VERIFIED)
    return _finish(plan, rows)


def override_pn_batch_pair(plan, measurement, p_file, n_file):
    if not isinstance(plan, PnBatchPlan) or plan._verified is not _VERIFIED or _signature(plan) != plan.signature or not _snapshot_is_current(plan):
        raise ValueError('Refresh P/N assignments before editing; matching inputs may have changed.')
    rows = []
    found = False
    for row in plan.rows:
        if row.measurement_file == measurement:
            found = True
            option = next((item for item in row.options if (item.p_file, item.n_file) == (p_file, n_file)), None)
            if option is None:
                raise ValueError('This pair is not compatible with the measurement and policy.')
            row = replace(row, p_file=p_file, n_file=n_file, warnings=tuple(dict.fromkeys((*option.warnings, *(w for w in row.warnings if w.startswith(('No verified capture batch', 'Segments use selected')))))),
                          evidence='Manual P/N pair assignment; review before generation.', timing_basis=option.timing_basis, manual=True)
        rows.append(row)
    if not found:
        raise ValueError('Select a measurement from this plan.')
    return _finish(plan, rows)


def pn_batch_plan_is_current(plan):
    try:
        return bool(isinstance(plan, PnBatchPlan) and plan.ready and _signature(plan) == plan.signature
                    and _snapshot_is_current(plan))
    except (OSError, ValueError, TypeError):
        return False
