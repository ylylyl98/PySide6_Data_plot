"""Read-only, reviewable measured-BN suggestions and confirmed-plan reuse.

A suggestion is never an approval. The caller shows batch summaries with
per-file details and explicitly confirms the snapshot, including its warnings.
Only ordinary measured BN CSVs inside this experiment can enter this service.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

from core.drr_bn_store import is_bn_calibration_source
from core.drr_bn_profile_cache import BN_GATE_PROFILE_CACHE
from core.drr_bn_batches import BATCH_POLICY_VERSION, BnMatchBatch, assess_bn_batches
from core.drr_composite_background import is_composite_background
from core.drr_measured_bn import MeasuredBnMatcher, is_measured_bn_name
from core.drr_sources import (
    DrrBackgroundResolution, DrrMeasurementAssignment, DrrSource,
    inspect_csv_gate_profile, preview_drr_sources, resolve_source_path,
)
from core.load_cancellation import LoadCancelled, check_load_cancelled
from core.provenance import _digest_signature


SCHEMA_VERSION = 2
DEFAULT_MAX_GAP_SECONDS = 86400.0
DEFAULT_MIN_MARGIN_SECONDS = 60.0
_PREPARED_BN_CONFIRMATION = object()


def _strings(value, field):
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) for v in value):
        raise ValueError(f'{field} must be a list of strings.')
    return tuple(value)


def _number(value, field, *, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{field} must be a finite nonnegative number.')
    return float(value)


@dataclass(frozen=True)
class BnMatchRow:
    measurement_file: str
    assignment: DrrMeasurementAssignment | None = None
    evidence: str = ''
    warnings: tuple[str, ...] = ()
    timing_basis: str = ''
    time_gap_seconds: float | None = None
    runner_up_gap_seconds: float | None = None
    unknown_conditions: tuple[str, ...] = ()
    candidate_count: int = 0
    batch_id: str = ''
    batch_status: str = ''
    anomaly: bool = False

    @property
    def baseline_file(self):
        return self.assignment.baseline_files[0] if self.assignment else ''

    def to_dict(self):
        return dict(
            measurement_file=self.measurement_file,
            assignment=self.assignment.to_dict() if self.assignment else None,
            evidence=self.evidence, warnings=list(self.warnings), timing_basis=self.timing_basis,
            time_gap_seconds=self.time_gap_seconds, runner_up_gap_seconds=self.runner_up_gap_seconds,
            unknown_conditions=list(self.unknown_conditions), candidate_count=self.candidate_count,
            batch_id=self.batch_id, batch_status=self.batch_status, anomaly=self.anomaly,
        )

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError('BN match row must be an object.')
        for field in ('measurement_file', 'evidence', 'timing_basis', 'batch_id', 'batch_status'):
            if not isinstance(value.get(field), str):
                raise ValueError(f'BN match row {field} must be a string.')
        count = value.get('candidate_count')
        if type(count) is not int or count < 0:
            raise ValueError('BN match candidate count must be a nonnegative integer.')
        if type(value.get('anomaly')) is not bool:
            raise ValueError('BN match anomaly flag must be boolean.')
        assignment = (DrrMeasurementAssignment.from_mapping(value['assignment'])
                      if value.get('assignment') is not None else None)
        if assignment and (assignment.measurement_file != value['measurement_file']
                           or assignment.baseline_mode != 'External'
                           or len(assignment.baseline_files) != 1
                           or assignment.baseline_which != 'all' or assignment.regions):
            raise ValueError('BN match row must contain one measured BN, using all frames.')
        return cls(
            value['measurement_file'], assignment, value['evidence'],
            _strings(value.get('warnings'), 'warnings'), value['timing_basis'],
            _number(value.get('time_gap_seconds'), 'time gap', optional=True),
            _number(value.get('runner_up_gap_seconds'), 'runner-up margin', optional=True),
            _strings(value.get('unknown_conditions'), 'unknown conditions'), count,
            value['batch_id'], value['batch_status'], value['anomaly'],
        )


@dataclass(frozen=True)
class BnMatchPlan:
    root: str
    selected: tuple[str, ...]
    candidate_files: tuple[str, ...]
    automatic_candidates: bool
    rows: tuple[BnMatchRow, ...]
    resolution: DrrBackgroundResolution
    signature: str
    warnings: tuple[str, ...] = ()
    # Absolute file paths and cheap change signatures, including Windows ChangeTime.
    snapshot: tuple[tuple[str, tuple[int, ...] | None], ...] = ()
    log_directories: tuple[str, ...] = ()
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS
    min_margin_seconds: float = DEFAULT_MIN_MARGIN_SECONDS
    batches: tuple[BnMatchBatch, ...] = ()
    policy_version: str = BATCH_POLICY_VERSION

    @property
    def ready(self):
        if (self.policy_version != BATCH_POLICY_VERSION or not self.batches or not self.resolution.resolved
                or not self.snapshot or any(stamp is None for _, stamp in self.snapshot)
                or tuple(row.measurement_file for row in self.rows) != self.selected
                or len(set(self.selected)) != len(self.selected)
                or tuple(row.assignment for row in self.rows) != self.resolution.assignments
                or any(row.assignment is None for row in self.rows)):
            return False
        members = tuple(name for batch in self.batches for name in batch.measurement_files)
        if (len(members) != len(self.selected) or len(set(members)) != len(members)
                or set(members) != set(self.selected)
                or len({batch.batch_id for batch in self.batches}) != len(self.batches)):
            return False
        row_map = {row.measurement_file: row for row in self.rows}
        for batch in self.batches:
            if (batch.status not in {'consensus', 'conflict', 'unverified'}
                    or (batch.status == 'consensus' and (not batch.verified_run or not batch.run_id
                        or not batch.baseline_file or batch.anomaly_measurements))
                    or (not batch.verified_run and len(batch.measurement_files) != 1)):
                return False
            for name in batch.measurement_files:
                row = row_map[name]
                if (row.batch_id != batch.batch_id or row.batch_status != batch.status
                        or row.anomaly != (name in batch.anomaly_measurements)
                        or (batch.baseline_file and row.baseline_file != batch.baseline_file)):
                    return False
        return True

    @property
    def requires_acknowledgement(self):
        return bool(self.warnings)

    def to_dict(self):
        return dict(
            schema_version=SCHEMA_VERSION, root=self.root, selected=list(self.selected),
            candidate_files=list(self.candidate_files), automatic_candidates=self.automatic_candidates,
            rows=[row.to_dict() for row in self.rows], resolution=self.resolution.to_dict(),
            signature=self.signature, warnings=list(self.warnings),
            snapshot=[[path, list(stamp) if stamp is not None else None] for path, stamp in self.snapshot],
            log_directories=list(self.log_directories), max_gap_seconds=self.max_gap_seconds,
            min_margin_seconds=self.min_margin_seconds,
            batches=[batch.to_dict() for batch in self.batches], policy_version=self.policy_version,
        )

    @classmethod
    def from_dict(cls, value):
        """Restore a versioned JSON object; freshness is checked separately.

        Cached approvals are useful only with exactly the original root, paths,
        policy, source signatures and timing-log inventory.
        """
        if not isinstance(value, dict) or type(value.get('schema_version')) is not int or value['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Unsupported BN match plan schema.')
        if value.get('policy_version') != BATCH_POLICY_VERSION:
            raise ValueError('Saved BN matching policy requires a new explicit review.')
        if not isinstance(value.get('root'), str) or not Path(value['root']).is_absolute():
            raise ValueError('BN match plan requires an absolute experiment root.')
        if type(value.get('automatic_candidates')) is not bool or not isinstance(value.get('signature'), str):
            raise ValueError('Invalid BN match plan identity.')
        selected = _strings(value.get('selected'), 'selected measurements')
        candidates = _strings(value.get('candidate_files'), 'candidate files')
        if not isinstance(value.get('rows'), list):
            raise ValueError('BN match rows must be a list.')
        rows = tuple(BnMatchRow.from_dict(row) for row in value['rows'])
        if tuple(row.measurement_file for row in rows) != selected:
            raise ValueError('BN match rows do not describe the selected measurements.')
        if any(row.baseline_file and row.baseline_file not in candidates for row in rows):
            raise ValueError('BN match assignment is outside the candidate pool.')
        if not isinstance(value.get('batches'), list):
            raise ValueError('BN batches must be a list.')
        batches = tuple(BnMatchBatch.from_dict(item) for item in value['batches'])
        members = tuple(name for batch in batches for name in batch.measurement_files)
        if (len(members) != len(selected) or len(set(members)) != len(members)
                or len(set(selected)) != len(selected) or set(members) != set(selected)
                or len({batch.batch_id for batch in batches}) != len(batches)):
            raise ValueError('BN batches must partition every selected measurement exactly once.')
        row_map = {row.measurement_file: row for row in rows}
        for batch in batches:
            if any(name not in candidates for name in batch.candidate_files):
                raise ValueError('BN batch candidate is outside the experiment candidate pool.')
            has_missing = any(row_map[name].assignment is None for name in batch.measurement_files)
            if (batch.status == 'unresolved') != has_missing:
                raise ValueError('BN batch unresolved status and per-file assignments disagree.')
            for name in batch.measurement_files:
                row = row_map[name]
                if (row.batch_id != batch.batch_id or row.batch_status != batch.status
                        or row.anomaly != (name in batch.anomaly_measurements)
                        or (batch.baseline_file and row.baseline_file != batch.baseline_file)):
                    raise ValueError('BN batch assessment and per-file preview disagree.')
        raw_resolution = value.get('resolution')
        if not isinstance(raw_resolution, dict):
            raise ValueError('BN match resolution must be an object.')
        assignments = tuple(row.assignment for row in rows if row.assignment is not None)
        serialized_assignments = raw_resolution.get('assignments')
        if not isinstance(serialized_assignments, list) or tuple(
                DrrMeasurementAssignment.from_mapping(item) for item in serialized_assignments) != assignments:
            raise ValueError('BN match resolution and preview disagree.')
        unresolved = _strings(raw_resolution.get('unresolved_measurements'), 'unresolved measurements')
        if unresolved != tuple(row.measurement_file for row in rows if row.assignment is None):
            raise ValueError('BN match unresolved measurements and preview disagree.')
        for field in ('reason', 'numerical_path'):
            if not isinstance(raw_resolution.get(field), str):
                raise ValueError(f'BN match resolution {field} must be a string.')
        raw_snapshot = value.get('snapshot')
        if not isinstance(raw_snapshot, list):
            raise ValueError('BN match snapshot must be a list.')
        snapshot = []
        for item in raw_snapshot:
            if (not isinstance(item, (list, tuple)) or len(item) != 2
                    or not isinstance(item[0], str) or not Path(item[0]).is_absolute()
                    or (item[1] is not None and (not isinstance(item[1], (list, tuple))
                        or len(item[1]) != 5 or any(type(v) is not int for v in item[1])))):
                raise ValueError('Invalid BN match file signature.')
            snapshot.append((item[0], tuple(item[1]) if item[1] is not None else None))
        plan = cls(
            value['root'], selected, candidates, value['automatic_candidates'], rows,
            DrrBackgroundResolution(assignments, raw_resolution['numerical_path'], raw_resolution['reason'], unresolved),
            value['signature'], _strings(value.get('warnings'), 'warnings'), tuple(snapshot),
            _strings(value.get('log_directories'), 'log directories'),
            _number(value.get('max_gap_seconds'), 'maximum time gap'),
            _number(value.get('min_margin_seconds'), 'minimum timing margin'),
            batches, value['policy_version'],
        )
        if _plan_signature(plan) != plan.signature:
            raise ValueError('BN match plan signature is invalid.')
        return plan


def _check(cancelled):
    check_load_cancelled()
    if cancelled is not None:
        if hasattr(cancelled, 'check'):
            cancelled.check()
        elif (cancelled.is_set() if hasattr(cancelled, 'is_set') else cancelled()):
            raise LoadCancelled()


def _candidate_names(root, candidate_files, cancelled=None):
    names = (item.source for item in preview_drr_sources(root, include_all=True)) if candidate_files is None else candidate_files
    found = {}
    for name in names:
        _check(cancelled)
        path = resolve_source_path(root, name)
        if not path.is_relative_to(root):
            raise ValueError(f'Measured BN candidate is outside the experiment folder: {name}')
        if path.suffix.casefold() == '.csv' and is_measured_bn_name(path.name):
            found[str(path).casefold()] = path.relative_to(root).as_posix()
    return tuple(found[key] for key in sorted(found))


def _log_directories(root, paths):
    directories = {root, root/'Initial Data', root/'Initial data'}
    for path in paths:
        for directory in (path.parent, path.parent.parent):
            if directory.is_relative_to(root):
                directories.add(directory)
    return tuple(str(path) for path in sorted(directories))


def _snapshot(paths, log_directories, cancelled=None):
    inventory = {str(path.resolve()).casefold(): path.resolve() for path in paths}
    for directory in log_directories:
        _check(cancelled)
        folder = Path(directory)
        if folder.is_dir():
            for path in folder.glob('*.experiment.events.jsonl'):
                _check(cancelled)
                if path.is_file():
                    inventory[str(path.resolve()).casefold()] = path.resolve()
    result = []
    for key in sorted(inventory):
        _check(cancelled)
        path = inventory[key]
        result.append((str(path), _digest_signature(path)))
    return tuple(result)


def _plan_signature(plan):
    payload = plan.to_dict()
    payload.pop('signature')
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _row_warnings(match, max_gap, min_margin):
    warnings = []
    if match.timing_basis != 'capture_started_utc':
        warnings.append('Capture logs are incomplete; timing uses file modification times, which can change when files are copied.')
    if match.unknown_conditions:
        warnings.append('Unknown acquisition conditions: ' + ', '.join(match.unknown_conditions) + '.')
    if match.time_gap_seconds is not None and match.time_gap_seconds > max_gap:
        warnings.append(f'Time gap {match.time_gap_seconds:g} s exceeds the review threshold {max_gap:g} s.')
    if match.runner_up_gap_seconds is not None and match.runner_up_gap_seconds < min_margin:
        warnings.append(f'The next candidate is only {match.runner_up_gap_seconds:g} s farther away; review threshold {min_margin:g} s.')
    return tuple(warnings)


def build_bn_match_plan(root, selected, *, candidate_files=None, cancelled=None,
                        max_gap_seconds=DEFAULT_MAX_GAP_SECONDS,
                        min_margin_seconds=DEFAULT_MIN_MARGIN_SECONDS):
    """Suggest one measured BN per source, preserving every row for review.

    ``None`` discovers a fresh bounded experiment inventory. An explicit pool,
    including an empty tuple, restricts suggestions to precisely that pool.
    Neither mode reads saved recipes nor falls back to P/N or generated data.
    """
    from dataclasses import replace

    root = Path(root).resolve()
    selected = tuple(dict.fromkeys(str(name) for name in selected))
    max_gap = _number(max_gap_seconds, 'maximum time gap')
    min_margin = _number(min_margin_seconds, 'minimum timing margin')
    automatic = candidate_files is None
    pool = None if automatic else tuple(candidate_files)
    candidates, rows, snapshot, directories, warnings, batch_records = (), [], (), (), [], ()

    def finish(reason=''):
        by_name = {row.measurement_file: row for row in rows}
        ordered = tuple(by_name.get(name, BnMatchRow(name, evidence=reason)) for name in selected)
        batches = batch_records
        if not batches:
            batches, annotated = assess_bn_batches(root, selected, ordered, {}, None,
                max_gap_seconds=max_gap, min_margin_seconds=min_margin)
            ordered = tuple(annotated)
        assignments = tuple(row.assignment for row in ordered if row.assignment is not None)
        unresolved = tuple(row.measurement_file for row in ordered if row.assignment is None)
        signatures = {(item.baseline_files, item.baseline_which) for item in assignments}
        resolution = DrrBackgroundResolution(
            assignments, 'unresolved' if unresolved or not assignments else ('common' if len(signatures) == 1 else 'heterogeneous'),
            reason or 'Review measured BN suggestions for every selected measurement.', unresolved,
        )
        all_warnings = tuple(dict.fromkeys((*warnings, *(warning for row in ordered for warning in row.warnings))))
        plan = BnMatchPlan(str(root), selected, candidates, automatic, ordered, resolution, '',
                           all_warnings, snapshot, directories, max_gap, min_margin, batches)
        return replace(plan, signature=_plan_signature(plan))

    try:
        _check(cancelled)
        if not selected:
            return finish('No measurements selected.')
        selected_paths = tuple(resolve_source_path(root, name) for name in selected)
        if any(not path.is_relative_to(root) for path in selected_paths):
            return finish('Selected measurement is outside the experiment folder.')
        candidates = _candidate_names(root, pool, cancelled)
        candidate_paths = tuple(resolve_source_path(root, name) for name in candidates)
        paths = tuple(dict.fromkeys((*selected_paths, *candidate_paths)))
        directories = _log_directories(root, paths)
        snapshot = _snapshot(paths, directories, cancelled)
        measurements, references = [], []

        def inspect(name):
            _check(cancelled)
            path = resolve_source_path(root, name)
            if path.suffix.casefold() != '.csv' or not path.is_file():
                raise ValueError('Not an available measurement CSV.')
            profile = BN_GATE_PROFILE_CACHE.load(path, inspect_csv_gate_profile,
                                                check=lambda: _check(cancelled))
            if len(profile.spectral_grid) < 2 or profile.frame_count is None or profile.frame_count < 1:
                raise ValueError('No verifiable spectral grid or frames.')
            return DrrSource(name, path.name, '', '', path.stat().st_mtime, False,
                             gate_varies=profile.varies, frame_count=profile.frame_count,
                             spectral_grid=tuple(profile.spectral_grid), gate_grid=profile.gate_grid,
                             gate_direction=profile.gate_direction, gate_labels=profile.gate_labels)

        for name in selected:
            try:
                measurements.append(inspect(name))
            except (OSError, ValueError, TypeError) as exc:
                rows.append(BnMatchRow(name, evidence=f'Cannot inspect measurement: {exc}'))
        selected_identities = {str(path).casefold() for path in selected_paths}
        for name, path in zip(candidates, candidate_paths):
            _check(cancelled)
            if str(path).casefold() in selected_identities:
                continue
            try:
                if (is_bn_calibration_source(root, name) or is_composite_background(path)
                        or any(part.casefold() in {'.drr_bn_calibration', 'pn backgrounds'} for part in path.parts)):
                    continue
                references.append(inspect(name))
            except (OSError, ValueError, TypeError) as exc:
                warnings.append(f'Cannot inspect BN candidate {name}: {exc}')
        matcher = MeasuredBnMatcher(root, measurements, references, cancelled=cancelled, include_capture_contexts=True)
        for measurement in measurements:
            _check(cancelled)
            match = matcher.match(measurement)
            _check(cancelled)
            row_warnings = _row_warnings(match, max_gap, min_margin) if match.assignment else ()
            rows.append(BnMatchRow(measurement.source, match.assignment, match.reason,
                                   row_warnings, match.timing_basis, match.time_gap_seconds,
                                   match.runner_up_gap_seconds, match.unknown_conditions, match.candidate_count))
        batch_records, rows = assess_bn_batches(root, selected, rows,
            {item.source: item for item in measurements}, matcher, max_gap_seconds=max_gap,
            min_margin_seconds=min_margin, check=lambda: _check(cancelled))
        _check(cancelled)
        if (candidates != _candidate_names(root, pool, cancelled)
                or snapshot != _snapshot(paths, directories, cancelled)):
            batch_records = ()
            rows[:] = [BnMatchRow(row.measurement_file, evidence='Sources, BN inventory or timing logs changed during matching; refresh and retry.') for row in rows]
            return finish('Sources, BN inventory or timing logs changed during matching; refresh and retry.')
        if any(row.assignment is None for row in rows):
            return finish('Some measurements have no unambiguous compatible measured BN. Review the individual rows.')
        return finish()
    except LoadCancelled:
        # A cancelled plan cannot be accepted, even if earlier rows succeeded.
        batch_records = ()
        rows[:] = [BnMatchRow(row.measurement_file, evidence='Measured-BN matching cancelled.') for row in rows]
        return finish('Measured-BN matching cancelled.')
    except (OSError, ValueError, TypeError) as exc:
        batch_records = ()
        rows[:] = [BnMatchRow(row.measurement_file, evidence=str(exc)) for row in rows]
        return finish(f'Measured-BN matching needs review: {exc}')


def plan_is_current(plan):
    """Verify an applicable snapshot without re-reading spectral CSV bodies."""
    try:
        if not isinstance(plan, BnMatchPlan) or not plan.ready or _plan_signature(plan) != plan.signature:
            return False
        root = Path(plan.root).resolve()
        paths = tuple(resolve_source_path(root, name) for name in (*plan.selected, *plan.candidate_files))
        if not paths or any(not path.is_relative_to(root) for path in paths):
            return False
        if plan.automatic_candidates and _candidate_names(root, None) != plan.candidate_files:
            return False
        if any(stamp is None for _, stamp in plan.snapshot):
            return False
        return (_log_directories(root, paths) == plan.log_directories
                and _snapshot(paths, plan.log_directories) == plan.snapshot)
    except (OSError, ValueError, TypeError, LoadCancelled):
        return False


@dataclass(frozen=True)
class PreparedBnConfirmation:
    """Worker-verified approval draft; it does not mutate a confirmed cache.

    The process-local seal deliberately cannot be restored from JSON. Persisted
    plans must pass worker verification again before they can be committed.
    """
    plan: BnMatchPlan
    directory_snapshot: tuple[tuple[str, tuple[int, ...] | None], ...]
    file_snapshot: tuple[tuple[str, tuple[int, ...] | None], ...]
    plan_signature: str
    warnings_acknowledged: bool = False
    _verified: object = field(default=None, repr=False, compare=False)


def _confirmation_directories(plan):
    """Snapshot inventory directories in a worker, including absent log roots.

    Checking every existing discovery directory captures new nested files and
    folders through their parent's change signature. The GUI later stats this
    fixed list; it never walks the directory tree or opens spectrum headers.
    """
    root = Path(plan.root)
    directories = {root, root/'Initial Data', root/'Initial data',
                   *(Path(path) for path in plan.log_directories)}
    for path in root.iterdir():
        _check(None)
        if path.name.casefold() == 'initial data' and path.is_dir():
            directories.add(path)
            for child in path.rglob('*'):
                _check(None)
                if child.is_dir():
                    directories.add(child)
    result = []
    for path in sorted(directories):
        _check(None)
        signature = _digest_signature(path)
        if signature is None and path.exists():
            raise ValueError(f'Cannot verify BN inventory directory: {path}')
        result.append((str(path), signature))
    return tuple(result)


def _fixed_signatures(snapshot):
    """Read only already-known paths; no resolution, discovery or CSV reads."""
    return tuple((path, _digest_signature(Path(path))) for path, _ in snapshot)


def confirmation_is_current(prepared):
    """Cheap final guard for a worker-prepared approval or accepted mapping."""
    try:
        if (not isinstance(prepared, PreparedBnConfirmation)
                or prepared._verified is not _PREPARED_BN_CONFIRMATION
                or prepared.plan.signature != prepared.plan_signature
                or prepared.file_snapshot != prepared.plan.snapshot
                or (prepared.plan.requires_acknowledgement and not prepared.warnings_acknowledged)):
            return False
        # Repeat directory signatures around file checks. An addition while
        # checking existing files must not slip through the acceptance guard.
        return (_fixed_signatures(prepared.directory_snapshot) == prepared.directory_snapshot
                and _fixed_signatures(prepared.file_snapshot) == prepared.file_snapshot
                and _fixed_signatures(prepared.directory_snapshot) == prepared.directory_snapshot)
    except (OSError, ValueError, TypeError):
        return False


class BnMatchCache:
    """Small cache of explicitly confirmed plans; never auto-confirms a plan."""

    def __init__(self, *, max_entries=16):
        self.max_entries = max(1, int(max_entries))
        self._plans = []

    def prepare_confirm(self, plan, *, warnings_acknowledged=False):
        """Verify in a worker without changing this cache or application state."""
        _check(None)
        if not isinstance(plan, BnMatchPlan) or not plan.ready:
            raise ValueError('BN match plan is incomplete or stale; refresh before confirming.')
        if plan.requires_acknowledgement and not warnings_acknowledged:
            raise ValueError('BN match warnings require explicit acknowledgement.')
        before = _confirmation_directories(plan)
        if not plan_is_current(plan):
            raise ValueError('BN match plan is incomplete or stale; refresh before confirming.')
        after = _confirmation_directories(plan)
        if before != after or _fixed_signatures(plan.snapshot) != plan.snapshot:
            raise ValueError('BN inputs or inventory changed during verification; refresh before confirming.')
        _check(None)
        return PreparedBnConfirmation(plan, after, plan.snapshot, plan.signature,
                                      bool(warnings_acknowledged), _PREPARED_BN_CONFIRMATION)

    def commit_confirm(self, prepared):
        """Accept a prepared draft using fast signatures only, after UI guards."""
        _check(None)
        if not confirmation_is_current(prepared):
            raise ValueError('BN inputs or inventory changed after verification; the plan is stale. Refresh before confirming.')
        plan = prepared.plan
        self._plans = [item for item in self._plans if item.signature != plan.signature]
        self._plans.append(plan)
        self._plans = self._plans[-self.max_entries:]
        return plan

    def confirm(self, plan, *, warnings_acknowledged=False):
        """Synchronous convenience API; GUI callers should use the two phases."""
        return self.commit_confirm(self.prepare_confirm(plan, warnings_acknowledged=warnings_acknowledged))

    def find(self, root, selected, *, candidate_files=None,
             max_gap_seconds=DEFAULT_MAX_GAP_SECONDS,
             min_margin_seconds=DEFAULT_MIN_MARGIN_SECONDS):
        root = Path(root).resolve()
        selected = tuple(dict.fromkeys(str(name) for name in selected))
        try:
            pool = None if candidate_files is None else _candidate_names(root, tuple(candidate_files))
            for plan in reversed(self._plans):
                if (plan.root == str(root) and plan.selected == selected
                        and plan.automatic_candidates == (pool is None)
                        and (pool is None or plan.candidate_files == pool)
                        and plan.max_gap_seconds == max_gap_seconds
                        and plan.min_margin_seconds == min_margin_seconds and plan_is_current(plan)):
                    return plan
        except (OSError, ValueError, TypeError, LoadCancelled):
            pass
        return None

    def to_dict(self):
        return dict(schema_version=SCHEMA_VERSION, plans=[plan.to_dict() for plan in self._plans])

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict) or type(value.get('schema_version')) is not int
                or value['schema_version'] != SCHEMA_VERSION or not isinstance(value.get('plans'), list)):
            raise ValueError('Unsupported confirmed BN cache schema.')
        cache = cls()
        for item in value['plans'][-cache.max_entries:]:
            plan = BnMatchPlan.from_dict(item)
            if plan.ready:
                cache._plans.append(plan)
        return cache
