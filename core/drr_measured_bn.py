"""Conservative per-measurement selection of an ordinary measured BN reference.

Create one ``MeasuredBnMatcher`` in the External matching worker. Saved/manual
recipes take precedence at the caller. ``applicable=False`` permits the generic
fallback; ``applicable=True`` without an assignment means timing needs review.
Only one BN CSV is selected, using all its frames, without pooling acquisitions.
No files or persistent application state are written.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
from threading import RLock

from core.drr_baseline_candidates import parse_drr_acquisition_conditions, spectral_grids_match
from core.drr_sources import (DrrMeasurementAssignment, resolve_source_path,
    extract_wavelength_center_nm, wavelength_centers_match)
from core.drr_match_conditions import CONDITION_LABELS, normalize_relaxed_conditions
from core.load_cancellation import LoadCancelled, check_load_cancelled
from core.provenance import _digest_signature


_BN_NAME = re.compile(r'(?<![a-z0-9])bn[ _-]*back(?:ground)?(?=$|[^a-z])', re.I)
_CONDITIONS = ('temperature', 'magnetic_field', 'polarization', 'rotation',
               'power', 'exposure', 'accumulation')
_TIMING_CACHE = OrderedDict()
_TIMING_LOCK = RLock()
_CACHE_MAX_ENTRIES = 16
_CACHE_MAX_SOURCES = 4096
_LOG_CACHE = OrderedDict()
_LOG_CACHE_LOCK = RLock()
_LOG_CACHE_MAX_BYTES = 32 * 1024 * 1024
_LOG_CACHE_MAX_ENTRIES = 128


def is_measured_bn_name(filename) -> bool:
    """Recognize BNback/BNbackground tokens, including 'BN back'."""
    return bool(_BN_NAME.search(Path(str(filename)).stem))


@dataclass(frozen=True)
class MeasuredBnMatch:
    assignment: DrrMeasurementAssignment | None = None
    reason: str = ''
    applicable: bool = False
    timing_basis: str = ''
    time_gap_seconds: float | None = None
    runner_up_gap_seconds: float | None = None
    unknown_conditions: tuple[str, ...] = ()
    candidate_count: int = 0
    relaxed_differences: tuple[str, ...] = ()


@dataclass(frozen=True)
class MeasuredBnCapture:
    run_id: str = ''
    experiment_id: str = ''
    capture_start: float | None = None
    capture_end: float | None = None
    verified_run: bool = False
    evidence: str = 'No verified capture-run identity; do not infer a batch from dates or filenames.'


@dataclass(frozen=True)
class MeasuredBnCandidateTime:
    source: str
    time_gap_seconds: float


@dataclass(frozen=True)
class _Timing:
    times: dict[str, float]
    basis: str
    error: str = ''
    captures: tuple[tuple[str, MeasuredBnCapture], ...] = ()


@dataclass(frozen=True)
class _Capture:
    stamp: float
    run_id: str | None
    experiment_id: str | None
    acquisition_id: str | None
    event_id: str | None
    point_index: int | None
    frames_requested: int | None


@dataclass(frozen=True)
class _LogCapture:
    output: str
    capture: _Capture | None
    error: str = ''


def _parse_log_capture(event):
    """Keep malformed capture evidence until its output is actually selected."""
    if not isinstance(event, dict) or event.get('event') != 'capture_started':
        return None
    output = event.get('output')
    if not isinstance(output, dict) or not isinstance(output.get('file'), str):
        return None
    name = output['file']
    try:
        dt = datetime.fromisoformat(str(event.get('started_utc', event.get('utc'))).replace('Z', '+00:00'))
        stamp = dt.timestamp()
        if dt.tzinfo is None or not math.isfinite(stamp) or stamp <= 0:
            raise ValueError('missing timezone or invalid capture time')
    except (TypeError, ValueError, OverflowError, OSError):
        return _LogCapture(name, None, 'Invalid capture time')
    context = event.get('context')
    point = context.get('point_index') if isinstance(context, dict) else None
    if point is not None and type(point) is not int:
        return _LogCapture(name, None, 'Invalid capture point index')
    def identifier(field):
        value = event.get(field)
        return value if isinstance(value, str) and value.strip() else None
    return _LogCapture(name, _Capture(
        stamp, identifier('run_id'), identifier('experiment_id'),
        identifier('acquisition_id'), identifier('event_id'),
        point if type(point) is int else None,
        event.get('capture_frames_requested') if type(event.get('capture_frames_requested')) is int else None))


def _log_capture_bytes(record):
    # Conservative accounting includes strings/fields even if Python shares
    # them. Cached tuples contain no references to the original JSON objects.
    size = sys.getsizeof(record) + sys.getsizeof(record.output) + sys.getsizeof(record.error)
    if record.capture is not None:
        size += sys.getsizeof(record.capture) + sys.getsizeof(record.capture.__dict__)
        size += sum(sys.getsizeof(value) for value in vars(record.capture).values())
    return size + sys.getsizeof(record.__dict__) + 8


def _capture_log_records(logfile, check):
    """Stream once, then share immutable events across measurement selections.

    Target resolution and point/run validation remain selection-specific. Only
    fully consumed, unchanged logs enter this bounded process-local cache.
    """
    check()
    signature = _stat(logfile)
    check()
    key = str(logfile)
    with _LOG_CACHE_LOCK:
        cached = _LOG_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            _LOG_CACHE.move_to_end(key)
        else:
            _LOG_CACHE.pop(key, None)
            cached = None
    if cached is not None:
        for record in cached[1]:
            check()
            yield record
        check()
        if _stat(logfile) != signature:
            raise OSError('Capture log changed during matching; refresh and retry.')
        check()
        return
    records, size = [], sys.getsizeof(())
    with logfile.open('r', encoding='utf-8-sig', errors='replace') as stream:
        for line in stream:
            check()
            if 'capture_started' not in line:
                continue
            try:
                record = _parse_log_capture(json.loads(line))
            except (TypeError, ValueError):
                continue
            if record is None:
                continue
            if records is not None:
                size += _log_capture_bytes(record)
                if size <= _LOG_CACHE_MAX_BYTES:
                    records.append(record)
                else:
                    records = None
            yield record
    check()
    if _stat(logfile) != signature:
        raise OSError('Capture log changed during matching; refresh and retry.')
    check()
    if records is not None:
        with _LOG_CACHE_LOCK:
            check()
            _LOG_CACHE[key] = (signature, tuple(records), size)
            _LOG_CACHE.move_to_end(key)
            while (len(_LOG_CACHE) > _LOG_CACHE_MAX_ENTRIES
                   or sum(item[2] for item in _LOG_CACHE.values()) > _LOG_CACHE_MAX_BYTES):
                _LOG_CACHE.popitem(last=False)


def _capture_start(records, frame_count):
    """Return a verified file-acquisition start, never merge different runs.

    Instrument experiment logs emit one capture_started per CSV point, with a
    run-wide acquisition_index that need not be contiguous for this file.
    Exact duplicate records (for example copied logs) describe the same event.
    Older logs with a single file-level event remain supported.
    """
    if not records:
        return None, ''
    if len(records) == 1:
        record = next(iter(records))
        if record.point_index is None:
            return record.stamp, ''
    runs = {(item.run_id, item.experiment_id) for item in records}
    if len(runs) != 1 or not any(next(iter(runs))):
        return None, 'Conflicting or unidentified capture runs'
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 1:
        return None, 'Cannot verify capture sequence without CSV frame count'
    if any(type(item.point_index) is not int or item.point_index < 1 for item in records):
        return None, 'Conflicting capture records without point indices'
    points = [item.point_index for item in records]
    acquisitions = [item.acquisition_id for item in records]
    if (len(set(points)) != len(points) or not all(acquisitions)
            or len(set(acquisitions)) != len(acquisitions)):
        return None, 'Conflicting repeated capture points or acquisition IDs'
    if (len(records) != frame_count or set(points) != set(range(1, frame_count+1))
            or any(item.frames_requested != 1 for item in records)):
        return None, 'Incomplete or incompatible capture point sequence'
    ordered = sorted(records, key=lambda item: item.point_index)
    if any(second.stamp < first.stamp for first, second in zip(ordered, ordered[1:])):
        return None, 'Conflicting capture point chronology'
    return ordered[0].stamp, ''


def _key(path):
    return str(path.resolve()).casefold()


def _stat(path):
    # Include Windows ChangeTime: st_ctime there is creation time and would
    # miss a same-size log rewrite whose modification timestamp was restored.
    signature = _digest_signature(path)
    if signature is None:
        raise OSError(f'Cannot verify timing source: {path}')
    return signature[1], signature[0], signature


def _condition_comparison(measurement, candidate, relaxed_conditions=()):
    first = parse_drr_acquisition_conditions(measurement.filename)
    second = parse_drr_acquisition_conditions(candidate.filename)
    differences = []
    unknown = []
    for key in ('sample', 'position', *_CONDITIONS):
        left, right = getattr(first, key), getattr(second, key)
        if left is None or right is None:
            unknown.append(key)
        # Legacy matching rejects missing sample/position, and rejects a
        # known value paired with an unknown setting. An explicit relaxation
        # admits that uncertainty but counts it as a reviewable difference.
        missing_identity = key in ('sample', 'position') and (left is None or right is None)
        mismatch = left != right
        if missing_identity or mismatch:
            if key not in relaxed_conditions:
                return None
            left_text = 'unknown' if left is None else left
            right_text = 'unknown' if right is None else right
            differences.append(f'{CONDITION_LABELS[key]}: measurement {left_text}, background {right_text}')
    left = getattr(measurement, 'wavelength_center_nm', None)
    right = getattr(candidate, 'wavelength_center_nm', None)
    if left is None:
        left = extract_wavelength_center_nm(measurement.filename)
    if right is None:
        right = extract_wavelength_center_nm(candidate.filename)
    # Legacy files without a named center remain eligible; the spectral grid
    # is independently required to match for every candidate.
    if left is not None and right is not None and not wavelength_centers_match(left, right):
        if 'wavelength_center_nm' not in relaxed_conditions:
            return None
        differences.append(f'{CONDITION_LABELS["wavelength_center_nm"]}: measurement {left}, background {right}')
    return tuple(differences), tuple(unknown)


def _conditions_match(measurement, candidate):
    return _condition_comparison(measurement, candidate) is not None


class MeasuredBnMatcher:
    """Lazy, cancellable matcher; timing is read once for this worker snapshot.

    Complete unambiguous capture_started timestamps take precedence. Verified
    per-point records from one run use the first point's acquisition start.
    With
    incomplete log coverage, every source uses filesystem mtime, explicitly
    reported as a proxy. Conflicting log records and equal nearest distances
    remain unresolved. Candidate suffix numbers never bind a measurement.
    """

    def __init__(self, root, measurements, candidates, *, cancelled=None, include_capture_contexts=False, relaxed_conditions=()):
        self.root = Path(root).resolve()
        self.measurements = tuple(measurements)
        self.candidates = tuple(candidates)
        self.cancelled = cancelled
        self.include_capture_contexts = include_capture_contexts
        self.relaxed_conditions = normalize_relaxed_conditions(relaxed_conditions)
        self._eligible = None
        self._timing = None
        self._paths = {}

    def _check(self):
        check_load_cancelled()
        token = self.cancelled
        if token is None:
            return
        if hasattr(token, 'check'):
            token.check()
        elif (token.is_set() if hasattr(token, 'is_set') else token()):
            raise LoadCancelled()

    def _path(self, source):
        if source.source not in self._paths:
            self._paths[source.source] = resolve_source_path(self.root, source.source)
        return self._paths[source.source]

    def _prepare(self):
        selected_paths = {_key(self._path(item)) for item in self.measurements}
        candidates = []
        seen = set()
        for item in self.candidates:
            self._check()
            if not is_measured_bn_name(item.filename) or item.gate_varies is not False:
                continue
            path = self._path(item)
            identity = _key(path)
            parts = {part.casefold() for part in path.parts}
            if (identity in seen or identity in selected_paths or path.suffix.casefold() != '.csv'
                    or not path.is_relative_to(self.root) or not path.is_file()
                    or '.drr_bn_calibration' in parts or 'pn backgrounds' in parts):
                continue
            seen.add(identity)
            candidates.append(item)
        eligible = {}
        for measurement in self.measurements:
            self._check()
            eligible[_key(self._path(measurement))] = tuple(
                item for item in candidates
                if _condition_comparison(measurement, item, self.relaxed_conditions) is not None
                and spectral_grids_match(measurement, item))
        self._eligible = eligible

    def _log_paths(self, paths):
        directories = {self.root, self.root/'Initial Data', self.root/'Initial data'}
        for path in paths:
            for directory in (path.parent, path.parent.parent):
                if directory.is_relative_to(self.root):
                    directories.add(directory)
        logs = {}
        for directory in sorted(directories):
            self._check()
            if not directory.is_dir():
                continue
            for path in directory.glob('*.experiment.events.jsonl'):
                resolved = path.resolve()
                if resolved.is_relative_to(self.root) and resolved.is_file():
                    logs[_key(resolved)] = resolved
        return tuple(logs[key] for key in sorted(logs))

    def _event_target(self, output, logfile, targets, basenames):
        raw = Path(output)
        if raw.is_absolute():
            identity = _key(raw)
            return identity if identity in targets else None
        # A qualified relative path must resolve to the supplied source. Do not
        # turn a path in another directory into a match merely by basename.
        if len(raw.parts) > 1:
            matches = {_key(parent/raw) for parent in (self.root, logfile.parent)} & targets
            return next(iter(matches)) if len(matches) == 1 else None
        matches = basenames.get(raw.name.casefold(), ())
        return matches[0] if len(matches) == 1 else None

    def _read_timing(self):
        paths = {}
        frame_counts = {}
        for measurement in self.measurements:
            self._check()
            identity = _key(self._path(measurement))
            eligible = self._eligible[identity]
            if eligible or self.include_capture_contexts:
                paths[identity] = self._path(measurement)
                frame_counts[identity] = measurement.frame_count
                paths.update((_key(self._path(item)), self._path(item)) for item in eligible)
                frame_counts.update((_key(self._path(item)), item.frame_count) for item in eligible)
        try:
            signatures = tuple((key, _stat(path)) for key,path in sorted(paths.items()))
            logs = self._log_paths(tuple(paths.values()))
            log_signatures = tuple((str(path), _stat(path)) for path in logs)
            cache_key = (signatures, log_signatures, tuple(sorted(frame_counts.items())), self.include_capture_contexts)
            with _TIMING_LOCK:
                cached = _TIMING_CACHE.get(cache_key)
                if cached is not None:
                    _TIMING_CACHE.move_to_end(cache_key)
                    return cached
            found = {identity: set() for identity in paths}
            targets = set(paths)
            target_cache = {}
            basenames = {}
            for identity,path in paths.items():
                basenames.setdefault(path.name.casefold(), []).append(identity)
            for logfile in logs:
                self._check()
                log_parent = _key(logfile.parent)
                for record in _capture_log_records(logfile, self._check):
                    # A file normally emits one event per point. Resolve its
                    # identity once per selection, including unrelated outputs.
                    # Basename ambiguity and relative paths depend on this
                    # selection and must never be inherited from another group.
                    target_key = (log_parent, record.output)
                    if target_key not in target_cache:
                        target_cache[target_key] = self._event_target(
                            record.output, logfile, targets, basenames)
                    target = target_cache[target_key]
                    if target is None:
                        continue
                    if record.error:
                        return _Timing({}, '', f'{record.error} for {paths[target].name}.')
                    found[target].add(record.capture)
            self._check()
            if (signatures != tuple((key, _stat(path)) for key,path in sorted(paths.items()))
                    or log_signatures != tuple((str(path), _stat(path)) for path in logs)):
                return _Timing({}, '', 'Measurement, BN reference or capture log changed during matching; refresh and retry.')
            starts, captures = {}, []
            for key, records in found.items():
                self._check()
                start, error = _capture_start(records, frame_counts[key])
                if error:
                    return _Timing({}, '', f'{error} for {paths[key].name}.')
                if start is not None:
                    starts[key] = start
                    runs = {(item.run_id, item.experiment_id) for item in records}
                    if len(runs) == 1:
                        run_id, experiment_id = next(iter(runs))
                        # Experiment IDs can span several runs or days. Only a
                        # verified acquisition-run ID can join separate files.
                        verified = bool(run_id)
                        captures.append((key, MeasuredBnCapture(
                            run_id or '', experiment_id or '', start,
                            max(item.stamp for item in records), verified,
                            ('Verified capture sequence identifies this acquisition run. End is the final recorded capture start, not a completion timestamp.'
                             if verified else 'Capture time exists without a verified acquisition-run ID; an experiment ID alone cannot join files.'))))
            if all(found.values()):
                timing = _Timing(starts, 'capture_started_utc', captures=tuple(captures))
            else:
                times = {key: stamp[1]/1e9 for key,stamp in signatures}
                if not all(math.isfinite(value) and value > 0 for value in times.values()):
                    return _Timing({}, '', 'File modification times are unavailable; choose the measured BN explicitly.')
                timing = _Timing(times, 'filesystem_mtime_proxy', captures=tuple(captures))
            with _TIMING_LOCK:
                if len(timing.times) <= _CACHE_MAX_SOURCES:
                    _TIMING_CACHE[cache_key] = timing
                    while (len(_TIMING_CACHE) > _CACHE_MAX_ENTRIES
                           or sum(len(item.times) for item in _TIMING_CACHE.values()) > _CACHE_MAX_SOURCES):
                        _TIMING_CACHE.popitem(last=False)
            return timing
        except OSError as exc:
            return _Timing({}, '', f'Cannot read measured-BN timing: {exc}')

    def capture_context(self, measurement) -> MeasuredBnCapture:
        """Reuse the same cached log parse used for matching; never scan again."""
        self._check()
        if self._eligible is None:
            self._prepare()
        if self._timing is None:
            self._timing = self._read_timing()
        if self._timing.error:
            return MeasuredBnCapture(evidence=self._timing.error)
        return dict(self._timing.captures).get(_key(self._path(measurement)), MeasuredBnCapture())

    def candidate_times(self, measurement) -> tuple[MeasuredBnCandidateTime, ...]:
        """Eligible candidate distances from the already verified worker clock."""
        self.capture_context(measurement)
        if self._timing.error:
            return ()
        identity = _key(self._path(measurement))
        if identity not in self._timing.times:
            return ()
        eligible = self._eligible.get(identity, ())
        if self.relaxed_conditions and eligible:
            best = min(self._condition_rank(measurement, item) for item in eligible)
            eligible = tuple(item for item in eligible if self._condition_rank(measurement, item) == best)
        return tuple(sorted((MeasuredBnCandidateTime(
            item.source, abs(self._timing.times[_key(self._path(item))]-self._timing.times[identity]))
            for item in eligible), key=lambda item: (item.time_gap_seconds, item.source)))

    def _condition_rank(self, measurement, item):
        differences, unknown = _condition_comparison(measurement, item, self.relaxed_conditions)
        return len(differences), len(unknown)

    def match(self, measurement, *, preferred_source=None) -> MeasuredBnMatch:
        try:
            self._check()
            if self._eligible is None:
                self._prepare()
            identity = _key(self._path(measurement))
            eligible = self._eligible.get(identity, ())
            if not eligible:
                return MeasuredBnMatch(reason='No constant measured BN with matching sample, position, settings and spectral grid.')
            if self._timing is None:
                self._timing = self._read_timing()
            timing = self._timing
            self._check()
            if timing.error:
                return MeasuredBnMatch(reason=timing.error, applicable=True)
            if preferred_source is not None:
                eligible = tuple(item for item in eligible if item.source == preferred_source)
                if not eligible:
                    return MeasuredBnMatch(reason='Manual BN is not an eligible candidate for this measurement.', applicable=True)
            elif self.relaxed_conditions:
                best_rank = min(self._condition_rank(measurement, item) for item in eligible)
                eligible = tuple(item for item in eligible if self._condition_rank(measurement, item) == best_rank)
            ranked = sorted(((abs(timing.times[_key(self._path(item))]-timing.times[identity]), item)
                             for item in eligible), key=lambda pair: pair[0])
            distance, best = ranked[0]
            if len(ranked) > 1 and math.isclose(distance, ranked[1][0], rel_tol=0, abs_tol=1e-6):
                return MeasuredBnMatch(reason='Equally near measured BN references; choose the reference explicitly.', applicable=True)
            basis = ('capture_started UTC (file acquisition start)' if timing.basis == 'capture_started_utc'
                     else 'file modification time proxy (incomplete capture logs)')
            selection = 'Manual selection' if preferred_source is not None else 'Nearest matching measured BN'
            reason = (f'{selection} by {basis}; difference {distance:g} s; '
                      'one CSV, all frames; timing does not establish optical stability')
            differences = _condition_comparison(measurement, best, self.relaxed_conditions)[0]
            if differences:
                reason += '; Relaxed acquisition differences: ' + '; '.join(differences)
            first = parse_drr_acquisition_conditions(measurement.filename)
            second = parse_drr_acquisition_conditions(best.filename)
            unknown = tuple(key for key in _CONDITIONS
                            if getattr(first, key) is None or getattr(second, key) is None)
            return MeasuredBnMatch(DrrMeasurementAssignment(
                measurement_file=measurement.source, baseline_mode='External',
                baseline_files=(best.source,), baseline_which='all', selection_reason=reason), reason, True,
                timing_basis=timing.basis, time_gap_seconds=distance,
                runner_up_gap_seconds=ranked[1][0]-distance if len(ranked) > 1 else None,
                unknown_conditions=unknown, candidate_count=len(ranked),
                relaxed_differences=differences)
        except LoadCancelled:
            return MeasuredBnMatch(reason='Measured-BN matching cancelled.', applicable=True)
