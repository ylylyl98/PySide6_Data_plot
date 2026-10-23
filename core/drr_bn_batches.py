"""Capture-run batches and conservative shared-candidate BN assessment.

Batch membership never depends on the chosen BN. Joint timing evidence cannot
overrule a conflicting individual nearest reference or certify optical stability.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math

from core.drr_baseline_candidates import parse_drr_acquisition_conditions
from core.drr_measured_bn import MeasuredBnCapture, _CONDITIONS


BATCH_POLICY_VERSION = 'verified_capture_run_joint_time_consensus_v1'
_STATUSES = {'consensus', 'conflict', 'unverified', 'unresolved'}


@dataclass(frozen=True)
class BnMatchBatch:
    batch_id: str
    measurement_files: tuple[str, ...]
    baseline_file: str = ''
    status: str = 'unverified'
    evidence: str = ''
    warnings: tuple[str, ...] = ()
    anomaly_measurements: tuple[str, ...] = ()
    capture_start: float | None = None
    capture_end: float | None = None
    run_id: str = ''
    experiment_id: str = ''
    verified_run: bool = False
    candidate_files: tuple[str, ...] = ()
    aggregate_time_gaps: tuple[tuple[str, float], ...] = ()

    @property
    def id(self):
        return self.batch_id

    @property
    def measurement_count(self):
        return len(self.measurement_files)

    def to_dict(self):
        return dict(batch_id=self.batch_id, measurement_files=list(self.measurement_files),
            baseline_file=self.baseline_file, status=self.status, evidence=self.evidence,
            warnings=list(self.warnings), anomaly_measurements=list(self.anomaly_measurements),
            capture_start=self.capture_start, capture_end=self.capture_end,
            run_id=self.run_id, experiment_id=self.experiment_id, verified_run=self.verified_run,
            candidate_files=list(self.candidate_files), aggregate_time_gaps=[list(item) for item in self.aggregate_time_gaps])

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError('BN batch must be an object.')
        for field in ('batch_id', 'baseline_file', 'status', 'evidence', 'run_id', 'experiment_id'):
            if not isinstance(value.get(field), str):
                raise ValueError(f'BN batch {field} must be a string.')
        if not value['batch_id'] or value['status'] not in _STATUSES or type(value.get('verified_run')) is not bool:
            raise ValueError('Invalid BN batch identity/status.')
        sequences = {}
        for field in ('measurement_files', 'warnings', 'anomaly_measurements', 'candidate_files'):
            raw = value.get(field)
            if not isinstance(raw, (list, tuple)) or any(not isinstance(item, str) for item in raw):
                raise ValueError(f'BN batch {field} must be a list of strings.')
            sequences[field] = tuple(raw)
        members = sequences['measurement_files']
        if not members or len(set(members)) != len(members) or not set(sequences['anomaly_measurements']).issubset(members):
            raise ValueError('Invalid BN batch measurement membership.')
        for field in ('capture_start', 'capture_end'):
            number = value.get(field)
            if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float))
                                       or not math.isfinite(number) or number <= 0):
                raise ValueError('Invalid BN batch capture timestamp.')
        if value['verified_run'] and not value['run_id']:
            raise ValueError('Verified BN batch requires a capture-run identity.')
        if (value['capture_start'] is not None and value['capture_end'] is not None
                and value['capture_end'] < value['capture_start']):
            raise ValueError('BN batch capture time range is reversed.')
        if value['status'] == 'consensus' and (not value['verified_run'] or not value['baseline_file'] or sequences['anomaly_measurements']):
            raise ValueError('BN batch consensus requires a verified run, one BN and no timing anomaly.')
        if value['status'] in ('conflict', 'unresolved') and value['baseline_file']:
            raise ValueError('A conflicting or unresolved BN batch cannot force a common reference.')
        if not value['verified_run'] and len(members) != 1:
            raise ValueError('Unknown capture runs must remain separate singletons.')
        if value['baseline_file'] and value['baseline_file'] not in sequences['candidate_files']:
            raise ValueError('BN batch consensus is outside shared candidate eligibility.')
        raw_gaps = value.get('aggregate_time_gaps')
        if not isinstance(raw_gaps, (list, tuple)):
            raise ValueError('BN joint candidate scores must be a list.')
        gaps = []
        for item in raw_gaps:
            if (not isinstance(item, (list, tuple)) or len(item) != 2 or item[0] not in sequences['candidate_files']
                    or isinstance(item[1], bool) or not isinstance(item[1], (int, float))
                    or not math.isfinite(item[1]) or item[1] < 0):
                raise ValueError('Invalid BN joint candidate timing evidence.')
            gaps.append((item[0], float(item[1])))
        if (len(set(sequences['candidate_files'])) != len(sequences['candidate_files'])
                or len({name for name, _ in gaps}) != len(gaps)
                or {name for name, _ in gaps} != set(sequences['candidate_files'])):
            raise ValueError('BN joint timing evidence must cover shared candidate eligibility exactly once.')
        if value['baseline_file'] and (not gaps or gaps[0][0] != value['baseline_file']):
            raise ValueError('BN batch consensus and joint timing winner disagree.')
        return cls(value['batch_id'], members, value['baseline_file'], value['status'], value['evidence'],
            sequences['warnings'], sequences['anomaly_measurements'], value.get('capture_start'), value.get('capture_end'),
            value['run_id'], value['experiment_id'], value['verified_run'], sequences['candidate_files'], tuple(gaps))


def _compatibility_key(source):
    conditions = parse_drr_acquisition_conditions(source.filename)
    return (tuple(getattr(conditions, field) for field in ('sample', 'position', *_CONDITIONS)),
            source.spectral_grid, source.frame_count, source.gate_grid,
            source.gate_direction, source.gate_labels)


def assess_bn_batches(root, selected, rows, measurements, matcher, *, max_gap_seconds,
                      min_margin_seconds, check=lambda: None):
    """Return an exact selected-file partition and annotated per-file rows.

    ``matcher`` supplies the existing cached log snapshot, not a second parser.
    A conflict never forces the aggregate winner onto an individual member.
    """
    row_map = {row.measurement_file: row for row in rows}
    groups, contexts, timing = {}, {}, {}
    for name in selected:
        check()
        source = measurements.get(name)
        context = matcher.capture_context(source) if matcher is not None and source is not None else MeasuredBnCapture()
        contexts[name] = context
        timing[name] = {item.source: item.time_gap_seconds for item in matcher.candidate_times(source)} if matcher is not None and source is not None else {}
        compatibility = _compatibility_key(source) if source is not None else ()
        key = ('run', context.run_id, context.experiment_id, compatibility) if context.verified_run else ('singleton', name)
        groups.setdefault(key, []).append(name)
    result, annotated = [], dict(row_map)
    for key, members in groups.items():
        check()
        verified = key[0] == 'run'
        context = contexts[members[0]]
        identity = json.dumps([str(root), key], separators=(',', ':'), ensure_ascii=True)
        batch_id = ('run-' if verified else 'single-')+hashlib.sha256(identity.encode()).hexdigest()[:16]
        shared = set(timing[members[0]])
        for name in members[1:]:
            shared.intersection_update(timing[name])
        gaps = tuple(sorted(((candidate, sum(timing[name][candidate] for name in members)) for candidate in shared),
                            key=lambda item: (item[1], item[0])))
        warnings, anomalies = [warning for name in members for warning in row_map[name].warnings], set()
        missing = [name for name in members if row_map[name].assignment is None]
        chosen = {row_map[name].baseline_file for name in members if row_map[name].assignment is not None}
        ambiguous = (len(gaps) > 1 and math.isclose(gaps[0][1], gaps[1][1], rel_tol=0, abs_tol=1e-6))
        for name in members:
            row = row_map[name]
            if row.time_gap_seconds is not None and row.time_gap_seconds > max_gap_seconds:
                anomalies.add(name)
                warnings.append('BN exceeds the maximum review time gap for '+name+'.')
            if row.runner_up_gap_seconds is not None and row.runner_up_gap_seconds < min_margin_seconds:
                anomalies.add(name)
                warnings.append('BN timing alternatives are close for '+name+'.')
        if not verified:
            warnings.append('No verified capture-run identity. This measurement is an unverified singleton; a shared date or BN does not establish a batch.')
        if missing:
            status, baseline = 'unresolved', ''
            anomalies.update(missing)
            warnings.append('Some batch members have no unambiguous compatible BN; all selected measurements are retained.')
        elif not gaps or len(chosen) != 1 or ambiguous or next(iter(chosen)) != gaps[0][0]:
            status, baseline = 'conflict', ''
            anomalies.update(members)
            warnings.append('Individual nearest BN choices and joint shared-candidate timing do not give one unambiguous consensus. Per-file suggestions are retained for explicit review; no majority choice was forced.')
        elif anomalies:
            status, baseline = 'conflict', ''
            warnings.append('Joint timing requires anomaly review; per-file suggestions are retained without forcing a batch reference.')
        else:
            status, baseline = ('consensus' if verified else 'unverified'), next(iter(chosen))
        starts = [contexts[name].capture_start for name in members if contexts[name].capture_start is not None]
        ends = [contexts[name].capture_end for name in members if contexts[name].capture_end is not None]
        evidence = ('Verified capture run and matching acquisition settings/grids define these selected batch members. '
                    if verified else 'Unverified capture identity; no multi-file batch was inferred. ')
        evidence += ('Shared eligible BNs were judged by the sum of absolute per-member time gaps. '
                     if gaps else 'No shared eligible candidate timing is available. ')
        evidence += 'Timing consensus does not establish optical stability or maximize physical SNR.'
        batch = BnMatchBatch(batch_id, tuple(members), baseline, status, evidence,
            tuple(dict.fromkeys(warnings)), tuple(name for name in members if name in anomalies),
            min(starts) if starts else None, max(ends) if ends else None,
            context.run_id if verified else '', context.experiment_id if verified else '', verified,
            tuple(candidate for candidate, _ in gaps), gaps)
        result.append(batch)
        for name in members:
            annotated[name] = replace(row_map[name], batch_id=batch_id, batch_status=status,
                anomaly=name in anomalies, warnings=tuple(dict.fromkeys((*row_map[name].warnings, *batch.warnings))))
    return tuple(result), [annotated[name] for name in selected]
