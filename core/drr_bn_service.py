"""Read native spectra for group BN comparison; never fit a displayed DR/R cube."""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np
import scipy

from core import processing_run as P
from core.drr_baseline_candidates import parse_drr_acquisition_conditions
from core.drr_bn_calibration import BnCalibrationSettings, evaluate_bn_group
from core.drr_bn_store import (DrrBnStore, canonical_json, file_digest, is_bn_calibration_source,
                               source_records, validate_distinct_measurement_contents, joint_validation_passed)
from core.drr_composite_background import is_composite_background
from core.drr_sources import inspect_csv_spectral_grid, resolve_source_path
from core.load_cancellation import LoadCancelled, check_load_cancelled


def _check(cancellation):
    check_load_cancelled()
    if cancellation is None:
        return
    if hasattr(cancellation, 'check'):
        cancellation.check()
    elif callable(cancellation) and cancellation():
        raise LoadCancelled()
    elif hasattr(cancellation, 'is_set') and cancellation.is_set():
        raise LoadCancelled()


def _emit(target, value):
    if target is not None:
        if hasattr(target, 'emit'):
            target.emit(value)
        else:
            target(value)


def _algorithm_identity():
    from core.drr_bn_fingerprint import module_fingerprint, python_runtime_identity
    names = ('core.drr_bn_service', 'core.drr_bn_calibration', 'core.drr_bn_joint', 'core.drr_bn_d2_objective',
             'core.processing', 'core.processing_run', 'core.loader',
             'core.drr_segmented_derivative', 'core.drr_sources',
             'core.drr_bn_store', 'core.drr_bn_fingerprint')
    return dict(protocol='d2_shrinkage_v1', fingerprint_schema_version=1,
                module_fingerprints={name: module_fingerprint(name) for name in names},
                python_runtime=python_runtime_identity(),
                numpy_version=np.__version__, scipy_version=scipy.__version__)


def _ordering(root, records, cancellation):
    """Use relevant local capture logs; otherwise label filesystem order a proxy.

    Only this experiment root and the selected files' in-project directories
    are searched, never sibling projects or an arbitrary filesystem ancestor.
    """
    root = Path(root).resolve()
    by_name = {}
    for item in records:
        by_name.setdefault(Path(item['path']).name, []).append(item['path'])
    directories = {root, root / 'Initial Data', root / 'Initial data'}
    for item in records:
        parent = resolve_source_path(root, item['path']).parent
        if parent.is_relative_to(root):
            directories.add(parent)
            if parent.parent.is_relative_to(root):
                directories.add(parent.parent)
    found = {item['path']: [] for item in records}
    evidence = []
    for directory in sorted(directories):
        if not directory.is_dir():
            continue
        for logfile in sorted(directory.glob('*.experiment.events.jsonl')):
            _check(cancellation)
            used = False
            with logfile.open('r', encoding='utf-8-sig', errors='replace') as stream:
                for line in stream:
                    _check(cancellation)
                    # Acquisition logs can contain large settings snapshots.
                    if 'capture_started' not in line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get('event') != 'capture_started':
                            continue
                        output = str(event.get('output', {}).get('file', ''))
                        matches = by_name.get(Path(output).name, ())
                        if len(matches) != 1:
                            continue
                        target = resolve_source_path(root, matches[0])
                        if Path(output).is_absolute() and Path(output).resolve() != target:
                            continue
                        stamp = event.get('started_utc', event.get('utc'))
                        value = datetime.fromisoformat(str(stamp).replace('Z', '+00:00')).timestamp()
                        found[matches[0]].append(value)
                        used = True
                    except (ValueError, TypeError, KeyError):
                        continue
            if used:
                evidence.append(dict(path=str(logfile), sha256=file_digest(logfile)))
    if all(found[item['path']] for item in records):
        times = {name: min(values) for name, values in found.items()}
        ordered = sorted(records, key=lambda item: (times[item['path']], item['path']))
        return ordered, dict(basis='capture_started_utc', source_logs=evidence,
                             first_capture_unix_seconds=times,
                             limitation='Logs identify saved output names; they do not prove immutable acquisition contents or stable experimental state.')
    ordered = sorted(records, key=lambda item: (item['mtime_ns'], item['path']))
    return ordered, dict(basis='filesystem_mtime_proxy', source_logs=evidence,
                         mtime_ns={item['path']: item['mtime_ns'] for item in records},
                         limitation='Missing complete capture logs: file modification order is a proxy, not verified acquisition chronology.')


def _compatibility(measurements, baseline_files):
    bn_pattern = re.compile(r'(?<![a-z0-9])bn(?:back(?:ground)?)(?=$|[^a-z])', re.I)
    if any(not bn_pattern.search(Path(name).stem) for name in baseline_files):
        raise ValueError('BN comparison requires measured BNback/BNbackground reference files; P/N or other generated backgrounds are not BN inputs.')
    parsed = {name: asdict(parse_drr_acquisition_conditions(name)) for name in (*measurements, *baseline_files)}
    unknown = set()
    for field in next(iter(parsed.values())):
        known = {v[field] for v in parsed.values() if v[field] is not None}
        if len(known) > 1:
            raise ValueError(f'Measurement and BN acquisition condition differs: {field}.')
        if any(v[field] is None for v in parsed.values()):
            unknown.add(field)
    return dict(parsed=parsed, unknown_fields=sorted(unknown),
                interpretation='Known conflicts rejected; unknown filename conditions are not asserted to match.')


def compare_bn_group(folder, measurements, baseline_files, which='all', settings=None, *,
                     y_axis='auto', progress=None, log=None, cancellation=None):
    root = Path(folder).resolve()
    measurements, baseline_files = tuple(measurements), tuple(baseline_files)
    _check(cancellation)
    if which not in ('first', 'last', 'all'):
        raise ValueError('BN frame method must be first, last or all.')
    if not measurements or not baseline_files:
        raise ValueError('Select a measurement group and its measured BN background first.')
    store = DrrBnStore(root)
    if any(is_bn_calibration_source(root, name) for name in baseline_files):
        if len(baseline_files) != 1:
            raise ValueError('Select one calibrated BN version to review its original reference.')
        version = store.version_for_csv(baseline_files[0])
        expected = {resolve_source_path(root, v['path']) for v in version.source_identity['measurements']}
        if expected != {resolve_source_path(root, name) for name in measurements}:
            raise ValueError('This calibrated BN belongs to another measurement group.')
        baseline_files = tuple(v['path'] for v in version.source_identity['baseline_files'])
        which = version.source_identity['baseline_which']
    if any(is_bn_calibration_source(root, name) or is_composite_background(resolve_source_path(root, name))
           for name in (*measurements, *baseline_files)):
        raise ValueError('Compare raw measurement spectra with measured BN files; generated or composite references cannot be recalibrated.')
    if {resolve_source_path(root, n) for n in measurements} & {resolve_source_path(root, n) for n in baseline_files}:
        raise ValueError('A measurement cannot be its own BN reference.')
    compatibility = _compatibility(measurements, baseline_files)
    if settings is None:
        config = BnCalibrationSettings()
    elif isinstance(settings, BnCalibrationSettings):
        config = settings
    else:
        config = BnCalibrationSettings(**dict(settings))
    settings_dict = json.loads(canonical_json(asdict(config)))
    m_records = source_records(root, measurements)
    validate_distinct_measurement_contents(m_records)
    b_records = source_records(root, baseline_files)
    ordered, ordering = _ordering(root, m_records, cancellation)
    ordered_names = [v['path'] for v in ordered]
    columns = tuple(inspect_csv_spectral_grid(resolve_source_path(root, measurements[0])))
    if len(set(np.round(columns, 9))) != len(columns):
        raise ValueError('BN comparison requires unique native spectral columns.')
    for name in (*measurements, *baseline_files):
        _check(cancellation)
        if tuple(inspect_csv_spectral_grid(resolve_source_path(root, name))) != columns:
            raise ValueError('All measurement and BN files must use the same complete ordered native spectral grid.')
    identity = dict(schema_version=2, root=str(root), measurements=m_records, baseline_files=b_records,
                    baseline_which=which, spectral_columns=list(columns), y_axis=str(y_axis),
                    repeat_order=ordered_names, ordering=ordering,
                    algorithm=_algorithm_identity())
    cached = store.find_matching(identity, settings_dict)
    if cached is not None:
        payload = store.load_result(cached)
        _emit(log, 'Reusing the unchanged group BN comparison; no new fit.')
        return payload
    _emit(log, 'Reading native measurement intensities and the selected BN frames.')
    snapshots = {}; data = []; gate_coordinates = None; e = None; g = None
    for i, name in enumerate(ordered_names):
        _check(cancellation)
        energy, bg, tg, bias, intensity = P._parse_raw_spectrum(resolve_source_path(root, name))
        resolved = P._resolve_axis_choice(y_axis=y_axis, vbg=bg, vtg=tg, vbias=bias,
                                          parts=re.findall(r'\$(.*?)\$', name), stem=Path(name).stem)
        gates = np.asarray(resolved['gate_axis'], float)
        coordinates = [np.asarray(bg), np.asarray(tg)] + ([] if bias is None else [np.asarray(bias)])
        if e is None:
            e, g, gate_coordinates = np.asarray(energy), gates, coordinates
        elif (not np.array_equal(e, energy) or not np.array_equal(g, gates)
              or len(coordinates) != len(gate_coordinates)
              or any(not np.array_equal(a, b, equal_nan=True) for a, b in zip(coordinates, gate_coordinates))):
            raise ValueError('Group repeats must share exact native energy and acquisition-order gate coordinates; no gate resampling is performed.')
        data.append(np.asarray(intensity, float))
        snapshots['measurement_' + str(i)] = np.asarray(intensity, float)
        snapshots['measurement_' + str(i) + '_coordinates'] = np.asarray(coordinates)
        _emit(progress, int(35 * (i + 1) / len(ordered_names)))
    selected_vectors = []; first_halves = []; last_halves = []; half_available = which == 'all'; bn_gates = None; bn_bias = None
    for i, name in enumerate(baseline_files):
        _check(cancellation)
        energy, bg, tg, bias, intensity = P._parse_raw_spectrum(resolve_source_path(root, name))
        if not np.array_equal(e, energy):
            raise ValueError('BN energy grid differs from the measurement native grid.')
        constants = tuple(float(np.asarray(v)[0]) for v in (bg, tg))
        if (not np.all(np.asarray(bg) == constants[0]) or not np.all(np.asarray(tg) == constants[1])
                or (bn_gates is not None and constants != bn_gates)):
            raise ValueError('Measured BN references must have one consistent constant gate condition.')
        bn_gates = constants
        if bias is not None and np.isfinite(bias).any():
            bias = np.asarray(bias, float)
            if (not np.isfinite(bias).all() or not np.all(bias == bias[0])
                    or (bn_bias is not None and float(bias[0]) != bn_bias)):
                raise ValueError('Measured BN references must have one consistent constant bias condition when Vbias is recorded.')
            bn_bias = float(bias[0])
        intensity = np.asarray(intensity, float)
        snapshots['baseline_' + str(i)] = intensity
        selected_vectors.append(intensity.mean(0) if which == 'all' else intensity[0 if which == 'first' else -1])
        if len(intensity) >= 2 and len(intensity) % 2 == 0:
            first_halves.append(intensity[:len(intensity)//2].mean(0))
            last_halves.append(intensity[len(intensity)//2:].mean(0))
        else:
            half_available = False
    background = np.mean(selected_vectors, axis=0)
    halves = np.stack([np.mean(first_halves, axis=0), np.mean(last_halves, axis=0)]) if half_available else None
    _emit(progress, 40)
    result = evaluate_bn_group(e, np.stack(data), background, gates=g, settings=config,
                               repeat_order=np.arange(len(data)), background_repeats=halves, cancellation=cancellation)
    _check(cancellation)
    # The selected candidate is stored for review even when the applied method
    # has conservatively fallen back to the original reference.
    report = dict(result.report)
    report['source_ordering'] = ordering
    report['acquisition_condition_check'] = compatibility
    report['background_frames'] = dict(which=which, file_weighting='equal weight per file after selected-frame averaging',
                                       independent_halves_available=half_available,
                                       limitation=('Disjoint frame halves test within-reference sensitivity; they are not independent acquisition sessions.'
                                                   if half_available else 'First/last selection or odd/fewer-than-two frame count: equal-weight BN halves unavailable.'))
    if result.eligible and not joint_validation_passed(report, settings_dict):
        raise ValueError('BN engine returned acceptance without complete joint DR/R and D2E evidence.')
    candidates = []
    for name in result.candidate_backgrounds:
        candidate_report = report.get('candidates', {}).get(name, {})
        score = candidate_report.get('selection', {})
        if name == result.selected_method:
            candidate_reason = '; '.join(result.reasons)
        elif name == 'original':
            candidate_reason = 'Original measured BN.'
        else:
            reasons = list(candidate_report.get('selection_joint', {}).get('reasons', []))
            reasons.extend(candidate_report.get('training_influence_audit', {}).get('reasons', []))
            parameter_reason = candidate_report.get('parameters', {}).get('reason')
            if parameter_reason:
                reasons.insert(0, parameter_reason)
            candidate_reason = '; '.join(dict.fromkeys(reasons)) or 'Selection-set comparison only; not the final validated choice.'
        candidates.append(dict(name=name, improvement_fraction=score.get('mean_improvement_fraction'),
                               family=candidate_report.get('parameters', {}).get('family', name),
                               correction_strength=candidate_report.get('parameters', {}).get('correction_strength'),
                               improvement_fraction_d2=candidate_report.get('selection_d2', {}).get('mean_improvement_fraction'),
                               joint_passed=bool(candidate_report.get('selection_joint', {}).get('passed', False)),
                               drr_noninferiority_passed=bool(candidate_report.get('selection_joint', {}).get('drr_noninferiority_passed', False)),
                               eligible=bool(result.eligible and name == result.selected_method),
                               reason=candidate_reason))
    summary = (f'{result.status}: {result.selected_method}. ' + '; '.join(result.reasons)).rstrip('. ')
    payload = dict(report=report, energy=e.copy(), gates=g.copy(), original_background=background.copy(),
                   candidate_background=result.candidate_backgrounds[result.selected_method].copy(),
                   candidate_backgrounds={k: v.copy() for k, v in result.candidate_backgrounds.items()},
                   mean_raw=np.mean(data, axis=0), eligible=bool(result.eligible), selected_candidate=result.selected_method,
                   summary=summary, candidates=candidates, settings=settings_dict, source_identity=identity,
                   status=result.status, reasons=list(result.reasons), input_snapshots=snapshots, cached=False, version_id=None)
    for item in (*m_records, *b_records):
        _check(cancellation)
        if file_digest(resolve_source_path(root, item['path'])) != item['sha256']:
            raise ValueError('Source changed during BN comparison: ' + item['path'])
    _emit(progress, 100)
    return payload
