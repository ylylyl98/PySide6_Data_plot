"""Immutable group-scoped BN comparisons and explicit activation records."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from core.drr_sources import portable_source_name, resolve_source_path
from core.load_cancellation import check_load_cancelled

BN_DIRECTORY = '.drr_bn_calibration'
BN_MARKER = 'BNCalibrationVersion'
BN_KIND = 'group_bn_calibration'
_VERIFIED_VERSION = object()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def optimization_policy_current(report):
    """Recognize this policy even when its comparison failed validation."""
    if not isinstance(report, dict):
        return False
    joint = report.get('joint_validation')
    return (report.get('schema_version') == 4
            and report.get('optimization_policy') == 'd2_shrinkage_v1'
            and isinstance(joint, dict) and joint.get('schema_version') == 3
            and joint.get('policy') == 'd2_shrinkage_v1')


def _candidate_family_strength(name):
    for family in ('local_shift', 'local_shift_gain'):
        if name == family:
            return family, 1.
        for suffix, strength in (('s25', .25), ('s50', .5), ('s75', .75)):
            if name == family + '_' + suffix:
                return family, strength
    raise ValueError('Unsupported BN candidate.')


def _probe_errors_valid(evidence, count):
    probes = evidence['probes']
    if len(probes) != count:
        return False
    for product in ('drr', 'd2'):
        values = [probe[product + '_error_fraction'] for probe in probes]
        maximum = evidence['max_' + product + '_error_fraction']
        if (not np.isfinite([maximum, *values]).all() or min(values) < 0
                or max(values) > .05 or not np.isclose(maximum, max(values), rtol=0, atol=1e-12)):
            return False
    return True


def _referenced_probes_valid(evidence, influence):
    """Bind real refits and selector reruns to the coarse/refined probes checked."""
    references = evidence['chosen_probe_references']
    if len(references) != 3 or len(evidence['probes']) != 3:
        return False
    seen, checked = set(), []
    for reference, probe in zip(references, evidence['probes']):
        source, index = reference['source'], reference['probe_index']
        if (source not in ('coarse', 'refinement') or type(index) is not int
                or (source, index) in seen):
            return False
        probes = influence['screening' if source == 'coarse' else 'refinement']['probes']
        if not 0 <= index < len(probes):
            return False
        seen.add((source, index))
        checked.append(probes[index])
        if (probe.get('probe_source') != source or probe.get('probe_index') != index
                or any(probe.get(key) != probes[index].get(key) for key in
                       ('center_energy', 'sigma_energy', 'sign'))):
            return False
    combined = influence['screening']['probes'] + influence['refinement']['probes']
    return all(max(p[product + '_error_fraction'] for p in checked)
               == max(p[product + '_error_fraction'] for p in combined)
               for product in ('drr', 'd2'))


def _refinement_passed(audit):
    """Require the declared native-grid coverage, including every generated probe."""
    refinement = audit['refinement']
    offsets, widths = [-1., -.5, 0., .5, 1.], [1., 1.25, 1.5, 2., 2.5, 3.]
    if (refinement.get('schema_version') != 1 or refinement.get('policy') != 'native_feature_grid_v1'
            or refinement.get('evaluated') is not True or refinement.get('passed') is not True
            or refinement.get('max_seed_centers') != 3 or refinement.get('max_probe_count') != 168
            or refinement.get('center_offsets_native_spacings') != offsets
            or refinement.get('sigma_native_spacings') != widths or refinement.get('signs') != [-1, 1]):
        return False
    positions, step = audit['positions_energy'], audit['native_median_spacing']
    bounds = refinement['center_bounds_energy']
    if (not np.isfinite(step) or step <= 0 or len(bounds) != 2
            or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]
            or positions[0] < bounds[0] or positions[-1] > bounds[1]):
        return False
    coarse = audit['screening']['probes']
    used = set()
    for i, probe in enumerate(coarse):
        scale, sign = (1.5, 3.)[(i % 4)//2], (-1, 1)[i % 2]
        expected = (positions[i//4], scale*step, sign)
        if (probe.get('probe_source') != 'coarse' or probe.get('probe_index') != i
                or probe.get('sigma_native_spacings') != scale
                or tuple(probe[key] for key in ('center_energy', 'sigma_energy', 'sign')) != expected):
            return False
        used.add(expected)
    ranked = sorted(range(17), key=lambda i: (
        -max(max(p['drr_error_fraction'], p['d2_error_fraction']) for p in coarse[4*i:4*i+4]), i))[:3]
    if (refinement['seed_center_indices'] != ranked
            or refinement['seed_centers_energy'] != [positions[i] for i in ranked]):
        return False
    expected_probes = []
    for seed in ranked:
        for offset in offsets:
            center = float(np.clip(positions[seed]+offset*step, *bounds))
            for scale in widths:
                for sign in (-1, 1):
                    key = (center, scale*step, sign)
                    if key not in used:
                        expected_probes.append(dict(center_energy=center, sigma_energy=scale*step,
                            sigma_native_spacings=scale, sign=sign, seed_center_index=seed,
                            center_offset_native_spacings=offset))
                        used.add(key)
    count = len(expected_probes)
    if (not 1 <= count <= 168 or refinement.get('probe_count') != count
            or refinement.get('expected_probe_count') != count or not _probe_errors_valid(refinement, count)):
        return False
    return all(probe.get('probe_source') == 'refinement' and probe.get('probe_index') == i
               and all(probe.get(key) == value for key, value in expected.items())
               for i, (probe, expected) in enumerate(zip(refinement['probes'], expected_probes)))


def _selection_sensitivity_passed(audit, selected, family, strength, split, influence):
    if (audit.get('schema_version') != 2 or audit.get('policy') != 'pre_final_selection_sensitivity_v2'
            or audit.get('passed') is not True or audit.get('selection_aware') is not True
            or audit.get('pre_final') is not True or audit.get('final_repeats_read') is not False
            or audit.get('training_only') is not False
            or audit.get('train_repeat_indices') != split['train']
            or audit.get('selection_repeat_indices') != split['selection']
            or audit.get('baseline_selected_method') != selected
            or audit.get('baseline_family') != family
            or audit.get('baseline_correction_strength') != strength
            or audit.get('amplitude') != 2e-4 or audit.get('max_relative_error') != .05
            or audit.get('count') != 3 or not _probe_errors_valid(audit, 3)):
        return False
    qualified = audit['baseline_qualified_candidates']
    if selected not in qualified or len(set(qualified)) != len(qualified):
        return False
    for name in qualified:
        _candidate_family_strength(name)
    if not _referenced_probes_valid(audit, influence):
        return False
    for probe in audit['probes']:
        if probe.get('passed') is not True or probe.get('baseline_selected_method') != selected:
            return False
        winner = probe['perturbed_selected_method']
        expected = ('original', 0.) if winner == 'original' else _candidate_family_strength(winner)
        if ((probe.get('perturbed_family'), probe.get('perturbed_correction_strength')) != expected
                or probe.get('original_fallback_jump_included') is not (winner == 'original')):
            return False
    return True


def joint_validation_passed(report, settings=None):
    """Only the current D2E-priority policy certifies a processed reference.

    Numerical decisions belong to the engine. This checks the saved evidence
    contract before any caller can activate or reuse its acceptance flag.
    """
    try:
        joint = report['joint_validation']
        derivative = report['derivative_settings']
        if (not optimization_policy_current(report) or report.get('selection_frozen') is not True
                or joint.get('passed') is not True
                or joint.get('d2_training_evaluated') is not True
                or joint.get('fixed_bn_basis') is not True
                or joint.get('derivative_checked') is not True
                or derivative.get('processing') != 'production SG energy derivative'):
            return False
        for role in ('selection', 'final'):
            if not all(joint[role].get(flag) is True for flag in
                       ('passed', 'derivative_checked', 'drr_noninferiority_passed', 'd2_benefit_passed')):
                return False
        for role in ('training', 'selection', 'final'):
            protection = joint['protection'][role]
            if (protection.get('passed') is not True or protection.get('derivative_checked') is not True
                    or any(protection[product].get('checked') is not True
                           or protection[product].get('passed') is not True for product in ('drr', 'd2'))):
                return False
        requested, used, poly = (derivative[key] for key in
                                ('window_length', 'used_window_length', 'polyorder'))
        if (any(type(v) is not int for v in (requested, used, poly)) or poly < 2
                or requested < 3 or used < 3 or used % 2 != 1 or used <= poly):
            return False
        if settings is not None and (settings.get('derivative_window') != requested
                                     or settings.get('derivative_polyorder') != poly):
            return False
        selected = report['selected_method']
        family, strength = _candidate_family_strength(selected)
        parameters = report['candidates'][selected]['parameters']
        if (parameters.get('family') != family or parameters.get('correction_strength') != strength):
            return False
        training = parameters['training_objective']
        if (training.get('policy') != 'd2_shrinkage_v1'
                or training.get('metric') != 'BN-projected production D2 residual'
                or training.get('d2_training_evaluated') is not True
                or training.get('fixed_bn_basis') is not True
                or training.get('basis_source') != 'BN-only shift/gain tangent responses'
                or training.get('context') != 'full training-repeat spectra; score band is not an independent energy holdout'
                or type(training.get('basis_rank')) is not int
                or not 1 <= training['basis_rank'] <= 2
                or (family == 'local_shift_gain' and training['basis_rank'] != 2)
                or training.get('used_window_length') != used
                or training.get('repeat_indices') != report['split']['train']):
            return False
        for key in ('repeat_indices', 'gate_indices', 'channel_indices', 'score_channel_indices'):
            indices = training[key]
            if (not isinstance(indices, list) or not indices
                    or any(type(v) is not int or v < 0 for v in indices)
                    or len(set(indices)) != len(indices)):
                return False
        if (training['channel_indices'] != list(range(len(training['channel_indices'])))
                or not set(training['score_channel_indices']).issubset(training['channel_indices'])):
            return False
        shrinkage = parameters['shrinkage']
        full = parameters['unshrunk_background']
        full_parameters = parameters['unshrunk_parameters']
        if (shrinkage.get('post_fit') is not True or shrinkage.get('alpha_is_fitted') is not False
                or shrinkage.get('training_objective_background') != 'unshrunk fitted BN'
                or full.get('point_count') != len(training['channel_indices'])
                or not isinstance(full.get('sha256'), str) or len(full['sha256']) != 64
                or any(character not in '0123456789abcdef' for character in full['sha256'])
                or full_parameters.get('optimizer_success') is not True
                or full_parameters.get('near_parameter_bound') is not False
                or any(full_parameters.get(key) != parameters.get(key) for key in ('shift_pixel', 'fine_gain'))):
            return False
        values = [full['log_correction_rms'], full['max_absolute_log_correction']]
        if not np.isfinite(values).all() or min(values) < 0 or values[0] > values[1] + 1e-12:
            return False
        bands = training['fit_bands']
        if not isinstance(bands, list) or not bands:
            return False
        scored = []
        for band in bands:
            channels, region = band['channel_indices'], band['region']
            if (not isinstance(channels, list) or len(channels) < 3
                    or any(type(i) is not int or i < 0 for i in channels)
                    or channels != sorted(set(channels)) or band['channel_count'] != len(channels)
                    or len(region) != 2 or not np.isfinite(region).all() or region[0] >= region[1]
                    or not np.isfinite(band['energy_span']) or band['energy_span'] <= 0
                    or band['energy_span'] > region[1]-region[0]+1e-12):
                return False
            scored.extend(channels)
        if (len(set(scored)) != len(scored)
                or set(scored) != set(training['score_channel_indices'])):
            return False
        if settings is not None and settings.get('train_regions'):
            if [list(band['region']) for band in bands] != [list(region) for region in settings['train_regions']]:
                return False
        bandwidth = training['bandwidth_check']
        span, footprint = bandwidth['scoring_energy_span'], bandwidth['sg_footprint_energy']
        if (bandwidth.get('passed') is not True
                or type(bandwidth.get('score_channel_count')) is not int
                or bandwidth.get('total_score_channel_count') != len(training['score_channel_indices'])
                or bandwidth['score_channel_count'] > bandwidth['total_score_channel_count']
                or type(bandwidth.get('required_channel_count')) is not int
                or bandwidth['required_channel_count'] < used
                or bandwidth['score_channel_count'] < bandwidth['required_channel_count']
                or not np.isfinite([span, footprint]).all()
                or min(span, footprint) <= 0 or footprint > span + 1e-12):
            return False
        audit = joint['influence_audit']
        if (audit.get('schema_version') != 3 or audit.get('policy') != 'training_feature_influence_shrinkage_v2'
                or audit.get('passed') is not True or audit.get('training_only') is not True
                or audit.get('evaluated') is not True
                or audit.get('family') != family or audit.get('correction_strength') != strength
                or audit.get('fit_jacobian_background') != 'unshrunk fitted BN'
                or audit.get('forcing_background') != 'unshrunk fitted BN'
                or audit.get('output_response_background') != 'fixed-alpha shrunk BN'
                or audit.get('output_response_strength') != strength
                or audit.get('probe_mode') != 'common_to_all_training_repeats_and_gates'
                or audit.get('selected_method') != selected
                or audit.get('train_repeat_indices') != report['split']['train']
                or audit.get('fit_gate_indices') != training['gate_indices']
                or audit.get('amplitude') != 2e-4 or audit.get('signs') != [-1, 1]
                or audit.get('sigma_native_spacings') != [1.5, 3]
                or audit.get('position_count') != 17 or audit.get('max_relative_error') != .05):
            return False
        checked_gates = audit['checked_gate_indices']
        positions = audit['positions_energy']
        if (not isinstance(checked_gates, list) or not checked_gates
                or checked_gates != list(range(len(checked_gates)))
                or not set(training['gate_indices']).issubset(checked_gates)
                or len(positions) != 17 or not np.isfinite(positions).all()
                or not np.all(np.diff(positions) > 0)):
            return False
        for phase, count in (('screening', 68), ('actual_refits', 3)):
            evidence = audit[phase]
            errors = [evidence['max_drr_error_fraction'], evidence['max_d2_error_fraction']]
            if (evidence.get('passed') is not True or len(evidence['probes']) != count
                    or not np.isfinite(errors).all() or min(errors) < 0 or max(errors) > .05
                    or not _probe_errors_valid(evidence, count)):
                return False
        if (audit['screening'].get('linear_approximation') is not True
                or audit['actual_refits'].get('count') != 3
                or not _refinement_passed(audit)
                or not _referenced_probes_valid(audit['actual_refits'], audit)):
            return False
        for probe in audit['actual_refits']['probes']:
            if (probe.get('passed') is not True or probe.get('optimizer_success') is not True
                    or probe.get('family') != family or probe.get('correction_strength') != strength):
                return False
        candidate_audit = report['candidates'][selected]['training_influence_audit']
        if candidate_audit != audit:
            return False
        if not _selection_sensitivity_passed(joint['selection_sensitivity_audit'], selected,
                                             family, strength, report['split'], audit):
            return False
        drr, d2 = report['final_confirmation'], report['final_confirmation_d2']
        repeats = drr['repeat_indices']
        if (len(repeats) < 2 or len(set(repeats)) != len(repeats)
                or repeats != d2['repeat_indices'] or repeats != report['split']['final']
                or set(repeats) & set(report['split']['train'] + report['split']['selection'])
                or set(report['split']['train']) & set(report['split']['selection'])
                or not report['split']['train'] or not report['split']['selection']):
            return False
        for comparison in (drr, d2):
            values = [comparison['mean_original_rms'], comparison['mean_candidate_rms'],
                      comparison['mean_improvement_fraction']]
            if not np.isfinite(values).all() or min(values[:2]) < 0:
                return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def comparison_key(identity, settings):
    """An order-independent source recipe with an explicit ordered split."""
    identity = dict(identity)
    for key in ('measurements', 'baseline_files'):
        identity[key] = sorted(identity[key], key=lambda item: item['path'])
    return hashlib.sha256(canonical_json(dict(source_identity=identity, settings=settings)).encode()).hexdigest()


def file_digest(path):
    """Hash actual bytes, including when size and mtime were restored."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            check_load_cancelled()
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Input changed while reading: {path.name}')
    return digest.hexdigest()


def source_records(folder, names):
    paths = [resolve_source_path(folder, name) for name in names]
    if len(set(paths)) != len(paths) or not paths:
        raise ValueError('Select distinct, nonempty source files.')
    return [dict(path=portable_source_name(folder, path), sha256=file_digest(path),
                 mtime_ns=path.stat().st_mtime_ns) for path in paths]


def _names_key(folder, names):
    names = sorted({os.path.normcase(str(resolve_source_path(folder, name))) for name in names})
    return hashlib.sha256(canonical_json(names).encode()).hexdigest()


def validate_distinct_measurement_contents(records):
    """Different filenames do not make byte-identical captures independent."""
    hashes = [item['sha256'] for item in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate measurement contents cannot count as independent BN calibration repeats.')


def is_bn_calibration_source(folder, source):
    """Also recognize copied generated CSVs so moving one cannot remove scope."""
    path = resolve_source_path(folder, source)
    if BN_DIRECTORY in path.parts:
        return True
    if path.suffix.lower() != '.csv' or not path.is_file():
        return False
    with path.open('r', encoding='utf-8-sig', errors='replace') as stream:
        return BN_MARKER in stream.readline()


@dataclass(frozen=True)
class BnCalibrationVersion:
    version_id: str
    csv_path: Path
    report_path: Path
    eligible: bool
    status: str
    validation_status: str
    fresh: bool
    selected_method: str
    summary: str
    report: dict
    settings: dict
    source_identity: dict
    reasons: tuple[str, ...] = ()
    report_sha256: str = ''
    _verified: object = None


class DrrBnStore:
    def __init__(self, folder):
        self.root = Path(folder).resolve()
        self.directory = self.root / BN_DIRECTORY
        self.manifest_path = self.directory / 'manifest.json'

    def _read(self):
        if not self.manifest_path.exists():
            return dict(schema_version=1, versions=[], selections={})
        try:
            data = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            if data['schema_version'] != 1 or not isinstance(data['versions'], list) or not isinstance(data['selections'], dict):
                raise ValueError('Invalid schema.')
            ids = [v['version_id'] for v in data['versions']]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate version ids.')
            return data
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'BN calibration history is unreadable: {exc}') from exc

    @contextmanager
    def _locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = self.directory / '.store.lock'
        try:
            stream = lock.open('x', encoding='utf-8')
        except FileExistsError as exc:
            raise ValueError('BN calibration history is busy; retry after the current save.') from exc
        try:
            with stream:
                stream.write(str(os.getpid()))
            yield
        finally:
            lock.unlink(missing_ok=True)

    def _atomic_json(self, path, data):
        temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(canonical_json(data) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            check_load_cancelled()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _path(self, name):
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError('BN version path escapes its managed directory.')
        return path

    def _input_status(self, identity):
        reasons = []
        algorithm = identity.get('algorithm')
        if identity.get('schema_version', 1) >= 2:
            # Reopening an active version must not bypass the same numerical
            # pipeline invalidation that protects the comparison cache.
            from core.drr_bn_service import _algorithm_identity
            if algorithm != _algorithm_identity():
                reasons.append('Numerical pipeline changed; repeat joint DR/R and D2E validation.')
        for item in (*identity['measurements'], *identity['baseline_files']):
            try:
                if file_digest(resolve_source_path(self.root, item['path'])) != item['sha256']:
                    reasons.append('Changed input: ' + item['path'])
            except (OSError, ValueError) as exc:
                reasons.append('Unavailable input: ' + item['path'] + ': ' + str(exc))
        ordering = identity.get('ordering', {})
        if ordering.get('basis') == 'filesystem_mtime_proxy':
            for name, expected in ordering.get('mtime_ns', {}).items():
                try:
                    if resolve_source_path(self.root, name).stat().st_mtime_ns != expected:
                        reasons.append('Filesystem ordering changed: ' + name)
                except OSError:
                    reasons.append('Filesystem ordering unavailable: ' + name)
        for item in ordering.get('source_logs', ()):
            try:
                if file_digest(item['path']) != item['sha256']:
                    reasons.append('Acquisition ordering log changed: ' + item['path'])
            except (OSError, ValueError):
                reasons.append('Acquisition ordering log unavailable: ' + item['path'])
        return not reasons, tuple(reasons)

    def _version(self, record):
        path = self._path(record['report_path'])
        try:
            if file_digest(path) != record['report_sha256']:
                raise ValueError('BN report was modified.')
            doc = json.loads(path.read_text(encoding='utf-8'))
            if doc['eligible']:
                validate_distinct_measurement_contents(doc['source_identity']['measurements'])
            csv_path = self._path(doc['csv_path'])
            if file_digest(csv_path) != doc['csv_sha256']:
                raise ValueError('BN candidate CSV was modified.')
            if file_digest(self._path(doc['arrays_path'])) != doc['arrays_sha256']:
                raise ValueError('BN comparison snapshot was modified.')
            fresh, reasons = self._input_status(doc['source_identity'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'Invalid saved BN calibration {record["version_id"]}: {exc}') from exc
        eligible = bool(doc['eligible'] and joint_validation_passed(doc['report'], doc['settings']))
        status = doc['status']
        if doc['eligible'] and not eligible:
            status = 'needs_joint_validation'
            reasons = (*reasons, 'Saved comparison requires current D2E-priority joint DR/R and D2E validation.')
        return BnCalibrationVersion(
            record['version_id'], csv_path, path, eligible,
            status if fresh else 'stale', status, fresh,
            doc['selected_candidate'], doc['summary'], doc['report'], doc['settings'],
            doc['source_identity'], reasons, record['report_sha256'], _VERIFIED_VERSION,
        )

    def get(self, version_id):
        record = next((v for v in self._read()['versions'] if v['version_id'] == version_id), None)
        if record is None:
            raise ValueError('Unknown BN calibration version: ' + str(version_id))
        return self._version(record)

    get_version = get

    def version_for_csv(self, path):
        path = resolve_source_path(self.root, path)
        record = next((v for v in self._read()['versions']
                       if self._path(v['version_id'] + '.csv') == path), None)
        if record is None:
            raise ValueError('BN candidate has no verified version in this experiment.')
        return self._version(record)

    def find_current(self, measurements, baseline_files=None, which=None):
        key = _names_key(self.root, measurements)
        bn_key = _names_key(self.root, baseline_files) if baseline_files is not None else None
        for record in reversed(self._read()['versions']):
            if record['group_key'] != key or (bn_key is not None and record['baseline_key'] != bn_key):
                continue
            if which is not None and record['baseline_which'] != which:
                continue
            return self._version(record)
        return None

    def find_matching(self, identity, settings):
        """Find any exact saved comparison, including a previous SG setting.

        New records carry a small index. Older records remain reviewable but
        cannot accidentally reuse the pre-joint algorithm's acceptance.
        """
        key = comparison_key(identity, settings)
        for record in reversed(self._read()['versions']):
            if record.get('comparison_key') != key:
                continue
            version = self._version(record)
            if version.fresh and comparison_key(version.source_identity, version.settings) == key:
                return version
        return None

    def load_result(self, version_or_id):
        version = self.get(getattr(version_or_id, 'version_id', version_or_id))
        doc = json.loads(version.report_path.read_text(encoding='utf-8'))
        with np.load(self._path(doc['arrays_path']), allow_pickle=False) as data:
            payload = {k: data[k].copy() for k in ('energy', 'gates', 'original_background', 'candidate_background', 'mean_raw')}
            payload['candidate_backgrounds'] = {name: data['candidate__' + name].copy() for name in doc['candidate_names']}
            payload['input_snapshots'] = {name: data['input__' + name].copy() for name in doc['input_snapshot_names']}
        payload.update({key: doc[key] for key in ('report', 'eligible', 'selected_candidate', 'summary', 'candidates',
                                                'settings', 'source_identity', 'status', 'reasons')})
        payload.update(eligible=version.eligible, status=version.status)
        if doc['eligible'] and not version.eligible:
            payload['reasons'] = list(dict.fromkeys([*payload['reasons'], *version.reasons]))
            payload['summary'] = 'Current D2E-priority joint DR/R and D2E validation required. ' + version.summary
            payload['candidates'] = [dict(item, eligible=False) for item in payload['candidates']]
        payload.update(cached=True, version_id=version.version_id)
        return payload

    def save_result(self, payload):
        identity = payload['source_identity']
        if payload['eligible']:
            validate_distinct_measurement_contents(identity['measurements'])
            if not joint_validation_passed(payload['report'], payload['settings']):
                raise ValueError('Eligible BN publication requires joint DR/R and D2E validation evidence.')
        if Path(identity['root']).resolve() != self.root:
            raise ValueError('BN comparison belongs to another experiment.')
        fresh, reasons = self._input_status(identity)
        if not fresh:
            raise ValueError('BN inputs changed during comparison: ' + '; '.join(reasons))
        energy = np.asarray(payload['energy'], float)
        candidate = np.asarray(payload['candidate_background'], float)
        columns = np.asarray(identity['spectral_columns'], float)
        if (candidate.shape != energy.shape or columns.shape != energy.shape or energy.ndim != 1
                or not np.all(np.diff(energy) > 0) or not np.isfinite(candidate).all() or np.any(candidate <= 0)):
            raise ValueError('BN candidate must be positive, finite and use the original complete energy grid.')
        original_energy = 1240 / columns if np.median(columns) > 20 else columns
        order = np.argsort(original_energy, kind='stable')
        # The normal parser rounds numeric wavelength headers to 9 decimals.
        parser_columns = np.round(columns, 9)
        parsed_energy = 1240 / parser_columns if np.median(columns) > 20 else parser_columns
        if not np.allclose(parsed_energy[order], energy, rtol=1e-10, atol=1e-10):
            raise ValueError('Candidate energy does not match the source native spectral columns.')
        if payload.get('cached') and payload.get('version_id'):
            return self.get(payload['version_id'])
        with self._locked():
            manifest = self._read()
            version_id = 'bn_' + uuid4().hex
            csv_path = self.directory / (version_id + '.csv')
            report_path = csv_path.with_suffix('.metadata.json')
            arrays_path = csv_path.with_suffix('.npz')
            values = np.empty_like(candidate); values[order] = candidate
            stream = io.StringIO(newline=''); writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['Vbg', 'Vtg', 'Vbias', BN_MARKER, *(format(v, '.17g') for v in columns)])
            writer.writerow([0, 0, 0, version_id, *(format(v, '.17g') for v in values)])
            arrays = {k: np.asarray(payload[k]) for k in ('energy', 'gates', 'original_background', 'candidate_background', 'mean_raw')}
            arrays.update({'candidate__' + k: v for k, v in payload['candidate_backgrounds'].items()})
            arrays.update({'input__' + k: v for k, v in payload.get('input_snapshots', {}).items()})
            if any(v.dtype.kind == 'O' for v in arrays.values()):
                raise ValueError('Object arrays are not allowed in a BN snapshot.')
            try:
                check_load_cancelled()
                with csv_path.open('xb') as target:
                    target.write(stream.getvalue().encode('utf-8'))
                with arrays_path.open('xb') as target:
                    np.savez_compressed(target, **arrays)
                doc = {key: payload[key] for key in ('report', 'eligible', 'selected_candidate', 'summary', 'candidates',
                                                     'settings', 'source_identity', 'status', 'reasons')}
                doc.update(schema_version=1, kind=BN_KIND, version_id=version_id,
                           created_at=datetime.now(timezone.utc).isoformat(), csv_path=csv_path.name,
                           csv_sha256=file_digest(csv_path), arrays_path=arrays_path.name,
                           arrays_sha256=file_digest(arrays_path), candidate_names=list(payload['candidate_backgrounds']),
                           input_snapshot_names=list(payload.get('input_snapshots', {})))
                self._atomic_json(report_path, doc)
                fresh, reasons = self._input_status(identity)
                if not fresh:
                    raise ValueError('BN inputs changed before publication: ' + '; '.join(reasons))
                record = dict(version_id=version_id, report_path=report_path.name, report_sha256=file_digest(report_path),
                              group_key=_names_key(self.root, [r['path'] for r in identity['measurements']]),
                              baseline_key=_names_key(self.root, [r['path'] for r in identity['baseline_files']]),
                              baseline_which=identity['baseline_which'], comparison_key=comparison_key(identity, payload['settings']))
                manifest['versions'].append(record)
                self._atomic_json(self.manifest_path, manifest)
            except BaseException:
                # These are this invocation's three unique files, never source paths.
                for path in (csv_path, arrays_path, report_path):
                    path.unlink(missing_ok=True)
                raise
        return self.get(version_id)

    def activate(self, version_or_id, measurements=None):
        version = self.get(getattr(version_or_id, 'version_id', version_or_id))
        return self.activate_verified(version, measurements=measurements)

    def activate_verified(self, version, measurements=None):
        """Small GUI-thread commit after worker verification and context guard.

        No spectrum reads here. Load entry points verify all bytes again before
        using the background, closing the verification-to-load race.
        """
        if not isinstance(version, BnCalibrationVersion) or version._verified is not _VERIFIED_VERSION:
            raise ValueError('Activation requires a version returned by store verification.')
        with self._locked():
            if not version.fresh or not version.eligible or version.validation_status != 'accepted':
                raise ValueError('Only a fresh, independently validated BN result can be activated.')
            expected = [v['path'] for v in version.source_identity['measurements']]
            measurements = expected if measurements is None else measurements
            if _names_key(self.root, measurements) != _names_key(self.root, expected):
                raise ValueError('This BN calibration belongs to a different measurement group.')
            payload = self._read()
            record = next((v for v in payload['versions'] if v['version_id'] == version.version_id), None)
            if record is None or record['report_sha256'] != version.report_sha256:
                raise ValueError('BN history changed after verification; verify this version again.')
            if record['group_key'] != _names_key(self.root, measurements):
                raise ValueError('Verified BN version belongs to a different measurement group.')
            payload['selections'][_names_key(self.root, expected)] = dict(mode='calibrated', version_id=version.version_id)
            self._atomic_json(self.manifest_path, payload)
            return version

    def deactivate(self, measurements, baseline_files=None, which=None, *, restore_original=False):
        if not measurements:
            return
        if restore_original and (not baseline_files or which not in ('first', 'last', 'all')):
            raise ValueError('Restoring original BN requires its original files and frame method.')
        with self._locked():
            payload = self._read()
            choice = dict(mode='original', restore_original=bool(restore_original))
            if baseline_files:
                choice.update(baseline_paths=[portable_source_name(self.root, resolve_source_path(self.root, n))
                                              for n in baseline_files], baseline_which=which)
            payload['selections'][_names_key(self.root, measurements)] = choice
            self._atomic_json(self.manifest_path, payload)

    def clear_activation(self, measurements):
        """Automatic fallback is not an explicit choice to disable future BN fits."""
        if not measurements or not self.manifest_path.exists():
            return False
        with self._locked():
            payload = self._read()
            key = _names_key(self.root, measurements)
            if payload['selections'].get(key, {}).get('mode') != 'calibrated':
                return False
            del payload['selections'][key]
            self._atomic_json(self.manifest_path, payload)
        return True

    def original_recipe(self, measurements):
        """Only an explicit restore outranks later automatic recipe guesses.

        Forgetting a calibration while choosing another raw/PN background uses
        the default marker and must not restore an older original BN instead.
        """
        choice = self._read()['selections'].get(_names_key(self.root, measurements), {})
        if choice.get('mode') != 'original' or not choice.get('restore_original'):
            return None
        files, which = choice.get('baseline_paths'), choice.get('baseline_which')
        if not files or which not in ('first', 'last', 'all'):
            raise ValueError('Saved original BN choice is incomplete; select its reference again.')
        return dict(baseline_files=tuple(files), baseline_which=which)

    def selection_state(self, measurements):
        if not measurements:
            return None
        return self._read()['selections'].get(_names_key(self.root, measurements), {}).get('mode')

    def has_original_choice(self, measurements, baseline_files=None, which=None):
        choice = self._read()['selections'].get(_names_key(self.root, measurements), {})
        if choice.get('mode') != 'original':
            return False
        if baseline_files is not None and choice.get('baseline_paths'):
            return choice.get('baseline_which') == which and _names_key(self.root, choice['baseline_paths']) == _names_key(self.root, baseline_files)
        return True

    def find_applied(self, measurements):
        if not measurements:
            return None
        choice = self._read()['selections'].get(_names_key(self.root, measurements), {})
        if choice.get('mode') != 'calibrated':
            return None
        version = self.get(choice['version_id'])
        if _names_key(self.root, measurements) != _names_key(
                self.root, [v['path'] for v in version.source_identity['measurements']]):
            raise ValueError('Saved BN association does not match its exact measurement group.')
        if not version.fresh or not version.eligible or version.validation_status != 'accepted':
            raise ValueError('The previously applied BN calibration is stale or not eligible; use the original BN or compare again.')
        return version

    def validate_usage(self, measurements, background, *, y_axis=None):
        version = self.version_for_csv(background)
        expected = [v['path'] for v in version.source_identity['measurements']]
        if _names_key(self.root, measurements) != _names_key(self.root, expected):
            raise ValueError('BN calibration is restricted to its exact measurement group; Keep background cannot reuse it for another group.')
        if not version.fresh or not version.eligible or version.validation_status != 'accepted':
            raise ValueError('BN calibration is stale or exploratory; the original BN remains the usable reference.')
        if y_axis is not None and version.source_identity.get('y_axis') != str(y_axis):
            raise ValueError('BN calibration uses a different gate axis; restore its original axis or compare this group again.')
        choice = self._read()['selections'].get(_names_key(self.root, measurements), {})
        if choice.get('mode') != 'calibrated' or choice.get('version_id') != version.version_id:
            raise ValueError('This BN comparison has not been activated for the selected group.')
        return version


def validate_bn_background_usage(folder, measurements, backgrounds, *, y_axis=None):
    """Check whole-group scope before any per-member loader can split it."""
    managed = [name for name in dict.fromkeys(backgrounds) if is_bn_calibration_source(folder, name)]
    if managed and len(set(backgrounds)) != 1:
        raise ValueError('A group-calibrated BN cannot be averaged or spliced with other backgrounds.')
    store = DrrBnStore(folder)
    for name in managed:
        store.validate_usage(measurements, name, y_axis=y_axis)
    # A composite snapshot must not launder a scoped BN into an unrestricted
    # background. Ordinary independent composite snapshots remain unchanged.
    from core.drr_composite_background import is_composite_background, read_composite_background
    seen = set()
    def check_composite(name):
        path = resolve_source_path(folder, name)
        if path in seen or not path.is_file() or not is_composite_background(path):
            return
        seen.add(path)
        document = read_composite_background(path)
        for source in document.sources:
            for original in source.files:
                if is_bn_calibration_source(folder, original):
                    raise ValueError('Group-calibrated BN must be used directly; composite wrapping cannot remove its group scope.')
                check_composite(original)
    for name in backgrounds:
        if name not in managed:
            check_composite(name)
