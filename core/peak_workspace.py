"""Immutable numeric snapshots and atomic persistence for Peak Analysis."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from core.peak_profiles import filter_presets, recommended_detection_method

from core.loader import DataCube

AXIS_FIELDS = ('gate_label', 'title', 'cbar_label', 'gate_unit', 'y_axis_semantic')


def session_directory():
    return Path(os.environ.get('LOCALAPPDATA', str(Path.home()/'.local/share'))) / 'DPTK' / 'Peak Analysis'


def fingerprint(cube, kind, channel, source):
    digest = hashlib.sha256(json.dumps([kind, channel, source, *[getattr(cube, k) for k in AXIS_FIELDS]],
                                      ensure_ascii=False).encode('utf-8'))
    x, z = np.asarray(cube.energy), np.asarray(cube.Z)
    if x.size > 1 and x[0] > x[-1]:
        x, z = x[::-1], z[:, ::-1]
    for value in (x, cube.gate, z):
        array = np.asarray(value, dtype='<f8', order='C')
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def create_dataset(cube, kind, channel, source, name, view=None):
    if kind not in {'PL', 'DRR'}:
        raise ValueError('Peak Analysis supports PL intensity and DRR extrema.')
    x, y, z = [np.array(a, dtype=float, copy=True) for a in (cube.energy, cube.gate, cube.Z)]
    if x.ndim != 1 or y.ndim != 1 or z.shape != (len(y), len(x)) or len(x) < 3 or not len(y):
        raise ValueError('A matching scan-by-energy matrix with at least three energy samples is required.')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Energy and scan coordinates must be finite.')
    if len(np.unique(y)) != len(y):
        raise ValueError('Repeated scan coordinates: separate or aggregate repeats before tracking.')
    if not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)):
        raise ValueError('Energy coordinates must be strictly monotonic.')
    # Preserve original row indexing for overlays; algorithms sort scan positions internally.
    if x[0] > x[-1]:
        x, z = x[::-1].copy(), z[:, ::-1].copy()
    for a in (x, y, z):
        a.setflags(write=False)
    snapshot = replace(cube, energy=x, gate=y, Z=z)
    settings = dict(x_min=float(x.min()), x_max=float(x.max()), y_min=float(y.min()), y_max=float(y.max()),
                    prominence=.05, min_distance_mev=2., max_shift_mev=5., min_width_mev=.5,
                    noise_sigma=3., max_gap=2, max_peaks=8, polarity='both' if kind == 'DRR' else 'peaks',
                    model='Lorentzian', detection_method=recommended_detection_method(kind,channel),
                    smoothing_window=11, candidate_floor=.01,
                    max_width_mev=0., max_per_row=0)
    settings.update(filter_presets(snapshot, kind, channel)['Balanced'])
    return dict(key=fingerprint(snapshot, kind, channel, source), kind=kind, channel=str(channel),
                source=str(source), name=str(name), cube=snapshot, settings=settings, branches=[], result=None,
                row=0, show_overlay=True, overlay_position='detected', view=deepcopy(view))


def validate_result(dataset, result):
    if result is None:
        return
    if result.get('dataset_key') != dataset['key']:
        raise ValueError('Analysis result identity does not match the dataset.')
    ids = {b['id'] for b in result.get('branches', [])}
    for point in result.get('points', []):
        row = point.get('row_index')
        if not isinstance(row, int) or not 0 <= row < len(dataset['cube'].gate):
            raise ValueError('Invalid result row.')
        if point.get('branch_id') not in ids or not np.isfinite([point['energy'], point['y']]).all():
            raise ValueError('Invalid branch or peak coordinates.')
        if point['y'] != float(dataset['cube'].gate[row]):
            raise ValueError('Peak row does not match its measured coordinate.')


def _json_clean(value):
    if isinstance(value, dict):
        return {k: _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(_json_clean(payload), stream, ensure_ascii=False, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_workspace(path, datasets, active_key=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    arrays, records = {}, []
    for index, d in enumerate(datasets):
        validate_result(d, d['result'])
        record = {k: v for k, v in d.items() if k != 'cube'}
        record['axis'] = {k: getattr(d['cube'], k) for k in AXIS_FIELDS}
        records.append(record)
        for field in ('energy', 'gate', 'Z'):
            arrays[f'{index}_{field}'] = getattr(d['cube'], field)
    arrays['metadata'] = np.array(json.dumps(_json_clean(dict(schema=1, datasets=records, active_key=active_key)), allow_nan=False))
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_workspace(path, *, fresh_snapshot=False):
    """Restore choices by default; app handoffs can apply current recommendations.

    The child owns handoff defaults so an already-running main app can keep
    supplying snapshots without reloading its imported analysis modules.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata'].item()))
            if meta['schema'] != 1:
                raise ValueError('Unsupported Peak Analysis session version.')
            datasets = []
            for index, record in enumerate(meta['datasets']):
                cube = DataCube(*(archive[f'{index}_{k}'] for k in ('energy', 'gate', 'Z')), **record['axis'])
                d = create_dataset(cube, record['kind'], record['channel'], record['source'], record['name'])
                if d['key'] != record['key']:
                    raise ValueError('Snapshot identity mismatch.')
                defaults=d['settings']
                d.update({k: v for k, v in record.items() if k not in {'cube', 'axis'}})
                d['settings']={**defaults,**d['settings']}
                # Opening a saved analysis must not silently refilter its data.
                # Old app handoffs with no results receive the current defaults.
                if 'min_snr' not in record['settings']:
                    d['settings'].update(min_snr=0.,neighbor_support=0)
                if fresh_snapshot and not (d.get('result') or d.get('candidates') or d['branches']):
                    d['settings'].update(filter_presets(d['cube'], d['kind'], d['channel'], d.get('product_label',''))['Balanced'])
                    d['settings']['detection_method']=recommended_detection_method(d['kind'],d['channel'],d.get('product_label',''))
                validate_result(d, d['result']); datasets.append(d)
        return datasets, meta.get('active_key')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Cannot read Peak Analysis session: {exc}') from exc
