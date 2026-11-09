"""Versioned, pickle-free persistence for completed power peak analyses."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import zlib

import numpy as np

from core.power_multi_peaks import MultiPeakSettings, PeakSeed, SpectrumFit, center_bounds
from core.power_peaks import PeakFit

_VERSION = 2
_FIELDS = ('power_uw', 'status', 'center_ev', 'fwhm_mev', 'fwhm_error_mev', 'area', 'height', 'rms')
_ARRAYS = ('x', 'fitted', 'residual', 'baseline')


def analysis_fingerprint(cubes, background):
    """Hash measured arrays and background for validating cached analyses."""
    h = hashlib.sha256()
    h.update(json.dumps(float(background), allow_nan=False).encode())
    for label in sorted(cubes):
        cube = cubes[label]
        h.update(json.dumps(label, ensure_ascii=True).encode())
        energy, gate, z = (np.asarray(getattr(cube, name), dtype='<f8') for name in ('energy', 'gate', 'Z'))
        if (energy.ndim != 1 or gate.ndim != 1 or z.shape != (len(gate), len(energy))
                or len(energy) < 2 or not np.all(np.isfinite(energy))
                or not np.all(np.diff(energy) > 0) or not np.all(np.isfinite(gate))):
            raise ValueError('Invalid cube axes or dimensions')
        for a in (energy, gate, z):
            h.update(repr(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _settings(data, energy):
    if data['model'] not in ('Lorentzian', 'Gaussian'):
        raise ValueError('Invalid model')
    seeds = tuple(PeakSeed(**seed) for seed in data['peaks'])
    settings = MultiPeakSettings(seeds, data['model'])
    center_bounds(settings, energy)
    return settings


def _peak_settings(rows, model):
    return MultiPeakSettings(tuple(PeakSeed(float(row["center"]), float(row["width"]) / 1000.,
        None if row.get("height") in (None, "") else float(row["height"]),
        None if row.get("lower") in (None, "") else float(row["lower"]),
        None if row.get("upper") in (None, "") else float(row["upper"])) for row in rows), model)


def save_analysis(path, cubes, background, rows, model, results, *, laws=(), law_error='', ranges=(), preferences=None):
    """Atomically save numeric results and metadata in a single NPZ file."""
    fingerprint = analysis_fingerprint(cubes, background)
    settings = _peak_settings(rows, model)
    for cube in cubes.values():
        _settings(asdict(settings), np.asarray(cube.energy))
    if set(results) != set(cubes):
        raise ValueError('Result channels do not match cubes')
    arrays = {}
    chunks = []
    offset = 0
    def pack_array(value):
        nonlocal offset
        value = np.asarray(value, dtype=float)
        if value.ndim != 1:
            raise ValueError('Invalid fit array')
        start = offset
        offset += len(value)
        chunks.append(value)
        return [start, len(value)]
    packed = {}
    for label, fits in results.items():
        cube = cubes[label]
        if len(fits) != len(cube.gate):
            raise ValueError('Result row count does not match cube')
        packed[label] = []
        for index, fit in enumerate(fits):
            item = dict(power_uw=float(fit.power_uw), status=str(fit.status), manual=bool(fit.manual),
                        rms=_number(fit.rms), parameters=list(fit.parameters), settings=asdict(fit.settings),
                        components=[{key: (str(getattr(c, key)) if key == 'status' else _number(getattr(c, key)))
                                     for key in _FIELDS} for c in fit.components])
            for name in _ARRAYS:
                value = getattr(fit, name)
                item[name] = None if value is None else pack_array(value)
            item['component_curves'] = [pack_array(curve) for curve in fit.component_curves]
            packed[label].append(item)
    arrays['data'] = np.concatenate(chunks) if chunks else np.empty(0, dtype=float)
    metadata = dict(version=_VERSION, fingerprint=fingerprint, rows=rows, model=model,
                    results=packed, laws=list(laws), law_error=str(law_error), ranges=list(ranges),
                    preferences=dict(preferences or {}))
    arrays['metadata'] = np.frombuffer(json.dumps(metadata, allow_nan=False, ensure_ascii=False).encode('utf-8'), dtype=np.uint8)
    # Validate the complete representation before replacing a previously valid cache.
    _unpack(metadata, arrays, cubes, fingerprint)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


def _validate_law_metadata(laws, ranges, cubes, count):
    def interval(item):
        if not isinstance(item, dict) or item.get('channel') not in cubes:
            raise ValueError('Invalid law channel')
        peak = item.get('peak')
        if isinstance(peak, bool) or not isinstance(peak, int) or not 1 <= peak <= count:
            raise ValueError('Invalid law peak')
        for field in ('lower', 'upper'):
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError('Invalid law interval')
        if not 0 < item['lower'] < item['upper']:
            raise ValueError('Invalid law interval')
        if 'suggested' in item and not isinstance(item['suggested'], bool):
            raise ValueError('Invalid suggested flag')

    for row in ranges:
        interval(row)
    for row in laws:
        interval(row)
        for field in ('observed_lower', 'observed_upper', 'alpha', 'alpha_error', 'amplitude', 'log_rms'):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError('Invalid law parameter')
        if (not 0 < row['observed_lower'] <= row['observed_upper'] or row['amplitude'] < 0
                or row['alpha_error'] < 0 or row['log_rms'] < 0):
            raise ValueError('Invalid law parameter')
        for field, minimum in (('n', 4), ('excluded', 0)):
            if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < minimum:
                raise ValueError('Invalid law count')
        if not isinstance(row['name'], str) or not isinstance(row['method'], str) or not isinstance(row['crosses_sources'], bool):
            raise ValueError('Invalid law metadata')


def _unpack(meta, arrays, cubes, fingerprint):
    if meta['version'] not in (1, _VERSION) or meta['fingerprint'] != fingerprint or set(meta['results']) != set(cubes):
        raise ValueError('Stale cache')
    if not isinstance(meta['laws'], list) or not isinstance(meta.get('ranges', []), list) or not isinstance(meta.get('preferences', {}), dict):
        raise ValueError('Invalid metadata')
    if any(not isinstance(k, str) or not isinstance(v, (str, bool, int, float, type(None)))
           or (isinstance(v, float) and not np.isfinite(v))
           for k, v in meta.get('preferences', {}).items()):
        raise ValueError('Invalid preferences')
    settings = _peak_settings(meta['rows'], meta['model'])
    count = len(settings.peaks)
    if not 1 <= count <= 8:
        raise ValueError('Invalid peak count')
    if not isinstance(meta['law_error'], str):
        raise ValueError('Invalid law error')
    _validate_law_metadata(meta['laws'], meta.get('ranges', []), cubes, count)
    data = np.asarray(arrays['data']) if meta['version'] == 2 else None
    if data is not None and (data.ndim != 1 or data.dtype.kind not in 'fiu'):
        raise ValueError('Invalid numeric data')
    results = {}
    for label, items in meta['results'].items():
        cube = cubes[label]
        energy, powers = np.asarray(cube.energy), np.asarray(cube.gate)
        _settings(asdict(settings), energy)
        if len(items) != len(powers):
            raise ValueError('Invalid row count')
        fits = []
        settings_cache = {}
        for i, item in enumerate(items):
            settings_key = json.dumps(item['settings'], sort_keys=True, allow_nan=False)
            if settings_key not in settings_cache:
                settings_cache[settings_key] = _settings(item['settings'], energy)
            fit_settings = settings_cache[settings_key]
            if len(fit_settings.peaks) != count or len(item['components']) != count:
                raise ValueError('Invalid component count')
            if not np.isfinite(item['power_uw']) or item['power_uw'] != powers[i]:
                raise ValueError('Invalid power')
            components = []
            for component in item['components']:
                fields = {key: (component[key] if key == 'status' else
                                (np.nan if component[key] is None else float(component[key]))) for key in _FIELDS}
                if fields['power_uw'] != powers[i] or any(np.isinf(v) for k, v in fields.items() if k != 'status'):
                    raise ValueError('Invalid component')
                components.append(PeakFit(**fields))
            params = tuple(float(v) for v in item['parameters'])
            if len(params) not in (0, 2 + 3 * count) or not np.all(np.isfinite(params)):
                raise ValueError('Invalid parameters')
            def array(key):
                if key is None:
                    return None
                if data is None:
                    value = np.asarray(arrays[key])
                else:
                    if (not isinstance(key, list) or len(key) != 2 or any(
                            isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in key)
                            or key[0] + key[1] > len(data)):
                        raise ValueError('Invalid array reference')
                    value = data[key[0]:key[0] + key[1]]
                if value.ndim != 1 or value.dtype.kind not in 'fiu' or not np.all(np.isfinite(value)):
                    raise ValueError('Invalid fit array')
                return value.astype(float, copy=True)
            values = {name: array(item[name]) for name in _ARRAYS}
            curves = tuple(array(key) for key in item['component_curves'])
            if params:
                x = values['x']
                if (x is None or len(x) == 0 or np.any(np.diff(x) <= 0)
                        or not np.all(np.isin(x, energy)) or any(values[n] is None or values[n].shape != x.shape for n in _ARRAYS)
                        or len(curves) != count or any(c is None or c.shape != x.shape for c in curves)):
                    raise ValueError('Inconsistent fit arrays')
            elif any(value is not None for value in values.values()) or curves:
                raise ValueError('Arrays without parameters')
            if item['rms'] is not None and not np.isfinite(float(item['rms'])):
                raise ValueError('Invalid RMS')
            fits.append(SpectrumFit(float(item['power_uw']), str(item['status']), tuple(components), fit_settings,
                                    bool(item['manual']), np.nan if item['rms'] is None else float(item['rms']),
                                    params, component_curves=curves, **values))
        results[label] = tuple(fits)
    return dict(rows=meta['rows'], model=meta['model'], settings=settings, results=results,
                laws=meta['laws'], law_error=meta['law_error'], ranges=meta.get('ranges', []),
                preferences=meta.get('preferences', {}), fingerprint=fingerprint)


def load_analysis(path, cubes, background):
    """Return a validated analysis, or None for stale, absent, or corrupt data."""
    try:
        fingerprint = analysis_fingerprint(cubes, background)
        with open(path, 'rb') as stream:
            with np.load(stream, allow_pickle=False) as archive:
                metadata = np.asarray(archive['metadata'])
                if metadata.dtype != np.uint8 or metadata.ndim != 1:
                    return None
                meta = json.loads(metadata.tobytes().decode('utf-8'))
                return _unpack(meta, archive, cubes, fingerprint)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, EOFError, UnicodeError, OverflowError, zipfile.BadZipFile, zlib.error):
        return None
