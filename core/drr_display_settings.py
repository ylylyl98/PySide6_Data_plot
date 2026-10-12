"""Portable DRR display history, separate from background/processing recipes."""
from copy import deepcopy
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4
from core.provenance import _digest_signature
from core.load_cancellation import check_load_cancelled

from core.drr_sources import (extract_wavelength_center_nm, resolve_source_path,
                             portable_source_name, _metadata_source_name,
                             _METADATA_PATHS, _history_file_signature, _HistoryPathScope)

VERSION = 1
INDEX_NAME = '.display_settings.json'
_WRITE_LOCK = RLock()
_CACHE_LOCK = RLock()
_INDEX_CACHE = OrderedDict()
_LEGACY_CACHE = OrderedDict()
_LEGACY_RECORD_CACHE = OrderedDict()


def source_key(folder, names):
    root = resolve_source_path(Path.cwd(), folder)
    paths = []
    for name in names:
        path = resolve_source_path(root, name)
        try:
            path = path.relative_to(root)
        except ValueError:
            pass
        paths.append(path.as_posix().casefold())
    return json.dumps(sorted(set(paths)), separators=(',', ':'))


def source_names(folder, names, provenance=()):
    """Use only verified canonical aliases; same basenames are not identities."""
    aliases = {}
    for record in provenance if isinstance(provenance, (list, tuple)) else ():
        if not isinstance(record, dict):
            if not hasattr(record, 'to_dict'):
                continue
            record = record.to_dict()
        digest = record.get('canonical_sha256')
        if (record.get('role') == 'measurement' and record.get('provenance_verified')
                and digest and digest == record.get('working_sha256_at_creation')
                and record.get('canonical_source_path') and record.get('working_copy_path')):
            aliases[str(resolve_source_path(folder, record['working_copy_path'])).casefold()] = record['canonical_source_path']
    result = []
    for name in names:
        path = resolve_source_path(folder, name)
        result.append(portable_source_name(folder, aliases.get(str(path).casefold(), path)))
    return result


def center_key(names, known=None):
    centers = [extract_wavelength_center_nm(name) for name in names]
    if centers and all(center is not None for center in centers):
        if max(centers) - min(centers) > 1e-6:
            return None
        known = centers[0]
    elif any(center is not None for center in centers):
        return None
    if known is None or not math.isfinite(float(known)) or float(known) <= 0:
        return None
    return format(float(known), '.9g')


def _read(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _index(path):
    path = Path(path)
    stamp = _signature(path)
    with _CACHE_LOCK:
        cached = _INDEX_CACHE.get(path)
        if cached is not None and cached[0] == stamp:
            _INDEX_CACHE.move_to_end(path)
            return cached[1]
        index = _read(path)
        if index.get('version') != VERSION:
            index = {'version': VERSION, 'files': {}, 'centers': {}}
        for name in ('files', 'centers'):
            if not isinstance(index.get(name), dict):
                index[name] = {}
        if _signature(path) == stamp:
            _INDEX_CACHE[path] = stamp, index
            _INDEX_CACHE.move_to_end(path)
            while len(_INDEX_CACHE) > 8:
                _INDEX_CACHE.popitem(last=False)
        return index


def _signature(path):
    path = Path(path)
    stamp = _digest_signature(path)
    if stamp is not None:
        return stamp
    try:
        stat = path.stat()
        # Still parse readable metadata when strong change stamps are not
        # supported. A unique stamp prevents reuse of these uncertain reads.
        return stat.st_mtime_ns, stat.st_size, stat.st_dev, stat.st_ino, object()
    except OSError:
        return None


def _metadata_files(directory):
    try:
        with os.scandir(directory) as entries:
            return tuple(sorted((entry.name, _signature(Path(entry.path))) for entry in entries
                                if entry.name.casefold().endswith('.metadata.json')))
    except OSError:
        return ()


def _legacy_state(payload, names):
    plot = payload.get('plot')
    if not isinstance(plot, dict) or not all(key in plot for key in ('xlim', 'ylim', 'vmin', 'vmax')):
        return None
    try:
        if len(plot['xlim']) != 2 or len(plot['ylim']) != 2:
            return None
        ranges = dict(zip(('xmin', 'xmax', 'ymin', 'ymax'), [*plot['xlim'], *plot['ylim']]))
        ranges.update(vmin=plot['vmin'], vmax=plot['vmax'])
        split = plot.get('split_scale') or {}
        if not isinstance(split, dict):
            return None
        processing = payload.get('processing') or {}
        if not isinstance(processing, dict):
            processing = {}
        count = 3 if split.get('split_x2') is not None else 2 if split else 1
        region = {key: split[key] for key in ('left_vmin', 'left_vmax', 'right_vmin',
                  'right_vmax', 'middle_vmin', 'middle_vmax') if split.get(key) is not None}
        if split:
            region['x0'] = split['split_x']
        if count == 3:
            region['x1'] = split['split_x2']
        return {'version': VERSION, 'files': names, 'center': center_key(names),
                'ranges': ranges, 'fixes': {}, 'raw': plot,
                'regions': {'count': count, 'boundary': split.get('show_boundary', True),
                            'drr': region, 'drr_fix': {}},
                'axis': {'y_spec': processing.get('y_axis', processing.get('y_axis_spec'))}}
    except (TypeError, KeyError, ValueError):
        return None


def current_revision(folder):
    """Identify snapshots changed while a load worker was reading history."""
    return _signature(Path(folder) / 'Processed Data' / 'DRR' / INDEX_NAME)


def find_candidates(folder, names, center=None, *, restore_files=True, restore_centers=True):
    """Read on the load worker; old export metadata also remains usable."""
    center = center_key(names, center)
    requested = bool(restore_files), bool(restore_centers)
    if not any(requested):
        return {'exact': None, 'template': None, 'center': center,
                'revision': None, 'requested': requested}
    folder = resolve_source_path(Path.cwd(), folder)
    revision = current_revision(folder)
    index = _index(folder / 'Processed Data' / 'DRR' / INDEX_NAME)
    key = source_key(folder, names) if restore_files else None
    exact = index['files'].get(key) if restore_files else None
    template = index['centers'].get(center) if center and restore_centers else None
    # A local index already contains the latest successful save. Scan old
    # metadata only for missing entries, never PNGs or source data tables.
    if (restore_files and exact is None) or (restore_centers and center and template is None):
        files, centers = _legacy_catalog(folder)
        if restore_files and exact is None:
            exact = files.get(key)
        if restore_centers and template is None:
            template = centers.get(center)
    return {'exact': deepcopy(exact), 'template': deepcopy(template), 'center': center,
            'revision': revision, 'requested': requested}


def _legacy_catalog(folder):
    directory = folder / 'Processed Data' / 'DRR'
    signature = _metadata_files(directory)
    with _CACHE_LOCK:
        cached = _LEGACY_CACHE.get(folder)
        if cached is not None:
            previous, dependencies, files, centers = cached
            if previous == signature and all(_history_file_signature(path) == stamp
                                             for path, stamp in dependencies):
                _LEGACY_CACHE.move_to_end(folder)
                return files, centers
        paths = {}
        token = _METADATA_PATHS.set(paths)
        try:
            files, centers = _parse_legacy_catalog(folder, signature)
        finally:
            _METADATA_PATHS.reset(token)
        dependencies = tuple((path, value[1]) for path, value in paths.items())
        if (_metadata_files(directory) == signature and all(
                _history_file_signature(path) == stamp for path, stamp in dependencies)):
            _LEGACY_CACHE[folder] = signature, dependencies, files, centers
            _LEGACY_CACHE.move_to_end(folder)
            while len(_LEGACY_CACHE) > 4:
                _LEGACY_CACHE.popitem(last=False)
        return files, centers


def _parse_legacy_catalog(folder, signature):
    directory = folder / 'Processed Data' / 'DRR'
    files, centers = {}, {}
    if signature:
        candidates, derivatives = [], []
        for name, stamp in signature:
            check_load_cancelled()
            path = directory / name
            parsed = _legacy_record(folder, path, stamp)
            if parsed is None:
                continue
            kind, record = parsed
            # Pair merging must not modify a cached individual sidecar.
            record = (*record[:-1], deepcopy(record[-1]))
            (derivatives if kind == 'second' else candidates).append(record)
        second_by_source = {}
        for _, _, key, _, state in sorted(derivatives, reverse=True):
            second_by_source.setdefault(key, state)
        for _, _, key, center, state in sorted(candidates, reverse=True):
            second = second_by_source.get(key)
            if second is not None and 'second' not in state:
                state['second'] = second['raw']
                state['regions']['drr_second'] = second['regions']['drr']
            files.setdefault(key, state)
            if center:
                centers.setdefault(center, state)
    return files, centers


def _legacy_record(folder, path, stamp):
    if stamp is None:
        return None
    shared = _METADATA_PATHS.get()
    key = folder, path.name
    cached = _LEGACY_RECORD_CACHE.get(key)
    if cached is not None and cached[0] == stamp and all(
            _history_file_signature(p) == value[1] for p, value in cached[1].items()):
        _, dependencies, result = cached
        _LEGACY_RECORD_CACHE.move_to_end(key)
        if shared is not None:
            for p, value in dependencies.items():
                shared.setdefault(p, value)
        return result
    dependencies = _HistoryPathScope(shared if shared is not None else {})
    token = _METADATA_PATHS.set(dependencies)
    try:
        result = _parse_legacy_record(folder, path, stamp)
    finally:
        _METADATA_PATHS.reset(token)
    if _signature(path) == stamp and all(
            _history_file_signature(p) == value[1] for p, value in dependencies.items()):
        _LEGACY_RECORD_CACHE[key] = stamp, dict(dependencies), result
        _LEGACY_RECORD_CACHE.move_to_end(key)
        while len(_LEGACY_RECORD_CACHE) > 4096:
            _LEGACY_RECORD_CACHE.popitem(last=False)
    return result


def _parse_legacy_record(folder, path, stamp):
    payload = _read(path)
    if payload.get('operation') not in ('DR/R', 'DRR') or payload.get('display_save_pending'):
        return None
    sources = payload.get('sources', payload.get('inputs', []))
    if not isinstance(sources, list):
        return None
    try:
        names = [_metadata_source_name(folder, item) for item in sources
                 if isinstance(item, dict) and item.get('role') == 'measurement']
        names = [name for name in names if isinstance(name, str) and name]
        if not names:
            return None
        names = source_names(folder, names, payload.get('provenance', ()))
        state = payload.get('ui_state')
        legacy = not isinstance(state, dict) or state.get('version') != VERSION
        if legacy:
            state = _legacy_state(payload, names)
        elif isinstance(state.get('files'), list) and all(isinstance(name, str) for name in state['files']):
            names = source_names(folder, state['files'], payload.get('provenance', ()))
        if state is None:
            return None
        processing = payload.get('processing') or {}
        order = processing.get('derivative_order') if isinstance(processing, dict) else None
        if not legacy or order in (None, 0, '0'):
            kind = 'raw'
        elif order in (2, '2'):
            kind = 'second'
        else:
            return None
        return kind, (stamp[0], path.name, source_key(folder, names),
                      center_key(names, state.get('center')), state)
    except (OSError, TypeError, ValueError):
        return None


def save_snapshot(folder, state):
    """Called only after image/data export succeeds; replace the index atomically."""
    directory = Path(folder) / 'Processed Data' / 'DRR'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / INDEX_NAME
    with _WRITE_LOCK:
        index = deepcopy(_index(path))
        index['files'][source_key(folder, state['files'])] = deepcopy(state)
        center = state.get('center')
        if center:
            index['centers'][center] = deepcopy(state)
        _write(path, index)


def clear_center(folder, center):
    """A tombstone also prevents legacy metadata from re-creating this template."""
    path = Path(folder) / 'Processed Data' / 'DRR' / INDEX_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        index = deepcopy(_index(path))
        index['centers'][center] = {'disabled': True}
        _write(path, index)


def complete_metadata_snapshots(data_paths):
    """Publish sidecar display history only when both paired products exist."""
    for data_path in data_paths:
        path = Path(data_path).with_suffix('.metadata.json')
        payload = _read(path)
        if payload.pop('display_save_pending', False):
            _write(path, payload)


def _write(path, payload):
    temporary = path.with_name(f'.display-settings-{uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
