"""Per-source Peak Analysis workspace locations and legacy migration.

These functions perform filesystem I/O and must be called from a worker. They
never modify the legacy workspace. Callers serialize operations for a workspace.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from core.peak_workspace import atomic_json, load_workspace, save_workspace
from core.peak_output_names import output_folder, output_stem


def _identity(descriptor):
    source = descriptor.get('source') or ''
    try:
        parsed = json.loads(source) if isinstance(source, str) else source
    except (ValueError, TypeError):
        parsed = source
    if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
        mode, primary, files = parsed
    else:
        mode, primary, files = descriptor.get('kind', ''), parsed, []
    processed = descriptor.get('processed_path')
    primary = str(processed or primary or '')
    # Explicit processed paths identify one product. Otherwise keep the whole
    # input selection: two combinations with the same first file are distinct.
    identity = (['inferred', primary, source] if processed and descriptor.get('processed_path_inferred')
                else ['', primary, []] if processed else [str(mode), primary, files])
    return primary, identity


def _legacy_history_path(descriptor, fallback_root):
    """Locate the former hashed nested layout for migration only."""
    primary, identity = _identity(descriptor)
    missing = not primary
    if missing:
        primary = str(Path(fallback_root) / 'unknown')
    source = Path(primary).expanduser()
    if not source.is_absolute():
        source = Path(fallback_root) / source
    source = Path(os.path.abspath(source))
    canonical = json.dumps([identity[0], str(source).casefold(), identity[2]], ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]
    stem = re.sub(r'[^\w.-]+', '_', source.stem, flags=re.UNICODE).strip('._')[:48] or 'source'
    parts = source.parts
    initial_index = next((i for i, part in enumerate(parts) if part.casefold() == 'initial data'), None)
    if initial_index is not None:
        root = Path(*parts[:initial_index])
        kind = str(descriptor.get('kind') or identity[0] or 'Unknown')
        kind = re.sub(r'[^\w.-]+', '_', kind).strip('._') or 'Unknown'
        parent = root / 'Processed Data' / kind
    elif source.suffix.casefold() == '.dat':
        parent = source.parent
    else:
        # A suffix denotes a file; otherwise the identity is a folder.
        parent = source.parent if source.suffix else source
        if missing:
            parent = Path(fallback_root)
    return parent / 'Peak Analysis' / f'{stem}-{digest}' / 'workspace.npz'


def history_path(descriptor, fallback_root):
    """Bind a source to a flat readable filename, reserving it persistently.

    Calls write a small index and must be serialized by the caller.
    """
    folder = output_folder(descriptor, fallback_root)
    stem = output_stem(descriptor)
    key = _binding_key(descriptor, fallback_root)
    index_path = Path(fallback_root) / 'peak-history-bindings.json'
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.is_file() else {}
    if key in index:
        return Path(index[key])
    used = {str(path).casefold() for path in index.values()}
    number = 1
    while True:
        suffix = '' if number == 1 else f'_{number:02d}'
        candidate = folder / f'{stem}_analysis_history{suffix}.npz'
        if str(candidate).casefold() not in used:
            if not candidate.exists():
                break
            try:
                records, _ = load_workspace(candidate)
                if records and all(_binding_key(d, fallback_root) == key for d in records):
                    break
            except (OSError, ValueError, KeyError, TypeError):
                pass  # An unrelated or unreadable file must never be overwritten.
        number += 1
    index[key] = str(candidate)
    atomic_json(index_path, index)
    return candidate


def _binding_key(descriptor, fallback_root):
    folder = output_folder(descriptor, fallback_root)
    primary, identity = _identity(descriptor)
    source = Path(primary).expanduser()
    if not source.is_absolute():
        source = Path(fallback_root) / source
    return json.dumps([str(folder).casefold(), identity[0], os.path.abspath(source).casefold(), identity[2]], ensure_ascii=False, sort_keys=True, default=str)


def save_histories(datasets, fallback_root, active_key=None):
    """Merge and save each group; return paths written, preserving old keys."""
    groups = {}
    for dataset in datasets:
        groups.setdefault(history_path(dataset, fallback_root), []).append(dataset)
    nested_index_path = Path(fallback_root) / 'peak-history-nested-migration.json'
    nested_done = set(json.loads(nested_index_path.read_text(encoding='utf-8'))) if nested_index_path.is_file() else set()
    for path, incoming in groups.items():
        old, old_active = load_workspace(path) if path.is_file() else ([], None)
        merged = {}
        newly_migrated = set()
        for dataset in incoming:
            legacy = _legacy_history_path(dataset, fallback_root)
            marker = str(legacy) + '\n' + str(path)
            if marker not in nested_done and marker not in newly_migrated and legacy.is_file():
                records, _ = load_workspace(legacy)
                merged.update({item['key']: item for item in records})
                newly_migrated.add(marker)
        merged.update({item['key']: item for item in old})
        merged.update({item['key']: item for item in incoming})
        selected = active_key if active_key in merged else old_active
        save_workspace(path, list(merged.values()), selected)
        if newly_migrated:
            nested_done.update(newly_migrated)
            atomic_json(nested_index_path, sorted(nested_done))
    if groups:
        index_path = Path(fallback_root) / 'peak-history-inferred.json'
        index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.is_file() else {}
        for dataset in datasets:
            if dataset.get('processed_path_inferred') and dataset.get('processed_path'):
                key = _inferred_key(dataset, fallback_root)
                index[key] = str(history_path(dataset, fallback_root))
        atomic_json(index_path, index)
        atomic_json(Path(fallback_root) / 'peak-history-last.json',
                    {'paths': [str(path) for path in groups], 'active_key': active_key})
    return list(groups)


def load_last_histories(fallback_root):
    """Restore the last saved set of file histories, or None if unavailable."""
    index_path = Path(fallback_root) / 'peak-history-last.json'
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text(encoding='utf-8'))
    datasets = []
    for name in index['paths']:
        path = Path(name)
        records, _ = load_workspace(path)
        for item in records:
            item['history_root'] = str(fallback_root)
        datasets.extend(records)
    return datasets, index.get('active_key')


def _logical_key(dataset):
    return json.dumps([str(dataset.get(k, '')) for k in ('source', 'kind', 'channel')], ensure_ascii=False)


def _inferred_key(dataset, fallback_root):
    path = Path(dataset['processed_path']).expanduser()
    if not path.is_absolute():
        path = Path(fallback_root) / path
    return json.dumps([_logical_key(dataset), os.path.abspath(path).casefold()], ensure_ascii=False)


def _legacy_cache(legacy_path, fallback_root):
    """Split the legacy workspace once; publish index after all writes succeed."""
    legacy_id = os.path.abspath(legacy_path)
    base = Path(fallback_root) / 'peak-history-migration-cache'
    old_root = base / hashlib.sha256(legacy_id.encode('utf-8')).hexdigest()
    catalog_path = base / 'catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8')) if catalog_path.is_file() else {}
    if (old_root / 'index.json').is_file():
        return json.loads((old_root / 'index.json').read_text(encoding='utf-8'))
    if legacy_id not in catalog:
        catalog[legacy_id] = f'workspace_{len(catalog) + 1:04d}'
        atomic_json(catalog_path, catalog)
    cache_root = base / catalog[legacy_id]
    index_path = cache_root / 'index.json'
    if index_path.is_file():
        return json.loads(index_path.read_text(encoding='utf-8'))
    datasets, _ = load_workspace(legacy_path)
    groups = {}
    for dataset in datasets:
        groups.setdefault(_logical_key(dataset), []).append(dataset)
    mapping = {}
    for number, (logical_key, records) in enumerate(groups.items(), 1):
        name = f'cache_{number:04d}.npz'
        save_workspace(cache_root / name, records, None)
        mapping[logical_key] = str(cache_root / name)
    atomic_json(index_path, mapping)
    return mapping


def load_histories(descriptors, fallback_root, legacy_path=None):
    """Load requested histories, optionally migrating matching legacy records.

    Return (datasets, report), where report has active_keys and migrated_paths.
    Existing per-file records win on a
    key collision. The migration index is written only after successful saves.
    """
    paths = list(dict.fromkeys(history_path(d, fallback_root) for d in descriptors))
    loaded, active = {}, {}
    for path in paths:
        loaded[path], active[path] = load_workspace(path) if path.is_file() else ([], None)
        for item in loaded[path]:
            item['history_root'] = str(fallback_root)
    # Preserve the old nested history and copy its records into the flat file.
    nested_index_path = Path(fallback_root) / 'peak-history-nested-migration.json'
    nested_done = set(json.loads(nested_index_path.read_text(encoding='utf-8'))) if nested_index_path.is_file() else set()
    for descriptor in descriptors:
        target = history_path(descriptor, fallback_root)
        old_path = _legacy_history_path(descriptor, fallback_root)
        marker = str(old_path) + '\n' + str(target)
        if marker in nested_done or not old_path.is_file() or old_path == target:
            continue
        records, _ = load_workspace(old_path)
        merged = {d['key']: d for d in records}
        existing = {d['key']: d for d in loaded[target]}
        merged.update(existing)
        if len(merged) > len(existing):
            save_workspace(target, list(merged.values()), active[target])
            loaded[target] = list(merged.values())
        nested_done.add(marker)
        atomic_json(nested_index_path, sorted(nested_done))
    inferred_index_path = Path(fallback_root) / 'peak-history-inferred.json'
    inferred_index = json.loads(inferred_index_path.read_text(encoding='utf-8')) if inferred_index_path.is_file() else {}
    for descriptor in descriptors:
        if not descriptor.get('processed_path') or descriptor.get('processed_path_inferred'):
            continue
        target = history_path(descriptor, fallback_root)
        preferred = dict(descriptor, processed_path=descriptor.get('processed_path_hint') or descriptor['processed_path'])
        old_path = inferred_index.get(_inferred_key(preferred, fallback_root))
        if not old_path or Path(old_path) == target:
            continue
        old, _ = load_workspace(old_path)
        existing = {d['key']: d for d in loaded[target]}
        additions = {}
        for item in old:
            if _logical_key(item) == _logical_key(descriptor):
                item = dict(item, processed_path=descriptor['processed_path'],
                            processed_path_inferred=False, history_root=str(fallback_root))
                additions[item['key']] = item
        additions.update(existing)
        if additions and len(additions) > len(existing):
            save_workspace(target, list(additions.values()), active[target])
            loaded[target] = list(additions.values())
    migrated_paths = []
    if legacy_path and Path(legacy_path).is_file():
        index_path = Path(fallback_root) / 'peak-history-migration.json'
        index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.is_file() else {'migrated': []}
        migrated = set(index.get('migrated', []))
        legacy_id = str(Path(legacy_path).resolve())
        def marker(path):
            matching = [d for d in descriptors if history_path(d, fallback_root) == path]
            identity = sorted((str(d.get('source', '')), str(d.get('kind', '')), str(d.get('channel', ''))) for d in matching)
            return legacy_id + '\n' + str(path) + '\n' + json.dumps(identity, ensure_ascii=False)
        pending = [p for p in paths if marker(p) not in migrated]
        if pending:
            cache = _legacy_cache(legacy_path, fallback_root)
            for path in pending:
                additions = []
                matching = [d for d in descriptors if history_path(d, fallback_root) == path]
                for descriptor in matching:
                    cache_path = cache.get(_logical_key(descriptor))
                    if not cache_path:
                        continue
                    records, _ = load_workspace(cache_path)
                    for dataset in records:
                        candidate = dict(dataset)
                        if descriptor.get('processed_path'):
                            candidate['processed_path'] = descriptor['processed_path']
                            candidate['processed_path_inferred'] = bool(descriptor.get('processed_path_inferred'))
                            if descriptor.get('processed_path_hint'):
                                candidate['processed_path_hint'] = descriptor['processed_path_hint']
                        candidate['history_root'] = str(fallback_root)
                        additions.append(candidate)
                existing = {d['key']: d for d in loaded[path]}
                merged = {d['key']: d for d in additions}
                merged.update(existing)
                if additions:
                    save_workspace(path, list(merged.values()), active[path])
                    loaded[path] = list(merged.values())
                    migrated_paths.append(str(path))
                migrated.add(marker(path))
                atomic_json(index_path, {'migrated': sorted(migrated)})
    return [item for path in paths for item in loaded[path]], {'active_keys': active, 'migrated_paths': migrated_paths}
