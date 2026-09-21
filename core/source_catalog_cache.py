"""Persistent catalog snapshots; source contents are never read for validation.

``read`` is deliberately optimistic for immediate display. ``refresh`` validates
the recursive inventory before reusing a snapshot, and builds only on changes.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
from collections import OrderedDict
from threading import RLock


_DRR_SNAPSHOTS = OrderedDict()
_SNAPSHOT_LOCK = RLock()


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _record_types():
    # Fixed imports only: cache files cannot select arbitrary Python objects.
    from core.data_io import PowerSeriesSource
    from core.processing import PowerSeriesFile
    from core.drr_sources import DrrSource
    return {cls.__name__: cls for cls in (PowerSeriesSource, PowerSeriesFile, DrrSource)}


def _encode(value, grids=None, grid_ids=None):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if is_dataclass(value) and type(value) in _record_types().values():
        encoded = {}
        for field in fields(value):
            item = getattr(value, field.name)
            if type(value).__name__ == 'DrrSource' and field.name == 'spectral_grid' and item and grids is not None:
                if item not in grid_ids:
                    grid_ids[item] = len(grids)
                    grids.append(_encode(item))
                encoded[field.name] = {'type': 'grid', 'value': grid_ids[item]}
            else:
                encoded[field.name] = _encode(item, grids, grid_ids)
        return {'type': type(value).__name__, 'value': encoded}
    if isinstance(value, dict):
        return {'type': 'dict', 'value': [[_encode(k, grids, grid_ids), _encode(v, grids, grid_ids)] for k, v in value.items()]}
    for cls, tag in ((tuple, 'tuple'), (list, 'list'), (set, 'set')):
        if isinstance(value, cls):
            return {'type': tag, 'value': [_encode(item, grids, grid_ids) for item in value]}
    raise TypeError(f'Unsupported catalog value: {type(value).__name__}')


def _decode(value, grids=()):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if not isinstance(value, dict) or set(value) != {'type', 'value'}:
        raise ValueError('Invalid catalog value')
    tag, payload = value['type'], value['value']
    if tag == 'grid':
        if type(payload) is not int or not 0 <= payload < len(grids):
            raise ValueError('Invalid grid reference')
        return grids[payload]
    if tag == 'dict' and isinstance(payload, list):
        return {_decode(k, grids): _decode(v, grids) for k, v in payload}
    if tag in ('tuple', 'list', 'set') and isinstance(payload, list):
        return {'tuple': tuple, 'list': list, 'set': set}[tag](_decode(v, grids) for v in payload)
    cls = _record_types().get(tag)
    if cls is not None and isinstance(payload, dict):
        return cls(**{k: _decode(v, grids) for k, v in payload.items()})
    raise ValueError('Unsupported catalog type')


class SourceCatalogCache:
    SCHEMA_VERSION = 2
    EXTENSIONS = frozenset({'.csv', '.dat', '.xlsx', '.xls', '.json', '.png',
                            '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.tsv', '.txt'})
    IMAGE_EXTENSIONS = frozenset({'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'})
    DATA_MODES = frozenset({'pl', 'mcd', 'compare', 'shg', 'shg processing',
                            'power', 'power dependent', 'drr'})

    def __init__(self, folder: str, namespace: str, cache_root: Path | None = None, *, inventory_provider: Callable[[], list] | None = None):
        self.folder = Path(folder).resolve()
        self.namespace = str(namespace)
        self._inventory_provider = inventory_provider
        # The application appends a legacy-discovery flag to the mode namespace.
        mode = self.namespace.casefold()
        if mode.endswith(('-0', '-1')):
            mode = mode[:-2]
        self._extensions = (self.EXTENSIONS - self.IMAGE_EXTENSIONS
                            if mode in self.DATA_MODES else self.EXTENSIONS)
        if cache_root is None:
            base = os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CACHE_HOME')
            cache_root = (Path(base) if base else Path.home() / '.cache') / 'PySide6_Data_Plot' / 'source-catalog'
        self.cache_root = Path(cache_root).resolve()
        self._identity = os.path.normcase(str(self.folder))
        digest = hashlib.sha256((self._identity + '\0' + self.namespace).encode()).hexdigest()
        self.path = self.cache_root / (digest + '.json')

    def inventory(self) -> list[list[Any]]:
        """List relative names, sizes and nanosecond mtimes; fail on partial scans."""
        if self._inventory_provider is not None:
            return self._inventory_provider()
        if not self.folder.is_dir():
            raise OSError(f'Source directory is unavailable: {self.folder}')
        records = []
        pending = [(str(self.folder), '')]
        while pending:
            directory, prefix = pending.pop()
            # Keep DirEntry metadata: on Windows enumeration already supplies
            # size/mtime, avoiding separate Path.is_symlink/stat calls per file.
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    relative = prefix + entry.name
                    if entry.is_dir(follow_symlinks=False):
                        if Path(entry.path).resolve() != self.cache_root:
                            pending.append((entry.path, relative + '/'))
                    elif os.path.splitext(entry.name)[1].lower() in self._extensions:
                        if Path(entry.path) == self.path:
                            continue
                        stat = entry.stat(follow_symlinks=False)
                        records.append([relative, stat.st_size, stat.st_mtime_ns])
        return sorted(records)

    def _read_record(self):
        try:
            signature = _signature(self.path)
            key = (self.path, self._identity, self.namespace)
            with _SNAPSHOT_LOCK:
                cached = _DRR_SNAPSHOTS.get(key)
                if cached is not None and cached[0] == signature:
                    _DRR_SNAPSHOTS.move_to_end(key)
                    return [row[:] for row in cached[1]], list(cached[2])
            record = json.loads(self.path.read_text(encoding='utf-8'))
            if (record['schema'] not in (1, self.SCHEMA_VERSION) or record['folder'] != self._identity
                    or record['namespace'] != self.namespace or not isinstance(record['inventory'], list)):
                return None
            grids = tuple(_decode(grid) for grid in record.get('grids', ()))
            if any(not isinstance(grid, tuple) or any(type(v) not in (int, float) for v in grid) for grid in grids):
                return None
            payload = _decode(record['payload'], grids)
            if _signature(self.path) != signature:
                return None
            # Upgrade old snapshots once; every later process reads shared grids.
            if record['schema'] == 1:
                self._write(record['inventory'], payload)
                return record['inventory'], payload
            from core.drr_sources import DrrSource
            if isinstance(payload, list) and len(payload) <= 10000 and all(isinstance(item, DrrSource) for item in payload):
                with _SNAPSHOT_LOCK:
                    _DRR_SNAPSHOTS[key] = (signature, [row[:] for row in record['inventory']], tuple(payload))
                    _DRR_SNAPSHOTS.move_to_end(key)
                    while len(_DRR_SNAPSHOTS) > 4:
                        _DRR_SNAPSHOTS.popitem(last=False)
            return record['inventory'], payload
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            return None

    def read(self) -> Any | None:
        """Return the last snapshot without scanning the source directory."""
        record = self._read_record()
        return record[1] if record is not None else None

    def refresh(self, builder: Callable[[], Any], *, force: bool = False,
                publish_cached: Callable[[Any], None] | None = None,
                accept_cached: Callable[[Any], bool] | None = None) -> Any:
        # One decoded record supplies both immediate preview and validation.
        record = self._read_record() if publish_cached is not None or not force else None
        if record is not None and accept_cached is not None and not accept_cached(record[1]):
            record = None
        if record is not None and publish_cached is not None:
            publish_cached(record[1])
        try:
            before = self.inventory()
        except OSError:
            return builder()
        if not force and record is not None and record[0] == before:
            return record[1]
        payload = builder()
        try:
            # Do not certify a catalog built across a concurrent acquisition.
            if self.inventory() == before:
                self._write(before, payload)
        except OSError:
            pass
        return payload

    def _write(self, inventory, payload):
        temporary = None
        try:
            grids = []
            record = {'schema': self.SCHEMA_VERSION, 'folder': self._identity,
                      'namespace': self.namespace, 'inventory': inventory, 'payload': _encode(payload, grids, {}),
                      'grids': grids}
            self.cache_root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.cache_root,
                                             suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(record, handle, ensure_ascii=False)
            os.replace(temporary, self.path)
        except (OSError, TypeError, ValueError, RecursionError):
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
