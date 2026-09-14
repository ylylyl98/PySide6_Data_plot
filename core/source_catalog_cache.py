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


def _record_types():
    # Fixed imports only: cache files cannot select arbitrary Python objects.
    from core.data_io import PowerSeriesSource
    from core.processing import PowerSeriesFile
    from core.drr_sources import DrrSource
    return {cls.__name__: cls for cls in (PowerSeriesSource, PowerSeriesFile, DrrSource)}


def _encode(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if is_dataclass(value) and type(value) in _record_types().values():
        return {'type': type(value).__name__, 'value': {
            field.name: _encode(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, dict):
        return {'type': 'dict', 'value': [[_encode(k), _encode(v)] for k, v in value.items()]}
    for cls, tag in ((tuple, 'tuple'), (list, 'list'), (set, 'set')):
        if isinstance(value, cls):
            return {'type': tag, 'value': [_encode(item) for item in value]}
    raise TypeError(f'Unsupported catalog value: {type(value).__name__}')


def _decode(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if not isinstance(value, dict) or set(value) != {'type', 'value'}:
        raise ValueError('Invalid catalog value')
    tag, payload = value['type'], value['value']
    if tag == 'dict' and isinstance(payload, list):
        return {_decode(k): _decode(v) for k, v in payload}
    if tag in ('tuple', 'list', 'set') and isinstance(payload, list):
        return {'tuple': tuple, 'list': list, 'set': set}[tag](_decode(v) for v in payload)
    cls = _record_types().get(tag)
    if cls is not None and isinstance(payload, dict):
        return cls(**{k: _decode(v) for k, v in payload.items()})
    raise ValueError('Unsupported catalog type')


class SourceCatalogCache:
    SCHEMA_VERSION = 1
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
        def onerror(error):
            raise error
        for directory, dirs, names in os.walk(self.folder, onerror=onerror, followlinks=False):
            parent = Path(directory)
            dirs[:] = [name for name in dirs
                       if not (parent / name).is_symlink()
                       and (parent / name).resolve() != self.cache_root]
            for name in names:
                path = parent / name
                if path.suffix.lower() not in self._extensions or path.is_symlink():
                    continue
                if path == self.path:
                    continue
                stat = path.stat()
                records.append([path.relative_to(self.folder).as_posix(), stat.st_size, stat.st_mtime_ns])
        return sorted(records)

    def _read_record(self):
        try:
            record = json.loads(self.path.read_text(encoding='utf-8'))
            if (record['schema'] != self.SCHEMA_VERSION or record['folder'] != self._identity
                    or record['namespace'] != self.namespace or not isinstance(record['inventory'], list)):
                return None
            return record['inventory'], _decode(record['payload'])
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            return None

    def read(self) -> Any | None:
        """Return the last snapshot without scanning the source directory."""
        record = self._read_record()
        return record[1] if record is not None else None

    def refresh(self, builder: Callable[[], Any], *, force: bool = False,
                publish_cached: Callable[[Any], None] | None = None) -> Any:
        # One decoded record supplies both immediate preview and validation.
        record = self._read_record() if publish_cached is not None or not force else None
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
            record = {'schema': self.SCHEMA_VERSION, 'folder': self._identity,
                      'namespace': self.namespace, 'inventory': inventory, 'payload': _encode(payload)}
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
