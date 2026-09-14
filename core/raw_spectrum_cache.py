"""Bounded process-local cache of parsed CSV arrays, before axis/background choices."""
from collections import OrderedDict
from threading import RLock

import numpy as np


class RawSpectrumCache:
    def __init__(self, max_bytes=128 * 1024 * 1024):
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = RLock()

    @staticmethod
    def _signature(path):
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns, stat.st_dev, stat.st_ino)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def load(self, path, version, parse):
        path = path.resolve()
        before = self._signature(path)
        key = (path, version, before)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry[0]

        # Parsing is intentionally outside the lock so independent files can load
        # concurrently. A file modified while reading is never published.
        arrays = parse(path)
        size = sum(a.nbytes for a in arrays if a is not None)
        if size > self.max_bytes:
            return arrays
        # Own each buffer, including legacy matrix slices, for exact accounting.
        frozen = tuple(None if a is None else np.array(a, copy=True) for a in arrays)
        for a in frozen:
            if a is not None:
                a.setflags(write=False)
        with self._lock:
            try:
                if self._signature(path) != before:
                    return arrays
            except OSError:
                return arrays
            for old_key in list(self._entries):
                if old_key[0] == path:
                    self._bytes -= self._entries.pop(old_key)[1]
            while self._entries and self._bytes + size > self.max_bytes:
                _, (_, removed_size) = self._entries.popitem(last=False)
                self._bytes -= removed_size
            self._entries[key] = (frozen, size)
            self._bytes += size
        return frozen
