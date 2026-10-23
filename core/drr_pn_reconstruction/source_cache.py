"""Bounded process-local reuse of source fits for exactly the same raw P/N.

Only the sample-independent preparation is cached. Callers retain the raw-file
and implementation checks before and after each reconstruction. No cache files
are written into the experiment, and no historical background is accepted.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import sys
from threading import RLock

import numpy as np


def _key(energy, pn_mean, pn_frames, identity):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False,
                                      separators=(",", ":")).encode("utf8"))
    for value in (energy, pn_mean, pn_frames):
        array = np.asarray(value)
        digest.update(repr((array.dtype.str, array.shape, array.strides)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _freeze_and_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
        return value.nbytes + 128
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_freeze_and_size(k, seen) + _freeze_and_size(v, seen)
                    for k, v in value.items())
    elif isinstance(value, (tuple, list)):
        size += sum(_freeze_and_size(v, seen) for v in value)
    return size


class SourcePreparationCache:
    """Own every cached buffer; give each reconstruction a private copy.

    Fits run outside the lock. A simultaneous request may compute the same fit
    twice, but cannot observe a partial entry or share mutable diagnostics.
    """

    def __init__(self, max_bytes=128 * 1024 * 1024, max_entries=4):
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max(0, int(max_entries))
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = RLock()

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def prepare(self, energy, pn_mean, pn_frames, *, identity, compute, check,
                force=False):
        check()
        key = _key(energy, pn_mean, pn_frames, identity)
        if not force:
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    result = copy.deepcopy(entry[0])
                else:
                    result = None
            if result is not None:
                check()
                return result, True, key
        result = compute()
        check()
        if _key(energy, pn_mean, pn_frames, identity) != key:
            raise ValueError("Raw P/N arrays changed during source preparation.")
        if self.max_bytes and self.max_entries:
            owned = copy.deepcopy(result)
            size = _freeze_and_size(owned)
            check()
            if size <= self.max_bytes:
                with self._lock:
                    previous = self._entries.pop(key, None)
                    if previous is not None:
                        self._bytes -= previous[1]
                    while self._entries and (self._bytes + size > self.max_bytes
                                             or len(self._entries) >= self.max_entries):
                        _, (_, removed_size) = self._entries.popitem(last=False)
                        self._bytes -= removed_size
                    self._entries[key] = (owned, size)
                    self._bytes += size
        return result, False, key


SOURCE_PREPARATION_CACHE = SourcePreparationCache()
