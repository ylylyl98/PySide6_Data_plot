"""Bounded process-local review results, identified by complete array contents."""
from collections import OrderedDict
import hashlib
import json
from threading import Lock

import numpy as np
from core.load_cancellation import check_load_cancelled


def review_key(metadata, *arrays):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for array in arrays:
        check_load_cancelled()
        canonical = np.ascontiguousarray(array, dtype='<f8')
        digest.update(str(canonical.shape).encode())
        digest.update(memoryview(canonical).cast('B'))
    return digest.hexdigest()


class ReviewMemo:
    """Store only small JSON results; no raw spectra or caller-owned objects."""
    def __init__(self, max_entries=128, max_bytes=2*1024*1024):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def get_or_compute(self, key, compute):
        check_load_cancelled()
        with self._lock:
            saved = self._entries.get(key)
            if saved is not None:
                self._entries.move_to_end(key)
        if saved is not None:
            check_load_cancelled()
            return json.loads(saved)
        result = compute()  # Independent workers need not wait on one slow file.
        check_load_cancelled()
        try:
            saved = json.dumps(result, allow_nan=False, separators=(',', ':')).encode()
        except (TypeError, ValueError):
            return result
        size = len(key) + len(saved)
        if size <= self.max_bytes and self.max_entries > 0:
            with self._lock:
                check_load_cancelled()
                old = self._entries.pop(key, None)
                if old is not None:
                    self._bytes -= len(key) + len(old)
                while self._entries and (len(self._entries) >= self.max_entries or self._bytes+size > self.max_bytes):
                    old_key, old_value = self._entries.popitem(last=False)
                    self._bytes -= len(old_key) + len(old_value)
                self._entries[key] = saved
                self._bytes += size
        return result
