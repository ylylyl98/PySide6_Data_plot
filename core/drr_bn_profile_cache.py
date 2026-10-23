"""Bounded, process-local reuse of complete measured-BN CSV inspections."""
from collections import OrderedDict
import math
from pathlib import Path
import sys
from threading import RLock

from core.drr_sources import DrrGateProfile
from core.load_cancellation import check_load_cancelled
from core.provenance import _digest_signature


def _finite(values):
    return isinstance(values, tuple) and all(not isinstance(value, bool) and math.isfinite(value) for value in values)


def _complete_profile(profile):
    """Incomplete or malformed reads may be returned, but never reused."""
    try:
        if (not isinstance(profile, DrrGateProfile) or profile.grid_complete is not True
                or type(profile.varies) is not bool or type(profile.frame_count) is not int
                or profile.frame_count < 1 or len(profile.spectral_grid) < 2
                or not _finite(profile.spectral_grid)
                or not isinstance(profile.gate_grid, tuple)
                or len(profile.gate_grid) != profile.frame_count or not profile.gate_grid):
            return False
        width = len(profile.gate_grid[0])
        return bool(width and all(len(row) == width and _finite(row) for row in profile.gate_grid)
                    and isinstance(profile.gate_labels, tuple) and len(profile.gate_labels) == width
                    and all(isinstance(label, str) for label in profile.gate_labels)
                    and isinstance(profile.gate_ranges, tuple) and len(profile.gate_ranges) == width
                    and all(len(pair) == 2 and _finite(pair) and pair[0] <= pair[1] for pair in profile.gate_ranges)
                    and _finite(profile.constant_values)
                    and len(profile.constant_values) == (0 if profile.varies else width))
    except (TypeError, ValueError, OverflowError):
        return False


def _owned_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, DrrGateProfile):
        size += _owned_bytes(vars(value), seen)
    elif isinstance(value, dict):
        size += sum(_owned_bytes(key, seen) + _owned_bytes(item, seen) for key, item in value.items())
    elif isinstance(value, tuple):
        size += sum(_owned_bytes(item, seen) for item in value)
    return size


class VerifiedGateProfileCache:
    """Only stable full inspections enter this cache; no persisted trust."""

    def __init__(self, *, max_entries=128, max_bytes=16 * 1024 * 1024):
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = RLock()

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def _discard(self, path, expected=None):
        entry = self._entries.get(path)
        if entry is not None and (expected is None or entry is expected):
            self._bytes -= self._entries.pop(path)[2]

    def load(self, path, inspect, *, check=check_load_cancelled):
        check()
        path = Path(path).resolve()
        before = _digest_signature(path)
        check()
        with self._lock:
            cached = self._entries.get(path)
            if cached is not None and (before is None or cached[0] != before):
                self._discard(path, cached)
                cached = None
        if cached is not None:
            after = _digest_signature(path)
            check()
            if before == after:
                with self._lock:
                    if self._entries.get(path) is cached:
                        self._entries.move_to_end(path)
                return cached[1]
            with self._lock:
                self._discard(path, cached)
            before = after

        # Keep file reads outside the lock. A concurrently edited or cancelled
        # inspection never populates the cache; the caller's plan snapshot also
        # checks the whole operation before allowing the result to be applied.
        profile = inspect(path, max_rows=None)
        check()
        after = _digest_signature(path)
        check()
        if before is None or before != after or not _complete_profile(profile):
            return profile
        size = _owned_bytes(profile)
        if size > self.max_bytes:
            return profile
        with self._lock:
            check()
            self._discard(path)
            while self._entries and (len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes):
                oldest = next(iter(self._entries))
                self._discard(oldest)
            self._entries[path] = (after, profile, size)
            self._bytes += size
        return profile


BN_GATE_PROFILE_CACHE = VerifiedGateProfileCache()
