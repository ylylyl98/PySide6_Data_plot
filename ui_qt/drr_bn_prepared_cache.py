"""Bounded private snapshots for the active group's validated BN products."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from threading import RLock

import numpy as np

from core.drr_bn_matching import confirmation_is_current
from core.load_cancellation import check_load_cancelled


def frozen_key(value):
    """Freeze the controlled LoadOptions/draft values without inspecting files."""
    if is_dataclass(value):
        return (type(value).__qualname__, tuple((item.name, frozen_key(getattr(value, item.name)))
                                                for item in fields(value)))
    if isinstance(value, dict):
        return tuple(sorted((key, frozen_key(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(frozen_key(item) for item in value)
    return value


def retained_array_bytes(value):
    """Count unique NumPy backing allocations, including strided/small views.

    A one-element view can retain a large base, while several views can share
    one allocation. The budget covers these buffers; small Python metadata is
    separately bounded by the entry limit.
    """
    seen, backings = set(), {}
    def visit(item):
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, np.ndarray):
            backing = item
            while True:
                if isinstance(backing, np.ndarray) and backing.base is not None:
                    backing = backing.base
                elif isinstance(backing, memoryview):
                    backing = backing.obj
                elif getattr(backing, 'base', None) is not None:
                    # NumPy stride helpers can insert a DummyArray owner.
                    backing = backing.base
                else:
                    break
            if isinstance(backing, (bytes, bytearray)):
                backings[id(backing)] = len(backing)
            elif isinstance(backing, np.ndarray):
                backings[id(backing)] = backing.nbytes
            else:
                try:
                    backings[id(backing)] = memoryview(backing).nbytes
                except TypeError:
                    backings[id(backing)] = item.nbytes
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (tuple, list, set, frozenset)):
            for child in item:
                visit(child)
        elif hasattr(item, '__dict__'):
            visit(vars(item))
    visit(value)
    return sum(backings.values())


def clone_prepared(prepared):
    """Copy mutable products together, preserving aliases and rebinding keys.

    Confirmations and plans are immutable, process-sealed values and must keep
    their identity. Never deepcopy their seal or freeze application arrays.
    """
    keys = tuple(prepared.derivatives)
    loaded, products = deepcopy((prepared.loaded, tuple(prepared.derivatives.values())))
    derivatives = {(id(loaded.cube), *key[1:]): value for key, value in zip(keys, products)}
    draft = replace(prepared.draft, fitted_result=deepcopy(prepared.draft.fitted_result))
    return replace(prepared, loaded=loaded, derivatives=derivatives, draft=draft)


class GroupBnPreparedCache:
    """At most three private products for one exact group/processing context.

    Activation returns an identity token. A superseded worker cannot activate
    an old scope or add its result to a newer group's cache. Copies occur only
    in workers; publishing a result never exposes the retained snapshot.
    """

    def __init__(self, *, max_bytes=96*1024*1024, max_entries=3):
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max(1, int(max_entries))
        self._lock = RLock()
        self._scope = None
        self._token = object()
        self._entries = OrderedDict()

    def activate(self, scope):
        with self._lock:
            if scope != self._scope:
                self._entries.clear()
                self._scope, self._token = scope, object()
            return self._token

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._scope, self._token = None, object()

    @property
    def entry_count(self):
        with self._lock:
            return len(self._entries)

    @property
    def total_bytes(self):
        with self._lock:
            return retained_array_bytes(tuple(self._entries.values()))

    def get(self, token, key):
        check_load_cancelled()
        with self._lock:
            if token is not self._token or key not in self._entries:
                return None
            prepared = self._entries.pop(key)
            self._entries[key] = prepared
        if not confirmation_is_current(prepared.confirmation):
            with self._lock:
                if token is self._token:
                    self._entries.clear()
            return None
        try:
            result = clone_prepared(prepared)
        except (MemoryError, TypeError):
            with self._lock:
                if token is self._token:
                    self._entries.pop(key, None)
            return None
        check_load_cancelled()
        if not confirmation_is_current(result.confirmation):
            with self._lock:
                if token is self._token:
                    self._entries.clear()
            return None
        with self._lock:
            return result if token is self._token else None

    def put(self, token, key, prepared):
        check_load_cancelled()
        with self._lock:
            if token is not self._token:
                return False
        if retained_array_bytes(prepared) > self.max_bytes:
            return False
        if not confirmation_is_current(prepared.confirmation):
            return False
        try:
            private = clone_prepared(prepared)
        except (MemoryError, TypeError):
            return False  # Caching is optional; the calculated result is usable.
        if retained_array_bytes(private) > self.max_bytes:
            return False
        check_load_cancelled()
        with self._lock:
            if token is not self._token:
                return False
            self._entries.pop(key, None)
            self._entries[key] = private
            while self._entries and (len(self._entries) > self.max_entries
                                     or retained_array_bytes(tuple(self._entries.values())) > self.max_bytes):
                self._entries.popitem(last=False)
        return True
