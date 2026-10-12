"""Byte-bounded derivative history with usable current full-resolution products."""
from collections import OrderedDict
from collections.abc import MutableMapping

import numpy as np


class DrrDerivativeCache(MutableMapping):
    """Retain the current raw/dE/second products and evict LRU history first.

    The budget covers retained derivative buffers, including current products.
    If the current dE/second pair alone exceeds it, no history is kept: those
    products remain usable so a full-resolution view cannot recompute forever.
    The raw source is already owned by LoadedState and is not counted twice.
    """

    def __init__(self, max_bytes=128 * 1024 * 1024, max_entries=12):
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max(3, int(max_entries))
        self._active = {}
        self._history = OrderedDict()

    @staticmethod
    def _slot(key):
        derivative = key[1]
        return 'raw' if derivative is None else 'first' if derivative == 1 else 'second'

    @staticmethod
    def _cost(key, value):
        cube, _window = value
        if id(cube) == key[0]:
            return 0
        arrays = [cube.energy, cube.gate, cube.Z, *getattr(cube, 'gate_coordinates', {}).values()]
        buffers = {}
        for array in arrays:
            if not isinstance(array, np.ndarray):
                continue
            while isinstance(array.base, np.ndarray):
                array = array.base
            buffers[id(array)] = array.nbytes
        return sum(buffers.values())

    @property
    def active_bytes(self):
        return sum(item[2] for item in self._active.values())

    @property
    def history_bytes(self):
        return sum(item[1] for item in self._history.values())

    @property
    def total_bytes(self):
        return self.active_bytes + self.history_bytes

    def __getitem__(self, key):
        active = self._active.get(self._slot(key))
        if active is not None and active[0] == key:
            return active[1]
        value, _cost = self._history.pop(key)
        self[key] = value
        return value

    def __setitem__(self, key, value):
        self._history.pop(key, None)
        slot = self._slot(key)
        previous = self._active.get(slot)
        if previous is not None and previous[0] != key:
            self._history[previous[0]] = (previous[1], previous[2])
        self._active[slot] = (key, value, self._cost(key, value))
        while self._history and (self.total_bytes > self.max_bytes or len(self) > self.max_entries):
            self._history.popitem(last=False)

    def __delitem__(self, key):
        active = self._active.get(self._slot(key))
        if active is not None and active[0] == key:
            del self._active[self._slot(key)]
        else:
            del self._history[key]

    def __contains__(self, key):
        active = self._active.get(self._slot(key))
        return (active is not None and active[0] == key) or key in self._history

    def __iter__(self):
        # Mapping clients can read/promote entries while iterating a snapshot.
        return iter((*self._history, *(item[0] for item in self._active.values())))

    def __len__(self):
        return len(self._active) + len(self._history)

    def clear(self):
        self._history.clear()
        self._active.clear()
