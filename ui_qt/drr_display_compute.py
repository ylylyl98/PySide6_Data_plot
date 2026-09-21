"""One in-flight transform, keyed by source, derivative and SG settings."""
from PySide6.QtCore import QObject, Signal, Slot
from core.processing import apply_sg_derivative_energy
from ui_qt.common import Worker


def compute(cube, key, *, progress, log):
    result, window = apply_sg_derivative_energy(
        cube, derivative=key[1], window_length=key[2], polyorder=key[3])
    result.gate_unit = getattr(cube, 'gate_unit', '')
    result.y_axis_semantic = getattr(cube, 'y_axis_semantic', '')
    return cube, key, result, window


class DrrDisplayCompute(QObject):
    ready = Signal()
    failed = Signal(str)

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.worker = None
        self.desired = None
        self.error = False

    def request(self, cube, derivative, window, poly):
        key = (id(cube), derivative, int(window), int(poly))
        if key in self.owner._drr_derivative_cache:
            return True
        self.desired = key
        if self.worker is None:
            self.error = False
            self.active_key = key
            self.worker = Worker(compute, cube, key)
            self.worker.signals.result.connect(self._result)
            self.worker.signals.error.connect(self._error)
            self.worker.signals.finished.connect(self._finished)
            self.owner.thread_pool.start(self.worker)
        return False

    @Slot(object)
    def _result(self, payload):
        cube, key, result, window = payload
        if (getattr(getattr(self.owner, 'loaded', None), 'cube', None) is not cube
                or self.desired is None or self.desired[0] != key[0]
                or self.desired[2:] != key[2:]):
            return
        # Raw-range work can require dE while the display requires d2E.
        # Both remain valid for identical source/SG settings; retain each
        # under its own key so neither request starves the other.
        cache = self.owner._drr_derivative_cache
        if len(cache) >= 12:
            cache.pop(next(iter(cache)))
        cache[key] = result, int(window)

    @Slot(str)
    def _error(self, message):
        self.error = True
        if (self.desired is not None and self.desired[2:] == self.active_key[2:]
                and id(getattr(getattr(self.owner, 'loaded', None), 'cube', None)) == self.active_key[0]):
            self.failed.emit(message)

    @Slot()
    def _finished(self):
        self.worker = None
        if (not self.error or self.desired != self.active_key
                or id(getattr(getattr(self.owner, 'loaded', None), 'cube', None)) != self.active_key[0]):
            self.ready.emit()
