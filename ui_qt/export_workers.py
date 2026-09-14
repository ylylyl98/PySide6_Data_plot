"""Owned Qt worker pool for standalone export windows."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot

from ui_qt.common import Worker


class _OwnedRunnable(QRunnable):
    def __init__(self, pool: "OwnedWorkerPool", worker: Worker):
        super().__init__()
        self._pool = pool
        self._worker = worker

    def run(self) -> None:
        try:
            self._worker.run()
        finally:
            self._pool._finished(self._worker)


class OwnedWorkerPool(QThreadPool):
    """Retain Worker and signal QObjects until queued callbacks are delivered."""

    _completed = Signal(object)

    def __init__(self, owner: QObject):
        super().__init__(owner)
        self._owner = owner
        self._workers: set[Worker] = set()
        self._completed.connect(self._retire, Qt.QueuedConnection)

    def start_worker(self, worker: Worker) -> None:
        worker.signals.setParent(self)
        self._workers.add(worker)
        super().start(_OwnedRunnable(self, worker))

    def _finished(self, worker: Worker) -> None:
        self._completed.emit(worker)

    @Slot(object)
    def _retire(self, worker: Worker) -> None:
        self._workers.discard(worker)
        worker.signals.deleteLater()
