"""Render a frozen Matplotlib figure without touching the live Qt canvas."""
import pickle
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication
from ui_qt.common import Worker


def render_snapshot(snapshot, path, options, *, progress, log):
    # Snapshot bytes are generated locally from our own figure, never from files
    # supplied by a user. Matplotlib's pickle excludes its live Qt canvas.
    figure = pickle.loads(snapshot)
    FigureCanvasAgg(figure)
    path = Path(path)
    temporary = path.with_name(f'.{path.stem}-{uuid4().hex}.tmp{path.suffix}')
    try:
        figure.savefig(temporary, **options)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        figure.clear()
    return str(path)


class FigureSaveJob(QObject):
    succeeded = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, figure, path, options):
        super().__init__(QApplication.instance())
        self.done = False
        self.error = None
        self.started = perf_counter()
        self.last_tick = self.started
        self.max_gui_gap = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._tick)
        options = dict(options)
        for key in ('dpi', 'facecolor', 'edgecolor', 'transparent', 'bbox', 'pad_inches'):
            options.setdefault('bbox_inches' if key == 'bbox' else key, mpl.rcParams[f'savefig.{key}'])
        try:
            snapshot = pickle.dumps(figure, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            self.deleteLater()
            raise
        self.snapshot_seconds = perf_counter() - self.started
        self.worker = Worker(render_snapshot, snapshot, str(path), options)
        self.worker.signals.result.connect(self._success)
        self.worker.signals.error.connect(self._failure)
        self.worker.signals.finished.connect(self._finish)
        app = QApplication.instance()
        if not hasattr(app, '_figure_save_pool'):
            app._figure_save_pool = QThreadPool(app)
            app._figure_save_pool.setMaxThreadCount(1)
            app._figure_save_jobs = set()
        app._figure_save_jobs.add(self)
        self.timer.start()
        # Start after callers have connected status handlers.
        QTimer.singleShot(0, lambda: app._figure_save_pool.start(self.worker))

    @Slot()
    def _tick(self):
        now = perf_counter()
        self.max_gui_gap = max(self.max_gui_gap, now - self.last_tick)
        self.last_tick = now

    @Slot(object)
    def _success(self, path):
        self.succeeded.emit(str(path))

    @Slot(str)
    def _failure(self, message):
        self.error = message
        self.failed.emit(message)

    @Slot()
    def _finish(self):
        self._tick()
        self.timer.stop()
        self.done = True
        self.elapsed = perf_counter() - self.started
        self.finished.emit()
        QApplication.instance()._figure_save_jobs.discard(self)
        self.worker = None
        self.setParent(None)


def save_figure_async(figure, path, **options):
    apply_font = getattr(figure.canvas, 'apply_ui_font', None)
    if callable(apply_font):
        apply_font()
    return FigureSaveJob(figure, path, options)
