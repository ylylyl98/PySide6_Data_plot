from __future__ import annotations

import gc
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QSettings, QThreadPool, QTimer, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from tests.ui_test_helpers import dispose_owned_window
from ui_qt import presentation_widget as presentation


class PresentationWorkerLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _wait_until(self, predicate) -> None:
        loop = QEventLoop()
        poll = QTimer(loop)
        poll.timeout.connect(lambda: loop.quit() if predicate() else None)
        watchdog = QTimer(loop)
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(loop.quit)
        poll.start(5)
        watchdog.start(3000)
        if not predicate():
            loop.exec()
        poll.stop()
        watchdog.stop()
        self.assertTrue(predicate(), "worker condition did not complete within 3 s")

    def _exercise_worker(self, kind: str, *, close_before_result: bool = False) -> None:
        # Gate run() after its completion signal. This deliberately lets the
        # GUI drop its worker reference before QThreadPool releases run's self.
        main_thread = threading.get_ident()
        destroyed_on = []
        run_started = threading.Event()
        run_completed = threading.Event()
        allow_result = threading.Event()
        allow_return = threading.Event()
        if not close_before_result:
            allow_result.set()
        original = getattr(presentation, kind)

        class GatedWorker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.signals.destroyed.connect(
                    lambda: destroyed_on.append(threading.get_ident()), Qt.DirectConnection
                )

            def run(self):
                run_started.set()
                if allow_result.wait(3):
                    super().run()
                run_completed.set()
                allow_return.wait(3)

        pool = QThreadPool()
        pool.setMaxThreadCount(1)
        with tempfile.TemporaryDirectory() as temporary:
            settings = QSettings(str(Path(temporary) / "settings.ini"), QSettings.IniFormat)
            with patch.dict(os.environ, LOCALAPPDATA=temporary), patch.object(
                presentation, "QSettings", return_value=settings
            ), patch.object(presentation, kind, GatedWorker):
                if kind == "_ThumbnailWorker":
                    owner = presentation.SlidePreview()
                    owner._thumbnail_pool = pool
                    image = QImage(16, 16, QImage.Format_RGB32)
                    image.fill(Qt.blue)
                    path = Path(temporary) / "preview.png"
                    self.assertTrue(image.save(str(path)))
                    start = lambda: owner._request_thumbnail(path)
                    finished = lambda: not owner._thumbnail_workers
                else:
                    owner = presentation.PresentationBuilderWidget()
                    owner._thread_pool = pool
                    owner.image_root_edit.setText(temporary)
                    start = owner.refresh_plots
                    finished = lambda: not owner._discovery_running
                try:
                    start()
                    self._wait_until(run_started.is_set)
                    if close_before_result:
                        dispose_owned_window(owner)
                        allow_result.set()
                    self._wait_until(lambda: run_completed.is_set() and finished())
                    allow_return.set()
                    self.assertTrue(pool.waitForDone(3000))
                    self.app.processEvents()
                    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                    # A signal QObject must be retired promptly in its GUI
                    # affinity thread, without relying on cyclic GC later.
                    self.assertEqual(destroyed_on, [main_thread])
                    if kind == "_ThumbnailWorker" and not close_before_result:
                        self.assertFalse(owner._pixmaps[str(path)].isNull())
                finally:
                    allow_result.set()
                    allow_return.set()
                    pool.waitForDone(3000)
                    self.app.processEvents()
                    dispose_owned_window(owner)
                    gc.collect()
                    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_thumbnail_signals_are_destroyed_on_gui_thread_after_worker_returns(self):
        self._exercise_worker("_ThumbnailWorker")

    def test_discovery_signals_are_retired_without_waiting_for_cyclic_gc(self):
        self._exercise_worker("_DiscoveryWorker")

    def test_thumbnail_can_finish_after_preview_has_been_destroyed(self):
        self._exercise_worker("_ThumbnailWorker", close_before_result=True)


if __name__ == "__main__":
    unittest.main()
