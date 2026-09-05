from __future__ import annotations

import os
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRunnable, QThreadPool
from PySide6.QtWidgets import QApplication

from ui_qt.common import Worker
from ui_qt.main_window import MainWindow


class _FlagRunnable(QRunnable):
    def __init__(self, flag: threading.Event) -> None:
        super().__init__()
        self.flag = flag

    def run(self) -> None:
        self.flag.set()


class _PostSignalsWorker(Worker):
    """Hold a Worker after Worker.run emits finished but before outer return."""

    def __init__(self, fn, returned: threading.Event, allow_return: threading.Event) -> None:
        super().__init__(fn)
        self.returned = returned
        self.allow_return = allow_return

    def run(self) -> None:
        super().run()
        self.returned.set()
        self.allow_return.wait(2.0)


class WorkerLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None), patch.object(
            MainWindow, "_schedule_automatic_update_check", lambda _self: None
        ):
            self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_main_window_uses_private_pool_and_keeps_worker_until_outer_run_returns(self) -> None:
        self.assertIsNot(self.window.thread_pool, QThreadPool.globalInstance())

        post_signals = threading.Event()
        allow_return = threading.Event()

        def task(*, progress, log):
            return None

        worker = _PostSignalsWorker(task, post_signals, allow_return)
        self.window.thread_pool.start(worker)
        self.assertTrue(post_signals.wait(1.0))
        # Worker.run() has emitted result/finished, but the outer QRunnable
        # has not returned.  The ownership registry must still retain it.
        self.assertIn(worker, self.window._owned_workers)
        allow_return.set()
        for _ in range(100):
            self.app.processEvents()
            if worker not in self.window._owned_workers:
                break
        self.assertNotIn(worker, self.window._owned_workers)

    def test_close_does_not_wait_on_unrelated_global_work(self) -> None:
        global_started = threading.Event()
        global_release = threading.Event()
        global_done = threading.Event()

        class _BlockedGlobalRunnable(QRunnable):
            def run(self) -> None:
                global_started.set()
                global_release.wait(2.0)
                global_done.set()

        global_pool = QThreadPool.globalInstance()
        global_pool.start(_BlockedGlobalRunnable())
        self.assertTrue(global_started.wait(1.0))
        try:
            self.window.close()
            self.assertFalse(global_done.is_set())
        finally:
            global_release.set()
            self.assertTrue(global_done.wait(1.0))

    def test_close_waits_for_owned_worker_before_window_teardown(self) -> None:
        entered = threading.Event()
        closing_seen = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        result_seen = threading.Event()

        def task(*, progress, log):
            entered.set()
            while not self.window._is_closing:
                if closing_seen.wait(0.01):
                    break
            closing_seen.set()
            release.wait(2.0)
            completed.set()

        worker = Worker(task)
        worker.signals.result.connect(lambda _result: result_seen.set())
        self.window.thread_pool.start(worker)
        self.assertTrue(entered.wait(1.0))

        # The close path must wait for this window's private pool.  Release the
        # worker only after close has marked the window closing.
        def release_when_closing() -> None:
            self.assertTrue(closing_seen.wait(1.0))
            release.set()

        releaser = threading.Thread(target=release_when_closing)
        releaser.start()
        self.window.close()
        releaser.join(1.0)
        self.assertTrue(completed.is_set())
        self.app.processEvents()
        self.assertTrue(result_seen.is_set())
        self.assertFalse(self.window._owned_workers)


if __name__ == "__main__":
    unittest.main()
