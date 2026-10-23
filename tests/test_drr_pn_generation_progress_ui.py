"""Generation progress lifecycle without reconstruction fitting or user data."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

# Import the module rather than its TestCase: unittest must not discover and
# rerun the entire older suite through an imported TestCase alias.
from tests import test_drr_pn_background_ui as _fixtures
from ui_qt.theme import install_theme


class PnGenerationProgressUiTests(unittest.TestCase):
    write = _fixtures.PnBackgroundDialogTests.write
    dialog = _fixtures.PnBackgroundDialogTests.dialog
    choose = _fixtures.PnBackgroundDialogTests.choose
    reconstruction_dialog = _fixtures.PnBackgroundDialogTests.reconstruction_dialog
    reconstruction_stub = _fixtures.PnBackgroundDialogTests.reconstruction_stub

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self):
        _fixtures.PnBackgroundDialogTests.setUp(self)
        self.release_events = []

    def tearDown(self):
        for event in self.release_events:
            event.set()
        _fixtures.PnBackgroundDialogTests.tearDown(self)

    def wait_until(self, predicate, *, timeout=10000):
        """Run Qt's event loop while worker Events release the Python GIL."""
        self.app.processEvents()
        if predicate():
            return
        loop = QEventLoop()
        tick = QTimer()
        tick.setInterval(10)
        tick.timeout.connect(lambda: loop.quit() if predicate() else None)
        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(loop.quit)
        tick.start()
        watchdog.start(timeout)
        loop.exec()
        tick.stop()
        watchdog.stop()
        self.assertTrue(predicate(), "Timed out waiting for a controlled worker/UI event")

    def wait(self, dialog, *, timeout=15):
        self.wait_until(lambda: dialog._worker is None, timeout=int(timeout * 1000))

    def gate(self):
        entered, release = threading.Event(), threading.Event()
        self.release_events.append(release)
        return entered, release

    @staticmethod
    def await_release(release):
        if not release.wait(10):
            raise TimeoutError("Test did not release the controlled worker")

    def assert_incomplete_stopped(self, dialog, outcome):
        self.assertEqual(dialog._generation_outcome, outcome)
        self.assertFalse(dialog._generation_active)
        self.assertFalse(dialog._progress_timer.isActive())
        self.assertLess(dialog.progress_bar.value(), 19)
        self.assertFalse(dialog.cancel_generation_btn.isEnabled())

    def test_idle_and_non_generation_work_do_not_show_progress(self):
        dialog = self.dialog()
        self.assertTrue(dialog.progress_panel.isHidden())
        self.assertFalse(dialog._progress_timer.isActive())
        entered, release = self.gate()
        consumed = Mock()

        def work():
            entered.set()
            self.await_release(release)
            return "history"

        dialog._start(work, consumed, "Loading history")
        self.wait_until(entered.is_set)
        self.assertTrue(dialog.progress_panel.isHidden())
        self.assertFalse(dialog._generation_active)
        release.set()
        self.wait(dialog)
        consumed.assert_called_once_with("history")
        self.assertTrue(dialog.progress_panel.isHidden())

    def test_success_reaches_19_only_after_consume_completes(self):
        dialog = self.dialog()
        entered, release = self.gate()
        consumed_at = []

        def work(progress, log):
            progress.emit(19)
            log.emit("Ready to save and preview")
            entered.set()
            self.await_release(release)
            return "candidate"

        def consume(value):
            consumed_at.append((value, dialog.progress_bar.value()))

        dialog._start(work, consume, "Generating", generation=True, report_progress=True)
        self.wait_until(lambda: entered.is_set() and "Ready to save" in dialog.status.text())
        self.assertFalse(dialog.progress_panel.isHidden())
        self.assertEqual(dialog.progress_bar.maximum(), 19)
        self.assertLess(dialog.progress_bar.value(), 19)
        self.assertTrue(dialog._generation_active)
        self.assertTrue(dialog._progress_timer.isActive())
        self.assertTrue(dialog.cancel_generation_btn.isEnabled())
        release.set()
        self.wait(dialog)
        self.assertEqual(len(consumed_at), 1)
        self.assertEqual(consumed_at[0][0], "candidate")
        self.assertLess(consumed_at[0][1], 19)
        self.assertEqual(dialog.progress_bar.value(), 19)
        self.assertEqual(dialog._generation_outcome, "success")
        self.assertFalse(dialog._generation_active)
        self.assertFalse(dialog._progress_timer.isActive())
        self.assertFalse(dialog.cancel_generation_btn.isEnabled())

    def test_save_worker_error_and_preview_consume_error_never_complete(self):
        for failure_at in ("worker", "consume"):
            with self.subTest(failure_at=failure_at):
                dialog = self.dialog()

                def work(progress, log):
                    progress.emit(19)
                    if failure_at == "worker":
                        raise OSError("Draft save failed")
                    return "saved draft"

                def consume(_value):
                    raise ValueError("Preview failed")

                dialog._start(work, consume, "Generating", generation=True, report_progress=True)
                self.wait(dialog)
                self.assert_incomplete_stopped(dialog, "error")
                self.assertIn("failed", dialog.status.text().lower())
                self.assertIsNone(dialog.selected_version)

    def test_cancel_rejects_late_progress_and_retry_resets_progress_and_clock(self):
        dialog = self.dialog()
        dialog.show()
        entered, release = self.gate()
        consumed = Mock()

        def work(progress, log):
            progress.emit(6)
            log.emit("Controlled stage")
            entered.set()
            self.await_release(release)
            progress.emit(19)
            log.emit("Late completion must not win")
            return "cancelled result"

        dialog._start(work, consumed, "Generating", generation=True, report_progress=True)
        self.wait_until(lambda: entered.is_set() and dialog.progress_bar.value() == 6)
        initial_elapsed = dialog.elapsed_label.text()
        self.wait_until(lambda: dialog._progress_clock.elapsed() >= 1100
                        and dialog.elapsed_label.text() != initial_elapsed, timeout=3000)
        worker = dialog._worker
        QTest.mouseClick(dialog.cancel_generation_btn, Qt.LeftButton)
        self.assertTrue(worker.cancellation.cancelled)
        self.assertTrue(dialog.isVisible())
        self.assertFalse(dialog._closed)
        # Cancellation is cooperative: report stopping until the worker exits,
        # while already refusing its subsequent queued progress and log events.
        self.assertIsNone(dialog._generation_outcome)
        self.assertFalse(dialog.cancel_generation_btn.isEnabled())
        self.assertLess(dialog.progress_bar.value(), 19)
        stopped_value, stopped_text = dialog.progress_bar.value(), dialog.status.text()
        worker.signals.progress.emit(19)
        worker.signals.log.emit("Late completion must not win")
        self.app.processEvents()
        self.assertEqual(dialog.progress_bar.value(), stopped_value)
        self.assertEqual(dialog.status.text(), stopped_text)
        release.set()
        self.wait(dialog)
        consumed.assert_not_called()
        self.assert_incomplete_stopped(dialog, "cancelled")

        entered_again, release_again = self.gate()

        def retry(progress, log):
            entered_again.set()
            self.await_release(release_again)
            progress.emit(18)
            return "new result"

        dialog._start(retry, consumed, "Retrying", generation=True, report_progress=True)
        self.wait_until(entered_again.is_set)
        self.assertEqual(dialog.progress_bar.value(), 0)
        self.assertIsNone(dialog._generation_outcome)
        self.assertTrue(dialog._generation_active)
        self.assertTrue(dialog._progress_timer.isActive())
        self.assertLess(dialog._progress_clock.elapsed(), 1000)
        release_again.set()
        self.wait(dialog)
        consumed.assert_called_once_with("new result")
        self.assertEqual(dialog._generation_outcome, "success")
        self.assertEqual(dialog.progress_bar.value(), 19)

    def test_stale_context_does_not_consume_or_report_success(self):
        dialog = self.dialog()
        entered, release = self.gate()
        consumed = Mock()

        def work(progress, log):
            progress.emit(7)
            entered.set()
            self.await_release(release)
            progress.emit(19)
            return "obsolete candidate"

        dialog._start(work, consumed, "Generating", generation=True, report_progress=True)
        self.wait_until(lambda: entered.is_set() and dialog.progress_bar.value() == 7)
        dialog.measurements = ()  # A different selection, without loading any file.
        release.set()
        self.wait(dialog)
        consumed.assert_not_called()
        self.assertNotEqual(dialog._generation_outcome, "success")
        self.assertFalse(dialog._generation_active)
        self.assertFalse(dialog._progress_timer.isActive())
        self.assertLess(dialog.progress_bar.value(), 19)

    def test_generate_button_connects_reconstruction_stages_to_saved_preview(self):
        dialog, names = self.reconstruction_dialog()
        api, _request, _document = self.reconstruction_stub(names)
        entered, release = self.gate()

        def generate(*_args, **kwargs):
            kwargs["progress"]("source_v12", "start")
            kwargs["progress"]("source_v12", "complete")
            entered.set()
            self.await_release(release)
            return api.generate_reconstruction_background.return_value

        api.generate_reconstruction_background.side_effect = generate
        dialog.show()
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api", return_value=api):
            QTest.mouseClick(dialog.generate_btn, Qt.LeftButton)
            self.wait_until(lambda: entered.is_set() and dialog.progress_bar.value() > 0)
            self.assertFalse(dialog.progress_panel.isHidden())
            self.assertTrue(dialog._generation_active)
            self.assertTrue(dialog._progress_timer.isActive())
            self.assertGreater(dialog.progress_bar.value(), 0)
            self.assertLess(dialog.progress_bar.value(), 19)
            release.set()
            self.wait(dialog)
        api.generate_reconstruction_background.assert_called_once()
        self.assertIsNotNone(dialog._current, dialog.status.text())
        self.assertTrue(dialog._current.is_draft)
        self.assertIsNotNone(dialog._payload)
        self.assertEqual(dialog._generation_outcome, "success")
        self.assertEqual(dialog.progress_bar.value(), 19)
        self.assertFalse(dialog._progress_timer.isActive())
        self.assertIsNone(dialog.selected_version)
        self.assertIsNone(self.store.find_applied(names))

    def test_preview_exception_clears_partial_state_and_disables_use(self):
        dialog = self.dialog()

        def work(progress, log):
            progress.emit(18)
            return "saved draft"

        def partially_loaded_preview(_value):
            # _loaded_history/_show_preview can mutate these before rendering
            # raises. A failed preview must not leave this draft applicable.
            dialog._refreshing = True
            dialog._current = SimpleNamespace(status="valid", fresh=True)
            dialog._payload = {"compatibility_error": ""}
            dialog.use_btn.setEnabled(True)
            raise ValueError("Unable to display saved preview")

        dialog._start(work, partially_loaded_preview, "Generating",
                      generation=True, report_progress=True)
        self.wait(dialog)
        self.assert_incomplete_stopped(dialog, "error")
        self.assertIsNone(dialog._current)
        self.assertIsNone(dialog._payload)
        self.assertFalse(dialog._refreshing)
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertIn("Unable to display saved preview", dialog.status.text())

    def test_incompatible_saved_preview_completes_but_cannot_be_applied(self):
        dialog, names = self.reconstruction_dialog()
        api, _request, _document = self.reconstruction_stub(names)
        load_preview = dialog._preview

        def incompatible_preview(version):
            payload = load_preview(version)
            payload["compatibility_error"] = "Native grids incompatible for this test"
            return payload

        dialog.show()
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api", return_value=api), \
                patch.object(dialog, "_preview", side_effect=incompatible_preview):
            QTest.mouseClick(dialog.generate_btn, Qt.LeftButton)
            self.wait(dialog)
        api.generate_reconstruction_background.assert_called_once()
        self.assertIsNotNone(dialog._current, dialog.status.text())
        self.assertTrue(dialog._payload["compatibility_error"])
        self.assertEqual(dialog._generation_outcome, "success")
        self.assertEqual(dialog.progress_bar.value(), 19)
        self.assertEqual(dialog.progress_bar.format(), "Completed · 100%")
        self.assertEqual(dialog.progress_phase_label.text(),
                         "Completed — cannot apply this background")
        self.assertFalse(dialog._progress_timer.isActive())
        self.assertFalse(dialog._generation_active)
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertIsNone(dialog.selected_version)
        self.assertIsNone(self.store.find_applied(names))


if __name__ == "__main__":
    unittest.main()
