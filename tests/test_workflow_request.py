import unittest
import threading
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QListWidget, QTabWidget
from PySide6.QtCore import Qt, QCoreApplication, QObject

from core.workflow_request import LatestRequest, RequestToken
from ui_qt.controllers_shg import ShgController
from ui_qt.controllers_compare import CompareController


class WorkflowRequestTests(unittest.TestCase):
    def test_latest_pending_replaces_intermediate_and_old_token_is_rejected(self):
        queue = LatestRequest[str]()
        token_a, start_a = queue.submit("PL", "folder", "A.csv")
        token_b, start_b = queue.submit("PL", "folder", "B.csv")
        token_c, start_c = queue.submit("PL", "folder", "C.csv")

        self.assertTrue(start_a)
        self.assertFalse(start_b)
        self.assertFalse(start_c)
        self.assertEqual(queue.pending_token, token_c)
        self.assertTrue(queue.accepts(token_a, "folder"))
        self.assertFalse(queue.accepts(token_b, "folder"))
        self.assertIsNone(queue.finish(token_b))
        self.assertEqual(queue.finish(token_a), "C.csv")
        self.assertEqual(queue.active_token, token_c)

    def test_same_active_or_pending_snapshot_is_not_submitted_twice(self):
        queue = LatestRequest[tuple[str, int]]()
        first, should_start = queue.submit("DRR", "one", ("A", 1))
        duplicate, duplicate_start = queue.submit("DRR", "one", ("A", 1))
        pending, pending_start = queue.submit("DRR", "one", ("B", 2))
        duplicate_pending, duplicate_pending_start = queue.submit("DRR", "one", ("B", 2))

        self.assertTrue(should_start)
        self.assertFalse(duplicate_start)
        self.assertFalse(pending_start)
        self.assertFalse(duplicate_pending_start)
        self.assertEqual(first, duplicate)
        self.assertEqual(pending, duplicate_pending)

    def test_folder_identity_is_part_of_acceptance(self):
        queue = LatestRequest[str]()
        token, _ = queue.submit("Compare", "folder-a", "source")
        self.assertFalse(queue.accepts(token, "folder-b"))
        self.assertTrue(queue.accepts(token, "folder-a"))

    def test_threaded_worker_releases_a_then_starts_only_latest_pending(self):
        queue = LatestRequest[str]()
        release_a = threading.Event()
        started = []

        token_a, should_start = queue.submit("PL", "folder", "A.csv")
        self.assertTrue(should_start)

        def worker_a():
            started.append("A.csv")
            release_a.wait(2.0)
            queue.finish(token_a)

        thread = threading.Thread(target=worker_a)
        thread.start()
        _token_b, _ = queue.submit("PL", "folder", "B.csv")
        token_c, _ = queue.submit("PL", "folder", "C.csv")
        release_a.set()
        thread.join(2.0)
        self.assertEqual(started, ["A.csv"])
        self.assertEqual(queue.active_token, token_c)
        self.assertFalse(queue.accepts(_token_b, "folder"))

    def test_shg_first_complete_single_selection_starts_load(self):
        QApplication.instance() or QApplication([])

        class Owner:
            current_folder = "folder"
            loaded = None

            def __init__(self):
                self.shg_files = QListWidget()
                self.shg_files.addItem("sample.csv")
                self.shg_files.item(0).setSelected(True)
                self.shg_workflow_tabs = QTabWidget()
                self.shg_workflow_tabs.addTab(QListWidget(), "Single")
                self.shg_workflow_tabs.addTab(QListWidget(), "Compare")
                self.shg_fit_branch_spin = type("Spin", (), {"setEnabled": lambda *_: None})()
                self.started = []

            def _start_load(self, mode):
                self.started.append(mode)

            @staticmethod
            def _selected(widget):
                return [item.text() for item in widget.selectedItems()]

            def _invalidate_export_move_sources(self):
                pass

            def _status(self, _message):
                pass

        owner = Owner()
        ShgController(owner)._on_shg_source_changed()
        self.assertEqual(owner.started, ["SHG Processing"])

    def test_compare_complete_group_assignment_starts_load_once(self):
        from pathlib import Path
        import tempfile
        from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QPlainTextEdit
        QApplication.instance() or QApplication([])

        class Owner:
            loaded = None
            current_folder = ""
            cmp_group_power_tolerance_percent = 5.0

            def __init__(self):
                self.cmp_channel_combos = {key: QComboBox() for key in ("KK", "KKp", "KpK", "KpKp")}
                self.cmp_display_preset_combo = QComboBox(); self.cmp_display_preset_combo.addItem("KK + KKp")
                self.cmp_show_checks = {}
                self.cmp_assignment_summary = QPlainTextEdit()
                for name, value in (("cmp_in_k_angle_spin", 0), ("cmp_in_kp_angle_spin", 45),
                                    ("cmp_out_k_angle_spin", 0), ("cmp_out_kp_angle_spin", 45),
                                    ("cmp_angle_tolerance_spin", 15)):
                    spin = QDoubleSpinBox(); spin.setValue(value); setattr(self, name, spin)
                self.started = []

            def _start_load(self, mode): self.started.append(mode)
            def _invalidate_export_move_sources(self): pass

        with tempfile.TemporaryDirectory() as folder:
            owner = Owner(); owner.current_folder = folder
            for name in ("a_PL.csv", "b_PL.csv"):
                Path(folder, name).write_text("x,y\n1,2\n", encoding="utf-8")
            for combo, name in ((owner.cmp_channel_combos["KK"], "a_PL.csv"),
                                (owner.cmp_channel_combos["KKp"], "b_PL.csv")):
                combo.addItem(name); combo.setCurrentText(name)
            CompareController(owner)._cmp_maybe_auto_load()
            self.assertEqual(owner.started, ["Compare"])

    def test_cross_workflow_finish_promotes_only_latest_captured_snapshot(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        QApplication.instance() or QApplication([])
        owner = QObject()
        token = RequestToken("PL", "folder", 1)
        pending = SimpleNamespace(mode="Compare", selected_files=["C.csv"])
        owner._is_closing = False
        owner._active_load_mode = "PL"
        owner._active_load_token = token
        owner._active_load_succeeded = False
        owner._load_in_progress = True
        owner._pending_load_mode = "Compare"
        owner._pending_load_options = pending
        owner._invalidated_load_modes = {"PL", "Compare"}
        owner._status_progress = SimpleNamespace(setVisible=lambda _value: None)
        owner._refresh_data_state_label = lambda: None
        owner._launch_load_options = Mock()
        owner._pl_auto_next_active = False
        owner._pl_auto_next_queue = []
        owner._load_requests = {"PL": LatestRequest()}
        owner._load_requests["PL"].submit("PL", "folder", "A.csv")
        from ui_qt.main_window import MainWindow
        MainWindow._on_load_finished(owner, request_token=token)
        QCoreApplication.instance().processEvents()
        owner._launch_load_options.assert_called_once_with(pending)
        owner.deleteLater()

    def test_real_main_window_first_result_draws_while_request_is_busy(self):
        from unittest.mock import patch
        from PySide6.QtWidgets import QApplication
        from ui_qt.main_window import LoadedState, MainWindow
        QApplication.instance() or QApplication([])
        window = MainWindow()
        try:
            tracker = LatestRequest()
            token, started = tracker.submit("PL", "", object())
            self.assertTrue(started)
            window._load_requests = {"PL": tracker}
            window._active_load_mode = "PL"
            window._active_load_token = token
            window._load_in_progress = True
            window._invalidated_load_modes = set()
            with patch.object(window, "_plot_mode") as draw:
                window._on_loaded(LoadedState(mode="PL", folder=""), request_token=token)
                draw.assert_called_once_with("PL", auto=True)
        finally:
            window.close()

    def test_clear_drops_pending_and_deferred_launch_cannot_restart(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        QApplication.instance() or QApplication([])
        owner = QObject()
        owner.current_folder = "folder"
        owner._is_closing = False
        owner._active_load_mode = "PL"
        owner._active_load_token = RequestToken("PL", "folder", 1)
        owner._active_load_succeeded = False
        owner._load_in_progress = True
        owner._load_lifecycle_generation = 0
        owner._pending_load_mode = "Compare"
        owner._pending_load_options = SimpleNamespace(mode="Compare", folder="folder")
        owner._invalidated_load_modes = set()
        owner._status_progress = SimpleNamespace(setVisible=lambda _value: None)
        owner._refresh_data_state_label = lambda: None
        owner._load_requests = {"PL": LatestRequest()}
        owner._load_requests["PL"].submit("PL", "folder", "A.csv")
        owner._pl_auto_next_active = False
        owner._pl_auto_next_queue = []
        owner._launch_load_options = Mock()
        MainWindow = __import__("ui_qt.main_window", fromlist=["MainWindow"]).MainWindow
        MainWindow._invalidate_active_load(owner)
        self.assertIsNone(owner._pending_load_options)
        MainWindow._on_load_finished(owner, request_token=owner._active_load_token)
        owner._is_closing = True
        QCoreApplication.instance().processEvents()
        owner._launch_load_options.assert_not_called()
        owner.deleteLater()

    def test_shg_export_waits_for_pending_reprocess(self):
        from unittest.mock import patch
        from ui_qt.main_window import LoadedState, MainWindow
        QApplication.instance() or QApplication([])
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="SHG Processing", folder="")
            window.last_plotted_mode = "SHG Processing"
            with patch.object(window, "_ensure_loaded_matches_ui_params", return_value=False), \
                    patch.object(window.shg_controller, "_shg_reprocess_pending_for_export", return_value=True), \
                    patch.object(window, "_status") as status:
                window._start_export("SHG Processing")
                status.assert_called_once()
            self.assertFalse(window._export_in_progress)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
