import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class SourceStatusTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self):
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            self.window = MainWindow()
        self.window.resize(1180, 820)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_range_center_request_survives_followup_log_change(self):
        for mode, prefix in (("PL", "pl"), ("DRR", "drr"), ("Compare", "cmp")):
            self.window._pending_range_refresh.clear()
            self.window.loaded = SimpleNamespace(mode=mode, cube=None, compare_cubes={})
            with patch.object(self.window, "_schedule_plot_redraw"):
                spin = getattr(self.window, f"{prefix}_spins")["xmin"]
                spin.setValue(float(spin.value()) + 0.25)
                log = getattr(self.window, f"{prefix}_log_chk")
                log.setChecked(not log.isChecked())
            self.assertTrue(self.window._pending_range_refresh.get(mode), mode)

    def test_shown_status_follows_shared_canvas_and_clear_removes_it(self):
        self.window._last_draw_identity = {"Compare": ("B.csv",)}
        self.window._shown_draw_mode = "PL"
        self.window._shown_draw_identity = ("A.csv",)
        compare_index = next(
            i for i in range(self.window.tabs.count())
            if self.window.tabs.tabText(i) == "Compare"
        )
        self.window.tabs.setCurrentIndex(compare_index)
        self.window._set_stage("Loaded")
        self.assertEqual(self.window.data_state_label.text(), "Plot not updated")
        self.assertIn("A.csv", self.window.folder_edit.toolTip())

        self.window.loaded = SimpleNamespace(mode="PL")
        self.window.last_plotted_mode = "PL"
        with patch.object(self.window.canvas, "draw_idle"):
            self.window.pl_controller._clear_pl_source()
        self.assertEqual(self.window._shown_draw_identity, ())
        self.assertTrue(self.window.data_state_label.isHidden())

    def test_mcd_selection_and_power_roles_have_real_source_identities(self):
        self.window.mcd_files.addItem("mcd.csv")
        self.window._restore_list_selection(self.window.mcd_files, ["mcd.csv"])
        self.assertEqual(self.window._selected_identity_for_status("MCD"), ("mcd.csv",))

        record = SimpleNamespace(file_name="kkp.csv")
        role_record = SimpleNamespace(file_name="kk.csv")
        loaded = SimpleNamespace(
            mode="Power Dependent",
            selected_files=["primary.csv"],
            baseline_files=[],
            compare_sources={},
            power_records=[record],
            power_results={"KK": SimpleNamespace(records=[role_record])},
        )
        self.assertEqual(
            self.window._shown_source_identity_for_loaded(loaded),
            ("primary.csv", "kkp.csv", "kk.csv"),
        )

    def test_failed_load_marks_status_without_replacing_shared_shown_identity(self):
        self.window.pl_files.addItem("B.csv")
        self.window._restore_list_selection(self.window.pl_files, ["B.csv"])
        self.window._shown_draw_mode = "PL"
        self.window._shown_draw_identity = ("A.csv",)
        self.window._set_stage("Loading...")
        with patch("ui_qt.main_window.QMessageBox.critical"):
            self.window._show_error("B load failed")
        self.window._on_load_finished()
        self.assertEqual(
            self.window.data_state_label.text(),
            "Load failed",
        )
        self.assertIn("B.csv", self.window.folder_edit.toolTip())
        self.assertIn("A.csv", self.window.folder_edit.toolTip())


if __name__ == "__main__":
    unittest.main()
