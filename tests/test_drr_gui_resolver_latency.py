from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow
from core.drr_sources import DrrMeasurementAssignment


class DrrGuiResolverLatencyTests(unittest.TestCase):
    """Guard the GUI boundary around DRR assignment resolution.

    These tests deliberately fail fast if the GUI-side resolver is entered.
    The expensive resolver reads source files; a real load worker remains the
    owner of that work.  The second test only stubs the load dispatch boundary
    so it can prove that parameter mismatch is handed off before any resolver
    call is made.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._old_qsettings_format = QSettings.defaultFormat()
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(self.root / "settings"))
        self.source = self.root / "measurement.csv"
        self.background = self.root / "background.csv"
        self.source.write_text("measurement\n", encoding="utf-8")
        self.background.write_text("background\n", encoding="utf-8")
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None), patch.object(
            MainWindow, "_schedule_automatic_update_check", lambda _self: None
        ):
            self.window = MainWindow()
        self.window.current_folder = str(self.root)
        self.assignment = DrrMeasurementAssignment(
            measurement_file=self.source.name,
            baseline_mode="Self (last frame)",
            baseline_files=(),
            baseline_which="last",
            selection_reason="test authoritative assignment",
        )
        self.window.loaded = LoadedState(
            mode="DRR",
            folder=str(self.root),
            primary_file=self.source.name,
            selected_files=[self.source.name],
            baseline_files=[],
            cube=SimpleNamespace(),
            drr_mode_label="DR/R Self",
            drr_baseline_text="Automatic",
            drr_baseline_which="last",
            drr_assignments=(self.assignment,),
            y_axis_spec="auto",
        )
        self.window.drr_selected_files = [self.source.name]
        self.window._drr_assignments = (self.assignment,)
        self.window._drr_assignments_automatic = True
        self.window.drr_baseline_combo.setCurrentText("Automatic")

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        QSettings.setDefaultFormat(self._old_qsettings_format)
        self.tmp.cleanup()

    def test_existing_automatic_assignment_does_not_resolve_on_gui(self) -> None:
        gui_thread = threading.get_ident()
        with patch(
            "ui_qt.main_window.resolve_drr_background_assignments",
            side_effect=AssertionError("automatic assignment was resolved on GUI thread"),
        ) as resolve, patch(
            "ui_qt.main_window.discover_drr_sources",
            side_effect=AssertionError("DRR discovery was entered on GUI thread"),
        ) as discover, patch.object(
            self.window, "_start_load", side_effect=AssertionError("unexpected reload")
        ):
            changed = self.window._ensure_loaded_matches_drr_params()

        self.assertFalse(changed)
        resolve.assert_not_called()
        discover.assert_not_called()
        self.assertEqual(threading.get_ident(), gui_thread)

    def test_baseline_change_dispatches_reload_before_resolver(self) -> None:
        self.window.drr_baseline_files_manual = [self.background.name]
        self.window.drr_baseline_combo.setCurrentText("External")
        self.window._drr_assignments = ()
        self.window._drr_assignments_automatic = False
        dispatches: list[tuple[str, int]] = []
        params = self.window.drr_controller._read_drr_params()
        self.assertEqual(tuple(params["selected_files"]), (self.source.name,))
        self.assertEqual(params["baseline_mode"], "External")
        self.assertEqual(tuple(params["baseline_files"]), (self.background.name,))

        def record_dispatch(mode: str) -> None:
            dispatches.append((mode, threading.get_ident()))

        with patch(
            "ui_qt.main_window.resolve_drr_background_assignments",
            side_effect=AssertionError("baseline change resolved synchronously"),
        ) as resolve, patch.object(self.window, "_start_load", side_effect=record_dispatch):
            changed = self.window._ensure_loaded_matches_drr_params()

        self.assertFalse(changed)
        self.assertEqual(dispatches, [("DRR", threading.get_ident())])
        resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
