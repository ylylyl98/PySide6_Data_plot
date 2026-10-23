from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow
from core.drr_sources import DrrMeasurementAssignment
from core import drr_sources
from core.loader import DataCube
import numpy as np
from ui_qt.controllers_drr import DrrController
from tests.ui_test_helpers import dispose_owned_window


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
            cube=DataCube(np.array([1., 2.]), np.array([0., 1.]), np.ones((2, 2)), 'Gate', 'DRR', 'DR/R'),
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
        dispose_owned_window(self.window)
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

    def test_restored_self_frame_choice_does_not_trigger_second_load(self):
        from dataclasses import replace
        for which in ("first", "all"):
            with self.subTest(which=which):
                assignment = replace(self.assignment, baseline_which=which)
                self.window.loaded.drr_assignments = (assignment,)
                self.window.loaded.drr_baseline_which = which
                resolved = drr_sources.DrrBackgroundResolution((assignment,))
                with patch.object(self.window, "_start_load") as start:
                    self.assertTrue(self.window.drr_controller._restore_saved_drr_recipe(resolved))
                    self.window._ensure_loaded_matches_drr_params()
                    start.assert_not_called()

    def test_default_selection_resolves_in_worker_and_newest_selection_wins(self):
        window = self.window
        window.loaded = None
        window._drr_assignments = ()
        window._drr_assignments_automatic = False
        window.drr_selected_files = []
        window._drr_baseline_user_selected = False
        window.drr_memory_files_chk.setChecked(False)
        window.drr_memory_centers_chk.setChecked(False)
        for name in ('first.csv', 'latest.csv'):
            (self.root / name).write_text('Vbg,Vtg,600,620,640,660,680,700,720,740\n'
                '0,0,10,12,14,16,18,20,22,24\n1,0,20,24,28,32,36,40,44,48\n')
        entered, release = threading.Event(), threading.Event()
        gui_thread = threading.get_ident()
        resolving, errors = [], []
        original = drr_sources.resolve_drr_background_assignments

        def resolve(folder, catalog, selected, **kwargs):
            resolving.append((tuple(selected), threading.get_ident()))
            if selected == ['first.csv'] and not kwargs:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('Test did not release resolver')
            return original(folder, catalog, selected, **kwargs)

        try:
            with patch.object(DrrController, '_open_drr_source_dialog',
                              side_effect=[['first.csv'], ['latest.csv']]), patch(
                    'ui_qt.main_window.resolve_drr_background_assignments', side_effect=resolve), patch(
                    'ui_qt.controllers_drr.resolve_drr_background_assignments',
                    side_effect=AssertionError('GUI resolver called')), patch.object(
                    window, '_show_error', side_effect=errors.append):
                window.drr_controller._edit_drr_measurements()
                self.assertTrue(entered.wait(3))
                QTest.qWait(20)  # The event loop stays responsive while resolution is blocked.
                self.assertTrue(window._load_in_progress)
                self.assertIn('Matching background', window.data_state_label.text())
                old_token = window._active_load_token
                old_cancel = window._active_load_cancel
                window.drr_controller._edit_drr_measurements()
                self.assertTrue(old_cancel.cancelled)
                self.assertEqual(window._pending_load_options.selected_files, ['latest.csv'])
                release.set()
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    QTest.qWait(10)
                    if window.loaded is not None and not window._load_in_progress:
                        break
                self.assertIsNotNone(window.loaded, errors)
                self.assertEqual(window.loaded.selected_files, ['latest.csv'])
                np.testing.assert_allclose(window.loaded.cube.Z, [[-.5]*8, [0.]*8])
                self.assertTrue(window._drr_baseline_user_selected)
                self.assertTrue(all(thread != gui_thread for _, thread in resolving))
                self.assertEqual(errors, [])
                current_label = window.data_state_label.text()
                window._on_fitted_load_progress(old_token, message='DRR load: Obsolete matching')
                self.assertEqual(window.data_state_label.text(), current_label)
        finally:
            release.set()
            window.thread_pool.waitForDone(5000)

    def test_deferred_saved_recipe_restores_controls_and_background_provenance(self):
        window = self.window
        window.loaded = None
        window._drr_assignments = ()
        window._drr_assignments_automatic = False
        window._drr_baseline_user_selected = False
        assignment = DrrMeasurementAssignment(self.source.name, 'External', (self.background.name,), 'all')
        resolution = drr_sources.DrrBackgroundResolution(assignments=(assignment,))
        captured = []
        signals = SimpleNamespace(emit=lambda *_: None)
        cube = DataCube(np.array([1., 2.]), np.array([0., 1.]), np.ones((2, 2)), 'Gate', 'DRR', 'DR/R')
        with patch.object(window.thread_pool, 'start', captured.append), patch(
                'ui_qt.main_window.resolve_drr_background_assignments', return_value=resolution) as resolve:
            window._start_load('DRR')
            resolve.assert_not_called()
            with patch.object(window, '_provenance_records_for_load', return_value=()) as provenance, patch(
                    'ui_qt.main_window.data_io.load_drr_resolved_cube', return_value=cube):
                loaded = window._load_task_data(captured[0].args[0], progress=signals, log=signals)
            self.assertEqual(provenance.call_args.args[0].baseline_files, [self.background.name])
            with patch.object(window, '_plot_mode'), patch.object(window, '_apply_auto_limits_for_loaded'):
                window._on_loaded(loaded, request_token=window._active_load_token)
            self.assertEqual(window.drr_baseline_combo.currentText(), 'External')
            self.assertEqual(window.drr_baseline_files_manual, [self.background.name])
            self.assertTrue(window._drr_assignments_automatic)
            self.assertFalse(window.drr_pin_baseline_chk.isEnabled())
            window._on_load_finished(request_token=window._active_load_token)

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
