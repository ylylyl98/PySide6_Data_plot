"""Reference-list editing and native Quick input with isolated settings/data."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickBackgroundSelectionTests(unittest.TestCase):
    def test_reference_selection_workspace(self):
        with TemporaryDirectory() as temporary:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", QT_SCALE_FACTOR="1",
                       MPLBACKEND="agg", MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary,
                       XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, QML_DISK_CACHE_PATH=temporary)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataclasses import replace
    from threading import Event
    import time
    from types import SimpleNamespace
    from unittest.mock import patch
    from PySide6.QtCore import QPointF, QSettings, Qt, QUrl
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
    from PySide6.QtQml import QQmlComponent, QQmlEngine
    from PySide6.QtQuick import QQuickItem, QQuickWindow
    from PySide6.QtTest import QTest
    from core.drr_sources import DrrMeasurementAssignment
    from ui_qt.quick.plot_surface import register_plot_surface
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
    from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft

    app = QGuiApplication([])
    if os.name == "nt":
        # The Windows offscreen plugin has no system font database.
        font = Path(os.environ["WINDIR"]) / "Fonts" / "segoeui.ttf"
        QFontDatabase.addApplicationFont(str(font))
        app.setFont(QFont("Segoe UI", 9))
    register_plot_surface()
    colors = dict(surface="#f3f5f8", control="#ffffff", text="#20242b", muted="#606978", border="#c6cbd3",
        accent="#0067c0", hover="#edf1f6", pressed="#e0e7ef", error="#b42318", disabled="#939aa5",
        selectedText="#ffffff", inset="#e9edf2", divider="#dce1e8", borderHover="#aeb7c4", borderDisabled="#dce1e8",
        controlDisabled="#edf0f4", focus="#20242b", accentText="#0067c0", selected="#e6f0fa", selectedBorder="#b4d6fa", errorSurface="#fdf1f0")

    def settle():
        deadline = time.monotonic() + .04
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.001)
        app.processEvents()

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory()
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name).resolve()
            self.folder = self.root / "measurements"
            self.folder.mkdir()
            self.names = tuple(f"YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_BNbackground_repeat_{index:02}.csv" for index in range(8))
            for name in (*self.names, "measurement.csv"):
                (self.folder / name).write_text("Vbg,Vtg,1.5,1.6\n0,0,10,20\n")
            self.external = self.root / "external_BNbackground.csv"
            self.external.write_text("Vbg,Vtg,1.5,1.6\n0,0,10,20\n")
            settings = QSettings(str(self.root / "test.ini"), QSettings.Format.IniFormat)
            self.bridge = DrrBackgroundBridge(settings=settings)
            self.addCleanup(self.bridge.close)
            self.context = BackgroundContext(str(self.folder), ("measurement.csv",))
            self.bridge.open(self.context, BackgroundDraft(kind="bn", which="all"))

        def wait(self):
            deadline = time.monotonic() + 5
            while self.bridge.snapshot["busy"] and time.monotonic() < deadline:
                settle()
            self.assertFalse(self.bridge.snapshot["busy"])
            self.assertFalse(self.bridge.snapshot["error"], self.bridge.snapshot["error"])

        def test_open_and_direct_browse_do_not_start_full_experiment_inspection(self):
            requested = []
            self.bridge.requestFiles.connect(lambda role, multiple: requested.append((role, multiple)))
            with patch("core.drr_sources.discover_drr_sources", side_effect=AssertionError("Unexpected full scan")) as scan:
                self.bridge.open(self.context, BackgroundDraft(kind="bn", which="all"))
                self.assertFalse(self.bridge.snapshot["busy"])
                self.bridge.invoke("browse:background-add")
                self.bridge.selectFiles("background-add", [self.names[0]])
                self.bridge.invoke("page:pn")
                self.assertFalse(self.bridge.snapshot["busy"])
                self.bridge.invoke("browse:P")
                self.bridge.selectFiles("P", ["Pbackground.csv"])
                self.assertEqual(self.bridge.draft.files, (self.names[0],))
                self.assertEqual(self.bridge.snapshot["pFiles"], ["Pbackground.csv"])
                self.assertEqual(requested, [("background-add", True), ("P", False)])
                scan.assert_not_called()

        def test_cancel_delayed_optional_catalog_does_not_overwrite_new_file_selection(self):
            started, release, finished = Event(), Event(), Event()
            self.addCleanup(release.set)

            def delayed_scan(*args, **kwargs):
                started.set()
                try:
                    if not release.wait(5):
                        raise AssertionError("Test did not release optional catalog scan")
                    return ()
                finally:
                    finished.set()

            with patch("core.drr_sources.discover_drr_sources", side_effect=delayed_scan) as scan:
                self.bridge.invoke("refresh")
                self.assertTrue(started.wait(2))
                self.assertTrue(self.bridge.snapshot["busy"])
                self.assertTrue(scan.call_args.kwargs["parallel_inspection"])
                self.assertIsNotNone(scan.call_args.kwargs["cache"])
                self.bridge.invoke("cancel-task")
                self.assertFalse(self.bridge.snapshot["busy"])
                self.bridge.selectFiles("background", [self.names[1]])
                release.set()
                self.assertTrue(finished.wait(2))
                settle()
            self.assertEqual(self.bridge.draft.files, (self.names[1],))
            self.assertFalse(self.bridge.snapshot["catalogLoaded"])
            self.assertEqual(self.bridge.snapshot["pnCatalog"], [])
            self.assertFalse(self.bridge.snapshot["busy"])
            self.assertFalse(self.bridge.snapshot["error"])

        def test_optional_catalog_cache_reinspects_only_changed_files(self):
            from core.drr_sources import inspect_csv_gate
            with patch("core.drr_sources.inspect_csv_gate", wraps=inspect_csv_gate) as inspect:
                self.bridge.invoke("refresh")
                self.wait()
                first_count = inspect.call_count
                self.assertGreater(first_count, 0)
                self.assertTrue(self.bridge.snapshot["catalogLoaded"])
                self.bridge.invoke("refresh")
                self.wait()
                self.assertEqual(inspect.call_count, first_count)
                with (self.folder / self.names[0]).open("a") as stream:
                    stream.write("1,1,11,21\n")
                self.bridge.invoke("refresh")
                self.wait()
                self.assertEqual(inspect.call_count, first_count + 1)

        def test_reopen_clears_catalog_rows_but_retains_current_saved_version(self):
            version = SimpleNamespace(version_id="current", number=1, created_at="2026-10-24T12:00:00Z",
                status="ready", status_reasons=(), pair=SimpleNamespace(p_files=("Pbackground.csv",), n_files=("Nbackground.csv",)))
            self.bridge._pn_catalog = ("obsolete row",)
            self.bridge._catalog_loaded = True
            self.bridge.open(self.context, BackgroundDraft(kind="pn", version=version, which="all"))
            self.assertFalse(self.bridge.snapshot["busy"])
            self.assertFalse(self.bridge.snapshot["catalogLoaded"])
            self.assertEqual(self.bridge.snapshot["pnCatalog"], [])
            self.assertEqual(self.bridge._versions, (version,))
            self.assertEqual(self.bridge.snapshot["choices"]["version"]["current"], 0)
            self.assertTrue(self.bridge.snapshot["canUseVersion"])

        def test_add_deduplicates_but_replace_retains_existing_picker_semantics(self):
            applied = []
            self.bridge.applyRequested.connect(applied.append)
            self.bridge.selectFiles("background", [self.names[0]])
            self.bridge.selectFiles("background-add", [QUrl.fromLocalFile(str(self.folder / self.names[0])),
                                                     self.names[1], QUrl.fromLocalFile(str(self.external))])
            self.assertEqual(self.bridge.draft.files, (self.names[0], self.names[1], str(self.external)))
            self.assertEqual(self.bridge.snapshot["referencePaths"][self.names[0]], str(self.folder / self.names[0]))
            self.bridge.selectFiles("background-add", [])
            self.assertEqual(len(self.bridge.draft.files), 3)
            self.bridge.removeBackgroundFile(self.names[1])
            self.assertEqual(self.bridge.draft.files, (self.names[0], str(self.external)))
            self.bridge.selectFiles("background", [self.names[2]])
            self.assertEqual(self.bridge.draft.files, (self.names[2],))
            self.bridge.clearBackgroundFiles()
            self.assertEqual(self.bridge.draft.files, ())
            self.assertFalse(applied)
            self.assertTrue((self.folder / self.names[1]).exists())
            self.assertTrue(self.external.exists())

        def test_matched_pool_edits_discard_obsolete_mapping_and_certificates(self):
            self.bridge.setChoice("method", 1)
            plan = SimpleNamespace(ready=True, rows=(), warnings=(), resolution=SimpleNamespace(assignments=()))
            assignment = DrrMeasurementAssignment("measurement.csv", "External", (self.names[0],), "all")
            self.bridge._draft = replace(self.bridge.draft, bn_candidates=(self.names[0],), bn_plan=plan,
                saved_assignments=(assignment,), calibration={"old": True}, bn_warnings_acknowledged=True,
                pn_warnings_acknowledged=True)
            self.bridge._bn_payload = {"eligible": True, "candidates": ()}
            self.bridge._flags["acknowledged"] = True
            self.bridge.selectFiles("background-add", [self.names[1]])
            self.assertEqual(self.bridge.draft.bn_candidates, self.names[:2])
            self.assertIsNone(self.bridge.draft.bn_plan)
            self.assertIsNone(self.bridge.draft.calibration)
            self.assertEqual(self.bridge.draft.saved_assignments, ())
            self.assertFalse(self.bridge.draft.bn_warnings_acknowledged)
            self.assertFalse(self.bridge.draft.pn_warnings_acknowledged)
            self.assertFalse(self.bridge.snapshot["flags"]["acknowledged"])
            self.assertFalse(self.bridge.snapshot["canUseCalibrated"])
            self.bridge.removeBackgroundFile(self.names[0])
            self.assertEqual(self.bridge.draft.bn_candidates, (self.names[1],))
            self.bridge.clearBackgroundFiles()
            self.assertEqual(self.bridge.draft.bn_candidates, ())
            self.assertFalse(self.bridge.snapshot["candidatePoolAutomatic"])
            self.bridge.invoke("match")
            deadline = time.monotonic() + 5
            while self.bridge.snapshot["busy"] and time.monotonic() < deadline:
                settle()
            self.assertFalse(self.bridge.snapshot["busy"])
            self.assertEqual(self.bridge.draft.bn_plan.candidate_files, ())
            self.assertFalse(self.bridge.draft.bn_plan.ready)

        def test_pn_pool_is_separate_and_all_file_edits_lock_during_application(self):
            self.bridge.setChoice("kind", 1)
            self.bridge.setChoice("method", 1)
            self.bridge.selectFiles("background", ["Pbackground.csv"])
            self.bridge.selectFiles("background-add", ["Pbackground_repeat.csv"])
            self.assertEqual(self.bridge.draft.pn_candidates, ("Pbackground.csv", "Pbackground_repeat.csv"))
            self.assertIsNone(self.bridge.draft.bn_candidates)
            self.bridge.application_started()
            self.bridge.selectFiles("background-add", ["Pbackground_other.csv"])
            self.bridge.removeBackgroundFile("Pbackground.csv")
            self.bridge.clearBackgroundFiles()
            self.assertEqual(self.bridge.draft.pn_candidates, ("Pbackground.csv", "Pbackground_repeat.csv"))
            self.bridge.application_finished()
            self.bridge.clearBackgroundFiles()
            self.assertEqual(self.bridge.draft.pn_candidates, ())

        def test_qml_source_tabs_tools_and_inline_reference_actions(self):
            self.bridge.selectFiles("background", list(self.names))
            # An existing generic reference has no forced BN/P/N relabel.
            self.bridge.setChoice("kind", 6)
            self.bridge.selectFiles("background", [self.names[0]])
            custom_draft = self.bridge.draft
            engine = QQmlEngine()
            warnings = []
            engine.warnings.connect(lambda messages: warnings.extend(message.toString() for message in messages))
            path = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml" / "DrrBackgroundSettings.qml"
            component = QQmlComponent(engine, QUrl.fromLocalFile(str(path)))
            item = component.createWithInitialProperties(dict(bridge=self.bridge, colors=colors))
            self.assertIsNotNone(item, "\n".join(error.toString() for error in component.errors()))
            window = QQuickWindow()
            self.addCleanup(window.close)
            item.setParentItem(window.contentItem())
            item.setWidth(1020)
            item.setHeight(800)
            window.resize(1020, 800)
            window.show()
            settle()

            def find(name):
                def visit(parent):
                    if parent.objectName() == name:
                        return parent
                    for child in parent.childItems():
                        found = visit(child)
                        if found is not None:
                            return found
                    return None
                found = visit(window.contentItem())
                self.assertIsNotNone(found, name)
                return found

            def click(button):
                point = button.mapToScene(QPointF(button.width() / 2, button.height() / 2)).toPoint()
                QTest.mouseClick(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
                settle()

            tabs = find("drr_background_source_tabs")
            source_keys = ("self", "bn", "p", "n", "pn", "fitted")
            source_buttons = [find("drr_background_source_" + key) for key in source_keys]
            self.assertEqual(tabs.property("currentIndex"), -1)
            self.assertEqual(self.bridge.draft, custom_draft)
            self.assertFalse(any(button.property("checked") for button in source_buttons))
            self.assertEqual([button.property("text") for button in source_buttons],
                             ["Self", "BN", "P", "N", "P + N", "Fitted"])
            click(source_buttons[1])
            self.assertEqual(self.bridge.draft.kind, "bn")
            self.assertEqual(self.bridge.draft.files, self.names)
            with patch.object(self.bridge, "setChoice", wraps=self.bridge.setChoice) as choose:
                click(source_buttons[1])
                choose.assert_not_called()
            click(source_buttons[0])
            self.assertEqual(self.bridge.draft.kind, "self")
            self.assertFalse(item.property("inlineReferencesVisible"))
            self.assertEqual(find("drr_background_choice_frame").property("count"), 2)
            source_buttons[0].forceActiveFocus()
            QTest.keyClick(window, Qt.Key.Key_Right)
            settle()
            self.assertEqual(self.bridge.draft.kind, "bn")
            self.assertEqual(self.bridge.draft.files, self.names)
            self.assertEqual(tabs.property("currentIndex"), 1)

            # Routine selection has one tools entry, with the empty preview
            # and its navigation hidden until a tool is explicitly opened.
            preview_pane = find("drr_background_preview_pane")
            tools_button = find("drr_background_tools")
            back = find("drr_background_page_settings")
            self.assertFalse(preview_pane.isVisible())
            self.assertTrue(tools_button.isVisible())
            self.assertFalse(back.isVisible())
            draft = self.bridge.draft
            click(tools_button)
            click(find("drr_background_tool_raw"))
            self.assertEqual(self.bridge.snapshot["page"], "raw")
            self.assertEqual(self.bridge.draft, draft)
            self.assertFalse(self.bridge.snapshot["catalogLoaded"])
            self.assertTrue(preview_pane.isVisible())
            self.assertTrue(back.isVisible())
            item.setProperty("invalidFields", {"test": True})
            settle()
            self.assertFalse(tools_button.isEnabled())
            self.assertFalse(back.isEnabled())
            item.setProperty("invalidFields", {})
            settle()
            click(back)
            self.assertEqual(self.bridge.snapshot["page"], "settings")
            self.assertEqual(self.bridge.draft, draft)
            self.assertFalse(preview_pane.isVisible())
            self.assertEqual(find("drr_background_cancel").property("text"), "Close")
            self.bridge._texts["report"] = "Synthetic last-preview diagnostic"
            self.bridge._refresh()
            settle()
            click(find("drr_background_diagnostics"))
            self.assertTrue(find("drr_background_settings_report").isVisible())
            self.assertFalse(preview_pane.isVisible())
            self.assertFalse(find("drr_background_plot").isVisible())
            self.assertEqual(self.bridge.draft, draft)
            click(find("drr_background_diagnostics"))
            self.assertFalse(find("drr_background_settings_report").isVisible())
            self.assertFalse(preview_pane.isVisible())
            self.bridge._texts["report"] = ""
            self.bridge._refresh()
            settle()

            deadline = time.monotonic() + 5
            while self.bridge.inlineFileSelectionBridge.snapshot["busy"] and time.monotonic() < deadline:
                settle()
            listing = find("drr_background_chosen_files")
            self.assertGreater(listing.height(), 170)
            self.assertEqual(listing.property("count"), len(self.names))
            click(find("drr_background_inline_remove_" + self.names[0]))
            self.assertEqual(self.bridge.draft.files, self.names[1:])
            requested = []
            self.bridge.requestFiles.connect(lambda role, multiple: requested.append((role, multiple)))
            click(find("drr_background_inline_browse"))
            self.assertEqual(len(requested), 1)
            self.assertTrue(requested[0][0].startswith("inline-catalog:"))
            self.assertTrue(requested[0][1])
            self.assertFalse(self.bridge.fileSelectionBridge.snapshot["selectionOpen"])
            self.assertEqual(self.bridge.draft.files, self.names[1:])
            capture = os.environ.get("DPTK_BACKGROUND_SELECTION_CAPTURE")
            if capture:
                window.grabWindow().save(capture)
            self.assertFalse(window.grabWindow().isNull())
            click(find("drr_background_inline_clear"))
            self.assertEqual(self.bridge.draft.files, ())
            self.assertFalse(find("drr_background_inline_clear").isEnabled())
            self.assertTrue(listing.isVisible())
            self.assertEqual(listing.property("count"), 0)
            self.assertFalse(warnings, "\n".join(warnings))
            item.setParentItem(None)
            item.deleteLater()
            settle()

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
