"""Real reference-file gestures, draft synchronization and complete clearing."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickBackgroundSelectionGestureProcessTests(unittest.TestCase):
    def test_reference_file_gestures_in_isolated_process(self):
        with TemporaryDirectory(prefix="quick-background-gestures-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                QT_SCALE_FACTOR="1", MPLBACKEND="agg", MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary,
                XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, QML_DISK_CACHE_PATH=temporary,
                PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BACKGROUND_GESTURE_TESTS=", result.stdout)
        print(result.stdout.strip())


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataclasses import replace
    from time import monotonic, sleep
    from types import SimpleNamespace

    from PySide6.QtCore import Q_ARG, QCoreApplication, QEvent, QMetaObject, QObject, QPointF, QSettings, Qt, QUrl
    from PySide6.QtGui import QGuiApplication, QMouseEvent
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from PySide6.QtTest import QTest

    from ui_qt.quick.appearance import quick_colors
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
    from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft
    from ui_qt.typography import install_application_font

    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    install_application_font(app)

    def visual_items(parent):
        yield parent
        for child in parent.childItems():
            yield from visual_items(child)

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="quick-background-gestures-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name)
            self.raw = self.root / "experiment" / "Initial Data" / "REF"
            self.raw.mkdir(parents=True)
            self.folder = self.root / "experiment"
            self.paths = [self.csv(f"sample_730nmc_BNbackground_repeat_{index:02}.csv")
                          for index in range(32)]
            self.measurement = self.csv("sample_730nmc_TG.csv")
            self.external = self.csv("external_BNbackground.csv", self.root / "outside")
            settings = QSettings(str(self.root / "settings.ini"), QSettings.IniFormat)
            self.background = DrrBackgroundBridge(settings=settings)
            self.addCleanup(self.background.close)
            self.context = BackgroundContext(str(self.folder), (self.portable(self.measurement),))
            self.background.open(self.context, BackgroundDraft(kind="bn", which="all"))
            self.applied = []
            self.background.applyRequested.connect(self.applied.append)
            self.window = self.engine = None
            self.warnings, self.wrappers = [], []
            self.timestamp = 1000

        def csv(self, name, folder=None):
            path = (folder or self.raw) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Vbg,Vtg,730,731\n0,0,10,20\n1,2,11,21\n", encoding="utf-8")
            return path.resolve()

        def portable(self, path):
            return path.relative_to(self.folder).as_posix() if path.is_relative_to(self.folder) else str(path)

        def wait(self, predicate, timeout=8):
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                app.processEvents()
                if predicate():
                    app.processEvents()
                    return
                sleep(.003)
            self.fail("Reference gesture UI did not reach its requested state")

        def show(self, *, modal=False, matched=False, selected=()):
            if matched:
                self.background.setChoice("method", 1)
            self.background.selectFiles("background", [str(path) for path in selected])
            if modal:
                self.background.invoke("choose-files")
                self.catalog = self.background.fileSelectionBridge
                body = """SourceCatalog {
                    bridge: window.bridge; colors: window.colors
                    parent: Basic.Overlay.overlay
                    Component.onCompleted: open()
                }"""
            else:
                self.background.setInlineSelectionVisible(True)
                self.catalog = self.background.inlineFileSelectionBridge
                body = """BackgroundFileBrowser {
                    bridge: window.bridge; colors: window.colors
                    anchors.fill: parent; anchors.margins: 12
                }"""
            self.modal = modal
            self.wait(lambda: self.catalog.snapshot["selectionOpen"] and not self.catalog.snapshot["busy"])
            self.engine = QQmlApplicationEngine()
            self.engine.warnings.connect(lambda messages: self.warnings.extend(message.toString() for message in messages))
            self.engine.setInitialProperties(dict(bridge=self.catalog, colors=quick_colors("light")))
            source = """import QtQuick
import QtQuick.Controls.Basic as Basic
Basic.ApplicationWindow {
    id: window
    required property var bridge
    required property var colors
    width: 1060; height: 600; visible: true
    color: colors.surface
    palette.window: colors.surface
    palette.windowText: colors.text
    palette.base: colors.control
    palette.text: colors.text
    palette.button: colors.control
    palette.buttonText: colors.text
    palette.highlight: colors.accent
    palette.highlightedText: colors.selectedText
    palette.placeholderText: colors.muted
    BODY
}""".replace("BODY", body)
            qml = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml"
            self.engine.loadData(source.encode(), QUrl.fromLocalFile(str(qml / "background-gestures-test.qml")))
            self.assertTrue(self.engine.rootObjects(), "\n".join(self.warnings))
            self.window = self.engine.rootObjects()[0]
            self.addCleanup(self.dispose)
            QTest.qWait(50)

        def dispose(self):
            if self.window is not None:
                self.window.setVisible(False)
            if self.engine is not None:
                self.engine.deleteLater()
                app.sendPostedEvents(None, QEvent.DeferredDelete)
                app.processEvents()
            self.wrappers.clear()
            self.assertFalse(self.warnings, "\n".join(self.warnings))

        def find(self, name):
            item = self.window.findChild(QObject, name)
            if item is None:
                item = next((child for child in visual_items(self.window.contentItem())
                             if child.objectName() == name), None)
            self.assertIsNotNone(item, name)
            self.wrappers.append(item)
            return item

        def row_name(self, path, *, chosen=False):
            prefix = ("source_catalog_chosen_" if chosen else "source_catalog_file_") if self.modal else (
                "drr_background_chosen_" if chosen else "drr_background_available_")
            return prefix + path.name

        def expose(self, path, *, chosen=False):
            view_name = ("source_catalog_chosen" if chosen else "source_catalog_files") if self.modal else (
                "drr_background_chosen_files" if chosen else "drr_background_available_files")
            view = self.find(view_name)
            model = self.catalog.chosenModel if chosen else self.catalog.filesModel
            index = model.indexOfId(str(path))
            self.assertGreaterEqual(index, 0)
            self.assertTrue(QMetaObject.invokeMethod(view, "positionViewAtIndex", Qt.DirectConnection,
                Q_ARG(int, index), Q_ARG(int, 0)))
            self.assertTrue(QMetaObject.invokeMethod(view, "forceLayout", Qt.DirectConnection))
            QTest.qWait(25)
            return self.find(self.row_name(path, chosen=chosen))

        def click(self, item):
            point = item.mapToScene(QPointF(item.width() * .55, item.height() / 2)).toPoint()
            QTest.mouseClick(self.window, Qt.LeftButton, Qt.NoModifier, point)
            QTest.qWait(25)

        def double_click(self, item):
            point = item.mapToScene(QPointF(item.width() * .55, item.height() / 2))
            global_point = self.window.mapToGlobal(point.toPoint()).toPointF()
            # Match QTest.mouseDClick's real window sequence, including both
            # presses. Sending only a synthetic double-click hides double toggles.
            for kind, buttons in ((QEvent.MouseButtonPress, Qt.LeftButton),
                                  (QEvent.MouseButtonRelease, Qt.NoButton),
                                  (QEvent.MouseButtonPress, Qt.LeftButton),
                                  (QEvent.MouseButtonDblClick, Qt.LeftButton),
                                  (QEvent.MouseButtonRelease, Qt.NoButton)):
                event = QMouseEvent(kind, point, global_point, Qt.LeftButton, buttons, Qt.NoModifier)
                self.timestamp += 20
                event.setTimestamp(self.timestamp)
                QCoreApplication.sendEvent(self.window, event)
                app.processEvents()
            QTest.qWait(25)

        def selected_names(self):
            return tuple(self.background._background_files())

        def accept_modal(self):
            if self.modal:
                self.click(self.find("source_catalog_load"))

        def assert_selected(self, paths):
            self.assertEqual(self.catalog.snapshot["selected"], [str(path) for path in paths])
            if not self.modal:
                self.assertEqual(self.selected_names(), tuple(self.portable(path) for path in paths))

        def test_inline_double_click_adds_once_and_repeated_add_preserves_matched_plan(self):
            self.show(matched=True, selected=self.paths[:1])
            self.double_click(self.expose(self.paths[1]))
            self.assert_selected(self.paths[:2])
            plan = SimpleNamespace(ready=True, rows=(), warnings=(), resolution=SimpleNamespace(assignments=()))
            self.background._draft = replace(self.background.draft, bn_plan=plan,
                calibration={"synthetic": "current"}, bn_warnings_acknowledged=True)
            self.background._refresh()
            before = self.background.draft
            self.double_click(self.expose(self.paths[1]))
            self.assert_selected(self.paths[:2])
            self.assertIs(self.background.draft, before)
            self.assertIs(self.background.draft.bn_plan, plan)
            self.assertEqual(self.applied, [])

        def test_inline_chosen_double_click_removes_and_checkbox_can_uncheck(self):
            self.show(selected=self.paths[:3])
            self.double_click(self.expose(self.paths[0], chosen=True))
            self.assert_selected(self.paths[1:3])
            row = self.expose(self.paths[1])
            self.click(self.find(row.objectName() + "_check"))
            self.assert_selected(self.paths[2:3])
            self.assertEqual(self.applied, [])

        def test_inline_keyboard_add_toggle_and_remove_use_focused_rows(self):
            self.show()
            row = self.expose(self.paths[0])
            row.forceActiveFocus(Qt.TabFocusReason)
            self.assertTrue(row.hasActiveFocus())
            QTest.keyClick(self.window, Qt.Key_Return)
            self.assert_selected(self.paths[:1])
            QTest.keyClick(self.window, Qt.Key_Return)
            self.assert_selected(self.paths[:1])
            QTest.keyClick(self.window, Qt.Key_Space)
            self.assert_selected([])
            QTest.keyClick(self.window, Qt.Key_Space)
            self.assert_selected(self.paths[:1])
            chosen = self.expose(self.paths[0], chosen=True)
            self.click(chosen)
            self.assertTrue(chosen.hasActiveFocus())
            QTest.keyClick(self.window, Qt.Key_Delete)
            self.assert_selected([])
            self.assertEqual(self.applied, [])

        def test_inline_clear_all_clears_hidden_and_offscreen_common_and_matched_files(self):
            self.show(selected=(*self.paths, self.external))
            for matched in (False, True):
                with self.subTest(matched=matched):
                    if matched:
                        self.background.setChoice("method", 1)
                        self.catalog.addExternal([str(path) for path in (*self.paths, self.external)])
                    self.catalog.setParameter("query", "repeat_31")
                    self.assertEqual(self.catalog.filesModel.rowCount(), 1)
                    self.assertEqual(self.catalog.chosenModel.rowCount(), 33)
                    chosen = self.find("drr_background_chosen_files")
                    self.assertGreater(chosen.property("contentHeight"), chosen.height())
                    clear = self.find("drr_background_inline_clear")
                    self.assertTrue(clear.isEnabled())
                    self.click(clear)
                    self.assert_selected([])
                    self.assertFalse(clear.isEnabled())
                    if matched:
                        self.assertEqual(self.background.draft.bn_candidates, ())
                        self.assertFalse(self.background.snapshot["candidatePoolAutomatic"])
                    self.assertTrue(all(path.is_file() for path in (*self.paths, self.external)))
            self.assertEqual(self.applied, [])

        def test_inline_refresh_busy_and_native_response_lifecycle_keep_selection_coherent(self):
            self.show(selected=self.paths[:2])
            self.catalog.invoke("browse")
            token = self.catalog.snapshot["selectionToken"]
            self.background.application_started()
            before = self.background.draft
            self.catalog.invoke("clear")
            self.catalog.selectFile(str(self.paths[0]), False)
            self.assertIs(self.background.draft, before)
            self.background.application_finished(error="Synthetic load failure")
            self.assertTrue(self.catalog.snapshot["editEnabled"])
            self.assertFalse(self.catalog.accept_external(token, [str(self.external)]))
            self.catalog.refresh(True)
            self.wait(lambda: not self.catalog.snapshot["busy"])
            self.click(self.find("drr_background_inline_remove_" + self.paths[0].name))
            self.assert_selected(self.paths[1:2])
            self.click(self.find("drr_background_inline_clear"))
            self.assert_selected([])
            self.assertEqual(self.applied, [])

        def test_modal_double_click_add_is_idempotent_and_accepts_current_pool(self):
            self.show(modal=True, matched=True, selected=self.paths[:1])
            self.double_click(self.expose(self.paths[1]))
            self.assert_selected(self.paths[:2])
            self.double_click(self.expose(self.paths[1]))
            self.assert_selected(self.paths[:2])
            self.accept_modal()
            self.assertEqual(self.selected_names(), tuple(self.portable(path) for path in self.paths[:2]))
            self.assertEqual(self.applied, [])

        def test_modal_repeated_add_and_accept_preserve_existing_match_and_calibration(self):
            self.show(modal=True, matched=True, selected=self.paths[:1])
            plan = SimpleNamespace(ready=True, rows=(), warnings=(), resolution=SimpleNamespace(assignments=()))
            self.background._draft = replace(self.background.draft, bn_plan=plan,
                calibration={"synthetic": "current"}, bn_warnings_acknowledged=True)
            self.background._refresh()
            before = self.background.draft
            self.double_click(self.expose(self.paths[0]))
            self.assert_selected(self.paths[:1])
            self.accept_modal()
            self.assertIs(self.background.draft, before)
            self.assertIs(self.background.draft.bn_plan, plan)
            self.assertEqual(self.applied, [])

        def test_modal_remove_and_clear_filtered_offscreen_files_update_draft_on_accept(self):
            self.show(modal=True, selected=(*self.paths, self.external))
            self.catalog.setParameter("query", "repeat_31")
            self.assertEqual(self.catalog.filesModel.rowCount(), 1)
            remove = self.find("source_catalog_remove_" + self.paths[0].name)
            self.click(remove)
            self.assertEqual(len(self.catalog.snapshot["selected"]), 32)
            self.assertNotIn(str(self.paths[0]), self.catalog.snapshot["selected"])
            self.click(self.find("source_catalog_clear"))
            self.assert_selected([])
            self.assertTrue(self.find("source_catalog_load").isEnabled())
            self.accept_modal()
            self.assertEqual(self.selected_names(), ())
            self.assertTrue(all(path.is_file() for path in (*self.paths, self.external)))
            self.assertEqual(self.applied, [])

        def test_modal_chosen_double_click_and_focused_delete_remove_only_target_references(self):
            self.show(modal=True, selected=self.paths[:3])
            self.double_click(self.expose(self.paths[0], chosen=True))
            self.assert_selected(self.paths[1:3])
            chosen = self.expose(self.paths[1], chosen=True)
            self.click(chosen)
            self.assertTrue(chosen.hasActiveFocus())
            QTest.keyClick(self.window, Qt.Key_Delete)
            self.assert_selected(self.paths[2:3])
            self.accept_modal()
            self.assertEqual(self.selected_names(), (self.portable(self.paths[2]),))
            self.assertEqual(self.applied, [])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    print(f"BACKGROUND_GESTURE_TESTS={result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
