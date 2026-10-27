"""Exercise the compact source picker actions against its real draft bridge."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import unittest

from PySide6.QtCore import QEvent, QMetaObject, QObject, QPointF, QSettings, Qt, QUrl, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtTest import QTest

from ui_qt.quick.appearance import quick_colors
from ui_qt.quick.source_catalog import SourceCatalogBridge
from ui_qt.typography import install_application_font


ROOT = Path(__file__).resolve().parents[1]
QML = ROOT / "ui_qt/quick/qml"


class _CatalogBridge(SourceCatalogBridge):
    def __init__(self, settings):
        super().__init__(settings=settings)
        self.calls = []

    @Slot(str)
    def invoke(self, action):
        self.calls.append(action)
        super().invoke(action)


def visual_items(parent):
    yield parent
    for child in parent.childItems():
        yield from visual_items(child)


class QuickSourceCatalogActionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QGuiApplication.instance() is None:
            QQuickStyle.setStyle("Basic")
        cls.app = QGuiApplication.instance() or QGuiApplication([])
        install_application_font(cls.app)

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="quick-catalog-actions-")
        self.root = Path(self.temporary.name).resolve()
        settings = QSettings(str(self.root / "settings.ini"), QSettings.IniFormat)
        self.bridge = _CatalogBridge(settings)
        self.paths = [str(self.root / f"measurement-{index}.csv") for index in (1, 2)]
        self.bridge._workflow = "drr"
        self.bridge._folder = str(self.root)
        self.bridge._selected_group = "group-one"
        self.bridge._catalog_validated = True
        self.bridge._catalog = dict(files=[dict(id=path, source=Path(path).name, name=Path(path).name,
            kind="measurement", status="new", saved=False, modified=1., processedAt="", background=False,
            detail="Synthetic measurement") for path in self.paths], groups=[dict(id="group-one", name="Synthetic group",
            session="Synthetic session", count=2, processedCount=0, detail="Reviewed measurements", members=self.paths)], sources=())
        self.bridge._refresh_models()
        self.accepted = []
        self.requests = []
        self.inspected_rows = []
        self.bridge.selectionAccepted.connect(lambda workflow, files: self.accepted.append((workflow, list(files))))
        self.bridge.dialogRequested.connect(lambda kind, multiple: self.requests.append((kind, multiple)))
        self.engine = QQmlApplicationEngine()
        self.warnings = []
        self.engine.warnings.connect(lambda messages: self.warnings.extend(item.toString() for item in messages))
        self.engine.setInitialProperties(dict(bridge=self.bridge, paletteColors=quick_colors("light")))
        self.engine.loadData(b'''import QtQuick
import QtQuick.Controls.Basic as Basic
Basic.ApplicationWindow {
    id: window
    required property var bridge
    required property var paletteColors
    readonly property var colors: paletteColors
    width: 1100; height: 780; visible: true
    color: colors.inset
    palette.window: colors.inset
    palette.windowText: colors.text
    palette.base: colors.control
    palette.text: colors.text
    palette.button: colors.control
    palette.buttonText: colors.text
    palette.light: colors.control
    palette.mid: colors.border
    palette.midlight: colors.pressed
    palette.dark: colors.border
    palette.highlight: colors.accent
    palette.highlightedText: colors.selectedText
    palette.placeholderText: colors.muted
    SourceCatalog {
        bridge: window.bridge; colors: window.colors
        parent: Basic.Overlay.overlay
        Component.onCompleted: open()
    }
}''', QUrl.fromLocalFile(str(QML / "source-catalog-actions-test.qml")))
        self.assertTrue(self.engine.rootObjects(), "\n".join(self.warnings))
        self.window = self.engine.rootObjects()[0]
        self.dialog = self.window.findChild(QObject, "quick_source_catalog")
        self.wait(lambda: self.dialog.property("visible"))
        QTest.qWait(40)

    def tearDown(self):
        self.window.setVisible(False)
        self.engine.deleteLater()
        self.bridge.close()
        self.app.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()
        self.inspected_rows.clear()
        self.temporary.cleanup()
        self.assertFalse(self.warnings, "\n".join(self.warnings))

    def wait(self, predicate):
        deadline = monotonic() + 3
        while monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            QTest.qWait(10)
        self.fail("Source picker did not reach the requested state")

    def item(self, name):
        obj = self.window.findChild(QObject, name)
        if obj is None:
            obj = next((item for item in visual_items(self.window.contentItem()) if item.objectName() == name), None)
        self.assertIsNotNone(obj, name)
        # Retain inspected Quick wrappers until their owning engine is disposed.
        self.inspected_rows.append(obj)
        return obj

    def click(self, item, *, double=False):
        point = item.mapToScene(QPointF(item.width() / 2, item.height() / 2)).toPoint()
        action = QTest.mouseDClick if double else QTest.mouseClick
        action(item.window(), Qt.LeftButton, Qt.NoModifier, point)
        QTest.qWait(30)

    def open_menu(self):
        self.click(self.item("source_catalog_selection_actions"))
        self.wait(lambda: self.item("source_catalog_selection_menu").property("visible"))

    def test_footer_is_compact_in_all_workflows_and_themes(self):
        footer = self.item("source_catalog_actions")
        for mode in ("light", "dark", "high-contrast"):
            self.window.setProperty("paletteColors", quick_colors(mode))
            QTest.qWait(20)
            labels = [item.property("text") for item in footer.childItems()
                      if item.property("visible") and item.metaObject().indexOfProperty("text") >= 0]
            self.assertEqual(labels, ["Browse files…", "Selection actions…", "Load selected", "Close"])
            self.assertTrue(self.item("source_catalog_close").property("visible"))
            capture = os.environ.get("DPTK_SOURCE_CATALOG_CAPTURE")
            if capture:
                path = Path(capture) / f"source-catalog-actions-{mode}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                self.assertTrue(self.window.grabWindow().save(str(path)))
        self.bridge.setParameter("savedOnly", True)
        self.assertFalse(self.item("source_catalog_selection_actions").property("visible"))
        self.bridge._workflow = "pl"
        self.bridge._refresh_models()
        self.assertFalse(self.item("source_catalog_selection_actions").property("visible"))

    def test_group_actions_keep_bridge_commands_and_disabled_state(self):
        self.open_menu()
        self.click(self.item("source_catalog_add_group"))
        self.assertIn("add-group", self.bridge.calls)
        self.assertEqual(self.bridge.snapshot["selected"], self.paths)
        self.open_menu()
        self.click(self.item("source_catalog_add_compatible"))
        self.assertIn("compatible", self.bridge.calls)
        self.bridge._busy = True
        self.bridge.snapshotChanged.emit()
        self.assertFalse(self.item("source_catalog_add_group").property("enabled"))
        self.assertFalse(self.item("source_catalog_add_compatible").property("enabled"))
        self.open_menu()
        calls = list(self.bridge.calls)
        self.click(self.item("source_catalog_add_group"))
        self.assertEqual(self.bridge.calls, calls)

    def test_clear_is_next_to_chosen_files_and_load_tracks_the_draft(self):
        clear = self.item("source_catalog_clear")
        load = self.item("source_catalog_load")
        self.assertFalse(clear.property("enabled"))
        self.assertFalse(load.property("enabled"))
        self.bridge.selectFile(self.paths[0], True)
        self.assertTrue(clear.property("enabled"))
        self.assertTrue(load.property("enabled"))
        self.click(clear)
        self.assertEqual(self.bridge.snapshot["selected"], [])
        self.assertFalse(load.property("enabled"))
        self.assertEqual(self.accepted, [])

    def test_close_button_footer_close_and_escape_reject_without_loading(self):
        self.bridge.selectFile(self.paths[0], True)
        for action in ("source_catalog_close", "source_catalog_footer_close", "escape"):
            with self.subTest(action=action):
                if not self.dialog.property("visible"):
                    QMetaObject.invokeMethod(self.dialog, "open")
                    self.wait(lambda: self.dialog.property("visible"))
                previous = self.bridge.calls.count("cancel")
                if action == "escape":
                    QTest.keyClick(self.window, Qt.Key_Escape)
                else:
                    self.click(self.item(action))
                self.wait(lambda: not self.dialog.property("visible"))
                self.assertEqual(self.bridge.calls.count("cancel"), previous + 1)
                self.assertEqual(self.accepted, [])

    def test_load_accepts_once_and_browse_does_not_commit(self):
        self.click(self.item("source_catalog_browse"))
        self.assertEqual(self.requests, [("catalog", True)])
        self.assertEqual(self.accepted, [])
        self.bridge.selectFile(self.paths[0], True)
        self.click(self.item("source_catalog_load"))
        self.wait(lambda: not self.dialog.property("visible"))
        self.assertEqual(self.accepted, [("drr", self.paths[:1])])
        self.assertNotIn("cancel", self.bridge.calls)

    def test_double_click_shortcuts_still_add_and_remove_draft_files(self):
        group = next(item for item in visual_items(self.window.contentItem())
                     if item.metaObject().indexOfProperty("highlighted") >= 0
                     and str(item.property("text")).startswith("Synthetic group"))
        self.inspected_rows.append(group)
        self.click(group, double=True)
        self.assertEqual(self.bridge.snapshot["selected"], self.paths)
        chosen = next(item for item in visual_items(self.window.contentItem())
                      if item.metaObject().indexOfProperty("highlighted") >= 0
                      and item.property("text") == "measurement-1.csv")
        # Keep inspected wrappers alive until the owning engine is disposed.
        self.inspected_rows.append(chosen)
        self.click(chosen, double=True)
        self.assertEqual(self.bridge.snapshot["selected"], self.paths[1:])
        self.assertEqual(self.accepted, [])

    def test_filename_double_click_checkbox_and_explicit_removal_share_one_draft(self):
        row = self.item("source_catalog_file_measurement-1.csv")
        self.inspected_rows.append(row)
        for _ in range(2):
            self.click(row, double=True)
            self.assertEqual(self.bridge.snapshot["selected"], self.paths[:1])
        self.click(self.item("source_catalog_file_measurement-1.csv_check"))
        self.assertEqual(self.bridge.snapshot["selected"], [])
        for number in (1, 2):
            self.click(self.item(f"source_catalog_file_measurement-{number}.csv"))
        self.assertEqual(self.bridge.snapshot["selected"], self.paths)
        remove = self.item("source_catalog_remove_measurement-1.csv")
        self.assertTrue(remove.isVisible())
        self.click(remove)
        self.assertEqual(self.bridge.snapshot["selected"], self.paths[1:])
        chosen = self.item("source_catalog_chosen_measurement-2.csv")
        self.inspected_rows.append(chosen)
        self.click(chosen)
        self.assertTrue(chosen.hasActiveFocus())
        QTest.keyClick(self.window, Qt.Key_Delete)
        self.assertEqual(self.bridge.snapshot["selected"], [])
        self.click(row)
        self.click(self.item("source_catalog_clear"))
        self.assertEqual(self.bridge.snapshot["selected"], [])
        self.assertEqual(self.accepted, [])


if __name__ == "__main__":
    unittest.main()
