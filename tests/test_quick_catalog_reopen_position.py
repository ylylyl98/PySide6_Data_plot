"""Center real QML picker rows on the latest successful source load."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickCatalogReopenPositionProcessTests(unittest.TestCase):
    def test_catalog_reopen_position_in_isolated_gui_process(self):
        with TemporaryDirectory(prefix="catalog-position-process-") as directory:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                QT_SCALE_FACTOR="1", MPLBACKEND="agg", MPLCONFIGDIR=directory, LOCALAPPDATA=directory,
                XDG_CONFIG_HOME=directory, XDG_CACHE_HOME=directory, QML_DISK_CACHE_PATH=directory,
                PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CATALOG_REOPEN_POSITION_TESTS=", result.stdout)
        print(result.stdout.strip())


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from time import monotonic

    from PySide6.QtCore import Q_ARG, QEvent, QMetaObject, QObject, QPoint, QPointF, QSettings, Qt, QUrl
    from PySide6.QtGui import QGuiApplication, QWheelEvent
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from PySide6.QtTest import QTest

    from ui_qt.quick.appearance import quick_colors
    from ui_qt.quick.source_catalog import SourceCatalogBridge
    from ui_qt.typography import install_application_font

    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    install_application_font(app)

    class ControlledCatalogBridge(SourceCatalogBridge):
        """Use production model updates; publish scan phases deterministically."""

        def refresh(self, force=False):
            self._generation += 1
            self._busy = True
            self._catalog_validated = False
            self._preserve_catalog = False
            self.snapshotChanged.emit()

        def _watch_refresh(self):
            pass

    def visual_items(parent):
        yield parent
        for child in parent.childItems():
            yield from visual_items(child)

    def entry(item):
        value = item.property("entry")
        return value.toVariant() if hasattr(value, "toVariant") else value

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="catalog-position-case-")
            self.base = Path(self.temporary.name).resolve()
            self.bridge = ControlledCatalogBridge(settings=QSettings(str(self.base / "settings.ini"), QSettings.IniFormat))
            self.bridge._workflow, self.bridge._folder, self.bridge._only_new = "drr", str(self.base), False
            self.catalog = self.fixture()
            self.bridge._catalog = self.catalog
            self.bridge._catalog_validated = True
            self.bridge._refresh_models()
            self.warnings, self.wrappers = [], []
            self.engine = QQmlApplicationEngine()
            self.engine.warnings.connect(lambda messages: self.warnings.extend(message.toString() for message in messages))
            self.engine.setInitialProperties(dict(bridge=self.bridge, paletteColors=quick_colors("light")))
            qml = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml"
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
    palette.highlight: colors.accent
    palette.highlightedText: colors.selectedText
    palette.placeholderText: colors.muted
    SourceCatalog { bridge: window.bridge; colors: window.colors; parent: Basic.Overlay.overlay }
}''', QUrl.fromLocalFile(str(qml / "catalog-reopen-position-test.qml")))
            self.assertTrue(self.engine.rootObjects(), "\n".join(self.warnings))
            self.window = self.engine.rootObjects()[0]
            self.dialog = self.item("quick_source_catalog")
            self.groups, self.files = self.item("source_catalog_groups"), self.item("source_catalog_files")

        def tearDown(self):
            self.window.setVisible(False)
            self.engine.deleteLater()
            self.bridge.close()
            app.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
            self.wrappers.clear()
            self.temporary.cleanup()
            self.assertFalse(self.warnings, "\n".join(self.warnings))

        def fixture(self):
            files, groups = [], []
            for group_number in range(30):
                members = []
                for file_number in range(36):
                    name = f"measurement-{group_number:02}-{file_number:02}.csv"
                    path = str(self.base / name)
                    members.append(path)
                    files.append(dict(id=path, source=name, name=name, kind="measurement", status="new",
                        saved=False, modified=1., processedAt="", background=False, detail="Synthetic measurement"))
                groups.append(dict(id=f"group-{group_number:02}", name=f"Acquisition group {group_number:02}",
                    session="Synthetic session", count=len(members), processedCount=0,
                    detail="Synthetic acquisition", members=members))
            return dict(files=files, groups=groups, sources=())

        def wait(self, predicate, message="Catalog QML did not settle", timeout=4):
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                app.processEvents()
                if predicate():
                    return
                QTest.qWait(5)
            self.fail(message)

        def item(self, name):
            item = self.window.findChild(QObject, name)
            if item is None:
                item = next((child for child in visual_items(self.window.contentItem()) if child.objectName() == name), None)
            self.assertIsNotNone(item, name)
            self.wrappers.append(item)
            return item

        def row(self, view, identity):
            for item in visual_items(view):
                if item.metaObject().indexOfProperty("entry") >= 0 and entry(item)["id"] == identity:
                    self.wrappers.append(item)
                    return item
            return None

        def row_visible(self, view, identity):
            item = self.row(view, identity)
            if item is None:
                return False
            y = item.mapToItem(view, QPointF(0, 0)).y()
            return y >= -1 and y + item.height() <= view.height() + 1

        def position(self, view, index):
            self.assertTrue(QMetaObject.invokeMethod(view, "positionViewAtIndex", Qt.DirectConnection,
                Q_ARG(int, index), Q_ARG(int, 0)))
            QTest.qWait(35)

        def click(self, item):
            point = item.mapToScene(QPointF(item.width() / 2, item.height() / 2)).toPoint()
            QTest.mouseClick(self.window, Qt.LeftButton, Qt.NoModifier, point)
            QTest.qWait(25)

        def open(self, selected, *, workflow="drr", saved_only=False):
            self.bridge.open(workflow, str(self.base), selected, saved_only)
            self.assertTrue(QMetaObject.invokeMethod(self.dialog, "open"))
            self.wait(lambda: self.dialog.property("visible"))
            QTest.qWait(60)

        def select_row(self, group_index, file_index):
            group = self.catalog["groups"][group_index]
            self.position(self.groups, group_index)
            self.click(self.row(self.groups, group["id"]))
            target = group["members"][file_index]
            self.position(self.files, file_index)
            self.click(self.row(self.files, target))
            self.assertIn(target, self.bridge.snapshot["selected"])
            self.assertEqual(self.bridge.snapshot["selectedGroup"], group["id"])
            return group["id"], target

        def select_lower_row(self):
            self.open([])
            return self.select_row(25, 30)

        def accept(self, *, succeeded=True):
            state = self.bridge.snapshot
            selected = list(state["selected"])
            self.click(self.item("source_catalog_load"))
            self.wait(lambda: not self.dialog.property("visible"))
            if succeeded:
                # The picker accepts a draft before the app's asynchronous load
                # succeeds. Only that successful completion records a bookmark.
                self.bridge.recordLoaded(state["workflow"], state["folder"], selected)
            return selected

        def reset_layout(self):
            # Model/layout recreation while hidden must not be the only place
            # that remembers where the last loaded source lives.
            self.bridge._catalog = dict(files=[], groups=[], sources=())
            self.bridge._refresh_models()
            QTest.qWait(30)
            self.bridge._catalog = self.catalog
            self.bridge._refresh_models()
            QTest.qWait(30)
            self.position(self.groups, 0)
            self.position(self.files, 0)

        def close_and_reset_layout(self):
            self.accept()
            self.reset_layout()

        def assert_target_visible(self, group, target):
            for view, identity, label in ((self.groups, group, "acquisition group"), (self.files, target, "measurement")):
                try:
                    self.wait(lambda: self.row_visible(view, identity), f"The remembered {label} remained offscreen")
                except AssertionError:
                    row = self.row(view, identity)
                    geometry = None if row is None else (row.mapToItem(view, QPointF(0, 0)).y(), row.height())
                    self.fail(f"The remembered {label} remained offscreen: target={identity}, "
                        f"row geometry={geometry}, viewport={view.height()}, contentY={view.property('contentY')}, "
                        f"currentIndex={view.property('currentIndex')}, pending={self.dialog.property('reopenPositionPending')}")

        def assert_row_centered(self, view, identity, label):
            row = self.row(view, identity)
            self.assertIsNotNone(row, f"The remembered {label} has no delegate: {identity}")
            center = row.mapToItem(view, QPointF(0, row.height() / 2)).y()
            self.assertAlmostEqual(center, view.height() / 2, delta=2,
                msg=f"The remembered {label} is visible but not centered: target={identity}, "
                f"row center={center}, viewport center={view.height() / 2}, "
                f"contentY={view.property('contentY')}")

        def assert_target_centered(self, group, target):
            self.assert_target_visible(group, target)
            self.assert_row_centered(self.groups, group, "acquisition group")
            self.assert_row_centered(self.files, target, "measurement")

        def test_reopen_centers_last_loaded_file_and_group_after_hidden_model_resets(self):
            group, target = self.select_lower_row()
            self.close_and_reset_layout()
            self.open([target])
            self.assert_target_centered(group, target)
            self.assertEqual(self.bridge.snapshot["selected"], [target])
            self.assertTrue(self.row(self.files, target).property("checked"))

        def test_main_clear_keeps_last_loaded_position_without_rechecking_the_file(self):
            group, target = self.select_lower_row()
            self.close_and_reset_layout()
            # The main Clear action passes an empty selection when reopening.
            self.open([])
            self.assert_target_centered(group, target)
            self.assertEqual(self.bridge.snapshot["selected"], [])
            self.assertFalse(self.row(self.files, target).property("checked"))
            self.assertFalse(self.item("source_catalog_load").property("enabled"))

        def test_incoming_selection_replaces_an_unrelated_previous_focus_and_group(self):
            self.select_lower_row()
            self.close_and_reset_layout()
            other_group = self.catalog["groups"][20]
            target = other_group["members"][29]
            self.bridge.recordLoaded("drr", str(self.base), [target])
            self.open([target])
            self.assertEqual(self.bridge.snapshot["focused"], target)
            self.assertEqual(self.bridge.snapshot["selectedGroup"], other_group["id"])
            self.assert_target_centered(other_group["id"], target)
            self.assertEqual(self.bridge.snapshot["selected"], [target])

        def test_repeated_cumulative_group_loads_move_bookmark_away_from_first_selection(self):
            first_group, first_file = self.select_lower_row()
            selected = self.accept()
            self.reset_layout()
            for group_index in (10, 17):
                with self.subTest(group_index=group_index):
                    group = self.catalog["groups"][group_index]
                    self.open(selected)
                    self.position(self.groups, group_index)
                    self.click(self.row(self.groups, group["id"]))
                    self.bridge.invoke("add-group")
                    self.assertEqual(self.bridge.snapshot["selected"][0], first_file)
                    selected = self.accept()
                    self.reset_layout()
                    self.open(selected)
                    focus = self.bridge.snapshot["focused"]
                    self.assertIn(focus, group["members"],
                        "The successful group load retained the first selection's bookmark")
                    self.assertEqual(self.bridge.snapshot["selectedGroup"], group["id"])
                    self.assertNotEqual(self.bridge.snapshot["selectedGroup"], first_group)
                    self.assert_target_visible(group["id"], focus)
                    # The group action can focus its first member, where normal
                    # list-edge clamping prevents exact file-row centering.
                    self.assert_row_centered(self.groups, group["id"], "latest loaded group")
                    self.click(self.item("source_catalog_footer_close"))
                    self.wait(lambda: not self.dialog.property("visible"))

        def test_cancelled_draft_browsing_does_not_replace_successful_load_bookmark(self):
            group, target = self.select_lower_row()
            self.close_and_reset_layout()
            self.open([target])
            _, draft = self.select_row(10, 18)
            self.assertEqual(self.bridge.snapshot["focused"], draft)
            self.click(self.item("source_catalog_footer_close"))
            self.wait(lambda: not self.dialog.property("visible"))
            self.reset_layout()
            # Clear in the main view removes all checkmarks; it must still use
            # the successful load rather than the cancelled picker's draft.
            self.open([])
            self.assertEqual(self.bridge.snapshot["focused"], target)
            self.assertEqual(self.bridge.snapshot["selectedGroup"], group)
            self.assert_target_centered(group, target)
            self.assertEqual(self.bridge.snapshot["selected"], [])
            self.assertFalse(self.row(self.files, target).property("checked"))

        def test_accepted_draft_without_success_does_not_replace_successful_load_bookmark(self):
            group, target = self.select_lower_row()
            self.close_and_reset_layout()
            self.open([target])
            self.select_row(10, 18)
            self.accept(succeeded=False)
            self.reset_layout()
            self.open([])
            self.assertEqual(self.bridge.snapshot["focused"], target)
            self.assertEqual(self.bridge.snapshot["selectedGroup"], group)
            self.assert_target_centered(group, target)
            self.assertEqual(self.bridge.snapshot["selected"], [])

        def test_cold_preview_missing_target_defers_position_until_final_catalog(self):
            group = self.catalog["groups"][26]
            target = group["members"][31]
            self.bridge._catalog = dict(files=[], groups=[], sources=())
            self.bridge._catalog_validated = False
            self.bridge._refresh_models()
            self.open([target])
            preview = dict(files=self.catalog["files"][:36], groups=self.catalog["groups"][:1], sources=())
            self.bridge._preview(self.bridge._generation, preview)
            QTest.qWait(70)
            self.assertEqual(self.bridge.filesModel.indexOfId(target), -1)
            self.bridge._finish(self.bridge._generation, self.catalog, "", 1., None)
            self.assert_target_visible(group["id"], target)
            self.assertEqual(self.bridge.snapshot["selectedGroup"], group["id"])
            self.assertEqual(self.bridge.snapshot["selected"], [target])
            # Once restoration finishes, a status-only final publication must
            # preserve subsequent browsing instead of dragging back to target.
            self.position(self.files, 2)
            before = self.files.property("contentY")
            changed = dict(self.catalog, files=[dict(row, detail="Updated history") for row in self.catalog["files"]])
            self.bridge._catalog = changed
            self.bridge._refresh_models()
            QTest.qWait(70)
            self.assertAlmostEqual(self.files.property("contentY"), before, delta=1.5)
            self.assertFalse(self.row_visible(self.files, target))

        def test_pl_incoming_selection_restores_the_direct_file_list(self):
            rows = [dict(row, kind="PL") for row in self.catalog["files"][:100]]
            target = rows[86]["id"]
            self.open([target], workflow="pl")
            self.bridge._finish(self.bridge._generation, dict(files=rows, groups=[], sources=()), "", 1., None)
            self.wait(lambda: self.row_visible(self.files, target), "The PL measurement remained offscreen")
            self.assertFalse(self.groups.property("visible"))
            self.assertEqual(self.bridge.snapshot["selected"], [target])
            self.assertTrue(self.row(self.files, target).property("checked"))

        def test_pl_warm_reopen_centers_last_loaded_file_after_hidden_reset(self):
            rows = [dict(row, kind="PL") for row in self.catalog["files"][:180]]
            catalog = dict(files=rows, groups=[], sources=())
            target = rows[146]["id"]
            self.open([], workflow="pl")
            self.bridge._finish(self.bridge._generation, catalog, "", 1., None)
            self.position(self.files, 146)
            self.click(self.row(self.files, target))
            self.accept()
            self.bridge._catalog = dict(files=[], groups=[], sources=())
            self.bridge._refresh_models()
            QTest.qWait(30)
            self.bridge._catalog = catalog
            self.bridge._refresh_models()
            self.position(self.files, 0)
            self.open([target], workflow="pl")
            self.wait(lambda: self.row_visible(self.files, target), "The PL file was not restored on reopen")
            self.assert_row_centered(self.files, target, "PL measurement")
            self.assertEqual(self.bridge.snapshot["focused"], target)
            self.assertTrue(self.row(self.files, target).property("checked"))

        def test_pl_cached_target_keeps_position_through_final_inventory(self):
            rows = [dict(row, kind="PL") for row in self.catalog["files"][:180]]
            target = rows[146]["id"]
            # The remembered file can be near the top of an optimistic cache;
            # newer acquisitions arrive ahead of it in the final inventory.
            self.open([target], workflow="pl")
            preview = dict(files=rows[144:156], groups=[], sources=())
            self.bridge._preview(self.bridge._generation, preview)
            QTest.qWait(70)
            self.assertTrue(self.dialog.property("reopenPositionPending"))
            self.bridge._finish(self.bridge._generation, dict(files=rows, groups=[], sources=()), "", 1., None)
            self.wait(lambda: self.row_visible(self.files, target), "Final PL inventory lost the remembered position")
            self.assert_row_centered(self.files, target, "PL measurement after inventory refresh")

        def test_pl_cached_target_is_revealed_while_full_inspection_is_still_busy(self):
            rows = [dict(row, kind="PL") for row in self.catalog["files"][:180]]
            target = rows[146]["id"]
            self.open([target], workflow="pl")
            self.bridge._preview(self.bridge._generation, dict(files=rows[100:160], groups=[], sources=()))
            self.wait(lambda: self.row_visible(self.files, target), "Cached PL target waited for a full scan")
            self.assertTrue(self.bridge.snapshot["busy"])
            self.assertTrue(self.dialog.property("reopenPositionPending"))
            self.bridge._finish(self.bridge._generation, dict(files=rows, groups=[], sources=()), "", 1., None)
            self.wait(lambda: not self.dialog.property("reopenPositionPending"))
            self.assert_row_centered(self.files, target, "PL measurement after slow inspection")

        def test_pl_browsing_cached_preview_cancels_late_reopen_position(self):
            rows = [dict(row, kind="PL") for row in self.catalog["files"][:180]]
            target, draft = rows[146]["id"], rows[2]["id"]
            self.open([target], workflow="pl")
            self.bridge._preview(self.bridge._generation, dict(files=rows[:12], groups=[], sources=()))
            self.wait(lambda: self.row_visible(self.files, draft))
            self.click(self.row(self.files, draft))
            self.assertFalse(self.dialog.property("reopenPositionPending"))
            self.bridge._finish(self.bridge._generation, dict(files=rows, groups=[], sources=()), "", 1., None)
            QTest.qWait(70)
            self.assertTrue(self.row_visible(self.files, draft))
            self.assertFalse(self.row_visible(self.files, target))
            self.assertEqual(self.bridge.snapshot["selected"], [draft])

        def test_filtered_target_finishes_without_mutating_search_or_selection(self):
            target = self.catalog["groups"][25]["members"][30]
            self.bridge.setParameter("query", "measurement-00")
            self.open([target])
            self.wait(lambda: not self.dialog.property("reopenPositionPending"))
            self.assertEqual(self.bridge.snapshot["query"], "measurement-00")
            self.assertEqual(self.bridge.snapshot["selected"], [target])
            self.assertEqual(self.bridge.filesModel.rowCount(), 36)
            self.assertEqual(self.bridge.filesModel.indexOfId(target), -1)
            self.assertEqual(self.bridge.snapshot["selectedGroup"], "")
            self.assertFalse(self.bridge.snapshot["reopenPositionPending"])

        def test_deleted_target_finishes_without_reintroducing_a_source_row(self):
            target = self.catalog["groups"][25]["members"][30]
            self.bridge._catalog = dict(self.catalog,
                files=[row for row in self.catalog["files"] if row["id"] != target])
            self.bridge._refresh_models()
            self.open([target])
            self.wait(lambda: not self.dialog.property("reopenPositionPending"))
            self.assertEqual(self.bridge.snapshot["selected"], [target])
            self.assertEqual(self.bridge.filesModel.indexOfId(target), -1)
            self.assertEqual(self.bridge.snapshot["query"], "")
            self.assertEqual(self.bridge.snapshot["selectedGroup"], "")
            self.assertFalse(self.bridge.snapshot["reopenPositionPending"])

        def test_group_click_during_cold_preview_cancels_the_pending_restore(self):
            target = self.catalog["groups"][26]["members"][31]
            self.bridge._catalog = dict(files=[], groups=[], sources=())
            self.bridge._catalog_validated = False
            self.bridge._refresh_models()
            self.open([target])
            preview = dict(files=self.catalog["files"][:36], groups=self.catalog["groups"][:1], sources=())
            self.bridge._preview(self.bridge._generation, preview)
            user_group = self.catalog["groups"][0]["id"]
            self.wait(lambda: self.row_visible(self.groups, user_group))
            self.click(self.row(self.groups, user_group))
            self.assertFalse(self.dialog.property("reopenPositionPending"))
            self.bridge._finish(self.bridge._generation, self.catalog, "", 1., None)
            QTest.qWait(70)
            self.assertEqual(self.bridge.snapshot["selectedGroup"], user_group)
            self.assertEqual(self.bridge.snapshot["selected"], [target])
            self.assertTrue(self.row_visible(self.groups, user_group))
            self.assertEqual(self.bridge.filesModel.indexOfId(target), -1)
            self.assertFalse(self.bridge.snapshot["reopenPositionPending"])

        def test_wheel_browsing_during_cold_preview_cancels_the_pending_restore(self):
            group = self.catalog["groups"][26]
            target = group["members"][31]
            self.bridge._catalog = dict(files=[], groups=[], sources=())
            self.bridge._catalog_validated = False
            self.bridge._refresh_models()
            self.open([target])
            preview = dict(files=self.catalog["files"][:20 * 36], groups=self.catalog["groups"][:20], sources=())
            self.bridge._preview(self.bridge._generation, preview)
            self.wait(lambda: self.groups.property("count") == 20)
            before = self.groups.property("contentY")
            point = self.groups.mapToScene(QPointF(self.groups.width() / 2, self.groups.height() / 2))
            global_point = self.window.mapToGlobal(point.toPoint())
            wheel = QWheelEvent(point, QPointF(global_point), QPoint(), QPoint(0, -120),
                Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
            app.sendEvent(self.window, wheel)
            self.wait(lambda: self.groups.property("contentY") > before + 1, "The real wheel event did not scroll the group list")
            self.wait(lambda: not self.dialog.property("reopenPositionPending"), "Wheel browsing did not cancel QML restoration")
            self.assertFalse(self.bridge.snapshot["reopenPositionPending"], "Wheel browsing did not cancel bridge restoration")
            QTest.qWait(90)
            position = self.groups.property("contentY")
            self.bridge._finish(self.bridge._generation, self.catalog, "", 1., None)
            QTest.qWait(90)
            self.assertEqual(self.bridge.snapshot["selectedGroup"], "")
            self.assertAlmostEqual(self.groups.property("contentY"), position, delta=1.5)
            self.assertFalse(self.row_visible(self.groups, group["id"]))
            self.assertEqual(self.bridge.snapshot["selected"], [target])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"CATALOG_REOPEN_POSITION_TESTS={result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
