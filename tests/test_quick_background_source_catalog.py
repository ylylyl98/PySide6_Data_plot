"""Background reference catalogs edit isolated drafts without inspecting spectra."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickBackgroundSourceCatalogProcessTests(unittest.TestCase):
    def test_background_catalog_in_isolated_gui_process(self):
        with TemporaryDirectory(prefix="quick-background-catalog-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                QT_SCALE_FACTOR="1", MPLBACKEND="agg", MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary,
                XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, QML_DISK_CACHE_PATH=temporary,
                PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BACKGROUND_CATALOG_TESTS=", result.stdout)
        print(result.stdout.strip())


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from threading import Event
    from dataclasses import replace
    from time import monotonic, sleep
    from types import SimpleNamespace
    from unittest.mock import patch

    from PySide6.QtCore import QEvent, QMetaObject, QObject, QPointF, QSettings, Qt, QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from PySide6.QtTest import QTest

    from tests.quick_selection_checks import assert_text_selection_palette
    from ui_qt.quick.appearance import quick_colors
    from ui_qt.quick.background_source_catalog import BackgroundSourceCatalogBridge
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
    from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft
    from ui_qt.typography import install_application_font

    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    install_application_font(app)

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="quick-background-catalog-")
            self.addCleanup(self.temporary.cleanup)
            self.base = Path(self.temporary.name)
            self.root = self.base / "experiment"
            self.raw = self.root / "Initial Data" / "REF"
            self.raw.mkdir(parents=True)
            self.bn = [self.csv(f"sample_730nmc_BNbackground_repeat{index:02}.csv") for index in (1, 2)]
            self.p = self.csv("sample_730nmc_Pbackground.csv")
            self.n = self.csv("sample_730nmc_Nbackground.csv")
            self.measurement = self.csv("sample_730nmc_TG_measurement.csv")
            self.external = self.csv("outside_BNbackground.csv", folder=self.base / "external")
            self.settings = QSettings(str(self.base / "settings.ini"), QSettings.IniFormat)
            self.background = DrrBackgroundBridge(settings=self.settings)
            self.addCleanup(self.background.close)
            self.context = BackgroundContext(str(self.root), (self.portable(self.measurement),))
            self.background.open(self.context, BackgroundDraft(kind="bn", files=(self.portable(self.bn[0]),), which="all"))
            self.catalog = self.background.fileSelectionBridge
            self.applied, self.requested, self.opened = [], [], []
            self.background.applyRequested.connect(self.applied.append)
            self.background.requestFiles.connect(lambda role, multiple: self.requested.append((role, multiple)))
            self.background.fileSelectionRequested.connect(lambda: self.opened.append(True))

        def csv(self, name, folder=None):
            path = (folder or self.raw) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Vbg,Vtg,730,731\n0,0,10,20\n1,2,11,21\n", encoding="utf-8")
            return path.resolve()

        def portable(self, path):
            return path.relative_to(self.root).as_posix()

        def wait(self, predicate, timeout=8):
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                app.processEvents()
                if predicate():
                    app.processEvents()
                    return
                sleep(.003)
            self.fail("Background catalog did not reach the requested state")

        def open_catalog(self, action="choose-files"):
            previous = len(self.opened)
            self.background.invoke(action)
            self.assertEqual(len(self.opened), previous + 1)
            self.assertTrue(self.catalog.snapshot["selectionOpen"])
            self.wait(lambda: not self.catalog.snapshot["busy"])
            self.assertFalse(self.catalog.snapshot["error"], self.catalog.snapshot["error"])

        def visible_paths(self):
            return {row["id"] for row in self.catalog.filesModel.rows}

        def test_metadata_inventory_search_and_multiple_selection_never_scan_spectra(self):
            with patch("core.drr_sources.discover_drr_sources", side_effect=AssertionError("Full experiment inspected")) as discover, \
                 patch("core.drr_sources.inspect_csv_gate", side_effect=AssertionError("CSV grid inspected")) as inspect, \
                 patch("core.drr_sources.inspect_csv_wavelength_center", side_effect=AssertionError("CSV axis inspected")) as axis, \
                 patch("core.drr_composite_background.is_composite_background", side_effect=AssertionError("CSV header inspected")) as header:
                self.open_catalog()
                self.assertEqual(self.visible_paths(), {str(path) for path in self.bn})
                self.assertEqual(self.catalog.groupsModel.rowCount(), 0)
                self.assertTrue(self.catalog.snapshot["backgroundSelection"])
                self.assertEqual(self.catalog.snapshot["acceptLabel"], "Use selected")
                self.catalog.setParameter("query", "repeat02")
                self.assertEqual(self.visible_paths(), {str(self.bn[1])})
                self.catalog.selectFile(str(self.bn[1]), True)
                self.assertEqual(set(self.catalog.snapshot["selected"]), {str(path) for path in self.bn})
                self.catalog.setParameter("query", "")
                self.catalog.setParameter("showAllFiles", True)
                self.assertTrue({str(self.p), str(self.n), str(self.measurement)} <= self.visible_paths())
                self.catalog.invoke("refresh")
                self.wait(lambda: not self.catalog.snapshot["busy"])
                discover.assert_not_called()
                inspect.assert_not_called()
                axis.assert_not_called()
                header.assert_not_called()
            self.assertEqual(self.background.draft.files, (self.portable(self.bn[0]),))
            self.assertEqual(self.applied, [])

        def test_cancel_discards_edits_and_accept_replaces_only_current_kind_draft(self):
            self.open_catalog()
            self.catalog.selectFile(str(self.bn[0]), False)
            self.catalog.selectFile(str(self.bn[1]), True)
            self.catalog.invoke("cancel")
            self.assertEqual(self.background.draft.files, (self.portable(self.bn[0]),))
            self.open_catalog()
            self.assertEqual(self.catalog.snapshot["selected"], [str(self.bn[0])])
            self.catalog.selectFile(str(self.bn[0]), False)
            self.catalog.selectFile(str(self.bn[1]), True)
            self.catalog.invoke("accept")
            self.assertFalse(self.catalog.snapshot["selectionOpen"])
            self.assertEqual(self.background.draft.files, (self.portable(self.bn[1]),))
            self.background.setChoice("kind", 1)
            self.open_catalog()
            self.assertEqual(self.visible_paths(), {str(self.p)})
            self.catalog.selectFile(str(self.p), True)
            self.catalog.invoke("accept")
            self.assertEqual(self.background.draft.kind, "p")
            self.assertEqual(self.background.draft.files, (self.portable(self.p),))
            self.background.setChoice("kind", 0)
            self.assertEqual(self.background.draft.files, (self.portable(self.bn[1]),))
            self.open_catalog()
            self.assertEqual(self.catalog.snapshot["selected"], [str(self.bn[1])])
            self.assertEqual(self.applied, [])

        def test_external_fallback_deduplicates_and_empty_clear_is_explicitly_accepted(self):
            self.open_catalog()
            self.catalog.invoke("browse")
            role, multiple = self.requested[-1]
            self.assertTrue(role.startswith("catalog:"))
            self.assertTrue(multiple)
            self.background.selectFiles(role, [QUrl.fromLocalFile(str(self.external)), str(self.bn[0]), str(self.external)])
            self.assertEqual(self.catalog.snapshot["selected"], [str(self.bn[0]), str(self.external)])
            self.assertEqual(self.background.draft.files, (self.portable(self.bn[0]),))
            self.catalog.invoke("accept")
            self.assertEqual(self.background.draft.files, (self.portable(self.bn[0]), str(self.external)))
            self.open_catalog()
            self.assertEqual(self.catalog.chosenModel.rowCount(), 2)
            self.catalog.invoke("clear")
            self.assertEqual(self.catalog.snapshot["selected"], [])
            self.assertTrue(self.catalog.snapshot["canAccept"])
            self.catalog.invoke("accept")
            self.assertEqual(self.background.draft.files, ())
            self.assertTrue(self.external.is_file())
            self.assertTrue(all(path.is_file() for path in self.bn))
            self.assertEqual(self.applied, [])

        def test_old_native_dialog_responses_cannot_edit_a_new_kind_or_closed_owner(self):
            self.open_catalog()
            self.catalog.invoke("browse")
            stale_role = self.requested[-1][0]
            self.background.setChoice("kind", 1)
            self.background.selectFiles(stale_role, [str(self.external)])
            self.assertEqual(self.background.draft.files, ())
            self.open_catalog()
            self.catalog.invoke("browse")
            current_role = self.requested[-1][0]
            self.assertNotEqual(stale_role, current_role)
            self.background.selectFiles(stale_role, [str(self.external)])
            self.assertEqual(self.catalog.snapshot["selected"], [])
            self.background.selectFiles(current_role, [QUrl.fromLocalFile(str(self.p))])
            self.assertEqual(self.catalog.snapshot["selected"], [str(self.p)])
            self.background.invoke("cancel")
            self.background.selectFiles(current_role, [str(self.n)])
            self.catalog.invoke("accept")
            self.assertEqual(self.background.draft.files, ())
            self.assertEqual(self.applied, [])

        def test_matched_pool_selection_invalidates_mapping_without_applying(self):
            self.background.setChoice("method", 1)
            self.background.selectFiles("background", [str(self.bn[0])])
            self.open_catalog()
            self.catalog.selectFile(str(self.bn[1]), True)
            self.catalog.invoke("accept")
            self.assertEqual(self.background.draft.bn_candidates, tuple(self.portable(path) for path in self.bn))
            self.assertIsNone(self.background.draft.bn_plan)
            self.assertFalse(self.background.draft.bn_warnings_acknowledged)
            self.background.setChoice("kind", 1)
            self.background.setChoice("kind", 0)
            self.assertEqual(self.background.snapshot["method"], "matched")
            self.assertEqual(self.background.snapshot["selectedCandidates"], [self.portable(path) for path in self.bn])
            self.open_catalog()
            self.assertEqual(set(self.catalog.snapshot["selected"]), {str(path) for path in self.bn})
            self.assertEqual(self.applied, [])

        def test_accepting_unchanged_selection_preserves_calibration_and_match_certificate(self):
            plan = SimpleNamespace(ready=True, rows=(), warnings=(), resolution=SimpleNamespace(assignments=()))
            calibration = {"certificate": "synthetic-current"}
            payload = {"eligible": True, "candidates": ()}
            self.background._draft = replace(self.background.draft, bn_candidates=(self.portable(self.bn[0]),),
                bn_plan=plan, calibration=calibration, bn_warnings_acknowledged=True)
            self.background._method = "matched"
            self.background._bn_payload = payload
            self.background._refresh()
            before = self.background.draft
            self.open_catalog()
            self.catalog.invoke("accept")
            self.assertIs(self.background.draft, before)
            self.assertIs(self.background.draft.bn_plan, plan)
            self.assertIs(self.background.draft.calibration, calibration)
            self.assertIs(self.background._bn_payload, payload)
            self.assertEqual(self.applied, [])

        def test_advanced_p_role_remains_single_validated_and_independent_of_n(self):
            self.background.setChoice("kind", 3)
            self.background.selectFiles("P", [str(self.p)])
            self.background.selectFiles("N", [str(self.n)])
            version = SimpleNamespace(version_id="current", number=1, created_at="2026-10-24T12:00:00Z",
                status="ready", status_reasons=())
            document = SimpleNamespace(sources=(), regions=())
            self.background._version, self.background._document = version, document
            self.background._refresh()
            self.open_catalog("choose-files:P")
            self.assertEqual(self.catalog.snapshot["selected"], [str(self.p)])
            self.assertEqual(self.visible_paths(), {str(self.p)})
            self.catalog.invoke("accept")
            self.assertIs(self.background._version, version)
            self.assertIs(self.background._document, document)
            self.open_catalog("choose-files:P")
            self.catalog.setParameter("showAllFiles", True)
            self.catalog.selectFile(str(self.n), True)
            self.assertEqual(self.catalog.snapshot["selected"], [str(self.n)])
            self.catalog.invoke("accept")
            self.assertEqual(self.background.snapshot["pFiles"], [self.portable(self.p)])
            self.assertEqual(self.background.snapshot["nFiles"], [self.portable(self.n)])
            self.assertIn("cannot be relabelled", self.background.snapshot["error"])
            self.open_catalog("choose-files:P")
            self.catalog.invoke("clear")
            self.assertTrue(self.catalog.snapshot["canAccept"])
            self.catalog.invoke("accept")
            self.assertEqual(self.background.snapshot["pFiles"], [])
            self.assertEqual(self.background.snapshot["nFiles"], [self.portable(self.n)])
            self.assertIsNone(self.background._version)
            self.assertIsNone(self.background._document)
            self.assertEqual(self.applied, [])

        def test_cancelled_inventory_cannot_replace_a_new_folder_session(self):
            from core.drr_sources import drr_source_paths
            entered, release = Event(), Event()
            self.addCleanup(release.set)
            alternate = self.base / "second-experiment"
            alternate_file = self.csv("second_Pbackground.csv", folder=alternate / "Initial Data" / "REF")

            def delayed(folder, **kwargs):
                if Path(folder).resolve() == self.root.resolve():
                    entered.set()
                    if not release.wait(8):
                        raise AssertionError("Inventory test was not released")
                return drr_source_paths(folder, **kwargs)

            chooser = BackgroundSourceCatalogBridge(settings=self.settings)
            self.addCleanup(chooser.close)
            with patch("core.drr_sources.drr_source_paths", side_effect=delayed):
                try:
                    chooser.open_selection(str(self.root), "bn", [str(self.bn[0])])
                    self.assertTrue(entered.wait(3))
                    chooser.invoke("cancel")
                    chooser.open_selection(str(alternate), "p", [str(alternate_file)])
                    release.set()
                    self.wait(lambda: not chooser.snapshot["busy"])
                finally:
                    release.set()
            self.assertEqual(chooser.snapshot["folder"], str(alternate.resolve()))
            self.assertEqual(chooser.snapshot["selected"], [str(alternate_file)])
            self.assertEqual({row["id"] for row in chooser.filesModel.rows}, {str(alternate_file)})

        def test_real_qml_selection_search_remove_clear_cancel_and_themes(self):
            self.open_catalog()
            engine = QQmlApplicationEngine()
            warnings = []
            engine.warnings.connect(lambda messages: warnings.extend(message.toString() for message in messages))
            engine.setInitialProperties(dict(bridge=self.catalog, paletteColors=quick_colors("light")))
            qml = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml"
            engine.loadData(b'''import QtQuick
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
    SourceCatalog {
        bridge: window.bridge; colors: window.colors
        parent: Basic.Overlay.overlay
        Component.onCompleted: open()
    }
}''', QUrl.fromLocalFile(str(qml / "background-source-catalog-test.qml")))
            self.assertTrue(engine.rootObjects(), "\n".join(warnings))
            window = engine.rootObjects()[0]
            wrappers = []

            def dispose():
                window.setVisible(False)
                engine.deleteLater()
                app.sendPostedEvents(None, QEvent.DeferredDelete)
                app.processEvents()
                wrappers.clear()
                self.assertFalse(warnings, "\n".join(warnings))

            self.addCleanup(dispose)

            def visual_items(parent):
                yield parent
                for child in parent.childItems():
                    yield from visual_items(child)

            def find(name):
                item = window.findChild(QObject, name)
                if item is None:
                    item = next((child for child in visual_items(window.contentItem()) if child.objectName() == name), None)
                self.assertIsNotNone(item, name)
                wrappers.append(item)
                return item

            def click(item):
                point = item.mapToScene(QPointF(item.width() / 2, item.height() / 2)).toPoint()
                QTest.mouseClick(window, Qt.LeftButton, Qt.NoModifier, point)
                QTest.qWait(30)

            dialog = find("quick_source_catalog")
            self.wait(lambda: dialog.property("visible"))
            QTest.qWait(40)
            self.assertIn("BN", dialog.property("title"))
            self.assertEqual(find("source_catalog_load").property("text"), "Use selected")
            self.assertFalse(find("source_catalog_selection_actions").property("visible"))
            hidden_filters = {"Saved DAT results", "Unprocessed only", "Include backgrounds", "All experiment folders"}
            self.assertFalse([item for item in visual_items(window.contentItem())
                if item.property("text") in hidden_filters and item.isVisible()])
            click(find("source_catalog_show_all"))
            self.assertIn(str(self.measurement), self.visible_paths())
            click(find("source_catalog_show_all"))
            self.assertNotIn(str(self.measurement), self.visible_paths())
            search = find("source_catalog_search")
            search.forceActiveFocus()
            for character in "repeat02":
                QTest.keyClick(window, ord(character), Qt.NoModifier)
            self.wait(lambda: self.catalog.filesModel.rowCount() == 1)
            click(find("source_catalog_file_" + self.bn[1].name))
            self.assertEqual(set(self.catalog.snapshot["selected"]), {str(path) for path in self.bn})
            click(find("source_catalog_remove_" + self.bn[0].name))
            self.assertEqual(self.catalog.snapshot["selected"], [str(self.bn[1])])
            self.catalog.setParameter("query", "")
            QTest.mouseMove(window, QPointF(500, 700).toPoint())
            QTest.qWait(30)
            for mode in ("light", "dark", "high-contrast"):
                window.setProperty("paletteColors", quick_colors(mode))
                QTest.qWait(30)
                assert_text_selection_palette(self, window, quick_colors(mode))
                for name in ("source_catalog_files", "source_catalog_chosen", "source_catalog_load"):
                    item = find(name)
                    self.assertGreater(item.width(), 30, name)
                    self.assertGreater(item.height(), 20, name)
                image = window.grabWindow()
                self.assertFalse(image.isNull())
                capture = os.environ.get("DPTK_BACKGROUND_CATALOG_CAPTURE")
                if capture:
                    destination = Path(capture) / f"background-source-catalog-{mode}.png"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    self.assertTrue(image.save(str(destination)))
            click(find("source_catalog_clear"))
            self.assertEqual(self.catalog.snapshot["selected"], [])
            self.assertTrue(find("source_catalog_load").property("enabled"))
            click(find("source_catalog_footer_close"))
            self.wait(lambda: not dialog.property("visible"))
            self.assertEqual(self.background.draft.files, (self.portable(self.bn[0]),))
            self.open_catalog()
            QMetaObject.invokeMethod(dialog, "open")
            self.wait(lambda: dialog.property("visible"))
            click(find("source_catalog_file_" + self.bn[1].name))
            click(find("source_catalog_load"))
            self.wait(lambda: not dialog.property("visible"))
            self.assertEqual(set(self.background.draft.files), {self.portable(path) for path in self.bn})
            self.assertEqual(self.applied, [])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"BACKGROUND_CATALOG_TESTS={result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
