"""Inline reference editing, lifecycle and compact Quick layouts on synthetic data."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickBackgroundInlineBrowserProcessTests(unittest.TestCase):
    def test_inline_background_browser_in_isolated_gui_process(self):
        with TemporaryDirectory(prefix="quick-background-inline-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                QT_SCALE_FACTOR="1", MPLBACKEND="agg", MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary,
                XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, QML_DISK_CACHE_PATH=temporary,
                PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BACKGROUND_INLINE_TESTS=", result.stdout)
        print(result.stdout.strip())


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataclasses import replace
    from time import monotonic, sleep
    from types import SimpleNamespace
    from unittest.mock import patch

    from PySide6.QtCore import QEvent, QPointF, QSettings, Qt, QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from PySide6.QtTest import QTest

    from ui_qt.quick.appearance import quick_colors
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
    from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft
    from ui_qt.quick.plot_surface import register_plot_surface
    from ui_qt.typography import install_application_font

    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    install_application_font(app)
    register_plot_surface()

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="quick-background-inline-")
            self.addCleanup(self.temporary.cleanup)
            self.base = Path(self.temporary.name).resolve()
            self.root = self.base / "experiment"
            self.raw = self.root / "Initial Data" / "REF"
            self.raw.mkdir(parents=True)
            self.bn = [self.csv("YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_"
                               f"BNbackground_reference_repeat_{index:02}.csv") for index in range(1, 13)]
            self.p = self.csv("sample_730nmc_Pbackground.csv")
            self.n = self.csv("sample_730nmc_Nbackground.csv")
            self.measurement = self.csv("sample_730nmc_TG_measurement.csv")
            self.external = self.csv("outside_BNbackground.csv", folder=self.base / "external")
            self.settings = QSettings(str(self.base / "settings.ini"), QSettings.IniFormat)
            self.background = DrrBackgroundBridge(settings=self.settings)
            self.addCleanup(self.background.close)
            self.context = BackgroundContext(str(self.root), (self.portable(self.measurement),))
            self.background.open(self.context, BackgroundDraft(kind="bn", files=(self.portable(self.bn[0]),), which="all"))
            self.catalog = self.background.inlineFileSelectionBridge
            self.applied, self.requested, self.modal_opened = [], [], []
            self.background.applyRequested.connect(self.applied.append)
            self.background.requestFiles.connect(lambda role, multiple: self.requested.append((role, multiple)))
            self.background.fileSelectionRequested.connect(lambda: self.modal_opened.append(True))

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
            self.fail("Inline background browser did not reach the requested state")

        def settle(self):
            QTest.qWait(45)
            app.processEvents()

        def open_inline(self):
            self.background.setInlineSelectionVisible(True)
            self.wait(lambda: self.catalog.snapshot["selectionOpen"] and not self.catalog.snapshot["busy"])
            self.assertFalse(self.catalog.snapshot["error"], self.catalog.snapshot["error"])

        def visible_paths(self):
            return {row["id"] for row in self.catalog.filesModel.rows}

        def test_metadata_only_inventory_and_edits_do_not_rescan_or_apply(self):
            from ui_qt.quick.background_source_catalog import BackgroundSourceCatalogBridge
            scan = BackgroundSourceCatalogBridge._scan_catalog
            opened = Path.open

            def metadata_open(path, *args, **kwargs):
                if path.suffix.casefold() == ".csv":
                    raise AssertionError("Reference browsing read CSV contents")
                return opened(path, *args, **kwargs)

            with patch("core.drr_sources.discover_drr_sources", side_effect=AssertionError("Full experiment inspected")) as discover, \
                 patch("core.drr_sources.inspect_csv_gate", side_effect=AssertionError("CSV grid inspected")) as inspect, \
                 patch("core.drr_sources.inspect_csv_wavelength_center", side_effect=AssertionError("CSV axis inspected")) as axis, \
                 patch("pathlib.Path.open", metadata_open), \
                 patch.object(BackgroundSourceCatalogBridge, "_scan_catalog", autospec=True, side_effect=scan) as scans:
                self.open_inline()
                self.assertEqual(self.visible_paths(), {str(path) for path in self.bn})
                initial_scans = scans.call_count
                self.assertGreater(initial_scans, 0)
                self.catalog.selectFile(str(self.bn[1]), True)
                self.assertEqual(self.background.draft.files, tuple(self.portable(path) for path in self.bn[:2]))
                self.catalog.selectFile(str(self.bn[0]), False)
                self.assertEqual(self.background.draft.files, (self.portable(self.bn[1]),))
                self.catalog.setParameter("query", "repeat_03")
                self.assertEqual(self.visible_paths(), {str(self.bn[2])})
                self.assertEqual(self.catalog.chosenModel.rowCount(), 1)
                self.catalog.setParameter("query", "")
                self.catalog.addExternal([QUrl.fromLocalFile(str(self.external))])
                self.assertEqual(self.background.draft.files, (self.portable(self.bn[1]), str(self.external)))
                self.catalog.invoke("clear")
                self.assertEqual(self.background.draft.files, ())
                self.background.setChoice("frame", 0)
                for kind in (1, 2, 3, 0):
                    self.background.setChoice("kind", kind)
                self.assertEqual(scans.call_count, initial_scans)
                discover.assert_not_called()
                inspect.assert_not_called()
                axis.assert_not_called()
            self.assertEqual(self.applied, [])
            self.assertEqual(self.modal_opened, [])
            self.assertFalse(self.background.snapshot["catalogLoaded"])
            self.assertFalse(self.settings.contains("drr/bn_auto_compare"))
            self.assertTrue(all(path.is_file() for path in (*self.bn, self.external)))

        def test_kind_and_method_drafts_remain_independent(self):
            self.open_inline()
            self.background.setChoice("method", 1)
            self.catalog.selectFile(str(self.bn[0]), True)
            self.catalog.selectFile(str(self.bn[1]), True)
            bn_pool = tuple(self.portable(path) for path in self.bn[:2])
            self.assertEqual(self.background.draft.bn_candidates, bn_pool)
            self.background.setChoice("kind", 1)
            self.assertEqual(self.catalog.snapshot["selected"], [])
            self.assertEqual(self.visible_paths(), {str(self.p)})
            self.catalog.selectFile(str(self.p), True)
            self.background.setChoice("kind", 2)
            self.assertEqual(self.catalog.snapshot["selected"], [])
            self.assertEqual(self.visible_paths(), {str(self.n)})
            self.catalog.selectFile(str(self.n), True)
            self.background.setChoice("kind", 3)
            self.assertEqual(self.catalog.snapshot["selected"], [])
            self.assertEqual(self.visible_paths(), {str(self.p), str(self.n)})
            self.background.setChoice("method", 1)
            self.catalog.selectFile(str(self.p), True)
            self.catalog.selectFile(str(self.n), True)
            pn_pool = (self.portable(self.p), self.portable(self.n))
            self.assertEqual(self.background.draft.pn_candidates, pn_pool)
            for kind, selected, method in ((0, bn_pool, "matched"),
                                           (1, (self.portable(self.p),), "common"),
                                           (2, (self.portable(self.n),), "common"),
                                           (3, pn_pool, "matched")):
                self.background.setChoice("kind", kind)
                self.assertEqual(self.background.snapshot["method"], method)
                self.assertEqual(self.background.draft.which, "all")
                self.assertEqual(tuple(self.background._background_files()), selected)
                self.assertEqual(self.catalog.snapshot["selected"], [str(self.root / name) for name in selected])
            self.assertEqual(self.applied, [])

        def test_noop_file_edit_preserves_current_match_and_calibration(self):
            plan = SimpleNamespace(ready=True, rows=(), warnings=(), resolution=SimpleNamespace(assignments=()))
            calibration = {"certificate": "synthetic-current"}
            payload = {"eligible": True, "candidates": ()}
            self.background._draft = replace(self.background.draft, bn_candidates=(self.portable(self.bn[0]),),
                bn_plan=plan, calibration=calibration, bn_warnings_acknowledged=True)
            self.background._method = "matched"
            self.background._bn_payload = payload
            self.background._refresh()
            self.open_inline()
            before = self.background.draft
            self.catalog.selectFile(str(self.bn[0]), True)
            self.catalog.addExternal([QUrl.fromLocalFile(str(self.bn[0]))])
            self.assertIs(self.background.draft, before)
            self.assertIs(self.background.draft.bn_plan, plan)
            self.assertIs(self.background.draft.calibration, calibration)
            self.assertIs(self.background._bn_payload, payload)
            self.catalog.selectFile(str(self.bn[1]), True)
            self.assertEqual(self.background.draft.bn_candidates, tuple(self.portable(path) for path in self.bn[:2]))
            self.assertIsNone(self.background.draft.bn_plan)
            self.assertIsNone(self.background.draft.calibration)
            self.assertFalse(self.background.draft.bn_warnings_acknowledged)
            self.assertFalse(self.background.snapshot["canUseCalibrated"])
            self.assertEqual(self.applied, [])

        def test_hidden_modes_stop_watches_and_ignore_stale_native_results(self):
            self.open_inline()

            def request():
                self.catalog.invoke("browse")
                role, multiple = self.requested[-1]
                self.assertTrue(role.startswith("inline-catalog:"))
                self.assertTrue(multiple)
                return role

            def stopped():
                self.assertFalse(self.catalog.snapshot["selectionOpen"])
                self.assertFalse(self.catalog._watch_active)
                self.assertFalse(self.catalog._watch_timer.isActive())
                self.assertEqual(self.catalog._watcher.directories(), [])

            for kind in (4, 5):
                stale = request()
                self.background.setChoice("kind", kind)
                stopped()
                before = self.background.draft
                self.background.selectFiles(stale, [str(self.external)])
                self.assertIs(self.background.draft, before)
                self.background.setChoice("kind", 0)
                self.assertTrue(self.catalog.snapshot["selectionOpen"])
            stale = request()
            self.background.invoke("page:raw")
            stopped()
            self.background.selectFiles(stale, [str(self.external)])
            self.background.invoke("page:settings")
            self.assertTrue(self.catalog.snapshot["selectionOpen"])
            stale = request()
            self.background.setChoice("method", 1)
            self.background.selectFiles(stale, [str(self.external)])
            self.assertNotIn(str(self.external), self.catalog.snapshot["selected"])
            current = request()
            self.background.selectFiles(current, [QUrl.fromLocalFile(str(self.external))])
            self.assertIn(str(self.external), self.background.draft.bn_candidates)
            stale = request()
            self.background.invoke("cancel")
            stopped()
            before = self.background.draft
            self.background.selectFiles(stale, [str(self.n)])
            self.assertIs(self.background.draft, before)
            self.assertEqual(self.applied, [])

        def test_application_blocks_inline_edits_until_finished(self):
            self.open_inline()
            before = self.background.draft
            self.background.application_started()
            self.assertFalse(self.catalog.snapshot["editEnabled"])
            self.catalog.selectFile(str(self.bn[1]), True)
            self.catalog.addExternal([str(self.external)])
            self.catalog.invoke("clear")
            self.assertIs(self.background.draft, before)
            self.background.application_finished(error="Synthetic failed load retains the editable draft")
            self.assertTrue(self.catalog.snapshot["editEnabled"])
            self.catalog.selectFile(str(self.bn[1]), True)
            self.assertEqual(self.background.draft.files, tuple(self.portable(path) for path in self.bn[:2]))
            self.assertEqual(self.applied, [])

        def test_real_qml_opens_inline_and_fills_two_columns_in_both_themes(self):
            engine = QQmlApplicationEngine()
            warnings = []
            engine.warnings.connect(lambda messages: warnings.extend(message.toString() for message in messages))
            engine.setInitialProperties(dict(bridge=self.background, paletteColors=quick_colors("light")))
            qml = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml"
            engine.loadData(b'''import QtQuick
import QtQuick.Controls.Basic as Basic
Basic.ApplicationWindow {
    id: window
    required property var bridge
    required property var paletteColors
    readonly property var colors: paletteColors
    width: 1060; height: 760; visible: true
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
    DrrBackgroundSettings {
        anchors.fill: parent
        bridge: window.bridge
        colors: window.colors
    }
}''', QUrl.fromLocalFile(str(qml / "background-inline-test.qml")))
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
                item = next((child for child in visual_items(window.contentItem())
                             if child.objectName() == name), None)
                self.assertIsNotNone(item, name)
                wrappers.append(item)
                return item

            def click(item):
                point = item.mapToScene(QPointF(item.width() / 2, item.height() / 2)).toPoint()
                QTest.mouseClick(window, Qt.LeftButton, Qt.NoModifier, point)
                self.settle()

            self.wait(lambda: self.catalog.snapshot["selectionOpen"] and not self.catalog.snapshot["busy"])
            browser = find("drr_background_inline_browser")
            self.assertTrue(browser.isVisible())
            self.assertEqual(self.modal_opened, [])
            self.assertFalse(self.background.fileSelectionBridge.snapshot["selectionOpen"])
            self.assertFalse(find("drr_background_preview_pane").isVisible())
            available = find("drr_background_available_files")
            chosen = find("drr_background_chosen_files")
            self.assertEqual(available.property("count"), len(self.bn))
            self.assertEqual(chosen.property("count"), 1)
            search = find("drr_background_inline_search")
            search.forceActiveFocus()
            for character in "repeat_02":
                QTest.keyClick(window, ord(character), Qt.NoModifier)
            self.wait(lambda: self.catalog.filesModel.rowCount() == 1)
            click(find("drr_background_available_" + self.bn[1].name))
            self.assertEqual(self.background.draft.files, tuple(self.portable(path) for path in self.bn[:2]))
            self.assertEqual(chosen.property("count"), 2)
            click(find("drr_background_inline_remove_" + self.bn[0].name))
            self.assertEqual(self.background.draft.files, (self.portable(self.bn[1]),))
            click(find("drr_background_inline_browse"))
            self.assertEqual(len(self.requested), 1)
            self.background.selectFiles(self.requested[0][0], [QUrl.fromLocalFile(str(self.external))])
            self.assertEqual(chosen.property("count"), 2)
            self.assertEqual(self.applied, [])
            click(find("drr_background_inline_clear"))
            self.assertEqual(self.background.draft.files, ())
            self.assertFalse(find("drr_background_inline_clear").isEnabled())
            self.catalog.setParameter("query", "")
            self.catalog.addExternal([str(path) for path in self.bn[:8]])
            self.settle()
            self.assertEqual(chosen.property("count"), 8)
            for width, height in ((1060, 760), (760, 600)):
                window.resize(width, height)
                self.settle()
                for mode in ("light", "dark"):
                    window.setProperty("paletteColors", quick_colors(mode))
                    self.settle()
                    image = window.grabWindow()
                    self.assertFalse(image.isNull())
                    capture = os.environ.get("DPTK_BACKGROUND_INLINE_CAPTURE")
                    if capture:
                        destination = Path(capture) / f"background-inline-{width}x{height}-{mode}.png"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        self.assertTrue(image.save(str(destination)))
                    with self.subTest(width=width, height=height, theme=mode):
                        left = available.mapToScene(QPointF())
                        right = chosen.mapToScene(QPointF())
                        print(f"BACKGROUND_INLINE_VIEWPORT={width}x{height}:{mode}:"
                              f"top={left.y():g},available={available.height():g},chosen={chosen.height():g}")
                        self.assertGreater(available.width(), 220)
                        self.assertGreater(chosen.width(), 220)
                        self.assertGreaterEqual(right.x(), left.x() + available.width())
                        self.assertLessEqual(abs(left.y() - right.y()), 2)
                        self.assertLess(left.y(), 220)
                        self.assertGreaterEqual(available.height(), height * .50)
                        self.assertGreaterEqual(chosen.height(), height * .50)
                        for name in ("drr_background_inline_browse", "drr_background_inline_clear",
                                     "drr_background_frame_summary", "drr_background_choice_method",
                                     "drr_background_cancel", "drr_background_apply"):
                            action = find(name)
                            self.assertTrue(action.isVisible(), name)
                            point = action.mapToScene(QPointF())
                            self.assertGreaterEqual(point.x(), 0, name)
                            self.assertLessEqual(point.x() + action.width(), width + 1, name)
                            self.assertGreaterEqual(point.y(), 0, name)
                            self.assertLessEqual(point.y() + action.height(), height + 1, name)
            self.assertEqual(find("drr_background_frame_summary").property("text"),
                             "Average all frames · Equal file weight")
            for index, which, summary in ((0, "first", "First"), (1, "last", "Last"), (2, "all", "Average all")):
                click(find("drr_background_tools"))
                submenu = next((child for child in visual_items(window.contentItem())
                                if child.property("text") == "Reference frames" and child.isVisible()), None)
                self.assertIsNotNone(submenu, "Reference frames submenu is reachable from Advanced tools")
                click(submenu)
                click(find("drr_background_reference_frame_" + str(index)))
                self.assertEqual(self.background.draft.which, which)
                self.assertIn(summary, find("drr_background_frame_summary").property("text"))
                self.assertEqual(self.background.draft.files, tuple(self.portable(path) for path in self.bn[:8]))
            for kind, references in (("p", (self.p,)), ("n", (self.n,)), ("pn", (self.p, self.n))):
                click(find("drr_background_source_" + kind))
                if kind == "pn":
                    # P+N now starts in the dedicated create/saved workflow;
                    # ordinary CSV assignment remains an explicit saved route.
                    self.assertTrue(find("drr_pn_simple_workflow").isVisible())
                    self.background.setPnMode("saved")
                    self.settle()
                    find("drr_pn_choose_saved_files").forceActiveFocus()
                    QTest.keyClick(window, Qt.Key_Space)
                    self.settle()
                self.assertTrue(find("drr_background_inline_browser").isVisible())
                self.assertTrue(self.catalog.snapshot["selectionOpen"])
                self.assertEqual(self.catalog.snapshot["selected"], [])
                for path in references:
                    click(find("drr_background_available_" + path.name))
                self.assertEqual(self.background.draft.files, tuple(self.portable(path) for path in references))
                self.assertEqual(self.background.draft.which, "all")
            click(find("drr_background_source_self"))
            self.assertFalse(any(child.objectName() == "drr_background_inline_browser" and child.isVisible()
                                 for child in visual_items(window.contentItem())))
            self.assertFalse(self.catalog.snapshot["selectionOpen"])
            click(find("drr_background_source_bn"))
            self.assertTrue(find("drr_background_inline_browser").isVisible())
            self.assertEqual(self.background.draft.files, tuple(self.portable(path) for path in self.bn[:8]))
            self.assertEqual(self.modal_opened, [])
            self.assertEqual(self.applied, [])
            self.assertFalse("matplotlib.backends.backend_qtagg" in sys.modules,
                             "Inline QML unexpectedly loaded the Widgets Matplotlib backend")

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    print(f"BACKGROUND_INLINE_TESTS={result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
