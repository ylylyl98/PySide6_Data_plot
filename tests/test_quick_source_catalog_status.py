"""Render catalog status cards and preserve the reader's place through updates."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickSourceCatalogStatusProcessTests(unittest.TestCase):
    def test_source_catalog_status_in_isolated_gui_process(self):
        with TemporaryDirectory(prefix="quick-catalog-status-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                QT_SCALE_FACTOR="1", MPLBACKEND="agg", MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary,
                XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, QML_DISK_CACHE_PATH=temporary,
                PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CATALOG_STATUS_TESTS=", result.stdout)
        print(result.stdout.strip())


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataclasses import replace
    from time import monotonic, sleep

    from PySide6.QtCore import Q_ARG, QEvent, QMetaObject, QObject, QPointF, QSettings, Qt, QUrl
    from PySide6.QtGui import QColor, QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from PySide6.QtTest import QTest

    from core.drr_sources import DrrSource
    from ui_qt.quick.appearance import quick_colors
    from ui_qt.quick.source_catalog import SourceCatalogBridge, _drr_catalog_rows
    from ui_qt.typography import install_application_font

    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    install_application_font(app)

    def visual_items(parent):
        yield parent
        for child in parent.childItems():
            yield from visual_items(child)

    def entry(item):
        value = item.property("entry")
        return value.toVariant() if hasattr(value, "toVariant") else value

    def luminance(value):
        color = QColor(value)
        channels = [component / 12.92 if component <= .04045 else ((component + .055) / 1.055) ** 2.4
            for component in (color.redF(), color.greenF(), color.blueF())]
        return sum(weight * component for weight, component in zip((.2126, .7152, .0722), channels))

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="quick-catalog-status-")
            self.base = Path(self.temporary.name).resolve()
            self.bridge = SourceCatalogBridge(settings=QSettings(str(self.base / "settings.ini"), QSettings.IniFormat))
            self.bridge._workflow, self.bridge._folder, self.bridge._only_new = "drr", str(self.base), False
            self.bridge._catalog_validated = True
            self.engine = QQmlApplicationEngine()
            self.warnings, self.wrappers = [], []
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
    SourceCatalog {
        bridge: window.bridge; colors: window.colors
        parent: Basic.Overlay.overlay
        Component.onCompleted: open()
    }
}''', QUrl.fromLocalFile(str(qml / "source-catalog-status-test.qml")))
            self.assertTrue(self.engine.rootObjects(), "\n".join(self.warnings))
            self.window = self.engine.rootObjects()[0]
            self.wait(lambda: self.item("quick_source_catalog").property("visible"))

        def tearDown(self):
            self.window.setVisible(False)
            self.engine.deleteLater()
            self.bridge.close()
            app.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
            self.wrappers.clear()
            self.temporary.cleanup()
            self.assertFalse(self.warnings, "\n".join(self.warnings))

        def wait(self, predicate, timeout=4):
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                app.processEvents()
                if predicate():
                    return
                sleep(.003)
            self.fail("Catalog QML did not reach the requested state")

        def item(self, name):
            item = self.window.findChild(QObject, name)
            if item is None:
                item = next((child for child in visual_items(self.window.contentItem()) if child.objectName() == name), None)
            self.assertIsNotNone(item, name)
            self.wrappers.append(item)
            return item

        def publish(self, catalog):
            self.bridge._catalog = catalog
            self.bridge._refresh_models()
            QTest.qWait(60)

        def source(self, name, group, processed):
            return DrrSource(source=name, filename=name, group_key=group, session_date="2026-10-24",
                modified_time=1., is_background=False, processed=processed, classification="measurement",
                classification_reason="Synthetic complete measurement", gate_varies=True, frame_count=2,
                size_bytes=64, wavelength_center_nm=730., gate_grid=((0., 0.), (1., 2.)),
                spectral_grid=(730., 731.), gate_direction="Vbg increasing, Vtg increasing",
                gate_ranges=((0., 1.), (0., 2.)), gate_labels=("Vbg", "Vtg"), grid_complete=True)

        def test_group_and_file_statuses_remain_readable_in_all_themes(self):
            long_name = "YZ383_pwn1_9T_1.67KREF_730nmc_0p05sx20_RotIn94deg_TG-1BG=15_PIXIS"
            sources = tuple(self.source(f"{long_name}_{group}_rep{repeat:02}.csv", long_name + "_" + group, processed)
                for group, values in (("Unprocessed experiment", (False, False)),
                    ("Partially processed experiment", (False, True)), ("Processed experiment", (True, True)))
                for repeat, processed in enumerate(values, 1))
            catalog = _drr_catalog_rows(self.base, sources)
            self.publish(catalog)
            expected = {long_name + "_Unprocessed experiment": ("new", "Unprocessed"),
                long_name + "_Partially processed experiment": ("partial", "Partial"),
                long_name + "_Processed experiment": ("processed", "Processed")}
            groups = self.item("source_catalog_groups")
            self.bridge.selectFile(catalog["files"][0]["id"], True)
            for mode in ("light", "dark", "high-contrast"):
                self.bridge.setParameter("selectedGroup", "")
                self.window.setProperty("paletteColors", quick_colors(mode))
                QTest.qWait(30)
                for row in catalog["groups"]:
                    card = self.item("source_catalog_group_" + row["id"])
                    badge = self.item("source_catalog_group_status_" + row["id"])
                    state, label = expected[row["name"]]
                    self.assertEqual(badge.property("status"), state)
                    self.assertIn(label, badge.property("text"))
                    self.assertGreater(card.height(), badge.height())
                    self.assertLessEqual(card.width(), groups.width() + 1)
                    foreground, background = badge.property("foreground"), badge.property("color")
                    self.assertIsNotNone(foreground)
                    self.assertIsNotNone(background)
                    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
                    self.assertGreaterEqual((light + .05) / (dark + .05), 4.5, f"{mode}: {state}")
                    for text in visual_items(badge):
                        if text.metaObject().indexOfProperty("contentWidth") >= 0:
                            self.assertLessEqual(text.property("contentWidth"), text.width() + 1, f"{mode}: {label}")
                            self.assertLessEqual(text.property("contentHeight"), text.height() + 1, f"{mode}: {label}")
                for index, row in enumerate(catalog["files"]):
                    self.assertTrue(QMetaObject.invokeMethod(self.item("source_catalog_files"), "positionViewAtIndex",
                        Qt.DirectConnection, Q_ARG(int, index), Q_ARG(int, 0)))
                    QTest.qWait(10)
                    badge = self.item("source_catalog_file_status_" + row["name"])
                    self.assertEqual(badge.property("status"), row["status"])
                selected_group = next(row["id"] for row in catalog["groups"]
                    if catalog["files"][0]["id"] in row["members"])
                self.bridge.setParameter("selectedGroup", selected_group)
                QTest.qWait(30)
                image = self.window.grabWindow()
                self.assertFalse(image.isNull())
                capture = os.environ.get("DPTK_SOURCE_CATALOG_STATUS_CAPTURE")
                if capture:
                    destination = Path(capture) / f"source-catalog-status-{mode}.png"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    self.assertTrue(image.save(str(destination)))

        def test_saved_and_unknown_statuses_have_explicit_labels(self):
            self.bridge._workflow, self.bridge._type = "pl", "All"
            files = [dict(id=str(self.base / (state + ".csv")), source=state + ".csv", name=state + ".csv",
                kind="PL", status=state, saved=state == "saved", modified=1., processedAt="",
                background=False, detail="Synthetic processing status") for state in ("new", "processed", "unknown", "saved")]
            self.publish(dict(files=files, groups=[], sources=()))
            for mode in ("light", "dark", "high-contrast"):
                self.window.setProperty("paletteColors", quick_colors(mode))
                self.bridge.setParameter("savedOnly", False)
                QTest.qWait(20)
                for state, label in (("new", "Unprocessed"), ("processed", "Processed"), ("unknown", "History unknown")):
                    badge = self.item("source_catalog_file_status_" + state + ".csv")
                    self.assertEqual(badge.property("status"), state)
                    self.assertEqual(badge.property("text"), label)
                self.bridge.setParameter("savedOnly", True)
                QTest.qWait(20)
                badge = self.item("source_catalog_file_status_saved.csv")
                self.assertEqual(badge.property("status"), "saved")
                self.assertEqual(badge.property("text"), "Saved result")

        def test_cached_gate_summaries_render_equal_and_different_measured_traces(self):
            gates = tuple(-1. + index / 270. for index in range(541))
            labels = ("Vbg_set", "Vbg_meas", "Vtg_set", "Vtg_meas")
            direction = "Vbg_set constant, Vbg_meas constant, Vtg_set increasing, Vtg_meas increasing"
            equal = replace(self.source("equal-set-measured.csv", "Equal set and measured gates", False),
                frame_count=541, gate_labels=labels, gate_direction=direction,
                gate_grid=tuple((15., 15., value, value) for value in gates),
                gate_ranges=((15., 15.), (15., 15.), (-1., 1.), (-1., 1.)))
            different = replace(equal, source="different-measured.csv", filename="different-measured.csv",
                group_key="Measured gate differs by 0.01 V", modified_time=.5,
                gate_grid=tuple((15., 15., value, value + .01) for value in gates),
                gate_ranges=((15., 15.), (15., 15.), (-1., 1.), (-.99, 1.01)))
            catalog = _drr_catalog_rows(self.base, (equal, different))
            self.publish(catalog)
            self.assertEqual(len(catalog["groups"]), 2)
            expected = {
                equal.group_key: "Vbg: 15; Vtg: -1–1 ↑ · 541 frames",
                different.group_key: "Vbg: 15; Vtg set: -1–1 ↑; Vtg measured: -0.99–1.01 ↑ · 541 frames",
            }
            for row in catalog["groups"]:
                card = self.item("source_catalog_group_" + row["id"])
                detail = entry(card)["detail"]
                self.assertEqual(detail, expected[row["name"]])
                self.assertIn(detail, card.property("text"))
                rendered = [child for child in visual_items(card)
                    if child.metaObject().indexOfProperty("text") >= 0 and child.property("text") == detail]
                self.assertTrue(rendered, "The actual group card must render its compact gate detail")
                self.assertTrue(any(child.property("visible") and child.height() > 0 for child in rendered))
                self.assertNotIn("(541, 541)", detail)
            for row in catalog["files"]:
                file_row = self.item("source_catalog_file_" + row["name"])
                detail = entry(file_row)["detail"]
                self.assertIn("541 frames", detail)
                if row["name"] == equal.filename:
                    self.assertIn("Vbg: 15; Vtg: -1–1 ↑", detail)
                    self.assertNotIn("Vtg set", detail)
                    self.assertNotIn("measured:", detail)
                else:
                    self.assertIn("Vtg set: -1–1 ↑; Vtg measured: -0.99–1.01 ↑", detail)
                self.bridge.selectFile(row["id"], True)
                self.wait(lambda: self.bridge.snapshot["detail"] == detail)
                self.wait(lambda: any(child.property("visible")
                    and child.metaObject().indexOfProperty("text") >= 0
                    and child.property("text") == detail
                    for child in visual_items(self.window.contentItem())))

            self.window.setProperty("paletteColors", quick_colors("light"))
            QTest.qWait(60)
            image = self.window.grabWindow()
            self.assertFalse(image.isNull())
            capture = os.environ.get("DPTK_SOURCE_CATALOG_STATUS_CAPTURE")
            if capture:
                destination = Path(capture) / "source-catalog-gate-summary-light.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                self.assertTrue(image.save(str(destination)))

        def scroll_fixture(self, separate_groups):
            sources = tuple(self.source(f"measurement-{index:03}.csv",
                f"Group {index:03}" if separate_groups else "One acquisition group", False) for index in range(80))
            self.publish(_drr_catalog_rows(self.base, sources))
            return sources

        def position(self, view, index):
            self.assertTrue(QMetaObject.invokeMethod(view, "positionViewAtIndex", Qt.DirectConnection,
                Q_ARG(int, index), Q_ARG(int, 0)))
            QTest.qWait(40)
            view.setProperty("contentY", view.property("contentY") + 13)
            QTest.qWait(30)

        def anchor(self, view, prefix):
            visible = []
            for item in visual_items(view):
                if item.objectName().startswith(prefix) and item.metaObject().indexOfProperty("entry") >= 0:
                    self.wrappers.append(item)
                    y = item.mapToItem(view, QPointF(0, 0)).y()
                    if y + item.height() > 0 and y < view.height():
                        visible.append((y, entry(item)["id"]))
            self.assertTrue(visible, prefix)
            y, identity = min(visible)
            return identity, y

        def assert_same_anchor(self, view, prefix, expected):
            QTest.qWait(90)
            identity, y = self.anchor(view, prefix)
            self.assertEqual(identity, expected[0])
            self.assertAlmostEqual(y, expected[1], delta=1.5)

        def test_group_scroll_current_group_and_selection_survive_inventory_resets(self):
            sources = self.scroll_fixture(True)
            view = self.item("source_catalog_groups")
            selected_group = self.bridge.groupsModel.rows[40]["id"]
            self.bridge.setParameter("selectedGroup", selected_group)
            selected = str(self.base / sources[40].source)
            self.bridge.selectFile(selected, True)
            self.position(view, 30)
            expected = self.anchor(view, "source_catalog_group_")
            added = self.source("new-first.csv", "AAA new first group", False)
            self.publish(_drr_catalog_rows(self.base, (added, *sources)))
            self.assert_same_anchor(view, "source_catalog_group_", expected)
            self.publish(_drr_catalog_rows(self.base, (added, *sources[5:])))
            self.assert_same_anchor(view, "source_catalog_group_", expected)
            self.assertEqual(self.bridge.snapshot["selectedGroup"], selected_group)
            self.assertEqual(self.bridge.snapshot["selected"], [selected])

        def test_file_scroll_selection_and_focus_survive_inventory_resets(self):
            sources = self.scroll_fixture(False)
            view = self.item("source_catalog_files")
            selected = str(self.base / sources[45].source)
            self.bridge.selectFile(selected, True)
            self.position(view, 30)
            expected = self.anchor(view, "source_catalog_file_")
            added = self.source("000-new-first.csv", "One acquisition group", False)
            self.publish(_drr_catalog_rows(self.base, (added, *sources)))
            self.assert_same_anchor(view, "source_catalog_file_", expected)
            self.publish(_drr_catalog_rows(self.base, (added, *sources[5:])))
            self.assert_same_anchor(view, "source_catalog_file_", expected)
            # Cached preview and final validation can publish in one event turn.
            self.bridge._catalog = _drr_catalog_rows(self.base, (added, *sources[3:]))
            self.bridge._refresh_models()
            self.bridge._catalog = _drr_catalog_rows(self.base, (added, *sources[7:]))
            self.bridge._refresh_models()
            self.assert_same_anchor(view, "source_catalog_file_", expected)
            self.assertEqual(self.bridge.snapshot["selected"], [selected])
            self.assertEqual(self.bridge.snapshot["focused"], selected)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"CATALOG_STATUS_TESTS={result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
