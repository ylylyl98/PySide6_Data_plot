"""Exercise the native Quick shell without the real app or persistent settings.

The plot type and QObject state are deliberately small fakes: these tests prove
the shell's input, layout and loading boundary without constructing MainWindow,
starting acquisition, discovering local data or touching production QSettings.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from time import monotonic
import unittest

from PySide6.QtCore import QObject, QPoint, Property, QUrl, Qt, Signal, Slot
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent, qmlRegisterType
from PySide6.QtQuick import QQuickItem, QQuickWindow
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ui_qt.quick.drr_display_host import display_colors


class _PlotSurface(QQuickItem):
    backendChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._backend = None

    @Property(QObject, notify=backendChanged)
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value):
        self._backend = value
        self.backendChanged.emit()


class _Bridge(QObject):
    snapshotChanged = Signal()
    colorsChanged = Signal()
    parameterUrlChanged = Signal()
    workspaceUrlChanged = Signal()
    workflowBridgeChanged = Signal()

    def __init__(self):
        super().__init__()
        self.calls = []
        self._parameter_url = QUrl()
        self._workspace_url = QUrl()
        self._snapshot = {
            "title": "Native shell test",
            "workflow": "pl",
            "workspaceTitle": "Photoluminescence",
            "sourcePath": "",
            "status": "Ready",
            "busy": False,
            "progress": 0,
            "controlsVisible": True,
            "railVisible": True,
            "resultsVisible": False,
            "logVisible": False,
            "sideBySide": False,
            "resultText": "Fit result: 1.23",
            "logText": "Source loaded",
            "workspaceMessage": "",
            "inspectionVisible": True,
            "spectrumVisible": True,
            "dividerY": .7,
            "heatmapFraction": .7,
            "parameterGroups": [{
                "title": "Inspection",
                "description": "Native core controls",
                "fields": [
                    {"key": "gate", "label": "Gate voltage", "type": "number",
                     "value": 0., "min": -5., "max": 5., "step": .25, "decimals": 2},
                    {"key": "normalize", "label": "Normalize", "type": "flag", "value": False},
                    {"key": "channel", "label": "Channel", "type": "choice", "value": "a",
                     "options": [{"label": "Channel A", "value": "a"}, {"label": "Channel B", "value": "b"}]},
                    {"key": "inspect", "label": "Inspect data", "type": "action"},
                ],
            }],
        }

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        return self._snapshot

    @Property("QVariantMap", notify=colorsChanged)
    def colors(self):
        return display_colors()

    @Property(QObject, constant=True)
    def plotBackend(self):
        return self

    @Property(QObject, notify=workflowBridgeChanged)
    def workflowBridge(self):
        return self

    @Property(QUrl, notify=parameterUrlChanged)
    def parameterUrl(self):
        return self._parameter_url

    @Property(QUrl, notify=workspaceUrlChanged)
    def workspaceUrl(self):
        return self._workspace_url

    def update(self, **values):
        self._snapshot = {**self._snapshot, **values}
        self.snapshotChanged.emit()

    @Slot(str)
    def selectWorkflow(self, key):
        self.calls.append(("workflow", key))
        self.update(workflow=key)

    @Slot(str)
    def invoke(self, command):
        self.calls.append(("invoke", command))
        if command == "toggle-results":
            self.update(resultsVisible=not self._snapshot["resultsVisible"])
        elif command == "toggle-log":
            self.update(logVisible=not self._snapshot["logVisible"])
        elif command == "toggle-spectrum":
            self.update(spectrumVisible=not self._snapshot["spectrumVisible"])

    @Slot(str)
    def setSourcePath(self, value):
        self.calls.append(("source", value))
        self.update(sourcePath=value)

    @Slot(QUrl)
    def openFile(self, value):
        self.calls.append(("file", value.toString()))

    @Slot(bool)
    def setControlsVisible(self, value):
        self.calls.append(("controls", value))
        self.update(controlsVisible=value)

    @Slot(bool)
    def setRailVisible(self, value):
        self.calls.append(("rail", value))
        self.update(railVisible=value)

    @Slot(str, "QVariant")
    def setParameter(self, key, value):
        self.calls.append(("parameter", key, value))
        if key in {"dividerY", "heatmapFraction"}:
            self.update(**{key: value})
        else:
            groups = deepcopy(self._snapshot["parameterGroups"])
            for group in groups:
                for field in group["fields"]:
                    if field["key"] == key:
                        field["value"] = value
            self.update(parameterGroups=groups)


class QuickApplicationShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # The Windows offscreen plugin has no system-font enumeration. Supply
        # one known font for useful screenshots without changing installed fonts.
        if os.name == "nt" and not QFontDatabase.families():
            font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
            if font_path.is_file():
                QFontDatabase.addApplicationFont(str(font_path))
                cls.app.setFont(QFont("Segoe UI", 9))
        qmlRegisterType(_PlotSurface, "Dptk", 1, 0, "PlotSurface")

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bridge = _Bridge()
        self.engine = QQmlApplicationEngine()
        self.warnings = []
        self.engine.warnings.connect(lambda messages: self.warnings.extend(item.toString() for item in messages))
        self.engine.setInitialProperties({"appBridge": self.bridge})
        path = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml" / "Application.qml"
        self.engine.load(QUrl.fromLocalFile(str(path)))
        self.assertTrue(self.engine.rootObjects(), "\n".join(self.warnings))
        self.window = self.engine.rootObjects()[0]
        QTest.qWait(30)
        self.addCleanup(self.dispose)

    def dispose(self):
        self.window.close()
        self.engine.deleteLater()
        self.app.processEvents()

    def item(self, name):
        result = self.window.findChild(QQuickItem, name)
        if result is None:
            # Repeater delegate ownership is separate from its visual parent.
            def visit(parent):
                for child in parent.childItems():
                    if child.objectName() == name:
                        return child
                    found = visit(child)
                    if found is not None:
                        return found
                return None
            result = visit(self.window.contentItem())
        self.assertIsNotNone(result, name)
        return result

    def click(self, name):
        item = self.item(name)
        point = item.mapToScene(item.boundingRect().center()).toPoint()
        QTest.mouseClick(self.window, Qt.LeftButton, Qt.NoModifier, point)
        QTest.qWait(20)

    def type_text(self, text):
        for character in text:
            modifier = Qt.ShiftModifier if character.isupper() else Qt.NoModifier
            QTest.keyClick(self.window, ord(character), modifier)

    def range_groups(self):
        return [
            {"key": "data", "title": "Data", "fields": [
                {"key": "selected", "type": "summary", "value": "Two selected measurement files"},
                {"key": "select", "type": "action", "label": "Select…"},
                {"key": "clear", "type": "action", "label": "Clear"},
                {"key": "saved", "type": "action", "label": "Saved results"},
            ]},
            {"key": "axes", "title": "Axes", "fields": [
                {"key": "xmin", "label": "Photon energy minimum", "type": "number", "pair": "x",
                 "pairLabel": "Photon energy (eV)", "value": 1.23456789012345, "decimals": 4, "step": .01},
                {"key": "xmax", "label": "Photon energy maximum", "type": "number", "pair": "x",
                 "pairLabel": "Photon energy (eV)", "value": 2.7, "decimals": 4, "step": .01},
                {"key": "sample", "label": "Sample label", "type": "text", "value": "Sample A"},
            ]},
            {"key": "advanced", "title": "Advanced settings", "advanced": True, "fields": [
                {"key": "correction", "label": "Apply correction", "type": "flag", "value": False},
            ]},
        ]

    def make_parameter_pane(self, groups, *, width=320, height=650, point_size=9):
        self.bridge.update(parameterGroups=groups)
        path = Path(__file__).resolve().parents[1] / "ui_qt/quick/qml/AppParameterPane.qml"
        component = QQmlComponent(self.engine, QUrl.fromLocalFile(str(path)))
        self._parameter_component = component
        pane = component.createWithInitialProperties({"bridge": self.bridge, "colors": self.bridge.colors, "groups": groups})
        self.assertIsNotNone(pane, "\n".join(error.toString() for error in component.errors()))
        window = QQuickWindow()
        window.resize(width, height)
        pane.setParentItem(window.contentItem())
        pane.setParent(window.contentItem())
        pane.setWidth(width)
        pane.setHeight(height)
        pane.setProperty("font", QFont("Segoe UI", point_size))
        def refresh():
            pane.setProperty("groups", self.bridge.snapshot["parameterGroups"])
        self.bridge.snapshotChanged.connect(refresh)
        def dispose_pane():
            self.bridge.snapshotChanged.disconnect(refresh)
            window.close()
            pane.setParentItem(None)
            pane.deleteLater()
            window.deleteLater()
            self.app.processEvents()
        self.addCleanup(dispose_pane)
        window.show()
        window.requestActivate()
        QTest.qWait(30)
        return window, pane

    def parameter_item(self, root, name):
        def visit(parent):
            if parent.objectName() == name:
                return parent
            for child in parent.childItems():
                found = visit(child)
                if found is not None:
                    return found
            return None
        result = visit(root)
        self.assertIsNotNone(result, name)
        return result

    def type_in_window(self, window, text):
        for character in text:
            QTest.keyClick(window, ord(character), Qt.NoModifier)

    def test_parameter_ranges_fit_320_and_240_then_stack_for_large_text(self):
        window, pane = self.make_parameter_pane(self.range_groups())
        minimum = self.parameter_item(pane, "app_parameter_xmin")
        maximum = self.parameter_item(pane, "app_parameter_xmax")
        pair = self.parameter_item(pane, "app_parameter_range_x")
        for width in (320, 240):
            with self.subTest(width=width):
                window.resize(width, 650)
                pane.setWidth(width)
                QTest.qWait(20)
                self.assertFalse(pair.property("stacked"))
                left = minimum.mapToScene(QPoint(0, 0))
                right = maximum.mapToScene(QPoint(0, 0))
                self.assertAlmostEqual(left.y(), right.y(), places=1)
                self.assertLess(left.x(), right.x())
                self.assertGreaterEqual(minimum.width(), 60)
                self.assertLessEqual(right.x() + maximum.width(), width)
        pane.setProperty("font", QFont("Segoe UI", 18))
        deadline = monotonic() + 2
        while maximum.mapToScene(QPoint(0, 0)).y() <= minimum.mapToScene(QPoint(0, 0)).y() and monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(pair.property("stacked"))
        self.assertGreater(maximum.mapToScene(QPoint(0, 0)).y(), minimum.mapToScene(QPoint(0, 0)).y())
        self.assertIs(self.parameter_item(pane, "app_parameter_xmin"), minimum)
        self.assertEqual(self.warnings, [])

    def test_parameter_draft_focus_and_full_precision_survive_snapshots(self):
        window, pane = self.make_parameter_pane(self.range_groups())
        minimum = self.parameter_item(pane, "app_parameter_xmin")
        maximum = self.parameter_item(pane, "app_parameter_xmax")
        self.bridge.calls.clear()
        minimum.forceActiveFocus()
        self.assertEqual(minimum.property("text"), "1.2346")
        self.assertEqual(self.bridge.snapshot["parameterGroups"][1]["fields"][0]["value"], 1.23456789012345)
        QTest.keyClick(window, Qt.Key_Tab)
        self.assertTrue(maximum.hasActiveFocus())
        QTest.qWait(10)
        self.assertFalse(any(call[0] == "parameter" for call in self.bridge.calls))
        minimum.forceActiveFocus()
        QTest.keyClick(window, Qt.Key_A, Qt.ControlModifier)
        draft = "1.23456789012346"
        self.type_in_window(window, draft)
        groups = deepcopy(self.bridge.snapshot["parameterGroups"])
        groups[1]["fields"][0]["value"] = 1.4
        self.bridge.update(parameterGroups=groups, status="Background refresh finished")
        QTest.qWait(20)
        self.assertIs(self.parameter_item(pane, "app_parameter_xmin"), minimum)
        self.assertTrue(minimum.hasActiveFocus())
        self.assertEqual(minimum.property("text"), draft)
        QTest.keyClick(window, Qt.Key_Return)
        QTest.qWait(10)
        self.assertIn(("parameter", "xmin", float(draft)), self.bridge.calls)
        maximum.forceActiveFocus()
        QTest.qWait(10)
        self.assertEqual(sum(call[:2] == ("parameter", "xmin") for call in self.bridge.calls), 1)
        self.assertEqual(self.bridge.snapshot["parameterGroups"][1]["fields"][0]["value"], float(draft))
        self.bridge.calls.clear()
        minimum.forceActiveFocus()
        QTest.keyClick(window, Qt.Key_A, Qt.ControlModifier)
        self.type_in_window(window, "nan")
        QTest.keyClick(window, Qt.Key_Return)
        self.assertFalse(any(call[0] == "parameter" for call in self.bridge.calls))
        self.assertTrue(minimum.parentItem().property("invalid"))
        QTest.keyClick(window, Qt.Key_Escape)
        self.assertEqual(minimum.property("text"), "1.2346")
        self.assertEqual(self.warnings, [])

    def test_advanced_expansion_and_text_drafts_survive_descriptor_updates(self):
        window, pane = self.make_parameter_pane(self.range_groups())
        heading = self.parameter_item(pane, "app_parameter_group_advanced")
        body = self.parameter_item(pane, "app_parameter_group_body_advanced")
        self.assertFalse(body.isVisible())
        heading.forceActiveFocus()
        QTest.keyClick(window, Qt.Key_Space)
        QTest.qWait(20)
        self.assertTrue(body.isVisible())
        label = self.parameter_item(pane, "app_parameter_sample")
        label.forceActiveFocus()
        QTest.keyClick(window, Qt.Key_A, Qt.ControlModifier)
        self.type_in_window(window, "New label")
        groups = deepcopy(self.bridge.snapshot["parameterGroups"])
        groups[2]["title"] = "Advanced settings (ready)"
        groups[1]["fields"][2]["value"] = "Updated elsewhere"
        self.bridge.update(parameterGroups=groups, status="Ready")
        QTest.qWait(20)
        self.assertIs(self.parameter_item(pane, "app_parameter_group_advanced"), heading)
        self.assertTrue(body.isVisible())
        self.assertIs(self.parameter_item(pane, "app_parameter_sample"), label)
        self.assertTrue(label.hasActiveFocus())
        self.assertEqual(label.property("text"), "New label")
        QTest.keyClick(window, Qt.Key_Return)
        self.assertIn(("parameter", "sample", "New label"), self.bridge.calls)
        self.assertEqual(self.warnings, [])

    def test_data_selection_actions_stay_visible_while_parameters_scroll(self):
        groups = self.range_groups()
        groups[1]["fields"].extend({"key": "extra" + str(index), "label": "Additional parameter " + str(index),
                                    "type": "number", "value": index} for index in range(12))
        window, pane = self.make_parameter_pane(groups, width=240, height=420)
        select = self.parameter_item(pane, "app_parameter_select")
        saved = self.parameter_item(pane, "app_parameter_saved")
        start = select.mapToScene(QPoint(0, 0))
        scroll = self.parameter_item(pane, "app_parameter_scroll")
        flickable = scroll.property("contentItem")
        self.assertGreater(flickable.property("contentHeight"), flickable.property("height"))
        flickable.setProperty("contentY", flickable.property("contentHeight") - flickable.property("height"))
        QTest.qWait(20)
        self.assertEqual(select.mapToScene(QPoint(0, 0)), start)
        self.assertTrue(select.isVisible())
        self.assertTrue(saved.isVisible())
        self.assertLess(saved.mapToScene(QPoint(0, 0)).y() + saved.height(), pane.height())
        QTest.mouseClick(window, Qt.LeftButton, Qt.NoModifier, select.mapToScene(select.boundingRect().center()).toPoint())
        self.assertIn(("invoke", "select"), self.bridge.calls)
        self.assertEqual(self.warnings, [])

    def test_shell_loads_native_workflows_and_routes_real_clicks(self):
        self.assertEqual(self.window.title(), "Native shell test")
        for key in ("pl", "drr", "cmp", "power", "mcd", "mcd_peak_shift", "shg", "slides", "tools"):
            self.assertIsNotNone(self.item("app_workflow_" + key))
        self.click("app_workflow_drr")
        self.assertIn(("workflow", "drr"), self.bridge.calls)
        self.assertTrue(self.item("app_workflow_drr").property("checked"))
        self.click("app_export")
        self.assertIn(("invoke", "export"), self.bridge.calls)
        self.click("app_plot_home")
        self.assertIn(("invoke", "home"), self.bridge.calls)
        self.assertEqual(self.warnings, [])

    def test_keyboard_source_and_descriptor_edits_reach_qobject(self):
        source = self.item("app_source_path")
        source.forceActiveFocus()
        self.type_text("D:/measurement")
        QTest.keyClick(self.window, Qt.Key_Return)
        self.assertIn(("source", "D:/measurement"), self.bridge.calls)
        number = self.item("app_parameter_gate")
        number.forceActiveFocus()
        QTest.keyClick(self.window, Qt.Key_A, Qt.ControlModifier)
        self.type_text("1.25")
        QTest.keyClick(self.window, Qt.Key_Return)
        self.assertIn(("parameter", "gate", 1.25), self.bridge.calls)
        self.assertIs(self.item("app_parameter_gate"), number)
        self.assertTrue(number.hasActiveFocus())
        QTest.keyClick(self.window, Qt.Key_Up)
        self.assertIn(("parameter", "gate", 1.5), self.bridge.calls)
        self.assertIs(self.item("app_parameter_gate"), number)
        self.click("app_parameter_normalize")
        self.assertIn(("parameter", "normalize", True), self.bridge.calls)
        self.click("app_parameter_inspect")
        self.assertIn(("invoke", "inspect"), self.bridge.calls)
        QTest.keyClick(self.window, Qt.Key_B, Qt.ControlModifier)
        self.assertFalse(self.bridge.snapshot["controlsVisible"])
        QTest.keyClick(self.window, Qt.Key_B, Qt.ControlModifier)
        self.assertTrue(self.bridge.snapshot["controlsVisible"])

    def test_side_by_side_renders_bridge_rail_policy_and_keeps_manual_restore(self):
        # The bridge owns automatic collapse and its restore state. QML must
        # not turn an automatic transition into a manual visibility override.
        self.bridge.update(sideBySide=True, railVisible=False)
        QTest.qWait(20)
        self.assertFalse(self.item("app_workflow_rail").isVisible())
        self.assertTrue(self.item("app_parameter_pane").isVisible())
        self.assertTrue(self.item("app_workflow_selector").isVisible())
        self.click("app_toggle_rail")
        self.assertTrue(self.item("app_workflow_rail").isVisible())
        self.bridge.update(status="Refreshed")
        self.app.processEvents()
        self.assertTrue(self.item("app_workflow_rail").isVisible())
        self.bridge.update(sideBySide=False)
        self.app.processEvents()
        self.assertTrue(self.item("app_workflow_rail").isVisible())
        self.assertTrue(self.item("app_parameter_pane").isVisible())

    def test_narrow_layout_preserves_controls_and_positive_plot_area(self):
        self.window.resize(720, 600)
        QTest.qWait(20)
        self.assertTrue(self.item("app_parameter_pane").isVisible())
        self.assertGreaterEqual(self.item("app_parameter_pane").width(), 220)
        self.assertGreater(self.item("app_plot_surface").width(), 150)
        self.assertGreater(self.item("app_plot_surface").height(), 200)
        for name in ("app_toggle_controls", "app_toggle_rail", "app_browse_folder", "app_open_file", "app_export"):
            item = self.item(name)
            self.assertTrue(item.isVisible())
            position = item.mapToScene(QPoint(0, 0))
            self.assertGreaterEqual(position.x(), 0, name)
            self.assertLessEqual(position.x() + item.width(), self.window.width() + 1, name)
        self.click("app_toggle_controls")
        self.assertFalse(self.item("app_parameter_pane").isVisible())
        # SplitView relayout is a deferred scene-graph polish. Under the full
        # suite a 20 ms input wait can precede that frame even after visible
        # has changed; await the actual geometry rather than a fixed delay.
        deadline = monotonic() + 2
        while self.item("app_plot_surface").width() <= 550 and monotonic() < deadline:
            QTest.qWait(10)
        self.assertGreater(self.item("app_plot_surface").width(), 550)

    def test_results_log_and_unavailable_workspace_are_explicit(self):
        self.click("app_toggle_results")
        self.assertTrue(self.item("app_bottom_panel").isVisible())
        self.assertEqual(self.item("app_panel_text").property("text"), "Fit result: 1.23")
        self.click("app_toggle_log")
        self.assertEqual(self.item("app_panel_text").property("text"), "Source loaded")
        self.click("app_close_panel")
        self.assertFalse(self.bridge.snapshot["logVisible"])
        self.assertTrue(self.item("app_bottom_panel").isVisible())
        self.bridge.update(workspaceMessage="SHG native processing is still being migrated.")
        self.app.processEvents()
        self.assertFalse(self.item("app_plot_surface").isVisible())
        self.assertTrue(self.item("app_workspace_message").isVisible())

    def test_native_inspection_divider_keyboard_drag_and_spectrum_toggle(self):
        handle = self.item("app_inspection_divider")
        self.assertAlmostEqual(handle.y() + handle.height() / 2,
            self.item("app_plot_surface").height() * self.bridge.snapshot["dividerY"])
        handle.forceActiveFocus()
        QTest.keyClick(self.window, Qt.Key_Up)
        self.assertIn(("parameter", "heatmapFraction", .72), self.bridge.calls)
        QTest.keyClick(self.window, Qt.Key_Home)
        self.assertIn(("parameter", "heatmapFraction", .7), self.bridge.calls)
        start = handle.mapToScene(handle.boundingRect().center()).toPoint()
        QTest.mousePress(self.window, Qt.LeftButton, Qt.NoModifier, start)
        QTest.mouseMove(self.window, start + QPoint(0, -35), 10)
        QTest.mouseRelease(self.window, Qt.LeftButton, Qt.NoModifier, start + QPoint(0, -35))
        self.assertTrue(any(call[:2] == ("parameter", "dividerY") for call in self.bridge.calls))
        self.click("app_toggle_spectrum")
        self.assertFalse(self.bridge.snapshot["spectrumVisible"])
        self.assertFalse(handle.isVisible())
        self.assertTrue(self.item("app_toggle_spectrum").isVisible())
        self.click("app_toggle_spectrum")
        self.assertTrue(self.bridge.snapshot["spectrumVisible"])
        self.assertTrue(handle.isVisible())

    def test_spectrum_restore_stays_in_the_same_corner_in_all_themes_and_sizes(self):
        from unittest.mock import patch
        from ui_qt.quick.appearance import quick_colors

        for theme in ("light", "dark", "high-contrast"):
            with patch(__name__ + ".display_colors", return_value=quick_colors(theme)):
                self.bridge.colorsChanged.emit()
                for width in (1320, 720):
                    with self.subTest(theme=theme, width=width):
                        self.window.resize(width, 600)
                        QTest.qWait(20)
                        toggle = self.item("app_toggle_spectrum")
                        parent = toggle.parentItem()
                        corner = (toggle.x() + toggle.width(), toggle.y() + toggle.height())
                        self.assertAlmostEqual(corner[0], parent.width() - 8)
                        self.assertAlmostEqual(corner[1], parent.height() - 8)
                        self.assertGreaterEqual(toggle.x(), 0)
                        self.assertGreaterEqual(toggle.y(), 0)
                        surface = self.item("app_plot_surface")
                        self.assertGreaterEqual(toggle.mapToScene(toggle.boundingRect().topLeft()).y(),
                            surface.mapToScene(surface.boundingRect().bottomRight()).y())
                        self.assertEqual(toggle.property("text"), "Hide spectrum")
                        self.assertEqual(toggle.property("background").property("color").alpha(), 255)
                        self.click("app_toggle_spectrum")
                        self.assertFalse(self.bridge.snapshot["spectrumVisible"])
                        self.assertEqual(toggle.property("text"), "Show spectrum")
                        self.assertTrue(toggle.property("primary"))
                        self.assertEqual((toggle.x() + toggle.width(), toggle.y() + toggle.height()), corner)
                        self.assertEqual(toggle.property("background").property("color").alpha(), 255)
                        self.click("app_toggle_spectrum")
                        self.assertTrue(self.bridge.snapshot["spectrumVisible"])
                        self.assertFalse(toggle.property("primary"))
                        self.assertEqual((toggle.x() + toggle.width(), toggle.y() + toggle.height()), corner)

        self.click("app_toggle_spectrum")
        self.item("app_toggle_spectrum").forceActiveFocus(Qt.TabFocusReason)
        QTest.keyClick(self.window, Qt.Key_Space)
        self.assertTrue(self.bridge.snapshot["spectrumVisible"])
        self.assertEqual(self.warnings, [])

    def test_typed_qml_parameter_loader_receives_workflow_bridge(self):
        page = Path(self.temporary.name) / "TypedParameters.qml"
        page.write_text('''import QtQuick
Rectangle {
    required property var bridge
    required property var colors
    objectName: "typed_parameter_page"
    color: colors.surface
    Component.onCompleted: bridge.invoke("typed-page-loaded")
}
''', encoding="utf-8")
        self.bridge._parameter_url = QUrl.fromLocalFile(str(page))
        self.bridge.parameterUrlChanged.emit()
        self.app.processEvents()
        self.assertIsNotNone(self.item("typed_parameter_page"))
        self.assertIn(("invoke", "typed-page-loaded"), self.bridge.calls)
        self.assertFalse(self.item("app_default_parameter_pane").isVisible())
        self.assertEqual(self.warnings, [])

    def test_workflow_keyboard_navigation_and_native_menu_shortcut(self):
        self.item("app_workflow_pl").forceActiveFocus()
        QTest.keyClick(self.window, Qt.Key_Down)
        self.assertTrue(self.item("app_workflow_drr").hasActiveFocus())
        QTest.keyClick(self.window, Qt.Key_Return)
        self.assertEqual(self.bridge.snapshot["workflow"], "drr")
        self.window.resize(720, 520)
        QTest.qWait(20)
        QTest.keyClick(self.window, Qt.Key_End)
        self.assertTrue(self.item("app_workflow_tools").hasActiveFocus())
        position = self.item("app_workflow_tools").mapToScene(QPoint(0, 0))
        self.assertLess(position.y() + self.item("app_workflow_tools").height(), self.window.height())
        QTest.keyClick(self.window, Qt.Key_F, Qt.AltModifier)
        QTest.qWait(20)
        menu = self.window.findChild(QObject, "app_file_menu")
        self.assertIsNotNone(menu)
        self.assertTrue(menu.property("visible"))
        QTest.keyClick(self.window, Qt.Key_Escape)
        self.assertFalse(menu.property("visible"))
        self.assertEqual(self.warnings, [])


if __name__ == "__main__":
    unittest.main()
