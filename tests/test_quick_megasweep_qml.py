"""Exercise Megasweep's Quick controls with a realistic observable bridge."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MegasweepQmlTest(unittest.TestCase):
    def test_workspace_controls_and_movie(self):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", PYTHONIOENCODING="utf-8")
        run = subprocess.run([sys.executable, __file__, "--qml-child"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=45)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


def child():
    sys.path.insert(0, str(ROOT))
    from PySide6.QtCore import QObject, QPointF, Property, Signal, Slot, QUrl, QSettings, Qt, qInstallMessageHandler
    from PySide6.QtGui import QFontMetricsF, QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtTest import QTest
    from ui_qt.quick.background_source_catalog import BackgroundSourceCatalogBridge
    from ui_qt.quick.plot_surface import register_plot_surface

    app = QGuiApplication([])
    register_plot_surface()
    errors = []
    def log(kind, context, message):
        if any(part in message for part in ("ReferenceError", "TypeError", "Binding loop", "Cannot assign", "Unable to assign")):
            errors.append(message)
    qInstallMessageHandler(log)
    class Bridge(QObject):
        changed = Signal()
        plotLimitsChanged = Signal()
        def __init__(self):
            super().__init__()
            self.bg = BackgroundSourceCatalogBridge(settings=QSettings("DPTK tests", "Megasweep QML"), parent=self)
            self.data = dict(mode="PL", values=dict(view="slice"), parameterGroups=[], busy=False,
                sourceFile="example.csv", sourcePath="", inspectionVisible=True, canExport=True, summary="Loaded", error="", status="Ready",
                movieOptions=dict(fps=30), moviePreviewFrameCount=4, moviePreviewFrameIndex=0, movieDuration=4/30,
                backgroundRole="custom", backgroundGroups=dict(custom=["c.csv"], p=["p1.csv", "p2.csv"], n=[], bn=["bn.csv"]),
                referenceParameterGroups=[dict(key="normalization", title="Reference scaling", fields=[
                    dict(key="auto_background_scale", label="Auto background scale", type="flag", value=True)])],
                backgroundRoleOptions=[dict(value="custom",label="Custom"),dict(value="p",label="P"),dict(value="n",label="N"),dict(value="bn",label="BN")])
            self.actions = []
            self.cmap = "RdBu_r"
            self.limit_rows = {}
            self.color_auto = True
            self.color_bounds = (-1.237e-125, 9.876e+113)
            self.data['referencePoint'] = dict(xName='Vbg', yName='Vtg', x=0., y=0.)
        @Property("QVariantMap", notify=changed)
        def snapshot(self): return self.data
        @Property(QObject, constant=True)
        def backgroundSelectionBridge(self): return self.bg
        @Property(QObject, constant=True)
        def plotBackend(self): return None
        @Slot("QVariant")
        def setPalette(self, colors): pass
        @Slot(str, "QVariant")
        def setParameter(self, key, value):
            if key == "mode": self.data["mode"] = value
            elif key == "cmap":
                self.cmap = value
                self.plotLimitsChanged.emit()
            elif key == "color_auto":
                self.color_auto = value
                self.plotLimitsChanged.emit()
            elif key == "background_role": self.data["backgroundRole"] = value
            elif key == "movie.previewFrame": self.data["moviePreviewFrameIndex"] = value
            elif key.startswith("movie."): self.data["movieOptions"][key[6:]] = value
            self.actions.append((key, value)); self.changed.emit()
        @Slot(result="QVariantList")
        def mapLimits(self):
            labels = {"X": "Doping_axis_b_set", "Y": "Vbias_axis_a_set", "Color": "RC peak-to-peak"}
            rows = [dict(key=axis.lower()+"_auto", axis=axis, label=labels[axis], automatic=self.color_auto if axis == "Color" else True,
                         reversed=False, low=self.color_bounds[0] if axis == "Color" else 0.,
                         high=self.color_bounds[1] if axis == "Color" else 2.) for axis in ("X", "Y", "Color")]
            for row in rows:
                row.update(self.limit_rows.get(row["key"], {}))
            rows[-1].update(cmap=self.cmap, colormaps=["RdBu_r", "RdBu"])
            return rows
        @Slot(str, float)
        def setReferenceCoordinate(self, axis, value):
            self.actions.append((axis, value))
            snapped = round(value)
            self.data['referencePoint'].update(x=snapped, y=snapped)
            self.changed.emit()
        @Slot("QVariantList", result=str)
        def applyMapLimits(self, rows):
            row, = rows
            if not row["automatic"] and float(row["low"]) >= float(row["high"]):
                return "Min must be below Max"
            self.limit_rows[row["key"]] = dict(row, low=float(row["low"]), high=float(row["high"]))
            self.actions.append((row["key"], row["automatic"]))
            self.plotLimitsChanged.emit()
            return ""
        @Slot(str)
        def invoke(self, name):
            self.actions.append((name, None))
            if name == "reflection-preview": self.data["reflectionPreviewVisible"] = True; self.changed.emit()
            elif name in ("return-to-map", "show-spectral-cut"):
                self.data["reflectionPreviewVisible"] = False
                self.data["values"]["view"] = "slice" if name == "show-spectral-cut" else "rc_p2p"
                self.changed.emit()
    bridge = Bridge()
    colors = dict(surface="#fff", control="#fff", text="#111", muted="#555", border="#ccc", accent="#067", hover="#eee", pressed="#ddd", disabled="#999", selectedText="#fff", inset="#eee", divider="#ddd", focus="#067", accentText="#067", selected="#def", selectedBorder="#bcd", error="#a00", controlDisabled="#eee", borderDisabled="#ddd")
    engine = QQmlApplicationEngine()
    engine.setInitialProperties(dict(bridge=bridge, colors=colors))
    qml = ROOT / "ui_qt" / "quick" / "qml"
    engine.loadData(b'''import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import "."
Window { id: win; required property var bridge; required property var colors; width: 1100; height: 760; visible: true
RowLayout { anchors.fill: parent
MegasweepControls { bridge: win.bridge; colors: win.colors; Layout.preferredWidth: 320; Layout.fillHeight: true }
MegasweepWorkspace { bridge: win.bridge; colors: win.colors; Layout.fillWidth: true; Layout.fillHeight: true }
} }''', QUrl.fromLocalFile(str(qml / "MegasweepTestHarness.qml")))
    assert engine.rootObjects(), "Failed to load Megasweep UI"
    window = engine.rootObjects()[0]
    app.processEvents()
    def find(name):
        def walk(item):
            if item.objectName() == name: return item
            for child in item.childItems() if hasattr(item, "childItems") else item.children():
                result = walk(child)
                if result is not None: return result
        result = walk(window.contentItem()) or window.findChild(QObject, name)
        assert result is not None, name
        return result
    def click(name):
        item = find(name)
        assert item.isVisible() and item.isEnabled(), name
        point = item.mapToScene(item.boundingRect().center()).toPoint()
        QTest.mouseClick(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        app.processEvents()
    # Both lines of each paired formula must fit in the selector and its popup.
    labels = ["D = TG + r × BG\nE = TG − r × BG", "D = r × TG + BG\nE = r × TG − BG"]
    bridge.data["parameterGroups"] = [dict(key="coordinates", fields=[dict(
        key="convention", type="choice", label="Doping (D) and E-field (E)", value=0,
        options=[dict(value=i, label=label) for i, label in enumerate(labels)])])]
    bridge.changed.emit()
    QTest.qWait(50)
    selector = find("app_parameter_convention")
    current = selector.property("contentItem")
    assert current.property("lineCount") == 2 and current.height() >= current.property("contentHeight")
    click("app_parameter_convention")
    QTest.qWait(50)
    from PySide6.QtQml import QQmlExpression
    expression = QQmlExpression(engine.rootContext(), selector,
        "[0, 1].every(i => { let row = popup.contentItem.itemAtIndex(i); "
        "return row && row.text === model[i] && row.implicitContentHeight >= contentItem.contentHeight; })")
    assert expression.evaluate()[0], expression.error().toString()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    bridge.data["parameterGroups"] = []
    bridge.changed.emit()
    app.processEvents()
    click("megasweep_mode_reflection")
    assert bridge.data["mode"] == "Reflection"
    sidebar = find("quick_megasweep_controls")
    sidebar_box = sidebar.mapRectToScene(sidebar.boundingRect())
    tabs = find("megasweep_modes")
    tabs_bottom = tabs.mapRectToScene(tabs.boundingRect()).bottom()
    for name in ("megasweep_select_files", "megasweep_background", "megasweep_update"):
        button = find(name)
        box = button.mapRectToScene(button.boundingRect())
        assert sidebar_box.contains(box) and box.top() >= tabs_bottom, name
    click("megasweep_reflection_preview")
    assert ("reflection-preview", None) in bridge.actions
    assert find("megasweep_preview_notice").isVisible()
    assert find("megasweep_reference_x").isVisible() and find("megasweep_reference_y").isVisible()
    def enter_reference(axis, keys, finish):
        field = find("megasweep_reference_" + axis)
        field.forceActiveFocus()
        QTest.keyClick(window, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        for key in keys:
            QTest.keyClick(window, key)
        QTest.keyClick(window, finish)
        app.processEvents()
    enter_reference('x', [Qt.Key.Key_1, Qt.Key.Key_Period, Qt.Key.Key_8], Qt.Key.Key_Return)
    assert ('x', 1.8) in bridge.actions
    assert all(find("megasweep_reference_" + a).property("text") == '2' for a in ('x', 'y'))
    enter_reference('y', [Qt.Key.Key_2, Qt.Key.Key_Period, Qt.Key.Key_1], Qt.Key.Key_Tab)
    assert ('y', 2.1) in bridge.actions and find("megasweep_reference_y").property("text") == '2'
    assert not find("megasweep_plot_limits").isVisible()
    assert find("megasweep_reflection_preview").property("checked") and find("megasweep_return_to_map").isVisible()
    click("megasweep_return_to_map")
    assert ("return-to-map", None) in bridge.actions
    assert find("megasweep_return_to_map").property("checked") and find("megasweep_reflection_preview").isVisible()
    for axis in ("x", "y", "color"):
        assert find("megasweep_limit_min_" + axis + "_auto").isVisible()
        assert find("megasweep_limit_max_" + axis + "_auto").isVisible()
    for width in (640, 800, 1440, 1100):
        window.setWidth(width)
        QTest.qWait(30)
        xy_bar = find("megasweep_coordinate_limits_bar")
        color_bar = find("megasweep_color_limits_bar")
        xy_box = xy_bar.mapRectToScene(xy_bar.boundingRect())
        color_box = color_bar.mapRectToScene(color_bar.boundingRect())
        limits = find("megasweep_plot_limits")
        limit_box = limits.mapRectToScene(limits.boundingRect())
        assert limit_box.contains(xy_box) and limit_box.contains(color_box)
        assert color_box.right() <= window.width()
        for axis, name in [('x', 'Doping_axis_b_set'), ('y', 'Vbias_axis_a_set'), ('color', 'RC peak-to-peak')]:
            title = find('megasweep_limit_title_' + axis + '_auto')
            assert name in title.property('text') and not title.property('truncated')
            assert title.property('contentWidth') <= title.width() + 1
            assert title.property('contentHeight') <= title.height() + 1
        assert not xy_box.intersects(color_box), (width, xy_box, color_box)
        if width == 1440:
            assert xy_box.right() <= color_box.left() and abs(xy_box.top() - color_box.top()) < 2, (limit_box, xy_box, color_box)
            assert 0 <= color_box.left() - xy_box.right() <= 9
            y_group = find('megasweep_limits_group_y_auto')
            assert xy_box.right() - y_group.mapRectToScene(y_group.boundingRect()).right() <= 8
        if width == 640:
            assert xy_box.bottom() <= color_box.top()
        for bound, expected in (('min', '-1.237e-125'), ('max', '9.876e+113')):
            editor = find("megasweep_limit_" + bound + "_color_auto")
            assert editor.property("text") == expected
            usable = editor.width() - editor.property("leftPadding") - editor.property("rightPadding")
            assert usable >= QFontMetricsF(editor.property("font")).horizontalAdvance(expected), (expected, usable)
            right = editor.mapToScene(QPointF(editor.width(), 0)).x()
            assert right <= window.width(), (width, right)
    click("megasweep_reverse_y_auto")
    assert ("y_reverse", True) in bridge.actions
    x_min = find("megasweep_limit_min_x_auto")
    x_min.forceActiveFocus()
    QTest.keyClick(window, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(window, Qt.Key.Key_1)
    click("megasweep_limit_auto_color_auto")
    click("megasweep_limit_auto_color_auto")
    assert ("color_auto", False) in bridge.actions and ("color_auto", True) in bridge.actions
    assert ("x_auto", False) in bridge.actions and x_min.property("text") == '1'
    x_min.forceActiveFocus()
    QTest.keyClick(window, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(window, Qt.Key.Key_0)
    QTest.keyClick(window, Qt.Key.Key_Return)
    assert bridge.limit_rows['x_auto']['low'] == 0
    assert window.findChild(QObject, 'megasweep_apply_limits') is None
    palette = find('megasweep_colormap_color_auto')
    assert palette.property('count') == 2 and palette.property('currentText') == 'RdBu_r'
    click('megasweep_colormap_color_auto')
    QTest.keyClick(window, Qt.Key.Key_Down)
    QTest.keyClick(window, Qt.Key.Key_Return)
    app.processEvents()
    assert ("cmap", "RdBu") in bridge.actions and palette.property('currentText') == 'RdBu'
    click("megasweep_spectral_cut")
    assert find("megasweep_spectral_cut").property("checked")
    click("megasweep_background")
    assert find("megasweep_background_browser").isVisible()
    assert find("megasweep_background_files_tab").property("checked")
    assert not find("megasweep_background_normalization").isVisible()
    role = find("megasweep_background_role")
    assert role.property("currentText") == "Custom · 1"
    click("megasweep_background_role"); QTest.keyClick(window, Qt.Key.Key_Down); QTest.keyClick(window, Qt.Key.Key_Return); app.processEvents()
    assert bridge.data["backgroundRole"] == "p" and role.property("currentText") == "P · 2"
    assert bridge.data["backgroundGroups"]["custom"] == ["c.csv"]
    click("megasweep_background_role"); QTest.keyClick(window, Qt.Key.Key_Up); QTest.keyClick(window, Qt.Key.Key_Return); app.processEvents()
    assert bridge.data["backgroundRole"] == "custom"
    assert find("megasweep_background_role_help").isVisible()
    for width in (640, 800, 1000, 1440):
        window.setWidth(width); window.setHeight(560); app.processEvents()
        for name in ("megasweep_background_role", "megasweep_background_role_help", "drr_background_inline_browse", "drr_background_available_files", "drr_background_chosen_files", "megasweep_background_close"):
            item = find(name); point = item.mapToScene(item.boundingRect().center())
            assert 0 <= point.x() < width and 0 <= point.y() < 560, (width, name, point)
    assert ("choose-background", None) in bridge.actions
    window.setWidth(640); window.setHeight(520); app.processEvents()
    normalization_tab = find("megasweep_background_normalization_tab")
    normalization_tab.forceActiveFocus()
    QTest.keyClick(window, Qt.Key.Key_Space); QTest.qWait(30)
    assert normalization_tab.property("checked")
    assert find("megasweep_background_normalization").isVisible()
    assert not find("megasweep_background_browser").isVisible()
    for name in ("megasweep_background_files_tab", "megasweep_background_normalization_tab", "app_parameter_auto_background_scale", "megasweep_background_close"):
        item = find(name)
        point = item.mapToScene(item.boundingRect().center())
        assert item.isVisible() and 0 <= point.x() < 640 and 0 <= point.y() < 520, (name, point)
    scale_checkbox = find("app_parameter_auto_background_scale")
    indicator = scale_checkbox.mapToScene(QPointF(10, scale_checkbox.height() / 2)).toPoint()
    QTest.mouseClick(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, indicator)
    app.processEvents()
    assert ("auto_background_scale", False) in bridge.actions, bridge.actions
    click("megasweep_background_files_tab")
    assert find("megasweep_background_browser").isVisible()
    assert role.property("currentText") == "Custom · 1"
    click("megasweep_background_normalization_tab")
    click("megasweep_background_close")
    click("megasweep_background")
    assert find("megasweep_background_files_tab").property("checked")
    assert find("megasweep_background_browser").isVisible()
    click("megasweep_background_close")
    assert "Custom (1)" in find("megasweep_background").property("text")
    click("megasweep_mode_transport")
    assert bridge.data["mode"] == "Transport"
    assert not find("megasweep_export_video").isVisible()
    assert not find("megasweep_background_browser").isVisible()
    click("megasweep_mode_reflection")
    assert not find("megasweep_background_browser").isVisible()
    click("megasweep_mode_pl")
    assert bridge.data["mode"] == "PL"
    assert window.findChild(QObject, "megasweep_export") is None
    for name in ("megasweep_select_files", "megasweep_update", "megasweep_slice_previous", "megasweep_slice_next", "megasweep_export_batch"):
        click(name)
    for action in ("select-files", "update", "slice-previous", "slice-next", "export-batch"):
        assert (action, None) in bridge.actions, action
    click("megasweep_export_video")
    assert find("megasweep_movie_dialog").property("visible")
    for width in (640, 800, 1000, 1440):
        window.setWidth(width); window.setHeight(560); app.processEvents()
        for name in ("megasweep_movie_first", "megasweep_movie_preview", "megasweep_movie_seek", "megasweep_movie_export"):
            item = find(name); point = item.mapToScene(item.boundingRect().center())
            assert 0 <= point.x() < width and 0 <= point.y() < 560, (width, name, point)
    window.setWidth(1100); window.setHeight(760); app.processEvents()
    for name in ("megasweep_movie_first", "megasweep_movie_last", "megasweep_movie_stride", "megasweep_movie_fps", "megasweep_movie_width", "megasweep_movie_height", "megasweep_movie_direction", "megasweep_movie_preview", "megasweep_movie_seek", "megasweep_movie_play", "megasweep_movie_export"):
        assert find(name).isVisible(), name
    first = find("megasweep_movie_first")
    first.forceActiveFocus(); app.processEvents()
    for key in (Qt.Key.Key_1, Qt.Key.Key_Period, Qt.Key.Key_2, Qt.Key.Key_5): QTest.keyClick(window, key)
    QTest.keyClick(window, Qt.Key.Key_Tab); app.processEvents()
    assert ("movie.first", 1.25) in bridge.actions
    first.forceActiveFocus(); QTest.keyClick(window, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier); QTest.keyClick(window, Qt.Key.Key_Backspace); QTest.keyClick(window, Qt.Key.Key_Tab); app.processEvents()
    assert ("movie.first", None) in bridge.actions
    fps = find("megasweep_movie_fps"); fps.forceActiveFocus(); QTest.keyClick(window, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    for key in (Qt.Key.Key_2, Qt.Key.Key_Period, Qt.Key.Key_5): QTest.keyClick(window, key)
    QTest.keyClick(window, Qt.Key.Key_Tab); app.processEvents()
    assert ("movie.fps", 25) in bridge.actions, (fps.property("text"), bridge.actions)
    bridge.data["busy"] = True; bridge.changed.emit(); app.processEvents()
    assert not first.isEnabled()
    bridge.data["busy"] = False; bridge.changed.emit(); app.processEvents()
    click("megasweep_movie_preview")
    assert ("movie-preview", None) in bridge.actions
    click("megasweep_movie_seek")
    assert any(key == "movie.previewFrame" for key, value in bridge.actions)
    bridge.actions.clear()
    click("megasweep_movie_play")
    QTest.qWait(100)
    assert any(key == "movie.previewFrame" for key, value in bridge.actions)
    assert not errors, errors
    window.close()

if __name__ == "__main__":
    if "--qml-child" in sys.argv: child()
    else: unittest.main()
