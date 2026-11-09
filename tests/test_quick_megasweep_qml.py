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
    from PySide6.QtCore import QObject, Property, Signal, Slot, QUrl, QSettings, Qt, qInstallMessageHandler
    from PySide6.QtGui import QGuiApplication
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
        def __init__(self):
            super().__init__()
            self.bg = BackgroundSourceCatalogBridge(settings=QSettings("DPTK tests", "Megasweep QML"), parent=self)
            self.data = dict(mode="PL", values=dict(view="slice"), parameterGroups=[], busy=False,
                sourceFile="example.csv", sourcePath="", canExport=True, summary="Loaded", error="", status="Ready",
                movieOptions=dict(fps=30), moviePreviewFrameCount=4, moviePreviewFrameIndex=0, movieDuration=4/30)
            self.actions = []
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
            elif key == "movie.previewFrame": self.data["moviePreviewFrameIndex"] = value
            elif key.startswith("movie."): self.data["movieOptions"][key[6:]] = value
            self.actions.append((key, value)); self.changed.emit()
        @Slot(str)
        def invoke(self, name):
            self.actions.append((name, None))
            if name == "reflection-preview": self.data["reflectionPreviewVisible"] = True; self.changed.emit()
            elif name == "return-to-map": self.data["reflectionPreviewVisible"] = False; self.changed.emit()
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
    click("megasweep_mode_reflection")
    assert bridge.data["mode"] == "Reflection"
    click("megasweep_reflection_preview")
    assert ("reflection-preview", None) in bridge.actions
    assert find("megasweep_preview_notice").isVisible()
    assert not find("megasweep_reflection_preview").isVisible()
    click("megasweep_return_to_map")
    assert ("return-to-map", None) in bridge.actions
    assert not find("megasweep_return_to_map").isVisible()
    click("megasweep_background")
    assert find("megasweep_background_browser").isVisible()
    for width in (640, 800, 1000, 1440):
        window.setWidth(width); window.setHeight(560); app.processEvents()
        for name in ("drr_background_inline_browse", "drr_background_available_files", "drr_background_chosen_files", "megasweep_background_close"):
            item = find(name); point = item.mapToScene(item.boundingRect().center())
            assert 0 <= point.x() < width and 0 <= point.y() < 560, (width, name, point)
    assert ("choose-background", None) in bridge.actions
    click("megasweep_background_close")
    click("megasweep_mode_transport")
    assert bridge.data["mode"] == "Transport"
    assert not find("megasweep_export_video").isVisible()
    assert not find("megasweep_background_browser").isVisible()
    click("megasweep_mode_reflection")
    assert not find("megasweep_background_browser").isVisible()
    click("megasweep_mode_pl")
    assert bridge.data["mode"] == "PL"
    for name in ("megasweep_select_files", "megasweep_update", "megasweep_slice_previous", "megasweep_slice_next", "megasweep_export", "megasweep_export_batch"):
        click(name)
    for action in ("select-files", "update", "slice-previous", "slice-next", "export", "export-batch"):
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
