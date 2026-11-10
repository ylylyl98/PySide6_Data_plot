"""Rendered shared-button interaction checks in an isolated Quick process."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ButtonAffordanceTests(unittest.TestCase):
    def test_quick_button_states(self):
        with tempfile.TemporaryDirectory(prefix="button-affordance-") as tmp:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                       XDG_CONFIG_HOME=tmp, XDG_CACHE_HOME=tmp, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, __file__, "--isolated"], cwd=Path(__file__).resolve().parents[1],
                                    env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import time
    from PySide6.QtCore import QObject, QPoint, Qt, QUrl
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication
    from PySide6.QtQml import QQmlComponent, QQmlEngine, QQmlProperty
    from PySide6.QtQuick import QQuickWindow
    from PySide6.QtTest import QTest
    from ui_qt.quick.appearance import quick_colors
    app = QGuiApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    app.setFont(QFont("Segoe UI", 10))
    directory = Path(__file__).resolve().parents[1] / "ui_qt/quick/qml"
    qml = '''import QtQuick
import QtQuick.Layouts
import "%s" as UI
Rectangle {
    id: root; width: 680; height: 260
    property var colors; property int clicks: 0; property int toggles: 0; color: colors.surface
    ColumnLayout { anchors.fill: parent; anchors.margins: 22; spacing: 14
        Text { text: "Button states"; color: root.colors.text; font.bold: true }
        RowLayout { spacing: 12
            UI.AppButton { objectName: "normal"; text: "Normal"; colors: root.colors; onClicked: root.clicks++ }
            UI.AppButton { objectName: "primary"; text: "Primary"; colors: root.colors; primary: true; onClicked: root.clicks++ }
            UI.AppButton { objectName: "checked"; text: "Checked"; colors: root.colors; checkable: true; checked: true; onClicked: root.clicks++ }
            UI.AppButton { objectName: "disabled"; text: "Disabled"; colors: root.colors; enabled: false; primary: true; onClicked: root.clicks++ }
        }
        RowLayout { spacing: 12
            UI.DisplayButton { objectName: "display"; text: "Display"; colors: root.colors; onClicked: root.clicks++ }
            UI.DisplayButton { objectName: "subtle"; text: "Subtle"; colors: root.colors; subtle: true; onClicked: root.clicks++ }
            UI.DisplayButton { objectName: "displaydisabled"; text: "Disabled"; colors: root.colors; subtle: true; enabled: false }
        }
        RowLayout { spacing: 12
            UI.AppParameterGroup { Layout.preferredWidth: 220; colors: root.colors; bridge: null; descriptors: ({}); schema: ({key:"fold", fields:[]}); descriptor: ({title:"Advanced settings", collapsible:true}); onToggleRequested: root.toggles++ }
            UI.AppParameterGroup { Layout.preferredWidth: 220; colors: root.colors; bridge: null; descriptors: ({}); schema: ({key:"plain", fields:[]}); descriptor: ({title:"Plain heading"}) }
        }
        Item { Layout.fillHeight: true }
    }
}''' % directory.as_uri()
    engine = QQmlEngine(); component = QQmlComponent(engine)
    component.setData(qml.encode(), QUrl.fromLocalFile(str(directory / "ButtonGallery.qml")))
    deadline = time.monotonic() + 10
    while component.isLoading() and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.005)
    if not component.isReady(): raise AssertionError(str(component.errors()))
    def settle():
        for _ in range(20): app.processEvents(); time.sleep(.005)
    for theme in ("light", "dark", "high-contrast"):
        colors = quick_colors(theme)
        item = component.createWithInitialProperties({"colors": colors})
        assert item is not None, str(component.errors())
        window = QQuickWindow(); window.resize(680, 260); item.setParentItem(window.contentItem()); window.show(); settle()
        def find(name):
            found = item.findChild(QObject, name)
            assert found is not None, name
            return found
        def bg(button): return button.property("background")
        def check_surface(name):
            box = bg(find(name)); color = box.property("color"); width = QQmlProperty.read(box, "border.width")
            assert QColor(color).alpha() == 255, (theme, name, color)
            assert width >= 1, (theme, name, width)
        for name in ("normal", "primary", "checked", "disabled", "display", "subtle", "displaydisabled", "app_parameter_group_fold"):
            check_surface(name)
        assert QColor(bg(find("app_parameter_group_plain")).property("color")).alpha() == 0
        assert QColor(bg(find("disabled")).property("color")) == QColor(colors["controlDisabled"])
        def center(button):
            p = button.mapToScene(button.boundingRect().center()); return QPoint(round(p.x()), round(p.y()))
        normal, primary, checked, disabled = (find(x) for x in ("normal", "primary", "checked", "disabled"))
        QTest.mouseClick(window, Qt.LeftButton, Qt.NoModifier, center(find("app_parameter_group_fold"))); settle(); assert item.property("toggles") == 1
        QTest.mouseMove(window, center(normal)); settle(); assert normal.property("hovered")
        QTest.mousePress(window, Qt.LeftButton, Qt.NoModifier, center(normal)); settle(); assert normal.property("down")
        QTest.mouseRelease(window, Qt.LeftButton, Qt.NoModifier, center(normal)); settle(); assert item.property("clicks") == 1
        QTest.mouseClick(window, Qt.LeftButton, Qt.NoModifier, center(checked)); settle(); assert not checked.property("checked")
        QTest.mouseClick(window, Qt.LeftButton, Qt.NoModifier, center(disabled)); settle(); assert item.property("clicks") == 2
        QTest.mouseMove(window, center(primary)); settle(); assert primary.property("hovered")
        hover_color = QColor(bg(primary).property("color"))
        QTest.mousePress(window, Qt.LeftButton, Qt.NoModifier, center(primary)); settle(); assert primary.property("down")
        assert QColor(bg(primary).property("color")) == QColor(colors["accentPressed"])
        QTest.mouseRelease(window, Qt.LeftButton, Qt.NoModifier, center(primary)); settle()
        assert hover_color == QColor(colors["accentHover"])
        normal.forceActiveFocus(Qt.TabFocusReason); settle(); assert normal.property("visualFocus")
        assert QQmlProperty.read(bg(normal), "border.width") >= 2
        if theme in ("light", "dark") and os.environ.get("DPTK_BUTTON_GALLERY"):
            checked.setProperty("checked", True)
            QTest.mouseMove(window, QPoint(670, 250)); window.contentItem().forceActiveFocus(); settle()
            folder = Path(os.environ["DPTK_BUTTON_GALLERY"])
            folder.mkdir(parents=True, exist_ok=True)
            assert window.grabWindow().save(str(folder / ("buttons-" + theme + ".png")))
        window.close(); item.deleteLater(); settle()


if __name__ == "__main__":
    if "--isolated" in sys.argv: isolated()
    else: unittest.main()
