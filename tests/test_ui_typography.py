"""Arial reaches native controls and QML text without altering typography."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QUrl
from PySide6.QtGui import QFont, QFontDatabase, QFontInfo
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout

from core.typography import FONT_FAMILY
from ui_qt.theme import install_theme
from ui_qt.typography import install_application_font

ROOT = Path(__file__).resolve().parents[1]


class UiTypographyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        previous = QFont(self.app.font())
        self.addCleanup(self.app.setFont, previous)

    def assertArial(self, font):
        self.assertEqual(font.family(), FONT_FAMILY)
        # Windows CI and this application host have installed Arial. Other
        # platforms may substitute an unavailable system font at render time.
        if sys.platform == "win32" or FONT_FAMILY in QFontDatabase.families():
            self.assertEqual(QFontInfo(font).family(), FONT_FAMILY)

    def test_font_helper_import_keeps_qt_and_science_lazy(self):
        code = """
import sys
from ui_qt.typography import install_application_font
assert not any(name in sys.modules for name in ('PySide6', 'matplotlib', 'numpy', 'scipy', 'pandas'))
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_quick_launchers_install_arial_before_the_event_loop(self):
        code = """
import runpy, sys
from PySide6.QtGui import QFontDatabase, QFontInfo, QGuiApplication
from core.typography import FONT_FAMILY
original_exec = QGuiApplication.exec
def checked_exec():
    font = QGuiApplication.instance().font()
    assert font.family() == FONT_FAMILY, font.toString()
    if sys.platform == 'win32' or FONT_FAMILY in QFontDatabase.families():
        assert QFontInfo(font).family() == FONT_FAMILY, QFontInfo(font).family()
    assert 'PySide6.QtWidgets' not in sys.modules
    return original_exec()
QGuiApplication.exec = staticmethod(checked_exec)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        for entry in ("run_quick.py", "run_drr_analysis_quick.py",
                      "run_peak_analysis_quick.py", "run_mcd_organizer_quick.py"):
            with self.subTest(entry=entry), TemporaryDirectory(prefix="arial-entry-") as temporary:
                environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                    MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary, XDG_CONFIG_HOME=temporary,
                    PYTHONIOENCODING="utf-8")
                result = subprocess.run([sys.executable, "-c", code, entry, "--check"],
                    cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_application_family_preserves_point_and_pixel_font_attributes(self):
        for pixels in (False, True):
            with self.subTest(pixels=pixels):
                original = QFont(self.app.font())
                original.setFamily("Sans Serif")
                original.setWeight(QFont.Weight.DemiBold)
                original.setItalic(True)
                if pixels:
                    original.setPixelSize(19)
                else:
                    original.setPointSizeF(12.5)
                self.app.setFont(original)
                install_application_font(self.app)
                actual = self.app.font()
                self.assertArial(actual)
                self.assertEqual(actual.pixelSize(), original.pixelSize())
                self.assertEqual(actual.pointSizeF(), original.pointSizeF())
                self.assertEqual(actual.weight(), original.weight())
                self.assertEqual(actual.italic(), original.italic())

    def test_widgets_and_dialogs_keep_arial_through_all_theme_modes(self):
        original = QFont(self.app.font())
        original.setPointSizeF(12.5)
        original.setWeight(QFont.Weight.Normal)
        self.app.setFont(original)
        manager = install_theme(self.app, mode="light")
        self.addCleanup(manager.set_mode, "light")
        dialog = QDialog()
        self.addCleanup(dialog.close)
        layout = QVBoxLayout(dialog)
        controls = [QLabel("Plot settings"), QLineEdit("Spectrum"), QPushButton("Save")]
        for control in controls:
            layout.addWidget(control)
        heading = QFont(controls[0].font())
        heading.setPointSizeF(15)
        heading.setBold(True)
        controls[0].setFont(heading)
        dialog.ensurePolished()
        for mode in ("light", "dark", "system", "high-contrast"):
            with self.subTest(mode=mode):
                manager.set_mode(mode)
                self.app.processEvents()
                self.assertArial(self.app.font())
                self.assertEqual(self.app.font().pointSizeF(), 12.5)
                self.assertEqual(self.app.font().weight(), QFont.Weight.Normal)
                for control in (dialog, *controls):
                    self.assertArial(control.font())
                self.assertEqual(controls[0].font().pointSizeF(), 15)
                self.assertTrue(controls[0].font().bold())

    def test_actual_quick_shell_text_and_popups_inherit_arial_across_themes(self):
        from ui_qt.quick.application_bridge import QuickApplicationBridge
        from ui_qt.quick.plot_surface import register_plot_surface

        install_application_font(self.app)
        with TemporaryDirectory(prefix="arial-qml-") as temporary:
            bridge = QuickApplicationBridge(settings=QSettings(str(Path(temporary) / "settings.ini"), QSettings.IniFormat))
            engine = QQmlApplicationEngine()
            warnings = []
            engine.warnings.connect(lambda errors: warnings.extend(error.toString() for error in errors))
            register_plot_surface()
            engine.setInitialProperties({"appBridge": bridge})
            try:
                engine.load(QUrl.fromLocalFile(str(ROOT / "ui_qt/quick/qml/Application.qml")))
                self.assertTrue(engine.rootObjects(), "\n".join(warnings))
                window = engine.rootObjects()[0]
                for mode in ("light", "dark", "system", "high-contrast"):
                    with self.subTest(mode=mode):
                        bridge._appearance.setMode(mode)
                        self.app.processEvents()
                        pending = [window]
                        visited = set()
                        text_count = 0
                        while pending:
                            item = pending.pop()
                            if item in visited:
                                continue
                            visited.add(item)
                            pending.extend(item.children())
                            if isinstance(item, QQuickItem):
                                pending.extend(item.childItems())
                            font = item.property("font")
                            if isinstance(font, QFont):
                                self.assertArial(font)
                                if isinstance(item.property("text"), str):
                                    text_count += 1
                        self.assertGreater(text_count, 25)
                self.assertEqual(warnings, [])
            finally:
                for window in engine.rootObjects():
                    window.close()
                engine.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                self.app.processEvents()
                bridge.close()


if __name__ == "__main__":
    unittest.main()
