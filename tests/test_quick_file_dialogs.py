"""Exercise picker directories and cancellation with only synthetic folders."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PySide6.QtCore import QMetaObject, QObject, QUrl
from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QJSValue
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtTest import QTest


class QuickFileDialogsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QGuiApplication.instance() is None:
            QQuickStyle.setStyle("Basic")
        cls.app = QGuiApplication.instance() or QGuiApplication([])
        if os.name == "nt" and not QFontDatabase.families():
            font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
            if font.is_file():
                QFontDatabase.addApplicationFont(str(font))
                cls.app.setFont(QFont("Segoe UI", 9))

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="quick-picker-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.a = self.root / "a"
        self.b = self.root / "b"
        for folder in (self.a, self.b):
            folder.mkdir()
            (folder / "child").mkdir()
            (folder / "measurement.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        self.engine = QQmlApplicationEngine()
        self.warnings = []
        self.engine.warnings.connect(lambda messages: self.warnings.extend(item.toString() for item in messages))
        qml_dir = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml"
        self.engine.loadData(b'''
            import QtQuick
            import QtQuick.Controls
            import QtQuick.Dialogs
            import "DialogState.js" as DialogState
            ApplicationWindow {
                id: window
                width: 800; height: 600; visible: true
                property url targetFolder
                property url suggestedFile
                property url acceptedFolder
                property int rejectedCount: 0
                function openFolder() {
                    DialogState.prepareFolder(folderPicker, targetFolder)
                    folderPicker.open()
                }
                function cancelFolderImmediately() {
                    openFolder()
                    folderPicker.reject()
                }
                function openSaveFile() {
                    filePicker.fileMode = FileDialog.SaveFile
                    DialogState.prepareFile(filePicker, targetFolder, suggestedFile)
                    filePicker.open()
                }
                function openMeasurements() {
                    filePicker.fileMode = FileDialog.OpenFiles
                    DialogState.prepareFile(filePicker, targetFolder)
                    filePicker.open()
                }
                AppFolderDialog {
                    id: folderPicker
                    objectName: "folder_picker"
                    onAccepted: window.acceptedFolder = selectedFolder
                    onRejected: window.rejectedCount++
                }
                FileDialog {
                    id: filePicker
                    objectName: "file_picker"
                    options: FileDialog.DontUseNativeDialog
                    nameFilters: ["CSV measurements (*.csv)"]
                }
            }
        ''', QUrl.fromLocalFile(str(qml_dir / "picker-test.qml")))
        self.assertTrue(self.engine.rootObjects(), "\n".join(self.warnings))
        self.window = self.engine.rootObjects()[0]
        self.folder = self.window.findChild(QObject, "folder_picker")
        self.file = self.window.findChild(QObject, "file_picker")
        self.addCleanup(self.dispose)
        QTest.qWait(30)

    def dispose(self):
        QMetaObject.invokeMethod(self.folder, "reject")
        QMetaObject.invokeMethod(self.file, "reject")
        self.window.close()
        self.engine.deleteLater()
        self.app.processEvents()

    def invoke(self, name):
        self.assertTrue(QMetaObject.invokeMethod(self.window, name))
        QTest.qWait(40)

    def target(self, folder):
        self.window.setProperty("targetFolder", QUrl.fromLocalFile(str(folder)))

    @staticmethod
    def active_dialog(wrapper):
        dialog = wrapper.property("_dialog")
        return dialog.toQObject() if isinstance(dialog, QJSValue) else dialog

    def assert_folder(self, obj, folder):
        self.assertEqual(Path(obj.property("currentFolder").toLocalFile()), folder)

    def test_folder_reopen_uses_the_new_requested_directory(self):
        self.target(self.a)
        self.invoke("openFolder")
        first = self.active_dialog(self.folder)
        self.assertTrue(self.folder.property("visible"))
        self.assert_folder(first, self.a)
        QMetaObject.invokeMethod(self.folder, "reject")
        QTest.qWait(30)
        self.assertIsNone(self.active_dialog(self.folder))

        self.target(self.b)
        self.invoke("openFolder")
        second = self.active_dialog(self.folder)
        self.assertIsNot(first, second)
        self.assert_folder(second, self.b)
        self.assert_folder(self.folder, self.b)
        self.assertEqual(self.window.property("rejectedCount"), 1)

    def test_folder_result_is_copied_before_result_signal_and_disposal(self):
        self.target(self.a)
        self.invoke("openFolder")
        selected = self.a / "child"
        dialog = self.active_dialog(self.folder)
        dialog.setProperty("selectedFolder", QUrl.fromLocalFile(str(selected)))
        QMetaObject.invokeMethod(dialog, "accept")
        self.assertEqual(Path(self.window.property("acceptedFolder").toLocalFile()), selected)
        self.assertIsNone(self.active_dialog(self.folder))
        self.assertFalse(self.folder.property("visible"))
        QTest.qWait(30)

    def test_folder_can_cancel_during_opening_and_reopen(self):
        self.target(self.a)
        self.invoke("cancelFolderImmediately")
        self.assertFalse(self.folder.property("visible"))
        self.assertIsNone(self.active_dialog(self.folder))
        self.assertEqual(self.window.property("rejectedCount"), 1)
        self.target(self.b)
        self.invoke("openFolder")
        self.assert_folder(self.active_dialog(self.folder), self.b)

    def test_folder_close_and_visible_false_release_the_instance(self):
        self.target(self.a)
        self.invoke("openFolder")
        QMetaObject.invokeMethod(self.folder, "close")
        QTest.qWait(30)
        self.assertIsNone(self.active_dialog(self.folder))
        self.assertFalse(self.folder.property("visible"))

        self.target(self.b)
        self.invoke("openFolder")
        self.assert_folder(self.active_dialog(self.folder), self.b)
        self.folder.setProperty("visible", False)
        QTest.qWait(30)
        self.assertIsNone(self.active_dialog(self.folder))
        self.assertEqual(self.window.property("rejectedCount"), 2)

        self.target(self.a)
        self.invoke("openFolder")
        self.assert_folder(self.active_dialog(self.folder), self.a)

    def test_file_request_drops_save_selection_before_changing_directory(self):
        self.target(self.b)
        previous = self.b / "export.csv"
        self.window.setProperty("suggestedFile", QUrl.fromLocalFile(str(previous)))
        self.invoke("openSaveFile")
        self.assertEqual(Path(self.file.property("selectedFile").toLocalFile()), previous)
        QMetaObject.invokeMethod(self.file, "reject")
        QTest.qWait(30)
        self.target(self.a)
        self.invoke("openMeasurements")
        self.assert_folder(self.file, self.a)
        urls = self.file.property("selectedFiles")
        if isinstance(urls, QJSValue):
            urls = urls.toVariant()
        self.assertTrue(all(Path(url.toLocalFile()).parent == self.a for url in urls))
        self.assertFalse(self.warnings, "\n".join(self.warnings))


if __name__ == "__main__":
    unittest.main()
