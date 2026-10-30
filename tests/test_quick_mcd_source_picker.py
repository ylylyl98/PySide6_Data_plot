"""Live MCD file selection: full long names, single-file drafts and explicit load."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickMcdSourcePickerTests(unittest.TestCase):
    def test_isolated_mcd_and_peak_file_pickers(self):
        with TemporaryDirectory() as directory:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                QT_SCALE_FACTOR="1", LOCALAPPDATA=directory, XDG_CONFIG_HOME=directory,
                QML_DISK_CACHE_PATH=directory, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                                    cwd=Path(__file__).resolve().parents[1], env=environment,
                                    capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import json
    import time
    from unittest.mock import patch
    from PySide6.QtCore import QObject, QSettings, QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlComponent, QQmlEngine
    from PySide6.QtQuick import QQuickWindow
    from ui_qt.typography import install_application_font
    from ui_qt.quick.mcd_bridge import McdBridge
    from ui_qt.quick.mcd_peak_bridge import McdPeakBridge
    from ui_qt.quick.plot_surface import register_plot_surface

    app = QGuiApplication([])
    install_application_font(app)
    register_plot_surface()
    colors = dict(surface="#f3f5f8", control="#ffffff", text="#20242b", muted="#606978", border="#c6cbd3",
        accent="#0067c0", hover="#edf1f6", pressed="#e0e7ef", error="#b42318", disabled="#939aa5",
        selectedText="#ffffff", inset="#e9edf2", divider="#dce1e8", borderHover="#aeb7c4",
        borderDisabled="#dce1e8", controlDisabled="#edf0f4", focus="#20242b", accentText="#0067c0",
        selected="#e6f0fa", selectedBorder="#b4d6fa", errorSurface="#fdf1f0")

    def settle(milliseconds=35):
        end = time.monotonic()+milliseconds/1000
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(.001)

    def find_visual(window, name):
        pending = [window.contentItem()]
        while pending:
            item = pending.pop()
            if item.objectName() == name:
                return item
            pending.extend(item.childItems())

    class Cases(unittest.TestCase):
        def test_mcd_and_peak_use_full_name_picker_and_explicit_selection(self):
            for Bridge, filename in ((McdBridge, "McdWorkspace.qml"), (McdPeakBridge, "McdPeakWorkspace.qml")):
                with self.subTest(page=filename), TemporaryDirectory() as directory:
                    folder = Path(directory).resolve()
                    source_folder = folder/"mcd"
                    source_folder.mkdir()
                    prefix = "MCD_" + "shared_filename_prefix_with_spectral_conditions_"*3
                    first = source_folder/(prefix+"device_gate_0_temperature_4K_first.csv")
                    second = source_folder/(prefix+"device_gate_1_temperature_4K_second.csv")
                    for path in (first, second):
                        path.write_text("cataloging must not parse spectra", encoding="utf-8")
                    history = folder/"Processed Data"/"MCD"
                    history.mkdir(parents=True)
                    (history/"second_MCD_settings.json").write_text(json.dumps(dict(
                        workflow="mcd", source_relative_path="mcd/"+second.name,
                        created_utc="2026-10-30T12:00:00Z")), encoding="utf-8")
                    bridge = Bridge(settings=QSettings(str(folder/"settings.ini"), QSettings.IniFormat))
                    engine = QQmlEngine()
                    warnings = []
                    engine.warnings.connect(lambda messages: warnings.extend(message.toString() for message in messages))
                    path = Path(__file__).resolve().parents[1]/"ui_qt"/"quick"/"qml"/filename
                    component = QQmlComponent(engine, QUrl.fromLocalFile(str(path)))
                    item = component.createWithInitialProperties(dict(bridge=bridge, colors=colors))
                    self.assertIsNotNone(item, "\n".join(error.toString() for error in component.errors()))
                    pane_component = QQmlComponent(engine, QUrl.fromLocalFile(str(path.with_name(filename.replace("Workspace", "Parameters")))))
                    pane = pane_component.createWithInitialProperties(dict(bridge=bridge, colors=colors))
                    self.assertIsNotNone(pane, "\n".join(error.toString() for error in pane_component.errors()))
                    window = QQuickWindow()
                    window.resize(1200, 760)
                    pane.setParentItem(window.contentItem())
                    pane.setWidth(240)
                    pane.setHeight(760)
                    item.setParentItem(window.contentItem())
                    item.setX(240)
                    item.setWidth(960)
                    item.setHeight(760)
                    window.show()
                    try:
                        with patch.object(bridge, "_queue_load") as load:
                            bridge.setExperimentFolder(str(folder))
                            bridge.openFiles("sources", [str(first)])
                            load.reset_mock()
                            settle(50)
                            self.assertIsNone(find_visual(window, "app_parameter_source"))
                            summary = find_visual(window, "app_parameter_sourceSummary")
                            self.assertTrue(summary.property("text").endswith("first.csv"))
                            self.assertFalse(summary.property("truncated"))
                            find_visual(window, "app_parameter_select-files").clicked.emit()
                            deadline = time.monotonic()+10
                            while (bridge._catalog_busy or len(bridge._catalog) < 2) and time.monotonic() < deadline:
                                settle()
                            settle(80)
                            picker = item.findChild(QObject, "mcd_source_catalog")
                            self.assertIsNotNone(picker)
                            self.assertTrue(picker.property("visible"))
                            files = find_visual(window, "mcd_source_files")
                            self.assertEqual(files.property("count"), 2)
                            search = find_visual(window, "mcd_source_search")
                            search.setProperty("text", "gate_1")
                            settle()
                            self.assertEqual(files.property("count"), 1)
                            model_changes = []
                            files.modelChanged.connect(lambda: model_changes.append(True))
                            bridge.setSourceDraft(str(second))
                            settle()
                            self.assertEqual(model_changes, [], "Selecting a row reset the file list")
                            self.assertEqual(find_visual(window, "mcd_source_full_path").property("text"), str(second))
                            load.assert_not_called()
                            self.assertEqual(bridge._source, str(first))
                            search.setProperty("text", "")
                            bridge.setParameter("historyFilter", "processed")
                            settle()
                            self.assertEqual(files.property("count"), 1)
                            bridge.setParameter("historyFilter", "all")
                            settle()
                            capture = os.environ.get("DPTK_MCD_PICKER_CAPTURE")
                            if capture:
                                window.grabWindow().save(capture+"-"+filename.removesuffix(".qml")+"-wide.png")
                            window.resize(480, 700)
                            pane.setVisible(False)
                            item.setX(0)
                            item.setWidth(480)
                            item.setHeight(700)
                            settle(100)
                            label = find_visual(window, "mcd_source_filename_"+str(second))
                            self.assertIsNotNone(label)
                            self.assertTrue(label.property("text").endswith("second.csv"))
                            self.assertFalse(label.property("truncated"))
                            self.assertGreater(label.property("lineCount"), 3)
                            if capture:
                                window.grabWindow().save(capture+"-"+filename.removesuffix(".qml")+"-compact.png")
                            find_visual(window, "mcd_source_cancel").clicked.emit()
                            settle()
                            self.assertFalse(picker.property("visible"))
                            self.assertEqual(bridge._source, str(first))
                            bridge.invoke("select-files")
                            settle()
                            self.assertEqual(bridge._source_draft, str(first))
                            bridge.setSourceDraft(str(second))
                            settle()
                            find_visual(window, "mcd_source_accept").clicked.emit()
                            settle()
                            self.assertEqual(bridge._source, str(second))
                            self.assertFalse(picker.property("visible"))
                            load.assert_called_once_with(immediate=True)
                        self.assertFalse(warnings, "\n".join(warnings))
                    finally:
                        window.close()
                        item.setParentItem(None)
                        pane.setParentItem(None)
                        bridge.close()
                        item.deleteLater()
                        pane.deleteLater()
                        engine.deleteLater()
                        settle()

    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(isolated())
    unittest.main()
