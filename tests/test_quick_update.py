"""Isolated native updater state/UI with fake network and no installer execution."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class NativeUpdateTests(unittest.TestCase):
    def test_native_help_and_update_lifecycle(self):
        with TemporaryDirectory(prefix="quick-update-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", PYTHONIOENCODING="utf-8",
                XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_UPDATE_TESTS=", result.stdout)


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import hashlib
    import importlib.abc
    import json
    import threading
    import time
    import traceback
    from dataclasses import replace
    from unittest.mock import Mock, patch
    class NoWidgets(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.main_window", "ui_qt.common"}:
                raise AssertionError("Native updater imported Widgets: " + fullname)
    sys.meta_path.insert(0, NoWidgets())
    from PySide6.QtCore import QCoreApplication, QEvent, QMetaObject, QObject, QSettings, QUrl, Qt, qInstallMessageHandler
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from core.update_checker import (CheckResult, DownloadResult, ReleaseInfo, RELEASES_URL, SUMS_FILENAME,
        check_for_update, download_installer, sha256_file)
    from ui_qt.quick.update_bridge import UpdateBridge, UpdateServices
    from ui_qt.quick.appearance import quick_colors
    from tests.test_update_checker import FakeOpener, make_release
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    python_errors, qml_errors = [], []
    def exception(kind, value, trace):
        python_errors.append("".join(traceback.format_exception(kind, value, trace)))
        traceback.print_exception(kind, value, trace)
    sys.excepthook = exception
    def qt_message(kind, context, message):
        if any(word in message for word in ("ReferenceError", "TypeError", "Binding loop", "Cannot assign", "Unable to assign", "Error:")):
            qml_errors.append(str(message))
            print(message, file=sys.stderr)
    old_handler = qInstallMessageHandler(qt_message)
    def wait(predicate, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.003)
        raise AssertionError("Timed out waiting for native update state")
    def find(root, name):
        if root.objectName() == name:
            return root
        children = root.childItems() if hasattr(root, "childItems") else root.children()
        for child in children:
            match = find(child, name)
            if match is not None:
                return match
        return None
    filename = "DPTK-Setup-v2.0.0-Windows-x64.exe"
    prefix = "https://github.com/ylylyl98/PySide6_Data_plot/releases/download/v2.0.0/"
    release_url = "https://github.com/ylylyl98/PySide6_Data_plot/releases/tag/v2.0.0"
    available = CheckResult("update_available", (1, 0, 0), (2, 0, 0),
        ReleaseInfo("v2.0.0", (2, 0, 0), False, False, release_url, "Release notes\nNew native frontend.", {}),
        prefix + filename, prefix + SUMS_FILENAME, "Version 2.0.0 is available.")

    class Cases(unittest.TestCase):
        def setUp(self):
            initial_python, initial_qml = len(python_errors), len(qml_errors)
            self.addCleanup(lambda: self.assertEqual(python_errors[initial_python:], [], "Uncaught update callback"))
            self.addCleanup(lambda: self.assertEqual(qml_errors[initial_qml:], [], "QML update error"))
            self.temporary = TemporaryDirectory(prefix="quick-update-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name)
            self.settings = QSettings(str(self.root / "isolated.ini"), QSettings.Format.IniFormat)
            self.settings.setValue("unrelated/preserved", "keep")
            self.body = b"fake installer data, never executed"
            self.digest = hashlib.sha256(self.body).hexdigest()
            self.check = Mock(return_value=available)
            self.download = Mock(side_effect=self.fake_download)
            self.checksum = Mock(side_effect=sha256_file)
            self.open_url = Mock(return_value=True)
            self.services = UpdateServices(self.check, self.download, self.checksum, self.open_url)
            self.bridge = UpdateBridge(settings=self.settings, services=self.services,
                current_version="1.0.0", download_directory=self.root / "downloads")
            self.addCleanup(self.bridge.close)
            self.status = []
            self.bridge.statusMessage.connect(self.status.append)
            # A constructor or forgotten default may never reach the network.
            self.network = patch("urllib.request.build_opener", side_effect=AssertionError("Unexpected real network"))
            self.network.start()
            self.addCleanup(self.network.stop)

        def fake_download(self, installer, sums, name, destination, *, progress):
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / name
            path.write_bytes(self.body)
            progress(25)
            progress(100)
            return DownloadResult("ok", str(path), self.digest, "Downloaded.")

        def idle(self):
            wait(lambda: not self.bridge.snapshot["busy"])

        def downloaded(self):
            self.bridge.invoke("check")
            self.idle()
            self.bridge.invoke("download")
            self.idle()
            self.assertEqual(self.bridge.snapshot["stage"], "downloaded")

        def test_defaults_setting_retention_and_no_automatic_network_in_constructor(self):
            app.processEvents()
            self.assertTrue(self.bridge.snapshot["automaticChecksEnabled"])
            self.check.assert_not_called()
            self.bridge.setAutomatic(False)
            self.bridge.scheduleAutomaticCheck(0)
            app.processEvents()
            self.check.assert_not_called()
            self.assertIs(self.settings.value("updates/check_automatically"), False)
            self.assertEqual(self.settings.value("unrelated/preserved"), "keep")
            for value, expected in (("false", False), ("0", False), ("true", True), ("yes", True), (None, True)):
                self.settings.setValue("updates/check_automatically", value)
                other = UpdateBridge(settings=self.settings, services=self.services, download_directory=self.root)
                try:
                    self.assertEqual(other.snapshot["automaticChecksEnabled"], expected)
                finally:
                    other.close()

        def test_automatic_check_is_silent_but_update_indicator_opens_details(self):
            self.bridge.scheduleAutomaticCheck(0)
            wait(lambda: self.check.call_count == 1 and not self.bridge.snapshot["busy"])
            self.assertFalse(self.bridge.snapshot["dialogOpen"])
            self.assertTrue(self.bridge.snapshot["indicatorVisible"])
            self.assertEqual(self.status, [])
            self.download.assert_not_called()
            self.open_url.assert_not_called()
            self.bridge.invoke("show-update")
            self.assertTrue(self.bridge.snapshot["dialogOpen"])
            self.assertTrue(self.bridge.snapshot["canDownload"])
            self.assertIn("New native", self.bridge.snapshot["releaseNotes"])
            self.bridge.invoke("release-notes")
            self.assertEqual(self.open_url.call_args.args[0].toString(), release_url)

        def test_manual_up_to_date_and_errors_automatic_error_silent(self):
            self.check.return_value = CheckResult("up_to_date", (1, 0, 0), message="You are running the latest version.")
            self.bridge.invoke("check")
            self.idle()
            self.assertEqual(self.bridge.snapshot["stage"], "up_to_date")
            self.assertTrue(self.bridge.snapshot["dialogOpen"])
            self.assertFalse(self.bridge.snapshot["indicatorVisible"])
            self.bridge.invoke("later")
            self.check.return_value = CheckResult("error", (1, 0, 0), message="Offline.\nPrivate diagnostic.")
            self.bridge.check(automatic=True)
            self.idle()
            self.assertEqual(self.bridge.snapshot["message"], "Offline.")
            self.assertFalse(self.bridge.snapshot["dialogOpen"])
            self.bridge.invoke("check")
            self.idle()
            self.assertTrue(self.bridge.snapshot["dialogOpen"])
            self.assertEqual(self.bridge.snapshot["error"], "Offline.")

        def test_verified_download_requires_install_and_final_confirmation(self):
            self.downloaded()
            self.assertEqual(self.bridge.snapshot["progress"], 100)
            self.open_url.assert_not_called()
            self.checksum.assert_not_called()
            self.bridge.invoke("confirm-install")
            self.open_url.assert_not_called()
            self.bridge.invoke("install")
            self.idle()
            self.assertTrue(self.bridge.snapshot["installConfirmationRequired"])
            self.open_url.assert_not_called()
            self.assertEqual(self.checksum.call_count, 1)
            self.bridge.invoke("confirm-install")
            self.idle()
            self.assertEqual(self.checksum.call_count, 2)
            self.assertEqual(self.open_url.call_count, 1)
            self.assertTrue(self.open_url.call_args.args[0].isLocalFile())
            self.assertEqual(self.bridge.snapshot["stage"], "launched")
            self.assertFalse(self.bridge.snapshot["dialogOpen"])
            self.bridge.invoke("confirm-install")
            self.assertEqual(self.open_url.call_count, 1)

        def test_changed_installer_cannot_launch_before_or_after_confirmation(self):
            self.downloaded()
            path = Path(self.bridge.snapshot["installerPath"])
            path.write_bytes(b"changed before Install now")
            self.bridge.invoke("install")
            self.idle()
            self.assertEqual(self.bridge.snapshot["stage"], "error")
            self.assertIn("checksum no longer matches", self.bridge.snapshot["message"])
            self.open_url.assert_not_called()
            self.downloaded()
            self.bridge.invoke("install")
            self.idle()
            self.assertTrue(self.bridge.snapshot["installConfirmationRequired"])
            path.write_bytes(b"changed while confirmation was visible")
            self.bridge.invoke("confirm-install")
            self.idle()
            self.assertEqual(self.bridge.snapshot["stage"], "error")
            self.open_url.assert_not_called()

        def test_missing_checksum_download_failure_and_launch_failure(self):
            self.download.return_value = None
            self.download.side_effect = lambda *args, **kwargs: DownloadResult("error", message="Checksum missing.")
            self.bridge.invoke("check")
            self.idle()
            self.bridge.invoke("download")
            self.idle()
            self.assertEqual(self.bridge.snapshot["error"], "Checksum missing.")
            self.assertFalse(self.bridge.snapshot["canInstall"])
            self.open_url.assert_not_called()
            self.download.side_effect = self.fake_download
            self.downloaded()
            self.bridge.invoke("install")
            self.idle()
            self.open_url.return_value = False
            self.bridge.invoke("confirm-install")
            self.idle()
            self.assertEqual(self.bridge.snapshot["stage"], "error")
            self.assertIn("could not be opened", self.bridge.snapshot["message"])

        def test_later_and_close_while_verifying_do_not_launch(self):
            self.downloaded()
            self.bridge.invoke("install")
            self.idle()
            self.bridge.invoke("later")
            self.assertFalse(self.bridge.snapshot["installConfirmationRequired"])
            self.bridge.invoke("confirm-install")
            self.open_url.assert_not_called()
            entered, release = threading.Event(), threading.Event()
            def blocked(path):
                entered.set()
                release.wait(4)
                return sha256_file(path)
            self.checksum.side_effect = blocked
            self.bridge.invoke("install")
            self.assertTrue(entered.wait(4))
            self.bridge.invoke("later")
            release.set()
            self.idle()
            self.assertFalse(self.bridge.snapshot["dialogOpen"])
            self.assertFalse(self.bridge.snapshot["installConfirmationRequired"])
            self.open_url.assert_not_called()

        def test_about_keeps_pending_update_and_works_during_auto_check(self):
            self.bridge.check(automatic=True)
            self.idle()
            self.bridge.invoke("about")
            self.assertEqual(self.bridge.snapshot["stage"], "about")
            self.assertIn("Version 1.0.0", self.bridge.snapshot["message"])
            self.assertFalse(self.bridge.snapshot["canDownload"])
            self.bridge.invoke("later")
            self.assertEqual(self.bridge.snapshot["stage"], "update_available")
            self.bridge.invoke("show-update")
            self.assertTrue(self.bridge.snapshot["canDownload"])

        def test_manual_check_promotes_running_auto_request_and_close_ignores_late_completion(self):
            entered, release, completed = threading.Event(), threading.Event(), threading.Event()
            def delayed(current):
                entered.set()
                release.wait(4)
                completed.set()
                return available
            self.check.side_effect = delayed
            self.bridge.check(automatic=True)
            self.assertTrue(entered.wait(4))
            self.bridge.invoke("check")
            self.assertTrue(self.bridge.snapshot["dialogOpen"])
            release.set()
            self.idle()
            self.assertEqual(self.check.call_count, 1)
            self.assertTrue(self.bridge.snapshot["dialogOpen"])
            self.assertTrue(self.bridge.snapshot["canDownload"])
            entered.clear()
            release.clear()
            completed.clear()
            self.bridge.check(automatic=True)
            self.assertTrue(entered.wait(4))
            changes = []
            self.bridge.snapshotChanged.connect(lambda: changes.append(True))
            self.bridge.close()
            release.set()
            wait(completed.is_set)
            self.assertEqual(changes, [])
            self.open_url.assert_not_called()

        def test_real_core_parser_and_verified_download_with_fake_network(self):
            raw_release = make_release("v2.0.0", assets=((filename, prefix + filename), (SUMS_FILENAME, prefix + SUMS_FILENAME)), html_url=release_url)
            opener = FakeOpener({RELEASES_URL: json.dumps([raw_release]), prefix + filename: self.body,
                prefix + SUMS_FILENAME: self.digest + "  " + filename})
            services = UpdateServices(check=lambda current: check_for_update(current, opener=opener),
                download=lambda installer, sums, name, destination, progress: download_installer(installer, sums, name, destination, opener=opener, progress=progress),
                checksum=self.checksum, open_url=self.open_url)
            self.bridge.services = services
            self.downloaded()
            self.assertEqual(len(opener.requests), 3)
            self.assertEqual(Path(self.bridge.snapshot["installerPath"]).read_bytes(), self.body)
            self.assertFalse(list((self.root / "downloads").glob("*.part")))
            self.open_url.assert_not_called()

        def test_native_qml_dialog_download_and_cancel_lifecycle(self):
            engine = QQmlApplicationEngine()
            engine.rootContext().setContextProperty("updateBackend", self.bridge)
            engine.rootContext().setContextProperty("updateColors", quick_colors("light"))
            directory = Path(__file__).resolve().parents[1] / "ui_qt/quick/qml"
            source = b'''import QtQuick
import QtQuick.Controls.Basic as Basic
Basic.ApplicationWindow {
    width: 860; height: 640; visible: true
    UpdateDialog { updateBridge: updateBackend; colors: updateColors }
}'''
            engine.loadData(source, QUrl.fromLocalFile(str(directory / "UpdateHarness.qml")))
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            self.addCleanup(lambda: (window.setVisible(False), engine.deleteLater(), app.processEvents()))
            self.bridge.invoke("about")
            dialog = window.findChild(QObject, "quick_update_dialog")
            wait(lambda: dialog.property("visible"))
            self.assertFalse(window.grabWindow().isNull())
            self.bridge.invoke("later")
            wait(lambda: not dialog.property("visible"))
            self.bridge.invoke("check")
            self.idle()
            wait(lambda: dialog.property("visible"))
            button = find(window.contentItem(), "quick_update_download")
            self.assertIsNotNone(button)
            self.assertTrue(QMetaObject.invokeMethod(button, "clicked", Qt.ConnectionType.DirectConnection))
            self.idle()
            self.assertEqual(self.bridge.snapshot["stage"], "downloaded")
            self.bridge.invoke("later")
            wait(lambda: not dialog.property("visible"))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    qInstallMessageHandler(old_handler)
    print("QUICK_UPDATE_TESTS=" + str(result.testsRun))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
