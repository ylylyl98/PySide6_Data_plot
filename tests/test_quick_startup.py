"""Exercise the real launcher before and after its deferred first plot."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from textwrap import dedent
import unittest


class QuickStartupTests(unittest.TestCase):
    def check_launcher(self, *, demo, entry="run_quick.py"):
        with TemporaryDirectory(prefix="quick-launch-check-") as directory:
            root = Path(directory)
            report = root / "report.json"
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLCONFIGDIR=str(root / "mpl"), LOCALAPPDATA=str(root), XDG_CONFIG_HOME=str(root))
            args = [sys.executable, entry, "--check", "--report", str(report)]
            if demo:
                args.append("--demo")
            process = subprocess.run(args, cwd=Path(__file__).resolve().parents[1], env=env,
                capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            evidence = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "ok", evidence)
        self.assertFalse(evidence["widgets_imported"])
        self.assertTrue(evidence["settings_isolated"])
        self.assertEqual(evidence["scientific_imports_at_shell_frame"], [])
        self.assertGreater(evidence["timing_ms"]["first_shell_frame_ms"], 0)
        if demo:
            self.assertGreater(evidence["timing_ms"]["first_plot_frame_ms"],
                evidence["timing_ms"]["first_shell_frame_ms"])

    def test_empty_shell_presents_without_scientific_imports(self):
        self.check_launcher(demo=False)

    def test_first_plot_loads_deferred_canvas_and_reaches_presented_frame(self):
        self.check_launcher(demo=True)

    def test_production_entry_defaults_to_native_shell_with_isolated_check_settings(self):
        self.check_launcher(demo=False, entry="run_qt.py")

    def test_production_entry_native_plot_uses_no_widgets(self):
        self.check_launcher(demo=True, entry="run_qt.py")

    def test_deckcraft_companion_listens_before_showing_a_plot(self):
        probe = dedent("""
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import Mock, patch

            from PySide6.QtCore import QTimer
            from PySide6.QtGui import QGuiApplication, QWindow
            from run_quick import main
            from ui_qt.quick.application_bridge import QuickApplicationBridge

            observed = {}
            role = Mock()
            origin = object()

            def start_service(bridge, *, companion=False):
                app = QGuiApplication.instance()
                window = next(window for window in app.topLevelWindows()
                              if window.objectName() == "quick_application_window")
                observed["companion"] = companion
                observed["minimized"] = window.visibility() == QWindow.Visibility.Minimized
                observed["isolated_settings"] = Path(bridge.settings.fileName()).name == "preview.ini"
                windows = Mock()
                def activate(source, hwnd):
                    observed["origin"] = source
                    observed["matching_window"] = hwnd == int(window.winId())
                    window.showNormal()
                    return True
                windows.activate_plot_window.side_effect = activate
                bridge._deckcraft_service = SimpleNamespace(windows=windows, close=lambda: None)
                def requested():
                    bridge._activate_figure_editor(SimpleNamespace(window=origin))
                    observed["shown"] = window.visibility() == QWindow.Visibility.Windowed
                    app.quit()
                QTimer.singleShot(0, requested)

            with patch("ui_qt.quick.deckcraft_instance.acquire_deckcraft_role", return_value=role) as gate, \
                 patch.object(QuickApplicationBridge, "_deckcraft_start", start_service), \
                 patch.object(QuickApplicationBridge, "startServices") as normal_services:
                assert main(["--deckcraft-companion"]) == 0
            assert observed == dict(companion=True, minimized=True, isolated_settings=True,
                                    origin=origin, matching_window=True, shown=True), observed
            gate.assert_called_once_with(companion=True, production=False)
            normal_services.assert_not_called()
            role.close.assert_called_once_with()
        """)
        with TemporaryDirectory(prefix="quick-companion-check-") as directory:
            environment = dict(
                os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                LOCALAPPDATA=directory, XDG_CONFIG_HOME=directory,
            )
            process = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=40,
            )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
