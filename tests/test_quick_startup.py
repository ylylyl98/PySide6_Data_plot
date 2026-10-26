"""Exercise the real launcher before and after its deferred first plot."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
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
