"""Public standalone names and frozen-compatible flags must start native UI."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class NativeEntryTests(unittest.TestCase):
    def test_public_analysis_scripts_and_flags_do_not_import_widgets(self):
        code = '''
import importlib.abc, runpy, sys
class NoWidgets(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'PySide6.QtWidgets', 'PySide6.QtQuickWidgets', 'ui_qt.main_window'}:
            raise AssertionError('Public entry imported Widgets: '+fullname)
sys.meta_path.insert(0, NoWidgets())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        for script, flag in (("run_peak_analysis.py", "--peak-analysis"),
                             ("run_drr_analysis.py", "--drr-analysis"),
                             ("run_mcd_organizer.py", "--mcd-organizer")):
            for arguments in ([script, "--check"], ["run_qt.py", flag, "--check"]):
                with self.subTest(arguments=arguments), TemporaryDirectory(prefix="native-public-entry-") as temporary:
                    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                        LOCALAPPDATA=temporary, MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary,
                        PYTHONIOENCODING="utf-8")
                    result = subprocess.run([sys.executable, "-c", code, *arguments],
                        cwd=Path(__file__).resolve().parents[1], env=environment,
                        capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
