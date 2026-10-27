"""Resource inventory and isolated source/frozen smoke contracts."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPECS = ("PySide6_Data_Plot.spec", "PySide6_Data_Plot_macos.spec")


def _spec_values(filename):
    """Inspect real Analysis arguments without requiring PyInstaller installed."""
    tree = ast.parse((ROOT / "packaging" / filename).read_text(encoding="utf-8"))
    namespace = {
        "project_root": ROOT,
        "str": str,
        "fluent_datas": [],
        "matplotlib_datas": [],
        "pptx_datas": [],
        "collect_submodules": lambda _package: [],
    }
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        names = {target.id for target in statement.targets if isinstance(target, ast.Name)}
        if "quick_qml_datas" in names:
            namespace["quick_qml_datas"] = eval(compile(ast.Expression(statement.value), filename, "eval"), namespace)
        if isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and statement.value.func.id == "Analysis":
            keywords = {argument.arg: argument.value for argument in statement.value.keywords}
            return {
                name: eval(compile(ast.Expression(keywords[name]), filename, "eval"), namespace)
                for name in ("datas", "hiddenimports")
            }
    raise AssertionError(f"No Analysis call found in {filename}")


class QuickDisplayPackagingTests(unittest.TestCase):
    def test_both_specs_preserve_entire_local_qml_tree(self):
        qml_root = ROOT / "ui_qt" / "quick" / "qml"
        self.assertTrue((qml_root / "Display.qml").is_file())
        qml_files = sorted(qml_root.rglob("*.qml"))
        self.assertGreaterEqual(len(qml_files), 2, "Smoke must package local QML components too")
        for filename in SPECS:
            with self.subTest(spec=filename):
                inventory = _spec_values(filename)
                matches = [(Path(source), destination) for source, destination in inventory["datas"] if destination == "ui_qt/quick/qml"]
                self.assertEqual(matches, [(qml_root, "ui_qt/quick/qml")])
                # PyInstaller expands a directory data source recursively. The
                # resulting URLs must keep local component imports unchanged.
                bundled = {
                    (Path(destination) / path.relative_to(source)).as_posix()
                    for source, destination in matches
                    for path in source.rglob("*")
                    if path.is_file()
                }
                for path in qml_files:
                    self.assertIn(path.relative_to(ROOT).as_posix(), bundled)

    def test_both_specs_activate_narrow_qt_quick_hooks(self):
        for filename in SPECS:
            with self.subTest(spec=filename):
                inventory = _spec_values(filename)
                self.assertTrue({"PySide6.QtQuickWidgets", "PySide6.QtQml", "PySide6.QtQuick"}.issubset(inventory["hiddenimports"]))
                source = (ROOT / "packaging" / filename).read_text(encoding="utf-8")
                self.assertNotIn('collect_all("PySide6")', source)
                self.assertNotIn('collect_submodules("PySide6")', source)
                self.assertNotIn('collect_data_files("PySide6")', source)

    def test_source_smoke_renders_actual_page_and_keeps_existing_settings(self):
        # This also exercises the command-line dispatch used by the frozen
        # executable. Qt state cannot leak into other in-process test modules.
        code = textwrap.dedent("""
            import os
            from pathlib import Path
            import sys
            from PySide6.QtCore import QSettings
            sentinel_dir = Path(sys.argv[1])
            QSettings.setDefaultFormat(QSettings.IniFormat)
            QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(sentinel_dir))
            settings = QSettings(QSettings.IniFormat, QSettings.UserScope, 'DPTK', 'PySide6_Data_Plot')
            settings.setValue('sentinel/key', 'untouched')
            settings.setValue('data/last_folder', str(sentinel_dir / 'production-do-not-read'))
            settings.sync()
            sentinel_path = Path(settings.fileName())
            before = sentinel_path.read_bytes()
            # Fail closed if a future smoke change misses either module's
            # QSettings constructor. On Windows setDefaultFormat() does not
            # redirect the common two-string overload away from the registry.
            import PySide6.QtCore as qtcore
            class IniOnlySettings(QSettings):
                def __init__(self, *args, **kwargs):
                    explicit_ini = bool(args) and args[0] == QSettings.IniFormat
                    filename_ini = len(args) >= 2 and isinstance(args[0], str) and args[1] == QSettings.IniFormat
                    if not (explicit_ini or filename_ini):
                        raise AssertionError('Smoke attempted implicit/native settings access')
                    super().__init__(*args, **kwargs)
            qtcore.QSettings = IniOnlySettings
            from run_qt import main
            sys.argv = ['run_qt.py', '--check-quick-display']
            result = main()
            assert result == 0, result
            assert QSettings.defaultFormat() == QSettings.IniFormat
            assert sentinel_path.read_bytes() == before
            raise SystemExit(result)
        """)
        with tempfile.TemporaryDirectory(prefix="dptk-quick-smoke-test-") as directory:
            environment = os.environ.copy()
            environment.update({
                "QT_QPA_PLATFORM": "offscreen",
                "QT_QUICK_BACKEND": "software",
                "LOCALAPPDATA": directory,
                "MPLCONFIGDIR": str(Path(directory) / "outer-mpl-cache"),
                "PYTHONPATH": str(ROOT),
            })
            result = subprocess.run(
                [sys.executable, "-c", code, directory],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        reports = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"backend":')]
        self.assertEqual(len(reports), 1, result.stdout)
        report = reports[0]
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["backend"], "quick")
        self.assertTrue(report["settings_isolated"])
        self.assertTrue(report["data_isolated"])
        self.assertTrue(report["qml"].endswith("Display.qml"))
        self.assertGreater(report["frame"]["sampled_colors"], 1)


if __name__ == "__main__":
    unittest.main()
