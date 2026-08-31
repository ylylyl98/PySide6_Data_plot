from __future__ import annotations

import unittest
from pathlib import Path


class WindowsPowerPointPackagingTests(unittest.TestCase):
    def test_windows_build_bundles_and_smoke_checks_powerpoint_bridge(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = (root / "packaging" / "PySide6_Data_Plot.spec").read_text(encoding="utf-8")
        build_script = (root / "build_windows.bat").read_text(encoding="utf-8")
        self.assertIn('get_pywin32_module_file_attribute("pythoncom")', spec)
        self.assertIn('get_pywin32_module_file_attribute("pywintypes")', spec)
        self.assertIn('"win32com.client"', spec)
        self.assertIn("--check-powerpoint-integration", build_script)

    def test_source_launcher_repairs_missing_powerpoint_bridge(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "Data_Plot_App.bat").read_text(encoding="utf-8")
        self.assertIn('import PySide6, pythoncom, win32com.client', launcher)
        self.assertIn('pip install -r requirements.txt', launcher)


if __name__ == "__main__":
    unittest.main()
