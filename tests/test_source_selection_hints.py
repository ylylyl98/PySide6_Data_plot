import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from ui_qt.main_window import MainWindow


class SourceSelectionHintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_shg_hint_describes_required_columns_and_root_availability(self):
        window = MainWindow()
        try:
            self.assertIn("measured-angle", window.shg_source_hint.text())
            self.assertIn("numeric wavelength", window.shg_source_hint.text())
            self.assertIn("All root CSVs", window.shg_source_hint.text())
        finally:
            window.close()
            window.deleteLater()


if __name__ == "__main__":
    unittest.main()
