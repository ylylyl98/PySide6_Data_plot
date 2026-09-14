import unittest
from PySide6.QtWidgets import QApplication

from ui_qt.mcd_source_summary import McdSourceSummary


class SourceSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_structured_fields_and_legacy_calls(self):
        widget = McdSourceSummary()
        widget.set_source(status="Processed", filename="x" * 200, saved_at="2026-09-12 12:34", tooltip="full path", badge_state="processed")
        self.assertEqual(widget.status_label.text(), "Processed")
        self.assertEqual(widget.filename_label.toolTip(), "full path")
        self.assertEqual(widget.time_label.text(), "2026-09-12 12:34")
        self.assertIn("Processed", widget.text())
        widget.set_status("New")
        self.assertEqual(widget.text(), "New")
        widget.setText("History unknown")
        self.assertEqual(widget.status_label.text(), "History unknown")


if __name__ == "__main__":
    unittest.main()
