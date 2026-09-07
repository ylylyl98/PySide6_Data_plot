import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox

from ui_qt.controllers_compare import CompareController


class _CompareOwner:
    def __init__(self):
        self.pl_available_files = [
            "sample_PL.csv",
            "sample_REF.csv",
            "sample_unknown.csv",
            "saved.dat",
        ]
        self.available_files = list(self.pl_available_files)
        self.cmp_source_filter_combo = QComboBox()
        self.cmp_source_filter_combo.addItem("PL raw sources", "pl")
        self.cmp_source_filter_combo.addItem("All raw data", "all")
        self.cmp_channel_combos = {key: QComboBox() for key in ("KK", "KKp", "KpK", "KpKp")}


class CompareSourceWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.owner = _CompareOwner()
        self.controller = CompareController(self.owner)

    def test_default_candidates_are_pl_only_and_exclude_dat(self):
        self.assertEqual(self.controller._cmp_assign_candidate_files(), ["sample_PL.csv"])

    def test_all_raw_data_includes_ref_and_unknown_but_excludes_dat(self):
        self.owner.cmp_source_filter_combo.setCurrentIndex(1)
        self.assertEqual(
            self.controller._cmp_assign_candidate_files(),
            ["sample_PL.csv", "sample_REF.csv", "sample_unknown.csv"],
        )

    def test_manual_assignment_survives_filter_change_with_exact_path(self):
        self.owner.cmp_source_filter_combo.setCurrentIndex(1)
        self.controller._cmp_set_channel_combo_items()
        combo = self.owner.cmp_channel_combos["KK"]
        combo.setCurrentText("sample_REF.csv")
        self.owner.cmp_source_filter_combo.setCurrentIndex(0)
        self.controller._cmp_set_channel_combo_items()
        self.assertEqual(combo.currentText(), "sample_REF.csv")
        self.assertIn("outside", combo.itemData(combo.findText("sample_REF.csv"), Qt.ToolTipRole))


if __name__ == "__main__":
    unittest.main()
