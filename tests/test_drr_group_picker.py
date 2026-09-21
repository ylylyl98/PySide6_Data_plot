import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from PySide6.QtWidgets import QApplication
from ui_qt.drr_group_picker import DrrGroupPicker,group_columns


class GroupPickerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_filter_preserves_checked_identity_and_selects_only_visible(self):
        paths=['YZ365_720nm_TG−1.087BG=18_avg1_DR_R.metadata.json','YZ365_760nm_TG−1.087BG=0_avg3_DR_R.metadata.json']
        dialog=DrrGroupPicker(paths)
        try:
            dialog.search.setText('720 TG-1.087BG=18');dialog.check_visible(True)
            self.assertEqual(dialog.selected_paths(),paths[:1])
            dialog.search.setText('760');dialog.check_visible(True)
            self.assertEqual(dialog.selected_paths(),paths)
            dialog.clear_checks();self.assertEqual(dialog.selected_paths(),[])
        finally:dialog.close()

    def test_condition_remains_whole_and_version_distinguishable(self):
        condition,measurement,processing,name=group_columns('YZ365_p5n2_1T_720nm_TG−1.087BG=18_avg3_DR_R_External_v2.metadata.json')
        self.assertEqual(condition,'TG−1.087BG=18')
        self.assertEqual(measurement,'YZ365_p5n2_1T_720nm')
        self.assertIn('v2',processing)
