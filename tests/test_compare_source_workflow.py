import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QPlainTextEdit

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
        self.cmp_display_preset_combo = QComboBox()
        self.cmp_display_preset_combo.addItem("KK + KKp")
        self.cmp_show_checks = {}
        self.cmp_assignment_summary = QPlainTextEdit()
        self.cmp_group_power_tolerance_percent = 5.0
        self.cmp_in_k_angle_spin = QDoubleSpinBox()
        self.cmp_in_kp_angle_spin = QDoubleSpinBox()
        self.cmp_out_k_angle_spin = QDoubleSpinBox()
        self.cmp_out_kp_angle_spin = QDoubleSpinBox()
        self.cmp_angle_tolerance_spin = QDoubleSpinBox()
        self.cmp_in_k_angle_spin.setValue(0.0)
        self.cmp_in_kp_angle_spin.setValue(45.0)
        self.cmp_out_k_angle_spin.setValue(0.0)
        self.cmp_out_kp_angle_spin.setValue(45.0)
        self.cmp_angle_tolerance_spin.setValue(15.0)
        self.loaded = None
        self._append_log = lambda *_args: None
        self._invalidate_export_move_sources = lambda: None


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

    def test_unlabeled_file_remains_manually_selectable(self):
        self.controller._cmp_set_channel_combo_items()
        combo = self.owner.cmp_channel_combos["KK"]
        self.assertGreaterEqual(combo.findText("sample_PL.csv"), 0)
        combo.setCurrentText("sample_PL.csv")
        self.assertEqual(combo.currentText(), "sample_PL.csv")

    def test_swap_kk_and_kkp_preserves_other_channels(self):
        for combo in self.owner.cmp_channel_combos.values():
            combo.addItems(["", "kk.csv", "kkp.csv", "kpk.csv", "kpkp.csv"])
        self.owner.cmp_channel_combos["KK"].setCurrentText("kk.csv")
        self.owner.cmp_channel_combos["KKp"].setCurrentText("kkp.csv")
        self.owner.cmp_channel_combos["KpK"].setCurrentText("kpk.csv")
        self.owner.cmp_channel_combos["KpKp"].setCurrentText("kpkp.csv")
        self.owner._cmp_update_assignment_summary = lambda: None
        self.controller._cmp_swap_kk_channels()
        self.assertEqual(self.owner.cmp_channel_combos["KK"].currentText(), "kkp.csv")
        self.assertEqual(self.owner.cmp_channel_combos["KKp"].currentText(), "kk.csv")
        self.assertEqual(self.owner.cmp_channel_combos["KpK"].currentText(), "kpk.csv")
        self.assertEqual(self.owner.cmp_channel_combos["KpKp"].currentText(), "kpkp.csv")

    def test_clear_group_clears_hidden_channel_state(self):
        for combo in self.owner.cmp_channel_combos.values():
            combo.addItems(["", "assigned.csv"])
            combo.setCurrentText("assigned.csv")
        self.owner.cmp_selected_group_key = "group-1"
        self.owner.cmp_selected_group_label = "Group 1"
        self.owner.cmp_selected_group_sources = ("assigned.csv",)
        self.owner._cmp_update_assignment_summary = lambda: None
        self.controller._cmp_clear_group()
        self.assertFalse(self.owner.cmp_selected_group_key)
        self.assertTrue(all(not combo.currentText() for combo in self.owner.cmp_channel_combos.values()))

    def _angle_owner(self, files):
        owner = _CompareOwner()
        owner.pl_available_files = list(files)
        owner.available_files = list(files)
        owner.cmp_source_filter_combo.setCurrentIndex(1)
        return owner, CompareController(owner)

    def test_auto_detect_infers_selected_group_without_mixing_stage_groups(self):
        files = [
            "YZ212_1077uW_Rot224deg_Stage200.csv",
            "YZ212_1078uW_Rot269deg_Stage200.csv",
            "YZ212_1077uW_Rot224deg_Stage3600.csv",
            "YZ212_1078uW_Rot269deg_Stage3600.csv",
        ]
        owner, controller = self._angle_owner(files)
        controller._cmp_set_channel_combo_items()
        controller._cmp_auto_assign_channels()

        self.assertEqual(owner.cmp_out_k_angle_spin.value(), 24.0)
        self.assertEqual(owner.cmp_out_kp_angle_spin.value(), 69.0)
        self.assertIn("stage200", owner.cmp_selected_group_key)
        self.assertEqual(
            owner.cmp_channel_combos["KK"].currentText(),
            "YZ212_1077uW_Rot224deg_Stage200.csv",
        )
        self.assertEqual(
            owner.cmp_channel_combos["KKp"].currentText(),
            "YZ212_1078uW_Rot269deg_Stage200.csv",
        )

    def test_manual_swap_survives_refresh_after_inferred_references(self):
        files = [
            "YZ212_1077uW_Rot224deg_Stage200.csv",
            "YZ212_1078uW_Rot269deg_Stage200.csv",
        ]
        owner, controller = self._angle_owner(files)
        controller._cmp_set_channel_combo_items()
        controller._cmp_auto_assign_channels()
        controller._cmp_swap_kk_channels()
        swapped = {
            key: combo.currentText() for key, combo in owner.cmp_channel_combos.items()
        }

        controller._cmp_auto_assign_channels()

        self.assertEqual(
            {key: combo.currentText() for key, combo in owner.cmp_channel_combos.items()},
            swapped,
        )

    def test_multi_cluster_group_does_not_guess_references(self):
        files = [
            "YZ212_1077uW_Rot224deg_Stage200.csv",
            "YZ212_1078uW_Rot250deg_Stage200.csv",
            "YZ212_1079uW_Rot269deg_Stage200.csv",
        ]
        owner, controller = self._angle_owner(files)
        controller._cmp_set_channel_combo_items()
        controller._cmp_auto_assign_channels()

        self.assertEqual(owner.cmp_out_k_angle_spin.value(), 0.0)
        self.assertEqual(owner.cmp_out_kp_angle_spin.value(), 45.0)
        self.assertFalse(owner.cmp_channel_combos["KK"].currentText())
        self.assertFalse(owner.cmp_channel_combos["KpK"].currentText())

    def test_angle_edit_reclassifies_without_inference(self):
        files = [
            "YZ212_1077uW_Rot224deg_Stage200.csv",
            "YZ212_1078uW_Rot269deg_Stage200.csv",
        ]
        owner, controller = self._angle_owner(files)
        controller._cmp_set_channel_combo_items()
        controller._cmp_auto_assign_channels()

        owner.cmp_out_k_angle_spin.setValue(10.0)
        controller._on_cmp_angle_reference_changed()

        self.assertEqual(owner.cmp_out_k_angle_spin.value(), 10.0)
        self.assertEqual(owner.cmp_out_kp_angle_spin.value(), 69.0)
        self.assertEqual(owner.cmp_channel_combos["KKp"].currentText(), files[1])


if __name__ == "__main__":
    unittest.main()
