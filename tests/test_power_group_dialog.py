import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget, QDialog
from PySide6.QtWidgets import QCheckBox, QComboBox
from unittest.mock import Mock, patch
from types import SimpleNamespace

from core import data_io
from ui_qt.power_group_dialog import PowerGroupDialog
from ui_qt.controllers_power import PowerController


class _Controller:
    def __init__(self, folder):
        self._owner = QWidget()
        self.current_folder = folder
        self._power_measurement_group_key = ""
        self._power_picker_status_filter = "All"
        self._power_include_legacy = False


def _table(folder, name, channel):
    path = Path(folder, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Power_uW,1.4,1.5,stage_pos\n1,2,3,0\n2,3,4,1\n", encoding="utf-8")


class PowerGroupDialogTests(unittest.TestCase):
    def test_separate_single_and_comparison_pickers(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('sample_KK_deg24.csv', 'sample_KKp_deg69.csv', 'waiting_deg24.csv', 'plain.csv'):
                _table(folder, name, '')
            controller = _Controller(folder)
            single = PowerGroupDialog(controller, mode='single')
            compare = PowerGroupDialog(controller, mode='compare')
            try:
                self.assertEqual(single.source_list.count(), 4)
                self.assertTrue(all(len(g.sources) == 1 for g in single._catalog_groups))
                self.assertEqual(single.selection()['action'], 'Single intensity')
                self.assertTrue(single.swap_button.isHidden())
                self.assertTrue(single.action_combo.isHidden())
                self.assertEqual(compare.source_list.count(), 1)
                compare.incomplete_check.setChecked(True)
                self.assertEqual(compare.source_list.count(), 2)
                for row, group in enumerate(compare._groups):
                    compare.source_list.setCurrentRow(row)
                    self.assertEqual(compare.selection()['action'], 'Compare intensity')
                    self.assertEqual(compare.ok_button.isEnabled(), len(group.sources) == 2)
                    if len(group.sources) == 1:
                        self.assertIn('Waiting for partner', compare.source_list.item(row).text())
                self.assertTrue(compare.action_combo.isHidden())
            finally:
                single.close(); compare.close(); controller._owner.close()

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_group_list_and_role_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            _table(folder, "sample_1uW_KK.csv", "KK")
            _table(folder, "sample_1uW_KKp.csv", "KKp")
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                self.assertEqual(dialog.source_list.count(), 1)
                self.assertEqual(dialog.action_combo.itemText(0), "Single intensity")
                self.assertEqual(dialog.action_combo.itemText(1), "Compare intensity")
                self.assertEqual(dialog.pair_combo.itemText(0), "Pair by Stage")
                dialog.source_list.setCurrentRow(0)
                self.assertFalse(dialog.assignment_box.isVisible())
                dialog.details_toggle.click()
                dialog.action_combo.setCurrentText("Compare intensity")
                self.assertTrue(dialog.kk_combo.isEnabled())
                self.assertTrue(dialog.kkp_combo.isEnabled())
            finally:
                dialog.close(); controller._owner.close()

    def test_quick_swap_remembers_angles_on_accept_and_cancel_keeps_saved_choice(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('sample_KK_deg24.csv', 'sample_KKp_deg69.csv'):
                _table(folder, name, '')
            manifest = Path(folder, '.power-selection.json')
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                self.assertTrue(dialog.swap_button.isEnabled())
                self.assertFalse(dialog.details_toggle.isChecked())
                dialog.swap_button.click()
                self.assertEqual(dialog.kk_combo.currentData(), 'csv::sample_KKp_deg69.csv')
                self.assertIn('KK: 69°', dialog.assignment_summary.text())
                self.assertIn('KKp: 24°', dialog.assignment_summary.text())
                self.assertFalse(manifest.exists())
                dialog._accept_checked()
                self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            finally:
                dialog.close(); controller._owner.close()
            saved = manifest.read_bytes()
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                self.assertEqual(dialog.kk_combo.currentData(), 'csv::sample_KKp_deg69.csv')
                self.assertEqual(dialog.kkp_combo.currentData(), 'csv::sample_KK_deg24.csv')
                dialog.swap_button.click()
                dialog.reject()
                self.assertEqual(manifest.read_bytes(), saved)
            finally:
                dialog.close(); controller._owner.close()

    def test_legacy_degree_pair_opens_without_manual_assignments(self):
        from PySide6.QtWidgets import QDoubleSpinBox
        with tempfile.TemporaryDirectory() as folder:
            for angle in (24, 69):
                _table(folder, f'yz212_deg{angle}_doping=-1_002.csv', '')
            controller = _Controller(folder)
            for name, value in (('cmp_in_k_angle_spin', 0), ('cmp_in_kp_angle_spin', 45),
                                ('cmp_out_k_angle_spin', 0), ('cmp_out_kp_angle_spin', 45)):
                spin = QDoubleSpinBox(controller._owner)
                spin.setValue(value)
                setattr(controller, name, spin)
            dialog = PowerGroupDialog(controller)
            try:
                self.assertEqual(dialog.source_list.count(), 1)
                self.assertEqual(dialog.ok_button.text(), 'Open comparison')
                self.assertTrue(dialog.ok_button.isEnabled())
                with patch.object(dialog, '_catalog', side_effect=AssertionError('Accept must not rescan the experiment')):
                    dialog._accept_checked()
                self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
                self.assertFalse(dialog.details_toggle.isChecked())
                self.assertEqual(dialog.selection()['KK'], 'csv::yz212_deg24_doping=-1_002.csv')
            finally:
                dialog.close(); controller._owner.close()

    def test_recent_measurements_limit_and_show_older(self):
        with tempfile.TemporaryDirectory() as folder:
            _table(folder, "sample_1uW_KK.csv", "KK")
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                base = dict(label="x", sources=("csv::x.csv",), mapping={}, duplicates={}, issues=(), status="New", context="x", power_min=1., power_max=2., power_count=2)
                dialog._catalog_groups = [SimpleNamespace(**base, key=f"x{i}", modified=i) for i in range(25)]
                dialog._selected_group_key = ""
                dialog._render_groups()
                self.assertEqual(dialog.source_list.count(), 20)
                self.assertFalse(dialog.show_older_button.isHidden())
                dialog.show_older_button.click()
                self.assertEqual(dialog.source_list.count(), 25)
            finally:
                dialog.close(); controller._owner.close()

    def test_ambiguous_resolve_expands_without_accepting(self):
        with tempfile.TemporaryDirectory() as folder:
            _table(folder, "sample_1uW_KK.csv", "KK")
            controller = _Controller(folder); dialog = PowerGroupDialog(controller)
            try:
                key = "csv::sample_1uW_KK.csv"
                group = SimpleNamespace(key="amb", label="amb", context="amb", sources=(key,), mapping={}, duplicates={"KK": (key,)}, issues=(), status="New", power_min=1., power_max=2., power_count=2)
                dialog._catalog_groups = [group]; dialog._sources = {key: dialog._sources[key]}; dialog._selected_group_key = "amb"; dialog._render_groups(); dialog.refresh = lambda: None
                dialog._accept_checked()
                self.assertTrue(dialog.details_toggle.isChecked())
                self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
            finally: dialog.close(); controller._owner.close()

    def test_manual_pairing_saves_on_accept_and_cancel_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            _table(folder, "sample_1uW_KK.csv", "KK"); _table(folder, "sample_1uW_KKp.csv", "KKp")
            controller = _Controller(folder); dialog = PowerGroupDialog(controller)
            try:
                dialog.source_list.setCurrentRow(0); dialog.details_toggle.setChecked(True)
                with patch("core.power_selection_store.save_power_selection") as save:
                    dialog._manual_edits.add(dialog._selected_group_key); dialog._accept_checked()
                    save.assert_called_once()
            finally: dialog.close(); controller._owner.close()
            controller = _Controller(folder); dialog = PowerGroupDialog(controller)
            try:
                dialog.details_toggle.setChecked(True); dialog._manual_edits.add(dialog._selected_group_key)
                with patch("core.power_selection_store.save_power_selection") as save:
                    dialog.reject(); save.assert_not_called()
            finally: dialog.close(); controller._owner.close()

    def test_real_manual_pairing_roundtrip_and_cancel(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('sample_KK.csv', 'sample_KK_Rot11deg.csv', 'sample_KKp.csv'):
                _table(folder, name, '')
            manifest = Path(folder, '.power-selection.json')
            chosen = 'csv::sample_KK_Rot11deg.csv'
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                self.assertEqual(dialog.source_list.count(), 1)
                self.assertEqual(dialog.ok_button.text(), 'Resolve pairing')
                dialog._accept_checked()
                self.assertTrue(dialog.details_toggle.isChecked())
                dialog.kk_combo.setCurrentIndex(dialog.kk_combo.findData(chosen))
                dialog.reject()
                self.assertFalse(manifest.exists())
            finally:
                dialog.close(); controller._owner.close()
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                dialog._accept_checked()
                dialog.kk_combo.setCurrentIndex(dialog.kk_combo.findData(chosen))
                dialog.details_toggle.setChecked(False)
                dialog._accept_checked()
                self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
                self.assertTrue(manifest.exists())
            finally:
                dialog.close(); controller._owner.close()
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                self.assertEqual(dialog.kk_combo.currentData(), chosen)
                self.assertEqual(dialog.ok_button.text(), 'Open comparison')
                dialog._accept_checked()
                self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
                self.assertFalse(dialog.details_toggle.isChecked())
            finally:
                dialog.close(); controller._owner.close()

    def test_filter_keeps_group_draft_and_cancel_does_not_change_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            _table(folder, "sample_1uW_KK.csv", "KK")
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                dialog.source_list.setCurrentRow(0)
                original = dialog.single_combo.currentData()
                dialog.filter_edit.setText("sample")
                dialog.refresh()
                self.assertEqual(dialog.single_combo.currentData(), original)
                dialog.legacy_check.setChecked(True)
                self.assertFalse(controller._power_include_legacy)
                dialog.reject()
                self.assertFalse(controller._power_include_legacy)
            finally:
                dialog.close(); controller._owner.close()

    def test_duplicate_role_requires_explicit_assignment(self):
        with tempfile.TemporaryDirectory() as folder:
            _table(folder, "sample_1uW_KK.csv", "KK")
            _table(folder, "sample_2uW_KK.csv", "KK")
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                dialog.source_list.setCurrentRow(0)
                dialog.action_combo.setCurrentText("Compare intensity")
                self.assertFalse(dialog.ok_button.isEnabled())
            finally:
                dialog.close(); controller._owner.close()

    def test_controller_applies_single_without_compare_and_loads_once(self):
        owner = QWidget(); owner.current_folder = ""; owner._power_include_legacy = False
        owner._invalidate_export_move_sources = Mock()
        owner._power_measurement_group_key = ""; owner._power_picker_status_filter = "All"
        owner.power_group_combo = QComboBox(); owner.power_kk_group_combo = QComboBox(); owner.power_kkp_group_combo = QComboBox()
        for combo in (owner.power_group_combo, owner.power_kk_group_combo, owner.power_kkp_group_combo):
            combo.addItem("—", ""); combo.addItem("source", "csv::source.csv")
        owner.power_compare_chk = QCheckBox(); owner._power_refresh_groups = Mock()
        owner._on_power_source_assignment_changed = Mock(); owner._power_set_view_mode = Mock(); owner._start_load = Mock()
        ctl = PowerController(owner)
        class Dialog(QDialog):
            def __init__(self, *_args, **_kwargs): super().__init__(owner)
            def open(self): self.accept()
            def selection(self): return {"group": "ctx", "action": "Single intensity", "single": "csv::source.csv", "KK": "", "KKp": "", "pairing": "Pair by Stage", "status": "All", "legacy": False}
        with patch("ui_qt.power_group_dialog.PowerGroupDialog", Dialog), patch.object(PowerController, "_power_refresh_groups", Mock()), patch.object(PowerController, "_on_power_source_assignment_changed", Mock()), patch.object(PowerController, "_power_set_view_mode", Mock()):
            ctl._power_choose_measurement_group()
            self.app.processEvents()
        self.assertFalse(owner.power_compare_chk.isChecked())
        owner._start_load.assert_called_once_with("Power Dependent")
        owner.close()

    def test_picker_lifetime_is_owned_until_deferred_close_handler(self):
        with tempfile.TemporaryDirectory() as folder:
            _table(folder, 'sample_KK.csv', '')
            owner = QWidget()
            owner.current_folder = folder
            controller = PowerController(owner)
            try:
                with patch.object(PowerController, '_apply_power_measurement_group') as apply:
                    controller._power_choose_measurement_group()
                    dialog = owner._power_group_dialog
                    self.assertTrue(dialog.isModal())
                    controller._power_choose_measurement_group()
                    self.assertIs(owner._power_group_dialog, dialog)
                    dialog.accept()
                    apply.assert_not_called()
                    self.assertIs(owner._power_group_dialog, dialog)
                    self.app.processEvents()
                    apply.assert_called_once_with(dialog)
                    self.assertIsNone(owner._power_group_dialog)
                    controller._power_choose_measurement_group()
                    owner._power_group_dialog.reject()
                    self.app.processEvents()
                    self.assertIsNone(owner._power_group_dialog)
                    self.assertEqual(apply.call_count, 1)
            finally:
                owner.close()

    def test_controller_applies_compare_and_loads_once(self):
        owner = QWidget(); owner.current_folder = ""; owner._power_include_legacy = False
        owner._invalidate_export_move_sources = Mock()
        owner._power_measurement_group_key = ""; owner._power_picker_status_filter = "All"
        owner.power_group_combo = QComboBox(); owner.power_kk_group_combo = QComboBox(); owner.power_kkp_group_combo = QComboBox()
        for combo in (owner.power_group_combo, owner.power_kk_group_combo, owner.power_kkp_group_combo):
            combo.addItem("—", ""); combo.addItem("source", "csv::source.csv"); combo.addItem("other", "csv::other.csv")
        owner.power_compare_chk = QCheckBox(); owner._power_refresh_groups = Mock(); owner._on_power_source_assignment_changed = Mock(); owner._power_set_view_mode = Mock(); owner._start_load = Mock()
        ctl = PowerController(owner)
        class Dialog(QDialog):
            def __init__(self, *_args, **_kwargs): super().__init__(owner)
            def open(self): self.accept()
            def selection(self): return {"group": "ctx", "action": "Compare intensity", "single": "", "KK": "csv::source.csv", "KKp": "csv::other.csv", "pairing": "Pair by Stage", "status": "All", "legacy": False}
        with patch("ui_qt.power_group_dialog.PowerGroupDialog", Dialog), patch.object(PowerController, "_power_refresh_groups", Mock()), patch.object(PowerController, "_on_power_source_assignment_changed", Mock()), patch.object(PowerController, "_power_set_view_mode", Mock()):
            ctl._power_choose_measurement_group()
            self.app.processEvents()
        self.assertTrue(owner.power_compare_chk.isChecked()); owner._start_load.assert_called_once_with("Power Dependent")
        owner.close()

    def test_vp_validation_and_deleted_source_cannot_accept(self):
        with tempfile.TemporaryDirectory() as folder:
            _table(folder, 'sample_KK.csv', 'KK')
            _table(folder, 'sample_KKp.csv', 'KKp')
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                dialog.source_list.setCurrentRow(0)
                dialog.action_combo.setCurrentText('VP')
                self.assertTrue(dialog.ok_button.isEnabled(), dialog.details_label.text())
                original = dialog.kkp_combo.currentData()
                Path(folder, 'sample_KKp.csv').unlink()
                dialog.refresh()
                self.assertEqual(dialog.kkp_combo.currentData(), original)
                self.assertIn('Missing', dialog.kkp_combo.currentText())
                self.assertFalse(dialog.ok_button.isEnabled())
                dialog._accept_checked()
                self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
            finally:
                dialog.close(); controller._owner.close()

    def test_power_vp_rejects_disjoint_ranges_but_comparison_is_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            _table(folder, 'sample_KK.csv', 'KK')
            Path(folder, 'sample_KKp.csv').write_text('Power_uW,1.4,1.5\n10,2,3\n20,3,4\n')
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                dialog.source_list.setCurrentRow(0)
                dialog.action_combo.setCurrentText('Compare intensity')
                self.assertTrue(dialog.ok_button.isEnabled())
                dialog.action_combo.setCurrentText('VP')
                dialog.pair_combo.setCurrentText('Power Interpolation')
                self.assertFalse(dialog.ok_button.isEnabled())
                self.assertIn('overlapping power', dialog.details_label.text())
            finally:
                dialog.close(); controller._owner.close()

    def test_hidden_group_and_explicit_clear_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('a_KK.csv', 'a_KKp.csv', 'b_KK.csv'):
                _table(folder, name, '')
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            try:
                dialog.source_list.setCurrentRow(0)
                selected = dialog._selected_group_key
                # Modification-time order need not put the a pair first.
                for row, group in enumerate(dialog._groups):
                    if 'csv::a_KK.csv' in group.sources:
                        dialog.source_list.setCurrentRow(row)
                        selected = dialog._selected_group_key
                        break
                dialog.kk_combo.setCurrentIndex(0)
                dialog.filter_edit.setText('b_KK')
                dialog.refresh()
                self.assertIsNone(dialog.source_list.currentItem())
                self.assertEqual(dialog._selected_group_key, selected)
                dialog.filter_edit.clear(); dialog.refresh()
                self.assertEqual(dialog._selected_group_key, selected)
                self.assertEqual(dialog.kk_combo.currentData(), '')
            finally:
                dialog.close(); controller._owner.close()

    def test_reopen_restores_roles_action_and_pairing(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('sample_1uW_KK.csv', 'sample_1uW_KKp.csv', 'sample_KKp.csv'):
                _table(folder, name, '')
            controller = _Controller(folder)
            controller.power_group_combo = QComboBox()
            controller.power_kk_group_combo = QComboBox()
            controller.power_kkp_group_combo = QComboBox()
            for combo, name in ((controller.power_group_combo, 'sample_1uW_KK.csv'),
                                (controller.power_kk_group_combo, 'sample_1uW_KK.csv'),
                                (controller.power_kkp_group_combo, 'sample_KKp.csv')):
                combo.addItem(name, 'csv::' + name)
            controller.power_compare_chk = QCheckBox(); controller.power_compare_chk.setChecked(True)
            controller._power_view = lambda: 'VP'
            controller.power_pair_mode_combo = QComboBox(); controller.power_pair_mode_combo.addItem('Power Interpolation')
            dialog = PowerGroupDialog(controller)
            try:
                self.assertEqual(dialog.action_combo.currentText(), 'VP')
                self.assertEqual(dialog.pair_combo.currentText(), 'Power Interpolation')
                self.assertEqual(dialog.kk_combo.currentData(), 'csv::sample_1uW_KK.csv')
                self.assertTrue(dialog.ok_button.isEnabled(), dialog.details_label.text())
            finally:
                dialog.close(); controller._owner.close()


if __name__ == "__main__":
    unittest.main()
