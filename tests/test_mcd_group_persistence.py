import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from PySide6.QtWidgets import QApplication
from tests.test_curie_weiss_ui import records
from core.mcd_extract import organize_mcd_series
from ui_qt.mcd_organizer_window import McdOrganizerWindow


class GroupPersistenceTests(unittest.TestCase):
    def test_concurrent_exclusions_merge_record_changes(self):
        from ui_qt.mcd_organizer_window import _merge_selection_changes
        self.assertEqual(_merge_selection_changes(['a', 'b', 'c'], ['a', 'c'], ['b', 'c']), ['c'])

    def test_two_windows_preserve_independent_edits_in_same_temperature_series(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            r = records(root)
            windows = [McdOrganizerWindow(root, auto_scan=False) for _ in range(2)]
            try:
                for w in windows:
                    w.compare_combo.setCurrentIndex(w.compare_combo.findData('Temperature'))
                    w._on_scan_result(w._scan_generation, (str(root.resolve()), r))
                    w._write_condition_selections()
                for w, record, group in zip(windows, r, ('Group 7', 'Group 8')):
                    w._energy_group_overrides[record.record_id] = group
                    w._energy_groups_changed()
                windows[0]._write_condition_selections()
                windows[0]._load_saved_condition_selections()
                restored = windows[0]._energy_groups(windows[0]._current_series().records)
                self.assertEqual([restored[x.record_id] for x in r[:2]], ['Group 7', 'Group 8'])
            finally:
                for w in windows:
                    w.close()

    def test_failed_save_keeps_previous_file_and_reports_error(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            r = records(root)
            w = McdOrganizerWindow(root, auto_scan=False)
            try:
                w.compare_combo.setCurrentIndex(w.compare_combo.findData('Temperature'))
                w._on_scan_result(w._scan_generation, (str(root.resolve()), r))
                w._write_condition_selections()
                before = w._selection_settings_path().read_bytes()
                w._energy_group_overrides[r[0].record_id] = 'Group 7'
                with patch('ui_qt.mcd_organizer_window.QSaveFile') as save:
                    save.return_value.open.return_value = False
                    save.return_value.errorString.return_value = 'Permission denied'
                    self.assertFalse(w._write_condition_selections())
                self.assertEqual(w._selection_settings_path().read_bytes(), before)
                self.assertIn('save failed', w.statusBar().currentMessage())
                self.assertTrue(w._write_condition_selections())
                self.assertEqual(w.statusBar().currentMessage(), '')
            finally:
                w.close()

    def test_stale_window_cannot_overwrite_newer_manual_group_on_close(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            r = records(root)
            first = McdOrganizerWindow(root, auto_scan=False)
            second = McdOrganizerWindow(root, auto_scan=False)
            try:
                for w in (first, second):
                    w._load_saved_condition_selections()
                    w.compare_combo.setCurrentIndex(w.compare_combo.findData('Temperature'))
                    w._on_scan_result(w._scan_generation, (str(root.resolve()), r))
                    w._write_condition_selections()
                first._energy_group_overrides[r[0].record_id] = 'Group 7'
                first._energy_groups_changed()
                second.close()
                first._load_saved_condition_selections()
                self.assertEqual(first._energy_groups(first._current_series().records)[r[0].record_id], 'Group 7')
            finally:
                first.close(); second.close()

    def test_fixed_count_limits_assignments_and_new_points_without_merging(self):
        with tempfile.TemporaryDirectory() as folder:
            r=records(Path(folder))
            w=McdOrganizerWindow(folder,auto_scan=False)
            try:
                w.records=r
                w.compare_combo.setCurrentIndex(w.compare_combo.findData('Temperature'))
                w._regroup()
                w._energy_group_overrides[r[0].record_id]='Group 7'
                w.fixed_group_count_chk.setChecked(True)
                w.group_count_spin.setValue(3)
                s=w._current_series()
                groups=w._energy_groups(s.records)
                self.assertTrue(groups[r[0].record_id].startswith('Unassigned '))
                self.assertEqual(w._assignable_group_names(s),['Group 1','Group 2','Group 3'])
                from unittest.mock import patch
                from PySide6.QtCore import Qt
                item=next(w.condition_list.item(i) for i in range(w.condition_list.count())
                          if w.condition_list.item(i).data(Qt.UserRole)==r[0].record_id)
                item.setSelected(True)
                with patch('ui_qt.mcd_organizer_window.QInputDialog.getItem',return_value=('Group 3',True)) as choose:
                    w._assign_selected_group()
                    self.assertFalse(choose.call_args.args[-1])
                self.assertEqual(w._energy_groups(s.records)[r[0].record_id],'Group 3')
                new=replace(r[0],record_id='new',center_ev=1.5,
                            acquisition_conditions={**r[0].acquisition_conditions,'T':(30.,30.)})
                w.records=r+[new];w._regroup()
                self.assertTrue(w._energy_groups(w._current_series().records)['new'].startswith('Unassigned '))
                w._write_condition_selections();w._load_saved_condition_selections()
                self.assertEqual(w._group_scope(w._current_series())['group_count'],3)
            finally:
                w.close();w.deleteLater()

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_refresh_adds_new_points_without_renumbering_and_restores_saved_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            r = records(root)
            w = McdOrganizerWindow(root, auto_scan=False)
            try:
                w.compare_combo.setCurrentIndex(w.compare_combo.findData('Temperature'))
                w.records = r
                w._regroup()
                old = w._energy_groups(w._current_series().records)
                w._energy_group_overrides[r[0].record_id] = 'Custom'
                w._energy_groups_changed()
                old = w._energy_groups(w._current_series().records)
                new = replace(r[0], record_id='new', center_ev=1.5,
                              acquisition_conditions={**r[0].acquisition_conditions, 'T': (30.,30.)})
                w.records = r+[new]
                w._regroup()
                groups = w._energy_groups(w._current_series().records)
                self.assertEqual({key:groups[key] for key in old}, old)
                self.assertEqual(groups['new'], 'Group 2')
                w._write_condition_selections()
                w._load_saved_condition_selections()
                self.assertEqual(w._energy_groups(w._current_series().records), groups)
            finally:
                w.close()
                w.deleteLater()

    def test_same_record_can_have_independent_groups_in_other_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            r = records(Path(folder))
            more = [replace(x, record_id=x.record_id+'f', acquisition_conditions={
                **x.acquisition_conditions, 'E-field': (1.,1.)}) for x in r]
            w = McdOrganizerWindow(folder, auto_scan=False)
            try:
                w.records = r+more
                w.compare_combo.setCurrentIndex(w.compare_combo.findData('Temperature'))
                w._regroup()
                w._energy_group_overrides[r[0].record_id] = 'Temperature custom'
                w._energy_groups_changed()
                w.compare_combo.setCurrentIndex(w.compare_combo.findData('E-field'))
                w._regroup()
                efield = next(s for s in w.series_groups if r[0] in s.records)
                self.assertNotEqual(w._energy_groups(efield.records)[r[0].record_id], 'Temperature custom')
                w.compare_combo.setCurrentIndex(w.compare_combo.findData('Temperature'))
                w._regroup()
                temp = next(s for s in w.series_groups if r[0] in s.records)
                self.assertEqual(w._energy_groups(temp.records)[r[0].record_id], 'Temperature custom')
            finally:
                w.close()
                w.deleteLater()

    def test_scan_and_reopen_keep_exclusions_names_and_separate_fixed_conditions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            r = records(root)
            extra = [replace(x, record_id=x.record_id+'other', source_file=x.source_file+'other', acquisition_conditions={
                **x.acquisition_conditions, 'E-field': (1.,1.)}) for x in r]
            w = McdOrganizerWindow(root, auto_scan=False)
            try:
                w._load_saved_condition_selections()
                w.compare_combo.setCurrentIndex(w.compare_combo.findData('Temperature'))
                w._on_scan_result(w._scan_generation, (str(root.resolve()), r+extra))
                s = w._current_series()
                w._energy_group_overrides[r[0].record_id] = 'Group 7'
                w._selected_record_ids[s.series_id].discard(r[1].record_id)
                before = w._energy_groups(s.records)
                other = next(s for s in w.series_groups if extra[0] in s.records)
                self.assertEqual(w._energy_groups(other.records)[extra[0].record_id], 'Group 1')
                w._write_condition_selections()
                new = replace(r[0], record_id='added', source_file='added.csv', acquisition_conditions={
                    **r[0].acquisition_conditions, 'T': (30.,30.)})
                w._on_scan_result(w._scan_generation, (str(root.resolve()), r+extra+[new]))
                s = next(s for s in w.series_groups if r[0] in s.records)
                after = w._energy_groups(s.records)
                self.assertEqual({k:after[k] for k in before}, before)
                self.assertNotIn(r[1].record_id, w._selected_record_ids[s.series_id])
                self.assertIn('added', w._selected_record_ids[s.series_id])
                w._write_condition_selections()
                reopened = McdOrganizerWindow(root, auto_scan=False)
                try:
                    reopened.compare_combo.setCurrentIndex(reopened.compare_combo.findData('Temperature'))
                    reopened._on_scan_result(reopened._scan_generation, (str(root.resolve()), r+extra+[new]))
                    s = next(s for s in reopened.series_groups if r[0] in s.records)
                    self.assertEqual(reopened._energy_groups(s.records), after)
                    self.assertNotIn(r[1].record_id, reopened._selected_record_ids[s.series_id])
                finally:
                    reopened.close()
                    reopened.deleteLater()
            finally:
                w.close()
                w.deleteLater()

    def test_close_but_distinct_condition_partitions_do_not_share_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            r = records(Path(folder))
            r = [replace(x, acquisition_conditions={**x.acquisition_conditions,
                         'Doping': (d,d)}) for x,d in zip(r, [0.,.009,.011,.011,.011,.011])]
            w = McdOrganizerWindow(folder, auto_scan=False)
            try:
                w.records = r
                w.compare_combo.setCurrentIndex(w.compare_combo.findData('Temperature'))
                w._regroup()
                self.assertEqual(len(w.series_groups), 2)
                a,b = [w._group_scope(s) for s in w.series_groups]
                self.assertIsNot(a,b)
                a['manual']['0'] = 'Custom'
                b['manual'].clear()
                self.assertEqual(a['manual']['0'], 'Custom')
            finally:
                w.close()
                w.deleteLater()
