import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from openpyxl import load_workbook
from core.mcd_extract import discover_processed_mcd, export_mcd_extract, SLOPE_METRICS
from ui_qt.mcd_organizer_window import McdOrganizerWindow
from tests import test_mcd_extract as extract_tests


class SlopeVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_independent_visibility_and_export_all_even_when_hidden(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = extract_tests.McdExtractTests()
            for name, f in [('a', 0), ('b', 10)]:
                fixture._write_result(root, name, energy=1.65, doping=6.3, efield=f)
            records = [replace(r, high_field_slopes={
                'high_negative': {'B increasing': .5, 'B decreasing': 1.},
                'high_positive': {'B increasing': 2., 'B decreasing': 4.},
            }) for r in discover_processed_mcd(root)]
            w = McdOrganizerWindow(root, auto_scan=False)
            try:
                w._on_scan_result(0, (str(root.resolve()), records))
                # The normal delayed preview initialization owns the canvases.
                self.app.processEvents()
                for check in w.slope_checks.values():
                    check.setChecked(True)
                w._update_preview()
                self.assertEqual(len(w._slope_lines), 6)
                self.assertEqual(w.details_table.columnCount(), 11)
                self.assertEqual(len(w.slope_figure.axes),2)
                axes = w.slope_figure.axes
                self.assertTrue(axes[0].get_shared_x_axes().joined(axes[0],axes[-1]))
                self.assertIn('saved integration center', axes[0].get_title(loc='left'))
                from matplotlib.backend_bases import MouseEvent
                w.slope_canvas.draw()
                px,py = axes[0].transData.transform((0,1.65))
                event = MouseEvent('motion_notify_event',w.slope_canvas,px,py)
                w._energy_plot_hover(event)
                self.assertIsNotNone(w._energy_hover)
                self.assertIn('Width: 5 meV',w._energy_hover.get_text())
                self.assertIn('Near zero - Positive high field',w._energy_hover.get_text())
                w.condition_list.setCurrentRow(1)
                click = MouseEvent('button_press_event',w.slope_canvas,px,py,button=1)
                w._energy_plot_click(click)
                self.assertEqual(w._focused_record_id,records[0].record_id)
                self.assertGreaterEqual(len(w._energy_focus_artists),4)
                values = next(artist for key,artist in w._slope_lines.items() if key[:2]==('low_minus_positive', 'B increasing')).get_offsets()[:,1]
                self.assertTrue(all(v == 0 for v in values))
                w.slope_checks['low_minus_negative'].setChecked(False)
                w._update_preview()
                self.assertEqual(len(w._slope_lines), 4)
                for check in w.slope_checks.values():
                    check.setChecked(False)
                w._update_preview()
                self.assertEqual(len(w._slope_lines), 0)
                self.assertEqual(len(w.slope_figure.axes),1)
                from unittest.mock import patch
                from PySide6.QtWidgets import QInputDialog
                for row in range(w.condition_list.count()):
                    w.condition_list.item(row).setSelected(True)
                before = {item.data(256) for item in w.condition_list.selectedItems()}
                w.palette_combo.setCurrentIndex((w.palette_combo.currentIndex()+1)%w.palette_combo.count())
                self.assertEqual(before,{item.data(256) for item in w.condition_list.selectedItems()})
                from matplotlib.colors import to_hex
                shades = w._preview_energy_colors(w._current_series())
                for row in range(w.condition_list.count()):
                    item = w.condition_list.item(row)
                    self.assertEqual(item.data(257),to_hex(shades[item.data(256)]))
                with patch.object(QInputDialog, 'getItem', return_value=('Joined', True)):
                    w._assign_selected_group()
                self.assertEqual(set(w._energy_groups(records).values()), {'Joined'})
                # A rendering failure must neither empty the list nor lose the assignment.
                count_before = w.condition_list.count()
                with patch.object(w, '_preview_energy_colors', side_effect=RuntimeError('test color failure')):
                    w._energy_groups_changed()
                self.assertEqual(w.condition_list.count(), count_before)
                self.assertFalse(w._list_refreshing)
                saved = json.loads(w._selection_settings_path().read_text(encoding='utf-8'))
                self.assertTrue(all(saved['energy_groups'][r.record_id]=='Joined' for r in records))
                self.assertIn('test color failure',w.selection_summary.text())
                w._energy_groups_changed()
                self.assertEqual(w.condition_list.count(),count_before)

                self.assertEqual(set(w._energy_group_overrides), {r.record_id for r in records})
                w._reset_series_groups()
                self.assertFalse(w._energy_group_overrides)
                w._energy_group_overrides[records[0].record_id] = 'Manual A'
                w._write_condition_selections()
                w._energy_group_overrides.clear()
                w._load_saved_condition_selections()
                self.assertEqual(w._energy_groups(records)[records[0].record_id],'Manual A')
                from unittest.mock import patch
                from PySide6.QtWidgets import QDialog, QPushButton
                def reset_dialog(dialog):
                    next(button for button in dialog.findChildren(QPushButton)
                         if button.text()=='Reset to automatic groups').click()
                    return QDialog.Accepted
                with patch.object(QDialog,'exec',reset_dialog):
                    w._edit_energy_groups()
                self.assertEqual(w._energy_group_overrides,{})
                with patch.object(QDialog,'exec',return_value=QDialog.Accepted):
                    w._edit_energy_groups()
                self.assertEqual(set(w._energy_group_overrides),{r.record_id for r in records})
                w._write_condition_selections()
                w._energy_group_overrides.clear()
                w._load_saved_condition_selections()
                self.assertEqual(set(w._energy_group_overrides),{r.record_id for r in records})

                w._energy_group_overrides[records[0].record_id]='Manual A'
                series = w._current_series()
                w._update_preview()
                w.mcd_group_combo.setCurrentIndex(w.mcd_group_combo.findData('Manual A'))
                w._update_preview()
                self.assertEqual(set(w._plot_artists), {records[0].record_id})
                plotted_ids = {r.record_id for members,metric,branch in w._energy_point_artists.values() for r in members}
                self.assertEqual(plotted_ids,{r.record_id for r in records})
                full = w._export_energy_groups([series])
                self.assertEqual(full[records[0].record_id],'Manual A')
                # UI hide state is not an export option. Even an Inc-only curve
                # export must retain all six slope values in the summary.
                paths = export_mcd_extract(records, root/'exports', branches=('B increasing',),energy_groups=w._energy_groups(records))
                self.assertEqual(len([k for k in paths if k.startswith('slope_')]),4)
                self.assertTrue(all(p.exists() for k,p in paths.items() if k.startswith('slope_')))
                self.assertTrue(paths['energy_png'].exists())
                self.assertEqual(len([k for k in paths if k.startswith('mcd_group_')]),2)
                book = load_workbook(paths['summary_xlsx'], data_only=True)
                rows = list(book['Slopes'].values)
                self.assertEqual(len(rows[0]), 11)
                self.assertEqual(list(rows[1][2:8]), [2, 3, 1.5, 2, 0, -1])
                self.assertEqual(rows[1][8],5)
                self.assertEqual(rows[1][9],'Manual A')
                self.assertEqual(rows[1][10],records[0].record_id)
                book.close()
                settings = json.loads(paths['settings'].read_text())
                self.assertEqual(settings['slope_metrics'], list(SLOPE_METRICS))
                self.assertIn('not fitted peak',settings['energy_semantics'])
            finally:
                w.close()
                w.deleteLater()

    def test_missing_high_fit_exports_na(self):
        from core.mcd_extract import compact_slope_table
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            extract_tests.McdExtractTests()._write_result(root, 'a', energy=1.65, doping=6.3, efield=0)
            records = discover_processed_mcd(root)
            table = compact_slope_table(records, 'E-field', tuple(SLOPE_METRICS))
            self.assertEqual(list(table.iloc[0, 4:8]), ['N/A']*4)
