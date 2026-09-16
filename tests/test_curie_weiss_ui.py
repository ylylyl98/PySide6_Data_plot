import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from core.mcd_extract import ProcessedMcdRecord, McdSeries
from ui_qt.curie_weiss_panel import CurieWeissPanel
from ui_qt.mcd_organizer_window import McdOrganizerWindow


def records(root):
    result = []
    for i, t in enumerate([3., 4., 6., 9., 15., 25.]):
        path = root / f'{i}.json'
        trace = root / f'{i}.csv'
        slope = 2 / (t + 4)
        with trace.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['B_increasing_T', 'corrected_signed_mean_increasing',
                             'B_decreasing_T', 'corrected_signed_mean_decreasing'])
            writer.writerows([[-.2, -.2*slope, .2, .2*slope], [0, 0, 0, 0],
                              [.2, .2*slope, -.2, -.2*slope]])
        result.append(ProcessedMcdRecord(str(i), path, trace, f'{i}.csv', 'test', '',
                      1.64, 5., 'mean', .2,
                      {'T': (t,t), 'Doping': (6.3,6.3), 'E-field': (0.,0.)},
                      {'T': 'measurement'}, slope, slope, t, t+.01))
    return result


class CurieWeissUiTests(unittest.TestCase):
    def test_inverse_default_exports_errors_and_can_switch_to_slope_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            r = records(Path(folder))
            for record in r:
                record.settings_path.write_text(json.dumps({'slopes': [dict(
                    branch='B increasing', region='low', status='ok',
                    center_ev=record.center_ev, width_mev=record.width_mev,
                    slope=record.increasing_slope_per_t, slope_se=.001)]}))
            panel = CurieWeissPanel()
            try:
                panel.set_series('s', r, {x.record_id:'G' for x in r}, ('B increasing',))
                panel.refit()
                result = panel.results['B increasing']
                self.assertEqual(result.get('fit_space'), 'inverse_slope')
                self.assertAlmostEqual(result['theta_k'], -4)
                self.assertEqual(len(result['inverse_slope_se']), 6)
                self.assertAlmostEqual(result['inverse_slope_se'][0], .01225)
                self.assertEqual(len(panel.figure.axes), 1)
                self.assertTrue(panel.figure.axes[0].containers)
                from types import SimpleNamespace
                from ui_qt.matplotlib_theme import apply_display_theme, display_theme_from_resolved, LIGHT_PUBLICATION_THEME
                annotation = panel.figure.axes[0].texts[0]
                dark = display_theme_from_resolved(SimpleNamespace(name='dark'))
                apply_display_theme(panel.figure, dark)
                self.assertEqual(annotation.get_color(), dark.text)
                apply_display_theme(panel.figure, LIGHT_PUBLICATION_THEME)
                self.assertEqual(annotation.get_color(), LIGHT_PUBLICATION_THEME.text)
                target = panel.export_to(Path(folder))
                payload = json.loads((target/'fit.json').read_text())
                self.assertEqual(payload['fit_method'], 'inverse')
                self.assertIn('inverse_slope_se', payload['points'][0])
                panel.method_combo.setCurrentIndex(panel.method_combo.findData('slope'))
                panel.refit()
                self.assertIn('slope space', panel.results['B increasing']['method'])
                wrong_window = replace(r[0], center_ev=1.65)
                self.assertIsNone(panel._saved_slope_se(wrong_window, 'B increasing', r[0].increasing_slope_per_t))
            finally:
                panel.close()
                panel.deleteLater()

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_group_range_background_and_export_are_live(self):
        with tempfile.TemporaryDirectory() as folder:
            r = records(Path(folder))
            panel = CurieWeissPanel()
            try:
                panel.set_series('series', r, {x.record_id: 'Group 1' for x in r},
                                 ('B increasing', 'B decreasing'))
                panel.refit()
                self.assertAlmostEqual(panel.results['B increasing']['theta_k'], -4., places=4)
                panel.t_min.setValue(6)
                panel.refit()
                self.assertEqual(panel.results['B increasing']['n'], 4)
                destination = panel.export_to(Path(folder))
                saved = json.loads((destination / 'fit.json').read_text())
                self.assertEqual(saved['results']['B increasing']['n'], 4)
                with (destination/'points.csv').open(encoding='utf-8-sig') as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 8)
                self.assertTrue((destination/'fit.png').is_file())
                panel.t_min.setValue(20)
                panel.refit()
                self.assertFalse(panel.results)
                self.assertFalse(panel.export_btn.isEnabled())
            finally:
                panel.close()
                panel.deleteLater()

    def test_organizer_selection_and_non_temperature_state_clear_results(self):
        with tempfile.TemporaryDirectory() as folder:
            window = McdOrganizerWindow(Path(folder), auto_scan=False)
            try:
                r = records(Path(folder))
                window.records = r
                window.compare_combo.setCurrentIndex(window.compare_combo.findData('Temperature'))
                window._update_preview()
                panel = window.curie_weiss_panel
                panel.refit()
                self.assertEqual(panel.results['B increasing']['n'], 6)
                series = window._current_series()
                window._selected_record_ids[series.series_id] = {x.record_id for x in r[1:]}
                window._apply_inclusion_visibility()
                panel.refit()
                self.assertEqual(panel.results['B increasing']['n'], 5)
                window.compare_combo.setCurrentIndex(window.compare_combo.findData('E-field'))
                window._update_preview()
                self.assertFalse(panel.results)
                self.assertFalse(panel.export_btn.isEnabled())
            finally:
                window.close()
                window.deleteLater()

    def test_out_of_range_width_does_not_block_and_empty_selection_clears(self):
        with tempfile.TemporaryDirectory() as folder:
            r = records(Path(folder))
            r[0] = replace(r[0], width_mev=10.)
            panel = CurieWeissPanel()
            try:
                panel.set_series('s', r, {x.record_id:'G' for x in r}, ('B increasing',))
                panel.t_min.setValue(4.)
                panel.refit()
                self.assertEqual(panel.results['B increasing']['n'], 5)
                panel.set_series('s', [], {}, ('B increasing',))
                panel.refit()
                self.assertFalse(panel.results)
                self.assertNotIn('θCW =', panel.summary.toPlainText())
            finally:
                panel.close()
                panel.deleteLater()

    def test_energy_groups_and_measured_temperature_do_not_mix(self):
        with tempfile.TemporaryDirectory() as folder:
            r = records(Path(folder))
            other = [replace(x, record_id=x.record_id+'b', center_ev=1.58,
                             increasing_slope_per_t=3/(x.temperature_setpoint_k-1)) for x in r]
            groups = {x.record_id:'A' for x in r} | {x.record_id:'B' for x in other}
            panel = CurieWeissPanel()
            try:
                panel.set_series('s', r+other, groups, ('B increasing',))
                panel.refit()
                self.assertAlmostEqual(panel.results['B increasing']['theta_k'], -4, places=4)
                panel.group_combo.setCurrentIndex(panel.group_combo.findData('B'))
                panel.refit()
                self.assertAlmostEqual(panel.results['B increasing']['theta_k'], 1, places=4)
                panel.temperature_combo.setCurrentIndex(panel.temperature_combo.findData('measured'))
                panel.refit()
                self.assertAlmostEqual(panel.results['B increasing']['theta_k'], 1.01, places=4)
                panel.set_series('s', [replace(x, temperature_measured_k=None) for x in other], groups,
                                 ('B increasing',))
                panel.refit()
                self.assertFalse(panel.results)
            finally:
                panel.close()
                panel.deleteLater()

    def test_refit_field_range_updates_points_diagnostics_and_export(self):
        from tests.test_mcd_slope_refit import write_curved_trace
        with tempfile.TemporaryDirectory() as folder:
            r = records(Path(folder))
            for record in r:
                write_curved_trace(record)
                # Give each temperature a different amplitude while keeping curvature.
                text = record.trace_path.read_text()
                rows = list(csv.reader(text.splitlines()))
                factor = 1/(record.temperature_setpoint_k + 4)
                for row in rows[1:]:
                    for i in (1,2,3,5,6,7):
                        row[i] = str(float(row[i])*factor)
                with record.trace_path.open('w', newline='') as handle:
                    csv.writer(handle).writerows(rows)
            panel = CurieWeissPanel()
            try:
                panel.set_series('s', r, {x.record_id:'G' for x in r}, ('B increasing','B decreasing'))
                panel.field_refit_chk.setChecked(True)
                panel.refit()
                first = next(p for p in panel.points if p['record_id']=='0' and p['branch']=='B increasing')
                self.assertAlmostEqual(first['slope'], 2/7)
                panel.b_min.setValue(-.3)
                panel.b_max.setValue(.3)
                # Export must flush the pending refit and use the new bounds.
                target = panel.export_to(Path(folder))
                payload = json.loads((target/'fit.json').read_text())
                self.assertEqual(payload['field_range_t'], [-.3,.3])
                first = next(p for p in payload['points'] if p['record_id']=='0' and p['branch']=='B increasing')
                self.assertAlmostEqual(first['slope'], 23/49)
                self.assertEqual(first['field_point_count'], 7)
                self.assertTrue((target/'field_points.csv').is_file())
                self.assertTrue((target/'field_fit_preview.png').is_file())
                panel.b_min.setValue(-.05)
                panel.b_max.setValue(.05)
                panel.refit()
                self.assertFalse(panel.results)
                self.assertFalse(panel.export_btn.isEnabled())
                self.assertTrue(panel.field_fits)
                self.assertTrue(all(f['slope'] is None for f in panel.field_fits))
                panel.field_refit_chk.setChecked(False)
                panel.refit()
                self.assertAlmostEqual(panel.results['B increasing']['theta_k'], -4, places=4)
                self.assertFalse(panel.field_fits)
            finally:
                panel.close()
                panel.deleteLater()


if __name__ == '__main__':
    unittest.main()
