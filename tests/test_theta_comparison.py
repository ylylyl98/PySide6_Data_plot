import csv
import tempfile
import unittest
from pathlib import Path
from tests.test_curie_weiss_ui import records
from tests.test_mcd_slope_refit import write_curved_trace
from core.theta_comparison import compare_theta


def entry(root, name, theta, field):
    folder = root/name
    folder.mkdir()
    result = records(folder)
    for record in result:
        write_curved_trace(record)
        rows = list(csv.reader(record.trace_path.read_text().splitlines()))
        for row in rows[1:]:
            for i in (1,2,3,5,6,7):
                row[i] = str(float(row[i])/(record.temperature_setpoint_k-theta))
        with record.trace_path.open('w', newline='') as f:
            csv.writer(f).writerows(rows)
    return dict(series_id=name, group='Group 1', fixed_conditions={'Doping':6.3,'E-field':field}, records=result)


class ThetaComparisonTests(unittest.TestCase):
    def test_hover_contains_cw_quality_for_coincident_points(self):
        from unittest.mock import patch
        from matplotlib.backend_bases import MouseEvent
        from PySide6.QtWidgets import QApplication
        from ui_qt.theta_comparison_dialog import ThetaComparisonDialog, draw_comparison
        app = QApplication.instance() or QApplication([])
        dialog = ThetaComparisonDialog([], ('B increasing',), '.')
        rows = [dict(series_id='a',group=group,doping=6.3,efield=10,
                     fixed_conditions={},branch='B increasing',is_primary=True,halfwidth_t=.2,
                     status='ok',theta_k=-4,ci95_low_k=None,ci95_high_k=None,r_squared=.876,n=6)
                for group in ('Group 1','Group 2')]
        try:
            dialog._point_artists = draw_comparison(dialog.figure, {'rows':rows})
            dialog.canvas.draw()
            x,y = dialog.figure.axes[0].transData.transform((10,-4))
            event = MouseEvent('motion_notify_event',dialog.canvas,x,y)
            with patch('ui_qt.theta_comparison_dialog.QToolTip.showText') as show:
                dialog._hover_point(event)
                text = show.call_args.args[1]
                for value in ('Group 1','Group 2','CW R² (1/s vs T) = 0.876','N = 6','unavailable / unreliable'):
                    self.assertIn(value,text)
            dialog._invalidate()
            self.assertEqual(dialog._point_artists,{})
        finally:
            dialog.close();dialog.deleteLater();app.processEvents()

    def test_reliable_auto_y_uses_visible_ci_bounds_and_falls_back(self):
        from matplotlib.figure import Figure
        from ui_qt.theta_comparison_dialog import draw_comparison
        rows = [dict(series_id=str(i),group=group,doping=6.3,efield=i,
                     fixed_conditions={},branch='B increasing',is_primary=True,halfwidth_t=.2,
                     status='ok',theta_k=theta,ci95_low_k=low,ci95_high_k=high)
                for i,(group,theta,low,high) in enumerate([
                    ('Group 1',-4,-8,2),('Group 1',-200,None,None),
                    ('Group 2',-100,-120,-80)])]
        fig = Figure()
        draw_comparison(fig, {'rows':rows}, visible_groups={'Group 1'}, reliable_y=True)
        low, high = fig.axes[0].get_ylim()
        self.assertLess(low,-8); self.assertGreater(high,2)
        self.assertGreater(low,-20)
        self.assertTrue(any('1 point' in t.get_text() for t in fig.axes[0].texts))
        draw_comparison(fig, {'rows':[rows[1]]}, reliable_y=True)
        self.assertTrue(any('No reliable CI' in t.get_text() for t in fig.axes[0].texts))
        low, high = fig.axes[0].get_ylim()
        self.assertLess(low,-200); self.assertGreater(high,-200)

    def test_branch_fill_ci_cross_and_manual_y_range(self):
        from matplotlib.figure import Figure
        from ui_qt.theta_comparison_dialog import draw_comparison
        rows = [dict(series_id=str(i),group='Group 1',doping=6.3,efield=i,
                     fixed_conditions={},branch=branch,is_primary=True,halfwidth_t=.2,
                     status='ok',theta_k=theta,ci95_low_k=low,ci95_high_k=high)
                for i,(branch,theta,low,high) in enumerate([
                    ('B increasing',-4,None,None),('B decreasing',-8,-10,-6),
                    ('B decreasing',-60,None,None)])]
        for axis in ('efield','sensitivity'):
            fig = Figure()
            draw_comparison(fig,{'rows':rows},axis,y_limits=(-25,5))
            ax = fig.axes[0]
            markers = [c.lines[0] for c in ax.containers]
            self.assertNotEqual(markers[0].get_markerfacecolor(),'none')
            self.assertEqual(markers[1].get_markerfacecolor(),'none')
            self.assertEqual(markers[1].get_marker(),'s')
            self.assertEqual(ax.get_ylim(),(-25,5))
            self.assertTrue(any('1 point' in t.get_text() for t in ax.texts))
            self.assertEqual(sum(t.get_text()=='×' for t in ax.texts),2)
            crosses = [t for t in ax.texts if t.get_text() == '×']
            for cross in crosses:
                self.assertEqual(cross.get_position(), (0, 0))
                self.assertEqual(cross.get_ha(), 'center')
                self.assertEqual(cross.get_va(), 'center')
            self.assertEqual([cross.xy[1] for cross in crosses], [-4, -60])
            self.assertNotIn('hollow: CI',ax.get_title())

    def test_all_groups_fit_and_visibility_preserve_results_and_separate_sensitivity(self):
        import time
        from dataclasses import replace
        from PySide6.QtWidgets import QApplication
        from ui_qt.theta_comparison_dialog import ThetaComparisonDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            entries = [entry(Path(folder), 'a', -4., 0.), entry(Path(folder), 'b', -4., 10.)]
            for e in entries:
                original = e['records']
                extra = [replace(r, record_id=r.record_id+'second') for r in original]
                e.update(label=e['series_id'], records=original+extra,
                         groups={r.record_id:g for records_,g in [(original,'Group 1'),(extra,'Group 2')] for r in records_})
            dialog = ThetaComparisonDialog(entries, ('B increasing',), folder)
            try:
                dialog.all_groups_chk.setChecked(True)
                dialog._run()
                deadline = time.monotonic()+15
                while dialog.worker is not None and time.monotonic()<deadline:
                    app.processEvents(); time.sleep(.01)
                self.assertIsNone(dialog.worker)
                self.assertEqual(sum(r['is_primary'] for r in dialog.result['rows']), 4)
                result = dialog.result
                dialog.axis_combo.setCurrentIndex(dialog.axis_combo.findData('sensitivity'))
                self.assertEqual(len(dialog.figure.axes[0].get_legend_handles_labels()[1]), 4)
                dialog.group_visibility_checks['Group 2'].setChecked(False)
                labels = dialog.figure.axes[0].get_legend_handles_labels()[1]
                self.assertEqual(len(labels), 2)
                self.assertTrue(all('Group 1' in label for label in labels))
                self.assertIs(dialog.result, result)
                self.assertTrue(dialog.save_btn.isEnabled())
                dialog.group_visibility_checks['Group 1'].setChecked(False)
                self.assertEqual(len(dialog.figure.axes[0].containers), 0)
            finally:
                dialog.close(); dialog.deleteLater(); app.processEvents()

    def test_bulk_group_selection_marks_missing_and_invalidates_results(self):
        from PySide6.QtWidgets import QApplication
        from ui_qt.theta_comparison_dialog import ThetaComparisonDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            entries = [entry(Path(folder), 'a', -4., 0.), entry(Path(folder), 'b', -4., 10.)]
            for e in entries:
                e.update(label=e['series_id'], groups={r.record_id:'Group 1' for r in e['records']})
            entries[0]['groups'][entries[0]['records'][0].record_id] = 'Group 2'
            dialog = ThetaComparisonDialog(entries, ('B increasing',), folder)
            try:
                dialog.result = {'old':True}
                dialog.bulk_group_combo.setCurrentIndex(dialog.bulk_group_combo.findData('Group 2'))
                dialog.bulk_group_btn.click()
                self.assertEqual(dialog.group_combos[0].currentData(), 'Group 2')
                self.assertIsNone(dialog.group_combos[1].currentData())
                self.assertIsNone(dialog.result)
                self.assertIn('1', dialog.status.text())
                dialog._run()
                self.assertIsNone(dialog.worker)
                dialog.bulk_group_combo.setCurrentIndex(dialog.bulk_group_combo.findData('Group 1'))
                dialog.bulk_group_btn.click()
                self.assertTrue(all(c.currentData() == 'Group 1' for c in dialog.group_combos))
            finally:
                dialog.close(); dialog.deleteLater(); app.processEvents()

    def test_plot_ignores_condition_roundoff_without_merging_distinct_values(self):
        from matplotlib.figure import Figure
        from ui_qt.theta_comparison_dialog import draw_comparison
        for axis, other in [('efield','doping'),('doping','efield')]:
            rows=[]
            for i,value in enumerate([6.3,6.299999999999998,6.300000000000002,6.31]):
                for branch in ('B increasing','B decreasing'):
                    rows.append(dict(series_id=str(i),is_primary=True,status='ok',
                        **{axis:float(i),other:value},branch=branch,halfwidth_t=.2,
                        fixed_conditions={'Vbias':-1e-16 if i%2 else 0.},
                        theta_k=-4.,ci95_low_k=-5.,ci95_high_k=-3.))
            fig=Figure()
            draw_comparison(fig,{'rows':rows},axis)
            ax=fig.axes[0]
            self.assertEqual(len(ax.get_legend_handles_labels()[1]),4)
            colors=[container.lines[0].get_color() for container in ax.containers]
            self.assertEqual(colors[0],colors[2])
            self.assertEqual(colors[0],colors[4])
            self.assertNotEqual(colors[0],colors[6])

    def test_dialog_background_run_export_and_invalidation(self):
        import time,json
        from PySide6.QtWidgets import QApplication
        from ui_qt.theta_comparison_dialog import ThetaComparisonDialog
        app=QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            entries=[entry(Path(folder),'a',-4.,0.),entry(Path(folder),'b',1.,10.)]
            for e in entries:
                e.update(label=e['series_id'],groups={r.record_id:e['group'] for r in e['records']})
            dialog=ThetaComparisonDialog(entries,('B increasing',),folder)
            try:
                dialog.sensitivity.setChecked(False)
                dialog._run()
                deadline=time.monotonic()+15
                while dialog.worker is not None and time.monotonic()<deadline:
                    app.processEvents();time.sleep(.01)
                self.assertIsNone(dialog.worker)
                self.assertIsNotNone(dialog.result,dialog.status.text())
                self.assertEqual(dialog.table.rowCount(),2)
                target=dialog.export_to(folder)
                self.assertEqual(target.parent.name,'Theta_CW_comparison')
                payload=json.loads((target/'comparison.json').read_text())
                self.assertEqual(len(payload['rows']),2)
                self.assertTrue((target/'theta_efield.png').is_file())
                dialog.halfwidth.setValue(.3)
                self.assertIsNone(dialog.result)
                self.assertFalse(dialog.save_btn.isEnabled())
            finally:
                dialog.close();dialog.deleteLater();app.processEvents()

    def test_batch_recovers_distinct_theta_and_reports_failed_sensitivity_ranges(self):
        with tempfile.TemporaryDirectory() as folder:
            entries = [entry(Path(folder),'a',-4.,0.),entry(Path(folder),'b',1.,10.)]
            out = compare_theta(entries, halfwidth=.2, t_min=3., t_max=25.,
                                branches=('B increasing','B decreasing'), sensitivity_widths=[.05,.3])
            main = [r for r in out['rows'] if r['is_primary']]
            self.assertEqual(len(main),4)
            for row in main:
                self.assertEqual(row['status'],'ok')
                self.assertAlmostEqual(row['theta_k'], -4 if row['series_id']=='a' else 1, places=5)
                self.assertEqual(row['n'],6)
            self.assertTrue(all(r['status']=='failed' for r in out['rows'] if r['halfwidth_t']==.05))
            self.assertEqual(len(out['points']), 72)

    def test_missing_curve_is_not_silently_dropped_from_one_condition(self):
        with tempfile.TemporaryDirectory() as folder:
            e = entry(Path(folder),'a',-4.,0.)
            e['records'][0].trace_path.unlink()
            out = compare_theta([e],halfwidth=.2,t_min=3.,t_max=25.,branches=('B increasing',))
            self.assertEqual(out['rows'][0]['status'],'failed')
            self.assertIsNone(out['rows'][0]['theta_k'])
