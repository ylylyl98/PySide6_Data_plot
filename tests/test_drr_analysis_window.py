import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
import time
from dataclasses import replace
import numpy as np
from PySide6.QtWidgets import QApplication
from core.loader import DataCube


class WorkspaceWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        from ui_qt.drr_analysis_window import DrrAnalysisWindow
        from core.drr_analysis_workspace import create_dataset
        self.window=DrrAnalysisWindow()
        x=np.linspace(1,1.1,101);z=np.array([np.exp(-((x-1.05)/.003)**2)]*3)
        self.a=create_dataset(DataCube(x,np.arange(3.),z,'Y','A','DR/R'),'A',{'measurement_files':['a.csv']})
        self.b=create_dataset(DataCube(x,np.arange(3.),z*.7,'Y','B','DR/R'),'B',{'measurement_files':['b.csv']})
        self.window.add_dataset(self.a);self.window.add_dataset(self.b)

    def tearDown(self):self.window.close();self.app.processEvents()

    def test_find_peaks_defaults_to_full_data_and_can_use_display_range(self):
        from unittest.mock import patch
        w=self.window;w.activate_dataset(self.a.key)
        w.analysis.bounds['x_min'].setValue(1.02)
        w.display_bounds['x_min'].setValue(1.03)
        with patch.object(w,'analyze_keys') as run:
            w.find_peaks_keys([self.a.key])
            self.assertAlmostEqual(self.a.settings.x_min,1.)
            run.assert_called_once_with([self.a.key])
        w.detection_scope.setCurrentIndex(1)
        with patch.object(w,'analyze_keys'):
            w.find_peaks_keys([self.a.key])
            self.assertAlmostEqual(self.a.settings.x_min,1.03)

    def test_neutral_candidate_view_does_not_copy_full_pool(self):
        from core.drr_peak_analysis import analyze_drr_peaks
        w=self.window;w.activate_dataset(self.a.key)
        result=analyze_drr_peaks(self.a.cube,self.a.settings,retain_candidates=True)
        self.assertIs(w.filtered_dataset_result(self.a.key,result),result)

    def test_workspace_restores_p2p_results_and_fixed_view_without_recalculation(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from ui_qt.drr_analysis_window import DrrAnalysisWindow
        w=self.window;w.open_range_amplitude();page=w.amplitude_page
        page.bounds[0].setValue(1.035);page.bounds[1].setValue(1.065)
        page.smoothing.setCurrentText('SG');page.sg_width.setValue(3.)
        page.auto_timer.stop()
        records=page.compute_records(list(w.datasets.values()),(1.035,1.065,0.,2.),False,smoothing=page.smoothing_settings())
        page.receive_records(records)
        page.set_display_bounds((1.02,1.08,0.,1.));page.apply_display_range();page.fixed_display.setChecked(True)
        page.select_y(1.)
        page.style_controls['width'].setValue(7.5)
        page.style_controls['legend_size'].setValue(11)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'workspace.npz';w.session_path=path
            self.assertTrue(w.save_workspace(quiet=True))
            restored=DrrAnalysisWindow()
            try:
                with patch('ui_qt.drr_p2p_batch.BatchRangeAmplitudePage.compute_records',side_effect=AssertionError('recalculated')):
                    restored.restore_session(path)
                p=restored.amplitude_page
                self.assertIsNotNone(p)
                self.assertTrue(p.fixed_display.isChecked())
                self.assertEqual(p.map_axis.get_xlim(),(1.02,1.08))
                self.assertEqual(p.map_axis.get_ylim(),(0.,1.))
                self.assertEqual(p.row_spin.value(),2)
                self.assertEqual(tuple(spin.value() for spin in p.bounds[:2]),(1.035,1.065))
                self.assertEqual(p.smoothing.currentText(),'SG')
                self.assertEqual(p.sg_unit.currentText(),'points')
                self.assertEqual(p.sg_width.value(),3.)
                self.assertEqual(p.style_controls['width'].value(),7.5)
                self.assertEqual(p.style_controls['legend_size'].value(),11)
                np.testing.assert_array_equal(p.records[0]['processed'].Z,page.records[0]['processed'].Z)
                np.testing.assert_array_equal(p.records[0]['result'],page.records[0]['result'])
            finally:restored.session_path=None;restored.close();w.session_path=None

    def test_p2p_display_enter_commits_and_updates_map_without_recalculation(self):
        from PySide6.QtTest import QTest
        from PySide6.QtCore import Qt
        w=self.window;w.show();w.open_range_amplitude();page=w.amplitude_page
        page.display_panel.show();self.app.processEvents()
        records=page.records;spin=page.display_bounds[0]
        spin.setFocus();spin.selectAll();QTest.keyClicks(spin,'1.02');QTest.keyClick(spin,Qt.Key_Return)
        QTest.qWait(250)
        self.assertEqual(spin.value(),1.02)
        self.assertAlmostEqual(page.map_axis.get_xlim()[0],1.02)
        self.assertIs(page.records,records)

    def test_p2p_calculation_enter_and_drag_update_all_files_automatically(self):
        from PySide6.QtTest import QTest
        from PySide6.QtCore import Qt
        w=self.window;w.show();w.open_range_amplitude();page=w.amplitude_page
        def finish():
            deadline=time.monotonic()+10
            while (page.auto_timer.isActive() or w.jobs) and time.monotonic()<deadline:
                self.app.processEvents();time.sleep(.005)
            self.assertFalse(w.jobs);self.assertFalse(page.auto_timer.isActive())
        spin=page.bounds[0];spin.setFocus();spin.selectAll()
        QTest.keyClicks(spin,'1.04');QTest.keyClick(spin,Qt.Key_Return);finish()
        self.assertEqual(len(page.records),2)
        self.assertTrue(all(r['bounds'][0]==1.04 for r in page.records))
        page.select_window(1.045,1.06);finish()
        self.assertTrue(all(r['bounds'][:2]==(1.045,1.06) for r in page.records))
        self.assertTrue(page.csv_button.isEnabled())

    def test_p2p_queues_latest_range_and_discards_old_completion(self):
        from unittest.mock import patch
        w=self.window;w.open_range_amplitude();page=w.amplitude_page
        old=page.records
        with patch.object(w,'start_job') as start:
            page.bounds[0].setValue(1.02);page.run_batch()
            callback=start.call_args.args[1]
            stale=page.compute_records(list(w.datasets.values()),(1.02,1.1,0,2),False)
            page.bounds[0].setValue(1.04)
            callback(stale)
            self.assertIs(page.records,old)
            self.assertTrue(page.auto_timer.isActive())
            self.assertFalse(page.csv_button.isEnabled())
            page.auto_timer.stop()

    def test_p2p_batch_runs_selected_files_and_keeps_comparison_on_dataset_change(self):
        w=self.window;w.activate_dataset(self.a.key);w.open_range_amplitude()
        page=w.amplitude_page
        # Only one selected file: P2P must still compare the entire workspace.
        w.list.clearSelection();w.list.item(0).setSelected(True)
        page.run_batch();self.finish_filter()
        self.assertEqual(len(page.records),2)
        self.assertTrue(page.csv_button.isEnabled())
        self.assertEqual(page.records[0]['bounds'][:2],page.records[1]['bounds'][:2])
        w.activate_dataset(self.b.key)
        self.assertIs(w.amplitude_page,page)
        self.assertEqual(len(page.records),2)
        w.remove_keys([self.b.key])
        self.assertEqual(len(page.records),1)
        self.assertFalse(page.csv_button.isEnabled())

    def test_p2p_tab_hides_peak_workflow_and_restores_it_on_return(self):
        w=self.window
        self.assertEqual(w.analysis_tabs.count(),2)
        w.analysis_tabs.setCurrentIndex(1)
        self.assertIsNotNone(w.amplitude_page)
        self.assertTrue(w.peak_sidebar.isHidden())
        self.assertIs(w.sidebar_stack.currentWidget(),w.p2p_sidebar)
        self.assertTrue(w.amplitude_page.sidebar.isAncestorOf(w.amplitude_page.batch_button))
        self.assertFalse(w.amplitude_page.isAncestorOf(w.amplitude_page.batch_button))
        self.assertTrue(all(button.isHidden() for button in w.peak_batch_buttons))
        w.analysis_tabs.setCurrentIndex(0)
        self.assertFalse(w.peak_sidebar.isHidden())
        self.assertTrue(all(not button.isHidden() for button in w.peak_batch_buttons))

    def test_color_limits_follow_display_and_toolbar_ranges_but_can_lock(self):
        w=self.window;w.activate_dataset(self.a.key);w.redraw_plot()
        axis=w._drr_heatmap_axes['raw'];original=axis.collections[0].get_clim()
        self.a.settings=replace(self.a.settings,x_min=1.,x_max=1.02)
        w.redraw_plot()
        self.assertEqual(w._drr_heatmap_axes['raw'].collections[0].get_clim(),original)
        w.set_display_range((1.,1.02,0.,2.));w.display_range_changed();w.timer.stop();w.redraw_plot()
        axis=w._drr_heatmap_axes['raw'];narrow=axis.collections[0].get_clim()
        self.assertLess(narrow[1],original[1]/10)
        w.lock_color_scale.setChecked(True)
        axis.set_xlim(1.,1.1);w.update_visible_color_scales()
        self.assertEqual(axis.collections[0].get_clim(),narrow)
        w.lock_color_scale.setChecked(False);w.update_visible_color_scales()
        self.assertGreater(axis.collections[0].get_clim()[1],narrow[1]*10)

    def test_color_scale_uses_visible_y_rows(self):
        w=self.window;z=self.a.cube.Z.copy();z[2]*=100
        self.a.cube=replace(self.a.cube,Z=z)
        w.activate_dataset(self.a.key);w.set_display_range((1.,1.1,0.,2.));w.redraw_plot()
        full=w._drr_heatmap_axes['raw'].collections[0].get_clim()[1]
        w.set_display_range((1.,1.1,0.,1.));w.redraw_plot()
        local=w._drr_heatmap_axes['raw'].collections[0].get_clim()[1]
        self.assertLess(local,full/10)

    def test_result_table_defers_content_sizing_until_all_cells_are_updated(self):
        from unittest.mock import patch
        from PySide6.QtWidgets import QHeaderView
        from core.drr_peak_analysis import analyze_drr_peaks
        w=self.window;w.activate_dataset(self.a.key)
        result=analyze_drr_peaks(self.a.cube,self.a.settings,retain_candidates=True)
        result.update(dataset_id=self.a.key,dataset_name=self.a.name,dataset_revision=self.a.revision,provenance=self.a.provenance)
        self.a.result=result;w.hydrate_result()
        table=w.analysis.table;header=table.horizontalHeader()
        modes=[header.sectionResizeMode(i) for i in range(table.columnCount())]
        original=table.setItem
        def set_item(row,column,item):
            self.assertNotEqual(header.sectionResizeMode(column),QHeaderView.ResizeToContents)
            self.assertTrue(table.signalsBlocked())
            return original(row,column,item)
        with patch.object(table,'setItem',side_effect=set_item):w.analysis.populate_table()
        self.assertEqual([header.sectionResizeMode(i) for i in range(table.columnCount())],modes)
        self.assertFalse(table.signalsBlocked())

    def finish_filter(self):
        deadline=time.monotonic()+10
        while (self.window.filter_timer.isActive() or self.window.jobs) and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.005)
        self.assertFalse(self.window.jobs);self.assertFalse(self.window.filter_timer.isActive())

    def test_metric_filter_reuses_candidates_and_survives_manual_exclusion(self):
        from unittest.mock import patch
        from core.drr_peak_analysis import analyze_drr_peaks
        w=self.window;w.activate_dataset(self.a.key)
        result=analyze_drr_peaks(self.a.cube,self.a.settings,retain_candidates=True)
        result.update(dataset_id=self.a.key,dataset_name=self.a.name,dataset_revision=self.a.revision,provenance=self.a.provenance)
        self.a.result=result;w.hydrate_result();revision=self.a.revision
        count=len(w.analysis.points);self.assertGreater(count,0)
        with patch('ui_qt.drr_analysis_window.analyze_drr_peaks',side_effect=AssertionError('redetected')):
            w.result_metrics['min_width_mev'].setValue(100)
            self.finish_filter()
            self.assertEqual(w.analysis.points,[])
            w.result_metrics['min_width_mev'].setValue(0)
            self.finish_filter()
            self.assertEqual(len(w.analysis.points),count)
            self.assertEqual(self.a.revision,revision)
            w.analysis.table.selectRow(0);w.analysis.exclude_selected()
            self.assertEqual(len(w.analysis.points),count-1)
        self.assertFalse(w.jobs)

    def test_result_interval_filters_without_recomputing_or_losing_candidates(self):
        from core.drr_analysis_workspace import analyze_dataset
        from core.drr_result_filter import filtered_result
        w=self.window;w.activate_dataset(self.a.key)
        self.a.result=analyze_dataset(self.a);w.hydrate_result()
        result=self.a.result;revision=self.a.revision;count=len(w.analysis.points)
        self.assertGreater(count,0)
        w.result_interval_selected(1.001,1.002)
        self.finish_filter()
        self.assertEqual(len(w.analysis.points),0)
        self.assertIs(self.a.result,result)
        self.assertEqual(self.a.revision,revision)
        self.assertFalse(w.jobs)
        exported=filtered_result(result,w.result_filters[self.a.key])
        self.assertEqual(sum(len(p['points']) for p in exported['products'].values()),0)
        self.assertEqual(len(exported['filtered_out_points']),count)
        w.result_filter_enabled.setChecked(False)
        self.finish_filter()
        self.assertEqual(len(w.analysis.points),count)

    def test_display_fixed_across_groups_does_not_change_detection(self):
        w=self.window;w.activate_dataset(self.a.key)
        w.display_bounds['x_min'].setValue(1.02)
        self.assertAlmostEqual(self.a.settings.x_min,1.)
        w.fixed_range.setChecked(True);w.activate_dataset(self.b.key);w.redraw_plot()
        self.assertAlmostEqual(self.b.settings.x_min,1.)
        self.assertAlmostEqual(w._drr_heatmap_ax.get_xlim()[0],1.02)
        self.assertFalse(w.jobs)

    def test_unlinked_display_preserves_peak_results_and_bounds(self):
        from core.drr_analysis_workspace import analyze_dataset
        w=self.window;w.activate_dataset(self.a.key)
        self.a.result=analyze_dataset(self.a);result=self.a.result
        w.follow.setChecked(False)
        w.display_bounds['x_min'].setValue(1.02);w.redraw_plot()
        self.assertIs(self.a.result,result)
        self.assertAlmostEqual(self.a.settings.x_min,1.)
        self.assertAlmostEqual(w._drr_heatmap_ax.get_xlim()[0],1.02)
        w.display_bounds['x_max'].setValue(1.01)
        self.assertAlmostEqual(w.plot_range[1],1.1)

    def test_close_and_restore_workspace(self):
        import tempfile
        from pathlib import Path
        from ui_qt.drr_analysis_window import DrrAnalysisWindow
        from core.drr_analysis_workspace import analyze_dataset
        w=self.window;w.activate_dataset(self.a.key)
        w.analysis.prominence.setValue(.21)
        w.follow.setChecked(False);w.fixed_range.setChecked(True)
        w.display_bounds['x_min'].setValue(1.02)
        self.a.result=analyze_dataset(self.a)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'workspace.npz';w.session_path=path
            w.close();w.session_path=None
            other=DrrAnalysisWindow()
            try:
                other.restore_session(path)
                self.assertEqual(len(other.datasets),2)
                self.assertEqual(other.active_key,self.a.key)
                self.assertAlmostEqual(other.analysis.prominence.value(),.21)
                self.assertFalse(other.follow.isChecked())
                self.assertTrue(other.fixed_range.isChecked())
                self.assertAlmostEqual(other._drr_heatmap_ax.get_xlim()[0],1.02)
                self.assertEqual(other.datasets[self.a.key].result,self.a.result)
            finally:other.session_path=None;other.close()

    def test_missing_persistence_path_cannot_silently_discard_data(self):
        from unittest.mock import patch
        from PySide6.QtGui import QCloseEvent
        w=self.window;w.persistence_required=True
        event=QCloseEvent()
        with patch.object(w,'save_workspace',return_value=False) as save:
            w.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertEqual(save.call_count,2)
        w.persistence_required=False

    def test_processed_group_picker_reuses_saved_data_and_deduplicates(self):
        from unittest.mock import patch
        from pathlib import Path
        w=self.window
        def complete(fn,callback,**kwargs):
            from unittest.mock import Mock
            callback(fn(progress=Mock(),log=Mock()))
        with patch.object(w,'start_job',side_effect=complete), patch('core.drr_processed_groups.discover_groups',return_value=[Path('group.metadata.json')]), patch('core.drr_processed_groups.load_group',return_value=self.a) as load:
            w.refresh_processed_groups()
            self.assertEqual(w.processed_groups.count(),1)
            w.processed_groups.item(0).setSelected(True)
            w.use_processed_groups()
            load.assert_called_once_with('group.metadata.json')
            self.assertEqual(len(w.datasets),2)
            self.assertEqual(w.active_key,self.a.key)

    def test_switch_restores_settings_without_mutating_other_dataset(self):
        w=self.window;w.activate_dataset(self.a.key)
        w.analysis.bounds['x_min'].setValue(1.02)
        w.activate_dataset(self.b.key)
        self.assertAlmostEqual(w.analysis.bounds['x_min'].value(),1.)
        w.activate_dataset(self.a.key)
        self.assertAlmostEqual(w.analysis.bounds['x_min'].value(),1.02)

    def test_range_change_redraws_without_running_analysis(self):
        w=self.window;w.activate_dataset(self.a.key)
        w.analysis.bounds['x_min'].setValue(1.02);w.redraw_plot()
        self.assertAlmostEqual(w._drr_heatmap_ax.get_xlim()[0],1.)
        self.assertIsNone(self.a.result)

    def test_duplicate_does_not_add_or_reset_settings(self):
        w=self.window;w.activate_dataset(self.a.key);w.analysis.prominence.setValue(.2)
        w.add_dataset(self.a)
        self.assertEqual(len(w.datasets),2)
        self.assertAlmostEqual(w.analysis.prominence.value(),.2)

    def test_late_result_is_rejected_after_settings_change(self):
        w=self.window;revision=self.a.revision
        from core.drr_analysis_workspace import apply_settings
        apply_settings(self.a,replace(self.a.settings,prominence=.2))
        w.accept_results([(self.a.key,revision,{'products':{}},None,self.a)])
        self.assertIsNone(self.a.result)

    def test_readded_dataset_does_not_accept_previous_instance_result(self):
        from core.drr_analysis_workspace import create_dataset
        w=self.window;old=self.a
        w.remove_keys([old.key])
        new=create_dataset(old.cube,old.name,old.provenance,old.settings)
        w.add_dataset(new)
        self.assertEqual(old.key,new.key)
        self.assertEqual(old.revision,new.revision)
        w.accept_results([(old.key,old.revision,{'products':{}},None,old)])
        self.assertIsNone(new.result)

    def test_detector_parameters_do_not_rebuild_colorplot(self):
        w=self.window;w.timer.stop()
        w.analysis.prominence.setValue(.23)
        self.assertFalse(w.timer.isActive())

    def test_gate_change_keeps_heatmap_axes(self):
        w=self.window;w.timer.stop();axis=w._drr_heatmap_ax
        w.gate_spin.setValue(1.)
        deadline=time.monotonic()+.3
        while time.monotonic()<deadline:self.app.processEvents();time.sleep(.01)
        self.assertIs(w._drr_heatmap_ax,axis)

    def test_cancel_then_analyze_again_resets_job_state(self):
        from unittest.mock import patch
        from threading import Event
        entered=Event();w=self.window
        def slow(cube,settings,cancelled=None,progress=None,**kwargs):
            entered.set();deadline=time.monotonic()+5
            while not cancelled() and time.monotonic()<deadline:time.sleep(.005)
            raise RuntimeError('Peak analysis cancelled.')
        with patch('ui_qt.drr_analysis_window.analyze_drr_peaks',side_effect=slow):
            w.analyze_keys([self.b.key]);self.assertTrue(entered.wait(2))
            self.assertFalse(w.analysis.analyze_button.isEnabled())
            w.cancel_job();deadline=time.monotonic()+5
            while w.jobs and time.monotonic()<deadline:self.app.processEvents();time.sleep(.005)
        self.assertFalse(w.jobs);self.assertTrue(w.job_status.text().startswith('Cancelled'))
        w.analyze_keys([self.b.key]);deadline=time.monotonic()+5
        while w.jobs and time.monotonic()<deadline:self.app.processEvents();time.sleep(.005)
        self.assertFalse(w.cancel.is_set());self.assertTrue(w.job_status.text().startswith('Completed'))

    def test_sg_edits_wait_for_explicit_preview(self):
        w=self.window;w.timer.stop();w.product(self.b,'second')
        previous=list(w.derivative_cache)
        w.drr_sg_window_spin.setValue(31)
        self.assertFalse(w.timer.isActive())
        self.assertEqual(list(w.derivative_cache),previous)
        w.product(self.b,'second')
        self.assertEqual(list(w.derivative_cache),previous)
        w.update_preview()
        self.assertIn((self.b.key,31,2),w.derivative_cache)

    def test_batch_parameter_change_invalidates_compare_preview(self):
        w=self.window
        from PySide6.QtCore import QItemSelectionModel
        w.list.setCurrentItem(w.list.item(0),QItemSelectionModel.ClearAndSelect)
        w.list.setCurrentItem(w.list.item(1),QItemSelectionModel.Select)
        w.display.setCurrentIndex(3);w.timer.stop()
        w.analysis.prominence.setValue(.23);w.apply_selected()
        self.assertTrue(w.timer.isActive())

    def test_independent_manual_views_restore(self):
        w=self.window;w.activate_dataset(self.a.key);w.follow.setChecked(False)
        w._drr_heatmap_ax.set_xlim(1.01,1.06)
        w.activate_dataset(self.b.key);w.follow.setChecked(False)
        w._drr_heatmap_ax.set_xlim(1.04,1.09)
        w.activate_dataset(self.a.key)
        self.assertEqual(tuple(w._drr_heatmap_ax.get_xlim()),(1.01,1.06))

    def test_batch_result_exports_and_multi_selection_survives_switch(self):
        from PySide6.QtCore import QItemSelectionModel
        from core.drr_analysis_workspace import export_summary
        import tempfile
        w=self.window
        w.list.setCurrentItem(w.list.item(0),QItemSelectionModel.ClearAndSelect)
        w.list.setCurrentItem(w.list.item(1),QItemSelectionModel.Select)
        self.assertEqual(len(w.selected_keys()),2)
        w.analyze_selected();deadline=time.monotonic()+10
        while w.jobs and time.monotonic()<deadline:self.app.processEvents();time.sleep(.01)
        self.assertFalse(w.jobs)
        self.assertIsNotNone(self.a.result)
        self.assertIsNotNone(self.b.result)
        self.assertEqual(w.progress_bar.value(),100)
        self.assertTrue(w.job_status.text().startswith('Completed'))
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(export_summary(folder,[self.a,self.b]).exists())
