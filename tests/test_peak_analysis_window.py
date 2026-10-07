import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.peak_workspace import create_dataset, load_workspace
from tests.test_peak_workspace import sample


class AnalysisWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui_qt.peak_analysis_window import PeakAnalysisWindow
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'workspace.npz'
        self.window = PeakAnalysisWindow(self.path)
        self.window.add_datasets([create_dataset(sample(), 'PL', 'PL', 'a.csv', 'PL sample')])

    def wait_idle(self):
        deadline = time.monotonic()+12
        while self.window.busy and time.monotonic()<deadline:
            self.app.processEvents(); time.sleep(.01)
        self.assertFalse(self.window.busy, 'Background analysis did not finish.')
        self.app.processEvents()

    def tearDown(self):
        self.wait_idle(); self.window.close(); self.wait_idle()
        self.app.processEvents(); self.window.deleteLater(); self.app.processEvents()
        self.temp.cleanup()

    def test_preview_track_fit_and_row_selection_keep_heatmap(self):
        w = self.window
        w.row_slider.setValue(4)
        axes = w.heat_ax
        w.preview(); self.wait_idle()
        self.assertEqual(w.controls.branches.count(), 2)
        w.analyze('track'); self.wait_idle()
        self.assertEqual(len(w.dataset['result']['points']), 18)
        w.analyze('fit'); self.wait_idle()
        self.assertEqual(len(w.dataset['result']['fits']), 9)
        axes = w.heat_ax
        w.row_slider.setValue(7)
        self.assertIs(w.heat_ax, axes)
        self.assertEqual(w.dataset['row'], 7)
        self.assertTrue(w.spectrum_ax.lines)

    def test_drr_hides_pl_fit_and_selection_does_not_rerun(self):
        w = self.window
        w.add_datasets([create_dataset(sample('DRR'), 'DRR', 'raw', 'b.csv', 'DRR sample')])
        w.show(); self.app.processEvents()
        self.assertFalse(w.controls.fit_group.isVisible())
        self.assertTrue(w.controls.polarity.isVisible())
        w.preview(); self.wait_idle(); w.analyze('track'); self.wait_idle()
        result = w.dataset['result']
        w.controls.branches.item(0).setCheckState(Qt.Unchecked)
        self.assertIs(w.dataset['result'], result)
        self.assertFalse(w.busy)

    def test_close_persists_and_restart_is_explicit(self):
        w = self.window
        w.preview(); self.wait_idle()
        w.controls.branches.item(0).setText('Exciton')
        w.request_restart(); self.wait_idle()
        datasets, active = load_workspace(self.path)
        self.assertEqual(datasets[0]['branches'][0]['name'], 'Exciton')
        self.assertTrue(w.restart_requested)
        self.assertEqual(active, datasets[0]['key'])

    def test_drr_products_use_tabs_and_keep_separate_branches(self):
        w=self.window
        raw=create_dataset(sample('DRR'),'DRR','raw','pair.csv','DRR · raw · Pair')
        second=create_dataset(sample('DRR'),'DRR','second','pair.csv','DRR · second · Pair')
        w.add_datasets([raw,second]);w.show();self.app.processEvents()
        self.assertEqual(w.dataset_tabs.count(),2)  # Initial PL plus one DRR file.
        self.assertTrue(w.product_tabs.isVisible())
        self.assertEqual([w.product_tabs.tabText(i) for i in range(2)],['ΔR/R','2nd derivative'])
        w.product_tabs.setCurrentIndex(0);w.preview();self.wait_idle()
        w.controls.branches.item(0).setText('Raw branch');key=w.dataset['key']
        w.row_slider.setValue(4)
        w.product_tabs.setCurrentIndex(1)
        self.assertEqual(w.dataset['key'],second['key'])
        self.assertEqual(w.dataset['row'],4)
        self.assertEqual(w.controls.branches.count(),0)
        w.product_tabs.setCurrentIndex(0)
        self.assertEqual(w.dataset['key'],key)
        self.assertEqual(w.controls.branches.item(0).text(),'Raw branch')
        self.assertFalse(w.busy)

    def test_drr_missing_product_is_disabled_and_does_not_use_another_file(self):
        w=self.window
        w.add_datasets([create_dataset(sample('DRR'),'DRR','second','other.csv','Other'),
                        create_dataset(sample('DRR'),'DRR','raw','only-raw.csv','Raw only')])
        self.assertFalse(w.product_tabs.isTabEnabled(1))
        self.assertEqual(w.dataset['source'],'only-raw.csv')

    def test_separate_handoffs_join_tabs_and_keep_previous_product_versions(self):
        w=self.window
        raw=create_dataset(sample('DRR'),'DRR','raw','a.csv','DRR · raw · A')
        second=create_dataset(sample('DRR'),'DRR','second','a.csv','DRR · second · A')
        for d in (raw,second):d['group_id']='source-revision'
        w.add_datasets([raw]);self.assertFalse(w.product_tabs.isTabEnabled(1))
        w.add_datasets([second]);self.assertTrue(w.product_tabs.isTabEnabled(0))
        self.assertEqual(w.dataset_tabs.count(),2)
        w.preview();self.wait_idle();w.controls.branches.item(0).setText('Previous derivative branch')
        cube=sample('DRR');cube.Z*=2
        updated=create_dataset(cube,'DRR','second','a.csv','DRR · second · A')
        updated['group_id']='source-revision';w.add_datasets([updated])
        self.assertEqual(w.dataset['key'],updated['key'])
        self.assertTrue(w.product_tabs.isTabEnabled(0))
        self.assertEqual(w.dataset_tabs.count(),3)
        w.dataset_tabs.setCurrentIndex(next(i for i in range(w.dataset_tabs.count()) if w.dataset_tabs.tabData(i)==second['key']))
        self.assertEqual(w.controls.branches.item(0).text(),'Previous derivative branch')

    def test_current_view_uses_child_zoom_and_survives_redraw(self):
        w=self.window
        w.heat_ax.set_xlim(1.64,1.67);w.heat_ax.set_ylim(-.6,.6)
        w.controls.scope.setCurrentIndex(1)
        self.assertAlmostEqual(w.dataset['settings']['x_min'],1.64)
        self.assertAlmostEqual(w.dataset['settings']['x_max'],1.67)
        w.row_slider.setValue(4)
        w.preview();self.wait_idle();w.analyze('track');self.wait_idle()
        self.assertEqual(len(w.dataset['branches']),1)
        self.assertEqual({p['row_index'] for p in w.dataset['result']['points']},{3,4,5})
        self.assertEqual(tuple(w.heat_ax.get_xlim()),(1.64,1.67))
        self.assertEqual(tuple(w.heat_ax.get_ylim()),(-.6,.6))

    def test_find_peaks_keeps_named_branches_missing_from_selected_row(self):
        w=self.window
        w.add_datasets([create_dataset(sample(missing=True),'PL','PL','missing.csv','Missing row')])
        w.row_slider.setValue(4);w.preview();self.wait_idle()
        w.controls.branches.item(0).setText('Exciton')
        before=[dict(b) for b in w.dataset['branches']]
        w.row_slider.setValue(6);w.preview();self.wait_idle()
        self.assertEqual(w.dataset['branches'],before)
        w.row_slider.setValue(4);w.preview();self.wait_idle()
        self.assertEqual(w.dataset['branches'],before)

    def test_close_during_initial_load_waits_for_worker(self):
        w=self.window;w.datasets.clear();w.show();self.app.processEvents()
        release=Event()
        try:
            w._start('Loading...',lambda cancel,progress:release.wait(4),lambda _:None)
            self.assertFalse(w.close())
            self.assertTrue(w.isVisible())
            release.set();self.wait_idle();self.app.processEvents()
            self.assertFalse(w.isVisible())
        finally:release.set()

    def test_cancelling_analysis_keeps_previous_result(self):
        w=self.window;w.preview();self.wait_idle();w.analyze('track');self.wait_idle()
        previous=w.dataset['result'];release=Event()
        try:
            w._start('Tracking...',lambda cancel,progress:release.wait(4),lambda _:w.dataset.update(result=None))
            w.cancel_button.click();release.set();self.wait_idle()
            self.assertIs(w.dataset['result'],previous)
            self.assertIn('Cancelled',w.status_label.text())
        finally:release.set()

    def test_find_overlays_entire_map_and_filters_keep_candidate_pool(self):
        w=self.window;w.preview();self.wait_idle()
        self.assertIsNotNone(w.dataset['result'], 'Find peaks must immediately produce an overlay')
        self.assertEqual({p['row_index'] for p in w.dataset['result']['points']},set(range(9)))
        self.assertTrue(any(a.get_gid()=='peak-candidates' for a in w.heat_ax.collections))
        pool=w.dataset['candidates'];before=len(w.dataset['result']['points'])
        w.controls.numeric['prominence'].setValue(1.);self.wait_idle()
        self.assertIs(w.dataset['candidates'],pool)
        self.assertLess(len(w.dataset['result']['points']),before)
        w.controls.numeric['prominence'].setValue(.05);self.wait_idle()
        self.assertEqual(len(w.dataset['result']['points']),before)
        self.assertIn('Savitzky',w.controls.method_description.text())

    def test_file_selection_is_a_tab_strip_not_a_dropdown(self):
        from PySide6.QtWidgets import QTabBar
        self.assertIsInstance(self.window.dataset_tabs,QTabBar)
        self.window.show();self.app.processEvents()
        self.assertFalse(self.window.dataset_tabs.isVisible())
        self.assertTrue(self.window.dataset_label.isVisible())
        self.window.add_datasets([create_dataset(sample('DRR'),'DRR','raw','b.csv','Other')])
        self.assertTrue(self.window.dataset_tabs.isVisible())

    def test_cached_filter_preserves_manually_picked_branch(self):
        from types import SimpleNamespace
        w=self.window;w.preview();self.wait_idle();w.row_slider.setValue(0)
        w.controls.seed.setChecked(True)
        w.plot_clicked(SimpleNamespace(xdata=1.648,ydata=1.,inaxes=w.spectrum_ax))
        manual=w.dataset['branches'][-1];identity=manual['id'];manual['name']='Manual exciton'
        w.controls.numeric['min_width_mev'].setValue(.6);self.wait_idle()
        self.assertTrue(w.dataset['manual_seeds'])
        self.assertEqual(next(b['name'] for b in w.dataset['branches'] if b['id']==identity),'Manual exciton')

    def test_product_recommendation_is_visible_and_keeps_manual_choice_on_reimport(self):
        w=self.window
        raw=create_dataset(sample('DRR'),'DRR','raw','pair.csv','DRR raw')
        second=create_dataset(sample('DRR'),'DRR','second','pair.csv','DRR derivative')
        w.add_datasets([raw,second])
        self.assertEqual(w.controls.method.currentIndex(),0)
        self.assertIn('Recommended: Local extrema',w.controls.method_description.text())
        w.controls.method.setCurrentIndex(1)
        self.assertEqual(w.dataset['settings']['detection_method'],'sg')
        w.product_tabs.setCurrentIndex(0)
        self.assertEqual(w.controls.method.currentIndex(),1)
        self.assertIn('Recommended: SG',w.controls.method_description.text())
        w.product_tabs.setCurrentIndex(1)
        self.assertEqual(w.controls.method.currentIndex(),1)
        w.add_datasets([create_dataset(sample('DRR'),'DRR','second','pair.csv','DRR derivative')])
        self.assertEqual(w.controls.method.currentIndex(),1)
        self.assertIn('Recommended: Local extrema',w.controls.method_description.text())
        self.assertFalse(w.busy)

    def test_quality_preset_enriches_legacy_cache_once_and_refilters_without_detection(self):
        from unittest.mock import patch
        from tests.test_peak_quality import QualityTests
        w=self.window;c=w.controls
        self.assertTrue(hasattr(c,'preset_buttons'), 'Quality preset buttons are missing')
        d=QualityTests().noisy_map();d['settings'].update(min_snr=0,neighbor_support=0)
        w.add_datasets([d]);w.preview();self.wait_idle()
        pool=d['candidates'];pool.pop('quality',None)
        for point in pool['points']:
            point.pop('snr',None);point.pop('noise_sigma',None)
        ids=[p['candidate_id'] for p in pool['points']];before=len(d['result']['points'])
        generation=w.generation
        with patch('core.drr_peak_analysis.find_peaks',side_effect=AssertionError('Preset reran detection')):
            c.preset_buttons['Balanced'].click();self.wait_idle()
            self.assertEqual(w.generation,generation+1,'A preset must launch one filter job')
            self.assertEqual(ids,[p['candidate_id'] for p in d['candidates']['points']])
            self.assertLess(len(d['result']['points']),before)
            self.assertTrue(c.preset_buttons['Balanced'].isChecked())
            cached=d['candidates']
            balanced_ids={p['candidate_id'] for p in d['result']['points']}
            c.max_count.setValue(1);self.wait_idle()
            self.assertFalse(c.preset_buttons['Balanced'].isChecked())
            self.assertLess(len(d['result']['points']),len(balanced_ids))
            c.preset_buttons['Balanced'].click();self.wait_idle()
            self.assertEqual({p['candidate_id'] for p in d['result']['points']},balanced_ids)
            c.preset_buttons['Strict'].click();self.wait_idle()
            self.assertIs(d['candidates'],cached)
            c.numeric['min_snr'].setValue(4.5);self.wait_idle()
            self.assertFalse(any(button.isChecked() for button in c.preset_buttons.values()))
            self.assertIn('Custom',c.preset_status.text())
            w.reset_filters();self.wait_idle()
            self.assertEqual(d['settings']['min_snr'],0)
            self.assertEqual(d['settings']['neighbor_support'],0)

    def test_filter_presets_follow_product_without_overwriting_manual_choices_on_switch(self):
        w = self.window; c = w.controls
        pl = w.dataset
        raw = create_dataset(sample('DRR'), 'DRR', 'raw', 'profiles.csv', 'Raw')
        second = create_dataset(sample('DRR'), 'DRR', 'second', 'profiles.csv', 'Second')
        w.add_datasets([raw, second])
        c.preset_buttons['Balanced'].click()
        self.assertEqual((second['settings']['min_snr'], second['settings']['prominence'],
                          second['settings']['neighbor_support']), (7., .1, 4))
        self.assertIn('2nd derivative', c.filter_heading.text())
        c.preset_buttons['Sensitive'].click()
        self.assertEqual((second['settings']['min_snr'], second['settings']['prominence'],
                          second['settings']['neighbor_support']), (5., .05, 3))
        self.assertAlmostEqual(second['settings']['min_width_mev'], .8)
        c.preset_buttons['Strict'].click()
        self.assertEqual((second['settings']['min_snr'], second['settings']['prominence'],
                          second['settings']['neighbor_support']), (9., .15, 5))
        c.numeric['min_snr'].setValue(6.2)
        second_settings = dict(second['settings'])
        w.product_tabs.setCurrentIndex(0); c.preset_buttons['Balanced'].click()
        self.assertEqual(raw['settings']['prominence'], .08)
        self.assertIn('ΔR/R', c.filter_heading.text())
        w.product_tabs.setCurrentIndex(1)
        self.assertEqual(second['settings'], second_settings)
        self.assertFalse(any(b.isChecked() for b in c.preset_buttons.values()))
        w.dataset_tabs.setCurrentIndex(next(i for i in range(w.dataset_tabs.count())
                                           if w.dataset_tabs.tabData(i) == pl['key']))
        c.preset_buttons['Balanced'].click()
        self.assertEqual(pl['settings']['prominence'], .05)
        self.assertIn('PL', c.filter_heading.text())
        c.smoothing.setValue(15); c.preset_buttons['Strict'].click()
        self.assertEqual(pl['settings']['smoothing_window'], 15)
        self.assertEqual(pl['settings']['prominence'], .075)
        self.assertAlmostEqual(pl['settings']['min_width_mev'], 1.6)

    def test_derivative_preprocessing_and_extra_smoothing_are_distinct(self):
        w=self.window;c=w.controls
        d=create_dataset(sample('DRR'),'DRR','second','sg.csv','DRR derivative')
        d['source_processing']=dict(method='Savitzky–Golay',derivative=2,window_samples=21,polyorder=3)
        w.add_datasets([d]);w.preview();self.wait_idle()
        self.assertTrue(hasattr(c,'input_processing'))
        self.assertIn('21 samples',c.input_processing.text())
        self.assertIn('order 3',c.input_processing.text())
        self.assertEqual(c.method.currentIndex(),0)
        self.assertEqual(c.method.tabText(1),'Extra SG')
        pool=d['candidates'];result=d['result']
        c.method.setCurrentIndex(1)
        self.assertIs(d['candidates'],pool);self.assertIs(d['result'],result)
        self.assertIn('Extra smoothing',c.method_description.text())
        self.assertIn('Detection settings changed',c.cache_status.text())
        w.add_datasets([create_dataset(sample('DRR'),'DRR','second','legacy.csv','Legacy')])
        self.assertIn('unavailable',c.input_processing.text())
        w.add_datasets([create_dataset(sample('DRR'),'DRR','raw','raw.csv','Raw')])
        self.assertEqual(c.method.tabText(1),'SG + extrema')
        self.assertTrue(c.input_processing.isHidden())

    def test_xy_ranges_refilter_cache_and_marker_size_only_updates_artists(self):
        from unittest.mock import patch
        from PySide6.QtWidgets import QTabBar
        w=self.window;c=w.controls;w.show();self.app.processEvents()
        w.preview();self.wait_idle();d=w.dataset;pool=d['candidates']
        self.assertIsInstance(c.scope,QTabBar)
        self.assertTrue(c.ranges_widget.isVisible())
        with patch('core.peak_candidates.detect_heatmap',side_effect=AssertionError('Range reran detection')):
            for key,value in [('x_min',1.64),('x_max',1.67),('y_min',-.6),('y_max',.6)]:
                c.bounds[key].setValue(value);self.wait_idle()
            self.assertEqual(c.scope.currentIndex(),2)
            self.assertIs(d['candidates'],pool)
            self.assertEqual({p['row_index'] for p in d['result']['points']},{3,4,5})
            self.assertTrue(all(1.64<=p['energy']<=1.67 for p in d['result']['points']))
            c.scope.setCurrentIndex(0);self.wait_idle()
            self.assertEqual(len(d['result']['points']),18)
            c.scope.setCurrentIndex(2);self.wait_idle()
            self.assertEqual({p['row_index'] for p in d['result']['points']},{3,4,5})
        self.assertTrue(hasattr(w,'marker_size'))
        axes=w.heat_ax;result=d['result'];generation=w.generation
        heatmaps=[a for a in axes.collections if a.get_gid()!='peak-candidates']
        with patch('ui_qt.peak_analysis_window.build_figure',side_effect=AssertionError('Size rebuilt heatmap')):
            w.marker_size.setValue(4);self.app.processEvents()
        self.assertIs(w.heat_ax,axes);self.assertIs(d['result'],result)
        self.assertIs(d['candidates'],pool);self.assertEqual(w.generation,generation)
        self.assertEqual(d['marker_size'],4)
        self.assertTrue(all(float(a.get_sizes()[0])==4 for a in axes.collections if a.get_gid()=='peak-candidates'))
        self.assertTrue(all(a in axes.collections for a in heatmaps))
        w.request_restart();self.wait_idle()
        saved,_=load_workspace(self.path)
        self.assertEqual(next(x for x in saved if x['key']==d['key'])['marker_size'],4)

    def test_legacy_custom_range_survives_scope_switch(self):
        w=self.window;d=create_dataset(sample(),'PL','PL','legacy.csv','Legacy')
        d['scope']=2;d['settings'].update(x_min=1.64,x_max=1.67,y_min=-.6,y_max=.6)
        w.add_datasets([d]);w.controls.scope.setCurrentIndex(0);w.controls.scope.setCurrentIndex(2)
        self.assertAlmostEqual(d['settings']['x_min'],1.64)
        self.assertAlmostEqual(d['settings']['y_max'],.6)

    def test_view_range_follows_zoom_once_after_gesture_without_redetection(self):
        from unittest.mock import patch
        w=self.window;w.preview();self.wait_idle();w.controls.scope.setCurrentIndex(1);self.wait_idle()
        pool=w.dataset['candidates'];generation=w.generation
        with patch('ui_qt.peak_analysis_window.detect_heatmap',side_effect=AssertionError('Zoom reran detection')):
            w.heat_ax.set_xlim(1.64,1.67);w.heat_ax.set_ylim(-.6,.6)
            deadline=time.monotonic()+3
            while w.generation==generation and time.monotonic()<deadline:
                self.app.processEvents();time.sleep(.01)
            self.wait_idle()
            self.assertAlmostEqual(w.controls.bounds['x_min'].value(),1.64)
            self.assertAlmostEqual(w.controls.bounds['y_max'].value(),.6)
            self.assertEqual(w.generation,generation+1)
            self.assertEqual({p['row_index'] for p in w.dataset['result']['points']},{3,4,5})
            self.assertIs(w.dataset['candidates'],pool)


if __name__ == '__main__':
    unittest.main()
