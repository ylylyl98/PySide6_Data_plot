import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
from types import SimpleNamespace
import numpy as np
from PySide6.QtWidgets import QApplication
from core.loader import DataCube
from core.drr_analysis_workspace import create_dataset
from core.drr_comparison_groups import product_id
from core.drr_workspace_session import load_session
from ui_qt.drr_analysis_window import DrrAnalysisWindow


class ComparisonUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_cross_group_range_locks_keep_display_and_recalculate_only_changed_calculation(self):
        from ui_qt.drr_comparison_workspace import carry_p2p_ranges
        w=DrrAnalysisWindow()
        x=np.linspace(1.,1.1,41)
        d=create_dataset(DataCube(x,np.arange(3.),np.array([np.sin(x*100)]*3),'Y','a','DR/R'),'a',{})
        try:
            w.add_dataset(d);w.open_range_amplitude();page=w.amplitude_page
            target=page.snapshot_state()
            page.set_display_bounds((1.02,1.08,0.,1.));page.fixed_display.setChecked(True)
            previous=page.snapshot_state()
            carried=carry_p2p_ranges(previous,target)
            self.assertFalse(carried.get('_recalculate_on_open',False))
            self.assertEqual(carried['display_bounds'],[1.02,1.08,0.,1.])
            page.fixed_calculation.setChecked(True)
            page.bounds[0].setValue(1.03);page.auto_timer.stop()
            w.replace_comparison(([d],{},d.key,{'p2p':target}))
            restored=w.amplitude_page
            self.assertTrue(restored.fixed_calculation.isChecked())
            self.assertTrue(restored.fixed_display.isChecked())
            self.assertEqual(restored.bounds[0].value(),1.03)
            self.assertEqual(restored.map_axis.get_xlim(),(1.02,1.08))
            self.assertTrue(restored.pending)
            self.assertTrue(restored.auto_timer.isActive())
            restored.auto_timer.stop()
            self.assertTrue(restored.snapshot_state()['fixed_calculation'])
            unlocked=carry_p2p_ranges({**previous,'fixed_display':False,'fixed_calculation':False},target)
            self.assertEqual(unlocked['bounds'],target['bounds'])
            self.assertEqual(unlocked['display_bounds'],target['display_bounds'])
        finally:
            w.comparison_key=None;w.session_path=None;w.persistence_required=False;w.close();self.app.processEvents()

    def test_unchanged_sources_reuse_group_cache_and_same_group_is_noop(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);meta=root/'a.metadata.json';dat=root/'a.dat'
            meta.write_text('{}');dat.write_text('original')
            x=np.linspace(1.,1.1,41)
            d=create_dataset(DataCube(x,np.arange(3.),np.array([np.sin(x*100)]*3),'Y','a','DR/R'),'a',{'metadata_path':str(meta)})
            pid=product_id(meta)
            w=DrrAnalysisWindow(session_path=root/'workspace.npz')
            w.comparison_catalog={'one':{'label':'one','members':[pid]}}
            w.comparison_entries={pid:{'path':str(meta)}}
            def run(fn,callback,**_):callback(fn(progress=SimpleNamespace(emit=lambda *_:None),log=SimpleNamespace(emit=lambda *_:None)))
            try:
                with patch.object(w,'start_job',side_effect=run),patch('core.drr_processed_groups.load_group',return_value=d) as load:
                    w.open_comparison('one');self.assertEqual(load.call_count,1)
                    w.open_range_amplitude();page=w.amplitude_page
                    w.open_comparison('one');self.assertIs(w.amplitude_page,page)
                    self.assertEqual(load.call_count,1)
                    w.persist_comparison(w.comparison_store(),'one',w.comparison_snapshot())
                    w.comparison_key=None
                    with patch.object(page.__class__,'compute_records',side_effect=AssertionError('Cached P2P recomputed')):
                        w.open_comparison('one')
                    self.assertEqual(load.call_count,1)
                    dat.write_text('changed source')
                    w.open_comparison('one');self.assertEqual(load.call_count,2)
            finally:
                w.comparison_key=None;w.session_path=None;w.persistence_required=False;w.close();self.app.processEvents()

    def test_open_button_and_double_click_do_not_open_membership_dialog(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidgetItem
        w=DrrAnalysisWindow()
        try:
            item=QListWidgetItem('Test group');item.setData(Qt.UserRole,'group');w.comparison_groups.addItem(item);w.comparison_groups.setCurrentItem(item)
            with patch.object(w,'open_comparison') as load,patch('ui_qt.drr_comparison_browser.ComparisonPreview',side_effect=AssertionError('Membership dialog opened')):
                w.open_group_button.click()
                w.comparison_groups.itemDoubleClicked.emit(item)
                self.assertEqual(load.call_count,2)
                load.assert_called_with('group')
            self.assertEqual(w.manage_group_button.text(),'Manage members…')
        finally:w.close();self.app.processEvents()

    def test_p2p_group_load_preserves_tab_and_skips_peak_previews(self):
        w=DrrAnalysisWindow()
        x=np.linspace(1.,1.1,41)
        datasets=[create_dataset(DataCube(x,np.arange(3.),np.array([np.sin(x*100)]*3),'Y',str(i),'DR/R'),str(i),{'measurement_files':[f'{i}.csv']}) for i in range(4)]
        try:
            w.add_dataset(datasets[0]);w.open_range_amplitude()
            with patch.object(w,'product',wraps=w.product) as product,patch.object(w,'activate_dataset',wraps=w.activate_dataset) as activate:
                w.replace_comparison((datasets,{},datasets[0].key,{'analysis_tab':0}))
                self.assertEqual(w.analysis_tabs.currentIndex(),1)
                self.assertEqual(activate.call_count,1)
                product.assert_not_called()
                self.assertEqual(len(w.amplitude_page.records),4)
            w.analysis_tabs.setCurrentIndex(0)
            self.assertIsNotNone(w._drr_heatmap_ax)
        finally:w.close();self.app.processEvents()

    def test_switch_exclude_restore_retains_results_and_ranges(self):
        with TemporaryDirectory() as folder:
            w=DrrAnalysisWindow(session_path=Path(folder)/'workspace.npz')
            x=np.linspace(1.,1.1,101)
            a=create_dataset(DataCube(x,np.arange(3.),np.array([np.sin(x*100)]*3),'Y','a','DR/R'),'a',{'metadata_path':str(Path(folder)/'a.metadata.json')})
            b=create_dataset(DataCube(x,np.arange(5.),np.array([np.sin(x*100)]*5),'Y','b','DR/R'),'b',{'metadata_path':str(Path(folder)/'b.metadata.json')})
            pa=product_id(a.provenance['metadata_path']);pb=product_id(b.provenance['metadata_path'])
            w.comparison_catalog={'one':{'label':'one','members':[pa,pb]},'two':{'label':'two','members':[a.provenance['metadata_path']]}}
            w.comparison_entries={pa:{'path':a.provenance['metadata_path']},pb:{'path':b.provenance['metadata_path']}}
            def run(fn,callback,**_):callback(fn(progress=SimpleNamespace(emit=lambda *_:None),log=SimpleNamespace(emit=lambda *_:None)))
            def load(path):return a if product_id(path)==pa else b
            try:
                with patch.object(w,'start_job',side_effect=run),patch('core.drr_processed_groups.load_group',side_effect=load):
                    w.open_comparison('one');self.assertEqual(len(w.datasets),2)
                    w.open_range_amplitude();page=w.amplitude_page
                    page.bounds[0].setValue(1.02);page.bounds[1].setValue(1.08);page.auto_timer.stop()
                    page.receive_records(page.compute_records(list(w.datasets.values()),(1.02,1.08,0.,4.),False))
                    page.set_display_bounds((1.01,1.09,0.,4.));page.fixed_display.setChecked(True);page.apply_display_range()
                    original=page.records[0]['result'].copy()
                    page.bounds[0].setValue(1.03);page.auto_timer.stop();page.run_batch()
                    archives=list((w.comparison_store().root/'history').rglob('*.npz'))
                    self.assertEqual(len(archives),1)
                    self.assertEqual(load_session(archives[0],with_display=True)[3]['p2p']['bounds'][0],1.02)
                    page.bounds[0].setValue(1.02);page.auto_timer.stop();page.run_batch()
                    w.open_comparison('two');self.assertEqual(len(w.datasets),1)
                    w.open_comparison('one');self.assertEqual(len(w.datasets),2)
                    self.assertEqual(w.amplitude_page.bounds[0].value(),1.02)
                    self.assertEqual(w.amplitude_page.map_axis.get_xlim(),(1.01,1.09))
                    np.testing.assert_equal(w.amplitude_page.records[0]['result'],original)
                    w.list.clearSelection()
                    for i in range(w.list.count()):
                        if w.list.item(i).data(256)==a.key:w.list.item(i).setSelected(True)
                    w.remove_selected();self.assertNotIn(a.key,w.datasets)
                    store=w.comparison_store();self.assertEqual(store.members('one',[pa,pb]),[pb])
                    self.assertTrue(store.dataset(a.key).exists())
                    w.open_comparison('one');self.assertEqual(set(w.datasets),{b.key})
                    store.reset('one');w.open_comparison('one');self.assertEqual(len(w.datasets),2)
                    self.assertEqual(len(w.amplitude_page.records),2)
                    self.assertGreater(len(list((store.root/'history').rglob('*.npz'))),0)
                    original_root=store.root
                    w.session_path=Path(folder)/'export'/'renamed.npz'
                    self.assertEqual(w.comparison_store().root,original_root)
                    old_b=b.key
                    b=create_dataset(DataCube(x,np.arange(5.),np.array([np.sin(x*100)+.1]*5),'Y','b','DR/R'),'b',b.provenance)
                    w.open_comparison('one')
                    self.assertNotIn(old_b,w.datasets)
                    self.assertIn(b.key,w.datasets)
                    self.assertTrue(w.amplitude_page.pending)
                    self.assertNotIn(old_b,{r['dataset'].key for r in w.amplitude_page.records})
            finally:
                w.comparison_key=None;w.session_path=None;w.persistence_required=False;w.close();self.app.processEvents()
