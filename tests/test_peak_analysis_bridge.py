import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from threading import Event
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar
from matplotlib.figure import Figure
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
from core.peak_workspace import create_dataset, fingerprint, load_workspace
from core.pl_peak_analysis import detect, track
from tests.test_peak_workspace import sample


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        from ui_qt.peak_analysis_bridge import PeakAnalysisBridge
        self.owner=QMainWindow();w=self.owner
        w.figure=Figure();w.canvas=ThemeAwareFigureCanvasQTAgg(w.figure);w.setCentralWidget(w.canvas)
        w.toolbar=QToolBar();w.addToolBar(w.toolbar)
        w.loaded=SimpleNamespace(mode='PL',cube=sample(),primary_file='a.csv',selected_files=['a.csv'])
        w.last_plotted_mode='PL';w._active_mode=lambda:w.loaded.mode;w._load_in_progress=False
        w._pl_last_plot_cube=w.loaded.cube;w._pl_heatmap_axes={'linear':w.figure.add_subplot(111)}
        w._status=lambda message:None
        self.bridge=PeakAnalysisBridge(w,w.toolbar)

    def wait(self):
        deadline=time.monotonic()+12
        idle=0
        while idle<2 and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.01)
            idle=idle+1 if self.bridge.job is None and not self.bridge.queued else 0
        self.assertIsNone(self.bridge.job)

    def tearDown(self):
        self.wait();self.owner.close();self.owner.deleteLater();self.app.processEvents()

    def test_snapshot_channels_are_actual_products_and_exclude_vp(self):
        w=self.owner
        self.assertEqual(self.bridge.sources()[0]['channel'],'PL')
        for mode, attr in [('DRR','_drr_plot_cubes'),('Compare','_cmp_active_cubes'),('Power Dependent','_power_active_cubes')]:
            w.loaded.mode=w.last_plotted_mode=mode
            setattr(w,attr,{'raw' if mode=='DRR' else 'KK':sample(), 'second' if mode=='DRR' else 'KKp':sample()})
            self.assertEqual(len(self.bridge.sources()),2)
        w._power_active_cubes={'VP':sample()}
        self.assertEqual(self.bridge.sources(),[])

    def test_derivative_provenance_uses_displayed_cache_not_pending_controls(self):
        w=self.owner;w.loaded.mode=w.last_plotted_mode='DRR';w._drr_plot_view='second'
        raw=sample('DRR');second=sample('DRR');second.Z*=2
        w.loaded.cube=raw;w._drr_plot_cubes={'raw':raw,'second':second}
        w._drr_derivative_cache={(id(raw),2,21,3):(second,19)}
        w.drr_sg_window_spin=SimpleNamespace(value=lambda:51)
        w.drr_sg_poly_spin=SimpleNamespace(value=lambda:4)
        sources={s['channel']:s for s in self.bridge.sources()}
        self.assertIn('source_processing',sources['second'])
        expected=dict(method='Savitzky–Golay',derivative=2,window_samples=19,polyorder=3)
        self.assertEqual(sources['second']['source_processing'],expected)
        self.assertNotIn('source_processing',sources['raw'])
        with tempfile.TemporaryDirectory() as folder, \
             patch('ui_qt.peak_analysis_bridge.session_directory',return_value=Path(folder)), \
             patch('ui_qt.peak_analysis_bridge.QProcess.startDetached',return_value=(True,123)) as launch:
            self.bridge.open();self.wait()
            arguments=launch.call_args.args[1]
            datasets,_=load_workspace(arguments[arguments.index('--snapshot')+1])
            self.assertEqual(next(d for d in datasets if d['channel']=='second')['source_processing'],expected)
        w._drr_derivative_cache.clear()
        self.assertNotIn('source_processing',next(s for s in self.bridge.sources() if s['channel']=='second'))

    def test_bridge_carries_each_drr_products_display_settings(self):
        from core.plotting import HeatmapParams,SplitColorScale,plot_heatmap
        from core.colormaps import register_colormaps,RDBU_R_P0P60_ID
        register_colormaps();w=self.owner
        w.loaded.mode=w.last_plotted_mode='DRR';w._drr_plot_view='raw'
        w._drr_plot_cubes={'raw':sample('DRR'),'second':sample('DRR')}
        w.figure.clear();w._drr_heatmap_axes={}
        for index,(channel,cmap,bounds) in enumerate([('raw','turbo',(-.4,1.2)),('second',RDBU_R_P0P60_ID,(-9.,13.))]):
            axis=w.figure.add_subplot(1,2,index+1);w._drr_heatmap_axes[channel]=axis
            plot_heatmap(axis,w._drr_plot_cubes[channel],HeatmapParams(channel,'Energy','Gate','DR/R',*bounds,
                (1.60,1.8),(-2,2),cmap=cmap,split_scale=SplitColorScale(1.7,*bounds,-.1,.2)))
        with tempfile.TemporaryDirectory() as folder,patch('ui_qt.peak_analysis_bridge.session_directory',return_value=Path(folder)), \
             patch('ui_qt.peak_analysis_bridge.QProcess.startDetached',return_value=(True,123)) as launch:
            self.bridge.open();self.wait()
            arguments=launch.call_args.args[1]
            datasets,active=load_workspace(arguments[arguments.index('--snapshot')+1])
            by_channel={d['channel']:d for d in datasets}
            self.assertEqual(active,by_channel['raw']['key'])
            self.assertEqual(by_channel['second']['view']['display']['cmap'],RDBU_R_P0P60_ID)
            self.assertEqual(by_channel['raw']['view']['display']['cmap'],'turbo')
            self.assertEqual(by_channel['second']['view']['display']['split_scale']['left_vmin'],-9.)
            self.assertEqual(by_channel['raw']['group_id'],by_channel['second']['group_id'])
            pair_group=by_channel['raw']['group_id']
            for channel in ('raw','second'):
                w._drr_plot_cubes={channel:by_channel[channel]['cube']}
                self.bridge.open();self.wait()
                arguments=launch.call_args.args[1]
                single,_=load_workspace(arguments[arguments.index('--snapshot')+1])
                self.assertEqual(single[0]['group_id'],pair_group)

    def test_pl_active_log_view_supplies_log_color_scale(self):
        from core.plotting import HeatmapParams,plot_heatmap
        w=self.owner;w.figure.clear();w._pl_heatmap_axes={};w._pl_active_scale=True
        for index,scale in enumerate(('linear','log')):
            axis=w.figure.add_subplot(1,2,index+1);w._pl_heatmap_axes[scale]=axis
            plot_heatmap(axis,w.loaded.cube,HeatmapParams(scale,'Energy','Gate','Intensity',.1,1.2,
                (1.60,1.8),(-2,2),cmap='plasma' if scale=='log' else 'viridis',log_scale=scale=='log'))
        display=self.bridge.sources()[0]['view']['display']
        self.assertEqual(display['cmap'],'plasma');self.assertTrue(display['log_scale'])

    def test_overlay_rejects_changed_data_but_survives_color_and_view(self):
        source=self.bridge.sources()[0]
        d=create_dataset(source['cube'],source['kind'],source['channel'],source['source'],source['name'])
        d['branches']=detect(d,4,d['settings']);d['result']=track(d,d['branches'],d['settings'])
        packet={k:v for k,v in d.items() if k!='cube'}
        self.bridge.accept_overlay(packet);self.wait()
        self.assertTrue(self.bridge.artists)
        self.owner._pl_heatmap_axes['linear'].set_xlim(1.62,1.79)
        self.bridge.refresh();self.wait();self.assertTrue(self.bridge.artists)
        self.owner._pl_last_plot_cube=sample();self.owner._pl_last_plot_cube.Z[0,0]+=1
        self.bridge.refresh();self.wait();self.assertEqual(self.bridge.artists,[])

    def test_decreasing_energy_content_identity_is_canonical(self):
        from dataclasses import replace
        cube=sample();reverse=replace(cube,energy=cube.energy[::-1],Z=cube.Z[:,::-1])
        d=create_dataset(reverse,'PL','PL','a.csv','a')
        self.assertEqual(fingerprint(reverse,'PL','PL','a.csv'),d['key'])

    def test_overlay_is_removed_when_displayed_values_change_in_place(self):
        source=self.bridge.sources()[0]
        d=create_dataset(source['cube'],source['kind'],source['channel'],source['source'],source['name'])
        d['branches']=detect(d,4,d['settings']);d['result']=track(d,d['branches'],d['settings'])
        self.bridge.accept_overlay({k:v for k,v in d.items() if k!='cube'});self.wait()
        self.assertTrue(self.bridge.artists)
        source['cube'].Z[:]=0
        self.bridge.refresh();self.wait()
        self.assertEqual(self.bridge.artists,[])

    def test_data_change_during_background_hash_is_checked_again(self):
        source=self.bridge.sources()[0]
        d=create_dataset(source['cube'],source['kind'],source['channel'],source['source'],source['name'])
        d['branches']=detect(d,4,d['settings']);d['result']=track(d,d['branches'],d['settings'])
        self.bridge.accept_overlay({k:v for k,v in d.items() if k!='cube'});self.wait()
        computed=Event();resume=Event()
        def delayed(*args):
            value=fingerprint(*args);computed.set();resume.wait(4);return value
        try:
            with patch('ui_qt.peak_analysis_bridge.fingerprint',side_effect=delayed):
                self.bridge.refresh();self.assertTrue(computed.wait(4))
                source['cube'].Z[:]=0
                self.bridge.refresh();resume.set();self.wait()
                self.assertEqual(self.bridge.artists,[])
        finally:resume.set()


class MainAppBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_queued_refresh_is_cancelled_when_owner_is_destroyed(self):
        import sys
        from PySide6.QtWidgets import QTabWidget
        from shiboken6 import delete
        from ui_qt.peak_analysis_bridge import PeakAnalysisBridge

        owner=QMainWindow()
        owner.canvas=ThemeAwareFigureCanvasQTAgg(Figure());owner.setCentralWidget(owner.canvas)
        toolbar=QToolBar(owner);owner.addToolBar(toolbar)
        tabs=QTabWidget(owner);owner._active_mode=lambda:tabs.currentIndex()
        bridge=PeakAnalysisBridge(owner,toolbar)
        errors=[]
        with patch.object(sys,'excepthook',side_effect=lambda *args:errors.append(args)):
            bridge.schedule_refresh()
            delete(owner)
            self.app.processEvents()
        self.assertEqual(errors,[],[str(error[1]) for error in errors])

    def test_worker_callbacks_are_ignored_after_owner_is_destroyed(self):
        import sys
        from shiboken6 import delete
        from ui_qt.peak_analysis_bridge import PeakAnalysisBridge

        for fail in (False,True):
            with self.subTest(fail=fail):
                owner=QMainWindow()
                owner.canvas=ThemeAwareFigureCanvasQTAgg(Figure());owner.setCentralWidget(owner.canvas)
                toolbar=QToolBar(owner);owner.addToolBar(toolbar)
                results=[];messages=[];errors=[];owner._status=messages.append
                bridge=PeakAnalysisBridge(owner,toolbar)
                def work():
                    if fail:raise ValueError('Analysis failed')
                    return 'Analysis completed'
                with patch.object(sys,'excepthook',side_effect=lambda *args:errors.append(args)):
                    bridge.start(work,results.append)
                    finished=bridge.pool.waitForDone(3000)
                    delete(owner)
                    self.app.processEvents()
                self.assertTrue(finished)
                self.assertEqual(results,[])
                self.assertEqual(messages,[])
                self.assertEqual(errors,[],[str(error[1]) for error in errors])

    def test_toolbar_launch_passes_current_map_without_opening_bottom_docks(self):
        from tests.test_pl_dual_view import PlDualViewTests
        fixture=PlDualViewTests();fixture.setUp()
        try:
            w=fixture.w;w._plot_mode('PL');w.canvas.draw();self.app.processEvents()
            bridge=w.peak_analysis_bridge
            before=(w.log_dock.isHidden(),w.results_dock.isHidden())
            with tempfile.TemporaryDirectory() as folder, \
                 patch('ui_qt.peak_analysis_bridge.session_directory',return_value=Path(folder)), \
                 patch('ui_qt.peak_analysis_bridge.QProcess.startDetached',return_value=(True,123)) as launch:
                bridge.button.click()
                deadline=time.monotonic()+12
                while bridge.job and time.monotonic()<deadline:
                    self.app.processEvents();time.sleep(.01)
                self.assertIsNone(bridge.job);launch.assert_called_once()
                arguments=launch.call_args.args[1]
                datasets,_=load_workspace(arguments[arguments.index('--snapshot')+1])
                self.assertEqual(len(datasets),1)
                self.assertEqual(datasets[0]['kind'],'PL')
                import numpy as np
                np.testing.assert_array_equal(datasets[0]['cube'].Z,w._pl_last_plot_cube.Z)
                self.assertEqual(before,(w.log_dock.isHidden(),w.results_dock.isHidden()))
        finally:fixture.doCleanups()


if __name__=='__main__':unittest.main()
