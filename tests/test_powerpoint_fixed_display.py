import os
from pathlib import Path
import subprocess
import sys
import unittest


class PowerPointFixedDisplayTests(unittest.TestCase):
    def test_in_isolated_gui_process(self):
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen', MPLBACKEND='agg')
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--isolated'],
            cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tempfile
    import time
    from dataclasses import replace
    import numpy as np
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QGuiApplication
    from core.loader import DataCube
    from core.plotting import HeatmapParams, SplitColorScale
    from ui_qt.quick.drr_display_state import DrrDisplayState
    from ui_qt.quick.powerpoint_application import restore_drr_figure
    _APP = QGuiApplication.instance() or QGuiApplication([])

    class PowerPointDrrFixTests(unittest.TestCase):
        def setUp(self):
            self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
            self.settings=QSettings(str(Path(self.tmp.name)/'prefs.ini'),QSettings.IniFormat)
            self.display=DrrDisplayState(settings=self.settings);self.addCleanup(self.display.close)
            x=np.linspace(1.,4.,7);y=np.linspace(-1.,1.,5)
            self.cube=DataCube(x,y,np.ones((5,7)), 'Gate','Old','DRR')
            self.display.sync_cube(self.cube,self.cube)
            self.params=HeatmapParams('new','Energy','Gate','DRR',0.,5.,(1.,4.),(-1.,1.))
            self.display._ppt_previous_y_identity=('gate','','')

        def test_manual_fixed_x_y_colors_override_saved_figure(self):
            state=self.display
            for key,value in [('axis.xmin','1.5'),('axis.xmax','3.5'),('axis.ymin','-0.5'),('axis.ymax','0.5'),('raw.vmin','0.5'),('raw.vmax','2')]:
                self.assertTrue(state.setNumber(key,value));self.assertTrue(state.setFlag('fix.'+key,True))
            restore_drr_figure(state,self.params)
            params=state.heatmap_params(self.cube)
            self.assertEqual(params.xlim,(1.5,3.5));self.assertEqual(params.ylim,(-.5,.5))
            self.assertEqual((params.vmin,params.vmax),(.5,2.))
            self.assertTrue(all(state._flags['fix.'+key] for key in ('axis.xmin','axis.xmax','raw.vmin','raw.vmax')))

        def test_borrowed_fixes_are_not_adopted(self):
            state=self.display
            state._memory._borrow('numbers','axis.xmin',1.5,origin='F')
            state._memory._borrow('flags','fix.axis.xmin',True,origin='F')
            restore_drr_figure(state,self.params)
            self.assertEqual(state.heatmap_params(self.cube).xlim,(1.,4.))
            self.assertFalse(state._flags['fix.axis.xmin'])

        def test_incompatible_y_lock_not_applied(self):
            state=self.display
            state.setNumber('axis.ymin','-.5');state.setFlag('fix.axis.ymin',True)
            state._ppt_previous_y_identity=('bias','v','bias')
            restore_drr_figure(state,self.params)
            self.assertEqual(state.heatmap_params(self.cube).ylim,(-1.,1.))
            self.assertFalse(state._flags['fix.axis.ymin'])
            self.assertIn('Y Fix not applied',state._memory_status)

        def test_manual_second_region_fix_maps_to_imported_raw(self):
            state=self.display
            state.setChoice('regionCount',1)
            state.setNumber('boundary.x0','2');state.setFlag('fix.boundary.x0',True)
            state.setNumber('second.left_vmin','-0.5');state.setFlag('fix.second.left_vmin',True)
            params=replace(self.params,split_scale=SplitColorScale(2.5,-1.,5.,-1.,5.))
            restore_drr_figure(state,params,product='second')
            result=state.heatmap_params(self.cube)
            self.assertEqual(result.split_scale.split_x,2.)
            self.assertEqual(result.split_scale.left_vmin,-.5)

        def test_one_fixed_bound_preserved_when_saved_peer_conflicts(self):
            state=self.display
            state.setNumber('axis.xmin','2.5');state.setFlag('fix.axis.xmin',True)
            restore_drr_figure(state,replace(self.params,xlim=(1.,2.)))
            self.assertEqual(state._numbers['axis.xmin'],2.5)
            self.assertGreater(state._numbers['axis.xmax'],2.5)
            self.assertTrue(state._flags['fix.axis.xmin'])

        def test_repeated_second_figure_keeps_user_fixed_imported_raw(self):
            state=self.display
            state.setNumber('second.left_vmin','-.5');state.setFlag('fix.second.left_vmin',True)
            params=replace(self.params,split_scale=SplitColorScale(2.5,-1.,5.,-1.,5.))
            restore_drr_figure(state,params,product='second')
            state._ppt_previous_figure_mode=True
            state._raw=self.cube
            restore_drr_figure(state,params,product='second')
            self.assertEqual(state.heatmap_params(self.cube).split_scale.left_vmin,-.5)
            self.assertTrue(state._flags['fix.raw.left_vmin'])

        def test_invalid_boundary_fails_without_partial_state_changes(self):
            state=self.display
            state.setChoice('regionCount',1)
            state.setNumber('boundary.x0','2');state.setFlag('fix.boundary.x0',True)
            prior=dict(state._numbers); flags=dict(state._flags)
            params=replace(self.params,xlim=(2.5,4.),split_scale=SplitColorScale(3.,-1.,5.,-1.,5.))
            with self.assertRaises(ValueError):restore_drr_figure(state,params)
            self.assertEqual(state._numbers,prior);self.assertEqual(state._flags,flags)

        def test_unused_region_values_not_validated(self):
            state=self.display;state._numbers['raw.middle_vmin']=3.;state._numbers['raw.middle_vmax']=1.
            restore_drr_figure(state,self.params)
            self.assertEqual(state.heatmap_params(self.cube).xlim,(1.,4.))

        def test_actual_powerpoint_open_applies_fixed_values(self):
            import json,time
            from dataclasses import asdict
            from PIL import Image
            from core.pptx_figure_index import FigureIndex
            from ui_qt.quick.application_bridge import QuickApplicationBridge
            folder=Path(self.tmp.name)/'Processed Data'/'DRR';folder.mkdir(parents=True)
            dat=folder/'map.dat';dat.write_text('X\t-1\t0\t1\n1\t1\t2\t3\n2\t2\t3\t4\n3\t3\t4\t5\n4\t4\t5\t6\n')
            png=folder/'map.png';Image.new('RGB',(10,10),'red').save(png)
            (folder/'map.metadata.json').write_text(json.dumps({'operation':'DR/R','outputs':[png.name,dat.name],
                'figures':{png.name:{'data_file':dat.name,'plot':asdict(self.params),'role':'second'}}}))
            record=FigureIndex(self.tmp.name).find_png(png)[0]
            bridge=QuickApplicationBridge(settings=self.settings)
            self.addCleanup(bridge.close)
            bridge.selectWorkflow('drr')
            bridge.set_cube(self.cube)
            state=bridge._drr_display
            state.setNumber('axis.xmin','1.5');state.setFlag('fix.axis.xmin',True)
            bridge._ppt_open(record)
            deadline=time.monotonic()+30
            while bridge._busy and time.monotonic()<deadline:
                _APP.processEvents();time.sleep(.01)
            self.assertFalse(bridge._busy)
            self.assertIsNotNone(bridge._ppt_record,bridge._status)
            self.assertEqual(state.heatmap_params(bridge._cube).xlim[0],1.5)
            self.assertTrue(state._flags['fix.axis.xmin'])

        def test_unfixed_derivative_color_uses_saved_figure(self):
            state=self.display
            state._second_auto_scale=False
            state._numbers.update({'second.vmin': -5., 'second.vmax': 10.})
            restore_drr_figure(state,self.params,product='second')
            params=state.heatmap_params(self.cube)
            self.assertEqual((params.vmin,params.vmax),(0.,5.))

        def test_all_fixed_regions_boundaries_and_saved_options(self):
            for product in ('raw', 'second'):
                with self.subTest(product=product):
                    state = DrrDisplayState(settings=QSettings(str(Path(self.tmp.name)/(product+'.ini')), QSettings.IniFormat))
                    try:
                        state.sync_cube(self.cube, self.cube)
                        state._ppt_previous_y_identity=('gate','','')
                        self.assertTrue(state.setChoice('regionCount', 2))
                        self.assertTrue(state.setNumber('boundary.x0', '2'))
                        self.assertTrue(state.setNumber('boundary.x1', '3'))
                        for key in ('boundary.x0', 'boundary.x1'):
                            self.assertTrue(state.setFlag('fix.'+key, True))
                        for index, side in enumerate(('left','middle','right')):
                            lo, hi = -(index+2.), index+2.
                            self.assertTrue(state.setNumber(product+'.'+side+'_vmin', str(lo)))
                            self.assertTrue(state.setNumber(product+'.'+side+'_vmax', str(hi)))
                            self.assertTrue(state.setFlag('fix.'+product+'.'+side+'_vmin', True))
                            self.assertTrue(state.setFlag('fix.'+product+'.'+side+'_vmax', True))
                        split=SplitColorScale(2.2,-10.,10.,-11.,11.,False,3.2,-12.,12.)
                        saved=replace(self.params,split_scale=split,cmap='viridis',center_zero=True,clip_outliers=False)
                        restore_drr_figure(state,saved,product=product)
                        got=state.heatmap_params(self.cube)
                        self.assertEqual((got.split_scale.split_x,got.split_scale.split_x2),(2.,3.))
                        self.assertEqual(got.split_scale.bounds,((-2.,2.),(-3.,3.),(-4.,4.)))
                        self.assertEqual((got.cmap,got.log_scale,got.center_zero,got.clip_outliers),('viridis',False,True,False))
                        for side in ('left','middle','right'):
                            for bound in ('vmin','vmax'):
                                self.assertTrue(state._flags['fix.raw.'+side+'_'+bound])
                        self.assertTrue(state._flags['fix.boundary.x0']);self.assertTrue(state._flags['fix.boundary.x1'])
                    finally: state.close()

        def test_saved_log_scale_and_boundary_visibility_without_locks(self):
            split=SplitColorScale(2.5,.1,5.,.2,6.,False)
            saved=replace(self.params,vmin=.1,vmax=6.,split_scale=split,cmap='plasma',log_scale=True,clip_outliers=True)
            restore_drr_figure(self.display,saved)
            got=self.display.heatmap_params(self.cube)
            self.assertTrue(got.log_scale);self.assertTrue(got.clip_outliers)
            self.assertEqual(got.cmap,'plasma');self.assertFalse(got.split_scale.show_boundary)
            self.assertEqual(got.split_scale.bounds,((.1,5.),(.2,6.)))

        def test_saved_region_count_and_boundary_visibility_override_current_options(self):
            state=self.display
            state.setChoice('regionCount',1)
            state.setNumber('boundary.x0','2')
            state.setFlag('fix.boundary.x0',True)
            state.setFlag('boundary.show',False)
            split=SplitColorScale(2.2,-1.,5.,-1.,5.,True,3.2,-1.,5.)
            restore_drr_figure(state,replace(self.params,split_scale=split))
            got=state.heatmap_params(self.cube)
            self.assertEqual(state._choices['regionCount'],2)
            self.assertTrue(got.split_scale.show_boundary)
            self.assertEqual(got.split_scale.split_x,2.)
            self.assertEqual(got.split_scale.split_x2,3.2)
            self.assertTrue(state._flags['fix.boundary.x0'])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PowerPointDrrFixTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    if '--isolated' in sys.argv:
        raise SystemExit(isolated())
    unittest.main()
