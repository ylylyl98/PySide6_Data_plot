"""Isolated Quick P/N batch selection and workflow tests."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class QuickPnBatchUiTests(unittest.TestCase):
    def test_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', QT_QUICK_BACKEND='software',
                       MPLBACKEND='agg', LOCALAPPDATA=tmp, XDG_CONFIG_HOME=tmp, XDG_CACHE_HOME=tmp)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--isolated'],
                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

def isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import time
    from threading import Event
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QSettings, QUrl
    from PySide6.QtGui import QGuiApplication, QFont, QFontDatabase
    from PySide6.QtQml import QQmlComponent, QQmlEngine
    from PySide6.QtQuick import QQuickWindow
    from ui_qt.quick.plot_surface import register_plot_surface
    from core.drr_pn_reconstruction import pipeline
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
    from ui_qt.quick.background_source_catalog import BackgroundSourceCatalogBridge
    app = QGuiApplication.instance() or QGuiApplication([])
    if os.name == 'nt':
        QFontDatabase.addApplicationFont(str(Path(os.environ['WINDIR'])/'Fonts'/'segoeui.ttf'))
        app.setFont(QFont('Segoe UI',9))
    register_plot_surface()
    class Cases(unittest.TestCase):
        def test_role_picker_multiple_opt_in_and_reset(self):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); chooser=BackgroundSourceCatalogBridge(settings=QSettings(str(root/'settings.ini'),QSettings.IniFormat))
                self.addCleanup(chooser.close)
                first,second,third=[str(root / ('Pbackground_%d.csv' % i)) for i in range(3)]
                requests=[]; accepted=[]
                chooser.dialogRequested.connect(lambda role,multiple: requests.append((role,multiple)))
                chooser.selectionAccepted.connect(lambda workflow,items: accepted.append(items))
                chooser.open_selection(root,'pn',[],role='P',multiple=True)
                chooser.selectFile(first,True);chooser.selectFile(second,True)
                chooser.addExternal([second,third]);chooser.invoke('browse')
                self.assertTrue(requests[-1][1]);self.assertTrue(chooser.snapshot['canAccept'])
                self.assertTrue(chooser.snapshot['multipleSelection'])
                chooser.invoke('accept');self.assertEqual(accepted[-1],[first,second,third])
                chooser.open_selection(root,'pn',[],role='P')
                chooser.selectFile(first,True);chooser.selectFile(second,True);chooser.invoke('browse')
                self.assertFalse(requests[-1][1]);self.assertFalse(chooser.snapshot['multipleSelection'])
                chooser.invoke('accept');self.assertEqual(accepted[-1],[second])
    class OwnerCases(unittest.TestCase):
        prefix='YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
        def setUp(self):
            self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
            self.root=Path(self.tmp.name)
            self.samples=[self.write('TG_rep01',1000),self.write('TG_rep02',2000)]
            self.ps=[self.write('Pbackground_001',900,-10),self.write('Pbackground_002',1900,-10)]
            self.ns=[self.write('Nbackground_001',905,10),self.write('Nbackground_002',1905,10)]
            self.owner=QuickApplicationBridge(settings=QSettings(str(self.root/'settings.ini'),QSettings.IniFormat))
            self.addCleanup(self.owner.close)
            patcher=patch.object(self.owner,'_schedule_plot');patcher.start();self.addCleanup(patcher.stop)
            self.owner.selectWorkflow('drr')
            self.owner.selectFiles([QUrl.fromLocalFile(str(self.root/name)) for name in self.samples])
            self.wait(lambda:not self.owner._pending and not self.owner.snapshot['busy'])
            self.owner.openDrrBackground();self.bg=self.owner.drrBackgroundBridge
            preview=patch.object(self.bg,'_draw_preview');preview.start();self.addCleanup(preview.stop)
            self.bg.setChoice('kind',3);self.bg.setPnMode('create')
        def write(self,token,stamp,gate=None):
            path=self.root/(self.prefix+token+'.csv');wave=np.linspace(630.,690.,41)
            gates=np.linspace(-2,2,9) if gate is None else np.full(4,gate)
            values=150+np.linspace(-1,1,len(wave))[None,:]**2+np.zeros((len(gates),len(wave)))
            if gate is None: values+=5*gates[:,None]
            np.savetxt(path,np.column_stack((gates,gates,values)),delimiter=',',header='Vbg,Vtg,'+','.join(map(str,wave)),comments='')
            os.utime(path,(stamp,stamp));return path.name
        def wait(self,predicate,timeout=30):
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                app.processEvents()
                if predicate(): app.processEvents();return
                time.sleep(.002)
            self.fail('Timed out: '+str(self.bg.snapshot if hasattr(self,'bg') else self.owner.snapshot))
        @staticmethod
        def stub(energy,pn_mean,frames,samples,**kwargs):
            if kwargs['method']=='conditioned_single':
                algorithm,protocol=pipeline.CONDITIONED_SINGLE_ALGORITHM_VERSION,pipeline.CONDITIONED_SINGLE_PROTOCOL
            else:
                algorithm,protocol=pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION,pipeline.CONDITIONED_BOUNDED_PROTOCOL
            return {'energy':energy.copy(),'background':pn_mean.mean(0)}, {'algorithm_version':algorithm,'settings':dict(protocol),'BN_used':False}
        def plan(self):
            self.bg.selectFiles('P',[str(self.root/name) for name in self.ps])
            self.bg.selectFiles('N',[str(self.root/name) for name in self.ns])
            self.wait(lambda:not self.bg.snapshot['pnBatchLoading'] and bool(self.bg.snapshot['pnBatchRows']))
            self.assertTrue(self.bg.snapshot['pnBatchActive'],self.bg.snapshot)
            self.assertEqual(self.bg.snapshot['pnBatchUnresolved'],0,self.bg.snapshot)
        def test_batch_generate_preview_unified_owner_apply(self):
            self.plan()
            rows=self.bg.snapshot['pnBatchRows']
            self.assertEqual(len(rows),2)
            self.assertNotEqual((rows[0]['p'],rows[0]['n']),(rows[1]['p'],rows[1]['n']))
            with patch.object(pipeline,'reconstruct_background',side_effect=self.stub):
                self.bg.invoke('pn-generate')
                self.wait(lambda:not self.bg.snapshot['busy'])
            self.assertEqual(self.bg.snapshot['error'],'')
            self.assertTrue(self.bg.snapshot['pnPreviewReady'])
            self.assertEqual(len(self.bg.snapshot['pnBatchPreviewItems']),2)
            self.bg.setPnBatchPreview(1);self.assertEqual(self.bg.snapshot['pnBatchPreviewIndex'],1)
            self.bg.acknowledgePnBatchWarnings(True)
            self.bg.invoke('pn-apply')
            self.wait(lambda:not self.bg.snapshot['busy'] and not self.owner._pending and not self.owner.snapshot['busy'])
            self.assertTrue(self.owner.snapshot['canExport'],self.owner.snapshot)
            self.assertEqual(len(self.owner._drr_background_recipe.assignments),2)

        def test_large_pair_pool_keeps_a_short_list_and_retains_manual_choice(self):
            self.ps += [self.write('Pbackground_00'+str(i), 400+i*25, -10) for i in range(3,7)]
            self.ns += [self.write('Nbackground_00'+str(i), 410+i*25, 10) for i in range(3,6)]
            self.plan()
            row = self.bg.snapshot['pnBatchRows'][0]
            self.assertEqual(row['totalOptions'], 30)
            self.assertTrue(row['moreOptions'])
            self.assertLessEqual(len(row['options']), 26)
            self.bg.setPnBatchPair(row['measurement'], 29)
            self.wait(lambda: not self.bg.snapshot['busy'])
            row = self.bg.snapshot['pnBatchRows'][0]
            self.assertEqual(row['optionIndices'][row['current']], 29)
            self.bg.showAllPnBatchPairs(row['measurement'])
            self.assertEqual(len(self.bg.snapshot['pnBatchRows'][0]['options']), 30)
            self.bg.refreshPnBatchAssignments()
            self.wait(lambda: not self.bg.snapshot['pnBatchLoading'])
            self.assertFalse(self.bg.snapshot['pnBatchRows'][0]['manual'])
        def test_rendered_collapsed_expanded_preview_and_theme_sizes(self):
            self.plan()
            light=dict(surface='#f3f5f8', control='#ffffff', text='#20242b', muted='#606978', border='#c6cbd3',
                accent='#0067c0', hover='#edf1f6', pressed='#e0e7ef', error='#b42318', disabled='#939aa5',
                selectedText='#ffffff', inset='#e9edf2', divider='#dce1e8', borderHover='#aeb7c4', borderDisabled='#e1e5eb',
                controlDisabled='#edf0f4', focus='#20242b', accentText='#0067c0', selected='#e6f0fa', selectedBorder='#b4d6fa', errorSurface='#fdf1f0')
            dark=dict(light,surface='#202020',control='#303030',text='#ffffff',muted='#b0b0b0',border='#606060',hover='#404040',pressed='#505050',inset='#252525',divider='#505050')
            engine=QQmlEngine();warnings=[]
            engine.warnings.connect(lambda messages:warnings.extend(m.toString() for m in messages))
            source=Path(__file__).resolve().parents[1]/'ui_qt'/'quick'/'qml'/'DrrBackgroundSettings.qml'
            component=QQmlComponent(engine,QUrl.fromLocalFile(str(source)))
            item=component.createWithInitialProperties(dict(bridge=self.bg,colors=light))
            self.assertIsNotNone(item,'\n'.join(e.toString() for e in component.errors()))
            window=QQuickWindow();item.setParentItem(window.contentItem());window.show()
            def close(engine=engine, component=component):
                window.close();item.setParentItem(None);item.deleteLater();app.processEvents()
            self.addCleanup(close)
            def find(name):
                stack=[item]
                while stack:
                    child=stack.pop()
                    if child.objectName()==name:return child
                    stack.extend(child.childItems())
                self.fail('Missing QML object '+name)
            def resize(width,height):
                window.resize(width,height);item.setWidth(width);item.setHeight(height)
                self.wait(lambda:not window.grabWindow().isNull())
                app.processEvents()
            for label,colors,size in [('light',light,(1060,800)),('dark',dark,(1060,800)),('narrow',light,(760,600)),('vertical',light,(600,600))]:
                item.setProperty('colors',colors);resize(*size)
                summary=find('drr_pn_batch_summary');self.assertTrue(summary.isVisible())
                self.assertIn('2/2',summary.property('text'))
                self.assertFalse(find('drr_pn_batch_rows').isVisible())
                self.assertTrue(window.grabWindow().save(str(Path(tempfile.gettempdir())/('drr-pn-batch-'+label+'.png'))))
            find('drr_pn_batch_expand').clicked.emit();app.processEvents()
            self.assertTrue(find('drr_pn_batch_rows').isVisible())
            self.assertTrue(find('drr_pn_batch_pair_0').isVisible())
            row=self.bg.snapshot['pnBatchRows'][0]
            target=next(i for i in range(len(row['options'])) if i != row['current'])
            find('drr_pn_batch_pair_0').activated.emit(target)
            self.wait(lambda:not self.bg.snapshot['busy'])
            self.assertTrue(self.bg.snapshot['pnBatchRows'][0]['manual'])
            with patch.object(pipeline,'reconstruct_background',side_effect=self.stub):
                self.bg.invoke('pn-generate');self.wait(lambda:not self.bg.snapshot['busy'])
            self.assertTrue(self.bg.snapshot['pnPreviewReady'],self.bg.snapshot['error'])
            self.assertTrue(find('drr_pn_batch_preview').isVisible())
            resize(1060,800)
            DrrBackgroundBridge._draw_preview(self.bg)
            self.wait(lambda:self.bg.plotBackend is not None and self.bg.plotBackend.frameGeneration > 0)
            first_generation=self.bg.plotBackend.frameGeneration
            self.bg.setPnBatchPreview(1)
            DrrBackgroundBridge._draw_preview(self.bg)
            self.wait(lambda:self.bg.plotBackend.frameGeneration > first_generation)
            self.assertEqual(self.bg.snapshot['pnBatchPreviewIndex'],1)
            self.assertTrue(window.grabWindow().save(str(Path(tempfile.gettempdir())/'drr-pn-batch-expanded.png')))
            self.assertTrue(window.grabWindow().save(str(Path(tempfile.gettempdir())/'drr-pn-batch-real-plot.png')))
            resize(600,600)
            self.assertTrue(window.grabWindow().save(str(Path(tempfile.gettempdir())/'drr-pn-batch-vertical-plot.png')))
            self.assertFalse(warnings,'\n'.join(warnings))

        def test_manual_override_invalidates_preview_and_requires_review(self):
            self.plan()
            row=self.bg.snapshot['pnBatchRows'][0]
            self.assertGreater(len(row['options']),1)
            target=next(i for i in range(len(row['options'])) if i != row['current'])
            self.bg.setPnBatchPair(row['measurement'],target)
            self.wait(lambda:not self.bg.snapshot['busy'])
            updated=self.bg.snapshot['pnBatchRows'][0]
            self.assertEqual(updated['current'],target)
            self.assertTrue(updated['manual'])
            self.assertFalse(self.bg.snapshot['pnCanApply'])
            self.assertFalse(self.bg.snapshot['pnBatchAcknowledged'])
        def test_old_planner_cannot_override_changed_source_pool(self):
            started,release=Event(),Event();calls=[]
            from core.drr_pn_batch_plan import build_pn_batch_plan as real_plan
            def delayed(*args,**kwargs):
                calls.append(tuple(kwargs['p_candidates']))
                if len(calls)==1:
                    started.set();release.wait(5)
                return real_plan(*args,**kwargs)
            with patch('core.drr_pn_batch_plan.build_pn_batch_plan',side_effect=delayed):
                self.bg.selectFiles('P',[str(self.root/name) for name in self.ps])
                self.bg.selectFiles('N',[str(self.root/name) for name in self.ns])
                self.wait(started.is_set)
                self.bg.selectFiles('P',[str(self.root/self.ps[1]),str(self.root/self.ps[0])])
                release.set()
                self.wait(lambda:len(calls)==2 and not self.bg.snapshot['pnBatchLoading'])
                self.assertEqual(calls[-1],tuple(reversed(self.ps)))
                self.assertEqual(self.bg._pn_batch_plan.p_candidates,tuple(reversed(self.ps)))

        def test_old_planner_result_does_not_replace_new_context(self):
            started,release=Event(),Event()
            from core.drr_pn_batch_plan import build_pn_batch_plan as real_plan
            def delayed(*args,**kwargs):
                started.set();release.wait(5);return real_plan(*args,**kwargs)
            with patch('core.drr_pn_batch_plan.build_pn_batch_plan',side_effect=delayed):
                self.bg.selectFiles('P',[str(self.root/name) for name in self.ps])
                self.bg.selectFiles('N',[str(self.root/name) for name in self.ns])
                self.wait(started.is_set)
                from ui_qt.quick.drr_background_workflow import BackgroundContext,BackgroundDraft
                new=BackgroundContext(str(self.root),(self.samples[0],))
                self.bg.open(new,BackgroundDraft(kind='pn'))
                release.set()
                deadline=time.monotonic()+.7
                while time.monotonic()<deadline: app.processEvents();time.sleep(.003)
                self.assertEqual(self.bg.context.measurements,(self.samples[0],))
                self.assertFalse(self.bg.snapshot['pnBatchRows'])
                self.assertFalse(self.bg.snapshot['pnBatchActive'])

        def test_failure_keeps_previous_owner_selection(self):
            self.plan(); prior=(self.owner._drr_background_recipe,self.owner._drr_products)
            with patch.object(pipeline,'reconstruct_background',side_effect=ValueError('forced failure')):
                self.bg.invoke('pn-generate');self.wait(lambda:not self.bg.snapshot['busy'])
            self.assertIn('forced failure',self.bg.snapshot['error'])
            self.assertFalse(self.bg.snapshot['pnCanApply'])
            self.assertEqual((self.owner._drr_background_recipe,self.owner._drr_products),prior)
        def test_legacy_one_pair_not_batch(self):
            self.bg.selectFiles('P',[str(self.root/self.ps[0])]);self.bg.selectFiles('N',[str(self.root/self.ns[0])])
            self.assertFalse(self.bg.snapshot['pnBatchActive'])
            self.assertTrue(self.bg.snapshot['pnCreateReady'])

    suite=unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(Cases),unittest.defaultTestLoader.loadTestsFromTestCase(OwnerCases)))
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
if __name__=='__main__':
    if '--isolated' in sys.argv: sys.exit(0 if isolated() else 1)
    unittest.main()
