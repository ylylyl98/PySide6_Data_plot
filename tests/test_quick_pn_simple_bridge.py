"""P/N simple workflow state in an isolated Qt process."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
class QuickPnSimpleProcessTest(unittest.TestCase):
    def test_isolated(self):
        with tempfile.TemporaryDirectory() as root:
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',MPLBACKEND='agg',LOCALAPPDATA=root,XDG_CONFIG_HOME=root)
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--isolated'],cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
def isolated():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from threading import Event
    import time
    from types import SimpleNamespace
    from unittest.mock import patch
    from PySide6.QtCore import QCoreApplication,QSettings
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
    from ui_qt.quick.drr_background_workflow import BackgroundContext,BackgroundDraft
    app=QCoreApplication.instance() or QCoreApplication([])
    class Cases(unittest.TestCase):
        def setUp(self):
            self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
            self.root=Path(self.tmp.name)
            self.bridge=DrrBackgroundBridge(settings=QSettings(str(self.root/'s.ini'),QSettings.Format.IniFormat));self.addCleanup(self.bridge.close)
        def wait(self,fn):
            deadline=time.monotonic()+8
            while not fn() and time.monotonic()<deadline: app.processEvents();time.sleep(.005)
            self.assertTrue(fn(),self.bridge.snapshot)
        def test_training_defaults_manual_edits_and_insufficient(self):
            b=self.bridge;names=('sample_rep01.csv','sample_rep02.csv','sample_rep03.csv')
            b.open(BackgroundContext(str(self.root),names),BackgroundDraft(kind='pn'))
            self.assertEqual(b.snapshot['pnMode'],'create')
            self.assertEqual(b._training,{names[0],names[2]})
            self.assertIn('1 and 3',b.snapshot['pnTrainingSummary'])
            b.setFlag('manual_training',True);b.setFlag('training:'+names[2],False);b.setFlag('training:'+names[1],True)
            b._auto_training();self.assertEqual(b._training,{names[0],names[1]})
            b.open(BackgroundContext(str(self.root),names[:1]),BackgroundDraft(kind='pn'))
            self.assertTrue(b.snapshot['pnTrainingReady']);self.assertIn('Single measurement',b.snapshot['pnTrainingSummary'])
        def test_history_cannot_overwrite_new_selection_or_previous_context(self):
            b=self.bridge;started,release=Event(),Event()
            version=SimpleNamespace(fresh=True,status='fresh',number=1,version_id='v1',csv_path='dummy',created_at='2025-01-01T00:00:00Z')
            def history(*args): started.set();release.wait(5);return (version,)
            with patch('core.drr_pn_store.DrrPnStore.list_versions',history),patch('core.drr_pn_store.DrrPnStore.find_applied',return_value=version),patch('core.drr_composite_background.read_composite_background',return_value=object()):
                b.open(BackgroundContext(str(self.root),('a_rep01.csv','a_rep02.csv')),BackgroundDraft(kind='pn'))
                self.wait(started.is_set)
                b.selectFiles('P',['Pbackground.csv'])
                release.set();self.wait(lambda:not b.snapshot['pnHistoryLoading'])
                self.assertIsNone(b._version);self.assertEqual(b._pn_files['P'],('Pbackground.csv',))
        def test_single_optin_and_raw_source_validation(self):
            from ui_qt.drr_pn_repeat_selection import choose_training_repeats
            raw='sample_rep01.csv'
            self.assertFalse(choose_training_repeats(self.root,(raw,)).usable)
            result=choose_training_repeats(self.root,(raw,),allow_single=True)
            self.assertTrue(result.usable)
            self.assertTrue(result.single_measurement)
            self.assertEqual(result.training_files,(raw,))
            self.assertFalse(choose_training_repeats(self.root,('Pbackground.csv',),allow_single=True).usable)
            self.assertFalse(choose_training_repeats(self.root,(raw,),excluded_files=(raw,),allow_single=True).usable)
            self.assertFalse(choose_training_repeats(self.root,('sample.dat',),allow_single=True).usable)

        def test_single_reconstruction_generates_one_training_input(self):
            b=self.bridge;name='sample_rep01.csv';calls=[]
            version=SimpleNamespace(fresh=True,status='fresh',number=1,version_id='v',status_reasons=(),created_at='2025-01-01T00:00:00Z')
            document=SimpleNamespace(sources=(),regions=())
            def generate(context,pair,**kwargs): calls.append(kwargs);return version,document,{}
            with patch('ui_qt.quick.drr_background_workflow.generate_pn',side_effect=generate),patch('core.drr_pn_store.DrrPnStore.list_versions',return_value=(version,)),patch('core.drr_pn_store.DrrPnStore.find_applied',return_value=None),patch.object(b,'_draw_preview'):
                b.open(BackgroundContext(str(self.root),(name,)),BackgroundDraft(kind='pn'))
                b.selectFiles('P',['Pbackground.csv']);b.selectFiles('N',['Nbackground.csv'])
                self.assertTrue(b.snapshot['pnCreateReady'])
                self.assertTrue(b.snapshot['pnSingleMeasurement'])
                self.assertEqual(b.snapshot['pnMethodLabel'],'Single measurement draft')
                b.invoke('pn-generate');self.wait(lambda:not b.snapshot['busy'])
                self.assertEqual(calls[0]['training_files'],(name,))
                self.assertTrue(b.snapshot['pnPreviewReady'])
                b.open(BackgroundContext(str(self.root),(name,'sample_rep02.csv')),BackgroundDraft(kind='pn'))
                self.assertFalse(b.snapshot['pnPreviewReady'])
                self.assertFalse(b.snapshot['pnSingleMeasurement'])
                self.assertEqual(len(b.snapshot['pnTrainingFiles']),2)

        def test_assisted_summary_uses_actual_backend_training_split(self):
            b=self.bridge;names=tuple(f'sample_rep{i:02}.csv' for i in range(1,5))
            b.open(BackgroundContext(str(self.root),names),BackgroundDraft(kind='pn'))
            b.setChoice('generation',2)
            self.assertTrue(b.snapshot['pnUsesMeasurements'])
            self.assertFalse(b.snapshot['pnTrainingRequired'])
            self.assertEqual(b.snapshot['pnTrainingFiles'],[names[0],names[2]])
            self.assertIn('2 training and 2 check',b.snapshot['pnTrainingSummary'])
            b.setChoice('generation',3)
            self.assertFalse(b.snapshot['pnUsesMeasurements'])
            self.assertEqual(b.snapshot['pnTrainingFiles'],[])

        def test_advanced_method_and_frames_are_not_overwritten(self):
            b=self.bridge;captured=[]
            version=SimpleNamespace(fresh=True,status='fresh',number=1,version_id='v',status_reasons=(),created_at='2025-01-01T00:00:00Z')
            document=SimpleNamespace(sources=(),regions=())
            def generate(context,pair,**kwargs): captured.append(kwargs);return version,document,{}
            with patch('ui_qt.quick.drr_background_workflow.generate_pn',side_effect=generate),patch('core.drr_pn_store.DrrPnStore.list_versions',return_value=(version,)),patch('core.drr_pn_store.DrrPnStore.find_applied',return_value=None),patch.object(b,'_draw_preview'):
                b.open(BackgroundContext(str(self.root),('sample_rep01.csv',)),BackgroundDraft(kind='pn'))
                b.selectFiles('P',['Pbackground.csv']);b.selectFiles('N',['Nbackground.csv'])
                for index,method in ((3,'pn-only'),(2,'assisted')):
                    b.setChoice('generation',index);b.setChoice('pFrame',0);b.setChoice('nFrame',1)
                    self.assertTrue(b.snapshot['pnCreateReady'])
                    self.assertFalse(b.snapshot['pnTrainingRequired'])
                    b.invoke('pn-generate');self.wait(lambda:not b.snapshot['busy'])
                    self.assertEqual((captured[-1]['mode'],captured[-1]['p_which'],captured[-1]['n_which']),(method,'first','last'))

        def test_history_cancelled_on_group_change_and_close(self):
            from core.load_cancellation import check_load_cancelled, LoadCancelled
            b=self.bridge; started=Event(); release=Event(); cancelled=Event()
            def history(*args):
                started.set();release.wait(5)
                try: check_load_cancelled()
                except LoadCancelled: cancelled.set();raise
                return ()
            with patch('core.drr_pn_store.DrrPnStore.list_versions',history):
                b.open(BackgroundContext(str(self.root),('a_rep01.csv','a_rep02.csv')),BackgroundDraft(kind='pn'))
                self.wait(started.is_set)
                b.open(BackgroundContext(str(self.root),('b_rep01.csv','b_rep02.csv')),BackgroundDraft(kind='self'))
                release.set();self.wait(cancelled.is_set)
                self.assertFalse(b.snapshot['pnHistoryLoading'])
                started.clear();release.clear();cancelled.clear()
                b.open(BackgroundContext(str(self.root),('a_rep01.csv','a_rep02.csv')),BackgroundDraft(kind='pn'))
                self.wait(started.is_set);b.invoke('cancel');release.set();self.wait(cancelled.is_set)
                self.assertFalse(b.snapshot['opened'])

        def test_external_candidate_pool_blocks_associated_version_restore(self):
            b=self.bridge; started=Event(); release=Event()
            version=SimpleNamespace(fresh=True,status='fresh',number=1,version_id='v1',csv_path='dummy',created_at='2025-01-01T00:00:00Z')
            plan=SimpleNamespace(rows=(),warnings=(),ready=False,resolution=SimpleNamespace(assignments=()))
            def history(*args): started.set();release.wait(5);return (version,)
            with patch('core.drr_pn_store.DrrPnStore.list_versions',history),patch('core.drr_pn_store.DrrPnStore.find_applied',return_value=version),patch('core.drr_composite_background.read_composite_background',return_value=SimpleNamespace(sources=(),regions=())),patch('core.drr_pn_matching.build_pn_match_plan',return_value=plan),patch.object(b,'_draw_preview'):
                draft=BackgroundDraft(kind='pn',assignment_method='matched',pn_candidates=('saved_composite.csv',))
                b.open(BackgroundContext(str(self.root),('a_rep01.csv','a_rep02.csv')),draft)
                self.assertTrue(b.snapshot['pnExternalReferenceActive'])
                self.assertEqual(b.snapshot['pnMode'],'saved')
                self.wait(started.is_set);release.set()
                self.wait(lambda:not b.snapshot['pnHistoryLoading'] and not b.snapshot['matching'])
                self.assertIsNone(b._version)
                self.assertIsNone(b._document)
                self.assertEqual(b.draft.pn_candidates,('saved_composite.csv',))
                self.assertTrue(b.snapshot['pnExternalReferenceActive'])
                self.assertEqual(b.snapshot['pnMode'],'saved')

        def test_generation_preview_apply_and_invalidation(self):
            b=self.bridge;names=('sample_rep01.csv','sample_rep02.csv','sample_rep03.csv')
            version=SimpleNamespace(fresh=True,status='fresh',number=1,version_id='v',status_reasons=(),created_at='2025-01-01T00:00:00Z')
            document=SimpleNamespace(sources=(),regions=());generated=[];prepared=object();events=[]
            def generate(context,pair,**kwargs): generated.append(kwargs);return version,document,{}
            with patch('ui_qt.quick.drr_background_workflow.generate_pn',side_effect=generate),patch('core.drr_pn_store.DrrPnStore.list_versions',return_value=(version,)),patch('core.drr_pn_store.DrrPnStore.find_applied',return_value=None),patch.object(b,'_draw_preview'),patch('ui_qt.quick.drr_background_bridge.prepare_application',return_value=prepared):
                b.open(BackgroundContext(str(self.root),names),BackgroundDraft(kind='pn'))
                b.selectFiles('P',['Pbackground.csv']);b.selectFiles('N',['Nbackground.csv'])
                self.assertTrue(b.snapshot['pnCreateReady'])
                b.invoke('pn-generate');self.wait(lambda:not b.snapshot['busy'])
                self.assertEqual(generated[0]['training_files'],(names[0],names[2]))
                self.assertEqual((generated[0]['p_which'],generated[0]['n_which']),('all','all'))
                self.assertTrue(b.snapshot['pnCanApply'])
                b.applyRequested.connect(events.append);b.invoke('pn-apply');self.wait(lambda:bool(events))
                self.assertIs(events[0],prepared)
                self.assertEqual(b.draft.assignment_method,'common')
                b.setFlag('manual_training',True)
                self.assertFalse(b.snapshot['pnCanApply'])
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases)).wasSuccessful()
if __name__=='__main__':
    if '--isolated' in sys.argv: sys.exit(0 if isolated() else 1)
    unittest.main()
