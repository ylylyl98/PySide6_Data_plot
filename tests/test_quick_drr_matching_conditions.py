"""Quick background matching condition choices in an isolated Qt process."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class ConditionsProcessTest(unittest.TestCase):
    def test_conditions(self):
        with tempfile.TemporaryDirectory() as root:
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', QT_QUICK_BACKEND='software', MPLBACKEND='agg', LOCALAPPDATA=root, XDG_CONFIG_HOME=root)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--isolated'], cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

def isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataclasses import replace
    from threading import Event
    import time
    from types import SimpleNamespace
    from unittest.mock import patch
    from PySide6.QtCore import QCoreApplication, QSettings
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
    from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft
    app = QCoreApplication.instance() or QCoreApplication([])
    class Cases(unittest.TestCase):
        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
            self.root = Path(self.tmp.name)
            self.b = DrrBackgroundBridge(settings=QSettings(str(self.root/'settings.ini'), QSettings.Format.IniFormat))
            self.addCleanup(self.b.close)
        def wait(self, predicate):
            deadline = time.monotonic()+10
            while not predicate() and time.monotonic()<deadline:
                app.processEvents(); time.sleep(.005)
            self.assertTrue(predicate(), self.b.snapshot)
        def test_real_sources_strict_then_allowed(self):
            prefix='YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
            def write(name, stamp, gate=0, varying=False):
                path=self.root/name
                path.write_text(f'Vbg,Vtg,740,760,780\n{gate},{gate},10,11,12\n{gate+int(varying)},{gate},11,12,13\n')
                os.utime(path,(stamp,stamp)); return name
            measurement=write(prefix+'TG_rep01.csv',1000,varying=True)
            context=BackgroundContext(str(self.root),(measurement,))
            for kind, suffix, gate in [('p','Pbackground',-10),('n','Nbackground',10),('bn','BNbackground',0)]:
                with self.subTest(kind=kind):
                    candidate=write(prefix.replace('94deg','95deg')+suffix+'.csv',900,gate)
                    field='bn_candidates' if kind=='bn' else 'pn_candidates'
                    self.b.open(context,BackgroundDraft(kind=kind,assignment_method='matched',**{field:(candidate,)}))
                    self.wait(lambda:not self.b.snapshot['matching'])
                    strict=self.b.draft.bn_plan or self.b.draft.pn_plan
                    self.assertFalse(strict.ready, self.b.snapshot)
                    self.b.setRelaxedCondition('rotation', True)
                    self.assertFalse(self.b.snapshot['applyEnabled'])
                    self.wait(lambda:not self.b.snapshot['matching'])
                    plan=self.b.draft.bn_plan or self.b.draft.pn_plan
                    self.assertTrue(plan.ready, self.b.snapshot)
                    self.assertEqual(plan.relaxed_conditions, ('rotation',))
                    self.assertEqual(plan.rows[0].baseline_file, candidate)
                    self.assertTrue(plan.warnings)
                    self.assertFalse(self.b.snapshot['applyEnabled'])
                    self.b.setFlag('acknowledged', True)
                    self.assertTrue(self.b.snapshot['applyEnabled'])
                    self.b.resetMatchingConditions()
                    self.assertFalse(self.b.snapshot['applyEnabled'])
                    self.wait(lambda:not self.b.snapshot['matching'])
                    self.assertFalse((self.b.draft.bn_plan or self.b.draft.pn_plan).ready)
        def test_advanced_choices_clear_matching_strategy(self):
            context=BackgroundContext(str(self.root),('m.csv',))
            def open_matched():
                draft=BackgroundDraft(kind='p',assignment_method='matched',pn_candidates=(),
                    relaxed_conditions=('rotation',),saved_matching_review={'old': True},
                    bn_warnings_acknowledged=True,pn_warnings_acknowledged=True)
                self.b.open(context,draft)
                self.b._flags['acknowledged']=True
            def verify(kind):
                draft=self.b.draft
                self.assertEqual(draft.kind,kind,self.b.snapshot['error'])
                self.assertEqual(draft.assignment_method,'common')
                self.assertEqual(self.b.snapshot['method'],'common')
                self.assertEqual(draft.relaxed_conditions,())
                self.assertIsNone(draft.saved_matching_review)
                self.assertIsNone(draft.pn_candidates)
                self.assertIsNone(draft.bn_candidates)
                self.assertFalse(draft.pn_warnings_acknowledged)
                self.assertFalse(self.b.snapshot['flags']['acknowledged'])
                self.assertTrue(self.b.snapshot['applyEnabled'])
            open_matched()
            self.b._version=SimpleNamespace(fresh=True,status_reasons=())
            self.b.invoke('use-version')
            verify('pn')
            open_matched()
            self.b._fit_reference=object()
            with patch.object(self.b, '_fit_recipe', return_value={'method': 'spline'}):
                self.b.invoke('use-fit')
            verify('fitted')
            for action in ('use-bn-original','use-bn-candidate'):
                open_matched()
                self.b._bn_payload={'eligible':True,'candidates':()}
                self.b.invoke(action)
                verify('bn')

        def test_advanced_choice_cancels_pending_old_match(self):
            started, release=Event(),Event()
            stale=SimpleNamespace(rows=(),warnings=(),ready=True,resolution=SimpleNamespace(assignments=()))
            def match(*args, **kwargs):
                started.set(); release.wait(5); return stale
            context=BackgroundContext(str(self.root),('m.csv',))
            with patch('core.drr_pn_matching.build_pn_match_plan',side_effect=match):
                self.b.open(context,BackgroundDraft(kind='p',assignment_method='matched',pn_candidates=('p.csv',)))
                self.wait(started.is_set)
                self.b._version=SimpleNamespace(fresh=True,status_reasons=())
                self.b.invoke('use-version')
                release.set()
                deadline=time.monotonic()+.5
                while time.monotonic()<deadline: app.processEvents(); time.sleep(.005)
                self.assertEqual(self.b.draft.kind,'pn')
                self.assertIsNone(self.b.draft.pn_plan)
                self.assertEqual(self.b.snapshot['method'],'common')
                self.assertTrue(self.b.snapshot['applyEnabled'])

        def test_bn_manual_override_and_incompatible_candidate(self):
            prefix='YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
            def write(name, stamp, varying=False, grid='740,760,780'):
                path=self.root/name
                path.write_text(f'Vbg,Vtg,{grid}\n0,0,10,11,12\n{int(varying)},0,11,12,13\n')
                os.utime(path,(stamp,stamp)); return name
            measurement=write(prefix+'TG_rep01.csv',1000,True)
            first=write(prefix+'BNbackground_01.csv',900)
            second=write(prefix+'BNbackground_02.csv',1300)
            invalid=write(prefix+'BNbackground_bad.csv',1400,grid='740,760,790')
            context=BackgroundContext(str(self.root),(measurement,))
            self.b.open(context,BackgroundDraft(kind='bn',assignment_method='matched',bn_candidates=(first,second,invalid)))
            self.wait(lambda:not self.b.snapshot['matching'])
            original=self.b.draft.bn_plan
            self.assertEqual(original.rows[0].baseline_file,first)
            self.assertIn(invalid,self.b.snapshot['mappings'][0]['candidates'])
            self.b.overrideMatch(measurement,second)
            self.assertFalse(self.b.snapshot['applyEnabled'])
            self.wait(lambda:not self.b.snapshot['matching'])
            self.assertEqual(self.b.draft.bn_plan.rows[0].baseline_file,second)
            self.assertFalse(self.b.draft.bn_warnings_acknowledged)
            chosen=self.b.draft.bn_plan
            self.b.overrideMatch(measurement,invalid)
            self.wait(lambda:not self.b.snapshot['matching'])
            self.assertIs(self.b.draft.bn_plan,chosen)
            self.assertTrue(self.b.snapshot['error'])

        def test_cancelled_bn_override_cannot_publish_old_policy(self):
            started, release = Event(), Event()
            old=SimpleNamespace(rows=(), warnings=(), ready=True, resolution=SimpleNamespace(assignments=()), candidate_files=('b.csv',))
            new=SimpleNamespace(rows=(), warnings=(), ready=True, resolution=SimpleNamespace(assignments=()), candidate_files=('b.csv',), relaxed_conditions=('rotation',))
            def override(*args, **kwargs):
                started.set(); release.wait(5); return old
            def match(*args, **kwargs): return new
            context=BackgroundContext(str(self.root),('m.csv',))
            with patch('core.drr_bn_matching.override_bn_match',side_effect=override), patch('core.drr_bn_matching.build_bn_match_plan',side_effect=match):
                self.b.open(context,BackgroundDraft(kind='bn',assignment_method='matched',bn_candidates=('b.csv',),bn_plan=old))
                self.b.overrideMatch('m.csv','b.csv')
                self.wait(started.is_set)
                self.b.setRelaxedCondition('rotation',True)
                release.set()
                self.wait(lambda:not self.b.snapshot['matching'])
                self.assertIs(self.b.draft.bn_plan,new)
                self.assertEqual(self.b.draft.relaxed_conditions,('rotation',))

        def test_old_worker_cannot_publish_after_condition_change(self):
            started=Event(); release=Event(); plans=[]
            def match(root, selected, **kw):
                policy=kw['relaxed_conditions']; plan=SimpleNamespace(rows=(), warnings=(), ready=True, resolution=SimpleNamespace(assignments=()), relaxed_conditions=policy)
                plans.append(plan)
                if not policy: started.set(); release.wait(5)
                return plan
            context=BackgroundContext(str(self.root),('measurement.csv',))
            with patch('core.drr_pn_matching.build_pn_match_plan',side_effect=match):
                self.b.open(context,BackgroundDraft(kind='p',assignment_method='matched',pn_candidates=('p.csv',)))
                self.wait(started.is_set)
                self.b.setRelaxedCondition('rotation',True)
                self.assertIsNone(self.b.draft.pn_plan)
                self.assertFalse(self.b.snapshot['applyEnabled'])
                release.set()
                self.wait(lambda:len(plans)==2 and not self.b.snapshot['matching'])
                self.assertIs(self.b.draft.pn_plan,plans[1])
                self.assertEqual(self.b.draft.relaxed_conditions,('rotation',))
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases)).wasSuccessful()

if __name__=='__main__':
    if '--isolated' in sys.argv: sys.exit(0 if isolated() else 1)
    unittest.main()
