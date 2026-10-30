"""Explicit background policy and debounced candidate matching."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class BackgroundPolicyTests(unittest.TestCase):
    def test_isolated_policy(self):
        with tempfile.TemporaryDirectory() as root:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", MPLBACKEND="agg", LOCALAPPDATA=root, XDG_CONFIG_HOME=root, XDG_CACHE_HOME=root)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"], cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

def _isolated():
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tempfile
    import time
    import unittest
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch
    from PySide6.QtCore import QCoreApplication, QSettings
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
    from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft


    class Cases(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.app = QCoreApplication.instance() or QCoreApplication([])

        def setUp(self):
            self.temp = tempfile.TemporaryDirectory()
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name)
            self.bridge = DrrBackgroundBridge(settings=QSettings(str(self.root / 'settings.ini'), QSettings.Format.IniFormat))
            self.addCleanup(self.bridge.close)
            self.context = BackgroundContext(str(self.root), ('measurement.csv',))

        def wait_for(self, predicate, timeout=3):
            deadline = time.monotonic() + timeout
            while not predicate() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.005)
            self.assertTrue(predicate())

        def test_multi_file_mode_is_explicit_and_preserves_candidates(self):
            b = self.bridge
            b.open(self.context, BackgroundDraft(kind='p'))
            b.selectFiles('background', ['a.csv', 'b.csv'])
            self.assertTrue(b.snapshot['recommendMatching'])
            self.assertEqual(b.draft.files, ('a.csv', 'b.csv'))
            b.setChoice('method', 1)
            self.assertEqual(b.draft.pn_candidates, ('a.csv', 'b.csv'))
            self.assertEqual(b.draft.assignment_method, 'matched')
            self.assertIs(b.source_drafts['p'], b.draft)
            b.setChoice('method', 0)
            self.assertEqual(b.draft.files, ('a.csv', 'b.csv'))
            self.assertFalse(b.snapshot['recommendMatching'])

        def test_open_matched_schedules_and_debounces_latest_pool(self):
            b = self.bridge
            calls = []
            plan = SimpleNamespace(rows=(), warnings=(), ready=True, resolution=SimpleNamespace(assignments=()))
            def match(root, selected, **kwargs):
                calls.append(kwargs['candidate_files'])
                return plan
            with patch('core.drr_pn_matching.build_pn_match_plan', side_effect=match):
                b.open(self.context, BackgroundDraft(kind='p', assignment_method='matched', pn_candidates=('a.csv',)))
                b.selectFiles('background-add', ['b.csv'])
                self.wait_for(lambda: len(calls) == 1 and not b.snapshot['matching'])
                self.assertEqual(calls, [('a.csv', 'b.csv')])
                self.assertIs(b.draft.pn_plan, plan)
                self.assertFalse(b.snapshot['busy'])
        def test_apply_requires_ready_mapping_and_acknowledged_warnings(self):
            from dataclasses import replace
            b = self.bridge
            b.open(self.context, BackgroundDraft(kind='p', assignment_method='matched', pn_candidates=()))
            self.assertFalse(b.snapshot['applyEnabled'])
            plan = SimpleNamespace(rows=(), warnings=('Review timing',), ready=False, resolution=SimpleNamespace(assignments=()))
            b._draft = replace(b.draft, pn_plan=plan)
            b._refresh()
            self.assertFalse(b.snapshot['applyEnabled'])
            plan.ready = True
            b._refresh()
            self.assertFalse(b.snapshot['applyEnabled'])
            b.setFlag('acknowledged', True)
            self.assertTrue(b.snapshot['applyEnabled'])

        def test_source_drafts_switch_and_rematch(self):
            b = self.bridge
            plans = []
            def match(root, selected, **kwargs):
                plan = SimpleNamespace(rows=(), warnings=(), ready=True, resolution=SimpleNamespace(assignments=()))
                plans.append(kwargs['candidate_files'])
                return plan
            p = BackgroundDraft(kind='p', assignment_method='matched', pn_candidates=('p.csv',))
            n = BackgroundDraft(kind='n', assignment_method='matched', pn_candidates=('n.csv',))
            with patch('core.drr_pn_matching.build_pn_match_plan', side_effect=match):
                b.open(self.context, p, source_drafts={'p': p, 'n': n})
                self.wait_for(lambda: len(plans) == 1 and not b.snapshot['matching'])
                existing = b.draft.pn_plan
                b.setChoice('method', 1)
                self.assertIs(b.draft.pn_plan, existing)
                b.setChoice('kind', 2)
                self.assertEqual(b.draft.pn_candidates, ('n.csv',))
                self.wait_for(lambda: len(plans) == 2 and not b.snapshot['matching'])
                self.assertEqual(plans, [('p.csv',), ('n.csv',)])

        def test_explicit_common_survives_empty_draft_reopen(self):
            b = self.bridge
            b.open(self.context, BackgroundDraft(kind='p'))
            b.setChoice('method', 0)
            draft = b.draft
            self.assertTrue(draft.assignment_method_explicit)
            b.open(self.context, draft)
            b.selectFiles('background', ['a.csv', 'b.csv'])
            self.assertFalse(b.snapshot['recommendMatching'])
            self.assertTrue(b.snapshot['applyEnabled'])

        def test_real_p_n_and_bn_matching(self):
            prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
            def write(suffix, stamp, gate, varying=False):
                name = prefix + suffix + '.csv'
                path = self.root / name
                path.write_text(f'Vbg,Vtg,740,760,780\n{gate},{gate},10,11,12\n{gate+1 if varying else gate},{gate},11,12,13\n')
                os.utime(path, (stamp, stamp))
                return name
            measurement = write('TG_rep01', 1000, 0, True)
            context = BackgroundContext(str(self.root), (measurement,))
            for kind, suffix, gate in (('p', 'Pbackground', -10), ('n', 'Nbackground', 10), ('bn', 'BNbackground', 0)):
                with self.subTest(kind=kind):
                    background = write(suffix, 900, gate)
                    field = 'bn_candidates' if kind == 'bn' else 'pn_candidates'
                    self.bridge.open(context, BackgroundDraft(kind=kind, assignment_method='matched', **{field: (background,)}))
                    self.assertFalse(self.bridge.snapshot['canFix'])
                    self.assertFalse(self.bridge.snapshot['applyEnabled'])
                    self.wait_for(lambda: not self.bridge.snapshot['matching'], timeout=8)
                    plan = self.bridge.draft.bn_plan if kind == 'bn' else self.bridge.draft.pn_plan
                    self.assertIsNotNone(plan, self.bridge.snapshot['error'])
                    self.assertEqual(plan.rows[0].baseline_file, background)
                    self.assertIn('Time source:', self.bridge.snapshot['mappings'][0]['detail'])
                    self.assertIn('Time gap:', self.bridge.snapshot['mappings'][0]['detail'])

        def test_obsolete_running_match_cannot_replace_latest(self):
            from threading import Event
            started, release = Event(), Event()
            b = self.bridge
            plans = []
            def match(root, selected, **kwargs):
                names = kwargs['candidate_files']
                plan = SimpleNamespace(rows=(), warnings=(), ready=True, resolution=SimpleNamespace(assignments=()), names=names)
                plans.append(plan)
                if len(plans) == 1:
                    started.set()
                    release.wait(3)
                return plan
            with patch('core.drr_pn_matching.build_pn_match_plan', side_effect=match):
                b.open(self.context, BackgroundDraft(kind='p', assignment_method='matched', pn_candidates=('a.csv',)))
                self.wait_for(started.is_set)
                self.assertFalse(b.snapshot['applyEnabled'])
                b.selectFiles('background-add', ['b.csv'])
                self.assertFalse(b.snapshot['applyEnabled'])
                release.set()
                self.wait_for(lambda: len(plans) == 2 and not b.snapshot['matching'])
                self.assertIs(b.draft.pn_plan, plans[1])
                self.assertEqual(plans[1].names, ('a.csv', 'b.csv'))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    if '--isolated' in sys.argv:
        sys.exit(0 if _isolated() else 1)
    unittest.main()
