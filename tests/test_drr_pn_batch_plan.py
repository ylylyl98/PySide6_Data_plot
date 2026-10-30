"""Joint P/N batch assignment preserves identity, review and local training."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.drr_pn_batch_plan import build_pn_batch_plan, override_pn_batch_pair, pn_batch_plan_is_current


class BatchPlanTests(unittest.TestCase):
    prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
    def csv(self, suffix, stamp, gate, varying=False, prefix=None, columns='740,760,780'):
        name = (prefix or self.prefix) + suffix + '.csv'
        path = self.root / name
        path.write_text('Vbg,Vtg,' + columns + '\n' + ''.join(
            f'{gate+i if varying else gate},{gate},10,11,12\n' for i in range(4)))
        os.utime(path, (stamp, stamp))
        return name
    def logs(self, pairs, verified=True):
        events = [dict(event='capture_started', output={'file': name}, experiment_id='experiment',
            started_utc=datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
            **({'run_id': 'run' if measurement else 'source'} if verified else {}))
            for name, stamp, measurement in pairs]
        (self.root / 'run.experiment.events.jsonl').write_text('\n'.join(map(json.dumps, events)))
    def test_verified_temporal_segments_pairing_and_single_route(self):
        m = [self.csv(f'TG_rep0{i}', 1000*i, 0, True) for i in range(1, 5)]
        p = [self.csv(f'Pbackground_00{i}', 1000*i-200, -10) for i in (1, 3)]
        n = [self.csv(f'Nbackground_00{i}', 1000*i-100, 10) for i in (1, 3)]
        self.logs([(name, stamp, True) for name, stamp in zip(m, (1000, 1200, 3000, 3200))] +
                  [(p[0], 800, False), (n[0], 900, False), (p[1], 2800, False), (n[1], 2900, False)])
        plan = build_pn_batch_plan(self.root, m, p_candidates=p, n_candidates=n)
        self.assertTrue(plan.ready, plan.rows)
        self.assertEqual([(r.p_file, r.n_file) for r in plan.rows], [(p[0], n[0])]*2 + [(p[1], n[1])]*2)
        self.assertEqual([s.measurement_files for s in plan.segments], [tuple(m[:2]), tuple(m[2:])])
        self.assertTrue(all(s.method == 'conditioned_bounded' for s in plan.segments))
        changed = override_pn_batch_pair(plan, m[1], p[1], n[1])
        self.assertTrue(changed.rows[1].manual)
        self.assertEqual(changed.segments[0].method, 'conditioned_single')
        self.assertTrue(pn_batch_plan_is_current(changed))
        self.assertFalse(pn_batch_plan_is_current(replace(changed, relaxed_conditions=('sample',))))
        (self.root / 'Processed Data' / 'DRR').mkdir(parents=True)
        self.assertTrue(pn_batch_plan_is_current(changed))
        new_log = self.root / 'new.experiment.events.jsonl'
        new_log.write_text('')
        self.assertFalse(pn_batch_plan_is_current(changed))
        new_log.unlink()
        self.assertTrue(pn_batch_plan_is_current(changed))
        (self.root / p[0]).write_text('changed')
        self.assertFalse(pn_batch_plan_is_current(changed))
    def test_selected_background_event_splits_same_pair_training(self):
        a = self.csv('TG_rep01', 1000, 0, True)
        b = self.csv('TG_rep02', 1200, 0, True)
        p = self.csv('Pbackground', 900, -10)
        n = self.csv('Nbackground', 910, 10)
        other = self.csv('Pbackground_other', 1100, -10)
        # This candidate lies between the measurements but is conditionally worse.
        self.logs([(a, 1000, True), (b, 1200, True), (p, 900, False),
                   (n, 910, False), (other, 1100, False)])
        # Force the same pair, without ignoring the intervening acquisition event.
        plan = build_pn_batch_plan(self.root, (a,b), p_candidates=(p,other), n_candidates=(n,))
        plan = override_pn_batch_pair(plan, b, p, n)
        self.assertEqual([segment.method for segment in plan.segments], ['conditioned_single'] * 2)
        self.assertTrue(any('unselected acquisition events' in w for w in plan.warnings))

    def test_unknown_run_never_combines_and_log_precedes_mtime(self):
        a = self.csv('TG_rep01', 1000, 0, True)
        b = self.csv('TG_rep02', 1001, 0, True)
        p = self.csv('Pbackground', 9999, -10)
        n = self.csv('Nbackground', 9998, 10)
        self.logs([(a, 1000, True), (b, 1010, True), (p, 900, False), (n, 910, False)], verified=False)
        plan = build_pn_batch_plan(self.root, (a, b), p_candidates=(p,), n_candidates=(n,))
        self.assertTrue(plan.ready)
        self.assertEqual([s.method for s in plan.segments], ['conditioned_single'] * 2)
        self.assertEqual(plan.rows[0].options[0].measurement_p_gap, 100)
        self.assertEqual(plan.rows[0].timing_basis, 'capture_started_utc')
    def test_strict_pair_precedes_relaxed_nearby_and_grid_role_hard(self):
        a = self.csv('TG_rep01', 1000, 0, True)
        p = self.csv('Pbackground_strict', 300, -10)
        n = self.csv('Nbackground_strict', 310, 10)
        altered = self.prefix.replace('9T', '8T')
        nearp = self.csv('Pbackground_near', 990, -10, prefix=altered)
        nearn = self.csv('Nbackground_near', 991, 10, prefix=altered)
        grid = self.csv('Pbackground_grid', 999, -10, columns='740,760,781')
        plan = build_pn_batch_plan(self.root, (a,), p_candidates=(p, nearp, grid), n_candidates=(n, nearn), relaxed_conditions=('magnetic_field',))
        self.assertEqual((plan.rows[0].p_file, plan.rows[0].n_file), (p, n))
        self.assertFalse(any(o.p_file == grid for o in plan.rows[0].options))
        relaxed = override_pn_batch_pair(plan, a, nearp, nearn)
        self.assertTrue(any('magnetic_field' in w for w in relaxed.rows[0].warnings))
    def test_same_size_rewrite_with_restored_mtime_invalidates_plan_and_override(self):
        a = self.csv('TG_rep01', 1000, 0, True)
        p1 = self.csv('Pbackground_1', 900, -10)
        p2 = self.csv('Pbackground_2', 1100, -10)
        n = self.csv('Nbackground', 1000, 10)
        plan = build_pn_batch_plan(self.root, (a,), p_candidates=(p1, p2), n_candidates=(n,))
        self.assertFalse(plan.ready)
        ready = override_pn_batch_pair(plan, a, p1, n)
        self.assertTrue(ready.ready)
        path = self.root / p1
        stamp = path.stat().st_mtime_ns
        original = path.read_bytes()
        altered = original.replace(b'10,11,12', b'20,11,12', 1)
        self.assertEqual(len(altered), len(original))
        path.write_bytes(altered)
        os.utime(path, ns=(stamp, stamp))
        # Simulate unchanged file metadata to establish the digest check itself.
        with patch('core.drr_pn_batch_plan._unchanged', return_value=True):
            self.assertFalse(pn_batch_plan_is_current(ready))
            with self.assertRaisesRegex(ValueError, 'Refresh'):
                override_pn_batch_pair(plan, a, p1, n)
        with self.assertRaisesRegex(ValueError, 'Refresh'):
            override_pn_batch_pair(plan, a, p1, n)

    def test_pair_requires_equal_even_source_frame_counts(self):
        a = self.csv('TG_rep01', 1000, 0, True)
        p = self.csv('Pbackground', 900, -10)
        n = self.csv('Nbackground', 900, 10)
        path = self.root / n
        with path.open('a') as stream:
            stream.write('10,10,10,11,12\n10,10,10,11,12\n')
        plan = build_pn_batch_plan(self.root, (a,), p_candidates=(p,), n_candidates=(n,))
        self.assertFalse(plan.ready)
        self.assertFalse(plan.rows[0].options)

    def test_ties_and_invalid_sources_remain_unresolved(self):
        a = self.csv('TG_rep01', 1000, 0, True)
        p1 = self.csv('Pbackground_1', 900, -10)
        p2 = self.csv('Pbackground_2', 1100, -10)
        n = self.csv('Nbackground', 1000, 10)
        plan = build_pn_batch_plan(self.root, (a,), p_candidates=(p1,p2), n_candidates=(n,))
        self.assertFalse(plan.ready)
        self.assertEqual(len(plan.rows[0].options), 2)
        self.assertTrue(override_pn_batch_pair(plan, a, p1, n).ready)
        invalid = build_pn_batch_plan(self.root, (a,), p_candidates=(n,), n_candidates=(p1,))
        self.assertFalse(invalid.ready)


if __name__ == '__main__': unittest.main()
