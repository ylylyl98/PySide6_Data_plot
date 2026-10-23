"""Verified acquisition-run grouping and conservative joint BN judgment."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from core.drr_bn_matching import BnMatchCache, BnMatchPlan, build_bn_match_plan, plan_is_current
from core.drr_measured_bn import MeasuredBnMatcher


class BnBatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_polH_1mW_'
        self.selected = [self.csv(f'TG_rep{index}.csv', stamp, varying=True)
                         for index, stamp in enumerate((1000, 1010, 1020), 1)]
        self.bn1 = self.csv('BNbackground_001.csv', 900)
        self.bn2 = self.csv('BNbackground_002.csv', 1600)

    def csv(self, suffix, stamp, *, varying=False, prefix=None, grid='740,760,780'):
        name = (prefix or self.prefix)+suffix
        (self.root/name).write_text(f'Vbg,Vtg,{grid}\n0,0,10,11,12\n'
                                   + ('1,1' if varying else '0,0')+',11,12,13\n')
        os.utime(self.root/name, (stamp, stamp))
        return name

    def logs(self, measurement_runs=None, *, experiment='experiment-A', no_run=False, timestamps=None):
        runs = measurement_runs or {name: 'run-A' for name in self.selected}
        stamps = {name: (self.root/name).stat().st_mtime for name in (*self.selected, self.bn1, self.bn2)}
        if timestamps:
            stamps.update(timestamps)
        events = []
        for name, stamp in stamps.items():
            event = dict(event='capture_started', output={'file': name},
                         experiment_id=experiment,
                         started_utc=datetime.fromtimestamp(stamp, timezone.utc).isoformat())
            if not no_run:
                event['run_id'] = runs.get(name, 'reference-run')
            events.append(event)
        path = self.root/'capture.experiment.events.jsonl'
        path.write_text('\n'.join(json.dumps(event) for event in events)+'\n')
        return path

    def plan(self, **kwargs):
        return build_bn_match_plan(self.root, self.selected, **kwargs)

    def assert_partition(self, plan):
        members = [name for batch in plan.batches for name in batch.measurement_files]
        self.assertCountEqual(members, self.selected)
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual([row.measurement_file for row in plan.rows], self.selected)

    def test_two_verified_runs_using_same_bn_remain_two_batches(self):
        self.logs({self.selected[0]: 'run-A', self.selected[1]: 'run-A', self.selected[2]: 'run-B'})
        plan = self.plan()
        self.assertTrue(plan.ready, plan.resolution.reason)
        self.assert_partition(plan)
        self.assertEqual([batch.measurement_count for batch in plan.batches], [2, 1])
        self.assertEqual([batch.baseline_file for batch in plan.batches], [self.bn1, self.bn1])
        self.assertEqual([batch.run_id for batch in plan.batches], ['run-A', 'run-B'])
        self.assertTrue(all(batch.status == 'consensus' and batch.verified_run for batch in plan.batches))
        self.assertNotEqual(plan.batches[0].id, plan.batches[1].id)
        self.assertEqual(dict(plan.batches[0].aggregate_time_gaps)[self.bn1], 210)

    def test_within_run_disagreement_keeps_all_per_file_suggestions_no_majority_force(self):
        self.logs(timestamps={self.selected[2]: 1500})
        plan = self.plan()
        self.assertTrue(plan.ready)
        self.assert_partition(plan)
        self.assertEqual(len(plan.batches), 1)
        batch = plan.batches[0]
        self.assertEqual(batch.status, 'conflict')
        self.assertEqual(batch.baseline_file, '')
        self.assertEqual(batch.aggregate_time_gaps[0][0], self.bn1)
        self.assertEqual([item.baseline_files for item in plan.resolution.assignments],
                         [(self.bn1,), (self.bn1,), (self.bn2,)])
        self.assertEqual(batch.anomaly_measurements, tuple(self.selected))
        self.assertTrue(all(row.anomaly and row.batch_status == 'conflict' for row in plan.rows))
        with self.assertRaisesRegex(ValueError, 'acknowledgement'):
            BnMatchCache().prepare_confirm(plan)
        cache = BnMatchCache()
        cache.confirm(plan, warnings_acknowledged=True)
        self.assertEqual(cache.find(self.root, self.selected), plan)

    def test_no_logs_or_experiment_id_without_run_id_never_merge_same_day_files(self):
        for mode in ('no_logs', 'experiment_only'):
            with self.subTest(mode=mode):
                if mode == 'experiment_only':
                    self.logs(no_run=True, timestamps={self.selected[2]: 90000, self.bn2: 90100})
                plan = self.plan()
                self.assertTrue(plan.ready)
                self.assert_partition(plan)
                self.assertEqual(len(plan.batches), 3)
                self.assertTrue(all(not batch.verified_run and batch.status == 'unverified' for batch in plan.batches))
                self.assertTrue(all(batch.measurement_count == 1 for batch in plan.batches))
                self.assertTrue(all(batch.baseline_file for batch in plan.batches))
                self.assertTrue(any('singleton' in warning for warning in plan.warnings))

    def test_mixed_acquisition_settings_and_grids_split_a_verified_run(self):
        other = self.csv('TG_other.csv', 1030, varying=True, prefix=self.prefix.replace('94deg', '49deg'))
        other_bn = self.csv('BNbackground_003.csv', 901, prefix=self.prefix.replace('94deg', '49deg'))
        changed_grid = self.csv('TG_grid.csv', 1040, varying=True, grid='740,760,781')
        grid_bn = self.csv('BNbackground_004.csv', 902, grid='740,760,781')
        self.selected.extend((other, changed_grid))
        self.logs()
        # Additional references receive verified capture starts as well.
        path = self.root/'extra.experiment.events.jsonl'
        path.write_text('\n'.join(json.dumps(dict(event='capture_started', output={'file': name},
            run_id='reference-run', started_utc=datetime.fromtimestamp(stamp, timezone.utc).isoformat()))
            for name, stamp in ((other_bn, 901), (grid_bn, 902))))
        plan = self.plan()
        self.assertTrue(plan.ready, plan.resolution.reason)
        self.assert_partition(plan)
        self.assertEqual(sorted(batch.measurement_count for batch in plan.batches), [1, 1, 3])
        self.assertTrue(all(batch.run_id == 'run-A' for batch in plan.batches))
        self.assertEqual(len({batch.batch_id for batch in plan.batches}), 3)

    def test_age_and_near_tie_are_anomalies_without_changing_individual_choices(self):
        self.logs()
        aged = self.plan(max_gap_seconds=50)
        self.assertTrue(aged.ready)
        self.assertEqual(aged.batches[0].status, 'conflict')
        self.assertEqual(aged.batches[0].anomaly_measurements, tuple(self.selected))
        close = self.plan(min_margin_seconds=1000)
        self.assertTrue(close.ready)
        self.assertEqual(close.batches[0].status, 'conflict')
        self.assertTrue(all(item.baseline_files == (self.bn1,) for item in close.resolution.assignments))

    def test_batch_context_and_joint_candidates_reuse_one_timing_parse(self):
        self.logs()
        original = MeasuredBnMatcher._read_timing
        calls = []
        def timed(matcher):
            calls.append(matcher)
            return original(matcher)
        with patch.object(MeasuredBnMatcher, '_read_timing', timed):
            plan = self.plan()
        self.assertTrue(plan.ready)
        self.assertEqual(len(calls), 1)
        self.assertEqual(plan.batches[0].capture_start, 1000)
        self.assertEqual(plan.batches[0].capture_end, 1020)

    def test_old_confirmed_schema_rejected_and_current_batch_snapshot_reusable(self):
        self.logs()
        plan = self.plan()
        cache = BnMatchCache()
        cache.confirm(plan)
        restored = BnMatchCache.from_dict(json.loads(json.dumps(cache.to_dict())))
        self.assertEqual(restored.find(self.root, self.selected), plan)
        old = plan.to_dict()
        old['schema_version'] = 1
        with self.assertRaisesRegex(ValueError, 'schema'):
            BnMatchPlan.from_dict(old)
        with self.assertRaisesRegex(ValueError, 'schema'):
            BnMatchCache.from_dict({'schema_version': 1, 'plans': []})
        log = self.root/'capture.experiment.events.jsonl'
        log.write_text(log.read_text().replace('run-A', 'run-B'))
        self.assertFalse(plan_is_current(plan))
        self.assertIsNone(restored.find(self.root, self.selected))

    def test_dataclass_ready_requires_current_policy_exact_partition_and_row_mapping(self):
        self.logs()
        plan = self.plan()
        corrupt = (
            replace(plan, batches=()), replace(plan, policy_version='old-policy'),
            replace(plan, batches=(replace(plan.batches[0], measurement_files=tuple(self.selected[:-1])),)),
            replace(plan, rows=(replace(plan.rows[0], batch_id='wrong'), *plan.rows[1:])),
        )
        for current in corrupt:
            with self.subTest(current=current.policy_version):
                self.assertFalse(current.ready)
                with self.assertRaisesRegex(ValueError, 'incomplete|stale'):
                    BnMatchCache().prepare_confirm(current, warnings_acknowledged=True)

    def test_serialized_batch_partition_and_consensus_checked_beyond_overall_hash(self):
        self.logs()
        plan = self.plan()
        mutations = (
            lambda p: p['batches'][0]['measurement_files'].pop(),
            lambda p: p['batches'][0].update(baseline_file=self.bn2),
            lambda p: p['batches'][0].update(baseline_file=''),
            lambda p: p['batches'][0].update(verified_run=False),
            lambda p: p['rows'][0].update(batch_id='foreign'),
            lambda p: p['batches'][0].update(anomaly_measurements=[self.selected[0]]),
        )
        for mutate in mutations:
            payload = json.loads(json.dumps(plan.to_dict()))
            mutate(payload)
            unsigned = dict(payload)
            unsigned.pop('signature')
            payload['signature'] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            with self.subTest(mutation=mutate):
                with self.assertRaises(ValueError):
                    BnMatchPlan.from_dict(payload)

    def test_cancel_and_missing_selected_sources_keep_partition_and_cannot_be_ready(self):
        stop = Event()
        stop.set()
        cancelled = self.plan(cancelled=stop)
        self.assert_partition(cancelled)
        self.assertFalse(cancelled.ready)
        self.assertFalse(cancelled.resolution.assignments)
        self.logs()
        (self.root/self.selected[-1]).unlink()
        partial = self.plan()
        self.assert_partition(partial)
        self.assertFalse(partial.ready)
        self.assertEqual(partial.resolution.unresolved_measurements, (self.selected[-1],))
        self.assertEqual(len(partial.resolution.assignments), 2)


if __name__ == '__main__':
    unittest.main()
