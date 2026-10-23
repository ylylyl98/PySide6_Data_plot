"""Unchanged acquisition logs can serve new groups without parsing them again."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import core.drr_measured_bn as bn
from core.load_cancellation import LoadCancelled
from tests import test_drr_measured_bn as fixtures


class BnLogIndexTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MeasuredBnTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        bn._LOG_CACHE.clear()
        bn._TIMING_CACHE.clear()
        self.addCleanup(bn._LOG_CACHE.clear)
        self.addCleanup(bn._TIMING_CACHE.clear)
        f = self.fixture
        self.log, self.events = f.point_logs(((f.old, 1000), (f.new, 90000), (f.bn1, 900), (f.bn2, 90100)))

    def match(self, measurements, target=None):
        f = self.fixture
        return f.matcher(measurements=measurements).match(target or measurements[0])

    def write_events(self, events):
        self.log.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')

    def test_different_measurement_group_reuses_log_without_reads_or_json_parse(self):
        f = self.fixture
        self.assertEqual(self.match((f.old,)).assignment.baseline_files, (f.bn1.source,))
        with patch.object(bn.json, 'loads', side_effect=AssertionError('Reparsed unchanged log')), \
             patch.object(Path, 'open', side_effect=AssertionError('Reopened unchanged log')):
            result = self.match((f.new,))
        self.assertEqual(result.assignment.baseline_files, (f.bn2.source,))
        self.assertEqual(result.timing_basis, 'capture_started_utc')

    def test_unrelated_invalid_event_remains_invalid_when_its_file_is_selected_later(self):
        f = self.fixture
        self.events[2]['started_utc'] = 'not-a-time'
        self.write_events(self.events)
        self.assertIsNotNone(self.match((f.old,)).assignment)
        with patch.object(bn.json, 'loads', side_effect=AssertionError('Reparsed unchanged log')):
            result = self.match((f.new,))
        self.assertIsNone(result.assignment)
        self.assertIn('Invalid capture time', result.reason)

    def test_target_resolution_is_rechecked_when_basename_becomes_ambiguous(self):
        f = self.fixture
        other = f.source('TG1BG=0_rep01_001.csv', 4000, constant=False, directory='Other')
        for event in self.events:
            if event['output']['file'] == f.old.source:
                event['output']['file'] = f.old.filename
        self.write_events(self.events)
        self.assertEqual(self.match((f.old,)).timing_basis, 'capture_started_utc')
        with patch.object(bn.json, 'loads', side_effect=AssertionError('Reparsed unchanged log')):
            result = self.match((f.old, other))
        self.assertEqual(result.timing_basis, 'filesystem_mtime_proxy')

    def test_same_size_rewrite_with_restored_mtime_invalidates_index(self):
        f = self.fixture
        self.assertEqual(self.match((f.old,)).assignment.baseline_files, (f.bn1.source,))
        before = self.log.stat()
        # Both timestamps have the same serialized length; ChangeTime/inode
        # evidence must invalidate this despite restoring the modification time.
        text = self.log.read_text(encoding='utf-8')
        for point in (0, 1):
            old = datetime.fromtimestamp(1000 + point, timezone.utc).isoformat()
            new = datetime.fromtimestamp(90000 + point, timezone.utc).isoformat()
            text = text.replace(old, new)
        self.log.write_text(text, encoding='utf-8')
        self.assertEqual(self.log.stat().st_size, before.st_size)
        os.utime(self.log, ns=(before.st_atime_ns, before.st_mtime_ns))
        with patch.object(bn.json, 'loads', wraps=json.loads) as parse:
            result = self.match((f.old,))
        self.assertGreater(parse.call_count, 0)
        self.assertEqual(result.assignment.baseline_files, (f.bn2.source,))

    def test_append_and_truncate_do_not_reuse_old_events(self):
        f = self.fixture
        self.assertIsNotNone(self.match((f.old,)).assignment)
        extra = dict(self.events[0], run_id='conflicting-run')
        self.write_events([*self.events, extra])
        self.assertIn('Conflicting', self.match((f.old,)).reason)
        self.log.write_text('', encoding='utf-8')
        self.assertEqual(self.match((f.old,)).timing_basis, 'filesystem_mtime_proxy')

    def test_cancelled_partial_parse_is_not_cached_and_cache_hits_remain_cancellable(self):
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls > 3:
                raise LoadCancelled()
        with self.assertRaises(LoadCancelled):
            list(bn._capture_log_records(self.log, cancel))
        self.assertNotIn(str(self.log), bn._LOG_CACHE)
        list(bn._capture_log_records(self.log, lambda: None))
        calls = 0
        with patch.object(bn.json, 'loads', side_effect=AssertionError('Reparsed unchanged log')):
            with self.assertRaises(LoadCancelled):
                list(bn._capture_log_records(self.log, cancel))

    def test_log_changed_during_parse_is_not_published(self):
        stream = bn._capture_log_records(self.log, lambda: None)
        next(stream)
        with self.log.open('a', encoding='utf-8') as output:
            output.write('\n{}\n')
        with self.assertRaisesRegex(OSError, 'changed during matching'):
            list(stream)
        self.assertNotIn(str(self.log), bn._LOG_CACHE)

    def test_cancellation_during_final_signature_check_does_not_publish(self):
        original_stat = bn._stat
        cancelled = False
        checks = 0
        def stat(path):
            nonlocal cancelled, checks
            checks += 1
            result = original_stat(path)
            if checks == 2:
                cancelled = True
            return result
        def check():
            if cancelled:
                raise LoadCancelled()
        with patch.object(bn, '_stat', side_effect=stat):
            with self.assertRaises(LoadCancelled):
                list(bn._capture_log_records(self.log, check))
        self.assertNotIn(str(self.log), bn._LOG_CACHE)

    def test_cache_eviction_and_oversize_log_do_not_change_results(self):
        list(bn._capture_log_records(self.log, lambda: None))
        other = self.log.with_name('another.experiment.events.jsonl')
        other.write_bytes(self.log.read_bytes())
        with patch.object(bn, '_LOG_CACHE_MAX_ENTRIES', 1):
            list(bn._capture_log_records(other, lambda: None))
        self.assertEqual(list(bn._LOG_CACHE), [str(other)])
        bn._LOG_CACHE.clear()
        with patch.object(bn, '_LOG_CACHE_MAX_BYTES', 1):
            records = list(bn._capture_log_records(self.log, lambda: None))
        self.assertEqual(len(records), len(self.events))
        self.assertFalse(bn._LOG_CACHE)

    def test_current_experiment_log_count_does_not_thrash_on_next_selection(self):
        # The current experiment contains 37 logs. A 32-entry LRU reparses
        # every log on each sequential sweep despite a sufficient byte budget.
        paths = [self.log]
        for number in range(36):
            path = self.log.with_name(f'run{number}.experiment.events.jsonl')
            path.write_bytes(self.log.read_bytes())
            paths.append(path)
        for path in paths:
            list(bn._capture_log_records(path, lambda: None))
        with patch.object(bn.json, 'loads', side_effect=AssertionError('Sequential cache thrash')):
            for path in paths:
                self.assertEqual(len(list(bn._capture_log_records(path, lambda: None))), len(self.events))
        self.assertLessEqual(sum(entry[2] for entry in bn._LOG_CACHE.values()), bn._LOG_CACHE_MAX_BYTES)

    def test_complete_hit_still_validates_frames_and_points_for_each_group(self):
        from dataclasses import replace
        f = self.fixture
        self.assertIsNotNone(self.match((f.old,)).assignment)
        new = replace(f.new, frame_count=3)
        with patch.object(bn.json, 'loads', side_effect=AssertionError('Reparsed unchanged log')):
            result = self.match((new,))
        self.assertIsNone(result.assignment)
        self.assertIn('Incomplete', result.reason)


if __name__ == '__main__':
    unittest.main()
