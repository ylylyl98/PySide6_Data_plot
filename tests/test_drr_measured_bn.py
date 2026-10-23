from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from core.drr_measured_bn import MeasuredBnMatcher, is_measured_bn_name
from core.drr_sources import DrrSource


class MeasuredBnTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
        self.old = self.source('TG1BG=0_rep01_001.csv', 1000, constant=False)
        self.new = self.source('TG1BG=0_rep01_009.csv', 90000, constant=False)
        self.bn1 = self.source('BNbackground_001.csv', 900)
        self.bn2 = self.source('BNbackground_002.csv', 90100)

    def source(self, suffix, stamp, *, constant=True, directory='', prefix=None, grid=(740.,760.,780.)):
        name = (prefix or self.prefix)+suffix
        path = self.root/directory/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('Vbg,Vtg,740,760,780\n0,0,10,11,12\n0,0,11,12,13\n',encoding='utf-8')
        os.utime(path,(stamp,stamp))
        return DrrSource(path.relative_to(self.root).as_posix(),name,'','',stamp,False,
                         gate_varies=False if constant else True,spectral_grid=grid,frame_count=2)

    def matcher(self, measurements=None, candidates=None, **kwargs):
        return MeasuredBnMatcher(self.root,measurements or (self.old,self.new),
                                 candidates if candidates is not None else (self.bn1,self.bn2),**kwargs)

    def logs(self, records, directory='', name='run.experiment.events.jsonl'):
        path = self.root/directory/name
        path.parent.mkdir(parents=True,exist_ok=True)
        events = []
        for source,stamp in records:
            target = str(self.root/source.source) if isinstance(source,DrrSource) else str(source)
            events.append(dict(event='capture_started',output={'file':target},
                               started_utc=datetime.fromtimestamp(stamp,timezone.utc).isoformat()))
        path.write_text('\n'.join(json.dumps(x) for x in events)+'\n',encoding='utf-8')
        return path

    def point_logs(self, records):
        """The acquisition service writes one started event per output row."""
        events = []
        for source, stamp in records:
            for point in range(1, source.frame_count+1):
                events.append(dict(
                    event='capture_started', event_id=f'{source.source}-event-{point}',
                    run_id='one-experiment', experiment_id='one-experiment',
                    acquisition_id=f'{source.source}-capture-{point}',
                    # This is global to the run, not a per-file row count.
                    acquisition_index=100+point*2, capture_frames_requested=1,
                    context={'point_index':point}, output={'file':source.source},
                    started_utc=datetime.fromtimestamp(stamp+point-1,timezone.utc).isoformat()))
        path = self.root/'points.experiment.events.jsonl'
        path.write_text('\n'.join(json.dumps(event) for event in events)+'\n', encoding='utf-8')
        return path, events

    def test_name_token_does_not_admit_pn_or_substrings(self):
        for name in ('x_BNbackground_002.csv','x_BNback.csv','x_BN back.csv','x_BN-back.csv'):
            self.assertTrue(is_measured_bn_name(name),name)
        for name in ('x_PNbackground.csv','x_Nbackground.csv','x_ABNbackground.csv','x_BNbackgate.csv'):
            self.assertFalse(is_measured_bn_name(name),name)

    def test_old_and_new_dates_select_one_bn_each_without_pooling(self):
        matcher = self.matcher()
        for measurement,expected in ((self.old,self.bn1),(self.new,self.bn2)):
            result = matcher.match(measurement)
            self.assertTrue(result.applicable)
            self.assertEqual(result.assignment.baseline_files,(expected.source,))
            self.assertEqual(result.assignment.baseline_which,'all')
            self.assertIn('proxy',result.reason)

    def test_capture_times_override_touched_mtime(self):
        os.utime(self.root/self.bn1.source,(89999,89999))
        os.utime(self.root/self.bn2.source,(99000,99000))
        self.logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
        result = self.matcher().match(self.new)
        self.assertEqual(result.assignment.baseline_files,(self.bn2.source,))
        self.assertIn('capture_started UTC',result.reason)
        self.assertEqual(result.timing_basis,'capture_started_utc')
        self.assertEqual(result.time_gap_seconds,100)
        self.assertEqual(result.runner_up_gap_seconds,89000)
        self.assertEqual(result.unknown_conditions,('polarization','power'))
        self.assertEqual(result.candidate_count,2)

    def test_proxy_match_metadata_and_single_candidate_gap(self):
        result = self.matcher(candidates=(self.bn2,)).match(self.new)
        self.assertEqual(result.timing_basis,'filesystem_mtime_proxy')
        self.assertEqual(result.time_gap_seconds,100)
        self.assertIsNone(result.runner_up_gap_seconds)
        self.assertEqual(result.unknown_conditions,('polarization','power'))
        self.assertEqual(result.candidate_count,1)

    def test_incomplete_logs_use_one_mtime_clock_for_every_source(self):
        self.logs(((self.new,910),(self.bn1,900)))
        result = self.matcher().match(self.new)
        self.assertEqual(result.assignment.baseline_files,(self.bn2.source,))
        self.assertIn('proxy',result.reason)

    def test_condition_grid_and_constant_gate_rejections(self):
        wrong_angle = self.source('BNbackground_007.csv',90001,prefix=self.prefix.replace('94deg','49deg'))
        unknown_angle = self.source('BNbackground_008.csv',90001,prefix=self.prefix.replace('_RotIn94deg',''))
        wrong_grid = replace(self.bn2,spectral_grid=(740.,760.,781.))
        unknown_constant = replace(self.bn2,gate_varies=None)
        for candidate in (wrong_angle,unknown_angle,wrong_grid,unknown_constant,replace(self.bn2,gate_varies=True)):
            with self.subTest(candidate=candidate):
                result = self.matcher(candidates=(candidate,)).match(self.new)
                self.assertFalse(result.applicable)
                self.assertIsNone(result.assignment)

    def test_unknown_sample_or_position_is_not_certified(self):
        for prefix in (self.prefix.replace('YZ383_',''),self.prefix.replace('p2n1_','')):
            candidate = self.source('BNbackground_008.csv',90001,prefix=prefix)
            self.assertFalse(self.matcher(candidates=(candidate,)).match(self.new).applicable)

    def test_equal_time_distance_is_unresolved_not_newest(self):
        os.utime(self.root/self.bn1.source,(89900,89900))
        result = self.matcher().match(self.new)
        self.assertTrue(result.applicable)
        self.assertIsNone(result.assignment)
        self.assertIn('Equally near',result.reason)

    def test_capture_conflicts_and_naive_timestamps_are_unresolved(self):
        log = self.logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100),(self.bn2,90200)))
        self.assertIn('Conflicting',self.matcher().match(self.new).reason)
        log.write_text(json.dumps(dict(event='capture_started',output={'file':str(self.root/self.new.source)},started_utc='2026-10-21T12:00:00')))
        self.assertIn('Invalid capture',self.matcher().match(self.new).reason)

    def test_complete_per_point_capture_run_uses_first_point(self):
        self.point_logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
        os.utime(self.root/self.bn1.source,(89999,89999))
        matcher = self.matcher()
        self.assertEqual(matcher.match(self.old).assignment.baseline_files,(self.bn1.source,))
        result = matcher.match(self.new)
        self.assertEqual(result.assignment.baseline_files,(self.bn2.source,))
        self.assertIn('file acquisition start',result.reason)
        self.assertIn('difference 100 s',result.reason)

    def test_duplicate_copied_logs_are_the_same_acquisition_events(self):
        path, _ = self.point_logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
        copy = self.root/'Initial Data'/'copy.experiment.events.jsonl'
        copy.parent.mkdir()
        copy.write_bytes(path.read_bytes())
        self.assertEqual(self.matcher().match(self.new).assignment.baseline_files,(self.bn2.source,))

    def test_per_point_target_resolution_is_bounded_by_distinct_outputs(self):
        sources = tuple(replace(self.source(suffix,stamp,constant=constant,directory='Initial Data/REF'),
                                frame_count=128)
                        for suffix,stamp,constant in (
                            ('TG1BG=0_rep01_001.csv',1000,False),
                            ('TG1BG=0_rep01_009.csv',90000,False),
                            ('BNbackground_001.csv',900,True),
                            ('BNbackground_002.csv',90100,True)))
        self.point_logs(tuple(zip(sources,(1000,90000,900,90100))))
        matcher = self.matcher(measurements=sources[:2],candidates=sources[2:])
        original = Path.resolve
        resolves = []
        def resolve(path,*args,**kwargs):
            resolves.append(path)
            return original(path,*args,**kwargs)
        with patch.object(Path,'resolve',resolve), \
                patch.object(matcher,'_event_target',wraps=matcher._event_target) as target:
            result = matcher.match(sources[1])
        self.assertEqual(result.assignment.baseline_files,(sources[3].source,))
        self.assertEqual(result.timing_basis,'capture_started_utc')
        self.assertEqual(target.call_count,4)
        self.assertLess(len(resolves),100)

    def test_repeated_unrelated_outputs_cache_target_misses(self):
        path, events = self.point_logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
        unrelated = 'Unrelated/other.csv'
        events.extend(dict(event='capture_started',output={'file':unrelated},started_utc='not-a-time')
                      for _ in range(256))
        path.write_text('\n'.join(json.dumps(event) for event in events),encoding='utf-8')
        matcher = self.matcher()
        with patch.object(matcher,'_event_target',wraps=matcher._event_target) as target:
            result = matcher.match(self.new)
        self.assertEqual(result.assignment.baseline_files,(self.bn2.source,))
        self.assertEqual(result.timing_basis,'capture_started_utc')
        self.assertEqual(sum(call.args[0] == unrelated for call in target.call_args_list),1)

    def test_cached_target_still_validates_later_capture_records(self):
        for defect,reason in (('time','Invalid capture time'),('point','Invalid capture point index')):
            with self.subTest(defect=defect):
                path,events = self.point_logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
                if defect == 'time':
                    events[-1]['started_utc'] = '2026-10-21T12:00:00'
                else:
                    events[-1]['context']['point_index'] = '2'
                path.write_text('\n'.join(json.dumps(event) for event in events),encoding='utf-8')
                result = self.matcher().match(self.new)
                self.assertIsNone(result.assignment)
                self.assertIn(reason,result.reason)

    def test_cancellation_still_checked_on_cached_target_hits(self):
        path,events = self.point_logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
        path.write_text('\n'.join(json.dumps(event) for event in events[:2]*128+events[2:]),encoding='utf-8')
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            return checks > 50
        matcher = self.matcher(cancelled=cancel)
        with patch.object(matcher,'_event_target',wraps=matcher._event_target) as target:
            result = matcher.match(self.new)
        self.assertIsNone(result.assignment)
        self.assertIn('cancelled',result.reason)
        self.assertEqual(target.call_count,1)

    def test_same_relative_output_in_different_log_directories_stays_distinct(self):
        first = self.source('TG1BG=0_rep01_009.csv',1000,constant=False,directory='First/REF')
        second = self.source('TG1BG=0_rep01_009.csv',90000,constant=False,directory='Second/REF')
        relative = 'REF/'+first.filename
        self.logs(((relative,1000),),directory='First')
        self.logs(((relative,90000),),directory='Second')
        self.logs(((self.bn1,900),(self.bn2,90100)))
        matcher = self.matcher(measurements=(first,second))
        with patch.object(matcher,'_event_target',wraps=matcher._event_target) as target:
            early = matcher.match(first)
            late = matcher.match(second)
        self.assertEqual(early.assignment.baseline_files,(self.bn1.source,))
        self.assertEqual(late.assignment.baseline_files,(self.bn2.source,))
        self.assertEqual(early.timing_basis,'capture_started_utc')
        self.assertEqual(late.timing_basis,'capture_started_utc')
        self.assertEqual(sum(call.args[0] == relative for call in target.call_args_list),2)

    def test_same_output_from_different_runs_does_not_choose_earliest(self):
        path, events = self.point_logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
        overwritten = [dict(event, run_id='another-run', experiment_id='another-run',
                            event_id=event['event_id']+'-again', acquisition_id=event['acquisition_id']+'-again')
                       for event in events if event['output']['file'] == self.bn2.source]
        path.write_text('\n'.join(json.dumps(event) for event in events+overwritten),encoding='utf-8')
        result = self.matcher().match(self.new)
        self.assertIsNone(result.assignment)
        self.assertIn('Conflicting',result.reason)

    def test_incomplete_or_repeated_points_and_reversed_times_are_unresolved(self):
        for defect in ('missing','repeated','reversed','missing_acquisition'):
            with self.subTest(defect=defect):
                path, events = self.point_logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
                if defect == 'missing':
                    events.pop()
                elif defect == 'repeated':
                    events[-1]['context']['point_index'] = 1
                elif defect == 'reversed':
                    events[-1]['started_utc'] = datetime.fromtimestamp(90099,timezone.utc).isoformat()
                else:
                    events[-1].pop('acquisition_id')
                path.write_text('\n'.join(json.dumps(event) for event in events),encoding='utf-8')
                result = self.matcher().match(self.new)
                self.assertIsNone(result.assignment)
                self.assertTrue(result.applicable)

    def test_frame_count_validation_is_part_of_cache_identity(self):
        self.point_logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
        self.assertIsNotNone(self.matcher().match(self.new).assignment)
        result = self.matcher(candidates=(self.bn1,replace(self.bn2,frame_count=3))).match(self.new)
        self.assertIsNone(result.assignment)
        self.assertIn('Incomplete',result.reason)

    def test_cancellation_before_or_during_read_is_unresolved(self):
        event = Event();event.set()
        result = self.matcher(cancelled=event).match(self.new)
        self.assertTrue(result.applicable);self.assertIn('cancelled',result.reason)
        event.clear()
        self.logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls > 8
        result = self.matcher(cancelled=cancel).match(self.new)
        self.assertIsNone(result.assignment);self.assertIn('cancelled',result.reason)

    def test_suffix_is_not_a_binding(self):
        renamed = self.source('BNbackground_009.csv',900)
        result = self.matcher(candidates=(renamed,self.bn2)).match(self.new)
        self.assertEqual(result.assignment.baseline_files,(self.bn2.source,))

    def test_foreign_absolute_same_basename_log_cannot_match(self):
        foreign = self.root.parent/'unrelated'/Path(self.new.source).name
        self.logs(((foreign,910),(self.bn1,900),(self.bn2,90100),(self.old,1000)))
        result = self.matcher().match(self.new)
        self.assertEqual(result.assignment.baseline_files,(self.bn2.source,))
        self.assertIn('proxy',result.reason)

    def test_ambiguous_bare_basename_does_not_invent_capture_coverage(self):
        duplicate = self.source('TG1BG=0_rep01_009.csv',1000,constant=False,directory='Other')
        self.logs(((self.new.filename,910),(self.bn1,900),(self.bn2,90100),(self.old,1000)))
        result = self.matcher(measurements=(self.old,self.new,duplicate)).match(self.new)
        self.assertEqual(result.assignment.baseline_files,(self.bn2.source,))
        self.assertIn('proxy',result.reason)

    def test_does_not_scan_parent_or_sibling_logs(self):
        parent_log = self.root.parent/f'{self.root.name}.experiment.events.jsonl'
        parent_log.write_text('foreign\n')
        self.addCleanup(parent_log.unlink,missing_ok=True)
        opened = []
        original = Path.open
        def record(path,*args,**kwargs):
            opened.append(path.resolve())
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',record):
            self.assertIsNotNone(self.matcher().match(self.new).assignment)
        self.assertNotIn(parent_log.resolve(),opened)
        self.assertTrue(all(path.is_relative_to(self.root) for path in opened))

    def test_timing_reads_once_per_worker_and_cache_invalidates(self):
        log = self.logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
        matcher = self.matcher()
        with patch.object(matcher,'_read_timing',wraps=matcher._read_timing) as read:
            matcher.match(self.old);matcher.match(self.new)
        self.assertEqual(read.call_count,1)
        with patch.object(Path,'open',side_effect=AssertionError('unchanged timing reread')):
            self.assertEqual(self.matcher().match(self.new).assignment.baseline_files,(self.bn2.source,))
        self.logs(((self.old,1000),(self.new,90000),(self.bn1,90001),(self.bn2,99000)))
        self.assertEqual(self.matcher().match(self.new).assignment.baseline_files,(self.bn1.source,))
        log.unlink()
        self.assertIn('proxy',self.matcher().match(self.new).reason)
        os.utime(self.root/self.bn1.source,(89999,89999))
        self.assertEqual(self.matcher().match(self.new).assignment.baseline_files,(self.bn1.source,))

    def test_generated_directory_and_selected_file_are_excluded(self):
        generated = self.source('BNbackground_999.csv',90001,directory='Processed Data/DRR/PN Backgrounds')
        managed = self.source('BNbackground_998.csv',90001,directory='.drr_bn_calibration')
        for candidates in ((generated,),(managed,),(replace(self.new,filename=self.bn2.filename,gate_varies=False),)):
            self.assertFalse(self.matcher(candidates=candidates).match(self.new).applicable)

    def test_same_size_log_rewrite_with_restored_mtime_invalidates_cache(self):
        log = self.logs(((self.old,1000),(self.new,90000),(self.bn1,900),(self.bn2,90100)))
        self.assertEqual(self.matcher().match(self.new).assignment.baseline_files,(self.bn2.source,))
        before = log.stat()
        self.logs(((self.old,1000),(self.new,90000),(self.bn1,90001),(self.bn2,99000)))
        self.assertEqual(log.stat().st_size,before.st_size)
        os.utime(log,ns=(before.st_atime_ns,before.st_mtime_ns))
        self.assertEqual(self.matcher().match(self.new).assignment.baseline_files,(self.bn1.source,))


if __name__ == '__main__':
    unittest.main()
