"""Measured BN suggestions are reviewable, bounded and reusable only unchanged."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from core.drr_bn_matching import (
    BnMatchCache, BnMatchPlan, build_bn_match_plan, confirmation_is_current, plan_is_current,
)
from core.drr_sources import inspect_csv_gate_profile


class BnMatchPlanTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
        self.measurement = self.write('TG1BG=0_rep01.csv', 1000, varying=True)
        self.first = self.write('BNbackground_001.csv', 900)
        self.second = self.write('BNbackground_002.csv', 1200)

    def write(self, suffix, stamp, *, varying=False, prefix=None, directory='', grid='740,760,780'):
        path = self.root/directory/((prefix or self.prefix)+suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'Vbg,Vtg,{grid}\n0,0,10,11,12\n'
                        + ('1,1,11,12,13\n' if varying else '0,0,11,12,13\n'), encoding='utf-8')
        os.utime(path, (stamp, stamp))
        return path.relative_to(self.root).as_posix()

    def plan(self, selected=None, **kwargs):
        return build_bn_match_plan(self.root, selected or (self.measurement,), **kwargs)

    def logs(self, records=None, name='run.experiment.events.jsonl'):
        records = records or ((self.measurement, 1000), (self.first, 900), (self.second, 1200))
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(dict(
            event='capture_started', output={'file': name},
            run_id='verified-test-run', experiment_id='verified-test-experiment',
            started_utc=datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
        )) for name, stamp in records)+'\n', encoding='utf-8')
        return path

    def test_no_logs_keeps_suggestion_but_requires_acknowledgement(self):
        plan = self.plan()
        self.assertTrue(plan.ready, plan.resolution.reason)
        self.assertEqual(plan.rows[0].baseline_file, self.first)
        self.assertEqual(plan.rows[0].timing_basis, 'filesystem_mtime_proxy')
        self.assertEqual(plan.rows[0].time_gap_seconds, 100)
        self.assertTrue(any('modification times' in warning for warning in plan.warnings))
        self.assertTrue(any('Unknown acquisition conditions' in warning for warning in plan.warnings))
        self.assertTrue(plan_is_current(plan))
        cache = BnMatchCache()
        with self.assertRaisesRegex(ValueError, 'acknowledgement'):
            cache.confirm(plan)
        self.assertIsNone(cache.find(self.root, (self.measurement,)))
        cache.confirm(plan, warnings_acknowledged=True)
        self.assertIs(cache.find(self.root, (self.measurement,)), plan)

    def test_auto_bn_assigned_source_is_bound_to_load_and_export(self):
        from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft, prepare_application
        from ui_qt.quick.drr_workflow import DrrRequest, export_drr, load_drr
        context = BackgroundContext(str(self.root), (self.measurement,), "tg", 3, 2)
        prepared = prepare_application(context, BackgroundDraft(kind="bn", bn_plan=self.plan(),
            bn_warnings_acknowledged=True))
        request = DrrRequest((str(self.root / self.measurement),), background_mode="external",
            y_axis="tg", window=3, experiment_folder=str(self.root), background_recipe=prepared.recipe)
        environment = dict(MPLBACKEND="agg", MPLCONFIGDIR=str(self.root / "mpl"),
            LOCALAPPDATA=str(self.root / "local"), XDG_CONFIG_HOME=str(self.root / "config"),
            XDG_CACHE_HOME=str(self.root / "cache"))
        with patch.dict(os.environ, environment), \
                patch("core.drr_bn_matching.build_bn_match_plan") as rematch, \
                patch("core.drr_sources.discover_drr_sources") as discover:
            products = load_drr(request)
            self.assertEqual({Path(item[0]).name for item in products.input_signatures},
                {self.measurement, self.first})
            source = self.root / self.first
            before = source.stat()
            source.write_bytes(source.read_bytes().replace(b"10,11,12", b"20,21,22"))
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "changed since loading"):
                export_drr(products, folder=str(self.root), raw_params=None, second_params=None, view="raw")
            rematch.assert_not_called()
            discover.assert_not_called()

    def test_auto_bn_assigned_source_change_during_load_is_rejected(self):
        from core.data_io import load_drr_resolved_cube
        from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft, load_with_recipe, prepare_application
        context = BackgroundContext(str(self.root), (self.measurement,), "tg", 3, 2)
        prepared = prepare_application(context, BackgroundDraft(kind="bn", bn_plan=self.plan(),
            bn_warnings_acknowledged=True))

        def load_then_replace(*args, **kwargs):
            cube = load_drr_resolved_cube(*args, **kwargs)
            source = self.root / self.first
            before = source.stat()
            source.write_bytes(source.read_bytes().replace(b"10,11,12", b"20,21,22"))
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            return cube

        with patch("core.data_io.load_drr_resolved_cube", side_effect=load_then_replace):
            with self.assertRaisesRegex(ValueError, "changed during loading"):
                load_with_recipe(context, prepared.recipe)

    def test_capture_logs_and_full_conditions_need_no_warnings(self):
        prefix = self.prefix+'polH_1mW_'
        measurement = self.write('TG_rep01.csv', 1000, prefix=prefix, varying=True)
        bn = self.write('BNbackground_001.csv', 900, prefix=prefix)
        self.logs(((measurement, 1000), (bn, 900)))
        plan = self.plan((measurement,), candidate_files=(bn,))
        self.assertTrue(plan.ready, plan.rows[0].evidence)
        self.assertEqual(plan.rows[0].timing_basis, 'capture_started_utc')
        self.assertFalse(plan.warnings)
        self.assertFalse(plan.requires_acknowledgement)
        cache = BnMatchCache()
        cache.confirm(plan)
        self.assertIs(cache.find(self.root, (measurement,), candidate_files=(bn,)), plan)

    def test_large_gap_and_close_alternative_warn_with_boundary_policy(self):
        # At either threshold the suggestion is not flagged for that policy.
        plan = self.plan(max_gap_seconds=100, min_margin_seconds=100)
        self.assertFalse(any('exceeds' in warning or 'next candidate' in warning for warning in plan.warnings))
        plan = self.plan(max_gap_seconds=99, min_margin_seconds=101)
        self.assertTrue(plan.ready)
        self.assertTrue(any('exceeds' in warning for warning in plan.warnings))
        self.assertTrue(any('next candidate' in warning for warning in plan.warnings))
        self.assertEqual(plan.rows[0].runner_up_gap_seconds, 100)

    def test_explicit_empty_pool_and_pn_names_never_fall_back(self):
        pn = self.write('Pbackground.csv', 999)
        for pool in ((), (pn,)):
            with self.subTest(pool=pool):
                plan = self.plan(candidate_files=pool)
                self.assertFalse(plan.ready)
                self.assertEqual(plan.candidate_files, ())
                self.assertEqual(plan.resolution.unresolved_measurements, (self.measurement,))
                self.assertFalse(plan.resolution.assignments)
                self.assertFalse(plan.resolution.self_fallback_allowed)

    def test_manual_pool_restricts_exact_candidates_and_is_part_of_cache_key(self):
        plan = self.plan(candidate_files=(self.second,))
        self.assertTrue(plan.ready)
        self.assertEqual(plan.rows[0].baseline_file, self.second)
        self.assertEqual(plan.candidate_files, (self.second,))
        cache = BnMatchCache()
        cache.confirm(plan, warnings_acknowledged=True)
        self.assertIs(cache.find(self.root, (self.measurement,), candidate_files=(self.second,)), plan)
        self.assertIsNone(cache.find(self.root, (self.measurement,)))
        self.assertIsNone(cache.find(self.root, (self.measurement,), candidate_files=(self.first,)))
        self.assertIsNone(cache.find(self.root, (self.measurement,), candidate_files=(self.second,), max_gap_seconds=5))

    def test_partial_failure_preserves_preview_and_prevents_apply(self):
        missing = self.prefix+'TG_missing.csv'
        plan = self.plan((self.measurement, missing))
        self.assertFalse(plan.ready)
        self.assertEqual(tuple(row.measurement_file for row in plan.rows), (self.measurement, missing))
        self.assertIsNotNone(plan.rows[0].assignment)
        self.assertIsNone(plan.rows[1].assignment)
        self.assertEqual(plan.resolution.unresolved_measurements, (missing,))
        self.assertFalse(plan_is_current(plan))
        with self.assertRaisesRegex(ValueError, 'incomplete or stale'):
            BnMatchCache().confirm(plan, warnings_acknowledged=True)

    def test_equal_and_conflicting_times_are_unresolved(self):
        os.utime(self.root/self.second, (1100, 1100))
        equal = self.plan()
        self.assertFalse(equal.ready)
        self.assertIn('Equally near', equal.rows[0].evidence)
        self.logs(((self.measurement, 1000), (self.first, 900), (self.second, 1200), (self.second, 1201)))
        conflicting = self.plan()
        self.assertFalse(conflicting.ready)
        self.assertIn('Conflicting', conflicting.rows[0].evidence)

    def test_wrong_grid_settings_and_generated_references_are_rejected(self):
        from core.drr_bn_store import BN_MARKER
        from core.drr_composite_background import COMPOSITE_MARKER
        wrong_grid = self.write('BNbackground_003.csv', 999, grid='740,760,781')
        wrong_angle = self.write('BNbackground_004.csv', 999, prefix=self.prefix.replace('94deg', '49deg'))
        generated = []
        for index, marker in enumerate((BN_MARKER, COMPOSITE_MARKER), 5):
            name = self.write(f'BNbackground_{index:03d}.csv', 999)
            path = self.root/name
            path.write_text(marker+','+path.read_text())
            generated.append(name)
        for candidate in (wrong_grid, wrong_angle, *generated):
            with self.subTest(candidate=candidate):
                plan = self.plan(candidate_files=(candidate,))
                self.assertFalse(plan.ready)
                self.assertFalse(plan.resolution.assignments)

    def test_ordinary_bn_outside_root_is_rejected(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        path = Path(other.name)/(self.prefix+'BNbackground.csv')
        path.write_bytes((self.root/self.first).read_bytes())
        plan = self.plan(candidate_files=(str(path),))
        self.assertFalse(plan.ready)
        self.assertIn('outside the experiment', plan.resolution.reason)

    def test_profiles_only_selected_measurements_and_bn_csvs(self):
        unrelated = self.root/'unrelated.csv'
        unrelated.write_text('Not a spectral CSV\n')
        reads = []
        def inspect(path, **kwargs):
            reads.append(Path(path).resolve().relative_to(self.root.resolve()).as_posix())
            return inspect_csv_gate_profile(path, **kwargs)
        with patch('core.drr_bn_matching.inspect_csv_gate_profile', side_effect=inspect):
            plan = self.plan()
        self.assertTrue(plan.ready, plan.resolution.reason)
        self.assertEqual(set(reads), {self.measurement, self.first, self.second})
        self.assertEqual(len(reads), 3)

    def test_selected_csv_and_candidate_modifications_invalidate(self):
        for name in (self.measurement, self.first, self.second):
            with self.subTest(name=name):
                plan = self.plan()
                path = self.root/name
                old = path.stat()
                # Change bytes, retaining size and mtime. Windows ChangeTime
                # (or POSIX ctime) must invalidate a confirmed assignment.
                path.write_bytes(path.read_bytes().replace(b'10,11,12', b'20,21,22'))
                os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
                self.assertFalse(plan_is_current(plan))
                path.write_bytes(path.read_bytes().replace(b'20,21,22', b'10,11,12'))

    def test_added_bn_invalidates_auto_but_respects_a_fixed_manual_pool(self):
        automatic = self.plan()
        manual = self.plan(candidate_files=(self.first,))
        new = self.write('BNbackground_003.csv', 999)
        self.assertFalse(plan_is_current(automatic))
        self.assertTrue(plan_is_current(manual))
        rebuilt = self.plan()
        self.assertEqual(rebuilt.rows[0].baseline_file, new)

    def test_changed_added_and_deleted_timing_logs_invalidate(self):
        plan = self.plan()
        log = self.logs()
        self.assertFalse(plan_is_current(plan))
        plan = self.plan()
        old = log.stat()
        log.write_text(log.read_text().replace('00:15:00', '00:15:01'))
        os.utime(log, ns=(old.st_atime_ns, old.st_mtime_ns))
        self.assertFalse(plan_is_current(plan))
        plan = self.plan()
        self.logs(name='Initial Data/late.experiment.events.jsonl')
        self.assertFalse(plan_is_current(plan))
        plan = self.plan()
        log.unlink()
        self.assertFalse(plan_is_current(plan))

    def test_new_nested_bn_is_discovered_and_new_logs_beside_it_invalidate(self):
        nested = self.write('BNbackground_003.csv', 999, directory='Initial Data/REF/session')
        plan = self.plan()
        self.assertEqual(plan.rows[0].baseline_file, nested)
        self.assertIn(nested, plan.candidate_files)
        self.logs(name='Initial Data/REF/session/late.experiment.events.jsonl')
        self.assertFalse(plan_is_current(plan))

    def test_inventory_change_during_matching_never_returns_ready_plan(self):
        def inspect(path, **kwargs):
            if Path(path).name == self.measurement:
                self.write('BNbackground_003.csv', 999)
            return inspect_csv_gate_profile(path, **kwargs)
        with patch('core.drr_bn_matching.inspect_csv_gate_profile', side_effect=inspect):
            plan = self.plan()
        self.assertFalse(plan.ready)
        self.assertIn('changed during matching', plan.resolution.reason)

    def test_cancelled_before_or_during_matching_cannot_be_applied(self):
        cancel = Event()
        cancel.set()
        plan = self.plan(cancelled=cancel)
        self.assertFalse(plan.ready)
        self.assertFalse(plan.resolution.assignments)
        cancel.clear()
        def inspect(path, **kwargs):
            profile = inspect_csv_gate_profile(path, **kwargs)
            cancel.set()
            return profile
        with patch('core.drr_bn_matching.inspect_csv_gate_profile', side_effect=inspect):
            plan = self.plan(cancelled=cancel)
        self.assertFalse(plan.ready)
        self.assertFalse(plan.resolution.assignments)
        self.assertIn('cancelled', plan.resolution.reason)

    def test_json_round_trip_preserves_signature_and_confirmed_reuse(self):
        self.logs()
        plan = self.plan()
        restored = BnMatchPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        self.assertEqual(restored, plan)
        self.assertTrue(plan_is_current(restored))
        cache = BnMatchCache()
        cache.confirm(restored, warnings_acknowledged=True)
        restored_cache = BnMatchCache.from_dict(json.loads(json.dumps(cache.to_dict())))
        self.assertEqual(restored_cache.find(self.root, (self.measurement,)), plan)
        self.assertIsNone(restored_cache.find(self.root, (self.measurement, self.first)))
        self.write('BNbackground_003.csv', 999)
        self.assertIsNone(restored_cache.find(self.root, (self.measurement,)))

    def test_json_rejects_unsupported_schema_and_tampered_approval(self):
        plan = self.plan()
        for mutate in (
                lambda payload: payload.update(schema_version=99),
                lambda payload: payload['rows'][0].update(time_gap_seconds=float('nan')),
                lambda payload: payload.update(warnings=[]),
                lambda payload: payload['rows'][0]['assignment'].update(baseline_files=[self.second]),
                lambda payload: payload['resolution'].update(unresolved_measurements=[self.measurement])):
            payload = json.loads(json.dumps(plan.to_dict()))
            mutate(payload)
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    BnMatchPlan.from_dict(payload)

    def test_policy_must_be_finite_nonnegative(self):
        for kwargs in ({'max_gap_seconds': -1}, {'min_margin_seconds': float('nan')}, {'max_gap_seconds': True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.plan(**kwargs)

    def test_prepare_confirmation_has_no_cache_mutation_and_commit_needs_no_discovery(self):
        plan = self.plan()
        cache = BnMatchCache()
        before = cache.to_dict()
        prepared = cache.prepare_confirm(plan, warnings_acknowledged=True)
        self.assertEqual(cache.to_dict(), before)
        self.assertIs(prepared.plan, plan)
        self.assertTrue(prepared.warnings_acknowledged)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for target in ('core.drr_bn_matching.plan_is_current', 'core.drr_bn_matching._candidate_names',
                           'core.drr_bn_matching._confirmation_directories',
                           'core.drr_bn_matching.inspect_csv_gate_profile', 'pathlib.Path.rglob', 'pathlib.Path.resolve'):
                stack.enter_context(patch(target, side_effect=AssertionError('Fast commit must use fixed path stats: '+target)))
            self.assertTrue(confirmation_is_current(prepared))
            self.assertIs(cache.commit_confirm(prepared), plan)
        self.assertEqual(cache.to_dict()['plans'], [plan.to_dict()])

    def test_unverified_confirmation_token_cannot_commit(self):
        cache = BnMatchCache()
        prepared = cache.prepare_confirm(self.plan(), warnings_acknowledged=True)
        invalid = replace(prepared, _verified=None)
        self.assertFalse(confirmation_is_current(invalid))
        with self.assertRaisesRegex(ValueError, 'stale'):
            cache.commit_confirm(invalid)
        self.assertFalse(cache.to_dict()['plans'])

    def test_fast_confirmation_rejects_changed_selected_bn_or_existing_log(self):
        log = self.logs()
        for path in (self.root/self.measurement, self.root/self.first, self.root/self.second, log):
            with self.subTest(path=path):
                plan = self.plan()
                cache = BnMatchCache()
                prepared = cache.prepare_confirm(plan, warnings_acknowledged=True)
                old = path.stat()
                content = path.read_bytes()
                path.write_bytes(content.replace(b'10,11,12', b'20,21,22') if path.suffix == '.csv'
                                 else content.replace(b'00:15:00', b'00:15:01'))
                os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
                self.assertFalse(confirmation_is_current(prepared))
                with self.assertRaisesRegex(ValueError, 'stale'):
                    cache.commit_confirm(prepared)
                self.assertFalse(cache.to_dict()['plans'])
                path.write_bytes(content)

    def test_fast_confirmation_rejects_new_bn_log_or_nested_inventory_directory(self):
        nested = self.root/'Initial Data'/'REF'/'existing'
        nested.mkdir(parents=True)
        changes = (
            lambda: self.write('BNbackground_003.csv', 999),
            lambda: self.logs(name='late.experiment.events.jsonl'),
            lambda: self.write('BNbackground_004.csv', 998, directory='Initial Data/REF/existing'),
            lambda: self.write('BNbackground_005.csv', 997, directory='Initial Data/REF/new/nested'),
            lambda: self.logs(name='Initial Data/REF/existing/late.experiment.events.jsonl'),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                cache = BnMatchCache()
                prepared = cache.prepare_confirm(self.plan(), warnings_acknowledged=True)
                change()
                self.assertFalse(confirmation_is_current(prepared))
                with self.assertRaisesRegex(ValueError, 'stale'):
                    cache.commit_confirm(prepared)
                self.assertFalse(cache.to_dict()['plans'])

    def test_prepare_rejects_inventory_addition_during_expensive_verification(self):
        cache = BnMatchCache()
        plan = self.plan()
        def verify_then_add(current):
            valid = plan_is_current(current)
            self.write('BNbackground_003.csv', 999)
            return valid
        with patch('core.drr_bn_matching.plan_is_current', side_effect=verify_then_add):
            with self.assertRaisesRegex(ValueError, 'during verification'):
                cache.prepare_confirm(plan, warnings_acknowledged=True)
        self.assertFalse(cache.to_dict()['plans'])

    def test_fast_confirmation_rechecks_directories_after_known_files(self):
        from core import drr_bn_matching as service
        cache = BnMatchCache()
        prepared = cache.prepare_confirm(self.plan(), warnings_acknowledged=True)
        original = service._fixed_signatures
        def read_then_add(snapshot):
            result = original(snapshot)
            if snapshot is prepared.file_snapshot:
                self.write('BNbackground_003.csv', 999)
            return result
        with patch.object(service, '_fixed_signatures', side_effect=read_then_add):
            self.assertFalse(confirmation_is_current(prepared))
        self.assertFalse(cache.to_dict()['plans'])


if __name__ == '__main__':
    unittest.main()
