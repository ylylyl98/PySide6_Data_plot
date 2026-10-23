"""Only stable, complete file inspections can accelerate measured-BN plans."""
from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core import drr_bn_matching as matching
from core import drr_bn_profile_cache as profiles
from core.drr_sources import DrrGateProfile, inspect_csv_gate_profile
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope
from core.provenance import _digest_signature


class BnProfileCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = profiles.VerifiedGateProfileCache()
        profiles.BN_GATE_PROFILE_CACHE.clear()
        self.addCleanup(profiles.BN_GATE_PROFILE_CACHE.clear)

    def write(self, name='source.csv', *, varying=False, rows=2):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ['Vbg,Vtg,740,760,780']
        data.extend(f'{index if varying else 0},0,10,11,12' for index in range(rows))
        path.write_text('\n'.join(data) + '\n', encoding='utf-8')
        return path

    def test_unchanged_full_grid_is_read_once_and_retains_every_frame(self):
        path = self.write(varying=True, rows=41)
        with patch.object(matching, 'inspect_csv_gate_profile', wraps=inspect_csv_gate_profile) as inspect:
            first = self.cache.load(path, inspect)
            second = self.cache.load(path, inspect)
        self.assertIs(second, first)
        self.assertEqual(inspect.call_count, 1)
        self.assertEqual(inspect.call_args.kwargs, {'max_rows': None})
        self.assertEqual(first.frame_count, 41)
        self.assertEqual(len(first.gate_grid), 41)
        self.assertEqual(first.gate_grid[-1], (40., 0.))
        self.assertTrue(first.grid_complete)

    def test_same_size_rewrite_with_restored_mtime_invalidates_strong_identity(self):
        path = self.write(varying=True)
        with patch.object(matching, 'inspect_csv_gate_profile', wraps=inspect_csv_gate_profile) as inspect:
            first = self.cache.load(path, inspect)
            original_stat, original_signature = path.stat(), _digest_signature(path)
            text = path.read_text(encoding='utf-8').replace('1,0,10', '9,0,10')
            path.write_text(text, encoding='utf-8')
            os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            self.assertEqual(path.stat().st_size, original_stat.st_size)
            self.assertNotEqual(_digest_signature(path), original_signature)
            second = self.cache.load(path, inspect)
        self.assertEqual(inspect.call_count, 2)
        self.assertEqual(first.gate_grid[-1][0], 1.)
        self.assertEqual(second.gate_grid[-1][0], 9.)

    def test_file_changed_during_inspection_is_not_published(self):
        path = self.write(varying=True)
        def inspect_then_change(target, **kwargs):
            result = inspect_csv_gate_profile(target, **kwargs)
            self.write(varying=True, rows=3)
            return result
        self.cache.load(path, inspect_then_change)
        with patch.object(matching, 'inspect_csv_gate_profile', wraps=inspect_csv_gate_profile) as inspect:
            current = self.cache.load(path, inspect)
        self.assertEqual(inspect.call_count, 1)
        self.assertEqual(current.frame_count, 3)

    def test_hit_changed_between_signature_checks_is_reinspected(self):
        path = self.write(varying=True)
        self.cache.load(path, inspect_csv_gate_profile)
        signature_calls = []
        def signature(target):
            signature_calls.append(target)
            if len(signature_calls) == 2:
                self.write(varying=True, rows=3)
            return _digest_signature(target)
        with patch.object(profiles, '_digest_signature', side_effect=signature), \
             patch.object(matching, 'inspect_csv_gate_profile', wraps=inspect_csv_gate_profile) as inspect:
            result = self.cache.load(path, inspect)
        self.assertEqual(inspect.call_count, 1)
        self.assertEqual(result.frame_count, 3)

    def test_deleted_file_cannot_return_cached_profile(self):
        path = self.write()
        self.cache.load(path, inspect_csv_gate_profile)
        path.unlink()
        with patch.object(matching, 'inspect_csv_gate_profile', wraps=inspect_csv_gate_profile) as inspect:
            missing = self.cache.load(path, inspect)
        self.assertIsNone(missing.frame_count)
        self.assertEqual(inspect.call_count, 1)
        path = self.write(rows=3)
        self.assertEqual(self.cache.load(path, inspect_csv_gate_profile).frame_count, 3)

    def test_incomplete_malformed_and_unverifiable_profiles_are_not_cached(self):
        valid = inspect_csv_gate_profile(self.write())
        malformed = (
            DrrGateProfile(None, None), replace(valid, grid_complete=False),
            replace(valid, frame_count=3), replace(valid, spectral_grid=(740., float('nan'))),
            replace(valid, gate_grid=((0.,), (0., 0.))),
            replace(valid, gate_ranges=((0., 0.),)),
        )
        for index, profile in enumerate(malformed):
            with self.subTest(profile=index):
                path = self.write(f'bad{index}.csv')
                with patch.object(matching, 'inspect_csv_gate_profile', return_value=profile) as inspect:
                    self.cache.load(path, inspect)
                    self.cache.load(path, inspect)
                self.assertEqual(inspect.call_count, 2)
        path = self.write('unverifiable.csv')
        with patch.object(profiles, '_digest_signature', return_value=None), \
             patch.object(matching, 'inspect_csv_gate_profile', wraps=inspect_csv_gate_profile) as inspect:
            self.cache.load(path, inspect)
            self.cache.load(path, inspect)
        self.assertEqual(inspect.call_count, 2)

    def test_real_truncated_spectral_row_is_never_cached(self):
        path = self.root / 'truncated.csv'
        path.write_text('Vbg,Vtg,740,760,780\n0,0,10,11\n', encoding='utf-8')
        with patch.object(matching, 'inspect_csv_gate_profile', wraps=inspect_csv_gate_profile) as inspect:
            first = self.cache.load(path, inspect)
            self.cache.load(path, inspect)
        self.assertFalse(first.grid_complete)
        self.assertEqual(inspect.call_count, 2)

    def test_cancellation_during_read_cannot_publish_a_profile(self):
        path = self.write(varying=True)
        cancellation = LoadCancellation()
        def cancelled_inspect(target, **kwargs):
            result = inspect_csv_gate_profile(target, **kwargs)
            cancellation.cancel()
            return result
        with load_scope(cancellation):
            with self.assertRaises(LoadCancelled):
                self.cache.load(path, cancelled_inspect)
        with patch.object(matching, 'inspect_csv_gate_profile', wraps=inspect_csv_gate_profile) as inspect:
            self.cache.load(path, inspect)
        self.assertEqual(inspect.call_count, 1)

    def test_cancelled_hit_does_not_return_a_cached_profile(self):
        path = self.write()
        self.cache.load(path, inspect_csv_gate_profile)
        cancellation = LoadCancellation()
        cancellation.cancel()
        with self.assertRaises(LoadCancelled), load_scope(cancellation):
            self.cache.load(path, lambda *_args, **_kwargs: self.fail('Cancelled cache hit inspected a file'))

    def test_cancellation_during_final_signature_cannot_publish_a_profile(self):
        path = self.write()
        cancellation = LoadCancellation()
        signatures = []
        def signature(target):
            result = _digest_signature(target)
            signatures.append(result)
            if len(signatures) == 2:
                cancellation.cancel()
            return result
        with patch.object(profiles, '_digest_signature', side_effect=signature), load_scope(cancellation):
            with self.assertRaises(LoadCancelled):
                self.cache.load(path, inspect_csv_gate_profile)
        self.assertFalse(self.cache._entries)
        with patch.object(matching, 'inspect_csv_gate_profile', wraps=inspect_csv_gate_profile) as inspect:
            self.cache.load(path, inspect)
        self.assertEqual(inspect.call_count, 1)

    def test_same_basename_in_distinct_directories_has_independent_profiles(self):
        first = self.write('one/source.csv', rows=2)
        second = self.write('two/source.csv', rows=3)
        self.assertEqual(self.cache.load(first, inspect_csv_gate_profile).frame_count, 2)
        self.assertEqual(self.cache.load(second, inspect_csv_gate_profile).frame_count, 3)
        self.assertEqual(self.cache.load(first, inspect_csv_gate_profile).frame_count, 2)

    def test_lru_entry_and_byte_limits_cause_reinspection(self):
        first, second = self.write('one.csv'), self.write('two.csv')
        for cache in (profiles.VerifiedGateProfileCache(max_entries=1),
                      profiles.VerifiedGateProfileCache(max_bytes=1)):
            with self.subTest(entries=cache.max_entries, bytes=cache.max_bytes):
                with patch.object(matching, 'inspect_csv_gate_profile', wraps=inspect_csv_gate_profile) as inspect:
                    cache.load(first, inspect)
                    cache.load(second, inspect)
                    cache.load(first, inspect)
                self.assertEqual(inspect.call_count, 3)

    def test_repeated_plans_reuse_profiles_without_changing_any_match_evidence(self):
        prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn49deg_'
        measurement = self.write(prefix + 'TG_rep01.csv', varying=True)
        first = self.write(prefix + 'BNbackground_001.csv')
        second = self.write(prefix + 'BNbackground_002.csv')
        for path, stamp in ((measurement, 1000), (first, 900), (second, 1200)):
            os.utime(path, (stamp, stamp))
        selected = (measurement.name,)
        with patch.object(matching, 'inspect_csv_gate_profile', wraps=inspect_csv_gate_profile) as inspect:
            initial = matching.build_bn_match_plan(self.root, selected)
            self.assertEqual(inspect.call_count, 3)
            repeated = matching.build_bn_match_plan(self.root, selected)
            self.assertEqual(inspect.call_count, 3)
        self.assertTrue(initial.ready)
        self.assertEqual(initial.to_dict(), repeated.to_dict())
        self.assertEqual(initial.rows[0].baseline_file, first.name)


if __name__ == '__main__':
    unittest.main()
