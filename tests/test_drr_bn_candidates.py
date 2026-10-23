"""Explicit BN browsing prioritizes the selected rotation without reading files."""
import builtins
from dataclasses import replace
from itertools import permutations
from pathlib import Path
import unittest
from unittest.mock import patch

from core.drr_baseline_candidates import DrrBaselineRecommendation
from core.drr_bn_candidates import bn_grid_state, rank_bn_candidates
from core.drr_sources import DrrSource


def source(rotation='RotIn49deg', *, token='BNbackground', mtime=100.,
           grid=(740., 760., 780.), name=None, directory='Initial Data/REF', temperature='1.67K'):
    filename = name or '_'.join(part for part in (
        'YZ383', 'p2n1', '9T', temperature, '730nmc', '0p3sx10', rotation, token) if part) + '.csv'
    return DrrSource(f'{directory}/{filename}', filename, '', '', mtime, token == 'BNbackground',
                     gate_varies=token != 'BNbackground', spectral_grid=tuple(grid))


def names(ranked):
    return tuple(item.source.source for item in ranked)


class BnCandidateOrderingTests(unittest.TestCase):
    def test_same49_precedes_newer94_and_selection_change_reverses_priority(self):
        measurement49 = source(token='TG_rep01')
        measurement94 = source('RotIn94deg', token='TG_rep01')
        bn49 = source(mtime=10.)
        bn94 = source('RotIn94deg', mtime=1000.)
        self.assertEqual(names(rank_bn_candidates((measurement49,), (bn94, bn49))), (bn49.source, bn94.source))
        self.assertEqual(names(rank_bn_candidates((measurement94,), (bn49, bn94))), (bn94.source, bn49.source))

    def test_filename_supplies_rotation_when_catalog_has_no_rotation_field_or_grid(self):
        measurement = source(token='TG_rep01', grid=())
        self.assertFalse(hasattr(measurement, 'rotation'))
        bn49 = source(mtime=10., grid=())
        bn94 = source('RotIn94deg', mtime=1000., grid=())
        ranked = rank_bn_candidates((measurement,), (bn94, bn49))
        self.assertEqual(names(ranked), (bn49.source, bn94.source))
        self.assertIn('Same angle', ranked[0].reason)
        self.assertIn('needs validation', ranked[0].reason)

    def test_unknown_candidate_rotation_is_between_same_and_known_conflict(self):
        measurement = source(token='TG_rep01')
        same = source(mtime=1.)
        unknown = source('', mtime=100.)
        conflict = source('RotIn94deg', mtime=1000.)
        ranked = rank_bn_candidates((measurement,), (conflict, unknown, same))
        self.assertEqual(names(ranked), (same.source, unknown.source, conflict.source))
        self.assertLess(ranked[0].tier, ranked[1].tier)
        self.assertLess(ranked[1].tier, ranked[2].tier)

    def test_unknown_measurement_rotation_does_not_claim_same_angle(self):
        measurement = source('', token='TG_rep01')
        older49, newer94 = source(mtime=10.), source('RotIn94deg', mtime=20.)
        ranked = rank_bn_candidates((measurement,), (older49, newer94))
        self.assertEqual(names(ranked), (newer94.source, older49.source))
        self.assertTrue(all('Same angle' not in item.reason for item in ranked))

    def test_mixed49_and94_measurements_never_claim_one_shared_rotation(self):
        measurements = (source(token='TG_rep01'), source('RotIn94deg', token='TG_rep02'))
        unknown = source('', mtime=1.)
        bn49, bn94 = source(mtime=10.), source('RotIn94deg', mtime=20.)
        ranked = rank_bn_candidates(measurements, (bn49, unknown, bn94))
        self.assertEqual(ranked[0].source, unknown)
        self.assertTrue(all('Same angle' not in item.reason for item in ranked))
        self.assertEqual({item.source.source for item in ranked}, {unknown.source, bn49.source, bn94.source})

    def test_rotin_and_rotout_are_distinct_and_missing_arms_are_unknown(self):
        measurement = source('RotIn49deg_RotOut10deg', token='TG_rep01')
        both = source('RotIn49deg_RotOut10deg', mtime=1.)
        missing = source('RotIn49deg', mtime=100.)
        wrong = source('RotIn49deg_RotOut94deg', mtime=1000.)
        ranked = rank_bn_candidates((measurement,), (wrong, missing, both))
        self.assertEqual(names(ranked), (both.source, missing.source, wrong.source))
        self.assertIn('not fully known', ranked[1].reason)
        # An equal numeric angle on the other optical arm is not a match.
        only_out = source('RotOut49deg')
        self.assertNotIn('Same angle', rank_bn_candidates((source(token='TG_rep01'),), (only_out,))[0].reason)

    def test_mixed_measurement_arm_sets_cannot_be_promoted_by_a_two_arm_candidate(self):
        measurements = (source('RotIn49deg', token='TG_rep01'), source('RotOut49deg', token='TG_rep02'))
        candidate = source('RotIn49deg_RotOut49deg')
        ranked = rank_bn_candidates(measurements, (candidate,))
        self.assertEqual(len(ranked), 1)
        self.assertNotIn('Same angle', ranked[0].reason)

    def test_only_bn_named_sources_remain_even_if_other_backgrounds_are_newer(self):
        measurement = source(token='TG_rep01')
        bn = source()
        bnback = source(token='BNback', mtime=20.)
        candidates = (source(token='Pbackground', mtime=1000.), source(token='Nback', mtime=1000.),
                      source(token='background', mtime=1000.), source(token='TG_rep02', mtime=1000.), bn, bnback)
        ranked = rank_bn_candidates((measurement,), candidates)
        self.assertEqual({item.source.source for item in ranked}, {bn.source, bnback.source})
        self.assertTrue(all(isinstance(item, DrrBaselineRecommendation) for item in ranked))

    def test_known_grid_conflict_is_last_even_for_the_same_angle(self):
        measurement = source(token='TG_rep01')
        bad_same = source(mtime=10000., grid=(740., 760., 781.))
        unknown_grid = source('', mtime=10., grid=())
        good_wrong_angle = source('RotIn94deg', mtime=20.)
        ranked = rank_bn_candidates((measurement,), (bad_same, good_wrong_angle, unknown_grid))
        self.assertEqual(ranked[-1].source, bad_same)
        self.assertIn('incompatible spectral grid', ranked[-1].reason)
        self.assertEqual(bn_grid_state((measurement,), bad_same), 2)
        self.assertEqual(bn_grid_state((measurement,), unknown_grid), 1)
        self.assertEqual(bn_grid_state((measurement,), good_wrong_angle), 0)

    def test_other_known_condition_agreement_precedes_newest_timestamp(self):
        measurement = source(token='TG_rep01')
        matching = source(mtime=1.)
        wrong_temperature = source(mtime=1000., temperature='4K')
        ranked = rank_bn_candidates((measurement,), (wrong_temperature, matching))
        self.assertEqual(names(ranked), (matching.source, wrong_temperature.source))

    def test_correct_temperature_with_unknown_grid_precedes_known_wrong_temperature(self):
        measurement = source(token='TG_rep01')
        correct_temperature = source(mtime=1., grid=())
        wrong_temperature = source(mtime=1000., temperature='4K')
        ranked = rank_bn_candidates((measurement,), (wrong_temperature, correct_temperature))
        self.assertEqual(names(ranked), (correct_temperature.source, wrong_temperature.source))
        self.assertIn('needs validation', ranked[0].reason)
        self.assertIn('different temperature', ranked[1].reason)

    def test_grid_and_rotation_agreement_must_cover_every_selected_measurement(self):
        measurement = source(token='TG_rep01')
        unknown = source('', token='TG_rep02', grid=())
        candidate = source()
        self.assertEqual(bn_grid_state((measurement, unknown), candidate), 1)
        self.assertNotIn('Same angle', rank_bn_candidates((measurement, unknown), (candidate,))[0].reason)
        different_grid = replace(measurement, source='second.csv', spectral_grid=(740., 760., 781.))
        self.assertEqual(bn_grid_state((measurement, different_grid), candidate), 2)

    def test_rotation_aliases_numeric_forms_and_arm_order_are_normalized(self):
        measurements = (source('RotIn49.0deg_RotOut10deg', token='TG_rep01'),
                        source('RotationOut10deg_RotationIn49deg', token='TG_rep02'))
        matching = source('RotOut10.0deg_RotIn49deg', mtime=1.)
        conflict = source('RotOut10deg_RotIn94deg', mtime=1000.)
        ranked = rank_bn_candidates(measurements, (conflict, matching))
        self.assertEqual(names(ranked), (matching.source, conflict.source))
        self.assertIn('Same angle', ranked[0].reason)

    def test_newest_then_name_and_full_path_break_ties_deterministically(self):
        measurement = source(token='TG_rep01')
        base = source(name='YZ383_p2n1_9T_1.67K_0p3sx10_RotIn49deg_BNbackground_A.csv', mtime=10.)
        later_name = replace(base, source=base.source.replace('_A.csv', '_Z.csv'), filename=base.filename.replace('_A.csv', '_Z.csv'))
        same_name_other_directory = replace(base, source='Initial Data/ZZ/' + base.filename)
        newest = replace(base, source='Initial Data/NEW/' + base.filename, modified_time=11.)
        expected = (newest.source, base.source, same_name_other_directory.source, later_name.source)
        for ordering in permutations((base, later_name, same_name_other_directory, newest)):
            self.assertEqual(names(rank_bn_candidates((measurement,), ordering)), expected)

    def test_empty_measurement_fallback_and_unknown_timestamps_are_deterministic(self):
        valid = source(mtime=1.)
        invalid = source('RotIn94deg', mtime=float('nan'))
        ranked = rank_bn_candidates((), (invalid, valid))
        self.assertEqual(names(ranked), (valid.source, invalid.source))
        self.assertTrue(all('Same angle' not in item.reason for item in ranked))

    def test_ranking_never_reads_files_resolves_paths_or_imports_gui(self):
        measurement = source(token='TG_rep01')
        bn49, bn94 = source(), source('RotIn94deg', mtime=1000.)
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith(('PySide6', 'ui_qt')):
                raise AssertionError('Ranking imported GUI code: ' + name)
            return original_import(name, *args, **kwargs)

        with patch('builtins.open', side_effect=AssertionError('File opened')), \
             patch('builtins.__import__', side_effect=guarded_import), \
             patch.object(Path, 'open', side_effect=AssertionError('Path opened')), \
             patch.object(Path, 'stat', side_effect=AssertionError('Path stat')), \
             patch.object(Path, 'resolve', side_effect=AssertionError('Path resolved')), \
             patch.object(Path, 'iterdir', side_effect=AssertionError('Directory read')):
            ranked = rank_bn_candidates((measurement,), (bn94, bn49))
        self.assertEqual(names(ranked), (bn49.source, bn94.source))


if __name__ == '__main__':
    unittest.main()
