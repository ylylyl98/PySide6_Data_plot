"""BN relaxed conditions require explicit policy and reviewable evidence."""
from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.drr_bn_matching import BnMatchCache, BnMatchPlan, build_bn_match_plan, plan_is_current, override_bn_match


class BnRelaxedConditionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.measurement = self.write('YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_TG_rep01', 1000, varying=True)

    def write(self, name, stamp, *, varying=False, grid='740,760,780'):
        path = self.root / (name + '.csv')
        path.write_text(f'Vbg,Vtg,{grid}\n0,0,10,11,12\n' +
                        ('1,1,11,12,13\n' if varying else '0,0,11,12,13\n'))
        os.utime(path, (stamp, stamp))
        return path.name

    def candidate(self, field='9T', stamp=900, suffix='001', **kwargs):
        return self.write(f'YZ383_p2n1_{field}_1.67KREF_730nmc_0p3sx10_RotIn94deg_BNbackground_{suffix}', stamp, **kwargs)

    def plan(self, **kwargs):
        return build_bn_match_plan(self.root, (self.measurement,), **kwargs)

    def test_default_strict_and_explicit_difference_warning(self):
        candidate = self.candidate('8T')
        self.assertFalse(self.plan().ready)
        plan = self.plan(relaxed_conditions=('magnetic_field',))
        self.assertTrue(plan.ready, plan.resolution.reason)
        self.assertEqual(plan.rows[0].baseline_file, candidate)
        self.assertTrue(any('Magnetic field' in w and '9' in w and '8' in w for w in plan.warnings), plan.warnings)
        self.assertFalse(self.plan(relaxed_conditions=('temperature',)).ready)

    def test_exact_match_precedes_nearer_relaxed_and_cache_policy_is_separate(self):
        relaxed = self.candidate('8T', 999)
        exact = self.candidate('9T', 500, '002')
        plan = self.plan(relaxed_conditions=('magnetic_field',))
        self.assertEqual(plan.rows[0].baseline_file, exact)
        self.assertEqual(plan.batches[0].candidate_files, (exact,))
        cache = BnMatchCache()
        cache.confirm(plan, warnings_acknowledged=True)
        self.assertIsNone(cache.find(self.root, (self.measurement,)))
        self.assertIsNotNone(cache.find(self.root, (self.measurement,), relaxed_conditions=('magnetic_field',)))
        self.assertNotEqual(relaxed, exact)

    def test_plan_serialization_signs_policy_and_rejects_tampering(self):
        self.candidate('8T')
        plan = self.plan(relaxed_conditions=('magnetic_field',))
        restored = BnMatchPlan.from_dict(plan.to_dict())
        self.assertTrue(plan_is_current(restored))
        payload = plan.to_dict()
        payload['relaxed_conditions'] = []
        with self.assertRaises(ValueError):
            BnMatchPlan.from_dict(payload)
        self.assertFalse(plan_is_current(replace(plan, relaxed_conditions=())))
        for invalid in ('magnetic_field', ('bogus',)):
            with self.assertRaises(ValueError):
                self.plan(relaxed_conditions=invalid)

    def test_manual_override_rebuilds_and_rejects_stale_or_ineligible(self):
        first = self.candidate('9T', 900)
        second = self.candidate('8T', 800, '002')
        plan = self.plan(relaxed_conditions=('magnetic_field',))
        selected = override_bn_match(plan, (self.measurement,), second)
        self.assertTrue(selected.ready)
        self.assertEqual(selected.rows[0].baseline_file, second)
        self.assertEqual(selected.manual_assignments, ((self.measurement, second),))
        self.assertNotEqual(selected.signature, plan.signature)
        self.assertTrue(any('Manual BN selection' in warning for warning in selected.warnings))
        self.assertTrue(any('Magnetic field' in warning for warning in selected.warnings))
        self.assertEqual(BnMatchPlan.from_dict(selected.to_dict()), selected)
        with self.assertRaises(ValueError):
            override_bn_match(self.plan(), (self.measurement,), second)
        with self.assertRaises(ValueError):
            override_bn_match(plan, (self.measurement,), 'unselected.csv')
        self.root.joinpath(first).write_text('changed')
        with self.assertRaises(ValueError):
            override_bn_match(plan, (self.measurement,), second)

    def test_manual_override_resolves_tied_unresolved_plan(self):
        first = self.candidate('9T', 900)
        self.candidate('9T', 1100, '002')
        plan = self.plan()
        self.assertFalse(plan.ready)
        selected = override_bn_match(plan, (self.measurement,), first)
        self.assertTrue(selected.ready)
        self.assertEqual(selected.rows[0].baseline_file, first)

    def test_missing_values_need_explicit_relaxation_and_rank_below_complete(self):
        prefixes = {
            'sample': 'p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_',
            'position': 'YZ383_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_',
            'temperature': 'YZ383_p2n1_9T_730nmc_0p3sx10_RotIn94deg_',
        }
        for key, prefix in prefixes.items():
            with self.subTest(key=key):
                candidate = self.write(prefix + 'BNbackground_001', 999)
                self.assertFalse(self.plan(candidate_files=(candidate,)).ready)
                plan = self.plan(candidate_files=(candidate,), relaxed_conditions=(key,))
                self.assertTrue(plan.ready, plan.resolution.reason)
                self.assertIn('unknown', plan.rows[0].assignment.selection_reason)
                self.assertTrue(any('unknown' in warning.lower() for warning in plan.warnings))
                exact = self.candidate('9T', 500, '002')
                plan = self.plan(candidate_files=(candidate, exact), relaxed_conditions=(key,))
                self.assertEqual(plan.rows[0].baseline_file, exact)
                self.root.joinpath(candidate).unlink()
                self.root.joinpath(exact).unlink()

    def test_nominal_center_relaxation_keeps_native_grid_strict(self):
        different = self.write('YZ383_p2n1_9T_1.67KREF_740nmc_0p3sx10_RotIn94deg_BNbackground_001', 900)
        self.assertFalse(self.plan().ready)
        plan = self.plan(relaxed_conditions=('wavelength_center_nm',))
        self.assertTrue(plan.ready, plan.resolution.reason)
        self.assertEqual(plan.rows[0].baseline_file, different)
        self.assertIn('730', plan.rows[0].assignment.selection_reason)
        self.assertIn('740', plan.rows[0].assignment.selection_reason)
        self.root.joinpath(different).unlink()
        self.write('YZ383_p2n1_9T_1.67KREF_740nmc_0p3sx10_RotIn94deg_BNbackground_002', 900,
                   grid='741,760,780')
        self.assertFalse(self.plan(relaxed_conditions=('wavelength_center_nm',)).ready)

    def test_wrong_grid_and_varying_or_wrong_role_remain_ineligible(self):
        self.candidate('8T', grid='741,760,780')
        self.candidate('8T', 800, '002', varying=True)
        self.write('YZ383_p2n1_8T_1.67KREF_730nmc_0p3sx10_RotIn94deg_Pbackground_003', 950)
        self.assertFalse(self.plan(relaxed_conditions=('magnetic_field',)).ready)


if __name__ == '__main__':
    unittest.main()
