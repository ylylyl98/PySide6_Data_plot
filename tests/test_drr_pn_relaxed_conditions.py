"""Relaxed P/N conditions remain explicit, reviewable and subordinate to strict matches."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

from core.drr_match_conditions import CONDITION_LABELS, normalize_relaxed_conditions
from core.drr_pn_matching import (build_pn_match_plan, override_pn_match,
    plan_is_current, prepare_pn_confirmation)


class PnRelaxedConditionsTests(unittest.TestCase):
    prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_polH_1mW_'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.measurement = self.write(self.prefix + 'TG.csv', 1000, varying=True)

    def write(self, name, stamp, *, varying=False, gate=-10, columns='740,760,780'):
        path = self.root / name
        path.write_text(f'Vbg,Vtg,{columns}\n{gate},{gate},10,11,12\n'
                        f'{gate+1 if varying else gate},{gate},11,12,13\n')
        os.utime(path, (stamp, stamp))
        return name

    def plan(self, candidates, **kwargs):
        return build_pn_match_plan(self.root, (self.measurement,), kind='p',
                                   candidate_files=candidates, **kwargs)

    def test_each_field_is_strict_by_default_and_reviewed_when_relaxed(self):
        changes = {'sample': ('YZ383', 'YZ384'), 'position': ('p2n1', 'p3n1'),
            'temperature': ('1.67KREF', '2KREF'), 'magnetic_field': ('9T', '8T'),
            'polarization': ('polH', 'polV'), 'rotation': ('RotIn94deg', 'RotIn49deg'),
            'power': ('1mW', '2mW'), 'exposure': ('0p3sx10', '0p4sx10'),
            'accumulation': ('0p3sx10', '0p3sx11'), 'wavelength_center_nm': ('730nmc', '735nmc')}
        self.assertEqual(set(changes), set(CONDITION_LABELS))
        for field, (old, new) in changes.items():
            with self.subTest(field=field):
                candidate = self.write(self.prefix.replace(old, new) + 'Pbackground.csv', 990)
                self.assertFalse(self.plan((candidate,)).ready)
                plan = self.plan((candidate,), relaxed_conditions=(field,))
                self.assertTrue(plan.ready, (field, plan.rows[0].evidence))
                self.assertEqual(plan.relaxed_conditions, (field,))
                self.assertTrue(any(field + ':' in w for w in plan.rows[0].warnings))
                self.assertIn(field + ':', plan.rows[0].assignment.selection_reason)
                with self.assertRaisesRegex(ValueError, 'acknowledge'):
                    prepare_pn_confirmation(plan)
                self.assertTrue(prepare_pn_confirmation(plan, warnings_acknowledged=True))

    def test_strict_precedes_nearer_relaxed_and_manual_override_keeps_warning(self):
        strict = self.write(self.prefix + 'Pbackground_strict.csv', 100)
        relaxed = self.write(self.prefix.replace('9T', '8T') + 'Pbackground_relaxed.csv', 999)
        plan = self.plan((strict, relaxed), relaxed_conditions=('magnetic_field',))
        self.assertEqual(plan.rows[0].baseline_file, strict)
        edited = override_pn_match(plan, self.measurement, relaxed)
        self.assertTrue(any('magnetic_field:' in w for w in edited.rows[0].warnings))
        self.assertIn('magnetic_field:', edited.rows[0].assignment.selection_reason)
        self.assertFalse(plan_is_current(replace(edited, relaxed_conditions=())))

    def test_role_and_grid_cannot_be_relaxed(self):
        wrong_role = self.write(self.prefix + 'Nbackground.csv', 990, gate=10)
        wrong_grid = self.write(self.prefix + 'Pbackground_grid.csv', 990, columns='740,760,781')
        plan = self.plan((wrong_role, wrong_grid), relaxed_conditions=tuple(CONDITION_LABELS))
        self.assertFalse(plan.ready)
        self.assertFalse(plan.rows[0].candidate_files)

    def test_policy_validation_and_canonical_order(self):
        self.assertEqual(normalize_relaxed_conditions(('power', 'sample', 'power')), ('sample', 'power'))
        for bad in ('power', ('unknown',), (None,), None):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                normalize_relaxed_conditions(bad)


if __name__ == '__main__':
    unittest.main()
