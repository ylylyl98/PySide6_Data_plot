from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_fitted_background import estimate_background, load_fitted_group, load_intensity, read_reference_raw
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope
from core.loader import DataCube
from tests.test_drr_fitted_background import SETTINGS, synthetic, write_csv


class GroupAverageBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        setting = patch.dict(os.environ, DPTK_FITTED_CACHE_DIR=str(self.folder/'cache'))
        setting.start()
        self.addCleanup(setting.stop)

    def write(self, name, cube):
        write_csv(self.folder/name, cube)
        return load_intensity(self.folder, name, y_axis='bg')

    def test_unequal_native_grids_align_raw_to_common_support_before_one_fit(self):
        energy_a = np.linspace(1.4, 2., 401)
        energy_b = np.linspace(1.45, 1.95, 321)
        gate_a = np.array([-2., -1., 0., 1., 2.])
        gate_b = np.array([-1.5, -.5, .5, 1.5])
        def frame(energy, gate, gain):
            values = gain*(10000+2000*energy+gate[:, None]*(100+30*energy))
            return DataCube(energy, gate, values, 'BG (V)', 'linear raw fixture', 'counts')
        first = self.write('a.csv', frame(energy_a, gate_a, 1.))
        second = self.write('b.csv', frame(energy_b, gate_b, 3.))
        selected_e = (first.energy >= second.energy.min()) & (first.energy <= second.energy.max())
        selected_g = (first.gate >= second.gate.min()) & (first.gate <= second.gate.max())
        energy, gate = first.energy[selected_e], first.gate[selected_g]
        second_energy_aligned = np.asarray([np.interp(energy, second.energy, row) for row in second.Z])
        second_aligned = np.asarray([np.interp(gate, second.gate, column) for column in second_energy_aligned.T]).T
        expected = replace(first, energy=energy, gate=gate,
                           Z=(first.Z[np.ix_(selected_g, selected_e)]+second_aligned)/2)
        from core import drr_background_cache as cache
        with patch.object(cache, 'estimate_background', wraps=estimate_background) as fit:
            actual, references = load_fitted_group(self.folder, ['a.csv', 'b.csv'], SETTINGS, y_axis='bg')
        self.assertEqual(len(references), 1)
        self.assertEqual(fit.call_count, 1)
        averaged_input = fit.call_args.args[0]
        np.testing.assert_array_equal(averaged_input.energy, energy)
        np.testing.assert_array_equal(averaged_input.gate, gate)
        np.testing.assert_allclose(averaged_input.Z, expected.Z, rtol=0, atol=1e-10)
        np.testing.assert_allclose(actual.Z, estimate_background(expected, SETTINGS)[0].Z, atol=1e-11)
        np.testing.assert_array_equal(actual.gate, [-1., 0., 1.])
        restored = read_reference_raw(self.folder, references[0], y_axis='bg')
        np.testing.assert_array_equal(restored.energy, actual.energy)
        np.testing.assert_array_equal(restored.gate, actual.gate)
        np.testing.assert_allclose(restored.Z, expected.Z, rtol=0, atol=1e-10)

    def test_no_common_gate_or_energy_support_is_an_error_not_variable_weights(self):
        raw, _, _ = synthetic()
        self.write('a.csv', raw)
        for other in (replace(raw, gate=raw.gate+10), replace(raw, energy=raw.energy+1)):
            self.write('b.csv', other)
            with self.subTest(gate=other.gate.tolist()), self.assertRaisesRegex(ValueError, '(?i)overlap|common'):
                load_fitted_group(self.folder, ['a.csv', 'b.csv'], SETTINGS, y_axis='bg')

    def test_invalid_member_samples_cannot_be_hidden_by_averaging_other_files(self):
        raw, _, _ = synthetic()
        self.write('a.csv', raw)
        for value in (0., -1., np.nan):
            values = raw.Z.copy()
            values[1, 30] = value
            self.write('b.csv', replace(raw, Z=values))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '(?i)finite|positive|missing'):
                load_fitted_group(self.folder, ['a.csv', 'b.csv'], SETTINGS, y_axis='bg')

    def test_reused_mean_fit_retains_all_members_and_inspector_detects_second_member_change(self):
        raw, _, _ = synthetic()
        self.write('a.csv', raw)
        self.write('b.csv', replace(raw, Z=2*raw.Z))
        result, first = load_fitted_group(self.folder, ['a.csv', 'b.csv'], SETTINGS, y_axis='bg')
        self.assertEqual(len(first), 1)
        self.assertFalse(first[0].cache_hit)
        with patch('core.drr_background_cache.estimate_background', side_effect=AssertionError('must reuse mean fit')):
            repeated, second = load_fitted_group(self.folder, ['a.csv', 'b.csv'], SETTINGS, y_axis='bg')
        self.assertEqual(len(second), 1)
        self.assertTrue(second[0].cache_hit)
        np.testing.assert_array_equal(repeated.Z, result.Z)
        metadata = json.dumps(second[0].metadata())
        self.assertIn('a.csv', metadata)
        self.assertIn('b.csv', metadata)
        self.write('b.csv', replace(raw, Z=2.01*raw.Z))
        with self.assertRaisesRegex(ValueError, '(?i)changed|reload'):
            read_reference_raw(self.folder, second[0], y_axis='bg')
        _, changed = load_fitted_group(self.folder, ['a.csv', 'b.csv'], SETTINGS, y_axis='bg')
        self.assertFalse(changed[0].cache_hit)

    def test_member_identity_invalidates_cache_even_if_mean_values_are_identical(self):
        raw, _, _ = synthetic()
        for name in ('a.csv', 'b.csv', 'c.csv'):
            self.write(name, raw)
        initial, first = load_fitted_group(self.folder, ['a.csv', 'b.csv'], SETTINGS, y_axis='bg')
        swapped, second = load_fitted_group(self.folder, ['a.csv', 'c.csv'], SETTINGS, y_axis='bg')
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        np.testing.assert_array_equal(initial.Z, swapped.Z)
        self.assertFalse(second[0].cache_hit)
        self.assertIn('c.csv', json.dumps(second[0].metadata()))

    def test_cancelled_raw_read_does_not_start_another_file_or_fit_partial_mean(self):
        raw, _, _ = synthetic()
        self.write('a.csv', raw)
        self.write('b.csv', raw)
        cancellation = LoadCancellation()
        from core import drr_fitted_background as background
        original = background.load_intensity
        def cancel_after_read(*args, **kwargs):
            result = original(*args, **kwargs)
            cancellation.cancel()
            return result
        with load_scope(cancellation), patch.object(background, 'load_intensity', side_effect=cancel_after_read) as load, \
                patch('core.drr_background_cache.estimate_background') as fit:
            with self.assertRaises(LoadCancelled):
                load_fitted_group(self.folder, ['a.csv', 'b.csv'], SETTINGS, y_axis='bg')
        self.assertEqual(load.call_count, 1)
        fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
