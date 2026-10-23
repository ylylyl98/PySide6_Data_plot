import json
import unittest
from unittest.mock import patch
import numpy as np
from core.drr_periodic_texture import periodic_texture, _coordinates, _select_model, _guard_long_gaps, _fit


def sample(*, uneven=False, chirp=False):
    energy = np.sort(1/np.linspace(1/1.8, 1/1.4, 801)) if uneven else np.linspace(1.4, 1.8, 801)
    gates = np.linspace(-1, 1, 24)
    x = (energy-1.6)/.4
    if chirp:
        texture = .004*np.sin(2*np.pi*(60*x+22*x*x))
    else:
        texture = .003*np.sin(2*np.pi*(energy-1.4)/.0062)+.0015*np.cos(2*np.pi*(energy-1.4)/.0091)
    values = texture[None, :]+.03*gates[:, None]+.02*(energy-1.6)**2
    values += np.random.default_rng(5).normal(0, .0001, values.shape)
    anchors = np.ones(values.shape, bool)
    anchors[:, (energy > 1.59) & (energy < 1.605)] = False
    return energy, values, anchors, texture


class PeriodicTextureTests(unittest.TestCase):
    def test_two_periods_predict_short_protected_gap_with_independent_holdouts(self):
        energy, values, anchors, truth = sample()
        original = values.copy()
        texture, coverage, report = periodic_texture(energy, values, anchors)
        self.assertEqual(report['status'], 'Applied')
        gap = ~anchors.any(axis=0)
        self.assertTrue(np.all(coverage[gap] > .99))
        self.assertLess(np.sqrt(np.mean((texture[gap]-truth[gap])**2)), .00015)
        for window in report['windows']:
            if window['status'] != 'Applied':
                continue
            self.assertEqual(len(window['validation_blocks']), 2)
            self.assertGreaterEqual(window['validated_gap_width_eV'], window['gap_width_eV'])
            for block in window['validation_blocks']:
                self.assertTrue(block['independent_frequency_selection'])
                self.assertTrue(block['independent_model_selection'])
                self.assertGreaterEqual(block['minimum_phase_cycles'], 2.)
        self.assertFalse(report['validation']['experimental_repeats'])
        np.testing.assert_array_equal(values, original)
        json.dumps(report, allow_nan=False)

    def test_nonuniform_energy_coordinates_preserve_prediction(self):
        energy, values, anchors, truth = sample(uneven=True)
        texture, coverage, report = periodic_texture(energy, values, anchors)
        self.assertEqual(report['status'], 'Applied')
        gap = ~anchors.any(axis=0)
        self.assertGreater(np.min(coverage[gap]), .99)
        self.assertLess(np.sqrt(np.mean((texture[gap]-truth[gap])**2)), .0002)
        for window in report['windows']:
            if window['status'] == 'Applied':
                self.assertTrue(all(block['minimum_phase_cycles'] >= 2 for block in window['validation_blocks']))

    def test_slow_chirp_is_predicted_only_after_holdout_validation(self):
        energy, values, anchors, truth = sample(chirp=True)
        texture, coverage, report = periodic_texture(energy, values, anchors)
        self.assertEqual(report['status'], 'Applied')
        gap = ~anchors.any(axis=0)
        self.assertLess(np.sqrt(np.mean((texture[gap]-truth[gap])**2)), .0003)
        self.assertTrue(np.all(coverage[gap] > .99))

    def test_protected_stationary_and_moving_peaks_never_enter_fitting(self):
        energy, values, anchors, _ = sample()
        baseline = periodic_texture(energy, values, anchors)
        modified = values.copy()
        for index in range(len(values)):
            center = 1.594+.006*index/(len(values)-1)
            peak = .15*np.exp(-((energy-center)/.002)**2)+.004*np.exp(-((energy-1.599)/.003)**2)
            modified[index, ~anchors[index]] += peak[~anchors[index]]
        changed = periodic_texture(energy, modified, anchors)
        np.testing.assert_array_equal(changed[0], baseline[0])
        np.testing.assert_array_equal(changed[1], baseline[1])
        gap = ~anchors.any(axis=0)
        corrected_difference = np.exp(modified[:, gap]-changed[0][gap])-np.exp(values[:, gap]-baseline[0][gap])
        expected_difference = np.exp(values[:, gap]-baseline[0][gap])*np.expm1(modified[:, gap]-values[:, gap])
        np.testing.assert_allclose(corrected_difference, expected_difference, atol=1e-15, rtol=1e-11)

    def test_noise_and_inconsistent_gate_phase_do_not_get_a_prediction(self):
        energy, values, anchors, _ = sample()
        rng = np.random.default_rng(92)
        noise = rng.normal(0, .002, values.shape)
        angle = 2*np.pi*(energy-1.4)/.0062
        inconsistent = .004*np.sin(angle+np.r_[np.full(12, -.7), np.full(12, .7)][:, None])
        for observed in (noise, inconsistent+noise*.05):
            texture, coverage, report = periodic_texture(energy, observed, anchors)
            np.testing.assert_array_equal(texture, 0.)
            np.testing.assert_array_equal(coverage, 0.)
            self.assertNotEqual(report['status'], 'Applied')
            self.assertTrue(report['reason'])

    def test_wide_unsupported_gap_remains_uncorrected(self):
        energy, values, _, _ = sample()
        anchors = np.ones(values.shape, bool)
        gap = (energy > 1.53) & (energy < 1.67)
        anchors[:, gap] = False
        texture, coverage, report = periodic_texture(energy, values, anchors)
        np.testing.assert_array_equal(texture[gap], 0.)
        np.testing.assert_array_equal(coverage[gap], 0.)
        self.assertGreater(report['maximum_requested_gap_width_eV'], .13)
        self.assertEqual(report['validated_gap_width_eV'], 0.)

    def test_training_model_selection_cannot_see_holdout_values(self):
        energy, values, anchors, _ = sample()
        select = (energy >= 1.53) & (energy <= 1.665)
        ee = energy[select]
        yy = np.mean(values[:, select], axis=0)
        support = anchors[0, select].copy()
        support[(ee > 1.545) & (ee < 1.579)] = False
        coordinate = _coordinates(ee, np.flatnonzero(select))
        first = _select_model(coordinate, yy, support, float(np.ptp(ee)), 3)
        altered = yy.copy()
        altered[~support] = 1000*np.sin(ee[~support]*17865)
        second = _select_model(coordinate, altered, support, float(np.ptp(ee)), 3)
        self.assertIsNotNone(first)
        self.assertEqual(first[:2], second[:2])
        np.testing.assert_array_equal(first[2]['prediction'], second[2]['prediction'])

    def test_masked_bands_do_not_lower_native_frequency_search_limit(self):
        x = np.linspace(-.5, .5, 241)
        frequency = 35.7
        truth = .004*np.sin(2*np.pi*frequency*x+.4)
        values = truth+.03*x*x+.02*x
        anchors = abs(x) > .15
        # A second continuous band is held out, leaving too few anchors for
        # count/5 to include the true frequency, despite unchanged sampling.
        anchors[(x > -.38) & (x < -.23)] = False
        self.assertLess(anchors.sum()/5, frequency)
        model = _fit(x, values, anchors, .13, 1, False)
        self.assertIsNotNone(model)
        self.assertLess(abs(model['components'][0][0]-frequency), .01)
        np.testing.assert_allclose(model['prediction'], truth, atol=1e-6)
        altered = values.copy()
        altered[~anchors] += 10*np.cos(43*x[~anchors])
        hidden = _fit(x, altered, anchors, .13, 1, False)
        np.testing.assert_array_equal(hidden['prediction'], model['prediction'])

    def test_chirped_model_respects_native_phase_sampling_on_uneven_grid(self):
        x = np.linspace(0, 1, 151)**1.15-.5
        phase = 2*np.pi*(20*x+6*x*x)
        values = .004*np.sin(phase)+.02*x*x
        model = _fit(x, values, np.ones(len(x), bool), .13, 1, True)
        self.assertIsNotNone(model)
        for frequency, chirp in model['components']:
            increments = np.diff(frequency*x+chirp*x*x)
            self.assertLessEqual(np.max(abs(increments)), .2+1e-12)
        self.assertLess(np.sqrt(np.mean((model['prediction']-.004*np.sin(phase))**2)), .0002)

    def test_single_energy_fallback_requires_all_four_predefined_holdouts(self):
        energy, values, anchors, _ = sample()
        truth = .004*np.sin(2*np.pi*(energy-1.4)/.0047)
        values = np.tile(truth, (len(values), 1))+.03*np.linspace(-1, 1, len(values))[:, None]
        with patch('core.drr_periodic_texture._select_model', return_value=None):
            texture, coverage, report = periodic_texture(energy, values, anchors)
        self.assertEqual(report['status'], 'Applied')
        window = next(window for window in report['windows'] if window['status'] == 'Applied')
        self.assertEqual(window['family'], 'energy_one')
        self.assertEqual([attempt['family'] for attempt in window['family_attempts']], ['general_periodic', 'energy_one'])
        self.assertEqual(window['family_attempts'][0]['status'], 'No resolved periodic component')
        self.assertEqual(len(window['validation_blocks']), 4)
        self.assertEqual([block['validation_family'] for block in window['validation_blocks']],
                         ['gap_matched', 'gap_matched', 'outer_wider', 'outer_wider'])
        self.assertTrue(all(block['passed'] and block['minimum_phase_cycles'] >= 2
                            for block in window['validation_blocks']))
        gap = ~anchors.any(axis=0)
        self.assertGreater(np.min(coverage[gap]), .99)
        np.testing.assert_allclose(texture[gap], truth[gap], atol=1e-6)
        modified = values.copy()
        modified[~anchors] += .1
        with patch('core.drr_periodic_texture._select_model', return_value=None):
            changed = periodic_texture(energy, modified, anchors)
        np.testing.assert_array_equal(changed[0], texture)
        json.dumps(report, allow_nan=False)

    def test_single_energy_fallback_rejects_incomplete_four_holdout_geometry(self):
        energy, values, anchors, _ = sample()
        truth = .004*np.sin(2*np.pi*(energy-1.4)/.0047)
        values = np.tile(truth, (len(values), 1))
        with patch('core.drr_periodic_texture._select_model', return_value=None), \
                patch('core.drr_periodic_texture._edge_holdouts', return_value=[]):
            texture, coverage, report = periodic_texture(energy, values, anchors)
        np.testing.assert_array_equal(texture, 0.)
        np.testing.assert_array_equal(coverage, 0.)
        self.assertIn('Incomplete predefined holdout geometry', report['reason'])
        attempt = report['windows'][0]['family_attempts'][-1]
        self.assertEqual(attempt['family'], 'energy_one')
        self.assertEqual([geometry['available_blocks'] for geometry in attempt['validation_geometry']], [2, 0])

    def test_adjacent_long_gap_has_smooth_zero_guard_not_a_hard_edge(self):
        energy = np.linspace(1.4, 1.8, 4001)
        support = np.ones(len(energy), bool)
        support[(energy >= 1.59) & (energy <= 1.70)] = False
        weight = _guard_long_gaps(energy, support, np.ones(len(energy)), .035)
        np.testing.assert_array_equal(weight[~support], 0.)
        left = np.flatnonzero(~support)[0]
        self.assertLess(weight[left-1], .0001)
        self.assertGreater(weight[left-60], .99)
        # At the joining point both numerical slope and curvature tend to zero;
        # an abrupt mask would instead leave an order-one second difference.
        curvature_step = abs(np.diff(weight, n=2)[left-2])
        self.assertLess(curvature_step, .0003)
        finer = np.linspace(1.4, 1.8, 8001)
        finer_support = ~((finer >= 1.59) & (finer <= 1.70))
        finer_weight = _guard_long_gaps(finer, finer_support, np.ones(len(finer)), .035)
        finer_left = np.flatnonzero(~finer_support)[0]
        self.assertLess(abs(np.diff(finer_weight, n=2)[finer_left-2]), .2*curvature_step)
        np.testing.assert_array_equal(support[(energy >= 1.59) & (energy <= 1.70)], False)

    def test_neighboring_short_predictions_blend_without_a_maximum_corner(self):
        energy, values, anchors, truth = sample()
        anchors[:] = True
        anchors[:, ((energy > 1.59) & (energy < 1.595)) | ((energy > 1.61) & (energy < 1.615))] = False
        texture, coverage, report = periodic_texture(energy, values, anchors)
        self.assertEqual(sum(item['status'] == 'Applied' for item in report['windows']), 2)
        overlap = (energy > 1.596) & (energy < 1.609)
        self.assertLess(np.max(abs(texture[overlap]-truth[overlap]*coverage[overlap])), .0002)
        midpoint = int(np.argmin(abs(energy-1.6025)))
        self.assertGreater(coverage[midpoint], .55)
        self.assertLess(coverage[midpoint], .9)
        # On this measured grid a max-of-windows join has a second difference
        # above .3. Smooth overlapping predictions remain below .12.
        self.assertLess(abs(np.diff(coverage, n=2)[midpoint-1]), .12)

    def test_rejects_malformed_inputs_and_honors_cancellation(self):
        energy, values, anchors, _ = sample()
        for maximum in (0, 6, True, 1.5):
            with self.assertRaises(ValueError):
                periodic_texture(energy, values, anchors, maximum_components=maximum)
        with self.assertRaises(ValueError):
            periodic_texture(energy[::-1], values, anchors)
        with self.assertRaises(ValueError):
            periodic_texture(energy, values, anchors.astype(int))
        with patch('core.drr_periodic_texture.check_load_cancelled', side_effect=RuntimeError('cancelled')):
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                periodic_texture(energy, values, anchors)


if __name__ == '__main__':
    unittest.main()
