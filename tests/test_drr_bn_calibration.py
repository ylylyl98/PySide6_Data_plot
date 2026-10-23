import json
import unittest
from dataclasses import replace

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from core.drr_bn_calibration import BnCalibrationSettings, evaluate_bn_group
from core.load_cancellation import LoadCancellation, LoadCancelled


def synthetic(shifts=.30, repeats=5, gain=1.07, seed=10, peak=None, *, localized=False,
              protected_regions=(), energy_max=2.10, drift_core=(1.900, 1.970), periods=(7.3, 12.1)):
    """Known reference texture plus sample envelope, optional real narrow peak."""
    rng = np.random.default_rng(seed)
    energy = np.linspace(1.50, energy_max, int(round((energy_max-1.50)/.0005))+1)
    gates = np.linspace(-1, 1, 9)
    index = np.arange(len(energy), dtype=float)
    log_background = 10 + .3*(energy-1.8) + .004*np.sin(index*2*np.pi/periods[0]) + .002*np.cos(index*2*np.pi/periods[1])
    background = np.exp(log_background)
    fine = log_background-savgol_filter(log_background, 101, 3)
    spline = CubicSpline(index, fine)
    # Independent raised-cosine ground truth, deliberately different from the
    # engine's quintic taper. A global drift is also retained as an adversarial
    # case: locally correcting it can create curvature at the support edges.
    envelope = np.ones_like(energy)
    if localized:
        envelope = (.5-.5*np.cos(np.pi*np.clip((energy-drift_core[0]+.01)/.01, 0, 1)))
        envelope *= .5-.5*np.cos(np.pi*np.clip((drift_core[1]+.01-energy)/.01, 0, 1))
        for lo, hi in protected_regions:
            distance = np.maximum(lo-.006-energy, energy-hi-.006)
            envelope *= .5-.5*np.cos(np.pi*np.clip(distance/.01, 0, 1))
    if np.isscalar(shifts):
        shifts = [shifts]*repeats
    values = []
    for shift in shifts:
        shifted = gain*spline(np.clip(index+shift, 0, len(index)-1))
        ref = background*np.exp(envelope*(shifted-fine))
        frames = []
        for gate in gates:
            sample_log = .02 + .01*(energy-1.9) + .002*gate*(energy-1.9)**2
            if peak is not None:
                center, width, amplitude = peak
                sample_log = sample_log + amplitude*np.exp(-.5*((energy-center)/width)**2)
            frames.append(ref*np.exp(sample_log+rng.normal(0, 2e-5, len(energy))))
        values.append(frames)
    # Independent synthetic reference noise, centered so the declared average
    # exactly reproduces the actual original background.
    noise = rng.normal(0, 2e-5, (20, len(energy)))
    noise -= noise.mean(0)
    reference_frames = background*(1+noise)
    return energy, np.array(values), background, gates, reference_frames


def localized_synthetic(*args, **kwargs):
    return synthetic(*args, localized=True, **kwargs)


WIDE_TRUSTED = BnCalibrationSettings(core_region=(1.900, 2.450), train_region=(1.900, 2.400),
    validation_region=(2.420, 2.450), trusted_regions=((1.88, 2.47),), uncertain_regions=())


def wide_synthetic(*args, **kwargs):
    """Declared wide pure-background control; no real group's ranges are changed."""
    return synthetic(*args, localized=True, energy_max=2.60, drift_core=(1.900, 2.450),
                     periods=(8.7, 14.2), **kwargs)


TRUSTED = BnCalibrationSettings(trusted_regions=((1.88, 1.99),), uncertain_regions=())


class BnCalibrationTests(unittest.TestCase):
    def evaluate(self, inputs, settings=WIDE_TRUSTED, **kwargs):
        e, z, b, g, frames = inputs
        return evaluate_bn_group(e, z, b, gates=g, settings=settings, background_repeats=frames, **kwargs)

    def test_known_drift_passes_independent_confirmation_and_preserves_inputs(self):
        inputs = wide_synthetic()
        snapshots = [v.copy() for v in inputs]
        result = self.evaluate(inputs)
        self.assertTrue(result.eligible, result.reasons)
        self.assertEqual(result.status, 'accepted')
        self.assertNotEqual(result.applied_method, 'original')
        self.assertGreater(result.report['final_confirmation']['mean_improvement_fraction'], .50)
        self.assertEqual(result.report['split']['train'], [0, 1])
        self.assertEqual(result.report['split']['selection'], [2])
        self.assertEqual(result.report['split']['final'], [3, 4])
        outside = (inputs[0] <= 1.89) | (inputs[0] >= 2.46)
        np.testing.assert_array_equal(result.applied_background[outside], inputs[2][outside])
        for actual, expected in zip(inputs, snapshots):
            np.testing.assert_array_equal(actual, expected)
        json.dumps(result.report, allow_nan=False)

    def test_four_repeats_use_one_train_one_select_two_final(self):
        result = self.evaluate(wide_synthetic(repeats=4))
        self.assertTrue(result.eligible, result.reasons)
        self.assertEqual(result.report['split']['train'], [0])
        self.assertEqual(result.report['split']['selection'], [1])
        self.assertEqual(result.report['split']['final'], [2, 3])

    def test_no_drift_keeps_original_background(self):
        inputs = wide_synthetic(shifts=0., gain=1.)
        result = self.evaluate(inputs)
        self.assertEqual(result.selected_method, 'original')
        self.assertEqual(result.applied_method, 'original')
        np.testing.assert_array_equal(result.applied_background, inputs[2])

    def test_real_narrow_peak_is_protected_including_native_derivative_stencil(self):
        settings = replace(WIDE_TRUSTED, protected_regions=((1.938, 1.942),))
        inputs = wide_synthetic(peak=(1.940, .0007, .07), protected_regions=settings.protected_regions)
        result = self.evaluate(inputs, settings)
        self.assertTrue(result.eligible, result.reasons)
        e, z, b, _, _ = inputs
        protected = (e >= 1.938) & (e <= 1.942)
        original = z[0, 0]/b-1
        corrected = z[0, 0]/result.applied_background-1
        np.testing.assert_array_equal(original[protected], corrected[protected])
        d_old = savgol_filter(original, 21, 2, deriv=2)
        d_new = savgol_filter(corrected, 21, 2, deriv=2)
        np.testing.assert_array_equal(d_old[protected], d_new[protected])

    def test_uncertain_region_blocks_automatic_application_even_with_lower_rms(self):
        settings = replace(WIDE_TRUSTED, uncertain_regions=((1.94, 2.02),))
        result = self.evaluate(wide_synthetic(), settings)
        self.assertNotEqual(result.selected_method, 'original')
        self.assertFalse(result.eligible)
        self.assertEqual(result.applied_method, 'original')
        self.assertIn('correction_overlaps_uncertain_region', result.reasons)

    def test_single_file_is_exploratory_not_independent_confirmation(self):
        result = self.evaluate(wide_synthetic(repeats=1))
        self.assertEqual(result.status, 'exploratory')
        self.assertFalse(result.eligible)
        self.assertEqual(result.applied_method, 'original')
        self.assertEqual(result.report['split']['final'], [])

    def test_final_holdout_cannot_change_selected_candidate_or_fitted_parameters(self):
        base = wide_synthetic()
        before = self.evaluate(base)
        altered = list(base)
        altered[1] = base[1].copy()
        opposite = wide_synthetic(shifts=[.30, .30, .30, -.35, -.35])[1]
        altered[1][3:] = opposite[3:]
        after = self.evaluate(altered)
        self.assertEqual(before.selected_method, after.selected_method)
        for method in ('local_shift', 'local_shift_gain'):
            self.assertEqual(before.report['candidates'][method]['parameters'], after.report['candidates'][method]['parameters'])
            np.testing.assert_array_equal(before.candidate_backgrounds[method], after.candidate_backgrounds[method])
        self.assertFalse(after.eligible)
        self.assertIn('final_holdout_benefit_not_confirmed', after.reasons)
        self.assertIn('temporal_shift_sign_reversal', after.reasons)

    def test_reference_half_instability_blocks_application(self):
        inputs = list(wide_synthetic())
        e, _, b, _, _ = inputs
        wave = .003*np.sin(np.arange(len(e))*2*np.pi/7.3 + .8)
        inputs[4] = np.stack([b*(1+wave), b*(1-wave)])
        result = self.evaluate(inputs)
        self.assertFalse(result.eligible)
        self.assertTrue(any(reason.startswith('background_') for reason in result.reasons), result.reasons)

    def test_missing_reference_repeats_is_explicitly_unverified(self):
        e, z, b, g, _ = wide_synthetic()
        result = evaluate_bn_group(e, z, b, gates=g, settings=WIDE_TRUSTED)
        self.assertFalse(result.eligible)
        self.assertIn('background_sensitivity_unverified', result.reasons)

    def test_invalid_grid_nonpositive_and_nonfinite_are_rejected(self):
        for problem in ('grid', 'zero', 'nan'):
            with self.subTest(problem=problem):
                inputs = list(synthetic())
                if problem == 'grid':
                    inputs[0][30] = inputs[0][29]
                elif problem == 'zero':
                    inputs[1][0, 0, 0] = 0
                else:
                    inputs[2][0] = np.nan
                result = self.evaluate(inputs)
                self.assertEqual(result.status, 'invalid')
                self.assertFalse(result.eligible)
                json.dumps(result.report, allow_nan=False)

    def test_descending_nonuniform_energy_and_explicit_repeat_order(self):
        inputs = list(wide_synthetic())
        inputs[0] = inputs[0]+.002*(inputs[0]-1.8)**2
        inputs[0] = inputs[0][::-1]
        inputs[1] = inputs[1][..., ::-1]
        inputs[2] = inputs[2][::-1]
        inputs[4] = inputs[4][..., ::-1]
        result = self.evaluate(inputs, repeat_order=[4, 3, 2, 1, 0])
        self.assertTrue(result.eligible, result.reasons)
        self.assertEqual(result.report['split']['train'], [4, 3])
        self.assertEqual(result.report['split']['final'], [1, 0])

    def test_two_independent_half_reference_means_are_supported(self):
        inputs = list(wide_synthetic())
        frames = inputs[4]
        inputs[4] = np.stack([frames[:10].mean(0), frames[10:].mean(0)])
        result = self.evaluate(inputs)
        self.assertTrue(result.eligible, result.reasons)

    def test_parameter_bound_prevents_automatic_application(self):
        result = self.evaluate(wide_synthetic(shifts=.8, gain=1.), replace(WIDE_TRUSTED, max_shift_pixels=.5))
        self.assertFalse(result.eligible)
        self.assertIn('selected_parameter_at_bound', result.reasons)

    def test_invalid_configuration_reports_json_safe_reason(self):
        result = self.evaluate(synthetic(), replace(TRUSTED, min_improvement_fraction=float('nan')))
        self.assertEqual(result.status, 'invalid')
        json.dumps(result.report, allow_nan=False)

    def test_insufficient_channels_are_rejected_without_window_retuning(self):
        inputs = list(synthetic())
        inputs[0] = inputs[0][::30]
        inputs[1] = inputs[1][..., ::30]
        inputs[2] = inputs[2][::30]
        inputs[4] = inputs[4][..., ::30]
        result = self.evaluate(inputs)
        self.assertEqual(result.status, 'invalid')
        self.assertIn('insufficient_template_bandwidth', result.reasons)

    def test_cooperative_cancellation_propagates(self):
        cancellation = LoadCancellation()
        cancellation.cancel()
        with self.assertRaises(LoadCancelled):
            self.evaluate(synthetic(), cancellation=cancellation)


if __name__ == '__main__':
    unittest.main()
