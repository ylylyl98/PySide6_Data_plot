import json
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_reference_texture import reference_texture, _calibrate, _score


def sample():
    energy = np.linspace(1.4, 1.8, 801)
    gate = np.linspace(-1, 1, 24)
    texture = (.004*np.sin(2*np.pi*(energy-1.4)/.0062)
               + .001*np.cos(2*np.pi*(energy-1.4)/.0091))
    reference = 9.8+.4*(energy-1.6)**2+texture
    reference = reference + np.random.default_rng(4).normal(0, .00008, (48, len(energy)))
    values = (10+1.05*texture[None, :]+.4*(energy-1.6)**2
              + .04*gate[:, None]+.03*gate[:, None]*(energy-1.6))
    anchors = np.ones(values.shape, bool)
    anchors[:, (energy > 1.594) & (energy < 1.606)] = False
    coverage = np.ones(len(energy))
    coverage[(energy > 1.589) & (energy < 1.611)] = 0.
    return energy, values, anchors, reference, coverage, texture


class ReferenceTextureTests(unittest.TestCase):
    def test_independent_reference_requires_all_six_shifted_holdouts(self):
        e, y, anchors, reference, shared, truth = sample()
        saved = [x.copy() for x in (e, y, anchors, reference, shared)]
        texture, coverage, report = reference_texture(e, y, anchors, e, reference, shared)
        self.assertEqual(report['status'], 'Applied')
        self.assertTrue(np.all(coverage[~anchors.any(axis=0)] > .99))
        self.assertLess(np.sqrt(np.mean((texture[~anchors.any(axis=0)]-1.05*truth[~anchors.any(axis=0)])**2)), .0004)
        self.assertLessEqual(np.max(abs(texture)), .02)
        np.testing.assert_array_equal(texture[shared == 1], 0.)
        window = next(w for w in report['windows'] if w['status'] == 'Applied')
        self.assertEqual(len(window['validation_blocks']), 6)
        self.assertEqual([v['center_shift_pixels'] for v in window['validation_blocks']], [-1, 0, 1]*2)
        self.assertTrue(all(v['passed'] and v['actual_span_eV']+1e-12 >= window['gap_width_eV']
                            and v['parameters_refitted_without_heldout_values']
                            for v in window['validation_blocks']))
        self.assertGreater(window['reference_half_frames']['correlation'], .99)
        for actual, expected in zip((e, y, anchors, reference, shared), saved):
            np.testing.assert_array_equal(actual, expected)
        json.dumps(report, allow_nan=False)

    def test_protected_sample_values_including_nan_never_enter_calibration(self):
        e, y, anchors, reference, shared, _ = sample()
        original = reference_texture(e, y, anchors, e, reference, shared)
        self.assertEqual(original[2]['status'], 'Applied')
        changed = y.copy()
        changed[~anchors] = np.nan
        result = reference_texture(e, changed, anchors, e, reference, shared)
        np.testing.assert_array_equal(result[0], original[0])
        np.testing.assert_array_equal(result[1], original[1])
        injected = y.copy()
        injected[~anchors] += .08
        result = reference_texture(e, injected, anchors, e, reference, shared)
        np.testing.assert_array_equal(result[0], original[0])
        measured = np.exp(injected-result[0])-np.exp(y-original[0])
        expected = np.exp(y-original[0])*np.expm1(injected-y)
        np.testing.assert_allclose(measured, expected, atol=1e-10, rtol=1e-11)

    def test_reference_protected_values_do_not_leak_into_detrending_or_template(self):
        e, y, anchors, reference, shared, _ = sample()
        ref_anchors = np.ones(reference.shape, bool)
        ref_anchors[:, (e > 1.535) & (e < 1.537)] = False
        with patch('core.drr_reference_texture.gate_anchor_masks', return_value=ref_anchors):
            baseline = reference_texture(e, y, anchors, e, reference, shared)
            changed = reference.copy()
            changed[~ref_anchors] += .4
            result = reference_texture(e, y, anchors, e, changed, shared)
        self.assertEqual(baseline[2]['status'], 'Applied')
        np.testing.assert_array_equal(result[0], baseline[0])
        np.testing.assert_array_equal(result[1], baseline[1])

    def test_raw_log_calibration_cannot_see_heldout_values(self):
        e, y, anchors, reference, _, _ = sample()
        template = np.median(reference, axis=0)
        context = (1.52, 1.68)
        train = (e > context[0]) & (e < context[1]) & ~((e > 1.54) & (e < 1.565))
        baseline = _calibrate(e, y, anchors, template, np.ones(len(e)), train, context)
        modified = y.copy()
        modified[:, ~train] = np.nan
        changed = _calibrate(e, modified, anchors, template, np.ones(len(e)), train, context)
        self.assertIsNotNone(baseline)
        self.assertEqual(changed['amplitude'], baseline['amplitude'])
        self.assertEqual(changed['shift_pixels'], baseline['shift_pixels'])
        np.testing.assert_array_equal(changed['prediction'], baseline['prediction'])

    def test_a_single_failed_predefined_shift_rejects_the_window(self):
        e, y, anchors, reference, shared, _ = sample()
        calls = []
        def one_failed_shift(*args):
            result = _score(*args)
            calls.append(result)
            if len(calls) == 6:
                result.update(passed=False, oscillatory_residual_ratio=.401)
            return result
        with patch('core.drr_reference_texture._score', side_effect=one_failed_shift):
            texture, coverage, report = reference_texture(e, y, anchors, e, reference, shared)
        self.assertEqual(len(calls), 6)
        np.testing.assert_array_equal(texture, 0.)
        np.testing.assert_array_equal(coverage, 0.)
        self.assertIn('predefined shifted holdout failed', report['reason'])

    def test_disagreeing_reference_frame_halves_are_rejected(self):
        e, y, anchors, reference, shared, truth = sample()
        reference[24:] += .004*np.sin(2*np.pi*(e-1.4)/.0062+1.1)-.004*np.sin(2*np.pi*(e-1.4)/.0062)
        texture, coverage, report = reference_texture(e, y, anchors, e, reference, shared)
        np.testing.assert_array_equal(texture, 0.)
        self.assertNotEqual(report['status'], 'Applied')
        self.assertTrue('halves disagree' in report.get('reason', '') or not report['windows'])

    def test_long_sample_gap_and_reference_protected_gap_remain_unfilled(self):
        e, y, anchors, reference, shared, _ = sample()
        long_gap = (e > 1.55) & (e < 1.65)
        shared[long_gap] = 0.
        anchors[:, long_gap] = False
        texture, coverage, report = reference_texture(e, y, anchors, e, reference, shared)
        np.testing.assert_array_equal(texture, 0.)
        self.assertIn('Enclosing shared gap exceeds 25 meV', report['reason'])
        e, y, anchors, reference, shared, _ = sample()
        ref_anchors = np.ones(reference.shape, bool)
        ref_anchors[:, shared == 0] = False
        with patch('core.drr_reference_texture.gate_anchor_masks', return_value=ref_anchors):
            texture, coverage, _ = reference_texture(e, y, anchors, e, reference, shared)
        np.testing.assert_array_equal(texture[shared == 0], 0.)
        np.testing.assert_array_equal(coverage[shared == 0], 0.)

    def test_inadequate_training_support_rejects_without_linalg_failure(self):
        e, y, anchors, reference, shared, _ = sample()
        anchors[:, (e < 1.58) | (e > 1.62)] = False
        texture, coverage, report = reference_texture(e, y, anchors, e, reference, shared)
        np.testing.assert_array_equal(texture, 0.)
        self.assertNotEqual(report['status'], 'Applied')
        self.assertTrue(report['reason'])
        json.dumps(report, allow_nan=False)

    def test_calibrated_reference_larger_than_two_percent_is_rejected_not_clipped(self):
        e, y, anchors, reference, shared, original = sample()
        strong = .018*np.sin(2*np.pi*(e-1.4)/.0062)
        reference += strong-original
        y += (1.4*strong-1.05*original)[None, :]
        # Isolate the final amplitude guard from the earlier peak detector.
        # Even a reference deemed peak-free must not exceed the correction cap.
        with patch('core.drr_reference_texture.gate_anchor_masks',
                   return_value=np.ones(reference.shape, bool)):
            texture, coverage, report = reference_texture(e, y, anchors, e, reference, shared)
        np.testing.assert_array_equal(texture, 0.)
        np.testing.assert_array_equal(coverage, 0.)
        self.assertIn('exceeds 0.02 log correction', report['reason'])
        self.assertTrue(all(v['passed'] for v in report['windows'][0]['validation_blocks']))

    def test_exact_grid_shape_and_finite_input_validation_and_cancellation(self):
        e, y, anchors, reference, shared, _ = sample()
        with self.assertRaisesRegex(ValueError, 'exactly matching'):
            reference_texture(e, y, anchors, e+1e-12, reference, shared)
        with self.assertRaisesRegex(ValueError, 'exactly matching'):
            reference_texture(e[::-1], y, anchors, e[::-1], reference, shared)
        with self.assertRaises(ValueError):
            reference_texture(e, y, anchors.astype(int), e, reference, shared)
        with self.assertRaises(ValueError):
            reference_texture(e, y, anchors, e, reference, shared+1)
        with patch('core.drr_reference_texture.check_load_cancelled', side_effect=RuntimeError('cancelled')):
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                reference_texture(e, y, anchors, e, reference, shared)
        texture, _, report = reference_texture(e, y, anchors, e, reference[:12], shared)
        np.testing.assert_array_equal(texture, 0.)
        self.assertIn('Insufficient', report['reason'])


if __name__ == '__main__':
    unittest.main()
