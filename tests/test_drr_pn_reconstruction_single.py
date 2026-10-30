"""Single-measurement construction is conditional evidence, never repeat validation."""
import json
import unittest
from unittest.mock import patch
import numpy as np

from core.drr_pn_reconstruction import pipeline
from core.drr_pn_reconstruction.regions_single_classifier import classify_single
from tests.test_drr_pn_texture_compatibility import synthetic_inputs


class SingleReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.e = np.linspace(1.5, 1.9, 301)
        e = self.e
        texture = .015*np.sin((e-1.5)*220)
        self.pn = np.exp(np.array([texture, texture]))
        self.aligned = np.log(self.pn)
        self.halves = np.stack((self.aligned, self.aligned+.0001*np.sin(150*e),
                                self.aligned-.0001*np.sin(150*e)))
        self.baseline = dict(energy=e, v32_effective_corrected_source_log=self.aligned,
                             v32_alignment_q_log=np.zeros(len(e)))
        self.control = dict(energy=e, source_weights=np.full((2, len(e)), .5),
                            background=np.exp(self.aligned.mean(axis=0)))
        self.samples = self.pn[0][None,None,:] * np.exp(np.zeros((1,9,len(e))))

    def classify(self, samples=None, halves=None):
        return classify_single(self.e, self.pn, self.samples if samples is None else samples,
            baseline=self.baseline, control=self.control,
            half_aligned_sources=self.halves if halves is None else halves)

    def test_common_texture_has_conditional_support_not_repeat_claim(self):
        arrays, report = self.classify()
        self.assertGreater(np.count_nonzero(arrays['v45_state'] == 3), 0)
        self.assertEqual(report['independent_sample_repeats'], 1)
        self.assertIn('not evaluable', report['cross_measurement_drift'])
        self.assertNotIn('v45_sample_repeat_noise', arrays)
        self.assertFalse(report['scientifically_validated'])
        json.dumps(report, allow_nan=False)

    def test_moving_branches_high_noise_and_single_spike_cannot_masquerade_as_stable(self):
        rng = np.random.default_rng(51)
        moving = np.copy(self.samples)
        for gate in range(9):
            center = 1.56 + gate*.035
            moving[0, gate] *= np.exp(.2*np.exp(-((self.e-center)/.007)**2))
        moving *= np.exp(rng.normal(0, .08, moving.shape))
        moving[0, 4, 150] *= 3
        arrays, _ = self.classify(moving)
        # Strong gate-varying / noisy structures must not become broad evidence.
        self.assertLess(np.count_nonzero(arrays['v45_state']), len(self.e)//4)

    def test_isolated_spike_and_fixed_resonance_remain_ambiguous(self):
        samples = self.samples.copy()
        samples[0, 4, 150] *= 10
        base, _ = self.classify()
        spike, _ = self.classify(samples)
        # A single gate spike cannot create new conditional support.
        self.assertFalse(np.any((spike['v45_state'] > 0) & (base['v45_state'] == 0)))
        fixed = self.samples * np.exp(.08*np.exp(-((self.e-1.7)/.009)**2))[None,None,:]
        arrays, report = self.classify(fixed)
        self.assertEqual(arrays['v45_state'].shape, self.e.shape)
        self.assertIn('fixed sample resonances', report['limitation'])
        self.assertFalse(report['scientifically_validated'])

    def test_contaminated_pn_source_is_not_selected_as_clean(self):
        halves = self.halves.copy()
        contamination = .03*np.sin(300*self.e)
        halves[:, 1, :] += contamination
        baseline = dict(self.baseline, v32_effective_corrected_source_log=halves[0])
        control = dict(self.control, background=np.exp(halves[0].mean(axis=0)))
        arrays, _ = classify_single(self.e, self.pn, self.samples, baseline=baseline,
            control=control, half_aligned_sources=halves)
        self.assertGreater(np.count_nonzero(arrays['v45_state'] == 1), 0)
        self.assertEqual(np.count_nonzero(arrays['v45_state'] == 3), 0)

    def test_odd_pn_frames_are_rejected(self):
        energy, frames, samples, _ = synthetic_inputs(n=301)
        with self.assertRaisesRegex(ValueError, 'even number'):
            pipeline.reconstruct_background(energy, frames[:, :3].mean(1), frames[:, :3],
                                            samples[:1], method='conditioned_single')

    def test_single_feature_protection_identifies_single_parent(self):
        energy, frames, samples, _ = synthetic_inputs(n=301)
        lo, hi = energy[100], energy[160]
        arrays, report = pipeline.reconstruct_background(energy, frames.mean(1), frames, samples[:1],
            method='conditioned_single', feature_protection=[dict(source='P', xmin_ev=lo, xmax_ev=hi)])
        self.assertEqual(str(arrays['construction_type']), 'conditioned_single_with_explicit_raw_source_protection')
        self.assertNotIn('bounded_prefix', report['final_array_contract'])
        self.assertIn('not evaluated', report['bounded_texture'])
        self.assertFalse(report['scientifically_validated'])

    def test_real_pipeline_native_grid_positive_and_explicit_limitations(self):
        energy, frames, samples, _ = synthetic_inputs(n=301)
        from core.drr_pn_reconstruction.render_conditioned import render_conditioned
        parents = []
        def record_parent(*args, **kwargs):
            value = render_conditioned(*args, **kwargs)
            parents.append(value['arrays']['background'].copy())
            return value
        with patch('core.drr_pn_reconstruction.render_conditioned.render_conditioned', side_effect=record_parent):
            arrays, report = pipeline.reconstruct_background(energy, frames.mean(1), frames,
                                                             samples[:1], method='conditioned_single')
        np.testing.assert_array_equal(arrays['background'], parents[0])
        np.testing.assert_array_equal(arrays['energy'], energy)
        self.assertTrue(np.all(np.isfinite(arrays['background']) & (arrays['background'] > 0)))
        self.assertEqual(report['algorithm_version'], pipeline.CONDITIONED_SINGLE_ALGORITHM_VERSION)
        self.assertEqual(report['independent_sample_repeats'], 1)
        self.assertIn('not evaluated', report['bounded_texture'])
        self.assertFalse(report['scientifically_validated'])
        # The sample division and energy derivatives retain the native axes and finite values.
        ratio = samples[0] / arrays['background'] - 1
        d2 = np.gradient(np.gradient(ratio, energy, axis=1), energy, axis=1)
        self.assertEqual(ratio.shape, samples[0].shape)
        self.assertTrue(np.all(np.isfinite(d2)))
        json.dumps(report, allow_nan=False)


if __name__ == '__main__': unittest.main()
