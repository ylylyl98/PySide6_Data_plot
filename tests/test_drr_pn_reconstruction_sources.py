"""Input-contract and physical bookkeeping checks for raw source reconstruction."""
import builtins
import inspect
import io
import os
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_pn_reconstruction.source_pipeline import prepare_sources, STAGE_FAMILIES
from core.drr_pn_reconstruction.source_runtime import cancellation_scope, least_squares


class RawSourceReconstructionTests(unittest.TestCase):
    @staticmethod
    def inputs(n=101, repeats=6):
        e = np.linspace(1.3, 2.2, n)
        common = 2.4 + .07*(e-1.7) + .03*(e-1.7)**2
        # Different smooth source responses, with frame drift symmetric about
        # their mean; half means are intentionally different from full means.
        base = np.exp(np.array([common+.02*(e-1.7), common-.02*(e-1.7)]))
        drift = np.linspace(-.001, .001, repeats)
        frames = base[:, None, :]*(1+drift[None, :, None])
        return e, frames.mean(1), frames

    def test_pure_readonly_no_io_and_half_mean_before_log(self):
        inputs = self.inputs()
        originals = [a.copy() for a in inputs]
        for a in inputs:
            a.setflags(write=False)
        def forbidden(*_a, **_kw):
            raise AssertionError("Source preparation may not read stored models or other files")
        with patch.object(builtins, "open", forbidden), patch.object(io, "open", forbidden), patch.object(os, "open", forbidden):
            result = prepare_sources(*inputs)
        for a, old in zip(inputs, originals):
            np.testing.assert_array_equal(a, old)
        self.assertEqual(tuple(result["stages"]), tuple(STAGE_FAMILIES))
        self.assertFalse(result["diagnostics"]["stored_model_used_as_input"])
        self.assertFalse(result["diagnostics"]["scientifically_validated"])
        self.assertEqual(result["diagnostics"]["half_frame_counts"], [3, 3])
        f = np.log(inputs[1])-result["corrected_source_log"]
        expected = np.log(inputs[2][:, :3].mean(1))-f
        np.testing.assert_array_equal(result["half_corrected_source_log"][0], expected)
        self.assertGreater(np.max(abs(result["half_corrected_source_log"][0]-result["corrected_source_log"])), 1e-4)
        np.testing.assert_allclose(result["corrected_source_log"][0]-result["corrected_source_log"][1],
                                   result["residual_log"]+result["alignment_q_log"], atol=1e-14, rtol=0)

    def test_shared_static_feature_is_not_automatically_removed(self):
        e, _, frames = self.inputs()
        common = np.exp(2+.06*np.exp(-.5*((e-1.75)/.012)**2))
        frames = np.broadcast_to(common, (2, 4, len(e))).copy()
        pn = frames.mean(1)
        result = prepare_sources(e, pn, frames)
        np.testing.assert_allclose(result["corrected_source_log"], np.log(pn), rtol=0, atol=1e-13)
        self.assertEqual(result["stages"]["v32"]["diagnostics"]["v32"]["accepted_count"], 0)

    def test_no_angle_template_or_external_model_argument(self):
        names = set(inspect.signature(prepare_sources).parameters)
        self.assertEqual(names, {"energy", "pn_mean", "pn_frames", "cancelled", "progress", "alignment_policy"})
        e, pn, frames = self.inputs(n=83, repeats=5)
        result = prepare_sources(e, pn, frames)
        np.testing.assert_array_equal(result["energy"], e)
        self.assertEqual(result["diagnostics"]["half_frame_counts"], [2, 3])
        self.assertTrue(np.all(np.isfinite(result["aligned_source_log"])))

    def test_invalid_arrays_rejected_before_fitting(self):
        e, pn, frames = self.inputs()
        bad = [
            (e[::-1], pn, frames),
            (e, pn*1.001, frames),
            (e, pn, frames[:, :3]),
            (e, -pn, -frames),
            (e, pn[:, :-1], frames),
        ]
        for values in bad:
            with self.subTest(shape=[a.shape for a in values]):
                with self.assertRaises(ValueError):
                    prepare_sources(*values)
        frames[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            prepare_sources(e, pn, frames)

    def test_cancellation_before_stage_and_inside_optimizer(self):
        with self.assertRaises(InterruptedError):
            prepare_sources(*self.inputs(), cancelled=lambda: True)
        count = 0
        def cancel_after_three():
            nonlocal count
            count += 1
            return count >= 3
        with self.assertRaises(InterruptedError), cancellation_scope(cancel_after_three):
            least_squares(lambda x: x-1, [0.])
        # Scope cleanup must not leak cancellation to a later call.
        fit = least_squares(lambda x: x-1, [0.])
        self.assertTrue(fit.success)


if __name__ == "__main__":
    unittest.main()
