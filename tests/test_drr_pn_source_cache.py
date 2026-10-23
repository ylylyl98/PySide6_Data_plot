"""Source reuse must not reuse sample decisions or weaken input identity."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_pn_reconstruction import pipeline
from core.drr_pn_reconstruction.source_cache import SourcePreparationCache


class SourcePreparationCacheTests(unittest.TestCase):
    def setUp(self):
        self.energy = np.linspace(1.4, 2.0, 31)
        self.frames = np.ones((2, 4, 31)) * 10
        self.mean = self.frames.mean(1)
        self.cache = SourcePreparationCache()
        self.calls = 0

    def compute(self):
        self.calls += 1
        return {"array": self.mean.copy(), "diagnostics": {"values": [1, 2]}}

    def get(self, **kwargs):
        options = dict(identity={"code": "a", "raw": "original-bytes"},
                       compute=self.compute, check=lambda: None)
        options.update(kwargs)
        return self.cache.prepare(self.energy, self.mean, self.frames, **options)

    def test_content_and_implementation_identity_invalidate(self):
        self.assertFalse(self.get()[1])
        self.assertTrue(self.get()[1])
        self.assertFalse(self.get(identity={"code": "b", "raw": "original-bytes"})[1])
        self.assertFalse(self.get(identity={"code": "a", "raw": "changed-header"})[1])
        self.frames[0, 0] += .1
        self.frames[0, 1] -= .1  # Same mean, distinct raw repeats.
        self.assertFalse(self.get()[1])
        self.energy += .001
        self.assertFalse(self.get()[1])
        self.assertEqual(self.calls, 5)

    def test_own_buffers_and_diagnostics_on_both_miss_and_hit(self):
        first, _, _ = self.get()
        first["array"][:] = -1
        first["diagnostics"]["values"].append(3)
        second, reused, _ = self.get()
        self.assertTrue(reused)
        np.testing.assert_array_equal(second["array"], self.mean)
        self.assertEqual(second["diagnostics"]["values"], [1, 2])
        second["array"].setflags(write=True)
        second["array"][:] = -2
        second["diagnostics"]["values"].clear()
        third, reused, _ = self.get()
        self.assertTrue(reused)
        np.testing.assert_array_equal(third["array"], self.mean)
        self.assertEqual(third["diagnostics"]["values"], [1, 2])
        self.assertEqual(self.calls, 1)

    def test_force_refit_and_clear(self):
        self.get()
        self.assertFalse(self.get(force=True)[1])
        self.assertTrue(self.get()[1])
        self.cache.clear()
        self.assertFalse(self.get()[1])
        self.assertEqual(self.calls, 3)

    def test_bounded_lru_and_oversize_results(self):
        self.cache = SourcePreparationCache(max_entries=2)
        self.get(identity="a")
        self.get(identity="b")
        self.assertTrue(self.get(identity="a")[1])
        self.get(identity="c")
        self.assertFalse(self.get(identity="b")[1])
        self.assertEqual(self.calls, 4)
        self.cache = SourcePreparationCache(max_bytes=1)
        self.assertFalse(self.get()[1])
        self.assertFalse(self.get()[1])

    def test_failure_and_cancellation_do_not_publish(self):
        def fail():
            raise ValueError("fit failed")
        with self.assertRaisesRegex(ValueError, "fit failed"):
            self.get(compute=fail)
        cancelled = False
        def check():
            if cancelled:
                raise InterruptedError("cancelled")
        def cancel_in_fit():
            nonlocal cancelled
            result = self.compute()
            cancelled = True
            return result
        with self.assertRaises(InterruptedError):
            self.get(compute=cancel_in_fit, check=check)
        self.assertFalse(self.get()[1])
        with self.assertRaises(InterruptedError):
            self.get(check=check)  # Even an available hit must honor cancel.

    def test_changed_arrays_during_fit_are_not_cached(self):
        def mutate():
            result = self.compute()
            self.frames[0, 0, 0] += 1
            return result
        with self.assertRaisesRegex(ValueError, "arrays changed"):
            self.get(compute=mutate)
        self.assertFalse(self.get()[1])

    def test_sample_decisions_are_recomputed_and_provenance_reports_reuse(self):
        samples = np.ones((2, 3, 31)) * 20
        def prepare(e, pn, frames, **_kwargs):
            self.calls += 1
            return dict(energy=e.copy(), source_feature_log=np.zeros_like(pn),
                        corrected_source_log=np.log(pn), aligned_source_log=np.log(pn),
                        stages={}, diagnostics={"raw_mean_refitted": True})
        def regions(e, pn, frames, current_samples, **_kwargs):
            # The final curve deliberately depends on the current sample.
            return {"full": {"background": pn.mean(0) + current_samples.mean((0, 1)),
                             "deferred_exclusive_mask": np.zeros(len(e), dtype=bool)},
                    "halves": {}}
        def render(e, *, baseline, **_kwargs):
            return {"arrays": baseline, "diagnostics": {}}
        with ExitStack() as stack:
            stack.enter_context(patch("core.drr_pn_reconstruction.source_pipeline.prepare_sources",
                                      side_effect=prepare))
            region_call = stack.enter_context(patch(
                "core.drr_pn_reconstruction.regions_pipeline.prepare_regions", side_effect=regions))
            stack.enter_context(patch("core.drr_pn_reconstruction.render_generalized.render_generalized",
                                      side_effect=render))
            def run(current, **kwargs):
                return pipeline.reconstruct_background(self.energy, self.mean, self.frames, current,
                    method="generalized", source_cache=self.cache,
                    source_cache_identity="same-raw-and-code", **kwargs)
            first, first_report = run(samples)
            second, second_report = run(samples * 2)
            self.assertEqual(self.calls, 1)
            self.assertEqual(region_call.call_count, 2)
            self.assertFalse(np.array_equal(first["background"], second["background"]))
            self.assertTrue(first_report["source_models_refitted_from_raw"])
            self.assertFalse(second_report["source_models_refitted_from_raw"])
            self.assertTrue(second_report["execution"]["source_preparation"]["reused"])
            self.assertFalse(second_report["source_preparation"]["raw_mean_refitted"])
            self.assertFalse(second_report["scientifically_validated"])
            third, third_report = run(samples * 2, force_source_refit=True)
            self.assertEqual(self.calls, 2)
            np.testing.assert_array_equal(second["background"], third["background"])
            self.assertFalse(third_report["execution"]["source_preparation"]["reused"])


if __name__ == "__main__":
    unittest.main()
