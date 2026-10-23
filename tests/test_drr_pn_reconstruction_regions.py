"""Portable regional recipe tests; no acquisition or research assets required."""
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_pn_reconstruction.regions_classifier import classify
from core.drr_pn_reconstruction.regions_helpers import local_quadratic
from core.drr_pn_reconstruction.regions_pipeline import prepare_regions
from core.drr_pn_reconstruction.regions_selection import (
    source_selection, resolve_regions, remove_unknown_excursions,
)
from core.drr_pn_reconstruction.render_pipeline import render_v66


def region_fixture():
    e = np.linspace(1.4, 2.1, 401)
    a = np.array([10 + .4*(e-1.7) + .002*np.sin(220*e),
                  10.001 + .3*(e-1.7) + .0015*np.cos(210*e)])
    q = .003 + .002*(e-1.7)
    c = a - np.array([-.5, .5])[:, None]*q
    a = c + np.array([-.5, .5])[:, None]*q
    w = np.ones(len(e))
    w[50:101] = 0
    w[180:231] = 0
    w[385:] = 0
    for first, last, p0 in ((30, 50, 1), (100, 120, 0), (160, 180, 1),
                             (230, 250, 0), (360, 385, 1)):
        t = np.linspace(0, 1, last-first+1)
        step = t**3*(10-15*t+6*t*t)
        w[first:last+1] = p0+(1-2*p0)*step
        w[first], w[last] = p0, 1-p0
    t = np.linspace(0, np.pi, 21)
    w[280:301] = 1-.8*np.sin(t)**2
    w[280] = w[300] = 1.
    weights = np.array([w, 1-w])
    state = np.full(len(e), 3, np.int8)
    state[281:300] = 0
    b = np.exp(np.sum(weights*a, axis=0))
    return e, dict(energy=e.copy(), background=b,
        baseline_reference_background=b.copy(), reference_source_weights=weights.copy(),
        original_corrected_source_log=c, original_aligned_source_log=a,
        alignment_q_log=q, source_weights=weights, state=state,
        raw_geometric_mean=np.exp(c.mean(0)), source_usable_masks=np.ones_like(a, bool))


class RegionalRecipeTests(unittest.TestCase):
    def test_measured_quadratic_reproduces_quadratic_on_nonuniform_axis(self):
        e = 1.4 + np.linspace(0, 1, 91)**1.2*.4
        values = np.array([2+3*e+4*e*e, 7-e+2*e*e])
        np.testing.assert_allclose(local_quadratic(e, values, .035), values, rtol=0, atol=3e-13)

    def test_model_risk_control_has_bounded_nonoverlapping_transitions(self):
        e = np.linspace(1.4, 1.8, 101)
        risk = np.ones((2, len(e)))
        risk[0, 30:70] = 2
        out = source_selection(e, risk)
        w = out["source_weights"]
        self.assertTrue(np.all((w >= 0) & (w <= 1)))
        np.testing.assert_array_equal(w.sum(0), 1.)
        self.assertEqual(w[0, 0], 1.)
        self.assertEqual(w[0, 50], 0.)
        self.assertEqual(w[0, -1], 1.)

    def test_only_all_unknown_same_source_excursion_is_removed(self):
        e, base = region_fixture()
        before = base["source_weights"].copy()
        result, diag = remove_unknown_excursions(e, before, base["state"])
        self.assertEqual(sum(row["applied"] for row in diag["decisions"]), 1)
        np.testing.assert_array_equal(result["source_weights"][0, 280:301], 1.)
        np.testing.assert_array_equal(before, base["source_weights"])
        state = base["state"].copy()
        state[290] = 2
        kept, _ = remove_unknown_excursions(e, before, state)
        np.testing.assert_array_equal(kept["source_weights"], before)

    def test_short_exclusive_island_deferred_but_sg_resolved_core_locked(self):
        e = np.linspace(1.4, 1.8, 101)
        a = np.array([10+.2*e, 10+.3*e])
        oldw = np.array([np.zeros(len(e)), np.ones(len(e))])
        states = np.zeros(len(e), np.int8)
        states[8:12] = 1
        states[40:66] = 1
        control = dict(energy=e, source_weights=oldw, background=np.exp(a[1]))
        baseline = dict(energy=e, original_corrected_source_log=a,
            original_aligned_source_log=a, alignment_q_log=np.zeros(len(e)),
            state=states, source_usable_masks=np.array([states == 1, states == 2]),
            raw_geometric_mean=np.exp(a.mean(0)))
        classifier = dict(energy=e, v45_state=states, v45_control_source_weights=oldw)
        out, diag = resolve_regions(e, baseline=baseline, control=control, classifier=classifier)
        np.testing.assert_array_equal(out["source_weights"][0, 8:12], 0.)
        np.testing.assert_array_equal(out["source_weights"][0, 40:66], 1.)
        self.assertEqual(len(diag["deferred_runs"]), 1)
        np.testing.assert_array_equal(out["background"][~out["local_support_mask"]],
                                      control["background"][~out["local_support_mask"]])

    def test_classifier_dynamic_gates_and_flat_unknown(self):
        e = np.linspace(1.4, 1.8, 101)
        pn = np.ones((2, len(e)))
        c = np.zeros_like(pn)
        source = dict(energy=e, v32_effective_corrected_source_log=c,
                      v32_alignment_q_log=np.zeros(len(e)))
        control = dict(energy=e, source_weights=np.full_like(pn, .5), background=np.ones(len(e)))
        out, diag = classify(e, pn, np.ones((2, 7, len(e))), baseline=source, control=control,
                             half_aligned_sources=np.zeros((3, 2, len(e))))
        self.assertEqual(diag["training_gates"], 7)
        np.testing.assert_array_equal(out["v45_state"], 0)
        self.assertFalse(out["source_usable_masks"].any())

    def test_prepare_regions_half_labels_and_current_source_binding(self):
        e = np.linspace(1.4, 1.8, 101)
        pn = np.ones((2,len(e)))
        frames = np.ones((2,8,len(e)))
        c = np.zeros_like(pn)
        q = np.zeros(len(e))
        source = dict(energy=e,pn_mean=pn,pn_frames=frames,
            corrected_source_log=c,aligned_source_log=c,alignment_q_log=q,
            half_corrected_source_log=np.zeros((2,2,len(e))),
            half_aligned_source_log=np.zeros((2,2,len(e))),half_pn_mean=np.ones((2,2,len(e))),
            stages={"v29":{"arrays":{"v29_curvature_risk":np.ones_like(pn)}},
                    "v32":{"arrays":{"energy":e,"v32_effective_corrected_source_log":c,
                         "v32_alignment_q_log":q,"v32_effective_residual_log":q}}})
        result = prepare_regions(e,pn,frames,np.ones((2,7,len(e))),source_result=source)
        self.assertEqual(set(result["halves"]), {"first_half", "last_half"})
        for half in result["halves"].values():
            np.testing.assert_array_equal(half["background"], 1.)
            np.testing.assert_array_equal(half["source_weights"], result["full"]["source_weights"])
        with self.assertRaises(AssertionError):
            prepare_regions(e,pn*2,frames,np.ones((2,7,len(e))),source_result=source)

    def test_complete_renderer_no_io_readonly_and_independent_half_anchor(self):
        e, base = region_fixture()
        snapshots = {key: value.tobytes() for key, value in base.items()}
        for value in base.values():
            value.setflags(write=False)
        with patch("builtins.open", side_effect=AssertionError("unexpected file input")), \
             patch("io.open", side_effect=AssertionError("unexpected file input")):
            result = render_v66(e, baseline=base)
        out = result["arrays"]
        self.assertTrue(np.all(np.isfinite(out["background"]) & (out["background"] > 0)))
        self.assertEqual(list(result["stages"]), ["v54", "v57_selection", "v63", "v64", "v65", "v66"])
        self.assertEqual(snapshots, {key: value.tobytes() for key, value in base.items()})
        shifted = {key: value.copy() for key, value in base.items()}
        shifted["original_corrected_source_log"] += .02
        shifted["original_aligned_source_log"] = shifted["original_corrected_source_log"] + np.array([-.5, .5])[:,None]*shifted["alignment_q_log"]
        shifted["background"] = np.exp(np.sum(shifted["source_weights"]*shifted["original_aligned_source_log"], axis=0))
        shifted["baseline_reference_background"] = shifted["background"].copy()
        shifted["raw_geometric_mean"] *= np.exp(.02)
        other = render_v66(e, baseline=shifted)["arrays"]
        np.testing.assert_allclose(other["background"]/out["background"], np.exp(.02), rtol=0, atol=1e-11)

    def test_unsupported_topology_and_cancellation_raise_without_fallback(self):
        e, base = region_fixture()
        with self.assertRaises(InterruptedError):
            render_v66(e, baseline=base, cancelled=lambda: True)
        base["source_weights"] = np.array([np.ones(len(e)), np.zeros(len(e))])
        base["reference_source_weights"] = base["source_weights"].copy()
        base["background"] = np.exp(base["original_aligned_source_log"][0])
        base["baseline_reference_background"] = base["background"].copy()
        with self.assertRaisesRegex(ValueError, "three frozen pure-P"):
            render_v66(e, baseline=base)


if __name__ == "__main__":
    unittest.main()
