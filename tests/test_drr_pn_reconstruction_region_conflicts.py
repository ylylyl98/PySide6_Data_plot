"""Small native-grid regressions for conservative regional authorization."""
import unittest

import numpy as np

from core.drr_pn_reconstruction.regions_selection import resolve_regions


def conflict_fixture(exclusive_runs, *, swap_sources=False):
    """Keep real alignment/control invariants without fitting or file inputs."""
    energy = np.linspace(1.4, 1.7, 301)
    states = np.zeros(len(energy), dtype=np.int8)
    for first, stop, state in exclusive_runs:
        states[first:stop] = state
    corrected = np.array([
        4.5 + .3 * energy + .01 * np.sin(25 * energy),
        4.7 - .1 * energy + .02 * np.cos(17 * energy),
    ])
    q = .03 + .02 * (energy - energy[0])
    weights = np.tile(np.array([[.375], [.625]]), (1, len(energy)))
    if swap_sources:
        states = np.where(states == 1, 2, np.where(states == 2, 1, states)).astype(np.int8)
        corrected = corrected[::-1].copy()
        q = -q
        weights = weights[::-1].copy()
    aligned = corrected + np.array([-.5, .5])[:, None] * q
    baseline = dict(
        energy=energy.copy(), state=states,
        original_corrected_source_log=corrected,
        original_aligned_source_log=aligned, alignment_q_log=q,
        raw_geometric_mean=np.exp(corrected.mean(axis=0)),
        source_usable_masks=np.array([states == 1, states == 2]),
    )
    control = dict(energy=energy.copy(), source_weights=weights,
                   background=np.exp(np.sum(weights * aligned, axis=0)))
    classifier = dict(energy=energy.copy(), v45_state=states.copy(),
                      v45_control_source_weights=weights.copy())
    return energy, baseline, control, classifier


class RegionalConflictTests(unittest.TestCase):
    def resolve_readonly(self, runs, *, swap_sources=False):
        energy, baseline, control, classifier = conflict_fixture(
            runs, swap_sources=swap_sources)
        inputs = [energy] + [value for mapping in (baseline, control, classifier)
                            for value in mapping.values()]
        snapshots = [value.tobytes() for value in inputs]
        for value in inputs:
            value.setflags(write=False)
        arrays, report = resolve_regions(
            energy, baseline=baseline, control=control, classifier=classifier)
        self.assertEqual(snapshots, [value.tobytes() for value in inputs])
        self.assertTrue(all(not value.flags.writeable for value in inputs))
        self.assertEqual(report["collar_width_eV"], .008)
        support = arrays["local_support_mask"]
        deferred = arrays["deferred_exclusive_mask"]
        self.assertFalse(np.any(support & deferred))
        # A tolerant float comparison would miss forbidden one-bit changes.
        for mask in (~support, deferred):
            for name in ("background", "source_weights"):
                self.assertEqual(arrays[name][..., mask].tobytes(),
                                 control[name][..., mask].tobytes())
        for row in report["deferred_runs"]:
            self.assertIs(row["new_switch_authorized"], False)
            self.assertIsInstance(row["reasons"], list)
            self.assertTrue(row["reasons"])
        for row in report["runs"]:
            self.assertIs(row["new_switch_authorized"], True)
            first, stop = row["first_index"], row["stop_exclusive"]
            np.testing.assert_array_equal(
                arrays["source_weights"][0, first:stop], row["target_P_weight"])
            self.assertEqual(row["support_bounds_eV"], [
                row["core_bounds_eV"][0] - .008,
                row["core_bounds_eV"][1] + .008,
            ])
        return arrays, report

    def assert_deferred(self, report, first, reason):
        row = next(row for row in report["deferred_runs"] if row["first_index"] == first)
        self.assertIn(reason, row["reasons"])

    def assert_source_swap_symmetric(self, normal, swapped):
        arrays, report = normal
        other, other_report = swapped
        for name in ("local_support_mask", "local_core_mask", "local_collar_mask",
                     "deferred_exclusive_mask", "local_run_core_indices"):
            np.testing.assert_array_equal(arrays[name], other[name])
        # Selection must be symmetric; arithmetic within authorized collars may
        # differ by floating-point rounding after complementing the weights.
        np.testing.assert_allclose(arrays["source_weights"][::-1],
                                   other["source_weights"], rtol=0, atol=3e-16)
        np.testing.assert_allclose(arrays["background"], other["background"],
                                   rtol=3e-15, atol=0)
        left = {row["first_index"]: row for row in report["deferred_runs"]}
        right = {row["first_index"]: row for row in other_report["deferred_runs"]}
        self.assertEqual(left.keys(), right.keys())
        for first, row in left.items():
            self.assertEqual(set(row["reasons"]), set(right[first]["reasons"]))
            self.assertEqual(row["state"], 3 - right[first]["state"])

    def test_nearby_short_exclusive_island_defers_conflict_not_safe_core(self):
        for island_state in (1, 2):
            with self.subTest(island_state=island_state):
                runs = [(40, 71, 1), (76, 79, island_state), (200, 231, 2)]
                normal = self.resolve_readonly(runs)
                arrays, report = normal
                self.assertEqual([row["first_index"] for row in report["runs"]], [200])
                self.assert_deferred(report, 76, "no_complete_nominal_SG21_window")
                self.assert_deferred(report, 40,
                                     "support_contacts_deferred_exclusive_evidence")
                if island_state == 2:
                    self.assert_deferred(report, 40,
                                         "fixed_collar_crosses_opposite_exclusive_evidence")
                self.assertTrue(arrays["local_core_mask"][200:231].all())
                self.assert_source_swap_symmetric(
                    normal, self.resolve_readonly(runs, swap_sources=True))

    def test_overlapping_requested_supports_defer_both_candidates(self):
        # The cores are 12 meV apart: neither 8 meV collar reaches the
        # other core, but their requested supports overlap by 4 meV.
        for second_state in (1, 2):
            with self.subTest(second_state=second_state):
                runs = [(40, 71, 1), (82, 113, second_state), (200, 231, 1)]
                normal = self.resolve_readonly(runs)
                _, report = normal
                self.assertEqual([row["first_index"] for row in report["runs"]], [200])
                for first in (40, 82):
                    self.assert_deferred(report, first, "overlapping_requested_supports")
                self.assert_source_swap_symmetric(
                    normal, self.resolve_readonly(runs, swap_sources=True))

    def test_sg21_gate_and_fixed_collar_are_not_relaxed_to_avoid_conflicts(self):
        # Twenty native points cannot contain a complete SG21 window; an
        # isolated run of exactly 21 points still has the existing authority.
        arrays, report = self.resolve_readonly([(40, 60, 1), (100, 121, 1)])
        self.assert_deferred(report, 40, "no_complete_nominal_SG21_window")
        self.assertEqual([row["first_index"] for row in report["runs"]], [100])
        self.assertEqual(
            report["runs"][0]["nominal_SG21_complete_window_center_indices"], [110])
        np.testing.assert_array_equal(arrays["source_weights"][0, 100:121], 1.)


if __name__ == "__main__":
    unittest.main()
