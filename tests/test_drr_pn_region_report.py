"""Saved provenance presentation must not certify a background's suitability."""
from copy import deepcopy
import unittest

import numpy as np

from core.drr_pn_region_report import describe_pn_regions


class PnRegionReportTests(unittest.TestCase):
    def setUp(self):
        self.energy = np.linspace(1.4, 2.0, 61)

    def test_missing_metadata_is_explicit_across_complete_axis(self):
        result = describe_pn_regions(self.energy, {})
        self.assertEqual(len(result.rows), 1)
        row = result.rows[0]
        self.assertEqual((row.xmin_ev, row.xmax_ev), (1.4, 2.0))
        self.assertEqual(row.support, "Not recorded")
        self.assertIn("not recorded", result.scientific_review)

    def test_legacy_pn_and_assisted_rows_preserve_evidence_and_gaps(self):
        report = dict(scientifically_validated=False,
            source_segments=[dict(source_key="P", xmin_ev=1.4, xmax_ev=1.55)],
            transitions=[dict(source_from="P", source_to="N", energy_bounds=[1.55, 1.65])],
            unsupported_regions=[[1.5, 1.6]], unresolved_regions=[[1.7, 1.8]],
            sample_protected_regions=[[1.6, 1.7]])
        original = deepcopy(report)
        result = describe_pn_regions(self.energy, report)
        self.assertEqual(report, original)
        self.assertTrue(any(row.source == "P" and row.operation == "Saved source segment" for row in result.rows))
        self.assertTrue(any(row.source == "P → N" for row in result.rows))
        self.assertTrue(any(row.support == "Unsupported" for row in result.rows))
        self.assertTrue(any(row.support == "Unresolved" for row in result.rows))
        self.assertTrue(any(row.support == "Protected / conditional" for row in result.rows))
        gaps = [r for r in result.rows if r.record_path == "metadata_gap"]
        self.assertEqual([(r.xmin_ev, r.xmax_ev) for r in gaps], [(1.65, 2.0)])
        self.assertIn("Not scientifically validated", result.scientific_review)

    def test_generalized_repairs_do_not_hide_deferred_or_opposite_evidence(self):
        report = {"render_diagnostics": {"full": {"repair_plan": [
            dict(source="P", core_indices=[10, 20], affected_bridge_support=[8, 22],
                 apply=True, reasons=[], opposite_exclusive_indices=[14]),
            dict(source="N", core_indices=[30, 40], apply=False,
                 reasons=["overlapping_requested_supports"]),
        ]}}}
        result = describe_pn_regions(self.energy, report)
        self.assertTrue(any(r.source == "P" and r.support == "Applied / conditional" for r in result.rows))
        deferred = next(r for r in result.rows if r.source == "Requested N")
        self.assertEqual(deferred.support, "Deferred / unresolved")
        self.assertIn("overlapping requested supports", deferred.reason)
        conflict = next(r for r in result.rows if r.operation == "Source conflict")
        self.assertEqual((conflict.xmin_ev, conflict.xmax_ev), (self.energy[14], self.energy[14]))
        self.assertEqual(conflict.support, "Unresolved")

    def test_conditioned_routes_and_unmet_intervals_are_visible(self):
        report = {"render_diagnostics": {"full": dict(
            regional_plan=[dict(source_index=0, core_indices=[10, 60], apply=False,
                                route="compact_only", reasons=["conditioned_compact_only"])],
            local_plan=[dict(source="P", core_indices=[20, 25], support_indices=[18, 27], apply=True)],
            unmet_exclusive_intervals=[dict(required_source="N", indices=[40, 41],
                inherited_selection="opposite", reason="Exclusive evidence remains unmet")])}}
        result = describe_pn_regions(self.energy, report)
        self.assertTrue(any(r.support == "Routed to compact repair" for r in result.rows))
        self.assertTrue(any(r.operation == "Compact repair" and r.support == "Applied / conditional" for r in result.rows))
        self.assertTrue(any(r.operation == "Compact repair support" for r in result.rows))
        self.assertTrue(any(r.source == "Required N" and r.support == "Unresolved" for r in result.rows))

    def test_selector_deferrals_expose_reasons_without_filling_construction_gaps(self):
        report = {"region_diagnostics": {"resolved_regions": {"deferred_runs": [
            dict(source="P", core_bounds_eV=[1.5, 1.6], nominal_SG21_supported=True,
                 new_switch_authorized=False,
                 reasons=["fixed_collar_crosses_opposite_exclusive_evidence", "overlapping_requested_supports"]),
            dict(source="N", core_bounds_eV=[1.7, 1.71], nominal_SG21_supported=False,
                 new_switch_authorized=False, reasons=["no_complete_nominal_SG21_window"]),
        ]}}}
        original = deepcopy(report)
        result = describe_pn_regions(self.energy, report)
        self.assertEqual(report, original)
        deferred = [row for row in result.rows if row.operation == "Source selection deferred"]
        self.assertEqual([(row.xmin_ev, row.xmax_ev, row.source) for row in deferred],
                         [(1.5, 1.6, "Requested P"), (1.7, 1.71, "Requested N")])
        self.assertTrue(all(row.support == "Control retained at selector only" for row in deferred))
        self.assertIn("fixed collar crosses opposite exclusive evidence", deferred[0].reason)
        self.assertIn("overlapping requested supports", deferred[0].reason)
        self.assertIn("SG21 window supported", deferred[0].reason)
        self.assertIn("no complete nominal SG21 window", deferred[1].reason)
        self.assertEqual(deferred[0].record_path, "region_diagnostics.resolved_regions.deferred_runs[0]")
        gaps = [row for row in result.rows if row.record_path == "metadata_gap"]
        self.assertEqual([(row.xmin_ev, row.xmax_ev) for row in gaps], [(1.4, 2.0)])
        self.assertTrue(any("do not describe final construction coverage" in note for note in result.notes))

    def test_selector_deferrals_preserve_parent_unmet_evidence_and_unknown_retention(self):
        report = dict(
            region_diagnostics={"resolved_regions": {"deferred_runs": [dict(
                source="P", core_bounds_eV=[1.8, 1.81], reasons=["overlapping_requested_supports"])]}},
            conditioned_parent_diagnostics={"full": {"unmet_exclusive_intervals": [dict(
                required_source="P", energy_bounds=[1.8, 1.81], reason="Inherited selector is mixed")]}})
        result = describe_pn_regions(self.energy, report)
        deferred = next(row for row in result.rows if row.operation == "Source selection deferred")
        self.assertEqual(deferred.support, "Selector retention not recorded")
        unmet = next(row for row in result.rows if row.operation == "Parent unmet source evidence")
        self.assertEqual(unmet.support, "Unresolved")
        self.assertEqual(unmet.source, "Required P")
        self.assertTrue(any(row.operation == "Parent construction not recorded"
                            and (row.xmin_ev, row.xmax_ev) == (1.4, 2.0) for row in result.rows))

    def test_malformed_selector_deferrals_do_not_infer_control_retention(self):
        for selector in (None, {"resolved_regions": None}, {"resolved_regions": {"deferred_runs": "invalid"}},
                         {"resolved_regions": {"deferred_runs": [None, dict(
                             source="P", core_bounds_eV=[1.8, 9.0], new_switch_authorized=False)]}}):
            with self.subTest(selector=selector):
                result = describe_pn_regions(self.energy, {"region_diagnostics": selector})
                self.assertFalse(any(row.operation == "Source selection deferred" for row in result.rows))
                self.assertTrue(any("could not be read" in note for note in result.notes))
                self.assertEqual([(row.xmin_ev, row.xmax_ev) for row in result.rows], [(1.4, 2.0)])

    def test_optional_correction_is_not_base_background_support(self):
        report = dict(region_applicability=dict(schema_version=1, coverage_fraction=0.4,
            regions=[dict(energy_bounds=[1.5, 1.7], source="P/N common fine",
                status="supported_candidate", correction_applied=True, reason="Frozen conditional evidence"),
                dict(energy_bounds=[1.4, 1.5], source="P/N common fine", status="boundary_guard",
                     correction_applied=False, reason="Native boundary guard")]))
        result = describe_pn_regions(self.energy, report)
        self.assertTrue(any(r.support == "Conditional correction candidate" for r in result.rows))
        self.assertTrue(any(r.support == "Boundary guard / no correction" and "does not establish" in r.reason for r in result.rows))
        gap = next(r for r in result.rows if r.record_path == "metadata_gap")
        self.assertEqual((gap.xmin_ev, gap.xmax_ev), (1.4, 2.0))
        self.assertTrue(any("40.0%" in note and "not a scientific pass rate" in note for note in result.notes))

    def test_bounded_parent_and_fine_records_remain_separate_without_mutation(self):
        report = dict(scientifically_validated=False,
            render_diagnostics={"full": dict(
                parent_diagnostics_path="conditioned_parent_diagnostics.full",
                correction_selected=True, final_polynomial_available=False)},
            conditioned_parent_diagnostics={"full": dict(
                regional_plan=[dict(source="P", core_indices=[10, 20],
                    affected_bridge_support=[8, 22], apply=True)],
                local_plan=[dict(source="N", core_indices=[30, 35],
                    support_indices=[28, 38], apply=True)],
                final_field_contract={"final_curvature_polynomial_coefficients": "Parent-only polynomial"})},
            region_applicability=dict(schema_version=1, regions=[dict(
                energy_bounds=[1.5, 1.75], source="P/N common fine", status="supported_candidate",
                correction_applied=True, reason="Conditional support")]))
        original = deepcopy(report)
        result = describe_pn_regions(self.energy, report)
        self.assertEqual(report, original)
        by_path = {row.record_path: row for row in result.rows}
        parent = by_path["conditioned_parent_diagnostics.full.regional_plan[0]"]
        self.assertEqual(parent.operation, "Parent regional repair")
        self.assertEqual(parent.support, "Applied to parent / conditional")
        local = by_path["conditioned_parent_diagnostics.full.local_plan[0]"]
        self.assertEqual(local.operation, "Parent compact repair")
        self.assertIn("conditioned_parent_diagnostics.full.regional_plan[0].affected_bridge_support", by_path)
        self.assertIn("conditioned_parent_diagnostics.full.local_plan[0].support_indices", by_path)
        fine = by_path["region_applicability.regions[0]"]
        self.assertEqual(fine.operation, "Optional fine correction")
        self.assertEqual(fine.support, "Conditional correction candidate")
        self.assertFalse(any(path.startswith("render_diagnostics.full") for path in by_path))
        self.assertTrue(any("parent polynomials and derivatives do not describe the final candidate" in n
                            for n in result.notes))
        # Optional correction cannot fill gaps in the parent's source records.
        gaps = [row for row in result.rows if row.record_path == "metadata_gap"]
        self.assertTrue(any(row.xmin_ev == self.energy[20] and row.xmax_ev == self.energy[30]
                            for row in gaps))
        self.assertTrue(all(row.operation == "Parent construction not recorded" for row in gaps))

    def test_bounded_parent_conflicts_and_deferred_routes_are_preserved(self):
        report = dict(conditioned_parent_diagnostics={"full": dict(
            regional_plan=[dict(source="P", core_indices=[10, 60], apply=False,
                route="compact_only", reasons=["conditioned_compact_only"],
                opposite_exclusive_indices=[44])],
            local_plan=[dict(source="P", core_indices=[30, 35], apply=False,
                reasons=["overlapping_support"])],
            unmet_exclusive_intervals=[dict(required_source="N", indices=[40, 41],
                reason="Exclusive evidence remains unmet")])})
        result = describe_pn_regions(self.energy, report)
        by_path = {row.record_path: row for row in result.rows}
        self.assertEqual(by_path["conditioned_parent_diagnostics.full.regional_plan[0]"].support,
                         "Routed to compact repair")
        self.assertEqual(by_path["conditioned_parent_diagnostics.full.local_plan[0]"].support,
                         "Deferred / unresolved")
        conflict = by_path["conditioned_parent_diagnostics.full.regional_plan[0].opposite_exclusive_indices[0]"]
        self.assertEqual(conflict.operation, "Parent source conflict")
        self.assertEqual(conflict.support, "Unresolved")
        unmet = by_path["conditioned_parent_diagnostics.full.unmet_exclusive_intervals[0]"]
        self.assertEqual(unmet.operation, "Parent unmet source evidence")
        self.assertEqual(unmet.source, "Required N")

    def test_missing_bounded_parent_is_not_filled_by_optional_correction_or_final_contract(self):
        for parent in (None, {}, {"full": None}, {"full": []}, {"full": {}}):
            report = dict(conditioned_parent_diagnostics=parent,
                render_diagnostics={"full": dict(
                    parent_diagnostics_path="conditioned_parent_diagnostics.full",
                    final_polynomial_available=False)},
                region_applicability=dict(schema_version=1, regions=[dict(
                    energy_bounds=[1.4, 2.0], source="P/N common fine", status="supported_candidate",
                    correction_applied=True)]))
            with self.subTest(parent=parent):
                original = deepcopy(report)
                result = describe_pn_regions(self.energy, report)
                self.assertEqual(report, original)
                gap = next(row for row in result.rows if row.record_path == "metadata_gap")
                self.assertEqual((gap.xmin_ev, gap.xmax_ev), (1.4, 2.0))
                self.assertEqual(gap.operation, "Parent construction not recorded")
                self.assertEqual(gap.support, "Not recorded")
                self.assertFalse(any(row.support == "Applied to parent / conditional" for row in result.rows))
        report.pop("conditioned_parent_diagnostics")
        result = describe_pn_regions(self.energy, report)
        self.assertTrue(any("Parent construction metadata is missing" in note for note in result.notes))

    def test_rejected_fine_correction_does_not_hide_parent_repairs_or_certify_them(self):
        report = dict(scientifically_validated=False,
            conditioned_parent_diagnostics={"full": dict(regional_plan=[dict(
                source="P", core_indices=[10, 20], apply=True)])},
            region_applicability=dict(schema_version=1, regions=[dict(
                energy_bounds=[1.4, 2.0], source="P/N common fine", status="proposal_rejected",
                correction_applied=False)]))
        result = describe_pn_regions(self.energy, report)
        self.assertTrue(any(row.support == "Applied to parent / conditional" for row in result.rows))
        fine = next(row for row in result.rows if row.operation == "Optional fine correction")
        self.assertEqual(fine.support, "Correction proposal rejected")
        self.assertIn("Parent retained; this does not establish its suitability", fine.reason)
        self.assertIn("Not scientifically validated", result.scientific_review)

    def test_unknown_validation_claim_never_certifies_current_measurement(self):
        for report in ({"scientifically_validated": True}, {"fresh": True, "applied": True},
                       {"scientifically_validated": "true"}):
            with self.subTest(report=report):
                result = describe_pn_regions(self.energy, report)
                self.assertIn("not recorded", result.scientific_review)
                self.assertNotIn("accepted", result.scientific_review)

    def test_bad_records_are_disclosed_without_inferred_support(self):
        report = dict(source_segments=[None, dict(source_key="P", xmin_ev=float("nan"), xmax_ev=1.8)],
            unresolved_regions="invalid", region_applicability={"schema_version": 7, "regions": []},
            render_diagnostics={"full": {"repair_plan": [dict(source_index=0.0, core_indices=[4, 9],
                apply=False, opposite_exclusive_indices=None), dict(source="P", core_indices=[-1, 90]) ]}})
        result = describe_pn_regions(self.energy, report)
        self.assertTrue(any("could not be read" in note for note in result.notes))
        self.assertTrue(any(r.source == "Requested Not recorded" for r in result.rows))
        self.assertFalse(any(r.support == "Applied / conditional" for r in result.rows))

    def test_nullable_and_unknown_fields_do_not_break_preview(self):
        for index in (0.0, True, [], "unknown"):
            report = {"render_diagnostics": {"full": {"repair_plan": [dict(
                source=None, source_index=index, core_indices=[10, 20], reasons=None)]}},
                "region_applicability": dict(schema_version=1, regions=[dict(
                    energy_bounds=[1.5, 1.6], status=[], source=None, reason=None)])}
            with self.subTest(index=index):
                result = describe_pn_regions(self.energy, report)
                self.assertTrue(any(r.source == "Requested Not recorded" and r.support == "Application not recorded"
                                    for r in result.rows))
                self.assertTrue(any(r.support == "Correction support not recorded" for r in result.rows))

    def test_real_pn_generator_metadata_can_be_described(self):
        from core.drr_pn_generator import generate_pn_arrays
        energy = np.linspace(1.4, 2.0, 181)
        p = 100 * (1 + .02*np.sin(energy*130))
        n = 110 * (1 + .02*np.sin(energy*130))
        _, report = generate_pn_arrays(energy, p, n)
        result = describe_pn_regions(energy, report)
        self.assertTrue(any(r.operation == "Saved source segment" for r in result.rows))
        self.assertFalse(any("could not be read" in note for note in result.notes))

    def test_invalid_axis_is_rejected(self):
        for energy in ([], [1.4], [1.5, 1.4], [1.4, float("nan")]):
            with self.subTest(energy=energy), self.assertRaises(ValueError):
                describe_pn_regions(energy, {})


if __name__ == "__main__":
    unittest.main()
