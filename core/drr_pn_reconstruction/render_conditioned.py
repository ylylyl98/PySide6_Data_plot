"""Condition regional repair on bounded geometry and current source evidence.

Reliable regions use the unchanged generalized-v1 kernels. Boundary/conflicting
regions can only receive existing compact evidence-local increments. The plan
depends on E, conditional states and weights, never on a background comparison.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .regions_helpers import runs
from .render_generalized import (
    _refresh_final_metadata, _render_generalized_with_policy, select_repair_regions,
)
from .render_local_evidence import _local_increment, plan_local_repairs
from .source_runtime import cancellation_scope, check_cancelled

RECIPE_VERSION = "pn_conditioned_regional_curvature_v1"
METADATA_REVISION = 1


def plan_conditioned_repairs(energy, weights, state, *, regional_plan=None):
    """Freeze regional/local routing and all cross-family exclusions at once.

    Existing v1 geometry failures remain failures. Only boundary/conflicting
    regions are redirected to compact repair. Local candidates are filtered
    before recomputing their mutual-overlap decisions, so unused local routes
    cannot veto another local repair. Applied regional supports have priority.
    """
    e = np.asarray(energy)
    state = np.asarray(state)
    regional = deepcopy(regional_plan if regional_plan is not None else
                        select_repair_regions(e, weights, state))
    protected = np.zeros(len(e), bool)
    protected_cells = np.zeros(len(e) - 1, bool)
    routed = []
    for i, row in enumerate(regional):
        first, last = row["core_indices"]
        boundary = bool(row.get("boundary_core", first == 0 or last == len(e) - 1))
        conflict = bool(row["opposite_exclusive_indices"])
        row["v1_apply_before_conditioning"] = bool(row["apply"])
        row["v1_reasons_before_conditioning"] = list(row["reasons"])
        if boundary or conflict:
            row["route"] = "compact_only"
            row["route_conditions"] = (["native_boundary_core"] if boundary else []) + (
                ["opposite_exclusive_inside_pure_region"] if conflict else [])
            row["apply"] = False
            row["reasons"] = sorted(set(row["reasons"] + ["conditioned_compact_only"]))
            routed.append(i)
        elif row["apply"]:
            row["route"] = "reliable_regional"
            row["route_conditions"] = ["bounded_without_opposite_exclusive"]
            lo, hi = row["affected_bridge_support"]
            protected[lo:hi + 1] = True
            protected_cells[lo:hi] = True
        else:
            row["route"] = "regional_deferred"
            row["route_conditions"] = ["existing_v1_geometry_not_supported"]

    local = []
    for candidate in plan_local_repairs(e, weights, state):
        a, b = candidate["core_indices"]
        parents = [i for i in routed
                   if regional[i]["core_indices"][0] <= a <= b <= regional[i]["core_indices"][1]]
        if not parents:
            continue
        row = deepcopy(candidate)
        row["regional_plan_indices"] = parents
        row["reasons"] = [r for r in row["reasons"] if r != "overlapping_requested_supports"]
        lo, hi = row["support_indices"]
        if 0 <= lo <= hi < len(e) and np.any(protected[lo:hi + 1]):
            row["reasons"].append("local_support_contacts_protected_regional_support")
        local.append(row)
    eligible = [i for i, row in enumerate(local) if not row["reasons"]]
    for k, i in enumerate(eligible):
        a, b = local[i]["support_indices"]
        for j in eligible[k + 1:]:
            c, d = local[j]["support_indices"]
            if min(b, d) > max(a, c):
                local[i]["reasons"].append("overlapping_requested_supports")
                local[j]["reasons"].append("overlapping_requested_supports")
    for row in local:
        row["reasons"] = sorted(set(row["reasons"]))
        row["apply"] = not row["reasons"]
        row["route"] = "compact_applied" if row["apply"] else "compact_deferred"
    return dict(regional_plan=regional, local_plan=local,
                protected_support_mask=protected, protected_cell_mask=protected_cells)


def _unmet_exclusive(e, weights, state):
    required = np.where(state == 1, weights[0], weights[1])
    exclusive = (state == 1) | (state == 2)
    mixed = exclusive & (required > 0) & (required < 1)
    opposite = exclusive & (required == 0)
    records = []
    for source in (0, 1):
        for kind, mask in (("mixed", mixed), ("opposite", opposite)):
            for first, stop in runs(mask & (state == source + 1)):
                first, stop = int(first), int(stop)
                records.append(dict(required_source="P" if source == 0 else "N",
                    inherited_selection=kind, indices=[first, stop - 1],
                    energy_bounds=e[[first, stop - 1]].tolist(),
                    reason="Exclusive evidence is not fully followed by the inherited selector; compact repairs do not override that selector."))
    records.sort(key=lambda row: row["indices"][0])
    return mixed, opposite, records


def render_conditioned(energy, *, baseline, cancelled=None, progress=None):
    """Build one experimental full/conditional-half result without file IO.

    Half inputs keep full states/weights. Their own C/A/reference recompute all
    coefficients. A compact repair's exact exterior is the *already reliable-
    repaired* parent; regional constant tail changes are not claimed absent.
    """
    e = np.asarray(energy)
    frozen = {}

    def policy(axis, weights, state, regional_plan):
        frozen.update(plan_conditioned_repairs(axis, weights, state, regional_plan=regional_plan))
        return frozen["regional_plan"]

    with cancellation_scope(cancelled):
        result = _render_generalized_with_policy(e, baseline=baseline,
            repair_plan_policy=policy, cancelled=cancelled, progress=progress)
        reliable = result["arrays"]
        current = reliable
        stages = result["stages"]
        stages["conditioned_reliable_parent"] = {"arrays": reliable,
            "diagnostics": {"regional_plan": frozen["regional_plan"],
                "meaning": "Current cubic/excursion parent after all permitted unchanged v1 regional operations."}}
        support = np.zeros(len(e), bool)
        interior = np.zeros(len(e), bool)
        cells = np.zeros(len(e) - 1, bool)
        local_core = np.zeros(len(e), bool)
        deferred = np.zeros(len(e), bool)
        regional_route = np.zeros(len(e), np.int8)
        local_route = np.zeros(len(e), np.int8)
        for row in frozen["regional_plan"]:
            lo, hi = row["core_indices"]
            regional_route[lo:hi + 1] = {"reliable_regional": 1, "compact_only": 2, "regional_deferred": 3}[row["route"]]
            if row["route"] == "regional_deferred":
                deferred[lo:hi + 1] = True
        local_records = []
        for i, row in enumerate(frozen["local_plan"]):
            check_cancelled()
            lo, hi = row["core_indices"]
            local_route[lo:hi + 1] = 1 if row["apply"] else 2
            if not row["apply"]:
                deferred[lo:hi + 1] = True
                continue
            name = f"conditioned_compact_{i:03d}"
            if progress is not None:
                progress(name, "start")
            current, diag = _local_increment(e, current, row)
            check_cancelled()
            stages[name] = {"arrays": current, "diagnostics": diag}
            if progress is not None:
                progress(name, "complete")
            a, b = row["support_indices"]
            support[a:b + 1] = True
            interior[a + 1:b] = True
            cells[a:b] = True
            local_core[lo:hi + 1] = True
            local_records.append(diag)

        final = dict(current)
        final.update(parent_background=reliable["background"].copy(),
            parent_background_log=reliable["background_log"].copy(),
            parent_integrated_log_polynomial_coefficients=reliable["integrated_log_polynomial_coefficients"].copy())
        final = _refresh_final_metadata(e, final)
        mixed, opposite, unmet_records = _unmet_exclusive(e, final["source_weights"], final["state"])
        unmet = mixed | opposite
        final.update(method_name=np.array(RECIPE_VERSION),
            construction_type=np.array("conditioned_regional_then_compact_source_shape_repairs"),
            diagnostic_metadata_revision=np.array(METADATA_REVISION),
            conditioned_regional_route=regional_route, conditioned_local_route=local_route,
            conditioned_protected_support_mask=frozen["protected_support_mask"].copy(),
            conditioned_protected_cell_mask=frozen["protected_cell_mask"].copy(),
            conditioned_local_support_mask=support, conditioned_local_interior_mask=interior,
            conditioned_local_cell_mask=cells, conditioned_local_core_mask=local_core,
            local_support_mask=support.copy(), local_interior_mask=interior.copy(),
            support_endpoint_indices=np.array([r["support_indices"] for r in frozen["local_plan"] if r["apply"]], int).reshape(-1, 2),
            conditioned_deferred_mask=deferred, conditioned_unmet_exclusive_mask=unmet,
            conditioned_mixed_exclusive_mask=mixed, conditioned_opposite_exclusive_mask=opposite,
            conditioned_reliable_parent_background=reliable["background"].copy(),
            conditioned_reliable_parent_background_log=reliable["background_log"].copy(),
            conditioned_reliable_parent_polynomial_coefficients=reliable["integrated_log_polynomial_coefficients"].copy(),
            conditioned_local_delta_log=final["background_log"] - reliable["background_log"],
            conditioned_regional_repair_source_indices=np.array([r["source_index"] for r in frozen["regional_plan"] if r["apply"]], np.int8),
            conditioned_regional_repair_core_indices=np.array([r["core_indices"] for r in frozen["regional_plan"] if r["apply"]], int).reshape(-1, 2),
            conditioned_local_repair_source_indices=np.array([r["source_index"] for r in frozen["local_plan"] if r["apply"]], np.int8),
            conditioned_local_repair_core_indices=np.array([r["core_indices"] for r in frozen["local_plan"] if r["apply"]], int).reshape(-1, 2),
            uncertain_mask=(final["state"] == 0) | deferred | unmet, seams=np.array([], float))
        for key in ("background", "background_log"):
            if not np.array_equal(final[key][~interior], reliable[key][~interior]):
                raise ArithmeticError("Compact repair changed its reliable parent's exterior")
            if not np.array_equal(final[key][frozen["protected_support_mask"]], reliable[key][frozen["protected_support_mask"]]):
                raise ArithmeticError("Compact repair changed a protected regional support")
        if not np.array_equal(final["integrated_log_polynomial_coefficients"][:, ~cells],
                              reliable["integrated_log_polynomial_coefficients"][:, ~cells]):
            raise ArithmeticError("Compact repair changed exterior polynomial cells")
        check_cancelled()

    regional_records = result["diagnostics"]["applied_repairs"]
    return {"arrays": final, "stages": stages, "diagnostics": dict(
        algorithm_version=RECIPE_VERSION, metadata_revision=METADATA_REVISION,
        scientifically_validated=False, BN_used=False, stored_model_or_background_used=False,
        source_fit_changed=False, source_state_changed=False,
        source_weights_changed_by_local_repair=False,
        regional_plan=frozen["regional_plan"], local_plan=frozen["local_plan"],
        regional_applied_repairs=regional_records, local_applied_repairs=local_records,
        regional_applied_count=len(regional_records), local_applied_count=len(local_records),
        applied_count=len(regional_records) + len(local_records),
        partial=bool(deferred.any() or unmet.any()), requested_repairs_deferred=bool(deferred.any()),
        unmet_exclusive_count=int(unmet.sum()), mixed_exclusive_count=int(mixed.sum()),
        opposite_exclusive_count=int(opposite.sum()), unmet_exclusive_intervals=unmet_records,
        no_supported_repairs=not bool(regional_records or local_records),
        route_semantics={"regional": "0 no request; 1 reliable regional; 2 compact-only region; 3 regional deferred",
            "local": "0 no local request; 1 compact applied core; 2 compact deferred core"},
        execution_order="Precomputed geometry/state-only routing; reliable v1 repairs first, then compact repairs, each in increasing native energy.",
        protected_regional_support_preserved_by_local=True,
        local_exterior_reliable_parent_bitwise=True,
        reference_semantics="Local parent is after reliable v1 operations; regional tail level changes remain possible. No new global anchor after that parent.",
        half_semantics="Frozen full states/weights determine identical routes; each half recomputes all coefficients from its own C/A/reference.",
        uncertainty_semantics="Unknown states, deferred requests and unmet exclusive source evidence, including mixed and opposite selections. Not a scientific verdict.",
        final_field_contract={"final_curvature_polynomial_coefficients": "Second derivative of final log PPoly.",
            "parent_and_delta_fields": "Final minus complete reliable-repaired parent, covering all compact operations.",
            "background_delta_log": "Final log background minus log current baseline reference.",
            "repair_log_delta": "Final log background minus log raw geometric mean.",
            "inherited_bridge_fields": "Provenance of the last v1 bridge; regional/local plans describe all actual supports."},
        stage_diagnostics={k: v["diagnostics"] for k, v in stages.items()},
        limitations=["Conditional source evidence does not certify clean spectra or dRR/d2 quality.",
            "Reliable means eligible for the fixed regional rule, not scientifically accepted.",
            "Boundary/conflicting regions never receive whole-region source-shape or tail propagation from their own repair.",
            "Fixed compact collars can still concentrate curvature or preserve inherited source defects.",
            "Short exclusive evidence remains unmet when inherited source weights disagree; it is not forced into a repair.",
            "Actual complete dRR and d2 evaluation is required; numeric continuity and old-batch identity are insufficient."])}
