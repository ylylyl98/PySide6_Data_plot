"""Source-symmetric, evidence-selected extension of the regional V66 recipe.

Region count, energy location and source labels are not recipe parameters. A
current conditional exclusive run must contain a complete nominal SG21 window
and agree with the selected pure source. Conflicting adjacent repairs are
reported and deferred together, never resolved by source-name preference.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PPoly, PchipInterpolator

from . import render_cubic, render_excursion, render_shape, render_right, render_left
from .regions_helpers import runs
from .regions_selection import remove_unknown_excursions
from .source_runtime import cancellation_scope, check_cancelled

RECIPE_VERSION = "pn_evidence_selected_regional_curvature_v1"
METADATA_REVISION = 2


def _refresh_final_metadata(e, arrays):
    """Refresh final-curve aliases without changing the constructed curve.

    ``delta_*`` is relative to the last construction parent. The complete
    method's reference delta and raw-PN delta have separate explicit fields.
    Ancestor selected/compensation curves are provenance, not final curvature.
    """
    out=dict(arrays)
    poly=PPoly(out["integrated_log_polynomial_coefficients"],e,extrapolate=False)
    parent=PPoly(out["parent_integrated_log_polynomial_coefficients"],e,extrapolate=False)
    delta_coeff=poly.c-parent.c
    delta=PPoly(delta_coeff,e,extrapolate=False)
    for key in ("curvature_polynomial_coefficients","parent_curvature_polynomial_coefficients",
                "delta_selected_curvature_polynomial_coefficients","compensation_curvature_polynomial_coefficients"):
        if key in out:out["generalized_inherited_"+key]=out.pop(key)
    changed=out["background"]!=out["parent_background"]
    out.update(construction_type=np.array("source_symmetric_regional_shape_and_curvature_bridges"),
        diagnostic_metadata_revision=np.array(METADATA_REVISION),
        final_curvature_polynomial_coefficients=poly.derivative(2).c,
        parent_final_curvature_polynomial_coefficients=parent.derivative(2).c,
        stitched_log_first_derivative=poly(e,1),stitched_log_second_derivative=poly(e,2),
        delta_integrated_log_polynomial_coefficients=delta_coeff,
        delta_closed_curvature_polynomial_coefficients=delta.derivative(2).c,
        delta_log_first_derivative=delta(e,1),delta_log_second_derivative=delta(e,2),
        delta_background_log=out["background_log"]-out["parent_background_log"],
        background_delta_log=out["background_log"]-np.log(out["baseline_reference_background"]),
        repair_log_delta=out["background_log"]-np.log(out["raw_geometric_mean"]),
        actual_change_mask=changed.copy(),modification_mask=changed.copy(),repair_mask=changed.copy(),
        changed_background_mask=changed.copy(),changed_curvature_cell_mask=np.any(delta_coeff!=0,axis=0),
        source_curvature_weights=out["source_weights"].copy(),
        weight_pchip_coefficients=PchipInterpolator(e,out["source_weights"][0]).c.copy())
    return out


def select_repair_regions(e, weights, state):
    """Return a fixed full/half-compatible plan using source states and geometry.

    The nominal uniform SG21 support is the same resolution test used by the
    existing regional selector. It is not a noise threshold or clean certificate.
    """
    e=np.asarray(e); weights=np.asarray(weights); state=np.asarray(state)
    n=len(e)
    if n<21 or weights.shape!=(2,n) or state.shape!=(n,):
        raise ValueError("Complete native weights/states with at least 21 points required")
    if not np.all(np.isfinite(e)) or np.any(np.diff(e)<=0):
        raise ValueError("Finite increasing native energy required")
    if not np.all(np.isfinite(weights)) or np.any((weights<0)|(weights>1)):
        raise ValueError("Invalid source weights")
    if not np.all(np.isin(state,[0,1,2,3])):
        raise ValueError("Unknown conditional state code")
    np.testing.assert_allclose(weights.sum(0),1,rtol=0,atol=2e-15)
    uniform=np.linspace(e[0],e[-1],n)
    span=20*float(e[-1]-e[0])/(n-1)
    mixed=[(int(l-1),int(r)) for l,r in runs((weights[0]>0)&(weights[0]<1))]
    rows=[]
    for source in (0,1):
        for l,r in runs(weights[source]==1):
            left,right=int(l),int(r-1)
            witnesses=[]
            for a,b in runs(state==source+1):
                if a<l or b>r:
                    continue
                centers=np.flatnonzero((uniform[:-20]>=e[a])&(uniform[20:]<=e[b-1]))+10
                if centers.size:
                    witnesses.append(dict(first_index=int(a),stop_exclusive=int(b),
                        complete_SG21_center_indices=centers.tolist()))
            if not witnesses:
                continue
            row=dict(source_index=source,source="P" if source==0 else "N",
                core_indices=[left,right],core_energy_bounds=e[[left,right]].tolist(),
                exclusive_witnesses=witnesses,apply=False,reasons=[],
                core_state_counts={str(k):int(np.sum(state[left:right+1]==k)) for k in range(4)},
                opposite_exclusive_indices=np.flatnonzero((state==2-source)&(np.arange(n)>=left)&(np.arange(n)<=right)).tolist())
            lbs=[x for x in mixed if x[1]==left]
            rbs=[x for x in mixed if x[0]==right]
            if len(lbs)>1 or len(rbs)>1 or (not lbs and left!=0) or (not rbs and right!=n-1):
                row["reasons"].append("boundary_or_nonisolated_core_without_two_mixed_bridges")
                rows.append(row)
                continue
            lb=lbs[0] if lbs else None
            rb=rbs[0] if rbs else None
            row["original_bridge_indices"]=[list(lb) if lb else None,list(rb) if rb else None]
            row["boundary_core"]=lb is None or rb is None
            if lb is None and rb is None:
                row["reasons"].append("entire_axis_pure_source_has_no_bridge_level_objective")
                rows.append(row)
                continue
            if (lb is not None and lb[0]<1) or (rb is not None and rb[1]>=n-1):
                row["reasons"].append("original_bridge_reaches_native_boundary")
                rows.append(row)
                continue
            el=int(np.searchsorted(e,e[left]-max(float(e[left]-e[lb[0]]),2*span),side="right")-1) if lb else 0
            er=int(np.searchsorted(e,e[right]+max(float(e[rb[1]]-e[right]),2*span),side="left")) if rb else n-1
            row["extended_bridge_indices"]=[[el,left] if lb else None,[right,er] if rb else None]
            row["affected_bridge_support"]=[el,er]
            if el<0 or er>=n:
                row["reasons"].append("two_SG21_extension_exceeds_native_domain")
            else:
                extension_sides=[]
                if lb: extension_sides.append(("left",el,lb[0]))
                if rb: extension_sides.append(("right",rb[1],er))
                for side,a,b in extension_sides:
                    if np.any(weights[source,a:b+1]!=0):
                        row["reasons"].append(side+"_extension_not_inside_neighbor_pure_source")
                    if np.any(state[a:b+1]==source+1):
                        row["reasons"].append(side+"_extension_conflicts_with_selected_source_evidence")
            rows.append(row)
    rows.sort(key=lambda row:row["core_indices"][0])
    feasible=[i for i,row in enumerate(rows) if not row["reasons"]]
    # Check all requested supports before construction; no greedy first-source win.
    for k,i in enumerate(feasible):
        a,b=rows[i]["affected_bridge_support"]
        for j in feasible[k+1:]:
            c,d=rows[j]["affected_bridge_support"]
            if min(b,d)>max(a,c):
                rows[i]["reasons"].append("shared_or_overlapping_repair_bridge")
                rows[j]["reasons"].append("shared_or_overlapping_repair_bridge")
    for row in rows:
        row["reasons"]=sorted(set(row["reasons"]))
        row["apply"]=not row["reasons"]
    return rows


def _canonical_source(baseline, source):
    """Make the selected source row zero; reverse q and state semantics too."""
    if source==0:
        return baseline
    b=dict(baseline)
    for key in ("original_corrected_source_log","original_aligned_source_log",
                "source_weights","reference_source_weights","source_usable_masks",
                "parent_source_weights","effective_source_weights"):
        if key not in baseline: continue
        b[key]=np.asarray(baseline[key])[::-1].copy()
    if "source_log_spline_coefficients" in baseline:
        b["source_log_spline_coefficients"]=np.asarray(baseline["source_log_spline_coefficients"])[:,:,::-1].copy()
    b["alignment_q_log"]=-np.asarray(baseline["alignment_q_log"])
    state=np.asarray(baseline["state"])
    b["state"]=np.where(state==1,2,np.where(state==2,1,state)).astype(state.dtype)
    return b


def _restore_sources(arrays, baseline, source):
    if source==0:
        return arrays
    for key in ("original_corrected_source_log","original_aligned_source_log",
                "alignment_q_log","state","reference_source_weights",
                "source_usable_masks","source_log_spline_coefficients"):
        if key not in arrays: continue
        arrays[key]=np.asarray(baseline[key]).copy()
    for key in ("source_weights","parent_source_weights","effective_source_weights","weight_delta"):
        if key in arrays:
            arrays[key]=arrays[key][::-1].copy()
    return arrays


def _render_generalized_with_policy(energy, *, baseline, repair_plan_policy=None,
                                    cancelled=None, progress=None):
    """Recompute all parameters from one current full/conditional-half baseline.

    Unsupported repairs remain explicitly deferred over the current cubic /
    excursion-closure baseline. No saved background or alternative algorithm is
    loaded. A half uses full states/weights and its own source/anchor numerics.
    """
    e=np.asarray(energy)
    np.testing.assert_array_equal(e,baseline["energy"])
    reference=np.asarray(baseline["baseline_reference_background"])
    np.testing.assert_array_equal(baseline["background"],reference)
    np.testing.assert_array_equal(baseline["reference_source_weights"],baseline["source_weights"])
    stages={}
    def run(name,fn):
        check_cancelled()
        if progress is not None: progress(name,"start")
        arrays,diag=fn()
        check_cancelled()
        stages[name]={"arrays":arrays,"diagnostics":diag}
        if progress is not None: progress(name,"complete")
        return arrays
    with cancellation_scope(cancelled):
        parent=run("cubic_reference",lambda:render_cubic.generate(e,baseline=baseline))
        selection=run("excursion_selection",lambda:remove_unknown_excursions(
            e,baseline["source_weights"],baseline["state"]))
        selected=dict(baseline)
        selected.update(source_weights=selection["source_weights"],
            parent_source_weights=baseline["source_weights"],reference_source_weights=baseline["source_weights"])
        current=run("excursion_closure",lambda:render_excursion.generate(
            e,baseline=selected,parent=parent,symmetric=True))
        stages["excursion_closure"]["diagnostics"]["limitations"]=[
            "The currently detected same-source all-unknown excursions receive compact two-moment closure.",
            "Closure can concentrate curvature inside a current excursion support.",
            "The current cubic reference remains fixed; no source or background quality is certified."]
        plan=select_repair_regions(e,current["source_weights"],current["state"])
        if repair_plan_policy is not None:
            plan=repair_plan_policy(e,current["source_weights"],current["state"],plan)
        deferred=np.zeros(len(e),bool)
        conflicting_evidence=np.zeros(len(e),bool)
        records=[]
        for number,row in enumerate(plan):
            check_cancelled()
            if not row["apply"]:
                l,r=row["core_indices"]
                deferred[l:r+1]=True
                continue
            src=row["source_index"]
            conflicting_evidence[row["opposite_exclusive_indices"]]=True
            shaped=run(f"region_{number:03d}_shape",lambda:render_shape.generate(
                e,baseline=current,source_index=src,core=row["core_indices"]))
            current={**current,**shaped}
            shape_diag=stages[f"region_{number:03d}_shape"]["diagnostics"]
            shape_diag["family"]="local_source_region_shape_increment"
            shape_diag["limitations"]=[
                "The selected source is conditionally supported by an exclusive subinterval, not proven clean across the entire pure run.",
                "The scalar minimizes bridge log-curvature change, not full sample dRR/d2 error.",
                "The original source/reference remain current inputs; this repair adds no new source fit."]
            # Relative bridge curvature is selected source versus the other
            # source plus the inherited cubic. Row swapping changes both sides.
            left=right=None
            if row["original_bridge_indices"][1] is not None:
                before=current
                canonical=_canonical_source(before,src)
                right=run(f"region_{number:03d}_right",lambda:render_right.generate(
                    e,baseline=canonical,region=row["original_bridge_indices"][1],
                    extension_mode="no_selected_exclusive"))
                right=_restore_sources(right,before,src)
                stages[f"region_{number:03d}_right"]["diagnostics"].update(
                    selected_source=row["source"],neighbor_source="N" if src==0 else "P",
                    new_curvature_weights="Selected-source quintic 1-to-0; the opposite source includes inherited cubic curvature.",
                    extension_state_unique=np.unique(before["state"][right["original_bridge_endpoint_indices"][1]:right["bridge_endpoint_indices"][1]+1]).tolist())
                current={**before,**right}
            if row["original_bridge_indices"][0] is not None:
                before=current
                canonical=_canonical_source(before,src)
                left=run(f"region_{number:03d}_left",lambda:render_left.generate(
                    e,baseline=canonical,region=row["original_bridge_indices"][0]))
                left=_restore_sources(left,before,src)
                stages[f"region_{number:03d}_left"]["diagnostics"].update(
                    selected_source=row["source"],neighbor_source="N" if src==0 else "P",
                    new_curvature_weights="Selected-source quintic 0-to-1; the opposite source includes inherited cubic curvature.",
                    extension_state_unique=np.unique(before["state"][left["bridge_endpoint_indices"][0]:left["original_bridge_endpoint_indices"][0]+1]).tolist())
                current={**before,**left}
            records.append(dict(source=row["source"],core_indices=row["core_indices"],
                exclusive_witnesses=row["exclusive_witnesses"],
                regional_log_constant=shape_diag["optimal_regional_log_constant"],
                boundary_core=row["boundary_core"],opposite_exclusive_indices=row["opposite_exclusive_indices"],
                right_log_level_offset=float(right["right_log_level_offset"]) if right else None,
                left_log_level_offset=float(left["left_log_level_offset"]) if left else None,
                right_bridge_indices=right["bridge_endpoint_indices"].tolist() if right else None,
                left_bridge_indices=left["bridge_endpoint_indices"].tolist() if left else None))
        current=_refresh_final_metadata(e,current)
        current.update(method_name=np.array(RECIPE_VERSION),
            generalized_deferred_mask=deferred,
            generalized_conflicting_source_evidence_mask=conflicting_evidence,
            uncertain_mask=(current["state"]==0)|deferred|conflicting_evidence,
            generalized_repair_source_indices=np.array([r["source_index"] for r in plan if r["apply"]],dtype=np.int8),
            generalized_repair_core_indices=np.array([r["core_indices"] for r in plan if r["apply"]],int).reshape(-1,2),
            generalized_initial_source_weights=selection["source_weights"].copy())
        check_cancelled()
    return {"arrays":current,"stages":stages,"diagnostics":dict(
        algorithm_version=RECIPE_VERSION,metadata_revision=METADATA_REVISION,scientifically_validated=False,BN_used=False,
        stored_model_or_background_used=False,repair_plan=plan,applied_repairs=records,
        partial=any(not r["apply"] for r in plan) or bool(conflicting_evidence.any()),
        applied_count=len(records),conflicting_exclusive_count=int(conflicting_evidence.sum()),
        no_supported_repairs=not bool(records),
        execution_order="Increasing native energy; all conflict decisions made before construction.",
        selection_rule="Pure selected source contains a current exclusive-state run with a complete nominal SG21 window.",
        extension_rule="Two nominal SG21 spans in neighboring opposite pure source, without selected-source exclusive contradiction.",
        half_semantics="Full conditional states and source weights; own C/A/reference and independently recomputed bridge coefficients.",
        source_weights_semantics="Curvature selectors, not pointwise intensity mixture.",
        final_field_contract={
            "final_curvature_polynomial_coefficients":"Exact second derivative of final integrated log PPoly.",
            "delta_fields":"Final curve minus last construction parent; parent_* identifies that stage.",
            "background_delta_log":"Final native log background minus log baseline_reference_background.",
            "repair_log_delta":"Final native log background minus log raw_geometric_mean.",
            "generalized_inherited_fields":"Ancestor selected/compensation curvature provenance, not final curve.",
            "bridge_geometry_fields":"Last bridge/shape stage; applied_repairs gives every regional operation."},
        stage_diagnostics={k:v["diagnostics"] for k,v in stages.items()},
        limitations=["Training sample consensus and conditional source states do not prove featurelessness.",
            "Unresolved source intervals remain explicit; same-source unknown removal is an engineering continuity choice.",
            "Boundary cores use only their existing bridge; entire-axis pure cores and overlapping repairs are deferred.",
            "Opposite-source short exclusive states inside a selected pure core remain unresolved evidence and are reported, not certified away.",
            "Multiple bridges propagate constant levels into tails and may change the full dRR baseline.",
            "Numeric closure and original-batch regression do not establish quality for another acquisition."])}


def render_generalized(energy, *, baseline, cancelled=None, progress=None):
    """Run the unchanged generalized v1 recipe, without conditional routing."""
    return _render_generalized_with_policy(
        energy, baseline=baseline, cancelled=cancelled, progress=progress)
