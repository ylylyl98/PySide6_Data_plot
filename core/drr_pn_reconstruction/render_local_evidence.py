"""Evidence-local source-shape increments with unchanged exterior background.

This is a separate experimental renderer. It does not revise the generalized
v1 or legacy V66 recipe. Each qualifying exclusive interval gets fixed two-SG21
collars; its source shape is joined through compact C2 increments. No level
offset is propagated into either tail and no global anchor is refitted.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PPoly

from . import render_cubic, render_excursion
from .render_shape import _hermite, _shift, _inner
from .regions_helpers import runs
from .regions_selection import remove_unknown_excursions
from .source_runtime import cancellation_scope, check_cancelled

RECIPE_VERSION="pn_local_exclusive_shape_increment_v1"
METADATA_REVISION=3


def plan_local_repairs(e,weights,state):
    """Use exclusive evidence itself, not the length of model-risk pure runs."""
    e=np.asarray(e);weights=np.asarray(weights);state=np.asarray(state);n=len(e)
    if n<21 or e.ndim!=1 or not np.all(np.isfinite(e)) or np.any(np.diff(e)<=0):
        raise ValueError("Finite increasing native energy with at least 21 samples required")
    if weights.shape!=(2,n) or state.shape!=(n,) or not np.all(np.isin(state,[0,1,2,3])):
        raise ValueError("Aligned native source weights and conditional states required")
    if not np.all(np.isfinite(weights)) or np.any((weights<0)|(weights>1)):
        raise ValueError("Invalid source weights")
    np.testing.assert_allclose(weights.sum(0),1,rtol=0,atol=2e-15)
    uniform=np.linspace(e[0],e[-1],n)
    collar=2*20*float(e[-1]-e[0])/(n-1)
    rows=[]
    for source in (0,1):
        for first,stop in runs(state==source+1):
            first,stop=int(first),int(stop)
            centers=np.flatnonzero((uniform[:-20]>=e[first])&(uniform[20:]<=e[stop-1]))+10
            if not centers.size:
                continue
            last=stop-1
            left=int(np.searchsorted(e,e[first]-collar,side="right")-1)
            right=int(np.searchsorted(e,e[last]+collar,side="left"))
            row=dict(source_index=source,source="P" if source==0 else "N",
                core_indices=[first,last],core_energy_bounds=e[[first,last]].tolist(),
                complete_SG21_center_indices=centers.tolist(),collar_width_eV=collar,
                support_indices=[left,right],apply=False,reasons=[])
            if not np.all(weights[source,first:stop]==1):
                row["reasons"].append("exclusive_core_not_selected_pure_source")
            if left<0 or right>=n:
                row["reasons"].append("fixed_collar_exceeds_native_domain")
            else:
                row["support_energy_bounds"]=e[[left,right]].tolist()
                if np.any(state[left:right+1]==2-source):
                    row["reasons"].append("fixed_collar_crosses_opposite_exclusive_evidence")
            rows.append(row)
    rows.sort(key=lambda row:row["core_indices"][0])
    eligible=[i for i,row in enumerate(rows) if not row["reasons"]]
    for k,i in enumerate(eligible):
        a,b=rows[i]["support_indices"]
        for j in eligible[k+1:]:
            c,d=rows[j]["support_indices"]
            if min(b,d)>max(a,c):
                rows[i]["reasons"].append("overlapping_requested_supports")
                rows[j]["reasons"].append("overlapping_requested_supports")
    for row in rows:
        row["reasons"]=sorted(set(row["reasons"]))
        row["apply"]=not row["reasons"]
    return rows


def _local_increment(e,baseline,row):
    """One compact source-shape repair; scalar objective is exactly quadratic."""
    source_index=row["source_index"]
    left,right=row["core_indices"]
    start,end=row["support_indices"]
    if not 0<=start<left<right<end<len(e):
        raise ValueError("Two complete fixed collars required")
    pc=np.asarray(baseline["integrated_log_polynomial_coefficients"])
    sc=np.asarray(baseline["source_log_spline_coefficients"])[:,:,source_index]
    if pc.shape!=(7,len(e)-1) or sc.shape!=(4,len(e)-1):
        raise ValueError("Native source and parent polynomial shapes mismatch")
    A=np.asarray(baseline["original_aligned_source_log"])
    parent=PPoly(pc,e,extrapolate=False);source=PPoly(sc,e,extrapolate=False)
    np.testing.assert_allclose(source(e),A[source_index],rtol=0,atol=1e-12)
    d0=np.zeros_like(pc[::-1]);mode=np.zeros_like(d0)
    objective=[]
    for side,(l,r) in enumerate(((start,left),(right,end))):
        H=float(e[r]-e[l])
        j0=[0.,0.,0.] if side==0 else [float(source(e[l],k)-parent(e[l],k)) for k in range(3)]
        j1=[float(source(e[r],k)-parent(e[r],k)) for k in range(3)] if side==0 else [0.,0.,0.]
        h0=_hermite(j0,j1,H)
        hm=np.array([0.,0.,0.,10.,-15.,6.]) if side==0 else np.array([1.,0.,0.,-10.,15.,-6.])
        aa=bb=cc=0.
        for i in range(l,r):
            check_cancelled()
            dx=float(e[i+1]-e[i]);t0=float((e[i]-e[l])/H)
            d0[:,i]=_shift(h0,t0,H);mode[:,i]=_shift(hm,t0,H)
            d=d0[2:,i]*np.arange(2,7)*np.arange(1,6)
            m=mode[2:,i]*np.arange(2,7)*np.arange(1,6)
            aa+=_inner(d,d,dx);bb+=_inner(d,m,dx);cc+=_inner(m,m,dx)
        objective.append((aa,bb,cc))
    Q=float(sum(t[2] for t in objective));v=float(sum(t[1] for t in objective))
    if not np.isfinite(Q) or Q<=0:
        raise ArithmeticError("Local bridge-level objective is not strictly convex")
    level=-v/Q
    for i in range(left,right):
        d0[:,i]=-pc[::-1,i];d0[:4,i]+=sc[::-1,i];mode[0,i]=1.
    coeff=pc.copy()
    coeff[:,start:end]+=(d0+level*mode)[::-1,start:end]
    coeff[:,left:right]=0.
    coeff[-4:,left:right]=sc[:,left:right]
    coeff[-1,left:right]+=level
    poly=PPoly(coeff,e,extrapolate=False)
    support=np.zeros(len(e),bool);support[start:end+1]=True
    interior=np.zeros(len(e),bool);interior[start+1:end]=True
    core=np.zeros(len(e),bool);core[left:right+1]=True
    L=np.asarray(baseline["background_log"]).copy()
    B=np.asarray(baseline["background"]).copy()
    L[interior]=poly(e[interior]);L[core]=A[source_index,core]+level
    B[interior]=np.exp(L[interior])
    if not np.all(np.isfinite(B)&(B>0)):
        raise ArithmeticError("Nonpositive/nonfinite local background")
    native_error=float(np.max(abs(poly(e)-L)))
    endpoint_error=[float(np.max(abs(poly(e[[start,end]],k)-parent(e[[start,end]],k)))) for k in range(3)]
    joins=[]
    for k in range(3):
        p=poly.derivative(k)
        endvalues=np.polynomial.polynomial.polyval(np.diff(e),p.c[::-1],tensor=False)
        joins.append(float(np.max(abs(endvalues[:-1]-p.c[-1,1:]))))
    if native_error>1e-10 or max(joins[0],endpoint_error[0])>1e-10 or max(joins[1],endpoint_error[1])>1e-8 or max(joins[2],endpoint_error[2])>1e-6:
        raise ArithmeticError("Local native polynomial or C2 closure failed")
    arrays=dict(baseline)
    arrays.update(background=B,background_log=L,integrated_log_polynomial_coefficients=coeff,
        stitched_log_first_derivative=poly(e,1),stitched_log_second_derivative=poly(e,2),
        local_evidence_support_mask=support,local_evidence_core_mask=core,
        local_evidence_regional_log_constant=np.array(level))
    diag=dict(source=row["source"],core_indices=[left,right],support_indices=[start,end],
        local_log_constant=level,objective_Q=Q,objective_linear_half=v,
        objective_stationarity=Q*level+v,C0_C1_C2_join_max=joins,
        endpoint_parent_jet_error=endpoint_error,native_PPoly_log_error=native_error,
        exterior_background_bitwise=bool(np.array_equal(B[~interior],baseline["background"][~interior])),
        exterior_coefficients_bitwise=bool(np.array_equal(coeff[:,:start],pc[:,:start]) and np.array_equal(coeff[:,end:],pc[:,end:])))
    return arrays,diag


def render_local_evidence(energy,*,baseline,cancelled=None,progress=None):
    """Return full arrays and explicit partial/support diagnostics; no file IO."""
    e=np.asarray(energy)
    np.testing.assert_array_equal(e,baseline["energy"])
    np.testing.assert_array_equal(baseline["background"],baseline["baseline_reference_background"])
    np.testing.assert_array_equal(baseline["source_weights"],baseline["reference_source_weights"])
    stages={}
    def run(name,fn):
        check_cancelled()
        if progress is not None:progress(name,"start")
        a,d=fn();check_cancelled()
        stages[name]={"arrays":a,"diagnostics":d}
        if progress is not None:progress(name,"complete")
        return a
    with cancellation_scope(cancelled):
        cubic=run("cubic_reference",lambda:render_cubic.generate(e,baseline=baseline))
        selection=run("excursion_selection",lambda:remove_unknown_excursions(e,baseline["source_weights"],baseline["state"]))
        selected=dict(baseline)
        selected.update(source_weights=selection["source_weights"],parent_source_weights=baseline["source_weights"])
        parent=run("local_parent",lambda:render_excursion.generate(e,baseline=selected,parent=cubic,symmetric=True))
        # The numerical kernel also serves the historical fixed-data recipe.
        # Describe its source-symmetric use here without editing that recipe.
        stages["local_parent"]["diagnostics"]["limitations"]=[
            "Same-source all-unknown excursions are closed on their currently detected supports.",
            "Local two-moment closure can concentrate curvature inside a current excursion support.",
            "The inherited cubic reference is not reanchored after an excursion or shape repair.",
            "Conditional source states and numerical closure do not establish full product quality."]
        current=parent
        plan=plan_local_repairs(e,current["source_weights"],current["state"])
        union=np.zeros(len(e),bool);core=np.zeros(len(e),bool);deferred=np.zeros(len(e),bool)
        records=[]
        for i,row in enumerate(plan):
            if not row["apply"]:
                l,r=row["core_indices"];deferred[l:r+1]=True
                continue
            current=run(f"local_repair_{i:03d}",lambda:_local_increment(e,current,row))
            union|=current["local_evidence_support_mask"]
            core|=current["local_evidence_core_mask"]
            records.append(stages[f"local_repair_{i:03d}"]["diagnostics"])
        final=dict(current)
        required_weight=np.where(final["state"]==1,final["source_weights"][0],final["source_weights"][1])
        exclusive=(final["state"]==1)|(final["state"]==2)
        mixed_exclusive=exclusive&(required_weight>0)&(required_weight<1)
        opposite_exclusive=exclusive&(required_weight==0)
        unmet_exclusive=mixed_exclusive|opposite_exclusive
        unmet_records=[]
        for source in (0,1):
            for kind,mask in (("mixed",mixed_exclusive),("opposite",opposite_exclusive)):
                for first,stop in runs(mask&(final["state"]==source+1)):
                    first,stop=int(first),int(stop)
                    unmet_records.append(dict(required_source="P" if source==0 else "N",
                        inherited_selection=kind,indices=[first,stop-1],
                        energy_bounds=e[[first,stop-1]].tolist(),
                        reason="Exclusive evidence is not fully followed by the inherited selector; the local repair does not override it."))
        unmet_records.sort(key=lambda row:row["indices"][0])
        delta=final["background_log"]-parent["background_log"]
        coeff=final["integrated_log_polynomial_coefficients"]
        delta_coeff=coeff-parent["integrated_log_polynomial_coefficients"]
        poly=PPoly(coeff,e,extrapolate=False)
        delta_poly=PPoly(delta_coeff,e,extrapolate=False)
        interior=np.zeros(len(e),bool);active_cells=np.zeros(len(e)-1,bool)
        endpoints=[]
        for row in plan:
            if row["apply"]:
                left,right=row["support_indices"]
                interior[left+1:right]=True;active_cells[left:right]=True
                endpoints.append([left,right])
        # These inherited fields describe only the parent's excursion. Keep
        # that provenance under explicit names rather than claiming that they
        # describe the newly repaired polynomial.
        for key in ("curvature_polynomial_coefficients",
                    "delta_selected_curvature_polynomial_coefficients",
                    "compensation_curvature_polynomial_coefficients",
                    "parent_curvature_polynomial_coefficients"):
            if key in final:
                final["local_evidence_parent_"+key]=final.pop(key)
        final.update(construction_type=np.array("compact_source_shape_increment_on_current_cubic_excursion_parent"),
            parent_background=parent["background"].copy(),parent_background_log=parent["background_log"].copy(),
            parent_integrated_log_polynomial_coefficients=parent["integrated_log_polynomial_coefficients"].copy(),
            parent_final_curvature_polynomial_coefficients=parent["final_curvature_polynomial_coefficients"].copy(),
            local_support_mask=union.copy(),local_interior_mask=interior,
            support_endpoint_indices=np.asarray(endpoints,int).reshape(-1,2),
            changed_curvature_cell_mask=active_cells,
            actual_change_mask=final["background"]!=parent["background"],
            modification_mask=final["background"]!=parent["background"],
            repair_mask=final["background"]!=parent["background"],
            delta_background_log=delta.copy(),
            background_delta_log=final["background_log"]-np.log(final["baseline_reference_background"]),
            repair_log_delta=final["background_log"]-np.log(final["raw_geometric_mean"]),
            delta_integrated_log_polynomial_coefficients=delta_coeff,
            final_curvature_polynomial_coefficients=poly.derivative(2).c,
            delta_closed_curvature_polynomial_coefficients=delta_poly.derivative(2).c,
            delta_log_first_derivative=delta_poly(e,1),delta_log_second_derivative=delta_poly(e,2),
            endpoint_jet_error=np.max(np.array([r["endpoint_parent_jet_error"] for r in records]),axis=0) if records else np.zeros(3),
            diagnostic_metadata_revision=np.array(METADATA_REVISION))
        final.update(method_name=np.array(RECIPE_VERSION),local_evidence_support_mask=union,
            local_evidence_core_mask=core,local_evidence_deferred_mask=deferred,
            local_evidence_unmet_exclusive_mask=unmet_exclusive,
            local_evidence_mixed_exclusive_mask=mixed_exclusive,
            local_evidence_opposite_exclusive_mask=opposite_exclusive,
            local_evidence_parent_background=parent["background"].copy(),
            local_evidence_parent_background_log=parent["background_log"].copy(),
            local_evidence_delta_log=delta,
            local_evidence_parent_polynomial_coefficients=parent["integrated_log_polynomial_coefficients"].copy(),
            local_evidence_repair_sources=np.array([r["source_index"] for r in plan if r["apply"]],np.int8),
            local_evidence_repair_cores=np.array([r["core_indices"] for r in plan if r["apply"]],int).reshape(-1,2),
            uncertain_mask=(final["state"]==0)|deferred|unmet_exclusive,seams=np.array([],float))
        outside=~union
        if not np.array_equal(final["background"][outside],parent["background"][outside]):
            raise ArithmeticError("Unexpected nonlocal background change")
    return {"arrays":final,"stages":stages,"diagnostics":dict(
        algorithm_version=RECIPE_VERSION,metadata_revision=METADATA_REVISION,scientifically_validated=False,BN_used=False,
        source_fit_changed=False,source_state_changed=False,source_weights_changed_by_local_repair=False,
        repair_plan=plan,applied_repairs=records,applied_count=len(records),
        partial=any(not r["apply"] for r in plan) or bool(unmet_exclusive.any()),
        requested_repairs_deferred=any(not r["apply"] for r in plan),
        unmet_exclusive_count=int(unmet_exclusive.sum()),
        mixed_exclusive_count=int(mixed_exclusive.sum()),opposite_exclusive_count=int(opposite_exclusive.sum()),
        unmet_exclusive_intervals=unmet_records,no_supported_repairs=not bool(records),
        uncertainty_semantics="Unknown states, deferred requested repairs, and exclusive evidence not fully followed by the inherited mixed/opposite selector. Not a quality verdict.",
        exterior_local_parent_background_bitwise=True,new_global_anchor=False,tail_level_propagation=False,
        source_weights_semantics="Inherited curvature selectors; local source-shape increments are separate from pointwise blending.",
        half_semantics="Fixed full states/weights, own C/A/reference and independently recomputed local coefficients.",
        stage_diagnostics={k:v["diagnostics"] for k,v in stages.items()},
        limitations=["Exclusive sample-consistency evidence remains conditional, not a proof of featurelessness.",
            "Two-SG21 collars are a fixed engineering scale; concentrated bridge curvature can still damage weak structure.",
            "Unsupported collars or opposing-source evidence cause explicit deferral, not a narrower substitute bridge.",
            "Exterior identity preserves this run's parent source/model defects as well as useful structure.",
            "No claim of full dRR/d2 quality follows from mathematical continuity or locality."])}
