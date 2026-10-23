"""V32: frozen V31 per-source residual curves on the frozen V29 baseline.

Pure NumPy construction; no fitting or data IO. Original fields retain their
original meaning; explicitly v32 effective fields describe the new background.
"""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "cc6e93464b5a0b4d26db5c0a6f51389fe6382a43de9d75f99e88b0e708b6344b"
import copy
import numpy as np

FAMILIES=("full_source_residual_cleanup",)
STABILITY_LIMIT=.25


def _curves(e,parameters,coefficients):
    parameters=np.asarray(parameters,dtype=float)
    coefficients=np.asarray(coefficients,dtype=float)
    if parameters.shape!=(2,) or coefficients.shape!=(2,2) or parameters[1]<=0:
        raise ValueError("Invalid frozen common-resonance parameters")
    if not np.isfinite(parameters).all() or not np.isfinite(coefficients).all():
        raise ValueError("Frozen fit contains nonfinite values")
    z=(e-parameters[0])/parameters[1]
    return coefficients@np.array([1/(1+z*z),z/(1+z*z)])


def frozen_source_stage(e,v31_diagnostics):
    """Reconstruct and check saved source curves without refitting any data."""
    e=np.asarray(e,dtype=float)
    cell=np.gradient(e)
    events=v31_diagnostics["v31"]["events"]
    curves=np.zeros((len(events),2,len(e)))
    applied=np.zeros(len(events),bool)
    audits=[]
    for i,row in enumerate(events):
        audit=dict(event_index=i,seed_energy=row["seed_energy"],v31_applied=bool(row["applied"]),
            v31_reasons=copy.deepcopy(row["reasons"]),applied=False,reasons=[])
        if not row["applied"]:audit["reasons"].append("not_accepted_by_frozen_V31")
        if not all(k in row for k in ("training","conditional_last50","cubic_fixed_shape","fit_point_indices")):
            audit["reasons"].append("missing_frozen_fit")
            audits.append(audit);continue
        train=row["training"];check=row["conditional_last50"];cubic=row["cubic_fixed_shape"]
        f=_curves(e,train["parameters"],train["coefficients"])
        h=_curves(e,check["parameters"],check["coefficients"])
        g=_curves(e,train["parameters"],cubic["coefficients"])
        curves[i]=f
        ids=np.asarray(row["fit_point_indices"],dtype=int)
        if ids.ndim!=1 or len(ids)==0 or np.any((ids<0)|(ids>=len(e))):
            raise ValueError("Invalid frozen fit point indices")
        w=cell[ids]/cell[ids].sum()
        norm=np.sqrt(np.sum(f[:,ids]**2*w,axis=1))
        half=np.sqrt(np.sum((h[:,ids]-f[:,ids])**2*w,axis=1))
        degree=np.sqrt(np.sum((g[:,ids]-f[:,ids])**2*w,axis=1))
        guard_defined=bool(np.all(norm>0))
        if not guard_defined:
            audit["reasons"].append("undefined_zero_source_norm")
            half_relative=None;degree_relative=None
        else:
            half_relative=half/norm;degree_relative=degree/norm
            if np.any(half_relative>STABILITY_LIMIT):audit["reasons"].append("source_half_frame_instability")
            if np.any(degree_relative>STABILITY_LIMIT):audit["reasons"].append("source_cubic_nuisance_instability")
        audit.update(training_parameters=copy.deepcopy(train["parameters"]),
            training_source_coefficients=copy.deepcopy(train["coefficients"]),
            conditional_parameters=copy.deepcopy(check["parameters"]),
            conditional_source_coefficients=copy.deepcopy(check["coefficients"]),
            cubic_source_coefficients=copy.deepcopy(cubic["coefficients"]),
            fit_point_indices=ids.tolist(),source_curve_weighted_RMS=norm.tolist(),
            source_half_frame_relative_RMS=None if half_relative is None else half_relative.tolist(),
            source_cubic_relative_RMS=None if degree_relative is None else degree_relative.tolist(),
            source_curve_fullaxis_minmax=[[float(v.min()),float(v.max())] for v in f])
        applied[i]=not audit["reasons"]
        audit["applied"]=bool(applied[i]);audits.append(audit)
    total=curves[applied].sum(0) if applied.any() else np.zeros((2,len(e)))
    return dict(v32_event_source_feature_log=curves,v32_event_applied=applied,
        v32_new_source_feature_log=total,
        diagnostics=dict(events=audits,accepted_count=int(applied.sum()),
            rejected_count=int((~applied).sum()),source_stability_limit=STABILITY_LIMIT,
            original_V31_accepted_indices=[i for i,r in enumerate(events) if r["applied"]],
            accepted_indices=np.flatnonzero(applied).tolist()))


def generate_all(e,pn,*,baseline,baseline_diagnostics,v31_diagnostics):
    e=np.asarray(e,dtype=float);pn=np.asarray(pn,dtype=float)
    if e.ndim!=1 or len(e)<3 or not np.all(np.isfinite(e)) or not np.all(np.diff(e)>0):
        raise ValueError("Expected finite strictly increasing native E")
    if pn.shape!=(2,len(e)) or not np.all(np.isfinite(pn)&(pn>0)):
        raise ValueError("Expected two finite positive native PN spectra")
    if not np.array_equal(e,baseline["energy"]):raise ValueError("Parent energy differs")
    source=np.asarray(baseline["source_feature_log"])
    weights=np.asarray(baseline["source_weights"])
    r=np.asarray(baseline["residual_intervention_native_log"])
    q=np.asarray(baseline["v29_alignment_q_log"])
    parent=np.asarray(baseline["background"])
    C=np.log(pn)-source
    if not np.allclose(C[0]-C[1],r+q,rtol=0,atol=6e-15):
        raise ValueError("Parent corrected source differential does not close")
    if not np.allclose(parent,np.exp(C.mean(0)+(weights[0]-.5)*r),rtol=3e-14,atol=0):
        raise ValueError("Parent is not the frozen aligned construction")
    if not np.allclose(weights.sum(0),1,rtol=0,atol=4e-15):
        raise ValueError("Frozen source weights do not sum to one")
    stage=frozen_source_stage(e,v31_diagnostics)
    F=stage["v32_new_source_feature_log"]
    delta=-np.sum(weights*F,axis=0)
    bg=parent*np.exp(delta) if stage["diagnostics"]["accepted_count"] else parent.copy()
    effective_source=source+F
    corrected=C-F
    effective_r=r-(F[0]-F[1])
    result={k:v.copy() for k,v in baseline.items() if isinstance(v,np.ndarray)}
    result.update({k:v.copy() for k,v in stage.items() if isinstance(v,np.ndarray)})
    result.update(energy=e.copy(),background=bg,baseline_background=parent.copy(),
        v32_parent_background=parent.copy(),v32_parent_corrected_source_log=C,
        v32_effective_source_feature_log=effective_source,
        v32_effective_corrected_source_log=corrected,
        v32_effective_residual_log=effective_r,v32_alignment_q_log=q.copy(),
        v32_background_delta_log=delta,
        v32_weighted_source_cleanup_log=-delta,
        repair_mask=abs(np.log(bg/np.sqrt(pn[0]*pn[1])))>1e-12,seams=np.array([],float))
    diag=copy.deepcopy(baseline_diagnostics)
    diag.update(family=FAMILIES[0],BN_used=False,historical_BN_used=False,sample_used=False,
        source_fits_frozen=True,source_assignments_frozen=True,weights_frozen=True,
        source_weights_interpretation="Frozen V29 aligned weights, now acting on v32_effective_residual_log; original residual fields remain unchanged.",
        v32=dict(**stage["diagnostics"],construction="B29*exp(-sum_s(w_s*F_s)); Ceff=C-F; r_eff=r-(F_P-F_N); q unchanged",
            bookkeeping="Old source_feature_log/r/component fields stay frozen; v32_effective_* fields define the new background.",
            all_parameters_reused_no_refit=True,
            conditional_frame_check_not_independent_generalization=True,
            source_cleanup_fullaxis_log_RMS=[float(np.sqrt(np.mean(v*v))) for v in F],
            source_cleanup_fullaxis_log_max=[float(np.max(abs(v))) for v in F],
            background_delta_log_RMS=float(np.sqrt(np.mean(delta*delta))),
            background_delta_log_max=float(np.max(abs(delta))),
            sample_to_background_gain_range=[float(np.exp(-delta).min()),float(np.exp(-delta).max())],
            effective_q_reconstruction_max=float(np.max(abs(corrected[0]-corrected[1]-effective_r-q)))))
    result["diagnostics"]=diag
    return {FAMILIES[0]:result}


def generate(e,pn,*,baseline,baseline_diagnostics,v31_diagnostics,family=FAMILIES[0]):
    if family not in FAMILIES:raise ValueError("Unknown frozen family")
    return generate_all(e,pn,baseline=baseline,baseline_diagnostics=baseline_diagnostics,
        v31_diagnostics=v31_diagnostics)[family]
