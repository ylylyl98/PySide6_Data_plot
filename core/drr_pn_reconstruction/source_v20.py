"""Frozen V20: one PN-only local residual-event pass after a supplied V12 result."""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "e815528517fed085a04cfc92ac6c2481c10100be7e7cc8d64bda35599327d9c9"
import copy
import hashlib
import json
import numpy as np
from .source_runtime import least_squares
from scipy.signal import find_peaks

from . import source_v7 as v7
Q = v7._quad
features_function = v7._v6._features
FAMILIES = ("residual_pass_fixed_weights", "residual_pass_updated_weights")
SETTINGS = dict(edge_height=2., edge_prominence=1.5, contrast_log=.006,
    contrast_radius_ev=.015, fit_radius_ev=.025, centre_radius_ev=.006,
    width_bounds_ev=[.0006,.008], starts_width_ev=[.0015,.003,.006],
    existing_applied_centre_exclusion_ev=.008, event_separation_ev=.012,
    max_events=8, minimum_feature_log=.005, minimum_rms_improvement=.25,
    minimum_template_transfer=.05, minimum_filtered_peak_log=.003,
    maximum_column_condition=1e4, attribution_bands_ev=[.025,.035],
    old_usable_weight=4., other_weight=1., exposure_scale_log=.015)

def _sha_array(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))

def _condition(a):
    norms=np.linalg.norm(a,axis=0)
    normalized=np.divide(a,norms[None,:],out=np.zeros_like(a),where=norms[None,:]>1e-16)
    singular=np.linalg.svd(normalized,compute_uv=False)
    condition=float(singular[0]/singular[-1]) if singular[-1]>1e-16 else None
    return condition,norms,singular

def detect_events(e,residual,old_records):
    smooth=Q(e,residual,.004)[0];edge=np.gradient(smooth,e)
    indices,props=find_peaks(abs(edge),height=2.,prominence=1.5)
    candidates=[]
    for j,i in enumerate(indices):
        local=abs(e-e[i])<=.015
        contrast=float(np.ptp(np.quantile(smooth[local],[.05,.95])))
        reasons=[]
        if contrast<.006:reasons.append("insufficient_contrast")
        if min(e[i]-e[0],e[-1]-e[i])<.025:reasons.append("endpoint_support")
        if any(r["applied"] and abs(e[i]-r["parameters"][0])<.008 for r in old_records):
            reasons.append("existing_applied_core_exclusion")
        candidates.append(dict(index=int(i),energy_ev=float(e[i]),edge_per_ev=float(edge[i]),
            prominence_per_ev=float(props["prominences"][j]),contrast_log=contrast,
            rejected_reasons=reasons))
    selected=[]
    for r in sorted(candidates,key=lambda v:-abs(v["edge_per_ev"])):
        if r["rejected_reasons"]:continue
        if any(abs(r["energy_ev"]-p["energy_ev"])<.012 for p in selected):
            r["rejected_reasons"].append("stronger_nearby_edge");continue
        if len(selected)>=8:r["rejected_reasons"].append("candidate_cap");continue
        selected.append(r)
    return sorted(selected,key=lambda r:r["energy_ev"]),candidates,smooth,edge

def _fit_event(e,residual,event):
    center=event["energy_ev"];mask=abs(e-center)<=.025
    ex=e[mask];y=residual[mask];x=(ex-center)/.025
    design=np.c_[np.ones(len(ex)),x,x*x]
    cell=np.gradient(ex);weight=np.sqrt(len(ex)*cell/cell.sum())
    nuisance0=np.linalg.lstsq(weight[:,None]*design,weight*y,rcond=None)[0]
    lo=np.array([center-.006,.0006,-3.,-6.,-.5,-.5,-.5])
    hi=np.array([center+.006,.008,3.,3.,.5,.5,.5])
    scale=np.array([.006,.003,.03,.03,.01,.01,.01])
    taper=np.ones((1,len(ex)))
    def fun(p):
        f,j=features_function(ex,p[:4],taper)
        return weight*(f[0]+design@p[4:]-y)
    def jac(p):
        f,j=features_function(ex,p[:4],taper)
        return weight[:,None]*np.c_[j[0],design]
    fits=[]
    for width in (.0015,.003,.006):
        start=np.r_[center,width,.03*np.sign(event["edge_per_ev"]),-.03,nuisance0]
        start=np.clip(start,lo+1e-10,hi-1e-10)
        fits.append(least_squares(fun,start,jac=jac,bounds=(lo,hi),x_scale=scale,
            loss="soft_l1",f_scale=.003,max_nfev=800,ftol=1e-9,xtol=1e-9,gtol=1e-9))
    best_index=min(range(3),key=lambda k:(fits[k].cost,np.linalg.norm(fits[k].x)))
    best=fits[best_index]
    full,_=features_function(e,best.x[:4],np.ones((1,len(e))))
    nuisance=design@best.x[4:]
    after=y-full[0,mask]-nuisance;before=y-design@nuisance0
    reduction=1-rms(after)/max(rms(before),1e-30)
    distances=np.minimum(best.x-lo,hi-best.x)/(hi-lo)
    bound=np.flatnonzero(distances<=.01).tolist()
    condition,norms,singular=_condition(jac(best.x))
    reasons=[]
    if not best.success:reasons.append("optimizer_not_converged")
    if bound:reasons.append("parameter_boundary")
    if condition is None or condition>1e4:reasons.append("ill_conditioned")
    if np.max(abs(full))<.005:reasons.append("weak_feature")
    if reduction<.25:reasons.append("insufficient_local_rms_improvement")
    report=dict(**copy.deepcopy(event),parameters=best.x[:4].tolist(),
        nuisance_coefficients=best.x[4:].tolist(),fit_window_ev=[float(ex[0]),float(ex[-1])],
        fit_point_count=int(mask.sum()),fit_usable=not reasons,fit_rejection_reasons=reasons,
        local_rms_before=rms(before),local_rms_after=rms(after),rms_improvement_fraction=reduction,
        maximum_feature_log=float(np.max(abs(full))),normalized_jacobian_condition=condition,
        jacobian_column_norms=norms.tolist(),jacobian_normalized_singular_values=singular.tolist(),
        bound_hits=bound,bound_distances=distances.tolist(),selected_start=best_index,
        starts=[dict(success=bool(f.success),status=int(f.status),cost=float(2*f.cost),
            optimality=float(f.optimality),nfev=int(f.nfev),parameters=f.x.tolist(),
            active_bounds=f.active_mask.tolist()) for f in fits])
    return full[0],mask,nuisance,report

def _prepare_source_targets(e,corrected_logs):
    """Compute the unchanged source bands once for a stage's fixed log input."""
    q4source=Q(e,corrected_logs,.004)
    return tuple(q4source-Q(e,corrected_logs,width) for width in (.025,.035))


def _attribute_event(e,corrected_logs,feature,mask,usable,*,prepared_targets=None):
    # Keep the original standalone path and arithmetic. Stage callers reuse
    # only targets whose energy/log input is identical for every event.
    if prepared_targets is None:
        q4source=Q(e,corrected_logs,.004)
    q4f=Q(e,feature,.004)[0]
    x=(e[mask]-np.mean(e[mask]))/.025
    reports=[];classes=[];coefficients=[];filtered_features=[];source_targets=[]
    for band,width in enumerate((.025,.035)):
        target=(q4source-Q(e,corrected_logs,width) if prepared_targets is None
                else prepared_targets[band])
        filtered=q4f-Q(e,feature,width)[0]
        transfer=float(np.linalg.norm(filtered[mask])/max(np.linalg.norm(q4f[mask]),1e-30))
        peak=float(np.max(abs(filtered[mask])))
        cell=np.gradient(e[mask]);weightbase=len(cell)*cell/cell.sum()
        coeff=[];source_reports=[];valid=True
        for source in range(2):
            weights=weightbase*(1+3*usable[source,mask].astype(float))
            design=np.c_[filtered[mask],np.ones(mask.sum()),x]
            weighted=design*np.sqrt(weights)[:,None];response=target[source,mask]*np.sqrt(weights)
            condition,norms,singular=_condition(weighted)
            pars=np.linalg.lstsq(weighted,response,rcond=None)[0]
            good=condition is not None and condition<=1e4
            valid&=good;coeff.append(float(pars[0]))
            source_reports.append(dict(source=source,coefficients=pars.tolist(),
                normalized_condition=condition,column_norms=norms.tolist(),
                normalized_singular_values=singular.tolist(),valid=good,
                fit_rms=rms(target[source,mask]-design@pars),high_weight_points=int(usable[source,mask].sum())))
        cp,cn=coeff
        label=-1
        eligible=valid and transfer>=.05 and peak>=.003
        if eligible:
            if cp>=.6 and abs(cn)<=.5*abs(cp):label=0
            elif cn<=-.6 and abs(cp)<=.5*abs(cn):label=1
        reports.append(dict(width_ev=width,transfer=transfer,peak_log=peak,eligible=eligible,
            source_fits=source_reports,classification=label))
        classes.append(label);coefficients.append(coeff);filtered_features.append(filtered);source_targets.append(target)
    agreed=classes[0]>=0 and classes[0]==classes[1]
    return (classes[0] if agreed else -1),reports,np.array(coefficients),np.array(filtered_features),np.array(source_targets)

def fit_residual_stage(e,pn,*,baseline,baseline_diagnostics):
    e=np.asarray(e,dtype=float);pn=np.asarray(pn,dtype=float)
    if e.ndim!=1 or pn.shape!=(2,len(e)) or len(e)<20 or not np.all(np.diff(e)>0):
        raise ValueError("Expected ascending native energy and P/N shape (2,E)")
    if not np.all(np.isfinite(pn)&(pn>0)):raise ValueError("P/N must be finite positive")
    np.testing.assert_array_equal(e,baseline["energy"])
    old_records=baseline_diagnostics["components"]
    old_residual=np.asarray(baseline["residual_intervention_native_log"])
    corrected_logs=np.log(pn)-baseline["source_feature_log"]
    events,detected,smooth,edge=detect_events(e,old_residual,old_records)
    records=[];features=[];masks=[];nuisances=[];coeffs=[];bands=[];targets=[]
    prepared_targets=_prepare_source_targets(e,corrected_logs) if events else None
    for event in events:
        feature,mask,nuisance,report=_fit_event(e,old_residual,event)
        source,attribution,coef,band,target=_attribute_event(e,corrected_logs,feature,mask,
            baseline["source_usable_masks"],prepared_targets=prepared_targets)
        applied=report["fit_usable"] and source>=0
        report.update(source=source if applied else -1,applied=bool(applied),
            source_projection=attribution,source_projection_coefficients=coef.tolist(),
            attribution_reason="matched_two_band_source" if applied else "fit_or_source_ambiguous",
            residual_pass=True,flank_calibration=dict(used=False))
        full_nuisance=np.full(len(e),np.nan);full_nuisance[mask]=nuisance
        records.append(report);features.append(feature);masks.append(mask);nuisances.append(full_nuisance)
        coeffs.append(coef);bands.append(band);targets.append(target)
    count=len(events);n=len(e)
    f=np.asarray(features).reshape(count,n)
    source_new=np.zeros_like(pn);differential_new=np.zeros(n);exposure=baseline["residual_source_exposure_log_squared"].copy()
    for feature,row in zip(f,records):
        if not row["applied"]:continue
        source=row["source"];source_new[source]+=(1 if source==0 else -1)*feature
        differential_new+=feature
        c,w,unused,unused2=row["parameters"]
        exposure[source]+=np.max(abs(feature))**2/(1+((e-c)/w)**2)
    reliability=1/(1+exposure/.015**2)
    updated_weights=reliability/reliability.sum(0,keepdims=True)
    return dict(e=e,pn=pn,baseline=baseline,baseline_diagnostics=baseline_diagnostics,
        detection=detected,smooth=smooth,edge=edge,records=records,features=f,
        fit_masks=np.asarray(masks,dtype=bool).reshape(count,n),
        nuisances=np.asarray(nuisances).reshape(count,n),
        coefficients=np.asarray(coeffs).reshape(count,2,2),
        band_features=np.asarray(bands).reshape(count,2,n),
        source_targets=np.asarray(targets).reshape(count,2,2,n),
        new_source_features=source_new,new_differential_feature=differential_new,
        new_residual=old_residual-differential_new,updated_exposure=exposure,updated_weights=updated_weights)

def render(bundle,*,family):
    if family not in FAMILIES:raise ValueError("Unknown family")
    old=bundle["baseline"];e=bundle["e"];pn=bundle["pn"];n=len(e)
    result={k:v.copy() for k,v in old.items() if isinstance(v,np.ndarray) and k not in ("diagnostics_json","family")}
    weights=(old["residual_source_weights"] if family==FAMILIES[0] else bundle["updated_weights"]).copy()
    delta=(weights[0]-.5)*bundle["new_residual"]
    log_change=-bundle["new_source_features"].mean(0)+delta-old["residual_intervention_delta_log"]
    new_background=old["background"]*np.exp(log_change)
    if not any(r["applied"] for r in bundle["records"]):
        new_background=old["background"].copy();log_change=np.zeros(n)
    source_features=old["source_feature_log"]+bundle["new_source_features"]
    corrected=pn*np.exp(-source_features)
    binary=old["v11_binary_background"]*np.exp(-bundle["new_source_features"].mean(0))
    np.testing.assert_allclose(new_background,binary*np.exp(delta),rtol=2e-14,atol=0)
    np.testing.assert_allclose(pn,corrected*np.exp(source_features),rtol=2e-15,atol=0)
    oldcount=len(bundle["baseline_diagnostics"]["components"])
    result.update(background=new_background,energy=e.copy(),corrected_sources=corrected,
        source_feature_log=source_features,pn_reconstruction=corrected*np.exp(source_features),
        conditional_full_source_feature_log=source_features.copy(),
        conditional_local_source_feature_log=source_features.copy(),
        source_weights=weights,weights=weights.copy(),residual_source_weights=weights.copy(),
        baseline_v12_background=old["background"].copy(),baseline_v11_binary_background=old["v11_binary_background"].copy(),
        v11_binary_background=binary,baseline_residual_delta_log=old["residual_intervention_delta_log"].copy(),
        baseline_residual_log=old["residual_intervention_native_log"].copy(),
        baseline_source_weights=old["residual_source_weights"].copy(),
        new_source_feature_log=bundle["new_source_features"].copy(),
        new_component_feature_log=bundle["features"].copy(),
        new_component_fit_mask=bundle["fit_masks"].copy(),
        new_component_nuisance_log=bundle["nuisances"].copy(),
        new_component_source_coefficients=bundle["coefficients"].copy(),
        new_component_band_features=bundle["band_features"].copy(),
        new_component_source_targets=bundle["source_targets"].copy(),
        new_component_source=np.array([r["source"] for r in bundle["records"]],dtype=int),
        new_component_applied=np.array([r["applied"] for r in bundle["records"]],dtype=bool),
        residual_intervention_native_log=bundle["new_residual"].copy(),
        residual_intervention_slow_log=Q(e,bundle["new_residual"],.035)[0],
        residual_intervention_selected_log=bundle["new_residual"].copy(),
        residual_intervention_delta_log=delta,
        residual_source_exposure_log_squared=(old["residual_source_exposure_log_squared"] if family==FAMILIES[0] else bundle["updated_exposure"]).copy(),
        differential_model_log=old["differential_model_log"]+bundle["new_differential_feature"],
        differential_residual_log=bundle["new_residual"].copy(),
        pass_detection_smooth_log=bundle["smooth"].copy(),pass_detection_edge_per_ev=bundle["edge"].copy(),
        log_change_from_v12=log_change,component_feature_log=np.r_[old["component_feature_log"],bundle["features"]],
        component_source=np.r_[old["component_source"],[r["source"] for r in bundle["records"]]],
        component_applied=np.r_[old["component_applied"],[r["applied"] for r in bundle["records"]]],
        component_tapers=np.r_[old["component_tapers"],np.ones_like(bundle["features"])])
    result["repair_log_delta"]=np.log(new_background/old["raw_geometric_mean"])
    result["repair_mask"]=abs(result["repair_log_delta"])>1e-12
    result["modification_mask"]=result["repair_mask"].copy();result["no_repair_mask"]=~result["repair_mask"]
    identity=dict(family=family,baseline_background_sha256=_sha_array(old["background"]),
        new_features_sha256=_sha_array(bundle["features"]),source_features_sha256=_sha_array(source_features),
        new_residual_sha256=_sha_array(bundle["new_residual"]),weights_sha256=_sha_array(weights))
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    diag=copy.deepcopy(bundle["baseline_diagnostics"])
    diag.update(family=family,settings=copy.deepcopy(SETTINGS),BN_used=False,historical_BN_used=False,
        sample_used=False,old_fit_refitted=False,old_source_decisions_changed=False,
        old_component_count=oldcount,components=copy.deepcopy(bundle["baseline_diagnostics"]["components"])+copy.deepcopy(bundle["records"]),
        residual_pass_detection=copy.deepcopy(bundle["detection"]),
        residual_pass_components=copy.deepcopy(bundle["records"]),
        residual_pass_applied_count=sum(r["applied"] for r in bundle["records"]),
        applied_component_count=sum(r["applied"] for r in bundle["baseline_diagnostics"]["components"]+bundle["records"]),
        construction=identity,construction_sha256=digest,model_sha256=digest,
        output_construction="old binary minus new source features, then exposure-weighted residual with applied new differential features removed",
        log_change_from_v12_rms=rms(log_change),log_change_from_v12_max=float(np.max(abs(log_change))),
        limitations="Old usable masks are weighting evidence only; common/anti-phase or unsupported source events remain unresolved. Full tails can change distant baseline. No sample or BN quality is certified.",
        inherited_variable_length_attribution_arrays="Existing V11 arrays describe old components only; new_component_* arrays describe this added pass.")
    result["construction_sha256"]=np.array(digest);result["model_sha256"]=np.array(digest)
    result["diagnostics"]=diag
    return result

def generate_all(e,pn,*,baseline,baseline_diagnostics):
    bundle=fit_residual_stage(e,pn,baseline=baseline,baseline_diagnostics=baseline_diagnostics)
    return {f:render(bundle,family=f) for f in FAMILIES}

def generate(e,pn,*,baseline,baseline_diagnostics,family=FAMILIES[0]):
    return render(fit_residual_stage(e,pn,baseline=baseline,baseline_diagnostics=baseline_diagnostics),family=family)
