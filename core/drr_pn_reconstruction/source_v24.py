"""V24: width-aware residual detection, frozen V20 fitting, post-fit de-duplication."""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "606fc56d6d35b8dcbb87a87936223d42ed7556a9118ab45b7a5e4dcb6761d9f4"
import copy,hashlib,json
import numpy as np
from scipy.signal import find_peaks

from . import source_v20 as v20
from . import source_v22 as v22
_sha_array=v20._sha_array
FAMILIES=("narrow_pass_fixed_weights","narrow_pass_post_cost_weights")
SETTINGS=dict(broad_width_boundary_ev=.010,narrow_protection_min_ev=.002,
    narrow_protection_max_ev=.008,narrow_protection_width_factor=2.,
    max_edges_to_fit=16,pre_fit_separation_ev=None,duplicate_min_distance_ev=.0015,
    duplicate_width_sum_factor=.5,fit_and_attribution="unchanged V20",
    source_conflicts_unresolved=True,old_refit=False,wide_joint_refit=False)

def detect(e,residual,records):
    smooth=v20.Q(e,residual,.004)[0];edge=np.gradient(smooth,e)
    indices,props=find_peaks(abs(edge),height=2.,prominence=1.5)
    candidates=[]
    for k,i in enumerate(indices):
        contrast=float(np.ptp(np.quantile(smooth[abs(e-e[i])<=.015],[.05,.95])))
        reasons=[];protected_by=[]
        if contrast<.006:reasons.append("insufficient_contrast")
        if min(e[i]-e[0],e[-1]-e[i])<.025:reasons.append("endpoint_support")
        for j,r in enumerate(records):
            if not r["applied"]:continue
            c,w=r["parameters"][:2]
            if w>=.010:continue
            radius=min(.008,max(.002,2*w))
            if abs(e[i]-c)<radius:protected_by.append(dict(component=j,center=c,width=w,radius=radius))
        if protected_by:reasons.append("existing_applied_narrow_core_protection")
        candidates.append(dict(index=int(i),energy_ev=float(e[i]),edge_per_ev=float(edge[i]),
            prominence_per_ev=float(props["prominences"][k]),contrast_log=contrast,
            rejected_reasons=reasons,protected_by=protected_by))
    survivors=sorted([r for r in candidates if not r["rejected_reasons"]],
        key=lambda r:(-abs(r["edge_per_ev"]),r["index"]))
    for row in survivors[16:]:row["rejected_reasons"].append("candidate_cap_16")
    selected=sorted(survivors[:16],key=lambda r:r["index"])
    counts=dict(detected=len(candidates),after_noncap_filters=len(survivors),
        fitted=len(selected),cap_rejected=max(0,len(survivors)-16),
        rejection_counts={reason:sum(reason in r["rejected_reasons"] for r in candidates)
            for reason in ("insufficient_contrast","endpoint_support","existing_applied_narrow_core_protection","candidate_cap_16")})
    return selected,candidates,counts,smooth,edge

def _near(a,b):
    ca,wa=a["parameters"][:2];cb,wb=b["parameters"][:2]
    return abs(ca-cb)<=max(.0015,.5*(wa+wb))

def deduplicate(records,old_records):
    records=copy.deepcopy(records);groups=[]
    for row in records:
        row["pre_dedup_source"]=row["source"];row["pre_dedup_applied"]=row["applied"]
        row["post_fit_rejection_reasons"]=[]
        if not row["applied"]:continue
        matched=[j for j,r in enumerate(old_records) if r["applied"] and r["parameters"][1]<.010 and _near(row,r)]
        if matched:
            conflict=any(old_records[j]["source"]!=row["source"] for j in matched)
            row["post_fit_rejection_reasons"].append("old_narrow_source_conflict" if conflict else "duplicate_of_old_narrow")
            row["matching_old_components"]=matched
            row["applied"]=False;row["source"]=-1
    pending=set(i for i,r in enumerate(records) if r["applied"])
    while pending:
        group={min(pending)};queue=list(group);pending-=group
        while queue:
            i=queue.pop()
            linked={j for j in pending if _near(records[i],records[j])}
            group|=linked;queue.extend(sorted(linked));pending-=linked
        ids=sorted(group);sources={records[i]["source"] for i in ids}
        if len(sources)>1:
            winner=None;reason="new_event_source_conflict"
            for i in ids:
                records[i]["applied"]=False;records[i]["source"]=-1
                records[i]["post_fit_rejection_reasons"].append(reason)
        else:
            winner=min(ids,key=lambda i:(-records[i]["rms_improvement_fraction"],
                -abs(records[i]["edge_per_ev"]),records[i]["index"]))
            reason="single_or_same_source_duplicate"
            for i in ids:
                if i!=winner:
                    records[i]["applied"]=False;records[i]["source"]=-1
                    records[i]["post_fit_rejection_reasons"].append("duplicate_of_new_event")
                    records[i]["duplicate_winner"]=winner
        groups.append(dict(members=ids,source_hypotheses=sorted(sources),winner=winner,reason=reason))
    return records,groups

def fit_residual_stage(e,pn,*,baseline,baseline_diagnostics):
    e=np.asarray(e,dtype=float);pn=np.asarray(pn,dtype=float)
    np.testing.assert_array_equal(e,baseline["energy"])
    if e.ndim!=1 or len(e)<20 or not np.all(np.isfinite(e)) or pn.shape!=(2,len(e)) or not np.all(np.diff(e)>0) or not np.all(np.isfinite(pn)&(pn>0)):
        raise ValueError("Expected ascending native E and positive PN")
    residual=baseline["residual_intervention_native_log"]
    corrected=np.log(pn)-baseline["source_feature_log"]
    events,candidates,counts,smooth,edge=detect(e,residual,baseline_diagnostics["components"])
    records=[];features=[];masks=[];nuisances=[];coeffs=[];bands=[];targets=[]
    prepared_targets=v20._prepare_source_targets(e,corrected) if events else None
    for event in events:
        f,mask,nuisance,row=v20._fit_event(e,residual,event)
        source,attribution,coef,band,target=v20._attribute_event(e,corrected,f,mask,
            baseline["source_usable_masks"],prepared_targets=prepared_targets)
        applied=bool(row["fit_usable"] and source>=0)
        row.update(source=source if applied else -1,applied=applied,source_projection=attribution,
            source_projection_coefficients=coef.tolist(),attribution_reason="V20_matched_two_band_source" if applied else "fit_or_source_ambiguous",
            residual_pass=True,v24_event=True,flank_calibration=dict(used=False))
        nu=np.full(len(e),np.nan);nu[mask]=nuisance
        records.append(row);features.append(f);masks.append(mask);nuisances.append(nu)
        coeffs.append(coef);bands.append(band);targets.append(target)
    records,groups=deduplicate(records,baseline_diagnostics["components"])
    count=len(records);n=len(e)
    f=np.asarray(features).reshape(count,n)
    source_new=np.zeros_like(pn);differential_new=np.zeros(n)
    for component,row in zip(f,records):
        if not row["applied"]:continue
        source_new[row["source"]]+=(1 if row["source"]==0 else -1)*component
        differential_new+=component
    corrected_new=corrected-source_new;residual_new=residual-differential_new
    post=v22.post_correction_weights(e,corrected_new)
    return dict(e=e,pn=pn,baseline=baseline,baseline_diagnostics=baseline_diagnostics,
        old_corrected=corrected,corrected=corrected_new,residual=residual_new,
        new_source_features=source_new,new_differential=differential_new,post=post,
        detection=candidates,detection_counts=counts,smooth=smooth,edge=edge,records=records,
        groups=groups,features=f,masks=np.asarray(masks,dtype=bool).reshape(count,n),
        nuisances=np.asarray(nuisances).reshape(count,n),coefficients=np.asarray(coeffs).reshape(count,2,2),
        bands=np.asarray(bands).reshape(count,2,n),targets=np.asarray(targets).reshape(count,2,2,n))

def render(bundle,*,family):
    if family not in FAMILIES:raise ValueError("Unknown V24 family")
    old=bundle["baseline"];old_diag=bundle["baseline_diagnostics"];e=bundle["e"];pn=bundle["pn"]
    weights=(old["residual_source_weights"] if family==FAMILIES[0] else bundle["post"]["weights"]).copy()
    delta=(weights[0]-.5)*bundle["residual"]
    change=-bundle["new_source_features"].mean(0)+delta-old["residual_intervention_delta_log"]
    background=old["background"]*np.exp(change)
    if family==FAMILIES[0] and not any(r["applied"] for r in bundle["records"]):
        background=old["background"].copy();change=np.zeros_like(e)
    features_total=old["source_feature_log"]+bundle["new_source_features"]
    binary=old["v11_binary_background"]*np.exp(-bundle["new_source_features"].mean(0))
    q_old=(bundle["old_corrected"][0]-bundle["old_corrected"][1])-old["residual_intervention_native_log"]
    q_new=(bundle["corrected"][0]-bundle["corrected"][1])-bundle["residual"]
    np.testing.assert_allclose(q_old,q_new,rtol=0,atol=5e-15)
    np.testing.assert_allclose(background,binary*np.exp(delta),rtol=2e-14,atol=0)
    result={k:v.copy() for k,v in old.items() if isinstance(v,np.ndarray) and k not in ("family","diagnostics_json")}
    result.update(energy=e.copy(),background=background,baseline_v21_fixed_background=old["background"].copy(),
        baseline_residual_source_weights=old["residual_source_weights"].copy(),
        source_weights=weights,weights=weights.copy(),residual_source_weights=weights.copy(),
        corrected_sources=np.exp(bundle["corrected"]),source_feature_log=features_total,
        pn_reconstruction=np.exp(bundle["corrected"]+features_total),
        conditional_full_source_feature_log=features_total.copy(),conditional_local_source_feature_log=features_total.copy(),
        v11_binary_background=binary,residual_intervention_native_log=bundle["residual"].copy(),
        residual_intervention_selected_log=bundle["residual"].copy(),residual_intervention_delta_log=delta,
        residual_intervention_slow_log=v20.Q(e,bundle["residual"],.035)[0],
        differential_model_log=old["differential_model_log"]+bundle["new_differential"],
        differential_residual_log=bundle["residual"].copy(),
        new_source_feature_log=old["new_source_feature_log"]+bundle["new_source_features"],
        new_component_feature_log=np.r_[old["new_component_feature_log"],bundle["features"]],
        new_component_fit_mask=np.r_[old["new_component_fit_mask"],bundle["masks"]],
        new_component_nuisance_log=np.r_[old["new_component_nuisance_log"],bundle["nuisances"]],
        new_component_source_coefficients=np.r_[old["new_component_source_coefficients"],bundle["coefficients"]],
        new_component_band_features=np.r_[old["new_component_band_features"],bundle["bands"]],
        new_component_source_targets=np.r_[old["new_component_source_targets"],bundle["targets"]],
        new_component_source=np.r_[old["new_component_source"],np.array([r["source"] for r in bundle["records"]],dtype=int)],
        new_component_applied=np.r_[old["new_component_applied"],np.array([r["applied"] for r in bundle["records"]],dtype=bool)],
        component_feature_log=np.r_[old["component_feature_log"],bundle["features"]],
        component_source=np.r_[old["component_source"],np.array([r["source"] for r in bundle["records"]],dtype=int)],
        component_applied=np.r_[old["component_applied"],np.array([r["applied"] for r in bundle["records"]],dtype=bool)],
        component_tapers=np.r_[old["component_tapers"],np.ones_like(bundle["features"])],
        v24_new_source_feature_log=bundle["new_source_features"].copy(),
        v24_component_feature_log=bundle["features"].copy(),
        v24_component_applied=np.array([r["applied"] for r in bundle["records"]],dtype=bool),
        v24_component_source=np.array([r["source"] for r in bundle["records"]],dtype=int),
        v24_corrected_source_logs=bundle["corrected"].copy(),
        v24_old_alignment_q_log=q_old,v24_new_alignment_q_log=q_new,
        v24_detection_smooth_log=bundle["smooth"].copy(),v24_detection_edge_per_ev=bundle["edge"].copy(),
        v24_post_source_bands=bundle["post"]["bands"].copy(),v24_post_structure_cost=bundle["post"]["costs"].copy(),
        v24_post_band_weights=bundle["post"]["weights_by_band"].copy(),
        v24_post_direction_agreement=bundle["post"]["agreement"].copy(),
        v24_post_consensus_P_weight=bundle["post"]["consensus"].copy(),
        v24_post_clipped_P_weight=bundle["post"]["clipped"].copy(),
        v24_post_smoothed_P_weight=bundle["post"]["smoothed"].copy(),
        v24_log_change_from_v21_fixed=change)
    result["repair_log_delta"]=np.log(background/old["raw_geometric_mean"])
    result["repair_mask"]=abs(result["repair_log_delta"])>1e-12
    result["modification_mask"]=result["repair_mask"].copy();result["no_repair_mask"]=~result["repair_mask"]
    if "source_residual_log" in result:result["legacy_source_residual_log_not_used"]=result.pop("source_residual_log")
    identity=dict(family=family,baseline_sha256=_sha_array(old["background"]),
        new_features_sha256=_sha_array(bundle["features"]),source_sum_sha256=_sha_array(bundle["new_source_features"]),
        weights_sha256=_sha_array(weights),residual_sha256=_sha_array(bundle["residual"]),settings=SETTINGS)
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    diag=copy.deepcopy(old_diag)
    records=copy.deepcopy(bundle["records"])
    diag.update(family=family,v24_settings=copy.deepcopy(SETTINGS),v24_detection=copy.deepcopy(bundle["detection"]),
        v24_detection_counts=copy.deepcopy(bundle["detection_counts"]),v24_event_records=records,
        v24_duplicate_groups=copy.deepcopy(bundle["groups"]),v24_applied_count=sum(r["applied"] for r in records),
        components=copy.deepcopy(old_diag["components"])+records,
        residual_pass_components=copy.deepcopy(old_diag["residual_pass_components"])+records,
        residual_pass_applied_count=sum(r["applied"] for r in old_diag["residual_pass_components"]+records),
        applied_component_count=sum(r["applied"] for r in old_diag["components"]+records),
        construction=identity,construction_sha256=digest,model_sha256=digest,
        BN_used=False,historical_BN_used=False,sample_used=False,old_fit_refitted=False,wide_joint_refit_used=False,
        old_source_decisions_unchanged=True,v24_delta_rms=v20.rms(change),v24_delta_max=float(np.max(abs(change))),
        alignment_invariance_max=float(np.max(abs(q_new-q_old))),
        output_construction="Frozen V21 fixed baseline, new source features removed once and new differential features removed from residual; fixed or frozen V22 post-cost weights.",
        source_claim="Matched bands and local Fano fit remain conditional evidence, not a physical identifiability certificate.",
        inheritance_note="Old exposure arrays are frozen descriptors, not the adaptive cost weights; v24_post_* are current costs.",
        inherited_variable_length_attribution_arrays="Original V11 arrays describe original components only. new_component_* concatenate V20/V21 and V24 events; v24_component_* describe this pass alone.")
    result["construction_sha256"]=np.array(digest);result["model_sha256"]=np.array(digest);result["diagnostics"]=diag
    return result

def generate_all(e,pn,*,baseline,baseline_diagnostics):
    bundle=fit_residual_stage(e,pn,baseline=baseline,baseline_diagnostics=baseline_diagnostics)
    return {f:render(bundle,family=f) for f in FAMILIES}

def generate(e,pn,*,baseline,baseline_diagnostics,family=FAMILIES[0]):
    return render(fit_residual_stage(e,pn,baseline=baseline,baseline_diagnostics=baseline_diagnostics),family=family)
