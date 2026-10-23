"""V21: frozen V20 event fits, alternate coefficient-space source hypothesis."""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "eca19de513949d7236d166723b341c2d8e6ecc8a8e5525c3632140e4bd60dd10"
import copy
import hashlib
import json
import numpy as np

from . import source_v20 as v20
_sha_array=v20._sha_array
FAMILIES=("source_hypothesis_fixed_weights","source_hypothesis_updated_weights")
SETTINGS=dict(maximum_cost_ratio=.5,maximum_differential_closure_error=.15,
    scope="only fit-usable two-band-eligible V20 strict-unknown events",
    strict_accepted_unchanged=True,all_fit_arrays_unchanged=True,
    costs_are_likelihoods=False,BN_used=False,sample_used=False)

def reconsider_sources(old_records):
    records=copy.deepcopy(old_records);reports=[]
    for k,row in enumerate(records):
        cp_cn=np.array(row["source_projection_coefficients"],dtype=float)
        jp=(cp_cn[:,0]-1)**2+cp_cn[:,1]**2
        jn=cp_cn[:,0]**2+(cp_cn[:,1]+1)**2
        winners=np.where(jp<jn,0,np.where(jn<jp,1,-1))
        ratio_pass=np.minimum(jp,jn)<=.5*np.maximum(jp,jn)
        closure=abs(cp_cn[:,0]-cp_cn[:,1]-1)
        closure_pass=closure<=.15
        eligible=bool(row["fit_usable"] and all(b["eligible"] for b in row["source_projection"]))
        considered=eligible and not row["applied"] and row["source"]==-1
        accepted=bool(considered and np.all(ratio_pass)&np.all(closure_pass)
            and winners[0]>=0 and winners[0]==winners[1])
        report=dict(component=k,strict_source=int(row["source"]),strict_applied=bool(row["applied"]),
            considered=considered,eligible=eligible,Jp=jp.tolist(),Jn=jn.tolist(),
            band_winners=winners.tolist(),cost_ratio_pass=ratio_pass.tolist(),
            differential_closure_error=closure.tolist(),closure_pass=closure_pass.tolist(),
            newly_assigned=accepted,source_after=int(winners[0]) if accepted else int(row["source"]))
        if accepted:
            row["source"]=int(winners[0]);row["applied"]=True
            row["attribution_reason"]="two_band_relative_source_hypothesis"
        row["v21_source_hypothesis"]=report
        reports.append(report)
    return records,reports

def build_bundle(e,pn,*,baseline_v12,baseline_diagnostics,v20_result,v20_diagnostics):
    np.testing.assert_array_equal(e,baseline_v12["energy"])
    np.testing.assert_array_equal(e,v20_result["energy"])
    features=v20_result["new_component_feature_log"]
    records,reports=reconsider_sources(v20_diagnostics["residual_pass_components"])
    source_new=np.zeros_like(pn);differential_new=np.zeros_like(e)
    exposure=baseline_v12["residual_source_exposure_log_squared"].copy()
    for feature,row in zip(features,records):
        if not row["applied"]:continue
        source=row["source"];source_new[source]+=(1 if source==0 else -1)*feature
        differential_new+=feature
        center,width=row["parameters"][:2]
        exposure[source]+=np.max(abs(feature))**2/(1+((e-center)/width)**2)
    reliability=1/(1+exposure/.015**2);weights=reliability/reliability.sum(0,keepdims=True)
    return dict(e=e,pn=pn,baseline=baseline_v12,baseline_diagnostics=baseline_diagnostics,
        detection=copy.deepcopy(v20_diagnostics["residual_pass_detection"]),
        smooth=v20_result["pass_detection_smooth_log"],edge=v20_result["pass_detection_edge_per_ev"],
        records=records,features=features,fit_masks=v20_result["new_component_fit_mask"],
        nuisances=v20_result["new_component_nuisance_log"],coefficients=v20_result["new_component_source_coefficients"],
        band_features=v20_result["new_component_band_features"],source_targets=v20_result["new_component_source_targets"],
        new_source_features=source_new,new_differential_feature=differential_new,
        new_residual=baseline_v12["residual_intervention_native_log"]-differential_new,
        updated_exposure=exposure,updated_weights=weights,hypothesis_reports=reports)

def render(bundle,*,family):
    if family not in FAMILIES:raise ValueError("Unknown V21 family")
    result=v20.render(bundle,family=v20.FAMILIES[FAMILIES.index(family)])
    reports=bundle["hypothesis_reports"]
    source=np.array([r["source"] for r in bundle["records"]],dtype=int)
    parent_construction=str(result["construction_sha256"])
    identity=dict(family=family,parent_construction_sha256=parent_construction,
        source_sha256=_sha_array(source),rule=SETTINGS)
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    result["construction_sha256"]=np.array(digest);result["model_sha256"]=np.array(digest)
    result["v21_newly_assigned"]=np.array([r["newly_assigned"] for r in reports],dtype=bool)
    result["v21_hypothesis_costs"]=np.array([[r["Jp"],r["Jn"]] for r in reports])
    result["diagnostics"].update(family=family,v21_settings=copy.deepcopy(SETTINGS),
        v21_source_hypotheses=copy.deepcopy(reports),
        v21_newly_assigned_count=sum(r["newly_assigned"] for r in reports),
        all_v20_fits_and_coefficients_frozen=True,old_strict_assignments_unchanged=True,
        construction=identity,construction_sha256=digest,model_sha256=digest,
        hypothesis_warning="Coefficient-space source distances are not physical source proof or likelihoods; common texture projection can bias both bands.")
    return result

def generate_all(e,pn,*,baseline_v12,baseline_diagnostics,v20_result,v20_diagnostics):
    bundle=build_bundle(e,pn,baseline_v12=baseline_v12,baseline_diagnostics=baseline_diagnostics,
        v20_result=v20_result,v20_diagnostics=v20_diagnostics)
    return {family:render(bundle,family=family) for family in FAMILIES}

def generate(e,pn,*,baseline_v12,baseline_diagnostics,v20_result,v20_diagnostics,family=FAMILIES[0]):
    return render(build_bundle(e,pn,baseline_v12=baseline_v12,baseline_diagnostics=baseline_diagnostics,
        v20_result=v20_result,v20_diagnostics=v20_diagnostics),family=family)
