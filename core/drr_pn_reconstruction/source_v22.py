"""V22: fixed corrected-source curves with post-correction structure-cost weights."""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "fb54ba9198a93df781a4b0ab6061633a2a8ccabf92558838cab9d3dfd5250ba8"
import copy
import hashlib
import json
import numpy as np

from . import source_v7 as v7
Q=v7._quad
FAMILIES=("old_aligned_weights","equal_aligned_weights","post_correction_structure_weights")
SETTINGS=dict(band_widths_ev=[.025,.035],fine_width_ev=.004,gaussian_sigma_ev=.004,
    cost_regularization_log=.003,minimum_source_weight=.2,maximum_source_weight=.8,
    same_preference_in_both_bands=True,post_consensus_gaussian_smoothing=True,
    cost_interpretation="local structure cost; not noise, probability or certified featurelessness",
    direct_blend_is_diagnostic_only=True)

def _sha_array(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))

def gaussian_kernel(e,sigma=.004):
    e=np.asarray(e,dtype=float)
    cell=np.gradient(e)
    kernel=np.exp(-.5*((e[:,None]-e[None,:])/sigma)**2)*cell[None,:]
    kernel/=kernel.sum(1,keepdims=True)
    return kernel,cell

def post_correction_weights(e,corrected_logs):
    kernel,cell=gaussian_kernel(e)
    q4=Q(e,corrected_logs,.004)
    bands=np.array([q4-Q(e,corrected_logs,w) for w in (.025,.035)])
    costs=np.einsum("ij,bsj->bsi",kernel,bands**2)
    rho=1/(.003**2+costs)
    weights_by_band=rho/rho.sum(1,keepdims=True)
    preference=weights_by_band[:,0]-.5
    agreement=preference[0]*preference[1]>0
    consensus=np.where(agreement,weights_by_band[:,0].mean(0),.5)
    clipped=np.clip(consensus,.2,.8)
    smooth=kernel@clipped
    weights=np.array([smooth,1-smooth])
    return dict(weights=weights,bands=bands,costs=costs,weights_by_band=weights_by_band,
        band_preference=preference,agreement=agreement,consensus=consensus,clipped=clipped,
        smoothed=smooth,kernel_row_sums=kernel.sum(1),
        disagreement_influence=kernel@(~agreement).astype(float),cell=cell)

def build_bundle(e,pn,*,baseline,baseline_diagnostics):
    e=np.asarray(e,dtype=float);pn=np.asarray(pn,dtype=float)
    if e.ndim!=1 or len(e)<20 or pn.shape!=(2,len(e)) or not np.all(np.diff(e)>0):
        raise ValueError("Expected ascending native E and P/N shape (2,E)")
    if not np.all(np.isfinite(pn)&(pn>0)):raise ValueError("PN must be finite positive")
    np.testing.assert_array_equal(e,baseline["energy"])
    c=np.log(pn)-baseline["source_feature_log"]
    mean=c.mean(0);difference=c[0]-c[1]
    residual=baseline["residual_intervention_native_log"].copy()
    q=difference-residual
    old_count=baseline_diagnostics["old_component_count"]
    old_applied=baseline["component_applied"][:old_count].astype(bool)
    old_unassigned=baseline["component_feature_log"][:old_count][~old_applied].sum(0)
    nuisance=baseline["differential_nuisance_log"].copy()
    rounding=q-nuisance-old_unassigned
    new_unassigned=baseline["new_component_feature_log"][~baseline["new_component_applied"].astype(bool)].copy()
    new_sum=new_unassigned.sum(0)
    aligned=c+np.array([-.5,.5])[:,None]*q
    weights=baseline["residual_source_weights"].copy()
    reference_log=mean+(weights[0]-.5)*residual
    np.testing.assert_allclose(np.log(baseline["background"]),reference_log,rtol=0,atol=8e-14)
    np.testing.assert_allclose(aligned[0]-aligned[1],residual,rtol=0,atol=5e-15)
    np.testing.assert_allclose(aligned.mean(0),mean,rtol=0,atol=5e-15)
    return dict(e=e,pn=pn,baseline=baseline,baseline_diagnostics=baseline_diagnostics,
        corrected=c,mean=mean,difference=difference,residual=residual,alignment=q,aligned=aligned,
        old_nuisance=nuisance,old_unassigned=old_unassigned,rounding=rounding,
        new_unassigned=new_unassigned,new_unassigned_sum=new_sum,
        residual_minus_new_accounting=residual-new_sum,
        old_weights=weights,post=post_correction_weights(e,c))

def render(bundle,*,family):
    if family not in FAMILIES:raise ValueError("Unknown V22 family")
    old=bundle["baseline"];e=bundle["e"];post=bundle["post"]
    if family==FAMILIES[0]:weights=bundle["old_weights"].copy()
    elif family==FAMILIES[1]:weights=np.full((2,len(e)),.5)
    else:weights=post["weights"].copy()
    change=(weights[0]-bundle["old_weights"][0])*bundle["residual"]
    background=old["background"].copy() if family==FAMILIES[0] else old["background"]*np.exp(change)
    aligned_log=bundle["mean"]+(weights[0]-.5)*bundle["residual"]
    direct_log=bundle["mean"]+(weights[0]-.5)*bundle["difference"]
    direct_minus_aligned=(weights[0]-.5)*bundle["alignment"]
    np.testing.assert_allclose(direct_log-aligned_log,direct_minus_aligned,rtol=0,atol=5e-15)
    result={k:v.copy() for k,v in old.items() if isinstance(v,np.ndarray) and k not in ("family","diagnostics_json")}
    if "source_residual_log" in result:
        result["legacy_source_residual_log_not_used"]=result.pop("source_residual_log")
    result.update(energy=e.copy(),background=background,baseline_v21_background=old["background"].copy(),
        source_weights=weights,weights=weights.copy(),residual_source_weights=weights.copy(),
        baseline_residual_source_weights=bundle["old_weights"].copy(),
        residual_intervention_delta_log=(weights[0]-.5)*bundle["residual"],
        post_true_corrected_source_logs=bundle["corrected"].copy(),
        post_corrected_mean_log=bundle["mean"].copy(),post_corrected_difference_log=bundle["difference"].copy(),
        post_aligned_source_logs=bundle["aligned"].copy(),post_alignment_q_log=bundle["alignment"].copy(),
        post_old_nuisance_log=bundle["old_nuisance"].copy(),
        post_old_unassigned_component_sum_log=bundle["old_unassigned"].copy(),
        post_alignment_rounding_log=bundle["rounding"].copy(),
        post_new_unassigned_component_log=bundle["new_unassigned"].copy(),
        post_new_unassigned_sum_accounting_only_log=bundle["new_unassigned_sum"].copy(),
        post_residual_minus_new_unassigned_accounting_only_log=bundle["residual_minus_new_accounting"].copy(),
        post_residual_log=bundle["residual"].copy(),
        post_source_band_log=post["bands"].copy(),
        post_common_band_log=post["bands"].mean(1),
        post_differential_band_log=post["bands"][:,0]-post["bands"][:,1],
        post_structure_cost=post["costs"].copy(),
        post_weights_by_band=post["weights_by_band"].copy(),
        post_band_preference=post["band_preference"].copy(),
        post_direction_agreement=post["agreement"].copy(),
        post_consensus_P_weight=post["consensus"].copy(),
        post_clipped_P_weight=post["clipped"].copy(),
        post_smoothed_P_weight=post["smoothed"].copy(),
        post_smoothing_disagreement_influence=post["disagreement_influence"].copy(),
        post_kernel_row_sums=post["kernel_row_sums"].copy(),post_cell_width_ev=post["cell"].copy(),
        post_aligned_log_background=aligned_log,
        diagnostic_direct_corrected_log_background=direct_log,
        diagnostic_direct_corrected_background=np.exp(direct_log),
        diagnostic_direct_minus_aligned_log=direct_minus_aligned,
        post_log_change_from_v21=change)
    result["repair_log_delta"]=np.log(background/old["raw_geometric_mean"])
    result["repair_mask"]=abs(result["repair_log_delta"])>1e-12
    result["modification_mask"]=result["repair_mask"].copy();result["no_repair_mask"]=~result["repair_mask"]
    identity=dict(family=family,baseline_background_sha256=_sha_array(old["background"]),
        source_corrected_log_sha256=_sha_array(bundle["corrected"]),residual_sha256=_sha_array(bundle["residual"]),
        weights_sha256=_sha_array(weights),post_cost_sha256=_sha_array(post["costs"]),settings=SETTINGS)
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    diagnostic=copy.deepcopy(bundle["baseline_diagnostics"])
    diagnostic.update(family=family,v22_settings=copy.deepcopy(SETTINGS),BN_used=False,
        historical_BN_used=False,sample_used=False,fit_refitted=False,source_decisions_changed=False,
        construction=identity,construction_sha256=digest,model_sha256=digest,
        v22_log_change_rms=rms(change),v22_log_change_max=float(np.max(abs(change))),
        corrected_difference_rms=rms(bundle["difference"]),residual_rms=rms(bundle["residual"]),
        alignment_q_rms=rms(bundle["alignment"]),alignment_rounding_max=float(np.max(abs(bundle["rounding"]))),
        direct_minus_aligned_rms=rms(direct_minus_aligned),direct_minus_aligned_max=float(np.max(abs(direct_minus_aligned))),
        post_preference_agreement_fraction=float(post["agreement"].mean()),
        post_source_weight_ranges=[[float(v.min()),float(v.max())] for v in post["weights"]],
        inherited_source_residual_used=False,
        structure_cost_is_noise_or_probability=False,featurelessness_certified=False,
        new_unassigned_accounting_warning="New event templates were independently fitted in overlapping windows; their sum is bookkeeping, not a joint decomposition.",
        output_construction="Frozen V21 background times exp((new_wP-old_wP)*unchanged_r21); same aligned-residual model.",
        direct_blend_warning="diagnostic_direct_corrected_background is not a fourth candidate; it adds (wP-.5)*q and can change slow response.",
        structure_cost_warning="A low local structure cost does not prove a source is clean; common or opposite-phase texture and true fixed resonances can bias the heuristic.")
    result["construction_sha256"]=np.array(digest);result["model_sha256"]=np.array(digest)
    result["diagnostics"]=diagnostic
    return result

def generate_all(e,pn,*,baseline,baseline_diagnostics):
    bundle=build_bundle(e,pn,baseline=baseline,baseline_diagnostics=baseline_diagnostics)
    return {f:render(bundle,family=f) for f in FAMILIES}

def generate(e,pn,*,baseline,baseline_diagnostics,family=FAMILIES[2]):
    return render(build_bundle(e,pn,baseline=baseline,baseline_diagnostics=baseline_diagnostics),family=family)
