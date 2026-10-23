"""V29: zero-refit risk weights for the frozen V24 aligned PN residual.

Pure numerical API. Importing or calling this module reads no data files.
"""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "69cf16b965cf46df2765b79db97c4885a0a53f47609b3c40bd8c4f9b74bc76b9"
import copy
import hashlib
import numpy as np

FAMILIES=(
    "all_applied_amplitude_weights",
    "all_applied_curvature_weights",
    "all_applied_curvature_smoothed_weights",
)
SETTINGS=dict(amplitude_floor_log=.015,curvature_scale_ev=.008,gaussian_sigma_ev=.002,
    component_selection="all_and_only_applied",curvature_kernel_power=3,
    weight_clipping=False,seam_masking=False,source_fits_frozen=True)

def _sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def gaussian_kernel(e,sigma=.002):
    """Positive native-cell row-normalized kernel; full support, no hard cutoff."""
    cell=np.gradient(e)
    kernel=np.exp(-.5*((e[:,None]-e[None,:])/sigma)**2)*cell[None,:]
    kernel/=kernel.sum(axis=1,keepdims=True)
    return kernel,cell

def _weights(risk,floor):
    rho=1/(1+risk/floor)
    return rho/rho.sum(axis=0,keepdims=True)

def generate_all(e,pn,*,baseline,baseline_diagnostics):
    e=np.asarray(e,dtype=float);pn=np.asarray(pn,dtype=float)
    if e.ndim!=1 or len(e)<3 or not np.all(np.isfinite(e)) or not np.all(np.diff(e)>0):
        raise ValueError("Expected finite strictly increasing native energy")
    if pn.shape!=(2,len(e)) or not np.all(np.isfinite(pn)&(pn>0)):
        raise ValueError("Expected two positive native-axis PN spectra")
    if not np.array_equal(e,baseline["energy"]):
        raise ValueError("Baseline energy differs")
    records=baseline_diagnostics["components"]
    f=np.asarray(baseline["component_feature_log"],dtype=float)
    applied=np.asarray(baseline["component_applied"],dtype=bool)
    source=np.asarray(baseline["component_source"],dtype=int)
    n=len(records)
    if f.shape!=(n,len(e)) or applied.shape!=(n,) or source.shape!=(n,) or not np.isfinite(f).all():
        raise ValueError("Frozen component arrays are inconsistent")
    old_weights=np.asarray(baseline["residual_source_weights"],dtype=float)
    residual=np.asarray(baseline["residual_intervention_native_log"],dtype=float)
    old_bg=np.asarray(baseline["background"],dtype=float)
    if old_weights.shape!=(2,len(e)) or residual.shape!=(len(e),) or old_bg.shape!=(len(e),):
        raise ValueError("Frozen residual, weights or background shape mismatch")
    if not np.all(np.isfinite(old_bg)&(old_bg>0)) or not np.isfinite(residual).all():
        raise ValueError("Invalid frozen background or residual")
    if not np.all((old_weights>=0)&(old_weights<=1)) or not np.allclose(old_weights.sum(0),1,rtol=0,atol=4e-15):
        raise ValueError("Invalid frozen weights")
    corrected=np.log(pn)-baseline["source_feature_log"]
    alignment=corrected[0]-corrected[1]-residual
    saved_q=baseline["v24_new_alignment_q_log"]
    if not np.allclose(alignment,saved_q,rtol=0,atol=5e-15):
        raise ValueError("PN or frozen alignment mismatch")
    expected=np.exp(corrected.mean(0)+(old_weights[0]-.5)*residual)
    if not np.allclose(old_bg,expected,rtol=2e-14,atol=0):
        raise ValueError("Baseline is not the expected aligned construction")
    amplitude=np.max(abs(f),axis=1)
    component_amplitude_risk=np.zeros_like(f)
    component_curvature_risk=np.zeros_like(f)
    amplitude_risk=np.zeros((2,len(e)))
    curvature_risk=np.zeros_like(amplitude_risk)
    source_sum=np.zeros_like(amplitude_risk)
    centers=np.zeros(n);widths=np.zeros(n)
    for i,row in enumerate(records):
        if bool(row["applied"])!=bool(applied[i]) or int(row["source"])!=int(source[i]):
            raise ValueError("Frozen record/array source or applied decision differs")
        centers[i],widths[i]=row["parameters"][:2]
        if not applied[i]:
            continue
        if source[i] not in (0,1) or not np.isfinite(widths[i]) or widths[i]<=0:
            raise ValueError("Invalid applied source or width")
        z=(e-centers[i])/widths[i]
        component_amplitude_risk[i]=amplitude[i]**2/(1+z*z)
        component_curvature_risk[i]=(amplitude[i]**2/widths[i]**4)/(1+z*z)**3
        amplitude_risk[source[i]]+=component_amplitude_risk[i]
        curvature_risk[source[i]]+=component_curvature_risk[i]
        source_sum[source[i]]+=(1 if source[i]==0 else -1)*f[i]
    if not np.allclose(source_sum,baseline["source_feature_log"],rtol=0,atol=5e-15):
        raise ValueError("Applied components do not reconstruct frozen source features")
    amp_floor=SETTINGS["amplitude_floor_log"]**2
    curv_floor=(SETTINGS["amplitude_floor_log"]/SETTINGS["curvature_scale_ev"]**2)**2
    amplitude_weights=_weights(amplitude_risk,amp_floor)
    curvature_weights=_weights(curvature_risk,curv_floor)
    kernel,cell=gaussian_kernel(e,SETTINGS["gaussian_sigma_ev"])
    smoothed_p=kernel@curvature_weights[0]
    smoothed_weights=np.array([smoothed_p,1-smoothed_p])
    families_weights=(amplitude_weights,curvature_weights,smoothed_weights)
    common=dict(
        v29_amplitude_risk=amplitude_risk,v29_curvature_risk=curvature_risk,
        v29_component_amplitude_risk=component_amplitude_risk,
        v29_component_curvature_risk=component_curvature_risk,
        v29_component_amplitude=amplitude,v29_component_center=centers,v29_component_width=widths,
        v29_amplitude_weights=amplitude_weights,v29_curvature_native_weights=curvature_weights,
        v29_curvature_smoothed_weights=smoothed_weights,
        v29_native_cell_width=cell,v29_gaussian_row_sum=kernel.sum(axis=1),
        v29_baseline_weights=old_weights,v29_alignment_q_log=np.asarray(saved_q),
    )
    output={}
    for family,weights in zip(FAMILIES,families_weights):
        change=(weights[0]-old_weights[0])*residual
        background=old_bg*np.exp(change)
        explicit=np.exp(corrected.mean(0)+(weights[0]-.5)*residual)
        if not np.allclose(background,explicit,rtol=2e-14,atol=0):
            raise ValueError("Changed aligned construction failed")
        # Preserve every old array, replacing only declared residual-weight outputs.
        result={k:v.copy() for k,v in baseline.items() if isinstance(v,np.ndarray)}
        result.update({k:v.copy() for k,v in common.items()})
        result.update(
            energy=e.copy(),background=background,baseline_background=old_bg.copy(),
            source_weights=weights.copy(),weights=weights.copy(),residual_source_weights=weights.copy(),
            residual_intervention_delta_log=(weights[0]-.5)*residual,
            background_delta_log=change,v29_weight_delta=weights-old_weights,
            raw_geometric_mean=np.sqrt(pn[0]*pn[1]),
            repair_mask=abs(np.log(background/np.sqrt(pn[0]*pn[1])))>1e-12,
            seams=np.array([],dtype=float),
        )
        diag=copy.deepcopy(baseline_diagnostics)
        diag.update(family=family,BN_used=False,historical_BN_used=False,sample_used=False,
            components=copy.deepcopy(records),source_assignments_frozen=True,source_fits_frozen=True,
            source_assignment_revalidated=False,source_weights_interpretation=
            "Heuristic component risk weights mix only frozen aligned differential residual, not raw source intensities.",
            v29=dict(settings=copy.deepcopy(SETTINGS),applied_indices=np.flatnonzero(applied).tolist(),
                excluded_unapplied_indices=np.flatnonzero(~applied).tolist(),
                amplitude_floor=amp_floor,curvature_floor=curv_floor,
                input_source_features_sha256=_sha(baseline["source_feature_log"]),
                input_component_features_sha256=_sha(f),input_residual_sha256=_sha(residual),
                input_alignment_sha256=_sha(saved_q),output_weights_sha256=_sha(weights),
                applied_components_reconstruct_source_max=float(np.max(abs(source_sum-baseline["source_feature_log"]))),
                explicit_formula_max_relative=float(np.max(abs(background/explicit-1))),
                weight_sum_error=float(np.max(abs(weights.sum(0)-1))),
                weight_range=[float(weights.min()),float(weights.max())],
                gaussian_constant_error=float(np.max(abs(kernel.sum(1)-1))),
                delta_log_rms=float(np.sqrt(np.mean(change**2))),delta_log_max=float(np.max(abs(change))),
                construction="V24post * exp((new wP-old wP)*frozen r); q/source features/fits unchanged",
                risk_caveat="Not noise or clean-source probability; shared residuals and derivative cross terms remain possible."),
        )
        result["diagnostics"]=diag
        output[family]=result
    return output

def generate(e,pn,*,baseline,baseline_diagnostics,family=FAMILIES[0]):
    if family not in FAMILIES:
        raise ValueError("Unknown frozen family")
    return generate_all(e,pn,baseline=baseline,baseline_diagnostics=baseline_diagnostics)[family]
