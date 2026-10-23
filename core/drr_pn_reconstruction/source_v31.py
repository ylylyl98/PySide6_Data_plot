"""V31 bounded current-PN common residual resonance experiment.

Pure numerical API: no data IO. Old fits, source labels, r/q and weights are
unchanged. Every shared correction remains an explicitly conditional model.
"""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "c546ebd9b5e7d20748c40be59f7bd49732bbc4bdc665d0edcabff6c9936ea470"
import copy
import numpy as np
from .source_runtime import least_squares
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix

FAMILIES = ("common_residual_resonance",)
SETTINGS = dict(narrow_width=.010, neighborhood_min=.012, neighborhood_widths=3.,
    detection_threshold=.003, max_groups=8, halfwindow=.035, center_radius=.006,
    width_bounds=(.0015,.015), width_starts=(.003,.006,.010), bound_fraction=.001,
    transfer_min=.1, condition_max=50., coherence_min=.9, train_improvement=.1,
    check_improvement=.05, center_stability=.001, relative_stability=.25)


def local_quadratic(e, values, width):
    """Exact existing measured-coordinate full-window/minimum7 definition."""
    e=np.asarray(e); values=np.asarray(values); n=len(e)
    left=np.searchsorted(e,e-width/2)
    right=np.searchsorted(e,e+width/2,side="right")
    narrow=right-left<7
    left[narrow]=np.clip(np.arange(n)[narrow]-3,0,n-7)
    right[narrow]=left[narrow]+7
    offsets=np.r_[0,np.cumsum(right-left)]
    weights=np.empty(offsets[-1]); columns=np.empty(offsets[-1],dtype=np.int32)
    for i,(a,b) in enumerate(zip(left,right)):
        x=(e[a:b]-e[i])/(width/2)
        design=np.column_stack((np.ones(len(x)),x,x*x))
        weights[offsets[i]:offsets[i+1]]=np.linalg.lstsq(design.T,[1.,0.,0.],rcond=None)[0]
        columns[offsets[i]:offsets[i+1]]=np.arange(a,b)
    projection=csr_matrix((weights,columns,offsets),shape=(n,n))
    shape=values.shape
    return (projection@values.reshape(-1,n).T).T.reshape(shape)


def _basis(e,theta):
    z=(e-theta[0])/theta[1]
    return np.column_stack((1/(1+z*z),z/(1+z*z)))


def _weighted_rms(x,cell):
    return float(np.sqrt(np.sum(np.asarray(x)**2*cell)/np.sum(cell)))


def _project_setup(e,y,cell,seed,degree):
    x=(e-seed)/SETTINGS["halfwindow"]
    poly=np.column_stack([x**j for j in range(degree+1)])
    sw=np.sqrt(cell/cell.mean())
    X=poly*sw[:,None]
    Q=np.linalg.qr(X,mode="reduced")[0]
    Y=y.T*sw[:,None]
    RY=Y-Q@(Q.T@Y)
    poly_only=np.linalg.lstsq(X,Y,rcond=None)[0]
    return dict(e=e,y=y,cell=cell,sw=sw,poly=poly,X=X,Q=Q,Y=Y,RY=RY,
        polynomial_only=poly@poly_only,polynomial_only_coefficients=poly_only.T)


def _linear(setup,theta):
    F=_basis(setup["e"],theta)
    FW=F*setup["sw"][:,None]
    RF=FW-setup["Q"]@(setup["Q"].T@FW)
    coeff=np.linalg.lstsq(RF,setup["RY"],rcond=None)[0]
    nuisance=np.linalg.lstsq(setup["X"],setup["Y"]-FW@coeff,rcond=None)[0]
    prediction=setup["poly"]@nuisance+F@coeff
    residual=setup["RY"]-RF@coeff
    norms=np.linalg.norm(RF,axis=0)
    raw_norms=np.linalg.norm(FW,axis=0)
    normalized=RF/np.maximum(norms,1e-30)
    singular=np.linalg.svd(RF,compute_uv=False)
    scaled_singular=np.linalg.svd(normalized,compute_uv=False)
    condition=float(scaled_singular[0]/max(scaled_singular[-1],1e-30))
    coef=coeff.T
    v=coef.mean(0); vcurve=RF@v
    projected=RF@coeff
    den=float(vcurve@vcurve)
    alpha=(projected.T@vcurve)/max(den,1e-30)
    cosine=float(projected[:,0]@projected[:,1]/max(
        np.linalg.norm(projected[:,0])*np.linalg.norm(projected[:,1]),1e-30))
    common_coef=float(np.min(alpha))*v
    common=F@common_coef
    return dict(theta=np.array(theta),coefficients=coef,nuisance_coefficients=nuisance.T,
        prediction=prediction.T,source_curves=(F@coeff).T,residual=residual.T,
        projected_norms=norms,raw_norms=raw_norms,transfer=norms/np.maximum(raw_norms,1e-30),
        singular_values=singular,normalized_singular_values=scaled_singular,
        condition=condition,source_cosine=cosine,source_projection=alpha,
        common_coefficients=common_coef,common_curve=common,objective=float(np.sum(residual**2)))


def _fit(e,y,cell,seed):
    setup=_project_setup(e,y,cell,seed,2)
    lower=np.array([seed-SETTINGS["center_radius"],SETTINGS["width_bounds"][0]])
    upper=np.array([seed+SETTINGS["center_radius"],SETTINGS["width_bounds"][1]])
    starts=[]; fits=[]
    for width in SETTINGS["width_starts"]:
        sol=least_squares(lambda th:_linear(setup,th)["residual"].ravel(),
            [seed,width],bounds=(lower,upper),x_scale=[.006,.006],
            max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10,loss="linear")
        fitted=_linear(setup,sol.x)
        proximity=np.minimum((sol.x-lower)/(upper-lower),(upper-sol.x)/(upper-lower))
        record=dict(initial_width=width,parameters=sol.x.tolist(),objective=fitted["objective"],
            success=bool(sol.success),status=int(sol.status),message=str(sol.message),
            nfev=int(sol.nfev),optimality=float(sol.optimality),
            normalized_bound_distance=proximity.tolist(),
            bound_hit=bool(np.any(proximity<=SETTINGS["bound_fraction"])))
        starts.append(record);fits.append(fitted)
    best=min(range(len(starts)),key=lambda i:(starts[i]["objective"],i))
    return fits[best],setup,dict(starts=starts,selected_start=best,selected=starts[best])


def _serial_fit(fit,run):
    return dict(parameters=fit["theta"].tolist(),coefficients=fit["coefficients"].tolist(),
        nuisance_coefficients=fit["nuisance_coefficients"].tolist(),
        projected_column_norms=fit["projected_norms"].tolist(),raw_column_norms=fit["raw_norms"].tolist(),
        projected_column_transfer=fit["transfer"].tolist(),
        projected_singular_values=fit["singular_values"].tolist(),
        normalized_projected_singular_values=fit["normalized_singular_values"].tolist(),
        normalized_projected_condition=fit["condition"],source_curve_cosine=fit["source_cosine"],
        source_projection_on_mean=fit["source_projection"].tolist(),
        common_coefficients=fit["common_coefficients"].tolist(),objective=fit["objective"],
        optimization=run)


def detect_events(e,corrected_train,records):
    band=local_quadratic(e,corrected_train,.004)-local_quadratic(e,corrected_train,.035)
    score=np.where(band[0]*band[1]>0,np.minimum(abs(band[0]),abs(band[1])),0.)
    neighborhood=np.zeros(len(e),bool); parents=[]
    for i,row in enumerate(records):
        c,w=row["parameters"][:2]
        if row["applied"] and w<=SETTINGS["narrow_width"]:
            mask=abs(e-c)<=max(SETTINGS["neighborhood_min"],SETTINGS["neighborhood_widths"]*w)
            neighborhood|=mask; parents.append((i,mask))
    starts=np.flatnonzero(np.diff(np.r_[False,neighborhood,False].astype(int))==1)
    stops=np.flatnonzero(np.diff(np.r_[False,neighborhood,False].astype(int))==-1)
    peaks=find_peaks(score)[0]
    events=[];groups=[]
    for gid,(a,b) in enumerate(zip(starts,stops)):
        ids=peaks[(peaks>=a)&(peaks<b)&(score[peaks]>=SETTINGS["detection_threshold"])]
        group=dict(group_id=gid,native_start=int(a),native_stop_exclusive=int(b),
            energy_limits=[float(e[a]),float(e[b-1])],
            parent_indices=[i for i,m in parents if m[a:b].any()],
            eligible_local_maxima=ids.tolist())
        if not len(ids):
            group["decision"]="no_coherent_extremum_above_fixed_threshold"
        else:
            index=min(ids,key=lambda i:(-score[i],i))
            group.update(decision="seed",seed_index=int(index),seed_strength=float(score[index]))
            events.append(dict(group_id=gid,seed_index=int(index),seed_energy=float(e[index]),
                seed_strength=float(score[index]),parent_indices=group["parent_indices"]))
        groups.append(group)
    events.sort(key=lambda row:(-row["seed_strength"],row["seed_index"]))
    selected=events[:SETTINGS["max_groups"]]
    for row in events[SETTINGS["max_groups"]:]:
        groups[row["group_id"]]["decision"]="fixed_group_cap"
    return selected,dict(groups=groups,groups_with_seed=len(events),
        groups_selected=len(selected),groups_capped=max(0,len(events)-SETTINGS["max_groups"])),band,score,neighborhood


def fit_common_stage(e,corrected_train,corrected_check,records):
    """Pure stage for conditional PN fits; useful for small independent controls."""
    e=np.asarray(e,dtype=float);cell=np.gradient(e)
    events,detection,band,score,neighborhood=detect_events(e,corrected_train,records)
    models=[];window_masks=[];rows=[]
    for event in events:
        seed=event["seed_energy"];mask=abs(e-seed)<=SETTINGS["halfwindow"]
        idx=np.flatnonzero(mask);row=copy.deepcopy(event);row["fit_window"]=[seed-.035,seed+.035]
        row.update(applied=False,reasons=[],fit_point_indices=idx.tolist())
        window_masks.append(mask);models.append(np.zeros(len(e)))
        if len(idx)<21 or np.sum(e[idx]<seed)<7 or np.sum(e[idx]>seed)<7:
            row["reasons"].append("insufficient_two_sided_native_support");rows.append(row);continue
        fit,setup,run=_fit(e[mask],corrected_train[:,mask],cell[mask],seed)
        check,check_setup,check_run=_fit(e[mask],corrected_check[:,mask],cell[mask],seed)
        cubic_setup=_project_setup(e[mask],corrected_train[:,mask],cell[mask],seed,3)
        cubic=_linear(cubic_setup,fit["theta"])
        train_sse=np.sum((corrected_train[:,mask]-setup["polynomial_only"].T)**2*cell[mask],axis=1)
        fit_sse=np.sum((corrected_train[:,mask]-fit["prediction"])**2*cell[mask],axis=1)
        check_poly_sse=np.sum((corrected_check[:,mask]-setup["polynomial_only"].T)**2*cell[mask],axis=1)
        check_fit_sse=np.sum((corrected_check[:,mask]-fit["prediction"])**2*cell[mask],axis=1)
        train_imp=1-fit_sse/np.maximum(train_sse,1e-30)
        check_imp=1-check_fit_sse/np.maximum(check_poly_sse,1e-30)
        scale=max(_weighted_rms(fit["common_curve"],cell[mask]),1e-30)
        repeat_curve_diff=_weighted_rms(check["common_curve"]-fit["common_curve"],cell[mask])/scale
        cubic_diff=_weighted_rms(cubic["common_curve"]-fit["common_curve"],cell[mask])/scale
        center_diff=float(abs(check["theta"][0]-fit["theta"][0]))
        width_diff=float(abs(check["theta"][1]/fit["theta"][1]-1))
        row.update(training=_serial_fit(fit,run),conditional_last50=_serial_fit(check,check_run),
            cubic_fixed_shape=dict(coefficients=cubic["coefficients"].tolist(),
                common_coefficients=cubic["common_coefficients"].tolist(),
                common_curve_relative_weighted_RMS_difference=cubic_diff,
                source_cosine=cubic["source_cosine"]),
            training_polynomial_SSE=train_sse.tolist(),training_resonance_SSE=fit_sse.tolist(),
            conditional_polynomial_SSE=check_poly_sse.tolist(),conditional_resonance_SSE=check_fit_sse.tolist(),
            train_relative_improvement=train_imp.tolist(),conditional_relative_improvement=check_imp.tolist(),
            stability=dict(center_difference_eV=center_diff,relative_width_difference=width_diff,
                common_curve_relative_weighted_RMS_difference=repeat_curve_diff),
            common_curve_training_weighted_RMS=scale)
        for label,f,rn in (("train",fit,run),("conditional",check,check_run)):
            if not rn["selected"]["success"]:row["reasons"].append(label+"_solver_failed")
            if rn["selected"]["bound_hit"]:row["reasons"].append(label+"_bound_hit")
            if np.min(f["transfer"])<SETTINGS["transfer_min"]:row["reasons"].append(label+"_low_projected_transfer")
            if f["condition"]>SETTINGS["condition_max"]:row["reasons"].append(label+"_ill_conditioned")
            if f["source_cosine"]<SETTINGS["coherence_min"] or np.min(f["source_projection"])<=0:
                row["reasons"].append(label+"_source_incoherent")
        if np.min(train_imp)<SETTINGS["train_improvement"]:row["reasons"].append("insufficient_train_improvement")
        if np.min(check_imp)<SETTINGS["check_improvement"]:row["reasons"].append("insufficient_conditional_improvement")
        if center_diff>SETTINGS["center_stability"]:row["reasons"].append("center_unstable")
        if width_diff>SETTINGS["relative_stability"]:row["reasons"].append("width_unstable")
        if repeat_curve_diff>SETTINGS["relative_stability"]:row["reasons"].append("common_amplitude_unstable")
        if cubic_diff>SETTINGS["relative_stability"]:row["reasons"].append("nuisance_degree_sensitive")
        if cubic["source_cosine"]<SETTINGS["coherence_min"] or np.min(cubic["source_projection"])<=0:
            row["reasons"].append("cubic_source_incoherent")
        models[-1]=_basis(e,fit["theta"])@fit["common_coefficients"]
        row["eligible_before_overlap"]=not row["reasons"];rows.append(row)
    ranking=sorted([i for i,r in enumerate(rows) if r.get("eligible_before_overlap",False)],
        key=lambda i:(-min(rows[i]["train_relative_improvement"]),-rows[i]["seed_strength"],rows[i]["seed_index"]))
    kept=[]
    for i in ranking:
        collisions=[j for j in kept if abs(rows[i]["seed_energy"]-rows[j]["seed_energy"])<=2*SETTINGS["halfwindow"]]
        if collisions:
            rows[i]["reasons"].append("overlapping_fit_window_duplicate")
            rows[i]["overlap_with_retained_event_indices"]=collisions
        else:
            rows[i]["applied"]=True;kept.append(i)
    models=np.array(models,dtype=float).reshape(-1,len(e))
    applied=np.array([row["applied"] for row in rows],bool)
    total=models[applied].sum(0) if applied.any() else np.zeros(len(e))
    return dict(common_feature_log=total,common_component_feature_log=models,
        common_component_applied=applied,v31_fit_window_mask=np.array(window_masks,bool).reshape(-1,len(e)),
        v31_detection_band=band,v31_detection_score=score,v31_detection_neighborhood_mask=neighborhood,
        diagnostics=dict(detection=detection,events=rows,accepted_event_indices=kept,
            overlap_selection_order=ranking,accepted_count=len(kept),
            rejected_count=len(rows)-len(kept),training_source="first half PN intensity mean",
            conditional_check_source="last50 PN intensity mean; full100 parent fits retained"))


def generate_all(e,pn,*,pn_frames,baseline,baseline_diagnostics):
    e=np.asarray(e,dtype=float);pn=np.asarray(pn,dtype=float);frames=np.asarray(pn_frames,dtype=float)
    if e.ndim!=1 or len(e)<21 or not np.all(np.isfinite(e)) or not np.all(np.diff(e)>0):
        raise ValueError("Expected finite strictly increasing native E")
    if pn.shape!=(2,len(e)) or frames.ndim!=3 or frames.shape[0]!=2 or frames.shape[2]!=len(e) or frames.shape[1]<4:
        raise ValueError("Expected PN(2,E) and at least four repeated frames per source")
    if not np.all(np.isfinite(pn)&(pn>0)) or not np.all(np.isfinite(frames)&(frames>0)):
        raise ValueError("Expected positive finite intensities")
    if not np.array_equal(e,baseline["energy"]):
        raise ValueError("Native energy differs from parent")
    if not np.allclose(frames.mean(1),pn,rtol=1e-13,atol=0):
        raise ValueError("Full100 intensity average differs from PN")
    original_features=np.asarray(baseline["source_feature_log"])
    C=np.log(pn)-original_features
    r=np.asarray(baseline["residual_intervention_native_log"])
    q=np.asarray(baseline["v29_alignment_q_log"])
    weights=np.asarray(baseline["source_weights"])
    if not np.allclose(C[0]-C[1],r+q,rtol=0,atol=6e-15):
        raise ValueError("Parent differential reconstruction failed")
    split=frames.shape[1]//2
    train=np.log(frames[:,:split].mean(1))-original_features
    check=np.log(frames[:,split:].mean(1))-original_features
    stage=fit_common_stage(e,train,check,baseline_diagnostics["components"])
    common=stage["common_feature_log"]
    oldbg=np.asarray(baseline["background"])
    bg=oldbg*np.exp(-common) if stage["diagnostics"]["accepted_count"] else oldbg.copy()
    effective=original_features+common[None,:]
    newC=C-common[None,:]
    result={k:v.copy() for k,v in baseline.items() if isinstance(v,np.ndarray)}
    result.update({k:v.copy() for k,v in stage.items() if isinstance(v,np.ndarray)})
    result.update(energy=e.copy(),background=bg,baseline_background=oldbg.copy(),
        effective_source_feature_log=effective,corrected_source_log=newC,
        v31_original_corrected_source_log=C,v31_train_corrected_source_log=train,
        v31_conditional_corrected_source_log=check,
        v31_background_delta_log=-common,
        repair_mask=abs(np.log(bg/np.sqrt(pn[0]*pn[1])))>1e-12,seams=np.array([],float))
    diag=copy.deepcopy(baseline_diagnostics)
    diag.update(family=FAMILIES[0],BN_used=False,historical_BN_used=False,sample_used=False,
        source_fits_frozen=True,source_assignments_frozen=True,
        source_weights_interpretation="Frozen V29 aligned-residual weights; additional equal shared correction changes common mean only.",
        v31=dict(settings=copy.deepcopy(SETTINGS),**stage["diagnostics"],
            construction="Bparent*exp(-common_feature_log); original source_feature_log stays frozen; effective source features include common.",
            global_common_gauge_ambiguity=True,
            conditional_frames_not_independent_validation=True,
            nuisance_never_in_background=True,
            differential_change_max=float(np.max(abs((newC[0]-newC[1])-(C[0]-C[1])))),
            common_curve_log_RMS=float(np.sqrt(np.mean(common**2))),
            common_curve_log_max=float(np.max(abs(common))),
            dRR_ratio_gain_exp_common_range=[float(np.exp(common).min()),float(np.exp(common).max())],
            log_gate_ratio_invariance_is_algebra_not_feature_preservation=True))
    result["diagnostics"]=diag
    return {FAMILIES[0]:result}


def generate(e,pn,*,pn_frames,baseline,baseline_diagnostics,family=FAMILIES[0]):
    if family not in FAMILIES:raise ValueError("Unknown frozen family")
    return generate_all(e,pn,pn_frames=pn_frames,baseline=baseline,
        baseline_diagnostics=baseline_diagnostics)[family]
