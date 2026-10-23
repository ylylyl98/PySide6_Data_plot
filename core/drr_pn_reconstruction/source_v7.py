"""V7 current-PN-only conditional source attribution and tail sensitivities."""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "5489af4393eb47e8179a0308e8ad2bbdf493cc1d23ccb2b9a5a9a02f06c3df3e"
import hashlib
import json
import numpy as np
from .source_runtime import least_squares
from scipy.signal import find_peaks

from . import source_v6 as _v6
V6_SHA256 = _v6.MIGRATED_FROM_SHA256
FAMILIES=('conditional_local_weighted','conditional_full_tail','opposite_full_tail')
SETTINGS=dict(ratio_round_decimals=12,edge_height=2.,edge_prominence=1.5,
    same_direction_suppression_ev=.014,opposite_direction_suppression_ev=.008,
    edge_contrast_log=.006,maximum_edges=16,maximum_components=8,
    pair_separation_ev=[.008,.060],pair_minimum_magnitude_ratio=.3,
    source_coefficient_minimum=.6,other_source_fraction_maximum=.5,
    flank_radius_factors=[1.,1.5,2.],maximum_flank_radius_ev=.180,
    minimum_flank_points_per_side=12,other_component_anchor_threshold_log=.005,
    fit_residual_anchor_threshold_log=.010,soft_l1_scale=.003,feature_penalty=.003,
    max_nfev=800,tolerance=1e-9,reliability_scale_log=.015,
    source_medium_guard_log=.015,isolated_fine_guard_log=.003,
    common_localized_guard_log=.010,fringe_period_range_ev=[.0015,.012],
    fringe_side_width_ev=.025,fringe_period_relative_tolerance=.30,
    fringe_min_autocorrelation=.35,fringe_min_source_correlation=.8)
_rms=_v6._rms
_sha_array=_v6._sha_array


def _quad(e,y,width):
    y=np.atleast_2d(y);out=np.empty_like(y)
    for i,center in enumerate(e):
        lo=np.searchsorted(e,center-width/2);hi=np.searchsorted(e,center+width/2,side='right')
        if hi-lo<7:lo=max(0,min(i-3,len(e)-7));hi=lo+7
        x=(e[lo:hi]-center)/(width/2);design=np.column_stack((np.ones(len(x)),x,x*x))
        out[:,i]=np.linalg.lstsq(design,y[:,lo:hi].T,rcond=None)[0][0]
    return out


def _automatic_support(e,logs):
    """Conditional featureless support; oscillations alone are not vetoed."""
    q4=_quad(e,logs,.004);q35=_quad(e,logs,.035);q140=_quad(e,logs,.140)
    fine=logs-q4;medium=q4-q140;texture=logs-q35;common_texture=texture.mean(axis=0)
    period=np.zeros((2,len(e)));rho=np.zeros((2,len(e)));source_corr=np.zeros(len(e));compatible=np.zeros(len(e),bool)
    def periodic(values,step):
        if len(values)<16:return 0.,0.
        values=values-np.mean(values)
        if _rms(values)<.0002:return 0.,0.
        maxlag=min(len(values)//2,int(.012/step))
        if maxlag<4:return 0.,0.
        ac=[]
        for lag in range(1,maxlag+1):
            a=values[:-lag];b=values[lag:];den=np.linalg.norm(a)*np.linalg.norm(b)
            ac.append(float(np.dot(a,b)/den) if den>0 else 0.)
        peaks,_=find_peaks(ac,prominence=.05)
        peaks=[int(i) for i in peaks if (i+1)*step>=.0015 and ac[i]>=.35]
        if not peaks:return 0.,0.
        k=max(peaks,key=lambda i:ac[i]);return (k+1)*step,ac[k]
    for i,center in enumerate(e):
        left=(e>=center-.025)&(e<center);right=(e>center)&(e<=center+.025);whole=left|right
        if min(left.sum(),right.sum())<16:continue
        step=float(np.median(np.diff(e[whole])))
        for side,mask in enumerate((left,right)):
            grid=np.arange(e[mask][0],e[mask][-1]+step/2,step)
            period[side,i],rho[side,i]=periodic(np.interp(grid,e[mask],common_texture[mask]),step)
        a=texture[0,whole]-np.mean(texture[0,whole]);b=texture[1,whole]-np.mean(texture[1,whole]);den=np.linalg.norm(a)*np.linalg.norm(b)
        source_corr[i]=np.dot(a,b)/den if den>0 else 0.
        amp=[_rms(common_texture[mask]) for mask in (left,right)]
        if min(period[:,i])>0 and source_corr[i]>=.8 and min(amp)>0:
            compatible[i]=(abs(period[0,i]-period[1,i])/max(period[:,i])<=.30 and max(amp)/min(amp)<=2.)
    common_local=(q4-q35).mean(axis=0);common_fine=fine.mean(axis=0)
    common_unknown=(~compatible)&((abs(common_local)>.010)|(abs(common_fine)>.003))
    usable=(abs(medium)<=.015)&((abs(fine)<=.003)|compatible[None,:])&(~common_unknown[None,:])
    reasons=np.zeros_like(logs,dtype=np.int8)
    reasons[abs(medium)>.015]=2
    reasons[(abs(fine)>.003)&(~compatible[None,:])]=3
    reasons[:,common_unknown]=4
    reasons[usable&compatible[None,:]]=1
    common_class=np.zeros(len(e),np.int8);common_class[compatible]=1;common_class[common_unknown]=2
    code=usable[0].astype(np.int8)+2*usable[1].astype(np.int8)
    starts=np.r_[0,np.flatnonzero(np.diff(code))+1];ends=np.r_[starts[1:],len(e)]
    segments=[dict(xmin_ev=float(e[a]),xmax_ev=float(e[b-1]),source_code=int(code[a]),
        label={0:'unknown_or_no_usable_source',1:'P_conditionally_usable',2:'N_conditionally_usable',3:'both_conditionally_usable'}[int(code[a])],
        evidence='multiscale source residuals and local periodic compatibility; not physical proof') for a,b in zip(starts,ends)]
    return dict(usable=usable,reasons=reasons,common_class=common_class,common_unknown=common_unknown,
        fringe_compatible=compatible,fringe_period_ev=period,fringe_autocorrelation=rho,
        fringe_source_correlation=source_corr,source_fine=fine,source_medium=medium,
        source_usage_code=code,segments=segments,q4logs=q4)


def _detect(e,ratio):
    smooth=_v6._q4(e,ratio);edge=np.gradient(smooth,e)
    indices,properties=find_peaks(abs(edge),height=2.,prominence=1.5)
    candidates=[]
    for k,i in enumerate(indices):
        contrast=float(np.ptp(np.quantile(smooth[abs(e-e[i])<=.015],[.05,.95])))
        if contrast<.006 or min(e[i]-e[0],e[-1]-e[i])<.060:continue
        candidates.append(dict(index=int(i),energy_ev=float(e[i]),edge_per_ev=float(edge[i]),
            contrast_log=contrast,prominence_per_ev=float(properties['prominences'][k])))
    kept=[]
    for row in sorted(candidates,key=lambda d:-abs(d['edge_per_ev'])):
        if len(kept)>=16:break
        if any(abs(row['energy_ev']-old['energy_ev']) < (.014 if row['edge_per_ev']*old['edge_per_ev']>0 else .008) for old in kept):continue
        kept.append(row)
    kept.sort(key=lambda d:d['energy_ev'])
    events=[];k=0
    while k<len(kept):
        a=kept[k]
        if k+1<len(kept):
            b=kept[k+1];sep=b['energy_ev']-a['energy_ev']
            relative=min(abs(a['edge_per_ev']),abs(b['edge_per_ev']))/max(abs(a['edge_per_ev']),abs(b['edge_per_ev']))
            paired=a['edge_per_ev']*b['edge_per_ev']<0 and .008<=sep<=.060 and relative>=.3
        else:paired=False
        if paired:
            inside=np.flatnonzero((e>=a['energy_ev'])&(e<=b['energy_ev']))
            index=inside[np.argmax(smooth[inside]) if a['edge_per_ev']>0 else np.argmin(smooth[inside])]
            width=.866*sep;center=float(e[index]);radius=max(.006,.25*sep)
            events.append(dict(energy_ev=center,kind='paired_symmetric_edges',
                edge_indices=[a['index'],b['index']],edge_energies_ev=[a['energy_ev'],b['energy_ev']],
                edge_per_ev=a['edge_per_ev'],strength=max(abs(a['edge_per_ev']),abs(b['edge_per_ev'])),
                initial_width_ev=width,center_radius_ev=radius,
                width_upper_ev=min(.060,max(.020,2*width)),initial_q=0.,
                initial_b=.2 if a['edge_per_ev']>0 else -.2))
            k+=2
        else:
            events.append(dict(energy_ev=a['energy_ev'],kind='single_asymmetric_edge',
                edge_indices=[a['index']],edge_energies_ev=[a['energy_ev']],edge_per_ev=a['edge_per_ev'],
                strength=abs(a['edge_per_ev']),initial_width_ev=.003,center_radius_ev=.006,
                width_upper_ev=.020,initial_q=-.3 if a['edge_per_ev']<0 else .3,initial_b=-.3))
            k+=1
    events=sorted(sorted(events,key=lambda d:-d['strength'])[:8],key=lambda d:d['energy_ev'])
    return events,dict(all_edge_candidates=candidates,retained_edges=kept,events=events),smooth,edge


def _fit(e,ratio,events):
    n=len(e);count=len(events);tapers=np.ones((count,n));x=2*(e-(e[0]+e[-1])/2)/(e[-1]-e[0])
    design=np.column_stack((np.ones(n),x,x*x));cell=np.gradient(e);weight=np.sqrt(n*cell/cell.sum())
    low=[];high=[];scale=[]
    for r in events:
        low.extend([r['energy_ev']-r['center_radius_ev'],.0006,-3.,-6.])
        high.extend([r['energy_ev']+r['center_radius_ev'],r['width_upper_ev'],3.,3.])
        scale.extend([r['center_radius_ev'],r['initial_width_ev'],1.,1.])
    low.extend([-.5]*3);high.extend([.5]*3);scale.extend([.03]*3)
    low=np.array(low);high=np.array(high);scale=np.array(scale)
    nuisance0=np.clip(np.linalg.lstsq(design,ratio,rcond=None)[0],-.49,.49)
    def residual(pars):
        components,unused=_v6._features(e,pars[:-3],tapers)
        return np.r_[weight*(components.sum(axis=0)+design@pars[-3:]-ratio),.003*components.ravel()]
    def jacobian(pars):
        unused,fjac=_v6._features(e,pars[:-3],tapers)
        jac=np.zeros((n*(count+1),4*count+3))
        for j in range(count):
            jac[:n,4*j:4*j+4]=weight[:,None]*fjac[j]
            jac[n+j*n:n+(j+1)*n,4*j:4*j+4]=.003*fjac[j]
        jac[:n,-3:]=weight[:,None]*design
        return jac
    fits=[]
    for factor in (.5,1.,2.):
        first=[]
        for r in events:first.extend([r['energy_ev'],r['initial_width_ev']*factor,r['initial_q'],r['initial_b']])
        start=np.clip(np.r_[first,nuisance0],low+1e-8,high-1e-8)
        fits.append(least_squares(residual,start,jac=jacobian,bounds=(low,high),x_scale=scale,
            loss='soft_l1',f_scale=.003,max_nfev=800,ftol=1e-9,xtol=1e-9,gtol=1e-9))
    chosen=min(range(len(fits)),key=lambda i:(fits[i].cost,np.linalg.norm(fits[i].x)))
    best=fits[chosen];components,unused=_v6._features(e,best.x[:-3],tapers)
    nuisance=design@best.x[-3:];model=components.sum(axis=0)+nuisance;error=ratio-model
    records=[]
    for k,event in enumerate(events):
        pars=best.x[4*k:4*k+4]
        distance=np.minimum(pars-low[4*k:4*k+4],high[4*k:4*k+4]-pars)/(high[4*k:4*k+4]-low[4*k:4*k+4])
        hits=np.flatnonzero(distance<=.01).tolist()
        core=abs(e-pars[0])<=max(.025,3*pars[1]);threshold=max(.005,.15*_rms((ratio-nuisance)[core]))
        reasons=[]
        if not best.success:reasons.append('optimizer_not_converged')
        if np.max(abs(components[k]))<.005:reasons.append('weak_fitted_component')
        if _rms(error[core])>threshold:reasons.append('differential_local_residual')
        if hits:reasons.append('parameter_boundary')
        records.append(dict(**event,parameters=pars.tolist(),bound_distance=distance.tolist(),
            bound_hits=hits,fit_usable=not reasons,fit_rejection_reasons=reasons,
            local_residual_log=_rms(error[core]),local_residual_threshold_log=threshold,
            maximum_feature_log=float(np.max(abs(components[k])))))
    singular=np.linalg.svd(jacobian(best.x)[:n]*scale[None,:],compute_uv=False)
    diagnostics=dict(selected_start=chosen,starts=[dict(success=bool(f.success),status=int(f.status),
        message=str(f.message),cost=float(2*f.cost),optimality=float(f.optimality),nfev=int(f.nfev),
        parameters=f.x.tolist(),active_bounds=f.active_mask.tolist()) for f in fits],
        scaled_jacobian_condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
        nuisance_coefficients=best.x[-3:].tolist(),differential_residual_rms_log=_rms(error))
    return components,nuisance,model,error,records,diagnostics


def _flanks(e,center,width,exclude):
    inner=max(.025,3*width);outer=max(.050,6*width);delta=e-center
    attempts=[]
    for factor in (1.,1.5,2.):
        a=inner*factor;b=min(.180,outer*factor)
        left=(delta>=-b)&(delta<=-a)&(~exclude)
        right=(delta>=a)&(delta<=b)&(~exclude)
        attempts.append(dict(factor=factor,inner_radius_ev=a,outer_radius_ev=b,
            left_points=int(left.sum()),right_points=int(right.sum())))
        if min(left.sum(),right.sum())>=12:return left|right,attempts,True
    return left|right,attempts,False


def _attribute(e,logs,components,records,automatic,error):
    count,n=components.shape;anchors=np.zeros((count,n),bool);source=np.full(count,-1,int)
    margins=np.zeros(count);tapers=np.zeros_like(components);projection_residuals=np.zeros((count,2,n))
    for k,row in enumerate(records):
        center,width,q,b=row['parameters'];inner=max(.025,3*width);outer=max(.050,6*width)
        tapers[k]=_v6._quintic((outer-abs(e-center))/(outer-inner))
        other=np.max(abs(np.delete(components,k,axis=0)),axis=0) if count>1 else np.zeros(n)
        exclude=(other>.005)|(~automatic['usable'].all(axis=0))|(abs(error)>.010)
        attempts=[];supported=False
        x=(e-center)/outer;design=np.column_stack((np.ones(n),x,x*x))
        for refinement in range(3):
            mask,tries,supported=_flanks(e,center,width,exclude);attempts.extend(tries)
            if not supported:break
            coefficients=np.linalg.lstsq(design[mask],logs[:,mask].T,rcond=None)[0]
            quality=np.all(abs(automatic['q4logs']-(design@coefficients).T)<=.010,axis=0)
            refined=mask&quality
            if min(np.sum(refined&(e<center)),np.sum(refined&(e>center)))>=12:
                mask=refined;break
            exclude|=mask&(~quality);supported=False
        anchors[k]=mask
        row.update(attribution_flank_attempts=attempts,source=-1,attribution='unresolved',
            support_ev=[float(center-outer),float(center+outer)],
            plateau_ev=[float(center-inner),float(center+inner)])
        if not supported:
            row['attribution_reason']='insufficient_two_sided_common_support';continue
        source_coeff=np.linalg.lstsq(design[mask],logs[:,mask].T,rcond=None)[0]
        local_residual=logs-(design@source_coeff).T
        basis=components[k]-design@np.linalg.lstsq(design[mask],components[k,mask],rcond=None)[0]
        core=abs(e-center)<=max(.020,2*width)
        denom=float(np.dot(basis[core],basis[core]))
        if denom<=1e-16:
            row['attribution_reason']='degenerate_projection_basis';continue
        coefficient=local_residual[:,core]@basis[core]/denom
        cp,cn=map(float,coefficient);projection_residuals[k]=local_residual
        p_margin=min(cp-.6,.5*abs(cp)-abs(cn));n_margin=min(-cn-.6,.5*abs(cn)-abs(cp))
        if cp>=.6 and abs(cn)<=.5*abs(cp):source[k]=0;margins[k]=p_margin
        elif cn<=-.6 and abs(cp)<=.5*abs(cn):source[k]=1;margins[k]=n_margin
        row.update(source=int(source[k]),source_projection_coefficients=[cp,cn],
            source_envelope_coefficients=source_coeff.tolist(),attribution_margin=float(margins[k]),
            attribution='P' if source[k]==0 else 'N' if source[k]==1 else 'unresolved',
            attribution_reason='conditional_local_source_residual_evidence' if source[k]>=0 else 'both_or_neither_source_compatible')
    return source,tapers,anchors,projection_residuals


def _reliability(e,corrected,components,records,source,tapers,error,automatic):
    n=len(e);severity=np.zeros((2,n));response_sum=np.zeros(n);response_weight=np.zeros(n)
    anchors=np.zeros_like(components,dtype=bool);reports=[]
    ratio=np.round(corrected[0]-corrected[1],12)
    for k,row in enumerate(records):
        if not row['applied']:
            reports.append(dict(component=k,used=False,reason='component_not_applied'));continue
        center,width,unused,unused2=row['parameters']
        mask,attempts,supported=_flanks(e,center,width,(abs(error)>.010)|(~automatic['usable'].all(axis=0)));anchors[k]=mask
        if not supported:
            reports.append(dict(component=k,used=False,reason='insufficient_two_sided_flanks',attempts=attempts));continue
        x=(e-center)/max(.050,6*width);design=np.column_stack((np.ones(n),x))
        coeff=np.linalg.lstsq(design[mask],ratio[mask],rcond=None)[0];response=design@coeff
        score=row['maximum_feature_log']**2*tapers[k]**2/(1+((e-center)/width)**2)
        severity[source[k]]+=score;response_sum+=score*response;response_weight+=score
        reports.append(dict(component=k,used=True,attempts=attempts,response_coefficients=coeff.tolist(),
            response_x_scale_ev=max(.050,6*width),anchor_rms_log=_rms((ratio-response)[mask])))
    alignment=np.divide(response_sum,response_weight,out=np.zeros(n),where=response_weight>0)
    reliability=1/(1+severity/.015**2);weights=reliability/reliability.sum(axis=0,keepdims=True)
    return weights,alignment,severity,anchors,reports


def generate(e,pn,*,family='conditional_local_weighted'):
    if family not in FAMILIES:raise ValueError('Unknown V7 family')
    e=np.array(e,float,copy=True);pn=np.array(pn,float,copy=True)
    if e.ndim!=1 or len(e)<20 or not np.all(np.isfinite(e)) or not np.all(np.diff(e)>0):raise ValueError('Increasing native energy required')
    if pn.shape!=(2,len(e)) or not np.all(np.isfinite(pn)&(pn>0)):raise ValueError('Positive finite PN required')
    logs=np.log(pn);raw_ratio=np.log(pn[0]/pn[1]);ratio=np.round(raw_ratio,12)
    baseline_log=logs.mean(axis=0);baseline=np.exp(baseline_log);n=len(e)
    automatic=_automatic_support(e,logs)
    events,detection,smoothed,edge=_detect(e,ratio)
    if events:
        components,nuisance,model,error,records,diag=_fit(e,ratio,events)
        source,tapers,attribution_anchors,source_local_residuals=_attribute(e,logs,components,records,automatic,error)
    else:
        components=tapers=np.empty((0,n));source=np.empty(0,int);records=[]
        attribution_anchors=np.empty((0,n),bool);source_local_residuals=np.empty((0,2,n))
        nuisance=np.full(n,float(np.median(ratio)));model=nuisance.copy();error=ratio-model
        diag=dict(selected_start=None,starts=[],differential_residual_rms_log=_rms(error))
    source_full=np.zeros_like(pn);source_local=np.zeros_like(pn);uncertain=np.zeros(n,bool)
    for k,row in enumerate(records):
        row['applied']=bool(row['fit_usable'] and source[k]>=0)
        if row['applied']:
            sign=1. if source[k]==0 else -1.
            source_full[source[k]]+=sign*components[k]
            source_local[source[k]]+=sign*components[k]*tapers[k]
        else:uncertain|=tapers[k]>0
    local_corrected=logs-source_local
    weighted,alignment,severity,weight_anchors,weight_reports=_reliability(e,local_corrected,components,records,source,tapers,error,automatic)
    weights=np.full_like(pn,.5)
    if family=='conditional_local_weighted':
        source_feature=source_local;weights=weighted
        output_log=np.sum(weights*(local_corrected+np.array([-.5,.5])[:,None]*alignment),axis=0)
        repair_mask=np.any(source_local!=0,axis=0)
    else:
        source_feature=source_full if family=='conditional_full_tail' else -source_full[::-1].copy()
        output_log=(logs-source_feature).mean(axis=0)
        repair_mask=np.any(source_feature!=0,axis=0)
    delta=output_log-baseline_log;background=np.exp(output_log);background[~repair_mask]=baseline[~repair_mask]
    corrected=logs-source_feature;source_residual=corrected-output_log[None,:]
    reconstruction=np.exp(output_log[None,:]+source_residual+source_feature)
    diag.update(family=family,status='experimental_conditional_PN_only_not_accepted',settings=SETTINGS,
        components=records,detection=detection,flank_calibration=weight_reports,
        automatic_support_segments=automatic['segments'],
        automatic_source_reason_codes={'0':'conditionally_smooth_usable','1':'fringe_compatible_usable',
            '2':'source_medium_anomaly','3':'isolated_fine_or_unverified_oscillation','4':'common_localized_structure_unknown'},
        automatic_source_confidence_levels={'0':'insufficient_or_anomalous','1':'conditional_multiscale_support','2':'conditional_support_with_repeated_fringe_evidence'},
        automatic_common_class_codes={'0':'no_detected_common_localized_anomaly','1':'fringe_compatible','2':'common_localized_structure_unknown'},
        automatic_support_interpretation='Conditional observable evidence, not a claim that weak common features are absent. No manually supplied clean interval.',
        detected_component_count=len(records),applied_component_count=sum(r['applied'] for r in records),
        input_hashes={'energy':_sha_array(e),'pn':_sha_array(pn)},
        ratio_quantization_max_error_log=float(np.max(abs(raw_ratio-ratio))),
        source_reconstruction_max_relative_error=float(np.max(abs(reconstruction/pn-1))),
        source_reconstruction_interpretation='Bookkeeping closure including source residual; not physical validation.',
        repair_fraction_quantiles={str(q):float(np.quantile(abs(background/baseline-1),q)) for q in (.5,.95,.99,1.)},
        BN_used=False,historical_BN_used=False,sample_used=False,
        source_attribution='Conditional same-flank source residual projection; no universal uniqueness.',
        common_feature_identifiability='Shared equal resonances cannot be identified; retained or marked unresolved.',
        shared_change_invariance='Ratio is numerically canonicalized. Source attribution can change under shared structures overlapping its local basis; finite tests required.',
        physical_metadata='Same sample and position, only gate/doping changed, per user.',
        no_global_response_correction=True,legacy_hard_seams_removed=True,
        limitations=['Analytic tails and local source envelopes can overlap true broad response.',
            'A low PN residual or reconstruction closure does not establish physical truth.',
            'Full-tail and opposite attribution are sensitivity records, not quality-selected alternatives.',
            'Full dRR and full d2E are parallel conditions with unchanged evaluation.'])
    return dict(energy=e,background=background,raw_geometric_mean=baseline,
        corrected_sources=np.exp(corrected),source_feature_log=source_feature,
        conditional_full_source_feature_log=source_full,conditional_local_source_feature_log=source_local,
        source_weights=weights,weights=weights.copy(),reliability_source_weights=weighted,
        flank_alignment_log=alignment,source_severity_log_squared=severity,
        component_feature_log=components,component_tapers=tapers,
        component_source=source,component_applied=np.array([r['applied'] for r in records],bool),
        attribution_anchor_masks=attribution_anchors,source_local_projection_residuals=source_local_residuals,
        flank_anchor_masks=weight_anchors,repair_log_delta=delta,repair_mask=repair_mask,
        modification_mask=repair_mask.copy(),no_repair_mask=~repair_mask,uncertain_mask=uncertain,
        common_feature_unidentifiable_mask=automatic['common_unknown'],
        source_usable_masks=automatic['usable'],featureless_candidate_masks=automatic['usable'].copy(),
        source_usable_reason_codes=automatic['reasons'],source_usage_code=automatic['source_usage_code'],
        source_confidence_level=np.where(automatic['usable'],np.where(automatic['fringe_compatible'][None,:],2,1),0).astype(np.int8),
        common_structure_class=automatic['common_class'],fringe_compatible_mask=automatic['fringe_compatible'],
        fringe_period_ev=automatic['fringe_period_ev'],fringe_autocorrelation=automatic['fringe_autocorrelation'],
        fringe_source_correlation=automatic['fringe_source_correlation'],
        fit_support_mask=np.any(tapers>0,axis=0) if len(tapers) else np.zeros(n,bool),
        ratio_log=raw_ratio,normalized_ratio_log=ratio,smoothed_ratio_log=smoothed,ratio_edge_per_ev=edge,
        differential_nuisance_log=nuisance,differential_model_log=model,differential_residual_log=error,
        source_residual_log=source_residual,pn_reconstruction=reconstruction,seams=np.empty(0),diagnostics=diag)
