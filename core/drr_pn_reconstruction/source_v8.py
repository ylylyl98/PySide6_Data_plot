"""V8 current-PN-only grouped source attribution, with auditable usable support."""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "cf8a35f3ffd6e9448fecd1e45aa9744cc521caa193fc3f9b6cb1d204f07ba5e9"
import copy
import hashlib
import json
import numpy as np

from . import source_v7 as _v7
V7_SHA256 = _v7.MIGRATED_FROM_SHA256
V6_SHA256=_v7.V6_SHA256
FAMILIES=_v7.FAMILIES
_rms=_v7._rms
_sha_array=_v7._sha_array
SETTINGS=dict(**_v7.SETTINGS,maximum_prune_rounds=3,
    common_amplitude_ratio_minimum=.5,common_local_correlation_minimum=.8,
    common_correlation_radius_ev=.010,group_anchor_extensions_ev=[.025,.050,.090,.140,.180],
    group_column_condition_maximum=1e4,paired_split_selection='unweighted_native_PN_residual_BIC')


def _automatic_support(e,logs):
    """Observable conditional support, including periodic prediction residuals."""
    previous=_v7._automatic_support(e,logs)
    fine=previous['source_fine'];medium=previous['source_medium']
    q35=_v7._quad(e,logs,.035);texture=logs-q35
    period=np.mean(previous['fringe_period_ev'],axis=0)
    compatible=previous['fringe_compatible'].copy()
    periodic_supported=compatible&(e-period>=e[0])&(e+period<=e[-1])
    periodic_residual=fine.copy();periodic_prediction=np.zeros_like(logs)
    for s in range(2):
        predicted=(np.interp(e-period,e,texture[s])+np.interp(e+period,e,texture[s]))/2
        periodic_prediction[s]=predicted
        periodic_residual[s,periodic_supported]=(texture[s]-predicted)[periodic_supported]
    unexplained=periodic_residual
    local_correlation=np.zeros((2,len(e)))
    shared_evidence=np.zeros((2,len(e)),bool)
    for band,values in enumerate((medium,unexplained)):
        threshold=.010 if band==0 else .003
        absolute=abs(values)
        point=(np.min(absolute,axis=0)>threshold)&(values[0]*values[1]>0)&(np.min(absolute,axis=0)>=.5*np.max(absolute,axis=0))
        for i,center in enumerate(e):
            mask=abs(e-center)<=.010
            a=values[0,mask]-np.mean(values[0,mask]);b=values[1,mask]-np.mean(values[1,mask])
            den=np.linalg.norm(a)*np.linalg.norm(b)
            local_correlation[band,i]=np.dot(a,b)/den if den>1e-24 else 0.
        shared_evidence[band]=point&(local_correlation[band]>=.8)
    common_unknown=shared_evidence.any(axis=0)
    usable=(abs(medium)<=.015)&(abs(unexplained)<=.003)&(~common_unknown[None,:])
    reasons=np.zeros_like(logs,dtype=np.int8)
    reasons[abs(medium)>.015]=2;reasons[abs(unexplained)>.003]=3
    reasons[:,common_unknown]=4
    reasons[usable&periodic_supported[None,:]]=1
    common_class=np.zeros(len(e),np.int8)
    common_class[periodic_supported]=1;common_class[common_unknown]=2
    code=usable[0].astype(np.int8)+2*usable[1].astype(np.int8)
    starts=np.r_[0,np.flatnonzero(np.diff(code))+1];ends=np.r_[starts[1:],len(e)]
    segments=[dict(xmin_ev=float(e[a]),xmax_ev=float(e[b-1]),source_code=int(code[a]),
        label={0:'unknown_or_no_usable_source',1:'P_conditionally_usable',2:'N_conditionally_usable',3:'both_conditionally_usable'}[int(code[a])],
        evidence='per-source multiscale and repeat-prediction residual; not physical proof') for a,b in zip(starts,ends)]
    previous.update(usable=usable,reasons=reasons,common_class=common_class,common_unknown=common_unknown,
        fringe_compatible=periodic_supported,source_usage_code=code,segments=segments,
        periodic_prediction=periodic_prediction,periodic_prediction_residual=periodic_residual,
        periodic_prediction_supported=periodic_supported,common_evidence=shared_evidence,
        common_local_correlation=local_correlation)
    return previous


def _split_events(detection):
    rows=sorted(sorted(detection['retained_edges'],key=lambda r:-abs(r['edge_per_ev']))[:8],key=lambda r:r['energy_ev'])
    return [dict(energy_ev=r['energy_ev'],kind='split_asymmetric_edge',
        edge_indices=[r['index']],edge_energies_ev=[r['energy_ev']],edge_per_ev=r['edge_per_ev'],
        strength=abs(r['edge_per_ev']),initial_width_ev=.003,center_radius_ev=.006,
        width_upper_ev=.020,initial_q=-.3 if r['edge_per_ev']<0 else .3,initial_b=-.3) for r in rows]


def _fit_selected(e,ratio):
    paired,detection,smoothed,edge=_v7._detect(e,ratio)
    choices=[]
    for name,initial in (('paired',paired),('split',_split_events(detection))):
        events=copy.deepcopy(initial);history=[]
        for stage in range(4):
            result=_v7._fit(e,ratio,events)
            components,nuisance,model,error,records,diag=result
            rejected=[i for i,r in enumerate(records) if not r['fit_usable']]
            history.append(dict(stage=stage,event_count=len(events),rejected_indices=rejected,
                components=copy.deepcopy(records),fit=copy.deepcopy(diag)))
            if not rejected or stage==3 or not events:break
            events=[r for i,r in enumerate(events) if i not in rejected]
        score=float(len(e)*np.log(max(np.mean(error**2),1e-24))+(4*len(events)+3)*np.log(len(e)))
        choices.append(dict(name=name,score=score,result=result,history=history))
    chosen=min(range(len(choices)),key=lambda i:(choices[i]['score'],i))
    result=choices[chosen]['result'];diag=result[-1]
    diag.update(selected_decomposition=choices[chosen]['name'],decomposition_selection=[
        dict(name=c['name'],bic=c['score'],history=c['history']) for c in choices])
    return (*result,detection,smoothed,edge)


def _groups(records):
    intervals=[]
    for k,r in enumerate(records):
        if not r['fit_usable']:continue
        c,w,unused,unused2=r['parameters'];radius=max(.050,6*w)
        intervals.append((c-radius,c+radius,k))
    intervals.sort();groups=[]
    for left,right,k in intervals:
        if not groups or left>groups[-1]['support'][1]:groups.append(dict(support=[left,right],components=[k]))
        else:
            groups[-1]['support'][1]=max(right,groups[-1]['support'][1]);groups[-1]['components'].append(k)
    for group in groups:
        group['core']=[min(records[k]['parameters'][0]-max(.020,2*records[k]['parameters'][1]) for k in group['components']),
                       max(records[k]['parameters'][0]+max(.020,2*records[k]['parameters'][1]) for k in group['components'])]
    return groups


def _attribute(e,logs,components,records,automatic,error):
    count,n=components.shape;anchors=np.zeros((count,n),bool);source=np.full(count,-1,int)
    tapers=np.zeros_like(components);projection_residuals=np.zeros((count,2,n));reports=[]
    for k,row in enumerate(records):
        c,w,unused,unused2=row['parameters'];inner=max(.025,3*w);outer=max(.050,6*w)
        tapers[k]=_v7._v6._quintic((outer-abs(e-c))/(outer-inner))
        row.update(source=-1,attribution='unresolved',attribution_reason='unsupported_differential_component',
            support_ev=[float(c-outer),float(c+outer)],plateau_ev=[float(c-inner),float(c+inner)])
    groups=_groups(records)
    for group_id,group in enumerate(groups):
        indices=group['components'];left,right=group['core'];center=(left+right)/2
        scale=max(.050,(right-left)/2);x=(e-center)/scale;design=np.column_stack((np.ones(n),x,x*x))
        core=(e>=left)&(e<=right)
        excluded=(~automatic['usable'].all(axis=0))|(abs(error)>.010)
        for other in groups:
            if other is group:continue
            excluded|=(e>=other['core'][0])&(e<=other['core'][1])
        attempts=[];supported=False;mask=np.zeros(n,bool)
        for extension in (.025,.050,.090,.140,.180):
            candidate=((e>=left-extension)&(e<left))|((e>right)&(e<=right+extension))
            mask=candidate&(~excluded)
            for refinement in range(3):
                counts=[int(np.sum(mask&(e<left))),int(np.sum(mask&(e>right)))]
                if min(counts)<12:break
                coeff=np.linalg.lstsq(design[mask],logs[:,mask].T,rcond=None)[0]
                quality=np.all(abs(automatic['q4logs']-(design@coeff).T)<=.010,axis=0)
                new=mask&quality
                if np.array_equal(new,mask):supported=True;break
                mask=new
            counts=[int(np.sum(mask&(e<left))),int(np.sum(mask&(e>right)))]
            if min(counts)>=12:
                coeff=np.linalg.lstsq(design[mask],logs[:,mask].T,rcond=None)[0]
                supported=bool(np.all(abs(automatic['q4logs'][:,mask]-(design@coeff).T[:,mask])<=.010))
            attempts.append(dict(extension_ev=extension,left_points=counts[0],right_points=counts[1],supported=supported))
            if supported:break
        report=dict(group=group_id,components=indices,core_ev=group['core'],support_ev=group['support'],core_span_ev=right-left,
            anchor_attempts=attempts,anchor_count=int(mask.sum()),supported=supported)
        reports.append(report)
        for k in indices:
            anchors[k]=mask;records[k].update(attribution_group=group_id,attribution_flank_attempts=attempts,
                attribution_reason='insufficient_group_two_sided_observed_support')
        if not supported:continue
        envelope=np.linalg.lstsq(design[mask],logs[:,mask].T,rcond=None)[0]
        local_residual=logs-(design@envelope).T
        basis=components[indices].T-design@np.linalg.lstsq(design[mask],components[indices][:,mask].T,rcond=None)[0]
        norms=np.linalg.norm(basis[core],axis=0)
        normalized=np.divide(basis[core],norms[None,:],out=np.zeros_like(basis[core]),where=norms[None,:]>1e-16)
        singular=np.linalg.svd(normalized,compute_uv=False)
        condition=float(singular[0]/singular[-1]) if len(singular) and singular[-1]>1e-16 else None
        report.update(column_normalized_condition=condition,source_envelope_coefficients=envelope.tolist(),
            source_envelope_x_center_ev=center,source_envelope_x_scale_ev=scale,
            source_envelope_anchor_residual_rms_log=[_rms((automatic['q4logs']-(design@envelope).T)[s,mask]) for s in range(2)])
        if condition is None or condition>1e4 or np.min(norms)<=1e-16:
            report['rejection']='degenerate_group_projection'
            for k in indices:records[k]['attribution_reason']='degenerate_group_projection'
            continue
        coeff=np.linalg.lstsq(basis[core],local_residual[:,core].T,rcond=None)[0].T
        report['source_projection_coefficients']=coeff.tolist()
        report['joint_projection_residual_rms_log']=[_rms(local_residual[s,core]-basis[core]@coeff[s]) for s in range(2)]
        for j,k in enumerate(indices):
            cp,cn=map(float,coeff[:,j]);projection_residuals[k]=local_residual
            if cp>=.6 and abs(cn)<=.5*abs(cp):source[k]=0
            elif cn<=-.6 and abs(cp)<=.5*abs(cn):source[k]=1
            margin=min(cp-.6,.5*abs(cp)-abs(cn)) if source[k]==0 else min(-cn-.6,.5*abs(cn)-abs(cp)) if source[k]==1 else 0.
            records[k].update(source=int(source[k]),source_projection_coefficients=[cp,cn],
                source_envelope_coefficients=envelope.tolist(),attribution_margin=float(margin),
                attribution='P' if source[k]==0 else 'N' if source[k]==1 else 'unresolved',
                attribution_reason='conditional_group_source_projection' if source[k]>=0 else 'both_or_neither_source_compatible')
    return source,tapers,anchors,projection_residuals,reports


def _reliability(e,corrected,components,records,source,tapers,attribution_anchors):
    n=len(e);severity=np.zeros((2,n));response_sum=np.zeros(n);response_weight=np.zeros(n)
    anchors=np.zeros_like(components,dtype=bool);reports=[];ratio=np.round(corrected[0]-corrected[1],12)
    for k,row in enumerate(records):
        if not row['applied']:
            reports.append(dict(component=k,used=False,reason='component_not_applied'));continue
        center,width,unused,unused2=row['parameters'];mask=attribution_anchors[k].copy();anchors[k]=mask
        x=(e-center)/max(.050,6*width);design=np.column_stack((np.ones(n),x))
        coeff=np.linalg.lstsq(design[mask],ratio[mask],rcond=None)[0];response=design@coeff
        score=row['maximum_feature_log']**2*tapers[k]**2/(1+((e-center)/width)**2)
        severity[source[k]]+=score;response_sum+=score*response;response_weight+=score
        reports.append(dict(component=k,used=True,source='accepted_group_attribution_anchors',
            response_coefficients=coeff.tolist(),response_x_scale_ev=max(.050,6*width),
            anchor_count=int(mask.sum()),anchor_rms_log=_rms((ratio-response)[mask])))
    alignment=np.divide(response_sum,response_weight,out=np.zeros(n),where=response_weight>0)
    reliability=1/(1+severity/.015**2);weights=reliability/reliability.sum(axis=0,keepdims=True)
    return weights,alignment,severity,anchors,reports


def fit_bundle(e,pn):
    """Fit current PN once. No BN or sample parameter is accepted."""
    e=np.array(e,float,copy=True);pn=np.array(pn,float,copy=True)
    if e.ndim!=1 or len(e)<20 or not np.all(np.isfinite(e)) or not np.all(np.diff(e)>0):raise ValueError('Increasing native energy required')
    if pn.shape!=(2,len(e)) or not np.all(np.isfinite(pn)&(pn>0)):raise ValueError('Positive finite PN required')
    logs=np.log(pn);raw_ratio=np.log(pn[0]/pn[1]);ratio=np.round(raw_ratio,12)
    automatic=_automatic_support(e,logs)
    components,nuisance,model,error,records,diag,detection,smoothed,edge=_fit_selected(e,ratio)
    source,tapers,attribution_anchors,source_local_residuals,groups=_attribute(e,logs,components,records,automatic,error)
    source_full=np.zeros_like(pn);source_local=np.zeros_like(pn);uncertain=automatic['common_unknown'].copy()
    for k,row in enumerate(records):
        row['applied']=bool(row['fit_usable'] and source[k]>=0)
        if row['applied']:
            sign=1. if source[k]==0 else -1.
            source_full[source[k]]+=sign*components[k]
            source_local[source[k]]+=sign*components[k]*tapers[k]
        else:uncertain|=tapers[k]>0
    weighted,alignment,severity,weight_anchors,weight_reports=_reliability(e,logs-source_local,components,records,source,tapers,attribution_anchors)
    identity=dict(energy_hash=_sha_array(e),pn_hash=_sha_array(pn),component_hash=_sha_array(components),
        source_hash=_sha_array(source),taper_hash=_sha_array(tapers),nuisance_hash=_sha_array(nuisance),
        anchors_hash=_sha_array(attribution_anchors),selected_decomposition=diag['selected_decomposition'])
    model_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    diag.update(components=records,detection=detection,attribution_groups=groups,
        flank_calibration=weight_reports,model_sha256=model_sha,model_hash_inputs=identity,
        detected_component_count=len(records),applied_component_count=sum(r['applied'] for r in records))
    return dict(e=e,pn=pn,logs=logs,raw_ratio=raw_ratio,ratio=ratio,automatic=automatic,
        components=components,nuisance=nuisance,model=model,error=error,records=records,diag=diag,
        smoothed=smoothed,edge=edge,source=source,tapers=tapers,attribution_anchors=attribution_anchors,
        source_local_residuals=source_local_residuals,source_full=source_full,source_local=source_local,
        uncertain=uncertain,weighted=weighted,alignment=alignment,severity=severity,
        weight_anchors=weight_anchors,model_sha=model_sha)


def render(bundle,*,family='conditional_local_weighted'):
    if family not in FAMILIES:raise ValueError('Unknown V8 family')
    b=bundle;e=b['e'];pn=b['pn'];logs=b['logs'];automatic=b['automatic'];records=b['records']
    baseline_log=logs.mean(axis=0);baseline=np.exp(baseline_log);n=len(e);weights=np.full_like(pn,.5)
    if family=='conditional_local_weighted':
        source_feature=b['source_local'];weights=b['weighted']
        output_log=np.sum(weights*(logs-source_feature+np.array([-.5,.5])[:,None]*b['alignment']),axis=0)
    else:
        source_feature=b['source_full'] if family=='conditional_full_tail' else -b['source_full'][::-1].copy()
        output_log=(logs-source_feature).mean(axis=0)
    repair_mask=np.any(source_feature!=0,axis=0);delta=output_log-baseline_log
    background=np.exp(output_log);background[~repair_mask]=baseline[~repair_mask]
    corrected=logs-source_feature;source_residual=corrected-output_log[None,:]
    reconstruction=np.exp(output_log[None,:]+source_residual+source_feature)
    diag=copy.deepcopy(b['diag'])
    diag.update(family=family,status='experimental_conditional_PN_only_not_accepted',settings=copy.deepcopy(SETTINGS),
        automatic_support_segments=copy.deepcopy(automatic['segments']),
        automatic_source_reason_codes={'0':'conditionally_smooth_usable','1':'repeated_fringe_usable_after_isolated_residual_check',
            '2':'source_medium_anomaly','3':'isolated_fine_or_unverified_oscillation','4':'common_localized_structure_unknown'},
        automatic_source_confidence_levels={'0':'insufficient_or_anomalous','1':'conditional_multiscale_support','2':'conditional_support_with_repeated_fringe_evidence'},
        automatic_common_class_codes={'0':'no_detected_common_localized_anomaly','1':'fringe_compatible','2':'common_localized_structure_unknown'},
        automatic_support_interpretation='Observable per-source evidence, not proof of no weak common feature; actual anchors consume these masks.',
        input_hashes={'energy':_sha_array(e),'pn':_sha_array(pn)},
        ratio_quantization_max_error_log=float(np.max(abs(b['raw_ratio']-b['ratio']))),
        source_reconstruction_max_relative_error=float(np.max(abs(reconstruction/pn-1))),
        source_reconstruction_interpretation='Bookkeeping closure including source residual; not physical validation.',
        repair_fraction_quantiles={str(q):float(np.quantile(abs(background/baseline-1),q)) for q in (.5,.95,.99,1.)},
        BN_used=False,historical_BN_used=False,sample_used=False,manual_clean_ranges_used=False,
        source_attribution='Conditional same-flank simultaneous group source residual projection; no universal uniqueness.',
        common_feature_identifiability='Equal common resonances cannot be uniquely separated; observable common anomalies are marked unknown.',
        shared_change_invariance='Rounded PN ratio fit is invariant; conditional source attribution requires finite envelope/fringe checks.',
        physical_metadata='Same sample and position, only gate/doping changed, per user.',
        no_global_response_correction=True,legacy_hard_seams_removed=True,
        limitations=['Analytic tails and local quadratic envelopes can overlap true broad response.',
            'Low PN residual, source reconstruction closure, or unchanged fallback do not establish accuracy.',
            'Full-tail and opposite attribution are sensitivity records, not quality-selected alternatives.',
            'Full dRR and full d2E are parallel conditions with unchanged external evaluation.'])
    result=dict(energy=e,background=background,raw_geometric_mean=baseline,
        corrected_sources=np.exp(corrected),source_feature_log=source_feature,
        conditional_full_source_feature_log=b['source_full'],conditional_local_source_feature_log=b['source_local'],
        source_weights=weights,weights=weights,reliability_source_weights=b['weighted'],
        flank_alignment_log=b['alignment'],source_severity_log_squared=b['severity'],
        component_feature_log=b['components'],component_tapers=b['tapers'],component_source=b['source'],
        component_applied=np.array([r['applied'] for r in records],bool),
        attribution_anchor_masks=b['attribution_anchors'],source_local_projection_residuals=b['source_local_residuals'],
        flank_anchor_masks=b['weight_anchors'],repair_log_delta=delta,repair_mask=repair_mask,
        modification_mask=repair_mask,no_repair_mask=~repair_mask,uncertain_mask=b['uncertain'],
        common_feature_unidentifiable_mask=automatic['common_unknown'],
        source_usable_masks=automatic['usable'],featureless_candidate_masks=automatic['usable'],
        source_usable_reason_codes=automatic['reasons'],source_usage_code=automatic['source_usage_code'],
        source_confidence_level=np.where(automatic['usable'],np.where(automatic['fringe_compatible'][None,:],2,1),0).astype(np.int8),
        common_structure_class=automatic['common_class'],fringe_compatible_mask=automatic['fringe_compatible'],
        fringe_period_ev=automatic['fringe_period_ev'],fringe_autocorrelation=automatic['fringe_autocorrelation'],
        fringe_source_correlation=automatic['fringe_source_correlation'],
        source_periodic_prediction_log=automatic['periodic_prediction'],
        source_periodic_prediction_residual_log=automatic['periodic_prediction_residual'],
        periodic_prediction_support_mask=automatic['periodic_prediction_supported'],
        common_anomaly_evidence=automatic['common_evidence'],common_local_correlation=automatic['common_local_correlation'],
        fit_support_mask=np.any(b['tapers']>0,axis=0) if len(b['tapers']) else np.zeros(n,bool),
        ratio_log=b['raw_ratio'],normalized_ratio_log=b['ratio'],smoothed_ratio_log=b['smoothed'],ratio_edge_per_ev=b['edge'],
        differential_nuisance_log=b['nuisance'],differential_model_log=b['model'],differential_residual_log=b['error'],
        source_residual_log=source_residual,pn_reconstruction=reconstruction,seams=np.empty(0),
        model_sha256=np.array(b['model_sha']))
    result={key:value.copy() for key,value in result.items()};result['diagnostics']=diag
    return result


def generate_all(e,pn):
    bundle=fit_bundle(e,pn)
    return {family:render(bundle,family=family) for family in FAMILIES}


def generate(e,pn,*,family='conditional_local_weighted'):
    if family not in FAMILIES:raise ValueError('Unknown V8 family')
    return render(fit_bundle(e,pn),family=family)
