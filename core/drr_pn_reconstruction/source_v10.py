"""V10 independent PN-only ablations; no composition or external-score selection."""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "d19d764527c6ad26366d3a74ea44b21989eddcb9defbcf9521b0b6909b1c1f12"
import copy
import hashlib
import json
import numpy as np

from . import source_v9 as _v9
V9_SHA256 = _v9.MIGRATED_FROM_SHA256
_v7=_v9._v7
V8_SHA256=_v9.V8_SHA256;V7_SHA256=_v9.V7_SHA256;V6_SHA256=_v9.V6_SHA256
FAMILIES=('v9_binary_full_tail','shared_amplitude_full_tail','pair_candidates_full_tail','iterated_support_full_tail')
_rms=_v9._rms;_sha_array=_v9._sha_array;_automatic_support=_v9._automatic_support
SETTINGS=dict(base_v9_settings=_v9.SETTINGS,shared_amplitude_rule='same-sign interval point nearest zero',
    pair_alternatives='one eligible adjacent merger at a time; no combinations',
    pair_separation_ev=[.008,.060],pair_old_amplitude_cutoff=.30,
    iteration_count=1,strong_maximum_feature_log_minimum=.20,strong_width_ev_maximum=.010,
    source_assignment_thresholds_unchanged=True,BN_or_sample_selection=False)


def _binary_coefficients(records):
    coeff=np.zeros((2,len(records)))
    for k,row in enumerate(records):
        if row['applied']:coeff[row['source'],k]=1. if row['source']==0 else -1.
    return coeff


def _shared(base):
    coeff=_binary_coefficients(base['records']);deltas=np.zeros((2,len(base['records'])))
    chosen=np.zeros(len(base['records']));reports=[]
    projection=base['projection_arrays']['attribution_coefficients_by_scale_source']
    for k,row in enumerate(base['records']):
        if not row['applied']:
            reports.append(dict(component=k,used=False,reason='V9_component_not_applied'));continue
        m0=.5 if row['source']==0 else -.5
        estimates=projection[:,:,k].mean(axis=1)-m0
        deltas[:,k]=estimates
        if estimates[0]*estimates[1]>0:
            chosen[k]=estimates[int(np.argmin(abs(estimates)))]
        reports.append(dict(component=k,used=bool(chosen[k]!=0),binary_common_coefficient=m0,
            raw_source_coefficients_by_scale=projection[:,:,k].tolist(),
            source_difference_coefficients_by_scale=(projection[:,0,k]-projection[:,1,k]).tolist(),
            common_delta_by_scale=estimates.tolist(),common_delta_interval=[float(estimates.min()),float(estimates.max())],
            common_delta_midpoint=float(estimates.mean()),common_delta_span=float(np.ptp(estimates)),
            selected_delta=float(chosen[k]),
            reason='same_direction_minimum_supported_amplitude' if chosen[k]!=0 else 'scale_interval_contains_zero'))
    updated=coeff+chosen[None,:]
    source_full=updated@base['components']
    np.testing.assert_allclose(source_full[0]-source_full[1],base['source_full'][0]-base['source_full'][1],rtol=0,atol=2e-15)
    bundle=dict(base);bundle['diag']=copy.deepcopy(base['diag']);bundle['source_full']=source_full
    bundle['source_local']=(updated[:,:,None]*base['components'][None,:,:]*base['tapers'][None,:,:]).sum(axis=1)
    extra=dict(shared_delta_by_scale=deltas,shared_delta_selected=chosen,
        shared_component_correction_log=chosen[:,None]*base['components'],
        component_source_coefficients_binary=coeff,component_source_coefficients_output=updated,
        shared_delta_eligible=np.array([r['applied'] for r in base['records']],bool))
    bundle['diag'].update(shared_amplitude_components=reports,
        shared_amplitude_uncertainty='Two coarse nuisance-scale estimates only; not a statistical confidence interval.',
        shared_amplitude_identifiability='Conditional same-shape common residual linked to a differential feature; purely equal PN structure is not identified.',
        continuous_common_amplitude_may_break_equivariance=True)
    return bundle,extra


def _pair_alternatives(base):
    events=copy.deepcopy(base['diag']['detection']['events'])
    edges=base['diag']['detection']['retained_edges'];e=base['e'];smooth=base['smoothed']
    singles={r['edge_indices'][0]:k for k,r in enumerate(events) if len(r['edge_indices'])==1}
    proposals=[]
    for a,b in zip(edges[:-1],edges[1:]):
        separation=b['energy_ev']-a['energy_ev']
        relative=min(abs(a['edge_per_ev']),abs(b['edge_per_ev']))/max(abs(a['edge_per_ev']),abs(b['edge_per_ev']))
        if not (a['edge_per_ev']*b['edge_per_ev']<0 and .008<=separation<=.060 and relative<.30):continue
        if a['index'] not in singles or b['index'] not in singles:continue
        inside=np.flatnonzero((e>=a['energy_ev'])&(e<=b['energy_ev']))
        index=inside[np.argmax(smooth[inside]) if a['edge_per_ev']>0 else np.argmin(smooth[inside])]
        width=.866*separation
        event=dict(energy_ev=float(e[index]),kind='additional_opposite_pair',
            edge_indices=[a['index'],b['index']],edge_energies_ev=[a['energy_ev'],b['energy_ev']],
            edge_per_ev=a['edge_per_ev'],strength=max(abs(a['edge_per_ev']),abs(b['edge_per_ev'])),
            initial_width_ev=width,center_radius_ev=max(.006,.25*separation),
            width_upper_ev=min(.060,max(.020,2*width)),initial_q=0.,
            initial_b=.2 if a['edge_per_ev']>0 else -.2)
        replace={singles[a['index']],singles[b['index']]}
        alternative=[r for k,r in enumerate(copy.deepcopy(events)) if k not in replace]+[event]
        alternative.sort(key=lambda r:r['energy_ev'])
        proposals.append(dict(name=f"extra_pair_{a['index']}_{b['index']}",events=alternative,
            replaced_event_indices=sorted(replace),edge_amplitude_ratio=relative,
            edge_energies_ev=event['edge_energies_ev']))
    return proposals


def _assembled(base,fit,name,selection):
    components,nuisance,model,error,records,diag=fit
    e=base['e'];logs=base['logs'];automatic=base['automatic']
    source,tapers,projection_reports,projection_arrays=_v9._projection(e,logs,components,records,automatic)
    source_full=np.zeros_like(logs);source_local=np.zeros_like(logs);uncertain=automatic['common_unknown'].copy()
    for k,row in enumerate(records):
        row['applied']=bool(row['fit_usable'] and source[k]>=0)
        if row['applied']:
            sign=1. if source[k]==0 else -1.
            source_full[source[k]]+=sign*components[k];source_local[source[k]]+=sign*components[k]*tapers[k]
        else:uncertain|=tapers[k]>0
    weighted,alignment,severity,weight_anchors,weight_reports=_v7._reliability(e,logs-source_local,components,records,source,tapers,error,automatic)
    identity=dict(energy_hash=_sha_array(e),pn_hash=_sha_array(base['pn']),component_hash=_sha_array(components),
        source_hash=_sha_array(source),taper_hash=_sha_array(tapers),nuisance_hash=_sha_array(nuisance),
        projection_coefficients_hash=_sha_array(projection_arrays['attribution_coefficients_by_scale_source']),
        selected_decomposition=name)
    model_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    diag.update(components=records,detection=copy.deepcopy(base['diag']['detection']),
        selected_decomposition=name,pair_candidate_selection=selection,source_projection=projection_reports,
        flank_calibration=weight_reports,model_sha256=model_sha,model_hash_inputs=identity,
        detected_component_count=len(records),applied_component_count=sum(r['applied'] for r in records),
        source_projection_nuisance_used_in_output=False,full_differential_components_retained=True,
        source_projection_operator='Same native Q4 on target, differential features and cubic nuisance.')
    bundle=dict(base)
    bundle.update(components=components,nuisance=nuisance,model=model,error=error,records=records,diag=diag,
        source=source,tapers=tapers,attribution_anchors=np.zeros_like(components,bool),
        source_local_residuals=np.zeros((len(components),2,len(e))),source_full=source_full,
        source_local=source_local,uncertain=uncertain,weighted=weighted,alignment=alignment,severity=severity,
        weight_anchors=weight_anchors,model_sha=model_sha,projection_arrays=projection_arrays)
    return bundle


def _paired(base):
    n=len(base['e']);base_score=float(n*np.log(max(np.mean(base['error']**2),1e-24))+(4*len(base['records'])+3)*np.log(n))
    proposals=_pair_alternatives(base)
    selections=[dict(name='original_V9_selected',bic=base_score,
        original_decomposition=base['diag']['selected_decomposition'],
        components=copy.deepcopy(base['records']),fit=copy.deepcopy(base['diag']))]
    winner=None;score=base_score
    for proposal in proposals:
        fit=_v7._fit(base['e'],base['ratio'],proposal['events'])
        trial_score=float(n*np.log(max(np.mean(fit[3]**2),1e-24))+(4*len(fit[4])+3)*np.log(n))
        selections.append(dict(name=proposal['name'],bic=trial_score,proposal=proposal,
            components=copy.deepcopy(fit[4]),fit=copy.deepcopy(fit[5])))
        if trial_score<score:score=trial_score;winner=(proposal['name'],fit)
    if winner is None:
        bundle=dict(base);bundle['diag']=copy.deepcopy(base['diag'])
        bundle['diag']['pair_candidate_selection']=selections
    else:bundle=_assembled(base,winner[1],winner[0],selections)
    bundle['diag'].update(additional_pair_candidate_count=len(proposals),pair_mergers_per_candidate=1,
        selected_additional_pair=None if winner is None else winner[0],
        original_V9_model_sha256=base['model_sha'])
    return bundle,dict(component_source_coefficients_output=_binary_coefficients(bundle['records']))


def _iterated(base):
    count=len(base['records']);strong=[k for k,r in enumerate(base['records'])
        if r['applied'] and r['maximum_feature_log']>=.20 and r['parameters'][1]<.010]
    correction=np.zeros_like(base['logs'])
    for k in strong:
        source=base['records'][k]['source'];correction[source]+=(1. if source==0 else -1.)*base['components'][k]
    corrected=base['logs']-correction
    automatic=_automatic_support(base['e'],corrected)
    remaining=[k for k in range(count) if k not in strong]
    records_remaining=[copy.deepcopy(base['records'][k]) for k in remaining]
    source_new,tapers_unused,projection_reports,projection_arrays=_v9._projection(base['e'],corrected,
        base['components'][remaining],records_remaining,automatic)
    bundle=dict(base);records=copy.deepcopy(base['records']);sources=base['source'].copy()
    source_full=base['source_full'].copy();source_local=base['source_local'].copy();additions=[]
    for j,k in enumerate(remaining):
        if records[k]['applied'] or not records[k]['fit_usable'] or source_new[j]<0:continue
        row=records_remaining[j];source=int(source_new[j]);sign=1. if source==0 else -1.
        center,width,unused,unused2=records[k]['parameters'];core=abs(base['e']-center)<=max(.004,2*width)
        donor=1-source
        additions.append(dict(component=k,source=source,
            reason='Previously unresolved fit-supported component agrees at both coarse scales after one strong-only correction.',
            donor_core_usable_before_fraction=float(np.mean(base['automatic']['usable'][donor,core])),
            donor_core_usable_after_fraction=float(np.mean(automatic['usable'][donor,core])),
            source_core_usable_before_fraction=float(np.mean(base['automatic']['usable'][source,core])),
            source_core_usable_after_fraction=float(np.mean(automatic['usable'][source,core])),
            projection_coefficients=row['source_projection_coefficients'],
            attribution_margin_by_scale=row['attribution_margin_by_scale']))
        sources[k]=source;source_full[source]+=sign*base['components'][k]
        source_local[source]+=sign*base['components'][k]*base['tapers'][k]
        records[k]=row;records[k].update(applied=True,attribution_reason='one_iteration_source_stable_after_strong_only_correction')
    uncertain=automatic['common_unknown'].copy()
    for k,row in enumerate(records):
        if not row['applied']:uncertain|=base['tapers'][k]>0
    diag=copy.deepcopy(base['diag'])
    diag.update(components=records,applied_component_count=sum(r['applied'] for r in records),
        support_iteration=dict(rounds=1,strong_component_indices=strong,remaining_component_indices=remaining,
            strong_definition='V9 applied, maximum abs feature>=.20log, fitted width<.010eV',
            projection_reports=projection_reports,additions=additions,
            conditional_support_warning='Self-corrected support is conditional model evidence, not proof of a physically featureless donor.'),
        source_projection_initial=copy.deepcopy(base['diag']['source_projection']))
    bundle.update(records=records,source=sources,source_full=source_full,source_local=source_local,
        automatic=automatic,uncertain=uncertain,diag=diag)
    extra=dict(iteration_strong_component_indices=np.array(strong,int),
        iteration_remaining_component_indices=np.array(remaining,int),
        iteration_strong_source_correction_log=correction,iteration_corrected_source_logs=corrected,
        source_usable_masks_before_iteration=base['automatic']['usable'],
        source_usable_masks_after_iteration=automatic['usable'],
        source_reason_codes_before_iteration=base['automatic']['reasons'],
        source_reason_codes_after_iteration=automatic['reasons'],
        component_source_coefficients_output=_binary_coefficients(records))
    for key,value in projection_arrays.items():extra['iteration_'+key]=value
    return bundle,extra


def fit_bundle(e,pn):
    base=_v9.fit_bundle(e,pn)
    shared,shared_extra=_shared(base);paired,paired_extra=_paired(base);iterated,iterated_extra=_iterated(base)
    return dict(base=base,variants={
        'v9_binary_full_tail':(base,dict(component_source_coefficients_output=_binary_coefficients(base['records']))),
        'shared_amplitude_full_tail':(shared,shared_extra),
        'pair_candidates_full_tail':(paired,paired_extra),
        'iterated_support_full_tail':(iterated,iterated_extra)})


def render(bundle,*,family='shared_amplitude_full_tail'):
    if family not in FAMILIES:raise ValueError('Unknown V10 family')
    model,extra=bundle['variants'][family]
    result=_v9.render(model,family='conditional_full_tail')
    for key,value in extra.items():result[key]=value.copy()
    construction=dict(family=family,underlying_model_sha256=model['model_sha'],
        source_correction_sha256=_sha_array(model['source_full']))
    construction_sha=hashlib.sha256(json.dumps(construction,sort_keys=True).encode()).hexdigest()
    result['construction_sha256']=np.array(construction_sha)
    result['baseline_V9_model_sha256']=np.array(bundle['base']['model_sha'])
    result['diagnostics'].update(family=family,settings=copy.deepcopy(SETTINGS),
        underlying_V9_or_pair_model_sha256=model['model_sha'],construction=construction,
        construction_sha256=construction_sha,baseline_V9_model_sha256=bundle['base']['model_sha'],
        ablations_are_not_composed=True,output_construction='Full native feature subtraction and unweighted corrected-PN log mean; spline nuisance never output.',
        legacy_local_weight_fields_used_in_output=False)
    if family=='iterated_support_full_tail':
        result['diagnostics']['automatic_support_interpretation']='After exactly one correction using stable strong features; conditional support, not independently proven clean. Before/after arrays saved.'
    if family=='shared_amplitude_full_tail':
        result['diagnostics']['source_attribution']='Original V9 source identity plus conservatively shared template amplitude; source difference remains unchanged.'
    return result


def generate_all(e,pn):
    bundle=fit_bundle(e,pn)
    return {family:render(bundle,family=family) for family in FAMILIES}


def generate(e,pn,*,family='shared_amplitude_full_tail'):
    if family not in FAMILIES:raise ValueError('Unknown V10 family')
    return render(fit_bundle(e,pn),family=family)
