"""V11 matched bandpass source inference, with low-transfer broad terms protected."""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "a23c6700794e7ed287ed9af72c5c997fd0e342e608423192e4b78c7966b0be33"
import copy
import hashlib
import json
import numpy as np

from . import source_v10 as _v10
V10_SHA256 = _v10.MIGRATED_FROM_SHA256
_v9=_v10._v9;_v7=_v10._v7
V9_SHA256=_v10.V9_SHA256;V8_SHA256=_v10.V8_SHA256;V7_SHA256=_v10.V7_SHA256;V6_SHA256=_v10.V6_SHA256
FAMILIES=('v10_pair_binary_full_tail','bandpass_binary_full_tail','bandpass_shared_full_tail')
_rms=_v10._rms;_sha_array=_v10._sha_array;_automatic_support=_v10._automatic_support
SETTINGS=dict(differential_model='frozen V10 finite-pair PN BIC selection',
    matched_bandpass_widths_ev=[.025,.035],fine_operator_width_ev=.004,
    minimum_template_norm_transfer=.05,minimum_filtered_template_peak_log=.003,
    maximum_normalized_column_condition=1e4,usable_point_weight=4.,other_point_weight=1.,
    common_delta_rule='same-sign two-band interval point nearest zero',
    ineligible_broad_behavior='retain V10 binary assignment; no new amplitude inference',
    common_multiplier_acceptance_unchanged=True)


def _bandpass(base):
    e=base['e'];logs=base['logs'];features=base['components'];n=len(e);count=len(features)
    q4source=_v7._quad(e,logs,.004)
    q4feature=_v7._quad(e,features,.004) if count else features.copy()
    cell=np.gradient(e)
    weights=(1.+3*base['automatic']['usable'].astype(float))*n*cell[None,:]/cell.sum()
    source_targets=np.empty((2,2,n));feature_bands=np.empty((2,count,n))
    coefficients=np.zeros((2,2,count));residuals=np.empty((2,2,n));valid=np.ones((2,2),bool)
    transfers=np.zeros((2,count));peaks=np.zeros((2,count));classifications=np.full((2,count),-1,int);reports=[]
    q4norm=np.linalg.norm(q4feature,axis=1)
    for band,width in enumerate((.025,.035)):
        target=q4source-_v7._quad(e,logs,width)
        filtered=q4feature-_v7._quad(e,features,width) if count else features.copy()
        source_targets[band]=target;feature_bands[band]=filtered
        transfers[band]=np.divide(np.linalg.norm(filtered,axis=1),q4norm,out=np.zeros(count),where=q4norm>1e-16)
        peaks[band]=np.max(abs(filtered),axis=1) if count else np.empty(0)
        source_reports=[]
        for source in range(2):
            design=filtered.T*np.sqrt(weights[source])[:,None]
            response=target[source]*np.sqrt(weights[source])
            norms=np.linalg.norm(design,axis=0)
            normalized=np.divide(design,norms[None,:],out=np.zeros_like(design),where=norms[None,:]>1e-16)
            if count:
                scaled,unused,rank,singular=np.linalg.lstsq(normalized,response,rcond=None)
                pars=np.divide(scaled,norms,out=np.zeros_like(scaled),where=norms>1e-16)
                condition=float(singular[0]/singular[-1]) if singular[-1]>1e-16 else None
                valid[band,source]=condition is not None and condition<=1e4 and bool(np.all(norms>1e-16))
            else:pars=np.empty(0);rank=0;singular=np.empty(0);condition=1.
            coefficients[band,source]=pars
            residuals[band,source]=target[source]-filtered.T@pars
            source_reports.append(dict(source='P' if source==0 else 'N',coefficients=pars.tolist(),
                column_norms=norms.tolist(),normalized_singular_values=singular.tolist(),rank=int(rank),
                column_normalized_condition=condition,valid=bool(valid[band,source]),
                source_band_residual_rms_log=_rms(residuals[band,source]),
                source_band_target_rms_log=_rms(target[source]),
                high_weight_point_count=int(base['automatic']['usable'][source].sum())))
        for k in range(count):
            cp,cn=coefficients[band,:,k]
            if not valid[band].all():continue
            if cp>=.6 and abs(cn)<=.5*abs(cp):classifications[band,k]=0
            elif cn<=-.6 and abs(cp)<=.5*abs(cn):classifications[band,k]=1
        reports.append(dict(operator=f'Q4-Q{int(width*1000)}',broad_width_ev=width,
            norm_transfer=transfers[band].tolist(),filtered_peak_log=peaks[band].tolist(),
            source_fits=source_reports,classification=classifications[band].tolist()))
    eligible=np.all(transfers>=.05,axis=0)&np.all(peaks>=.003,axis=0)&valid.all()
    records=copy.deepcopy(base['records']);sources=base['source'].copy()
    for k,row in enumerate(records):
        old_source=int(sources[k]);old_applied=bool(row['applied'])
        agreement=classifications[0,k]>=0 and classifications[0,k]==classifications[1,k]
        if eligible[k]:
            sources[k]=classifications[0,k] if agreement and row['fit_usable'] else -1
            row['applied']=bool(sources[k]>=0 and row['fit_usable'])
            row.update(source=int(sources[k]),attribution='P' if sources[k]==0 else 'N' if sources[k]==1 else 'unresolved',
                attribution_reason='matched_bandpass_two_scale_source_agreement' if row['applied'] else 'bandpass_source_ambiguous_or_fit_unsupported',
                source_projection_coefficients=coefficients[:,:,k].tolist(),
                source_projection_classifications=classifications[:,k].tolist())
        margins=[]
        for band in range(2):
            cp,cn=coefficients[band,:,k]
            margins.append(float(min(cp-.6,.5*abs(cp)-abs(cn))) if classifications[band,k]==0 else
                float(min(-cn-.6,.5*abs(cn)-abs(cp))) if classifications[band,k]==1 else None)
        row.update(baseline_source=old_source,baseline_applied=old_applied,
            bandpass_eligible=bool(eligible[k]),bandpass_norm_transfer=transfers[:,k].tolist(),
            bandpass_filtered_peak_log=peaks[:,k].tolist(),
            bandpass_source_coefficients=coefficients[:,:,k].tolist(),
            bandpass_source_classification=classifications[:,k].tolist(),
            bandpass_source_margin_by_band=margins,
            bandpass_source_agreement=bool(agreement),
            bandpass_behavior='new_bandpass_inference' if eligible[k] else 'retain_baseline_binary_low_transfer')
    binary=_v10._binary_coefficients(records)
    source_full=binary@features;source_local=(binary[:,:,None]*features[None,:,:]*base['tapers'][None,:,:]).sum(axis=1)
    uncertain=base['automatic']['common_unknown'].copy()
    for k,row in enumerate(records):
        if not row['applied']:uncertain|=base['tapers'][k]>0
    arrays=dict(attribution_source_weights=weights,attribution_feature_basis_q4=q4feature,
        attribution_coefficients_by_scale_source=coefficients,attribution_classification_by_scale=classifications,
        attribution_source_residual_log=residuals,attribution_scale_source_valid=valid,
        attribution_nuisance_log=np.zeros((2,2,n)),bandpass_source_target_log=source_targets,
        bandpass_feature_basis_log=feature_bands,bandpass_template_norm_transfer=transfers,
        bandpass_template_peak_log=peaks,bandpass_template_eligible=eligible,
        component_source_coefficients_output=binary,
        baseline_attribution_coefficients_by_scale_source=base['projection_arrays']['attribution_coefficients_by_scale_source'])
    identity=dict(baseline_model_sha256=base['model_sha'],bandpass_coefficients_sha256=_sha_array(coefficients),
        eligible_sha256=_sha_array(eligible),source_sha256=_sha_array(sources))
    model_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    diag=copy.deepcopy(base['diag'])
    diag.update(components=records,applied_component_count=sum(r['applied'] for r in records),
        source_projection_baseline=copy.deepcopy(base['diag'].get('source_projection',[])),
        source_projection=reports,source_projection_nuisance_used_in_output=False,
        source_projection_operator='Q4-Q25 / Q4-Q35 applied identically to raw source logs and every feature.',
        bandpass_model_identity=identity,model_sha256=model_sha,
        bandpass_transfer_warning='Low-transfer columns remain auxiliary fitted columns but cannot drive a new source or common-amplitude correction.',
        baseline_pair_model_sha256=base['model_sha'])
    bundle=dict(base);bundle.update(source=sources,records=records,source_full=source_full,source_local=source_local,
        uncertain=uncertain,projection_arrays=arrays,diag=diag,model_sha=model_sha)
    return bundle


def _shared(binary):
    coeff=_v10._binary_coefficients(binary['records']);projection=binary['projection_arrays']['attribution_coefficients_by_scale_source']
    eligible=binary['projection_arrays']['bandpass_template_eligible']
    chosen=np.zeros(len(binary['records']));deltas=np.zeros((2,len(chosen)));reports=[]
    for k,row in enumerate(binary['records']):
        if not (eligible[k] and row['applied']):
            reports.append(dict(component=k,used=False,reason='ineligible_transfer_or_no_stable_source'));continue
        m0=.5 if row['source']==0 else -.5
        estimates=projection[:,:,k].mean(axis=1)-m0;deltas[:,k]=estimates
        if estimates[0]*estimates[1]>0:chosen[k]=estimates[int(np.argmin(abs(estimates)))]
        reports.append(dict(component=k,used=bool(chosen[k]!=0),binary_common_coefficient=m0,
            bandpass_source_coefficients=projection[:,:,k].tolist(),
            source_difference_coefficients_by_band=(projection[:,0,k]-projection[:,1,k]).tolist(),
            delta_by_band=estimates.tolist(),delta_interval=[float(estimates.min()),float(estimates.max())],
            delta_midpoint=float(estimates.mean()),delta_span=float(np.ptp(estimates)),selected_delta=float(chosen[k])))
    updated=coeff+chosen[None,:];source_full=updated@binary['components']
    np.testing.assert_allclose(source_full[0]-source_full[1],binary['source_full'][0]-binary['source_full'][1],rtol=0,atol=2e-15)
    bundle=dict(binary);bundle['diag']=copy.deepcopy(binary['diag'])
    bundle['source_full']=source_full
    bundle['source_local']=(updated[:,:,None]*binary['components'][None,:,:]*binary['tapers'][None,:,:]).sum(axis=1)
    bundle['projection_arrays']={key:value.copy() for key,value in binary['projection_arrays'].items()}
    bundle['projection_arrays'].update(component_source_coefficients_output=updated,
        component_source_coefficients_binary=coeff,shared_delta_by_band=deltas,
        shared_delta_selected=chosen,shared_component_correction_log=chosen[:,None]*binary['components'])
    bundle['diag'].update(shared_amplitude_components=reports,
        shared_amplitude_uncertainty='Two matched-bandpass estimates only; not a statistical confidence interval.',
        continuous_common_amplitude_may_break_equivariance=True)
    return bundle


def fit_bundle(e,pn):
    base_v9=_v9.fit_bundle(e,pn);base,_extra=_v10._paired(base_v9)
    binary=_bandpass(base);shared=_shared(binary)
    return dict(base=base,variants={'v10_pair_binary_full_tail':base,
        'bandpass_binary_full_tail':binary,'bandpass_shared_full_tail':shared})


def render(bundle,*,family='bandpass_shared_full_tail'):
    if family not in FAMILIES:raise ValueError('Unknown V11 family')
    model=bundle['variants'][family];result=_v9.render(model,family='conditional_full_tail')
    construction=dict(family=family,model_sha256=model['model_sha'],source_correction_sha256=_sha_array(model['source_full']))
    construction_sha=hashlib.sha256(json.dumps(construction,sort_keys=True).encode()).hexdigest()
    result['construction_sha256']=np.array(construction_sha);result['baseline_pair_model_sha256']=np.array(bundle['base']['model_sha'])
    result['diagnostics'].update(family=family,settings=copy.deepcopy(SETTINGS),construction=construction,
        construction_sha256=construction_sha,baseline_pair_model_sha256=bundle['base']['model_sha'],
        legacy_local_weight_fields_used_in_output=False,
        output_construction='Raw full-feature subtraction; unweighted corrected-PN log mean. No filtered PN or nuisance substituted.')
    if family!='v10_pair_binary_full_tail':
        result['diagnostics']['source_attribution']='Matched local bandpass inference for sufficient-transfer components; broad low-transfer terms retain baseline binary assignment.'
    return result


def generate_all(e,pn):
    bundle=fit_bundle(e,pn)
    return {family:render(bundle,family=family) for family in FAMILIES}


def generate(e,pn,*,family='bandpass_shared_full_tail'):
    if family not in FAMILIES:raise ValueError('Unknown V11 family')
    return render(fit_bundle(e,pn),family=family)
