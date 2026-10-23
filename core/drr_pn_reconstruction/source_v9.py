"""V9 PN-only coarse-nuisance-stable source attribution; nuisance is never output."""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "568808534d60f99b6f6f981e2fd09a086f70a6452a20eed71c63e245c3b65be3"
import copy
import hashlib
import json
import numpy as np
from scipy.interpolate import BSpline

from . import source_v8 as _v8
V8_SHA256 = _v8.MIGRATED_FROM_SHA256
_v7=_v8._v7
V7_SHA256=_v8.V7_SHA256;V6_SHA256=_v8.V6_SHA256
FAMILIES=_v8.FAMILIES;_rms=_v8._rms;_sha_array=_v8._sha_array
_automatic_support=_v8._automatic_support
SETTINGS=dict(**_v7.SETTINGS,pruning=False,attribution_spline_scales_ev=[.120,.180],
    source_usable_projection_weight=4.,other_projection_weight=1.,
    feature_projection_condition_maximum=1e4,projection_operator='same native Q4 for target and ALL columns',
    decomposition_selection='unweighted_native_PN_residual_BIC')


def _fit_selected(e,ratio):
    paired,detection,smoothed,edge=_v7._detect(e,ratio)
    choices=[]
    for name,events in (('paired',paired),('split',_v8._split_events(detection))):
        result=_v7._fit(e,ratio,events)
        error=result[3]
        score=float(len(e)*np.log(max(np.mean(error**2),1e-24))+(4*len(events)+3)*np.log(len(e)))
        choices.append(dict(name=name,score=score,result=result))
    chosen=min(range(len(choices)),key=lambda i:(choices[i]['score'],i))
    result=choices[chosen]['result'];diag=result[-1]
    diag.update(selected_decomposition=choices[chosen]['name'],decomposition_selection=[
        dict(name=c['name'],bic=c['score'],components=copy.deepcopy(c['result'][4]),fit=copy.deepcopy(c['result'][5])) for c in choices])
    return (*result,detection,smoothed,edge)


def _projection(e,logs,components,records,automatic):
    count,n=components.shape;source=np.full(count,-1,int);tapers=np.zeros_like(components)
    for k,row in enumerate(records):
        c,w,unused,unused2=row['parameters'];inner=max(.025,3*w);outer=max(.050,6*w)
        tapers[k]=_v7._v6._quintic((outer-abs(e-c))/(outer-inner))
        row.update(source=-1,attribution='unresolved',attribution_reason='scale_unstable_or_unsupported_projection',
            support_ev=[float(c-outer),float(c+outer)],plateau_ev=[float(c-inner),float(c+inner)])
    qtarget=automatic['q4logs'];qfeatures=_v7._quad(e,components,.004) if count else components.copy()
    cell=np.gradient(e);weight=(1.+3*automatic['usable'].astype(float))*len(e)*cell[None,:]/cell.sum()
    coefficients=np.zeros((2,2,count));classifications=np.full((2,count),-1,int)
    residuals=np.zeros((2,2,n));nuisances=np.zeros_like(residuals)
    reports=[];arrays=dict(attribution_source_weights=weight,attribution_feature_basis_q4=qfeatures)
    valid=np.ones((2,2),bool)
    for scale_index,spacing in enumerate((.120,.180)):
        internal=np.arange(e[0]+spacing,e[-1]-1e-12,spacing)
        knots=np.r_[np.repeat(e[0],4),internal,np.repeat(e[-1],4)]
        basis=BSpline.design_matrix(e,knots,3).toarray()
        qbasis=_v7._quad(e,basis.T,.004).T
        matrix=np.column_stack((qfeatures.T,qbasis))
        scale_reports=[]
        for s in range(2):
            rootweight=np.sqrt(weight[s]);design=rootweight[:,None]*matrix;target=rootweight*qtarget[s]
            norms=np.linalg.norm(design,axis=0)
            normalized=np.divide(design,norms[None,:],out=np.zeros_like(design),where=norms[None,:]>1e-16)
            solution,unused,rank,singular=np.linalg.lstsq(normalized,target,rcond=None)
            pars=np.divide(solution,norms,out=np.zeros_like(solution),where=norms>1e-16)
            coefficients[scale_index,s]=pars[:count]
            nuisances[scale_index,s]=qbasis@pars[count:]
            residuals[scale_index,s]=qtarget[s]-matrix@pars
            smooth=rootweight[:,None]*qbasis;features=rootweight[:,None]*qfeatures.T
            projected=features-smooth@np.linalg.lstsq(smooth,features,rcond=None)[0]
            feature_norms=np.linalg.norm(projected,axis=0)
            pn=np.divide(projected,feature_norms[None,:],out=np.zeros_like(projected),where=feature_norms[None,:]>1e-16)
            sv=np.linalg.svd(pn,compute_uv=False)
            condition=float(sv[0]/sv[-1]) if len(sv) and sv[-1]>1e-16 else (1. if count==0 else None)
            full_condition=float(singular[0]/singular[-1]) if singular[-1]>1e-16 else None
            valid[scale_index,s]=condition is not None and condition<=1e4 and (count==0 or bool(np.all(feature_norms>1e-16)))
            scale_reports.append(dict(source='P' if s==0 else 'N',coefficients=pars.tolist(),
                feature_coefficients=pars[:count].tolist(),nuisance_coefficients=pars[count:].tolist(),
                column_norms=norms.tolist(),nuisance_residualized_feature_norms=feature_norms.tolist(),
                feature_condition=condition,full_normalized_condition=full_condition,rank=int(rank),
                source_residual_rms_log=_rms(residuals[scale_index,s]),valid=bool(valid[scale_index,s]),
                high_weight_point_count=int(automatic['usable'][s].sum())))
        for k in range(count):
            cp,cn=coefficients[scale_index,:,k]
            if not valid[scale_index].all():continue
            if cp>=.6 and abs(cn)<=.5*abs(cp):classifications[scale_index,k]=0
            elif cn<=-.6 and abs(cp)<=.5*abs(cn):classifications[scale_index,k]=1
        reports.append(dict(knot_spacing_ev=spacing,knots=knots.tolist(),source_fits=scale_reports,
            classification=classifications[scale_index].tolist()))
        arrays[f'attribution_nuisance_basis_q4_{int(spacing*1000)}mev']=qbasis
        arrays[f'attribution_knot_vector_{int(spacing*1000)}mev']=knots
    for k,row in enumerate(records):
        agreement=classifications[0,k]>=0 and classifications[0,k]==classifications[1,k]
        if agreement and row['fit_usable']:source[k]=classifications[0,k]
        margins=[]
        for scale in range(2):
            cp,cn=coefficients[scale,:,k]
            if classifications[scale,k]==0:margins.append(float(min(cp-.6,.5*abs(cp)-abs(cn))))
            elif classifications[scale,k]==1:margins.append(float(min(-cn-.6,.5*abs(cn)-abs(cp))))
            else:margins.append(None)
        row.update(source=int(source[k]),source_projection_coefficients=coefficients[:,:,k].tolist(),
            source_projection_classifications=classifications[:,k].tolist(),attribution_scale_agreement=bool(agreement),
            attribution_margin_by_scale=margins,
            attribution='P' if source[k]==0 else 'N' if source[k]==1 else 'unresolved',
            attribution_reason='same_source_at_both_coarse_spline_scales' if source[k]>=0 else 'fit_unsupported' if not row['fit_usable'] else 'ambiguous_or_scale_disagreement')
    arrays.update(attribution_coefficients_by_scale_source=coefficients,
        attribution_classification_by_scale=classifications,attribution_source_residual_log=residuals,
        attribution_nuisance_log=nuisances,attribution_scale_source_valid=valid)
    return source,tapers,reports,arrays


def fit_bundle(e,pn):
    e=np.array(e,float,copy=True);pn=np.array(pn,float,copy=True)
    if e.ndim!=1 or len(e)<20 or not np.all(np.isfinite(e)) or not np.all(np.diff(e)>0):raise ValueError('Increasing native energy required')
    if pn.shape!=(2,len(e)) or not np.all(np.isfinite(pn)&(pn>0)):raise ValueError('Positive finite PN required')
    logs=np.log(pn);raw_ratio=np.log(pn[0]/pn[1]);ratio=np.round(raw_ratio,12)
    automatic=_automatic_support(e,logs)
    components,nuisance,model,error,records,diag,detection,smoothed,edge=_fit_selected(e,ratio)
    source,tapers,projection_reports,projection_arrays=_projection(e,logs,components,records,automatic)
    source_full=np.zeros_like(pn);source_local=np.zeros_like(pn);uncertain=automatic['common_unknown'].copy()
    for k,row in enumerate(records):
        row['applied']=bool(row['fit_usable'] and source[k]>=0)
        if row['applied']:
            sign=1. if source[k]==0 else -1.
            source_full[source[k]]+=sign*components[k];source_local[source[k]]+=sign*components[k]*tapers[k]
        else:uncertain|=tapers[k]>0
    weighted,alignment,severity,weight_anchors,weight_reports=_v7._reliability(e,logs-source_local,components,records,source,tapers,error,automatic)
    identity=dict(energy_hash=_sha_array(e),pn_hash=_sha_array(pn),component_hash=_sha_array(components),
        source_hash=_sha_array(source),taper_hash=_sha_array(tapers),nuisance_hash=_sha_array(nuisance),
        projection_coefficients_hash=_sha_array(projection_arrays['attribution_coefficients_by_scale_source']),
        selected_decomposition=diag['selected_decomposition'])
    model_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    diag.update(components=records,detection=detection,source_projection=projection_reports,
        flank_calibration=weight_reports,model_sha256=model_sha,model_hash_inputs=identity,
        detected_component_count=len(records),applied_component_count=sum(r['applied'] for r in records),
        source_projection_nuisance_used_in_output=False,
        source_projection_operator='native Q4 applied identically to PN logs, differential features and spline basis',
        full_differential_components_retained=True)
    return dict(e=e,pn=pn,logs=logs,raw_ratio=raw_ratio,ratio=ratio,automatic=automatic,
        components=components,nuisance=nuisance,model=model,error=error,records=records,diag=diag,
        smoothed=smoothed,edge=edge,source=source,tapers=tapers,
        attribution_anchors=np.zeros_like(components,bool),source_local_residuals=np.zeros((len(components),2,len(e))),
        source_full=source_full,source_local=source_local,uncertain=uncertain,weighted=weighted,
        alignment=alignment,severity=severity,weight_anchors=weight_anchors,model_sha=model_sha,
        projection_arrays=projection_arrays)


def render(bundle,*,family='conditional_local_weighted'):
    if family not in FAMILIES:raise ValueError('Unknown V9 family')
    result=_v8.render(bundle,family=family)
    for key,value in bundle['projection_arrays'].items():result[key]=value.copy()
    result['diagnostics'].update(settings=copy.deepcopy(SETTINGS),
        source_attribution='Same assignment in jointly fitted source features plus cubic nuisance at 120/180meV; nuisance never substituted into PN.',
        attribution_anchor_interpretation='V9 uses all observed source points with usable-dependent weights. Empty legacy attribution-anchor array; actual projection weights and optional alignment flanks saved.')
    return result


def generate_all(e,pn):
    bundle=fit_bundle(e,pn)
    return {family:render(bundle,family=family) for family in FAMILIES}


def generate(e,pn,*,family='conditional_local_weighted'):
    if family not in FAMILIES:raise ValueError('Unknown V9 family')
    return render(fit_bundle(e,pn),family=family)
