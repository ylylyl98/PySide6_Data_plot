"""Shared low-order BN intensity trend, with same-scan numerical checks.

The selected fitting windows are assumptions about sample baseline, not an
independent measurement of the optical background or physical peak fidelity.
"""
from dataclasses import asdict
import json
import numpy as np
from core.drr_bn_calibration import BnCalibrationResult, _check
from core.drr_bn_settings import BnCalibrationSettings


def _fit(x, y, mask, degree, cancellation):
    if np.sum(mask)<max(12,degree+3): raise ValueError('insufficient_window_points')
    A=np.vander(x[mask],degree+1,increasing=True); v=y[mask]
    coef=np.linalg.lstsq(A,v,rcond=None)[0]
    for _ in range(35):
        _check(cancellation);r=v-A@coef;scale=max(1.4826*np.median(abs(r-np.median(r))),1e-8)
        w=np.minimum(1.,1.345*scale/np.maximum(abs(r),1e-15))
        new=np.linalg.lstsq(A*np.sqrt(w[:,None]),v*np.sqrt(w),rcond=None)[0]
        if np.max(abs(new-coef))<1e-11:coef=new;break
        coef=new
    if not np.isfinite(coef).all(): raise ValueError('nonfinite_coefficients')
    return coef


def _model(x, coeff):
    return np.polynomial.polynomial.polyval(x,coeff)


def _masks(e, regions):
    masks=[(e>=a)&(e<=b) for a,b in regions]
    if len(masks)<2 or any(np.count_nonzero(m)<12 for m in masks): raise ValueError('insufficient_window_points')
    mask=np.logical_or.reduce(masks)
    midpoint=(e[0]+e[-1])/2
    if not np.any(mask & (e<midpoint)) or not np.any(mask & (e>midpoint)):raise ValueError('requires_two_sided_windows')
    return masks,mask


def _evaluate_degree(e,x,y,regions,degree,limit,cancellation):
    masks,mask=_masks(e,regions); coeff=_fit(x,y,mask,degree,cancellation); predicted=_model(x,coeff)
    blocks=[[],[]]
    for m in masks:
        ids=np.flatnonzero(m);first,second=np.array_split(ids,2);blocks[0].extend(first);blocks[1].extend(second)
    cv=[]
    for k in (0,1):
        train=np.zeros(len(e),bool);test=np.zeros(len(e),bool);train[blocks[k]]=True;test[blocks[1-k]]=True
        c=_fit(x,y,train,degree,cancellation);res=y[test]-_model(x[test],c)
        cv.append(dict(original_rms=float(np.sqrt(np.mean(y[test]**2))),candidate_rms=float(np.sqrt(np.mean(res**2))),test_count=int(test.sum())))
    original=float(np.sqrt(np.mean([v['original_rms']**2 for v in cv])))
    candidate=float(np.sqrt(np.mean([v['candidate_rms']**2 for v in cv])))
    variants=[]; center=(e>=e[0]+.15*np.ptp(e))&(e<=e[-1]-.15*np.ptp(e))
    # Both windows must remain represented. These are predetermined window
    # perturbations, not optimized to improve fit.
    for fraction,shift in ((.1,0),(.2,0),(0,.08),(0,-.08)):
        alt=[]
        for lo,hi in regions:
            width=hi-lo;alt.append((max(float(e[0]),lo+width*(fraction+shift)),min(float(e[-1]),hi-width*(fraction-shift))))
        try:
            _,am=_masks(e,alt);c=_fit(x,y,am,degree,cancellation); diff=np.expm1(np.clip(_model(x,c)-predicted,-50,50));whole=float(np.max(abs(diff)));middle=float(np.max(abs(diff[center])))
            variants.append(dict(regions=[list(v) for v in alt],full_axis_max_relative=whole,center_max_relative=middle,coefficients=c.tolist()))
        except ValueError:raise ValueError('window_perturbation_insufficient')
    sensitivity=max(v['full_axis_max_relative'] for v in variants)
    return dict(coefficients=coeff.tolist(),degree=degree,cv_blocks=cv,cv_original_rms=original,cv_candidate_rms=candidate,
        cv_improvement_fraction=float((original-candidate)/original) if original>1e-12 else 0.,
        sensitivity_variants=variants,max_sensitivity=float(sensitivity),sensitivity_pass=bool(sensitivity<=limit),
        gain_min=float(np.exp(np.min(predicted))),gain_max=float(np.exp(np.max(predicted))))


def trend_validation_passed(report,settings=None):
    """Check recorded same-scan evidence; caller must verify the array itself."""
    if isinstance(report, dict) and report.get('optimization_policy') == 'shared_bn_trend_v2':
        from core.drr_bn_trend_adaptive import adaptive_validation_passed
        return adaptive_validation_passed(report, settings)
    try:
        saved=report['settings'];v=report['trend_validation'];scope=report['single_pair_scope'];name=report['selected_method'];models=report['models'];m=models[name]
        if settings is not None:
            expected=asdict(settings) if isinstance(settings,BnCalibrationSettings) else dict(settings)
            if json.dumps(saved,sort_keys=True)!=json.dumps(expected,sort_keys=True):return False
        if not (report['schema_version']==1 and report['optimization_policy']=='shared_bn_trend_v1' and report['measurement_count']==1 and report['independent_validation'] is False and report['shared_across_gates'] is True and report['candidate_generated'] is True and saved['single_measurement'] is True and saved['single_trend_enabled'] is True and scope['mode']=='full_range' and scope['model']=='shared_trend_normalization' and name in ('trend_linear','trend_quadratic') and v['passed'] is True): return False
        if report['selected_candidate']!=name or not all(v[k] is True for k in ('finite_positive','same_scan_cv_improved','window_sensitivity_pass')) or v['independent_validation'] is not False:return False
        lim=saved['single_trend_max_sensitivity'];order=saved['single_trend_order'];degree=1 if name=='trend_linear' else 2
        coord=report['coordinate'];center=coord['center_ev'];scale=coord['scale_ev'];regions=report['background_regions_ev']
        if not np.isfinite([center,scale]).all() or scale<=0 or len(regions)<2:return False
        if any(len(r)!=2 or not np.isfinite(r).all() or r[0]>=r[1] or r[0]<center-scale-1e-12 or r[1]>center+scale+1e-12 for r in regions):return False
        if any(a[1]>=b[0] for a,b in zip(regions,regions[1:])) or not (regions[0][0]<center<regions[-1][1]):return False
        requested=saved['single_trend_regions']
        if requested:
            if report['background_regions_source']!='user' or not np.allclose(regions,sorted(requested),rtol=0,atol=1e-12):return False
        else:
            expected=[[center-scale,center-scale+.3*scale],[center+scale-.3*scale,center+scale]]
            if report['background_regions_source']!='auto_edges' or not np.allclose(regions,expected,rtol=0,atol=1e-12):return False
        params=report['candidates'][name]['parameters']; coeff=params['coefficients']
        if len(coeff)!=degree+1 or not np.isfinite(coeff).all() or not np.isclose(params['energy_center_ev'],center):return False
        if not np.allclose(coeff,[v/scale**i for i,v in enumerate(m['coefficients'])],rtol=1e-10,atol=1e-12):return False
        if order not in ('auto','linear','quadratic') or (order!='auto' and name!='trend_'+order) or m['degree']!=degree or len(m['coefficients'])!=degree+1:return False
        vals=[lim,m['gain_min'],m['gain_max'],m['cv_original_rms'],m['cv_candidate_rms'],m['cv_improvement_fraction'],m['max_sensitivity'],*m['coefficients']]
        if not np.isfinite(vals).all() or not (0<lim<1 and 0<m['gain_min']<=m['gain_max'] and m['cv_original_rms']>1e-12 and m['cv_candidate_rms']>=0 and m['cv_candidate_rms']<m['cv_original_rms'] and 0<=m['max_sensitivity']<=lim and m['sensitivity_pass'] is True):return False
        if not np.isclose(m['cv_improvement_fraction'],1-m['cv_candidate_rms']/m['cv_original_rms'],rtol=1e-9,atol=1e-12):return False
        blocks=m['cv_blocks'];vs=m['sensitivity_variants']
        if len(blocks)!=2 or len(vs)!=4:return False
        for q in blocks:
            if q['test_count']<12 or not np.isfinite([q['original_rms'],q['candidate_rms']]).all() or min(q['original_rms'],q['candidate_rms'])<0:return False
        if not np.isclose(m['cv_original_rms'],np.sqrt(np.mean([q['original_rms']**2 for q in blocks]))) or not np.isclose(m['cv_candidate_rms'],np.sqrt(np.mean([q['candidate_rms']**2 for q in blocks]))):return False
        for q in vs:
            if len(q['coefficients'])!=degree+1 or not np.isfinite([q['full_axis_max_relative'],q['center_max_relative'],*q['coefficients']]).all() or not 0<=q['center_max_relative']<=q['full_axis_max_relative']<=lim:return False
        if not np.isclose(m['max_sensitivity'],max(q['full_axis_max_relative'] for q in vs)):return False
        if order=='auto' and degree==2:
            linear=models['trend_linear']
            if not m['cv_candidate_rms']<=.85*linear['cv_candidate_rms']:return False
        return True
    except (KeyError,TypeError,ValueError,OverflowError,IndexError):return False


def evaluate_bn_trend(energy, measurements, background, *, gates, settings=None, background_repeats=None, cancellation=None):
    config=settings or BnCalibrationSettings()
    if not isinstance(config,BnCalibrationSettings):config=BnCalibrationSettings(**dict(config))
    if config.single_trend_adaptive:
        from core.drr_bn_trend_adaptive import evaluate_adaptive_trend
        return evaluate_adaptive_trend(energy, measurements, background, gates=gates, settings=config,
                                       background_repeats=background_repeats, cancellation=cancellation)
    _check(cancellation);e=np.asarray(energy,float);z=np.asarray(measurements,float);b=np.asarray(background,float);g=np.asarray(gates,float)
    report=dict(schema_version=1,optimization_policy='shared_bn_trend_v1',measurement_count=1,independent_validation=False,
        shared_across_gates=True,candidate_generated=False,settings=asdict(config),joint_validation={'passed':False},
        single_pair_scope={'mode':'full_range','model':'shared_trend_normalization'},models={},
        limitation='Same-scan baseline assumption; shared gain changes absolute DR/R and may absorb shared sample features. No independent physical validation.')
    def finish(arrays,selected,eligible,reasons):
        applied=selected if eligible else 'original';report['selected_method']=selected;report['selected_candidate']=selected;report['reasons']=list(reasons)
        return BnCalibrationResult(b.copy(),arrays,selected,applied,arrays[applied].copy(),bool(eligible),'single_pair' if eligible else 'preview_only',tuple(reasons),report)
    arrays={'original':b.copy()}
    def fail(reason):report['trend_validation']={'passed':False,'reason':reason};return finish(arrays,'original',False,[reason])
    if z.ndim==3 and z.shape[0]==1:z=z[0]
    if (not config.single_measurement or not config.single_trend_enabled or e.ndim!=1 or b.shape!=e.shape or g.ndim!=1 or z.shape!=(len(g),len(e)) or len(g)<1 or len(e)<60 or not all(np.isfinite(t).all() for t in (e,z,b,g)) or np.any(z<=0) or np.any(b<=0) or not (np.all(np.diff(e)>0) or np.all(np.diff(e)<0))):return fail('invalid_input_or_scope')
    if config.single_trend_order not in ('auto','linear','quadratic') or not np.isfinite(config.single_trend_max_sensitivity) or not 0<config.single_trend_max_sensitivity<1:return fail('invalid_settings')
    if background_repeats is not None:
        r=np.asarray(background_repeats,float)
        if r.ndim!=2 or r.shape[1]!=len(e) or not np.isfinite(r).all() or np.any(r<=0):return fail('invalid_background_repeats')
        report['background_repeats']={'count':len(r),'used_for_fit':False}
    reverse=e[0]>e[-1];ee=e[::-1] if reverse else e;zz=z[:,::-1] if reverse else z;bb=b[::-1] if reverse else b
    try:
        if config.single_trend_regions:regions=sorted(tuple(map(float,r)) for r in config.single_trend_regions);source='user_supplied_background_assumption'
        else:
            width=.15*np.ptp(ee);regions=[(float(ee[0]),float(ee[0]+width)),(float(ee[-1]-width),float(ee[-1]))];source='default_outer_15_percent_background_assumption'
        if len(regions)<2 or any(len(r)!=2 or not np.isfinite(r).all() or r[0]<ee[0] or r[1]>ee[-1] or r[0]>=r[1] for r in regions) or any(a[1]>=b_[0] for a,b_ in zip(regions,regions[1:])):return fail('invalid_background_regions')
        center=float((ee[0]+ee[-1])/2);scale=float(np.ptp(ee)/2);x=(ee-center)/scale;y=np.log(np.median(zz,axis=0))-np.log(bb)
        models={};candidates={}
        for degree,name in ((1,'trend_linear'),(2,'trend_quadratic')):
            model=_evaluate_degree(ee,x,y,regions,degree,config.single_trend_max_sensitivity,cancellation);loggain=_model(x,model['coefficients'])
            with np.errstate(over='raise',invalid='raise'):candidate=bb*np.exp(loggain)
            if not np.isfinite(candidate).all() or np.any(candidate<=0):raise ValueError('invalid_candidate')
            models[name]=model;candidates[name]=candidate[::-1].copy() if reverse else candidate.copy()
        report.update(models=models,candidate_generated=True,coordinate={'center_ev':center,'scale_ev':scale},background_regions=[list(r) for r in regions],region_source=source,
            background_regions_ev=[list(r) for r in regions],background_regions_source='user' if config.single_trend_regions else 'auto_edges',
            cv_method='two contiguous halves per background region, both directions; same scan',sensitivity_method='window edges shrunk 10/20% and shifted +/-8%, all regions')
        report['candidates']={'original':{'parameters':{}}}
        for name, model in models.items():
            report['candidates'][name]={'parameters':{'coefficients':[v/scale**i for i,v in enumerate(model['coefficients'])],
                'energy_center_ev':center},
                'selection':{'mean_improvement_fraction':model['cv_improvement_fraction']},
                'selection_d2':{'mean_improvement_fraction':None},
                'selection_joint':{'passed':False,'drr_noninferiority_passed':False,
                    'reasons':['shared trend normalization; D2 not certified']}}
        arrays.update(candidates);selected='trend_'+config.single_trend_order
        if config.single_trend_order=='auto':
            q=models['trend_quadratic'];l=models['trend_linear'];selected='trend_quadratic' if q['sensitivity_pass'] and q['cv_candidate_rms']<=.85*l['cv_candidate_rms'] else 'trend_linear'
        report['model_selection']={'order_requested':config.single_trend_order,'selected':selected,'reason':('quadratic_same_scan_cv_improves_15_percent_and_stable' if config.single_trend_order=='auto' and selected=='trend_quadratic' else 'linear_default_or_quadratic_unstable_or_insufficient_improvement' if config.single_trend_order=='auto' else 'explicit_order')}
        m=models[selected];passed=bool(m['sensitivity_pass'] and m['cv_original_rms']>1e-12 and m['cv_candidate_rms']<m['cv_original_rms'])
        report['trend_validation']={'passed':passed,'finite_positive':True,'same_scan_cv_improved':m['cv_candidate_rms']<m['cv_original_rms'],'window_sensitivity_pass':m['sensitivity_pass'],'independent_validation':False}
        return finish(arrays,selected,passed,[] if passed else ['same_scan_checks_failed'])
    except (ValueError,TypeError,IndexError,FloatingPointError,np.linalg.LinAlgError) as exc:return fail(str(exc))
