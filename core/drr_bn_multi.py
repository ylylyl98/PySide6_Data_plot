"""Same-scan, multi-BN texture calibration with full-axis numerical guards.

The original measured background supplies the broad envelope. The only fine
structure bases come from the selected measured BN spectra and their first
pixel derivatives. Same-scan validation cannot prove physical peak fidelity.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from core.drr_bn_calibration import BnCalibrationResult, _check
from core.drr_bn_joint import production_d2
from core.drr_bn_settings import BnCalibrationSettings
from core.drr_bn_single import _full_axis_drr_check


def multi_pair_validation_passed(report, settings=None):
    """Validate same-scan numerical evidence, never independent fidelity."""
    try:
        saved=report['settings']; v=report['multi_pair_validation']; m=report['model']
        if settings is not None:
            expected=asdict(settings) if isinstance(settings,BnCalibrationSettings) else dict(settings)
            if json.dumps(saved,sort_keys=True)!=json.dumps(expected,sort_keys=True): return False
        der=report['derivative_settings']; bounds=m['correction_limit']
        values=(v['original_texture_rms'],v['candidate_texture_rms'],m['max_log_correction'],bounds,
                m['alpha_min'],m['alpha_max'],*m['coefficients'])
        return (report['schema_version']==1 and report['optimization_policy']=='multi_bn_single_v1'
            and report['measurement_count']==1 and report['independent_validation'] is False
            and saved['single_measurement'] is True and saved['single_full_range'] is True
            and m['basis_source']=='measured_bn_only' and m['knots']==6 and m['template_count']>=2
            and m['template_rank']>=1 and len(m['coefficients'])==m['template_count']*12
            and np.isfinite(values).all() and 0<=m['alpha_min']<=m['alpha_max']<=1
            and 0<bounds<=.2 and 0<=m['max_log_correction']<=bounds+1e-12
            and m['score_metric']=='weighted_bn_template_highpass_rms'
            and m['template_window']==saved['template_window'] and m['template_polyorder']==saved['template_polyorder']
            and der['processing']=='production SG energy derivative'
            and der['window_length']==saved['derivative_window'] and der['polyorder']==saved['derivative_polyorder']
            and isinstance(der['used_window_length'],int) and der['used_window_length']>0
            and all(v[k] is True for k in ('passed','finite_positive','all_gate_d2_check','all_gate_drr_check','same_scan_improvement'))
            and v['d2_windows_checked']>0 and v['d2_failed_windows']==0
            and len(v['drr_regions'])==4 and all(q['passed'] is True and q['gate_count']>0 for q in v['drr_regions'])
            and v['original_texture_rms']>0 and v['candidate_texture_rms']>=0
            and v['candidate_texture_rms']<v['original_texture_rms']*(1-saved['min_improvement_fraction']))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def evaluate_bn_multi(energy, measurements, background, *, gates, templates, settings=None, cancellation=None):
    config=settings or BnCalibrationSettings()
    if not isinstance(config,BnCalibrationSettings): config=BnCalibrationSettings(**dict(config))
    _check(cancellation)
    e=np.asarray(energy,float); z=np.asarray(measurements,float); b=np.asarray(background,float)
    g=np.asarray(gates,float); ts=np.asarray(templates,float)
    report=dict(schema_version=1,optimization_policy='multi_bn_single_v1',measurement_count=1,
        independent_validation=False,settings=asdict(config),joint_validation={'passed':False},
        limitation='Same-scan fit: physical peaks correlated with BN may be absorbed; no independent validation.',
        derivative_settings=dict(processing='production SG energy derivative',window_length=config.derivative_window,
                                 polyorder=config.derivative_polyorder),candidates={})
    def finish(candidate,eligible,reasons,validation=None,model=None):
        selected='multi_pair' if candidate is not None and not np.array_equal(candidate,b) else 'original'
        applied='multi_pair' if eligible else 'original'; arrays={'original':b.copy()}
        report['multi_pair_validation']=validation or {'passed':False}; report['model']=model or {}
        report['reasons']=list(reasons)
        report['candidates']['original']=dict(parameters={},selection={},selection_d2={},selection_joint={'passed':False})
        if candidate is not None:
            arrays['multi_pair']=candidate.copy(); v=validation or {}
            old=v.get('original_texture_rms',0.); new=v.get('candidate_texture_rms',old)
            improvement=(old-new)/old if old else 0.
            report['candidates']['multi_pair']=dict(parameters={'family':'multi_bn_texture'},
                selection={'mean_improvement_fraction':None},selection_d2={'mean_improvement_fraction':None},
                selection_joint={'passed':eligible,'drr_noninferiority_passed':bool(v.get('all_gate_drr_check',False)),
                                 'reasons':list(reasons)},texture_improvement_fraction=improvement)
        return BnCalibrationResult(b.copy(),arrays,selected,applied,arrays[applied].copy(),bool(eligible),
             'single_pair' if eligible else 'original',tuple(reasons),report)
    if z.ndim==3 and z.shape[0]==1:z=z[0]
    if (not config.single_measurement or not config.single_full_range or e.ndim!=1 or g.ndim!=1
        or ts.ndim!=2 or ts.shape[0]<2 or ts.shape[1:]!=e.shape or b.shape!=e.shape
        or z.shape!=(len(g),len(e)) or len(g)<1 or len(e)<max(config.template_window,config.derivative_window+2)
        or not all(np.isfinite(v).all() for v in (e,z,b,g,ts)) or np.any(z<=0) or np.any(b<=0)
        or np.any(ts<=0) or not (np.all(np.diff(e)>0) or np.all(np.diff(e)<0))):
        return finish(None,False,['invalid_multi_bn_input_or_scope'])
    if not np.allclose(b, np.mean(ts,axis=0), rtol=1e-10, atol=1e-8):
        return finish(None,False,['background_is_not_template_group_mean'])
    if (not isinstance(config.template_window,int) or config.template_window<3 or config.template_window%2!=1
        or not isinstance(config.template_polyorder,int) or not 0<=config.template_polyorder<config.template_window):
        return finish(None,False,['invalid_template_settings'])
    if (not np.isfinite(config.min_improvement_fraction) or not 0<=config.min_improvement_fraction<1
        or not np.isfinite(config.max_local_degradation_fraction) or config.max_local_degradation_fraction<0
        or not np.isfinite(config.max_absolute_d2_degradation) or config.max_absolute_d2_degradation<0
        or not np.isfinite(config.drr_noninferiority_fraction) or config.drr_noninferiority_fraction<0
        or not np.isfinite(config.drr_noninferiority_absolute) or config.drr_noninferiority_absolute<0
        or config.local_validation_window<3):return finish(None,False,['invalid_settings'])
    reverse=e[0]>e[-1]
    if reverse:e=e[::-1].copy();z=z[:,::-1].copy();b=b[::-1].copy();ts=ts[:,::-1].copy()
    def done(candidate,eligible,reasons,validation=None,model=None):
        nonlocal b
        if reverse:b=b[::-1].copy();candidate=None if candidate is None else candidate[::-1].copy()
        return finish(candidate,eligible,reasons,validation,model)
    logts=np.log(ts); fine=logts-savgol_filter(logts,config.template_window,config.template_polyorder,axis=1)
    differences=fine[1:]-fine[0]; sig=np.linalg.svd(differences,compute_uv=False)
    rank=int(np.sum(sig>max(sig[0]*1e-5,1e-8))) if len(sig) else 0
    if rank<1:return done(None,False,['no_distinct_bn_texture'])
    pix=np.arange(len(e)); breaks=np.linspace(0,len(e)-1,4)
    knot_vector=np.r_[np.repeat(breaks[0],4),breaks[1:-1],np.repeat(breaks[-1],4)]
    W=BSpline.design_matrix(pix,knot_vector,3).toarray(); n=6
    template=np.concatenate((fine,np.gradient(fine,axis=1)),axis=0)
    A=np.column_stack([template[j]*W[:,k] for j in range(len(template)) for k in range(n)])
    prior=np.zeros(A.shape[1]);prior[:n*len(ts)]=1/len(ts)
    D=np.zeros((len(template)*(n-2),A.shape[1]))
    for j in range(len(template)):
        for k in range(n-2):D[j*(n-2)+k,j*n+k:j*n+k+3]=[1,-2,1]
    high=np.log(z);high=high-savgol_filter(high,config.template_window,config.template_polyorder,axis=1);target=np.median(high,axis=0)
    mad=np.median(abs(high-target),axis=0);median_mad=np.median(mad);bw=np.ones(len(e))
    for ids in np.array_split(np.arange(len(e)),6):
        scale=max(np.sqrt(np.mean(target[ids]**2)),.25*np.sqrt(np.mean(target**2)),1e-5);bw[ids]=1/scale
    bw/=np.median(bw);weights=bw**2/(1+(mad/max(median_mad*3,1e-8))**2)
    sc=max(float(np.median(np.sum(A*A,axis=0))),1e-12); coef=prior.copy()
    try:
        for _ in range(12):
            _check(cancellation)
            res=A@coef-target;rs=max(1.4826*np.median(abs(res-np.median(res))),1e-6)
            hub=np.minimum(1,2*rs/np.maximum(abs(res),1e-12));w=weights*hub
            lhs=A.T@(w[:,None]*A)+sc*.03*(.05*np.eye(A.shape[1])+D.T@D)
            coef=np.linalg.solve(lhs,A.T@(w*target)+sc*.03*.05*prior)
    except np.linalg.LinAlgError:return done(None,False,['bn_fit_singular'])
    if not np.isfinite(coef).all():return done(None,False,['bn_fit_nonfinite'])
    logbase=np.log(b);baseline_fine=logbase-savgol_filter(logbase,config.template_window,config.template_polyorder)
    delta=(A@coef)-baseline_fine
    # Data-dependent scalar bound preserves smoothness (no per-pixel clipping).
    variation=float(np.max(np.sqrt(np.mean((fine-baseline_fine)**2,axis=1))))
    limit=float(min(.2,max(.02,3*variation)))
    maximum=float(np.max(abs(delta)))
    if maximum>limit:delta*=limit/maximum
    baseline_ratio=z/b; baseline_norm=baseline_ratio/np.median(baseline_ratio,axis=1,keepdims=True)-1
    original_d2,used=production_d2(e,baseline_ratio-1,config);report['derivative_settings']['used_window_length']=int(used)
    spline=CubicSpline(pix,baseline_fine);strength=np.ones(len(e)); span=max(int(config.local_validation_window),int(used))
    starts=sorted(set([*range(0,max(1,len(e)-span+1),max(1,span//2)),max(0,len(e)-span)]))
    def evaluate(alpha):
        trial=b*np.exp(alpha*delta); ratio=z/trial; d2,_=production_d2(e,ratio-1,config)
        norm=ratio/np.median(ratio,axis=1,keepdims=True)-1
        checks=_full_axis_drr_check(baseline_norm,norm,e,baseline_fine,spline,config)
        failed=[]; wanted=np.ones(len(e))
        for start in starts:
            stop=min(len(e),start+span)
            old=np.sqrt(np.mean(original_d2[:,start:stop]**2,axis=1));new=np.sqrt(np.mean(d2[:,start:stop]**2,axis=1))
            ratio_degrade=float(np.max((new-old)/(config.max_local_degradation_fraction*old+config.max_absolute_d2_degradation+1e-15)))
            if ratio_degrade>1:
                failed.append(dict(energy_range=[float(e[start]),float(e[stop-1])],worst_ratio=ratio_degrade))
                left=max(0,start-used//2-2);right=min(len(e),stop+used//2+2)
                wanted[left:right]=np.minimum(wanted[left:right],max(.01,min(1.,.65/ratio_degrade)))
        for q in checks:
            if not q['passed']:
                lo,hi=q['region'];mask=(e>=lo)&(e<=hi);wanted[mask]=np.minimum(wanted[mask],.75)
        return trial,checks,failed,wanted
    for iteration in range(61):
        _check(cancellation)
        candidate,checks,failed,wanted=evaluate(strength)
        if not failed and all(q['passed'] for q in checks):break
        next_strength=gaussian_filter1d(strength*wanted,sigma=20,mode='nearest')
        if np.max(abs(next_strength-strength))<1e-8:break
        strength=next_strength
    # The final attenuation may have been updated on the last allowed loop.
    candidate,checks,failed,_=evaluate(strength)
    original_score=float(np.sqrt(np.average((baseline_fine-target)**2,weights=weights)))
    candidate_score=float(np.sqrt(np.average((baseline_fine+strength*delta-target)**2,weights=weights)))
    max_correction=float(np.max(abs(strength*delta)))
    improvement=original_score>1e-10 and candidate_score<original_score*(1-config.min_improvement_fraction)
    no_change=max_correction<1e-7
    finite=bool(np.isfinite(candidate).all() and np.all(candidate>0))
    passed=bool(finite and not failed and all(q['passed'] for q in checks) and improvement and not no_change)
    validation=dict(passed=passed,finite_positive=finite,all_gate_d2_check=not failed,
        all_gate_drr_check=all(q['passed'] for q in checks),same_scan_improvement=bool(improvement and not no_change),
        d2_windows_checked=len(starts),d2_failed_windows=len(failed),d2_failure_details=failed,drr_regions=checks,
        original_texture_rms=original_score,candidate_texture_rms=candidate_score,independent_validation=False)
    model=dict(basis_source='measured_bn_only',knots=6,template_count=len(ts),template_rank=rank,
        template_singular_values=sig.tolist(),coefficients=coef.tolist(),correction_limit=limit,max_log_correction=max_correction,
        alpha_min=float(np.min(strength)),alpha_max=float(np.max(strength)),alpha_mean=float(np.mean(strength)),
        alpha_sha256=hashlib.sha256(np.asarray(strength,dtype='<f8').tobytes()).hexdigest(),iterations=iteration+1,
        score_metric='weighted_bn_template_highpass_rms',regularization=.03,
        template_window=config.template_window,template_polyorder=config.template_polyorder,
        energy_range=[float(e.min()),float(e.max())],protected_regions=[],fit_method='energy-balanced robust IRLS with smooth BN-only basis')
    reasons=['same_scan_only_no_independent_peak_validation'] if passed else ['multi_bn_candidate_failed_checks']
    return done(candidate,passed,reasons,validation,model)
