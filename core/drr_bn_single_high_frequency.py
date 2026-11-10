"""Conservative same-scan high-frequency refinement of a BN phase result.

The additional texture is estimated from the measurement's gate-shared
response, not solely from BN. It can therefore alter real gate-common peaks.
No independent physical validation or activation occurs here.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from core.drr_bn_calibration import _check
from core.drr_bn_joint import production_d2
from core.drr_bn_settings import BnCalibrationSettings

CUTOFF = .20
TRANSITION = .008
PADDING = 256


def _filter(values):
    y=np.pad(np.asarray(values,float),PADDING,mode='reflect')
    f=np.fft.rfftfreq(len(y))
    return np.fft.irfft(np.fft.rfft(y)*(.5*(1+np.tanh((f-CUTOFF)/TRANSITION))),n=len(y))[PADDING:-PADDING]


def refine_shared_high_frequency(e, z, original, phase, fine, settings, cancellation=None):
    """Return (background, info), keeping phase on any failed qualification.

    ``original`` is the unmodified BN denominator, ``phase`` is an already
    guarded candidate. Every final gate and window is checked against original.
    """
    _check(cancellation)
    config=settings if isinstance(settings,BnCalibrationSettings) else BnCalibrationSettings(**dict(settings))
    energy=np.asarray(e,float); data=np.asarray(z,float); base=np.asarray(original,float)
    phase=np.asarray(phase,float); fine=np.asarray(fine,float)
    info=dict(schema_version=1,passed=False,source='sample_shared_high_frequency',cutoff=CUTOFF,transition=TRANSITION,
              estimator='gate_log_mean_v1',shrink_policy='monotone_smoothed_factor_v1',baseline_source='phase_candidate',
              cutoff_cycles_per_pixel=CUTOFF,transition_cycles_per_pixel=TRANSITION,
              padding=PADDING,independent_validation=False,
              limitation='Same-scan shared signal may contain physical peaks; no independent validation.')
    def fallback(reason):
        info['reason']=reason
        return phase.copy(),info
    if (energy.ndim!=1 or data.ndim!=2 or data.shape[1:]!=energy.shape or len(data)<1
        or base.shape!=energy.shape or phase.shape!=energy.shape or fine.shape!=energy.shape
        or len(energy)<=PADDING or len(energy)<max(config.template_window,config.derivative_window+2)
        or not all(np.isfinite(a).all() for a in (energy,data,base,phase,fine))
        or np.any(data<=0) or np.any(base<=0) or np.any(phase<=0)
        or not (np.all(np.diff(energy)>0) or np.all(np.diff(energy)<0))
        or not isinstance(config.template_window,int) or config.template_window<3 or config.template_window%2!=1
        or not isinstance(config.template_polyorder,int) or not 0<=config.template_polyorder<config.template_window
        or not np.isfinite(config.min_improvement_fraction) or not 0<=config.min_improvement_fraction<1
        or not np.isfinite(config.max_local_degradation_fraction) or config.max_local_degradation_fraction<0
        or not np.isfinite(config.max_absolute_d2_degradation) or config.max_absolute_d2_degradation<0
        or not np.isfinite(config.drr_noninferiority_fraction) or config.drr_noninferiority_fraction<0
        or not np.isfinite(config.drr_noninferiority_absolute) or config.drr_noninferiority_absolute<0
        or config.local_validation_window<3):
        return fallback('invalid_input_or_settings')
    if len(data)<3:
        return fallback('insufficient_gates_for_shared_response')
    reverse=energy[0]>energy[-1]
    if reverse:
        energy=energy[::-1].copy();data=data[:,::-1].copy();base=base[::-1].copy();phase=phase[::-1].copy();fine=fine[::-1].copy()
    def done(candidate):return candidate[::-1].copy() if reverse else candidate.copy()
    # The mean is linear across gates and avoids rank/mask switching under
    # a small moving peak. It remains a sample-derived, same-scan estimate.
    logs=np.log(data);_,used=production_d2(energy,logs,config)
    high=logs-savgol_filter(logs,config.template_window,config.template_polyorder,axis=1)-fine
    target=np.mean(high,axis=0)
    change=np.log(phase/base);target-=change-savgol_filter(change,config.template_window,config.template_polyorder)
    delta=_filter(target)
    initial_delta=delta.copy()
    old_score=float(np.sqrt(np.mean(_filter(initial_delta)**2)))
    info.update(mask_fraction=0.0,used_window=int(used),original_hf_rms=old_score,original_shared_hf_rms=old_score)
    if not np.isfinite(delta).all() or old_score<1e-9:
        info['reason']='no_resolvable_high_frequency_response';return done(phase),info
    # Bound the complete correction without introducing pixelwise clipping.
    max_abs=float(np.max(abs(delta))); limit=.2
    if max_abs>limit:delta*=limit/max_abs
    old_ratio=data/base;old_raw=old_ratio-1
    old_norm=old_ratio/np.median(old_ratio,axis=1,keepdims=True)-1
    old_raw_d2,_=production_d2(energy,old_raw,config)
    old_norm_d2,_=production_d2(energy,old_norm,config)
    spline=CubicSpline(np.arange(len(energy)),fine)
    from core.drr_bn_single import _full_axis_drr_check, _projected_drr_check
    x=np.linspace(-1,1,len(energy));q,_=np.linalg.qr(np.column_stack((np.ones(len(x)),x,x*x)))
    response=np.stack((-spline(np.arange(len(energy)),1),-fine))
    span=min(len(energy),max(int(config.local_validation_window),int(used)));stride=max(1,span//2)
    starts=sorted(set([*range(0,len(energy)-span+1,stride),len(energy)-span]))
    def evaluate(alpha):
        _check(cancellation)
        candidate=phase*np.exp(alpha*delta);ratio=data/candidate
        raw=ratio-1;normalized=ratio/np.median(ratio,axis=1,keepdims=True)-1
        raw_d2,_=production_d2(energy,raw,config);norm_d2,_=production_d2(energy,normalized,config)
        failures=[];desired=np.ones(len(energy))
        for start in starts:
            stop=start+span
            for name,old,new in (('raw',old_raw_d2,raw_d2),('normalized',old_norm_d2,norm_d2)):
                before=np.sqrt(np.mean(old[:,start:stop]**2,axis=1));after=np.sqrt(np.mean(new[:,start:stop]**2,axis=1))
                tolerance=config.max_local_degradation_fraction*before+config.max_absolute_d2_degradation
                violation=after-before-tolerance
                if np.any(violation>0):
                    j=int(np.argmax(violation));factor=min(1.,.98/max(float(np.max((after-before)/np.maximum(tolerance,1e-15))),1.000000001))
                    left=max(0,start-used//2-2);right=min(len(energy),stop+used//2+2)
                    desired[left:right]=np.minimum(desired[left:right],factor)
                    failures.append(dict(kind=name,energy_range=[float(energy[start]),float(energy[stop-1])],
                        gate_index=j,old_rms=float(before[j]),new_rms=float(after[j]),allowed=float(tolerance[j])))
        regional=_full_axis_drr_check(old_norm,normalized,energy,fine,spline,config)
        for item in regional:
            if not item['passed']:
                lo,hi=item['region'];m=(energy>=lo)&(energy<=hi);desired[m]=np.minimum(desired[m],.9)
        whole=_projected_drr_check(old_norm,normalized,response,np.ones(len(energy),bool),q,config)
        if not whole['passed']:desired=np.minimum(desired,.9)
        return candidate,failures,regional,whole,desired
    alpha=np.ones(len(energy));iterations=0
    for iterations in range(1,81):
        candidate,failures,regional,whole,desired=evaluate(alpha)
        if not failures and all(v['passed'] for v in regional) and whole['passed']:break
        updated=alpha*gaussian_filter1d(desired,sigma=20,mode='nearest')
        if np.max(abs(updated-alpha))<1e-8:break
        alpha=updated
    candidate,failures,regional,whole,_=evaluate(alpha)
    new_score=float(np.sqrt(np.mean(_filter(initial_delta-alpha*delta)**2)))
    improvement=bool(np.isfinite(new_score) and new_score<old_score*(1-config.min_improvement_fraction))
    finite=bool(np.isfinite(candidate).all() and np.all(candidate>0))
    success=bool(finite and not failures and all(q['passed'] for q in regional) and whole['passed'] and improvement
                 and np.max(abs(np.log(candidate/phase)))>1e-9)
    info.update(passed=success,reason='accepted' if success else 'final_guards_or_hf_improvement_failed',
        finite_positive=finite,alpha_min=float(alpha.min()),alpha_max=float(alpha.max()),alpha_mean=float(alpha.mean()),
        strength_min=float(alpha.min()),strength_max=float(alpha.max()),strength_mean=float(alpha.mean()),
        iterations=iterations,failed_d2_windows=failures,checked_d2_windows=len(starts),
        all_gate_raw_d2_pass=not any(f['kind']=='raw' for f in failures),
        all_gate_normalized_d2_pass=not any(f['kind']=='normalized' for f in failures),
        full_axis_drr_pass=all(q['passed'] for q in regional),full_axis_drr_reports=regional,
        whole_axis_drr_pass=bool(whole['passed']),whole_axis_drr_report=whole,whole_axis_drr_projection=whole,
        candidate_hf_rms=new_score,candidate_shared_hf_rms=new_score,hf_improvement_pass=improvement,improvement_pass=improvement,
        max_log_refinement=float(np.max(abs(np.log(candidate/phase)))),correction_limit=limit)
    return done(candidate if success else phase),info
