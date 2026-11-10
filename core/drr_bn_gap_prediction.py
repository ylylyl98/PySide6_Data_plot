"""Experimental sample-assisted energy-gap BN prediction, preview only.

This does not provide independent validation or activate a background. The
sample-wide centering and subsequent high-frequency stage share information
across the excluded gap; this is not a strict statistical holdout.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from scipy.signal import hilbert, savgol_filter
from core.drr_bn_calibration import BnCalibrationResult, _check
from core.drr_bn_joint import production_d2
from core.drr_bn_settings import BnCalibrationSettings
from core.drr_bn_single import _full_axis_drr_check
from core.drr_bn_single_high_frequency import _filter, CUTOFF, TRANSITION

@dataclass(frozen=True)
class GapPredictionSettings:
    gap_ev: float = .018
    shoulder_ev: float = .004
    ridge: float = .1
    nodes: int = 4
    continuum_nodes: int = 20
    hop: int = 8
    include_high_frequency: bool = True


def _spline(x, nodes):
    knots=np.r_[[0]*4,np.linspace(0,len(x)-1,nodes-2)[1:-1],[len(x)-1]*4]
    return BSpline.design_matrix(x,knots,3).toarray()


def _basis(fine, nodes):
    n=len(fine); x=np.arange(n); pad=256
    f=np.fft.rfftfreq(n+2*pad); spectrum=np.fft.rfft(np.pad(fine,pad,mode='reflect'))
    W=_spline(x,nodes); fields=[]
    for lo,hi in zip((.012,.035,.065,.11),(.035,.065,.11,.20)):
        w=.5*(1+np.tanh((f-lo)/.003))*.5*(1-np.tanh((f-hi)/.003))
        band=np.fft.irfft(spectrum*w,n=n+2*pad)[pad:-pad]
        analytic=hilbert(np.pad(band,pad,mode='reflect'))[pad:-pad]
        fields.extend([W*analytic.real[:,None],W*analytic.imag[:,None]])
    return np.column_stack(fields)


def fit_energy_gap(energy, measurements, background, *, gap_settings=None, settings=None, cancellation=None):
    """Fit an ascending-energy prediction. Returns preview and diagnostics."""
    cfg=settings or BnCalibrationSettings(); opt=gap_settings or GapPredictionSettings()
    if not isinstance(cfg, BnCalibrationSettings): cfg=BnCalibrationSettings(**dict(cfg))
    e=np.asarray(energy,float); z=np.asarray(measurements,float); b=np.asarray(background,float)
    if (e.ndim!=1 or z.ndim!=2 or z.shape[1]!=len(e) or b.shape!=e.shape or len(z)<1 or
        len(e)<max(cfg.template_window,100) or not np.all(np.diff(e)>0) or
        not all(np.isfinite(a).all() for a in (e,z,b)) or np.any(z<=0) or np.any(b<=0)):
        raise ValueError('Expected positive finite measurements on an ascending energy grid')
    if (not np.isfinite([opt.gap_ev,opt.shoulder_ev,opt.ridge]).all() or min(opt.gap_ev,opt.shoulder_ev,opt.ridge)<=0 or
        not all(isinstance(v,int) for v in (opt.nodes,opt.continuum_nodes,opt.hop)) or
        opt.nodes<4 or opt.continuum_nodes<4 or max(opt.nodes,opt.continuum_nodes)>=len(e) or opt.hop<1 or opt.hop>=len(e)):
        raise ValueError('Invalid gap prediction settings')
    _check(cancellation)
    logb=np.log(b); fine=logb-savgol_filter(logb,cfg.template_window,cfg.template_polyorder)
    if np.sqrt(np.mean(fine*fine))<1e-10:
        return b.copy(),dict(reason='flat_bn_texture',effective_minimum_gap_ev=float(opt.gap_ev))
    n=len(e);x=np.arange(n); X=_basis(fine,opt.nodes); C=_spline(x,opt.continuum_nodes); A=np.column_stack((X,C));p=X.shape[1]
    rows=np.log(z)-np.log(b)[None,:];rows-=np.mean(rows,axis=1,keepdims=True); y=np.mean(rows,axis=0)
    for _ in range(50):
        _check(cancellation);w=1/np.sqrt(1+((rows-y)/.02)**2);nxt=np.sum(w*rows,axis=0)/np.sum(w,axis=0)
        if np.max(abs(nxt-y))<1e-10: break
        y=nxt
    w=np.mean(w,axis=0); norm=float(np.sum(X*X)/p); penalty=np.r_[np.full(p,opt.ridge*norm),np.full(C.shape[1],1e-8)]
    anchors=np.unique(np.r_[np.arange(0,n,opt.hop),n-1]); maxhop=float(np.max(np.diff(e[anchors])))
    # The extra anchor spacing prevents interpolation from importing a sample
    # within gap_ev of any interpolated output point.
    exclusion=opt.gap_ev+maxhop; coeff=[];conditions=[]
    for anchor in anchors:
        _check(cancellation);t=np.clip((abs(e-e[anchor])-exclusion)/opt.shoulder_ev,0,1);train=t**3*(10-15*t+6*t*t)
        if np.count_nonzero(train)>A.shape[1]:
            beta=np.r_[np.zeros(p),np.full(C.shape[1],np.mean(y))]
            for _ in range(12):
                ww=w/np.sqrt(1+((y-A@beta)/.01)**2)*train
                mat=A.T@(A*ww[:,None])+np.diag(penalty);rhs=A.T@(y*ww)
                scale=np.sqrt(np.diag(mat));scaled=mat/scale[:,None]/scale[None,:]
                nxt=np.linalg.solve(scaled,rhs/scale)/scale
                if np.max(abs(nxt-beta))<1e-6: break
                beta=nxt
            coeff.append(nxt[:p]);conditions.append(float(np.linalg.cond(scaled)))
        else:
            raise ValueError('Insufficient training outside energy gap')
    fields=np.column_stack([np.interp(x,anchors,np.asarray(coeff)[:,j]) for j in range(p)])
    delta=np.sum(X*fields,axis=1);delta*=min(1.,.10/max(float(np.max(abs(delta))),1e-12))
    with np.errstate(over='raise', invalid='raise'):
        candidate=b*np.exp(delta)
    if not np.isfinite(candidate).all() or np.any(candidate<=0):
        raise FloatingPointError('Nonfinite or nonpositive gap background')
    hf_used=bool(opt.include_high_frequency and len(z)>=3)
    if hf_used:
        target=np.mean(np.log(z)-np.log(candidate)[None,:],axis=0);target-=savgol_filter(target,cfg.template_window,cfg.template_polyorder)
        with np.errstate(over='raise', invalid='raise'):
            candidate=candidate*np.exp(_filter(target))
    if not np.isfinite(candidate).all() or np.any(candidate<=0):
        raise FloatingPointError('Nonfinite or nonpositive preview background')
    return candidate,dict(reason='preview',effective_minimum_gap_ev=float(opt.gap_ev),anchor_exclusion_ev=float(exclusion),
        max_anchor_spacing_ev=maxhop,anchor_count=len(anchors),scaled_condition_max=max(conditions),
        scaled_condition_median=float(np.median(conditions)),max_gap_log_change=float(np.max(abs(delta))),
        high_frequency_source='sample_shared_response_not_gap_excluded' if hf_used else ('insufficient_gates' if opt.include_high_frequency else 'disabled'),
        high_frequency_cutoff_cycles_per_pixel=CUTOFF,high_frequency_transition_cycles_per_pixel=TRANSITION)


def evaluate_bn_gap(energy, measurements, background, *, gates, settings=None, background_repeats=None, cancellation=None):
    cfg=settings or BnCalibrationSettings(); _check(cancellation)
    if not isinstance(cfg, BnCalibrationSettings): cfg=BnCalibrationSettings(**dict(cfg))
    e=np.asarray(energy,float);z=np.asarray(measurements,float);b=np.asarray(background,float);g=np.asarray(gates,float)
    report={'schema_version':1,'optimization_policy':'gap_prediction_experimental_v1','independent_validation':False,
        'measurement_count':1,'model':'energy_gap_prediction','candidate_generated':False,
        'source':'measured_bn_and_sample_assisted','fit_contract':{'bn_filter_bands_cycles_per_pixel':[[.012,.035],[.035,.065],[.065,.11],[.11,.20]],'band_transition_cycles_per_pixel':.003,'padding_pixels':256,'template_window':cfg.template_window,'template_polyorder':cfg.template_polyorder,'shared_robust_scale':.02,'fit_robust_scale':.01,'smooth_continuum_nodes':20,'maximum_log_gap_correction':.10},'gap_settings':asdict(GapPredictionSettings()),'settings':asdict(cfg),
        'limitations':['Preview only; never automatically applied.','Sample-wide log centering and high-frequency stage share information across the gap; this is not a strict holdout.','Physical sample features can affect the fitted background.'],
        'joint_validation':{'passed':False}}
    def done(candidate,reasons):
        return BnCalibrationResult(b.copy(),{'original':b.copy(),'gap_preview':candidate.copy()},'gap_preview','original',b.copy(),False,'preview_only',tuple(reasons),report)
    if z.ndim==3 and z.shape[0]==1:z=z[0]
    if (e.ndim!=1 or b.shape!=e.shape or g.ndim!=1 or z.shape!=(len(g),len(e)) or len(g)<1 or len(e)<max(cfg.template_window,100) or
        not all(np.isfinite(a).all() for a in (e,z,b,g)) or np.any(z<=0) or np.any(b<=0) or not (np.all(np.diff(e)>0) or np.all(np.diff(e)<0))):
        report['reason']='invalid_input';return done(b,['invalid_input'])
    reverse=e[0]>e[-1];ee=e[::-1] if reverse else e;zz=z[:,::-1] if reverse else z;bb=b[::-1] if reverse else b
    if background_repeats is not None:
        r=np.asarray(background_repeats,float)
        if r.ndim!=2 or r.shape[1]!=len(e) or not np.isfinite(r).all() or np.any(r<=0):
            report['reason']='invalid_background_repeats';return done(b,['invalid_background_repeats'])
        report['background_repeats']={'count':len(r),'used_for_fitting':False}
    try:
        candidate,info=fit_energy_gap(ee,zz,bb,settings=cfg,cancellation=cancellation)
        report['fit']=info
        if info['reason']=='flat_bn_texture':return done(b,['flat_bn_texture'])
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            old=zz/bb;new=zz/candidate; oldnorm=old/np.median(old,axis=1,keepdims=True)-1;newnorm=new/np.median(new,axis=1,keepdims=True)-1
        if not all(np.isfinite(a).all() for a in (old,new,oldnorm,newnorm)): raise FloatingPointError('Nonfinite ratio during validation')
        report['candidate_generated']=True
        d0,used=production_d2(ee,old-1,cfg);d1,_=production_d2(ee,new-1,cfg);n0,_=production_d2(ee,oldnorm,cfg);n1,_=production_d2(ee,newnorm,cfg)
        span=min(len(ee),max(int(cfg.local_validation_window),int(used)));starts=sorted(set([*range(0,len(ee)-span+1,max(1,span//2)),len(ee)-span]));fail=[]
        for start in starts:
            _check(cancellation);stop=start+span
            for kind,left,right in (('raw',d0,d1),('normalized',n0,n1)):
                before=np.sqrt(np.mean(left[:,start:stop]**2,axis=1));after=np.sqrt(np.mean(right[:,start:stop]**2,axis=1));allow=cfg.max_local_degradation_fraction*before+cfg.max_absolute_d2_degradation;bad=np.flatnonzero(after>before+allow)
                if len(bad):
                    i=int(bad[np.argmax((after-before-allow)[bad])]);fail.append({'kind':kind,'energy_range':[float(ee[start]),float(ee[stop-1])],'gate_index':i,'failing_gate_count':len(bad),'old_rms':float(before[i]),'new_rms':float(after[i]),'allowed':float(allow[i])})
        fine=np.log(bb)-savgol_filter(np.log(bb),cfg.template_window,cfg.template_polyorder)
        checks=_full_axis_drr_check(oldnorm,newnorm,ee,fine,CubicSpline(np.arange(len(ee)),fine),cfg)
        report['validation']={'passed':not fail and all(q['passed'] for q in checks),'all_gate_raw_d2_pass':not any(v['kind']=='raw' for v in fail),'all_gate_normalized_d2_pass':not any(v['kind']=='normalized' for v in fail),'failed_d2_windows':fail,'checked_d2_windows':len(starts),'full_axis_drr_reports':checks,'full_axis_drr_pass':all(q['passed'] for q in checks),'derivative_window_used':used}
        return done(candidate[::-1] if reverse else candidate,['experimental_preview_only'])
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        report['candidate_generated']=False;report['reason']='fit_failed';report['error']=str(exc);return done(b,['fit_failed'])
