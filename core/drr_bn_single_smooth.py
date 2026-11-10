"""Low-dimensional full-axis measured-BN texture model for single scans."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline, BSpline
from scipy.optimize import least_squares
from core.drr_bn_joint import production_d2
from core.drr_bn_calibration import _check


def energy_balanced_bn_d2_score(e, fine, old_d2, new_d2, settings):
    """Full-gate D2 score on the same six BN-only bases used during fitting.

    Each window is normalized by its own original all-gate projection RMS,
    so each contributes equally as a relative change. A numerically negligible
    original window cannot define a stable relative score and is rejected.
    This in-sample metric is not a physical truth test.
    """
    pix=np.arange(len(e),dtype=float)
    breakpoints=np.linspace(0,len(e)-1,4)
    knot_vector=np.r_[np.repeat(breakpoints[0],4),breakpoints[1:-1],np.repeat(breakpoints[-1],4)]
    matrix=BSpline.design_matrix(pix,knot_vector,3).toarray()
    spline=CubicSpline(pix,fine,bc_type='natural',extrapolate=True)
    fields=np.concatenate((-matrix.T*spline(pix,1),-matrix.T*fine),axis=0)
    fields_d2,_=production_d2(e,fields,settings)
    blocks=[]
    for ids in np.array_split(np.arange(len(e)),6):
        x=np.linspace(-1,1,len(ids)); q,_=np.linalg.qr(np.column_stack((np.ones(len(ids)),x,x*x)))
        f=fields_d2[:,ids]; f=f-(f@q)@q.T
        m=f.T; norms=np.linalg.norm(m,axis=0); valid=norms>1e-10
        if not np.any(valid): continue
        u,s,_=np.linalg.svd(m[:,valid]/norms[valid],full_matrices=False)
        basis=u[:,s>s[0]*1e-5]
        def projection(v):
            v=v[:,ids]; return (v-(v@q)@q.T)@basis/np.sqrt(len(ids))
        before=projection(old_d2); after=projection(new_d2)
        scale=float(np.sqrt(np.mean(before**2)))
        if not np.isfinite(scale) or scale <= 1e-5:
            return None
        relative=float(np.sqrt(np.mean(after**2)))/scale
        if not np.isfinite(relative):
            return None
        blocks.append(dict(energy_range=[float(e[ids[0]]),float(e[ids[-1]])],
            original=1.0, candidate=relative, scale=scale))
    if len(blocks) != 6: return None
    return dict(metric='energy_balanced_bn_projected_d2_rms',
        original=float(np.mean([v['original'] for v in blocks])),
        candidate=float(np.mean([v['candidate'] for v in blocks])),
        aggregation='equal-weight six windows; each is candidate all-gate RMS divided by original all-gate RMS',
        windows=blocks)


def fit_smooth_texture(e, z, b, fine, settings, *, cancellation=None, segment_seeds=None):
    """Fit six smooth knots per parameter; return a candidate and diagnostics.

    Only the measured BN defines the projection basis. The input scan is used
    for fitting, so correlated sample features remain fundamentally ambiguous.
    """
    pix = np.arange(len(e), dtype=float)
    breakpoints = np.linspace(0, len(e)-1, 4)
    knot_vector = np.r_[np.repeat(breakpoints[0],4),breakpoints[1:-1],np.repeat(breakpoints[-1],4)]
    matrix = BSpline.design_matrix(pix,knot_vector,3).toarray()
    knots = np.array([np.mean(knot_vector[i+1:i+4]) for i in range(6)])
    spline = CubicSpline(pix, fine, bc_type='natural', extrapolate=True)
    chosen = np.unique(np.linspace(0, len(z)-1, min(settings.gate_count,len(z))).astype(int))
    raw = z[chosen]/b
    old = raw/np.median(raw,axis=1,keepdims=True)-1
    old_d2, _ = production_d2(e,old,settings)
    # Differential response fields are generated entirely from measured BN.
    fields = np.concatenate((-matrix.T*spline(pix,1),-matrix.T*fine),axis=0)
    fields_d2, _ = production_d2(e,fields,settings)
    # Give equal weight to each of six contiguous energy windows. Each window
    # gets its own BN-only orthonormal basis and scale; no strong region can
    # dominate merely because its curvature has greater units or magnitude.
    blocks=[]
    for ids in np.array_split(np.arange(len(e)),6):
        x=np.linspace(-1,1,len(ids)); q,_=np.linalg.qr(np.column_stack((np.ones(len(ids)),x,x*x)))
        def basis(fields):
            f=fields[:,ids]; f=f-(f@q)@q.T
            m=f.T; norms=np.linalg.norm(m,axis=0); valid=norms>1e-10
            if not np.any(valid): return None
            u,s,_=np.linalg.svd(m[:,valid]/norms[valid],full_matrices=False)
            return u[:,s>s[0]*1e-5]
        b1,b2=basis(fields),basis(fields_d2)
        if b1 is None or b2 is None:
            continue
        def projection(v,basis,q=q,ids=ids):
            v=v[:,ids]; v=v-(v@q)@q.T
            return v@basis/np.sqrt(len(ids))
        p1,p2=projection(old,b1),projection(old_d2,b2)
        scale1=max(float(np.sqrt(np.mean(p1*p1))),1e-7)
        scale2=max(float(np.sqrt(np.mean(p2*p2))),1e-5)
        blocks.append((ids,q,b1,b2,scale1,scale2))
    if len(blocks) != 6: return None,dict(reason='no_resolvable_bn_basis')
    def project(v,basis,q,ids):
        v=v[:,ids]; v=v-(v@q)@q.T
        return v@basis/np.sqrt(len(ids))
    upper_shift = min(float(settings.single_max_shift_pixels),10.)
    low_gain, high_gain = settings.single_smooth_gain_bounds
    def candidate(p):
        # Nonnegative partition-of-unity basis preserves coefficient bounds
        # continuously, with no derivative kinks from clipping the fields.
        shift=matrix@p[:6]
        gain=matrix@p[6:]
        # Natural cubic extrapolation is confined to <= shift_limit pixels;
        # full-axis local D2 checks include both measured endpoints.
        return b*np.exp(gain*spline(pix+shift)-fine)
    def residual(p):
        _check(cancellation)
        trial=candidate(p); ratio=z[chosen]/trial
        drr=ratio/np.median(ratio,axis=1,keepdims=True)-1
        d2,_=production_d2(e,drr,settings)
        components=[]
        for ids,q,b1,b2,scale1,scale2 in blocks:
            r1=project(drr,b1,q,ids)/scale1
            r2=project(d2,b2,q,ids)/scale2
            components.extend((r1.ravel()/np.sqrt(r1.size),r2.ravel()/np.sqrt(r2.size)))
        # Penalize curvature and size rather than a freely varying correction.
        regularizer=np.r_[.12*np.diff(p[:6],2)/upper_shift,
                          .12*np.diff(p[6:],2)/(high_gain-low_gain),
                          .03*p[:6]/upper_shift, .03*(p[6:]-1)/(high_gain-low_gain)]
        return np.r_[*components,regularizer]
    lower=np.r_[np.full(6,-upper_shift),np.full(6,low_gain)]
    upper=np.r_[np.full(6,upper_shift),np.full(6,high_gain)]
    initial=np.r_[np.zeros(6),np.ones(6)]
    if segment_seeds:
        centers=[]; shifts=[]; gains=[]
        for item in segment_seeds:
            center=np.mean(item['region']); centers.append(center)
            shifts.append(0. if item.get('at_parameter_bound') else float(item.get('shift_pixels',0.)))
            gains.append(1. if item.get('at_parameter_bound') else float(item.get('fine_gain',1.)))
        if centers:
            order=np.argsort(centers); knot_e=np.interp(knots,pix,e)
            initial[:6]=np.interp(knot_e,np.asarray(centers)[order],np.asarray(shifts)[order])
            initial[6:]=np.interp(knot_e,np.asarray(centers)[order],np.asarray(gains)[order])
    initial=np.clip(initial,lower+1e-8,upper-1e-8)
    fit=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',f_scale=.1,
                      x_scale='jac',ftol=1e-5,xtol=1e-6,gtol=1e-6,max_nfev=100)
    fitted=candidate(fit.x)
    return fitted,dict(success=bool(fit.success), evaluations=int(fit.nfev),
        node_count=6, basis_source='measured_bn_only', basis_method='clamped_cubic_bspline',
        shift_knots=fit.x[:6].tolist(), gain_knots=fit.x[6:].tolist(),
        knot_pixels=knots.tolist(), shift_limit_pixels=upper_shift,
        edge_policy='natural cubic extrapolation up to bounded shift; endpoints are locally screened',
        drr_basis_rank=[int(item[2].shape[1]) for item in blocks], d2_basis_rank=[int(item[3].shape[1]) for item in blocks],
        objective='Six energy-balanced BN-only projections of normalized DR/R and production D2, robust loss and smoothness regularization',
        independent_validation=False)
