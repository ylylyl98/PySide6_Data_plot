"""Matched-input CW sensitivity comparison; never replaces the primary fit."""
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import chi2
from core.curie_weiss import fit_curie_weiss, fit_inverse_curie_weiss


def _weighted(t, s, errors, base):
    if errors.shape != s.shape or not np.all(np.isfinite(errors) & (errors > 0)):
        raise ValueError('Positive finite slope SE required for every point; no points dropped.')
    y=s-base
    if not (np.all(y>0) or np.all(y<0)):
        raise ValueError('Slopes change sign or reach zero; inspect the optical response first.')
    if np.ptp(y)<=max(np.max(abs(y))*1e-10,1e-15):
        raise ValueError('Constant response: theta is not identifiable.')
    lo,hi=np.log(1e-7),np.log(max(1.,float(np.ptp(t)))*1e7)
    def solve(x):
        theta=float(t.min()-np.exp(x)); z=1/(t-theta)
        a=float(np.sum(z*y/errors**2)/np.sum((z/errors)**2))
        return float(np.sum(((y-a*z)/errors)**2)),a,theta
    grid=np.linspace(lo,hi,800);cost=np.array([solve(x)[0] for x in grid])
    candidates=[lo,hi]
    for j in range(1,len(grid)-1):
        if cost[j]<=cost[j-1] and cost[j]<=cost[j+1]:
            candidates.append(minimize_scalar(lambda x:solve(x)[0],bounds=(grid[j-1],grid[j+1]),method='bounded').x)
    x=min(candidates,key=lambda x:solve(x)[0]);q,a,theta=solve(x)
    boundary=bool(abs(x-lo)<1e-4 or abs(x-hi)<1e-4)
    def threshold(value):
        return solve(value)[0]-q-chi2.ppf(.95,1)
    def endpoint(values):
        previous=x
        for value in values:
            if threshold(value)>0:
                crossing=brentq(threshold,min(previous,value),max(previous,value),xtol=1e-12)
                return float(t.min()-np.exp(crossing))
            previous=value
        return None
    lower=endpoint(grid[grid>x])
    upper=endpoint(grid[grid<x][::-1])
    warnings=['Conditional profile 95% interval; assumes independent Gaussian slope errors; excludes systematic errors.']
    if lower is None or upper is None:
        warnings.append('Profile interval unbounded in scan / reaches pole boundary; theta not reliably constrained.')
    if boundary: warnings.append('Boundary solution: no finite identifiable theta reported.')
    if np.any(abs(y)<=1.96*errors): warnings.append('Some slope uncertainties include zero; reciprocal is unreliable.')
    reduced=q/(len(t)-2)
    if chi2.sf(q,len(t)-2)<.05: warnings.append('CW model mismatch under the supplied slope errors.')
    return dict(theta_k=None if boundary else theta,amplitude=a,background=base,ci95_low_k=lower,ci95_high_k=upper,
                reduced_chi2=reduced,status='diagnostic only' if boundary or lower is None or upper is None or chi2.sf(q,len(t)-2)<.05 else 'estimated',
                diagnostics=' '.join(warnings))


def compare_cw_methods(temperature, slope, slope_se, *, background_mode='zero', background=0.):
    """Three rows, identical input points; failures stay visible, no fallback weights."""
    t,s=np.asarray(temperature,float),np.asarray(slope,float)
    errors=np.asarray(slope_se,float)
    rows=[]
    for key,label,weights in [('inverse','Inverse slope / equal weights','equal in inverse space'),
                              ('slope','Slope / equal weights','equal in slope space'),
                              ('weighted','Slope / SE weighted','1/slope_se^2')]:
        row=dict(method=key,label=label,weights=weights,n=len(t),theta_k=None,amplitude=None,background=None,ci95_low_k=None,
                 ci95_high_k=None,reduced_chi2=None,status='unavailable',diagnostics='')
        try:
            if background_mode not in ('zero','fixed'):
                raise ValueError('Select zero or independently fixed background for matched comparison.')
            if t.ndim!=1 or s.shape!=t.shape or len(t)<3 or not np.all(np.isfinite(t)&(t>0)) or not np.all(np.isfinite(s)):
                raise ValueError('Need at least three finite positive temperatures and finite slopes.')
            if len(np.unique(np.round(t,8)))!=len(t):raise ValueError('Select one point per distinct temperature.')
            base=0. if background_mode=='zero' else float(background)
            if not np.isfinite(base):raise ValueError('Background must be finite.')
            if key=='weighted':row.update(_weighted(t,s,errors,base))
            else:
                try:
                    f=(fit_inverse_curie_weiss(t,s,slope_se=errors,background_mode=background_mode,background=base)
                       if key=='inverse' else fit_curie_weiss(t,s,background_mode=background_mode,background=base))
                except ValueError as exc:
                    if key=='inverse' and 'pole is not below' in str(exc):
                        m,b=np.polyfit(t,1/(s-base),1)
                        row.update(theta_k=float(-b/m),amplitude=float(1/m),background=base,status='diagnostic only',diagnostics='Unconstrained linear intercept; pole within/above selected temperatures. Not a valid paramagnetic CW result.')
                        rows.append(row);continue
                    raise
                ci=f['theta_ci95_k']
                row.update(theta_k=f['theta_k'],amplitude=f['amplitude'],background=f['background'],ci95_low_k=ci[0] if ci else None,ci95_high_k=ci[1] if ci else None,
                           status='estimated' if ci else 'diagnostic only',diagnostics=' '.join(f['warnings'])+' '+f['uncertainty'])
                if key=='slope' and any('at a fit bound' in w for w in f['warnings']):
                    row.update(theta_k=None,amplitude=None,status='unavailable')
        except (ValueError,np.linalg.LinAlgError) as exc:row['diagnostics']=str(exc)
        rows.append(row)
    return rows
