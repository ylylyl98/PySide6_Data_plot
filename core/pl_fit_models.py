"""PL-only asymmetric peak model and masked constrained least squares.

Parameter layout: intercept, slope, (height, center, total FWHM)*N, then
asymmetry for the designated component when enabled. Positive asymmetry widens
the left half. Fixed centers are absent from the solver's free vector.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from core.power_multi_peaks import SpectrumFit, MultiPeakSettings
from core.power_peaks import PeakFit, full_peak_area, propagated_error


@dataclass(frozen=True)
class PLSeed:
    center_ev: float
    fwhm_ev: float
    height: float | None = None
    center_min_ev: float | None = None
    center_max_ev: float | None = None
    center_fixed: bool = False


def _component(x, height, center, width, family, asym=0.):
    sides = np.where(x < center, 1. + asym, 1. - asym)
    t = 2 * (x-center) / (width * sides)
    return height / (1+t*t) if family == 'Lorentzian' else height*np.exp(-np.log(2.)*t*t)


def evaluate_component(x, fit, index):
    """Evaluate one baseline-free component for old and new fit records."""
    parameters = fit['parameters']
    offset = 2 + 3 * index
    family = 'Gaussian' if 'Gaussian' in fit['model'] else 'Lorentzian'
    asym = parameters[-1] if fit['model'].startswith('Asymmetric ') and index == fit.get('asymmetric_index') else 0.
    return _component(np.asarray(x, float), *parameters[offset:offset+3], family, asym)


def evaluate_fit(x, fit):
    """Evaluate the full linear-intensity model for an old or new record."""
    x = np.asarray(x, float)
    parameters = fit['parameters']
    count = (len(parameters) - 2 - int(fit['model'].startswith('Asymmetric '))) // 3
    out = parameters[0] + parameters[1]*(x-fit['origin'])
    for index in range(count):
        out = out + evaluate_component(x, fit, index)
    return out


def fit_pl_model(energy, spectrum, gate, seeds, model, *, asymmetric_index=None, cancelled=None):
    """Fit raw PL data; return a SpectrumFit plus free-parameter metadata."""
    xall = np.asarray(energy, float); yall = np.asarray(spectrum, float)
    if model not in ('Lorentzian', 'Gaussian', 'Asymmetric Lorentzian'):
        raise ValueError('Unknown PL peak model.')
    if xall.ndim != 1 or xall.shape != yall.shape or np.any(~np.isfinite(xall)) or np.any(np.diff(xall) <= 0):
        raise ValueError('Spectrum needs a matching, increasing, finite energy axis.')
    if not 1 <= len(seeds) <= 8:
        raise ValueError('Use 1–8 peak seeds.')
    asym = model.startswith('Asymmetric ')
    if asym and (asymmetric_index is None or not 0 <= asymmetric_index < len(seeds)):
        raise ValueError('Select a valid asymmetric component.')
    valid = np.isfinite(yall); x, y = xall[valid], yall[valid]
    components = tuple(PeakFit(float(gate), 'insufficient finite points') for _ in seeds)
    settings = MultiPeakSettings(tuple(seeds), model)
    result = SpectrumFit(float(gate), 'insufficient finite points', components, settings)
    if len(x) < max(8, 3*len(seeds)+6):
        return result, dict(n_vary=0, fixed_centers=[])
    scale=float(np.ptp(y)); dx=float(np.median(np.diff(x))); span=float(x[-1]-x[0]);origin=float(np.mean(x))
    if scale <= np.finfo(float).eps * max(1., float(np.max(np.abs(y)))):
        result.status='flat spectrum'
        for component in components: component.status=result.status
        return result, dict(n_vary=0, fixed_centers=[])
    edge=max(2,len(x)//20); slope=(np.median(y[-edge:])-np.median(y[:edge]))/span
    baseline=float(np.median(y[:edge])+slope*(origin-np.mean(x[:edge])))
    p0=[baseline/scale,slope/scale]; lower=[-np.inf,-np.inf]; upper=[np.inf,np.inf]; fixed=[]
    for i, seed in enumerate(seeds):
        lo=xall[0] if seed.center_min_ev is None else max(xall[0],seed.center_min_ev)
        hi=xall[-1] if seed.center_max_ev is None else min(xall[-1],seed.center_max_ev)
        if not np.isfinite(seed.center_ev) or not lo < hi or not lo <= seed.center_ev <= hi:
            raise ValueError(f'Peak {i+1}: center must lie within valid center bounds.')
        h=seed.height if seed.height is not None and np.isfinite(seed.height) and seed.height>0 else max(float(np.interp(seed.center_ev,x,y)-baseline-slope*(seed.center_ev-origin)),scale*.02)
        p0.extend([h/scale,seed.center_ev,float(np.clip(seed.fwhm_ev,dx*.51,span*.99))]);lower.extend([0.,lo,dx*.5]);upper.extend([np.inf,hi,span])
        if seed.center_fixed: fixed.append(3+3*i)
    if asym: p0.append(0.);lower.append(-.8);upper.append(.8)
    p0=np.asarray(p0,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    free=np.array([i for i in range(len(p0)) if i not in fixed],int)
    def unpack(values):
        params=p0.copy();params[free]=values;return params
    def fun(values):
        if cancelled and cancelled(): raise InterruptedError('Peak fitting cancelled.')
        params=unpack(values); record=dict(parameters=params,model=model,origin=origin,asymmetric_index=asymmetric_index)
        return evaluate_fit(x,record)-y/scale
    meta=dict(n_vary=len(free),fixed_centers=[i for i,s in enumerate(seeds) if s.center_fixed],asymmetric_index=asymmetric_index if asym else None)
    try:
        sol=least_squares(fun,p0[free],bounds=(lower[free],upper[free]),x_scale='jac',max_nfev=3000)
        if not sol.success: raise RuntimeError(sol.message)
        params=unpack(sol.x); actual=params.copy();actual[:2]*=scale;actual[2:2+3*len(seeds):3]*=scale
        record=dict(parameters=actual,model=model,origin=origin,asymmetric_index=asymmetric_index)
        result.x=x;result.fitted=evaluate_fit(x,record);result.residual=y-result.fitted;result.rms=float(np.sqrt(np.mean(result.residual**2)))
        result.parameters=tuple(float(v) for v in actual);result.baseline=actual[0]+actual[1]*(x-origin)
        result.spectrum_area_roi=float(np.trapezoid(y-result.baseline,x));result.spectrum_area_roi_domain=(float(x[0]),float(x[-1]));result.spectrum_area_roi_method='Trapezoidal integration of finite measured samples minus fitted linear baseline'
        covariance=None;uncertainty='unavailable: singular covariance'
        if len(x)>len(free) and np.linalg.matrix_rank(sol.jac)==len(free):
            jtj=sol.jac.T@sol.jac
            if np.linalg.cond(jtj)<1e14:
                covfree=np.linalg.inv(jtj)*(float(sol.fun@sol.fun)/(len(x)-len(free)))
                full=np.zeros((len(p0),len(p0)));full[np.ix_(free,free)]=covfree
                units=np.ones(len(p0));units[:2]=scale;units[2:2+3*len(seeds):3]=scale
                covariance=full*np.outer(units,units);uncertainty='ok'
        result.covariance=covariance;curves=[]
        for i,(seed,component) in enumerate(zip(seeds,components)):
            k=2+3*i;height,center,width=actual[k:k+3];curve=evaluate_component(x,record,i);curves.append(curve);component.center_ev=float(center);component.rms=result.rms;component.uncertainty_status=(('user fixed center' if uncertainty == 'ok' else 'user fixed center; ' + uncertainty) if seed.center_fixed else uncertainty)
            if width<=dx or width>=span*.95:component.status='width unresolved or reaches spectrum limit'
            elif not seed.center_fixed and min(center-lower[k+1],upper[k+1]-center)<dx*.25:component.status='center reaches tracking limit'
            elif asym and i==asymmetric_index and min(actual[-1]+.8,.8-actual[-1])<.005:component.status='asymmetry reaches limit'
            elif covariance is None:component.status='width uncertainty too large'
            elif not np.isfinite(covariance[k+2,k+2]) or np.sqrt(max(covariance[k+2,k+2],0))>width:component.status='width uncertainty too large'
            elif height<=3*result.rms:component.status='peak below residual noise'
            else:
                component.status='ok';component.fwhm_mev=float(width*1000);component.fwhm_error_mev=float(np.sqrt(max(covariance[k+2,k+2],0))*1000);component.height=float(height);component.center_error_ev=0. if seed.center_fixed else float(np.sqrt(max(covariance[k+1,k+1],0)));component.height_error=float(np.sqrt(max(covariance[k,k],0)));component.area=float(np.trapezoid(curve,x));component.fit_area_full,factor=full_peak_area(height,width,'Gaussian' if 'Gaussian' in model else 'Lorentzian')
                grad=np.zeros(len(actual));grad[k]=factor*width;grad[k+2]=factor*height;component.fit_area_full_error=propagated_error(grad,covariance)
                grad=np.zeros(len(actual))
                for j in (k,k+1,k+2,*([len(actual)-1] if asym and i==asymmetric_index else [])):
                    step=max(abs(actual[j])*1e-5,1e-9);plus=actual.copy();minus=actual.copy();plus[j]+=step;minus[j]-=step
                    rec1={**record,'parameters':plus};rec2={**record,'parameters':minus};grad[j]=(np.trapezoid(evaluate_component(x,rec1,i),x)-np.trapezoid(evaluate_component(x,rec2,i),x))/(2*step)
                component.area_error=propagated_error(grad,covariance)
        result.component_curves=tuple(curves);count=sum(c.status=='ok' for c in components);result.status='ok' if count==len(components) else f'{count}/{len(components)} peaks valid'
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        result.status=f'fit failed: {exc}'
        for c in components:c.status=result.status
    return result,meta
