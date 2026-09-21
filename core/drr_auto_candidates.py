"""Heuristic candidate evidence; never a probability of physical reality."""
import numpy as np
from bisect import bisect_left,bisect_right
from scipy.signal import savgol_filter,savgol_coeffs


def derivative_noise_floor(x,raw,window,polyorder):
    """Approximate propagation of robust raw noise through the SG derivative."""
    delta=np.diff(raw);delta=delta[np.isfinite(delta)]
    if len(delta)<4 or not window:return 0.
    sigma=float(np.median(abs(delta-np.median(delta)))/(.67448975*np.sqrt(2)))
    step=float((x[-1]-x[0])/(len(x)-1))
    coefficients=savgol_coeffs(window,polyorder,deriv=2,delta=step)
    return sigma*float(np.linalg.norm(coefficients))


def assess_row(x,row,points,settings,noise_floor=0.):
    from core.drr_peak_analysis import _detect
    for point in points:
        point.update(confidence='low',noise_ratio=0.,scale_support=0,neighbor_support=0,
                     quality_reason='Not supported across smoothing scales')
    finite=np.isfinite(row);edges=np.diff(np.r_[False,finite,False].astype(int))
    for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        xx=x[start:end];values=row[start:end]
        if len(xx)<7 or np.ptp(values)==0:continue
        grid=np.linspace(xx[0],xx[-1],len(xx));uniform=np.interp(grid,xx,values)
        windows=sorted({min(w,len(xx) if len(xx)%2 else len(xx)-1) for w in (5,9,15)})
        signals=[savgol_filter(uniform,w,2,mode='interp') for w in windows]
        residual=uniform-signals[len(signals)//2]
        sigma=float(np.median(abs(residual-np.median(residual)))/.67448975)
        segment_floor=noise_floor(start,end) if callable(noise_floor) else noise_floor
        sigma=max(sigma,segment_floor,float(np.ptp(uniform))*1e-10,np.finfo(float).tiny)
        candidates=[_detect(xx,np.interp(xx,grid,signal),settings,True) for signal in signals]
        raw_by_polarity={polarity:sorted([p for p in points if p['polarity']==polarity and xx[0]<=p['energy']<=xx[-1]],key=lambda p:p['energy']) for polarity in ('peak','dip')}
        raw_energies={polarity:[p['energy'] for p in group] for polarity,group in raw_by_polarity.items()}
        stable_energies=[{polarity:sorted(p['energy'] for p in group if p['polarity']==polarity and p['prominence']>=sigma) for polarity in ('peak','dip')} for group in candidates]
        spacing=float(np.median(np.diff(xx)))
        for peak in candidates[len(candidates)//2]:
            tolerance=max(2*spacing,min(peak['width_mev']/4000,5*spacing))
            lo,hi=peak['energy']-tolerance,peak['energy']+tolerance;polarity=peak['polarity']
            support=sum(bisect_right(group[polarity],hi)>bisect_left(group[polarity],lo) for group in stable_energies)
            energies=raw_energies[polarity]
            nearby=raw_by_polarity[polarity][bisect_left(energies,lo):bisect_right(energies,hi)]
            if not nearby:continue
            point=min(nearby,key=lambda p:abs(p['energy']-peak['energy']))
            ratio=float(peak['prominence']/sigma)
            if ratio<=point['noise_ratio']:continue
            resolved=peak['width_mev']>=2*spacing*1000
            interior=xx[0]+spacing*windows[-1]/2<peak['energy']<xx[-1]-spacing*windows[-1]/2
            evidence=support>=2 and resolved and interior
            point.update(noise_ratio=ratio,scale_support=int(support),auto_width_mev=peak['width_mev'],
                         auto_prominence=peak['prominence'],auto_prominence_fraction=peak['prominence_fraction'],
                         auto_energy=peak['energy'],auto_eligible=bool(evidence and ratio>=2.5),
                         confidence='supported' if evidence and ratio>=5 else 'low',
                         quality_reason='Stable and above estimated noise' if evidence and ratio>=5 else 'Weak, narrow, edge-adjacent or scale-unstable')


def assess_neighbors(points,check_cancelled):
    rows={}
    for point in points:
        if point.get('auto_eligible'):rows.setdefault(point['row_index'],[]).append(point)
    # Compute support first; promotion does not recursively amplify weak noise.
    for row,group in rows.items():
        check_cancelled()
        for point in group:
            tolerance=min(3.,max(.5,point.get('auto_width_mev',0)/4))/1000
            neighbors=sum(any(p['polarity']==point['polarity'] and abs(p.get('auto_energy',p['energy'])-point.get('auto_energy',point['energy']))<=tolerance
                              for p in rows.get(adjacent,[])) for adjacent in (row-1,row+1))
            point['neighbor_support']=neighbors
            if neighbors==2 and point['confidence']=='low':
                point.update(confidence='supported',quality_reason='Weak but stable across scales and both adjacent Y rows')
