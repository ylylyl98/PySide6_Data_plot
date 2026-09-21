"""Windowed max-minus-min amplitude, without peak detection."""
import numpy as np


def smooth_energy_cube(cube,window_mev,polyorder=2,*,unit='meV'):
    """SG on full finite segments, regridded in energy; never bridge gaps.

    With meV, convert the requested span to an odd sample count. With points,
    use the supplied odd count. Cap at segment length; unresolved segments are NaN.
    """
    from dataclasses import replace
    from scipy.signal import savgol_filter
    if not np.isfinite(window_mev) or window_mev<=0 or polyorder<1:raise ValueError('Invalid SG settings.')
    if unit not in ('points','meV'):raise ValueError('Unknown SG window unit.')
    if unit=='points' and (int(window_mev)!=window_mev or int(window_mev)%2==0 or window_mev<=polyorder):
        raise ValueError('SG points must be an odd integer greater than the polynomial order.')
    order=np.argsort(cube.energy);x=cube.energy[order]
    if len(x)<3 or not np.isfinite(x).all() or np.any(np.diff(x)<=0):raise ValueError('SG requires distinct finite Energy samples.')
    output=np.full_like(cube.Z,np.nan,dtype=float);windows=[];spans=[]
    for row_index,row in enumerate(cube.Z[:,order]):
        edges=np.diff(np.r_[False,np.isfinite(row),False].astype(int))
        for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            n=end-start
            if n<polyorder+2:continue
            grid=np.linspace(x[start],x[end-1],n);step=grid[1]-grid[0]
            window=int(window_mev) if unit=='points' else max(polyorder+2,2*int(round(window_mev/1000/step/2))+1)
            if window%2==0:window+=1
            window=min(window,n if n%2 else n-1)
            if window<=polyorder:continue
            values=np.interp(grid,x[start:end],row[start:end])
            filtered=savgol_filter(values,window,polyorder,mode='interp')
            output[row_index,order[start:end]]=np.interp(x[start:end],grid,filtered)
            windows.append(window);spans.append((window-1)*step*1000)
    return replace(cube,Z=output),dict(method='SG',unit=unit,window_points=int(window_mev) if unit=='points' else None,window_mev=float(window_mev) if unit=='meV' else None,polyorder=int(polyorder),
        window_points_min=min(windows,default=0),window_points_max=max(windows,default=0),
        actual_span_mev_min=min(spans,default=0),actual_span_mev_max=max(spans,default=0))


def range_amplitude(cube, x_min, x_max, y_min, y_max):
    """Columns: Y, p2p, min, max, energy_at_min, energy_at_max, finite_count.

    Matches Megasweep's finite-window max-minus-min for valid spectra. Rows
    with fewer than two finite samples are missing, rather than false zeros.
    """
    if not np.isfinite([x_min,x_max,y_min,y_max]).all() or x_min>=x_max or y_min>y_max:
        raise ValueError('Invalid Energy or Y range.')
    ix=(cube.energy>=x_min)&(cube.energy<=x_max)
    iy=(cube.gate>=y_min)&(cube.gate<=y_max)
    if not ix.any() or not iy.any():raise ValueError('No samples in the selected range.')
    x=cube.energy[ix];z=cube.Z[np.ix_(iy,ix)];finite=np.isfinite(z)
    count=finite.sum(axis=1);valid=count>=2
    low=np.argmin(np.where(finite,z,np.inf),axis=1)
    high=np.argmax(np.where(finite,z,-np.inf),axis=1)
    rows=np.arange(len(z));minimum=z[rows,low];maximum=z[rows,high]
    output=np.column_stack((cube.gate[iy],maximum-minimum,minimum,maximum,x[low],x[high],count))
    output[~valid,1:6]=np.nan
    return output
