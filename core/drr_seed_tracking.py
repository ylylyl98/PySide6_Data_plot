"""Conservative single-branch tracking from an explicit seed, with gaps."""
from dataclasses import asdict
import numpy as np
from scipy.signal import find_peaks


def _candidates(x, row, s):
    sign=1 if s.polarity == 'peaks' else -1
    finite=np.isfinite(row)
    if finite.sum()<5: return []
    dif=np.diff(row); dif=dif[np.isfinite(dif)]
    if not len(dif): return []
    noise=float(np.median(abs(dif-np.median(dif)))/(.67448975*np.sqrt(2)))
    if s.source == 'second':
        # SG introduces correlated noise: adjacent differences alone underestimate
        # its amplitude. A robust row-scale floor intentionally rejects crowded
        # or weak curvature features rather than calling them reliable peaks.
        values=row[finite]
        noise=max(noise,float(np.median(abs(values-np.median(values)))/.67448975))
    threshold=max(s.prominence*float(np.ptp(row[finite])),s.noise_sigma*noise,np.finfo(float).eps*max(1,float(np.max(abs(row[finite]))))*100)
    edges=np.diff(np.r_[False,finite,False].astype(int)); points=[]
    for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        inds,props=find_peaks(sign*row[start:end],prominence=threshold,width=(None,None))
        for k,i in enumerate(inds+start):
            left=np.interp(props['left_ips'][k]+start,np.arange(len(x)),x)
            right=np.interp(props['right_ips'][k]+start,np.arange(len(x)),x)
            width=float(right-left)
            if width*1000<s.min_width_mev: continue
            # Compare shapes in an energy-centered window; normalize offset and scale.
            offsets=np.linspace(-max(width, .001),max(width,.001),25)
            if x[i]+offsets[0]<x[start] or x[i]+offsets[-1]>x[end-1]:continue
            shape=np.interp(x[i]+offsets,x[start:end],sign*row[start:end]);shape-=shape.mean()
            norm=np.linalg.norm(shape)
            if norm==0:continue
            points.append(dict(energy=float(x[i]),amplitude=float(row[i]),prominence=float(props['prominences'][k]),
                polarity='peak' if sign==1 else 'dip',width_mev=width*1000,noise_sigma_estimate=noise,
                _shape=shape/norm))
    return points


def track_seed(cube,s,ix,iy,check,progress=None):
    from core.drr_peak_analysis import compute_second_derivative_row
    x=np.asarray(cube.energy)[ix]; ys=np.asarray(cube.gate); cache={}
    ordered=sorted(map(int,iy),key=lambda i:ys[i]); seed_pos=min(range(len(ordered)),key=lambda k:abs(ys[ordered[k]]-s.seed_y))
    def candidates(pos):
        check();i=ordered[pos]
        if i not in cache:
            row=cube.Z[i] if s.source=='raw' else compute_second_derivative_row(cube,i,s)[0]
            cache[i]=_candidates(x,np.asarray(row)[ix],s)
            if progress is not None:progress(min(99,int(100*len(cache)/len(ordered))))
        return cache[i]
    near=sorted([p for p in candidates(seed_pos) if abs(p['energy']-s.seed_energy)<=s.max_shift_mev/1000],key=lambda p:abs(p['energy']-s.seed_energy))
    if not near:raise ValueError('No sufficiently wide, noise-qualified extremum near the seed. Choose a clearer seed or inspect the source/polarity.')
    if len(near)>1 and abs(near[1]['energy']-s.seed_energy)-abs(near[0]['energy']-s.seed_energy)<.0003:
        raise ValueError('Ambiguous seed: select one branch more precisely.')
    seed=near[0]; picked={seed_pos:seed}; reasons={};segment_ids={seed_pos:1}; next_segment=2
    for direction in (-1,1):
        prev=seed;gap=0;segment=1
        for pos in range(seed_pos+direction,len(ordered) if direction==1 else -1,direction):
            options=[]
            for p in candidates(pos):
                shift=abs(p['energy']-prev['energy'])/(s.max_shift_mev/1000)
                ratio=p['width_mev']/prev['width_mev'];strength=p['prominence']/prev['prominence']
                corr=float(np.dot(p['_shape'],prev['_shape']))
                if shift<=1 and .4<=ratio<=2.5 and .2<=strength<=5 and corr>=.65:
                    options.append((shift+(.5*(1-corr)),p))
            options.sort(key=lambda v:v[0])
            ambiguous=len(options)>1 and options[1][0]-options[0][0]<.25
            if not options or ambiguous:
                reasons[ordered[pos]]='ambiguous branch' if ambiguous else 'no qualified nearby extremum'
                gap+=1
                if gap>s.max_gap:break
                continue
            if gap:
                segment=next_segment;next_segment+=1
            prev=options[0][1];picked[pos]=prev;segment_ids[pos]=segment;gap=0
    points=[]
    for pos,p in sorted(picked.items(),key=lambda pair:ordered[pair[0]]):
        i=ordered[pos]
        points.append({k:v for k,v in p.items() if not k.startswith('_')}|dict(row_index=i,y=float(ys[i]),status='accepted',track_id=segment_ids[pos]))
    check()
    if progress is not None:progress(100)
    return dict(schema_version=1,settings=asdict(s),y_label=str(cube.gate_label),y_values=[float(ys[i]) for i in iy],
        products={s.source:dict(points=points,processing='Seeded local-extremum tracking; width, noise, shape and displacement gates; no interpolation across missing rows',
            seed_row=ordered[seed_pos],unmatched_rows=[dict(row_index=int(i),reason=reasons.get(i,'outside connected seed branch')) for i in iy if i not in {p['row_index'] for p in points}])})
