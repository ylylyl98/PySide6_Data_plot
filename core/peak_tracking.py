"""Shared seed identity and conservative multi-branch extrema tracking."""
from copy import deepcopy
from dataclasses import replace
from uuid import uuid4

import numpy as np

from core.drr_peak_analysis import PeakAnalysisSettings, analyze_drr_peaks
from core.peak_candidates import detect_heatmap, filter_heatmap


def detector_settings(settings, **overrides):
    values = {k: v for k, v in settings.items() if k in PeakAnalysisSettings.__dataclass_fields__}
    return PeakAnalysisSettings(**{**values, 'source': 'raw', **overrides})


def detect_extrema(dataset, row, settings, *, polarity):
    cube = dataset['cube']; row = int(row)
    if not 0 <= row < len(cube.gate):
        raise ValueError('Select a measured spectrum first.')
    one = replace(cube, gate=cube.gate[row:row+1], Z=cube.Z[row:row+1])
    s = detector_settings(settings, mode='all', polarity=polarity,
                          y_min=float(cube.gate[row]), y_max=float(cube.gate[row]))
    points = analyze_drr_peaks(one, s)['products']['raw']['points']
    previous = dataset.get('branches', []); used = set(); out = []
    positions = {p['branch_id']:p['energy'] for p in (dataset.get('result') or {}).get('points',[])
                 if p['row_index']==row and p['status']=='accepted'}
    def distance(branch,energy):
        return abs(positions.get(branch['id'],branch['seed_energy'])-energy)
    next_color = max((b['color'] for b in previous), default=-1) + 1
    for p in points:
        if p['width_mev'] < settings['min_width_mev']:
            continue
        matches = [b for b in previous if b['id'] not in used and b['polarity'] == p['polarity']
                   and distance(b,p['energy']) < settings['max_shift_mev']/1000]
        b = deepcopy(min(matches, key=lambda b: distance(b,p['energy']))) if matches else None
        if b is None:
            b = dict(id=uuid4().hex, name=f"{'P' if p['polarity']=='peak' else 'D'}{next_color+1}",
                     color=next_color, enabled=True, polarity=p['polarity'])
            next_color += 1
        used.add(b['id']); b.update(seed_energy=p['energy'], seed_y=float(cube.gate[row]),
                                   seed_row=row, width_mev=p['width_mev'])
        out.append(b)
    return out


def track_extrema(dataset, branches, settings, cancelled=None, progress=None):
    selected = deepcopy([b for b in branches if b.get('enabled', True)])
    if not selected:
        raise ValueError('Find or add a peak, then select at least one branch.')
    points, notices = [], []
    for index, branch in enumerate(selected):
        if cancelled and cancelled():
            raise RuntimeError('Peak analysis cancelled.')
        s = detector_settings(settings, mode='seed', seed_energy=branch['seed_energy'],
                              seed_y=branch['seed_y'], polarity=branch['polarity']+'s',min_width_mev=max(.001,settings['min_width_mev']))
        try:
            result = analyze_drr_peaks(dataset['cube'], s, cancelled=cancelled,
                progress=(lambda n, i=index: progress(int((i+n/100)*100/len(selected)))) if progress else None)
        except ValueError as exc:
            notices.append(f"{branch['name']}: {exc}")
            continue
        for p in result['products']['raw']['points']:
            points.append({**p, 'branch_id': branch['id'], 'segment': p['track_id']})
    # If independent seeds converge on the same sample, do not assign two identities.
    occupied = {}
    dx = float(np.median(np.diff(dataset['cube'].energy)))
    for p in points:
        key = (p['row_index'], p['polarity'])
        for other in occupied.get(key, []):
            if abs(other['energy']-p['energy']) < dx*.5:
                other['status'] = p['status'] = 'uncertain'
        occupied.setdefault(key, []).append(p)
    if progress:
        progress(100)
    return dict(dataset_key=dataset['key'], branches=selected, points=points, fits=[],
                settings=deepcopy(settings), notices=notices, method='Seeded extrema tracking')
