"""PL seed tracking and bounded multi-peak fits in linear intensity units."""
from copy import deepcopy
import numpy as np

from core.peak_tracking import detect_extrema, track_extrema
from core.power_multi_peaks import MultiPeakSettings, PeakSeed, fit_multi_spectrum
from core.peak_workspace import _json_clean


def detect(dataset, row, settings):
    return detect_extrema(dataset, row, settings, polarity='peaks')


def track(dataset, branches, settings, cancelled=None, progress=None):
    return track_extrema(dataset, branches, settings, cancelled, progress)


def fit(dataset, result, settings, cancelled=None, progress=None):
    if dataset['kind'] != 'PL':
        raise ValueError('Gaussian/Lorentzian fitting is available for PL intensity.')
    result = deepcopy(result); cube = dataset['cube']
    mask = (cube.energy >= settings['x_min']) & (cube.energy <= settings['x_max'])
    x = cube.energy[mask]
    by_row = {}
    for point in result['points']:
        if point['status'] == 'accepted':
            by_row.setdefault(point['row_index'], []).append(point)
    if not by_row:
        raise ValueError('Track selected branches before fitting.')
    result['fits'] = []
    for n, (row, points) in enumerate(sorted(by_row.items())):
        if cancelled and cancelled():
            raise RuntimeError('Peak fitting cancelled.')
        points.sort(key=lambda p: p['energy'])
        seeds = tuple(PeakSeed(p['energy'], max(p['width_mev']/1000, .0001)) for p in points)
        fitted = fit_multi_spectrum(x, cube.Z[row, mask], cube.gate[row], MultiPeakSettings(seeds, settings['model']))
        for p, component in zip(points, fitted.components):
            p.update(fit_status=component.status, fit_center=component.center_ev,
                     fwhm_mev=component.fwhm_mev, fwhm_error_mev=component.fwhm_error_mev,
                     height=component.height, area=component.area, rms=fitted.rms)
        if fitted.parameters:
            result['fits'].append(dict(row_index=row, parameters=list(fitted.parameters),
                origin=float(np.mean(x[np.isfinite(cube.Z[row, mask])])),
                x_min=float(x[0]), x_max=float(x[-1]), model=settings['model'],
                branch_ids=[p['branch_id'] for p in points], status=fitted.status))
        if progress:
            progress(int((n+1)*100/len(by_row)))
    if cancelled and cancelled():
        raise RuntimeError('Peak fitting cancelled.')
    actual_detection={k:result['settings'][k] for k in ('detection_method','smoothing_window','candidate_floor') if k in result['settings']}
    result['settings'] = {**deepcopy(settings),**actual_detection}
    result['method'] = result['method']+' + bounded multi-peak '+settings['model']+' fit'
    return _json_clean(result)
