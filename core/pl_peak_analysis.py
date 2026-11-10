"""PL seed tracking and bounded multi-peak fits in linear intensity units."""
from copy import deepcopy
import numpy as np

from core.peak_tracking import detect_extrema, track_extrema
from core.pl_fit_quality import fit_pl_spectrum
from core.peak_workspace import _json_clean

PL_FIT_VERSION = 5


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
    manual_ids = {b['id'] for b in dataset.get('branches', []) if b.get('manual')}
    for n, (row, points) in enumerate(sorted(by_row.items())):
        if cancelled and cancelled():
            raise RuntimeError('Peak fitting cancelled.')
        points.sort(key=lambda p: p['energy'])
        fitted, selected, quality = fit_pl_spectrum(x, cube.Z[row, mask], cube.gate[row],
                                                    points, settings['model'], manual_ids | {p['branch_id'] for p in points if p.get('manual')}, cancelled)
        selected_ids = {id(point) for point in selected}
        # Results can be refitted directly. Clear all previous fit outputs so a
        # pruned/failed component never carries stale numeric values.
        fit_fields = ('fit_status', 'fit_center', 'center_error_ev', 'fwhm_mev',
                      'fwhm_error_mev', 'height', 'height_error', 'area', 'area_error',
                      'fit_area_full', 'fit_area_full_error', 'uncertainty_status', 'rms',
                      'fit_quality_status', 'fit_local_residual_sigma', 'fit_height_to_local_noise')
        for point in points:
            for field in fit_fields:
                point.pop(field, None)
        for point in points:
            if id(point) not in selected_ids:
                point.update(fit_status='model_rejected', fit_center=None, fwhm_mev=None,
                             fit_quality_status=quality['status'])
        finite_x = x[np.isfinite(cube.Z[row, mask])]
        for index, (p, component) in enumerate(zip(selected, fitted.components)):
            status = component.status
            local_sigma = None
            local_snr = None
            if fitted.residual is not None and len(fitted.parameters) > 2 + 3 * index:
                radius = max(float(p['width_mev']) / 1000, 4 * float(np.median(np.diff(x))))
                local = fitted.residual[np.abs(finite_x - p['energy']) <= radius]
                if len(local) >= 8:
                    differences = np.diff(local)
                    local_sigma = float(np.median(np.abs(differences - np.median(differences))) / (.67448975 * np.sqrt(2)))
                    if local_sigma > 0 and np.isfinite(local_sigma):
                        local_snr = float(fitted.parameters[2 + 3 * index] / local_sigma)
            # Local differences reduce sensitivity to a slowly varying model
            # residual, but correlated noise can make this estimate optimistic.
            # It is review evidence, never a promotion to a valid physical fit.
            if status == 'peak below residual noise' and local_snr is not None and local_snr >= 5:
                status = 'review: local signal with high global residual'
            if status == 'ok' and p['branch_id'] in quality['local_protected_ids']:
                status = 'review: local-only model evidence'
            if (status == 'ok' and p.get('candidate_origin') in ('curvature', 'shoulder')
                    and abs(component.center_ev - p['energy']) > .5 * float(p['width_mev']) / 1000):
                status = 'review: shoulder center shifted'
            # A highly structured, large residual is a model adequacy warning,
            # separate from optimizer convergence and component bounds.
            if (status == 'ok' and quality['rms_fraction'] is not None
                    and quality['rms_fraction'] > .05 and quality['residual_lag1'] > .5):
                status = 'review: structured residual'
            p.update(fit_status=status, fit_center=component.center_ev,
                     fit_quality_status=quality['status'], fit_local_residual_sigma=local_sigma,
                     fit_height_to_local_noise=local_snr,
                     fwhm_mev=component.fwhm_mev, fwhm_error_mev=component.fwhm_error_mev,
                     height=component.height, area=component.area, rms=fitted.rms,
                     center_error_ev=component.center_error_ev, height_error=component.height_error,
                     fit_area_full=component.fit_area_full, fit_area_full_error=component.fit_area_full_error,
                     area_error=component.area_error, uncertainty_status=component.uncertainty_status)
        if fitted.parameters:
            result['fits'].append(dict(row_index=row, parameters=list(fitted.parameters),
                origin=float(np.mean(x[np.isfinite(cube.Z[row, mask])])),
                x_min=float(x[0]), x_max=float(x[-1]), model=settings['model'],
                branch_ids=[p['branch_id'] for p in selected],
                status=('review fit' if quality['status'] == 'review residual structure'
                        or any(str(p.get('fit_status', '')).startswith('review:') for p in selected)
                        else fitted.status), numerical_status=fitted.status, fit_quality=quality,
                **({'asymmetric_index': fitted.pl_meta['asymmetric_index'],
                    'asymmetric_branch_id': selected[fitted.pl_meta['asymmetric_index']]['branch_id']}
                   if getattr(fitted, 'pl_meta', {}).get('asymmetric_index') is not None else {}),
                n_vary=getattr(fitted, 'pl_meta', {}).get('n_vary', len(fitted.parameters)),
                fixed_center_branch_ids=[selected[i]['branch_id'] for i in getattr(fitted, 'pl_meta', {}).get('fixed_centers', [])],
                covariance=fitted.covariance.tolist() if fitted.covariance is not None else None,
                spectrum_area_roi=fitted.spectrum_area_roi,
                spectrum_area_roi_domain=fitted.spectrum_area_roi_domain,
                spectrum_area_roi_method=fitted.spectrum_area_roi_method))
        if progress:
            progress(int((n+1)*100/len(by_row)))
    if cancelled and cancelled():
        raise RuntimeError('Peak fitting cancelled.')
    actual_detection={k:result['settings'][k] for k in ('detection_method','smoothing_window','candidate_floor') if k in result['settings']}
    result['settings'] = {**deepcopy(settings),**actual_detection}
    result['pl_fit_version'] = PL_FIT_VERSION
    result['method'] = result['method']+' + bounded multi-peak '+settings['model']+' fit'
    return _json_clean(result)
