"""Worker-side group checks and explicit method comparison, without UI state."""
from dataclasses import asdict, replace
import numpy as np
from core.drr_fitted_background import load_intensity, settings_for_source, source_signature
from core.drr_background_cache import fit_cached
from core.drr_background_quality import suggest_regions
from core.drr_sources import resolve_source_path
from core.load_cancellation import check_load_cancelled
from core.drr_background_memo import ReviewMemo, review_key

VALIDATION_VERSION = 2
_validation_memo = ReviewMemo()


def blocked_validation(cube, reference):
    """Hold out small contiguous anchor bands on at most nine representative gates."""
    from core.drr_background_cache import CACHE_VERSION
    key = review_key(dict(version=VALIDATION_VERSION, fit_version=CACHE_VERSION, settings=asdict(reference.settings)),
                     cube.energy, cube.gate, cube.Z, reference.fit_mask, reference.anchor_masks())
    return _validation_memo.get_or_compute(key, lambda: _blocked_validation(cube, reference))


def _blocked_validation(cube, reference):
    from core.drr_fitted_background import estimate_background
    indices = np.unique(np.linspace(0, len(cube.gate)-1, min(9, len(cube.gate))).astype(int))
    sample = replace(cube, gate=cube.gate[indices], Z=cube.Z[indices])
    mask = reference.fit_mask
    starts = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)) == -1)
    blocks = sorted(zip(starts, ends), key=lambda pair: pair[1]-pair[0], reverse=True)[:3]
    errors = []
    for start, end in blocks:
        check_load_cancelled()
        if end-start < 16:
            continue
        center = (start+end)//2
        half = min((end-start)//6, max(2, int(round(.004/np.median(np.diff(cube.energy))))))
        lo, hi = center-half, center+half
        settings = replace(reference.settings, automatic_protection=False, protected_regions=reference.settings.protected_regions+
                           ((float(cube.energy[lo]), float(cube.energy[hi])),))
        anchors = None
        if reference.gate_fit_mask is not None:
            anchors = reference.anchor_masks()[indices].copy()
            anchors[:, lo:hi+1] = False
            settings = replace(reference.settings, shared_texture=False)
        try:
            # These temporary fits supply prediction errors only. Final model
            # fits still screen every gate; repeated peak screening here adds
            # no information to the held-out score.
            _, fitted = estimate_background(sample, settings, assess_quality=False, anchor_masks=anchors)
            error = sample.Z[:, lo:hi+1]/fitted.background()[:, lo:hi+1]-1
            errors.append(error[reference.anchor_masks()[indices, lo:hi+1]])
        except ValueError:
            continue  # Insufficient support is reported as unavailable, never a good score.
    errors = [error for error in errors if error.size]
    return dict(rms_percent=float(100*np.sqrt(np.mean(np.concatenate(errors)**2))) if errors else None,
                blocks=len(errors), gates=len(indices))


def check_group(folder, files, axis, recipe, generation, *, progress, log):
    results = []
    for index, source in enumerate(files):
        check_load_cancelled()
        try:
            path = resolve_source_path(folder, source)
            signature = source_signature(path)
            raw = load_intensity(folder, source, y_axis=axis)
            regions = recipe.get('protected_regions_by_file', {}).get(source, recipe.get('protected_regions', ()))
            reference = None
            if regions or recipe.get('automatic_protection', False):
                _, reference = fit_cached(raw, settings_for_source(recipe, source), source=source,
                    signature=signature, source_path=path, y_axis=axis)
                report = reference.diagnostics
            else:
                report = {**suggest_regions(raw), 'status': 'Needs regions',
                          'warnings': ['Review suggested regions or draw protection, then fit.']}
            if source_signature(path) != signature:
                raise ValueError('Source changed during check; retry after acquisition finishes.')
            results.append(dict(source=source, report=report, reference=reference))
        except (OSError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
            results.append(dict(source=source, report=dict(status='Error', warnings=[str(exc)]), reference=None))
        progress.emit(int(100*(index+1)/len(files)))
    return dict(kind='checks', generation=generation, results=results)


def compare_methods(folder, source, axis, settings, generation, window, polyorder, *, progress, log):
    from core.processing import apply_sg_derivative_energy
    path = resolve_source_path(folder, source)
    signature = source_signature(path)
    raw = load_intensity(folder, source, y_axis=axis)
    cubes, references = [], []
    alternate = replace(settings, method='polynomial' if settings.method == 'spline' else 'spline')
    for chosen in (settings, alternate):
        check_load_cancelled()
        cube, reference = fit_cached(raw, chosen, source=source, signature=signature, source_path=path, y_axis=axis)
        cubes.append(cube)
        references.append(reference)
    derivatives = [apply_sg_derivative_energy(cube, derivative=2, window_length=window, polyorder=polyorder)
                   for cube in cubes]
    primary, other = references
    order = np.argsort(raw.energy)
    raw = replace(raw, energy=raw.energy[order], Z=raw.Z[:, order])
    validation = [blocked_validation(raw, ref) for ref in references]
    percent = 100*(other.background()/primary.background()-1)
    protected = ~primary.fit_mask
    difference = derivatives[1][0].Z-derivatives[0][0].Z
    rms = float(np.sqrt(np.mean(derivatives[0][0].Z**2)))
    if source_signature(path) != signature:
        raise ValueError('Source changed during comparison; retry after acquisition finishes.')
    return dict(kind='compare', generation=generation,
        raw=raw, reference=primary, alternate=other, alternate_background=other.background(),
        derivatives=[item[0] for item in derivatives],
        metrics=dict(background_max_difference_percent=float(np.max(abs(percent[:, protected]))) if protected.any() else None,
                     derivative_relative_rms_percent=float(100*np.sqrt(np.mean(difference**2))/rms) if rms > 1e-12 else None,
                     sg_window=derivatives[0][1], sg_polyorder=polyorder, validation=validation))
