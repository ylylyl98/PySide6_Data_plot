"""Fast, conservative review cues; these are not confidence intervals or peak assignments."""
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from core.load_cancellation import check_load_cancelled
from core.drr_background_memo import ReviewMemo, review_key

SCREEN_VERSION = 1
_screen_memo = ReviewMemo()


def merge_regions(regions):
    merged = []
    for lo, hi in sorted((float(a), float(b)) for a, b in regions):
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError('Protected intervals must be finite and increasing.')
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def suggest_regions(cube, existing=(), *, feature_width=.035):
    """Screen every gate for positive AND negative narrow spectral structure.

    Work on a uniform energy grid. A 35 meV SG local quadratic removes smooth
    structure; a noise-scaled threshold (at least 0.75% log intensity) flags
    departures spanning >= 3 samples. Add 5 meV margins, never shrink manual
    regions. Broad resonances can still be missed; suggestions require review.
    """
    energy = np.asarray(cube.energy, float)
    raw = np.asarray(cube.Z, float)
    order = np.argsort(energy)
    energy, raw = energy[order], raw[:, order]
    if len(energy) < 24 or not np.isfinite(raw).all() or np.any(raw <= 0) or np.any(np.diff(energy) <= 0):
        raise ValueError('Region suggestions require finite positive spectra on distinct energy coordinates.')
    manual = merge_regions(existing)
    key = review_key(dict(version=SCREEN_VERSION, feature_width=feature_width), energy, raw)
    report = _screen_memo.get_or_compute(key, lambda: _screen_features(energy, raw, feature_width))
    candidates = report.pop('candidates')
    additions = [region for region in candidates
                 if not any(lo <= region[0] and hi >= region[1] for lo, hi in manual)]
    return dict(suggested_regions=[list(pair) for pair in additions],
                merged_regions=[list(pair) for pair in merge_regions((*manual, *additions))], **report)


def automatic_regions(cube):
    """Combine narrow and wider feature screens before automatic application."""
    return merge_regions(region for width in (.035, .070)
                         for region in suggest_regions(cube, feature_width=width)['suggested_regions'])


def _screen_threshold(values, window):
    differences = abs(np.diff(values, n=2, axis=1))
    noise = 1.4826*np.median(differences, axis=1)/np.sqrt(6)
    local_noise = 1.4826*np.median(differences, axis=0)/np.sqrt(6)
    local_noise = np.pad(median_filter(local_noise, size=max(7, window//2 | 1), mode='nearest'), 1, mode='edge')
    return np.maximum(.0075, 6*np.maximum(noise[:, None], local_noise[None, :])), noise


def _screen_features(energy, raw, feature_width=.035):
    """Protection-independent screening, shared by previews and model changes."""
    grid = np.linspace(energy[0], energy[-1], len(energy))
    step = grid[1] - grid[0]
    log_rows = []
    for row in raw:
        check_load_cancelled()
        log_rows.append(np.interp(grid, energy, np.log(row)))
    values = np.asarray(log_rows)
    window = min(len(grid) if len(grid) % 2 else len(grid)-1, max(7, int(round(feature_width/step)) | 1))
    smooth = savgol_filter(values, window, 2, axis=1)
    high = values - smooth
    # Photon noise grows in relative size toward the dim spectral wings. A
    # single row-wide threshold otherwise labels runs of wing noise as peaks.
    threshold, noise = _screen_threshold(values, window)
    significant = abs(high) > threshold
    edge = window//2
    significant[:, :edge] = significant[:, -edge:] = False
    # Keep runs within a single gate before taking the union. Scattered noise
    # from unrelated frames must not form an apparent continuous resonance.
    counts = np.zeros(len(grid), int)
    strong = np.zeros(len(grid), bool)
    affected = []
    for index, row in enumerate(significant):
        check_load_cancelled()
        starts = np.flatnonzero(np.diff(np.r_[False, row, False].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, row, False].astype(int)) == -1)
        found = False
        for start, end in zip(starts, ends):
            if end-start >= 3:
                counts[start:end] += 1
                # Retain strong one-gate features, while rejecting isolated
                # threshold crossings accumulated over a large scan.
                if np.max(abs(high[index, start:end])/threshold[index, start:end]) >= 2.5:
                    strong[start:end] = True
                found = True
        if found:
            affected.append(index)
    hits = (counts >= min(3, len(raw))) | strong
    starts = np.flatnonzero(np.diff(np.r_[False, hits, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, hits, False].astype(int)) == -1)
    candidates = merge_regions((max(grid[1], grid[a]-.005), min(grid[-2], grid[b-1]+.005))
                               for a, b in zip(starts, ends))
    return dict(candidates=[list(pair) for pair in candidates],
                screened_gates=len(raw), feature_gate_count=len(affected),
                noise_percent=float(100*np.median(noise)),
                suggestion_note='Heuristic narrow-feature screening of all gates; broad or weak peaks may be missed.')


def inspect_background(cube, reference):
    report = suggest_regions(cube, reference.settings.protected_regions)
    if reference.gate_fit_mask is not None:
        protected = ~reference.gate_fit_mask.all(axis=0)
        report['suggested_regions'] = [pair for pair in report['suggested_regions']
            if np.any((cube.energy >= pair[0]) & (cube.energy <= pair[1]) & ~protected)]
        report['merged_regions'] = [list(pair) for pair in reference.settings.protected_regions]
    contrast = cube.Z/reference.background()-1
    mask = reference.anchor_masks()
    rms = 100*np.sqrt(np.sum(np.where(mask, contrast**2, 0), axis=1)/mask.sum(axis=1))
    worst = int(np.argmax(rms))
    warnings = []
    if reference.settings.automatic_protection and not reference.settings.protected_regions:
        warnings.append('No narrow peaks detected automatically; inspect broad or weak features before interpreting the result.')
    if report['suggested_regions']:
        warnings.append('Spectral features extend outside protection; inspect suggested regions.')
    if mask.mean() < .35:
        warnings.append('Less than 35% of the spectrum supports the background.')
    if reference.diagnostics.get('shared_texture', {}).get('status') == 'Applied':
        warnings.append('Shared fine texture corrected outside protected peaks; inspect stationary weak features with correction off.')
    if float(np.max(rms)) > max(3., 6*report['noise_percent']):
        warnings.append('Large residual outside protection; inspect missing peaks or background shape.')
    if float(np.max(rms)) > max(1., 3*float(np.median(rms))):
        warnings.append('Some gates fit much worse than the median; shared shape may be insufficient.')
    report.update(status='Review' if warnings else 'Checked', warnings=warnings,
                  automatic_protection=reference.settings.automatic_protection,
                  effective_protected_regions=[list(pair) for pair in reference.settings.protected_regions],
                  anchor_fraction=float(mask.mean()), minimum_gate_anchor_fraction=float(mask.mean(axis=1).min()),
                  pooled_anchor_fraction=float(reference.fit_mask.mean()), max_gate_rms_percent=float(rms[worst]),
                  median_gate_rms_percent=float(np.median(rms)), worst_gate=float(cube.gate[worst]),
                  worst_gate_index=worst,
                  caveat='Checked means no heuristic alert, not a validated background under protected peaks.')
    return report
