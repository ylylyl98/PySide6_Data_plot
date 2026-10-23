"""Fixed-pipeline, per-repeat/per-gate checks for one shared calibrated BN.

All thresholds are operational guardrails in DR/R or eV^-2. They do not
estimate a noise floor, establish physical truth, or permit a noise exception.
"""
from __future__ import annotations

import numpy as np


def products(energy, measurements, background, repeat_ids, settings):
    """Process complete spectra, batching every gate through the real SG path."""
    drr = np.asarray(measurements[repeat_ids], float) / background - 1
    d2, used = production_d2(energy, drr, settings)
    return drr, d2, used


def production_d2(energy, profiles, settings):
    """The same full-axis production derivative for measured and BN-only rows."""
    from core.loader import DataCube
    from core.processing import apply_sg_derivative_energy

    drr = np.asarray(profiles, float)
    shape = drr.shape
    # The loader's production energy cubes are ascending. Normalize a public
    # API caller's descending grid before entering that same regridding path.
    descending = energy[0] > energy[-1]
    axis = np.asarray(energy)[::-1] if descending else np.asarray(energy)
    rows = drr[..., ::-1] if descending else drr
    cube = DataCube(axis, np.arange(int(np.prod(shape[:-1]))),
                    rows.reshape(-1, shape[-1]), 'Repeat / gate', 'BN validation', 'DR/R')
    derivative, used = apply_sg_derivative_energy(
        cube, derivative=2, window_length=settings.derivative_window,
        polyorder=settings.derivative_polyorder)
    d2 = np.asarray(derivative.Z).reshape(shape)
    if descending:
        d2 = d2[..., ::-1]
    if not np.isfinite(drr).all() or not np.isfinite(d2).all():
        raise FloatingPointError('Nonfinite production BN derivative')
    return d2, used


def residual_rms(values, energy, mask):
    """Quadratic-detrended RMS without reducing gates before measurement."""
    x = np.asarray(energy)[mask]
    if len(x) < 4 or np.ptp(x) <= 0:
        raise ValueError('Insufficient joint scoring bandwidth')
    x = 2 * (x - x.min()) / np.ptp(x) - 1
    design = np.column_stack((np.ones(len(x)), x, x*x))
    q, _ = np.linalg.qr(design)
    profiles = np.asarray(values)[..., mask]
    residual = profiles - (profiles @ q) @ q.T
    return np.sqrt(np.mean(residual*residual, axis=-1))


def repeat_scores(values, energy, mask, gate_ids):
    """Legacy-shaped per-repeat summary; gate acceptance uses unreduced scores."""
    return np.median(residual_rms(values[:, gate_ids], energy, mask), axis=1)


def _gate_diagnostics(old, new, energy, mask, repeat_ids, settings, *, d2=False):
    # Curvature itself is the displayed quantity. Removing another polynomial
    # here could conceal a broad false D2 feature, even when the selection
    # objective legitimately focuses on higher-frequency residual texture.
    before = (np.sqrt(np.mean(old[..., mask]**2, axis=-1)) if d2
              else residual_rms(old, energy, mask))
    after = (np.sqrt(np.mean(new[..., mask]**2, axis=-1)) if d2
             else residual_rms(new, energy, mask))
    absolute = (settings.max_absolute_d2_degradation if d2
                else settings.max_absolute_drr_degradation)
    delta = after - before
    relative = delta / np.maximum(before, 1e-12)
    bad = (delta > absolute) & (relative > settings.max_local_degradation_fraction)
    return dict(passed=not bool(np.any(bad)), repeat_indices=list(repeat_ids),
                metric=('Production D2 RMS without detrending, (DR/R)/eV^2' if d2
                        else 'Quadratic-detrended DR/R RMS, dimensionless'),
                original_rms=before.tolist(), candidate_rms=after.tolist(),
                degradation_fraction=relative.tolist(), failing_repeat_gate_indices=[
                    [int(repeat_ids[r]), int(g)] for r, g in np.argwhere(bad)])


def _local_diagnostics(old, new, energy, repeat_ids, settings, *, d2=False):
    """Overlapping fixed windows cover every energy sample and every gate."""
    n = len(energy)
    width = min(n, settings.local_validation_window)
    step = max(1, width // 2)
    starts = sorted(set(list(range(0, n-width+1, step)) + [n-width]))
    failures, failure_count, maximum_fraction = [], 0, 0.
    for start in starts:
        mask = np.zeros(n, bool)
        mask[start:start+width] = True
        item = _gate_diagnostics(old, new, energy, mask, repeat_ids, settings, d2=d2)
        maximum_fraction = max(maximum_fraction, float(np.max(item['degradation_fraction'])))
        for repeat, gate in item['failing_repeat_gate_indices']:
            failure_count += 1
            if len(failures) < 100:
                failures.append(dict(repeat_index=repeat, gate_index=gate,
                                     energy_min=float(np.min(energy[mask])),
                                     energy_max=float(np.max(energy[mask]))))
    return dict(passed=failure_count == 0, window_points=width, step_points=step,
                metric=('Production D2 RMS without detrending, (DR/R)/eV^2' if d2
                        else 'Quadratic-detrended DR/R RMS, dimensionless'),
                window_count=len(starts), checked_gate_count=old.shape[1],
                checked_repeat_indices=list(repeat_ids), checked_energy_points=n,
                maximum_degradation_fraction=maximum_fraction,
                failing_window_gate_count=failure_count, failures=failures)


def fidelity(old_drr, new_drr, old_d2, new_d2, energy, support, repeat_ids, used_window, settings):
    """Check actual protected values, untouched DR/R, and local edge behavior.

    The SG footprint is only used to identify far-away untouched output. It
    never substitutes for the actual full-axis protected derivative check.
    Local windows also inspect the footprint, all tapers, and support edges.
    """
    e = np.asarray(energy)
    protected = np.zeros(len(e), bool)
    for lo, hi in settings.protected_regions:
        protected |= (e >= lo) & (e <= hi)
    outside = np.asarray(support) <= 0
    # Regridding's cubic interpolation has tails. Its guard is checked against
    # actual output below, rather than assumed to produce exact identity.
    radius = (used_window//2 + 8) * float(np.max(abs(np.diff(e))))
    active_e = e[~outside]
    far_outside = outside.copy()
    if len(active_e):
        far_outside &= (e < active_e.min()-radius) | (e > active_e.max()+radius)
    delta_drr = np.abs(new_drr-old_drr)
    delta_d2 = np.abs(new_d2-old_d2)

    def unchanged(mask, delta, tolerance, *, require_evidence=True):
        if not np.any(mask):
            return dict(passed=not require_evidence, checked=False, channel_count=0,
                        max_absolute_change=None, absolute_tolerance=tolerance)
        changes = np.max(delta[..., mask], axis=-1)
        bad = changes > tolerance
        return dict(passed=not bool(np.any(bad)), checked=True, channel_count=int(mask.sum()),
                    max_absolute_change=float(np.max(changes)),
                    per_repeat_gate_max_absolute_change=changes.tolist(),
                    absolute_tolerance=tolerance, failing_repeat_gate_indices=[
                        [int(repeat_ids[r]), int(g)] for r, g in np.argwhere(bad)])

    protection = dict(
        drr=unchanged(protected, delta_drr, settings.protected_drr_absolute_tolerance),
        d2=unchanged(protected, delta_d2, settings.protected_d2_absolute_tolerance),
        outside_support_drr=unchanged(outside, delta_drr, 0.),
        outside_derivative_footprint_d2=unchanged(
            far_outside, delta_d2, settings.protected_d2_absolute_tolerance),
        derivative_footprint_guard_energy=radius,
        local_drr=_local_diagnostics(old_drr, new_drr, e, repeat_ids, settings),
        local_d2=_local_diagnostics(old_d2, new_d2, e, repeat_ids, settings, d2=True))
    protection['passed'] = all(v['passed'] for v in protection.values() if isinstance(v, dict))
    protection['derivative_checked'] = True
    return protection


def joint_checks(old_products, new_products, energy, score_mask, support, repeat_ids, settings):
    old_drr, old_d2, used = old_products
    new_drr, new_d2, new_used = new_products
    if used != new_used:
        raise ValueError('BN derivative settings changed between candidates')
    gates_drr = _gate_diagnostics(old_drr, new_drr, energy, score_mask, repeat_ids, settings)
    gates_d2 = _gate_diagnostics(old_d2, new_d2, energy, score_mask, repeat_ids, settings, d2=True)
    protection = fidelity(old_drr, new_drr, old_d2, new_d2, energy, support,
                          repeat_ids, used, settings)
    reasons = []
    if not gates_drr['passed']:
        reasons.append('drr_gate_degradation')
    if not gates_d2['passed']:
        reasons.append('d2_gate_degradation')
    if not protection['passed']:
        reasons.append('full_axis_fidelity_not_confirmed')
    return dict(passed=not reasons, reasons=reasons, derivative_checked=True,
                all_gate_drr=gates_drr, all_gate_d2=gates_d2, protection=protection)
