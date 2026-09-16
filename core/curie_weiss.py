"""Exploratory Curie–Weiss fits to an optical susceptibility proxy.

Fit slope space (equal weights), not its noisy reciprocal. Amplitude may be
negative because the optical sign is arbitrary. The pole must stay below the
lowest fitted temperature: the model describes the paramagnetic regime.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t


def fit_inverse_curie_weiss(temperature, slope, *, slope_se=None,
                            background_mode='zero', background=0.):
    """OLS of 1/(s-s0) = m*T+b; theta=-b/m, with propagated display errors.

    Follows Tang et al., doi:10.1038/s41565-022-01309-8, inverse-response
    presentation. The caption does not specify regression weights; equal
    weights are an explicit implementation choice, not a claimed replication.
    """
    t, s = np.asarray(temperature, float), np.asarray(slope, float)
    if background_mode not in ('zero', 'fixed'):
        raise ValueError('Inverse linear fit requires zero or fixed background.')
    if t.ndim != 1 or s.shape != t.shape or len(t) < 3:
        raise ValueError('Need at least 3 temperature points.')
    if not np.all(np.isfinite(t)) or np.any(t <= 0) or not np.all(np.isfinite(s)):
        raise ValueError('Temperatures must be positive kelvin; slopes must be finite.')
    if len(np.unique(np.round(t, 8))) != len(t):
        raise ValueError('Select one point per distinct temperature.')
    base = 0. if background_mode == 'zero' else float(background)
    y = s-base
    if not np.isfinite(base) or not (np.all(y > 0) or np.all(y < 0)):
        raise ValueError('Background-subtracted slopes must have one nonzero sign.')
    inverse = 1/y
    if not np.all(np.isfinite(inverse)) or np.ptp(inverse) <= max(np.max(np.abs(inverse))*1e-10, 1e-15):
        raise ValueError('Inverse slopes are constant or nonfinite; theta is not identifiable.')
    centered = t-t.mean()
    m = float(centered @ (inverse-inverse.mean()) / (centered @ centered))
    b = float(inverse.mean()-m*t.mean())
    if abs(m)*max(float(np.ptp(t)), 1.) <= np.max(np.abs(inverse))*1e-12:
        raise ValueError('Inverse slope is nearly flat; theta is not identifiable.')
    theta, amplitude = -b/m, 1/m
    if theta >= float(t.min()):
        raise ValueError('Fitted pole is not below the selected temperatures; choose a paramagnetic range.')
    predicted_inverse = m*t+b
    residual_inverse = inverse-predicted_inverse
    rss = float(residual_inverse @ residual_inverse)
    dof = len(t)-2
    # Centered regression avoids cancellation in the intercept covariance.
    cov = np.diag([rss/dof/float(centered @ centered), rss/dof/len(t)])
    gradient = np.array([float(inverse.mean())/m**2, -1/m])
    se = float(np.sqrt(max(0., gradient @ cov @ gradient)))
    radius = float(student_t.ppf(.975, dof)*se)
    ci = [theta-radius, theta+radius]
    warnings = []
    if len(t) < 6:
        warnings.append('Few temperatures: exploratory fit; add higher-temperature data.')
    if abs(m) <= student_t.ppf(.975, dof)*np.sqrt(cov[0, 0]):
        warnings.append('Linear coefficient is consistent with zero; theta ratio is poorly constrained.')
        ci = None
    errors = np.full(len(t), np.nan) if slope_se is None else np.asarray(slope_se, float)
    if errors.shape != s.shape:
        raise ValueError('Slope uncertainties must match the temperature points.')
    valid = np.isfinite(errors) & (errors >= 0)
    inverse_errors = [float(e/v**2) if ok else None for e, v, ok in zip(errors, y, valid)]
    if not np.all(valid):
        warnings.append('Some slope errors unavailable; missing error bars are omitted. Regression uses equal weights.')
    if np.any(valid & (errors >= np.abs(y))):
        warnings.append('Slope uncertainty reaches zero; reciprocal error propagation is unreliable.')
        ci = None
    r2 = 1-rss/float(np.sum((inverse-inverse.mean())**2))
    if r2 < .9:
        warnings.append('Weak Curie–Weiss agreement; inspect temperature range.')
    predicted = base+amplitude/(t-theta)
    residual = s-predicted
    interpretation = ('Unresolved interaction sign' if ci is None or ci[0] <= 0 <= ci[1]
                      else 'FM interaction tendency' if theta > 0 else 'AFM interaction tendency')
    return dict(theta_k=float(theta), theta_se_k=se, theta_ci95_k=ci,
                amplitude=float(amplitude), background=base, background_mode=background_mode,
                n=len(t), dof=dof, r_squared=r2, fit_space='inverse_slope',
                linear_slope=m, linear_intercept=b, inverse_slope=inverse.tolist(),
                inverse_slope_se=inverse_errors, predicted_inverse=predicted_inverse.tolist(),
                inverse_residuals=residual_inverse.tolist(), inverse_residual_rms=float(np.sqrt(rss/len(t))),
                residual_rms=float(np.sqrt(np.mean(residual**2))),
                temperature_min_k=float(t.min()), temperature_max_k=float(t.max()),
                predicted=predicted.tolist(), residuals=residual.tolist(),
                interpretation=interpretation, warnings=warnings,
                method='Unweighted linear least squares in inverse-slope space',
                uncertainty='Residual-scaled linear covariance; delta-method Student-t 95% theta interval; excludes systematic errors')


def fit_curie_weiss(temperature, slope, *, background_mode='zero', background=0.):
    t = np.asarray(temperature, dtype=float)
    y = np.asarray(slope, dtype=float)
    if background_mode not in ('zero', 'fixed', 'fit'):
        raise ValueError('Unknown background mode.')
    p = 3 if background_mode == 'fit' else 2
    if t.ndim != 1 or y.shape != t.shape or len(t) <= p:
        raise ValueError(f'Need at least {p + 1} temperature points.')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or np.any(t <= 0):
        raise ValueError('Temperatures must be positive kelvin; slopes must be finite.')
    if len(np.unique(np.round(t, 8))) != len(t):
        raise ValueError('Multiple windows/repeats at one temperature: select one per temperature.')
    if not np.isfinite(background):
        raise ValueError('Background must be finite.')
    base = 0. if background_mode == 'zero' else float(background)
    adjusted = y - base if background_mode != 'fit' else y
    if np.ptp(y) <= max(np.max(np.abs(y)) * 1e-10, 1e-15):
        raise ValueError('Slopes are constant; Weiss temperature is not identifiable.')
    if background_mode != 'fit' and not (np.all(adjusted > 0) or np.all(adjusted < 0)):
        raise ValueError('Background-subtracted slopes must have one nonzero sign.')
    scale = float(np.max(np.abs(y)))
    span = float(np.ptp(t))
    tscale = max(float(np.max(t)), span, 1.)
    pole_max = float(np.min(t)) - max(1e-8, span * 1e-8)
    pole_min = -tscale * 1e5

    # For each theta, amplitude and background are linear parameters. Solve
    # them analytically and optimize only log(Tmin-theta). This bounded search
    # avoids multi-second unconstrained iterations for almost linear data.
    yn = y / scale
    tmin = float(np.min(t))
    lower, upper = np.log(tmin - pole_max), np.log(tmin - pole_min)

    def solve(log_gap):
        theta = tmin - np.exp(log_gap)
        z = 1 / (t - theta)
        if p == 3:
            centered = z - np.mean(z)
            a = float(centered @ (yn - np.mean(yn)) / (centered @ centered))
            offset = float(np.mean(yn) - a * np.mean(z))
        else:
            offset = base / scale
            a = float(z @ (yn - offset) / (z @ z))
        residual = offset + a * z - yn
        return float(residual @ residual), a, float(theta), offset

    grid = np.linspace(lower, upper, 160)
    costs = np.array([solve(value)[0] for value in grid])
    candidates = [float(grid[0]), float(grid[-1])]
    minima = [i for i in range(1, len(grid)-1) if costs[i] <= costs[i-1] and costs[i] <= costs[i+1]]
    for index in sorted(minima, key=lambda i: costs[i])[:4]:
        fitted = minimize_scalar(lambda value: solve(value)[0],
                                 bounds=(grid[index-1], grid[index+1]), method='bounded',
                                 options={'maxiter': 100, 'xatol': 1e-12})
        if fitted.success:
            candidates.append(float(fitted.x))
    best_gap = min(candidates, key=lambda value: solve(value)[0])
    cost, a, theta, normalized_offset = solve(best_gap)
    amplitude = float(a * scale)
    offset = float(normalized_offset * scale)
    predicted = offset + amplitude / (t - theta)
    residual = y - predicted
    rss = float(residual @ residual)
    dof = len(t) - p
    warnings = []
    if len(t) < 6:
        warnings.append('Few temperatures: exploratory fit; add higher-temperature data.')
    # Local covariance is only meaningful away from singular/active boundaries.
    jac = np.column_stack([1/(t-theta), a/(t-theta)**2] + ([np.ones_like(t)] if p == 3 else []))
    condition = float(np.linalg.cond(jac))
    reliable = condition < 1e9 and lower + 1e-5 < best_gap < upper - 1e-5
    stderr = None
    ci = None
    if reliable:
        _, singular, vt = np.linalg.svd(jac, full_matrices=False)
        covariance = (vt.T / singular ** 2) @ vt * cost / dof
        stderr = float(np.sqrt(max(0., covariance[1, 1])))
        radius = float(student_t.ppf(.975, dof) * stderr)
        ci = [theta - radius, theta + radius]
    else:
        warnings.append('Parameters are poorly constrained or at a fit bound; uncertainty unavailable.')
    r2 = 1 - rss / float(np.sum((y - np.mean(y)) ** 2))
    if r2 < .9:
        warnings.append('Weak Curie–Weiss agreement; inspect residuals and temperature range.')
    if ci is not None and ci[1] >= float(np.min(t)):
        warnings.append('Local confidence interval approaches the pole constraint; interpret cautiously.')
    if ci is None or ci[0] <= 0 <= ci[1]:
        interpretation = 'Unresolved interaction sign'
    else:
        interpretation = 'FM interaction tendency' if theta > 0 else 'AFM interaction tendency'
    return dict(theta_k=theta, theta_se_k=stderr, theta_ci95_k=ci,
                amplitude=amplitude, background=offset, background_mode=background_mode,
                n=len(t), dof=dof, r_squared=r2, residual_rms=float(np.sqrt(rss / len(t))),
                temperature_min_k=float(np.min(t)), temperature_max_k=float(np.max(t)),
                predicted=predicted.tolist(), residuals=residual.tolist(),
                interpretation=interpretation, warnings=warnings,
                method='Unweighted nonlinear least squares in slope space',
                uncertainty='Local covariance, residual-scaled Student-t 95% interval; excludes systematic errors')
