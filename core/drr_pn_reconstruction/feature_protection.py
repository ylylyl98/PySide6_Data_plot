"""Explicit raw-source log-curvature protection with compact C2 transitions.

Only the parent and raw P/N intensity means enter this operation. A requested
source is a user choice, not a clean-source classification. No sample, BN,
stationarity score, weak-peak threshold, or source-feature fit is used.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline, PPoly


FEATURE_PROTECTION_PROTOCOL = {
    "version": "explicit_raw_source_curvature_protection_v1",
    "source": "raw P/N mean intensity, before source-feature correction",
    "gauge": "full-axis native trapezoid-weighted least-squares affine projection of the integrated log change",
    "guard": "one complete SG21 uniform-grid span on each side of the requested interval",
    "transition": "one additional SG21 span per side; quintic convex blend of parent/source log curvature",
    "integration": "exact piecewise-polynomial multiplication and two antiderivatives",
    "SG21_span": "20*(energy[-1]-energy[0])/(len(energy)-1)",
    "minimum_transition_native_points": 4,
    "half_policy": "same full-data global affine parameters and geometry; integrate own raw-source and parent curvature",
    "boundary_policy": "require support strictly inside the measured domain; reject touching or overlapping supports",
    "exterior_policy": "preserve parent log curvature; intensity changes by affine log gain/tilt on each exterior component",
    "scientifically_validated": False,
}


def normalize_feature_protection(regions):
    """Return sorted JSON rows; reject malformed, overlapping or touching cores."""
    if regions is None:
        return []
    if not isinstance(regions, (tuple, list)):
        raise ValueError("Feature protection must be a list of region objects")
    result = []
    for row in regions:
        if not isinstance(row, Mapping) or set(row) != {"xmin_ev", "xmax_ev", "source"}:
            raise ValueError("Each protection region needs only xmin_ev, xmax_ev and source")
        try:
            if any(isinstance(row[key], (bool, np.bool_)) for key in ("xmin_ev", "xmax_ev")):
                raise ValueError
            lo, hi = float(row["xmin_ev"]), float(row["xmax_ev"])
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Protection bounds must be finite energies") from exc
        if not np.isfinite([lo, hi]).all() or not lo < hi:
            raise ValueError("Protection bounds must be finite and increasing")
        source = row["source"]
        if not isinstance(source, str) or source.strip().upper() not in ("P", "N"):
            raise ValueError("Protection source must be P or N")
        result.append(dict(xmin_ev=lo, xmax_ev=hi, source=source.strip().upper()))
    result.sort(key=lambda row: (row["xmin_ev"], row["xmax_ev"], row["source"]))
    if any(left["xmax_ev"] >= right["xmin_ev"] for left, right in zip(result, result[1:])):
        raise ValueError("Protection regions must not overlap or touch")
    return result


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("Feature protection cancelled")


def _axis_hash(e):
    return hashlib.sha256(np.ascontiguousarray(e, dtype="<f8").tobytes()).hexdigest()


def _geometry(e, regions):
    span = float(20 * (e[-1] - e[0]) / (len(e) - 1))
    geometries = []
    for row in regions:
        lo, hi = row["xmin_ev"], row["xmax_ev"]
        plateau = [lo-span, hi+span]
        support = [plateau[0]-span, plateau[1]+span]
        if support[0] <= e[0] or support[1] >= e[-1]:
            raise ValueError("Protection requires complete SG21 guards and transitions strictly inside the measured domain")
        if not np.any((e >= lo) & (e <= hi)):
            raise ValueError("Protection region contains no native energy points")
        counts = [int(np.count_nonzero((e >= a) & (e <= b))) for a, b in (
            (support[0], plateau[0]), (plateau[1], support[1]))]
        if min(counts) < FEATURE_PROTECTION_PROTOCOL["minimum_transition_native_points"]:
            raise ValueError("Protection transitions need at least four native points each")
        geometry = dict(source=row["source"], requested_bounds_eV=[lo, hi],
            plateau_bounds_eV=plateau, support_bounds_eV=support,
            SG21_full_uniform_span_ev=span, transition_native_counts=counts)
        if geometries and geometries[-1]["support_bounds_eV"][1] >= support[0]:
            raise ValueError("Expanded feature-protection supports must not overlap or touch")
        geometries.append(geometry)
    return geometries


def _profile(x, geometry):
    """Return the quintic profile and its two analytic energy derivatives."""
    left, right = geometry["support_bounds_eV"]
    a, b = geometry["plateau_bounds_eV"]
    w, d1, d2 = np.zeros((3, len(x)))
    w[(x >= a) & (x <= b)] = 1.
    for mask, t, width, direction in (
        ((x > left) & (x < a), (x-left)/(a-left), a-left, 1.),
        ((x > b) & (x < right), (right-x)/(right-b), right-b, -1.),
    ):
        t = t[mask]
        u = np.minimum(t, 1-t)
        s = u**3 * (10 + u*(-15+6*u))
        w[mask] = np.where(t <= .5, s, 1-s)
        d1[mask] = direction*30*t*t*(1-t)**2/width
        d2[mask] = 60*t*(1-t)*(1-2*t)/(width*width)
    return w, d1, d2


def _curvature_polynomial(e, parent, sources, geometries, cancelled):
    """Exact quintic-weight times piecewise-linear curvature, degree at most 6."""
    bounds = [value for row in geometries for key in ("support_bounds_eV", "plateau_bounds_eV")
              for value in row[key]]
    knots = np.unique(np.r_[e, bounds])
    coefficients = np.zeros((7, len(knots)-1))
    region_index = 0
    for index, (lo, hi) in enumerate(zip(knots[:-1], knots[1:])):
        if index % 128 == 0:
            _check(cancelled)
        mid = (lo+hi)/2
        while (region_index < len(geometries)
               and mid >= geometries[region_index]["support_bounds_eV"][1]):
            region_index += 1
        if region_index == len(geometries):
            break
        row = geometries[region_index]
        left, right = row["support_bounds_eV"]
        a, b = row["plateau_bounds_eV"]
        if mid <= left:
            continue
        if a <= mid <= b:
            weight = Polynomial([1.])
        else:
            u = (Polynomial([lo-left, 1.])/(a-left) if mid < a
                 else Polynomial([right-lo, -1.])/(right-b))
            weight = 10*u**3-15*u**4+6*u**5
        source = sources[row["source"]]
        difference = Polynomial([float(source(lo, 2)-parent(lo, 2)),
                                 float(source(mid, 3)-parent(mid, 3))])
        product = weight*difference
        coefficients[-len(product.coef):, index] = product.coef[::-1]
    return PPoly(coefficients, knots, extrapolate=False)


def _boundary_jet_residuals(polynomial, bounds):
    """Left/right value, slope and curvature jumps at exact construction joins."""
    residuals = np.zeros((len(bounds), 3))
    for index, bound in enumerate(bounds):
        knot = int(np.searchsorted(polynomial.x, bound))
        width = polynomial.x[knot]-polynomial.x[knot-1]
        for derivative in range(3):
            pp = polynomial.derivative(derivative)
            residuals[index, derivative] = (np.polyval(pp.c[:, knot-1], width)
                                           - pp.c[-1, knot])
    return residuals


def apply_feature_protection(energy, parent_background, pn_mean, regions, *,
                             frozen_parameters=None, cancelled=None):
    """Return positive native background, numeric arrays, and JSON-safe report.

    Requested cores plus a complete nominal SG21 guard retain raw-source log
    curvature. On each exterior component the parent's log curvature is retained,
    while its intensity can change by affine log gain/tilt. The transition is a
    convex log-curvature mixture; it has no forced intensity closure terms.
    Exact polynomial integration and a full-axis affine projection determine the
    intensity curve. Neither curvature preservation nor C2 continuity certifies
    sample DRR derivatives: full products and global gain changes require review.

    Pass the full report's ``frozen_parameters`` for conditional half replay.
    This freezes geometry and global integration offset/tilt; each half uses its
    own source and parent curvatures. The parent can indirectly depend on training
    samples, although this function takes no sample arrays and performs no fits
    of sample features or source features.
    Empty regions return an exact parent copy and do not construct splines.
    """
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable or None")
    _check(cancelled)
    requested = normalize_feature_protection(regions)
    e = np.asarray(energy, dtype=float)
    parent = np.asarray(parent_background, dtype=float)
    pn = np.asarray(pn_mean, dtype=float)
    if (e.ndim != 1 or len(e) < 21 or not np.isfinite(e).all()
            or not np.all(np.diff(e) > 0)):
        raise ValueError("Finite increasing energy with at least 21 points required")
    if parent.shape != e.shape or not np.all(np.isfinite(parent) & (parent > 0)):
        raise ValueError("Positive finite native parent background required")
    if pn.shape != (2, len(e)) or not np.all(np.isfinite(pn) & (pn > 0)):
        raise ValueError("Positive finite raw P/N means with shape (2, energy) required")
    geometries = _geometry(e, requested)
    identity = dict(schema_version=1, protocol=FEATURE_PROTECTION_PROTOCOL["version"],
                    energy_sha256=_axis_hash(e), regions=deepcopy(requested), geometry=deepcopy(geometries))
    parameters = None
    center = float((e[0]+e[-1])/2)
    if frozen_parameters is not None:
        if not isinstance(frozen_parameters, Mapping) or any(frozen_parameters.get(k) != v for k, v in identity.items()):
            raise ValueError("Frozen feature-protection identity/geometry mismatch")
        try:
            parameters = np.asarray(frozen_parameters["global_affine_parameters"], dtype=float)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("Invalid frozen feature-protection affine parameters") from exc
        if (parameters.shape != (3,) or not np.isfinite(parameters).all()
                or parameters[2] != center):
            raise ValueError("Invalid frozen feature-protection affine parameters")
        parameters = parameters.copy()

    n, count = len(e), len(requested)
    weights = np.zeros((count, n))
    core_mask = np.zeros(n, bool)
    plateau_mask = np.zeros(n, bool)
    support_mask = np.zeros(n, bool)
    delta = np.zeros(n)
    boundary_residuals = np.zeros((count, 4, 3))
    background = parent.copy()
    source_parameters = np.zeros((count, 3))
    rows = []
    if count:
        lp = np.log(parent)
        parent_spline = CubicSpline(e, lp, extrapolate=False)
        source_splines = {source: CubicSpline(e, np.log(pn[0 if source == "P" else 1]), extrapolate=False)
                          for source in {row["source"] for row in geometries}}
        curvature = _curvature_polynomial(e, parent_spline, source_splines, geometries, cancelled)
        integrated = curvature.antiderivative(2)
        q = integrated(e)
        if parameters is None:
            spacing = np.diff(e)
            quadrature = np.r_[spacing[0]/2, (spacing[:-1]+spacing[1:])/2, spacing[-1]/2]
            design = np.column_stack([np.ones(n), e-center])
            root_weight = np.sqrt(quadrature)
            affine, _, rank, _ = np.linalg.lstsq(design*root_weight[:, None], -q*root_weight, rcond=None)
            if rank != 2:
                raise ArithmeticError("Feature-protection global affine projection lacks rank")
            parameters = np.r_[affine, center]
        delta = q+parameters[0]+parameters[1]*(e-parameters[2])
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            changed = delta != 0
            background[changed] = np.exp(lp[changed]+delta[changed])
        if not np.all(np.isfinite(background) & (background > 0)):
            raise ValueError("Feature protection did not produce positive finite intensity")
        delta_coefficients = integrated.c.copy()
        delta_coefficients[-2] += parameters[1]
        delta_coefficients[-1] += parameters[0]+parameters[1]*(integrated.x[:-1]-parameters[2])
        delta_polynomial = PPoly(delta_coefficients, integrated.x, extrapolate=False)
        curvature_delta = curvature(e)
    else:
        parameters = np.array([0., 0., center]) if parameters is None else parameters
        curvature_delta = np.zeros(n)
        curvature = PPoly(np.zeros((7, n-1)), e, extrapolate=False)
        delta_polynomial = PPoly(np.zeros((9, n-1)), e, extrapolate=False)
    for index, geometry in enumerate(geometries):
        _check(cancelled)
        raw_source = pn[0 if geometry["source"] == "P" else 1]
        source_log = np.log(raw_source)
        source_spline = source_splines[geometry["source"]]
        lo, hi = geometry["requested_bounds_eV"]
        a, b = geometry["plateau_bounds_eV"]
        left, right = geometry["support_bounds_eV"]
        source_center = (lo+hi)/2
        param = np.array([float(parent_spline(source_center)+delta_polynomial(source_center)-source_spline(source_center)),
                          float(parent_spline(source_center, 1)+delta_polynomial(source_center, 1)-source_spline(source_center, 1)),
                          source_center])
        source_parameters[index] = param
        gauge = param[0]+param[1]*(e-param[2])
        w, _, _ = _profile(e, geometry)
        weights[index] = w
        active, plateau, core = w > 0, w == 1, (e >= lo) & (e <= hi)
        target = source_log+gauge
        core_mask |= core
        plateau_mask |= plateau
        support_mask |= active
        joins = np.array([left, a, b, right])
        boundary_residuals[index] = _boundary_jet_residuals(delta_polynomial, joins)
        rows.append({**deepcopy(geometry), "source_status": "raw_mean_intensity",
            "source_affine_parameters": param.tolist(), "affine_parameter_order": ["offset_log", "tilt_log_per_ev", "center_ev"],
            "source_affine_parameters_derived_from_integrated_curve": True,
            "plateau_log_shape_error_max": float(np.max(abs(np.log(background[plateau])-target[plateau]))),
            "boundary_C0_C1_C2_residual_max": np.max(abs(boundary_residuals[index]), axis=0).tolist(),
            "source_core_gain_minmax": [float(np.min(background[core]/raw_source[core])), float(np.max(background[core]/raw_source[core]))],
            "support_applies_to": "log curvature; intensity change is global",
            "status": "explicit_raw_source_protection", "scientifically_validated": False})
    _check(cancelled)
    if not count:
        # Preserve the original dtype and bit pattern for the disabled default.
        background = np.array(parent_background, copy=True)
    frozen = {**identity, "global_affine_parameters": parameters.tolist()}
    arrays = dict(energy=e.copy(), weights=weights, core_mask=core_mask,
        plateau_mask=plateau_mask, curvature_support_mask=support_mask, delta_log=delta,
        curvature_delta=curvature_delta, global_affine_parameters=parameters.copy(),
        source_affine_parameters=source_parameters, boundary_log_jet_residuals=boundary_residuals,
        curvature_knots=curvature.x.copy(), curvature_delta_coefficients=curvature.c.copy(),
        delta_log_polynomial_coefficients=delta_polynomial.c.copy())
    report = dict(protocol=deepcopy(FEATURE_PROTECTION_PROTOCOL), enabled=bool(count),
        scientifically_validated=False, automatic_source_selection=False,
        sample_used=False, BN_used=False, source_feature_fit_used=False,
        parent_indirect_sample_dependence="The supplied parent may depend on training samples; this operation takes no sample arrays or sample-feature objective.",
        regions=rows, frozen_parameters=frozen, affine_parameters_reused=frozen_parameters is not None,
        outside_support_native_parent_exact=bool(np.array_equal(background[~support_mask], parent[~support_mask])),
        outside_support_parent_log_curvature_preserved=True,
        convex_log_curvature_mixture=True, positive_finite=True,
        whole_axis_gain_minmax=[float(np.min(background/parent)), float(np.max(background/parent))],
        global_affine_parameters=parameters.tolist(),
        array_contract={"weights": "region x native energy; quintic log-curvature mixture",
            "delta_log": "applied log increment relative to the supplied parent",
            "curvature_support_mask": "support of log-curvature change, not support of intensity change",
            "source_affine_parameters": "region x [offset_log, tilt_log_per_ev, center_ev] relative to raw source on its plateau",
            "global_affine_parameters": "frozen full-data integration gauge [offset_log, tilt_log_per_ev, center_ev]",
            "delta_log_polynomial_coefficients": "descending PPoly coefficients of final log increment on curvature_knots, including global affine gauge",
            "boundary_log_jet_residuals": "region x [outer-left, plateau-left, plateau-right, outer-right] x derivative 0/1/2",
            "final_polynomial_available": False},
        limitations=["Raw source choice is explicit and is not scientific certification of that source or the protected features.",
            "Affine log gain/tilt preserves source log curvature on the plateau, not absolute sample DRR amplitudes or derivatives.",
            "Exterior parent log curvature is preserved, but exterior intensity and sample DRR derivatives can change through global gain/tilt.",
            "SG21 guards describe nominal uniform-window support; sample differentiation also extends changes beyond the curvature support, and full-axis cubic regridding has global coefficients.",
            "C2 joins do not prove the absence of transition artifacts. Review complete fixed-processing products and held-out repeats."])
    return background, arrays, report
