"""Bounded compact log-background edits with a soft raw P/N curvature target.

Only the supplied parent and raw P/N means enter the construction. Numerical
feasibility is separate from the declared approximation budget, and neither is
scientific acceptance of the source. No sample feature is fitted here.
"""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline, CubicSpline
from scipy.optimize import lsq_linear

from .feature_protection import _check, _geometry, normalize_feature_protection


COMPACT_PROTECTION_PROTOCOL = {
    "version": "bounded_compact_raw_curvature_v1",
    "experimental_offline": True,
    "source": "CubicSpline of log(raw P/N mean)-log(parent), on the full native energy axis",
    "geometry": "same requested core, SG21 guard plateau, and additional SG21 transition support as explicit raw-source protection",
    "SG21_span": "20*(energy[-1]-energy[0])/(len(energy)-1)",
    "basis": "degree-5 open clamped B-spline in u=(E-support_left)/support_width; first and last three coefficients are exactly zero",
    "minimum_knot_spacing_SG21_spans": 0.5,
    "knot_policy": "uniform support intervals; floor(support_width/(0.5*SG21_span)) intervals, giving spacing no smaller than 0.5*SG21_span",
    "objective": "integral_support (h_uu-support_width**2*CubicSpline(log(raw)-log(parent))_EE)**2 du; no taper or additional regularizer",
    "quadrature": "four-point Gauss-Legendre on the union of native, basis, and requested/plateau/support breaks; exact for the piecewise degree-6 squared residual up to floating-point error",
    "coefficient_bounds": "every free coefficient in [-log(1.05),+log(1.05)]; partition of unity also bounds the continuous h",
    "log_correction_cap": float(np.log(1.05)),
    "continuous_gain_bounds": [float(1/1.05), 1.05],
    "exterior_policy": "copy native parent values exactly outside the open support; compact h,h_E,h_EE vanish at both support boundaries",
    "relative_curvature_rms_budget": 0.50,
    "fit_acceptance": "core, plateau, and full support each require relative curvature RMS error <=0.50; a near-zero target permits only machine-scale absolute error",
    "budget_interpretation": "predeclared approximation budget, not a confidence interval or scientific acceptance",
    "affine_roundoff_multiplier": 64,
    "solver_tolerance": 1e-12,
    "solver_max_iterations": 500,
    "rejected_fit_policy": "return the actual bounded numerical candidate with diagnostics; caller must retain Parent for adoption when fit_accepted is false",
    "scientifically_validated": False,
    "production_default_allowed": False,
}


def _basis(geometry):
    left, right = geometry["support_bounds_eV"]
    width = right-left
    span = geometry["SG21_full_uniform_span_ev"]
    minimum_spacing = COMPACT_PROTECTION_PROTOCOL["minimum_knot_spacing_SG21_spans"]*span
    intervals = max(1, int(np.floor(width/minimum_spacing)))
    breaks = np.linspace(0., 1., intervals+1)
    degree = 5
    knots = np.r_[np.zeros(degree+1), breaks[1:-1], np.ones(degree+1)]
    ncoef = len(knots)-degree-1
    free_indices = np.arange(3, ncoef-3)
    return knots, BSpline(knots, np.eye(ncoef)[:, free_indices], degree,
                          extrapolate=False), free_indices


def _affine_roundoff(log_ratio, e, log_scale):
    """Recognize an affine log ratio before interpolation amplifies roundoff."""
    x = (e-e[0])/(e[-1]-e[0])-.5
    design = np.column_stack((np.ones(len(e)), x))
    affine = np.linalg.lstsq(design, log_ratio, rcond=None)[0]
    error = float(np.max(abs(log_ratio-design@affine)))
    tolerance = float(COMPACT_PROTECTION_PROTOCOL["affine_roundoff_multiplier"]
                      * np.finfo(float).eps*max(1., log_scale))
    return error <= tolerance, error, tolerance


def _quadrature(e, geometry, knots, cancelled):
    """Keep every source polynomial break, including very short native cells."""
    left, right = geometry["support_bounds_eV"]
    width = right-left
    geometry_breaks = [v for key in ("requested_bounds_eV", "plateau_bounds_eV",
                                     "support_bounds_eV") for v in geometry[key]]
    breaks = np.unique(np.r_[0., 1., (e[(e > left) & (e < right)]-left)/width,
                              knots, (np.asarray(geometry_breaks)-left)/width])
    breaks = breaks[(breaks >= 0.) & (breaks <= 1.)]
    gaussian_x, gaussian_w = leggauss(4)
    mid = (breaks[:-1]+breaks[1:])/2
    half = np.diff(breaks)/2
    _check(cancelled)
    u = (mid[:, None]+half[:, None]*gaussian_x).ravel()
    weights = (half[:, None]*gaussian_w).ravel()
    if not np.all(np.isfinite(u)) or np.any(weights <= 0):
        raise ArithmeticError("Compact protection has invalid integration cells")
    return u, weights, breaks


def _fit_metrics(u, weights, target, actual, geometry):
    left, right = geometry["support_bounds_eV"]
    width = right-left
    result = {}
    # h_uu is dimensionless. The only absolute zero-target allowance is a fixed
    # floating-point relative-scale floor, not a feature/noise threshold.
    machine_tolerance = float(64*np.finfo(float).eps
                              * max(1., float(np.max(abs(target)))))
    for name, bounds in (("core", geometry["requested_bounds_eV"]),
                         ("plateau", geometry["plateau_bounds_eV"]),
                         ("support", geometry["support_bounds_eV"])):
        a, b = (np.asarray(bounds)-left)/width
        mask = (u >= a) & (u <= b)
        w = weights[mask]
        if not len(w) or not np.sum(w) > 0:
            raise ArithmeticError("Compact protection fit domain has no quadrature weight")
        residual_rms = float(np.sqrt(np.sum(w*(actual[mask]-target[mask])**2)/np.sum(w)))
        target_rms = float(np.sqrt(np.sum(w*target[mask]**2)/np.sum(w)))
        near_zero = target_rms <= machine_tolerance
        relative = None if near_zero else float(residual_rms/target_rms)
        accepted = (residual_rms <= machine_tolerance if near_zero else relative
                    <= COMPACT_PROTECTION_PROTOCOL["relative_curvature_rms_budget"])
        result[name] = dict(target_rms_u2=target_rms, residual_rms_u2=residual_rms,
            relative_rms_error=relative, near_zero_target=bool(near_zero),
            machine_absolute_tolerance_u2=machine_tolerance, accepted=bool(accepted))
    return result


def apply_compact_protection(energy, parent_background, pn_mean, regions, *, cancelled=None):
    """Return the actual bounded candidate, numeric arrays, and JSON diagnostics.

    The continuous model is ``parent(E)*exp(h(E))``. At native points outside
    every open support the output is a direct copy of the supplied parent. A
    converged fit that misses its approximation budget is still returned for
    diagnosis, with ``fit_accepted=False``; it must not be adopted by a caller.
    Geometry/input errors and numerical solve failures raise. Cancellation is
    propagated. Empty regions return an exact parent copy without a spline.
    """
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable or None")
    _check(cancelled)
    requested = normalize_feature_protection(regions)
    e = np.asarray(energy, dtype=float)
    parent = np.asarray(parent_background, dtype=float)
    pn = np.asarray(pn_mean, dtype=float)
    if (e.ndim != 1 or len(e) < 21 or not np.isfinite(e).all()
            or np.any(np.diff(e) <= 0)):
        raise ValueError("Finite increasing energy with at least 21 points required")
    if parent.shape != e.shape or not np.all(np.isfinite(parent) & (parent > 0)):
        raise ValueError("Positive finite native parent background required")
    if pn.shape != (2, len(e)) or not np.all(np.isfinite(pn) & (pn > 0)):
        raise ValueError("Positive finite raw P/N means with shape (2, energy) required")
    geometries = _geometry(e, requested)
    n = len(e)
    delta = np.zeros(n)
    curvature_delta = np.zeros(n)
    target_curvature_delta = np.zeros(n)
    core_mask, plateau_mask, support_mask = np.zeros((3, n), dtype=bool)
    region_support_masks = np.zeros((len(geometries), n), dtype=bool)
    jets = np.zeros((len(geometries), 2, 3))
    rows, all_knots, all_coefficients = [], [], []
    background = parent.copy()
    cap = COMPACT_PROTECTION_PROTOCOL["log_correction_cap"]
    lp = np.log(parent) if geometries else None
    for index, geometry in enumerate(geometries):
        _check(cancelled)
        lo, hi = geometry["requested_bounds_eV"]
        a, b = geometry["plateau_bounds_eV"]
        left, right = geometry["support_bounds_eV"]
        width = right-left
        active = (e > left) & (e < right)
        core_mask |= (e >= lo) & (e <= hi)
        plateau_mask |= (e >= a) & (e <= b)
        support_mask |= active
        region_support_masks[index] = active
        raw_log = np.log(pn[0 if geometry["source"] == "P" else 1])
        log_ratio = raw_log-lp
        affine_noop, affine_error, affine_tolerance = _affine_roundoff(
            log_ratio, e, max(float(np.max(abs(raw_log))), float(np.max(abs(lp)))))
        knots, basis, free_indices = _basis(geometry)
        coefficients = np.zeros(len(knots)-6)
        u, weights, integration_breaks = _quadrature(e, geometry, knots, cancelled)
        source_spline = None
        if affine_noop:
            target = np.zeros(len(u))
            actual = np.zeros(len(u))
            solver_detail = dict(success=True, status="affine_roundoff_exact_noop",
                                 iterations=0, optimality=0., cost=0.)
        else:
            source_spline = CubicSpline(e, log_ratio, extrapolate=False)
            target = width**2*source_spline(left+width*u, 2)
            design = basis(u, 2)
            root_weight = np.sqrt(weights)
            _check(cancelled)
            solved = lsq_linear(design*root_weight[:, None], target*root_weight,
                bounds=(-cap, cap), method="trf", lsq_solver="exact",
                tol=COMPACT_PROTECTION_PROTOCOL["solver_tolerance"],
                max_iter=COMPACT_PROTECTION_PROTOCOL["solver_max_iterations"])
            _check(cancelled)
            if not solved.success or not np.isfinite(solved.x).all():
                raise ArithmeticError("Compact protection bounded least-squares did not converge")
            coefficients[free_indices] = solved.x
            if np.max(abs(coefficients)) > cap:
                raise ArithmeticError("Compact protection solver exceeded its coefficient bounds")
            actual = design@solved.x
            solver_detail = dict(success=True, status=int(solved.status),
                iterations=int(solved.nit), optimality=float(solved.optimality),
                cost=float(solved.cost))
        if not np.isfinite(target).all() or not np.isfinite(actual).all():
            raise ArithmeticError("Compact protection curvature target or fit is nonfinite")
        spline = BSpline(knots, coefficients, 5, extrapolate=False)
        native_u = (e[active]-left)/width
        delta[active] = spline(native_u)
        curvature_delta[active] = spline(native_u, 2)/width**2
        if source_spline is not None:
            target_curvature_delta[active] = source_spline(e[active], 2)
        changed = active & (delta != 0)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            background[changed] = parent[changed]*np.exp(delta[changed])
        for derivative in range(3):
            jets[index, :, derivative] = spline([0., 1.], derivative)/width**derivative
        metrics = _fit_metrics(u, weights, target, actual, geometry)
        reasons = [name+"_curvature_rms_budget_exceeded" for name, metric in metrics.items()
                   if not metric["accepted"]]
        active_bound_mask = abs(coefficients[free_indices]) >= cap-64*np.finfo(float).eps
        rows.append({**deepcopy(geometry), "degree": 5, "knots_u": knots.tolist(),
            "coefficients": coefficients.tolist(), "coordinate": "u=(E-support_left)/support_width",
            "log_correction_cap": cap, "continuous_gain_bounds": [float(1/1.05), 1.05],
            "source_status": "raw_mean_intensity", "fit_accepted": not reasons,
            "fit_rejection_reasons": reasons, "curvature_fit_metrics": metrics,
            "active_bound_count": int(np.count_nonzero(active_bound_mask)),
            "active_bound_coefficient_indices": free_indices[active_bound_mask].tolist(),
            "fixed_zero_coefficient_indices": [0, 1, 2, len(coefficients)-3,
                                                 len(coefficients)-2, len(coefficients)-1],
            "basis_interval_count": int(len(np.unique(knots))-1),
            "basis_spacing_ev": float(width/(len(np.unique(knots))-1)),
            "integration_cell_count": int(len(integration_breaks)-1),
            "affine_roundoff_exact_noop": bool(affine_noop),
            "affine_log_ratio_max_residual": affine_error,
            "affine_log_ratio_machine_tolerance": affine_tolerance,
            "solver": solver_detail,
            "boundary_log_jets_E": jets[index].tolist(),
            "boundary_C0_C1_C2_residual_max": np.max(abs(jets[index]), axis=0).tolist(),
            "scientifically_validated": False})
        all_knots.append(knots)
        all_coefficients.append(coefficients)
    _check(cancelled)
    if not np.all(np.isfinite(background) & (background > 0)):
        raise ArithmeticError("Compact protection did not produce positive finite intensity")
    # Disabled and affine-only requests preserve the supplied dtype and bits.
    if not np.any(delta):
        background = np.array(parent_background, copy=True)
    counts = [len(value) for value in all_coefficients]
    knot_counts = [len(value) for value in all_knots]
    padded_coefficients = np.zeros((len(rows), max(counts, default=0)))
    padded_knots = np.zeros((len(rows), max(knot_counts, default=0)))
    for index, (knots, coefficients) in enumerate(zip(all_knots, all_coefficients)):
        padded_knots[index, :len(knots)] = knots
        padded_coefficients[index, :len(coefficients)] = coefficients
    arrays = dict(energy=e.copy(), delta_log=delta, curvature_delta=curvature_delta,
        target_curvature_delta=target_curvature_delta, core_mask=core_mask,
        plateau_mask=plateau_mask, support_mask=support_mask,
        curvature_support_mask=support_mask.copy(), region_support_masks=region_support_masks,
        boundary_log_jet_residuals=jets, basis_knots_u=padded_knots,
        basis_knot_counts=np.asarray(knot_counts, dtype=int),
        basis_coefficients=padded_coefficients,
        basis_coefficient_counts=np.asarray(counts, dtype=int))
    reasons = [dict(region_index=index, reasons=row["fit_rejection_reasons"])
               for index, row in enumerate(rows) if not row["fit_accepted"]]
    report = dict(protocol=deepcopy(COMPACT_PROTECTION_PROTOCOL), enabled=bool(rows),
        construction_mode="compact_bounded", regions=rows, numeric_valid=True,
        fit_accepted=not reasons, fit_rejection_reasons=reasons,
        scientifically_validated=False, production_default_allowed=False,
        automatic_source_selection=False, automatic_application=False,
        sample_used=False, BN_used=False, source_feature_fit_used=False,
        parent_indirect_sample_dependence="The supplied parent may depend on training samples; this operation takes no sample arrays or sample-feature objective.",
        outside_support_native_parent_exact=bool(np.array_equal(background[~support_mask], parent[~support_mask])),
        outside_support_parent_log_curvature_preserved=True,
        continuous_log_correction_cap=cap, positive_finite=True,
        whole_axis_gain_minmax=[float(np.min(background/parent)), float(np.max(background/parent))],
        array_contract=dict(delta_log="applied compact log increment relative to parent",
            curvature_delta="analytic native h_EE; zero outside open supports",
            target_curvature_delta="raw log-ratio CubicSpline curvature inside support; affine roundoff is canonicalized to zero",
            support_mask="union of open supports of both intensity and log-curvature edits",
            boundary_log_jet_residuals="region x [support_left,support_right] x derivative_E 0/1/2; exterior jets are zero",
            basis_coefficients="zero-padded full coefficient vectors; slice using basis_coefficient_counts",
            basis_knots_u="zero-padded full knot vectors in u; slice using basis_knot_counts"),
        limitations=["Numerical feasibility and the approximation budget are not physical validation of raw P/N or of sample DRR derivatives.",
            "The soft curvature target is generally incompatible with compact intensity closure; fit_accepted=false requires retention of Parent for adoption.",
            "C2 compact closure and bounded gain do not prove absence of transition artifacts. Review full fixed-processing sample products and independent repeats.",
            "Full-axis cubic regridding can couple derivative outputs beyond the native intensity support."])
    json.dumps(report, allow_nan=False)
    return background, arrays, report
