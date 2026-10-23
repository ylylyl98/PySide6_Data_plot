# Migrated numerical kernel. Version names describe recipe lineage, not stored inputs.
"""Mirror wider left source-curvature bridge; no file/sample/BN access.

The fixed two-window support is an engineering hypothesis, not a cleanliness
test. Source curves retain their native fine structure. Only construction
weights, curvature closure and the propagated left constant level change.
"""
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, PPoly

FAMILIES = ("wider_left_curvature_free_level",)


def generate(e, *, baseline, region):
    e = np.asarray(e)
    if e.ndim != 1 or len(e) < 4 or not np.isfinite(e).all() or np.any(np.diff(e) <= 0):
        raise ValueError("Complete finite increasing native energy required")
    np.testing.assert_array_equal(e, baseline["energy"])
    old_left, right = map(int, region)
    if not 0 <= old_left < right < len(e):
        raise ValueError("Explicit valid native bridge endpoints required")
    sg21_span = 20*float(e[-1]-e[0])/(len(e)-1)
    requested_left_energy = e[right]-max(float(e[right]-e[old_left]), 2*sg21_span)
    left = int(np.searchsorted(e, requested_left_energy, side="right")-1)
    if left < 0:
        raise ValueError("Fixed two-SG21-span extension exceeds the measured domain")
    A = np.asarray(baseline["original_aligned_source_log"])
    parent_weights = np.asarray(baseline["source_weights"])
    weights = np.array(parent_weights, copy=True)
    old_coeff = np.asarray(baseline["integrated_log_polynomial_coefficients"])
    old_log, old_B = baseline["background_log"], baseline["background"]
    if A.shape != (2, len(e)) or weights.shape != A.shape or old_coeff.shape != (7, len(e)-1):
        raise ValueError("Invalid native source/weight/polynomial shape")
    if not np.isfinite(A).all() or not np.isfinite(old_coeff).all() or not np.all(old_B > 0):
        raise ValueError("Sources and positive background must be finite")
    np.testing.assert_array_equal(parent_weights[:, right], [1., 0.])
    np.testing.assert_array_equal(parent_weights[0, left:old_left+1], 0.)
    if np.any(np.asarray(baseline["state"])[left:old_left+1] == 1):
        raise ValueError("Extension conflicts with a frozen P-exclusive point; no fallback")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("Weights outside [0,1]")
    np.testing.assert_allclose(weights.sum(0), 1., rtol=0, atol=2e-15)
    t_native = (e[left:right+1]-e[left])/(e[right]-e[left])
    new_p = t_native**3*(10-15*t_native+6*t_native**2)
    new_p[0], new_p[-1] = 0., 1.
    weights[0, left:right+1] = new_p
    weights[1, left:right+1] = 1-new_p
    old = PPoly(old_coeff, e, extrapolate=False)
    np.testing.assert_allclose(old(e), old_log, rtol=0, atol=1e-10)
    source = CubicSpline(e, A.T, axis=0, extrapolate=False)
    W = PchipInterpolator(e, weights[0], extrapolate=False)
    anchor = baseline["cubic_anchor_coefficients"]
    center, scale = float(baseline["anchor_coordinate_center"]), float(baseline["anchor_coordinate_scale"])
    if np.asarray(anchor).shape != (4,) or not scale > 0:
        raise ValueError("Frozen cubic coefficients and normalization required")
    u = (e[:-1]-center)/scale
    anchor_dd = np.array([(2*anchor[2]+6*anchor[3]*u)/scale**2,
                         np.full(len(e)-1, 6*anchor[3]/scale**3)])
    dd = source.derivative(2).c[::-1]
    ndd = dd[:, :, 1]+anchor_dd
    ha = np.zeros((5, len(e)-1))
    for i in range(4):
        for j in range(2):
            ha[i+j] += W.c[::-1][i]*(dd[j, :, 0]-ndd[j])
    ha[:2] += ndd
    h0 = PPoly(ha[::-1], e, extrapolate=False)
    expected_curv = old(e[[left, right]], 2)
    endpoint_curv_error = h0(e[[left, right]])-expected_curv
    if np.max(abs(endpoint_curv_error)) > 1e-6:
        raise ValueError("Frozen endpoint source/cubic curvature is incompatible with the parent")
    H, dx = float(e[right]-e[left]), np.diff(e)
    orders = np.arange(5)[:, None]
    hi = np.sum(ha*dx**(orders+1)/(orders+1), axis=0)
    slope_defect = float(old(e[right], 1)-old(e[left], 1)-hi[left:right].sum())
    amp = 6*slope_defect/H
    correction = np.zeros((3, len(e)-1))
    t = (e[left:right]-e[left])/H
    correction[:, left:right] = np.array([
        amp*t*(1-t), amp*(1-2*t)/H, np.full(right-left, -amp/H**2)])
    coeff = old_coeff.copy()
    level, slope = float(old(e[right])), float(old(e[right], 1))
    for i in range(right-1, left-1, -1):
        curvature = ha[:, i].copy()
        curvature[:3] += correction[:, i]
        ascending = np.zeros(7)
        ascending[2:] = curvature/(np.arange(1, 6)*np.arange(2, 7))
        slope -= float(np.sum(curvature*dx[i]**np.arange(1, 6)/np.arange(1, 6)))
        level -= slope*dx[i]+float(np.sum(ascending[2:]*dx[i]**np.arange(2, 7)))
        ascending[:2] = level, slope
        coeff[:, i] = ascending[::-1]
    left_offset = level-float(old(e[left]))
    coeff[-1, :left] += left_offset
    poly = PPoly(coeff, e, extrapolate=False)
    L = np.array(old_log, copy=True)
    L[left+1:right] = poly(e[left+1:right])
    L[:left+1] += left_offset
    B = np.array(old_B, copy=True)
    B[left+1:right] = np.exp(L[left+1:right])
    B[:left+1] *= np.exp(left_offset)
    if not np.all(np.isfinite(B) & (B > 0)):
        raise ArithmeticError("Nonfinite/nonpositive background")
    np.testing.assert_allclose(poly(e), L, rtol=0, atol=1e-10)
    jumps = []
    for nu in range(3):
        p = poly.derivative(nu)
        end = np.polynomial.polynomial.polyval(dx, p.c[::-1], tensor=False)
        jumps.append(float(np.max(abs(end[:-1]-p.c[-1, 1:]))))
    if jumps[0] > 1e-10 or jumps[1] > 1e-8 or jumps[2] > 1e-6:
        raise ArithmeticError("C2 construction check failed")
    copied = ("original_corrected_source_log", "original_aligned_source_log", "alignment_q_log",
              "state", "source_weights", "reference_source_weights", "baseline_reference_background",
              "cubic_anchor_coefficients", "anchor_coordinate_center", "anchor_coordinate_scale")
    arrays = {k: np.array(baseline[k], copy=True) for k in copied}
    arrays["source_weights"] = weights.copy()
    bridge_mask = np.zeros(len(e), bool)
    bridge_mask[left:right+1] = True
    arrays.update(
        energy=e.copy(), background=B, background_log=L, parent_background=old_B.copy(),
        parent_background_log=old_log.copy(), method_name=np.array(FAMILIES[0]),
        parent_source_weights=parent_weights.copy(), effective_source_weights=weights.copy(),
        weight_delta=weights-parent_weights,
        polynomial_knots=e.copy(), integrated_log_polynomial_coefficients=coeff,
        parent_integrated_log_polynomial_coefficients=old_coeff.copy(),
        original_bridge_curvature_coefficients=ha[::-1].copy(),
        closure_curvature_coefficients=correction[::-1],
        bridge_endpoint_indices=np.array([left, right]), local_support_mask=bridge_mask,
        original_bridge_endpoint_indices=np.array([old_left, right]),
        extension_mask=(np.arange(len(e)) >= left) & (np.arange(len(e)) < old_left),
        nominal_SG21_span_ev=np.array(sg21_span),
        changed_background_mask=B != old_B, left_continuation_mask=np.arange(len(e)) <= left,
        background_delta_log=L-old_log, left_log_level_offset=np.array(left_offset),
        stitched_log_first_derivative=poly(e, 1), stitched_log_second_derivative=poly(e, 2),
        seams=np.array([]))
    diag = dict(
        family=FAMILIES[0], status="exploratory_not_evaluated",
        region_indices=[left, right], energy_bounds=e[[left, right]].tolist(), span_ev=H,
        original_region_indices=[old_left, right],
        original_energy_bounds=e[[old_left, right]].tolist(),
        nominal_SG21_span_ev=sg21_span, engineering_span_multiplier=2,
        requested_left_energy=float(requested_left_energy),
        extension_state_unique=np.unique(baseline["state"][left:old_left+1]).tolist(),
        new_curvature_weights="Native quintic 0-to-1 samples, bounded PCHIP interpolation",
        original_endpoint_jets=[[float(old(e[i], k)) for k in range(3)] for i in (left, right)],
        curvature_endpoint_compatibility_error=endpoint_curv_error.tolist(),
        source_curvature_integral=float(hi[left:right].sum()), slope_defect=slope_defect,
        signed_peak_added_log_curvature=amp/4, left_log_level_offset=left_offset,
        left_intensity_ratio=float(np.exp(left_offset)),
        full_delta_log_range=[float((L-old_log).min()), float((L-old_log).max())],
        C0_C1_C2_join_max=jumps, left_slope_closure_error=float(slope-old(e[left], 1)),
        right_values_preserved=True, source_C_A_q_state_frozen=True,
        source_weights_changed=True, parent_weights_saved=True,
        prior_cubic_retained_on_left=True, new_global_anchor=False, BN_used=False,
        sample_used=False, no_new_source_fit=True, left_shape_preserved_up_to_constant=True,
        limitation="The wider bridge may spread the stripe into a broad lobe and change the low-energy denominator scale. Source curvature is not a sample-quality target; full products decide acceptance.")
    return arrays, diag
