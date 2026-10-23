# Migrated numerical kernel. Version names describe recipe lineage, not stored inputs.
"""Local curvature increment on an explicit, unchanged V54 parent curve."""
import numpy as np
from scipy.interpolate import PchipInterpolator, PPoly

FAMILIES = ("local_excursion_curvature_closure",)
CONSTRUCTION_TYPE = "V54_parent_plus_compact_two_moment_excursion_curvature_increment"


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def generate(e, *, baseline, parent, symmetric=False):
    """Pure arrays; no new source fit, anchor estimation, or exterior adjustment."""
    e = np.asarray(e)
    if e.ndim != 1 or len(e) < 4 or not np.all(np.diff(e) > 0):
        raise ValueError("Full increasing native energy required")
    np.testing.assert_array_equal(e, baseline["energy"])
    np.testing.assert_array_equal(e, parent["energy"])
    A = np.asarray(baseline["original_aligned_source_log"])
    C = np.asarray(baseline["original_corrected_source_log"])
    q = np.asarray(baseline["alignment_q_log"])
    w = np.asarray(baseline["source_weights"])
    oldw = np.asarray(baseline["parent_source_weights"])
    refw = np.asarray(baseline["reference_source_weights"])
    refB = np.asarray(baseline["baseline_reference_background"])
    state = np.asarray(baseline["state"])
    if A.shape != (2, len(e)) or w.shape != A.shape:
        raise ValueError("Wrong source or weight dimensions")
    np.testing.assert_array_equal(A, C + np.array([-.5, .5])[:, None] * q)
    np.testing.assert_array_equal(oldw, refw)
    np.testing.assert_array_equal(parent["source_weights"], oldw)
    for key in ("original_corrected_source_log", "original_aligned_source_log",
                "alignment_q_log", "state"):
        np.testing.assert_array_equal(parent[key], baseline[key])
    np.testing.assert_array_equal(parent["baseline_background"], refB)
    np.testing.assert_allclose(np.log(refB), np.sum(refw * A, axis=0), rtol=0, atol=2e-13)
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(w)) or not np.all((w >= 0) & (w <= 1)):
        raise ValueError("Invalid source or weight")
    np.testing.assert_allclose(w.sum(0), 1, rtol=0, atol=2e-15)
    changed = np.any(w != oldw, axis=0)
    if np.any(state[changed] != 0) or np.any(changed & ((oldw[0] == 0) | (oldw[0] == 1))):
        raise ValueError("Only the frozen all-unknown mixed excursion may change")
    # Reuse the frozen source interpolant; only the PCHIP selector differs.
    sc = np.asarray(parent["source_log_spline_coefficients"])
    dw = (PchipInterpolator(e, w[0]).c - PchipInterpolator(e, oldw[0]).c)[::-1]
    source_dd_difference = np.stack(
        [2 * (sc[1, :, 0] - sc[1, :, 1]), 6 * (sc[0, :, 0] - sc[0, :, 1])])
    dh = np.zeros((5, len(e) - 1))
    for i in range(4):
        for j in range(2):
            dh[i + j] += dw[i] * source_dd_difference[j]
    active = np.any(dh != 0, axis=0)
    runs = _runs(active)
    if len(runs) > 1 and not symmetric:
        raise ValueError("Unexpected multiple changed-cell supports; no merge/fallback")
    support = np.zeros(len(e), bool)
    interior = np.zeros(len(e), bool)
    compensation = np.zeros_like(dh)
    increment = np.zeros((7, len(e) - 1))  # ascending powers within each native cell
    records = []
    powers = np.arange(5)
    dhpoly = PPoly(dh[::-1], e, extrapolate=False)
    for left, right in runs:
        if left == 0 or right == len(e) - 1:
            raise ValueError("This local experiment requires interior supports")
        endpoint_source_valid = (oldw[0,left] == oldw[0,right] and oldw[0,left] in (0.,1.)) if symmetric else oldw[0,left] == oldw[0,right] == 1
        if not (endpoint_source_valid and np.all(state[left+1:right] == 0)):
            raise ValueError("Support is not the prescribed P-to-N-to-P unknown excursion")
        H = float(e[right] - e[left])
        dx = np.diff(e[left:right+1])
        hs = dh[:, left:right]
        inc1 = np.sum(hs * dx[None] ** (powers[:, None] + 1) / (powers[:, None] + 1), axis=0)
        inc2 = np.sum(hs * dx[None] ** (powers[:, None] + 2)
                      / ((powers[:, None] + 1) * (powers[:, None] + 2)), axis=0)
        raw_slope = np.r_[0., np.cumsum(inc1)]
        raw_log = np.r_[0., np.cumsum(raw_slope[:-1] * dx + inc2)]
        delta0, delta1 = -float(raw_log[-1]), -float(raw_slope[-1])
        a = 60 * delta0 / H**2 - 24 * delta1 / H
        b = 60 * delta1 / H - 120 * delta0 / H**2
        # c(t)=t(1-t)(a+b*t), translated exactly to native cell origins.
        cglobal = np.array([0., a/H, (b-a)/H**2, -b/H**3])
        offsets = e[left:right] - e[left]
        compensation[0, left:right] = np.polynomial.polynomial.polyval(offsets, cglobal)
        compensation[1, left:right] = np.polynomial.polynomial.polyval(offsets, np.arange(1,4)*cglobal[1:])
        compensation[2, left:right] = cglobal[2] + 3*cglobal[3]*offsets
        compensation[3, left:right] = cglobal[3]
        kh = hs + compensation[:, left:right]
        ki1 = np.sum(kh * dx[None] ** (powers[:, None]+1) / (powers[:, None]+1), axis=0)
        ki2 = np.sum(kh * dx[None] ** (powers[:, None]+2)
                     / ((powers[:, None]+1)*(powers[:, None]+2)), axis=0)
        kslope = np.r_[0., np.cumsum(ki1)]
        klog = np.r_[0., np.cumsum(kslope[:-1]*dx+ki2)]
        increment[0, left:right] = klog[:-1]
        increment[1, left:right] = kslope[:-1]
        increment[2:, left:right] = kh / ((powers[:,None]+1)*(powers[:,None]+2))
        if abs(klog[-1]) > 1e-10 or abs(kslope[-1]) > 1e-8:
            raise ArithmeticError("Local two-moment closure failed")
        if max(abs(dhpoly(e[left])), abs(dhpoly(e[right]))) > 1e-6:
            raise ArithmeticError("Changed curvature does not vanish at support endpoints")
        support[left:right+1] = True
        interior[left+1:right] = True
        records.append(dict(left_index=int(left), right_index=int(right),
            energy_bounds=[float(e[left]), float(e[right])], width=H,
            raw_increment_right_log=float(raw_log[-1]), raw_increment_right_slope=float(raw_slope[-1]),
            required_delta0=delta0, required_delta1=delta1,
            compensation_a=float(a), compensation_b=float(b),
            right_log_closure=float(klog[-1]), right_slope_closure=float(kslope[-1]),
            endpoint_source="P" if oldw[0,left]==1 else "N",
            target="Remove a same-source unknown excursion; closure curvature is not a measured source feature."))
    delta_coeff = increment[::-1]
    pc = np.asarray(parent["integrated_log_polynomial_coefficients"])
    if pc.shape != delta_coeff.shape:
        raise ValueError("Expected V54 degree-six native PPoly")
    coeff = pc.copy()
    coeff[:, active] += delta_coeff[:, active]
    poly = PPoly(coeff, e, extrapolate=False)
    parent_poly = PPoly(pc, e, extrapolate=False)
    delta_poly = PPoly(delta_coeff, e, extrapolate=False)
    # Keep stored exterior and endpoint values bitwise; polynomial closure is tolerance-based.
    L = np.asarray(parent["background_log"]).copy()
    B = np.asarray(parent["background"]).copy()
    L[interior] = poly(e[interior])
    B[interior] = np.exp(L[interior])
    if not np.all(np.isfinite(B) & (B > 0)):
        raise ArithmeticError("Nonpositive/nonfinite candidate")
    delta_log = L - parent["background_log"]
    cell = np.gradient(e)
    native_error = float(np.max(abs(poly(e)-L)))
    joins = []
    for nu in range(3):
        pp = poly.derivative(nu)
        at_right = np.polynomial.polynomial.polyval(np.diff(e), pp.c[::-1], tensor=False)
        joins.append(float(np.max(abs(at_right[:-1]-pp.c[-1,1:]))))
    endpoints = [j for r in runs for j in r]
    endpoint_error = [float(np.max(abs(poly(e[endpoints],nu=nu)-parent_poly(e[endpoints],nu=nu))))
                      if endpoints else 0. for nu in range(3)]
    if native_error > 1e-10 or joins[0] > 1e-10 or joins[1] > 1e-8 or joins[2] > 1e-6:
        raise ArithmeticError("Native PPoly/C2 check failed")
    arrays = {k:np.asarray(baseline[k]).copy() for k in (
        "original_corrected_source_log","original_aligned_source_log","alignment_q_log","state",
        "source_usable_masks","hard_lock_mask","raw_geometric_mean","source_difference_log") if k in baseline}
    arrays.update(energy=e.copy(), background=B, background_log=L,
        construction_type=np.array(CONSTRUCTION_TYPE), seams=np.array([],float),
        baseline_background=refB.copy(), baseline_reference_background=refB.copy(),
        parent_background=np.asarray(parent["background"]).copy(),
        parent_background_log=np.asarray(parent["background_log"]).copy(),
        parent_source_weights=oldw.copy(), reference_source_weights=refw.copy(),
        source_weights=w.copy(), effective_source_weights=w.copy(), source_curvature_weights=w.copy(),
        weight_delta=w-oldw, weight_changed_mask=changed, weight_changed_indices=np.flatnonzero(changed),
        changed_weight_states=state[changed].copy(), local_support_mask=support,
        local_interior_mask=interior, changed_curvature_cell_mask=active,
        support_endpoint_indices=np.asarray(runs,dtype=int).reshape(-1,2),
        actual_change_mask=B!=parent["background"], modification_mask=B!=parent["background"],
        repair_mask=B!=parent["background"], uncertain_mask=state==0,
        delta_background_log=delta_log, background_delta_log=L-np.log(refB),
        repair_log_delta=L-np.log(baseline["raw_geometric_mean"]),
        polynomial_knots=e.copy(), integrated_log_polynomial_coefficients=coeff,
        parent_integrated_log_polynomial_coefficients=pc.copy(),
        delta_integrated_log_polynomial_coefficients=delta_coeff,
        delta_selected_curvature_polynomial_coefficients=dh[::-1],
        compensation_curvature_polynomial_coefficients=compensation[::-1],
        delta_closed_curvature_polynomial_coefficients=(dh+compensation)[::-1],
        parent_curvature_polynomial_coefficients=np.asarray(parent["curvature_polynomial_coefficients"]).copy(),
        curvature_polynomial_coefficients=np.asarray(parent["curvature_polynomial_coefficients"])+(dh[::-1]),
        parent_final_curvature_polynomial_coefficients=np.asarray(parent["final_curvature_polynomial_coefficients"]).copy(),
        final_curvature_polynomial_coefficients=np.asarray(parent["final_curvature_polynomial_coefficients"])+(dh+compensation)[::-1],
        source_log_spline_coefficients=sc.copy(), weight_pchip_coefficients=PchipInterpolator(e,w[0]).c.copy(),
        cubic_anchor_coefficients=np.asarray(parent["cubic_anchor_coefficients"]).copy(),
        anchor_coordinate_center=np.asarray(parent["anchor_coordinate_center"]).copy(),
        anchor_coordinate_scale=np.asarray(parent["anchor_coordinate_scale"]).copy(),
        anchor_log_curve=np.asarray(parent["anchor_log_curve"]).copy(),
        anchor_log_curvature=np.asarray(parent["anchor_log_curvature"]).copy(),
        native_cell_weights=cell, stitched_log_first_derivative=poly(e,nu=1),
        stitched_log_second_derivative=poly(e,nu=2),
        delta_log_first_derivative=delta_poly(e,nu=1), delta_log_second_derivative=delta_poly(e,nu=2),
        endpoint_jet_error=np.asarray(endpoint_error))
    diag = dict(family=FAMILIES[0], construction_type=CONSTRUCTION_TYPE,
        status="actual_local_increment_trial_not_accepted", scientific_acceptance=False,
        BN_used=False, new_sample_read=False, no_new_source_fit=True, new_anchor_fit=False,
        inherited_V54_cubic_anchor_frozen=True, source_C_A_q_state_frozen=True,
        source_weights_from_V57=True, original_V50_reference_retained=True,
        support_records=records, support_count=len(records),
        changed_weight_count=int(changed.sum()), actual_changed_background_count=int((B!=parent["background"]).sum()),
        native_PPoly_log_error=native_error, C0_C1_C2_join_max=joins,
        endpoint_parent_jet_error=endpoint_error,
        exterior_background_bitwise_preserved=bool(np.array_equal(B[~interior],parent["background"][~interior])),
        exterior_polynomial_coefficients_preserved=bool(np.array_equal(coeff[:,~active],pc[:,~active])),
        weighted_mean_log_change=float(cell@delta_log/cell.sum()),
        delta_log_RMS=float(np.sqrt(cell@(delta_log**2)/cell.sum())),
        delta_log_range=[float(delta_log.min()),float(delta_log.max())],
        limitations=[
            "The new weights select the raw curvature increment only; compensation is additional artificial curvature.",
            "Exterior V54 values and jets are preserved; the original full-domain mean log anchor is not reimposed.",
            "Local two-moment closure can concentrate a derivative lobe inside the 1.84-eV excursion.",
            "This is conditional on frozen source fits and selection, not a featureless-source certificate.",
            "Numerical closure does not establish full dRR or SG11/21 scientific quality."])
    return arrays, diag
