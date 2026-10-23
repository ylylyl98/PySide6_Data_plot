"""Current model-risk control and conditional interval selection.

Model risk is a control selector, not a certificate of a clean source.
"""
import numpy as np
from .regions_helpers import runs, smoothstep

HALFWIDTH_EV = .008
COLLAR_EV = .008

def source_selection(e, risk):
    """Deterministic minimum-risk source and nonoverlapping C2 transitions."""
    hard = np.where(risk[0] <= risk[1], 0, 1).astype(np.int8)
    edges = np.flatnonzero(hard[1:] != hard[:-1])
    centers = (e[edges] + e[edges + 1]) / 2
    widths = np.full(len(centers), HALFWIDTH_EV)
    if len(centers):
        widths = np.minimum(widths, np.minimum(centers-e[0], e[-1]-centers))
        if len(centers) > 1:
            widths[1:] = np.minimum(widths[1:], np.diff(centers)/2)
            widths[:-1] = np.minimum(widths[:-1], np.diff(centers)/2)
    supports = np.column_stack((centers-widths, centers+widths))
    wp = (hard == 0).astype(float)
    transition = np.zeros(len(e), bool)
    for j, (left, right) in enumerate(supports):
        mask = (e > left) & (e < right)
        t = (e[mask]-left)/(right-left)
        # Symmetric evaluation of the same quintic avoids near-one cancellation.
        u = np.minimum(t, 1-t)
        s = u**3 * (10 + u*(-15+6*u))
        s = np.where(t <= .5, s, 1-s)
        before = float(hard[edges[j]] == 0)
        after = 1-before
        wp[mask] = before + (after-before)*s
        transition |= mask
    identity = hard.copy()
    identity[transition] = -1
    return dict(source_weights=np.array([wp, 1-wp]),
        v35_hard_source_identity=hard, v35_source_identity=identity,
        v35_transition_mask=transition, v35_switch_energy=centers,
        v35_switch_halfwidth=widths, v35_switch_support=supports,
        v35_switch_native_left_index=edges)


def resolve_regions(e, *, baseline, control, classifier):
    """Apply isolated fixed supports; defer conflicts without changing evidence.

    Deferred cores keep the control selection at this stage. Later renderers
    still construct their own background and report unmet exclusive evidence.
    No file access, fitting, boundary adjustment or implicit conflict winner.
    """
    e = np.asarray(e)
    n = len(e)
    if e.ndim != 1 or n < 21 or not np.all(np.diff(e) > 0):
        raise ValueError("Complete original increasing native axis required")
    for data in (baseline, control, classifier):
        np.testing.assert_array_equal(data["energy"], e)
    states = np.asarray(classifier["v45_state"])
    np.testing.assert_array_equal(states, baseline["state"])
    C, A, q = (np.asarray(baseline[k]) for k in (
        "original_corrected_source_log", "original_aligned_source_log", "alignment_q_log"))
    np.testing.assert_array_equal(A, C+np.array([-.5, .5])[:, None]*q)
    old_weights = np.asarray(control["source_weights"])
    oldw, oldB = old_weights[0], np.asarray(control["background"])
    np.testing.assert_array_equal(old_weights, classifier["v45_control_source_weights"])
    if not np.allclose(np.log(oldB), np.sum(old_weights*A, axis=0), rtol=0, atol=1e-13):
        raise ValueError("V36 background is inconsistent with the frozen aligned sources")
    rows = []
    deferred_rows = []
    uniform = np.linspace(e[0], e[-1], n)
    # The nominal SG21 window, not the global resampling dependency footprint.
    left_window_edges, right_window_edges = uniform[:-20], uniform[20:]
    for state in (1, 2):
        for first, stop in runs(states == state):
            target = float(state == 1)
            if not np.any(oldw[first:stop] != target):
                continue
            a, b = float(e[first]), float(e[stop-1])
            centers = np.flatnonzero((left_window_edges >= a) & (right_window_edges <= b)) + 10
            row = dict(source="P" if state == 1 else "N", state=int(state),
                first_index=int(first), stop_exclusive=int(stop), target_P_weight=target,
                core_bounds_eV=[a, b], support_bounds_eV=[a-COLLAR_EV, b+COLLAR_EV],
                nominal_SG21_complete_window_center_indices=centers.tolist(),
                nominal_SG21_supported=bool(len(centers)),
                new_switch_authorized=bool(len(centers)),
                reasons=[] if len(centers) else ["no_complete_nominal_SG21_window"])
            (rows if len(centers) else deferred_rows).append(row)
    rows.sort(key=lambda row: row["first_index"])
    short_deferred = np.zeros(n, bool)
    for row in deferred_rows:
        short_deferred[row["first_index"]:row["stop_exclusive"]] = True
    # Freeze every requested support before deciding which can be applied.
    # Reject both sides of an overlap, including a candidate already rejected
    # for another reason: its exclusive core must remain at the control value.
    for j, row in enumerate(rows):
        left, right = row["support_bounds_eV"]
        support = (e > left) & (e < right)
        opposite = np.flatnonzero(support & (states == 3-row["state"]))
        row["opposite_exclusive_indices"] = opposite.tolist()
        if opposite.size:
            row["reasons"].append("fixed_collar_crosses_opposite_exclusive_evidence")
        contacted = np.flatnonzero(support & short_deferred)
        row["contacted_deferred_exclusive_indices"] = contacted.tolist()
        if contacted.size:
            row["reasons"].append("support_contacts_deferred_exclusive_evidence")
        for other in rows[j+1:]:
            ol, oh = other["support_bounds_eV"]
            if min(right, oh) > max(left, ol):
                for candidate, neighbor in ((row, other), (other, row)):
                    candidate["reasons"].append("overlapping_requested_supports")
                    candidate.setdefault("overlapping_core_indices", []).append(
                        [neighbor["first_index"], neighbor["stop_exclusive"]])
    authorized = []
    for row in rows:
        row["reasons"] = sorted(set(row["reasons"]))
        row["new_switch_authorized"] = not row["reasons"]
        (authorized if row["new_switch_authorized"] else deferred_rows).append(row)
    rows = authorized
    deferred_rows.sort(key=lambda row: row["first_index"])
    w = oldw.copy()
    core = np.zeros(n, bool)
    profiles = []
    for row in rows:
        a, b = row["core_bounds_eV"]
        left, right = row["support_bounds_eV"]
        phi = np.zeros(n)
        at_core = (e >= a) & (e <= b)
        at_left = (e > left) & (e < a)
        at_right = (e > b) & (e < right)
        phi[at_core] = 1.
        phi[at_left] = smoothstep((e[at_left]-left)/COLLAR_EV)
        phi[at_right] = smoothstep((right-e[at_right])/COLLAR_EV)
        changed = phi > 0
        w[changed] = (1-phi[changed])*oldw[changed]+phi[changed]*row["target_P_weight"]
        core |= at_core
        profiles.append(phi)
    profiles = np.asarray(profiles).reshape(len(rows), n)
    support = np.any(profiles > 0, axis=0) if len(rows) else np.zeros(n, bool)
    collar = support & ~core
    weights = old_weights.copy()
    weights[0, support] = w[support]
    weights[1, support] = 1-w[support]
    if not np.all((weights >= 0) & (weights <= 1)):
        raise ArithmeticError("Local convex combination exceeds source bounds")
    if not np.all(w[core & (states == 1)] == 1) or not np.all(w[core & (states == 2)] == 0):
        raise ArithmeticError("An authorized full core was not retained exactly")
    deferred = np.zeros(n, bool)
    for row in deferred_rows:
        deferred[row["first_index"]:row["stop_exclusive"]] = True
    if np.any(deferred & support):
        raise ArithmeticError("Authorized support changes a deferred source island")
    d = A[0]-A[1]
    delta = (w-oldw)*d
    B = oldB.copy()
    changed = delta != 0
    B[changed] *= np.exp(delta[changed])
    assert np.array_equal(B[~support], oldB[~support])
    assert np.array_equal(weights[:, ~support], old_weights[:, ~support])
    arrays = dict(energy=e.copy(), background=B, source_weights=weights, seams=np.array([], float),
        baseline_background=oldB.copy(), raw_geometric_mean=baseline["raw_geometric_mean"].copy(),
        repair_log_delta=np.log(B/baseline["raw_geometric_mean"]), repair_mask=changed,
        modification_mask=changed, uncertain_mask=states == 0, state=states.copy(),
        source_usable_masks=baseline["source_usable_masks"].copy(),
        hard_lock_mask=core.copy(),
        region_authorization_mask=core.copy(), deferred_exclusive_mask=deferred,
        original_corrected_source_log=C.copy(), original_aligned_source_log=A.copy(),
        alignment_q_log=q.copy(), source_difference_log=d.copy(),
        control_source_weights=old_weights.copy(), background_delta_log=delta,
        local_core_mask=core, local_collar_mask=collar, local_support_mask=support,
        local_bump_profiles=profiles,
        local_run_source=np.array([0 if r["state"] == 1 else 1 for r in rows], np.int8),
        local_run_core_indices=np.array([[r["first_index"], r["stop_exclusive"]] for r in rows], int).reshape(-1, 2),
        local_run_core_bounds=np.array([r["core_bounds_eV"] for r in rows], float).reshape(-1, 2),
        local_run_support_bounds=np.array([r["support_bounds_eV"] for r in rows], float).reshape(-1, 2))
    diag = dict(family="resolved_region_source_correction", runs=rows,
        deferred_runs=deferred_rows, collar_width_eV=COLLAR_EV,
        conflict_policy="defer_all_conflicting_fixed_supports_v1",
        deferred_retention_scope="Control weights and background at this selector stage only; later rendering and optional texture correction have separate rules.",
        nominal_SG21_window_span_eV=float(20*(e[-1]-e[0])/(n-1)),
        all_authorized_cores_exact=True, outside_support_bitwise_control=True,
        source_status="conditional; training sample consistency is not a clean-source certificate",
        skipped_historical_LP=True, skipped_derivative_diagnostics_only=True)
    return arrays, diag


def remove_unknown_excursions(e, weights, state):
    """Retain the adopted same-source/all-unknown rule; do not relabel clean."""
    e=np.asarray(e); state=np.asarray(state); n=len(e)
    parent_weights=np.asarray(weights)
    weights=parent_weights.copy()
    parent_w=parent_weights[0]
    parent_mixed=(parent_w>0)&(parent_w<1)
    removed=np.zeros(n,bool)
    decisions=[]
    parent_supports=[]
    for number,(start,stop) in enumerate(runs(parent_mixed),1):
        if start==0 or stop==n:
            raise ValueError("Parent mixed support has no two pure endpoints")
        left,right=int(start-1),int(stop)
        if parent_w[left] not in (0.,1.) or parent_w[right] not in (0.,1.):
            raise ValueError("Parent endpoints must be pure")
        parent_supports.append([left,right])
        same=bool(parent_w[left]==parent_w[right])
        unknown=bool(np.all(state[start:stop]==0))
        apply=same and unknown
        if apply:
            weights[0,left:right+1]=parent_w[left]
            weights[1,left:right+1]=1-parent_w[left]
            removed[left:right+1]=True
        decisions.append(dict(number=number,row_zero_based=number-1,
            endpoint_indices=[left,right],bounds_eV=e[[left,right]].tolist(),
            endpoint_sources=["P" if parent_w[i]==1 else "N" for i in (left,right)],
            interior_state_counts={str(k):int(np.sum(state[start:stop]==k)) for k in range(4)},
            same_pure_endpoint_source=same,all_interior_unknown=unknown,applied=apply,
            reason=("same pure endpoints and every interior state unknown" if apply else
                    "different endpoint sources" if not same else "interior not entirely unknown")))
    return dict(energy=e.copy(),source_weights=weights,
        parent_source_weights=parent_weights.copy(),excursion_removed_mask=removed,
        weight_changed_mask=np.any(weights!=parent_weights,axis=0),
        parent_support_endpoint_indices=np.asarray(parent_supports,int).reshape(-1,2)), dict(decisions=decisions)
