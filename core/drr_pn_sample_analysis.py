"""Experimental sample-informed source selection, never sample background fitting.

Only the two supplied PN curves provide output intensity/texture. Sample log
bands compare source compatibility and veto unsupported selection changes. A
repeatable stationary sample feature is not thereby a clean reference. This is
not the prepared V66 research procedure or a scientific acceptance test.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from core.drr_auto_composite import AutoCompositeSettings, _dilate, _screen, _select_block
from core.drr_background_structure import _local_quadratic_trend
from core.drr_composite_background import CompositeSource
from core.drr_pn_generator import (
    PnGeneratorSettings, _axis, _curvature_bridges, _jumps, _regions, _runs,
    _settings, _transition_weights, default_settings,
    dependency_paths as _pn_dependency_paths,
)
from core.load_cancellation import check_load_cancelled


ALGORITHM_VERSION = "pn_sample_assisted_curvature_v1"
DEFAULT_SETTINGS = default_settings()


def dependency_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), *_pn_dependency_paths())


def _local_average(e, values, width):
    """Positive native-cell local average on the last axis; no interpolation."""
    cell = np.gradient(e)
    left = np.searchsorted(e, e-width/2)
    right = np.searchsorted(e, e+width/2, side="right")
    sums = np.concatenate((np.zeros((*values.shape[:-1], 1)),
                           np.cumsum(values*cell, axis=-1)), axis=-1)
    denominator = np.r_[0., np.cumsum(cell)]
    return (sums[..., right]-sums[..., left])/(denominator[right]-denominator[left])


def _bands(e, logs, widths):
    flat = logs.reshape(-1, len(e))
    trends = []
    for width in widths:
        check_load_cancelled()
        trends.append(_local_quadratic_trend(e, flat, width).reshape(logs.shape))
    return np.array([trends[0]-trends[1], trends[1]-trends[2]])


def _sample_evidence(e, raw, samples, settings):
    # These fixed engineering scales follow the existing PN screen. They are
    # not fitted noise levels or evidence that a common shape is instrumental.
    widths = (min(.004, settings.widths_ev[0]/4),
              settings.widths_ev[0], settings.widths_ev[-1])
    comparison_width = min(.015, settings.minimum_segment_ev)
    pn_band = _bands(e, np.log(raw), widths)                    # band, source, E
    sample_band = _bands(e, np.log(samples), widths)            # band, repeat, gate, E
    repeats, gates = samples.shape[:2]
    errors = np.empty((repeats, 2, 2, len(e)))                  # repeat, band, source, E
    for r in range(repeats):
        for band in range(2):
            check_load_cancelled()
            for source in range(2):
                squared = (sample_band[band, r]-pn_band[band, source])**2
                # Median of per-gate *costs*, never a median spectrum in B.
                errors[r, band, source] = np.median(
                    np.sqrt(np.maximum(0., _local_average(e, squared, comparison_width))), axis=0)
    repeat_center = sample_band.mean(axis=1, keepdims=True)
    repeat_rms = np.sqrt(np.maximum(0., _local_average(
        e, (sample_band-repeat_center)**2, comparison_width)))
    noise = np.median(repeat_rms, axis=(1, 2))                 # conditional repeat disagreement
    gate_range = np.ptp(sample_band, axis=2)
    floor = settings.feature_threshold/10
    dynamic = (np.min(gate_range, axis=1) > np.maximum(floor, 3*noise)).any(axis=0)
    if gates == 1:
        dynamic[:] = False
    repeat_spread = np.max(np.ptp(sample_band, axis=1), axis=1)
    unstable = (repeat_spread > settings.feature_threshold/4).any(axis=0)
    protected = _dilate(e, dynamic | unstable, max(widths)/2 + comparison_width/2)
    supported = (e >= e[0]+max(widths)/2+comparison_width/2)
    supported &= e <= e[-1]-max(widths)/2-comparison_width/2
    for width in widths:
        count = np.searchsorted(e, e+width/2, side="right")-np.searchsorted(e, e-width/2)
        # The native trend's seven-point fallback must not masquerade as full
        # support at a physical scale too small for this detector grid.
        supported &= count >= 7
    margin = errors[:, :, 1]-errors[:, :, 0]                  # positive favours P
    budget = 3*noise + 128*np.finfo(float).eps
    informative = np.sqrt(_local_average(e, (pn_band[:, 0]-pn_band[:, 1])**2,
                                        comparison_width)) > budget
    p_witness = ((margin >= -budget[None]).all(axis=(0, 1))
                 & ((margin > budget[None]) & informative[None]).any(axis=1).all(axis=0))
    n_witness = ((margin <= budget[None]).all(axis=(0, 1))
                 & ((margin < -budget[None]) & informative[None]).any(axis=1).all(axis=0))
    eligible = supported & ~protected
    if repeats < 2:
        eligible[:] = False
    p_witness &= eligible
    n_witness &= eligible
    if np.any(p_witness & n_witness):
        raise ValueError("Conflicting sample compatibility witnesses.")
    # Only penalize the less compatible source, with bounded influence. PN
    # screening remains independent and retains all suspect/unknown labels.
    advantage = np.median(abs(margin), axis=0).max(axis=0)
    penalty = np.minimum(3., np.log1p((advantage/(settings.feature_threshold/4))**2))
    cost = np.zeros((2, len(e)))
    cost[1, p_witness] = penalty[p_witness]
    cost[0, n_witness] = penalty[n_witness]
    return dict(sample_band_log=sample_band, pn_band_log=pn_band,
        sample_compatibility_rms=errors, sample_compatibility_margin=margin,
        sample_repeat_disagreement=noise, sample_gate_band_range=gate_range,
        sample_dynamic_mask=dynamic, sample_repeat_unstable_mask=unstable,
        sample_protected_mask=protected, sample_supported_mask=supported,
        sample_eligible_mask=eligible, sample_P_witness=p_witness,
        sample_N_witness=n_witness, sample_selection_cost=cost,
        sample_band_widths_ev=np.array(widths),
        sample_comparison_width_ev=np.array(comparison_width))


def _protected_weights(e, selected, locked, halfwidth):
    """Fit collars between protected native points, never through their core.

    Unsupported optional islands can be pruned. If two required cores have no
    intermediate native point, there is no faithful C2 bridge on this grid and
    generation fails explicitly instead of overriding a protected source.
    """
    selected = selected.copy()
    selected[locked >= 0] = locked[locked >= 0]
    pruned = []
    for _ in range(len(e)):
        cross = np.flatnonzero(selected[1:] != selected[:-1])+1
        centers = (e[cross-1]+e[cross])/2
        supports = []
        failed = None
        for number, (index, center) in enumerate(zip(cross, centers)):
            check_load_cancelled()
            lo = e[0] if number == 0 else (centers[number-1]+center)/2
            hi = e[-1] if number+1 == len(centers) else (center+centers[number+1])/2
            left = max(0, int(np.searchsorted(e, max(lo, center-halfwidth))))
            right = min(len(e)-1, int(np.searchsorted(e, min(hi, center+halfwidth), side="right")-1))
            lp = np.flatnonzero(locked[left:index] >= 0)+left
            rp = np.flatnonzero(locked[index:right+1] >= 0)+index
            if len(lp):
                left = int(lp[-1])
            if len(rp):
                right = int(rp[0])
            if left >= index or right < index or right-left < 2:
                failed = number
                break
            supports.append((left, right))
        if failed is None:
            break
        edges = np.r_[0, cross, len(e)]
        candidates = []
        for segment in (failed, failed+1):
            a, b = edges[segment:segment+2]
            if not np.any(locked[a:b] >= 0):
                candidates.append((e[b-1]-e[a], int(a), int(b)))
        if not candidates:
            raise ValueError("Unresolved protected source cores: no smooth native transition fits between them; no background was generated.")
        _, a, b = min(candidates)
        replacement = selected[a-1] if a else selected[b]
        selected[a:b] = replacement
        pruned.append(dict(indices=[a, b-1], reason="Optional source island cannot accommodate a protected smooth transition."))
    else:
        raise ValueError("Protected source transition construction did not terminate.")
    weights = (selected == 0).astype(float)
    for (left, right), index in zip(supports, cross):
        t = (e[left:right+1]-e[left])/(e[right]-e[left])
        phi = t**3*(10+t*(-15+6*t))
        weights[left:right+1] = phi if selected[index] == 0 else 1-phi
    if np.any(weights[locked == 0] != 1) or np.any(weights[locked == 1] != 0):
        raise ValueError("A transition would cross a protected source core.")
    if any(a1 < b0 for (_, b0), (a1, _) in zip(supports, supports[1:])):
        raise ValueError("Protected source transitions overlap.")
    return weights, supports, selected, pruned


def generate_sample_assisted_arrays(energy, p, n, samples, gate, *, settings=None):
    """Return (native arrays, draft report), with no IO or input mutation.

    ``samples`` is (repeat, gate, energy), all on the supplied native axis.
    Cancellation uses core.load_cancellation.load_scope, as the PN array API.
    Sample arrays may contain one repeat/gate; insufficient independent repeats
    disable sample-driven selection and protection. The PN-only core-safety
    renderer can still differ from the older unprotected PN-only comparator.
    Gate values must be finite and distinct;
    gate order and repeat order have no privileged reference gate.
    """
    check_load_cancelled()
    e = _axis(energy)
    settings = _settings(settings)
    if settings.minimum_segment_ev > np.ptp(e) or 2*settings.transition_halfwidth_ev >= np.ptp(e):
        raise ValueError("Segment/transition settings must fit within the actual energy span.")
    raw = np.array([p, n], float, copy=True)
    if raw.shape != (2, len(e)) or not np.all(np.isfinite(raw) & (raw > 0)):
        raise ValueError("P and N must be finite, positive native energy vectors.")
    sample = np.array(samples, float, copy=True)
    gates = np.array(gate, float, copy=True)
    if (sample.ndim != 3 or sample.shape[-1] != len(e) or min(sample.shape[:2]) < 1
            or not np.all(np.isfinite(sample) & (sample > 0))):
        raise ValueError("samples must be finite positive intensities with shape (repeats, gates, energy).")
    if (gates.ndim != 1 or len(gates) != sample.shape[1] or not np.isfinite(gates).all()
            or len(np.unique(gates)) != len(gates)):
        raise ValueError("gate must contain one distinct finite value per sample gate.")
    columns = tuple(format(x, ".17g") for x in e)
    sources = tuple(CompositeSource("PN"[i], "PN"[i], (), "all", e, columns, raw[i]) for i in range(2))
    screen = _screen(e, sources, AutoCompositeSettings(
        feature_threshold=settings.feature_threshold, minimum_segment_ev=settings.minimum_segment_ev,
        widths_ev=settings.widths_ev), ())
    pn_selected = _select_block(e, raw, screen["cost"], screen["valid"], settings.minimum_segment_ev, 0)
    pn_w, pn_supports = _transition_weights(e, pn_selected, settings.transition_halfwidth_ev)
    pn_poly, _, _, _, _ = _curvature_bridges(e, np.log(raw), pn_w, pn_supports)
    evidence = _sample_evidence(e, raw, sample, settings)
    exclusive = np.array([screen["supported"].all(0) & ~screen["suspect"][s]
                          & screen["suspect"][1-s] for s in range(2)])
    pn_locked = np.full(len(e), -1, int)
    pn_locked[exclusive[0]], pn_locked[exclusive[1]] = 0, 1
    safe_w, safe_supports, safe_selected, _ = _protected_weights(
        e, pn_selected, pn_locked, settings.transition_halfwidth_ev)
    safe_poly, _, _, _, _ = _curvature_bridges(e, np.log(raw), safe_w, safe_supports)
    locked = np.full(len(e), -1, int)
    # Preserve PN-only pure-source choice where a moving or unstable sample
    # shape could otherwise drive selection. Old mixed regions remain eligible
    # only for bridge geometry, never sample-derived costs.
    protected = evidence["sample_protected_mask"]
    if sample.shape[0] >= 2:
        locked[protected & (safe_w == 1)] = 0
        locked[protected & (safe_w == 0)] = 1
    locked[exclusive[0]], locked[exclusive[1]] = 0, 1
    eligible = screen["valid"].copy()
    eligible[1, locked == 0] = False
    eligible[0, locked == 1] = False
    cost = screen["cost"] + evidence["sample_selection_cost"]
    # A content-based tie convention avoids assigning physical priority to the
    # letter P on sample-informed selection. Identical sources are equivalent.
    if sample.shape[0] < 2 or (not evidence["sample_selection_cost"].any()
                             and np.array_equal(locked, pn_locked)):
        w, supports, selected, pruned = safe_w, safe_supports, safe_selected, []
    else:
        total = cost @ np.gradient(e)
        default = int(total[1] < total[0]) if evidence["sample_selection_cost"].any() else 0
        selected = _select_block(e, raw, cost, eligible, settings.minimum_segment_ev, default)
        w, supports, selected, pruned = _protected_weights(e, selected, locked, settings.transition_halfwidth_ev)
    poly, source, bridges, anchor, reference = _curvature_bridges(e, np.log(raw), w, supports)
    L = poly(e)
    with np.errstate(over="ignore", under="ignore"):
        background = np.exp(L)
    if not np.all(np.isfinite(background) & (background > 0)):
        raise ValueError("Source levels produce nonfinite/zero generated intensity; no clipped result is returned.")
    uncertain = ~screen["supported"].all(0) | screen["suspect"].all(0) | protected
    if sample.shape[0] < 2:
        uncertain[:] = True
    records = []
    edges = np.r_[e[0], (e[:-1]+e[1:])/2, e[-1]]
    for s in range(2):
        for a, b in _runs(w == (1 if s == 0 else 0)):
            shift = L[a:b]-np.log(raw[s, a:b])
            records.append(dict(source_key="PN"[s], indices=[int(a), int(b-1)],
                xmin_ev=float(edges[a]), xmax_ev=float(edges[b]), log_scale=float(shift.mean()),
                intensity_scale=float(np.exp(shift.mean())), shape_residual_max=float(np.max(abs(shift-shift.mean())))))
    records.sort(key=lambda row: row["xmin_ev"])
    arrays = dict(energy=e, background=background, background_log=L, raw_sources=raw,
        source_weights=np.array([w, 1-w]), hard_source_identity=selected,
        screen_score=screen["score"].copy(), screen_supported=screen["supported"].copy(),
        source_suspect_mask=screen["suspect"].copy(), uncertain_mask=uncertain,
        transition_endpoint_indices=np.asarray(supports, int).reshape(-1, 2),
        polynomial_knots=e.copy(), background_log_polynomial_coefficients=poly.c.copy(),
        source_log_polynomial_coefficients=source.c.copy(), reference_log=reference,
        global_log_anchor=np.array(anchor), pn_only_background=np.exp(pn_poly(e)),
        pn_only_source_weights=np.array([pn_w, 1-pn_w]), pn_only_hard_source_identity=pn_selected,
        pn_safety_only_background=np.exp(safe_poly(e)),
        pn_safety_only_source_weights=np.array([safe_w, 1-safe_w]),
        pn_safety_only_hard_source_identity=safe_selected,
        source_exclusive_core_mask=exclusive, protected_source_identity=locked,
        sample_gate=gates, combined_selection_cost=cost, **evidence)
    changed = w != pn_w
    assistance_changed = w != safe_w
    fallback = ("single_repeat_PN_core_safety_only" if sample.shape[0] < 2 else
                "no_fully_supported_sample_band" if not evidence["sample_supported_mask"].any() else
                "no_consistent_matching_witness_protection_only" if not evidence["sample_selection_cost"].any() else None)
    report = dict(schema_version=1, report_kind="pn_generated_draft", algorithm=ALGORITHM_VERSION,
        algorithm_version=ALGORITHM_VERSION, method=ALGORITHM_VERSION, status="draft_requires_preview",
        scientifically_validated=False, v64_equivalent=False, v66_equivalent=False, BN_used=False,
        sample_used=True, samples_analyzed=True, sample_intensity_in_background=False, new_source_fit=False,
        sample_matching_cost_applied=bool(evidence["sample_selection_cost"].any()),
        sample_protection_applied=bool(sample.shape[0] >= 2 and np.any((locked >= 0) & (pn_locked < 0))),
        sample_selection_influenced=bool(assistance_changed.any()), assistance_applied=bool(assistance_changed.any()),
        assistance_semantics="Weight change relative to PN-only core-safety construction; may come from protection or matching, not proof of improvement.",
        fallback_reason=fallback,
        fitted_global_log_anchor=True, settings={**asdict(settings), "widths_ev": list(settings.widths_ev)},
        input_source_keys=["P", "N"], points=len(e), resampled=False,
        sample_repeats=sample.shape[0], sample_gates=sample.shape[1],
        sample_selection_enabled=sample.shape[0] >= 2,
        degradation_reasons=(["Single repeat: sample matching/protection disabled; only PN core-safety construction is used."]
                             if sample.shape[0] < 2 else []) +
                            (["Single gate: no evidence about moving structure across gates."] if sample.shape[1] < 2 else []) +
                            (["No fully supported multiscale sample comparison points."] if not evidence["sample_supported_mask"].any() else []),
        source_segments=records, transitions=bridges, pruned_optional_segments=pruned,
        unsupported_regions=_regions(e, ~screen["supported"].all(0)), unresolved_regions=_regions(e, uncertain),
        sample_protected_regions=_regions(e, protected), global_log_anchor=anchor,
        C2_join_error=_jumps(poly), protected_core_weights_exact=True,
        source_weights_semantics="Curvature construction weights; not pointwise intensity mixture or cleanliness probability.",
        pn_only_comparison=dict(algorithm_version="pn_regional_curvature_v1", changed_weight_points=int(changed.sum()),
            changed_regions=_regions(e, changed), max_absolute_log_background_change=float(np.max(abs(L-pn_poly(e))))),
        pn_core_safety_comparison=dict(geometry_only_changed_points=int(np.count_nonzero(safe_w != pn_w)),
            sample_assistance_changed_points=int(assistance_changed.sum()),
            max_absolute_log_background_change=float(np.max(abs(L-safe_poly(e))))),
        compatibility_rule=dict(band_widths_ev=evidence["sample_band_widths_ev"].tolist(),
            comparison_width_ev=float(evidence["sample_comparison_width_ev"]),
            dynamic_log_floor=settings.feature_threshold/10, repeat_instability_log_floor=settings.feature_threshold/4,
            maximum_cost_penalty=3., threshold_status="Fixed engineering heuristics, not calibrated physical/noise acceptance limits."),
        level_drift=dict(max_absolute_pure_log_scale=max((abs(row["log_scale"]) for row in records), default=0.),
            reference_log_difference_range=[float(np.min(L-reference)), float(np.max(L-reference))]),
        limitations=[
            "Experimental raw-array source selection, not the V64/V66 prepared research algorithm or a validated background.",
            "Matching stationary texture or a low PN screen score is not proof of a featureless source; shared peaks stay unresolved.",
            "Sample band comparisons affect source choice only; no gate median or fitted sample signal is subtracted from the output.",
            "Moving/repeat-inconsistent regions freeze PN-only pure choices, not the unknown physical background; finite SG windows can still be affected.",
            "Exclusive PN cores are heuristic protection, not physical cleanliness certification; impossible smooth core transitions fail explicitly.",
            "Pure segments retain source shape only up to a regional constant; closure and the global PN anchor can change levels elsewhere.",
            "C2 continuity and compatibility costs do not establish dRR or d2 quality; preview both complete products before use."])
    check_load_cancelled()
    return arrays, report
