"""Experimental training-only comparison of parent and actual local P/N edits.

The normalization-transport score is a numerical proxy, not a physical feature
truth test. This module performs no file I/O, changes no default, and returns
individual proposals only. A caller must check their final combined product.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json

import numpy as np

from core.processing import _sg_derivative_energy_rows
from .feature_protection import apply_feature_protection


COMPARISON_PROTOCOL = {
    "version": "actual_local_parent_pn_transport_comparison_v1_1",
    "score_method": "RMS(D2(C/B)-D2(C)/B)/RMS(D2(C)/B), C=S-mean_gate(S); full native-axis production SG21/O2 before domain masking",
    "rank": "Parent, actual locally constructed P, actual locally constructed N; raw P/N scores are references only",
    "domains": "feature core and full constructor curvature support; combined ranking score is their arithmetic mean",
    "improvement_margin": "max(abs(delta_repeat0-delta_repeat1)/2,64*float64_eps*max(1,abs(parent_scores),abs(candidate_scores))); delta=parent-candidate; a repeat-spread tolerance, not a statistical confidence interval",
    "acceptance": "Same strict best actual source in both repeats; improvement over parent exceeds its repeat-spread margin in both repeats and both domains; improvement over the other valid source exceeds that margin for the combined ranking score",
    "maximum_transport": 0.25,
    "minimum_pattern_cosine": 0.9,
    "pattern_checks": "Both domains: raw P/N and actual P/N cross-source patterns where available; chosen/parent cross-background patterns per repeat; chosen cross-repeat patterns",
    "stationary": "Retain parent; full requested source support must not touch any ambiguous_stationary core or anchor interval",
    "diagnostic_construction": "Construct every geometrically available local P/N candidate, including stationary or conflicted features; numeric validity does not permit adoption",
    "diagnostic_ranking": "Per-repeat minimum mean(core,full_support) transport among finite valid actual Parent/P/N candidates; exact ties or undefined scores have no unique best; this is not a suggestion or physical truth",
    "stationary_chord_method": "Full native-axis production SG21/O2 D2(mean_gate(S)/B-1); signed center minus linearly interpolated fixed left/right anchors, each training repeat separately; nearest native points, no fitting or peak maximization",
    "composition": "No feature merging, no overlap resolution, no automatic application; caller must check final combined regions, stationary features, boundaries, and whole-axis changes",
    "scientifically_validated": False,
    "production_default_allowed": False,
}


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("Feature background comparison cancelled")


def _constructor_for_mode(mode):
    if mode == "integrated_curvature":
        return apply_feature_protection
    if mode == "compact_bounded":
        from .compact_protection import apply_compact_protection
        return apply_compact_protection
    raise ValueError("Unknown local construction mode")


def _inputs(energy, training_samples, cancelled):
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable or None")
    _check(cancelled)
    e, samples = np.asarray(energy, float), np.asarray(training_samples, float)
    if e.ndim != 1 or len(e) < 41 or not np.isfinite(e).all() or np.any(np.diff(e) <= 0):
        raise ValueError("Finite increasing energy with at least 41 points required")
    if samples.ndim != 3 or samples.shape[0] != 2 or samples.shape[1] < 3 or samples.shape[2] != len(e):
        raise ValueError("Exactly two training repeats with at least three gates required")
    if not np.isfinite(samples).all() or np.any(samples <= 0):
        raise ValueError("Training intensities must be positive and finite")
    return e, samples


def _background(background, n):
    values = np.asarray(background, float)
    if values.shape != (n,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Background must be positive finite native intensity")
    return values


def _bounds(bounds, e, label):
    if not isinstance(bounds, (list, tuple, np.ndarray)) or len(bounds) != 2:
        raise ValueError(f"{label} needs two finite increasing energies")
    try:
        values = [float(x) for x in bounds]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {label}") from exc
    if (any(isinstance(x, (bool, np.bool_)) for x in bounds)
            or not np.isfinite(values).all() or not values[0] < values[1]
            or values[0] < e[0] or values[1] > e[-1]):
        raise ValueError(f"{label} must lie inside the measured axis")
    if not np.any((e >= values[0]) & (e <= values[1])):
        raise ValueError(f"{label} contains no native energy point")
    return values


def _d2(values, e, cancelled):
    _check(cancelled)
    shape = values.shape
    result = _sg_derivative_energy_rows(values.reshape(-1, len(e)), e, 21, 2, 2).reshape(shape)
    _check(cancelled)
    return result


def _field(centered, raw_d2, background, e, cancelled):
    dynamic = _d2(centered/background, e, cancelled)
    intrinsic = raw_d2/background
    return dynamic, intrinsic


def _scores(field, mask):
    dynamic, intrinsic = field
    expected = intrinsic[..., mask]
    numerator = np.sqrt(np.mean((dynamic[..., mask]-expected)**2, axis=(1, 2)))
    denominator = np.sqrt(np.mean(expected**2, axis=(1, 2)))
    result = np.full(2, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def score_dynamic_background(energy, training_samples, background, bounds, *, cancelled=None):
    """Return the two training proxy scores after full-axis SG21 processing."""
    e, samples = _inputs(energy, training_samples, cancelled)
    bg = _background(background, len(e))
    lo, hi = _bounds(bounds, e, "Score bounds")
    centered = samples-samples.mean(axis=1, keepdims=True)
    raw_d2 = _d2(centered, e, cancelled)
    return _scores(_field(centered, raw_d2, bg, e, cancelled), (e >= lo) & (e <= hi))


def improvement_margin(scores_parent, scores_candidate):
    """Deterministic two-repeat spread tolerance; this is not a confidence CI."""
    parent, candidate = np.asarray(scores_parent, float), np.asarray(scores_candidate, float)
    if parent.shape != (2,) or candidate.shape != (2,) or not np.isfinite([parent, candidate]).all():
        raise ValueError("Two finite scores per method required for a repeat-spread margin")
    delta = parent-candidate
    floor = 64*np.finfo(float).eps*max(1., float(np.max(abs(parent))), float(np.max(abs(candidate))))
    return max(float(abs(delta[0]-delta[1])/2), floor)


def _cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip(np.dot(a, b)/denominator, -1., 1.)) if denominator > 0 else None


def _plain_scores(scores):
    return [float(x) if np.isfinite(x) else None for x in scores]


def _chord_geometry(row, e):
    a, center, b = [int(np.argmin(abs(e-value))) for value in
                    (row["anchor_bounds_eV"][0], row["center_ev"], row["anchor_bounds_eV"][1])]
    valid = a < center < b
    return dict(indices=[a, center, b], native_anchor_bounds_eV=[float(e[a]), float(e[b])],
                native_center_ev=float(e[center]), valid=valid,
                reason=None if valid else "native_anchors_do_not_strictly_surround_center")


def _signed_chord(stationary, geometry, e):
    if not geometry["valid"]:
        return np.full(2, np.nan)
    a, center, b = geometry["indices"]
    fraction = (e[center]-e[a])/(e[b]-e[a])
    return stationary[:, center]-(1-fraction)*stationary[:, a]-fraction*stationary[:, b]


def _numerical_ranking(actual, valid, names):
    composite = actual.mean(axis=2)
    winners, ties = [], []
    for repeat in range(2):
        available = [method for method in range(3) if valid[method] and np.isfinite(composite[method, repeat])]
        best = min((composite[method, repeat] for method in available), default=np.inf)
        tied = [names[method] for method in available if composite[method, repeat] == best]
        ties.append(tied if len(tied) > 1 else [])
        winners.append(tied[0] if len(tied) == 1 else None)
    return winners, bool(winners[0] is not None and winners[0] == winners[1]), ties


def _feature_rows(features, e):
    if not isinstance(features, (list, tuple)):
        raise ValueError("Features must be a list of detector records")
    rows, ids = [], set()
    for feature in features:
        if not isinstance(feature, Mapping):
            raise ValueError("Each feature must be a detector record")
        identity = feature.get("id")
        if isinstance(identity, bool) or not isinstance(identity, (str, int)) or str(identity) in ids:
            raise ValueError("Distinct string or integer feature IDs required")
        ids.add(str(identity))
        if feature.get("branch") not in ("dynamic", "stationary"):
            raise ValueError("Feature branch must be dynamic or stationary")
        bounds = _bounds(feature.get("bounds_eV"), e, "Feature bounds")
        core = _bounds(feature.get("core_bounds_eV", bounds), e, "Feature core")
        anchors = _bounds(feature.get("anchor_bounds_eV", bounds), e, "Feature anchors")
        if core[0] < bounds[0] or core[1] > bounds[1]:
            raise ValueError("Feature core must be within feature bounds")
        center = float(feature.get("center_ev", (core[0]+core[1])/2))
        if not np.isfinite(center) or not bounds[0] <= center <= bounds[1]:
            raise ValueError("Feature center must lie within feature bounds")
        rows.append(dict(id=identity, branch=feature["branch"], bounds_eV=bounds,
                         core_bounds_eV=core, anchor_bounds_eV=anchors, center_ev=center,
                         detector_status=str(feature.get("status", "not_recorded"))))
    return rows


def compare_feature_backgrounds(energy, training_samples, pn_mean, parent_background, features, *,
                               construction_mode="integrated_curvature", cancelled=None):
    """Compare actual individual edits without applying or merging proposals.

    Candidate arrays preserve the input feature order and IDs. Invalid P/N
    candidates contain a parent placeholder and have ``valid=False``; their
    scores are NaN in arrays and null in JSON. Parent is a legitimate choice.
    The opt-in compact constructor can produce a diagnostic candidate whose
    curvature fit is rejected. Such candidates stay visible but cannot be
    proposed for adoption. The default constructor is unchanged.
    """
    e, samples = _inputs(energy, training_samples, cancelled)
    constructor = _constructor_for_mode(construction_mode)
    parent = _background(parent_background, len(e))
    pn = np.asarray(pn_mean, float)
    if pn.shape != (2, len(e)) or not np.isfinite(pn).all() or np.any(pn <= 0):
        raise ValueError("P/N means must be positive finite arrays with shape (2, energy)")
    rows = _feature_rows(features, e)
    count = len(rows)
    backgrounds = np.broadcast_to(parent, (count, 3, len(e))).copy()
    valid = np.zeros((count, 3), bool)
    valid[:, 0] = True
    accepted_fit = valid.copy()
    score_array = np.full((count, 5, 2, 2), np.nan)
    stationary_chords = np.full((count, 5, 2), np.nan)
    core_masks = np.zeros((count, len(e)), bool)
    support_masks = np.zeros_like(core_masks)
    span = float(20*(e[-1]-e[0])/(len(e)-1))
    mean_sample = samples.mean(axis=1)
    centered = samples-mean_sample[:, None]
    raw_d2 = _d2(centered, e, cancelled)
    parent_field = _field(centered, raw_d2, parent, e, cancelled)
    raw_fields = [_field(centered, raw_d2, bg, e, cancelled) for bg in pn]
    stationary_references = [_d2(mean_sample/bg-1, e, cancelled) for bg in (parent, *pn)]
    cache, comparisons, suggested, suggested_ids = {}, [], [], []
    constructor_calls = 0
    names = ("Parent", "P", "N")
    for index, row in enumerate(rows):
        _check(cancelled)
        lo, hi = row["bounds_eV"]
        support = [lo-2*span, hi+2*span]
        core_mask = (e >= row["core_bounds_eV"][0]) & (e <= row["core_bounds_eV"][1])
        support_mask = (e >= support[0]) & (e <= support[1])
        core_masks[index], support_masks[index] = core_mask, support_mask
        masks = (core_mask, support_mask)
        chord_geometry = _chord_geometry(row, e)
        for method, field in ((0, parent_field), (3, raw_fields[0]), (4, raw_fields[1])):
            for domain, mask in enumerate(masks):
                score_array[index, method, :, domain] = _scores(field, mask)
        for method, stationary in zip((0, 3, 4), stationary_references):
            stationary_chords[index, method] = _signed_chord(stationary, chord_geometry, e)
        conflicts = []
        for other in rows:
            if other["detector_status"] != "ambiguous_stationary":
                continue
            for key in ("core_bounds_eV", "anchor_bounds_eV"):
                a, b = other[key]
                if a <= support[1] and b >= support[0]:
                    conflicts.append(dict(feature_id=other["id"], interval=key, bounds_eV=[a, b]))
        construction_blockers, adoption_blockers = [], []
        if row["branch"] != "dynamic":
            adoption_blockers.append("stationary_source_not_identified")
        if support[0] <= e[0] or support[1] >= e[-1]:
            construction_blockers.append("insufficient_full_constructor_support")
        if conflicts:
            adoption_blockers.append("full_support_intersects_ambiguous_stationary")
        blocked = construction_blockers+adoption_blockers
        fields = [parent_field, None, None]
        candidates = []
        for method, name in enumerate(names):
            failures = list(construction_blockers) if method else []
            detail = None
            if method and not failures:
                key = (lo, hi, name)
                if key not in cache:
                    try:
                        output, _, detail = constructor(e, parent, pn,
                            [dict(xmin_ev=lo, xmax_ev=hi, source=name)], cancelled=cancelled)
                        output = _background(output, len(e))
                        actual_support = detail["regions"][0]["support_bounds_eV"]
                        if not np.allclose(actual_support, support, rtol=0, atol=4*np.finfo(float).eps*max(1., abs(lo), abs(hi))):
                            raise ValueError("Constructor support differs from declared comparison geometry")
                        if construction_mode == "compact_bounded" and not isinstance(detail.get("fit_accepted"), bool):
                            raise ValueError("Compact constructor did not declare fit acceptance")
                        # Cache the one-dimensional construction, not its full
                        # repeat/gate derivative maps for every local feature.
                        cache[key] = (output.copy(), detail, None)
                    except (ValueError, ArithmeticError) as exc:
                        cache[key] = (None, None, str(exc))
                    constructor_calls += 1
                output, detail, error = cache[key]
                if not error:
                    try:
                        field = _field(centered, raw_d2, output, e, cancelled)
                        stationary = _d2(mean_sample/output-1, e, cancelled)
                    except (ValueError, ArithmeticError) as exc:
                        error = str(exc)
                if error:
                    failures.append("construction_failed: "+error)
                else:
                    backgrounds[index, method] = output
                    valid[index, method] = True
                    accepted_fit[index, method] = (detail["fit_accepted"]
                        if construction_mode == "compact_bounded" else True)
                    fields[method] = field
                    stationary_chords[index, method] = _signed_chord(stationary, chord_geometry, e)
                    for domain, mask in enumerate(masks):
                        score_array[index, method, :, domain] = _scores(field, mask)
            scores = score_array[index, method]
            candidates.append(dict(source=name, valid=bool(valid[index, method]),
                constructed=bool(valid[index, method]), fit_accepted=bool(accepted_fit[index, method]),
                fit_rejection_reasons=(deepcopy(detail.get("fit_rejection_reasons", [])) if detail else []),
                status="evaluated" if valid[index, method] else "not_constructed",
                reasons=failures, training_scores_core=_plain_scores(scores[:, 0]),
                training_scores_full_support=_plain_scores(scores[:, 1]),
                training_combined_scores=_plain_scores(scores.mean(axis=1)),
                training_stationary_signed_chords=_plain_scores(stationary_chords[index, method]),
                constructor_report=deepcopy(detail)))
        reasons, chosen, pattern_checks = list(blocked), "Parent", []
        actual = score_array[index, :3]
        numerical_best, numerical_consistent, numerical_ties = _numerical_ranking(actual, valid[index], names)
        eligible = [method for method in (1, 2) if valid[index, method] and np.isfinite(actual[method]).all()]
        if not blocked and np.isfinite(actual[0]).all():
            composite = actual.mean(axis=2)
            ranks = np.where(np.isfinite(composite), composite, np.inf)
            winners = np.argmin(ranks, axis=0)
            winner = int(winners[0])
            if winner == 0 or not np.all(winners == winner):
                reasons.append("parent_best_or_repeat_preferences_disagree")
            elif winner not in eligible:
                reasons.append("constructed_candidate_scores_undefined")
            elif not accepted_fit[index, winner]:
                reasons.append("numerical_best_candidate_curvature_fit_rejected")
            else:
                failures = []
                margins = []
                for domain in range(2):
                    margin = improvement_margin(actual[0, :, domain], actual[winner, :, domain])
                    margins.append(margin)
                    if not np.all(actual[0, :, domain]-actual[winner, :, domain] > margin):
                        failures.append(("core" if domain == 0 else "full_support")+"_does_not_improve_parent_beyond_repeat_spread")
                other = 3-winner
                if other in eligible:
                    margin = improvement_margin(composite[other], composite[winner])
                    if not np.all(composite[other]-composite[winner] > margin):
                        failures.append("source_preference_is_tied_within_repeat_spread")
                if not np.all(actual[winner] <= COMPARISON_PROTOCOL["maximum_transport"]):
                    failures.append("constructed_transport_exceeds_limit")
                for domain, mask in zip(("core", "full_support"), masks):
                    for repeat in range(2):
                        checks = [("raw_P_N", raw_fields[0][0][repeat], raw_fields[1][0][repeat]),
                                  ("chosen_parent", fields[winner][0][repeat], parent_field[0][repeat])]
                        if fields[1] is not None and fields[2] is not None:
                            checks.append(("constructed_P_N", fields[1][0][repeat], fields[2][0][repeat]))
                        for kind, a, b in checks:
                            pattern_checks.append(dict(domain=domain, comparison=kind, repeat=repeat,
                                                       cosine=_cosine(a[..., mask], b[..., mask])))
                    pattern_checks.append(dict(domain=domain, comparison="chosen_repeats", repeat=None,
                        cosine=_cosine(fields[winner][0][0, :, mask], fields[winner][0][1, :, mask])))
                if any(item["cosine"] is None or item["cosine"] < COMPARISON_PROTOCOL["minimum_pattern_cosine"] for item in pattern_checks):
                    failures.append("dynamic_pattern_agreement_below_limit")
                candidates[winner]["parent_improvement_margin_core_full_support"] = margins
                candidates[winner]["selection_gate_failures"] = failures
                if failures:
                    reasons.extend(failures)
                else:
                    chosen = names[winner]
                    reasons.append("actual_local_candidate_improves_parent_in_both_repeats_and_domains")
                    suggested.append(dict(xmin_ev=lo, xmax_ev=hi, source=chosen))
                    suggested_ids.append(row["id"])
        elif not blocked:
            reasons.append("parent_transport_undefined")
        comparisons.append({**deepcopy(row), "full_support_bounds_eV": support,
            "score_method": COMPARISON_PROTOCOL["score_method"], "candidates": candidates,
            "raw_reference_scores": {name: dict(training_scores_core=_plain_scores(score_array[index, j, :, 0]),
                training_scores_full_support=_plain_scores(score_array[index, j, :, 1]),
                training_stationary_signed_chords=_plain_scores(stationary_chords[index, j])) for name, j in (("P", 3), ("N", 4))},
            "stationary_chord_geometry": chord_geometry,
            "stationary_chord_method": COMPARISON_PROTOCOL["stationary_chord_method"],
            "construction_blockers": construction_blockers, "adoption_blockers": adoption_blockers,
            "fit_rejected_sources": [names[j] for j in (1, 2) if valid[index, j] and not accepted_fit[index, j]],
            "numerical_best_source_by_repeat": numerical_best,
            "numerical_best_source_consistent": numerical_consistent,
            "numerical_tied_sources_by_repeat": numerical_ties,
            "stationary_conflicts": conflicts, "pattern_checks": pattern_checks,
            "status": "retain_parent" if chosen == "Parent" else "local_proposal_unvalidated",
            "suggested_source": chosen, "retain_parent": chosen == "Parent", "reasons": reasons,
            "reason": "; ".join(reasons), "scientifically_validated": False})
    ids = np.asarray([row["id"] for row in rows], dtype=np.int64 if all(isinstance(row["id"], int) for row in rows) else str)
    arrays = dict(energy=e.copy(), candidate_local_ids=ids, local_candidate_backgrounds=backgrounds,
                  local_candidate_valid=valid, transport_scores=score_array,
                  local_candidate_constructed=valid.copy(), local_candidate_accepted_fit=accepted_fit,
                  stationary_signed_chords=stationary_chords,
                  core_masks=core_masks, full_support_masks=support_masks)
    protocol = {**deepcopy(COMPARISON_PROTOCOL), "construction_mode": construction_mode,
                "fit_acceptance": "Constructed compact candidates remain diagnostics when their source-curvature fit is rejected; rejected fits cannot be adopted"}
    if construction_mode == "compact_bounded":
        protocol["version"] = "actual_local_parent_pn_transport_comparison_v1_2_compact"
    report = dict(protocol=protocol, comparison_features=comparisons,
                  suggested_regions=suggested, suggested_feature_ids=suggested_ids,
                  scientifically_validated=False, automatic_application=False,
                  production_default_allowed=False, sample_used=True, training_repeats=2,
                  heldout_used=False, BN_used=False, features_merged=False,
                  constructor_calls=constructor_calls, retained_parent_count=count-len(suggested),
                  array_contract=dict(candidate_axis="input feature order; IDs retain original type in report",
                      background_methods=list(names), score_methods=[*names, "raw_P", "raw_N"],
                      transport_scores="feature, method, repeat, domain(core/full_support); undefined scores NaN",
                      stationary_signed_chords="feature, same method axis as transport_scores, supplied training-repeat order; fixed native anchor/center chord; undefined scores NaN",
                      candidate_validity="numeric construction validity only; stationary/conflict adoption blockers can retain Parent despite valid diagnostic candidates",
                      local_candidate_accepted_fit="feature, Parent/P/N; engineering fit acceptance only, not physical validation; false for missing constructions",
                      invalid_candidate="parent placeholder with valid=False; scores are undefined, never ranked"),
                  limitations=["A lower normalization-transport proxy is not physical feature truth.",
                      "Repeat-spread margins are deterministic tolerances, not statistical confidence intervals.",
                      "Only individual edits are compared. Overlaps, combined effects, stationary-feature changes, seams, and full-axis gain require caller validation.",
                      "Raw P/N, parent, and constructed candidates can all contain unwanted structure; retaining parent is not a certificate of cleanliness."])
    report["input_array_sha256"] = {name: hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()
        for name, value in (("energy", e), ("training_samples", samples), ("pn_mean", pn), ("parent_background", parent))}
    json.dumps(report, allow_nan=False)
    _check(cancelled)
    return arrays, report
