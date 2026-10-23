"""Experimental local-feature proposal and comparison; never saves or applies.

The caller supplies a current, matched parent background. This module preserves
local feature identities, permits the parent to win, and checks the actual
combined result. Numerical guards do not identify stationary-feature truth.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from core.processing import _sg_derivative_energy_rows


PROTOCOL = {
    "version": "pn_local_feature_threeway_experimental_v2_1",
    "processing": "complete native axis, production SG21/O2 second derivative",
    "inputs": "matched parent, original PN frames, two training samples only",
    "source_candidates": "parent, actual locally constructed raw-P curvature, actual locally constructed raw-N curvature",
    "comparison_scope": "construct diagnostic candidates whenever geometry permits; stationary uncertainty blocks adoption, not comparison",
    "noise": "quadrature of 1.4826 MAD(training difference/2) and 1.4826 MAD(PN half difference*sqrt(n1*n2)/(n1+n2))",
    "feature_identity": "local features are never merged because construction supports overlap",
    "overlap_policy": "exclude every member of intersecting expanded-support clusters; retain every local feature record",
    "combination": "construct all remaining proposals; recheck every original core and full support on the combined background; any failure retains the exact parent",
    "stationary_guard": "do not erase or reverse a resolved parent's local chord; max PN stationary noise plus parent-repeat noise is a conservative proxy, not a calibrated uncertainty",
    "stationary_truth": "source disagreement remains unresolved; no automatic label from strength, repetition, or stationarity",
    "automatic_application": False,
    "scientifically_validated": False,
}


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("Local feature comparison cancelled")


def _d2(values, energy, cancelled):
    _check(cancelled)
    values = np.asarray(values, float)
    result = _sg_derivative_energy_rows(values.reshape(-1, len(energy)), energy, 21, 2, 2)
    _check(cancelled)
    return result.reshape(values.shape)


def _mad(values):
    return float(1.4826*np.median(abs(values-np.median(values))))


def _hash(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _validated(energy, pn_frames, training_samples, parent_background, cancelled):
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable or None")
    _check(cancelled)
    e, frames, samples, parent = (np.asarray(value, float) for value in
        (energy, pn_frames, training_samples, parent_background))
    if e.ndim != 1 or len(e) < 41 or not np.isfinite(e).all() or np.any(np.diff(e) <= 0):
        raise ValueError("At least 41 finite, increasing native energy points are required")
    if frames.ndim != 3 or frames.shape[0] != 2 or frames.shape[1] < 4 or frames.shape[2] != len(e):
        raise ValueError("Original PN frames must have shape (2, frames>=4, energy)")
    if samples.ndim != 3 or samples.shape[0] != 2 or samples.shape[1] < 3 or samples.shape[2] != len(e):
        raise ValueError("Exactly two training repeats with at least three gate rows are required")
    if parent.shape != e.shape:
        raise ValueError("Parent background must match the complete native energy axis")
    if any(not np.all(np.isfinite(value) & (value > 0)) for value in (frames, samples, parent)):
        raise ValueError("Positive finite PN, sample and parent intensities are required")
    return e, frames, samples, parent


def _prepare_profiles(e, frames, samples, cancelled):
    pn = frames.mean(axis=1)
    full = _d2(samples[:, None]/pn[None, :, None]-1, e, cancelled)
    stationary = full.mean(axis=2)
    dynamic = full-stationary[:, :, None]
    split = frames.shape[1]//2
    halves = np.stack((frames[:, :split].mean(axis=1), frames[:, split:].mean(axis=1)))
    half_maps = _d2(samples.mean(axis=0)[None, None]/halves[:, :, None]-1, e, cancelled)
    half_stationary = half_maps.mean(axis=2)
    half_dynamic = half_maps-half_stationary[:, :, None]
    factor = float(np.sqrt(split*(frames.shape[1]-split))/frames.shape[1])
    noises = []
    for values, half_values in ((stationary, half_stationary), (dynamic, half_dynamic)):
        noise = np.empty(2)
        for source in range(2):
            noise[source] = max(float(np.hypot(_mad((values[0, source]-values[1, source])/2),
                                                _mad((half_values[0, source]-half_values[1, source])*factor))),
                64*np.finfo(float).eps*max(1., float(np.max(abs(values[:, source])))))
        noises.append(noise)
    return pn, stationary, np.sqrt(np.mean(dynamic**2, axis=2)), noises, factor


def _independent_supports(e, proposed, feature_ids):
    """No signal-dependent subset search or hidden feature/region merging."""
    if len(proposed) != len(feature_ids):
        raise ValueError("Suggested regions must preserve their feature IDs")
    span = 20*(e[-1]-e[0])/(len(e)-1)
    rows = sorted(zip(proposed, feature_ids), key=lambda item: item[0]["xmin_ev"])
    excluded, bad = [], set()
    supports = [(row["xmin_ev"]-2*span, row["xmax_ev"]+2*span) for row, _ in rows]
    for i, (lo, hi) in enumerate(supports):
        if not e[0] < lo < hi < e[-1]:
            bad.add(i)
            excluded.append(dict(feature_id=rows[i][1], reason="incomplete construction support"))
        for j in range(i):
            if lo <= supports[j][1] and hi >= supports[j][0]:
                bad.update((i, j))
    for i in sorted(bad):
        if not any(row["feature_id"] == rows[i][1] for row in excluded):
            excluded.append(dict(feature_id=rows[i][1], reason="overlapping construction supports; local feature identities retained"))
    return [row for i, (row, _) in enumerate(rows) if i not in bad], [fid for i, (_, fid) in enumerate(rows) if i not in bad], excluded


def _cosine(a, b):
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip(np.ravel(a) @ np.ravel(b)/denominator,-1.,1.)) if denominator else None


def _combined_checks(e, samples, parent, candidate, features, selected_ids, stationary_noise, cancelled):
    from .feature_comparison import score_dynamic_background, improvement_margin
    parent_maps = _d2(samples/parent-1, e, cancelled)
    candidate_maps = _d2(samples/candidate-1, e, cancelled)
    centered = samples-samples.mean(axis=1, keepdims=True)
    parent_dynamic = _d2(centered/parent, e, cancelled)
    candidate_dynamic = _d2(centered/candidate, e, cancelled)
    span = 20*(e[-1]-e[0])/(len(e)-1)
    records, failures = [], []
    for feature in features:
        if feature["id"] not in selected_ids:
            continue
        core = feature.get("core_bounds_eV", feature["bounds_eV"])
        support = [feature["bounds_eV"][0]-2*span, feature["bounds_eV"][1]+2*span]
        row = dict(feature_id=feature["id"], intervals={})
        for label, bounds in (("core", core), ("support", support)):
            old = score_dynamic_background(e, samples, parent, bounds, cancelled=cancelled)
            new = score_dynamic_background(e, samples, candidate, bounds, cancelled=cancelled)
            finite = bool(np.isfinite(old).all() and np.isfinite(new).all())
            margin = improvement_margin(old, new) if finite else None
            mask = (e >= bounds[0]) & (e <= bounds[1])
            cosines = [_cosine(parent_dynamic[i, :, mask], candidate_dynamic[i, :, mask]) for i in range(2)]
            repeat_cosine = _cosine(candidate_dynamic[0, :, mask], candidate_dynamic[1, :, mask])
            ok = bool(finite
                      and np.all(np.asarray(old)-new > margin) and np.all(np.asarray(new) <= .25)
                      and all(value is not None and value >= .9 for value in cosines+[repeat_cosine]))
            row["intervals"][label] = dict(bounds_eV=list(bounds),
                parent_scores=[float(value) if np.isfinite(value) else None for value in old],
                candidate_scores=[float(value) if np.isfinite(value) else None for value in new], improvement_margin=margin,
                parent_candidate_cosines=cosines, candidate_repeat_cosine=repeat_cosine, passed=ok)
            if not ok:
                failures.append(dict(feature_id=feature["id"], reason=f"combined candidate failed {label} comparison"))
        records.append(row)
    # Check local stationary directions; gain/tilt and full maps remain visible
    # in evaluation even when this limited direction guard passes.
    parent_stationary = parent_maps.mean(axis=1)
    candidate_stationary = candidate_maps.mean(axis=1)
    noise = float(np.hypot(_mad((parent_stationary[0]-parent_stationary[1])/2), max(stationary_noise)))
    stationary_records = []
    for feature in features:
        if feature["branch"] != "stationary":
            continue
        lo, hi = feature["anchor_bounds_eV"]
        indices = [int(np.argmin(abs(e-value))) for value in (lo, feature["center_ev"], hi)]
        a, center, b = indices
        if not a < center < b:
            continue
        fraction = (e[center]-e[a])/(e[b]-e[a])
        contrasts = [values[:, center]-(1-fraction)*values[:, a]-fraction*values[:, b]
                     for values in (parent_stationary, candidate_stationary)]
        old, new = (float(value.mean()) for value in contrasts)
        resolved = abs(old) >= 5*noise
        ok = not resolved or (abs(new) >= 5*noise and np.sign(old) == np.sign(new))
        stationary_records.append(dict(feature_id=feature["id"], parent_chord=old, candidate_chord=new,
            noise_proxy=noise, parent_resolved=resolved, direction_retained=bool(ok)))
        if not ok:
            failures.append(dict(feature_id=feature["id"], reason="combined candidate erases or reverses resolved stationary parent chord"))
    return dict(passed=not failures, feature_checks=records, stationary_checks=stationary_records,
        failures=failures, whole_axis_gain_minmax=[float(np.min(candidate/parent)), float(np.max(candidate/parent))],
        scope="Training-only numerical guards; direction preservation is not physical validation")


def analyze_local_feature_sources(energy, pn_frames, training_samples, parent_background, *,
                                  construction_mode="integrated_curvature", cancelled=None):
    """Return an unsaved experimental background, local evidence and decisions.

    Samples are exactly two training repeats. Held-out data, BN, angles and
    hand-selected energy windows are not accepted. The parent must belong to
    this same measurement group and source/frame selection; array validation
    alone cannot establish its provenance. The existing application default
    remains unchanged, and stationary ambiguity remains a review requirement.
    """
    from .feature_detection import detect_local_features
    from .feature_comparison import compare_feature_backgrounds, _constructor_for_mode
    constructor = _constructor_for_mode(construction_mode)
    e, frames, samples, parent = _validated(energy, pn_frames, training_samples, parent_background, cancelled)
    input_hashes = {key: _hash(value) for key, value in (
        ("energy", e), ("pn_frames", frames), ("training_samples", samples), ("parent_background", parent))}
    pn, stationary, dynamic, noises, factor = _prepare_profiles(e, frames, samples, cancelled)
    detection_arrays, detector = detect_local_features(e, stationary, dynamic, *noises, cancelled=cancelled)
    features = detector["local_features"]
    comparison_arrays, comparison = compare_feature_backgrounds(e, samples, pn, parent, features,
        construction_mode=construction_mode, cancelled=cancelled)
    proposed = comparison["suggested_regions"]
    proposed_ids = comparison["suggested_feature_ids"]
    ready, ready_ids, exclusions = _independent_supports(e, proposed, proposed_ids)
    candidate, construction_arrays, construction = constructor(e, parent, pn, ready, cancelled=cancelled)
    proposed_combined = candidate.copy()
    proposed_construction = deepcopy(construction)
    checks = (_combined_checks(e, samples, parent, candidate, features, ready_ids, noises[0], cancelled)
              if ready else dict(passed=False, failures=[], scope="No non-overlapping local source proposals to combine"))
    if ready and construction_mode == "compact_bounded" and not construction["fit_accepted"]:
        checks["passed"] = False
        checks["failures"].append(dict(reason="combined compact source-curvature fit rejected"))
    applied, applied_ids = ready, ready_ids
    if not checks["passed"]:
        candidate, construction_arrays, construction = constructor(e, parent, pn, [], cancelled=cancelled)
        applied, applied_ids = [], []
    protocol = {**deepcopy(PROTOCOL), "construction_mode": construction_mode}
    if construction_mode == "compact_bounded":
        protocol["version"] = "pn_local_feature_threeway_experimental_v3_compact"
    report = dict(algorithm_version=protocol["version"], protocol=protocol,
        detector=detector, comparison=comparison, local_features=features, regions=features, detected_features=features,
        automatic_regions=applied, scientifically_validated=False, production_default_allowed=False,
        BN_used=False, heldout_used=False, angle_used=False, sample_used=True, parent_background_used=True,
        input_array_sha256=input_hashes, pn_frame_count=int(frames.shape[1]),
        pn_half_frame_counts=[frames.shape[1]//2, frames.shape[1]-frames.shape[1]//2], pn_half_noise_factor=factor,
        experiment_construction=dict(proposed_regions=proposed, proposed_feature_ids=proposed_ids,
            nonoverlapping_regions=ready, nonoverlapping_feature_ids=ready_ids, exclusions=exclusions,
            applied_regions=applied, applied_feature_ids=applied_ids, combination_check=checks,
            constructor_report=construction, proposed_constructor_report=proposed_construction,
            candidate_changed=not np.array_equal(parent, candidate),
            renderer=construction_mode+"; threeway local comparison and final training-only guards"),
        limitation="Resolving local features and retaining the parent can prevent numerical regressions; neither proves correct stationary source choice or arbitrary-pair generalization.")
    arrays = dict(energy=e.copy(), background=candidate, parent_background=parent.copy(),
        proposed_combined_background=proposed_combined, stationary_profiles=stationary,
        dynamic_profiles=dynamic, stationary_noise_PN=noises[0], dynamic_noise_PN=noises[1])
    arrays.update({"detection_"+key: value for key, value in detection_arrays.items()})
    arrays.update(comparison_arrays)
    arrays.update({"construction_"+key: value for key, value in construction_arrays.items()})
    assert input_hashes == {key: _hash(value) for key, value in (
        ("energy", e), ("pn_frames", frames), ("training_samples", samples), ("parent_background", parent))}
    _check(cancelled)
    json.dumps(report, allow_nan=False)
    return arrays, report
