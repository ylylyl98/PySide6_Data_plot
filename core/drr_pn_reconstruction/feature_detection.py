"""Experimental local derivative-feature detection, with no source selection.

Inputs are already processed by full-axis SG21/order-2 differentiation. Local
extrema of these profiles are not asserted to be physical sample peaks. In
particular, stationary P/N agreement or repeatability cannot establish truth.
This module does not read raw files, backgrounds, BN, angles, or held-out data.
"""
from copy import deepcopy

import numpy as np
from scipy.signal import find_peaks, peak_widths


LOCAL_FEATURE_PROTOCOL = {
    "version": "local_derivative_extrema_v1",
    "experimental": True,
    "expected_processing": "full-axis SG21, polynomial order 2, second derivative",
    "SG21_full_span": "20*(energy[-1]-energy[0])/(n-1)",
    "extrema": "stationary positive and negative separately; dynamic RMS positive extrema only",
    "aggregate": "stationary repeat arithmetic mean; dynamic root mean square of repeat RMS profiles",
    "anchors": "nearest opposite local extremum on each side within two SG21 full spans; otherwise measured local-window or axis boundary, explicitly flagged",
    "minimum_prominence_snr": 5.,
    "minimum_width": "local half-prominence width >= 2*max(local native spacing, SG uniform spacing)",
    "edge_policy": "retain and reject axis extrema, candidates whose nominal SG21 half-window touches the axis, or candidates without two distinct measured anchors",
    "plateau_policy": "retain flat-top edges, use their middle native index; do not invent subpeaks",
    "source_deduplication": "same branch and polarity, at most one native index apart, mutually unique opposite-source neighbor, overlapping local basins containing both centers; retain every observation",
    "center_policy": "matched center nearest the unweighted mean energy, lower native index on a tie; never choose by amplitude",
    "no_support_merge": True,
    "scientifically_validated": False,
    "automatic_application": False,
}


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("Local feature detection cancelled")


def _peak_observations(values):
    peaks, info = find_peaks(values, plateau_size=(1, None))
    observations = [(int(p), int(lo), int(hi), False) for p, lo, hi in
                    zip(peaks, info["left_edges"], info["right_edges"])]
    # scipy deliberately excludes endpoints; retain these as rejected evidence.
    nonflat = np.flatnonzero(values != values[0])
    if len(nonflat) and values[0] > values[nonflat[0]]:
        hi = int(nonflat[0]-1)
        observations.append((hi//2, 0, hi, True))
    nonflat = np.flatnonzero(values != values[-1])
    if len(nonflat) and values[-1] > values[nonflat[-1]]:
        lo = int(nonflat[-1]+1)
        observations.append(((lo+len(values)-1)//2, lo, len(values)-1, True))
    return sorted(observations)


def _raw_candidates(e, aggregate, noise, branch, span, first_id, cancelled):
    result = []
    uniform_step = span/20
    for source in range(2):
        for polarity in ((1, -1) if branch == "stationary" else (1,)):
            _check(cancelled)
            signed = polarity*aggregate[source]
            _, valleys = find_peaks(-signed, plateau_size=(1, None))
            for center, flat_lo, flat_hi, axis_peak in _peak_observations(signed):
                if len(result) % 128 == 0:
                    _check(cancelled)
                window_lo = max(0, int(np.searchsorted(e, e[center]-2*span, side="left")))
                window_hi = min(len(e)-1, int(np.searchsorted(e, e[center]+2*span, side="right")-1))
                left = valleys["right_edges"]
                right = valleys["left_edges"]
                left = left[(left < flat_lo) & (left >= window_lo)]
                right = right[(right > flat_hi) & (right <= window_hi)]
                lo = int(left[-1]) if len(left) else window_lo
                hi = int(right[0]) if len(right) else window_hi
                kinds = ["local_valley" if len(left) else "axis_boundary" if lo == 0 else "local_window_boundary",
                         "local_valley" if len(right) else "axis_boundary" if hi == len(e)-1 else "local_window_boundary"]
                failures = []
                if axis_peak:
                    failures.append("axis_extremum")
                if e[center]-span/2 <= e[0] or e[center]+span/2 >= e[-1]:
                    failures.append("nominal_SG21_footprint_touches_axis")
                valid_anchors = lo < flat_lo <= center <= flat_hi < hi
                if not valid_anchors:
                    failures.append("insufficient_two_sided_local_anchors")
                prominence = float(signed[center]-max(signed[lo], signed[hi])) if valid_anchors else 0.
                prominence = max(0., prominence)
                snr = prominence/float(noise[source])
                if snr < LOCAL_FEATURE_PROTOCOL["minimum_prominence_snr"]:
                    failures.append("local_prominence_below_5_noise")
                native_step = float(max(e[center]-e[max(0, center-1)],
                                        e[min(len(e)-1, center+1)]-e[center]))
                minimum_width = 2*max(native_step, uniform_step)
                core = [float(e[center]), float(e[center])]
                if prominence > 0 and valid_anchors:
                    widths = peak_widths(signed, [center], rel_height=.5,
                        prominence_data=(np.array([prominence]), np.array([lo], dtype=np.intp), np.array([hi], dtype=np.intp)))
                    core = np.interp([widths[2][0], widths[3][0]], np.arange(len(e)), e).tolist()
                width = core[1]-core[0]
                if width < minimum_width:
                    failures.append("half_prominence_width_below_two_sampling_steps")
                result.append(dict(raw_candidate_id=first_id+len(result), branch=branch,
                    source=("P", "N")[source], source_index=source, polarity=polarity,
                    center_index=center, center_ev=float(e[center]),
                    bounds_eV=[float(e[lo]), float(e[hi])], anchor_bounds_eV=[float(e[lo]), float(e[hi])],
                    anchor_indices=[lo, hi], anchor_kinds=kinds,
                    anchor_truncated=any(kind != "local_valley" for kind in kinds),
                    core_bounds_eV=core, half_prominence_width_ev=float(width),
                    plateau_index_bounds=[flat_lo, flat_hi], plateau_bounds_eV=[float(e[flat_lo]), float(e[flat_hi])],
                    local_prominence=prominence, prominence_snr=float(snr), noise_scale=float(noise[source]),
                    minimum_sampling_width_ev=float(minimum_width),
                    accepted=not failures, rejection_reasons=failures,
                    source_channel=f"{branch}_{('P', 'N')[source]}_{'positive' if polarity == 1 else 'negative'}"))
    return result


def _groups(raw, cancelled):
    accepted = [row for row in raw if row["accepted"]]
    neighbors = {}
    for index, row in enumerate(accepted):
        if index % 64 == 0:
            _check(cancelled)
        nearby = []
        for other in accepted:
            if (row["branch"] != other["branch"] or row["polarity"] != other["polarity"]
                    or row["source"] == other["source"] or abs(row["center_index"]-other["center_index"]) > 1):
                continue
            lo = max(row["anchor_indices"][0], other["anchor_indices"][0])
            hi = min(row["anchor_indices"][1], other["anchor_indices"][1])
            common_left = max(row["anchor_bounds_eV"][0], other["anchor_bounds_eV"][0])
            common_right = min(row["anchor_bounds_eV"][1], other["anchor_bounds_eV"][1])
            if (lo < min(row["center_index"], other["center_index"])
                    and hi > max(row["center_index"], other["center_index"])
                    and common_left <= min(row["core_bounds_eV"][0], other["core_bounds_eV"][0])
                    and common_right >= max(row["core_bounds_eV"][1], other["core_bounds_eV"][1])):
                nearby.append(other)
        neighbors[row["raw_candidate_id"]] = nearby
    groups, used = [], set()
    for row in accepted:
        key = row["raw_candidate_id"]
        if key in used:
            continue
        group = [row]
        peers = neighbors[key]
        if len(peers) == 1 and len(neighbors[peers[0]["raw_candidate_id"]]) == 1:
            group.append(peers[0])
        used.update(item["raw_candidate_id"] for item in group)
        groups.append(group)
    return groups


def _channel_evidence(e, profiles, aggregate, noise, center, lo, hi):
    fraction = (e[center]-e[lo])/(e[hi]-e[lo])
    contrasts = profiles[:, :, center]-(profiles[:, :, lo]+fraction*(profiles[:, :, hi]-profiles[:, :, lo]))
    aggregate_contrast = aggregate[:, center]-(aggregate[:, lo]+fraction*(aggregate[:, hi]-aggregate[:, lo]))
    directions = {}
    for j, source in enumerate(("P", "N")):
        contrast = float(aggregate_contrast[j])
        resolved = abs(contrast) >= 5*noise[j]
        repeat_directions = ["positive" if v >= 5*noise[j] else "negative" if v <= -5*noise[j] else "unresolved" for v in contrasts[:, j]]
        directions[source] = dict(direction=("positive" if contrast > 0 else "negative") if resolved else "unresolved",
                                  chord_contrast=contrast, contrast_snr=float(abs(contrast)/noise[j]),
                                  repeat_signed_contrasts=contrasts[:, j].tolist(), repeat_directions=repeat_directions,
                                  repeat_agreement=repeat_directions[0] == repeat_directions[1], noise_scale=float(noise[j]))
    return contrasts, directions


def detect_local_features(energy, stationary_profiles, dynamic_profiles,
                          stationary_noise, dynamic_noise, *, cancelled=None):
    """Detect local extrema/basins without selecting a scientifically valid source.

    Profiles have shape ``(2 training repeats, 2 raw P/N sources, energy)``.
    Dynamic profiles are each repeat/source gate-centered D2's gate RMS. Noise
    arrays contain two positive finite global scaled-MAD estimates from the caller.
    The caller is responsible for the declared SG21/O2 processing provenance.

    Bounds are observed local anchors; core bounds are measured half-prominence
    crossings, never expanded construction guards. Nearby features are retained
    even when their processing windows overlap. Features absent as separate
    extrema after processing cannot be recovered by this detector.
    """
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable or None")
    _check(cancelled)
    e = np.asarray(energy, dtype=float)
    stationary = np.asarray(stationary_profiles, dtype=float)
    dynamic = np.asarray(dynamic_profiles, dtype=float)
    noises = (np.asarray(stationary_noise, dtype=float), np.asarray(dynamic_noise, dtype=float))
    if e.ndim != 1 or len(e) < 21 or not np.isfinite(e).all() or not np.all(np.diff(e) > 0):
        raise ValueError("Finite ascending energy with at least 21 native points required")
    if any(value.shape != (2, 2, len(e)) or not np.isfinite(value).all() for value in (stationary, dynamic)):
        raise ValueError("Finite profiles with shape (2 repeats, 2 P/N sources, energy) required")
    if np.any(dynamic < 0):
        raise ValueError("Dynamic RMS profiles must be nonnegative")
    if any(value.shape != (2,) or not np.all(np.isfinite(value) & (value > 0)) for value in noises):
        raise ValueError("Each noise array needs two positive finite source estimates")
    aggregates = (stationary[0]*.5+stationary[1]*.5, np.hypot(dynamic[0], dynamic[1])/np.sqrt(2))
    span = float(20*(e[-1]-e[0])/(len(e)-1))
    raw = _raw_candidates(e, aggregates[0], noises[0], "stationary", span, 0, cancelled)
    raw += _raw_candidates(e, aggregates[1], noises[1], "dynamic", span, len(raw), cancelled)
    # Preserve evidence at every observation's own geometry, including rejected
    # candidates. A deduplicated row additionally provides common-coordinate
    # comparisons; it must not erase the second source's measured center.
    for index, row in enumerate(raw):
        if index % 128 == 0:
            _check(cancelled)
        lo, hi = row["anchor_indices"]
        center = row["center_index"]
        row["repeat_source_signed_contrasts"] = None
        row["source_directions"] = None
        if lo < center < hi:
            k = 0 if row["branch"] == "stationary" else 1
            contrasts, directions = _channel_evidence(e, (stationary, dynamic)[k], aggregates[k], noises[k], center, lo, hi)
            row["repeat_source_signed_contrasts"] = contrasts.tolist()
            row["source_directions"] = directions
    features = []
    for observations in _groups(raw, cancelled):
        _check(cancelled)
        branch = observations[0]["branch"]
        k = 0 if branch == "stationary" else 1
        lo = max(row["anchor_indices"][0] for row in observations)
        hi = min(row["anchor_indices"][1] for row in observations)
        # There are at most two adjacent native centers: their mean is exactly
        # halfway between them, so choose the lower index without roundoff ties.
        center = min(row["center_index"] for row in observations)
        contrasts, directions = _channel_evidence(e, (stationary, dynamic)[k], aggregates[k], noises[k], center, lo, hi)
        labels = [directions[source]["direction"] for source in ("P", "N")]
        if branch == "stationary":
            conflict = labels[0] != labels[1] and any(label != "unresolved" for label in labels)
            status = "ambiguous_stationary" if conflict else "stationary_consensus_requires_review" if labels[0] != "unresolved" else "stationary_unresolved_requires_review"
        else:
            status = "dynamic_candidate_requires_review"
        features.append(dict(id=-1, branch=branch, polarity=observations[0]["polarity"],
            center_index=center, center_ev=float(e[center]),
            bounds_eV=[float(e[lo]), float(e[hi])], anchor_bounds_eV=[float(e[lo]), float(e[hi])],
            anchor_indices=[lo, hi],
            core_bounds_eV=[min(row["core_bounds_eV"][0] for row in observations), max(row["core_bounds_eV"][1] for row in observations)],
            raw_candidate_ids=[row["raw_candidate_id"] for row in observations], observations=deepcopy(observations),
            source_channels=[row["source_channel"] for row in observations],
            source_directions=directions, repeat_source_signed_contrasts=contrasts.tolist(),
            status=status, suggested_source=None, scientifically_validated=False,
            anchor_truncated=any(row["anchor_truncated"] for row in observations),
            source_match="same_polarity_mutual_unique_native_neighbor" if len(observations) == 2 else "unmatched_observation",
            reason="Local source-dependent derivative feature; physical origin and correct denominator require independent review"))
    features.sort(key=lambda row: (row["center_ev"], row["branch"], row["polarity"], row["raw_candidate_ids"]))
    for index, row in enumerate(features):
        row["id"] = index
        distances = [abs(row["center_ev"]-other["center_ev"]) for other in features
                     if other is not row and other["branch"] == row["branch"] and other["center_ev"] != row["center_ev"]]
        distance = min(distances) if distances else None
        row["nearest_distinct_center_distance_ev"] = distance
        row["shares_nominal_SG21_window_with_neighbor"] = distance is not None and distance < span
    support_mask, core_mask = np.zeros(len(e), bool), np.zeros(len(e), bool)
    for row in features:
        support_mask |= (e >= row["bounds_eV"][0]) & (e <= row["bounds_eV"][1])
        core_mask |= (e >= row["core_bounds_eV"][0]) & (e <= row["core_bounds_eV"][1])
    _check(cancelled)
    arrays = dict(energy=e.copy(), stationary_mean_PN_d2=aggregates[0], dynamic_mean_PN_rms_d2=aggregates[1],
                  stationary_noise_PN=noises[0].copy(), dynamic_noise_PN=noises[1].copy(),
                  feature_support_mask=support_mask, feature_core_mask=core_mask,
                  raw_center_indices=np.array([row["center_index"] for row in raw], dtype=int),
                  feature_center_indices=np.array([row["center_index"] for row in features], dtype=int))
    report = dict(protocol=deepcopy(LOCAL_FEATURE_PROTOCOL), regions=features, local_features=features,
                  local_features_alias_regions=True, raw_candidates=raw,
                  rejected_candidate_ids=[row["raw_candidate_id"] for row in raw if not row["accepted"]],
                  counts=dict(raw_candidates=len(raw), accepted_observations=sum(row["accepted"] for row in raw),
                              rejected_observations=sum(not row["accepted"] for row in raw), local_features=len(features)),
                  resolution=dict(SG21_full_span_ev=span, SG_uniform_step_ev=span/20,
                                  anchor_search_radius_ev=2*span,
                                  interpretation="Two sampling steps are a numerical width requirement, not a certified resolving power. Separate detected extrema within one SG window can be coupled; a processed single extremum is not deblended."),
                  scientifically_validated=False, automatic_application=False, experimental_offline=True,
                  automatic_regions=[], BN_used=False, heldout_used=False, angle_used=False,
                  limitations=["Local prominence detects derivative structure, not physical truth or a valid denominator.",
                               "Stationary agreement and repeat agreement are not scientific acceptance.",
                               "Window-boundary anchors are explicitly truncated local basins; they do not measure a complete broad feature.",
                               "Opposite-polarity source observations are not merged, and local_features is not a count of independent physical peaks."])
    return arrays, report
