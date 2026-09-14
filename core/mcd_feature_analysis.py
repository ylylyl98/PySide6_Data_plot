"""Pure selected-feature analysis for the unified MCD page.

This helper is intentionally small: catalog detection remains separate from a
selected seeded trajectory fit, and no Qt objects are accessed.  The returned
mapping contains only JSON-compatible values so workers/exporters can pass it
across a boundary without carrying live figures or widgets.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import copy
from types import SimpleNamespace

import numpy as np

from core.mcd_analysis import (
    AnalysisFeature,
    FeatureLink,
    associate_features,
    infer_valley_mapping,
    split_same_branch_tracks,
    _coerce_features,
)
from core.mcd_peak_shift import analyze_local_peak_shift, analyze_peak_shift


def _json_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _feature(value: Any) -> AnalysisFeature | None:
    return _coerce_features((value,))[0] if _coerce_features((value,)) else None


def _channel_source(feature: AnalysisFeature, requested: str) -> str:
    source = str(feature.source or "raw pos").strip().casefold()
    request = str(requested or "Raw").strip().casefold()
    channel = "neg" if source.endswith("neg") else "pos"
    if request in {"raw", "raw spectrum"}:
        corrected = False
    elif request.startswith("correct") or request in {"corrected spectrum", "corrected average"}:
        corrected = True
    else:
        corrected = "corrected" in source
    return f"corrected {channel}" if corrected else f"raw {channel}"


def _result_for_channel(result: Any, channel_source: str) -> Any:
    """Return a shallow result view whose pair B grid matches the channel."""
    view = copy.copy(result)
    channel = "neg" if channel_source.casefold().endswith("neg") else "pos"
    field_name = f"pair_b_{channel}"
    if hasattr(result, field_name):
        fields = np.asarray(getattr(result, field_name), dtype=float).ravel()
        if fields.size:
            setattr(view, "pair_b", fields)
    return view


def _point_dict(point: Any, reference_override: float | None = None) -> dict[str, Any]:
    energy = _json_float(point.energy_ev)
    delta = _json_float(point.delta_energy_ev)
    if reference_override is not None and energy is not None and str(point.status) in {"tracked", "ok"}:
        delta = energy - reference_override
    return {
        "field_t": _json_float(point.field_t),
        "energy_ev": energy,
        "delta_energy_ev": delta,
        "status": str(point.status),
    }


def _track_dict(track: Any, feature_id: str, channel: str, method: str, reference_override: float | None = None) -> dict[str, Any]:
    return {
        "feature_id": feature_id,
        "channel": channel,
        "method": method,
        "branch": str(track.branch),
        "feature_kind": str(track.feature_kind),
        "peak_id": int(track.peak_id),
        "reference_energy_ev": reference_override if reference_override is not None else _json_float(track.reference_energy_ev),
        "reference_field_t": None if reference_override is not None else _json_float(track.reference_field_t),
        "reference_method": "manual" if reference_override is not None else str(getattr(track, "reference_method", "unknown")),
        "points": [_point_dict(point, reference_override) for point in track.points],
        "quality": str(track.quality),
    }


def _constrain_track_window(track: Any, window: tuple[float, float]) -> Any:
    """Keep a selected trajectory while marking out-of-window samples gaps."""
    low, high = window
    source_points = list(track.points)
    valid = [
        point for point in source_points
        if point.energy_ev is not None and str(point.status) in {"tracked", "ok"}
        and np.isfinite(float(point.energy_ev)) and low <= float(point.energy_ev) <= high
    ]
    reference: float | None = None
    reference_method = "unavailable"
    exact = [point for point in valid if np.isfinite(float(point.field_t)) and abs(float(point.field_t)) <= 1e-9]
    if exact:
        point = min(exact, key=lambda item: abs(float(item.field_t)))
        reference = float(point.energy_ev)
        reference_method = "exact 0 T"
    else:
        ordered = sorted((point for point in valid if np.isfinite(float(point.field_t))), key=lambda item: float(item.field_t))
        for left, right in zip(ordered[:-1], ordered[1:]):
            if float(left.field_t) < 0.0 < float(right.field_t):
                ratio = -float(left.field_t) / (float(right.field_t) - float(left.field_t))
                reference = float(left.energy_ev) + (float(right.energy_ev) - float(left.energy_ev)) * ratio
                reference_method = "interpolated near-zero"
                break
    points = []
    for point in source_points:
        energy = point.energy_ev
        in_window = energy is not None and np.isfinite(float(energy)) and low <= float(energy) <= high
        if in_window:
            delta = float(energy) - reference if reference is not None and str(point.status) in {"tracked", "ok"} else None
            points.append(SimpleNamespace(field_t=point.field_t, branch=getattr(point, "branch", track.branch), energy_ev=energy, delta_energy_ev=delta, status=point.status))
        else:
            points.append(SimpleNamespace(field_t=point.field_t, branch=getattr(point, "branch", track.branch), energy_ev=None, delta_energy_ev=None, status="window gap"))
    return SimpleNamespace(
        peak_id=track.peak_id,
        branch=track.branch,
        points=tuple(points),
        reference_energy_ev=reference,
        reference_field_t=0.0 if reference is not None else None,
        reference_method=reference_method,
        quality=track.quality,
        feature_kind=track.feature_kind,
    )


def _analysis_dict(analysis: Any, feature_id: str, channel: str, method: str, tracks: Sequence[Any], reference_override: float | None = None) -> dict[str, Any]:
    return {
        "feature_id": feature_id,
        "channel": channel,
        "method": method,
        "fields_t": [_json_float(value) for value in np.asarray(analysis.fields_t, dtype=float)],
        "branches": [str(value) for value in np.asarray(analysis.branches, dtype=object)],
        "source": str(analysis.source),
        "tracking_method": str(analysis.tracking_method),
        # Only tracks that passed the selected feature's kind/branch/energy
        # gate belong in this result.  The full detector inventory is kept in
        # the catalog and must not leak into an export of a seeded selection.
        "tracks": [_track_dict(track, feature_id, channel, method, reference_override) for track in tracks],
        "locator_energy_ev": _json_float(analysis.locator_energy_ev),
        "fit_window_ev": None if analysis.fit_window_ev is None else [_json_float(value) for value in analysis.fit_window_ev],
    }


def _supported_related(selected: AnalysisFeature, candidates: Sequence[AnalysisFeature], tolerance: float) -> list[tuple[float, AnalysisFeature]]:
    related: list[tuple[float, AnalysisFeature]] = []
    selected_channel = "neg" if selected.source.casefold().endswith("neg") else "pos"
    for candidate in candidates:
        if candidate.id == selected.id or candidate.domain != "spectrum" or candidate.kind != selected.kind:
            continue
        channel = "neg" if candidate.source.casefold().endswith("neg") else "pos"
        if channel == selected_channel or abs(candidate.energy_ev - selected.energy_ev) > tolerance:
            continue
        if selected.branch and candidate.branch and selected.branch != candidate.branch and selected.branch not in {"B sweep", "all"} and candidate.branch not in {"B sweep", "all"}:
            continue
        common = set(round(value, 10) for value in selected.support_fields) & set(round(value, 10) for value in candidate.support_fields)
        interval = 0.0
        if selected.support_interval_t and candidate.support_interval_t:
            interval = max(0.0, min(selected.support_interval_t[1], candidate.support_interval_t[1]) - max(selected.support_interval_t[0], candidate.support_interval_t[0]))
        if len(common) < 2 and interval <= 0:
            continue
        related.append((abs(candidate.energy_ev - selected.energy_ev), candidate))
    return sorted(related, key=lambda item: (item[0], item[1].id))


def _run_selected(
    result: Any,
    feature: AnalysisFeature,
    *,
    method: str,
    channel_source: str,
    half_width_ev: float,
    search_window_ev: tuple[float, float] | None = None,
    prominence_fraction: float | None = None,
) -> Any:
    view = _result_for_channel(result, channel_source)
    field = feature.field_t
    if field is None:
        field = float(np.median(feature.support_fields)) if feature.support_fields else 0.0
    if method.casefold().strip() in {"local fit", "local mixed fit", "local"}:
        window = search_window_ev or (float(feature.energy_ev - half_width_ev), float(feature.energy_ev + half_width_ev))
        return analyze_local_peak_shift(view, source=channel_source, seed_energy_ev=float(feature.energy_ev), locator_energy_ev=float(feature.energy_ev), feature_kind=feature.kind, window_ev=(float(window[0]), float(window[1])))
    options: dict[str, Any] = {
        "source": channel_source,
        "tracking_method": method,
        "seed_energy_ev": float(feature.energy_ev),
        "seed_half_width_ev": float(half_width_ev),
        "seed_field_t": float(field),
        "max_peaks": 1,
    }
    if prominence_fraction is not None:
        options["prominence_fraction"] = float(prominence_fraction)
    return analyze_peak_shift(view, **options)


def analyze_selected_feature(
    result: Any,
    feature: AnalysisFeature | Mapping[str, Any],
    *,
    candidates: Sequence[AnalysisFeature | Mapping[str, Any]] = (),
    method: str = "Raw spectrum",
    source: str = "Raw",
    half_width_ev: float = 0.005,
    manual_links: Sequence[Any] = (),
    search_window_ev: Sequence[float] | None = None,
    prominence_fraction: float | None = None,
    reference_energy_ev: float | None = None,
) -> dict[str, Any]:
    """Analyze only the selected catalog feature and supported trajectories.

    MCD selections first use the supplied catalog and conservative association;
    ambiguous or unmatched links return without running a seeded fit. Spectrum
    selections run the selected channel and a supported counterpart channel.
    ``Local mixed fit`` is run only when explicitly requested.
    """
    selected = _feature(feature)
    normalized_window: tuple[float, float] | None = None
    if search_window_ev is not None:
        try:
            bounds = tuple(float(value) for value in search_window_ev)
        except (TypeError, ValueError):
            bounds = ()
        if len(bounds) != 2 or not all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            return {"selected_feature": selected.to_dict() if selected else None, "method": str(method), "source": str(source), "tracks": [], "analysis_results": {}, "links": [], "status": "unsupported", "reason": "search_window_ev must be finite increasing bounds"}
        normalized_window = (bounds[0], bounds[1])
    normalized_reference = _json_float(reference_energy_ev)
    if reference_energy_ev is not None and normalized_reference is None:
        return {"selected_feature": selected.to_dict() if selected else None, "method": str(method), "source": str(source), "tracks": [], "analysis_results": {}, "links": [], "status": "unsupported", "reason": "reference_energy_ev must be finite"}
    base = {"selected_feature": selected.to_dict() if selected else None, "method": str(method), "source": str(source), "tracks": [], "analysis_results": {}, "links": [], "status": "unmatched"}
    if normalized_window is not None:
        base["search_window_ev"] = [normalized_window[0], normalized_window[1]]
    if prominence_fraction is not None:
        try:
            prominence = float(prominence_fraction)
        except (TypeError, ValueError):
            prominence = float("nan")
        if not np.isfinite(prominence) or prominence < 0:
            base["status"] = "unsupported"
            base["reason"] = "prominence_fraction must be finite and non-negative"
            return base
        base["prominence_fraction"] = prominence
    if normalized_reference is not None:
        base["reference_energy_ev"] = normalized_reference
    if selected is None:
        base["reason"] = "selected feature is invalid"
        return base
    catalog = tuple(item for item in (_feature(value) for value in candidates) if item is not None)
    targets: list[AnalysisFeature] = []
    links: list[FeatureLink] = []
    if selected.domain == "mcd":
        spectrum = tuple(item for item in catalog if item.domain == "spectrum" and item.source.casefold().startswith("raw"))
        association = associate_features((selected,), spectrum, manual_links=manual_links, energy_tolerance_ev=max(float(half_width_ev) * 2.0, .012))
        links.extend(association.links)
        if association.ambiguous:
            base["status"] = "ambiguous"
            base["reason"] = "MCD candidate has ambiguous supported spectral associations"
        else:
            targets.extend(item for item in spectrum if any(link.mcd_id == selected.id and link.spectrum_id == item.id and link.status in {"automatic", "manual"} for link in association.links))
            if not targets:
                base["status"] = "unmatched"
                base["reason"] = "no supported raw spectral candidate"
    elif selected.domain == "spectrum":
        targets.append(selected)
        related = _supported_related(selected, catalog, max(float(half_width_ev) * 2.0, .012))
        channels: dict[str, list[tuple[float, AnalysisFeature]]] = {}
        for distance, item in related:
            channel = "neg" if item.source.casefold().endswith("neg") else "pos"
            channels.setdefault(channel, []).append((distance, item))
        for channel, items in sorted(channels.items()):
            if len(items) > 1 and abs(items[1][0] - items[0][0]) <= .05:
                ids = tuple(item.id for _, item in items)
                links.append(FeatureLink(selected.id, None, "spectrum_to_channel", "ambiguous", None, (f"multiple supported {channel} candidates",), ids))
            else:
                item = items[0][1]
                targets.append(item)
                links.append(FeatureLink(selected.id, item.id, "spectrum_to_channel", "automatic", float(items[0][0]), ("supported counterpart channel",), (item.id,)))
    else:
        base["status"] = "unsupported"
        base["reason"] = f"unsupported feature domain: {selected.domain}"
    if not targets:
        base["links"] = [link.to_dict() for link in links]
        return base
    for target in targets:
        channel_source = _channel_source(target, source)
        try:
            run_half_width = float(half_width_ev)
            if normalized_window is not None:
                # Keep the catalog feature as the seed center.  A single
                # half-width cannot encode asymmetric bounds, so widen only
                # enough to cover both sides and apply the exact bounds below.
                run_half_width = max(abs(float(target.energy_ev) - normalized_window[0]), abs(normalized_window[1] - float(target.energy_ev)))
                if run_half_width <= 0 or not np.isfinite(run_half_width):
                    base["status"] = "unsupported"
                    base["reason"] = "search_window_ev does not contain a valid seed span"
                    continue
            analysis = _run_selected(result, target, method=method, channel_source=channel_source, half_width_ev=run_half_width, search_window_ev=normalized_window, prominence_fraction=prominence_fraction)
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            base["status"] = "unsupported"
            base["reason"] = f"selected analysis unavailable: {str(exc).splitlines()[0]}"
            continue
        filtered = []
        for track in analysis.tracks:
            if str(track.feature_kind) != str(target.kind):
                continue
            if target.branch and target.branch not in {"B sweep", "all"} and str(track.branch) != target.branch:
                continue
            constrained = _constrain_track_window(track, normalized_window) if normalized_window is not None else track
            energies = [point.energy_ev for point in constrained.points if point.energy_ev is not None and point.status in {"tracked", "ok"}]
            if energies and min(abs(float(value) - target.energy_ev) for value in energies) <= max(float(half_width_ev) * 4.0, .03):
                filtered.append(constrained)
        # A channel can have multiple explicitly linked targets.  Include the
        # stable feature id and method in the key so a later target cannot
        # overwrite an earlier same-channel result.
        key = f"{channel_source}|feature={target.id}|method={method}"
        base["analysis_results"][key] = _analysis_dict(analysis, target.id, channel_source, method, filtered, normalized_reference)
        for track in filtered:
            base["tracks"].append(_track_dict(track, target.id, channel_source, method, normalized_reference))
    if base["tracks"]:
        base["status"] = "ok"
    elif normalized_window is not None and base["status"] == "unmatched":
        base["reason"] = "search window contains no selected feature trajectory"
    elif base["status"] not in {"ambiguous", "unmatched", "unsupported"}:
        base["status"] = "unmatched"
        base["reason"] = "seeded analysis returned no selected feature trajectory"
    base["links"] = [link.to_dict() for link in links]
    return base


def _track_object(track: Mapping[str, Any]) -> Any:
    """Adapt one JSON track to the conservative core splitting API."""
    points = tuple(SimpleNamespace(
        field_t=point.get("field_t"),
        energy_ev=point.get("energy_ev"),
        status=point.get("status", "tracked"),
    ) for point in track.get("points", ()) if isinstance(point, Mapping))
    return SimpleNamespace(branch=str(track.get("branch", "")), points=points)


def _channel_name(value: Any) -> str | None:
    text = str(value or "").casefold().strip()
    if text.endswith("pos") or text in {"k", "positive"}:
        return "pos"
    if text.endswith("neg") or text in {"kp", "kprime", "negative"}:
        return "neg"
    return None


def _linked_track_pairs(payload: Mapping[str, Any], reference_id: str | None) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], bool, str]:
    tracks = [item for item in payload.get("tracks", ()) if isinstance(item, Mapping)]
    by_branch_channel_id: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for track in tracks:
        channel = _channel_name(track.get("channel"))
        if channel is None:
            continue
        key = (str(track.get("branch", "")), channel, str(track.get("feature_id", "")))
        by_branch_channel_id.setdefault(key, []).append(track)
    groups: dict[str, set[str]] = {}
    ambiguous = False
    unpaired_reason = ""
    for raw_link in payload.get("links", ()):
        if not isinstance(raw_link, Mapping):
            continue
        root = str(raw_link.get("mcd_id", ""))
        if reference_id is not None and root != reference_id:
            continue
        status = str(raw_link.get("status", "")).casefold()
        if status == "ambiguous":
            ambiguous = True
            continue
        relation = str(raw_link.get("relation", "")).casefold()
        # MCD-to-spectrum links establish candidate associations, but do not
        # establish a K/K' pair.  Valley splitting requires a separate
        # explicit channel-pair relation from the selected feature workflow.
        if relation not in {"spectrum_to_channel", "valley_pair", "valley_pairing", "k_kp_pair"}:
            if relation == "mcd_to_spectrum":
                unpaired_reason = "MCD associations do not establish a valley pair"
            continue
        spectrum_id = raw_link.get("spectrum_id")
        if status not in {"automatic", "manual"} or spectrum_id in {None, ""}:
            continue
        groups.setdefault(root, set()).update((root, str(spectrum_id)))
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for ids in groups.values():
        for branch in sorted({key[0] for key in by_branch_channel_id if key[2] in ids}):
            pos = [track for key, values in by_branch_channel_id.items() if key[:2] == (branch, "pos") and key[2] in ids for track in values]
            neg = [track for key, values in by_branch_channel_id.items() if key[:2] == (branch, "neg") and key[2] in ids for track in values]
            if len(pos) == 1 and len(neg) == 1:
                if str(pos[0].get("feature_kind", "peak")) != str(neg[0].get("feature_kind", "peak")):
                    unpaired_reason = "linked candidates have different feature kinds"
                elif str(pos[0].get("method", "")) != str(neg[0].get("method", "")):
                    unpaired_reason = "linked candidates use different analysis methods"
                else:
                    pairs.append((pos[0], neg[0]))
            elif pos or neg:
                ambiguous = True
    return pairs, ambiguous, unpaired_reason


def _reference_samples(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, list[tuple[float, float, float]]]:
    by_branch: dict[str, list[tuple[float, float, float]]] = {}
    for pos, neg in pairs:
        pos_points = {round(float(point.get("field_t")), 10): point for point in pos.get("points", ()) if isinstance(point, Mapping) and point.get("field_t") is not None and point.get("energy_ev") is not None and str(point.get("status", "tracked")) in {"tracked", "ok"} and float(point.get("field_t")) > 0}
        neg_points = {round(float(point.get("field_t")), 10): point for point in neg.get("points", ()) if isinstance(point, Mapping) and point.get("field_t") is not None and point.get("energy_ev") is not None and str(point.get("status", "tracked")) in {"tracked", "ok"} and float(point.get("field_t")) > 0}
        branch = str(pos.get("branch", ""))
        for field in sorted(set(pos_points) & set(neg_points)):
            by_branch.setdefault(branch, []).append((float(field), float(pos_points[field]["energy_ev"]), float(neg_points[field]["energy_ev"])))
    return by_branch


def enrich_valley_analysis(
    payload: Mapping[str, Any],
    *,
    mapping_mode: str = "unknown",
    k_channel: str | None = None,
    reference_id: str | None = None,
) -> dict[str, Any]:
    """Attach an explicit K/K' mapping and conservative E_K-E_Kp splitting.

    Only feature pairs represented by automatic/manual links are considered.
    Mapping inference is opt-in via ``mapping_mode='energy_based'``; an unknown
    mapping or ambiguous relation returns no splitting rows.
    """
    enriched = copy.deepcopy(dict(payload))
    mode = str(mapping_mode or "unknown").casefold().replace("-", "_").replace(" ", "_")
    if reference_id is None:
        selected = enriched.get("selected_feature")
        if isinstance(selected, Mapping) and selected.get("id"):
            reference_id = str(selected["id"])
    pairs, pair_ambiguous, pair_unpaired_reason = _linked_track_pairs(enriched, reference_id)
    fixed_modes = {"fixed", "manual", "calibrated", "swap", "fixed_mapping", "manual_mapping", "calibrated_mapping", "manual_swap", "fixed_pos", "fixed_neg", "calibrated_pos", "calibrated_neg"}
    energy_modes = {"energy", "energy_based", "energy_mapping", "inferred"}
    mapping: dict[str, Any]
    if mode in fixed_modes or any(token in mode for token in ("fixed", "manual", "calibrated", "swap")):
        channel = _channel_name(k_channel)
        if channel is None and mode.endswith(("_pos", "_neg")):
            channel = mode.rsplit("_", 1)[-1]
        if channel is None:
            mapping = {"status": "unknown", "mode": mode, "k_channel": None, "kp_channel": None, "reference_id": reference_id, "reason": "fixed mapping requires explicit k_channel"}
        else:
            mapping = {"status": "fixed", "mode": mode, "k_channel": channel, "kp_channel": "neg" if channel == "pos" else "pos", "reference_id": reference_id, "reason": "explicit channel mapping"}
    elif mode in energy_modes:
        samples_by_branch = _reference_samples(pairs)
        samples = max(samples_by_branch.values(), key=len, default=[])
        evidence = {"fields": [item[0] for item in samples], "pos": [item[1] for item in samples], "neg": [item[2] for item in samples]}
        inferred = infer_valley_mapping(reference_id, evidence)
        mapping = inferred.to_dict()
        mapping["mode"] = mode
    else:
        mapping = {"status": "unknown", "mode": mode or "unknown", "k_channel": None, "kp_channel": None, "reference_id": reference_id, "reason": "no explicit or opt-in energy mapping"}
    enriched["mapping"] = mapping
    enriched["splitting"] = []
    if mapping.get("status") not in {"fixed", "inferred"}:
        enriched["splitting_status"] = "ambiguous" if pair_ambiguous else "unpaired" if pair_unpaired_reason else "unknown"
        enriched["splitting_reason"] = "ambiguous linked feature relation" if pair_ambiguous else pair_unpaired_reason if pair_unpaired_reason else "valley mapping is unknown"
        return enriched
    if pair_ambiguous:
        enriched["splitting_status"] = "ambiguous"
        enriched["splitting_reason"] = "ambiguous linked feature relation"
        return enriched
    if pair_unpaired_reason:
        enriched["splitting_status"] = "unpaired"
        enriched["splitting_reason"] = pair_unpaired_reason
        return enriched
    if not pairs:
        enriched["splitting_status"] = "insufficient"
        enriched["splitting_reason"] = "no explicit linked peak pair"
        return enriched
    k_name, kp_name = mapping["k_channel"], mapping["kp_channel"]
    records: list[dict[str, Any]] = []
    statuses: list[str] = []
    for left, right in pairs:
        left_channel = _channel_name(left.get("channel"))
        k_track, kp_track = (left, right) if left_channel == k_name else (right, left) if left_channel == kp_name else (None, None)
        if k_track is None:
            continue
        split = split_same_branch_tracks(_track_object(k_track), _track_object(kp_track), branch=str(k_track.get("branch", "")))
        status = "ok" if split.status == "ok" else "insufficient"
        statuses.append(status)
        records.append({
            "pair_id": f"{k_track.get('feature_id', '')}|{kp_track.get('feature_id', '')}|{k_track.get('method', '')}",
            "method": k_track.get("method", ""),
            "selected_channel": k_track.get("channel", ""),
            "counterpart_channel": kp_track.get("channel", ""),
            "selected_feature_id": k_track.get("feature_id", ""),
            "counterpart_feature_id": kp_track.get("feature_id", ""),
            "selected_peak_id": k_track.get("peak_id"),
            "counterpart_peak_id": kp_track.get("peak_id"),
            "branch": k_track.get("branch", ""),
            "status": status,
            "convention": "E_K-E_Kp",
            "reason": "" if split.status == "ok" else split.reason or "no matching measured fields",
            "points": [point.to_dict() for point in split.points],
        })
    enriched["splitting"] = records
    enriched["splitting_status"] = "ok" if any(status == "ok" for status in statuses) else "insufficient"
    if enriched["splitting_status"] != "ok":
        enriched["splitting_reason"] = "no exact shared measured B values"
    return enriched
