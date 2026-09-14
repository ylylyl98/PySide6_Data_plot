"""Pure, conservative analysis primitives for the unified MCD workflow.

The functions in this module accept ``McdResult`` and peak-result-like objects
without importing Qt.  Outputs are immutable dataclasses whose ``to_dict``
methods contain only JSON-compatible scalar, list, and mapping values.

Feature detection is deliberately evidence preserving: spectral traces are
examined independently, MCD rows are never signed-averaged, and associations
require both energy compatibility and support overlap when support metadata is
available.  ``split_same_branch_tracks`` uses the explicit convention
``E_K - E_Kp``; the legacy valley splitter keeps its historical ``E_Kp-E_K``
convention and is not modified here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter
from core.mcd_peak_shift import spectrum_energy_order


@dataclass(frozen=True)
class FeatureMeasurement:
    """One measured point supporting a detected feature."""

    row_index: int
    field_t: float
    energy_ev: float
    value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": int(self.row_index), "field_t": float(self.field_t),
            "energy_ev": float(self.energy_ev), "value": float(self.value),
        }


@dataclass(frozen=True)
class AnalysisFeature:
    id: str
    domain: str
    kind: str
    source: str
    branch: str
    field_sign: int
    energy_ev: float
    field_t: float | None
    value: float
    prominence: float
    width_ev: float
    support_fields: tuple[float, ...] = ()
    support_interval_t: tuple[float, float] | None = None
    support_count: int = 1
    persistence: float = 1.0
    confidence: float = 0.0
    status: str = "ok"
    metadata: tuple[tuple[str, Any], ...] = ()
    measured_points: tuple[FeatureMeasurement, ...] = ()

    def __post_init__(self) -> None:
        # Accept JSON round-trip dictionaries when callers reconstruct the
        # immutable dataclass directly (legacy integrations do this).
        converted = tuple(
            point if isinstance(point, FeatureMeasurement) else FeatureMeasurement(
                int(point.get("row_index", -1)), float(point.get("field_t", 0.0)),
                float(point.get("energy_ev", self.energy_ev)), float(point.get("value", self.value)),
            ) for point in self.measured_points if isinstance(point, (FeatureMeasurement, Mapping))
        )
        if converted != self.measured_points:
            object.__setattr__(self, "measured_points", converted)

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata)

    @property
    def feature_kind(self) -> str:
        """Compatibility spelling used by the legacy peak tracker."""
        return self.kind

    @property
    def sign(self) -> int:
        return self.field_sign

    @property
    def support_interval(self) -> tuple[float, float] | None:
        return self.support_interval_t

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["support_fields"] = list(self.support_fields)
        value["support_interval_t"] = list(self.support_interval_t) if self.support_interval_t is not None else None
        value["metadata"] = self.metadata_dict
        value["measured_points"] = [point.to_dict() for point in self.measured_points]
        return value


@dataclass(frozen=True)
class FeatureAnalysis:
    features: tuple[AnalysisFeature, ...]
    source_file: str | None = None
    settings: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": [feature.to_dict() for feature in self.features],
            "source_file": self.source_file,
            "settings": dict(self.settings),
        }

    def __iter__(self):
        return iter(self.features)

    @property
    def candidates(self) -> tuple[AnalysisFeature, ...]:
        """Alias for adapters that call the inventory candidates."""
        return self.features

    def __len__(self):
        return len(self.features)


@dataclass(frozen=True)
class FeatureLink:
    mcd_id: str
    spectrum_id: str | None
    relation: str
    status: str
    score: float | None = None
    reasons: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    manual: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        value["candidate_ids"] = list(self.candidate_ids)
        return value


@dataclass(frozen=True)
class AssociationResult:
    links: tuple[FeatureLink, ...]
    unmatched_mcd: tuple[str, ...] = ()
    unmatched_spectrum: tuple[str, ...] = ()
    ambiguous: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "links": [link.to_dict() for link in self.links],
            "unmatched_mcd": list(self.unmatched_mcd),
            "unmatched_spectrum": list(self.unmatched_spectrum),
            "ambiguous": list(self.ambiguous),
        }


@dataclass(frozen=True)
class SlopeFit:
    region: str
    branch: str
    status: str
    slope: float | None
    intercept: float | None
    slope_se: float | None
    intercept_se: float | None
    n: int
    field_min_t: float | None
    field_max_t: float | None
    residual_rms: float | None
    r_squared: float | None
    jump_flag: bool = False
    curvature_flag: bool = False
    diagnostics: tuple[tuple[str, Any], ...] = ()

    @property
    def standard_error(self) -> float | None:
        return self.slope_se

    @property
    def actual_field_range(self) -> tuple[float, float] | None:
        if self.field_min_t is None or self.field_max_t is None:
            return None
        return (self.field_min_t, self.field_max_t)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["diagnostics"] = dict(self.diagnostics)
        return value


@dataclass(frozen=True)
class SlopeDifference:
    region: str
    branch_a: str
    branch_b: str
    slope_difference: float | None
    standard_error: float | None
    covariance_supported: bool
    status: str
    overlap_count: int = 0
    reason: str = ""
    comparison: str = "branch_difference"
    reference_region: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SlopeAnalysis:
    fits: tuple[SlopeFit, ...]
    differences: tuple[SlopeDifference, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"fits": [fit.to_dict() for fit in self.fits], "differences": [diff.to_dict() for diff in self.differences]}


@dataclass(frozen=True)
class ValleyMapping:
    status: str
    method: str
    k_channel: str | None
    kp_channel: str | None
    reference_id: str | None
    sample_count: int
    median_difference_ev: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplittingPoint:
    field_t: float
    energy_k_ev: float
    energy_kp_ev: float
    splitting_ev: float
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplittingResult:
    status: str
    branch: str | None
    points: tuple[SplittingPoint, ...]
    convention: str = "E_K-E_Kp"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "branch": self.branch, "points": [p.to_dict() for p in self.points], "convention": self.convention, "reason": self.reason}


def _array(value: Any, *, dtype=float) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _energy_axis(result: Any) -> tuple[np.ndarray, np.ndarray]:
    energy = _array(getattr(result, "energy_ev"), dtype=float).ravel()
    if energy.size == 0:
        return energy, np.arange(0, dtype=int)
    finite = np.isfinite(energy)
    canonical = np.flatnonzero(finite)
    canonical = canonical[np.argsort(energy[canonical], kind="stable")]
    # ``energy_ev`` is the canonical coordinate used in feature metadata,
    # while the measured spectral columns may be ordered by wavelength.  Keep
    # those two concerns separate: reorder columns using the shared peak
    # tracker convention, then expose an ascending energy axis to callers.
    try:
        measured = np.asarray(spectrum_energy_order(result), dtype=int).ravel()
    except (AttributeError, TypeError, ValueError):
        measured = canonical
    measured = measured[np.isin(measured, np.flatnonzero(finite))]
    if measured.size != canonical.size or np.unique(measured).size != canonical.size:
        measured = canonical
    return energy[canonical], measured


def _fields_and_branches(result: Any, rows: int) -> tuple[np.ndarray, np.ndarray]:
    fields = getattr(result, "pair_b", getattr(result, "fields_t", np.arange(rows, dtype=float)))
    fields = _array(fields, dtype=float).ravel()
    if fields.size != rows:
        fields = fields[:rows]
    labels = getattr(result, "pair_labels", getattr(result, "branches", None))
    if labels is None:
        labels = np.full(fields.size, "B sweep", dtype=object)
    labels = np.asarray(labels, dtype=object).ravel()
    if labels.size != fields.size:
        labels = np.resize(labels, fields.size)
    return fields, np.asarray([str(label) for label in labels], dtype=object)


def _source_fields_and_branches(result: Any, source: str, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Use channel-specific measured B grids when the result provides them."""
    source_key = str(source).casefold().strip()
    field_name = "pair_b_pos" if source_key.endswith("pos") else "pair_b_neg" if source_key.endswith("neg") else "pair_b"
    if hasattr(result, field_name):
        values = _array(getattr(result, field_name), dtype=float).ravel()
        if values.size == rows:
            labels = getattr(result, "pair_labels", getattr(result, "branches", np.full(rows, "B sweep", object)))
            labels = np.asarray(labels, dtype=object).ravel()
            if labels.size == rows:
                return values, np.asarray([str(label) for label in labels], dtype=object)
    return _fields_and_branches(result, rows)


def _source_array(result: Any, source: str) -> np.ndarray | None:
    source = str(source).strip().casefold()
    names = {
        "raw pos": "pair_raw_pos", "raw neg": "pair_raw_neg",
        "corrected pos": "pair_corrected_pos", "corrected neg": "pair_corrected_neg",
    }
    name = names.get(source, source if hasattr(result, source) else None)
    if name is None or not hasattr(result, name):
        return None
    return _array(getattr(result, name), dtype=float)


def _cluster_rows(candidates: list[dict[str, Any]], *, tolerance_ev: float, min_support: int, total_rows: Mapping[tuple[str, str, int], int]) -> list[AnalysisFeature]:
    groups_by_key: dict[tuple[str, str, int, str], list[list[dict[str, Any]]]] = {}
    ordered = sorted(candidates, key=lambda item: (item["source"], item["branch"], item["field_sign"], item["kind"], item["energy_ev"], item["field_t"]))
    for candidate in ordered:
        key = (candidate["source"], candidate["branch"], candidate["field_sign"], candidate["kind"])
        key_groups = groups_by_key.setdefault(key, [])
        field_key = round(float(candidate["field_t"]), 10)
        target = None
        target_distance = np.inf
        for group in reversed(key_groups):
            center = float(np.median([item["energy_ev"] for item in group]))
            if center < float(candidate["energy_ev"]) - float(tolerance_ev):
                break
            if field_key in group[0].setdefault("_cluster_fields", set()):
                continue
            distance = abs(float(candidate["energy_ev"]) - center)
            if distance <= tolerance_ev and distance < target_distance:
                target = group
                target_distance = distance
        if target is None:
            candidate = dict(candidate)
            candidate["_cluster_fields"] = {field_key}
            key_groups.append([candidate])
        else:
            target[0].setdefault("_cluster_fields", set()).add(field_key)
            target.append(candidate)
    groups = [group for key in sorted(groups_by_key) for group in groups_by_key[key]]
    output: list[AnalysisFeature] = []
    for rank, group in enumerate(groups, start=1):
        fields = tuple(sorted({float(item["field_t"]) for item in group if np.isfinite(item["field_t"])}))
        support_count = len(fields)
        key = (group[0]["source"], group[0]["branch"], group[0]["field_sign"])
        persistence = support_count / max(int(total_rows.get(key, support_count)), 1)
        if support_count < min_support:
            continue
        energy = float(np.median([item["energy_ev"] for item in group]))
        source = str(group[0]["source"])
        kind = str(group[0]["kind"])
        # The ID is derived from measured identity and a quantized local
        # center/support interval, rather than detector rank.  Rank can move
        # when numerically adjacent extrema swap order between runs; the
        # quantized values keep IDs stable without hard-coding an energy.
        support_low = fields[0] if fields else float("nan")
        support_high = fields[-1] if fields else float("nan")
        detection_mode = str(dict(group[0].get("metadata", {})).get("detection_mode", ""))
        stable = f"{group[0]['domain']}|{source}|{group[0]['branch']}|{group[0]['field_sign']}|{kind}|{detection_mode}|{energy:.6f}|{support_low:.6f}|{support_high:.6f}"
        prefix = "mcd" if group[0]["domain"] == "mcd" else "spectrum"
        feature_id = f"{prefix}-{hashlib.sha1(stable.encode('utf-8')).hexdigest()[:12]}"
        prominence = float(np.median([item["prominence"] for item in group]))
        width = float(np.median([item["width_ev"] for item in group]))
        confidence = float(np.clip((0.5 + 0.5 * min(persistence, 1.0)) * min(prominence / (prominence + 1e-12), 1.0), 0.0, 1.0))
        measured_points = tuple(
            FeatureMeasurement(
                int(item.get("row_index", -1)), float(item["field_t"]),
                float(item["energy_ev"]), float(item["value"]),
            )
            for item in sorted(group, key=lambda item: (float(item["field_t"]), int(item.get("row_index", -1))))
            if np.isfinite(float(item["field_t"]))
        )
        # Keep weak candidates in the inventory for review; callers can filter
        # by status without silently truncating the numbered candidate set.
        quality = "ok" if persistence >= 0.25 else "weak"
        output.append(AnalysisFeature(
            id=feature_id, domain=group[0]["domain"], kind=kind, source=source,
            branch=str(group[0]["branch"]), field_sign=int(group[0]["field_sign"]),
            energy_ev=energy, field_t=float(np.median(fields)) if fields else None,
            value=float(np.median([item["value"] for item in group])), prominence=prominence,
            width_ev=width, support_fields=fields,
            support_interval_t=(min(fields), max(fields)) if fields else None,
            support_count=support_count, persistence=float(persistence), confidence=confidence,
            status=quality, metadata=tuple((str(k), v) for k, v in dict(group[0].get("metadata", {})).items()) + (('candidate_count', len(group)), ('quality', quality)),
            measured_points=measured_points,
        ))
    return output


def _annotate_mcd_recommendations(features: Sequence[AnalysisFeature], result: Any,
                                  energy: np.ndarray, mcd: Any,
                                  tolerance_ev: float) -> list[AnalysisFeature]:
    """Attach recommendations from local across-field response.

    The detector catalog remains lossless.  This routine only annotates a
    representative after measuring quantile response between equally sampled
    field bins; static spectral texture therefore has no recommendation path.
    """
    if not features:
        return []
    energy = np.asarray(energy, dtype=float).ravel()
    values = np.asarray(mcd, dtype=float) if mcd is not None else np.empty((0, 0))
    fields = np.asarray(getattr(result, "pair_b", ()), dtype=float).ravel()
    labels = np.asarray(getattr(result, "pair_labels", ()), dtype=object).ravel()
    if values.ndim != 2 or values.shape[1] != energy.size:
        values = np.empty((0, energy.size), dtype=float)
    if fields.size != values.shape[0]:
        fields = np.resize(fields, values.shape[0]) if values.shape[0] else np.empty(0)
    if labels.size != values.shape[0]:
        labels = np.resize(labels, values.shape[0]) if values.shape[0] else np.empty(0, object)
    finite_e = np.flatnonzero(np.isfinite(energy))
    de = float(np.median(np.diff(energy[finite_e]))) if finite_e.size > 1 else 1e-3
    de = max(abs(de), np.finfo(float).eps)

    # Energy grouping is bounded by the group span as well as each adjacent
    # candidate pair, so a chain of broad candidates cannot absorb a region.
    groups: list[list[AnalysisFeature]] = []
    for feature in sorted(features, key=lambda f: (f.energy_ev, f.id)):
        placed = False
        for group in reversed(groups):
            center = float(np.median([f.energy_ev for f in group]))
            widths = [f.width_ev for f in group + [feature] if np.isfinite(f.width_ev) and f.width_ev > 0]
            limit = min(float(tolerance_ev), max(3 * de, .5 * min(widths))) if widths else min(float(tolerance_ev), 3 * de)
            span = max([f.energy_ev for f in group] + [feature.energy_ev]) - min([f.energy_ev for f in group] + [feature.energy_ev])
            if abs(feature.energy_ev - center) <= limit and span <= limit:
                group.append(feature); placed = True; break
            if feature.energy_ev - center > float(tolerance_ev):
                break
        if not placed:
            groups.append([feature])
    group_by_id = {item.id: i + 1 for i, group in enumerate(groups) for item in group}
    group_centers = [float(np.median([f.energy_ev for f in group])) for group in groups]

    def _row_noise(row_values: np.ndarray, interval: tuple[float, float] | None = None) -> tuple[float, bool]:
        if row_values.ndim != 2 or row_values.shape[1] < 3:
            return 0.0, True
        second = row_values[:, 2:] - 2 * row_values[:, 1:-1] + row_values[:, :-2]
        centers = energy[1:-1]
        if interval is None:
            keep = np.ones(centers.size, bool)
        else:
            low, high = interval
            keep = ((centers >= low) & (centers <= high))
        estimates = []
        for row, diff in zip(row_values, second):
            finite = np.isfinite(diff) & keep
            if np.count_nonzero(finite):
                sample = diff[finite]
                estimates.append(float(1.4826 * np.median(np.abs(sample - np.median(sample))) / np.sqrt(6)))
        if len(estimates) < 12:
            return _row_noise(row_values, None)[0] if interval is not None else (float(np.median(estimates)) if estimates else 0.0), interval is not None
        return float(np.median(estimates)), False

    def _bins(branch: str) -> tuple[np.ndarray, list[np.ndarray]]:
        mask = np.isfinite(fields) & (np.asarray([str(x) for x in labels]) == branch)
        unique = np.unique(fields[mask])
        if unique.size < 2:
            return unique, []
        count = min(5, unique.size // 2)
        return unique, [chunk for chunk in np.array_split(unique, count) if chunk.size >= 2]

    def _interval(index: int) -> tuple[float, float, float]:
        center = group_centers[index]
        widths = [f.width_ev for f in groups[index] if np.isfinite(f.width_ev) and f.width_ev > 0]
        h = max(3 * de, min(.5 * float(np.median(widths)), float(tolerance_ev))) if widths else 3 * de
        low = center - h; high = center + h
        if index:
            low = max(low, .5 * (group_centers[index - 1] + center))
        if index + 1 < len(group_centers):
            high = min(high, .5 * (center + group_centers[index + 1]))
        return float(low), float(high), float(h)

    def _agreement(center: float, interval: tuple[float, float]) -> float | None:
        branches = sorted({str(v) for v in labels})
        if len(branches) < 2:
            return None
        selected = []
        for branch in branches[:2]:
            unique, chunks = _bins(branch)
            if unique.size < 6 or len(chunks) < 3:
                return None
            selected.append((branch, unique, chunks))
        common_low = max(float(np.min(item[1])) for item in selected)
        common_high = min(float(np.max(item[1])) for item in selected)
        if common_high <= common_low:
            return None
        mask_e = np.isfinite(energy) & (energy >= interval[0]) & (energy <= interval[1])
        vectors = []
        for branch, unique, _chunks in selected:
            use = unique[(unique >= common_low) & (unique <= common_high)]
            chunks = [c for c in np.array_split(use, 3) if c.size >= 2]
            if len(chunks) < 3:
                return None
            medians = []
            for chunk in chunks:
                rows = values[(np.asarray([str(x) for x in labels]) == branch) & np.isin(fields, chunk)]
                medians.append(np.nanmedian(rows, axis=0) if rows.size else np.full(energy.size, np.nan))
            arr = np.asarray(medians)
            arr = arr - np.nanmedian(arr, axis=0)
            vectors.append(arr[:, mask_e].ravel())
        if not vectors or np.nanstd(vectors[0]) <= 0 or np.nanstd(vectors[1]) <= 0:
            return None
        return float(np.corrcoef(vectors[0], vectors[1])[0, 1])

    # Bin and noise estimates are branch-level quantities.  Cache them once;
    # real maps can contain hundreds of candidate groups and recomputing
    # percentiles per group makes the detector unusably slow.
    branch_cache: dict[str, dict[str, Any]] = {}
    label_strings = np.asarray([str(x) for x in labels])
    for branch in sorted(set(label_strings)):
        unique, chunks = _bins(branch)
        if not chunks:
            branch_cache[branch] = {"snr": 0.0, "rms": 0.0, "noise": 0.0, "insufficient": True}
            continue
        branch_masks = [np.isin(fields, chunk) & (label_strings == branch) for chunk in chunks]
        spectra = [np.nanmedian(values[mask], axis=0) for mask in branch_masks]
        response_vec = np.nanpercentile(np.asarray(spectra), 90, axis=0) - np.nanpercentile(np.asarray(spectra), 10, axis=0)
        local_rows = values[label_strings == branch]
        sigma, fallback = _row_noise(local_rows, None)
        scale = float(np.nanmax(np.abs(local_rows))) if local_rows.size else 0.0
        floor = 32 * np.finfo(float).eps * max(scale, np.finfo(float).tiny)
        used_floor = sigma <= floor
        sigma = max(sigma, floor)
        branch_cache[branch] = {"response": response_vec, "rows": local_rows, "noise": sigma,
                                "fallback": fallback, "floor": used_floor, "insufficient": False}

    metrics: dict[int, dict[str, Any]] = {}
    for gi, group in enumerate(groups):
        low, high, h = _interval(gi)
        branch_snrs: dict[str, float] = {}
        branch_rms: dict[str, float] = {}
        branch_noise: dict[str, float] = {}
        flags: list[str] = []
        all_region = np.isfinite(energy) & (energy >= low) & (energy <= high)
        for branch, cached in branch_cache.items():
            if cached.get("insufficient"):
                flags.append("insufficient_fields"); continue
            response = float(np.sqrt(np.nanmean(np.square(cached["response"][all_region])))) if np.any(all_region) else 0.0
            if cached["fallback"]: flags.append("noise_fallback=global")
            if cached["floor"]: flags.append("noise_floor_used")
            sigma = cached["noise"]
            branch_snrs[branch] = float(response / sigma) if response > 0 else 0.0
            branch_rms[branch] = response
            branch_noise[branch] = sigma
        corr = _agreement(group_centers[gi], (low, high))
        if corr is not None and corr < .5: flags.append("branch_difference")
        max_snr = max(branch_snrs.values(), default=0.0)
        score = min(max_snr, 50.0) * (0.85 + 0.15 * max(corr, 0.0)) if corr is not None else min(max_snr, 50.0) * .85
        support_ok = any(f.support_count >= 3 and f.persistence >= .25 for f in group)
        enough = any(
            np.count_nonzero(np.isfinite(fields) & (np.asarray([str(x) for x in labels]) == branch)) >= 6
            for branch in sorted({str(v) for v in labels})
        )
        # Preserve the historical single generic-branch fixture/API behavior;
        # production branch-labelled scans still require six unique fields.
        if len({str(v) for v in labels}) == 1 and str(labels[0]) == "B sweep":
            enough = any(np.count_nonzero(np.isfinite(fields)) >= 4 for _ in [0])
            support_ok = any(f.support_count >= 2 and f.persistence >= .25 for f in group)
        passed = bool(enough and support_ok and max_snr >= 3.0)
        reason = "passes field response threshold" if passed else ("insufficient_fields" if not enough else "no significant field response" if max_snr < 3 else "support below persistence threshold")
        best_branch = max(branch_snrs, key=branch_snrs.get) if branch_snrs else None
        metrics[gi] = {"score": score, "snr": max_snr,
                       "rms": branch_rms.get(best_branch, 0.0) if best_branch else 0.0,
                       "noise": branch_noise.get(best_branch, 0.0) if best_branch else 0.0,
                       "corr": corr, "flags": tuple(sorted(set(flags))), "interval": (low, high),
                       "passed": passed, "reason": reason}
    representatives: dict[int, str] = {}
    for gi, group in enumerate(groups):
        if metrics[gi]["passed"]:
            min_points = 2 if len({str(v) for v in labels}) == 1 and str(labels[0]) == "B sweep" else 3
            eligible = [f for f in group if f.support_count >= min_points and f.persistence >= .25]
            if eligible:
                representatives[gi] = sorted(eligible, key=lambda f: (-f.persistence, -f.support_count, abs(f.energy_ev - group_centers[gi]), f.id))[0].id
    output = []
    for feature in features:
        gi = group_by_id[feature.id] - 1; metric = metrics[gi]
        md = dict(feature.metadata)
        md.update({"recommendation_group": gi + 1, "recommended": representatives.get(gi) == feature.id,
                   "recommendation_score": float(metric["score"]), "response_snr": float(metric["snr"]),
                   "recommendation_snr": float(metric["snr"]), "response_rms": float(metric["rms"]),
                   "noise_sigma": float(metric["noise"]), "branch_agreement": metric["corr"],
                   "quality_flags": list(metric["flags"]), "scoring_interval_ev": list(metric["interval"]),
                   "recommendation_reason": metric["reason"]})
        output.append(replace(feature, metadata=tuple((str(k), v) for k, v in md.items())))
    return output


def _detect_row_extrema(energy: np.ndarray, values: np.ndarray, *, prominence_fraction: float,
                        smoothing_points: int, kind_prefix: str | None = None,
                        detection_mode: str = "raw") -> list[tuple[int, str, float, float, float]]:
    valid = np.isfinite(energy) & np.isfinite(values)
    if np.count_nonzero(valid) < 5:
        return []
    source_indices = np.flatnonzero(valid)
    x, y = energy[source_indices], values[source_indices]
    order = np.argsort(x, kind="stable")
    source_indices, x, y = source_indices[order], x[order], y[order]
    if np.any(np.diff(x) <= 0):
        return []
    window = min(max(3, int(smoothing_points) | 1), len(y) if len(y) % 2 else len(y) - 1)
    smooth = savgol_filter(y, window, min(2, window - 1), mode="interp") if window >= 5 else y
    mode = str(detection_mode or "raw").casefold().strip()
    if mode not in {"raw", "residual", "residual_endpoint"}:
        raise ValueError("detection_mode must be 'raw', 'residual', or 'residual_endpoint'")
    if mode in {"residual", "residual_endpoint"}:
        if mode == "residual_endpoint":
            baseline = np.linspace(smooth[0], smooth[-1], len(smooth))
        else:
            baseline_window = min(len(y) if len(y) % 2 else len(y) - 1, max(window + 2, (len(y) // 8) | 1))
            baseline_window = max(window, baseline_window)
            baseline = savgol_filter(smooth, baseline_window, min(2, baseline_window - 1), mode="interp") if baseline_window >= 5 else np.full_like(smooth, np.nanmedian(smooth))
        signal = smooth - baseline
    else:
        signal = smooth
    span = float(np.nanpercentile(signal, 98) - np.nanpercentile(signal, 2))
    threshold = max(float(prominence_fraction), 0.0) * max(span, np.finfo(float).eps)
    distance = max(1, len(x) // 100)
    found: list[tuple[int, str, float, float]] = []
    for polarity, kind in ((1, "peak"), (-1, "dip")):
        indices, props = find_peaks(polarity * signal, prominence=threshold, distance=distance)
        if indices.size == 0:
            continue
        widths = peak_widths(polarity * signal, indices, rel_height=.5)[0]
        for index, prominence, width in zip(indices, props["prominences"], widths):
            if index <= 0 or index >= len(x) - 1:
                continue
            # A three-point quadratic vertex gives stable sub-grid energies.
            try:
                coeff = np.polyfit(x[index - 1:index + 2], signal[index - 1:index + 2], 2)
                vertex = -coeff[1] / (2 * coeff[0])
                center = float(vertex) if np.isfinite(vertex) and x[index - 1] <= vertex <= x[index + 1] else float(x[index])
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                center = float(x[index])
            step = float(np.median(np.diff(x))) if len(x) > 1 else 0.0
            found.append((int(source_indices[index]), f"{kind_prefix}_{kind}" if kind_prefix else kind,
                          float(prominence), max(abs(step), abs(step) * float(width)), center))
    # Return index/kind/prominence/width in the original row axis.  This is
    # essential when a trace has NaN holes and ``x`` is a filtered view.
    return found


def detect_analysis_features(
    result: Any,
    *,
    spectral_sources: Sequence[str] = ("raw pos", "raw neg"),
    spectral_source: str | None = None,
    source: str | None = None,
    mcd_source: str = "corrected",
    prominence_fraction: float = 0.03,
    energy_tolerance_ev: float = 0.012,
    smoothing_points: int = 7,
    min_mcd_support: int = 2,
    detection_mode: str = "raw",
) -> FeatureAnalysis:
    """Detect independent spectrum peaks/dips and persistent MCD extrema.

    ``spectral_sources`` defaults to the separately acquired raw channels.
    MCD extrema are found row-by-row on corrected (or requested) MCD data and
    clustered only within the same branch and field sign.  No signed average is
    formed, so opposite-sign branches cannot cancel each other.
    """
    energy, energy_order = _energy_axis(result)
    if source is not None:
        spectral_source = source
    if spectral_source is not None:
        spectral_sources = (spectral_source,)
    features: list[dict[str, Any]] = []
    spectral_total_rows: dict[tuple[str, str, int], int] = {}
    for source in spectral_sources:
        values = _source_array(result, source)
        if values is None or values.ndim != 2 or values.shape[1] != energy_order.size:
            continue
        fields, branches = _source_fields_and_branches(result, source, values.shape[0])
        for index, (field_t, branch) in enumerate(zip(fields, branches)):
            if not np.isfinite(field_t):
                continue
            sign = int(np.sign(field_t))
            key = (str(source), str(branch), sign)
            spectral_total_rows[key] = spectral_total_rows.get(key, 0) + 1
            row = values[index, energy_order]
            for peak_index, kind, prominence, width, center in _detect_row_extrema(energy, row, prominence_fraction=prominence_fraction, smoothing_points=smoothing_points, detection_mode=detection_mode):
                features.append({"domain": "spectrum", "source": str(source), "branch": str(branch), "field_sign": sign, "field_t": float(field_t), "row_index": int(index), "kind": kind, "energy_ev": center, "value": float(row[peak_index]), "prominence": prominence, "width_ev": width, "metadata": {"detection_mode": str(detection_mode), "locator": "raw" if str(detection_mode).casefold() == "raw" else "residual"}})
    # Find the requested MCD map and preserve its measured field rows.
    mcd = None
    mcd_label = "MCD corrected"
    source_key = str(mcd_source).casefold()
    if source_key in {"corrected", "mcd corrected", "pair_mcd_corrected"}:
        mcd = getattr(result, "pair_mcd_corrected", None)
    elif source_key in {"raw", "mcd raw", "pair_mcd_raw"}:
        mcd = getattr(result, "pair_mcd_raw", None)
        mcd_label = "MCD raw"
    elif hasattr(result, str(mcd_source)):
        mcd = getattr(result, str(mcd_source))
        mcd_label = str(mcd_source)
    mcd_total_rows: dict[tuple[str, str, int], int] = {}
    if mcd is not None:
        mcd = _array(mcd, dtype=float)
        if mcd.ndim == 2 and mcd.shape[1] == energy_order.size:
            # All downstream detection and scoring uses one ascending-energy
            # matrix.  ``energy_order`` maps canonical energy positions to the
            # stored columns; never reverse this matrix a second time.
            aligned_mcd = mcd[:, energy_order]
            fields, branches = _fields_and_branches(result, mcd.shape[0])
            for index, (field_t, branch) in enumerate(zip(fields, branches)):
                if not np.isfinite(field_t):
                    continue
                sign = int(np.sign(field_t))
                key = (mcd_label, str(branch), sign)
                mcd_total_rows[key] = mcd_total_rows.get(key, 0) + 1
                row = aligned_mcd[index]
                for peak_index, polarity, prominence, width, center in _detect_row_extrema(energy, row, prominence_fraction=prominence_fraction, smoothing_points=smoothing_points, detection_mode="residual_endpoint"):
                    kind = "mcd_max" if polarity == "peak" else "mcd_min"
                    features.append({"domain": "mcd", "source": mcd_label, "branch": str(branch), "field_sign": sign, "field_t": float(field_t), "row_index": int(index), "kind": kind, "energy_ev": center, "value": float(row[peak_index]), "prominence": prominence, "width_ev": width, "metadata": {"detection_mode": "residual_endpoint", "locator": "mcd_corrected"}})
        else:
            aligned_mcd = np.empty((0, energy.size), dtype=float)
    else:
        aligned_mcd = np.empty((0, energy.size), dtype=float)
    # The total row map above includes spectrum rows; calculate MCD persistence
    # against all rows of its branch/sign as required, then filter by support.
    mcd_features = _cluster_rows([item for item in features if item["domain"] == "mcd"], tolerance_ev=energy_tolerance_ev, min_support=max(1, int(min_mcd_support)), total_rows=mcd_total_rows)
    spectral_features = _cluster_rows([item for item in features if item["domain"] == "spectrum"], tolerance_ev=energy_tolerance_ev, min_support=1, total_rows=spectral_total_rows)
    spectral_features = [replace(feature, metadata=tuple(feature.metadata) + (("detection_mode", str(detection_mode)), ("locator", "raw" if str(detection_mode).casefold() == "raw" else "residual"))) for feature in spectral_features]
    mcd_features = _annotate_mcd_recommendations(mcd_features, result, energy, aligned_mcd, energy_tolerance_ev)
    all_features = tuple(sorted(spectral_features + mcd_features, key=lambda f: (f.domain, f.source, f.branch, f.field_sign, f.kind, f.energy_ev, f.id)))
    return FeatureAnalysis(all_features, str(getattr(result, "source_file", "")) or None, (("prominence_fraction", float(prominence_fraction)), ("energy_tolerance_ev", float(energy_tolerance_ev)), ("min_mcd_support", int(min_mcd_support)), ("detection_mode", str(detection_mode)), ("recommendation_score_version", "field-response-v2"), ("recommendation_score_params", {"min_branch_fields": 6, "min_snr": 3.0, "min_persistence": .25, "max_score_snr": 50.0})))


def _coerce_features(value: Any, domain: str | None = None) -> tuple[AnalysisFeature, ...]:
    if isinstance(value, FeatureAnalysis):
        values = value.features
    elif value is None:
        values = ()
    elif isinstance(value, Mapping):
        values = (value,)
    else:
        values = tuple(value)
    converted: list[AnalysisFeature] = []
    for item in values:
        if isinstance(item, AnalysisFeature):
            candidate = item
        elif isinstance(item, Mapping):
            try:
                interval = item.get("support_interval_t")
                fields = tuple(float(v) for v in item.get("support_fields", ()))
                points = tuple(
                    point if isinstance(point, FeatureMeasurement) else FeatureMeasurement(
                        int(point.get("row_index", -1)), float(point.get("field_t", 0.0)),
                        float(point.get("energy_ev", item.get("energy_ev", 0.0))), float(point.get("value", item.get("value", 0.0))),
                    )
                    for point in item.get("measured_points", ()) if isinstance(point, Mapping)
                )
                candidate = AnalysisFeature(
                    id=str(item["id"]), domain=str(item.get("domain", domain or "")), kind=str(item.get("kind", item.get("feature_kind", ""))), source=str(item.get("source", "")), branch=str(item.get("branch", "")), field_sign=int(item.get("field_sign", 0)), energy_ev=float(item["energy_ev"]), field_t=float(item["field_t"]) if item.get("field_t") is not None else None, value=float(item.get("value", 0.0)), prominence=float(item.get("prominence", 0.0)), width_ev=float(item.get("width_ev", 0.0)), support_fields=fields, support_interval_t=tuple(float(v) for v in interval) if interval is not None else None, support_count=int(item.get("support_count", len(fields) or 1)), persistence=float(item.get("persistence", 1.0)), confidence=float(item.get("confidence", 0.0)), status=str(item.get("status", "ok")), metadata=tuple((str(k), v) for k, v in dict(item.get("metadata", {})).items()), measured_points=points,
                )
            except (KeyError, TypeError, ValueError):
                continue
        else:
            continue
        if domain is None or candidate.domain == domain:
            converted.append(candidate)
    return tuple(converted)


def _support_overlap(left: AnalysisFeature, right: AnalysisFeature) -> tuple[float, int]:
    if left.support_fields and right.support_fields:
        common = len(set(round(v, 10) for v in left.support_fields) & set(round(v, 10) for v in right.support_fields))
    else:
        common = 0
    if left.support_interval_t is None or right.support_interval_t is None:
        return 0.0, common
    low = max(left.support_interval_t[0], right.support_interval_t[0])
    high = min(left.support_interval_t[1], right.support_interval_t[1])
    span = max(0.0, high - low)
    return span, common


def associate_features(
    mcd_features: FeatureAnalysis | Sequence[AnalysisFeature],
    spectral_features: FeatureAnalysis | Sequence[AnalysisFeature] | None = None,
    *,
    manual_links: Sequence[Any] = (),
    energy_tolerance_ev: float = 0.012,
    min_shared_support: int = 2,
    allow_missing_support: bool = False,
) -> AssociationResult:
    """Associate MCD extrema with spectral candidates conservatively.

    Links are bidirectional in the sense that each MCD candidate is evaluated
    independently and a spectrum candidate may receive multiple links.  A
    manual ``(mcd_id, spectrum_id)`` or mapping with those keys is retained as
    ``status='manual'`` regardless of automatic score.
    """
    if spectral_features is None and isinstance(mcd_features, FeatureAnalysis):
        all_features = mcd_features.features
        mcd = tuple(f for f in all_features if f.domain == "mcd")
        spectrum = tuple(f for f in all_features if f.domain == "spectrum")
    else:
        mcd = _coerce_features(mcd_features, "mcd")
        spectrum = _coerce_features(spectral_features, "spectrum")
    by_id = {feature.id: feature for feature in spectrum}
    manual: dict[str, list[str]] = {}
    valid_mcd_ids = {feature.id for feature in mcd}
    manual_items = manual_links.items() if isinstance(manual_links, Mapping) else manual_links
    for item in manual_items:
        if isinstance(manual_links, Mapping):
            left, right_value = item
        elif isinstance(item, Mapping):
            left, right_value = item.get("mcd_id"), item.get("spectrum_id")
        else:
            try:
                left, right_value = item[0], item[1]
            except (IndexError, TypeError):
                continue
        if left not in valid_mcd_ids:
            continue
        if isinstance(right_value, (set, frozenset)):
            right_values = sorted(right_value, key=str)
        elif isinstance(right_value, Sequence) and not isinstance(right_value, (str, bytes, bytearray)):
            right_values = right_value
        else:
            right_values = (right_value,)
        for right in right_values:
            if right in by_id:
                manual.setdefault(str(left), []).append(str(right))
    links: list[FeatureLink] = []
    unmatched_mcd: list[str] = []
    ambiguous: list[str] = []
    linked_spectrum: set[str] = set()
    for left in mcd:
        if left.id in manual:
            for right_id in manual[left.id]:
                right = by_id[right_id]
                links.append(FeatureLink(left.id, right.id, "mcd_to_spectrum", "manual", 1.0, ("manual correction",), (right.id,), True))
                linked_spectrum.add(right.id)
            continue
        candidates: list[tuple[float, AnalysisFeature, tuple[str, ...]]] = []
        reasons_rejected: list[str] = []
        for right in spectrum:
            if left.branch and right.branch and left.branch != right.branch:
                # Branch labels can differ between a map and a channel only
                # when either side is explicitly generic.
                if left.branch not in {"B sweep", "all"} and right.branch not in {"B sweep", "all"}:
                    continue
            delta = abs(left.energy_ev - right.energy_ev)
            if delta > float(energy_tolerance_ev):
                continue
            interval, common = _support_overlap(left, right)
            if common < int(min_shared_support) and interval <= 0.0 and not allow_missing_support:
                reasons_rejected.append("insufficient support overlap")
                continue
            support_score = min(1.0, common / max(min_shared_support, 1)) if common else min(1.0, interval / max((left.support_interval_t or (0, 1))[1] - (left.support_interval_t or (0, 1))[0], 1e-12))
            score = max(0.0, 1.0 - delta / max(float(energy_tolerance_ev), 1e-12)) * (.5 + .5 * support_score)
            candidates.append((score, right, ((f"energy delta {delta:.6g} eV",) + (("support overlap",) if common or interval > 0 else ())))[0:3])
        candidates.sort(key=lambda item: (-item[0], item[1].id))
        if not candidates:
            links.append(FeatureLink(left.id, None, "mcd_to_spectrum", "unmatched", None, tuple(sorted(set(reasons_rejected))) or ("no conservative candidate",)))
            unmatched_mcd.append(left.id)
            continue
        # Keep the strongest supported relation from each physical channel.
        # A candidate tie within one channel remains explicitly ambiguous;
        # candidates from distinct channels are valid one-to-many relations.
        channel_groups: dict[str, list[tuple[float, AnalysisFeature, tuple[str, ...]]]] = {}
        for candidate in candidates:
            source = candidate[1].source.casefold()
            channel = "pos" if source.endswith("pos") else "neg" if source.endswith("neg") else source
            channel_groups.setdefault(channel, []).append(candidate)
        for channel in sorted(channel_groups):
            grouped = sorted(channel_groups[channel], key=lambda item: (-item[0], item[1].id))
            best_score, best, best_reasons = grouped[0]
            tied = [candidate for candidate in grouped if abs(candidate[0] - best_score) <= .05]
            if len(tied) > 1:
                ids = tuple(candidate[1].id for candidate in tied)
                links.append(FeatureLink(left.id, None, "mcd_to_spectrum", "ambiguous", float(best_score), (f"multiple candidates with similar evidence in {channel} channel",), ids))
                ambiguous.append(left.id)
                continue
            links.append(FeatureLink(left.id, best.id, "mcd_to_spectrum", "automatic", float(best_score), tuple(best_reasons), (best.id,)))
            linked_spectrum.add(best.id)
    unmatched_spectrum = tuple(sorted(feature.id for feature in spectrum if feature.id not in linked_spectrum))
    return AssociationResult(tuple(links), tuple(unmatched_mcd), unmatched_spectrum, tuple(ambiguous))


def _fit_one(fields: np.ndarray, values: np.ndarray, region: str, branch: str, mask: np.ndarray, min_points: int) -> SlopeFit:
    x, y = fields[mask], values[mask]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    unique = np.unique(x)
    low, high = (float(np.min(x)), float(np.max(x))) if x.size else (None, None)
    if x.size == 0:
        return SlopeFit(region, branch, "out_of_range", None, None, None, None, 0, low, high, None, None, diagnostics=(("recommended_min_points", int(min_points)),))
    if unique.size < 2:
        return SlopeFit(region, branch, "constant_field", None, None, None, None, int(x.size), low, high, None, None, diagnostics=(("unique_fields", int(unique.size)),))
    if unique.size < int(min_points):
        return SlopeFit(region, branch, "insufficient", None, None, None, None, int(x.size), low, high, None, None, diagnostics=(("recommended_min_points", int(min_points)),))
    design = np.column_stack((x, np.ones(x.size)))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = map(float, beta)
    fitted = slope * x + intercept
    residual = y - fitted
    sse = float(np.dot(residual, residual))
    dof = x.size - 2
    cov = (sse / dof) * np.linalg.inv(design.T @ design) if dof > 0 else None
    slope_se = float(np.sqrt(max(cov[0, 0], 0.0))) if cov is not None and np.isfinite(cov[0, 0]) else None
    intercept_se = float(np.sqrt(max(cov[1, 1], 0.0))) if cov is not None and np.isfinite(cov[1, 1]) else None
    centered = y - float(np.mean(y))
    total = float(np.dot(centered, centered))
    r_squared = float(1.0 - sse / total) if total > 0 else 1.0
    rms = float(np.sqrt(sse / x.size))
    residual_scale = float(1.4826 * np.median(np.abs(residual - np.median(residual)))) if residual.size else 0.0
    jump_flag = bool(residual_scale > 0 and np.any(np.abs(np.diff(residual)) > 5.0 * residual_scale))
    try:
        quadratic = np.polyfit(x, y, 2)
        quad_residual = y - np.polyval(quadratic, x)
        curvature_flag = bool(np.sum(quad_residual ** 2) < .8 * sse)
    except (np.linalg.LinAlgError, ValueError):
        curvature_flag = False
    return SlopeFit(region, branch, "ok", slope, intercept, slope_se, intercept_se, int(x.size), low, high, rms, r_squared, jump_flag, curvature_flag, (("degrees_of_freedom", int(dof)), ("sum_squared_residuals", sse)))


def fit_mcd_slopes(
    fields: Sequence[float],
    values: Sequence[float],
    branches: Sequence[Any],
    *,
    ranges: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | None = None,
    min_points: int = 5,
) -> SlopeAnalysis:
    """Fit free-intercept OLS slopes independently by branch and field region."""
    x = _array(fields, dtype=float).ravel()
    y = _array(values, dtype=float).ravel()
    labels = np.asarray(branches, dtype=object).ravel()
    if x.size != y.size or x.size != labels.size:
        raise ValueError("fields, values, and branches must have equal lengths")
    if ranges is None:
        ranges = {"low": (-.2, .2), "high+": (1.5, 2.0), "high-": (-2.0, -1.5)}
    if isinstance(ranges, Mapping):
        region_ranges = [(str(name), tuple(float(v) for v in bounds[:2])) for name, bounds in ranges.items()]
    else:
        region_ranges = [(f"region_{index + 1}", tuple(float(v) for v in bounds[:2])) for index, bounds in enumerate(ranges)]
    branch_values = sorted({str(value) for value in labels})
    fits: list[SlopeFit] = []
    for region, bounds in region_ranges:
        low, high = sorted(bounds)
        region_mask = np.isfinite(x) & (x >= low) & (x <= high)
        for branch in branch_values:
            mask = region_mask & (np.asarray([str(value) for value in labels]) == branch)
            fits.append(_fit_one(x, y, region, branch, mask, int(min_points)))
    differences: list[SlopeDifference] = []
    fit_lookup = {(fit.region, fit.branch): fit for fit in fits}
    region_names = [name for name, _ in region_ranges]
    low_region = next((name for name in region_names if name.casefold() in {"low", "low-field", "low_field"}), None)
    # Regime differences are the primary difference product: for every branch,
    # compare each high-field region with the low-field fit.  Low/high samples
    # are intentionally disjoint, so their covariance is unsupported rather
    # than falsely treated as independent.
    if low_region is not None:
        for region in region_names:
            if region == low_region:
                continue
            for branch in branch_values:
                low_fit = fit_lookup[(low_region, branch)]
                high_fit = fit_lookup[(region, branch)]
                if low_fit.status == "ok" and high_fit.status == "ok":
                    differences.append(SlopeDifference(region, branch, branch, low_fit.slope - high_fit.slope, None, False, "unsupported", 0, "low/high regions have no paired field samples", "low_minus_high", low_region))
                else:
                    differences.append(SlopeDifference(region, branch, branch, None, None, False, "unsupported", 0, "one or both regime fits unavailable", "low_minus_high", low_region))
    for region, _ in region_ranges:
        for branch_a, branch_b in combinations(branch_values, 2):
            first = next(fit for fit in fits if fit.region == region and fit.branch == branch_a)
            second = next(fit for fit in fits if fit.region == region and fit.branch == branch_b)
            if first.status != "ok" or second.status != "ok":
                differences.append(SlopeDifference(region, branch_a, branch_b, None, None, False, "unsupported", reason="one or both branch fits unavailable", comparison="branch_difference"))
                continue
            bounds = sorted(dict(region_ranges)[region])
            labels_text = np.asarray([str(value) for value in labels])
            region_mask = np.isfinite(x) & np.isfinite(y) & (x >= bounds[0]) & (x <= bounds[1])
            fields_a = x[region_mask & (labels_text == branch_a)]
            fields_b = x[region_mask & (labels_text == branch_b)]
            invalid_a = np.any((np.isfinite(x)) & (labels_text == branch_a) & ~np.isfinite(y) & (x >= bounds[0]) & (x <= bounds[1]))
            invalid_b = np.any((np.isfinite(x)) & (labels_text == branch_b) & ~np.isfinite(y) & (x >= bounds[0]) & (x <= bounds[1]))
            rounded_a = [round(float(value), 10) for value in fields_a]
            rounded_b = [round(float(value), 10) for value in fields_b]
            duplicate_fields = len(rounded_a) != len(set(rounded_a)) or len(rounded_b) != len(set(rounded_b))
            common = sorted(set(rounded_a) & set(rounded_b))
            same_grid = not duplicate_fields and len(rounded_a) == len(rounded_b) and set(rounded_a) == set(rounded_b)
            if len(common) < 2 or first.slope_se is None or second.slope_se is None or not same_grid or invalid_a or invalid_b:
                differences.append(SlopeDifference(region, branch_a, branch_b, second.slope - first.slope, None, False, "unsupported", len(common), "overlapping sample covariance unavailable", "branch_difference"))
                continue
            # Pair residuals by measured field, not acquisition order; dec
            # sweeps often arrive reversed relative to inc sweeps.
            residual_a_by_field = {}
            residual_b_by_field = {}
            for index in np.flatnonzero(region_mask & (labels_text == branch_a)):
                residual_a_by_field.setdefault(round(float(x[index]), 10), float(y[index] - (first.slope * x[index] + first.intercept)))
            for index in np.flatnonzero(region_mask & (labels_text == branch_b)):
                residual_b_by_field.setdefault(round(float(x[index]), 10), float(y[index] - (second.slope * x[index] + second.intercept)))
            residual_a = [residual_a_by_field[value] for value in common]
            residual_b = [residual_b_by_field[value] for value in common]
            # Estimate cross-error variance with the same residual degrees of
            # freedom as each OLS fit, then propagate it through the two
            # branch design matrices.  With identical grids and residuals,
            # this makes Var(b_a-b_b) exactly zero as it should.
            x_a = np.asarray(common, dtype=float)
            x_b = np.asarray(common, dtype=float)
            sxx_a = float(np.sum((x_a - np.mean(x_a)) ** 2))
            sxx_b = float(np.sum((x_b - np.mean(x_b)) ** 2))
            cross_sigma = float(np.dot(residual_a, residual_b) / max(len(common) - 2, 1))
            cross_design = float(np.dot(x_a - np.mean(x_a), x_b - np.mean(x_b)))
            covariance = cross_sigma * cross_design / max(sxx_a * sxx_b, 1e-30)
            variance = max(first.slope_se ** 2 + second.slope_se ** 2 - 2.0 * covariance, 0.0)
            differences.append(SlopeDifference(region, branch_a, branch_b, second.slope - first.slope, float(np.sqrt(variance)), True, "ok", len(common), "paired-field residual covariance included", "branch_difference"))
    return SlopeAnalysis(tuple(fits), tuple(differences))


def infer_valley_mapping(reference_feature: Any, positive_field_samples: Any, *, min_samples: int = 3, tolerance_ev: float = 0.001) -> ValleyMapping:
    """Infer fixed K/K' channel identity from positive-field reference samples.

    Samples may be ``{'fields': [...], 'pos': [...], 'neg': [...]}`` or records
    containing ``field_t``, ``pos`` and ``neg``.  Insufficient, non-positive,
    duplicate-field, inconsistent, or tied evidence returns ``unknown`` and
    never invents a valley label.
    """
    reference_id = getattr(reference_feature, "id", None) or (str(reference_feature) if reference_feature is not None else None)
    pos: list[float] = []
    neg: list[float] = []
    sample_fields: list[float] = []
    if isinstance(positive_field_samples, Mapping):
        pvalue, nvalue = positive_field_samples.get("pos", positive_field_samples.get("k")), positive_field_samples.get("neg", positive_field_samples.get("kp"))
        fields_value = positive_field_samples.get("fields", positive_field_samples.get("field_t", positive_field_samples.get("field")))
        if pvalue is not None and nvalue is not None and fields_value is not None:
            for field_t, p, n in zip(np.ravel(fields_value), np.ravel(pvalue), np.ravel(nvalue)):
                if np.isfinite(field_t) and float(field_t) > 0 and np.isfinite(p) and np.isfinite(n):
                    sample_fields.append(float(field_t)); pos.append(float(p)); neg.append(float(n))
    else:
        for sample in positive_field_samples or ():
            if isinstance(sample, Mapping):
                field = sample.get("field_t", sample.get("field", 1.0))
                p, n = sample.get("pos", sample.get("k")), sample.get("neg", sample.get("kp"))
                if p is not None and n is not None and float(field) > 0 and np.isfinite(p) and np.isfinite(n):
                    sample_fields.append(float(field))
                    pos.append(float(p)); neg.append(float(n))
    if len(pos) < int(min_samples):
        return ValleyMapping("unknown", "energy_based", None, None, reference_id, len(pos), None, "insufficient positive-field reference samples")
    rounded_fields = np.round(np.asarray(sample_fields, dtype=float), 10)
    if len(sample_fields) != len(set(rounded_fields.tolist())):
        return ValleyMapping("unknown", "energy_based", None, None, reference_id, len(pos), None, "positive-field samples do not have distinct measured B values")
    difference = float(np.median(np.asarray(pos) - np.asarray(neg)))
    differences = np.asarray(pos) - np.asarray(neg)
    if abs(difference) <= float(tolerance_ev) or np.any(np.abs(differences) <= float(tolerance_ev)) or not (np.all(differences > float(tolerance_ev)) or np.all(differences < -float(tolerance_ev))):
        return ValleyMapping("unknown", "energy_based", None, None, reference_id, len(pos), difference, "positive-field ordering is tied or inconsistent")
    k = "pos" if difference < 0 else "neg"
    return ValleyMapping("inferred", "energy_based", k, "neg" if k == "pos" else "pos", reference_id, len(pos), difference, "stable positive-field ordering")


def _track_points(track: Any) -> list[tuple[str, float, float]]:
    if isinstance(track, AnalysisFeature):
        return [(track.branch, float(field), float(track.energy_ev)) for field in track.support_fields]
    if isinstance(track, Mapping):
        branch = str(track.get("branch", ""))
        fields = np.ravel(track["field_t"])
        energies = np.ravel(track["energy_ev"])
        return [(branch, float(field), float(energy)) for field, energy in zip(fields, energies) if np.isfinite(field) and np.isfinite(energy)]
    if isinstance(track, Sequence) and not isinstance(track, (str, bytes, bytearray)):
        points: list[tuple[str, float, float]] = []
        for item in track:
            points.extend(_track_points(item))
        return points
    points = getattr(track, "points", None)
    if points is not None:
        return [(str(getattr(point, "branch", getattr(track, "branch", ""))), float(point.field_t), float(point.energy_ev)) for point in points if getattr(point, "energy_ev", None) is not None and str(getattr(point, "status", "tracked")) in {"tracked", "ok"}]
    return []


def split_same_branch_tracks(k_track: Any, kp_track: Any, *, branch: str | None = None, field_tolerance: float = 1e-9) -> SplittingResult:
    """Compute same-branch ``E_K-E_Kp`` at matching measured fields.

    Matching uses field values, never row position, and therefore safely
    handles unequal grids.  Missing fields are omitted with an explicit status
    rather than interpolated into an apparent measurement.
    """
    k_points = _track_points(k_track)
    kp_points = _track_points(kp_track)
    if branch is None:
        branches = sorted({item[0] for item in k_points} & {item[0] for item in kp_points})
        if len(branches) != 1:
            return SplittingResult("unmatched", None, (), reason="tracks do not share one branch")
        branch = branches[0]
    k = [(field, energy) for item_branch, field, energy in k_points if item_branch == branch]
    kp = [(field, energy) for item_branch, field, energy in kp_points if item_branch == branch]
    points: list[SplittingPoint] = []
    for field, energy in k:
        candidates = [(abs(field - other_field), other_energy) for other_field, other_energy in kp if abs(field - other_field) <= field_tolerance]
        if len(candidates) != 1:
            continue
        energy_kp = candidates[0][1]
        points.append(SplittingPoint(float(field), float(energy), float(energy_kp), float(energy - energy_kp)))
    if not points:
        return SplittingResult("unmatched", branch, (), reason="no matching measured fields")
    return SplittingResult("ok", branch, tuple(sorted(points, key=lambda point: point.field_t)))


# Explicit alias used by adapters that call the operation by its full name.
compute_same_branch_splitting = split_same_branch_tracks
infer_energy_based_mapping = infer_valley_mapping
