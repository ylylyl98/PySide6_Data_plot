"""Strict DRR baseline eligibility and deterministic recommendations.

The catalog already caches each CSV's spectral header in :class:`DrrSource`.
This module deliberately uses that cached axis and never loads spectra while a
picker is being refreshed.  Acquisition conditions are parsed conservatively
from the filename conventions used by the instrument exports; an absent
condition is unknown and therefore cannot make a candidate rank as a match.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from core.drr_sources import DrrSource, _known_acquisition_conditions


SPECTRAL_GRID_RTOL = 1e-9
SPECTRAL_GRID_ATOL = 1e-10
_BACK_NAME_RE = re.compile(
    r"(?:^|[_\-\s])[np]?back(?:ground)?\d*(?=$|[_\-\s(\[])", re.IGNORECASE
)


def _named_constant_background(candidate: DrrSource) -> bool:
    return candidate.gate_varies is False and bool(_BACK_NAME_RE.search(Path(candidate.filename).stem))


@dataclass(frozen=True)
class DrrAcquisitionConditions:
    sample: str | None = None
    position: str | None = None
    temperature: str | None = None
    magnetic_field: str | None = None
    polarization: str | None = None
    rotation: str | None = None
    power: str | None = None
    exposure: str | None = None
    accumulation: str | None = None


@dataclass(frozen=True)
class DrrBaselineRecommendation:
    source: DrrSource
    tier: int
    reason: str
    time_difference_seconds: float | None = None


_SAMPLE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:sample|specimen|device)[_\-:=\s]+(?P<value>[A-Za-z0-9]+)",
    re.IGNORECASE,
)
_YZ_SAMPLE_RE = re.compile(r"(?<![A-Za-z0-9])(?P<value>YZ\d+)(?![A-Za-z0-9])", re.IGNORECASE)
_FIELD_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:b|field|mag)(?:[_\-:=\s]*)"
    r"(?P<value>[+\-]?\d+(?:[pP.]\d+)?)\s*(?P<unit>mT|T)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_BARE_FIELD_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<value>[+\-]?\d+(?:[pP.]\d+)?)\s*T(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_POL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:pol|polarization|polarisation)[_\-:=\s]*(?P<value>[A-Za-z0-9+\-.]+)",
    re.IGNORECASE,
)
_ROT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<arm>rot(?:ation)?(?:in|out)?)[_\-:=\s]*"
    r"(?P<value>[+\-]?\d+(?:[pP.]\d+)?)\s*(?:deg|degree|°)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_POWER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:power[_\-:=\s]*)?"
    r"(?P<value>[+\-]?\d+(?:[pP.]\d+)?)\s*(?P<unit>μW|µW|uW|uw|mW|W)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_EXPOSURE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<value>\d+(?:[pP.]\d+)?)\s*"
    r"(?P<unit>ms|us|μs|µs|s)\s*(?:x\s*(?P<accum>\d+))?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _number(value: str) -> float:
    return float(value.replace("p", ".").replace("P", "."))


def _canonical_number(value: float) -> str:
    return f"{value:.12g}"


def _match_value(pattern: re.Pattern[str], stem: str) -> str | None:
    match = pattern.search(stem)
    return match.group("value").casefold() if match else None


def _canonical_exposure(stem: str) -> tuple[str | None, str | None]:
    match = _EXPOSURE_RE.search(stem)
    if not match:
        return None, None
    value = _number(match.group("value"))
    unit = match.group("unit").casefold().replace("μ", "u").replace("µ", "u")
    factor = {"s": 1.0, "ms": 1e-3, "us": 1e-6}[unit]
    exposure = _canonical_number(value * factor) + "s"
    accumulation = match.group("accum")
    return exposure, (str(int(accumulation)) if accumulation else None)


@lru_cache(maxsize=2048)
def parse_drr_acquisition_conditions(filename: str | Path) -> DrrAcquisitionConditions:
    stem = Path(filename).stem
    known = _known_acquisition_conditions(filename)
    sample_match = _SAMPLE_RE.search(stem) or _YZ_SAMPLE_RE.search(stem)
    sample = sample_match.group("value").casefold() if sample_match else None
    point = known.get("point")
    temperature = (
        _canonical_number(_number(known["temperature"])) + "K"
        if known.get("temperature") is not None else None
    )

    field_match = _FIELD_RE.search(stem) or _BARE_FIELD_RE.search(stem)
    magnetic_field = None
    if field_match:
        value = _number(field_match.group("value"))
        unit = (field_match.groupdict().get("unit") or "T").casefold()
        if unit == "mt":
            value /= 1000.0
        magnetic_field = _canonical_number(value) + "T"

    pol_match = _POL_RE.search(stem)
    polarization = pol_match.group("value").casefold() if pol_match else None
    rotations = []
    for rot_match in _ROT_RE.finditer(stem):
        arm = rot_match.group("arm").casefold()
        rotations.append(arm + "=" + _canonical_number(_number(rot_match.group("value"))))
    rotation = ";".join(rotations) if rotations else None
    power_match = _POWER_RE.search(stem)
    power = None
    if power_match:
        value = _number(power_match.group("value"))
        unit = power_match.group("unit").casefold().replace("μ", "u").replace("µ", "u")
        factor = {"w": 1.0, "mw": 1e-3, "uw": 1e-6}[unit]
        power = _canonical_number(value * factor) + "W"
    exposure, accumulation = _canonical_exposure(stem)
    # Existing parser includes the accumulation suffix; retain our normalized
    # unit value while preserving an explicit xN as a separate condition.
    if exposure is None and known.get("exposure"):
        exposure = known["exposure"]

    return DrrAcquisitionConditions(
        sample=sample,
        position=point,
        temperature=temperature,
        magnetic_field=magnetic_field,
        polarization=polarization,
        rotation=rotation,
        power=power,
        exposure=exposure,
        accumulation=accumulation,
    )


def spectral_grids_match(
    measurement: DrrSource,
    candidate: DrrSource,
    *,
    rtol: float = SPECTRAL_GRID_RTOL,
    atol: float = SPECTRAL_GRID_ATOL,
) -> bool:
    """Return true only for equal length, ordered, finite spectral arrays."""
    first = np.asarray(getattr(measurement, "spectral_grid", ()), dtype=float).ravel()
    second = np.asarray(getattr(candidate, "spectral_grid", ()), dtype=float).ravel()
    return bool(
        first.size >= 2
        and first.shape == second.shape
        and np.all(np.isfinite(first))
        and np.all(np.isfinite(second))
        and np.allclose(first, second, rtol=rtol, atol=atol)
    )


def filter_baseline_candidates(
    measurements: Sequence[DrrSource], candidates: Sequence[DrrSource]
) -> tuple[DrrSource, ...]:
    """Filter each file independently against every selected measurement."""
    if not measurements:
        return ()
    return tuple(
        candidate
        for candidate in candidates
        if all(spectral_grids_match(measurement, candidate) for measurement in measurements)
    )


def _condition_state(expected: str | None, current: str | None) -> int:
    """0=known match, 1=known mismatch, 2=unknown."""
    if expected is None or current is None:
        return 2
    return 0 if expected == current else 1


def _aggregate_condition(measurements: Sequence[DrrSource], key: str) -> str | None:
    values = [getattr(parse_drr_acquisition_conditions(item.filename), key) for item in measurements]
    if not values or any(value is None for value in values) or len(set(values)) != 1:
        return None
    return values[0]


def _aggregate_conditions(measurements: Sequence[DrrSource]) -> dict[str, str | None]:
    keys = ("sample", "position", "temperature", "magnetic_field", "polarization", "rotation", "power", "exposure", "accumulation")
    parsed = [parse_drr_acquisition_conditions(item.filename) for item in measurements]
    result: dict[str, str | None] = {}
    for key in keys:
        values = [getattr(item, key) for item in parsed]
        result[key] = (
            values[0]
            if values and all(value is not None for value in values) and len(set(values)) == 1
            else None
        )
    return result


def _time_difference(measurement: DrrSource, candidate: DrrSource) -> float | None:
    first = float(getattr(measurement, "modified_time", float("nan")))
    second = float(getattr(candidate, "modified_time", float("nan")))
    if not math.isfinite(first) or not math.isfinite(second) or first <= 0.0 or second <= 0.0:
        return None
    return abs(second - first)


def _recommendation(
    measurements: Sequence[DrrSource],
    candidate: DrrSource,
    *,
    expected: dict[str, str | None] | None = None,
) -> DrrBaselineRecommendation:
    keys = ("sample", "position", "temperature", "magnetic_field", "polarization", "rotation", "power", "exposure", "accumulation")
    expected = expected or _aggregate_conditions(measurements)
    current = parse_drr_acquisition_conditions(candidate.filename)
    states = tuple(_condition_state(expected[key], getattr(current, key)) for key in keys)
    if states[0] == states[1] == 0:
        tier, label = 0, "same sample/position"
    elif states[2] == states[3] == 0:
        tier, label = 1, "same temperature/magnetic field"
    elif any(states[index] == 0 for index in (4, 5, 6)):
        tier, label = 2, "matching optical setting"
    elif states[7] == states[8] == 0:
        tier, label = 3, "same exposure/accumulation"
    else:
        tier, label = 4, "no known condition match"
    unknown_fields = [
        key.replace("_", " ")
        for key, state in zip(keys, states)
        if state == 2
    ]
    gap = min(
        (value for value in (_time_difference(measurement, candidate) for measurement in measurements) if value is not None),
        default=None,
    )
    if gap is None:
        time_text = "Modified time difference unavailable"
    else:
        time_text = f"Modified time difference {gap:g} s (acquisition time unavailable)"
    reason = f"{label}; {time_text}"
    if _named_constant_background(candidate):
        reason = "back/background filename + confirmed constant gate; " + reason
    if unknown_fields:
        reason += "; unknown: " + ", ".join(dict.fromkeys(unknown_fields))
    return DrrBaselineRecommendation(candidate, tier, reason, gap)


def rank_baseline_candidates(
    measurements: Sequence[DrrSource], candidates: Sequence[DrrSource]
) -> tuple[DrrBaselineRecommendation, ...]:
    """Prefer named constant backgrounds, then conditions, then newest timestamp."""
    if not measurements:
        return ()
    eligible = filter_baseline_candidates(measurements, candidates)
    expected = _aggregate_conditions(measurements)
    ranked = [_recommendation(measurements, candidate, expected=expected) for candidate in eligible]
    keys = ("sample", "position", "temperature", "magnetic_field", "polarization", "rotation", "power", "exposure", "accumulation")
    parsed_candidates = {
        item.source.source: parse_drr_acquisition_conditions(item.source.filename)
        for item in ranked
    }

    def sort_key(item: DrrBaselineRecommendation):
        current = parsed_candidates[item.source.source]
        states = tuple(
            _condition_state(expected[key], getattr(current, key))
            for key in keys
        )
        # Sample and position are one priority level.  A missing member of
        # that pair cannot promote a candidate for a multi-measurement set.
        sample_position = (
            0 if states[0] == states[1] == 0
            else 2 if states[0] == 2 or states[1] == 2
            else 1
        )
        return (
            0 if _named_constant_background(item.source) else 1,
            (sample_position, *states[2:]),
            -item.source.modified_time
            if math.isfinite(item.source.modified_time) and item.source.modified_time > 0
            else float("inf"),
            item.source.filename.casefold(),
            item.source.source.casefold(),
        )

    return tuple(sorted(ranked, key=sort_key))


def candidate_recommendation_map(
    measurements: Sequence[DrrSource], candidates: Sequence[DrrSource]
) -> dict[str, DrrBaselineRecommendation]:
    return {item.source.source: item for item in rank_baseline_candidates(measurements, candidates)}
