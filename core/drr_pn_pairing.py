"""Recommend independent P/N source groups without selecting or applying them.

This is a catalog-only operation: it neither reads spectra nor generates a
background.  Unknown acquisition conditions remain visible as unknown.  Gate
values may differ between P and N; they are not pairing conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

from core.drr_baseline_candidates import (
    DrrAcquisitionConditions, parse_drr_acquisition_conditions, spectral_grids_match,
)
from core.drr_sources import (
    DrrSource, extract_wavelength_center_nm, group_drr_sources, wavelength_centers_match,
)
from core.load_cancellation import check_load_cancelled


_ROLE = re.compile(
    r"(?<![A-Za-z0-9])(?P<role>[PN])back(?:ground)?\d*(?=$|[_\-\s(\[])",
    re.IGNORECASE,
)
_CONDITIONS = tuple(field.name for field in fields(DrrAcquisitionConditions))


@dataclass(frozen=True)
class DrrPnPair:
    """Ordered repeat groups.  P and N are never averaged together."""

    p_files: tuple[str, ...]
    n_files: tuple[str, ...]

    def __post_init__(self):
        for name in ("p_files", "n_files"):
            values = getattr(self, name)
            if (not isinstance(values, (tuple, list)) or not values
                    or any(not isinstance(value, str) or not value.strip() for value in values)):
                raise ValueError("A P/N pair needs nonempty P and N file groups.")
            values = tuple(values)
            if len({value.replace("\\", "/").casefold() for value in values}) != len(values):
                raise ValueError("A P/N source group cannot contain duplicate files.")
            object.__setattr__(self, name, values)
        if ({value.replace("\\", "/").casefold() for value in self.p_files}
                & {value.replace("\\", "/").casefold() for value in self.n_files}):
            raise ValueError("P and N must be independent source files.")

    @property
    def key(self) -> str:
        """Stable catalog identity; the store additionally resolves root paths."""
        payload = {"P": self.p_files, "N": self.n_files}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DrrPnPairRecommendation:
    pair: DrrPnPair
    label: str
    reason: str
    condition_differences: tuple[str, ...] = ()
    unknown_conditions: tuple[str, ...] = ()
    matched_conditions: tuple[str, ...] = ()
    rank: tuple[int, ...] = ()

    @property
    def requires_review(self) -> bool:
        return bool(self.condition_differences or self.unknown_conditions)


def detect_pn_role(filename: str | Path) -> str | None:
    """Recognize explicit Pback/Pbackground or Nback/Nbackground tokens.

    BNbackground, a position such as p2n1, and ambiguous names containing both
    role tokens are deliberately not accepted.
    """
    roles = {match.group("role").upper() for match in _ROLE.finditer(Path(filename).stem)}
    return next(iter(roles)) if len(roles) == 1 else None


def describe_manual_pair(
    pair: DrrPnPair, measurements: Sequence[str | Path | DrrSource] = (),
) -> str:
    """Describe filename conditions without I/O, selection or an eligibility gate.

    Manual and historical sources need not use the role naming convention.
    Gate values are deliberately absent: opposite P/N doping is expected.
    This description cannot verify native grids or physical cleanliness.
    """
    def parse(value):
        name = value.filename if isinstance(value, DrrSource) else str(value)
        parsed = parse_drr_acquisition_conditions(name)
        return {**{key: getattr(parsed, key) for key in _CONDITIONS},
                "wavelength_center_nm": extract_wavelength_center_nm(name)}

    if isinstance(measurements, (str, Path, DrrSource)):
        measurements = (measurements,)
    roles = {"P": [parse(name) for name in pair.p_files],
             "N": [parse(name) for name in pair.n_files]}
    samples = [parse(name) for name in measurements]
    keys = (*_CONDITIONS, "wavelength_center_nm")
    conflicts, differences, unknown = [], [], []

    def known(rows, key):
        return tuple(dict.fromkeys(row[key] for row in rows if row[key] is not None))

    def label(values):
        return ", ".join(map(str, values))

    for key in keys:
        p, n, target = known(roles["P"], key), known(roles["N"], key), known(samples, key)
        for role, rows in roles.items():
            values = known(rows, key)
            if any(not _equal(key, first, second) for first in values for second in values):
                conflicts.append(f"{role} repeat {key}: [{label(values)}]")
            if any(row[key] is None for row in rows):
                unknown.append(f"{role} {key}")
            if target and any(not _equal(key, value, expected) for value in values for expected in target):
                differences.append(f"{role} {key}: [{label(values)}], measurement [{label(target)}]")
        if any(not _equal(key, first, second) for first in p for second in n):
            conflicts.append(f"{key}: P [{label(p)}], N [{label(n)}]")
        if samples and any(row[key] is None for row in samples):
            unknown.append(f"measurement {key}")
    parts = ["Manual P/N choice; P and N remain independent source groups"]
    parts.append("Known P/N conflicts: " + "; ".join(conflicts) if conflicts
                 else "No known filename-condition conflict within the pair")
    if differences:
        parts.append("Measurement differences: " + "; ".join(differences))
    elif samples:
        parts.append("No known filename-condition difference from the measurement selection")
    if unknown:
        parts.append("Unknown (not a confirmed match or a blocker): " + ", ".join(unknown))
    parts.append("Generation requires an explicit Generate action; no automatic P/N averaging or fallback")
    return "; ".join(parts) + "."


def _conditions(source: DrrSource) -> dict[str, str | float | None]:
    parsed = parse_drr_acquisition_conditions(source.filename)
    return {**{name: getattr(parsed, name) for name in _CONDITIONS},
            "wavelength_center_nm": source.wavelength_center_nm}


def _equal(name, first, second):
    if name == "wavelength_center_nm":
        return wavelength_centers_match(float(first), float(second))
    return first == second


def _compatible(first: DrrSource, second: DrrSource) -> bool:
    if not spectral_grids_match(first, second):
        return False
    a, b = _conditions(first), _conditions(second)
    return all(a[key] is None or b[key] is None or _equal(key, a[key], b[key]) for key in a)


def _role_groups(catalog: Sequence[DrrSource], role: str) -> tuple[tuple[DrrSource, ...], ...]:
    sources = [source for source in catalog
               if detect_pn_role(source.filename) == role
               and Path(source.filename).suffix.casefold() == ".csv"
               and source.gate_varies is not True
               and spectral_grids_match(source, source)]
    # Existing repeat grouping also keeps different same-role gate conditions
    # and sessions separate.  It is NOT used to compare P with N.
    groups = []
    for group in group_drr_sources(sources):
        partitions: list[list[DrrSource]] = []
        for source in sorted(group.files, key=lambda item: item.source.casefold()):
            check_load_cancelled()
            partition = next((part for part in partitions
                              if all(_compatible(source, other) for other in part)), None)
            if partition is None:
                partitions.append([source])
            else:
                partition.append(source)
        groups.extend(tuple(part) for part in partitions)
    return tuple(groups)


def recommend_pn_pairs(
    measurements: Sequence[DrrSource], catalog: Sequence[DrrSource],
) -> tuple[DrrPnPairRecommendation, ...]:
    """Return all compatible pairs ranked against selected measurements.

    Pair-internal known condition conflicts and different ordered spectral
    grids are excluded.  Differences from measurements remain explicit in
    the recommendation, rather than silently selecting a mismatched pair.
    No measurements is allowed for a source-only generation workflow.
    """
    selected = {source.source for source in measurements}
    candidates = [source for source in catalog if source.source not in selected]
    p_groups, n_groups = _role_groups(candidates, "P"), _role_groups(candidates, "N")
    ranked = []
    for p_group in p_groups:
        for n_group in n_groups:
            check_load_cancelled()
            sources = (*p_group, *n_group)
            if not all(_compatible(p, n) for p in p_group for n in n_group):
                continue
            if measurements and not all(spectral_grids_match(sample, source)
                                        for sample in measurements for source in sources):
                continue
            pair = DrrPnPair(tuple(item.source for item in p_group),
                             tuple(item.source for item in n_group))
            current = [_conditions(source) for source in sources]
            expected = [_conditions(source) for source in measurements]
            differences, unknown, matched, ranks = [], [], [], []
            for name in (*_CONDITIONS, "wavelength_center_nm"):
                values = [item[name] for item in current]
                targets = [item[name] for item in expected]
                known_values = [value for value in values if value is not None]
                known_targets = [value for value in targets if value is not None]
                conflicts = [(target, value) for target in known_targets for value in known_values
                             if not _equal(name, target, value)]
                if conflicts:
                    sample_values = ", ".join(dict.fromkeys(map(str, known_targets)))
                    pair_values = ", ".join(dict.fromkeys(map(str, known_values)))
                    differences.append(f"{name}: measurement [{sample_values}], P/N [{pair_values}]")
                    ranks.append(2)
                elif (not expected or any(value is None for value in (*values, *targets))):
                    unknown.append(name)
                    ranks.append(1)
                else:
                    matched.append(name)
                    ranks.append(0)
            p_label = ", ".join(source.filename for source in p_group)
            n_label = ", ".join(source.filename for source in n_group)
            reason = "same ordered spectral grid; P and N remain separate"
            if matched:
                reason += "; matched: " + ", ".join(matched)
            if differences:
                reason += "; differences: " + "; ".join(differences)
            if unknown:
                reason += "; unknown: " + ", ".join(unknown)
            if any(source.gate_varies is None for source in sources):
                reason += "; source constant-gate status is unverified"
            # Total conflicts first; then sample/position, acquisition settings.
            # Stable filename ordering is only a display order, never a choice.
            rank = (len(differences), *ranks, len(unknown))
            ranked.append(DrrPnPairRecommendation(
                pair, f"P: {p_label}\nN: {n_label}", reason, tuple(differences),
                tuple(unknown), tuple(matched), rank,
            ))
    return tuple(sorted(ranked, key=lambda item: (item.rank, item.pair.p_files, item.pair.n_files)))
