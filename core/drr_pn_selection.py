"""Catalog-only P/N file choices and manual time/condition comparisons.

Every row is one original CSV, never an averaged repeat group. Times are file
modification times from the catalog, not acquisition timestamps. A metadata
comparison is a UI convenience, not proof of completed acquisition, optical purity,
or scientific suitability. Generate must still verify the current raw files.
The existing ``drr_pn_pairing`` recommendation API is deliberately unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import PurePosixPath
import re
from typing import Callable, Sequence

from core.drr_baseline_candidates import (
    DrrAcquisitionConditions, parse_drr_acquisition_conditions, spectral_grids_match,
)
from core.drr_pn_pairing import detect_pn_role
from core.drr_sources import DrrSource, extract_wavelength_center_nm, wavelength_centers_match
from core.load_cancellation import LoadCancelled, check_load_cancelled


TIME_LABEL = "File modified time"
_CONDITIONS = tuple(field.name for field in fields(DrrAcquisitionConditions))
_BN = re.compile(r"(?<![A-Za-z0-9])BN(?:back(?:ground)?)?\d*(?=$|[_\-\s(\[])", re.I)
_BACKGROUND = re.compile(r"(?<![A-Za-z0-9])(?:[PN]?back(?:ground)?|bg)\d*(?=$|[_\-\s(\[])", re.I)
_SAMPLE = re.compile(r"(?<![A-Za-z0-9])(?:sample|measurement|data)(?=$|[_\-\s(\[])", re.I)
_ROLE_TOKEN = re.compile(r"(?<![A-Za-z0-9])([PN])back(?:ground)?(?P<number>\d*)(?=$|[_\-\s(\[])", re.I)


@dataclass(frozen=True)
class PnSourceEntry:
    source: DrrSource
    role: str
    role_basis: str
    modified_time: float | None
    matches_measurements: bool
    role_conflict: bool
    reason: str
    condition_differences: tuple[str, ...] = ()
    unknown_conditions: tuple[str, ...] = ()
    time_label: str = TIME_LABEL


@dataclass(frozen=True)
class PnCounterpartCandidate:
    entry: PnSourceEntry
    time_difference_seconds: float | None
    grid_matches: bool
    condition_differences: tuple[str, ...]
    unknown_conditions: tuple[str, ...]
    frame_counts_match: bool | None
    reason: str


def _check(cancelled):
    check_load_cancelled()
    if cancelled is not None and cancelled():
        raise LoadCancelled("P/N source selection cancelled")


def _name(value):
    return str(value).replace("\\", "/")


def _identity(value):
    return _name(value).casefold()


def _mtime(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _order(entry):
    return (-(entry.modified_time or -math.inf), entry.source.filename.casefold(),
            _identity(entry.source.source))


def _conditions(source):
    # The legacy point parser accepts a leading p-word. A role token such as
    # Pbackground must not become a spurious position when no position exists.
    neutral = _ROLE_TOKEN.sub(lambda match: "background" + match.group("number"), source.filename)
    parsed = parse_drr_acquisition_conditions(neutral)
    center = source.wavelength_center_nm
    if center is None:
        center = extract_wavelength_center_nm(source.filename)
    return {**{key: getattr(parsed, key) for key in _CONDITIONS}, "wavelength_center_nm": center}


def _compare(first, second):
    a, b = _conditions(first), _conditions(second)
    differences, unknown = [], []
    for key in a:
        if a[key] is None or b[key] is None:
            unknown.append(key)
        elif not (wavelength_centers_match(a[key], b[key]) if key == "wavelength_center_nm"
                  else a[key] == b[key]):
            differences.append(f"{key}: {a[key]} / {b[key]}")
    return differences, unknown


def _dual_gate_sign(source, cancelled):
    """Use named constant BG/TG columns; never infer roles from position names."""
    channels = {"BG": [], "TG": []}
    for index, label in enumerate(source.gate_labels):
        normalized = re.sub(r"[^a-z0-9]", "", label.casefold())
        for role, prefix in (("BG", "vbg"), ("TG", "vtg")):
            if normalized.startswith(prefix) or normalized in {role.casefold(), role.casefold() + "v"}:
                channels[role].append(index)
    if not source.gate_grid or not all(channels.values()):
        return None, "constant BG/TG polarity unknown"
    signs = []
    for role, indices in channels.items():
        values = []
        for row in source.gate_grid:
            _check(cancelled)
            try:
                values.extend(float(row[index]) for index in indices)
            except (IndexError, TypeError, ValueError):
                return None, "incomplete BG/TG metadata"
        if not values or not all(math.isfinite(value) for value in values):
            return None, "nonfinite BG/TG metadata"
        tolerance = 1e-9 * max(1.0, *(abs(value) for value in values))
        if max(values) - min(values) > tolerance:
            return "varying", f"{role} varies or duplicate gate channels disagree"
        value = values[0]
        signs.append(-1 if value < -tolerance else 1 if value > tolerance else 0)
    if signs == [-1, -1]:
        return "P", "constant BG and TG are both negative"
    if signs == [1, 1]:
        return "N", "constant BG and TG are both positive"
    if signs[0] * signs[1] < 0:
        return "mixed", "constant BG/TG have opposite signs; P/N polarity is ambiguous"
    return None, "BG/TG includes zero; polarity is not inferred"


def build_pn_source_catalog(
    catalog: Sequence[DrrSource], measurements: Sequence[DrrSource] = (), *,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[PnSourceEntry, ...]:
    """Return per-file role rows, newest file-modified time first.

    Explicit Pback/Nback names take precedence; contradictory gate metadata
    stays visible as a role conflict. An anonymous *background candidate* needs
    constant, same-sign BG and TG to infer P (negative) or N (positive). BN,
    known sample files, and generated/processed-directory artifacts are omitted.
    Known measurement conflicts remain visible for explicit manual replacement;
    ``matches_measurements`` means grid-compatible with no known conflict, not
    that missing acquisition conditions have been verified. ``processed`` on a
    raw DrrSource only means it was previously analyzed and does not exclude it.
    """
    _check(cancelled)
    selected = {_identity(source.source) for source in measurements}
    result, seen = [], set()
    for source in catalog:
        _check(cancelled)
        path = PurePosixPath(_name(source.source))
        stem = PurePosixPath(_name(source.filename)).stem
        key = _identity(source.source)
        if key in seen or key in selected:
            continue
        seen.add(key)
        if (path.suffix.casefold() != ".csv" or _BN.search(stem)
                or any(part.casefold() == "processed data" for part in path.parts)
                or "saved composite" in source.classification_reason.casefold()
                or source.metadata_role == "measurement"):
            continue
        explicit = detect_pn_role(source.filename)
        named_background = bool(_BACKGROUND.search(stem))
        if len({match.group(1).upper() for match in _ROLE_TOKEN.finditer(stem)}) > 1:
            continue
        if not explicit and (not source.is_background and not named_background or _SAMPLE.search(stem)):
            continue
        sign, gate_reason = _dual_gate_sign(source, cancelled)
        if explicit:
            role, basis = explicit, "explicit_name"
        elif sign in {"P", "N"} and source.gate_varies is False and source.grid_complete:
            role, basis = sign, "constant_dual_gate"
        else:
            continue
        reasons = [f"{role} from explicit background name" if explicit else f"{role} inferred from {gate_reason}"]
        role_conflict = False
        if explicit:
            reasons.append(gate_reason)
            if sign in {"mixed", "varying"} or sign in {"P", "N"} and sign != role:
                role_conflict = True
                reasons.append("name/gate evidence conflict or ambiguity; review the selected role")
        if source.gate_varies is not False:
            reasons.append("constant-gate status is not verified")
        if not source.grid_complete:
            reasons.append("catalog grid completeness is not verified")
        count = source.frame_count
        if not isinstance(count, int) or isinstance(count, bool) or count < 4 or count % 2:
            reasons.append("current reconstruction requires a known even frame count of at least four")
        differences, unknown = [], []
        matching = spectral_grids_match(source, source)
        if not matching:
            reasons.append("native spectral grid is unavailable")
        for sample in measurements:
            _check(cancelled)
            if not spectral_grids_match(source, sample):
                matching = False
                differences.append(f"native spectral grid differs from {sample.filename}")
            conflict, absent = _compare(source, sample)
            differences.extend(conflict)
            unknown.extend(absent)
        if differences:
            matching = False
            reasons.append("measurement conflicts: " + "; ".join(dict.fromkeys(differences)))
        if unknown:
            reasons.append("unknown conditions: " + ", ".join(dict.fromkeys(unknown)))
        reasons.append("catalog metadata does not prove acquisition is complete")
        result.append(PnSourceEntry(source, role, basis, _mtime(source.modified_time),
            matching, role_conflict, "; ".join(reasons), tuple(dict.fromkeys(differences)),
            tuple(dict.fromkeys(unknown))))
    _check(cancelled)
    return tuple(sorted(result, key=_order))


def pn_source_matches_role(source: DrrSource | str, role: str) -> bool:
    """Guard manual P/N choices while retaining genuinely unlabelled imports.

    Known BN, opposite or ambiguous explicit role names cannot be relabelled.
    A catalog source can additionally carry a role inferred from dual gates.
    This does not impose reconstruction frame counts or perform file I/O.
    """
    filename = source.filename if isinstance(source, DrrSource) else str(source)
    stem = PurePosixPath(_name(filename)).stem
    roles = {match.group(1).upper() for match in _ROLE_TOKEN.finditer(stem)}
    if _BN.search(stem) or roles and roles != {role}:
        return False
    if roles or not isinstance(source, DrrSource):
        return True
    entries = build_pn_source_catalog((source,))
    return not entries or entries[0].role == role


def describe_pn_pair(
    selected: PnSourceEntry, other: PnSourceEntry, *,
    cancelled: Callable[[], bool] | None = None,
) -> PnCounterpartCandidate:
    """Describe two manually chosen opposite sources, with no pairing decision.

    Numeric filename suffixes, directories and session days never bind files or
    veto a comparison. Time is an absolute file-modification-time difference;
    there is no proximity threshold, ranking, automatic choice or acceptance.
    Known acquisition/grid conflicts remain explicit rather than being hidden.
    """
    _check(cancelled)
    if (not isinstance(selected, PnSourceEntry) or not isinstance(other, PnSourceEntry)
            or {selected.role, other.role} != {"P", "N"}
            or _identity(selected.source.source) == _identity(other.source.source)):
        raise ValueError("Select two independent opposite-role catalog entries.")
    difference = (abs(selected.modified_time - other.modified_time)
                  if selected.modified_time is not None and other.modified_time is not None else None)
    grid_matches = spectral_grids_match(selected.source, other.source)
    conflicts, unknown = _compare(selected.source, other.source)
    counts = (selected.source.frame_count, other.source.frame_count)
    frame_match = counts[0] == counts[1] if all(
        isinstance(count, int) and not isinstance(count, bool) and count > 0 for count in counts) else None
    reasons = [f"file modified time difference {difference:g} s" if difference is not None
               else "file modified time difference unknown"]
    reasons.append("same ordered native spectral grid" if grid_matches else "native spectral grid differs or is unavailable")
    if conflicts:
        reasons.append("known condition conflicts: " + "; ".join(conflicts))
    else:
        reasons.append("no known acquisition-condition conflict")
    if unknown:
        reasons.append("unknown conditions: " + ", ".join(unknown))
    if selected.role_conflict or other.role_conflict:
        reasons.append("selected role conflicts with or is ambiguous from the gate metadata")
    reasons.append("frame counts unknown" if frame_match is None else
                   "equal frame counts" if frame_match else f"different frame counts: {counts[0]} / {counts[1]}")
    reasons.append("manual choice; file modified time is not acquisition time or scientific validation")
    _check(cancelled)
    return PnCounterpartCandidate(other, difference, grid_matches, tuple(conflicts), tuple(unknown),
                                frame_match, "; ".join(reasons))


def describe_pn_counterparts(
    selected: PnSourceEntry, entries: Sequence[PnSourceEntry], *,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[PnCounterpartCandidate, ...]:
    """Describe all opposite rows in their existing order for manual selection.

    Includes different repeat numbers, dates, directories and known conditions.
    This helper supplies display metadata only; it never selects or filters a
    counterpart, and never reorders the chronological P/N tab rows by proximity.
    """
    _check(cancelled)
    if not isinstance(selected, PnSourceEntry) or selected.role not in {"P", "N"}:
        raise ValueError("Select one catalog P/N source entry.")
    comparisons = []
    for entry in entries:
        _check(cancelled)
        if entry.role == selected.role or _identity(entry.source.source) == _identity(selected.source.source):
            continue
        comparisons.append(describe_pn_pair(selected, entry, cancelled=cancelled))
    _check(cancelled)
    return tuple(comparisons)
