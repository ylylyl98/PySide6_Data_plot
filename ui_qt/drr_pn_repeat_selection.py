"""Choose the existing two-repeat P/N default from the current selection.

This helper does not discover files or inspect CSV contents. Catalog metadata
can identify saved backgrounds; original-file and native-grid validation still
belongs to the reconstruction request's existing preflight.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import re
from typing import Sequence

from core.drr_baseline_candidates import (
    DrrAcquisitionConditions, parse_drr_acquisition_conditions,
)
from core.drr_sources import (
    DrrSource, extract_wavelength_center_nm, is_background_name,
    measurement_group_key, resolve_source_path,
)
from core.processing import split_group_and_sort_key


_OTHER_REPEAT_SUFFIX = re.compile(
    r"[_\-](?:run|scan|repeat|part)(?P<repeat>\d+)$", re.IGNORECASE,
)


@dataclass(frozen=True)
class TrainingRepeatSelection:
    training_files: tuple[str, ...] = ()
    check_files: tuple[str, ...] = ()
    reason: str = ""

    @property
    def usable(self) -> bool:
        return len(self.training_files) == 2


def _identity(folder: str | Path, name: str | Path) -> str:
    return str(resolve_source_path(folder, name)).replace("\\", "/").casefold()


def _repeat(name: str) -> tuple[str, tuple[int, int]] | None:
    basename = Path(name).name
    _group, order = split_group_and_sort_key(basename)
    if order[0] < 2:
        return measurement_group_key(basename).casefold(), (order[1], order[2])
    match = _OTHER_REPEAT_SUFFIX.search(Path(basename).stem)
    if match:
        return measurement_group_key(basename).casefold(), (int(match["repeat"]), 0)
    return None


def choose_training_repeats(
    folder: str | Path,
    measurements: Sequence[str | Path],
    *,
    catalog: Sequence[DrrSource] = (),
    excluded_files: Sequence[str | Path] = (),
) -> TrainingRepeatSelection:
    """Choose sorted repeats 1 and 3, or both when exactly two are selected.

    Numbered repeats are ordered by repeat/subrepeat number, independently of
    selection order. Unknown or mixed naming retains the caller's group order.
    The caller supplies the measurement group; unknown naming is not evidence
    of incompatibility. Explicit repeat groups or known acquisition conditions
    that conflict are rejected. Invalid selections are never silently filtered.

    Paths are resolved only for aliases and catalog lookup. No directory scan,
    file-content read, sample ranking or reconstruction setting is involved.
    """
    if isinstance(measurements, (str, Path)):
        return TrainingRepeatSelection(reason="Select at least two distinct raw measurement repeats.")
    names = tuple(str(name) for name in measurements)

    def problem(reason: str) -> TrainingRepeatSelection:
        return TrainingRepeatSelection(check_files=names, reason=reason)

    if len(names) < 2 or any(not name.strip() for name in names):
        return problem("Select at least two distinct raw measurement repeats.")
    try:
        identities = tuple(_identity(folder, name) for name in names)
        excluded = {_identity(folder, name) for name in excluded_files}
        catalog_by_path: dict[str, list[DrrSource]] = {}
        for source in catalog:
            catalog_by_path.setdefault(_identity(folder, source.source), []).append(source)
    except (OSError, ValueError, RuntimeError):
        return problem("Selected measurement paths could not be resolved.")
    if len(set(identities)) != len(identities):
        return problem("Selected measurement repeats contain duplicate file paths; select each original only once.")
    if excluded.intersection(identities):
        return problem("P, N and sample measurement repeats must be distinct original files.")
    if any(Path(name).suffix.casefold() != ".csv" for name in names):
        return problem("Automatic training requires original intensity CSV measurement repeats.")
    for name, identity in zip(names, identities):
        sources = catalog_by_path.get(identity, ())
        if is_background_name(name) or any(
            source.is_background
            or source.classification in {"background", "likely_background"}
            or source.metadata_role == "background"
            for source in sources
        ):
            return problem("Select raw measurement repeats; backgrounds and saved composites cannot be training samples.")

    repeats = tuple(_repeat(name) for name in names)
    if len({repeat[0] for repeat in repeats if repeat is not None}) > 1:
        return problem("Select repeats from one measurement group; numbered filenames identify different groups.")
    # Remove repeat suffixes before parsing: sample_rep01 otherwise looks like
    # a named specimen called rep01 instead of the same sample's first repeat.
    conditions = tuple(parse_drr_acquisition_conditions(measurement_group_key(name) + ".csv") for name in names)
    for field in fields(DrrAcquisitionConditions):
        known = {getattr(condition, field.name) for condition in conditions}
        known.discard(None)
        if len(known) > 1:
            label = field.name.replace("_", " ")
            return problem(f"Select repeats from one measurement group; known {label} conditions differ.")
    centers = [extract_wavelength_center_nm(name) for name in names]
    centers = [value for value in centers if value is not None]
    if centers and max(centers) - min(centers) > 1e-6:
        return problem("Select repeats from one measurement group; wavelength centers differ.")

    ordered = list(range(len(names)))
    numbered = all(repeat is not None for repeat in repeats)
    if numbered:
        ordered.sort(key=lambda index: (repeats[index][1], identities[index]))
    selected = (ordered[0], ordered[2 if len(ordered) >= 3 else 1])
    training = tuple(names[index] for index in selected)
    checks = tuple(name for index, name in enumerate(names) if index not in selected)
    order_description = "repeat-number order" if numbered else "the selected measurement order"
    policy = ("Automatically uses both selected measurement repeats." if len(names) == 2 else
              f"Automatically uses repeats 1 and 3 in {order_description}; other selected files are checks.")
    return TrainingRepeatSelection(training, checks, policy)
