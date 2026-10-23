"""Portable external-background regions on the photon-energy axis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DrrBackgroundRegion:
    xmin_ev: float
    xmax_ev: float
    baseline_files: tuple[str, ...]
    baseline_which: str = "last"

    def __post_init__(self):
        if (isinstance(self.xmin_ev, bool) or isinstance(self.xmax_ev, bool)
                or not isinstance(self.xmin_ev, (int, float, np.number))
                or not isinstance(self.xmax_ev, (int, float, np.number))
                or not np.isfinite([self.xmin_ev, self.xmax_ev]).all()
                or self.xmin_ev >= self.xmax_ev):
            raise ValueError("Each background region needs two finite, increasing energy limits.")
        if (not isinstance(self.baseline_files, (list, tuple)) or not self.baseline_files
                or any(not isinstance(name, str) or not name.strip() for name in self.baseline_files)):
            raise ValueError("Each background region requires a list of background files.")
        if not isinstance(self.baseline_which, str) or self.baseline_which.casefold() not in {"first", "last", "all"}:
            raise ValueError("Background region frame method must be first, last or all.")
        object.__setattr__(self, "xmin_ev", float(self.xmin_ev))
        object.__setattr__(self, "xmax_ev", float(self.xmax_ev))
        object.__setattr__(self, "baseline_files", tuple(self.baseline_files))
        object.__setattr__(self, "baseline_which", self.baseline_which.casefold())

    def to_dict(self):
        return {"xmin_ev": self.xmin_ev, "xmax_ev": self.xmax_ev,
                "baseline_files": list(self.baseline_files), "baseline_which": self.baseline_which}

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("A background region must be an object.")
        try:
            return cls(value["xmin_ev"], value["xmax_ev"], value["baseline_files"],
                       value.get("baseline_which", "last"))
        except KeyError as exc:
            raise ValueError(f"Background region is missing {exc.args[0]}.") from exc


def normalize_regions(regions: Sequence[DrrBackgroundRegion]) -> tuple[DrrBackgroundRegion, ...]:
    if not isinstance(regions, (list, tuple)):
        raise ValueError("Background regions must be a list.")
    ordered = tuple(sorted((DrrBackgroundRegion.from_mapping(item) for item in regions),
                           key=lambda item: item.xmin_ev))
    if any(left.xmax_ev > right.xmin_ev for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Background regions must not overlap.")
    return ordered


def region_masks(energy, regions: Sequence[DrrBackgroundRegion]):
    """Validate coverage and return disjoint masks in the original axis order.

    Regions are [xmin, xmax), except that the final spectral sample belongs to
    a region ending exactly at the spectral maximum. Gaps use the default.
    """
    axis = np.asarray(energy, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all() or np.unique(axis).size != axis.size:
        raise ValueError("Background regions require a finite, distinct energy axis.")
    ordered = normalize_regions(regions)
    minimum, maximum = float(np.min(axis)), float(np.max(axis))
    masks = []
    for region in ordered:
        if region.xmin_ev < minimum or region.xmax_ev > maximum:
            raise ValueError("Background region limits must lie within the measurement energy range.")
        mask = (axis >= region.xmin_ev) & (axis < region.xmax_ev)
        if region.xmax_ev == maximum:
            mask |= axis == maximum
        if not mask.any():
            raise ValueError("Each background region must contain at least one measured energy sample.")
        masks.append(mask)
    return tuple(masks)


def background_boundaries(assignment, energy) -> tuple[float, ...]:
    """Return interior recipe switches, excluding adjoining identical recipes."""
    if not assignment.regions:
        return ()
    region_masks(energy, assignment.regions)
    minimum, maximum = float(np.min(energy)), float(np.max(energy))
    default = (assignment.baseline_files, assignment.baseline_which)
    events = {}
    for region in assignment.regions:
        recipe = (region.baseline_files, region.baseline_which)
        events.setdefault(region.xmin_ev, [default, default])[1] = recipe
        events.setdefault(region.xmax_ev, [default, default])[0] = recipe
    return tuple(sorted(point for point, (left, right) in events.items()
                        if minimum < point < maximum and left != right))
