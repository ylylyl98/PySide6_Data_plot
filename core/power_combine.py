"""Combine measured power sweeps without interpolating their power axes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_io import PowerSeriesResult
from core.loader import DataCube
from core.processing import PowerSweepPoint


def combine_power_sweeps(first: PowerSeriesResult, second: PowerSeriesResult,
                         *, duplicate_policy: str = "first") -> PowerSeriesResult:
    return combine_many_power_sweeps((first, second), duplicate_policy=duplicate_policy)


def combine_many_power_sweeps(results, *, duplicate_policy="first"):
    """Use exact power equality for duplicates; interpolate energy only in overlap.

    No intensity normalization, power interpolation, or extrapolation is applied.
    Stage is retained for averages only when all contributing stages agree.
    """
    if duplicate_policy not in {"first", "second", "average"}:
        raise ValueError("Choose first, second, or average for matching powers.")
    results = tuple(results)
    if len(results) < 2 or len({r.group_key for r in results}) != len(results):
        raise ValueError("Select at least two different sweeps.")
    from core.power_workflow import source_rows
    seen = set()
    for result in results:
        rows = source_rows(result)
        if seen & rows:
            raise ValueError("These inputs contain the same original measurement rows. Combining would double count them.")
        seen.update(rows)
    for result in results:
        cube = result.cube
        energy = np.asarray(cube.energy, float)
        powers = np.asarray(cube.gate, float)
        if (energy.size < 2 or not np.all(np.isfinite(energy))
                or np.any(np.diff(energy) <= 0)):
            raise ValueError("Sweeps need increasing, finite energy axes.")
        if (not powers.size or not np.all(np.isfinite(powers))
                or len(result.records) != powers.size
                or cube.Z.shape != (powers.size, energy.size)):
            raise ValueError("Sweep spectra, power points, and source records must match.")
    e1 = np.asarray(results[0].cube.energy, float)
    energy = e1[(e1 >= max(r.cube.energy[0] for r in results)) & (e1 <= min(r.cube.energy[-1] for r in results))]
    if energy.size < 2:
        raise ValueError("Sweeps need at least two spectral points in their shared energy range.")
    by_power = {}
    for result in results:
        for power, spectrum, record in zip(result.cube.gate, result.cube.Z, result.records):
            row = np.interp(energy, result.cube.energy, spectrum)
            by_power.setdefault(float(power), []).append((row, record, result.group_key))
    spectra, records = [], []
    for power, candidates in sorted(by_power.items()):
        chosen = (candidates if duplicate_policy == "average" else
                  [candidates[0] if duplicate_policy == "first" else candidates[-1]])
        rows = np.asarray([entry[0] for entry in chosen])
        finite = np.isfinite(rows)
        counts = finite.sum(axis=0)
        row = np.divide(np.where(finite, rows, 0).sum(axis=0), counts,
                        out=np.full(energy.size, np.nan), where=counts > 0)
        if not np.any(np.isfinite(row)):
            raise ValueError(f"No finite spectrum in the shared energy range at {power:g} uW.")
        spectra.append(row)
        stages = [entry[1].stage for entry in chosen]
        stage = stages[0] if all(value == stages[0] for value in stages) else None
        provenance = json.dumps({
            "duplicate_policy": duplicate_policy,
            "sources": [{"file": record.file_name,
                         "row": getattr(record, "row_index", None),
                         "group": group, "power_uW": power,
                         "stage": record.stage,
                         "prior_provenance": getattr(record, "source_provenance", "")}
                        for _, record, group in chosen],
            "energy_alignment": "first sweep grid within shared range; linear interpolation",
            "intensity_scaling": "none during merge; prior corrections recorded in source provenance",
        }, ensure_ascii=False)
        records.append(PowerSweepPoint("combined.csv", power, stage, len(records) + 2,
                                       "stage_pos", provenance))
    cube = DataCube(energy.copy(), np.asarray(sorted(by_power)), np.vstack(spectra),
                    "Power (uW)", "Combined " + " + ".join(r.cube.title for r in results),
                    results[0].cube.cbar_label)
    return PowerSeriesResult(cube, "combined", tuple(records), {})


def save_combined_power_sweep(path: Path, result: PowerSeriesResult, *, unique: bool = False) -> Path:
    """Write a reloadable table with per-point provenance; never replace a file."""
    columns = {"Power_uW": result.cube.gate,
               "stage_pos": [record.stage for record in result.records],
               "source_provenance": [record.source_provenance for record in result.records]}
    columns.update({format(float(energy), ".17g"): result.cube.Z[:, index]
                    for index, energy in enumerate(result.cube.energy)})
    frame = pd.DataFrame(columns)
    for index in range(10000):
        candidate = path if index == 0 else path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
        try:
            frame.to_csv(candidate, index=False, mode="x")
        except FileExistsError:
            if not unique:
                raise
        else:
            return candidate
    raise FileExistsError("Could not allocate a new combined-sweep filename.")
