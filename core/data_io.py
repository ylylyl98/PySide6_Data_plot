from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from core.file_ops import archive_selected, list_root_csvs
from core.loader import (
    DataCube,
    build_external_baseline,
    load_drr_avg,
    load_pl,
)
from core.processing import PowerSeriesFile, group_power_series_files, power_group_title


DEFAULT_ARCHIVE = "Initial data after processing"
DEFAULT_PROCESSED = "Processed Data"


@dataclass(frozen=True)
class CompareSelection:
    kk: str | None = None
    kkp: str | None = None
    kpk: str | None = None
    kpkp: str | None = None
    visible_order: tuple[str, ...] = ()

    def as_pairs(self) -> Dict[str, str]:
        raw: Dict[str, str] = {}
        if self.kk:
            raw["KK"] = self.kk
        if self.kkp:
            raw["KKp"] = self.kkp
        if self.kpk:
            raw["KpK"] = self.kpk
        if self.kpkp:
            raw["KpKp"] = self.kpkp
        order = self.visible_order if self.visible_order else ("KK", "KKp", "KpK", "KpKp")
        pairs: Dict[str, str] = {}
        for key in order:
            if key in raw:
                pairs[key] = raw[key]
        return pairs

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict[str, str],
        *,
        visible_order: Sequence[str] | None = None,
    ) -> "CompareSelection":
        return cls(
            kk=mapping.get("KK"),
            kkp=mapping.get("KKp"),
            kpk=mapping.get("KpK"),
            kpkp=mapping.get("KpKp"),
            visible_order=tuple(visible_order or ()),
        )


@dataclass(frozen=True)
class PowerSeriesResult:
    cube: DataCube
    group_key: str
    records: tuple[PowerSeriesFile, ...]
    groups: Dict[str, tuple[PowerSeriesFile, ...]]


def list_csv_files(folder: str) -> List[str]:
    return list_root_csvs(folder)


def move_selected_to_archive(folder: str, file_names: Sequence[str], archive_name: str = DEFAULT_ARCHIVE) -> int:
    return archive_selected(folder, list(file_names), archive_name)


def load_pl_cube(folder: str, file_name: str, *, log_scale: bool = False, y_axis: str = "auto") -> DataCube:
    return load_pl(folder, file_name, log_scale=log_scale, y_axis=y_axis)


def load_drr_self_cube(
    folder: str,
    files: Sequence[str],
    *,
    use_first_frame: bool,
    y_axis: str = "auto",
    derivative: int | None = None,
) -> DataCube:
    bg_mode = "self_first" if use_first_frame else "self_last"
    return load_drr_avg(
        folder,
        files,
        bg_mode=bg_mode,
        y_axis=y_axis,
        derivative=derivative,
    )


def load_drr_external_cube(
    folder: str,
    files: Sequence[str],
    baseline_files: Sequence[str],
    *,
    baseline_which: str,
    y_axis: str = "auto",
    derivative: int | None = None,
) -> DataCube:
    baseline = build_external_baseline(folder, baseline_files, which=baseline_which)
    return load_drr_avg(
        folder,
        files,
        bg_mode="external",
        y_axis=y_axis,
        external_vector=np.asarray(baseline["I0"], float),
        external_energy=np.asarray(baseline["energy"], float),
        derivative=derivative,
    )


def load_compare_cubes(
    folder: str,
    selection: CompareSelection,
    *,
    log_scale: bool = False,
    y_axis: str = "auto",
) -> Dict[str, DataCube]:
    return {name: load_pl(folder, fn, log_scale=log_scale, y_axis=y_axis) for name, fn in selection.as_pairs().items()}


def get_power_series_groups(files: Sequence[str]) -> Dict[str, tuple[PowerSeriesFile, ...]]:
    return {key: tuple(records) for key, records in group_power_series_files(files).items()}


def _spectrum_from_cube(cube: DataCube) -> np.ndarray:
    z = np.asarray(cube.Z, float)
    if z.ndim != 2:
        raise ValueError(f"Power-dependent input must be a 2D PL matrix, got shape {z.shape}.")
    if z.shape[1] != np.asarray(cube.energy).size and z.shape[0] == np.asarray(cube.energy).size:
        z = z.T
    if z.shape[1] != np.asarray(cube.energy).size:
        raise ValueError(f"Power-dependent input Z shape {z.shape} does not match energy axis.")
    if z.shape[0] == 1:
        return z[0, :].astype(float, copy=True)
    return np.nanmean(z, axis=0).astype(float, copy=True)


def load_power_series_cube(
    folder: str,
    files: Sequence[str],
    *,
    group_key: str | None = None,
    y_axis: str = "auto",
) -> PowerSeriesResult:
    groups = get_power_series_groups(files)
    if not groups:
        raise ValueError("No power-dependent groups found. Filenames must contain a power token such as 37.96uW.")

    selected_key = group_key if group_key in groups else ""
    if not selected_key:
        selected_key = sorted(groups.keys(), key=lambda key: (-len(groups[key]), key))[0]
    records = groups[selected_key]
    if not records:
        raise ValueError("Selected power-dependent group is empty.")

    ref_energy: np.ndarray | None = None
    spectra: list[np.ndarray] = []
    powers: list[float] = []
    for record in records:
        cube = load_pl(folder, record.file_name, log_scale=False, y_axis=y_axis)
        energy = np.asarray(cube.energy, float).ravel()
        spectrum = _spectrum_from_cube(cube)
        if ref_energy is None:
            ref_energy = energy.copy()
        elif energy.shape != ref_energy.shape or not np.allclose(energy, ref_energy, rtol=1e-7, atol=1e-10):
            finite = np.isfinite(energy) & np.isfinite(spectrum)
            if np.count_nonzero(finite) < 2:
                raise ValueError(f"Cannot interpolate spectrum from {record.file_name}: insufficient finite points.")
            spectrum = np.interp(ref_energy, energy[finite], spectrum[finite], left=np.nan, right=np.nan)
        spectra.append(np.asarray(spectrum, float))
        powers.append(float(record.power_uW))

    if ref_energy is None:
        raise ValueError("No spectra loaded for selected power group.")

    z = np.vstack(spectra)
    cube = DataCube(
        energy=ref_energy,
        gate=np.asarray(powers, float),
        Z=z,
        gate_label="Power (uW)",
        title=power_group_title(selected_key),
        cbar_label="PL (a.u.)",
    )
    return PowerSeriesResult(cube=cube, group_key=selected_key, records=records, groups=groups)


