from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from core.file_ops import _nat_key, archive_selected, list_root_csvs
from core.loader import (
    DataCube,
    XLSX_Y_LABEL_OPTIONS,
    build_external_baseline,
    is_xlsx_map_file,
    load_drr_avg,
    load_pl,
    load_xlsx_map,
    resolve_xlsx_y_label,
)
from core import processing_run as processing_impl
from core.processing import (
    PowerSeriesFile,
    PowerSweepPoint,
    group_power_series_files,
    power_group_title,
)
from core.shg import ShgSweepData, inspect_shg_csv, load_shg_sweep_csv


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
    records: tuple[PowerSeriesFile | PowerSweepPoint, ...]
    groups: Dict[str, tuple[PowerSeriesFile | PowerSweepPoint, ...]]
    sources: Dict[str, "PowerSeriesSource"] | None = None


@dataclass(frozen=True)
class PowerSeriesSource:
    key: str
    title: str
    source_format: str
    file_name: str | None = None
    records: tuple[PowerSeriesFile, ...] = ()


POWER_SWEEP_KEY_PREFIX = "csv::"


def power_sweep_source_key(file_name: str) -> str:
    return f"{POWER_SWEEP_KEY_PREFIX}{Path(file_name).name}"


def is_power_sweep_source_key(key: str) -> bool:
    return str(key).startswith(POWER_SWEEP_KEY_PREFIX)


def power_sweep_file_from_key(key: str) -> str:
    if not is_power_sweep_source_key(key):
        raise ValueError(f"Not a power-sweep CSV source key: {key}")
    return Path(str(key)[len(POWER_SWEEP_KEY_PREFIX):]).name


def list_csv_files(folder: str) -> List[str]:
    return list_root_csvs(folder)


def list_map_input_files(folder: str) -> List[str]:
    """Return root-level CSV and XLSX map filenames, naturally sorted."""
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []
    try:
        names = {f.name for f in p.glob("*.csv")} | {f.name for f in p.glob("*.xlsx")}
    except OSError:
        return []
    return sorted(names, key=_nat_key)


def list_shg_csv_files(folder: str, files: Sequence[str]) -> List[str]:
    return [file_name for file_name in files if inspect_shg_csv(folder, file_name)]


def load_shg_sweep(folder: str, file_name: str) -> ShgSweepData:
    return load_shg_sweep_csv(folder, file_name)


def move_selected_to_archive(folder: str, file_names: Sequence[str], archive_name: str = DEFAULT_ARCHIVE) -> int:
    return archive_selected(folder, list(file_names), archive_name)


def load_pl_cube(folder: str, file_name: str, *, log_scale: bool = False, y_axis: str = "auto") -> DataCube:
    return load_pl(folder, file_name, log_scale=log_scale, y_axis=y_axis)


def load_drr_map_cube(folder: str, file_name: str, *, y_axis: str = "auto") -> DataCube:
    """Display a precomputed dR/R XLSX map directly, without ratio processing."""
    return load_pl(folder, file_name, log_scale=False, y_axis=y_axis)


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
    pairs = selection.as_pairs()
    effective_y_axis = processing_impl.resolve_shared_y_axis_request(list(pairs.values()), y_axis)
    return {
        name: load_pl(folder, fn, log_scale=log_scale, y_axis=effective_y_axis)
        for name, fn in pairs.items()
    }


def get_power_series_groups(files: Sequence[str]) -> Dict[str, tuple[PowerSeriesFile, ...]]:
    return {key: tuple(records) for key, records in group_power_series_files(files).items()}


def _find_table_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    return processing_impl._find_col_by_priority(list(columns), list(candidates))


def _power_table_columns(columns: Sequence[str]) -> tuple[str | None, str | None, list[str]]:
    cols = [str(column).strip() for column in columns]
    power_col = _find_table_column(
        cols,
        ["power_uw", "power (uw)", "poweruw", "laser_power_uw", "laserpoweruw", "power"],
    )
    stage_col = _find_table_column(cols, ["stage_pos", "stage position", "stagepos", "stage"])
    spectrum_cols = [
        column for column in cols if processing_impl._parse_spec_axis_from_colname(column) is not None
    ]
    return power_col, stage_col, spectrum_cols


def inspect_power_sweep_csv(folder: str, file_name: str) -> bool:
    """Return True when a CSV header describes a single-file power sweep."""
    path = Path(folder) / Path(file_name).name
    if not path.is_file():
        return False
    try:
        sep = processing_impl._guess_sep_from_first_line(path)
        columns = pd.read_csv(path, sep=sep, nrows=0).columns
        power_col, _stage_col, _spectrum_cols = _power_table_columns(columns)
        # Keep malformed candidates visible in the UI so selecting one produces
        # the loader's actionable missing-spectrum error.
        return power_col is not None
    except Exception:
        return False


def get_power_series_sources(folder: str, files: Sequence[str]) -> Dict[str, PowerSeriesSource]:
    """Discover table-backed power sweeps first, followed by legacy filename groups."""
    sources: Dict[str, PowerSeriesSource] = {}
    table_files: set[str] = set()
    for file_name in files:
        if inspect_power_sweep_csv(folder, file_name):
            table_files.add(Path(file_name).name)
            key = power_sweep_source_key(file_name)
            sources[key] = PowerSeriesSource(
                key=key,
                title=power_group_title(key),
                source_format="table",
                file_name=Path(file_name).name,
            )
    legacy_files = [file_name for file_name in files if Path(file_name).name not in table_files]
    for key, records in get_power_series_groups(legacy_files).items():
        sources[key] = PowerSeriesSource(
            key=key,
            title=power_group_title(key),
            source_format="legacy",
            records=tuple(records),
        )
    return sources


def _power_sweep_signature(folder: str, file_name: str) -> tuple[int, int]:
    path = Path(folder) / Path(file_name).name
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found in folder root: {path}")
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=128)
def _load_power_sweep_csv_cached(
    folder: str,
    file_name: str,
    signature: tuple[int, int],
) -> tuple[DataCube, tuple[PowerSweepPoint, ...]]:
    del signature
    path = Path(folder) / Path(file_name).name
    sep = processing_impl._guess_sep_from_first_line(path)
    frame = pd.read_csv(path, sep=sep)
    frame.columns = [str(column).strip() for column in frame.columns]
    power_col, stage_col, spectrum_cols = _power_table_columns(frame.columns)
    if power_col is None:
        raise ValueError(
            f"Power sweep CSV {file_name!r} is missing a Power_uW column."
        )
    if len(spectrum_cols) < 2:
        raise ValueError(
            f"Power sweep CSV {file_name!r} needs at least two numeric wavelength/energy columns."
        )

    # Pandas may suffix duplicate headers (for example 752.58 and 752.58.1).
    # Keep the duplicate column containing the most finite measurements.
    spectral_groups: dict[float, list[str]] = {}
    for column in spectrum_cols:
        value = processing_impl._parse_spec_axis_from_colname(column)
        if value is not None:
            spectral_groups.setdefault(round(float(value), 9), []).append(column)
    chosen_columns: list[str] = []
    spectral_values: list[float] = []
    for value, candidates in spectral_groups.items():
        chosen = max(
            candidates,
            key=lambda column: int(
                np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)).sum()
            ),
        )
        chosen_columns.append(chosen)
        spectral_values.append(float(value))

    spectrum_axis = np.asarray(spectral_values, dtype=float)
    z = frame[chosen_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_columns = np.any(np.isfinite(z), axis=0)
    spectrum_axis = spectrum_axis[finite_columns]
    z = z[:, finite_columns]
    if spectrum_axis.size < 2:
        raise ValueError(
            f"Power sweep CSV {file_name!r} has fewer than two spectral columns containing numeric data."
        )

    power = pd.to_numeric(frame[power_col], errors="coerce").to_numpy(dtype=float)
    invalid_power = np.flatnonzero(~np.isfinite(power))
    if invalid_power.size:
        rows = ", ".join(str(int(index) + 2) for index in invalid_power[:8])
        suffix = "..." if invalid_power.size > 8 else ""
        raise ValueError(f"Power sweep CSV {file_name!r} has invalid Power_uW values on row(s) {rows}{suffix}.")
    invalid_spectra = np.flatnonzero(~np.any(np.isfinite(z), axis=1))
    if invalid_spectra.size:
        rows = ", ".join(str(int(index) + 2) for index in invalid_spectra[:8])
        suffix = "..." if invalid_spectra.size > 8 else ""
        raise ValueError(f"Power sweep CSV {file_name!r} has no numeric spectrum on row(s) {rows}{suffix}.")
    unique_power, counts = np.unique(power, return_counts=True)
    duplicates = unique_power[counts > 1]
    if duplicates.size:
        values = ", ".join(f"{value:.6g}" for value in duplicates[:8])
        suffix = "..." if duplicates.size > 8 else ""
        raise ValueError(
            f"Power sweep CSV {file_name!r} contains duplicate Power_uW values ({values}{suffix}); "
            "use unique power values so spectra are not silently averaged."
        )

    stage_values = (
        pd.to_numeric(frame[stage_col], errors="coerce").to_numpy(dtype=float)
        if stage_col is not None
        else np.full(power.shape, np.nan, dtype=float)
    )
    energy = 1240.0 / spectrum_axis if float(np.nanmedian(spectrum_axis)) > 20.0 else spectrum_axis.copy()
    energy_order = np.argsort(energy)
    energy = energy[energy_order]
    z = z[:, energy_order]
    power_order = np.argsort(power, kind="stable")
    power = power[power_order]
    z = z[power_order, :]
    stage_values = stage_values[power_order]

    records = tuple(
        PowerSweepPoint(
            file_name=Path(file_name).name,
            power_uW=float(power[index]),
            stage=(float(stage_values[index]) if np.isfinite(stage_values[index]) else None),
            row_index=int(power_order[index]) + 2,
            stage_column=stage_col,
        )
        for index in range(power.size)
    )
    cube = DataCube(
        energy=np.asarray(energy, dtype=float),
        gate=np.asarray(power, dtype=float),
        Z=np.asarray(z, dtype=float),
        gate_label="Power (uW)",
        title=power_group_title(power_sweep_source_key(file_name)),
        cbar_label="PL (a.u.)",
    )
    return cube, records


def load_power_sweep_csv(folder: str, file_name: str) -> tuple[DataCube, tuple[PowerSweepPoint, ...]]:
    signature = _power_sweep_signature(folder, file_name)
    cube, records = _load_power_sweep_csv_cached(folder, Path(file_name).name, signature)
    copied_cube = DataCube(
        energy=np.asarray(cube.energy, dtype=float).copy(),
        gate=np.asarray(cube.gate, dtype=float).copy(),
        Z=np.asarray(cube.Z, dtype=float).copy(),
        gate_label=cube.gate_label,
        title=cube.title,
        cbar_label=cube.cbar_label,
    )
    return copied_cube, tuple(records)


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
    sources = get_power_series_sources(folder, files)
    if not sources:
        raise ValueError(
            "No power-dependent data found. Use a CSV with a Power_uW column or filenames containing power such as 37.96uW."
        )

    selected_key = group_key if group_key in sources else ""
    if not selected_key:
        table_keys = [key for key, source in sources.items() if source.source_format == "table"]
        if table_keys:
            selected_key = sorted(table_keys)[0]
        else:
            selected_key = sorted(
                sources,
                key=lambda key: (-len(sources[key].records), key),
            )[0]
    selected_source = sources[selected_key]
    if selected_source.source_format == "table":
        assert selected_source.file_name is not None
        cube, records = load_power_sweep_csv(folder, selected_source.file_name)
        groups: Dict[str, tuple[PowerSeriesFile | PowerSweepPoint, ...]] = {
            key: tuple(source.records) for key, source in sources.items()
        }
        groups[selected_key] = records
        return PowerSeriesResult(
            cube=cube,
            group_key=selected_key,
            records=records,
            groups=groups,
            sources=sources,
        )

    groups = {key: tuple(source.records) for key, source in sources.items()}
    records = groups[selected_key]
    if not records:
        raise ValueError("Selected power-dependent group is empty.")

    effective_y_axis = processing_impl.resolve_shared_y_axis_request(
        [record.file_name for record in records],
        y_axis,
    )

    ref_energy: np.ndarray | None = None
    spectra: list[np.ndarray] = []
    powers: list[float] = []
    for record in records:
        cube = load_pl(folder, record.file_name, log_scale=False, y_axis=effective_y_axis)
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
    return PowerSeriesResult(
        cube=cube,
        group_key=selected_key,
        records=records,
        groups=groups,
        sources=sources,
    )


