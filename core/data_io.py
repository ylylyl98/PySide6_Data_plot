from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from core.file_ops import archive_all, archive_selected, list_root_csvs, restore_all
from core.loader import (
    DataCube,
    build_external_baseline,
    load_drr_avg,
    load_pl,
    peek_y_axis_options,
)


DEFAULT_ARCHIVE = "Initial data after processing"
DEFAULT_PROCESSED = "Processed Data"


@dataclass(frozen=True)
class CompareSelection:
    kk: str
    kkp: str
    kpk: str | None = None
    kpkp: str | None = None

    def as_pairs(self) -> Dict[str, str]:
        pairs: Dict[str, str] = {"KK": self.kk, "KKp": self.kkp}
        if self.kpk:
            pairs["KpK"] = self.kpk
        if self.kpkp:
            pairs["KpKp"] = self.kpkp
        return pairs


def processed_dir(folder: str, processed_name: str = DEFAULT_PROCESSED) -> Path:
    out = Path(folder) / processed_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_csv_files(folder: str) -> List[str]:
    return list_root_csvs(folder)


def move_all_to_archive(folder: str, archive_name: str = DEFAULT_ARCHIVE) -> int:
    return archive_all(folder, archive_name)


def restore_all_from_archive(folder: str, archive_name: str = DEFAULT_ARCHIVE) -> int:
    return restore_all(folder, archive_name)


def move_selected_to_archive(folder: str, file_names: Sequence[str], archive_name: str = DEFAULT_ARCHIVE) -> int:
    return archive_selected(folder, list(file_names), archive_name)


def load_pl_cube(folder: str, file_name: str, *, log_scale: bool = False) -> DataCube:
    return load_pl(folder, file_name, log_scale=log_scale)


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
        derivative=derivative,
    )


def load_compare_cubes(
    folder: str,
    selection: CompareSelection,
    *,
    log_scale: bool = False,
) -> Dict[str, DataCube]:
    return {name: load_pl(folder, fn, log_scale=log_scale) for name, fn in selection.as_pairs().items()}


def get_y_axis_options(folder: str, file_name: str) -> tuple[list[str], str]:
    return peek_y_axis_options(folder, file_name)


def write_heatmap_csv(path: str | Path, energy: Iterable[float], gate: Iterable[float], z: np.ndarray) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    e = np.asarray(list(energy), float).ravel()
    g = np.asarray(list(gate), float).ravel()
    zm = np.asarray(z, float)
    if zm.shape != (g.size, e.size):
        if zm.shape == (e.size, g.size):
            zm = zm.T
        else:
            raise ValueError(f"Z shape {zm.shape} does not match gate/energy axes.")

    table = np.empty((g.size + 1, e.size + 1), float)
    table[0, 0] = np.nan
    table[0, 1:] = e
    table[1:, 0] = g
    table[1:, 1:] = zm

    np.savetxt(out_path, table, delimiter=",", fmt="%.10g")
    return out_path
