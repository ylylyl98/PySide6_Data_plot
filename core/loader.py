from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from core import processing_run as P


@dataclass
class DataCube:
    energy: np.ndarray
    gate: np.ndarray
    Z: np.ndarray
    gate_label: str
    title: str
    cbar_label: str


def load_pl(user_folder: str, file_name: str, *, log_scale: bool = False) -> DataCube:
    res = P.process_pl(
        user_folder=user_folder,
        file=file_name,
        plot_interactive=False,
        save_png=False,
        save_dat_file=False,
        move_original=False,
        pl_scales=("log" if log_scale else "linear",),
        open_both_interactive=False,
    )
    return DataCube(
        energy=res["energy"],
        gate=res["gate_axis"],
        Z=res["Z"],
        gate_label=res.get("gate_label", "Gate (V)"),
        title=res.get("title", file_name),
        cbar_label="PL (a.u.)",
    )


def build_external_baseline(user_folder: str, files: Sequence[str], *, which: str = "last") -> dict:
    # Returns dict with keys: energy, I0
    energy, I0 = P.build_external_baseline_avg(
        user_folder=user_folder,
        files_zero=list(files),
        which=which,
        save_npz=None,
    )
    return {"energy": np.asarray(energy), "I0": np.asarray(I0)}


def load_drr_avg(
    user_folder: str,
    files: Sequence[str],
    *,
    bg_mode: str,
    external_vector: Optional[np.ndarray] = None,
    derivative: Optional[int] = None,
    dE_window_pts: int = 20,
    dE_polyorder: int = 2,
    dE_oversample: float = 1.0,
    dE_interp_kind: str = "cubic",
) -> DataCube:
    res = P.process_ref_avg(
        user_folder=user_folder,
        files=list(files),
        bg_mode=bg_mode,
        external_vector=external_vector,
        use_global_background=False,
        plot_interactive=False,
        save_png=False,
        save_dat_file=False,
        move_original=False,
        derivative=derivative,
        dE_window_pts=dE_window_pts,
        dE_polyorder=dE_polyorder,
        dE_oversample=dE_oversample,
        dE_interp_kind=dE_interp_kind,
        center_zero=True,
    )

    # use Z_out (DR/R or derivative)
    cbar = "DR/R" if derivative is None else ("d(DR/R)/dE" if derivative == 1 else "d²(DR/R)/dE²")
    return DataCube(
        energy=res["energy"],
        gate=res["gate_axis"],
        Z=res["Z_out"],
        gate_label=res.get("gate_label", "Gate (V)"),
        title=res.get("title", "DR/R"),
        cbar_label=cbar,
    )
