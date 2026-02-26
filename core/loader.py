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


def _validate_cube_arrays(energy: np.ndarray, gate: np.ndarray, Z: np.ndarray, *, context: str) -> None:
    e = np.asarray(energy).ravel()
    g = np.asarray(gate).ravel()
    z = np.asarray(Z)

    if e.size == 0 or g.size == 0 or z.size == 0:
        raise ValueError(f"{context}: empty energy/gate/Z data.")
    if z.ndim != 2:
        raise ValueError(f"{context}: Z must be 2D, got shape {z.shape}.")
    if z.shape not in {(g.size, e.size), (e.size, g.size)}:
        raise ValueError(
            f"{context}: Z shape {z.shape} does not match gate ({g.size}) x energy ({e.size})."
        )

def peek_y_axis_options(user_folder: str, file_name: str):
    # Preferred: use the implementation that owns _load_canonical (processing_run)
    if hasattr(P, "peek_y_axis_options"):
        return P.peek_y_axis_options(user_folder, file_name)

    # Fallback: derive from _load_canonical if it exists there
    d = P._load_canonical(user_folder, file_name, y_axis="auto")  # type: ignore[attr-defined]
    opts = d.get("available_axes", ["Vbg", "Vtg"])
    default = d.get("default_axis", opts[0] if opts else "Vtg")
    if default not in opts and opts:
        default = opts[0]
    return opts, default


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
    _validate_cube_arrays(res["energy"], res["gate_axis"], res["Z"], context="PL load")
    return DataCube(
        energy=res["energy"],
        gate=res["gate_axis"],
        Z=res["Z"],
        gate_label=res.get("gate_label", "Gate (V)"),
        title=res.get("title", file_name),
        cbar_label="PL (a.u.)",
    )


def build_external_baseline(user_folder: str, files: Sequence[str], *, which: str = "last") -> dict:
    """
    Returns dict with keys: energy, I0

    which:
      - "first" : use first frame in each file
      - "last"  : use last frame in each file
      - "all"   : average ALL frames within each file, then average across files
    """
    if not files:
        raise ValueError("build_external_baseline: 'files' is empty.")

    w = (which or "last").strip().lower()
    alias = {
        "first": "first",
        "1st": "first",
        "start": "first",

        "last": "last",
        "end": "last",

        "all": "all",
        "avg": "all",
        "mean": "all",
        "all_frames": "all",
        "all frames": "all",
        "frames": "all",
    }
    w = alias.get(w, w)
    if w not in ("first", "last", "all"):
        raise ValueError(f"Unknown which='{which}'. Use 'first', 'last', or 'all'.")

    energy, I0 = P.build_external_baseline_avg(
        user_folder=user_folder,
        files_zero=list(files),
        which=w,
        save_npz=None,
    )
    return {"energy": np.asarray(energy), "I0": np.asarray(I0)}


def load_drr_avg(
    user_folder: str,
    files: Sequence[str],
    *,
    bg_mode: str,
    y_axis: str = "auto",  
    external_vector: Optional[np.ndarray] = None,
    derivative: Optional[int] = None,
    dE_window_pts: int = 20,
    dE_polyorder: int = 2,
    dE_oversample: float = 1.0,
    dE_interp_kind: str = "cubic",
    dE_origin_like: bool = False,        
    dE_pad_flat_edges: bool = True,      
) -> DataCube:
    res = P.process_ref_avg(
        user_folder=user_folder,
        files=list(files),
        bg_mode=bg_mode,
        y_axis=y_axis,  
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
        dE_origin_like=dE_origin_like,
        dE_pad_flat_edges=dE_pad_flat_edges,
        center_zero=True,
    )

    _validate_cube_arrays(res["energy"], res["gate_axis"], res["Z_out"], context="DRR load")

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
