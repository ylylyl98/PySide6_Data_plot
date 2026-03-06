from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter

from core.loader import DataCube


REP_SUB_SUFFIX_RE = re.compile(r"_rep(?P<rep>\d+)_(?P<sub>\d+)$", re.IGNORECASE)
REP_ONLY_SUFFIX_RE = re.compile(r"(?:\$_|_)rep(?P<rep>\d{1,3})$", re.IGNORECASE)
RUN_SUFFIX_RE = re.compile(r"(?:\$_|_)(?P<run>\d{3,})$")


@dataclass(frozen=True)
class Limits:
    vmin: float
    vmax: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def split_group_and_sort_key(filename: str) -> tuple[str, tuple[int, int, int]]:
    stem = Path(filename).stem

    m = REP_SUB_SUFFIX_RE.search(stem)
    if m:
        condition_key = re.sub(r"_rep\d+_\d+$", "", stem, flags=re.IGNORECASE)
        return condition_key, (0, int(m.group("rep")), int(m.group("sub")))

    m = REP_ONLY_SUFFIX_RE.search(stem)
    if m:
        prefix = stem[: m.start()]
        return prefix, (1, int(m.group("rep")), 0)

    m = RUN_SUFFIX_RE.search(stem)
    if m:
        prefix = stem[: m.start()]
        return prefix, (1, int(m.group("run")), 0)

    return stem, (2, 0, 0)


def group_measurement_files(files: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[tuple[tuple[int, int, int], str]]] = {}
    for file_name in files:
        gk, sort_key = split_group_and_sort_key(file_name)
        groups.setdefault(gk, []).append((sort_key, file_name))

    out: Dict[str, List[str]] = {}
    for key in sorted(groups):
        out[key] = [name for _, name in sorted(groups[key], key=lambda item: item[0])]
    return out


def compute_auto_limits(cube: DataCube, *, log_scale: bool = False, low: float = 0.01, high: float = 99.99) -> Limits:
    z = np.asarray(cube.Z, float)
    e = np.asarray(cube.energy, float).ravel()
    g = np.asarray(cube.gate, float).ravel()
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ValueError("Data contains no finite values.")

    if log_scale:
        pos = z[np.isfinite(z) & (z > 0)]
        if pos.size:
            vmin, vmax = np.nanpercentile(pos, [low, high])
            vmin = float(max(vmin, 1e-12))
            vmax = float(max(vmax, vmin * 1.01))
        else:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        vmin, vmax = np.nanpercentile(finite, [low, high])
        vmin, vmax = float(vmin), float(vmax)

    return Limits(
        vmin=vmin,
        vmax=vmax,
        xmin=float(np.nanmin(e)),
        xmax=float(np.nanmax(e)),
        ymin=float(np.nanmin(g)),
        ymax=float(np.nanmax(g)),
    )


def nearest_gate_spectrum(cube: DataCube, gate_value: float) -> tuple[float, np.ndarray]:
    gate = np.asarray(cube.gate, float).ravel()
    idx = int(np.argmin(np.abs(gate - float(gate_value))))
    return float(gate[idx]), np.asarray(cube.Z, float)[idx, :]


def subtract_background(
    z: np.ndarray,
    energy: np.ndarray,
    *,
    method: str,
    p_low: float,
    roi: tuple[float, float] | None,
    clip_to_zero: bool,
) -> np.ndarray:
    z = np.asarray(z, float)
    e = np.asarray(energy, float).ravel()

    if method == "none":
        out = z.copy()
    elif method == "scalar_percentile":
        finite = z[np.isfinite(z)]
        bg = float(np.nanpercentile(finite, p_low)) if finite.size else 0.0
        out = z - bg
    elif method == "roi_median_scalar":
        if roi is None:
            out = z.copy()
        else:
            r0, r1 = roi
            mask = (e >= min(r0, r1)) & (e <= max(r0, r1))
            roi_vals = z[:, mask] if mask.any() else z
            finite = roi_vals[np.isfinite(roi_vals)]
            bg = float(np.nanmedian(finite)) if finite.size else 0.0
            out = z - bg
    else:
        bg_e = np.nanpercentile(z, p_low, axis=0)
        out = z - bg_e[None, :]

    if clip_to_zero:
        out = np.maximum(out, 0.0)
    return out


def vp_map(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.asarray(a, float) + np.asarray(b, float)
    out = (np.asarray(a, float) - np.asarray(b, float)) / np.where(denom > eps, denom, np.nan)
    return np.clip(out, -1.0, 1.0)


def vp_curve_vs_gate(
    energy: np.ndarray,
    gate: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    roi: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(energy, float).ravel()
    g = np.asarray(gate, float).ravel()
    r0, r1 = roi
    mask = (e >= min(r0, r1)) & (e <= max(r0, r1))
    if not mask.any():
        mask = np.ones_like(e, dtype=bool)
    ia = np.trapezoid(np.asarray(a, float)[:, mask], e[mask], axis=1)
    ib = np.trapezoid(np.asarray(b, float)[:, mask], e[mask], axis=1)
    vp = (ia - ib) / np.where((ia + ib) > 1e-12, (ia + ib), np.nan)
    return g, np.clip(vp, -1.0, 1.0)


def _clamp_sg_window(requested: int, *, n_energy: int, polyorder: int) -> int:
    win = int(requested)
    if win % 2 == 0:
        win += 1
    min_valid = max(3, polyorder + 2)
    if min_valid % 2 == 0:
        min_valid += 1
    max_valid = max(min_valid, int(n_energy) if int(n_energy) % 2 == 1 else int(n_energy) - 1)
    win = max(min_valid, min(win, max_valid))
    if win % 2 == 0:
        win = max(min_valid, win - 1)
    return int(win)


def clamp_sg_window(requested: int, *, n_energy: int, polyorder: int) -> int:
    return _clamp_sg_window(requested, n_energy=n_energy, polyorder=polyorder)


def apply_sg_derivative_energy(
    cube: DataCube,
    *,
    derivative: int | None,
    window_length: int,
    polyorder: int = 2,
) -> tuple[DataCube, int]:
    if derivative not in (None, 1, 2):
        raise ValueError("derivative must be None, 1, or 2.")
    if derivative is None:
        return cube, int(window_length)

    z = np.asarray(cube.Z, float)
    e = np.asarray(cube.energy, float).ravel()
    if z.ndim != 2 or e.size < 5:
        raise ValueError("Derivative needs 2D data with at least 5 energy points.")

    used_win = _clamp_sg_window(int(window_length), n_energy=e.size, polyorder=int(polyorder))
    delta = float(np.nanmedian(np.diff(e)))
    if not np.isfinite(delta) or delta == 0:
        delta = float(np.nanmean(np.diff(e)))
    if not np.isfinite(delta) or delta == 0:
        delta = 1.0

    z_out = savgol_filter(
        z,
        window_length=int(used_win),
        polyorder=int(polyorder),
        deriv=int(derivative),
        delta=delta,
        axis=1,
        mode="interp",
    )
    cbar = "d(DR/R)/dE" if int(derivative) == 1 else "d2(DR/R)/dE2"
    title = cube.title if cube.title else "DR/R"
    title = f"{title} ({cbar})"
    return (
        DataCube(
            energy=np.asarray(cube.energy, float).copy(),
            gate=np.asarray(cube.gate, float).copy(),
            Z=np.asarray(z_out, float).copy(),
            gate_label=cube.gate_label,
            title=title,
            cbar_label=cbar,
        ),
        int(used_win),
    )
