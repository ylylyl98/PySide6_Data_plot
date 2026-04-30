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


def _parse_angle_token(text: str) -> float:
    return float(str(text).replace("p", ".").replace("P", "."))


def parse_compare_in_out_angles(file_name: str) -> tuple[float | None, float | None]:
    text = str(Path(file_name).stem)
    number = r"([\-+]?\d+(?:[pP\.]\d+)?)"
    unit = r"\s*(?:deg|degree)(?=$|[^A-Za-z0-9])"
    named_patterns = (
        (
            rf"in[^0-9\-+]*{number}{unit}.*out[^0-9\-+]*{number}{unit}",
            False,
        ),
        (
            rf"out[^0-9\-+]*{number}{unit}.*in[^0-9\-+]*{number}{unit}",
            True,
        ),
    )
    for pattern, reverse in named_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        first = _parse_angle_token(match.group(1))
        second = _parse_angle_token(match.group(2))
        return (second, first) if reverse else (first, second)

    rot1 = re.search(rf"rot\s*1[^0-9\-+]*{number}{unit}", text, flags=re.IGNORECASE)
    rot2 = re.search(rf"rot\s*2[^0-9\-+]*{number}{unit}", text, flags=re.IGNORECASE)
    if rot1 and rot2:
        return _parse_angle_token(rot1.group(1)), _parse_angle_token(rot2.group(1))

    return None, None


def parse_compare_gate_condition(file_name: str) -> str:
    stem = str(Path(file_name).stem)
    coeff = r"(?:\d+(?:[pP\.]\d+)?)?"
    pattern = re.compile(
        rf"(?<![A-Za-z0-9])({coeff}\s*TG\s*[+-]\s*{coeff}\s*BG\s*=\s*0(?:\.0+)?)(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    match = pattern.search(stem)
    if not match:
        return ""
    token = re.sub(r"\s+", "", match.group(1))
    token = re.sub("tg", "TG", token, flags=re.IGNORECASE)
    token = re.sub("bg", "BG", token, flags=re.IGNORECASE)
    return token


def classify_compare_channel(
    file_name: str,
    *,
    in_k_angle: float,
    out_k_angle: float,
    tolerance: float = 45.0,
) -> str | None:
    in_angle, out_angle = parse_compare_in_out_angles(file_name)
    if in_angle is None or out_angle is None:
        return None
    in_is_k = abs(((float(in_angle) - float(in_k_angle) + 180.0) % 360.0) - 180.0) <= float(tolerance)
    out_is_k = abs(((float(out_angle) - float(out_k_angle) + 180.0) % 360.0) - 180.0) <= float(tolerance)
    if in_is_k and out_is_k:
        return "KK"
    if in_is_k and (not out_is_k):
        return "KKp"
    if (not in_is_k) and out_is_k:
        return "KpK"
    return "KpKp"


def coherent_compare_auto_assignment(
    file_names: Sequence[str],
    *,
    in_k_angle: float,
    out_k_angle: float,
) -> tuple[dict[str, str], dict[str, list[str]], str, list[str]]:
    groups: dict[str, list[tuple[int, str, str]]] = {}
    for idx, file_name in enumerate(file_names):
        key = classify_compare_channel(
            file_name,
            in_k_angle=in_k_angle,
            out_k_angle=out_k_angle,
        )
        if key is None:
            continue
        gate_key = parse_compare_gate_condition(file_name) or "__ungrouped__"
        groups.setdefault(gate_key, []).append((idx, key, file_name))

    if not groups:
        return {}, {}, "", []

    def _score(item: tuple[str, list[tuple[int, str, str]]]) -> tuple[int, int, int, int]:
        _gate_key, records = item
        keys = {key for _idx, key, _file_name in records}
        has_kk_pair = int("KK" in keys and "KKp" in keys)
        first_idx = min(idx for idx, _key, _file_name in records)
        return (has_kk_pair, len(keys), len(records), -first_idx)

    selected_gate, records = max(groups.items(), key=_score)
    found: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for _idx, key, file_name in sorted(records, key=lambda item: item[0]):
        if key in found:
            duplicates.setdefault(key, [found[key]]).append(file_name)
            continue
        found[key] = file_name

    gate_groups = [gate for gate in groups if gate != "__ungrouped__"]
    gate_label = "" if selected_gate == "__ungrouped__" else selected_gate
    return found, duplicates, gate_label, gate_groups


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


def background_correct_cube(cube: DataCube, background: float, *, title: str | None = None) -> DataCube:
    z = np.asarray(cube.Z, float) - float(background)
    return DataCube(
        energy=np.asarray(cube.energy, float).copy(),
        gate=np.asarray(cube.gate, float).copy(),
        Z=z,
        gate_label=cube.gate_label,
        title=title if title is not None else cube.title,
        cbar_label="PL corr. (a.u.)",
    )


def estimate_constant_background(
    cubes: Sequence[DataCube] | Dict[str, DataCube],
    *,
    percentile: float = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> float:
    """Estimate one scalar background from finite values across cubes."""
    iterable = cubes.values() if isinstance(cubes, dict) else cubes
    vals: list[np.ndarray] = []
    for cube in iterable:
        z = np.asarray(cube.Z, float)
        e = np.asarray(cube.energy, float).ravel()
        g = np.asarray(cube.gate, float).ravel()
        if z.shape != (g.size, e.size):
            roi = z
        else:
            x_mask = np.ones(e.shape, dtype=bool)
            y_mask = np.ones(g.shape, dtype=bool)
            if xlim is not None:
                x0, x1 = sorted((float(xlim[0]), float(xlim[1])))
                x_mask = (e >= x0) & (e <= x1)
            if ylim is not None:
                y0, y1 = sorted((float(ylim[0]), float(ylim[1])))
                y_mask = (g >= y0) & (g <= y1)
            roi = z[np.ix_(y_mask, x_mask)] if np.any(y_mask) and np.any(x_mask) else z
        finite = roi[np.isfinite(roi)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return 0.0
    p = float(np.clip(percentile, 0.0, 100.0))
    return float(np.nanpercentile(np.concatenate(vals), p))


def _require_compatible_grid(a: DataCube, b: DataCube, *, context: str) -> None:
    ea = np.asarray(a.energy, float).ravel()
    eb = np.asarray(b.energy, float).ravel()
    ga = np.asarray(a.gate, float).ravel()
    gb = np.asarray(b.gate, float).ravel()
    za = np.asarray(a.Z, float)
    zb = np.asarray(b.Z, float)
    if ea.shape != eb.shape or ga.shape != gb.shape or za.shape != zb.shape:
        raise ValueError(f"{context}: KK and KKp grids do not have matching shapes.")
    if not np.allclose(ea, eb, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{context}: KK and KKp energy axes do not match.")
    if not np.allclose(ga, gb, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{context}: KK and KKp gate axes do not match.")


def valley_polarization_cube(
    kk_cube: DataCube,
    kkp_cube: DataCube,
    *,
    background: float,
    denom_eps: float = 1e-12,
    title: str = "Valley Polarization",
) -> DataCube:
    kk_corr = background_correct_cube(kk_cube, background, title="KK")
    kkp_corr = background_correct_cube(kkp_cube, background, title="KKp")
    _require_compatible_grid(kk_corr, kkp_corr, context="VP")
    vp = vp_map(kk_corr.Z, kkp_corr.Z, eps=denom_eps)
    return DataCube(
        energy=np.asarray(kk_corr.energy, float).copy(),
        gate=np.asarray(kk_corr.gate, float).copy(),
        Z=vp,
        gate_label=kk_corr.gate_label,
        title=title,
        cbar_label="VP",
    )


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
