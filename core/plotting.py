from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

from core.loader import DataCube
from core.processing import nearest_gate_spectrum


COMPARE_PANEL_ORDER = ("KK", "KKp", "KpK", "KpKp")


@dataclass
class HeatmapParams:
    title: str
    xlabel: str
    ylabel: str
    cbar_label: str
    vmin: float
    vmax: float
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    cmap: str = "viridis"
    log_scale: bool = False
    center_zero: bool = False
    clip_outliers: bool = False


def _norm_from_params(z: np.ndarray, params: HeatmapParams):
    if params.log_scale:
        vmin = max(float(params.vmin), 1e-12)
        vmax = max(float(params.vmax), vmin * 1.01)
        return LogNorm(vmin=vmin, vmax=vmax, clip=True)
    if params.center_zero and params.vmin < 0 < params.vmax:
        return TwoSlopeNorm(vmin=params.vmin, vcenter=0.0, vmax=params.vmax)
    return Normalize(vmin=params.vmin, vmax=params.vmax)


def _sanitize_limits(requested: tuple[float, float], data_min: float, data_max: float) -> tuple[float, float]:
    lo_req, hi_req = float(requested[0]), float(requested[1])
    dmin = float(data_min)
    dmax = float(data_max)
    if not np.isfinite(dmin) or not np.isfinite(dmax):
        dmin, dmax = 0.0, 1.0
    if dmin == dmax:
        dmin -= 0.5
        dmax += 0.5
    lo = lo_req if np.isfinite(lo_req) else dmin
    hi = hi_req if np.isfinite(hi_req) else dmax
    if lo == hi:
        pad = max(1e-9, abs(lo) * 1e-6, (dmax - dmin) * 1e-3)
        lo -= pad
        hi += pad
    return (lo, hi)


def _axis_edges_from_centers(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, float).ravel()
    if v.size == 0:
        raise ValueError("Axis cannot be empty.")
    if v.size == 1:
        pad = max(1e-9, abs(float(v[0])) * 1e-6, 0.5)
        return np.asarray([float(v[0]) - pad, float(v[0]) + pad], float)
    mids = 0.5 * (v[:-1] + v[1:])
    first = float(v[0]) - 0.5 * float(v[1] - v[0])
    last = float(v[-1]) + 0.5 * float(v[-1] - v[-2])
    return np.concatenate(([first], mids, [last]))


def plot_heatmap(ax: Axes, cube: DataCube, params: HeatmapParams):
    z = np.asarray(cube.Z, float)
    if params.clip_outliers:
        z = np.minimum(z, params.vmax)
    norm = _norm_from_params(z, params)
    e = np.asarray(cube.energy, float).ravel()
    g = np.asarray(cube.gate, float).ravel()
    e_edges = _axis_edges_from_centers(e)
    g_edges = _axis_edges_from_centers(g)
    im = ax.pcolormesh(
        e_edges,
        g_edges,
        z,
        shading="flat",
        cmap=params.cmap,
        norm=norm,
    )
    # Guard matplotlib cursor formatting against inf/nan ranges that can raise in colorizer.
    try:
        im._format_cursor_data_override = lambda value: (
            "n/a" if not np.isfinite(float(value)) else f"{float(value):.6g}"
        )
    except Exception:
        pass
    ax.set_title(params.title)
    ax.set_xlabel(params.xlabel)
    ax.set_ylabel(params.ylabel)
    xmin, xmax = _sanitize_limits(params.xlim, float(np.nanmin(cube.energy)), float(np.nanmax(cube.energy)))
    ymin, ymax = _sanitize_limits(params.ylim, float(np.nanmin(cube.gate)), float(np.nanmax(cube.gate)))
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    return im


def plot_pl(ax: Axes, cube: DataCube, params: HeatmapParams):
    return plot_heatmap(ax, cube, params)


def plot_drr(ax: Axes, cube: DataCube, params: HeatmapParams):
    return plot_heatmap(ax, cube, params)


def plot_compare_panel(ax: Axes, label: str, cube: DataCube, params: HeatmapParams):
    panel = HeatmapParams(
        title=cube.title,
        xlabel=params.xlabel,
        ylabel=params.ylabel,
        cbar_label=params.cbar_label,
        vmin=params.vmin,
        vmax=params.vmax,
        xlim=params.xlim,
        ylim=params.ylim,
        cmap=params.cmap,
        log_scale=params.log_scale,
        center_zero=params.center_zero,
        clip_outliers=params.clip_outliers,
    )
    im = plot_heatmap(ax, cube, panel)
    ax.text(
        0.98,
        0.98,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#111",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.78),
        zorder=40,
    )
    return im


def plot_spectrum(ax: Axes, cube: DataCube, gate_value: float, *, ylabel: str):
    gate_used, y = nearest_gate_spectrum(cube, gate_value)
    x = np.asarray(cube.energy, float).ravel()
    ax.plot(x, np.asarray(y, float), linewidth=1.3)
    ax.set_title(f"Spectrum @ {gate_used:.6g} V")
    ax.set_xlabel("Photon Energy (eV)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_vp(ax: Axes, energy: np.ndarray, gate: np.ndarray, vp_z: np.ndarray):
    e = np.asarray(energy, float).ravel()
    g = np.asarray(gate, float).ravel()
    z = np.asarray(vp_z, float)
    im = ax.pcolormesh(
        _axis_edges_from_centers(e),
        _axis_edges_from_centers(g),
        z,
        shading="flat",
        cmap="RdBu_r",
        norm=Normalize(vmin=-1.0, vmax=1.0),
    )
    ax.set_title("Valley Polarization")
    ax.set_xlabel("Photon Energy (eV)")
    ax.set_ylabel("Gate (V)")
    return im


def render_compare_grid(fig, cubes: Dict[str, DataCube], params: HeatmapParams):
    keys = [key for key in COMPARE_PANEL_ORDER if key in cubes]
    if len(keys) <= 2:
        axes = fig.subplots(1, len(keys))
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(fig.subplots(2, 2)).ravel()
    images = []
    for idx, key in enumerate(keys):
        im = plot_compare_panel(axes[idx], key, cubes[key], params)
        images.append(im)
    return images
