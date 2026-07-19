from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

from core.loader import DataCube


COMPARE_PANEL_ORDER = ("KK", "KKp", "KpK", "KpKp")


@dataclass
class SplitColorScale:
    split_x: float
    left_vmin: float
    left_vmax: float
    right_vmin: float
    right_vmax: float
    show_boundary: bool = True


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
    y_axis_log: bool = False
    center_zero: bool = False
    clip_outliers: bool = False
    split_scale: SplitColorScale | None = None


@dataclass
class HeatmapRender:
    primary: object
    secondary: object | None = None
    split_x: float | None = None

    @property
    def is_split(self) -> bool:
        return self.secondary is not None and self.split_x is not None


def _norm_from_params(z: np.ndarray, params: HeatmapParams):
    if params.log_scale:
        vmin = max(float(params.vmin), 1e-12)
        vmax = max(float(params.vmax), vmin * 1.01)
        return LogNorm(vmin=vmin, vmax=vmax, clip=True)
    if params.center_zero and params.vmin < 0 < params.vmax:
        return TwoSlopeNorm(vmin=params.vmin, vcenter=0.0, vmax=params.vmax)
    return Normalize(vmin=params.vmin, vmax=params.vmax)


def _norm_from_bounds(
    z: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    log_scale: bool,
    center_zero: bool,
):
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Each color-scale region requires finite limits with vmin < vmax.")
    if log_scale:
        if vmin <= 0.0:
            raise ValueError("Log color scale requires a positive vmin in both x regions.")
        return LogNorm(vmin=vmin, vmax=vmax, clip=True)
    if center_zero and vmin < 0.0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def resolve_split_boundary(energy: np.ndarray, split_x: float) -> tuple[int, float]:
    """Snap an x split to the nearest interior heatmap-cell boundary."""
    values = np.asarray(energy, float).ravel()
    if values.size < 2:
        raise ValueError("Split color scale requires at least two x data columns.")
    if np.any(~np.isfinite(values)):
        raise ValueError("Split color scale requires finite x coordinates.")
    edges = _axis_edges_from_centers(values)
    interior = edges[1:-1]
    index = int(np.argmin(np.abs(interior - float(split_x)))) + 1
    return index, float(edges[index])


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


def _axis_edges_from_centers(values: np.ndarray, *, log_scale: bool = False) -> np.ndarray:
    v = np.asarray(values, float).ravel()
    if v.size == 0:
        raise ValueError("Axis cannot be empty.")
    if log_scale:
        if np.any(~np.isfinite(v)) or np.any(v <= 0):
            raise ValueError("Log axis requires finite positive coordinates.")
        if v.size == 1:
            factor = 10.0 ** 0.05
            return np.asarray([float(v[0]) / factor, float(v[0]) * factor], float)
        mids = np.sqrt(v[:-1] * v[1:])
        first = float(v[0]) / np.sqrt(float(v[1]) / float(v[0]))
        last = float(v[-1]) * np.sqrt(float(v[-1]) / float(v[-2]))
        return np.concatenate(([first], mids, [last]))
    if v.size == 1:
        pad = max(1e-9, abs(float(v[0])) * 1e-6, 0.5)
        return np.asarray([float(v[0]) - pad, float(v[0]) + pad], float)
    mids = 0.5 * (v[:-1] + v[1:])
    first = float(v[0]) - 0.5 * float(v[1] - v[0])
    last = float(v[-1]) + 0.5 * float(v[-1] - v[-2])
    return np.concatenate(([first], mids, [last]))


def plot_heatmap(ax: Axes, cube: DataCube, params: HeatmapParams):
    z = np.asarray(cube.Z, float)
    e = np.asarray(cube.energy, float).ravel()
    g = np.asarray(cube.gate, float).ravel()
    e_edges = _axis_edges_from_centers(e)
    g_edges = _axis_edges_from_centers(g, log_scale=bool(params.y_axis_log))
    split = params.split_scale
    if split is None:
        if params.clip_outliers:
            z = np.minimum(z, params.vmax)
        norm = _norm_from_params(z, params)
        images = [
            ax.pcolormesh(
                e_edges,
                g_edges,
                z,
                shading="flat",
                cmap=params.cmap,
                norm=norm,
            )
        ]
        applied_split = None
    else:
        xlo, xhi = sorted((float(params.xlim[0]), float(params.xlim[1])))
        if not xlo < float(split.split_x) < xhi:
            raise ValueError("Split position x0 must be strictly between xmin and xmax.")
        split_index, applied_split = resolve_split_boundary(e, float(split.split_x))
        if not xlo < applied_split < xhi:
            raise ValueError(
                "The nearest data-cell boundary for x0 is outside the visible x range. "
                "Move x0 farther from xmin or xmax."
            )
        left_z = np.asarray(z, float).copy()
        right_z = np.asarray(z, float).copy()
        if params.clip_outliers:
            left_z = np.minimum(left_z, float(split.left_vmax))
            right_z = np.minimum(right_z, float(split.right_vmax))
        left_mask = np.zeros(left_z.shape, dtype=bool)
        right_mask = np.zeros(right_z.shape, dtype=bool)
        left_mask[:, split_index:] = True
        right_mask[:, :split_index] = True
        left_norm = _norm_from_bounds(
            left_z,
            vmin=split.left_vmin,
            vmax=split.left_vmax,
            log_scale=bool(params.log_scale),
            center_zero=bool(params.center_zero),
        )
        right_norm = _norm_from_bounds(
            right_z,
            vmin=split.right_vmin,
            vmax=split.right_vmax,
            log_scale=bool(params.log_scale),
            center_zero=bool(params.center_zero),
        )
        images = [
            ax.pcolormesh(
                e_edges,
                g_edges,
                np.ma.array(left_z, mask=left_mask),
                shading="flat",
                cmap=params.cmap,
                norm=left_norm,
            ),
            ax.pcolormesh(
                e_edges,
                g_edges,
                np.ma.array(right_z, mask=right_mask),
                shading="flat",
                cmap=params.cmap,
                norm=right_norm,
            ),
        ]
        if split.show_boundary:
            ax.axvline(
                applied_split,
                color="#202020",
                linewidth=0.9,
                linestyle="--",
                alpha=0.8,
                zorder=15,
            )
    # Guard matplotlib cursor formatting against inf/nan ranges that can raise in colorizer.
    for image in images:
        try:
            image._format_cursor_data_override = lambda value: (
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
    if params.y_axis_log:
        if ymin <= 0 or ymax <= 0:
            positive = g[np.isfinite(g) & (g > 0)]
            if positive.size == 0:
                raise ValueError("Log power axis requires positive power values.")
            ymin = float(np.nanmin(positive))
            ymax = float(np.nanmax(positive))
        ax.set_yscale("log")
    ax.set_ylim((ymin, ymax))
    return HeatmapRender(
        primary=images[0],
        secondary=(images[1] if len(images) > 1 else None),
        split_x=applied_split,
    )


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
        y_axis_log=params.y_axis_log,
        center_zero=params.center_zero,
        clip_outliers=params.clip_outliers,
        split_scale=params.split_scale,
    )
    render = plot_heatmap(ax, cube, panel)
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
    return render


