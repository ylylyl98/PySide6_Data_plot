from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm


@dataclass
class HeatmapConfig:
    title: str
    xlabel: str = "Energy"
    ylabel: str = "Gate (V)"
    cbar_label: str = ""
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    center_zero: bool = False
    log_scale: bool = False
    xlim: Tuple[Optional[float], Optional[float]] = (None, None)
    ylim: Tuple[Optional[float], Optional[float]] = (None, None)


def _auto_limits(Z: np.ndarray, center_zero: bool) -> tuple[float, float]:
    z = np.asarray(Z, float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return -1.0, 1.0
    lo = float(np.nanpercentile(z, 1))
    hi = float(np.nanpercentile(z, 99))
    if center_zero:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def build_heatmap_fig(energy: np.ndarray, gate: np.ndarray, Z: np.ndarray, cfg: HeatmapConfig):
    energy = np.asarray(energy).ravel()
    gate = np.asarray(gate).ravel()
    Z = np.asarray(Z)

    vmin, vmax = cfg.vmin, cfg.vmax
    if vmin is None or vmax is None:
        alo, ahi = _auto_limits(Z, cfg.center_zero)
        vmin = alo if vmin is None else vmin
        vmax = ahi if vmax is None else vmax

    # Choose normalization
    norm = None
    if cfg.log_scale:
        # LogNorm requires positive values
        zpos = Z[np.isfinite(Z) & (Z > 0)]
        floor = float(np.nanmin(zpos)) if zpos.size else 1e-12
        vmin_eff = max(floor, float(vmin) if vmin is not None else floor)
        vmax_eff = max(vmin_eff * 1.01, float(vmax) if vmax is not None else vmin_eff * 100)
        norm = LogNorm(vmin=vmin_eff, vmax=vmax_eff)
    else:
        if cfg.center_zero:
            norm = TwoSlopeNorm(vmin=float(vmin), vcenter=0.0, vmax=float(vmax))
        else:
            norm = Normalize(vmin=float(vmin), vmax=float(vmax))

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[float(energy[0]), float(energy[-1]), float(gate[0]), float(gate[-1])],
        norm=norm,
    )

    ax.set_title(cfg.title)
    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)

    if cfg.xlim != (None, None):
        lo, hi = cfg.xlim
        if lo is not None or hi is not None:
            ax.set_xlim(left=lo, right=hi)
    if cfg.ylim != (None, None):
        lo, hi = cfg.ylim
        if lo is not None or hi is not None:
            ax.set_ylim(bottom=lo, top=hi)

    cbar = fig.colorbar(im, ax=ax)
    if cfg.cbar_label:
        cbar.set_label(cfg.cbar_label)

    return fig


def build_spectrum_fig(energy: np.ndarray, Z_row: np.ndarray, *, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 3.0), constrained_layout=True)
    ax.plot(energy, Z_row)
    ax.set_title(title)
    ax.set_xlabel("Energy")
    ax.set_ylabel(ylabel)
    return fig


def save_fig_png(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    return out_path
