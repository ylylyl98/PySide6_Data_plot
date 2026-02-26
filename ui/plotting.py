# ui/plotting.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import re
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm
from matplotlib.ticker import FuncFormatter, NullLocator, NullFormatter
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker
import textwrap


# -----------------------------
# Small helpers
# -----------------------------

def _prettify_title(text: str) -> str:
    """Shorten common tokens for display without changing meaning."""
    if not text:
        return ""

    t = str(text)

    # separators -> spaces
    t = t.replace("~", " ").replace("_", " ")

    # 14.1degree / 42degree -> 14.1 deg / 42 deg
    t = re.sub(r'(\d+(?:\.\d+)?)\s*degree\b', r'\1 deg', t, flags=re.IGNORECASE)

    # 1000ms -> 1s, 500ms -> 0.5s, keep others reasonable
    def _ms_to_s(m):
        ms = float(m.group(1))
        s = ms / 1000.0
        if abs(s - round(s)) < 1e-9:
            s_txt = str(int(round(s)))
        else:
            s_txt = f"{s:g}"
        return f"{s_txt}s"

    # handle "...msx" specifically (drop the trailing 'x')
    t = re.sub(r'(\d+(?:\.\d+)?)\s*msx\b', _ms_to_s, t, flags=re.IGNORECASE)

    # handle normal "...ms" even if followed by something (e.g. "ms_", "ms-", etc.)
    t = re.sub(r'(\d+(?:\.\d+)?)\s*ms\b', _ms_to_s, t, flags=re.IGNORECASE)

    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _wrap_title_to_fig_span(
    fig,
    text: str,
    left_x: float,
    right_x: float,
    *,
    fontsize=16,
    weight="bold",
    max_lines=3,
):
    if not text:
        return ""

    # <<< NEW: abbreviate/clean before wrapping >>>
    text = _prettify_title(text)

    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        wpx, _ = fig.canvas.get_width_height()
    except Exception:
        renderer = None
        wpx = 800

    max_w_px = max(40, (right_x - left_x) * wpx)

    if renderer is None:
        return "\n".join(textwrap.wrap(text, width=40)[:max_lines])

    fp = FontProperties(size=fontsize, weight=weight)
    words = text.split()

    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip() if cur else w
        tw, _, _ = renderer.get_text_width_height_descent(trial, fp, ismath=False)
        if tw <= max_w_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)

    return "\n".join(lines)


# -----------------------------
# Public API
# -----------------------------
@dataclass
class HeatmapConfig:
    title: str = ""
    xlabel: str = "Energy"
    ylabel: str = "Gate (V)"
    cbar_label: str = ""

    vmin: Optional[float] = None
    vmax: Optional[float] = None
    center_zero: bool = False
    log_scale: bool = False

    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None

    cmap: str = "viridis"
    cbar_tick_format: Optional[str] = None   # e.g. "%.3f"
    cbar_integer: bool = False               # for PL integer ticks
    # Keep figure size fixed
    figsize: Tuple[float, float] = (8.0, 6.2)
    dpi: int = 150


def build_heatmap_fig(E, G, Z, cfg):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.ticker import FuncFormatter, NullLocator, NullFormatter

    E = np.asarray(E, float).ravel()
    G = np.asarray(G, float).ravel()
    Z = np.asarray(Z)

    # ---- fixed figure size (never changes) ----
    fig = plt.figure(figsize=getattr(cfg, "figsize", (8.0, 6.2)),
                     dpi=getattr(cfg, "dpi", 150),
                     facecolor="white")

    # ---- fixed heatmap axes box (never changes) ----
    # Header band is everything above axpos.y1
    ax = fig.add_axes([0.08, 0.12, 0.84, 0.72])  # left, bottom, width, height
    axpos = ax.get_position()

    # vmin/vmax safety
    vmin = float(cfg.vmin) if cfg.vmin is not None else float(np.nanmin(Z[np.isfinite(Z)]))
    vmax = float(cfg.vmax) if cfg.vmax is not None else float(np.nanmax(Z[np.isfinite(Z)]))
    if vmax <= vmin:
        vmax = vmin * 1.01 if vmin != 0 else 1.0

    cmap = getattr(cfg, "cmap", "viridis")

    if cfg.log_scale:
        # enforce vmin>0 for log
        if vmin <= 0:
            pos = np.asarray(Z, float)
            pos = pos[np.isfinite(pos) & (pos > 0)]
            vmin = float(np.nanmin(pos)) if pos.size else 1e-12
        vmin = max(vmin, 1e-12)
        vmax = max(vmax, vmin * 1.01)
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    else:
        if bool(getattr(cfg, "center_zero", False)) and (vmin < 0.0 < vmax):
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[float(E.min()), float(E.max()), float(G.min()), float(G.max())],
        cmap=cmap,
        norm=norm,
    )

    # ---- fonts: title/labels 16, ticks 14 ----
    ax.set_xlabel(cfg.xlabel, fontsize=16)
    ax.set_ylabel(cfg.ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

    if cfg.xlim is not None:
        ax.set_xlim(cfg.xlim)
    if cfg.ylim is not None:
        ax.set_ylim(cfg.ylim)

    # ---- colorbar: shorter + lower, top-right, horizontal ----
    cbar_w = 0.24 * axpos.width   # shorter to avoid title collision
    cbar_h = 0.018               # thinner
    cbar_x = axpos.x1 - cbar_w
    cbar_y = axpos.y1 + 0.004    # closer to axes top (not too high)

    # Title should only use space left of the colorbar
    title_left = axpos.x0
    title_right = cbar_x - 0.01

    # ---- wrapped title in header band (no overlap) ----
    title_wrapped = _wrap_title_to_fig_span(
        fig, getattr(cfg, "title", ""),
        title_left, title_right,
        fontsize=16, max_lines=3
    )

    # Use tighter line spacing so 3 lines fit in header band
    fig.text(
        title_left, 0.95,
        title_wrapped,
        ha="left", va="top",
        fontsize=16, fontweight="bold",
        linespacing=1.0,
    )

    # ---- colorbar ----
    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")

    # exactly 3 ticks: vmin, mid, vmax

    # ---- ticks + formatting (make LOG look like LINEAR labels) ----
    tick_fmt = getattr(cfg, "cbar_tick_format", None)
    tick_int = bool(getattr(cfg, "cbar_integer", False))

    def _fmt_sci(x, _pos=None):
        if not np.isfinite(x):
            return ""
        x = float(x)
        if x == 0.0:
            return "0"
        s = f"{x:.0e}"
        mant, exp = s.split("e")
        exp_i = int(exp)
        return f"{mant}e{exp_i}"  # e.g. 3e2, 1e3

    def _fmt_compact(x, _pos=None):
        if not np.isfinite(x):
            return ""
        axx = abs(float(x))
        if axx < 1e-12:
            return "0"
        return f"{x:.4g}" if 1e-2 <= axx < 1e4 else f"{x:.0e}".replace("+0", "+").replace("-0", "-")

    use_sci_always = (
        bool(tick_int)
        or (isinstance(tick_fmt, str) and tick_fmt.lower() in ("sci", "scientific"))
    )

    # 3-tick policy for BOTH modes (log uses geometric mid)
    if cfg.log_scale:
        mid = float((vmin * vmax) ** 0.5)
        cb.locator = mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0))
        cb.update_ticks()

    else:
        if bool(getattr(cfg, "center_zero", False)) and (vmin < 0.0 < vmax) and not cfg.log_scale:
            mid = 0.0
        else:
            mid = float(0.5 * (vmin + vmax))

    ticks = [vmin, mid, vmax]
    cb.set_ticks(ticks)

    formatter = FuncFormatter(_fmt_sci if use_sci_always else _fmt_compact)
    cb.ax.xaxis.set_major_formatter(formatter)
    cb.formatter = formatter
    cb.update_ticks()

    # Safety: remove any offset/math text
    cb.ax.xaxis.get_offset_text().set_visible(False)

    cb.ax.xaxis.set_minor_locator(NullLocator())
    cb.ax.xaxis.set_minor_formatter(NullFormatter())


    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(top=True, 
                    labeltop=True, 
                    bottom=False, 
                    labelbottom=False, 
                    pad=1, 
                    labelsize=14)

    # cbar label text size = 16
    cb.ax.set_title(getattr(cfg, "cbar_label", ""), 
                    fontsize=16, 
                    fontweight="bold", 
                    loc="right", 
                    pad=0)

    return fig


def build_spectrum_fig(x, y, *, title: str = "", xlabel: str = "Energy", ylabel: str = ""):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()

    fig = plt.figure(figsize=(8.8, 2.6), dpi=150, facecolor="white")
    ax = fig.add_axes([0.10, 0.18, 0.86, 0.72])

    ax.plot(x, y, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    return fig


def save_fig_png(fig, out_path: Union[str, Path], *, dpi: Optional[int] = None) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Crop outer whitespace without changing internal axes geometry
    fig.savefig(
        out_path,
        dpi=fig.dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.005,
    )
    return str(out_path)

