from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from matplotlib.figure import Figure
from matplotlib.pyplot import close

from core.loader import DataCube
from core.plotting import HeatmapParams, plot_drr, plot_pl, render_compare_grid


EXPORT_WIDTH_PX = 1600
EXPORT_HEIGHT_PX = 1200
EXPORT_DPI = 200


def _new_export_figure(
    width_px: int = EXPORT_WIDTH_PX,
    height_px: int = EXPORT_HEIGHT_PX,
    dpi: int = EXPORT_DPI,
) -> Figure:
    return Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)


def export_fixed_figure(path: str | Path, draw_fn: Callable[[Figure], None], *, dpi: int = EXPORT_DPI) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = _new_export_figure(dpi=dpi)
    try:
        draw_fn(fig)
        fig.tight_layout()
        fig.savefig(out, dpi=dpi)
    finally:
        close(fig)
    return out


def export_fixed_plot(
    path: str | Path,
    *,
    mode: str,
    params: HeatmapParams,
    cube: DataCube | None = None,
    compare_cubes: Dict[str, DataCube] | None = None,
    dpi: int = EXPORT_DPI,
) -> Path:
    mode_u = mode.upper()

    def _draw(fig: Figure):
        if mode_u == "PL":
            if cube is None:
                raise ValueError("PL export requires a loaded cube.")
            ax = fig.subplots(1, 1)
            im = plot_pl(ax, cube, params)
            fig.colorbar(im, ax=ax, shrink=0.9, label=params.cbar_label)
        elif mode_u == "DRR":
            if cube is None:
                raise ValueError("DRR export requires a loaded cube.")
            ax = fig.subplots(1, 1)
            im = plot_drr(ax, cube, params)
            fig.colorbar(im, ax=ax, shrink=0.9, label=params.cbar_label)
        elif mode_u == "COMPARE":
            if not compare_cubes:
                raise ValueError("Compare export requires compare cubes.")
            images = render_compare_grid(fig, compare_cubes, params)
            if images:
                fig.colorbar(images[0], ax=fig.axes, shrink=0.8, label=params.cbar_label)
        else:
            raise ValueError(f"Unsupported export mode: {mode}")

    return export_fixed_figure(path, _draw, dpi=dpi)
