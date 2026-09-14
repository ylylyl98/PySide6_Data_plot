from __future__ import annotations

import io
import math
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from core import export
from core.loader import DataCube
from core.plotting import HeatmapParams


def _legacy_wrap_title_to_fig_span(
    fig,
    text: str,
    left_x: float,
    right_x: float,
    *,
    fontsize: int = 16,
    weight: str = "bold",
    max_lines: int = 3,
) -> str:
    """Reference implementation used to compare rendered output."""
    if not text:
        return ""
    text = export._prettify_title(text)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    wpx, _ = fig.canvas.get_width_height()
    max_w_px = max(40, (right_x - left_x) * wpx)
    fp = export.FontProperties(size=fontsize, weight=weight)
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = (cur + " " + word).strip() if cur else word
        tw, _, _ = renderer.get_text_width_height_descent(trial, fp, ismath=False)
        if tw <= max_w_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return "\n".join(lines)


def _params(title: str) -> HeatmapParams:
    return HeatmapParams(
        title=title,
        xlabel="Energy (eV)",
        ylabel="Gate (V)",
        cbar_label="Signal",
        vmin=-2.0,
        vmax=3.0,
        xlim=(1.0, 4.0),
        ylim=(-1.0, 2.0),
    )


def _cube(title: str) -> DataCube:
    return DataCube(
        energy=np.array([1.0, 2.0, 3.0, 4.0]),
        gate=np.array([-1.0, 0.5, 2.0]),
        Z=np.array(
            [[-2.0, -1.0, 0.0, 1.0], [0.5, 1.5, 2.5, 3.0], [3.0, 2.0, 1.0, -0.5]],
        ),
        gate_label="Gate (V)",
        title=title,
        cbar_label="Signal",
    )


def _render_png(cube: DataCube, params: HeatmapParams, figsize: tuple[float, float]) -> bytes:
    with warnings.catch_warnings(), patch.object(export, "EXPORT_FIGSIZE", figsize):
        warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")
        fig = export._build_streamlit_style_heatmap_fig(cube, params, drr=False)
    try:
        output = io.BytesIO()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")
            fig.savefig(
                output,
                format="png",
                dpi=fig.dpi,
                facecolor=fig.get_facecolor(),
                edgecolor="none",
                bbox_inches="tight",
                pad_inches=0.005,
            )
        return output.getvalue()
    finally:
        plt.close(fig)


class DatPngSaveOptimizationTests(unittest.TestCase):
    def test_save_heatmap_dat_preserves_special_float_bytes_without_numpy_scalar_ufunc(self) -> None:
        energy = np.array([-0.0, 1.234567890123, np.finfo(float).max, np.nan])
        gate = np.array([np.inf, -np.inf, -0.0])
        z = np.array(
            [
                [np.nan, np.inf, -np.inf, -0.0],
                [1.234567890123, -np.finfo(float).max, 0.0, np.nan],
                [np.inf, -np.inf, np.finfo(float).tiny, -0.0],
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "special.dat"
            with patch.object(export.np, "isfinite", side_effect=AssertionError("scalar np.isfinite call")):
                export.save_heatmap_dat_streamlit(path, energy, gate, z, header_lines=["unicode=中文"])

            expected_rows = [
                [
                    "Photon energy",
                    *[
                        "nan" if not math.isfinite(float(value)) else f"{float(value):.10g}"
                        for value in gate
                    ],
                ],
                *[
                    [
                        "nan" if not math.isfinite(float(energy_value)) else f"{float(energy_value):.10g}",
                        *[
                            "nan" if not math.isfinite(float(value)) else f"{float(value):.10g}"
                            for value in row
                        ],
                    ]
                    for energy_value, row in zip(energy, z.T)
                ],
            ]
            expected = "# unicode=中文\n" + "\n".join("\t".join(row) for row in expected_rows) + "\n"
            expected_bytes = expected.replace("\n", "\r\n" if os.name == "nt" else "\n").encode("utf-8")
            self.assertEqual(path.read_bytes(), expected_bytes)

    def test_title_measurement_uses_renderer_without_full_canvas_draw(self) -> None:
        fig = Figure(figsize=(8.0, 6.2), dpi=150)
        canvas = FigureCanvasAgg(fig)
        draw_calls = 0
        original_draw = canvas.draw

        def counted_draw(*args, **kwargs):
            nonlocal draw_calls
            draw_calls += 1
            return original_draw(*args, **kwargs)

        with warnings.catch_warnings(), patch.object(canvas, "draw", counted_draw):
            warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")
            wrapped = export._wrap_title_to_fig_span(
                fig,
                "A long title with 中文 unicode and enough words to wrap",
                0.1,
                0.8,
                fontsize=16,
                weight="bold",
            )
            normal = export._wrap_title_to_fig_span(
                fig,
                "A normal weight title with several words",
                0.1,
                0.8,
                fontsize=11,
                weight="normal",
            )

        self.assertTrue(wrapped)
        self.assertTrue(normal)
        self.assertEqual(draw_calls, 0)
        plt.close(fig)

    def test_title_measurement_falls_back_without_renderer(self) -> None:
        fig = SimpleNamespace(canvas=SimpleNamespace())
        wrapped = export._wrap_title_to_fig_span(
            fig,
            "A title that should use the safe fallback",
            0.1,
            0.8,
            fontsize=11,
            weight="normal",
            max_lines=2,
        )
        self.assertEqual(wrapped, "A title that should use the safe\nfallback")

    def test_title_optimization_keeps_real_png_bytes_for_titles_and_sizes(self) -> None:
        titles = (
            "Short title",
            "A long title with 中文 unicode and enough words to wrap across the available span",
        )
        sizes = ((4.0, 3.0), (8.0, 6.2), (12.0, 8.0))
        for title in titles:
            cube = _cube(title)
            params = _params(title)
            for figsize in sizes:
                with patch.object(export, "_wrap_title_to_fig_span", _legacy_wrap_title_to_fig_span):
                    legacy = _render_png(cube, params, figsize)
                optimized = _render_png(cube, params, figsize)
                self.assertEqual(optimized, legacy, msg=f"title={title!r}, figsize={figsize!r}")


if __name__ == "__main__":
    unittest.main()
