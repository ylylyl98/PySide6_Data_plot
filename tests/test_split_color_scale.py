from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from core.loader import DataCube
from core.export import _build_streamlit_style_heatmap_fig
from core.plotting import HeatmapParams, SplitColorScale, plot_heatmap, resolve_split_boundary


class SplitColorScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cube = DataCube(
            energy=np.array([1.0, 2.0, 3.0, 4.0]),
            gate=np.array([0.0, 1.0]),
            Z=np.array([[1.0, 2.0, 100.0, 200.0], [2.0, 4.0, 120.0, 240.0]]),
            gate_label="TG (V)",
            title="Split test",
            cbar_label="Intensity",
        )

    def _params(self, **changes) -> HeatmapParams:
        values = dict(
            title="Split test",
            xlabel="Energy (eV)",
            ylabel="TG (V)",
            cbar_label="Intensity",
            vmin=0.0,
            vmax=250.0,
            xlim=(1.0, 4.0),
            ylim=(0.0, 1.0),
            cmap="viridis",
            split_scale=SplitColorScale(
                split_x=2.6,
                left_vmin=0.0,
                left_vmax=5.0,
                right_vmin=90.0,
                right_vmax=250.0,
            ),
        )
        values.update(changes)
        return HeatmapParams(**values)

    def test_boundary_snaps_to_nearest_cell_edge(self) -> None:
        index, boundary = resolve_split_boundary(self.cube.energy, 2.6)
        self.assertEqual(index, 2)
        self.assertAlmostEqual(boundary, 2.5)

    def test_two_regions_use_independent_norms_and_nonoverlapping_masks(self) -> None:
        fig, ax = plt.subplots()
        render = plot_heatmap(ax, self.cube, self._params())
        self.assertTrue(render.is_split)
        self.assertAlmostEqual(render.split_x, 2.5)
        self.assertEqual((render.primary.norm.vmin, render.primary.norm.vmax), (0.0, 5.0))
        self.assertEqual((render.secondary.norm.vmin, render.secondary.norm.vmax), (90.0, 250.0))

        left_mask = np.ma.getmaskarray(render.primary.get_array()).reshape(self.cube.Z.shape)
        right_mask = np.ma.getmaskarray(render.secondary.get_array()).reshape(self.cube.Z.shape)
        self.assertTrue(np.all(~left_mask[:, :2]))
        self.assertTrue(np.all(left_mask[:, 2:]))
        self.assertTrue(np.all(right_mask[:, :2]))
        self.assertTrue(np.all(~right_mask[:, 2:]))
        self.assertFalse(np.any(~left_mask & ~right_mask))
        self.assertFalse(np.any(left_mask & right_mask))
        plt.close(fig)

    def test_split_position_must_be_inside_visible_x_range(self) -> None:
        params = self._params(
            split_scale=SplitColorScale(4.0, 0.0, 5.0, 90.0, 250.0)
        )
        fig, ax = plt.subplots()
        with self.assertRaisesRegex(ValueError, "strictly between"):
            plot_heatmap(ax, self.cube, params)
        plt.close(fig)

    def test_log_split_requires_positive_region_minima(self) -> None:
        fig, ax = plt.subplots()
        with self.assertRaisesRegex(ValueError, "positive vmin"):
            plot_heatmap(ax, self.cube, self._params(log_scale=True))
        plt.close(fig)

    def test_export_contains_two_labeled_colorbars(self) -> None:
        fig = _build_streamlit_style_heatmap_fig(self.cube, self._params(), drr=False)
        self.assertEqual(len(fig.axes), 3)
        titles = [axis.get_title() for axis in fig.axes[1:]]
        self.assertTrue(any("x ≤ 2.5" in title for title in titles))
        self.assertTrue(any("x ≥ 2.5" in title for title in titles))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
