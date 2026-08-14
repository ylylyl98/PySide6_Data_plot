from __future__ import annotations

import hashlib
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from core.colormaps import (
    CUSTOM_COLORMAPS,
    RDBU_R_P0P45_ID,
    RDBU_R_P0P45_LABEL,
    RDBU_R_P0P45_RGB,
    RDBU_R_P0P60_ID,
    RDBU_R_P0P60_LABEL,
    RDBU_R_P0P60_RGB,
    STANDARD_COLORMAPS,
    register_colormaps,
    resolve_cmap,
)
from core.export import _build_streamlit_style_heatmap_fig
from core.loader import DataCube
from core.plotting import HeatmapParams, SplitColorScale, plot_heatmap
from ui_qt.main_window import MainWindow


EXPECTED_LABELS = [
    "Default",
    RDBU_R_P0P45_LABEL,
    RDBU_R_P0P60_LABEL,
    *STANDARD_COLORMAPS,
]


class ColormapTableTests(unittest.TestCase):
    def test_exact_palette_tables(self) -> None:
        expected = (
            (
                RDBU_R_P0P45_RGB,
                "466fb0e65cae9ea65d22c781efd67d57f6622c828fd443bdecfea2b2e06088c9",
                (5, 48, 97),
                (167, 208, 228),
                (247, 183, 153),
                (103, 0, 31),
            ),
            (
                RDBU_R_P0P60_RGB,
                "4de0f1eef686229cbb37147a92bfb2dbb596a91d162d690f1e66017974738609",
                (5, 48, 97),
                (240, 244, 246),
                (248, 242, 239),
                (103, 0, 31),
            ),
        )
        for colors, digest, first, middle_left, middle_right, last in expected:
            self.assertEqual(len(colors), 256)
            self.assertTrue(
                all(isinstance(channel, int) and 0 <= channel <= 255 for rgb in colors for channel in rgb)
            )
            packed = bytes(channel for rgb in colors for channel in rgb)
            self.assertEqual(hashlib.sha256(packed).hexdigest(), digest)
            self.assertEqual(colors[0], first)
            self.assertEqual(colors[127], middle_left)
            self.assertEqual(colors[128], middle_right)
            self.assertEqual(colors[-1], last)

    def test_registry_ids_labels_and_resolution(self) -> None:
        self.assertEqual(
            CUSTOM_COLORMAPS,
            (
                (RDBU_R_P0P45_ID, RDBU_R_P0P45_LABEL),
                (RDBU_R_P0P60_ID, RDBU_R_P0P60_LABEL),
            ),
        )
        register_colormaps()
        register_colormaps()
        for cmap_id, label, colors in (
            (RDBU_R_P0P45_ID, RDBU_R_P0P45_LABEL, RDBU_R_P0P45_RGB),
            (RDBU_R_P0P60_ID, RDBU_R_P0P60_LABEL, RDBU_R_P0P60_RGB),
        ):
            cmap = matplotlib.colormaps[cmap_id]
            self.assertEqual(cmap.name, cmap_id)
            self.assertEqual(cmap.N, 256)
            for index in (0, 127, 128, 255):
                self.assertTrue(np.allclose(cmap(index)[:3], np.asarray(colors[index]) / 255.0))
            self.assertEqual(resolve_cmap(label, "turbo"), cmap_id)
            self.assertEqual(resolve_cmap(cmap_id, "turbo"), cmap_id)
        self.assertEqual(resolve_cmap("viridis", "turbo"), "viridis")
        self.assertEqual(resolve_cmap("Default", "RdBu_r"), "RdBu_r")


class ColormapUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_all_selectors_have_required_order_and_legacy_defaults(self) -> None:
        settings = QSettings(MainWindow.SETTINGS_ORG, MainWindow.SETTINGS_APP)
        before = {key: settings.value(key) for key in settings.allKeys()}
        window = MainWindow()
        try:
            expected_defaults = {
                "pl_cmap": "turbo",
                "drr_cmap": "RdBu_r",
                "cmp_cmap": "turbo",
                "mcd_cmap": "RdBu_r",
                "power_cmap": "turbo",
            }
            for attribute, expected_default in expected_defaults.items():
                combo = getattr(window, attribute)
                self.assertEqual([combo.itemText(i) for i in range(combo.count())], EXPECTED_LABELS)
                self.assertEqual(combo.currentText(), "Default")
                self.assertEqual(combo.itemData(1), RDBU_R_P0P45_ID)
                self.assertEqual(combo.itemData(2), RDBU_R_P0P60_ID)
                self.assertEqual(window._resolved_cmap(combo), expected_default)
                combo.setCurrentIndex(1)
                self.assertEqual(window._resolved_cmap(combo), RDBU_R_P0P45_ID)
                combo.setCurrentIndex(2)
                self.assertEqual(window._resolved_cmap(combo), RDBU_R_P0P60_ID)
        finally:
            window.close()
        settings.sync()
        self.assertEqual({key: settings.value(key) for key in settings.allKeys()}, before)


class ColormapRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        register_colormaps()
        self.cube = DataCube(
            energy=np.array([1.0, 2.0, 3.0, 4.0]),
            gate=np.array([-1.0, 1.0]),
            Z=np.array([[-2.0, -1.0, 1.0, 2.0], [-3.0, -1.5, 1.5, 3.0]]),
            gate_label="Gate (V)",
            title="Custom cmap",
            cbar_label="Signal",
        )

    def _params(self, cmap: str, *, split: bool = False) -> HeatmapParams:
        return HeatmapParams(
            title="Custom cmap",
            xlabel="Energy (eV)",
            ylabel="Gate (V)",
            cbar_label="Signal",
            vmin=-3.0,
            vmax=3.0,
            xlim=(1.0, 4.0),
            ylim=(-1.0, 1.0),
            cmap=cmap,
            split_scale=(SplitColorScale(2.5, -3.0, 0.0, 0.0, 3.0) if split else None),
        )

    def test_plotting_changes_only_color_lookup(self) -> None:
        original = self.cube.Z.copy()
        for index, cmap_id in enumerate((RDBU_R_P0P45_ID, RDBU_R_P0P60_ID)):
            fig, ax = plt.subplots()
            render = plot_heatmap(ax, self.cube, self._params(cmap_id, split=index == 0))
            self.assertEqual(render.primary.cmap.name, cmap_id)
            self.assertEqual((render.primary.norm.vmin, render.primary.norm.vmax), (-3.0, 0.0) if index == 0 else (-3.0, 3.0))
            if render.secondary is not None:
                self.assertEqual(render.secondary.cmap.name, cmap_id)
                self.assertEqual((render.secondary.norm.vmin, render.secondary.norm.vmax), (0.0, 3.0))
            self.assertTrue(np.array_equal(self.cube.Z, original))
            plt.close(fig)

    def test_export_figure_keeps_custom_colormap(self) -> None:
        for cmap_id in (RDBU_R_P0P45_ID, RDBU_R_P0P60_ID):
            fig = _build_streamlit_style_heatmap_fig(self.cube, self._params(cmap_id), drr=False)
            mappables = [*fig.axes[0].collections, *fig.axes[0].images]
            self.assertTrue(mappables)
            self.assertEqual(mappables[0].cmap.name, cmap_id)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
