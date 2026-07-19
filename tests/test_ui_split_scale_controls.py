from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QScrollArea, QSplitter, QWidget

from core.loader import DataCube
from ui_qt.main_window import LoadedState, MainWindow, UI_METRICS


class SplitScaleControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            self.window = MainWindow()
        self.window.resize(1180, 820)
        self.window.show()
        self.window.pl_split_scale_chk.setChecked(True)
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.close()

    def test_split_controls_fit_at_minimum_sidebar_width(self) -> None:
        splitter = self.window.findChild(QSplitter)
        self.assertIsNotNone(splitter)
        left_panel = splitter.widget(0)
        self.assertEqual(left_panel.width(), UI_METRICS["left_width"])
        splitter.setSizes([400, 1200])
        self.app.processEvents()
        self.assertEqual(left_panel.width(), UI_METRICS["left_width"])

        panel = self.window.pl_split_scale_panel
        visible_children = panel.findChildren(
            QWidget, options=Qt.FindDirectChildrenOnly
        )
        self.assertTrue(visible_children)
        right_edge = max(child.geometry().right() for child in visible_children)
        self.assertLessEqual(right_edge, panel.contentsRect().right())

    def test_fixed_sidebar_uses_complete_compact_tab_labels(self) -> None:
        tab_bar = self.window.tabs.tabBar()
        self.assertFalse(tab_bar.usesScrollButtons())
        self.assertEqual(tab_bar.elideMode(), Qt.ElideNone)
        self.assertEqual(
            [self.window.tabs.tabText(i) for i in range(self.window.tabs.count())],
            ["PL", "DRR", "Compare", "Power", "SHG", "Tools"],
        )
        expected_modes = [
            "PL",
            "DRR",
            "Compare",
            "Power Dependent",
            "SHG Processing",
            None,
        ]
        for index, expected in enumerate(expected_modes):
            self.window.tabs.setCurrentIndex(index)
            self.assertEqual(self.window._active_mode(), expected)

    def test_drr_summary_wraps_the_complete_first_filename(self) -> None:
        full_name = (
            "YZ364_0Tpa_3.6KREF_700nmc_0p1sx20_"
            "TG-1.05BG=0_extra_long_measurement_name.csv"
        )
        self.window.drr_selected_files = [full_name, "second.csv"]
        self.window._update_drr_selection_labels()

        displayed = self.window.drr_measurement_summary.text().replace("\u200b", "")
        self.assertIn(full_name, displayed)
        self.assertTrue(self.window.drr_measurement_summary.wordWrap())
        self.assertIn(full_name, self.window.drr_measurement_summary.toolTip())

    def test_tall_tab_pages_scroll_without_horizontal_clipping(self) -> None:
        self.window.tabs.setCurrentWidget(self.window.drr_tab_scroll)
        self.window.resize(1180, 700)
        self.app.processEvents()

        scroll = self.window.drr_tab_scroll
        self.assertIsInstance(scroll, QScrollArea)
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
        self.assertEqual(
            scroll.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff
        )

        scroll.ensureWidgetVisible(self.window.drr_analysis_text)
        self.app.processEvents()
        self.assertGreater(scroll.verticalScrollBar().value(), 0)

    def test_auto_region_preserves_each_fixed_bound(self) -> None:
        cube = DataCube(
            energy=np.array([0.0, 1.0, 2.0, 3.0]),
            gate=np.array([0.0, 1.0]),
            Z=np.array([[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]]),
            gate_label="Gate",
            title="Split controls",
            cbar_label="Intensity",
        )
        self.window.pl_spins["xmin"].setValue(0.0)
        self.window.pl_spins["xmax"].setValue(3.0)
        self.window.pl_spins["ymin"].setValue(0.0)
        self.window.pl_spins["ymax"].setValue(1.0)
        self.window.pl_split_spins["x0"].setValue(1.5)
        self.window.pl_split_spins["left_vmin"].setValue(-7.0)
        self.window.pl_split_spins["left_vmax"].setValue(100.0)
        self.window.pl_split_fix_checks["left_vmin"].setChecked(True)
        self.window.loaded = LoadedState(mode="PL", folder="", cube=cube)

        with patch.object(self.window, "_on_split_scale_param_changed"):
            self.window._auto_split_vrange("pl", "left")

        self.assertEqual(self.window.pl_split_spins["left_vmin"].value(), -7.0)
        self.assertLess(self.window.pl_split_spins["left_vmax"].value(), 5.0)

        self.window.pl_split_fix_checks["left_vmax"].setChecked(True)
        self.app.processEvents()
        self.assertFalse(self.window.pl_split_auto_left_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
