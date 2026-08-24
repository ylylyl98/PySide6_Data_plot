from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
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
            ["PL", "DRR", "Compare", "Power", "MCD", "SHG", "Slides", "Tools"],
        )
        expected_modes = [
            "PL",
            "DRR",
            "Compare",
            "Power Dependent",
            "MCD",
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

    @staticmethod
    def _write_drr_measurement(path: Path) -> None:
        path.write_text(
            "Vbg,Vtg,740,760,780\n0,0,1,2,3\n1,0,2,3,4\n",
            encoding="utf-8",
        )

    def test_drr_refresh_adds_new_file_to_a_fully_selected_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            initial = Path(tmp) / "Initial Data"
            initial.mkdir()
            first = initial / "sample_760nmc_rep1_1.csv"
            second = initial / "sample_760nmc_rep1_2.csv"
            self._write_drr_measurement(first)

            self.window._set_current_folder(tmp, remember=False)
            self.assertEqual(self.window.drr_selected_files, [])
            self.window.drr_selected_files = [
                "Initial Data/sample_760nmc_rep1_1.csv"
            ]

            self._write_drr_measurement(second)
            self.window._refresh_file_lists(auto=True)

            self.assertEqual(
                set(self.window.drr_selected_files),
                {
                    "Initial Data/sample_760nmc_rep1_1.csv",
                    "Initial Data/sample_760nmc_rep1_2.csv",
                },
            )

    def test_drr_refresh_preserves_a_deliberately_selected_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            initial = Path(tmp) / "Initial Data"
            initial.mkdir()
            first = initial / "sample_760nmc_rep1_1.csv"
            second = initial / "sample_760nmc_rep1_2.csv"
            third = initial / "sample_760nmc_rep1_3.csv"
            self._write_drr_measurement(first)
            self._write_drr_measurement(second)

            self.window._set_current_folder(tmp, remember=False)
            chosen = "Initial Data/sample_760nmc_rep1_1.csv"
            self.window.drr_selected_files = [chosen]
            self._write_drr_measurement(third)

            self.window._refresh_file_lists(auto=True)

            self.assertEqual(self.window.drr_selected_files, [chosen])

    def test_drr_toolbar_load_refreshes_catalog_before_loading(self) -> None:
        with (
            patch.object(self.window, "_active_mode", return_value="DRR"),
            patch.object(self.window, "_refresh_file_lists") as refresh,
            patch.object(self.window, "_start_load") as load,
        ):
            self.window._toolbar_load()

        refresh.assert_called_once_with(auto=True)
        load.assert_called_once_with("DRR")

    def test_second_save_is_ignored_while_an_export_is_running(self) -> None:
        self.window._export_in_progress = True
        with (
            patch.object(self.window, "_status") as status,
            patch.object(self.window.thread_pool, "start") as start,
        ):
            self.window._start_export("DRR")

        status.assert_called_once_with("Save already in progress.")
        start.assert_not_called()

    def test_switching_experiment_folders_clears_all_stale_workflow_state(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            for folder in (Path(first_tmp), Path(second_tmp)):
                (folder / "same_name.csv").write_text(
                    "Vbg,Vtg,740,760,780\n0,0,1,2,3\n1,0,2,3,4\n",
                    encoding="utf-8",
                )
            self.window._set_current_folder(first_tmp, remember=False)
            self.window._restore_list_selection(self.window.pl_files, ["same_name.csv"])
            self.window._restore_list_selection(self.window.shg_files, ["same_name.csv"])
            self.window.drr_selected_files = ["same_name.csv"]
            self.window.loaded = LoadedState(mode="PL", folder=first_tmp, cube=None)
            self.window.last_plotted_mode = "PL"

            self.window._set_current_folder(second_tmp, remember=False)

            self.assertEqual(self.window._selected(self.window.pl_files), [])
            self.assertEqual(self.window._selected(self.window.mcd_files), [])
            self.assertEqual(self.window._selected(self.window.shg_files), [])
            self.assertEqual(self.window.drr_selected_files, [])
            self.assertIsNone(self.window.loaded)
            self.assertIsNone(self.window.last_plotted_mode)

    def test_power_and_shg_sources_remain_unselected_after_folder_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("sample_1uW.csv", "sample_2uW.csv"):
                (folder / name).write_text(
                    "Vbg,Vtg,740,760,780\n0,0,1,2,3\n1,0,2,3,4\n",
                    encoding="utf-8",
                )

            self.window._set_current_folder(tmp, remember=False)

            self.assertEqual(self.window._power_selected_group_key(), "")
            self.assertEqual(self.window._shg_selected_file(), "")
            self.assertEqual(self.window._shg_compare_files(), ("", ""))

    def test_compare_auto_assignment_leaves_duplicate_channel_unassigned(self) -> None:
        with (
            patch.object(self.window, "_cmp_assign_candidate_files", return_value=[]),
            patch(
                "ui_qt.main_window.coherent_compare_auto_assignment",
                return_value=(
                    {"KK": "ambiguous.csv"},
                    {"KK": ["first.csv", "second.csv"]},
                    None,
                    [],
                ),
            ),
            patch.object(self.window, "_on_cmp_plot_param_changed"),
        ):
            self.window._cmp_auto_assign_channels()

        self.assertEqual(self.window.cmp_channel_combos["KK"].currentText(), "")

    def test_unchanged_repeat_save_is_suppressed_for_other_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.csv"
            self._write_drr_measurement(source)
            cube = DataCube(
                energy=np.asarray([1.0, 2.0]),
                gate=np.asarray([0.0, 1.0]),
                Z=np.asarray([[1.0, 2.0], [2.0, 3.0]]),
                gate_label="Gate",
                title="PL",
                cbar_label="Intensity",
            )
            self.window.current_folder = tmp
            self.window.loaded = LoadedState(
                mode="PL",
                folder=tmp,
                primary_file=source.name,
                selected_files=[source.name],
                cube=cube,
            )
            self.window.pl_split_scale_chk.setChecked(False)
            self.window.last_plotted_mode = "PL"
            with patch.object(self.window.thread_pool, "start") as first_start:
                self.window._start_export("PL")
            first_start.assert_called_once()
            self.window._on_export_done(
                {
                    "out_folder": str(Path(tmp) / "Processed Data" / "PL"),
                    "folder": tmp,
                    "mode": "PL",
                }
            )

            with (
                patch.object(self.window.thread_pool, "start") as second_start,
                patch.object(self.window, "_status") as status,
            ):
                self.window._start_export("PL")

            second_start.assert_not_called()
            status.assert_called_with(
                "This unchanged result is already saved; no duplicate created."
            )

    def test_compact_sidebar_avoids_default_scrolling_and_horizontal_clipping(self) -> None:
        self.window.tabs.setCurrentWidget(self.window.drr_tab_scroll)
        self.window.resize(1180, 700)
        self.app.processEvents()

        scroll = self.window.drr_tab_scroll
        self.assertIsInstance(scroll, QScrollArea)
        self.assertEqual(scroll.verticalScrollBar().maximum(), 0)
        self.assertEqual(
            scroll.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff
        )

    def test_workflow_navigation_is_above_the_collapsible_sidebar(self) -> None:
        self.assertFalse(self.window.tabs.tabBar().isVisible())
        self.assertEqual(self.window.workflow_tabs.count(), self.window.tabs.count())
        self.window.workflow_tabs.setCurrentIndex(1)
        self.assertEqual(self.window._active_mode(), "DRR")

        self.window.sidebar_toggle_btn.setChecked(False)
        self.app.processEvents()
        self.assertFalse(self.window.left_panel.isVisible())
        self.window.sidebar_toggle_btn.setChecked(True)
        self.app.processEvents()
        self.assertTrue(self.window.left_panel.isVisible())

    def test_slides_workspace_is_full_width_and_owns_its_build_controls(self) -> None:
        slides_index = next(
            index
            for index in range(self.window.tabs.count())
            if self.window.tabs.tabText(index) == "Slides"
        )
        self.window.workflow_tabs.setCurrentIndex(slides_index)
        QApplication.processEvents()
        self.assertIs(self.window.workspace_stack.currentWidget(), self.window.presentation_widget)
        self.assertFalse(self.window.sidebar_toggle_btn.isEnabled())
        self.assertFalse(self.window.load_action.isEnabled())
        self.assertFalse(self.window.plot_action.isEnabled())
        self.assertFalse(self.window.save_action.isEnabled())
        counts = [
            self.window.presentation_widget.images_per_slide_combo.itemData(index)
            for index in range(self.window.presentation_widget.images_per_slide_combo.count())
        ]
        self.assertEqual(counts, list(range(1, 13)))
        self.assertIn("never alter the PNG", self.window.presentation_widget.caption_combo.toolTip())

    def test_enabling_drr_split_centers_boundary_and_scales_both_regions(self) -> None:
        cube = DataCube(
            energy=np.array([0.0, 1.0, 2.0, 3.0]),
            gate=np.array([0.0, 1.0]),
            Z=np.array([[1.0, 2.0, 100.0, 200.0], [3.0, 4.0, 300.0, 400.0]]),
            gate_label="Gate",
            title="DRR split",
            cbar_label="DR/R",
        )
        self.window.loaded = LoadedState(mode="DRR", folder="", cube=cube)
        for key, value in (("xmin", 0.0), ("xmax", 3.0), ("ymin", 0.0), ("ymax", 1.0)):
            self.window._set_spin_value_silent(self.window.drr_spins[key], value)
        self.window.drr_split_spins["x0"].setValue(0.25)

        with patch.object(self.window, "_on_drr_plot_param_changed"):
            self.window.drr_split_scale_chk.setChecked(True)

        self.assertAlmostEqual(self.window.drr_split_spins["x0"].value(), 1.5)
        self.assertLess(self.window.drr_split_spins["left_vmax"].value(), 5.0)
        self.assertGreater(self.window.drr_split_spins["right_vmin"].value(), 90.0)

    def test_disabling_drr_split_restores_single_scale_and_replots(self) -> None:
        cube = DataCube(
            energy=np.array([0.0, 1.0, 2.0, 3.0]),
            gate=np.array([0.0, 1.0]),
            Z=np.array([[1.0, 2.0, 100.0, 200.0], [3.0, 4.0, 300.0, 400.0]]),
            gate_label="Gate",
            title="DRR split",
            cbar_label="DR/R",
        )
        self.window.loaded = LoadedState(mode="DRR", folder="", cube=cube)
        for key, value in (("xmin", 0.0), ("xmax", 3.0), ("ymin", 0.0), ("ymax", 1.0)):
            self.window._set_spin_value_silent(self.window.drr_spins[key], value)
        with patch.object(self.window, "_on_drr_plot_param_changed"):
            self.window.drr_split_scale_chk.setChecked(True)
        self.window._set_spin_value_silent(self.window.drr_spins["vmin"], -999.0)
        self.window._set_spin_value_silent(self.window.drr_spins["vmax"], 999.0)

        with patch.object(self.window, "_on_drr_plot_param_changed") as replot:
            self.window.drr_split_scale_chk.setChecked(False)

        replot.assert_called_once()
        self.assertGreater(self.window.drr_spins["vmin"].value(), -999.0)
        self.assertLess(self.window.drr_spins["vmax"].value(), 999.0)

    def test_drr_derivative_refreshes_unfixed_color_limits(self) -> None:
        energy = np.linspace(0.0, 4.0, 21)
        cube = DataCube(
            energy=energy,
            gate=np.array([0.0, 1.0]),
            Z=np.vstack((energy**2, 2.0 * energy**2)),
            gate_label="Gate",
            title="DRR derivative",
            cbar_label="DR/R",
        )
        self.window.loaded = LoadedState(mode="DRR", folder="", cube=cube)
        for key, value in (("xmin", 0.0), ("xmax", 4.0), ("ymin", 0.0), ("ymax", 1.0)):
            self.window._set_spin_value_silent(self.window.drr_spins[key], value)
        self.window._refresh_automatic_ranges("DRR")
        original_vmax = self.window.drr_spins["vmax"].value()

        with patch.object(self.window, "_plot_mode"):
            self.window.drr_derivative_combo.setCurrentText("dE")

        self.assertNotAlmostEqual(self.window.drr_spins["vmax"].value(), original_vmax)

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

    def test_measurement_change_clears_unpinned_external_background(self) -> None:
        self.window.drr_selected_files = ["old_760nmc_rep1.csv"]
        self.window.drr_baseline_files_manual = ["old_760nmc_back.csv"]
        blocked = self.window.drr_baseline_combo.blockSignals(True)
        self.window.drr_baseline_combo.setCurrentText("External")
        self.window.drr_baseline_combo.blockSignals(blocked)
        self.window.drr_pin_baseline_chk.setChecked(False)
        self.window.loaded = LoadedState(mode="DRR", folder="", cube=None)

        with patch.object(
            self.window,
            "_open_drr_source_dialog",
            return_value=["new_760nmc_rep1.csv"],
        ), patch.object(self.window, "_start_load") as start_load:
            self.window._edit_drr_measurements()

        self.assertEqual(self.window.drr_baseline_files_manual, [])
        self.assertIsNone(self.window.loaded)
        start_load.assert_not_called()

    def test_processed_measurement_restores_its_saved_background_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            processed = root / "Processed Data" / "DRR"
            initial.mkdir()
            processed.mkdir(parents=True)
            measurement = initial / "sample_760nmc_rep1.csv"
            background = initial / "sample_760nmc_back.csv"
            self._write_drr_measurement(measurement)
            background.write_text(
                "Vbg,Vtg,740,760,780\n0,0,1,2,3\n0,0,2,3,4\n",
                encoding="utf-8",
            )
            (processed / "sample.metadata.json").write_text(
                json.dumps(
                    {
                        "operation": "DR/R",
                        "sources": [
                            {"role": "measurement", "source_path": str(measurement)},
                            {"role": "background", "source_path": str(background)},
                        ],
                        "processing": {
                            "baseline_selection": "External",
                            "baseline_which": "all",
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.window._set_current_folder(tmp, remember=False)
            selected = "Initial Data/sample_760nmc_rep1.csv"

            with patch.object(
                self.window,
                "_open_drr_source_dialog",
                return_value=[selected],
            ), patch.object(self.window, "_start_load") as start_load:
                self.window._edit_drr_measurements()

            self.assertEqual(self.window.drr_selected_files, [selected])
            self.assertEqual(
                self.window.drr_baseline_files_manual,
                ["Initial Data/sample_760nmc_back.csv"],
            )
            self.assertEqual(self.window.drr_baseline_combo.currentText(), "External")
            self.assertEqual(
                self.window.drr_baseline_combine_combo.currentText(),
                "Average all frames in each file, then average files",
            )
            start_load.assert_called_once_with("DRR")

    def test_constant_gate_background_selection_defaults_to_all_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "back_1.csv"
            second = root / "back_2.csv"
            for path in (first, second):
                path.write_text(
                    "Vbg,Vtg,740,760,780\n0,0,1,2,3\n0,0,2,3,4\n",
                    encoding="utf-8",
                )
            self.window._set_current_folder(tmp, remember=False)
            self.window.drr_baseline_files_manual = [first.name, second.name]
            self.window.drr_baseline_combine_combo.setCurrentText(
                "Last frame from each file, then average"
            )

            accepted = self.window._apply_drr_background_gate_default()

            self.assertTrue(accepted)
            self.assertEqual(
                self.window.drr_baseline_combine_combo.currentText(),
                "Average all frames in each file, then average files",
            )

    def test_measurement_change_retains_explicitly_pinned_background(self) -> None:
        self.window.drr_selected_files = ["old_760nmc_rep1.csv"]
        self.window.drr_baseline_files_manual = ["old_760nmc_back.csv"]
        blocked = self.window.drr_baseline_combo.blockSignals(True)
        self.window.drr_baseline_combo.setCurrentText("External")
        self.window.drr_baseline_combo.blockSignals(blocked)
        self.window.drr_pin_baseline_chk.setChecked(True)

        with patch.object(
            self.window,
            "_open_drr_source_dialog",
            return_value=["new_760nmc_rep1.csv"],
        ), patch.object(self.window, "_start_load") as start_load:
            self.window._edit_drr_measurements()

        self.assertEqual(
            self.window.drr_baseline_files_manual, ["old_760nmc_back.csv"]
        )
        start_load.assert_called_once_with("DRR")


if __name__ == "__main__":
    unittest.main()
