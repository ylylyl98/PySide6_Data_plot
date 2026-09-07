from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QListWidget, QScrollArea, QSplitter, QStyle, QStyleOptionSpinBox, QToolButton, QWidget

from core.loader import DataCube
from core.drr_sources import DrrSource
from ui_qt.main_window import LoadedState, MainWindow, UI_METRICS
from ui_qt.theme import install_theme
from tests.ui_test_helpers import wait_for_file_catalog


class SplitScaleControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        # Match packaged startup: Fluent light theme is installed before any
        # real MainWindow is constructed so QSS-driven size hints are active.
        install_theme(cls.app, mode="light")

    def setUp(self) -> None:
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            self.window = MainWindow()
        self.window.resize(1180, 820)
        self.window.show()
        self.window.pl_split_scale_chk.setChecked(True)
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def _wait_for_drr_catalog(self) -> None:
        self._wait_for_file_catalog()
        # File discovery is intentionally asynchronous and GitHub's Windows
        # runners can be busy while the full Qt suite is running.
        for _ in range(500):
            self.app.processEvents()
            if not self.window._drr_refresh_running:
                return
            QTest.qWait(10)
        self.fail("Timed out waiting for the DRR catalog refresh")

    def _wait_for_file_catalog(self) -> None:
        wait_for_file_catalog(self.window)

    def test_split_controls_fit_at_minimum_sidebar_width(self) -> None:
        splitter = self.window.findChild(QSplitter)
        self.assertIsNotNone(splitter)
        left_panel = splitter.widget(0)
        self.assertEqual(left_panel.width(), UI_METRICS["left_width"])
        self.assertGreaterEqual(left_panel.minimumWidth(), 300)
        self.assertGreater(left_panel.maximumWidth(), left_panel.minimumWidth())
        splitter.setSizes([UI_METRICS["sidebar_min_width"] - 80, 1200])
        self.app.processEvents()
        self.assertGreaterEqual(left_panel.width(), UI_METRICS["sidebar_min_width"])

        panel = self.window.pl_split_scale_panel
        visible_children = panel.findChildren(
            QWidget, options=Qt.FindDirectChildrenOnly
        )
        self.assertTrue(visible_children)
        right_edge = max(child.geometry().right() for child in visible_children)
        self.assertLessEqual(right_edge, panel.contentsRect().right())

    def test_axis_range_rows_keep_fix_and_auto_controls_contained(self) -> None:
        """Real PL/DRR/Compare/Power rows stay contained on the production platform."""
        if QApplication.platformName() == "offscreen":
            env = os.environ.copy()
            env["QT_QPA_PLATFORM"] = "windows"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "tests.test_ui_split_scale_controls.SplitScaleControlTests.test_axis_range_rows_keep_fix_and_auto_controls_contained",
                    "-v",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                timeout=180,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            return

        self.window.workspace_splitter.setSizes([UI_METRICS["left_width"], 900])
        self.assertEqual(self.window.left_panel.width(), UI_METRICS["left_width"])

        for mode, prefix in (("PL", "pl"), ("DRR", "drr"), ("Compare", "cmp"), ("Power", "power")):
            tab_index = next(i for i in range(self.window.tabs.count()) if self.window.tabs.tabText(i) == mode)
            self.window.tabs.setCurrentIndex(tab_index)
            scroll = getattr(self.window, f"{prefix}_tab_scroll")
            manual_heads = [
                button
                for button in scroll.widget().findChildren(QToolButton)
                if button.text() == "Manual plot ranges"
            ]
            self.assertEqual(len(manual_heads), 1)
            manual_heads[0].setChecked(True)
            scroll.widget().layout().activate()
            self.app.processEvents()
            axis_box = next(
                group for group in scroll.widget().findChildren(QWidget)
                if getattr(group, "title", lambda: None)() == "Axis Ranges"
            )
            self.assertTrue(axis_box.isVisible())
            spins = getattr(self.window, f"{prefix}_spins")
            fixes = getattr(self.window, f"{prefix}_fix_checks")

            for axis, auto_name in (("v", "auto_v"), ("x", "auto_x"), ("y", "auto_y")):
                first = spins[f"{axis}min"]
                second = spins[f"{axis}max"]
                first_fix = fixes[f"{axis}min"]
                second_fix = fixes[f"{axis}max"]
                auto = getattr(self.window, f"{prefix}_{auto_name}_btn")
                row = first.parentWidget()
                self.assertIs(row, second.parentWidget())
                self.assertIs(row, first_fix.parentWidget())
                self.assertIs(row, second_fix.parentWidget())
                self.assertIs(row, auto.parentWidget())
                self.assertTrue(axis_box.contentsRect().contains(row.geometry()))
                self.assertGreater(row.width(), 0)

                row_widgets = [
                    row.layout().itemAt(index).widget()
                    for index in range(row.layout().count())
                    if row.layout().itemAt(index).widget() is not None
                    and row.layout().itemAt(index).widget().isVisible()
                ]
                expected_widgets = [first, first_fix, second, second_fix, auto]
                label_widget = row.layout().labelWidget() if hasattr(row.layout(), "labelWidget") else None
                if label_widget is not None and label_widget.isVisible():
                    expected_widgets.insert(0, label_widget)
                self.assertEqual(row_widgets[:len(expected_widgets)], expected_widgets)
                for child in row_widgets:
                    self.assertTrue(row.contentsRect().contains(child.geometry()))
                self.assertLessEqual(
                    max(child.geometry().bottom() for child in row_widgets),
                    row.contentsRect().bottom(),
                )
                for index, previous in enumerate(row_widgets):
                    for current in row_widgets[index + 1 :]:
                        self.assertFalse(previous.geometry().intersects(current.geometry()))
                self.assertGreaterEqual(
                    auto.width(), auto.fontMetrics().horizontalAdvance(auto.text()) + 12
                )
                self.assertGreaterEqual(
                    first_fix.width(), first_fix.fontMetrics().horizontalAdvance(first_fix.text()) + 20
                )
                self.assertGreaterEqual(first_fix.width(), first_fix.minimumSizeHint().width())
                self.assertTrue(row.contentsRect().contains(auto.geometry()))
                self.assertTrue(axis_box.contentsRect().contains(row.geometry()))

                if prefix == "power":
                    themed_min_height = max(
                        spin.minimumSizeHint().height() for spin in (first, second)
                    )
                    for spin in (first, second):
                        self.assertGreaterEqual(
                            spin.height(),
                            themed_min_height,
                            f"{mode} {axis} spinbox is below themed minimum height",
                        )

            # Verify the same production spinbox style in normal and read-only states.
            spin = spins["vmin"]
            original_read_only = spin.isReadOnly()
            original_property = spin.property("readOnly")
            try:
                for read_only in (False, True):
                    spin.setReadOnly(read_only)
                    spin.setProperty("readOnly", read_only)
                    spin.style().unpolish(spin)
                    spin.style().polish(spin)
                    self.app.processEvents()
                    option = QStyleOptionSpinBox()
                    spin.initStyleOption(option)
                    edit = spin.style().subControlRect(
                        QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxEditField, spin
                    )
                    up = spin.style().subControlRect(
                        QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, spin
                    )
                    down = spin.style().subControlRect(
                        QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxDown, spin
                    )
                    self.assertFalse(edit.intersects(up))
                    self.assertFalse(edit.intersects(down))
                    self.assertTrue(spin.rect().contains(up))
                    self.assertTrue(spin.rect().contains(down))
            finally:
                spin.setReadOnly(original_read_only)
                spin.setProperty("readOnly", original_property)
                spin.style().unpolish(spin)
                spin.style().polish(spin)
                self.app.processEvents()

    def test_axis_range_rows_fit_ordinary_formatted_values_at_minimum_sidebar(self) -> None:
        """Negative ordinary bounds remain fully visible beside steppers and actions."""
        self.window.workspace_splitter.setSizes([UI_METRICS["left_width"], 900])
        self.assertEqual(self.window.left_panel.width(), UI_METRICS["left_width"])
        for mode, prefix in (("PL", "pl"), ("DRR", "drr"), ("Compare", "cmp"), ("Power", "power")):
            tab_index = next(i for i in range(self.window.tabs.count()) if self.window.tabs.tabText(i) == mode)
            self.window.tabs.setCurrentIndex(tab_index)
            page = self.window.tabs.widget(tab_index)
            head = next((button for button in page.findChildren(QToolButton) if button.text() == "Manual plot ranges"), None)
            if head is not None:
                head.setChecked(True)
            self.app.processEvents()
            spins = getattr(self.window, f"{prefix}_spins")
            for axis in ("v", "x", "y"):
                first, second = spins[f"{axis}min"], spins[f"{axis}max"]
                first.setValue(-12.0)
                second.setValue(12.0)
                self.app.processEvents()
                for spin in (first, second):
                    line = spin.lineEdit()
                    margins = line.textMargins()
                    available = line.contentsRect().width() - margins.left() - margins.right()
                    self.assertGreaterEqual(
                        available,
                        line.fontMetrics().horizontalAdvance(spin.text()),
                        f"{mode} {axis} {spin.text()}",
                    )

    def test_empty_canvas_guidance_is_passive_and_tracks_scientific_axes(self) -> None:
        overlay = self.window.empty_canvas_overlay
        self.assertTrue(overlay.isVisible())
        self.assertTrue(overlay.testAttribute(Qt.WA_TransparentForMouseEvents))
        self.assertTrue(overlay.accessibleName())
        self.assertIn("Load", overlay.text())
        self.assertIn("Plot", overlay.text())

        self.window.figure.add_subplot(111)
        self.window.canvas.draw()
        self.app.processEvents()
        self.assertFalse(overlay.isVisible())

        self.window.figure.clear()
        self.window.canvas.draw()
        self.app.processEvents()
        self.assertTrue(overlay.isVisible())

    def test_data_source_spacing_is_tight_at_default_sidebar_width(self) -> None:
        self.window.show()
        self.app.processEvents()
        margins = self.window.data_source_context.layout().contentsMargins()
        self.assertEqual((margins.left(), margins.top(), margins.right(), margins.bottom()), (0, 0, 0, 0))
        self.assertEqual(self.window.data_source_context.layout().spacing(), 0)

    def test_sidebar_can_resize_within_bounds_and_canvas_remains_dominant(self) -> None:
        splitter = self.window.workspace_splitter
        splitter.setSizes([UI_METRICS["left_width"] + 80, 900])
        self.app.processEvents()
        self.assertEqual(splitter.sizes()[0], UI_METRICS["left_width"] + 80)
        self.assertGreater(splitter.sizes()[1], splitter.sizes()[0])

        splitter.setSizes([UI_METRICS["sidebar_max_width"] + 200, 900])
        self.app.processEvents()
        self.assertLessEqual(self.window.left_panel.width(), UI_METRICS["sidebar_max_width"])

    def test_sidebar_collapse_restore_remembers_last_expanded_width(self) -> None:
        splitter = self.window.workspace_splitter
        expanded_width = UI_METRICS["left_width"] + 60
        splitter.setSizes([expanded_width, 900])
        self.app.processEvents()
        remembered = splitter.sizes()[0]
        self.window.sidebar_toggle_btn.setChecked(False)
        self.app.processEvents()
        self.assertFalse(self.window.left_panel.isVisible())
        self.window.sidebar_toggle_btn.setChecked(True)
        self.app.processEvents()
        self.assertEqual(splitter.sizes()[0], remembered)

    def test_resizable_sidebar_uses_complete_compact_tab_labels(self) -> None:
        tab_bar = self.window.tabs.tabBar()
        self.assertFalse(tab_bar.usesScrollButtons())
        self.assertEqual(tab_bar.elideMode(), Qt.ElideNone)
        self.assertEqual(
            [self.window.tabs.tabText(i) for i in range(self.window.tabs.count())],
            ["PL", "DRR", "Compare", "Power", "MCD", "MCD Peak Shift", "SHG", "Slides", "Tools"],
        )
        expected_modes = [
            "PL",
            "DRR",
            "Compare",
            "Power Dependent",
            "MCD",
            None,  # MCD Peak Shift
            "SHG Processing",
            None,  # Slides
            None,  # Tools
        ]
        for index, expected in enumerate(expected_modes):
            self.window.tabs.setCurrentIndex(index)
            self.assertEqual(self.window._active_mode(), expected)

    def test_mcd_organizer_launcher_lives_on_tools_page(self) -> None:
        labels = [
            self.window.tabs.tabText(index) for index in range(self.window.tabs.count())
        ]
        mcd_page = self.window.tabs.widget(labels.index("MCD"))
        tools_page = self.window.tabs.widget(labels.index("Tools"))
        self.assertFalse(mcd_page.isAncestorOf(self.window.mcd_extract_btn))
        self.assertTrue(tools_page.isAncestorOf(self.window.mcd_extract_btn))

    def test_drr_summary_wraps_the_complete_first_filename(self) -> None:
        full_name = (
            "YZ364_0Tpa_3.6KREF_700nmc_0p1sx20_"
            "TG-1.05BG=0_extra_long_measurement_name.csv"
        )
        self.window.drr_selected_files = [full_name, "second.csv"]
        self.window.drr_controller._update_drr_selection_labels()

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
            self._wait_for_drr_catalog()
            self.assertEqual(self.window.drr_selected_files, [])
            self.window.drr_selected_files = [
                "Initial Data/sample_760nmc_rep1_1.csv"
            ]

            self._write_drr_measurement(second)
            self.window._refresh_file_lists(auto=True)
            self._wait_for_drr_catalog()

            self.assertEqual(
                set(self.window.drr_selected_files),
                {
                    "Initial Data/sample_760nmc_rep1_1.csv",
                    "Initial Data/sample_760nmc_rep1_2.csv",
                },
            )

    def test_drr_source_search_is_debounced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            initial = Path(tmp) / "Initial Data"
            initial.mkdir()
            self._write_drr_measurement(initial / "sample_760nmc_rep1.csv")
            self.window._set_current_folder(tmp, remember=False)
            self._wait_for_drr_catalog()

            observed: dict[str, int] = {}

            def fake_exec(dialog: QDialog) -> int:
                filter_edit = dialog.findChild(QLineEdit)
                group_list = next(
                    widget for widget in dialog.findChildren(QListWidget) if widget.count()
                )
                self.assertIsNotNone(filter_edit)
                self.assertGreater(group_list.count(), 0)
                filter_edit.setText("query-that-does-not-match")
                self.app.processEvents()
                observed["before"] = group_list.count()
                QTest.qWait(230)
                self.app.processEvents()
                observed["after"] = group_list.count()
                return QDialog.Rejected

            with patch.object(QDialog, "exec", fake_exec):
                self.window.drr_controller._open_drr_source_dialog(
                    title="Choose DRR files", selected=[], baseline_mode=False
                )

            self.assertGreater(observed["before"], 0)
            self.assertEqual(observed["after"], 0)

    def test_drr_catalog_discovery_runs_off_the_gui_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entered = threading.Event()
            release = threading.Event()
            source = DrrSource(
                source="Initial Data/async_760nmc.csv",
                filename="async_760nmc.csv",
                group_key="async_760nmc",
                session_date="2026-08-26",
                modified_time=1.0,
                is_background=False,
            )

            def delayed_discovery(_folder, *, cache=None):
                entered.set()
                release.wait(2.0)
                return [source]

            self.window.current_folder = str(root)
            with patch(
                "ui_qt.main_window.discover_drr_sources",
                side_effect=delayed_discovery,
            ):
                self.window._refresh_file_lists()
                self._wait_for_file_catalog()
                self.assertTrue(entered.wait(1.0))
                # The worker is intentionally blocked; reaching this point
                # proves the refresh call itself did not perform discovery.
                self.assertFalse(release.is_set())
                self.assertTrue(self.window._drr_refresh_running)

                release.set()
                self._wait_for_drr_catalog()

            self.assertEqual(self.window.drr_available_sources, [source])

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
            self._wait_for_drr_catalog()
            chosen = "Initial Data/sample_760nmc_rep1_1.csv"
            self.window.drr_selected_files = [chosen]
            self._write_drr_measurement(third)

            self.window._refresh_file_lists(auto=True)
            self._wait_for_drr_catalog()

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
            self._wait_for_file_catalog()
            self.window._restore_list_selection(self.window.pl_files, ["same_name.csv"])
            self.window._restore_list_selection(self.window.shg_files, ["same_name.csv"])
            self.window.drr_selected_files = ["same_name.csv"]
            self.window.loaded = LoadedState(mode="PL", folder=first_tmp, cube=None)
            self.window.last_plotted_mode = "PL"

            self.window._set_current_folder(second_tmp, remember=False)
            self._wait_for_file_catalog()

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

            self.assertEqual(self.window.power_controller._power_selected_group_key(), "")
            self.assertEqual(self.window.shg_controller._shg_selected_file(), "")
            self.assertEqual(self.window.shg_controller._shg_compare_files(), ("", ""))

    def test_compare_auto_assignment_leaves_duplicate_channel_unassigned(self) -> None:
        with (
            patch.object(type(self.window.compare_controller), "_cmp_assign_candidate_files", return_value=[]),
            patch(
                "ui_qt.controllers_compare.coherent_compare_auto_assignment",
                return_value=(
                    {"KK": "ambiguous.csv"},
                    {"KK": ["first.csv", "second.csv"]},
                    None,
                    [],
                ),
            ),
            patch.object(type(self.window.compare_controller), "_on_cmp_plot_param_changed"),
        ):
            self.window.compare_controller._cmp_auto_assign_channels()

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
        panels_height = self.window.menu_toolbar_host.panels_toolbar.height()
        self.window.resize(1180, self.window.minimumHeight() + panels_height)
        self.app.processEvents()

        scroll = self.window.drr_tab_scroll
        self.assertIsInstance(scroll, QScrollArea)
        self.assertEqual(
            scroll.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff
        )
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
        self.assertLessEqual(scroll.widget().width(), scroll.viewport().width())

        # The themed compact page may need vertical scrolling, but every
        # visible control remains reachable without horizontal clipping.
        for control in (
            self.window.drr_baseline_combo,
            self.window.drr_yaxis_combo,
            self.window.drr_derivative_combo,
        ):
            scroll.ensureWidgetVisible(control)
            self.app.processEvents()
            top_left = control.mapTo(scroll.viewport(), control.rect().topLeft())
            bottom_right = control.mapTo(scroll.viewport(), control.rect().bottomRight())
            self.assertTrue(scroll.viewport().rect().contains(top_left))
            self.assertTrue(scroll.viewport().rect().contains(bottom_right))

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

    def test_source_toolbar_action_is_visible_for_scientific_workflows_and_hidden_for_slides(self) -> None:
        self.assertTrue(self.window.menu_toolbar_host.source_widget_action.isVisible())
        slides_index = next(i for i in range(self.window.tabs.count()) if self.window.tabs.tabText(i) == "Slides")
        self.window.tabs.setCurrentIndex(slides_index)
        self.app.processEvents()
        self.assertFalse(self.window.menu_toolbar_host.source_widget_action.isVisible())
        self.assertFalse(self.window.menu_toolbar_host.source_separator_action.isVisible())
        self.window.tabs.setCurrentIndex(0)
        self.app.processEvents()
        self.assertTrue(self.window.menu_toolbar_host.source_widget_action.isVisible())
        self.assertTrue(self.window.menu_toolbar_host.source_separator_action.isVisible())

    def test_drag_feedback_still_targets_the_existing_folder_edit(self) -> None:
        self.window.current_folder = ""
        self.window._set_drop_highlight(True)
        self.assertEqual(self.window.folder_edit.property("appRole"), "dropTarget")
        self.window._set_drop_highlight(False)
        self.assertIsNone(self.window.folder_edit.property("appRole"))

    def test_slides_workspace_is_full_width_and_owns_its_build_controls(self) -> None:
        slides_index = next(
            index
            for index in range(self.window.tabs.count())
            if self.window.tabs.tabText(index) == "Slides"
        )
        self.window.workflow_tabs.setCurrentIndex(slides_index)
        QApplication.processEvents()
        self.assertIs(self.window.workspace_stack.currentWidget(), self.window.presentation_widget)
        self.assertFalse(self.window.left_panel.isVisible())
        self.assertFalse(self.window.data_source_context.isVisible())
        self.assertFalse(self.window.sidebar_toggle_btn.isEnabled())
        self.assertFalse(self.window.load_action.isEnabled())
        self.assertFalse(self.window.plot_action.isEnabled())
        self.assertFalse(self.window.save_action.isEnabled())
        counts = [
            self.window.presentation_widget.images_per_slide_combo.itemData(index)
            for index in range(self.window.presentation_widget.images_per_slide_combo.count())
        ]
        self.assertEqual(counts, [0, *range(1, 13)])
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

        with patch.object(type(self.window.drr_controller), "_on_drr_plot_param_changed"):
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
        with patch.object(type(self.window.drr_controller), "_on_drr_plot_param_changed"):
            self.window.drr_split_scale_chk.setChecked(True)
        self.window._set_spin_value_silent(self.window.drr_spins["vmin"], -999.0)
        self.window._set_spin_value_silent(self.window.drr_spins["vmax"], 999.0)

        with patch.object(type(self.window.drr_controller), "_on_drr_plot_param_changed") as replot:
            self.window.drr_split_scale_chk.setChecked(False)

        replot.assert_called_once()
        self.assertGreater(self.window.drr_spins["vmin"].value(), -999.0)
        self.assertLess(self.window.drr_spins["vmax"].value(), 999.0)

    def test_drr_xmin_change_refreshes_ranges_with_centered_split(self) -> None:
        cube = DataCube(
            energy=np.array([0.0, 1.0, 2.0, 3.0]),
            gate=np.array([0.0, 1.0]),
            Z=np.asarray([[1.0, 2.0, 100.0, 200.0], [3.0, 4.0, 300.0, 400.0]]),
            gate_label="Gate",
            title="DRR range refresh",
            cbar_label="DR/R",
        )
        self.window.loaded = LoadedState(mode="DRR", folder="", cube=cube)
        for key, value in (("xmin", 0.0), ("xmax", 3.0), ("ymin", 0.0), ("ymax", 1.0)):
            self.window._set_spin_value_silent(self.window.drr_spins[key], value)
        self.window.drr_split_scale_chk.setChecked(True)
        self.window._set_spin_value_silent(self.window.drr_split_spins["x0"], 0.5)
        spin = self.window.drr_spins["xmin"]
        with (
            patch.object(self.window, "_schedule_plot_redraw"),
        ):
            spin.setValue(1.5)

        self.assertAlmostEqual(self.window.drr_split_spins["x0"].value(), 2.5)

    def test_compare_xmin_signal_source_centers_split_range(self) -> None:
        cube = DataCube(
            energy=np.array([0.0, 1.0, 2.0, 3.0]),
            gate=np.array([0.0, 1.0]),
            Z=np.asarray([[1.0, 2.0, 100.0, 200.0], [3.0, 4.0, 300.0, 400.0]]),
            gate_label="Gate",
            title="Compare range refresh",
            cbar_label="Intensity",
        )
        self.window.loaded = LoadedState(
            mode="Compare", folder="", compare_cubes={"KK": cube, "KKp": cube}
        )
        self.window.available_files = ["kk.csv", "kkp.csv"]
        self.window.compare_controller._cmp_set_channel_combo_items()
        self.window.cmp_channel_combos["KK"].setCurrentText("kk.csv")
        self.window.cmp_channel_combos["KKp"].setCurrentText("kkp.csv")
        for key, value in (("xmin", 0.0), ("xmax", 3.0), ("ymin", 0.0), ("ymax", 1.0)):
            self.window._set_spin_value_silent(self.window.cmp_spins[key], value)
        self.window.cmp_split_scale_chk.setChecked(True)
        self.window._set_spin_value_silent(self.window.cmp_split_spins["x0"], 0.5)
        with patch.object(self.window, "_schedule_plot_redraw"):
            self.window.cmp_spins["xmin"].setValue(1.5)

        self.assertAlmostEqual(self.window.cmp_split_spins["x0"].value(), 2.5)

    def test_compare_auto_background_toggle_refreshes_color_range(self) -> None:
        cube = DataCube(
            energy=np.array([0.0, 1.0, 2.0, 3.0]),
            gate=np.array([0.0, 1.0]),
            Z=np.asarray([[10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]]),
            gate_label="Gate",
            title="Compare background refresh",
            cbar_label="Intensity",
        )
        self.window.loaded = LoadedState(
            mode="Compare", folder="", compare_cubes={"KK": cube, "KKp": cube}
        )
        self.window.available_files = ["kk.csv", "kkp.csv"]
        self.window.compare_controller._cmp_set_channel_combo_items()
        self.window.cmp_channel_combos["KK"].setCurrentText("kk.csv")
        self.window.cmp_channel_combos["KKp"].setCurrentText("kkp.csv")
        for key, value in (("xmin", 0.0), ("xmax", 3.0), ("ymin", 0.0), ("ymax", 1.0)):
            self.window._set_spin_value_silent(self.window.cmp_spins[key], value)
        self.window.cmp_vp_auto_background_chk.setChecked(False)
        self.window._set_spin_value_silent(self.window.cmp_spins["vmin"], -999.0)
        self.window._set_spin_value_silent(self.window.cmp_spins["vmax"], 999.0)

        self.window.cmp_vp_auto_background_chk.setChecked(True)

        self.assertGreater(self.window.cmp_spins["vmin"].value(), -999.0)
        self.assertLess(self.window.cmp_spins["vmax"].value(), 999.0)

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
            type(self.window.drr_controller),
            "_open_drr_source_dialog",
            return_value=["new_760nmc_rep1.csv"],
        ), patch.object(self.window, "_start_load") as start_load:
            self.window.drr_controller._edit_drr_measurements()

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
            self._wait_for_file_catalog()
            self._wait_for_drr_catalog()
            selected = "Initial Data/sample_760nmc_rep1.csv"

            with patch.object(
                type(self.window.drr_controller),
                "_open_drr_source_dialog",
                return_value=[selected],
            ), patch.object(self.window, "_start_load") as start_load:
                self.window.drr_controller._edit_drr_measurements()

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
            self._wait_for_file_catalog()
            self.window.drr_baseline_files_manual = [first.name, second.name]
            self.window.drr_baseline_combine_combo.setCurrentText(
                "Last frame from each file, then average"
            )

            accepted = self.window.drr_controller._apply_drr_background_gate_default()

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
            type(self.window.drr_controller),
            "_open_drr_source_dialog",
            return_value=["new_760nmc_rep1.csv"],
        ), patch.object(self.window, "_start_load") as start_load:
            self.window.drr_controller._edit_drr_measurements()

        self.assertEqual(
            self.window.drr_baseline_files_manual, ["old_760nmc_back.csv"]
        )
        start_load.assert_called_once_with("DRR")


class WindowLifecycleTests(unittest.TestCase):
    """Regression: pending plot redraw timers must never fire after close."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_closing_window_cancels_pending_plot_redraw(self) -> None:
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            window = MainWindow()
        window.resize(1180, 820)
        window.show()

        energy = np.linspace(0.0, 4.0, 21)
        cube = DataCube(
            energy=energy,
            gate=np.array([0.0, 1.0]),
            Z=np.vstack((energy**2, 2.0 * energy**2)),
            gate_label="Gate",
            title="DRR derivative",
            cbar_label="DR/R",
        )
        window.loaded = LoadedState(mode="DRR", folder="", cube=cube)
        for key, value in (("xmin", 0.0), ("xmax", 4.0), ("ymin", 0.0), ("ymax", 1.0)):
            window._set_spin_value_silent(window.drr_spins[key], value)
        window._refresh_automatic_ranges("DRR")

        redraw_calls: list[str] = []
        original_run = window._run_scheduled_plot_redraw

        def spy(mode: str) -> None:
            redraw_calls.append(mode)
            return original_run(mode)

        window._run_scheduled_plot_redraw = spy

        with patch.object(window, "_plot_mode") as plot_mode:
            window.drr_derivative_combo.setCurrentText("dE")
            timer = window._plot_redraw_timers.get("DRR")
            self.assertIsNotNone(timer)
            self.assertTrue(timer.isActive())

            window.close()
            # Closing must disarm the pending redraw timer.
            self.assertFalse(timer.isActive())

            window.deleteLater()
            self.app.processEvents()
            QTest.qWait(200)  # Give any surviving timer a chance to fire.
            self.app.processEvents()

            plot_mode.assert_not_called()

        self.assertEqual(redraw_calls, [])

    def test_automatic_update_check_does_not_fire_after_close(self) -> None:
        calls: list[str] = []
        with (
            patch.object(MainWindow, "_restore_last_folder", lambda _self: None),
            patch.object(MainWindow, "_auto_update_check_enabled", lambda _self: True),
            patch.object(MainWindow, "_run_automatic_update_check", lambda _self: calls.append("fired")),
        ):
            window = MainWindow()
        window.show()

        window.close()
        window.deleteLater()
        self.app.processEvents()
        QTest.qWait(1700)  # Well past the 1500 ms startup delay.
        self.app.processEvents()

        self.assertEqual(calls, [])

    def test_automatic_update_check_fires_once_after_delay_while_alive(self) -> None:
        calls: list[str] = []
        with (
            patch.object(MainWindow, "_restore_last_folder", lambda _self: None),
            patch.object(MainWindow, "_auto_update_check_enabled", lambda _self: True),
            patch.object(MainWindow, "_run_automatic_update_check", lambda _self: calls.append("fired")),
        ):
            window = MainWindow()
        window.show()

        self.assertEqual(calls, [])
        QTest.qWait(1100)  # Before the 1500 ms startup delay.
        self.app.processEvents()
        self.assertEqual(calls, [])
        QTest.qWait(700)  # Total ~1800 ms: the check must have fired once.
        self.app.processEvents()
        self.assertEqual(calls, ["fired"])

        window.close()
        window.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
