from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import ui_qt.presentation_widget as presentation_widget
from ui_qt.presentation_widget import PresentationBuilderWidget


class SlidesLatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._settings_tmp = tempfile.TemporaryDirectory()
        settings_path = str(Path(self._settings_tmp.name) / "settings.ini")
        self._settings_patcher = patch.object(
            presentation_widget,
            "QSettings",
            lambda *args, **kwargs: QSettings(settings_path, QSettings.IniFormat),
        )
        self._settings_patcher.start()
        self.widget = PresentationBuilderWidget()

    def tearDown(self) -> None:
        self.widget.close()
        self._settings_patcher.stop()
        self._settings_tmp.cleanup()

    def test_bulk_append_is_ordered_unique_and_planned_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"plot_{index:02d}.png" for index in range(30)]
            with patch.object(
                presentation_widget,
                "plan_presentation_slides",
                wraps=presentation_widget.plan_presentation_slides,
            ) as planner:
                added = self.widget._append_queue_paths([paths[4], paths[1], paths[4], *paths[5:]])
                self.assertEqual(planner.call_count, 1)
                # A selection-only preview consumes the valid plan snapshot.
                self.widget._update_preview()
                self.widget._slide_title(0, 1)
                self.assertEqual(planner.call_count, 1)
            self.assertEqual(added, 27)
            self.assertEqual(self.widget._queued_paths(), [path.resolve() for path in [paths[4], paths[1], *paths[5:]]])

    def test_remove_clear_readd_and_move_keep_queue_order_and_rebuild_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"plot_{index}.png" for index in range(4)]
            self.widget._append_queue_paths(paths)
            with patch.object(self.widget, "_rebuild_plan", wraps=self.widget._rebuild_plan) as rebuild:
                self.widget.queue_list.item(1).setSelected(True)
                self.widget._remove_selected()
                self.assertEqual(rebuild.call_count, 1)
            self.assertEqual(self.widget._queued_paths(), [paths[0].resolve(), paths[2].resolve(), paths[3].resolve()])

            self.widget._clear_queue()
            self.assertEqual(self.widget._queued_paths(), [])
            self.widget._append_queue_paths([paths[3], paths[0]])
            self.assertEqual(self.widget._queued_paths(), [paths[3].resolve(), paths[0].resolve()])

            self.widget.queue_list.item(0).setSelected(True)
            self.widget._move_selected(1)
            self.assertEqual(self.widget._queued_paths(), [paths[0].resolve(), paths[3].resolve()])

    def test_group_and_title_changes_rebuild_the_single_final_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"run_D{index}_F0_plot.png" for index in range(3)]
            self.widget._append_queue_paths(paths)
            self.widget._slide_groups()
            with patch.object(
                presentation_widget,
                "plan_presentation_slides",
                wraps=presentation_widget.plan_presentation_slides,
            ) as planner:
                self.widget.title_edit.setText("Experiment")
                self.widget.group_by_combo.setCurrentIndex(self.widget.group_by_combo.findData("queue"))
                self.assertGreaterEqual(planner.call_count, 1)
                self.assertEqual(self.widget._slide_title(0, 1), "Experiment")
                count = planner.call_count
                self.widget._update_preview()
                self.assertEqual(planner.call_count, count)

    def test_preview_option_signals_use_cached_snapshot_without_argument_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.widget._append_queue_paths([root / "plot.png"])
            self.widget.caption_combo.setCurrentIndex(self.widget.caption_combo.findData("full"))
            self.widget.panel_labels_chk.setChecked(True)
            self.widget.slide_list.setCurrentRow(0)
            self.assertEqual(self.widget.preview._caption_mode, "full")
            self.assertTrue(self.widget.preview._panel_labels)


if __name__ == "__main__":
    unittest.main()
