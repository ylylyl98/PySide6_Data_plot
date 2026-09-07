from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import QAbstractItemView, QApplication

from ui_qt.presentation_widget import PresentationBuilderWidget


class PresentationWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.widget = PresentationBuilderWidget()
        self.widget.resize(1280, 760)
        self.widget.show()

    def tearDown(self) -> None:
        self.widget.close()

    def _wait_for_discovery(self) -> None:
        for _ in range(100):
            self.app.processEvents()
            if not self.widget._discovery_running:
                return
            QThreadPool.globalInstance().waitForDone(10)
            self.app.processEvents()
        self.fail("plot discovery worker did not finish")

    def test_plot_workspace_defaults_and_optional_controls_are_compact(self) -> None:
        self.assertFalse(self.widget.panel_labels_chk.isChecked())
        self.assertEqual(self.widget.images_per_slide_combo.currentData(), 0)
        self.assertEqual(self.widget.caption_combo.currentData(), "none")
        self.assertEqual(self.widget.group_by_combo.currentData(), "doping")
        self.assertIn("never adds unselected energies", self.widget.group_by_combo.toolTip())
        self.assertFalse(self.widget.advanced_widget.isVisible())
        sizes = self.widget.workspace_splitter.sizes()
        self.assertGreater(sizes[0], sizes[1])
        self.assertGreater(sizes[0], sizes[2])
        self.widget.advanced_btn.setChecked(True)
        self.app.processEvents()
        self.assertTrue(self.widget.advanced_widget.isVisible())

    @staticmethod
    def _png(root: Path, relative: str, modified: float) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (120, 80), (30, 90, 180)).save(path)
        os.utime(path, (modified, modified))
        return path.resolve()

    def test_names_are_complete_and_newest_first_without_horizontal_scroll(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self._png(root, "PL/run/old_plot.png", 100)
            long_name = "newest_" + "very_long_measurement_name_" * 8 + ".png"
            newest = self._png(root, f"PL/run/{long_name}", 200)
            self.widget.image_root_edit.setText(str(root))
            self.widget.refresh_plots()
            self._wait_for_discovery()

            self.assertEqual(Path(self.widget.available_list.item(0).data(Qt.UserRole)), newest)
            visible = self.widget.available_list.item(0).text().replace("\u200b", "")
            self.assertIn(long_name, visible)
            self.assertNotIn("...", visible)
            self.assertEqual(self.widget.available_list.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff)
            self.assertEqual(
                self.widget.available_list.verticalScrollMode(),
                QAbstractItemView.ScrollPerPixel,
            )

            self.widget.sort_combo.setCurrentIndex(self.widget.sort_combo.findData("oldest"))
            self.assertEqual(Path(self.widget.available_list.item(0).data(Qt.UserRole)), old)

    def test_mcd_filters_and_newest_matching_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older_combo = self._png(root, "MCD/older/run_MCD_Combo_map.png", 100)
            self._png(root, "MCD/older/run_MCD_vs_B_E1.png", 101)
            newest_combo = self._png(root, "MCD/newer/run_MCD_Combo_map.png", 200)
            newest_trace = self._png(root, "MCD/newer/run_MCD_vs_B_E2.png", 201)
            self.widget.image_root_edit.setText(str(root))
            self.widget.refresh_plots()
            self._wait_for_discovery()

            self.widget.mcd_type_combo.setCurrentIndex(
                self.widget.mcd_type_combo.findData("mcd_combo")
            )
            shown = {
                Path(self.widget.available_list.item(index).data(Qt.UserRole))
                for index in range(self.widget.available_list.count())
            }
            self.assertEqual(shown, {older_combo, newest_combo})

            self.widget._add_newest_mcd_pair()
            self.assertEqual(self.widget._queued_paths(), [newest_combo, newest_trace])
            self.assertTrue(self.widget.mcd_folder_status.isVisible())
            self.assertIn("same folder", self.widget.mcd_folder_status.text())
            for index in range(self.widget.queue_list.count()):
                text = self.widget.queue_list.item(index).text().replace("\u200b", "")
                self.assertIn(self.widget._queued_paths()[index].name, text)
                self.assertIn("Folder:", text)
                self.assertRegex(text, r"F\d+")
                self.assertNotIn("...", text)
            self.assertEqual(self.widget.queue_list.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff)

            different_trace = self._png(root, "MCD/different/run_MCD_vs_B_E3.png", 300)
            self.widget.refresh_plots()
            self._wait_for_discovery()
            self.widget._clear_queue()
            self.widget._append_queue_path(newest_combo)
            self.widget._append_queue_path(different_trace)
            self.assertIn("different folders", self.widget.mcd_folder_status.text())

    def test_auto_layout_keeps_up_to_twelve_plots_on_one_slide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self._png(root, f"PL/run/plot_{index:02d}.png", index + 1) for index in range(13)]
            for path in paths:
                self.widget._append_queue_path(path)

            self.assertEqual([len(group) for group in self.widget._slide_groups()], [12, 1])
            self.widget._clear_queue()
            for path in paths[:7]:
                self.widget._append_queue_path(path)
            self.assertEqual([len(group) for group in self.widget._slide_groups()], [7])

            self.widget.images_per_slide_combo.setCurrentIndex(
                self.widget.images_per_slide_combo.findData(6)
            )
            self.assertEqual([len(group) for group in self.widget._slide_groups()], [6, 1])

    def test_doping_grouping_combines_repeated_doping_but_only_queued_energies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doping_one_first = self._png(
                root, "MCD/run_D1_F0/run_D1_F0_MCD_vs_B_E1.57eV_W5meV.png", 1
            )
            doping_two = self._png(
                root, "MCD/run_D2_F0/run_D2_F0_MCD_vs_B_E1.60eV_W5meV.png", 2
            )
            doping_one_return = self._png(
                root, "MCD/run_D1_F20/run_D1_F20_MCD_vs_B_E1.64eV_W5meV.png", 3
            )
            unselected_energy = self._png(
                root, "MCD/run_D1_F20/run_D1_F20_MCD_vs_B_E1.66eV_W5meV.png", 4
            )
            for path in (doping_one_first, doping_two, doping_one_return):
                self.widget._append_queue_path(path)

            groups = self.widget._slide_groups()
            self.assertEqual(groups, [[doping_one_first, doping_one_return], [doping_two]])
            self.assertNotIn(unselected_energy, [path for group in groups for path in group])
            self.assertIn("Doping = 1 V", self.widget._slide_title(0, 2))

    def test_selection_status_tracks_scrolling_and_can_return_to_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(30):
                self._png(root, f"PL/run/plot_{index:02d}.png", index + 1)
            self.widget.image_root_edit.setText(str(root))
            self.widget.refresh_plots()
            self._wait_for_discovery()

            selected = self.widget.available_list.item(0)
            selected.setSelected(True)
            self.app.processEvents()
            self.assertIn("selected PNG is visible", self.widget.available_selection_status.text())

            scrollbar = self.widget.available_list.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            self.app.processEvents()
            self.assertIn("selected PNG is above", self.widget.available_selection_status.text())

            self.widget._show_last_selected()
            self.app.processEvents()
            self.assertIn("selected PNG is visible", self.widget.available_selection_status.text())
            self.assertTrue(selected.isSelected())

    def test_filter_rebuild_preserves_selected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._png(root, "PL/run/first_plot.png", 1)
            second = self._png(root, "PL/run/second_plot.png", 2)
            self.widget.image_root_edit.setText(str(root))
            self.widget.refresh_plots()
            self._wait_for_discovery()
            self.widget.available_list.item(0).setSelected(True)
            selected_path = Path(self.widget.available_list.item(0).data(Qt.UserRole))

            # Apply the filter synchronously in the test; normal typing uses
            # the widget's short debounce timer.
            self.widget.search_edit.setText("plot")
            self.widget._apply_filters()

            self.assertEqual(
                [Path(item.data(Qt.UserRole)) for item in self.widget.available_list.selectedItems()],
                [selected_path],
            )
            self.assertEqual({first, second}, {
                Path(self.widget.available_list.item(i).data(Qt.UserRole))
                for i in range(self.widget.available_list.count())
            })


if __name__ == "__main__":
    unittest.main()
