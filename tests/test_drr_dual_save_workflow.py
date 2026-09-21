from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from core.loader import DataCube
from core.drr_sources import DrrSource
from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow


class DrrDualSaveWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # The organization/application overload still uses the Windows registry,
        # even after setDefaultFormat. Keep every window in its own test file.
        self.settings = QSettings(str(self.root / 'settings.ini'), QSettings.IniFormat)
        self.settings_patch = patch('ui_qt.main_window.QSettings', return_value=self.settings)
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.source = self.root / "measurement.csv"
        self.source.write_text("synthetic source\n", encoding="utf-8")
        energy = np.linspace(-2.0, 2.0, 21)
        self.cube = DataCube(
            energy=energy,
            gate=np.asarray([-1.0, 0.0, 1.0]),
            Z=np.vstack([energy**2, energy**2 + 1.0, energy**2 - 2.0]),
            gate_label="Gate",
            title="Synthetic DR/R",
            cbar_label="DR/R",
        )
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None), patch.object(
            MainWindow, "_schedule_automatic_update_check", lambda _self: None
        ):
            self.window = MainWindow()
        self.window.current_folder = ""
        self.window.loaded = LoadedState(
            mode="DRR",
            folder=str(self.root),
            primary_file=self.source.name,
            selected_files=[self.source.name],
            cube=self.cube,
            drr_mode_label="DR/R Map",
            drr_baseline_text="None",
        )
        index = next(
            index for index in range(self.window.tabs.count())
            if self.window.tabs.tabText(index) == "DRR"
        )
        self.window.tabs.setCurrentIndex(index)
        self.app.processEvents()
        self.assertEqual(self.window._active_mode(), "DRR")
        self.window.last_plotted_mode = "DRR"
        self.window._ensure_loaded_matches_ui_params = lambda _mode: True
        self.results: list[dict] = []
        self.errors: list[str] = []
        original_done = self.window._on_export_done

        def on_done(result: object) -> None:
            self.results.append(result)
            original_done(result)

        def on_error(message: str) -> None:
            self.errors.append(message)

        self.window._on_export_done = on_done
        self.window._on_export_error = on_error

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.tmp.cleanup()

    def _run_save(self, *, view: str = "raw", side_by_side: bool = False) -> dict:
        self.window._drr_plot_view = view
        self.window._drr_side_by_side = side_by_side
        previous_results = len(self.results)
        previous_errors = len(self.errors)
        self.window._start_export("DRR")
        self.assertTrue(self.window.thread_pool.waitForDone(5000))
        for _ in range(100):
            self.app.processEvents()
            if len(self.results) > previous_results or len(self.errors) > previous_errors:
                break
        self.assertEqual(len(self.errors), previous_errors, self.errors)
        self.assertGreater(len(self.results), previous_results, "background export did not deliver a result")
        return self.results[-1]

    def _output_dir(self) -> Path:
        return self.root / "Processed Data" / "DRR"

    def _configure_three_regions(self):
        w = self.window
        for key, value in dict(xmin=-2, xmax=2, ymin=-1, ymax=1, vmin=-5, vmax=5).items():
            w._set_spin_value_silent(w.drr_spins[key], value)
        w.drr_region_count_combo.setCurrentIndex(2)
        w._set_spin_value_silent(w.drr_split_spins['x0'], -.5)
        w._set_spin_value_silent(w.drr_split_spins['x1'], .5)
        for spins in (w.drr_split_spins, w.drr_second_split_spins):
            for side in ('left', 'middle', 'right'):
                w._set_spin_value_silent(spins[side + '_vmin'], -5)
                w._set_spin_value_silent(spins[side + '_vmax'], 5)

    def test_three_region_save_both_writes_both_real_pngs(self):
        self._configure_three_regions()
        self._run_save()
        paths = list(self._output_dir().glob('*.png'))
        self.assertEqual(len(paths), 2)
        for path in paths:
            self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_three_region_toolbar_png_saves_snapshot(self):
        self._configure_three_regions()
        self.test_toolbar_png_saves_snapshot_in_background()

    def test_toolbar_png_saves_snapshot_in_background(self):
        import time
        self.window._plot_mode('DRR')
        target = self.root / 'toolbar.png'
        reference = self.root / 'reference.png'
        def choose(*args):
            self.window.figure.savefig(reference)
            return str(target), 'PNG'
        with patch('ui_qt.main_window.QFileDialog.getSaveFileName', side_effect=choose), patch(
                'ui_qt.main_window.QMessageBox.critical', side_effect=lambda *args: self.fail(str(args[2]))):
            self.window.toolbar.save_figure()
        job = self.window.toolbar._png_save_job
        self.window.figure.clear()
        deadline = time.monotonic() + 10
        while not job.done and time.monotonic() < deadline:
            self.app.processEvents(); time.sleep(.005)
        self.assertTrue(job.done)
        self.assertIsNone(job.error)
        self.assertEqual(target.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        from matplotlib.image import imread
        np.testing.assert_array_equal(imread(reference), imread(target))

    def test_save_reuses_display_derivative(self):
        self.window.drr_controller._drr_cube_with_metadata(2)
        with patch('ui_qt.main_window.apply_sg_derivative_energy',
                   side_effect=AssertionError('recomputed displayed derivative')):
            self._run_save()

    def test_drr_save_refreshes_history_without_directory_scan(self):
        self.window.current_folder = str(self.root)
        self.window.drr_available_sources = [DrrSource(
            source=self.source.name, filename=self.source.name, group_key='measurement',
            session_date='2026-09-19', modified_time=0, is_background=False)]
        with patch.object(self.window, '_refresh_file_lists') as scan:
            self._run_save()
            self.window.thread_pool.waitForDone(5000)
            self.app.processEvents()
            scan.assert_not_called()
        self.assertTrue(self.window.drr_available_sources[0].processed)
        self.assertEqual(self.window.drr_available_sources[0].metadata_role, 'measurement')

    def test_changed_sg_parameters_recompute_derivative(self):
        self.window.drr_controller._drr_cube_with_metadata(2)
        self.window.drr_sg_window_spin.setValue(9)
        from core.processing import apply_sg_derivative_energy
        with patch('ui_qt.main_window.apply_sg_derivative_energy',
                   wraps=apply_sg_derivative_energy) as compute:
            self._run_save()
        compute.assert_called_once()
        self.assertEqual(compute.call_args.kwargs['window_length'], 9)

    def test_saved_history_does_not_update_another_folder(self):
        self.window.current_folder = str(self.root / 'other')
        with patch('ui_qt.main_window.refresh_drr_source_history') as refresh:
            self.window._refresh_drr_saved_history(str(self.root))
            refresh.assert_not_called()

    def test_cleaned_source_copies_require_directory_refresh(self):
        self.window.current_folder = str(self.root)
        with patch.object(self.window, '_refresh_file_lists') as scan:
            self.window._on_export_done(dict(mode='DRR', folder=str(self.root), cleaned=1))
            scan.assert_called_once_with(auto=True, mode='DRR')

    def test_plain_save_reports_queue_and_output_stages(self):
        ranges = self.window._mode_spins('DRR')
        for key, value in [('xmin', -2), ('xmax', 2), ('ymin', -1), ('ymax', 1)]:
            ranges[key].setValue(value)
        self._run_save()
        messages = '\n'.join(self.window.log_lines)
        for stage in ('queued for', 'Computing second derivative', 'Raw: Writing DAT',
                      'Raw: Rendering PNG', 'd2E: Writing DAT', 'd2E: Rendering PNG',
                      'worker completed in'):
            self.assertIn(stage, messages)

    def test_real_background_save_from_each_visible_drr_view_writes_pair(self) -> None:
        ranges = self.window._mode_spins("DRR")
        ranges["xmin"].setValue(-1.5)
        ranges["xmax"].setValue(1.5)
        ranges["ymin"].setValue(-0.75)
        ranges["ymax"].setValue(0.75)
        for view, side_by_side in (("raw", False), ("second", False), ("raw", True)):
            with self.subTest(view=view, side_by_side=side_by_side):
                with patch(
                    "core.export._save_heatmap_png",
                    side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b"PNG"),
                ):
                    self.window._last_export_request_key = ""
                    result_count = len(self.results)
                    self._run_save(view=view, side_by_side=side_by_side)
                    self.assertEqual(len(self.results), result_count + 1)
                output_dir = self._output_dir()
                self.assertEqual(len(list(output_dir.glob("*.png"))), 2)
                self.assertEqual(len(list(output_dir.glob("*.dat"))), 2)
        metadata = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in self._output_dir().glob("*.metadata.json")
        ]
        self.assertEqual(len(metadata), 2)
        self.assertEqual({tuple(item["plot"]["xlim"]) for item in metadata}, {(-1.5, 1.5)})
        self.assertEqual({tuple(item["plot"]["ylim"]) for item in metadata}, {(-0.75, 0.75)})
        raw_plot = next(item["plot"] for item in metadata if item["processing"]["derivative_order"] is None)
        second_plot = next(item["plot"] for item in metadata if item["processing"]["derivative_order"] == 2)
        self.assertEqual(second_plot["cmap"], "dptk_rdbu_r_p0p60")
        self.assertNotEqual(raw_plot["cmap"], second_plot["cmap"])

    def test_same_session_save_repairs_missing_second_dat(self) -> None:
        with patch(
            "core.export._save_heatmap_png",
            side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b"PNG"),
        ):
            first = self._run_save(view="raw")
            output_dir = self._output_dir()
            second_dat = next(path for path in output_dir.glob("*.dat") if "d2E" in path.stem)
            second_dat.unlink()
            second = self._run_save(view="raw")
        self.assertEqual(second["save_status"], "updated")
        self.assertEqual(len(list(output_dir.glob("*.png"))), 2)
        self.assertEqual(len(list(output_dir.glob("*.dat"))), 2)
        self.assertEqual(first["out_folder"], second["out_folder"])

    def test_view_change_does_not_create_new_products(self) -> None:
        with patch(
            "core.export._save_heatmap_png",
            side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b"PNG"),
        ):
            first = self._run_save(view="raw")
            output_dir = self._output_dir()
            first_names = sorted(path.name for path in output_dir.iterdir())
            self.window._drr_plot_view = "second"
            second = self._run_save(view="second")
            second_names = sorted(path.name for path in output_dir.iterdir())
        self.assertEqual(first["save_status"], "created")
        self.assertEqual(second["save_status"], "reused")
        self.assertEqual(first_names, second_names)

    def test_sg_snapshot_is_unchanged_by_later_ui_changes(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        original_derivative = __import__(
            "ui_qt.main_window", fromlist=["apply_sg_derivative_energy"]
        ).apply_sg_derivative_energy
        snapshot_window = int(self.window.drr_sg_window_spin.value())

        def blocked_derivative(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(3.0))
            return original_derivative(*args, **kwargs)

        with patch(
            "core.export._save_heatmap_png",
            side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b"PNG"),
        ), patch(
            "ui_qt.main_window.apply_sg_derivative_energy",
            side_effect=blocked_derivative,
        ):
            self.window._start_export("DRR")
            self.assertTrue(entered.wait(3.0))
            snapshot_window = int(self.window.drr_sg_window_spin.value())
            self.window.drr_sg_window_spin.setValue(max(5, snapshot_window - 2))
            release.set()
            self.assertTrue(self.window.thread_pool.waitForDone(5000))
            for _ in range(100):
                self.app.processEvents()
                if self.results or self.errors:
                    break
        self.assertFalse(self.errors, self.errors)
        self.assertEqual(len(self.results), 1)
        second_metadata = next(
            json.loads(path.read_text(encoding="utf-8"))
            for path in self._output_dir().glob("*.metadata.json")
            if json.loads(path.read_text(encoding="utf-8"))["processing"]["derivative_order"] == 2
        )
        self.assertEqual(second_metadata["processing"]["savgol_window"], snapshot_window)

    def test_unsupported_second_derivative_never_reports_success(self) -> None:
        self.window.loaded.cube = DataCube(
            energy=np.asarray([0.0, 1.0, 2.0, 3.0]),
            gate=np.asarray([0.0]),
            Z=np.asarray([[1.0, 2.0, 3.0, 4.0]]),
            gate_label="Gate",
            title="Too short",
            cbar_label="DR/R",
        )
        with patch(
            "core.export._save_heatmap_png",
            side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b"PNG"),
        ):
            self.window._drr_plot_view = "raw"
            self.window._start_export("DRR")
            self.assertTrue(self.window.thread_pool.waitForDone(5000))
            for _ in range(100):
                self.app.processEvents()
                if self.results or self.errors:
                    break
        self.assertFalse(self.results)
        self.assertTrue(self.errors)
        output_dir = self._output_dir()
        self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
