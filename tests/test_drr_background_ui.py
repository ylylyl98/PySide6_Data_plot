from __future__ import annotations

import os
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.drr_sources import (
    DrrBackgroundResolution,
    DrrMeasurementAssignment,
    discover_drr_sources,
    portable_source_name,
    resolve_drr_background_assignments,
)
from core import data_io
from core.loader import DataCube
from core.plotting import HeatmapParams
from ui_qt.common import ExportOptions, LoadOptions, LoadedState
from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class DrrBackgroundUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self) -> None:
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    @staticmethod
    def _cube() -> DataCube:
        return DataCube(
            energy=np.array([0.0, 1.0, 2.0]),
            gate=np.array([0.0]),
            Z=np.array([[1.0, 2.0, 3.0]]),
            gate_label="Gate",
            title="DRR",
            cbar_label="DR/R",
        )

    def test_frame_selection_clears_stale_per_measurement_recipe(self) -> None:
        stale = DrrMeasurementAssignment(
            "measurement.csv", baseline_mode="External", baseline_files=("old-bg.csv",),
            baseline_which="last",
        )
        self.window.drr_selected_files = ["measurement.csv"]
        self.window.drr_baseline_files_manual = ["old-bg.csv"]
        self.window._drr_assignments = (stale,)
        self.window._drr_assignments_automatic = False
        self.window.drr_baseline_combo.setCurrentText("External")

        self.window.drr_baseline_combine_combo.setCurrentText(
            "First frame from each file, then average"
        )

        self.assertEqual(self.window._drr_assignments, ())
        self.assertTrue(self.window.drr_baseline_combine_combo.isEnabled())

    def test_automatic_backgrounds_disable_common_pin_and_show_mapping(self) -> None:
        assignments = (
            DrrMeasurementAssignment(
                "A.csv", baseline_mode="External", baseline_files=("BG1.csv",),
                baseline_which="last",
            ),
            DrrMeasurementAssignment(
                "B.csv", baseline_mode="External", baseline_files=("BG2.csv",),
                baseline_which="first",
            ),
        )
        result = DrrBackgroundResolution(assignments=assignments, numerical_path="heterogeneous")
        self.window.current_folder = ""
        self.window.drr_selected_files = ["A.csv", "B.csv"]
        with patch(
            "ui_qt.controllers_drr.resolve_drr_background_assignments",
            return_value=result,
        ):
            self.assertTrue(self.window.drr_controller._guess_drr_background_for_selection())

        self.assertFalse(self.window.drr_pin_baseline_chk.isEnabled())
        self.assertFalse(self.window.drr_baseline_combine_combo.isEnabled())
        self.assertIn("A.csv", self.window.drr_baseline_summary.toolTip())
        self.assertIn("BG2.csv", self.window.drr_baseline_summary.toolTip())

    def test_reload_of_legacy_self_state_materializes_assignments(self) -> None:
        self.window.current_folder = str(Path.cwd())
        self.window.drr_selected_files = ["measurement.csv"]
        self.window.loaded = LoadedState(
            mode="DRR", folder=self.window.current_folder,
            primary_file="measurement.csv", selected_files=["measurement.csv"],
            cube=self._cube(), drr_baseline_text="Self (last frame)",
        )
        assignment = DrrMeasurementAssignment(
            "measurement.csv", baseline_mode="Self (last frame)", baseline_which="last",
        )
        resolved = DrrBackgroundResolution(assignments=(assignment,), numerical_path="common")
        replacement = self._cube()
        with (
            patch("ui_qt.main_window.resolve_drr_background_assignments", return_value=resolved) as resolve,
            patch("ui_qt.main_window.data_io.load_drr_resolved_cube", return_value=replacement) as load,
            patch("ui_qt.controllers_drr.DrrController._drr_missing_sources", return_value=[]),
            patch.object(self.window.thread_pool, "start") as start,
        ):
            self.assertFalse(self.window._ensure_loaded_matches_drr_params())
            self.assertEqual(start.call_count, 1)
            options = start.call_args.args[0].args[0]
            refreshed = self.window._load_task(
                options,
                progress=SimpleNamespace(emit=lambda *_args: None),
                log=SimpleNamespace(emit=lambda *_args: None),
            )
            token = self.window._active_load_token
            self.window._on_loaded(refreshed, request_token=token)
            self.window._on_load_finished(request_token=token)

        self.assertGreaterEqual(resolve.call_count, 1)
        load.assert_called_once()
        self.assertEqual(tuple(self.window.loaded.drr_assignments), (assignment,))
        self.assertEqual(
            self.window.loaded.drr_background_selection["drr_background_assignments"],
            [assignment.to_dict()],
        )

    def test_export_uses_current_assignments_over_stale_metadata(self) -> None:
        assignment = DrrMeasurementAssignment(
            "measurement.csv", baseline_mode="External", baseline_files=("new-bg.csv",),
            baseline_which="first",
        )
        loaded = LoadedState(
            mode="DRR", folder=str(Path.cwd()), primary_file="measurement.csv",
            selected_files=["measurement.csv"], baseline_files=["new-bg.csv"],
            cube=self._cube(), drr_mode_label="DR/R External",
            drr_baseline_text="External", drr_baseline_which="first",
            drr_assignments=(assignment,),
            drr_background_selection={
                "selection_method": "old", "numerical_path": "old",
                "drr_background_assignments": [{"measurement": "measurement.csv", "baseline_files": ["old-bg.csv"]}],
            },
        )
        # Derivative/export views are freshly allocated DataCubes and may not
        # carry the numerical path attribute from the loaded cube.
        loaded.cube.drr_numerical_path = "heterogeneous"
        class FakePaths(dict):
            save_status = "created"

        paths = FakePaths(
            png=Path.cwd() / "result.png",
            dat=Path.cwd() / "result.dat",
        )
        captured: dict[str, object] = {}

        def fake_export(*_args, **kwargs):
            captured.update(kwargs)
            return paths

        class FakePairPaths(dict):
            save_status = "created"

        pair_paths = FakePairPaths(
            raw_png=Path.cwd() / "result_DR_R.png",
            raw_dat=Path.cwd() / "result_DR_R.dat",
            second_png=Path.cwd() / "result_d2E.png",
            second_dat=Path.cwd() / "result_d2E.dat",
        )
        with patch(
            "ui_qt.main_window.export_drr_pair_pngs_and_dat",
            side_effect=lambda *_args, **kwargs: captured.update(kwargs) or pair_paths,
        ):
            MainWindow._export_task(
                self.window,
                loaded,
                ExportOptions(
                    mode="DRR", params=HeatmapParams(
                        "DRR", "Energy", "Gate", "DR/R", -1.0, 1.0,
                        (0.0, 2.0), (0.0, 0.0),
                    ),
                    drr_cube=self._cube(), drr_raw_cube=self._cube(),
                    drr_second_cube=self._cube(),
                    drr_raw_params=HeatmapParams(
                        "DRR", "Energy", "Gate", "DR/R", -1.0, 1.0,
                        (0.0, 2.0), (0.0, 0.0),
                    ),
                    drr_second_params=HeatmapParams(
                        "DRR", "Energy", "Gate", "d2(DR/R)/dE2", -1.0, 1.0,
                        (0.0, 2.0), (0.0, 0.0),
                    ),
                ),
                progress=SimpleNamespace(emit=lambda *_args: None),
                log=SimpleNamespace(emit=lambda *_args: None),
            )

        processing = captured["metadata_processing_raw"]
        self.assertEqual(processing["drr_background_assignments"], [assignment.to_dict()])
        self.assertEqual(processing["selection_method"], "explicit_per_measurement")
        self.assertEqual(processing["numerical_path"], "heterogeneous")
        self.assertIn("without extrapolation", processing["average_method"])

    @staticmethod
    def _write_csv(path: Path, rows: list[tuple[float, float]]) -> None:
        path.write_text(
            "Vbg,Vtg," + ",".join(str(700 + index) for index in range(10)) + "\n"
            + "\n".join(
                f"{gate},0," + ",".join(str(value) for _ in range(10))
                for gate, value in rows
            ) + "\n",
            encoding="utf-8",
        )

    def test_real_external_frame_change_changes_loaded_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement, background = root / "measurement.csv", root / "background.csv"
            self._write_csv(measurement, [(0.0, 20.0)])
            self._write_csv(background, [(0.0, 10.0), (0.0, 20.0)])

            self.window.current_folder = str(root)
            self.window.drr_selected_files = [measurement.name]
            self.window.drr_baseline_files_manual = [background.name]
            self.window.drr_baseline_combo.setCurrentText("External")
            self.window.drr_baseline_combine_combo.setCurrentText(
                "Last frame from each file, then average"
            )
            initial = DrrMeasurementAssignment(
                measurement.name, "External", (background.name,), "last"
            )
            with patch.object(self.window, "_provenance_records_for_load", return_value=()):
                self.window.loaded = self.window._load_task(
                    LoadOptions(
                        mode="DRR", folder=str(root), selected_files=[measurement.name],
                        baseline_files=[background.name], pl_log_scale=False,
                        drr_baseline_text="External", drr_baseline_which="last",
                        compare_log_scale=False, drr_assignments=(initial,),
                    ),
                    progress=SimpleNamespace(emit=lambda *_args: None),
                    log=SimpleNamespace(emit=lambda *_args: None),
                )
            last = self.window.loaded.cube
            self.assertTrue(np.allclose(last.Z, 0.0))
            self.window._drr_assignments = (initial,)
            self.window._drr_assignments_automatic = False

            self.window.drr_baseline_combine_combo.setCurrentText(
                "First frame from each file, then average"
            )
            self.assertEqual(self.window._drr_assignments, ())
            captured = []
            with patch.object(self.window.thread_pool, "start", captured.append):
                self.assertFalse(self.window._ensure_loaded_matches_drr_params())
            self.assertEqual(len(captured), 1)
            options = captured[0].args[0]
            refreshed = self.window._load_task(
                options,
                progress=SimpleNamespace(emit=lambda *_args: None),
                log=SimpleNamespace(emit=lambda *_args: None),
            )
            token = self.window._active_load_token
            self.window._on_loaded(refreshed, request_token=token)
            self.window._on_load_finished(request_token=token)
            first = self.window.loaded.cube

            self.assertTrue(np.allclose(first.Z, 1.0))

    def test_default_self_load_and_reload_plot_without_external_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement = root / "measurement.csv"
            self._write_csv(measurement, [(0.0, 20.0), (1.0, 40.0)])
            w = self.window
            blocked = w.drr_baseline_combo.blockSignals(True)
            w.drr_baseline_combo.setCurrentText("Self (last frame)")
            w.drr_baseline_combo.blockSignals(blocked)
            w.current_folder = str(root)
            w.drr_selected_files = [measurement.name]
            w.drr_available_sources = discover_drr_sources(root)
            w.tabs.setCurrentIndex(next(i for i in range(w.tabs.count()) if w.tabs.tabText(i) == "DRR"))
            self.assertEqual(w.drr_baseline_combo.currentText(), "Self (last frame)")
            self.assertFalse(w._drr_baseline_user_selected)
            for _ in range(2):
                captured = []
                with patch.object(w.thread_pool, "start", captured.append):
                    w._start_load("DRR")
                self.assertTrue(captured, w.statusBar().currentMessage())
                options = captured[0].args[0]
                loaded = w._load_task(
                    options, progress=SimpleNamespace(emit=lambda *_: None),
                    log=SimpleNamespace(emit=lambda *_: None),
                )
                np.testing.assert_allclose(loaded.cube.Z, [[-0.5] * 10, [0.0] * 10])
                self.assertFalse(w._drr_assignments_automatic)
                token = w._active_load_token
                w._on_loaded(loaded, request_token=token)
                w._on_load_finished(request_token=token)
                self.assertEqual(w.last_plotted_mode, "DRR")
                self.assertTrue(w._drr_heatmap_axes)

    def test_default_self_does_not_replace_missing_saved_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement = root / "measurement.csv"
            self._write_csv(measurement, [(0.0, 20.0), (1.0, 40.0)])
            processed = root / "Processed Data" / "DRR"
            processed.mkdir(parents=True)
            (processed / "saved.metadata.json").write_text(json.dumps({
                "operation": "DR/R",
                "sources": [{"role": "measurement", "source_path": str(measurement)}],
                "processing": {"drr_background_assignments": [{
                    "measurement": measurement.name, "baseline_mode": "External",
                    "baseline_files": ["missing_back.csv"], "baseline_which": "last",
                }]},
            }), encoding="utf-8")
            w = self.window
            w.current_folder = str(root)
            w.drr_selected_files = [measurement.name]
            w.drr_available_sources = discover_drr_sources(root)
            captured = []
            with patch.object(w.thread_pool, "start", captured.append):
                w._start_load("DRR")
            self.assertFalse(captured)
            self.assertIn("missing", w.statusBar().currentMessage().lower())

    def test_returning_to_self_after_empty_external_selection_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement = root / "measurement.csv"
            self._write_csv(measurement, [(0.0, 20.0), (1.0, 40.0)])
            w = self.window
            blocked = w.drr_baseline_combo.blockSignals(True)
            w.drr_baseline_combo.setCurrentText("Self (last frame)")
            w.drr_baseline_combo.blockSignals(blocked)
            w.current_folder = str(root)
            w.drr_selected_files = [measurement.name]
            w.drr_available_sources = discover_drr_sources(root)
            w.loaded = LoadedState(
                mode="DRR", folder=str(root), primary_file=measurement.name,
                selected_files=[measurement.name], cube=self._cube(),
                drr_baseline_text="Self (last frame)",
            )

            w.drr_baseline_combo.setCurrentText("External")
            self.assertIsNone(w.loaded)

            with patch.object(MainWindow, "_start_load") as start_load:
                for baseline in ("Self (first frame)", "Self (last frame)"):
                    w.drr_baseline_combo.setCurrentText(baseline)
                    start_load.assert_called_once_with("DRR")
                    start_load.reset_mock()

    def test_empty_external_selection_invalidates_active_drr_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement = root / "measurement.csv"
            self._write_csv(measurement, [(0.0, 20.0), (1.0, 40.0)])
            w = self.window
            w.current_folder = str(root)
            w.drr_selected_files = [measurement.name]
            w.drr_available_sources = discover_drr_sources(root)
            captured = []
            with patch.object(w.thread_pool, "start", captured.append):
                blocked = w.drr_baseline_combo.blockSignals(True)
                w.drr_baseline_combo.setCurrentText("Self (last frame)")
                w.drr_baseline_combo.blockSignals(blocked)
                w._start_load("DRR")
                token = w._active_load_token
                w._pending_load_mode = "DRR"
                w._pending_load_options = SimpleNamespace()
                stale = LoadedState(
                    mode="DRR", folder=str(root), primary_file=measurement.name,
                    selected_files=[measurement.name], cube=self._cube(),
                    drr_baseline_text="Self (last frame)",
                )
                w.drr_baseline_combo.setCurrentText("External")
                w._on_loaded(stale, request_token=token)

            self.assertIsNone(w.loaded)
            self.assertIn("DRR", w._invalidated_load_modes)
            self.assertIsNone(w._pending_load_mode)
            self.assertIsNone(w._pending_load_options)
            w._on_load_finished(request_token=token)
            self.assertFalse(w._load_in_progress)

    def test_returning_to_self_does_not_reload_when_suspended_or_unselected(self) -> None:
        w = self.window
        w.drr_baseline_combo.setCurrentText("External")
        with patch.object(MainWindow, "_start_load") as start_load:
            w.drr_baseline_combo.setCurrentText("Self (last frame)")
            start_load.assert_not_called()

            w.drr_selected_files = ["measurement.csv"]
            w.current_folder = str(Path.cwd())
            w._suspend_drr_autoplot = False
            next_index = (w.drr_cmap.currentIndex() + 1) % max(1, w.drr_cmap.count())
            w.drr_cmap.setCurrentIndex(next_index)
            start_load.assert_not_called()

            w._suspend_drr_autoplot = True
            w.drr_baseline_combo.setCurrentText("External")
            w.drr_baseline_combo.setCurrentText("Self (last frame)")
            start_load.assert_not_called()

    def test_real_external_to_self_load_uses_self_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement, background = root / "measurement.csv", root / "background.csv"
            self._write_csv(measurement, [(0.0, 20.0), (1.0, 40.0)])
            self._write_csv(background, [(0.0, 10.0)])

            self.window.current_folder = str(root)
            self.window.drr_selected_files = [measurement.name]
            self.window.drr_baseline_files_manual = [background.name]
            self.window.drr_baseline_combo.setCurrentText("External")
            self.window.drr_baseline_combine_combo.setCurrentText(
                "Last frame from each file, then average"
            )
            external_assignment = DrrMeasurementAssignment(
                measurement.name, "External", (background.name,), "last"
            )
            with patch.object(self.window, "_provenance_records_for_load", return_value=()):
                self.window.loaded = self.window._load_task(
                    LoadOptions(
                        mode="DRR", folder=str(root), selected_files=[measurement.name],
                        baseline_files=[background.name], pl_log_scale=False,
                        drr_baseline_text="External", drr_baseline_which="last",
                        compare_log_scale=False, drr_assignments=(external_assignment,),
                    ),
                    progress=SimpleNamespace(emit=lambda *_args: None),
                    log=SimpleNamespace(emit=lambda *_args: None),
                )
            self.window._drr_assignments = (external_assignment,)
            self.window._drr_assignments_automatic = False
            self.window.drr_baseline_combo.setCurrentText("Self (last frame)")
            self.assertEqual(self.window._drr_assignments, ())
            captured = []
            with patch.object(self.window.thread_pool, "start", captured.append):
                self.assertFalse(self.window._ensure_loaded_matches_drr_params())
            self.assertEqual(len(captured), 1)
            options = captured[0].args[0]
            refreshed = self.window._load_task(
                options,
                progress=SimpleNamespace(emit=lambda *_args: None),
                log=SimpleNamespace(emit=lambda *_args: None),
            )
            token = self.window._active_load_token
            self.window._on_loaded(refreshed, request_token=token)
            self.window._on_load_finished(request_token=token)
            self_mode = self.window.loaded.cube
            external = data_io.load_drr_resolved_cube(
                root, [measurement.name], (external_assignment,)
            )

            self.assertTrue(np.allclose(external.Z, [[1.0] * 10, [3.0] * 10]))
            self.assertTrue(np.allclose(self_mode.Z, [[-0.5] * 10, [0.0] * 10]))

    def test_real_selection_load_reload_and_export_preserve_distinct_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            processed = root / "Processed Data" / "DRR"
            initial.mkdir()
            processed.mkdir(parents=True)
            names = {
                key: initial / f"{key}_760nmc_REF.csv"
                for key in ("A", "B", "BG1_back", "BG2_back")
            }
            self._write_csv(names["A"], [(0.0, 20.0), (1.0, 20.0)])
            self._write_csv(names["B"], [(0.0, 40.0), (1.0, 40.0)])
            self._write_csv(names["BG1_back"], [(0.0, 10.0)])
            self._write_csv(names["BG2_back"], [(0.0, 40.0)])
            a_name = portable_source_name(root, names["A"])
            b_name = portable_source_name(root, names["B"])
            bg1_name = portable_source_name(root, names["BG1_back"])
            bg2_name = portable_source_name(root, names["BG2_back"])
            assignments_json = [
                {"measurement": a_name, "baseline_mode": "External", "baseline_files": [bg1_name], "baseline_which": "last"},
                {"measurement": b_name, "baseline_mode": "External", "baseline_files": [bg2_name], "baseline_which": "last"},
            ]
            (processed / "saved.metadata.json").write_text(json.dumps({
                "operation": "DR/R",
                "sources": [
                    {"role": "measurement", "source_path": str(names["A"])},
                    {"role": "measurement", "source_path": str(names["B"])},
                    {"role": "background", "source_path": str(names["BG1_back"])},
                    {"role": "background", "source_path": str(names["BG2_back"])},
                ],
                "processing": {"drr_background_assignments": assignments_json},
            }), encoding="utf-8")

            selected = [a_name, b_name]
            self.window.current_folder = str(root)
            self.window.drr_selected_files = selected
            self.window.drr_available_sources = discover_drr_sources(root)
            captured: dict[str, object] = {}
            with patch.object(self.window.thread_pool, "start", lambda worker: captured.setdefault("worker", worker)):
                self.window._start_load("DRR")
            options = captured["worker"].args[0]
            self.assertEqual(
                [item.baseline_files for item in options.drr_assignments],
                [(bg1_name,), (bg2_name,)],
            )
            loaded = self.window._load_task(
                options,
                progress=SimpleNamespace(emit=lambda *_args: None),
                log=SimpleNamespace(emit=lambda *_args: None),
            )
            self.assertEqual(loaded.cube.drr_numerical_path, "heterogeneous")
            self.assertTrue(np.allclose(loaded.cube.Z, 0.5))
            self.window._on_loaded(loaded)
            self.window._on_load_finished(request_token=self.window._active_load_token)

            # Force a real reload through the current UI selection state and
            # verify controller and LoadedState assignment identity stay equal.
            self.window.drr_selected_files = [b_name, a_name]
            self.window._drr_assignments = tuple(reversed(loaded.drr_assignments))
            self.window._drr_assignments_automatic = True
            captured = []
            with patch.object(self.window.thread_pool, "start", captured.append):
                self.assertFalse(self.window._ensure_loaded_matches_drr_params())
            self.assertEqual(len(captured), 1)
            options = captured[0].args[0]
            refreshed = self.window._load_task(
                options,
                progress=SimpleNamespace(emit=lambda *_args: None),
                log=SimpleNamespace(emit=lambda *_args: None),
            )
            token = self.window._active_load_token
            self.window._on_loaded(refreshed, request_token=token)
            self.window._on_load_finished(request_token=token)
            self.assertEqual(tuple(self.window._drr_assignments), tuple(self.window.loaded.drr_assignments))
            self.assertEqual(self.window.loaded.drr_background_selection["numerical_path"], "heterogeneous")

            MainWindow._export_task(
                self.window, self.window.loaded,
                ExportOptions(
                    mode="DRR",
                    params=HeatmapParams(
                        "DRR", "Energy", "Gate", "DR/R", -1.0, 1.0,
                        (700.0, 709.0), (0.0, 1.0),
                    ),
                    drr_cube=self.window.loaded.cube,
                    drr_raw_cube=self.window.loaded.cube,
                    drr_raw_params=HeatmapParams(
                        "DRR", "Energy", "Gate", "DR/R", -1.0, 1.0,
                        (700.0, 709.0), (0.0, 1.0),
                    ),
                    drr_second_cube=self.window.loaded.cube,
                ),
                progress=SimpleNamespace(emit=lambda *_args: None),
                log=SimpleNamespace(emit=lambda *_args: None),
            )
            metadata_files = list((root / "Processed Data" / "DRR").glob("*.metadata.json"))
            self.assertTrue(metadata_files)
            saved_path = max(metadata_files, key=lambda path: path.stat().st_mtime_ns)
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            saved_processing = saved["processing"]
            self.assertEqual(
                [item["baseline_files"] for item in saved_processing["drr_background_assignments"]],
                [[bg2_name], [bg1_name]],
            )
            restored = resolve_drr_background_assignments(root, discover_drr_sources(root), [b_name, a_name])
            self.assertTrue(restored.resolved)
            self.assertEqual(
                [item.baseline_files for item in restored.assignments],
                [(bg2_name,), (bg1_name,)],
            )
            self.assertIn("without extrapolation", saved_processing["average_method"])

    def test_precomputed_map_export_omits_background_assignments(self) -> None:
        loaded = LoadedState(
            mode="DRR", folder=str(Path.cwd()), primary_file="map.xlsx",
            selected_files=["map.xlsx"], cube=self._cube(), drr_mode_label="DR/R Map",
            drr_baseline_text="None", drr_baseline_which="last",
        )
        captured: dict[str, object] = {}
        class FakePaths(dict):
            save_status = "created"
        fake_paths = FakePaths(png=Path.cwd() / "map.png", dat=Path.cwd() / "map.dat")
        class FakePairPaths(dict):
            save_status = "created"

        pair_paths = FakePairPaths(
            raw_png=Path.cwd() / "map_DR_R.png",
            raw_dat=Path.cwd() / "map_DR_R.dat",
            second_png=Path.cwd() / "map_d2E.png",
            second_dat=Path.cwd() / "map_d2E.dat",
        )
        with patch(
            "ui_qt.main_window.export_drr_pair_pngs_and_dat",
            side_effect=lambda *_args, **kwargs: captured.update(kwargs) or pair_paths,
        ):
            MainWindow._export_task(
                self.window,
                loaded,
                ExportOptions(
                    mode="DRR", params=HeatmapParams(
                        "DRR", "Energy", "Gate", "DR/R", -1.0, 1.0,
                        (0.0, 2.0), (0.0, 0.0),
                    ),
                    drr_cube=self._cube(), drr_raw_cube=self._cube(),
                    drr_second_cube=self._cube(),
                    drr_raw_params=HeatmapParams(
                        "DRR", "Energy", "Gate", "DR/R", -1.0, 1.0,
                        (0.0, 2.0), (0.0, 0.0),
                    ),
                    drr_second_params=HeatmapParams(
                        "DRR", "Energy", "Gate", "d2(DR/R)/dE2", -1.0, 1.0,
                        (0.0, 2.0), (0.0, 0.0),
                    ),
                ),
                progress=SimpleNamespace(emit=lambda *_args: None),
                log=SimpleNamespace(emit=lambda *_args: None),
            )
        self.assertNotIn("drr_background_assignments", captured["metadata_processing_raw"])

    def test_unresolved_automatic_reload_clears_old_view_and_never_uses_union(self) -> None:
        self.window.current_folder = str(Path.cwd())
        self.window.drr_selected_files = ["A.csv", "B.csv"]
        self.window.drr_baseline_files_manual = ["old-union.csv"]
        self.window.drr_baseline_combo.setCurrentText("External")
        self.window.loaded = LoadedState(
            mode="DRR", folder=self.window.current_folder,
            primary_file="A.csv", selected_files=["A.csv", "B.csv"],
            baseline_files=["old-union.csv"], cube=self._cube(),
            drr_baseline_text="Automatic", drr_assignments=(),
        )
        self.window._drr_assignments = ()
        self.window._drr_assignments_automatic = True
        unresolved = DrrBackgroundResolution(
            reason="recorded baseline missing", unresolved_measurements=("A.csv", "B.csv")
        )
        with (
            patch("ui_qt.main_window.resolve_drr_background_assignments", return_value=unresolved) as resolve,
            patch.object(self.window.thread_pool, "start") as start,
        ):
            self.window._start_load("DRR")

        resolve.assert_called_once()
        start.assert_not_called()
        self.assertIsNone(self.window.loaded)
        self.assertTrue(self.window._drr_assignments_automatic)
        self.assertEqual(self.window.drr_baseline_files_manual, [])

    def test_export_stops_when_reload_invalidates_loaded_drr_view(self) -> None:
        """A missing source discovered during export preparation must not crash."""
        self.window.current_folder = str(Path.cwd())
        self.window.drr_selected_files = ["missing-measurement.csv"]
        self.window.loaded = LoadedState(
            mode="DRR", folder=self.window.current_folder,
            primary_file="map.xlsx", selected_files=["map.xlsx"], cube=self._cube(),
            drr_mode_label="DR/R Map", drr_baseline_text="None",
        )
        self.window.last_plotted_mode = "DRR"
        with patch.object(self.window, "_show_error") as show_error:
            self.window._start_export("DRR")
        show_error.assert_not_called()
        self.assertIsNone(self.window.loaded)
        self.assertFalse(self.window._export_in_progress)


if __name__ == "__main__":
    unittest.main()
