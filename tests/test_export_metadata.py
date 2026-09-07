from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import core.export as export_module
from core.export import export_drr_png_and_dat, export_pl_pngs_and_dat, write_export_metadata
from core.loader import DataCube
from core.plotting import HeatmapParams
from core.provenance import verify_initial_data_working_file


class ExportMetadataTests(unittest.TestCase):
    def test_mcd_center_change_reuses_map_and_hashes_source_once_per_save(self) -> None:
        def render(path, *_args, **_kwargs):
            Path(path).write_bytes(b"png")

        export_module._cached_file_sha256.cache_clear()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.export._save_heatmap_png", side_effect=render
        ), patch(
            "core.export._source_descriptor",
            wraps=export_module._source_descriptor,
        ) as describe_source:
            source = Path(tmp) / "mcd" / "sample.csv"
            source.parent.mkdir()
            source.write_text("raw MCD", encoding="utf-8")
            cube = DataCube(
                energy=np.asarray([1.6, 1.7]), gate=np.asarray([-1.0, 1.0]),
                Z=np.asarray([[-0.1, 0.1], [0.1, -0.1]]),
                gate_label="B field", title="MCD", cbar_label="MCD",
            )
            params = HeatmapParams(
                title="MCD", xlabel="Energy", ylabel="B field", cbar_label="MCD",
                vmin=-0.1, vmax=0.1, xlim=(1.6, 1.7), ylim=(-1.0, 1.0),
            )
            common = dict(
                cube=cube,
                params=params,
                export_base="sample_MCD_Combo",
                processed_name=str(Path("Processed Data") / "MCD" / "sample_MCD"),
                drr_style=False,
                metadata_input_files=(("measurement", str(source)),),
                metadata_processing={"map": "Combo", "mcd_settings": {"correction": "pair"}},
                reuse_existing_analysis=True,
            )
            first = export_drr_png_and_dat(tmp, **common)
            second = export_drr_png_and_dat(tmp, **common)

            self.assertEqual(first["png"], second["png"])
            self.assertEqual(first["dat"], second["dat"])
            self.assertEqual(second.save_status, "reused")
            self.assertEqual(len(list(first["dat"].parent.glob("*.dat"))), 1)
            self.assertEqual(len(list(first["png"].parent.glob("*.png"))), 1)
            self.assertEqual(describe_source.call_count, 2)
            cache = export_module._cached_file_sha256.cache_info()
            self.assertEqual(cache.misses, 1)
            self.assertGreaterEqual(cache.hits, 1)

    def test_pl_logical_outputs_share_collision_safe_stem(self) -> None:
        cube = DataCube(
            energy=np.asarray([0.0, 1.0]), gate=np.asarray([0.0]),
            Z=np.asarray([[1.0, 2.0]]), gate_label="Y", title="PL", cbar_label="PL",
        )
        params = HeatmapParams(
            title="PL", xlabel="X", ylabel="Y", cbar_label="PL",
            vmin=0.0, vmax=2.0, xlim=(0.0, 1.0), ylim=(0.0, 1.0),
        )
        with tempfile.TemporaryDirectory() as tmp, patch("core.export._save_heatmap_png"):
            first = export_pl_pngs_and_dat(
                tmp, "sample.csv", cube_linear=cube, cube_log=cube,
                params_linear=params, params_log=params,
            )
            second = export_pl_pngs_and_dat(
                tmp, "sample.csv", cube_linear=cube, cube_log=cube,
                params_linear=params, params_log=params,
            )
            self.assertEqual(first["dat"].stem, "sample_PL_linear")
            self.assertEqual(second["dat"].stem, "sample_PL_01_linear")
            self.assertEqual(second["png_linear"].stem, "sample_PL_01_linear")
            self.assertEqual(second["png_log"].stem, "sample_PL_01_log")
            payload = json.loads(second["dat"].with_suffix(".metadata.json").read_text())
            self.assertEqual(
                {item["filename"] for item in payload["output_manifest"]},
                {path.name for path in second.values()},
            )

    def test_metadata_contains_structured_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            folder = Path(tmp)
            source = folder / "working copy.csv"
            source.write_text("raw", encoding="utf-8")
            out = folder / "result.dat"
            write_export_metadata(
                str(folder), [out], operation="PL", input_files=(("measurement", source.name),)
            )
            payload = json.loads(out.with_suffix(".metadata.json").read_text(encoding="utf-8"))
            item = payload["sources"][0]
            self.assertEqual(payload["workflow"], "PL")
            self.assertEqual(item["filename"], source.name)
            self.assertEqual(item["processing_input_path"], str(source))
            self.assertEqual(len(item["sha256"]), 64)

    def test_export_does_not_move_source_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "working copy.csv"
            source.write_text("raw", encoding="utf-8")
            processed = folder / "Processed Data"
            processed.mkdir()
            cube = DataCube(
                energy=np.asarray([0.0, 1.0]),
                gate=np.asarray([0.0]),
                Z=np.asarray([[1.0, 2.0]]),
                gate_label="Y",
                title="PL",
                cbar_label="PL",
            )
            params = HeatmapParams(
                title="PL", xlabel="X", ylabel="Y", cbar_label="PL",
                vmin=0.0, vmax=2.0, xlim=(0.0, 1.0), ylim=(0.0, 1.0),
            )
            from ui_qt.main_window import ExportOptions, LoadedState, MainWindow

            loaded = LoadedState(
                mode="PL", folder=str(folder), primary_file=source.name,
                selected_files=[source.name], cube=cube,
            )
            options = ExportOptions(
                mode="PL", params=params,
                params_linear=params, params_log=params,
            )
            paths = {
                "png_linear": processed / "result_linear.png",
                "png_log": processed / "result_log.png",
                "dat": processed / "result.dat",
            }
            with patch("ui_qt.main_window.data_io.load_pl_cube", return_value=cube), \
                 patch("ui_qt.main_window.export_pl_pngs_and_dat", return_value=paths), \
                 patch("ui_qt.main_window.data_io.move_selected_to_archive") as move:
                result = MainWindow._export_task(
                    object(), loaded, options, progress=Mock(), log=Mock()
                )
            self.assertEqual(result["moved"], 0)
            self.assertFalse(result["auto_moved"])
            self.assertFalse(move.called)
            self.assertTrue(source.exists())
            self.assertFalse((folder / "Initial data after processing" / source.name).exists())

    def test_verified_pl_copy_is_cleaned_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            initial = folder / "Initial Data"
            initial.mkdir()
            source = folder / "sample.csv"
            source.write_text("raw", encoding="utf-8")
            (initial / source.name).write_text("raw", encoding="utf-8")
            record = verify_initial_data_working_file(source, folder, workflow="PL", role="measurement")
            cube = DataCube(
                energy=np.asarray([0.0, 1.0]), gate=np.asarray([0.0]),
                Z=np.asarray([[1.0, 2.0]]), gate_label="Y", title="PL", cbar_label="PL",
            )
            params = HeatmapParams(
                title="PL", xlabel="X", ylabel="Y", cbar_label="PL",
                vmin=0.0, vmax=2.0, xlim=(0.0, 1.0), ylim=(0.0, 1.0),
            )
            from ui_qt.main_window import ExportOptions, LoadedState, MainWindow
            loaded = LoadedState(
                mode="PL", folder=str(folder), primary_file=source.name,
                selected_files=[source.name], cube=cube, provenance_records=(record,),
            )
            paths = {"png_linear": folder / "Processed Data" / "a.png", "png_log": folder / "Processed Data" / "b.png", "dat": folder / "Processed Data" / "a.dat"}
            with patch("ui_qt.main_window.data_io.load_pl_cube", return_value=cube), \
                 patch("ui_qt.main_window.export_pl_pngs_and_dat", return_value=paths):
                result = MainWindow._export_task(
                    object(), loaded,
                    ExportOptions(mode="PL", params=params, params_linear=params, params_log=params, cleanup_verified_sources=True),
                    progress=Mock(), log=Mock(),
                )
            self.assertEqual(result["cleaned"], 1)
            self.assertFalse(source.exists())
            self.assertTrue((initial / source.name).exists())

    def test_drr_export_writes_reproducibility_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            measurement = folder / "measurement.csv"
            background = folder / "background.csv"
            measurement.write_text("measurement", encoding="utf-8")
            background.write_text("background", encoding="utf-8")
            params = HeatmapParams(
                title="DR/R",
                xlabel="Photon energy",
                ylabel="Gate",
                cbar_label="DR/R",
                vmin=-1.0,
                vmax=1.0,
                xlim=(0.0, 2.0),
                ylim=(0.0, 1.0),
                center_zero=True,
            )
            cube = DataCube(
                energy=np.asarray([0.0, 1.0, 2.0]),
                gate=np.asarray([0.0]),
                Z=np.asarray([[0.1, 0.2, 0.3]]),
                gate_label="Gate",
                title="DR/R",
                cbar_label="DR/R",
            )

            paths = export_drr_png_and_dat(
                str(folder),
                cube=cube,
                params=params,
                export_base="sample_avg2_DR_R_External",
                metadata_input_files=(
                    ("measurement", measurement.name),
                    ("background", background.name),
                ),
                metadata_processing={
                    "mode": "DR/R External",
                    "average_count": 2,
                    "baseline_which": "all",
                },
            )

            metadata_path = paths["dat"].with_suffix(".metadata.json")
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["operation"], "DR/R")
            self.assertEqual(payload["processing"]["average_count"], 2)
            self.assertEqual(
                [item["role"] for item in payload["inputs"]],
                ["measurement", "background"],
            )
            self.assertTrue(all(item["exists"] for item in payload["inputs"]))
            self.assertIn(paths["dat"].name, payload["outputs"])
            self.assertIn(paths["png"].name, payload["outputs"])

    def test_repeated_identical_drr_export_reuses_the_existing_result(self) -> None:
        def render(path, *_args, **_kwargs):
            Path(path).write_bytes(b"png")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.export._save_heatmap_png", side_effect=render
        ):
            source = Path(tmp) / "sample.csv"
            source.write_text("raw", encoding="utf-8")
            cube = DataCube(
                energy=np.asarray([1.0, 2.0]), gate=np.asarray([0.0]),
                Z=np.asarray([[0.1, 0.2]]), gate_label="Gate", title="DR/R", cbar_label="DR/R",
            )
            params = HeatmapParams(
                title="DR/R", xlabel="Energy", ylabel="Gate", cbar_label="DR/R",
                vmin=-1.0, vmax=1.0, xlim=(1.0, 2.0), ylim=(0.0, 1.0),
            )
            output_folder = str(Path("Processed Data") / "DRR")

            first = export_drr_png_and_dat(
                tmp, cube=cube, params=params, export_base="sample", processed_name=output_folder,
                metadata_input_files=(("measurement", source.name),),
                metadata_processing={"mode": "Self"},
            )
            second = export_drr_png_and_dat(
                tmp, cube=cube, params=params, export_base="sample", processed_name=output_folder,
                metadata_input_files=(("measurement", source.name),),
                metadata_processing={"mode": "Self"},
            )

            self.assertEqual(first["dat"].parent, Path(tmp) / "Processed Data" / "DRR")
            self.assertEqual(first["dat"].stem, "sample")
            self.assertEqual(second["dat"], first["dat"])
            self.assertEqual(second["png"], first["png"])
            self.assertEqual(second.save_status, "reused")
            self.assertEqual(len(list(first["dat"].parent.glob("*.dat"))), 1)
            self.assertEqual(len(list(first["dat"].parent.glob("*.png"))), 1)

    def test_drr_plot_change_updates_png_without_duplicate_dat(self) -> None:
        def render(path, *_args, **_kwargs):
            Path(path).write_bytes(b"png")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.export._save_heatmap_png", side_effect=render
        ):
            source = Path(tmp) / "sample.csv"
            source.write_text("raw", encoding="utf-8")
            cube = DataCube(
                energy=np.asarray([1.0, 2.0]), gate=np.asarray([0.0]),
                Z=np.asarray([[0.1, 0.2]]), gate_label="Gate", title="DR/R", cbar_label="DR/R",
            )
            first_params = HeatmapParams(
                title="DR/R", xlabel="Energy", ylabel="Gate", cbar_label="DR/R",
                vmin=-1.0, vmax=1.0, xlim=(1.0, 2.0), ylim=(0.0, 1.0),
            )
            changed_params = HeatmapParams(
                **{**first_params.__dict__, "vmin": -0.5, "vmax": 0.5}
            )
            kwargs = {
                "cube": cube,
                "export_base": "sample",
                "metadata_input_files": (("measurement", source.name),),
                "metadata_processing": {"mode": "Self"},
            }

            first = export_drr_png_and_dat(tmp, params=first_params, **kwargs)
            second = export_drr_png_and_dat(tmp, params=changed_params, **kwargs)

            self.assertEqual(second.save_status, "updated")
            self.assertEqual(second["dat"], first["dat"])
            self.assertEqual(len(list(second["dat"].parent.glob("*.dat"))), 1)
            payload = json.loads(second["dat"].with_suffix(".metadata.json").read_text())
            self.assertEqual(payload["plot"]["vmin"], -0.5)
            self.assertEqual(payload["plot"]["vmax"], 0.5)

    def test_drr_repeat_recognizes_metadata_written_before_fingerprints(self) -> None:
        def render(path, *_args, **_kwargs):
            Path(path).write_bytes(b"png")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.export._save_heatmap_png", side_effect=render
        ):
            source = Path(tmp) / "sample.csv"
            source.write_text("raw", encoding="utf-8")
            cube = DataCube(
                energy=np.asarray([1.0, 2.0]), gate=np.asarray([0.0]),
                Z=np.asarray([[0.1, 0.2]]), gate_label="Gate", title="DR/R", cbar_label="DR/R",
            )
            params = HeatmapParams(
                title="DR/R", xlabel="Energy", ylabel="Gate", cbar_label="DR/R",
                vmin=-1.0, vmax=1.0, xlim=(1.0, 2.0), ylim=(0.0, 1.0),
            )
            kwargs = {
                "cube": cube,
                "params": params,
                "export_base": "sample",
                "metadata_input_files": (("measurement", source.name),),
                "metadata_processing": {"mode": "Self"},
            }
            first = export_drr_png_and_dat(tmp, **kwargs)
            metadata_path = first["dat"].with_suffix(".metadata.json")
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload.pop("analysis_fingerprint")
            payload.pop("plot_fingerprint")
            metadata_path.write_text(json.dumps(payload), encoding="utf-8")

            second = export_drr_png_and_dat(tmp, **kwargs)

            self.assertEqual(second.save_status, "reused")
            self.assertEqual(second["dat"], first["dat"])

    def test_changed_drr_processing_creates_a_distinct_result(self) -> None:
        def render(path, *_args, **_kwargs):
            Path(path).write_bytes(b"png")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.export._save_heatmap_png", side_effect=render
        ):
            source = Path(tmp) / "sample.csv"
            source.write_text("raw", encoding="utf-8")
            cube = DataCube(
                energy=np.asarray([1.0, 2.0]), gate=np.asarray([0.0]),
                Z=np.asarray([[0.1, 0.2]]), gate_label="Gate", title="DR/R", cbar_label="DR/R",
            )
            params = HeatmapParams(
                title="DR/R", xlabel="Energy", ylabel="Gate", cbar_label="DR/R",
                vmin=-1.0, vmax=1.0, xlim=(1.0, 2.0), ylim=(0.0, 1.0),
            )
            common = {
                "cube": cube,
                "params": params,
                "export_base": "sample",
                "metadata_input_files": (("measurement", source.name),),
            }

            first = export_drr_png_and_dat(
                tmp, metadata_processing={"mode": "Self", "derivative_order": None}, **common
            )
            second = export_drr_png_and_dat(
                tmp, metadata_processing={"mode": "Self", "derivative_order": 1}, **common
            )

            self.assertEqual(first["dat"].stem, "sample")
            self.assertEqual(second["dat"].stem, "sample_01")
            self.assertEqual(second.save_status, "created")

    def test_changed_drr_source_contents_create_a_distinct_result(self) -> None:
        def render(path, *_args, **_kwargs):
            Path(path).write_bytes(b"png")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.export._save_heatmap_png", side_effect=render
        ):
            source = Path(tmp) / "sample.csv"
            source.write_text("first acquisition", encoding="utf-8")
            cube = DataCube(
                energy=np.asarray([1.0, 2.0]), gate=np.asarray([0.0]),
                Z=np.asarray([[0.1, 0.2]]), gate_label="Gate", title="DR/R", cbar_label="DR/R",
            )
            params = HeatmapParams(
                title="DR/R", xlabel="Energy", ylabel="Gate", cbar_label="DR/R",
                vmin=-1.0, vmax=1.0, xlim=(1.0, 2.0), ylim=(0.0, 1.0),
            )
            kwargs = {
                "cube": cube,
                "params": params,
                "export_base": "sample",
                "metadata_input_files": (("measurement", source.name),),
                "metadata_processing": {"mode": "Self"},
            }

            first = export_drr_png_and_dat(tmp, **kwargs)
            source.write_text("second acquisition", encoding="utf-8")
            second = export_drr_png_and_dat(tmp, **kwargs)

            self.assertEqual(first["dat"].stem, "sample")
            self.assertEqual(second["dat"].stem, "sample_01")
            self.assertEqual(second.save_status, "created")


if __name__ == "__main__":
    unittest.main()
