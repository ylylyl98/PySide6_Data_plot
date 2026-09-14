from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.loader import DataCube
from core.plotting import HeatmapParams, SplitColorScale
from core.processing import apply_sg_derivative_energy
from core.export import export_drr_pair_pngs_and_dat
from ui_qt.common import ExportOptions, LoadedState
from ui_qt.main_window import MainWindow


def _cube(values: np.ndarray, title: str) -> DataCube:
    return DataCube(
        energy=np.linspace(-2.0, 2.0, values.shape[1]),
        gate=np.linspace(-1.0, 1.0, values.shape[0]),
        Z=values,
        gate_label="Gate",
        title=title,
        cbar_label="DR/R" if "raw" in title else "d2(DR/R)/dE2",
    )


def _params(cube: DataCube, vmin: float, vmax: float) -> HeatmapParams:
    return HeatmapParams(
        title=cube.title, xlabel="Photon Energy (eV)", ylabel="Gate",
        cbar_label=cube.cbar_label, vmin=vmin, vmax=vmax,
        xlim=(-2.0, 2.0), ylim=(-1.0, 1.0), cmap="RdBu_r",
        log_scale=False, y_axis_log=False, center_zero=True, clip_outliers=False,
    )


class DrrDualExportTests(unittest.TestCase):
    def test_pair_helper_returns_four_paths_and_per_product_metadata(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.export._save_heatmap_png", side_effect=lambda path, *_a, **_k: Path(path).write_bytes(b"png")
        ):
            root = Path(tmp)
            source = root / "measurement.csv"
            source.write_text("source\n", encoding="utf-8")
            raw_values = np.asarray([[-2.0, -1.0, 0.0, 1.0, 2.0]]) ** 2
            raw_values = np.vstack([raw_values, raw_values + 1.0, raw_values - 2.0])
            raw_cube = _cube(raw_values, "raw")
            second_cube, _used_window = apply_sg_derivative_energy(
                raw_cube, derivative=2, window_length=5, polyorder=2
            )
            result = export_drr_pair_pngs_and_dat(
                str(root), raw_cube=raw_cube, second_cube=second_cube,
                raw_params=_params(raw_cube, -1.0, 5.0),
                second_params=_params(second_cube, -3.0, 3.0),
                raw_export_base="sample_DR_R", second_export_base="sample_d2E",
                metadata_input_files=(("measurement", source.name),),
                metadata_processing_raw={"derivative_order": None},
                metadata_processing_second={"derivative_order": 2},
            )

            self.assertEqual(set(result), {"raw_png", "raw_dat", "second_png", "second_dat"})
            self.assertEqual(result.save_status, "created")
            self.assertEqual(len(list((root / "Processed Data").glob("*.png"))), 0)
            out = root / "Processed Data" / "DRR"
            self.assertEqual(len(list(out.glob("*.png"))), 2)
            self.assertEqual(len(list(out.glob("*.dat"))), 2)
            self.assertEqual(np.loadtxt(result["raw_dat"], skiprows=1)[:, 1:].T.tolist(), raw_values.tolist())
            self.assertTrue(
                np.allclose(
                    np.loadtxt(result["second_dat"], skiprows=1)[:, 1:].T,
                    second_cube.Z,
                )
            )
            self.assertEqual(
                json.loads(result["raw_dat"].with_suffix(".metadata.json").read_text())["processing"]["derivative_order"],
                None,
            )
            self.assertEqual(
                json.loads(result["second_dat"].with_suffix(".metadata.json").read_text())["processing"]["derivative_order"],
                2,
            )

    def test_pair_helper_reuses_and_repairs_each_product(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.export._save_heatmap_png", side_effect=lambda path, *_a, **_k: Path(path).write_bytes(b"png")
        ) as render:
            root = Path(tmp)
            source = root / "measurement.csv"
            source.write_text("source\n", encoding="utf-8")
            raw = _cube(np.arange(15.0).reshape(3, 5), "raw")
            second = _cube(np.full((3, 5), 2.0), "second")
            kwargs = dict(
                folder=str(root), raw_cube=raw, second_cube=second,
                raw_params=_params(raw, -1.0, 1.0), second_params=_params(second, -3.0, 3.0),
                raw_export_base="sample_DR_R", second_export_base="sample_d2E",
                metadata_input_files=(("measurement", source.name),),
                metadata_processing_raw={"derivative_order": None},
                metadata_processing_second={"derivative_order": 2},
            )
            first = export_drr_pair_pngs_and_dat(**kwargs)
            png_mtimes = {key: first[key].stat().st_mtime_ns for key in ("raw_png", "second_png")}
            dat_mtimes = {key: first[key].stat().st_mtime_ns for key in ("raw_dat", "second_dat")}
            second_result = export_drr_pair_pngs_and_dat(**kwargs)
            self.assertEqual(second_result.save_status, "reused")
            self.assertEqual(render.call_count, 2)
            self.assertEqual(png_mtimes, {key: second_result[key].stat().st_mtime_ns for key in png_mtimes})
            self.assertEqual(dat_mtimes, {key: second_result[key].stat().st_mtime_ns for key in dat_mtimes})

            first["second_dat"].unlink()
            repaired = export_drr_pair_pngs_and_dat(**kwargs)
            self.assertEqual(repaired.save_status, "updated")
            self.assertTrue(repaired["second_dat"].exists())
            self.assertEqual(repaired["raw_dat"], first["raw_dat"])

    def test_pair_helper_validates_both_products_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = _cube(np.ones((3, 5)), "raw")
            malformed = DataCube(
                energy=np.arange(5.0), gate=np.arange(3.0), Z=np.ones((2, 2)),
                gate_label="Gate", title="second", cbar_label="d2(DR/R)/dE2",
            )
            with self.assertRaises(ValueError):
                export_drr_pair_pngs_and_dat(
                    str(root), raw_cube=raw, second_cube=malformed,
                    raw_params=_params(raw, -1.0, 1.0), second_params=_params(raw, -1.0, 1.0),
                    raw_export_base="sample_DR_R", second_export_base="sample_d2E",
                )
            self.assertFalse((root / "Processed Data" / "DRR").exists())

    def test_second_settings_change_keeps_raw_result_and_creates_derivative_result(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.export._save_heatmap_png", side_effect=lambda path, *_a, **_k: Path(path).write_bytes(b"png")
        ):
            root = Path(tmp)
            source = root / "measurement.csv"
            source.write_text("source\n", encoding="utf-8")
            raw_values = np.asarray([np.linspace(-3.0, 3.0, 7) ** 2])
            raw = _cube(raw_values, "raw")
            d2_w5, _ = apply_sg_derivative_energy(raw, derivative=2, window_length=5, polyorder=2)
            first = export_drr_pair_pngs_and_dat(
                str(root), raw_cube=raw, second_cube=d2_w5,
                raw_params=_params(raw, -1.0, 10.0), second_params=_params(d2_w5, -3.0, 3.0),
                raw_export_base="sample_DR_R", second_export_base="sample_d2E",
                metadata_input_files=(("measurement", source.name),),
                metadata_processing_raw={"derivative_order": None, "savgol_window": 5, "savgol_polyorder": 2},
                metadata_processing_second={"derivative_order": 2, "savgol_window": 5, "savgol_polyorder": 2},
            )
            raw_metadata = json.loads(first["raw_dat"].with_suffix(".metadata.json").read_text())

            d2_w7, _ = apply_sg_derivative_energy(raw, derivative=2, window_length=7, polyorder=2)
            changed = export_drr_pair_pngs_and_dat(
                str(root), raw_cube=raw, second_cube=d2_w7,
                raw_params=_params(raw, -1.0, 10.0), second_params=_params(d2_w7, -3.0, 3.0),
                raw_export_base="sample_DR_R", second_export_base="sample_d2E",
                metadata_input_files=(("measurement", source.name),),
                metadata_processing_raw={"derivative_order": None, "savgol_window": 5, "savgol_polyorder": 2},
                metadata_processing_second={"derivative_order": 2, "savgol_window": 7, "savgol_polyorder": 2},
            )
            self.assertEqual(changed.save_status, "created")
            self.assertEqual(changed["raw_dat"], first["raw_dat"])
            self.assertEqual(changed["raw_png"], first["raw_png"])
            self.assertNotEqual(changed["second_dat"], first["second_dat"])
            self.assertEqual(
                json.loads(changed["raw_dat"].with_suffix(".metadata.json").read_text()),
                raw_metadata,
            )
            changed_metadata = json.loads(changed["second_dat"].with_suffix(".metadata.json").read_text())
            self.assertEqual(changed_metadata["processing"]["savgol_window"], 7)

    def test_paired_export_writes_raw_and_second_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "measurement.csv"
            source.write_text("source\n", encoding="utf-8")
            raw = _cube(np.arange(15.0).reshape(3, 5), "raw")
            second = _cube(np.full((3, 5), 2.0), "second")
            loaded = LoadedState(
                mode="DRR", folder=str(root), primary_file=source.name,
                selected_files=[source.name], cube=raw,
            )
            options = ExportOptions(
                mode="DRR", params=_params(raw, -1.0, 1.0), drr_cube=raw,
                drr_raw_cube=raw, drr_second_cube=second,
                drr_raw_params=_params(raw, -1.0, 1.0),
                drr_second_params=_params(second, -3.0, 3.0),
                drr_derivative_order=None, drr_sg_window=9, drr_sg_polyorder=2,
                drr_raw_sg_window=9, drr_raw_sg_polyorder=2,
                drr_second_sg_window=9, drr_second_sg_polyorder=2,
            )
            result = MainWindow._export_task(
                object(), loaded, options,
                progress=SimpleNamespace(emit=lambda *_args: None),
                log=SimpleNamespace(emit=lambda *_args: None),
            )
            out = root / "Processed Data" / "DRR"
            self.assertEqual(len(list(out.glob("*.png"))), 2)
            self.assertEqual(len(list(out.glob("*.dat"))), 2)
            self.assertTrue(any("d2E" in path.stem for path in out.glob("*.dat")))
            self.assertTrue(any("DR_R" in path.stem for path in out.glob("*.dat")))
            metadata = [json.loads(path.read_text(encoding="utf-8")) for path in out.glob("*.metadata.json")]
            derivatives = {entry["processing"]["derivative_order"] for entry in metadata}
            self.assertEqual(derivatives, {None, 2})
            self.assertEqual(result["mode"], "DRR")

    def test_paired_export_preserves_second_split_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "measurement.csv"
            source.write_text("source\n", encoding="utf-8")
            raw = _cube(np.arange(15.0).reshape(3, 5), "raw")
            second = _cube(np.linspace(-2.0, 2.0, 15).reshape(3, 5), "second")
            second_params = _params(second, -2.0, 2.0)
            second_params.split_scale = SplitColorScale(
                split_x=0.0, left_vmin=-2.0, left_vmax=-0.1,
                right_vmin=0.1, right_vmax=2.0, show_boundary=True,
            )
            loaded = LoadedState(
                mode="DRR", folder=str(root), primary_file=source.name,
                selected_files=[source.name], cube=raw,
            )
            options = ExportOptions(
                mode="DRR", params=_params(raw, -1.0, 1.0), drr_cube=raw,
                drr_raw_cube=raw, drr_second_cube=second,
                drr_raw_params=_params(raw, -1.0, 1.0),
                drr_second_params=second_params,
                drr_second_auto_scale=False,
            )
            result = MainWindow._export_task(
                object(), loaded, options,
                progress=SimpleNamespace(emit=lambda *_args: None),
                log=SimpleNamespace(emit=lambda *_args: None),
            )
            second_dat = next((path for path in (root / "Processed Data" / "DRR").glob("*d2E*.dat")), None)
            self.assertIsNotNone(second_dat)
            metadata = json.loads(second_dat.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["plot"]["split_scale"]["left_vmax"], -0.1)
            self.assertEqual(metadata["plot"]["split_scale"]["right_vmin"], 0.1)


if __name__ == "__main__":
    unittest.main()
