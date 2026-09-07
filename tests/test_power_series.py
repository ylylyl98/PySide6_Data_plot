import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

import numpy as np

from core.export import export_power_series_png_and_dat, export_power_vp_pngs_and_dat, power_series_export_base
from core.loader import DataCube
from core.plotting import HeatmapParams
from core.processing import (
    PowerSeriesFile,
    group_power_series_files,
    parse_power_series_file,
    power_group_title,
    power_stage_paired_vp_cubes,
    power_valley_polarization_cube,
)
from core import data_io


FILES = [
    "YZ247_pX2_3.6KPL_730nm37.96uW_865nmc_5sx1_Rot1195p8deg_Rot2145deg_Stage5_TG=BG=0.csv",
    "YZ247_pX2_3.6KPL_730nm48.46uW_865nmc_5sx1_Rot1195p8deg_Rot2145deg_Stage4_TG=BG=0.csv",
    "YZ247_pX2_3.6KPL_730nm92.59uW_865nmc_5sx1_Rot1195p8deg_Rot295deg_Stage1_TG=BG=0.csv",
    "YZ247_pX2_3.6KPL_730nm61.95uW_865nmc_5sx1_Rot1195p8deg_Rot295deg_Stage3_TG=BG=0.csv",
]


class PowerSeriesParserTests(unittest.TestCase):
    def test_parser_removes_power_and_stage_from_group_title(self) -> None:
        parsed = parse_power_series_file(FILES[0])
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertAlmostEqual(parsed.power_uW, 37.96)
        self.assertEqual(parsed.stage, 5.0)
        self.assertEqual(
            parsed.title,
            "YZ247 pX2 3.6KPL 730nm 865nmc 5sx1 Rot1195p8deg Rot2145deg TG=BG=0",
        )

    def test_grouping_keeps_rotation_groups_separate_and_sorts_power(self) -> None:
        groups = group_power_series_files(FILES)
        self.assertEqual(len(groups), 2)
        titles = {power_group_title(key): [r.power_uW for r in records] for key, records in groups.items()}
        self.assertEqual(
            titles["YZ247 pX2 3.6KPL 730nm 865nmc 5sx1 Rot1195p8deg Rot2145deg TG=BG=0"],
            [37.96, 48.46],
        )
        self.assertEqual(
            titles["YZ247 pX2 3.6KPL 730nm 865nmc 5sx1 Rot1195p8deg Rot295deg TG=BG=0"],
            [61.95, 92.59],
        )


class PowerSeriesLoaderExportTests(unittest.TestCase):
    def test_power_series_can_export_into_group_package(self) -> None:
        cube = DataCube(
            np.asarray([1.0, 2.0]), np.asarray([10.0, 20.0]),
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "Power (uW)", "KK", "PL",
        )
        params = HeatmapParams(
            title="KK", xlabel="Energy", ylabel="Power", cbar_label="PL",
            vmin=0.0, vmax=4.0, xlim=(1.0, 2.0), ylim=(10.0, 20.0),
        )
        records = (PowerSeriesFile("a.csv", 10.0, 1.0, "rot2145", "KK"),)
        with tempfile.TemporaryDirectory() as tmp, patch("core.export._save_heatmap_png"):
            paths = export_power_series_png_and_dat(
                tmp, cube=cube, params=params, records=records,
                group_key="rot2145", y_axis_log=False,
                processed_name="Processed Data/Power Dependence/rot2145_PowerDep",
                metadata_extra={"package": "rot2145_PowerDep"},
            )
            expected = Path(tmp) / "Processed Data" / "Power Dependence" / "rot2145_PowerDep"
            self.assertEqual(paths["dat"].parent, expected)
            metadata = json.loads(paths["dat"].with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["package"], "rot2145_PowerDep")
            self.assertEqual(metadata["processing"]["group_key"], "rot2145")
            self.assertEqual(metadata["processing"]["power_values_uW"], [10.0])

    def test_legacy_power_group_uses_one_filename_derived_gate_axis(self) -> None:
        files = [
            "device_1uW_Rot220p5deg_Stage0_TG-0.8BG=0.csv",
            "device_2uW_Rot220p5deg_Stage1_TG-0.8BG=0.csv",
        ]
        cube = DataCube(
            np.array([1.0, 2.0]),
            np.array([0.0]),
            np.array([[1.0, 2.0]]),
            "TG+0.8BG (V)",
            "PL",
            "PL (a.u.)",
        )
        with patch.object(data_io, "load_pl", return_value=cube) as mock_load_pl:
            data_io.load_power_series_cube("unused", files, y_axis="auto")
        self.assertEqual(len(mock_load_pl.call_args_list), 2)
        self.assertTrue(
            all(call.kwargs["y_axis"] == "linear:1,0.8,0" for call in mock_load_pl.call_args_list)
        )

    def test_single_csv_power_sweep_uses_power_column_and_numeric_spectral_headers(self) -> None:
        fixture_folder = Path(__file__).resolve().parent / "fixtures"
        file_name = "power_sweep_table.csv"

        self.assertTrue(data_io.inspect_power_sweep_csv(str(fixture_folder), file_name))
        result = data_io.load_power_series_cube(str(fixture_folder), [file_name])

        self.assertEqual(result.group_key, f"csv::{file_name}")
        self.assertEqual(result.cube.gate.tolist(), [10.0, 20.0])
        self.assertEqual(result.cube.gate_label, "Power (uW)")
        self.assertTrue(np.all(np.diff(result.cube.energy) > 0))
        self.assertTrue(np.allclose(result.cube.Z[0], [100.0, 101.0, 102.0]))
        self.assertEqual([record.stage for record in result.records], [25.0, 0.0])
        self.assertEqual([record.row_index for record in result.records], [3, 2])

    def test_single_csv_power_sweep_rejects_duplicate_power_values(self) -> None:
        fixture_folder = Path(__file__).resolve().parent / "fixtures"
        with self.assertRaisesRegex(ValueError, "duplicate Power_uW"):
            data_io.load_power_series_cube(
                str(fixture_folder),
                ["power_sweep_duplicate.csv"],
            )

    def test_single_csv_power_sweep_reports_missing_spectral_headers(self) -> None:
        fixture_folder = Path(__file__).resolve().parent / "fixtures"
        file_name = "power_sweep_missing_spectrum.csv"
        self.assertTrue(data_io.inspect_power_sweep_csv(str(fixture_folder), file_name))
        with self.assertRaisesRegex(ValueError, "at least two numeric wavelength/energy columns"):
            data_io.load_power_series_cube(str(fixture_folder), [file_name])

    def test_single_csv_power_sweep_stage_rows_support_vp_pairing(self) -> None:
        fixture_folder = Path(__file__).resolve().parent / "fixtures"
        result = data_io.load_power_series_cube(str(fixture_folder), ["power_sweep_table.csv"])
        shifted = DataCube(
            result.cube.energy.copy(),
            result.cube.gate.copy(),
            result.cube.Z * 0.5,
            result.cube.gate_label,
            "KKp",
            result.cube.cbar_label,
        )
        _kk, _kkp, vp, pairs = power_stage_paired_vp_cubes(
            result.cube,
            result.records,
            shifted,
            result.records,
            background=0.0,
        )
        self.assertEqual([pair.stage for pair in pairs], [25.0, 0.0])
        self.assertEqual([(pair.kk_row, pair.kkp_row) for pair in pairs], [(3, 3), (2, 2)])
        self.assertTrue(np.allclose(vp.Z, 1.0 / 3.0))

    def test_single_csv_power_sweep_export_uses_source_stem_and_row_provenance(self) -> None:
        fixture_folder = Path(__file__).resolve().parent / "fixtures"
        result = data_io.load_power_series_cube(str(fixture_folder), ["power_sweep_table.csv"])
        cube = result.cube
        params = HeatmapParams(
            title=cube.title,
            xlabel="Photon Energy (eV)",
            ylabel=cube.gate_label,
            cbar_label=cube.cbar_label,
            vmin=float(np.nanmin(cube.Z)),
            vmax=float(np.nanmax(cube.Z)),
            xlim=(float(np.nanmin(cube.energy)), float(np.nanmax(cube.energy))),
            ylim=(float(np.nanmin(cube.gate)), float(np.nanmax(cube.gate))),
            cmap="turbo",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_power_series_png_and_dat(
                tmp,
                cube=cube,
                params=params,
                records=result.records,
                group_key=result.group_key,
                y_axis_log=False,
            )
            self.assertTrue(paths["png"].name.startswith("power_sweep_table_PowerDep_YLin"))
            exported = paths["dat"].read_text(encoding="utf-8")
            self.assertIn("# source_csv=power_sweep_table.csv", exported)
            self.assertIn("# power_column=Power_uW", exported)
            self.assertIn("# stage_column=stage_pos", exported)
            self.assertIn("row=3; power_uW=10.0; stage=25.0", exported)

    def test_loader_stacks_one_spectrum_per_file_sorted_by_power(self) -> None:
        energy = np.array([1.0, 2.0, 3.0])

        def fake_load_pl(_folder: str, file_name: str, **_kwargs) -> DataCube:
            parsed = parse_power_series_file(file_name)
            assert parsed is not None
            z = np.array([[parsed.power_uW, parsed.power_uW + 1.0, parsed.power_uW + 2.0]])
            return DataCube(energy, np.array([0.0]), z, "Gate", file_name, "PL")

        with patch.object(data_io, "load_pl", side_effect=fake_load_pl):
            result = data_io.load_power_series_cube("unused", FILES, group_key=None)

        self.assertEqual(result.cube.gate.tolist(), [37.96, 48.46])
        self.assertTrue(np.allclose(result.cube.Z[:, 0], [37.96, 48.46]))
        self.assertEqual(
            result.cube.title,
            "YZ247 pX2 3.6KPL 730nm 865nmc 5sx1 Rot1195p8deg Rot2145deg TG=BG=0",
        )

    def test_export_uses_powerdep_axis_tag_and_clean_title(self) -> None:
        cube = DataCube(
            np.array([1.0, 2.0]),
            np.array([10.0, 20.0]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            "Power (uW)",
            "YZ247 pX2 3.6KPL 730nm 865nmc 5sx1 Rot1195p8deg Rot2145deg TG=BG=0",
            "PL (a.u.)",
        )
        params = HeatmapParams(
            title=cube.title,
            xlabel="Photon Energy (eV)",
            ylabel=cube.gate_label,
            cbar_label=cube.cbar_label,
            vmin=1.0,
            vmax=4.0,
            xlim=(1.0, 2.0),
            ylim=(10.0, 20.0),
            cmap="turbo",
            log_scale=False,
            y_axis_log=True,
        )
        group_key = "YZ247_pX2_3.6KPL_730nm_865nmc_5sx1_Rot1195p8deg_Rot2145deg_TG=BG=0"
        self.assertTrue(power_series_export_base(group_key, y_axis_log=True).endswith("_PowerDep_YLog"))
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_power_series_png_and_dat(
                tmp,
                cube=cube,
                params=params,
                records=(),
                group_key=group_key,
                y_axis_log=True,
            )
            self.assertTrue(paths["png"].name.endswith("_PowerDep_YLog.png"))
            text = paths["dat"].read_text()
            lines = text.splitlines()
            data_lines = [line for line in lines if not line.startswith("#")]
            self.assertIn("# mode=Power Dependent", lines)
            self.assertIn("# corrected=True", lines)
            self.assertEqual(data_lines[0], "Photon energy\t10\t20")
            self.assertEqual(data_lines[1], "1\t1\t3")
            self.assertEqual(data_lines[2], "2\t2\t4")

    def test_power_vp_interpolates_slightly_different_power_axes_after_background(self) -> None:
        energy = np.array([1.0, 2.0])
        kk = DataCube(
            energy,
            np.array([10.0, 20.0, 30.0]),
            np.array([[11.0, 21.0], [21.0, 31.0], [31.0, 41.0]]),
            "Power (uW)",
            "KK",
            "PL",
        )
        kkp = DataCube(
            energy,
            np.array([10.2, 20.2, 30.2]),
            np.array([[6.1, 16.1], [11.1, 21.1], [16.1, 26.1]]),
            "Power (uW)",
            "KKp",
            "PL",
        )
        kk_corr, kkp_corr, vp = power_valley_polarization_cube(kk, kkp, background=1.0, title="VP")
        self.assertTrue(np.allclose(vp.gate, np.array([20.0, 30.0])))
        self.assertEqual(vp.Z.shape, (2, 2))
        expected_kk_e0 = np.array([20.0, 30.0])
        expected_kkp_e0 = np.array([10.0, 15.0])
        expected = (expected_kk_e0 - expected_kkp_e0) / (expected_kk_e0 + expected_kkp_e0)
        self.assertTrue(np.allclose(kk_corr.Z[:, 0], expected_kk_e0))
        self.assertTrue(np.allclose(kkp_corr.Z[:, 0], expected_kkp_e0))
        self.assertTrue(np.allclose(vp.Z[:, 0], expected))

    def test_stage_paired_power_vp_matches_stage_not_duplicate_power(self) -> None:
        energy = np.array([1.0, 2.0])
        kk = DataCube(
            energy,
            np.array([0.01, 0.01]),
            np.array([[11.0, 21.0], [31.0, 41.0]]),
            "Power (uW)",
            "KK",
            "PL",
        )
        kkp = DataCube(
            energy,
            np.array([0.01, 0.01]),
            np.array([[6.0, 16.0], [16.0, 26.0]]),
            "Power (uW)",
            "KKp",
            "PL",
        )
        kk_records = (
            PowerSeriesFile("kk_stage50.csv", 0.01, 50.0, "kk", "kk"),
            PowerSeriesFile("kk_stage47.csv", 0.01, 47.0, "kk", "kk"),
        )
        kkp_records = (
            PowerSeriesFile("kkp_stage47.csv", 0.01, 47.0, "kkp", "kkp"),
            PowerSeriesFile("kkp_stage50.csv", 0.01, 50.0, "kkp", "kkp"),
        )
        kk_corr, kkp_corr, vp, pairs = power_stage_paired_vp_cubes(
            kk,
            kk_records,
            kkp,
            kkp_records,
            background=1.0,
            title="VP",
        )
        self.assertEqual([pair.stage for pair in pairs], [47.0, 50.0])
        self.assertTrue(np.allclose(vp.gate, [0.01, 0.01]))
        self.assertTrue(np.allclose(kk_corr.Z[:, 0], [30.0, 10.0]))
        self.assertTrue(np.allclose(kkp_corr.Z[:, 0], [5.0, 15.0]))
        expected = np.array([(30.0 - 5.0) / (30.0 + 5.0), (10.0 - 15.0) / (10.0 + 15.0)])
        self.assertTrue(np.allclose(vp.Z[:, 0], expected))

    def test_power_vp_export_writes_corrected_channels_and_vp(self) -> None:
        energy = np.array([1.0, 2.0])
        power = np.array([10.0, 20.0])
        kk = DataCube(energy, power, np.array([[10.0, 20.0], [30.0, 40.0]]), "Power (uW)", "KK", "PL corr. (a.u.)")
        kkp = DataCube(energy, power, np.array([[5.0, 10.0], [15.0, 20.0]]), "Power (uW)", "KKp", "PL corr. (a.u.)")
        vp = DataCube(energy, power, np.array([[0.33, 0.33], [0.33, 0.33]]), "Power (uW)", "VP", "VP")
        params = HeatmapParams(
            title="VP",
            xlabel="Photon Energy (eV)",
            ylabel="Power (uW)",
            cbar_label="VP",
            vmin=-1.0,
            vmax=1.0,
            xlim=(1.0, 2.0),
            ylim=(10.0, 20.0),
            cmap="RdBu_r",
            y_axis_log=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_power_vp_pngs_and_dat(
                tmp,
                kk_cube=kk,
                kkp_cube=kkp,
                vp_cube=vp,
                params=params,
                kk_group_key="rot2145",
                kkp_group_key="rot295",
                kk_records=(),
                kkp_records=(),
                y_axis_log=False,
                background=2.0,
                pairing_mode="stage",
                stage_pairs=(),
            )
            self.assertIn("VP_png", paths)
            self.assertIn("KK_dat", paths)
            self.assertIn("KKp_dat", paths)
            for key in ("KK_dat", "KKp_dat", "VP_dat"):
                text = paths[key].read_text()
                lines = text.splitlines()
                self.assertTrue(lines[0].startswith("# mode=Power Dependent"))
                self.assertIn("Photon energy\t10\t20", lines)
            self.assertIn("1\t0.33\t0.33", paths["VP_dat"].read_text())
            metadata = json.loads(paths["VP_dat"].with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["workflow"], "Power Dependent VP")
            self.assertEqual(metadata["dataset_type"], "derived_analysis_2d")
            self.assertEqual(metadata["processing"]["kk_group_key"], "rot2145")
            self.assertEqual(metadata["processing"]["kkp_group_key"], "rot295")
            self.assertEqual(metadata["output_manifest"][1]["filename"], paths["VP_dat"].name)


if __name__ == "__main__":
    unittest.main()
