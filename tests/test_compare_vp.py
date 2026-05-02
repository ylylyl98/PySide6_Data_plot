import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.export_legacy import compare_source_title, export_compare_panels, vp_compare_export_base, vp_compare_title
from core.loader import DataCube
from core.plotting import HeatmapParams
from core.processing import (
    background_correct_cube,
    coherent_compare_auto_assignment,
    estimate_constant_background,
    nearest_gate_spectrum,
    parse_compare_gate_condition,
    parse_compare_in_out_angles,
    valley_polarization_cube,
)


class CompareAngleParserTests(unittest.TestCase):
    def test_rot1_rot2_p_decimal_angles(self) -> None:
        name = "YZ247_pX2_3.6KPL_730nm_2sx1_Rot1195p8deg_Rot295deg_Stage.csv"
        self.assertEqual(parse_compare_in_out_angles(name), (195.8, 95.0))

    def test_rot2_145_angle(self) -> None:
        name = "YZ247_pX2_3.6KPL_730nm_2sx1_Rot1195p8deg_Rot2145deg_Stage.csv"
        self.assertEqual(parse_compare_in_out_angles(name), (195.8, 145.0))

    def test_legacy_in_out_degree_pattern(self) -> None:
        name = "scan_in 195.000 degree_out 95.000 degree.csv"
        self.assertEqual(parse_compare_in_out_angles(name), (195.0, 95.0))

    def test_gate_condition_parser(self) -> None:
        name = "YZ247_Rot1195p8deg_Rot2145deg_Stage50_TG-BG=0.csv"
        self.assertEqual(parse_compare_gate_condition(name), "TG-BG=0")

    def test_coherent_auto_assignment_does_not_mix_gate_groups(self) -> None:
        files = [
            "YZ247_pX2_3.6KPL_730nm5.37uW_865nmc_2sx1_Rot1195p8deg_Rot295deg_Stage50_TG+BG=0.csv",
            "YZ247_pX2_3.6KPL_730nm5.40uW_865nmc_2sx1_Rot1195p8deg_Rot2145deg_Stage50_TG-BG=0.csv",
            "YZ247_pX2_3.6KPL_730nm5.57uW_865nmc_2sx1_Rot1195p8deg_Rot295deg_Stage50_TG-BG=0.csv",
        ]
        found, duplicates, gate_group, gate_groups = coherent_compare_auto_assignment(
            files,
            in_k_angle=195.0,
            out_k_angle=95.0,
        )
        self.assertEqual(gate_group, "TG-BG=0")
        self.assertIn("TG+BG=0", gate_groups)
        self.assertFalse(duplicates)
        self.assertEqual(found["KK"], files[2])
        self.assertEqual(found["KKp"], files[1])


class CompareVpTests(unittest.TestCase):
    def test_auto_constant_background_estimate_uses_shared_low_percentile(self) -> None:
        energy = np.array([1.0, 2.0, 3.0])
        gate = np.array([0.0, 1.0])
        kk = DataCube(energy, gate, np.array([[10.0, 12.0, 100.0], [11.0, 13.0, 200.0]]), "TG (V)", "KK", "PL")
        kkp = DataCube(energy, gate, np.array([[20.0, 22.0, 300.0], [21.0, 23.0, 400.0]]), "TG (V)", "KKp", "PL")

        self.assertEqual(estimate_constant_background({"KK": kk, "KKp": kkp}, percentile=0.0), 10.0)
        self.assertEqual(estimate_constant_background({"KK": kk, "KKp": kkp}, percentile=100.0), 400.0)

    def test_background_corrected_cube_is_used_for_linecut_values(self) -> None:
        energy = np.array([1.0, 2.0])
        gate = np.array([0.0, 1.0])
        cube = DataCube(energy, gate, np.array([[5.0, 9.0], [3.0, 7.0]]), "TG (V)", "KK", "PL")
        corrected = background_correct_cube(cube, 1.0)
        _gate, spectrum = nearest_gate_spectrum(corrected, 0.0)
        self.assertTrue(np.allclose(corrected.Z, np.array([[4.0, 8.0], [2.0, 6.0]])))
        self.assertTrue(np.allclose(spectrum, np.array([4.0, 8.0])))

    def test_vp_uses_background_corrected_inputs(self) -> None:
        energy = np.array([1.0, 2.0])
        gate = np.array([0.0, 1.0])
        kk = DataCube(energy, gate, np.array([[5.0, 9.0], [3.0, 7.0]]), "TG (V)", "KK", "PL")
        kkp = DataCube(energy, gate, np.array([[3.0, 5.0], [3.0, 1.0]]), "TG (V)", "KKp", "PL")
        vp = valley_polarization_cube(kk, kkp, background=1.0)
        expected = np.array([[2.0 / 6.0, 4.0 / 12.0], [0.0 / 4.0, 6.0 / 6.0]])
        self.assertTrue(np.allclose(vp.Z, expected, equal_nan=True))


class CompareExportTests(unittest.TestCase):
    def test_vp_filename_keeps_shared_rot_and_gate_tokens(self) -> None:
        sources = {
            "KK": "YZ247_pX2_3.6KPL_730nm5.57uW_865nmc_2sx1_Rot1195p8deg_Rot295deg_TG-BG=0.csv",
            "KKp": "YZ247_pX2_3.6KPL_730nm5.40uW_865nmc_2sx1_Rot1195p8deg_Rot2145deg_TG-BG=0.csv",
        }
        base = vp_compare_export_base(sources, 0.0, "linear")
        self.assertIn("pX2", base)
        self.assertIn("Rot1195p8deg", base)
        self.assertIn("TG-BG=0", base)
        self.assertIn("P5.57-5.40uW", base)
        self.assertIn("Bkg0", base)
        self.assertTrue(base.endswith("_C_Lin"))
        title = vp_compare_title(sources, 0.0, "linear")
        self.assertIn("P5.57-5.40uW", title)
        self.assertTrue(title.endswith("_Bkg0_Lin"))

    def test_compare_export_writes_corrected_channels_and_vp_only(self) -> None:
        energy = np.array([1.0, 2.0])
        gate = np.array([0.0, 1.0])
        cubes = {
            "KK": DataCube(energy, gate, np.array([[5.0, 9.0], [3.0, 7.0]]), "TG (V)", "KK", "PL"),
            "KKp": DataCube(energy, gate, np.array([[3.0, 5.0], [3.0, 1.0]]), "TG (V)", "KKp", "PL"),
        }
        sources = {
            "KK": "YZ247_pX2_3.6KPL_730nm1062.21uW_865nmc_2sx1_Rot1195p8deg_Rot295deg_TG-BG=0.csv",
            "KKp": "YZ247_pX2_3.6KPL_730nm1056.37uW_865nmc_2sx1_Rot1195p8deg_Rot2145deg_TG-BG=0.csv",
        }
        params = HeatmapParams(
            title="Compare",
            xlabel="Photon Energy (eV)",
            ylabel="TG (V)",
            cbar_label="PL corr. (a.u.)",
            vmin=0.1,
            vmax=10.0,
            xlim=(1.0, 2.0),
            ylim=(0.0, 1.0),
            cmap="turbo",
            log_scale=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_compare_panels(
                tmp,
                cubes=cubes,
                source_files=sources,
                params=params,
                scale_tag="linear",
                clip_outliers=False,
                correction_background=1.0,
                export_vp=True,
            )
            names = sorted(path.name for path in paths)
            self.assertEqual(len(names), 6)
            self.assertFalse(any("CompareGrid" in name or "CompareLinecut" in name for name in names))
            self.assertTrue(any(name.startswith("KK_") and name.endswith("_C_Lin.png") for name in names))
            self.assertTrue(any(name.startswith("KKp_") and name.endswith("_C_Lin.dat") for name in names))
            self.assertTrue(any(name.startswith("VP_") and "_Bkg1_C_Lin.png" in name for name in names))

            kk_dat = next(Path(tmp, "Processed Data", name) for name in names if name.startswith("KK_") and name.endswith(".dat"))
            text = kk_dat.read_text()
            self.assertIn(f"title={compare_source_title(sources['KK'])}", text)
            self.assertIn("background_constant=1.0", text)
            self.assertIn("\t4", text)
            vp_dat = next(Path(tmp, "Processed Data", name) for name in names if name.startswith("VP_") and name.endswith(".dat"))
            self.assertIn("title=VP_", vp_dat.read_text())


if __name__ == "__main__":
    unittest.main()
