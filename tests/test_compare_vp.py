import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.export import export_compare_panels, vp_compare_export_base, vp_compare_title
from core.loader import DataCube
from core.plotting import HeatmapParams
from core.processing import (
    background_correct_cube,
    circular_angle_distance,
    classify_angle_state,
    classify_compare_channel,
    coherent_compare_auto_assignment,
    estimate_constant_background,
    infer_compare_angle_references,
    nearest_gate_spectrum,
    parse_compare_gate_condition,
    parse_compare_in_out_angles,
    parse_compare_rotation_angles,
    valley_polarization_cube,
)


class CompareAngleParserTests(unittest.TestCase):
    EXACT_ROT2_FILES = [
        "YZD344_pen3_6KPL_690nm1508.355uW_880nmc_1sx1_Rot220p5deg_Stage0_TG-0.8BG=0.csv",
        "YZD344_pen3_6KPL_690nm1518.648uW_880nmc_1sx1_Rot265p5deg_Stage0_TG+0.8BG=0.csv",
    ]

    def test_rot1_rot2_p_decimal_angles(self) -> None:
        name = "YZ247_pX2_3.6KPL_730nm_2sx1_Rot1195p8deg_Rot295deg_Stage.csv"
        self.assertEqual(parse_compare_in_out_angles(name), (195.8, 95.0))

    def test_rot2_145_angle(self) -> None:
        name = "YZ247_pX2_3.6KPL_730nm_2sx1_Rot1195p8deg_Rot2145deg_Stage.csv"
        self.assertEqual(parse_compare_in_out_angles(name), (195.8, 145.0))

    def test_legacy_in_out_degree_pattern(self) -> None:
        name = "scan_in 195.000 degree_out 95.000 degree.csv"
        self.assertEqual(parse_compare_in_out_angles(name), (195.0, 95.0))

    def test_rot1_and_rot2_are_detected_independently(self) -> None:
        self.assertEqual(parse_compare_in_out_angles("scan_Rot120p5deg.csv"), (20.5, None))
        self.assertEqual(parse_compare_in_out_angles("scan_Rot220p5deg.csv"), (None, 20.5))
        self.assertEqual(parse_compare_rotation_angles("scan_Rot2-12.25degree.csv").rot2, -12.25)

    def test_exact_rot2_filenames_classify_with_fixed_input_k(self) -> None:
        self.assertEqual(parse_compare_in_out_angles(self.EXACT_ROT2_FILES[0]), (None, 20.5))
        self.assertEqual(parse_compare_in_out_angles(self.EXACT_ROT2_FILES[1]), (None, 65.5))
        channels = [
            classify_compare_channel(name, in_k_angle=0.0, out_k_angle=20.5)
            for name in self.EXACT_ROT2_FILES
        ]
        self.assertEqual(channels, ["KK", "KKp"])
        self.assertIsNone(
            classify_compare_channel(
                self.EXACT_ROT2_FILES[0],
                in_k_angle=0.0,
                out_k_angle=20.5,
                fixed_missing_arm=None,
            )
        )

    def test_exact_complementary_gate_files_form_vp_pair(self) -> None:
        found, duplicates, gate_group, gate_groups = coherent_compare_auto_assignment(
            self.EXACT_ROT2_FILES,
            in_k_angle=0.0,
            out_k_angle=20.5,
        )
        self.assertEqual(found, {"KK": self.EXACT_ROT2_FILES[0], "KKp": self.EXACT_ROT2_FILES[1]})
        self.assertFalse(duplicates)
        self.assertEqual(gate_group, "TG-0.8BG=0")
        self.assertEqual(gate_groups, ["TG-0.8BG=0", "TG+0.8BG=0"])

    def test_explicit_kp_reference_classifies_variable_rot2_angles(self) -> None:
        cases = {
            65.5: "KK",
            25.0: "KKp",
            20.5: "KKp",
            15.0: "KKp",
        }
        for angle, expected in cases.items():
            with self.subTest(angle=angle):
                token = str(angle).replace(".", "p")
                self.assertEqual(
                    classify_compare_channel(
                        f"scan_Rot2{token}deg.csv",
                        in_k_angle=0.0,
                        in_kp_angle=45.0,
                        out_k_angle=60.5,
                        out_kp_angle=20.0,
                        tolerance=12.0,
                    ),
                    expected,
                )

    def test_explicit_references_reject_ambiguous_and_distant_angles(self) -> None:
        ambiguous = classify_angle_state(
            40.0,
            k_angle=60.0,
            kp_angle=20.0,
            tolerance=25.0,
            ambiguity_margin=1.0,
        )
        self.assertIsNone(ambiguous.state)
        self.assertEqual(ambiguous.reason, "ambiguous")
        distant = classify_angle_state(
            40.0,
            k_angle=65.0,
            kp_angle=15.0,
            tolerance=10.0,
        )
        self.assertIsNone(distant.state)
        self.assertEqual(distant.reason, "outside-tolerance")
        self.assertEqual(circular_angle_distance(358.0, 2.0), 4.0)

    def test_explicit_in_and_out_references_support_all_four_channels(self) -> None:
        references = {
            "in_k_angle": 195.0,
            "in_kp_angle": 150.0,
            "out_k_angle": 60.0,
            "out_kp_angle": 20.0,
            "tolerance": 12.0,
        }
        cases = {
            "scan_Rot1196deg_Rot261deg.csv": "KK",
            "scan_Rot1196deg_Rot219deg.csv": "KKp",
            "scan_Rot1151deg_Rot261deg.csv": "KpK",
            "scan_Rot1151deg_Rot219deg.csv": "KpKp",
        }
        for file_name, expected in cases.items():
            self.assertEqual(classify_compare_channel(file_name, **references), expected)

    def test_angle_inference_clusters_drift_without_fixed_separation(self) -> None:
        files = [
            "scan_Rot215deg.csv",
            "scan_Rot220p5deg.csv",
            "scan_Rot225deg.csv",
            "scan_Rot260p5deg.csv",
            "scan_Rot265p5deg.csv",
        ]
        inferred = infer_compare_angle_references(
            files,
            in_k_anchor=0.0,
            out_k_anchor=60.5,
            cluster_tolerance=12.0,
        )
        self.assertEqual(len(inferred.rot2_clusters), 2)
        self.assertAlmostEqual(float(inferred.out_k), 63.0, places=1)
        self.assertAlmostEqual(float(inferred.out_kp), 20.2, places=1)

    def test_explicit_references_assign_screenshot_pair_with_20p5_as_kkp(self) -> None:
        found, _duplicates, _gate_group, _gate_groups = coherent_compare_auto_assignment(
            self.EXACT_ROT2_FILES,
            in_k_angle=0.0,
            in_kp_angle=45.0,
            out_k_angle=60.5,
            out_kp_angle=20.5,
            tolerance=12.0,
        )
        self.assertEqual(found["KK"], self.EXACT_ROT2_FILES[1])
        self.assertEqual(found["KKp"], self.EXACT_ROT2_FILES[0])

    def test_gate_condition_parser(self) -> None:
        name = "YZ247_Rot1195p8deg_Rot2145deg_Stage50_TG-BG=0.csv"
        self.assertEqual(parse_compare_gate_condition(name), "TG-BG=0")

    def test_gate_condition_parser_normalizes_unicode_minus(self) -> None:
        name = "YZ364_Rot1195p8deg_Rot2145deg_TG−1.1BG=0.csv"
        self.assertEqual(parse_compare_gate_condition(name), "TG-1.1BG=0")

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

    def test_auto_assignment_uses_coherent_stage_pair_at_angle_boundary(self) -> None:
        files = [
            "YZ247_pX2_300g_3.6KPL_730nm17.83uW_855nmc_5sx1_Rot1195p8deg_Rot2140deg_Stage50_TG-BG=0.csv",
            "YZ247_pX2_300g_3.6KPL_730nm370.87uW_855nmc_5sx1_Rot1195p8deg_Rot2140deg_Stage22_TG-BG=0.csv",
            "YZ247_pX2_300g_3.6KPL_730nm373.39uW_855nmc_5sx1_Rot1195p8deg_Rot295deg_Stage22_TG-BG=0.csv",
        ]
        self.assertEqual(
            classify_compare_channel(files[1], in_k_angle=195.0, out_k_angle=95.0),
            "KKp",
        )
        found, duplicates, gate_group, _gate_groups = coherent_compare_auto_assignment(
            files,
            in_k_angle=195.0,
            out_k_angle=95.0,
        )
        self.assertEqual(gate_group, "TG-BG=0")
        self.assertEqual(found["KK"], files[2])
        self.assertEqual(found["KKp"], files[1])
        self.assertIn("KKp", duplicates)


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
            lines = text.splitlines()
            data_lines = [line for line in lines if not line.startswith("#")]
            self.assertIn("# panel=KK", lines)
            self.assertIn("# corrected=True", lines)
            self.assertEqual(data_lines[0], "Photon energy\t0\t1")
            self.assertEqual(data_lines[1], "1\t4\t2")
            self.assertEqual(data_lines[2], "2\t8\t6")
            vp_dat = next(Path(tmp, "Processed Data", name) for name in names if name.startswith("VP_") and name.endswith(".dat"))
            vp_lines = vp_dat.read_text().splitlines()
            self.assertEqual(vp_lines[0], "# panel=VP")
            self.assertIn("Photon energy\t0\t1", vp_lines)


if __name__ == "__main__":
    unittest.main()
