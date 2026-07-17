from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import data_io
import core.loader as loader
import core.processing_run as processing_run

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "y_axis"


class CanonicalYAxisResolutionTests(unittest.TestCase):
    def test_exact_decimal_bg_condition_uses_complementary_axis(self) -> None:
        file_name = "YZD344_Rot220p5deg_Stage0_TG-0.8BG=0.csv"
        with tempfile.TemporaryDirectory() as folder:
            shutil.copyfile(FIXTURES / "plain_scan.csv", Path(folder) / file_name)
            d = processing_run._load_canonical(folder, file_name, y_axis="auto")
        self.assertTrue(np.allclose(d["gate_axis"], np.array([10.0, 11.8, 13.6])))
        self.assertEqual(d["gate_label"], "TG+0.8BG (V)")

    def test_canonical_linear_request_keeps_clean_coefficient_label(self) -> None:
        d = processing_run._load_canonical(
            str(FIXTURES), "plain_scan.csv", y_axis="linear:1,0.8,0"
        )
        self.assertTrue(np.allclose(d["gate_axis"], np.array([10.0, 11.8, 13.6])))
        self.assertEqual(d["gate_label"], "TG+0.8BG (V)")

    def test_shared_auto_axis_uses_reference_condition_for_complementary_pair(self) -> None:
        files = [
            "scan_Rot220p5deg_TG-0.8BG=0.csv",
            "scan_Rot265p5deg_TG+0.8BG=0.csv",
        ]
        self.assertEqual(processing_run.resolve_shared_y_axis_request(files), "linear:1,0.8,0")

    def test_shared_auto_axis_is_stable_when_channel_order_is_reversed(self) -> None:
        files = [
            "scan_Rot265p5deg_TG+0.8BG=0.csv",
            "scan_Rot220p5deg_TG-0.8BG=0.csv",
        ]
        self.assertEqual(processing_run.resolve_shared_y_axis_request(files), "linear:1,0.8,0")

    def test_shared_auto_axis_rejects_different_coefficients(self) -> None:
        with self.assertRaisesRegex(ValueError, "metadata conflict"):
            processing_run.resolve_shared_y_axis_request(
                ["scan_TG-0.8BG=0.csv", "scan_TG-1.1BG=0.csv"]
            )

    def test_title_keeps_gate_condition_outside_structured_metadata(self) -> None:
        file_name = "$Yuze320$~$sense$~$1.67K700nm50msx20x1$_0.95TG-BG=20_001.csv"
        expected = "Yuze320~sense~1.67K700nm50msx20x1~0.95TG-BG=20"

        with tempfile.TemporaryDirectory() as folder:
            shutil.copyfile(FIXTURES / "plain_scan.csv", Path(folder) / file_name)
            d = processing_run._load_canonical(folder, file_name, y_axis="auto")

        self.assertEqual(d["title_name"], expected)

    def test_title_does_not_duplicate_wrapped_gate_condition(self) -> None:
        file_name = "$Yuze320$~$sense$~$0.95TG-BG=20$_001.csv"
        self.assertEqual(
            processing_run._title_from_filename(file_name),
            "Yuze320~sense~0.95TG-BG=20",
        )

    def test_plain_filename_title_is_unchanged(self) -> None:
        self.assertEqual(processing_run._title_from_filename("plain_scan_001.csv"), "plain_scan_001")

    def test_auto_ratio_from_structured_metadata(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "sample$0.9TG+BG$.csv", y_axis="auto")
        expected = np.array([8.8, 8.9, 9.0])
        self.assertTrue(np.allclose(d["gate_axis"], expected))
        self.assertEqual(d["gate_label"], "0.9TG-BG (V)")

    def test_auto_bg_coefficient_from_structured_metadata(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "sample$TG+1.1BG=0$.csv", y_axis="auto")
        expected = np.array([9.8, 9.9, 10.0])
        self.assertTrue(np.allclose(d["gate_axis"], expected))
        self.assertEqual(d["gate_label"], "TG-1.1BG (V)")

    def test_auto_ratio_from_stem_fallback(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "sample_0.9TG+BG_scan.csv", y_axis="auto")
        expected = np.array([8.8, 8.9, 9.0])
        self.assertTrue(np.allclose(d["gate_axis"], expected))
        self.assertEqual(d["gate_label"], "0.9TG-BG (V)")

    def test_auto_bg_coefficient_from_stem_fallback(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "sample_TG+1.1BG=0_scan.csv", y_axis="auto")
        expected = np.array([9.8, 9.9, 10.0])
        self.assertTrue(np.allclose(d["gate_axis"], expected))
        self.assertEqual(d["gate_label"], "TG-1.1BG (V)")

    def test_auto_minus_gate_condition_from_stem_uses_complementary_plus_axis(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "sample_TG-BG=0_scan.csv", y_axis="auto")
        expected = np.array([10.0, 12.0, 14.0])
        self.assertTrue(np.allclose(d["gate_axis"], expected))
        self.assertEqual(d["gate_label"], "TG+BG (V)")

    def test_auto_minus_bg_coefficient_from_stem_uses_complementary_plus_axis(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "sample_TG-1.1BG=0_scan.csv", y_axis="auto")
        expected = np.array([10.0, 12.1, 14.2])
        self.assertTrue(np.allclose(d["gate_axis"], expected))
        self.assertEqual(d["gate_label"], "TG+1.1BG (V)")

    def test_unicode_minus_condition_uses_complementary_plus_axis(self) -> None:
        file_name = "YZ364_0Tpa_3.6KREF_700nmc_0p1sx10_TG−1.1BG=0.csv"
        with tempfile.TemporaryDirectory() as folder:
            shutil.copyfile(FIXTURES / "plain_scan.csv", Path(folder) / file_name)
            d = processing_run._load_canonical(folder, file_name, y_axis="auto")
        self.assertTrue(np.allclose(d["gate_axis"], np.array([10.0, 12.1, 14.2])))
        self.assertEqual(d["gate_label"], "TG+1.1BG (V)")

    def test_auto_minus_tg_coefficient_from_stem_uses_complementary_plus_axis(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "sample_0.9TG-BG=0_scan.csv", y_axis="auto")
        expected = np.array([9.0, 10.9, 12.8])
        self.assertTrue(np.allclose(d["gate_axis"], expected))
        self.assertEqual(d["gate_label"], "0.9TG+BG (V)")

    def test_metadata_has_priority_over_stem_fallback(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "device$bgonly$_0.9TG+BG_scan.csv", y_axis="auto")
        self.assertTrue(np.allclose(d["gate_axis"], np.array([0.0, 1.0, 2.0])))
        self.assertEqual(d["gate_label"], "BG (V)")

    def test_auto_tgonly_from_stem(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "deviceA_tgonly_5K.csv", y_axis="auto")
        self.assertTrue(np.allclose(d["gate_axis"], np.array([10.0, 11.0, 12.0])))
        self.assertEqual(d["gate_label"], "TG (V)")

    def test_auto_bgonly_from_stem(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "run3_bgonly_repeat.csv", y_axis="auto")
        self.assertTrue(np.allclose(d["gate_axis"], np.array([0.0, 1.0, 2.0])))
        self.assertEqual(d["gate_label"], "BG (V)")

    def test_auto_legacy_fallback_when_no_match(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "plain_scan.csv", y_axis="auto")
        self.assertTrue(np.allclose(d["gate_axis"], np.array([10.0, 11.0, 12.0])))
        self.assertEqual(d["gate_label"], "Top gate (V)")

    def test_manual_tg(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "plain_scan.csv", y_axis="tg")
        self.assertTrue(np.allclose(d["gate_axis"], np.array([10.0, 11.0, 12.0])))
        self.assertEqual(d["gate_label"], "TG (V)")

    def test_manual_bg(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "plain_scan.csv", y_axis="bg")
        self.assertTrue(np.allclose(d["gate_axis"], np.array([0.0, 1.0, 2.0])))
        self.assertEqual(d["gate_label"], "BG (V)")

    def test_manual_bias(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "plain_scan.csv", y_axis="bias")
        self.assertTrue(np.allclose(d["gate_axis"], np.array([0.2, 0.3, 0.4])))
        self.assertEqual(d["gate_label"], "Bias (V)")

    def test_manual_linear_combination(self) -> None:
        d = processing_run._load_canonical(str(FIXTURES), "plain_scan.csv", y_axis="linear:0.9,-1,0")
        expected = np.array([8.8, 8.9, 9.0])
        self.assertTrue(np.allclose(d["gate_axis"], expected))
        self.assertEqual(d["gate_label"], "0.9TG-BG (V)")

    def test_manual_bias_missing_column_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no bias column exists"):
            processing_run._load_canonical(str(FIXTURES), "plain_no_bias.csv", y_axis="bias")


class SharedPipelinePropagationTests(unittest.TestCase):
    def setUp(self) -> None:
        for fn in (loader._peek_y_axis_options_cached, loader._load_pl_cached):
            clear = getattr(fn, "cache_clear", None)
            if callable(clear):
                clear()

    def test_pl_loader_uses_shared_y_axis_override(self) -> None:
        with patch.object(loader.P, "process_pl") as mock_process:
            mock_process.return_value = {
                "energy": np.array([1.0, 2.0]),
                "gate_axis": np.array([0.0, 1.0]),
                "Z": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "gate_label": "TG (V)",
                "title": "Title",
            }
            loader.load_pl(str(Path(__file__).resolve().parent / "fixtures"), "sample.csv", y_axis="tg")
            self.assertEqual(mock_process.call_args.kwargs["y_axis"], "tg")

    def test_pl_loader_auto_resolves_decimal_bg_condition(self) -> None:
        file_name = "scan_TG-0.8BG=0.csv"
        with tempfile.TemporaryDirectory() as folder:
            shutil.copyfile(FIXTURES / "plain_scan.csv", Path(folder) / file_name)
            with patch.object(loader.P, "process_pl") as mock_process:
                mock_process.return_value = {
                    "energy": np.array([1.0]),
                    "gate_axis": np.array([0.0]),
                    "Z": np.array([[1.0]]),
                    "gate_label": "TG+0.8BG (V)",
                    "title": "PL",
                }
                loader.load_pl(folder, file_name, y_axis="auto")
        self.assertEqual(mock_process.call_args.kwargs["y_axis"], "linear:1,0.8,0")

    def test_drr_loader_uses_shared_y_axis_override(self) -> None:
        with patch.object(loader.P, "process_ref_avg") as mock_process:
            mock_process.return_value = {
                "energy": np.array([1.0, 2.0]),
                "gate_axis": np.array([0.0, 1.0]),
                "Z_out": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "gate_label": "BG (V)",
                "title": "DRR",
            }
            loader.load_drr_avg("folder", ["a.csv"], bg_mode="self_last", y_axis="bg")
        self.assertEqual(mock_process.call_args.kwargs["y_axis"], "bg")

    def test_compare_loader_propagates_y_axis_to_each_cube(self) -> None:
        selection = data_io.CompareSelection(kk="a.csv", kkp="b.csv")
        with patch("core.data_io.load_pl") as mock_load_pl:
            mock_load_pl.side_effect = [
                loader.DataCube(np.array([1.0]), np.array([0.0]), np.array([[1.0]]), "TG (V)", "A", "PL"),
                loader.DataCube(np.array([1.0]), np.array([0.0]), np.array([[1.0]]), "TG (V)", "B", "PL"),
            ]
            data_io.load_compare_cubes("folder", selection, y_axis="tg")
        calls = mock_load_pl.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call.kwargs["y_axis"] == "tg" for call in calls))

    def test_compare_auto_resolves_one_reference_axis_for_both_files(self) -> None:
        selection = data_io.CompareSelection(
            kk="scan_Rot220p5deg_TG-0.8BG=0.csv",
            kkp="scan_Rot265p5deg_TG+0.8BG=0.csv",
        )
        cube = loader.DataCube(
            np.array([1.0]), np.array([0.0]), np.array([[1.0]]), "TG+0.8BG (V)", "A", "PL"
        )
        with patch("core.data_io.load_pl", return_value=cube) as mock_load_pl:
            data_io.load_compare_cubes("folder", selection, y_axis="auto")
        self.assertEqual(len(mock_load_pl.call_args_list), 2)
        self.assertTrue(
            all(call.kwargs["y_axis"] == "linear:1,0.8,0" for call in mock_load_pl.call_args_list)
        )

    def test_drr_auto_resolves_filename_axis_before_processing(self) -> None:
        with patch.object(loader.P, "process_ref_avg") as mock_process:
            mock_process.return_value = {
                "energy": np.array([1.0]),
                "gate_axis": np.array([0.0]),
                "Z_out": np.array([[1.0]]),
                "gate_label": "TG+0.8BG (V)",
                "title": "DRR",
            }
            loader.load_drr_avg(
                "folder", ["scan_TG-0.8BG=0.csv"], bg_mode="self_last", y_axis="auto"
            )
        self.assertEqual(mock_process.call_args.kwargs["y_axis"], "linear:1,0.8,0")

    def test_drr_auto_resolves_unicode_minus_filename_axis(self) -> None:
        with patch.object(loader.P, "process_ref_avg") as mock_process:
            mock_process.return_value = {
                "energy": np.array([1.0]),
                "gate_axis": np.array([0.0]),
                "Z_out": np.array([[1.0]]),
                "gate_label": "TG+1.1BG (V)",
                "title": "DRR",
            }
            loader.load_drr_avg(
                "folder",
                ["YZ364_0Tpa_3.6KREF_700nmc_0p1sx10_TG−1.1BG=0.csv"],
                bg_mode="self_last",
                y_axis="auto",
            )
        self.assertEqual(mock_process.call_args.kwargs["y_axis"], "linear:1,1.1,0")


if __name__ == "__main__":
    unittest.main()
