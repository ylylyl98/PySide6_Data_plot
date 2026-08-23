from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np

import core.loader as loader

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        for fn in (loader._peek_y_axis_options_cached, loader._load_pl_cached):
            clear = getattr(fn, "cache_clear", None)
            if callable(clear):
                clear()

    def test_csv_signature_errors_for_missing_paths(self) -> None:
        with self.assertRaises(FileNotFoundError):
            loader._csv_signature("Z:/missing/folder", "a.csv")

        with self.assertRaises(FileNotFoundError):
            loader._csv_signature(str(FIXTURES), "missing.csv")

    def test_nested_initial_data_csv_can_be_loaded_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "Initial Data" / "old session"
            nested.mkdir(parents=True)
            shutil.copy2(FIXTURES / "y_axis" / "plain_scan.csv", nested / "sample.csv")

            cube = loader.load_pl(str(root), "Initial Data/old session/sample.csv")

            self.assertGreater(cube.energy.size, 0)
            self.assertGreater(cube.gate.size, 0)

    def test_peek_y_axis_options_falls_back_to_load_canonical(self) -> None:
        with patch.object(
            loader.P,
            "_load_canonical",
            return_value={"available_axes": ["Vbg", "Vtg"], "default_axis": "Vtg"},
        ) as mock_load:
            opts, default = loader.peek_y_axis_options(str(FIXTURES), "sample.csv")

            self.assertEqual(opts, ["Vbg", "Vtg"])
            self.assertEqual(default, "Vtg")
            mock_load.assert_called_once()

    def test_load_pl_returns_cube_with_copied_arrays(self) -> None:
        energy = np.array([1.0, 2.0])
        gate = np.array([0.0, 1.0])
        z = np.array([[1.0, 2.0], [3.0, 4.0]])

        with patch.object(
            loader.P,
            "process_pl",
            return_value={
                "energy": energy,
                "gate_axis": gate,
                "Z": z,
                "gate_label": "Gate",
                "title": "Title",
            },
        ):
            cube = loader.load_pl(str(FIXTURES), "sample.csv", log_scale=False)

        self.assertTrue(np.array_equal(cube.energy, energy))
        self.assertTrue(np.array_equal(cube.gate, gate))
        self.assertTrue(np.array_equal(cube.Z, z))
        self.assertIsNot(cube.energy, energy)
        self.assertIsNot(cube.gate, gate)
        self.assertIsNot(cube.Z, z)

    def test_load_pl_raises_for_invalid_z_shape(self) -> None:
        with patch.object(
            loader.P,
            "process_pl",
            return_value={
                "energy": np.array([1.0, 2.0]),
                "gate_axis": np.array([0.0, 1.0]),
                "Z": np.array([1.0, 2.0, 3.0]),
            },
        ):
            with self.assertRaises(ValueError):
                loader.load_pl(str(FIXTURES), "sample.csv")

    def test_external_drr_baseline_uses_energy_alignment(self) -> None:
        canonical = {
            "energy": np.array([1.0, 2.0, 3.0]),
            "gate_axis": np.array([0.0]),
            "Z": np.array([[11.0, 22.0, 33.0]]),
            "gate_label": "Gate",
            "title_name": "scan",
            "stem": "scan",
        }

        with patch.object(loader.P, "_load_canonical", return_value=canonical):
            result = loader.P.process_ref_avg(
                "unused",
                ["scan.csv"],
                bg_mode="external",
                external_vector=np.array([10.0, 30.0]),
                external_energy=np.array([1.0, 3.0]),
                use_global_background=False,
                plot_interactive=False,
                save_png=False,
                save_dat_file=False,
                move_original=False,
            )

        self.assertTrue(np.allclose(result["Z_out"], np.array([[0.1, 0.1, 0.1]])))

    def test_external_drr_baseline_rejects_insufficient_energy_overlap(self) -> None:
        canonical = {
            "energy": 1240.0 / np.array([800.0, 760.0, 740.0, 700.0]),
            "gate_axis": np.array([0.0]),
            "Z": np.array([[11.0, 22.0, 33.0, 44.0]]),
            "gate_label": "Gate",
            "title_name": "scan",
            "stem": "scan",
        }

        with patch.object(loader.P, "_load_canonical", return_value=canonical):
            with self.assertRaisesRegex(ValueError, "not compatible"):
                loader.P.process_ref_avg(
                    "unused",
                    ["scan.csv"],
                    bg_mode="external",
                    external_vector=np.array([10.0, 20.0]),
                    external_energy=1240.0 / np.array([751.0, 749.0]),
                    use_global_background=False,
                    plot_interactive=False,
                    save_png=False,
                    save_dat_file=False,
                    move_original=False,
                )

    def test_external_drr_baseline_rejects_different_spectral_center(self) -> None:
        measurement_energy = np.array([1240.0 / 780.0, 1240.0 / 760.0, 1240.0 / 740.0])
        canonical = {
            "energy": measurement_energy,
            "gate_axis": np.array([0.0]),
            "Z": np.array([[11.0, 22.0, 33.0]]),
            "gate_label": "Gate",
            "title_name": "scan",
            "stem": "scan",
        }
        baseline_energy = np.array([1240.0 / 660.0, 1240.0 / 640.0, 1240.0 / 620.0])

        with patch.object(loader.P, "_load_canonical", return_value=canonical):
            with self.assertRaisesRegex(ValueError, "wavelength center"):
                loader.P.process_ref_avg(
                    "unused",
                    ["scan.csv"],
                    bg_mode="external",
                    external_vector=np.array([10.0, 20.0, 30.0]),
                    external_energy=baseline_energy,
                    use_global_background=False,
                    plot_interactive=False,
                    save_png=False,
                    save_dat_file=False,
                    move_original=False,
                )

    def test_external_background_files_must_share_spectral_center(self) -> None:
        first = {
            "energy": 1240.0 / np.array([780.0, 760.0, 740.0]),
            "Z": np.array([[1.0, 2.0, 3.0]]),
        }
        second = {
            "energy": 1240.0 / np.array([660.0, 640.0, 620.0]),
            "Z": np.array([[1.0, 2.0, 3.0]]),
        }

        with patch.object(loader.P, "_load_canonical", side_effect=[first, second]):
            with self.assertRaisesRegex(ValueError, "different wavelength centers"):
                loader.P.build_external_baseline_avg(
                    "unused", ["first.csv", "second.csv"], which="last"
                )


if __name__ == "__main__":
    unittest.main()
