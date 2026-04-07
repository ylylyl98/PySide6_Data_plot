from __future__ import annotations

import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
