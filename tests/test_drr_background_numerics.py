import unittest
from unittest.mock import patch

import numpy as np

import core.loader as loader
from core.drr_sources import DrrMeasurementAssignment


def _canonical(energy, gate, values, title):
    return {
        "energy": np.asarray(energy, dtype=float),
        "gate_axis": np.asarray(gate, dtype=float),
        "Z": np.asarray(values, dtype=float),
        "gate_label": "Gate (V)",
        "title_name": title,
        "stem": title,
    }


class DrrBackgroundNumericsTests(unittest.TestCase):
    @staticmethod
    def _loader_for(canonical):
        def fake_load(_folder, name, **_kwargs):
            return canonical[name]

        return fake_load

    def test_common_resolved_result_is_exactly_the_legacy_result(self):
        canonical = {
            "a.csv": _canonical(
                [1.0, 2.0, 3.0], [0.0, 1.0],
                [[20.0, np.nan, 40.0], [10.0, 10.0, np.nan]], "a",
            ),
            "b.csv": _canonical(
                [1.0, 2.5, 3.0], [0.0, 1.0],
                [[30.0, 50.0, 60.0], [10.0, 20.0, 30.0]], "b",
            ),
            "bg.csv": _canonical(
                [1.0, 2.0, 3.0], [0.0], [[10.0, 20.0, 30.0]], "bg",
            ),
        }
        fake_load = self._loader_for(canonical)

        for mode, legacy_mode in (
            ("External", "external"),
            ("Self (first frame)", "self_first"),
            ("Self (last frame)", "self_last"),
        ):
            with self.subTest(mode=mode):
                assignments = tuple(
                    DrrMeasurementAssignment(
                        name,
                        mode,
                        ("bg.csv",) if mode == "External" else (),
                        "last",
                    )
                    for name in ("a.csv", "b.csv")
                )
                with patch.object(loader.P, "_load_canonical", side_effect=fake_load):
                    resolved = loader.load_drr_resolved_avg(
                        "unused", ["a.csv", "b.csv"], assignments=assignments,
                    )
                kwargs = {}
                if mode == "External":
                    with patch.object(loader.P, "_load_canonical", side_effect=fake_load):
                        baseline = loader.build_external_baseline("unused", ["bg.csv"])
                    kwargs.update(
                        external_vector=baseline["I0"],
                        external_energy=baseline["energy"],
                    )
                with patch.object(loader.P, "_load_canonical", side_effect=fake_load):
                    legacy = loader.load_drr_avg(
                        "unused", ["a.csv", "b.csv"], bg_mode=legacy_mode, **kwargs,
                    )

                self.assertTrue(np.array_equal(resolved.energy, legacy.energy))
                self.assertTrue(np.array_equal(resolved.gate, legacy.gate))
                self.assertTrue(np.array_equal(resolved.Z, legacy.Z, equal_nan=True))

    def test_external_multibg_uses_equal_file_weight_for_each_frame_recipe(self):
        canonical = {
            "bg_a.csv": _canonical(
                [2.0, 2.001, 2.002], [0.0, 1.0, 2.0],
                [[10.0, 20.0, 30.0], [20.0, 30.0, 40.0]], "bg_a",
            ),
            "bg_b.csv": _canonical(
                [2.0, 2.001, 2.002], [0.0, 1.0, 2.0],
                [[100.0, 200.0, 300.0], [200.0, 300.0, 400.0], [300.0, 400.0, 500.0]], "bg_b",
            ),
        }
        expected = {
            "first": np.array([55.0, 110.0, 165.0]),
            "last": np.array([160.0, 215.0, 270.0]),
            "all": np.array([107.5, 162.5, 217.5]),
        }
        fake_load = self._loader_for(canonical)
        for which, want in expected.items():
            with self.subTest(which=which), patch.object(
                loader.P, "_load_canonical", side_effect=fake_load
            ):
                result = loader.build_external_baseline(
                    "unused", ["bg_a.csv", "bg_b.csv"], which=which,
                )
            self.assertTrue(np.array_equal(result["I0"], want))

    def test_heterogeneous_path_corrects_each_member_before_energy_alignment(self):
        canonical = {
            "a.csv": _canonical([1.0, 3.0, 4.0], [0.0], [[10.0, 10.0, 10.0]], "a"),
            "b.csv": _canonical([1.0, 2.0, 4.0], [0.0], [[20.0, 100.0, 20.0]], "b"),
            "bg_a.csv": _canonical([1.0, 3.0, 4.0], [0.0], [[10.0, 10.0, 10.0]], "bg_a"),
            "bg_b.csv": _canonical([1.0, 2.0, 4.0], [0.0], [[10.0, 100.0, 10.0]], "bg_b"),
        }
        assignments = (
            DrrMeasurementAssignment("a.csv", "External", ("bg_a.csv",), "last"),
            DrrMeasurementAssignment("b.csv", "External", ("bg_b.csv",), "last"),
        )
        with patch.object(loader.P, "_load_canonical", side_effect=self._loader_for(canonical)):
            cube = loader.load_drr_resolved_avg("unused", ["a.csv", "b.csv"], assignments=assignments)

        self.assertTrue(np.allclose(cube.Z, [[0.5, 0.25, 0.5]]))
        self.assertEqual(cube.drr_numerical_path, "heterogeneous")

    def test_multibg_recipe_keeps_first_baseline_grid_before_measurement_alignment(self):
        measurement_energy = [2.0, 2.001, 2.002, 2.003, 2.004]
        canonical = {
            "a.csv": _canonical(measurement_energy, [0.0], [[20.0] * 5], "a"),
            "b.csv": _canonical(measurement_energy, [0.0], [[40.0] * 5], "b"),
            "bg_a.csv": _canonical([2.0, 2.002, 2.004], [0.0], [[10.0, 10.0, 10.0]], "bg_a"),
            "bg_b.csv": _canonical(measurement_energy, [0.0], [[10.0, 30.0, 10.0, 30.0, 10.0]], "bg_b"),
            "bg_c.csv": _canonical(measurement_energy, [0.0], [[20.0] * 5], "bg_c"),
        }
        assignments = (
            DrrMeasurementAssignment("a.csv", "External", ("bg_a.csv", "bg_b.csv"), "last"),
            DrrMeasurementAssignment("b.csv", "External", ("bg_c.csv",), "last"),
        )
        with patch.object(loader.P, "_load_canonical", side_effect=self._loader_for(canonical)):
            cube = loader.load_drr_resolved_avg("unused", ["a.csv", "b.csv"], assignments=assignments)

        self.assertTrue(np.allclose(cube.Z, np.ones((1, 5))))

    def test_heterogeneous_result_preserves_reversed_energy_and_gate_axes(self):
        canonical = {
            "a.csv": _canonical(
                [3.0, 2.0, 1.0], [1.0, 0.0],
                [[20.0, np.nan, 60.0], [10.0, np.nan, 30.0]], "a",
            ),
            "bg.csv": _canonical(
                [3.0, 2.0, 1.0], [0.0], [[10.0, 20.0, 30.0]], "bg_a",
            ),
            "b.csv": _canonical(
                [3.0, 2.0, 1.0], [0.0, 1.0],
                [[40.0, np.nan, 20.0], [20.0, np.nan, 20.0]], "b",
            ),
            "bg_b.csv": _canonical(
                [3.0, 2.0, 1.0], [0.0], [[20.0, 40.0, 10.0]], "bg_b",
            ),
        }
        assignments = (
            DrrMeasurementAssignment("a.csv", "External", ("bg.csv",), "last"),
            DrrMeasurementAssignment("b.csv", "External", ("bg_b.csv",), "last"),
        )
        with patch.object(loader.P, "_load_canonical", side_effect=self._loader_for(canonical)):
            cube = loader.load_drr_resolved_avg("unused", ["a.csv", "b.csv"], assignments=assignments)

        self.assertTrue(np.array_equal(cube.energy, np.array([3.0, 2.0, 1.0])))
        self.assertTrue(np.array_equal(cube.gate, np.array([1.0, 0.0])))
        self.assertTrue(np.array_equal(cube.Z, [[0.5, np.nan, 1.0], [0.5, np.nan, 0.5]], equal_nan=True))
        self.assertEqual(cube.drr_numerical_path, "heterogeneous")

    def test_heterogeneous_external_baseline_with_no_energy_overlap_is_rejected(self):
        energy = [2.0, 2.0001, 2.0002]
        canonical = {
            "a.csv": _canonical(energy, [0.0], [[20.0] * 3], "a"),
            "b.csv": _canonical(energy, [0.0], [[40.0] * 3], "b"),
            "bg_good.csv": _canonical(energy, [0.0], [[10.0] * 3], "bg_good"),
            "bg_bad.csv": _canonical([2.0003, 2.0004], [0.0], [[20.0, 20.0]], "bg_bad"),
        }
        assignments = (
            DrrMeasurementAssignment("a.csv", "External", ("bg_good.csv",), "last"),
            DrrMeasurementAssignment("b.csv", "External", ("bg_bad.csv",), "last"),
        )
        with patch.object(loader.P, "_load_canonical", side_effect=self._loader_for(canonical)):
            with self.assertRaisesRegex(ValueError, "overlap|coverage|compatible"):
                loader.load_drr_resolved_avg("unused", ["a.csv", "b.csv"], assignments=assignments)

    def test_heterogeneous_measurement_outside_reference_grid_is_rejected(self):
        first_energy = [2.0, 2.0001, 2.0002]
        second_energy = [2.0003, 2.0004, 2.0005]
        canonical = {
            "a.csv": _canonical(first_energy, [0.0], [[20.0] * 3], "a"),
            "b.csv": _canonical(second_energy, [0.0], [[40.0] * 3], "b"),
            "bg_a.csv": _canonical(first_energy, [0.0], [[10.0] * 3], "bg_a"),
            "bg_b.csv": _canonical(second_energy, [0.0], [[20.0] * 3], "bg_b"),
        }
        assignments = (
            DrrMeasurementAssignment("a.csv", "External", ("bg_a.csv",), "last"),
            DrrMeasurementAssignment("b.csv", "External", ("bg_b.csv",), "last"),
        )
        with patch.object(loader.P, "_load_canonical", side_effect=self._loader_for(canonical)):
            with self.assertRaisesRegex(ValueError, "overlap|coverage|compatible"):
                loader.load_drr_resolved_avg("unused", ["a.csv", "b.csv"], assignments=assignments)

    def test_identical_multibg_content_with_different_recipes_stays_on_common_path(self):
        canonical = {
            "a.csv": _canonical(
                [1.0, 2.0, 3.0], [0.0], [[20.0, 40.0, 60.0]], "a",
            ),
            "b.csv": _canonical(
                [1.0, 2.5, 3.0], [0.0], [[40.0, 60.0, 80.0]], "b",
            ),
            "bg_first.csv": _canonical(
                [1.0, 2.0, 3.0], [0.0, 1.0],
                [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], "bg_first",
            ),
            "bg_all.csv": _canonical(
                [1.0, 2.0, 3.0], [0.0, 1.0],
                [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], "bg_all",
            ),
        }
        assignments = (
            DrrMeasurementAssignment("a.csv", "External", ("bg_first.csv",), "first"),
            DrrMeasurementAssignment("b.csv", "External", ("bg_all.csv",), "all"),
        )
        fake_load = self._loader_for(canonical)
        with patch.object(loader.P, "_load_canonical", side_effect=fake_load):
            resolved = loader.load_drr_resolved_avg(
                "unused", ["a.csv", "b.csv"], assignments=assignments,
            )
            baseline = loader.build_external_baseline("unused", ["bg_first.csv"], which="first")
            legacy = loader.load_drr_avg(
                "unused", ["a.csv", "b.csv"], bg_mode="external",
                external_vector=baseline["I0"], external_energy=baseline["energy"],
            )

        self.assertEqual(resolved.drr_numerical_path, "common")
        self.assertTrue(np.array_equal(resolved.Z, legacy.Z, equal_nan=True))

    def test_mixed_self_and_external_assignments_average_on_first_member_grid(self):
        canonical = {
            "a.csv": _canonical(
                [1.0, 2.0, 3.0], [0.0, 1.0],
                [[20.0, 40.0, 20.0], [10.0, 20.0, 10.0]], "a",
            ),
            "b.csv": _canonical(
                [1.0, 2.5, 3.0], [0.0, 1.0],
                [[40.0, 40.0, 40.0], [40.0, 40.0, 40.0]], "b",
            ),
            "bg.csv": _canonical(
                [1.0, 2.5, 3.0], [0.0], [[20.0, 20.0, 20.0]], "bg",
            ),
        }
        assignments = (
            DrrMeasurementAssignment("a.csv", "Self (last frame)"),
            DrrMeasurementAssignment("b.csv", "External", ("bg.csv",), "last"),
        )
        with patch.object(loader.P, "_load_canonical", side_effect=self._loader_for(canonical)):
            cube = loader.load_drr_resolved_avg("unused", ["a.csv", "b.csv"], assignments=assignments)

        self.assertTrue(np.allclose(cube.Z, [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]))
        self.assertTrue(np.array_equal(cube.energy, np.array([1.0, 2.0, 3.0])))


if __name__ == "__main__":
    unittest.main()
