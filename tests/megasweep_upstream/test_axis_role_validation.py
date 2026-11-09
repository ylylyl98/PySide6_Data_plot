# Adapted from Megasweep Analysis commit 2612f6f.
import os
import tempfile
import unittest

import pandas as pd

from core.megasweep.megasweep_analysis import (
    classify_axis_role,
    guess_sweep_axis_columns,
    load_megasweep_csv,
    validate_axis_selection,
)


class AxisRoleValidationTests(unittest.TestCase):
    def test_classifies_realistic_gate_axis_names(self):
        self.assertEqual("tg", classify_axis_role("vtg_axis_a_set"))
        self.assertEqual("bg", classify_axis_role("vbg_axis_b_set"))

    def test_classifies_realistic_transformed_axis_names(self):
        self.assertEqual("doping", classify_axis_role("doping_set"))
        self.assertEqual("efield", classify_axis_role("efield_set"))

    def test_axis_order_markers_do_not_override_physical_roles(self):
        self.assertEqual("doping", classify_axis_role("Doping_axis_a_set"))
        self.assertEqual("vbias", classify_axis_role("Vbias_axis_b_set"))

    def test_accepts_expected_raw_gate_assignment(self):
        result = validate_axis_selection("vbg_axis_b_set", "vtg_axis_a_set")
        self.assertTrue(result["ok"])
        self.assertEqual("raw_gate", result["mode"])

    def test_blocks_swapped_raw_gate_assignment(self):
        result = validate_axis_selection("vtg_axis_a_set", "vbg_axis_b_set")
        self.assertFalse(result["ok"])
        self.assertEqual("swapped_raw_gate", result["mode"])

    def test_accepts_expected_transformed_assignment(self):
        result = validate_axis_selection("doping_set", "efield_set")
        self.assertTrue(result["ok"])
        self.assertEqual("transformed", result["mode"])

    def test_blocks_swapped_transformed_assignment(self):
        result = validate_axis_selection("efield_set", "doping_set")
        self.assertFalse(result["ok"])
        self.assertEqual("swapped_transformed", result["mode"])

    def test_accepts_doping_bias_sweep(self):
        result = validate_axis_selection("Doping_axis_a_set", "Vbias_axis_b_set")
        self.assertTrue(result["ok"])
        self.assertEqual("doping_bias", result["mode"])

    def test_blocks_swapped_doping_bias_sweep(self):
        result = validate_axis_selection("Vbias_axis_b_set", "Doping_axis_a_set")
        self.assertFalse(result["ok"])
        self.assertEqual("swapped_doping_bias", result["mode"])

    def test_prefers_explicit_sweep_axes_over_derived_metadata(self):
        columns = [
            "Doping_axis_a_set",
            "Vbias_axis_b_set",
            "Vtg_set",
            "Vbg_set",
            "Vbias_set",
            "Doping_set",
            "Efield_set",
            "Vbg_meas",
            "Vtg_meas",
            "Vbias_meas",
            "Ibg_A",
            "Itg_A",
            "Ibias_A",
        ]
        self.assertEqual(
            ("Doping_axis_a_set", "Vbias_axis_b_set"),
            guess_sweep_axis_columns(columns),
        )

    def test_preserves_supported_raw_gate_orientation(self):
        self.assertEqual(
            ("Vbg_axis_b_set", "Vtg_axis_a_set"),
            guess_sweep_axis_columns(["Vtg_axis_a_set", "Vbg_axis_b_set"]),
        )

    def test_loaded_doping_bias_axes_are_generic_not_de_transformed(self):
        frame = pd.DataFrame({
            "Doping_axis_a_set": [-2.0, -2.0, -1.0, -1.0],
            "Vbias_axis_b_set": [0.0, -0.005, 0.0, -0.005],
            "Doping_set": [-2.0, -2.0, -1.0, -1.0],
            "Efield_set": [30.0, 30.0, 30.0, 30.0],
            "657.957": [1.0, 2.0, 3.0, 4.0],
            "658.02": [5.0, 6.0, 7.0, 8.0],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "doping_bias.csv")
            frame.to_csv(path, index=False)
            data = load_megasweep_csv(
                path,
                x_col="Doping_axis_a_set",
                y_col="Vbias_axis_b_set",
            )

        self.assertEqual("generic", data["axis_space"])
        self.assertEqual(os.path.abspath(path), data["source_path"])
        self.assertEqual(2, len(set(data["x_data"])))
        self.assertEqual(2, len(set(data["y_data"])))

    def test_allows_unknown_custom_assignment_with_warning_mode(self):
        result = validate_axis_selection("axis_0", "axis_1")
        self.assertTrue(result["ok"])
        self.assertEqual("unknown", result["mode"])


if __name__ == "__main__":
    unittest.main()
