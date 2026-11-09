# Adapted from Megasweep Analysis commit 2612f6f.
import unittest

import numpy as np

from core.megasweep.megasweep_analysis import find_all_cut_values


class BatchLineCutValueTests(unittest.TestCase):
    def _endpoint_adjusted_sweep(self):
        regular = np.round(-8.0 + 0.15 * np.arange(127), 6)
        axis = np.r_[regular, 11.0]
        bg, tg = np.meshgrid(axis, axis, indexing="xy")
        return bg.ravel(), tg.ravel()

    def test_doping_discovery_rejects_endpoint_only_lattice(self):
        x_data, y_data = self._endpoint_adjusted_sweep()

        filtered = find_all_cut_values(
            x_data,
            y_data,
            "doping",
            ratio=1.0,
            epsilon=0.0,
        )
        unfiltered = find_all_cut_values(
            x_data,
            y_data,
            "doping",
            ratio=1.0,
            epsilon=0.0,
            dominant_lattice_only=False,
        )

        rounded_filtered = {round(value, 6) for value in filtered}
        rounded_unfiltered = {round(value, 6) for value in unfiltered}

        self.assertIn(12.5, rounded_filtered)
        self.assertIn(12.65, rounded_filtered)
        self.assertNotIn(12.6, rounded_filtered)
        self.assertIn(12.6, rounded_unfiltered)
        self.assertEqual(251, len(filtered))

    def test_efield_discovery_keeps_regular_diagonal_family(self):
        x_data, y_data = self._endpoint_adjusted_sweep()

        values = find_all_cut_values(
            x_data,
            y_data,
            "efield",
            ratio=1.0,
            epsilon=0.0,
        )

        rounded = {round(value, 6) for value in values}
        self.assertIn(0.0, rounded)
        self.assertEqual(251, len(values))


if __name__ == "__main__":
    unittest.main()
