"""Regressions for mixed sampling grids and repeats acquired across midnight."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from core.drr_sources import (
    compatible_drr_repeats, discover_drr_sources, group_drr_sources,
)


class DrrGroupGridTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def write_csv(self, name, gates, axis=(740, 760, 780)):
        (self.root / name).write_text(
            "Vbg,Vtg," + ",".join(map(str, axis)) + "\n"
            + "".join(f"{gate},0,1,2,3\n" for gate in gates), encoding="utf-8",
        )

    def catalog(self):
        return discover_drr_sources(self.root, include_history=False)

    @staticmethod
    def members(groups):
        return {frozenset(s.filename for s in g.files) for g in groups}

    def test_same_range_with_different_step_counts_stays_separate(self):
        self.write_csv("sample.csv", (-1, 0, 1))
        self.write_csv("sample_rep01.csv", (-1, -.5, 0, .5, 1))
        self.write_csv("sample_rep02.csv", (-1, -.5, 0, .5, 1))
        groups = group_drr_sources(self.catalog())
        self.assertEqual(self.members(groups), {
            frozenset({"sample.csv"}),
            frozenset({"sample_rep01.csv", "sample_rep02.csv"}),
        })
        self.assertEqual(len({g.key for g in groups}), 2)

    def test_same_count_range_and_direction_do_not_hide_different_interior_points(self):
        self.write_csv("sample_rep01.csv", (-1, 0, 1))
        self.write_csv("sample_rep02.csv", (-1, .25, 1))
        self.assertEqual(len(group_drr_sources(self.catalog())), 2)

    def test_same_gate_grid_with_different_spectral_sampling_stays_separate(self):
        self.write_csv("sample_rep01.csv", (-1, 0, 1))
        self.write_csv("sample_rep02.csv", (-1, 0, 1), axis=(740, 761, 780))
        self.assertEqual(len(group_drr_sources(self.catalog())), 2)

    def test_complete_repeats_stay_together_across_midnight_and_catalog_reordering(self):
        for i in range(1, 5):
            self.write_csv(f"sample_rep0{i}.csv", (-1, 0, 1))
        catalog = [replace(s, session_date="2026-10-04" if s.filename < "sample_rep03.csv"
                           else "2026-10-05", modified_time=float(i),
                           processed=s.filename == "sample_rep01.csv")
                   for i, s in enumerate(sorted(self.catalog(), key=lambda s: s.filename))]
        before = group_drr_sources(catalog[:2])
        groups = group_drr_sources(catalog)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].files), 4)
        self.assertEqual(groups[0].processed_count, 1)
        self.assertEqual(groups[0].session_date, "2026-10-05")
        self.assertEqual(groups[0].key, before[0].key)
        self.assertEqual(groups, group_drr_sources(list(reversed(catalog))))

    def test_incomplete_grid_is_not_added_to_a_complete_group(self):
        self.write_csv("sample_rep01.csv", (-1, 0, 1))
        self.write_csv("sample_rep02.csv", (-1, 0, 1))
        catalog = [replace(s, grid_complete=False) if s.filename.endswith("02.csv") else s
                   for s in self.catalog()]
        self.assertEqual(len(group_drr_sources(catalog)), 2)

    def test_background_sessions_keep_dates_for_time_based_baseline_matching(self):
        self.write_csv("sample_back_rep01.csv", (0, 0, 0))
        self.write_csv("sample_back_rep02.csv", (0, 0, 0))
        catalog = [replace(s, session_date="2026-10-04" if s.filename.endswith("01.csv")
                           else "2026-10-05") for s in self.catalog()]
        groups = group_drr_sources(catalog)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(group.is_background for group in groups))
        self.assertEqual({group.session_date for group in groups}, {"2026-10-04", "2026-10-05"})

    def test_group_grid_tolerance_matches_compatible_repeats(self):
        self.write_csv("sample_rep01.csv", (-1, 0, 1))
        self.write_csv("sample_rep02.csv", (-1 + 5e-10, 0, 1 + 5e-10))
        catalog = self.catalog()
        self.assertEqual(len(compatible_drr_repeats(catalog, "sample_rep01.csv")), 2)
        self.assertEqual(len(group_drr_sources(catalog)), 1)

    def test_growing_file_joins_repeats_only_after_its_grid_matches(self):
        self.write_csv("sample_rep01.csv", (-1, 0, 1))
        self.write_csv("sample_rep02.csv", (-1, 0))
        self.assertEqual(len(group_drr_sources(self.catalog())), 2)
        self.write_csv("sample_rep02.csv", (-1, 0, 1))
        self.assertEqual(len(group_drr_sources(self.catalog())), 1)


if __name__ == "__main__":
    unittest.main()
