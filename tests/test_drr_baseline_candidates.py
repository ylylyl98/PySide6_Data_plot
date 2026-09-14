from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from core import data_io
from core.drr_baseline_candidates import (
    filter_baseline_candidates,
    rank_baseline_candidates,
    spectral_grids_match,
)
from core.drr_sources import DrrSource, inspect_csv_spectral_grid, group_drr_sources


def _source(name: str, *, grid=(740.0, 760.0, 780.0), mtime=100.0, background=True, gate_ranges=()) -> DrrSource:
    return DrrSource(
        source=name,
        filename=name.rsplit("/", 1)[-1],
        group_key=name,
        session_date="2026-01-01",
        modified_time=mtime,
        is_background=background,
        spectral_grid=tuple(grid),
        gate_ranges=tuple(gate_ranges),
    )


class DrrBaselineCandidateTests(unittest.TestCase):
    def test_same_priority_baselines_use_newest_full_timestamp_not_time_distance(self) -> None:
        measurement = _source("measurement.csv", background=False, mtime=1800000000.)
        older = _source("a_back.csv", mtime=1800000001.)
        newer = _source("z_back.csv", mtime=1800000001.75)
        ranked = rank_baseline_candidates((measurement,), (older, newer))
        self.assertEqual([item.source for item in ranked], [newer, older])

    def test_group_processing_reference_order_is_independent_of_timestamp(self) -> None:
        older = replace(_source("a.csv", mtime=1800000001.), group_key="same")
        newer = replace(_source("z.csv", mtime=1800000001.75), group_key="same")
        self.assertEqual(group_drr_sources((older, newer))[0].files, (older, newer))

    def test_named_constant_background_precedes_condition_matches(self) -> None:
        measurement = _source("YZ303_p5n1_3.6KPL_0T.csv", background=False)
        match = replace(_source("YZ303_p5n1_3.6KREF_0T.csv"), gate_varies=True)
        background = replace(_source("other_BACK.csv"), gate_varies=False)
        ranked = rank_baseline_candidates((measurement,), (match, background))
        self.assertEqual([item.source for item in ranked], [background, match])
        self.assertIn("constant gate", ranked[0].reason)

    def test_background_priority_requires_name_and_confirmed_constant_gate(self) -> None:
        measurement = _source("YZ303_p5n1_3.6KPL_0T.csv", background=False)
        match = _source("YZ303_p5n1_3.6KREF_0T.csv")
        for name, varies in (("other_back.csv", True), ("other_back.csv", None),
                             ("other.csv", False), ("other_backgate.csv", False)):
            with self.subTest(name=name, varies=varies):
                candidate = replace(_source(name), gate_varies=varies)
                self.assertEqual(rank_baseline_candidates((measurement,), (candidate, match))[0].source, match)

    def test_named_constant_background_still_requires_matching_spectral_grid(self) -> None:
        measurement = _source("measurement.csv", background=False)
        background = replace(_source("other_background.csv", grid=(740., 760., 781.)), gate_varies=False)
        self.assertEqual(rank_baseline_candidates((measurement,), (background,)), ())

    def test_named_constant_backgrounds_keep_condition_order(self) -> None:
        measurement = _source("YZ303_p5n1_3.6KPL_0T.csv", background=False)
        same = replace(_source("YZ303_p5n1_3.6KREF_0T_nback.csv"), gate_varies=False)
        other = replace(_source("other_background.csv"), gate_varies=False)
        ranked = rank_baseline_candidates((measurement,), (other, same))
        self.assertEqual([item.source for item in ranked], [same, other])

    def test_grid_match_requires_same_count_order_and_values(self) -> None:
        measurement = _source("m.csv", background=False)
        self.assertTrue(spectral_grids_match(measurement, _source("same.csv")))
        self.assertFalse(spectral_grids_match(measurement, _source("count.csv", grid=(740, 780))))
        self.assertFalse(spectral_grids_match(measurement, _source("order.csv", grid=(780, 760, 740))))
        self.assertFalse(spectral_grids_match(measurement, _source("value.csv", grid=(740, 760.001, 780))))
        self.assertFalse(spectral_grids_match(measurement, _source("unknown.csv", grid=())))

    def test_filter_is_per_file_and_does_not_admit_a_mixed_grid_group(self) -> None:
        measurements = (
            _source("YZ303_pe2_3.6KPL_0T.csv", background=False),
            _source("YZ303_pe2_3.6KPL_0T_rep2.csv", background=False),
        )
        good = _source("YZ303_pe2_3.6KREF_0T.csv")
        wrong = _source("YZ303_pe2_3.6KREF_0T_rep2.csv", grid=(740, 760, 781))
        self.assertEqual(filter_baseline_candidates(measurements, (good, wrong)), (good,))

    def test_rank_prefers_explicit_sample_position_then_conditions_and_reports_mtime(self) -> None:
        measurement = _source("YZ303_pe2_3.6KPL_0T_RotOut90deg_0p08sx10.csv", background=False, mtime=100)
        same = _source("YZ303_pe2_3.6KREF_0T_RotOut90deg_0p08sx10.csv", mtime=300)
        same_conditions = _source("YZ999_pe9_3.6KREF_0T_RotOut0deg_0p08sx10.csv", mtime=101)
        unknown = _source("background.csv", mtime=102)
        ranked = rank_baseline_candidates((measurement,), (unknown, same_conditions, same))
        self.assertEqual([item.source.source for item in ranked], [same.source, same_conditions.source, unknown.source])
        self.assertIn("same sample/position", ranked[0].reason)
        self.assertIn("Modified time difference", ranked[-1].reason)
        self.assertIn("acquisition time unavailable", ranked[-1].reason)

    def test_matching_normalized_exposure_units_and_condition_priority(self) -> None:
        measurement = _source("YZ303_p5n1_3.6KPL_0T_RotOut90deg_0p08sx10.csv", background=False, mtime=100)
        same_position = _source("YZ303_p5n1_3.6KREF_0T_RotOut90deg_80msx10.csv", mtime=1)
        different_position = _source("YZ303_p9n1_3.6KREF_0T_RotOut90deg_80msx10.csv", mtime=101)
        ranked = rank_baseline_candidates((measurement,), (different_position, same_position))
        self.assertEqual([item.source.source for item in ranked], [same_position.source, different_position.source])

    def test_same_position_still_prefers_matching_temperature_before_time(self) -> None:
        measurement = _source("YZ303_p5n1_3.6KPL_0T.csv", background=False, mtime=100)
        same_position_wrong_temp = _source("YZ303_p5n1_4KREF_0T.csv", mtime=101)
        same_position_same_temp = _source("YZ303_p5n1_3.6KREF_0T.csv", mtime=300)
        ranked = rank_baseline_candidates((measurement,), (same_position_wrong_temp, same_position_same_temp))
        self.assertEqual([item.source.source for item in ranked], [same_position_same_temp.source, same_position_wrong_temp.source])

    def test_multi_measurement_conditions_are_aggregated_conservatively(self) -> None:
        measurements = (
            _source("YZ303_p5n1_3.6KPL_0T.csv", background=False),
            _source("YZ303_p5n2_3.6KPL_0T.csv", background=False),
        )
        candidate = _source("YZ303_p5n1_3.6KREF_0T.csv")
        ranked = rank_baseline_candidates(measurements, (candidate,))
        self.assertIn("unknown", ranked[0].reason.casefold())

    def test_matching_axis_can_have_different_gate_ranges(self) -> None:
        measurement = _source("m.csv", background=False, gate_ranges=((0, 1),))
        candidate = _source("b.csv", gate_ranges=((10, 20),))
        self.assertEqual(filter_baseline_candidates((measurement,), (candidate,)), (candidate,))

    def test_external_header_reader_keeps_ordered_wavelength_array(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.csv"
            path.write_text("Vbg,Vtg,740,760,780\n0,0,1,2,3\n", encoding="utf-8")
            self.assertEqual(inspect_csv_spectral_grid(path), (740.0, 760.0, 780.0))

    def test_external_loader_rejects_grid_mismatch_before_interpolation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement = root / "m_700nmc.csv"
            baseline = root / "b_760nmc.csv"
            measurement.write_text("Vbg,Vtg,740,760,780\n0,0,1,2,3\n", encoding="utf-8")
            baseline.write_text("Vbg,Vtg,740,760,781\n0,0,1,2,3\n", encoding="utf-8")
            with patch("core.data_io.build_external_baseline") as build, patch("core.data_io.load_drr_avg") as load:
                with self.assertRaisesRegex(ValueError, "exact spectral grid"):
                    data_io.load_drr_external_cube(
                        str(root), [measurement.name], [baseline.name], baseline_which="last"
                    )
                build.assert_not_called()
                load.assert_not_called()

    def test_external_loader_uses_array_when_filename_centers_disagree(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement = root / "m_700nmc.csv"
            baseline = root / "b_760nmc.csv"
            content = "Vbg,Vtg,740,760,780\n0,0,1,2,3\n"
            measurement.write_text(content, encoding="utf-8")
            baseline.write_text(content, encoding="utf-8")
            sentinel = object()
            with patch("core.data_io.build_external_baseline", return_value={"I0": np.ones(3), "energy": np.arange(3.)}) as build, patch("core.data_io.load_drr_avg", return_value=sentinel) as load:
                result = data_io.load_drr_external_cube(
                    str(root), [measurement.name], [baseline.name], baseline_which="last"
                )
            self.assertIs(result, sentinel)
            build.assert_called_once()
            load.assert_called_once()

    def test_unknown_metadata_does_not_count_as_condition_match(self) -> None:
        measurement = _source("sampleA_p5n1_3.6KPL_0T.csv", background=False)
        candidate = _source("p5n1_3.6KREF_0T.csv")
        ranked = rank_baseline_candidates((measurement,), (candidate,))
        self.assertIn("unknown", ranked[0].reason.casefold())

    def test_zero_mtime_is_unknown_time_metadata(self) -> None:
        measurement = _source("sampleA_p5n1.csv", background=False, mtime=0)
        candidate = _source("background.csv", mtime=0)
        ranked = rank_baseline_candidates((measurement,), (candidate,))
        self.assertIsNone(ranked[0].time_difference_seconds)
        self.assertIn("unavailable", ranked[0].reason.casefold())


if __name__ == "__main__":
    unittest.main()
