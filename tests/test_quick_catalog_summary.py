"""Presentation uses complete cached traces without changing scientific grouping."""
from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

from core.drr_sources import DrrSource, group_drr_sources
from ui_qt.quick.catalog_summary import (
    format_frame_count,
    group_catalog_detail,
    group_gate_summary,
    source_catalog_detail,
    source_gate_summary,
)


def _source(**changes):
    source = DrrSource(
        source="a.csv", filename="a.csv", group_key="sample", session_date="2026-10-24",
        modified_time=1., is_background=False, classification_reason="Measurement",
        frame_count=3, gate_labels=("Vtg_set", "Vtg_meas", "Vbg"),
        gate_grid=((-1., -1., 0.), (0., 0., 0.), (1., 1., 0.)),
        gate_ranges=((-1., 1.), (-1., 1.), (0., 0.)),
        gate_direction="Vtg_set increasing, Vtg_meas increasing, Vbg constant",
        spectral_grid=(700., 701.), wavelength_center_nm=700., grid_complete=True,
    )
    return replace(source, **changes)


class QuickCatalogSummaryTests(TestCase):
    def test_certified_equal_pair_shows_one_gate_and_constant_value(self):
        source = _source()
        with patch("builtins.open", side_effect=AssertionError("Summary must not read files")):
            self.assertEqual(source_gate_summary(source), "Vtg: -1–1 ↑; Vbg: 0")
            self.assertEqual(source_catalog_detail(source),
                "2026-10-24 · Measurement\nVtg: -1–1 ↑; Vbg: 0 · 3 frames · 700 nm")
        self.assertEqual(source.gate_labels, ("Vtg_set", "Vtg_meas", "Vbg"))
        self.assertEqual(source.gate_grid[0], (-1., -1., 0.))

    def test_same_bounds_and_direction_do_not_hide_different_interior_values(self):
        source = _source(gate_grid=((-1., -1., 0.), (0., .25, 0.), (1., 1., 0.)))
        self.assertEqual(source_gate_summary(source),
            "Vtg set: -1–1 ↑; Vtg measured: -1–1 ↑; Vbg: 0")

    def test_different_ranges_remain_distinct(self):
        source = _source(gate_grid=((-1., -.9, 0.), (0., .1, 0.), (1., 1.1, 0.)),
            gate_ranges=((-1., 1.), (-.9, 1.1), (0., 0.)))
        self.assertIn("Vtg measured: -0.9–1.1 ↑", source_gate_summary(source))
        self.assertIn("Vtg set: -1–1 ↑", source_gate_summary(source))

    def test_unknown_partial_malformed_or_nonfinite_grid_cannot_certify_pair(self):
        for changes in (
            {"gate_grid": ()}, {"grid_complete": False}, {"frame_count": None},
            {"frame_count": 4}, {"gate_grid": ((0., 0.),) * 3},
            {"gate_grid": ((float("nan"), float("nan"), 0.),) * 3},
            {"gate_grid": ((float("inf"), float("inf"), 0.),) * 3},
        ):
            with self.subTest(changes=changes):
                result = source_gate_summary(_source(**changes))
                self.assertIn("Vtg set:", result)
                self.assertIn("Vtg measured:", result)

    def test_label_spelling_and_order_keep_other_channels(self):
        source = _source(gate_labels=("Vbias measured", "Vbias setpoint", "Vtg"),
            gate_direction="Vbias measured decreasing, Vbias setpoint decreasing, Vtg constant")
        self.assertEqual(source_gate_summary(source), "Vbias: -1–1 ↓; Vtg: 0")
        self.assertIn("Vtg set:", source_gate_summary(_source(gate_labels=("Vtg_set", "Vbg_meas", "Vbg"))))

    def test_group_requires_certification_in_every_member(self):
        source = _source()
        group = group_drr_sources([source, replace(source, source="b.csv", filename="b.csv")])[0]
        self.assertEqual(group_gate_summary(group), "Vtg: -1–1 ↑; Vbg: 0")
        incomplete = replace(source, source="b.csv", grid_complete=False)
        with self.subTest("A later member can withhold certification"):
            result = group_gate_summary(replace(group, files=(source, incomplete)))
            self.assertIn("Vtg set:", result)
            self.assertIn("Vtg measured:", result)
        different = replace(source, gate_grid=((-1., -1., 0.), (0., .25, 0.), (1., 1., 0.)))
        self.assertIn("Vtg measured:", group_gate_summary(replace(group, files=(source, different))))
        self.assertIn("Vtg measured:", group_gate_summary(replace(group, files=())))

    def test_presentation_does_not_merge_scientific_groups(self):
        source = _source()
        different = replace(source, source="b.csv", filename="b.csv",
            gate_grid=((-1., -1., 0.), (.25, .25, 0.), (1., 1., 0.)))
        groups = group_drr_sources([source, different])
        self.assertEqual(len(groups), 2)
        keys = [group.key for group in groups]
        self.assertEqual([group_gate_summary(group) for group in groups], ["Vtg: -1–1 ↑; Vbg: 0"] * 2)
        self.assertEqual([group.key for group in groups], keys)

    def test_frame_counts_preserve_zero_and_collapse_equal_range(self):
        for value, expected in (
            ((541, 541), "541 frames"), ((540, 541), "540–541 frames"),
            (541, "541 frames"), (0, "0 frames"), (1, "1 frame"), (None, "? frames"),
        ):
            with self.subTest(value=value):
                self.assertEqual(format_frame_count(value), expected)
        group = group_drr_sources([_source()])[0]
        self.assertEqual(group_catalog_detail(replace(group, frame_count_range=(541, 541))),
            "Vtg: -1–1 ↑; Vbg: 0 · 541 frames")

    def test_missing_metadata_and_legacy_direction_still_have_useful_detail(self):
        source = _source(gate_labels=(), gate_ranges=(), gate_grid=(), gate_direction="mixed",
            frame_count=None, wavelength_center_nm=None, session_date="", classification_reason="")
        self.assertEqual(source_catalog_detail(source), "mixed · ? frames · ? nm")
