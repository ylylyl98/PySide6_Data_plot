"""Manual P/N time comparison must never bind sequence numbers or pick a file."""
from dataclasses import fields, replace
from pathlib import Path
import unittest
from unittest.mock import patch

from core import drr_pn_selection as selection
from core.drr_pn_selection import (
    TIME_LABEL, build_pn_source_catalog, describe_pn_pair, describe_pn_counterparts,
)
from core.drr_sources import DrrSource
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


def source(role, *, suffix="_001", time=1_700_000_000.0, gates=None, directory="Initial Data/REF", **changes):
    token = "data" if role == "sample" else role
    name = f"YZ383_p2n1_1.67K_9T_RotIn49deg_730nmc_0.3sx10_{token}_PIXIS{suffix}.csv"
    if gates is None:
        gates = (-10., -10.) if role.casefold().startswith("pback") else (10., 10.)
    row = (*gates, 0.)
    item = DrrSource(f"{directory}/{name}", name, "catalog-group", "2026-10-20", time,
        role not in {"sample", "data"}, gate_varies=False, frame_count=4,
        spectral_grid=(700., 710., 720.), gate_labels=("Vbg", "Vtg", "Vbias"),
        gate_grid=(row,) * 4, gate_ranges=tuple((value, value) for value in row),
        wavelength_center_nm=730., grid_complete=True)
    return replace(item, **changes)


def row_for(entries, item):
    return next(entry for entry in entries if entry.source.source == item.source)


class PnSourceSelectionTests(unittest.TestCase):
    def test_each_repeat_remains_one_time_sorted_row_unknown_last(self):
        p1, p2 = source("Pbackground", suffix="_001", time=100), source("Pbackground", suffix="_002", time=300)
        n = source("Nbackground", time=200)
        unknown = source("Pbackground", suffix="_003", time=float("nan"))
        entries = build_pn_source_catalog([p1, n, unknown, p2, p1])
        self.assertEqual([entry.source for entry in entries], [p2, n, p1, unknown])
        self.assertTrue(all(entry.time_label == TIME_LABEL for entry in entries))
        self.assertIsNone(entries[-1].modified_time)
        self.assertEqual(len([entry for entry in entries if entry.role == "P"]), 3)

    def test_equal_time_order_is_stable_without_choice_or_binding_fields(self):
        p, n1, n2 = source("Pbackground"), source("Nbackground"), source("Nback")
        first = build_pn_source_catalog([n1, p, n2])
        self.assertEqual(first, build_pn_source_catalog([n2, p, n1]))
        comparisons = describe_pn_counterparts(row_for(first, p), first)
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(item.time_difference_seconds == 0 for item in comparisons))
        self.assertNotIn("auto_source", {field.name for field in fields(type(comparisons[0]))})
        self.assertNotIn("acquisition_suffix", {field.name for field in fields(type(first[0]))})
        self.assertFalse(hasattr(selection, "match_pn_counterpart"))

    def test_explicit_role_and_gate_conflicts_are_reported_without_reassignment(self):
        for gates in ((10., 10.), (-10., 10.)):
            with self.subTest(gates=gates):
                p, n = source("Pbackground", gates=gates), source("Nbackground")
                entries = build_pn_source_catalog([p, n])
                named = row_for(entries, p)
                self.assertEqual(named.role, "P")
                self.assertEqual(named.role_basis, "explicit_name")
                self.assertTrue(named.role_conflict)
                self.assertIn("name/gate", named.reason)
                comparison = describe_pn_pair(named, row_for(entries, n))
                self.assertIn("role conflicts", comparison.reason)
                self.assertEqual(comparison.entry.source, n)

    def test_anonymous_background_needs_both_named_constant_gate_signs(self):
        p = source("background", gates=(-5., -7.), suffix="_001")
        n = source("background", gates=(5., 7.), suffix="_002")
        entries = build_pn_source_catalog([p, n])
        self.assertEqual(row_for(entries, p).role, "P")
        self.assertEqual(row_for(entries, n).role, "N")
        self.assertTrue(all(entry.role_basis == "constant_dual_gate" for entry in entries))
        for changes in (
            {"gate_labels": ("gate 1", "gate 2", "bias")},
            {"gate_grid": ((-5., 0., 0.),) * 4},
            {"gate_grid": ((-5., 7., 0.),) * 4},
            {"gate_grid": ((-5., -7., 0.), (-6., -7., 0.))},
            {"gate_varies": None}, {"grid_complete": False},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(build_pn_source_catalog([replace(p, **changes)]), ())

    def test_explicit_names_preserve_zero_and_unknown_gate_polarity(self):
        for p in (source("Pbackground", gates=(-10., 0.)),
                  source("Pbackground", gate_labels=(), gate_grid=())):
            entry = build_pn_source_catalog([p])[0]
            self.assertEqual(entry.role, "P")
            self.assertEqual(entry.role_basis, "explicit_name")
            self.assertFalse(entry.role_conflict)
            self.assertIn("polarity", entry.reason)

    def test_BN_samples_sweeps_and_saved_products_are_excluded(self):
        objects = [source("BNbackground"), source("BN"), source("data", is_background=True),
                   source("sample"), source("background", is_background=False, gate_varies=True),
                   source("Pbackground", directory="Processed Data/DRR/PN Backgrounds"),
                   source("Pbackground", classification_reason="saved composite background"),
                   source("Pbackground", metadata_role="measurement"),
                   source("Pbackground_Nbackground")]
        self.assertEqual(build_pn_source_catalog(objects), ())
        self.assertEqual(len(build_pn_source_catalog([source("Pbackground", processed=True)])), 1)

    def test_measurement_rows_are_excluded_and_known_conflicts_stay_visible(self):
        sample, p, n = source("sample"), source("Pbackground"), source("Nbackground")
        mismatch = replace(n, source=n.source.replace("49deg", "94deg"), filename=n.filename.replace("49deg", "94deg"))
        entries = build_pn_source_catalog([sample, p, n, mismatch], [sample])
        self.assertEqual(len(entries), 3)
        self.assertTrue(row_for(entries, p).matches_measurements)
        self.assertFalse(row_for(entries, mismatch).matches_measurements)
        self.assertIn("rotation", row_for(entries, mismatch).reason)
        comparisons = describe_pn_counterparts(row_for(entries, p), entries)
        self.assertEqual(len(comparisons), 2)
        bad = next(item for item in comparisons if item.entry.source == mismatch)
        self.assertTrue(bad.condition_differences)

    def test_cross_001_002_is_available_with_its_actual_time_difference(self):
        p = source("Pbackground", suffix="_001")
        near = source("Nbackground", suffix="_002", time=p.modified_time + 10)
        far_same_number = source("Nbackground", suffix="_001", time=p.modified_time + 1000)
        entries = build_pn_source_catalog([p, near, far_same_number])
        comparisons = describe_pn_counterparts(row_for(entries, p), entries)
        # Original modified-time order is preserved, not reordered by suffix or proximity.
        self.assertEqual([item.entry.source for item in comparisons], [far_same_number, near])
        self.assertEqual([item.time_difference_seconds for item in comparisons], [1000, 10])
        self.assertTrue(all(item.grid_matches and not item.condition_differences for item in comparisons))
        chosen = describe_pn_pair(row_for(entries, p), row_for(entries, near))
        self.assertEqual(chosen.entry.source, near)
        self.assertIn("manual choice", chosen.reason)

    def test_dates_directories_unrelated_numbers_and_long_gap_do_not_filter(self):
        p = source("Pbackground", suffix="_rep2_001")
        n = source("Nbackground", suffix="_999", directory="other-run", session_date="2026-10-25",
                   time=p.modified_time + 5 * 86400)
        entries = build_pn_source_catalog([p, n])
        comparisons = describe_pn_counterparts(row_for(entries, p), entries)
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0].entry.source, n)
        self.assertEqual(comparisons[0].time_difference_seconds, 5 * 86400)
        self.assertFalse(comparisons[0].condition_differences)

    def test_unknown_times_stay_unknown_not_zero(self):
        p = source("Pbackground")
        for value in (None, 0, float("nan"), float("inf")):
            n = source("Nbackground", time=value)
            entries = build_pn_source_catalog([p, n])
            comparison = describe_pn_pair(row_for(entries, p), row_for(entries, n))
            self.assertIsNone(comparison.time_difference_seconds)
            self.assertIn("difference unknown", comparison.reason)
        n = source("Nbackground", time=p.modified_time - 17)
        entries = build_pn_source_catalog([p, n])
        self.assertEqual(describe_pn_pair(row_for(entries, p), row_for(entries, n)).time_difference_seconds, 17)
        self.assertEqual(describe_pn_pair(row_for(entries, n), row_for(entries, p)).time_difference_seconds, 17)

    def test_grid_and_known_condition_conflicts_are_comparison_data_not_choice(self):
        p, n = source("Pbackground"), source("Nbackground")
        for changes, grid_ok in (({"spectral_grid": (720., 710., 700.)}, False),
                                 ({"spectral_grid": ()}, False),
                                 ({"filename": n.filename.replace("9T", "8T")}, True)):
            entries = build_pn_source_catalog([p, replace(n, **changes)])
            comparison = describe_pn_counterparts(row_for(entries, p), entries)[0]
            self.assertEqual(comparison.grid_matches, grid_ok)
            if grid_ok:
                self.assertTrue(comparison.condition_differences)
                self.assertIn("known condition conflicts", comparison.reason)
            else:
                self.assertIn("grid differs or is unavailable", comparison.reason)

    def test_unknown_conditions_do_not_create_false_Pbackground_position(self):
        p = replace(source("Pbackground"), filename="Pbackground_001.csv")
        n = replace(source("Nbackground"), filename="Nbackground_999.csv")
        entries = build_pn_source_catalog([p, n], [source("sample")])
        self.assertTrue(all(entry.matches_measurements for entry in entries))
        self.assertIn("position", entries[0].unknown_conditions)
        comparison = describe_pn_pair(row_for(entries, p), row_for(entries, n))
        self.assertFalse(comparison.condition_differences)
        self.assertIn("rotation", comparison.unknown_conditions)
        self.assertIn("not acquisition time", comparison.reason)

    def test_partial_and_unequal_frame_counts_remain_visible(self):
        p, n = source("Pbackground"), source("Nbackground")
        for count, expected in ((None, None), (2, False), (15, False), (6, False), (4, True)):
            entries = build_pn_source_catalog([p, replace(n, frame_count=count)])
            comparison = describe_pn_counterparts(row_for(entries, p), entries)[0]
            self.assertEqual(comparison.frame_counts_match, expected)
            self.assertIn("does not prove acquisition", row_for(entries, p).reason)

    def test_catalog_and_comparisons_do_no_io_and_leave_inputs_unchanged(self):
        p, n, sample = source("Pbackground"), source("Nbackground"), source("sample")
        inputs = [p, n]
        before = tuple(inputs)
        with patch("builtins.open", side_effect=AssertionError("No files may be read")), \
                patch.object(Path, "stat", side_effect=AssertionError("No stat")), \
                patch.object(Path, "resolve", side_effect=AssertionError("No path I/O")):
            entries = build_pn_source_catalog(inputs, [sample])
            comparison = describe_pn_counterparts(row_for(entries, p), entries)[0]
            self.assertEqual(comparison.entry.source, n)
        self.assertEqual(tuple(inputs), before)

    def test_callback_and_existing_load_scope_cancellation(self):
        p, n = source("Pbackground"), source("Nbackground")
        entries = build_pn_source_catalog([p, n])
        with self.assertRaises(LoadCancelled):
            build_pn_source_catalog([p, n], cancelled=lambda: True)
        with self.assertRaises(LoadCancelled):
            describe_pn_counterparts(entries[0], entries, cancelled=lambda: True)
        with self.assertRaises(LoadCancelled):
            describe_pn_pair(row_for(entries, p), row_for(entries, n), cancelled=lambda: True)
        calls = 0
        def cancel_during_gate_scan():
            nonlocal calls
            calls += 1
            return calls == 5
        with self.assertRaises(LoadCancelled):
            build_pn_source_catalog([p, n], cancelled=cancel_during_gate_scan)
        token = LoadCancellation()
        with load_scope(token):
            token.cancel()
            with self.assertRaises(LoadCancelled):
                build_pn_source_catalog([p, n])

    def test_only_independent_opposite_roles_form_a_pair_description(self):
        entries = build_pn_source_catalog([source("Pbackground"), source("Pbackground", suffix="_002")])
        self.assertEqual(describe_pn_counterparts(entries[0], entries), ())
        with self.assertRaises(ValueError):
            describe_pn_pair(entries[0], entries[1])


if __name__ == "__main__":
    unittest.main()
