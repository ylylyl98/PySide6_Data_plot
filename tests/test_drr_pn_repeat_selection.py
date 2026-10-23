"""Deterministic P/N defaults use only the current measurement selection."""
from dataclasses import replace
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_reconstruction import app
from core.drr_sources import DrrSource
from ui_qt.drr_pn_repeat_selection import choose_training_repeats


class TrainingRepeatSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def choose(self, names, **kwargs):
        return choose_training_repeats(self.root, names, **kwargs)

    def source(self, name, **overrides):
        source = DrrSource(name, Path(name).name, "group", "2026-10-21", 10., False)
        return replace(source, **overrides)

    def test_default_first_and_third_numbered_repeats_ignore_selection_order(self):
        names = ("sample_rep01.csv", "sample_rep02.csv", "sample_rep03_007.csv", "sample_rep10.csv")
        for order in itertools.permutations(names):
            with self.subTest(order=order):
                result = self.choose(order)
                self.assertTrue(result.usable, result.reason)
                self.assertEqual(result.training_files, (names[0], names[2]))
                self.assertEqual(result.check_files, tuple(name for name in order if name not in result.training_files))

    def test_repeat_subnumbers_use_numeric_order(self):
        result = self.choose(("sample_rep10_001.csv", "sample_rep02_010.csv", "sample_rep02_003.csv"))
        self.assertEqual(result.training_files, ("sample_rep02_003.csv", "sample_rep10_001.csv"))

    def test_other_existing_group_suffixes_are_numbered_repeats(self):
        for suffix in ("run", "scan", "repeat", "part", ""):
            names = tuple(f"measurement_{suffix}{index:03}.csv" for index in (10, 2, 1))
            with self.subTest(suffix=suffix):
                self.assertEqual(self.choose(names).training_files, (names[2], names[0]))

    def test_two_selected_repeats_both_used_and_one_never_duplicated(self):
        two = self.choose(("sample_rep03.csv", "sample_rep01.csv"))
        self.assertEqual(two.training_files, ("sample_rep01.csv", "sample_rep03.csv"))
        self.assertEqual(two.check_files, ())
        for names in ((), ("sample_rep01.csv",), "sample_rep01.csv"):
            with self.subTest(names=names):
                one = self.choose(names)
                self.assertFalse(one.usable)
                self.assertEqual(one.training_files, ())
                self.assertIn("two distinct", one.reason)

    def test_arbitrary_unknown_names_keep_existing_caller_order(self):
        names = ("z.csv", "sample.csv", "a.csv", "repeat.csv")
        result = self.choose(names)
        self.assertEqual(result.training_files, ("z.csv", "a.csv"))
        self.assertEqual(result.check_files, ("sample.csv", "repeat.csv"))
        self.assertTrue(self.choose(("sample.csv", "repeat.csv")).usable)
        mixed = ("sample_rep03.csv", "unnumbered.csv", "sample_rep01.csv")
        self.assertEqual(self.choose(mixed).training_files, (mixed[0], mixed[2]))

    def test_duplicate_path_aliases_are_rejected_without_silent_filtering(self):
        name = "sample_rep01.csv"
        for alias in (name, str(self.root / name), "./" + name,
                      "nested/../" + name, name.upper()):
            names = (name, "sample_rep02.csv", alias)
            with self.subTest(alias=alias):
                result = self.choose(names)
                self.assertFalse(result.usable)
                self.assertIn("duplicate file paths", result.reason)
                self.assertEqual(result.check_files, names)

    def test_selected_pn_sources_cannot_become_training_measurements(self):
        result = self.choose(("P.csv", "sample_rep01.csv"), excluded_files=(str(self.root / "P.csv"), "N.csv"))
        self.assertFalse(result.usable)
        self.assertIn("P, N and sample", result.reason)

    def test_background_names_and_catalog_roles_are_rejected(self):
        for name in ("Pbackground_rep01.csv", "Nback_rep01.csv", "dark.csv", "sample_BG.csv", "sample_REF.csv"):
            with self.subTest(name=name):
                result = self.choose((name, "sample_rep02.csv"))
                self.assertFalse(result.usable)
                self.assertIn("backgrounds and saved composites", result.reason)
        for overrides in ({"is_background": True}, {"classification": "background"},
                          {"classification": "likely_background"}, {"metadata_role": "background"}):
            with self.subTest(overrides=overrides):
                catalog = (self.source(str(self.root / "sample_rep01.csv"), **overrides),)
                result = self.choose(("sample_rep01.csv", "sample_rep02.csv"), catalog=catalog)
                self.assertFalse(result.usable)

    def test_unrelated_catalog_entries_never_add_measurements_or_change_default(self):
        names = ("sample_rep01.csv", "sample_rep02.csv", "sample_rep03.csv")
        catalog = tuple(self.source(name) for name in ("other_rep01.csv", "other_rep02.csv"))
        self.assertEqual(self.choose(names, catalog=catalog), self.choose(names))

    def test_structured_repeat_groups_and_known_conditions_cannot_be_mixed(self):
        cases = (
            (("sample_rep01.csv", "other_rep02.csv"), "different groups"),
            (("first_RotIn17.5deg.csv", "second_RotIn49deg.csv"), "rotation conditions differ"),
            (("first_1K.csv", "second_2K.csv"), "temperature conditions differ"),
            (("first_730nmc.csv", "second_760nmc.csv"), "wavelength centers differ"),
            (("first_0.3sx10.csv", "second_0.4sx10.csv"), "exposure conditions differ"),
        )
        for names, reason in cases:
            with self.subTest(names=names):
                result = self.choose(names)
                self.assertFalse(result.usable)
                self.assertIn(reason, result.reason)

    def test_matching_filename_conditions_and_unknowns_are_permitted(self):
        names = ("YZ383_p2n1_1.67KREF_9T_RotIn17.5deg_730nmc_0.3sx10_rep03.csv",
                 "YZ383_p2n1_1.67KREF_9T_RotIn17.5deg_730nmc_0.3sx10_rep01.csv")
        self.assertEqual(self.choose(names).training_files, names[::-1])
        self.assertTrue(self.choose(("first_RotIn17.5deg.csv", "unknown.csv")).usable)

    def test_non_original_extensions_rejected(self):
        for bad in ("sample.xlsx", "sample.metadata.json"):
            with self.subTest(bad=bad):
                result = self.choose((bad, "sample.csv"))
                self.assertFalse(result.usable)
                self.assertIn("original intensity CSV", result.reason)

    def test_chooser_does_not_scan_or_read_files_or_spectra(self):
        with patch.object(Path, "open", side_effect=AssertionError("file read")), \
             patch.object(Path, "iterdir", side_effect=AssertionError("directory scan")), \
             patch.object(Path, "glob", side_effect=AssertionError("directory glob")), \
             patch.object(Path, "rglob", side_effect=AssertionError("recursive directory glob")):
            result = self.choose(("sample_rep03.csv", "sample_rep02.csv", "sample_rep01.csv"))
        self.assertEqual(result.training_files, ("sample_rep01.csv", "sample_rep03.csv"))

    def test_default_choice_preserves_existing_generation_fingerprint(self):
        names = tuple(f"measurement_rep0{i}.csv" for i in range(1, 5))
        for name in ("P.csv", "N.csv", *names):
            (self.root / name).write_text("Vbg,Vtg,650,700\n0,0,100,110\n", encoding="utf8")
        pair = DrrPnPair(("P.csv",), ("N.csv",))
        manual = app.prepare_reconstruction_request(self.root, pair, names,
            training_files=(names[0], names[2]), method="conditioned_bounded")
        automatic = self.choose(names[::-1], excluded_files=(*pair.p_files, *pair.n_files))
        request = app.prepare_reconstruction_request(self.root, pair, names[::-1],
            training_files=automatic.training_files, method="conditioned_bounded")
        self.assertEqual(request.key.fingerprint, manual.key.fingerprint)
        self.assertEqual(request.settings, manual.settings)
        self.assertEqual(request.training_files, manual.training_files)


if __name__ == "__main__":
    unittest.main()
