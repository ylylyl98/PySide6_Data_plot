from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from core.drr_sources import (
    assess_background_gate_files,
    discover_drr_sources,
    extract_wavelength_center_nm,
    find_saved_drr_recipe,
    guess_drr_background,
    group_drr_sources,
    inspect_csv_gate,
    inspect_csv_wavelength_center,
    is_background_name,
    newest_measurement_group,
    portable_source_name,
    validate_named_wavelength_centers,
)
from core.provenance import verify_initial_data_working_file


class DrrSourceCatalogTests(unittest.TestCase):
    def test_discovers_root_and_nested_initial_data_without_processed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data" / "session-a"
            processed = root / "Processed Data" / "DRR"
            initial.mkdir(parents=True)
            processed.mkdir(parents=True)
            (root / "working.csv").write_text("root", encoding="utf-8")
            (initial / "sample_rep1_1.csv").write_text("raw", encoding="utf-8")
            (processed / "result.csv").write_text("output", encoding="utf-8")

            sources = discover_drr_sources(root)

            self.assertEqual(
                {source.source for source in sources},
                {"working.csv", "Initial Data/session-a/sample_rep1_1.csv"},
            )

    def test_groups_repeats_by_condition_and_acquisition_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir()
            first = initial / "sample_5K_rep1_1.csv"
            second = initial / "sample_5K_rep1_2.csv"
            first.write_text("a", encoding="utf-8")
            second.write_text("b", encoding="utf-8")
            timestamp = 1_700_000_000
            os.utime(first, (timestamp, timestamp))
            os.utime(second, (timestamp + 5, timestamp + 5))

            groups = group_drr_sources(discover_drr_sources(root))

            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].title, "sample_5K")
            self.assertEqual(len(groups[0].files), 2)

    def test_background_detection_does_not_hide_previous_measurements(self) -> None:
        self.assertTrue(is_background_name("sample_back_run1.csv"))
        self.assertTrue(is_background_name("sample_BG.csv"))
        self.assertFalse(is_background_name("sample_run1.csv"))

    def test_extracts_wavelength_center_from_measurement_filename(self) -> None:
        self.assertEqual(
            extract_wavelength_center_nm("YZ365_p2_1.67KREF_760nmc_0p05sx20.csv"),
            760.0,
        )
        self.assertEqual(extract_wavelength_center_nm("sample_632p8nm_center_back.csv"), 632.8)
        self.assertIsNone(extract_wavelength_center_nm("sample_632.8nm_laser.csv"))
        self.assertIsNone(extract_wavelength_center_nm("sample_back.csv"))

    def test_spectral_axis_supplies_center_when_filename_only_has_laser_nm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_632.8nm_laser.csv"
            path.write_text(
                "Vbg,Vtg,740,760,780\n0,0,1,2,3\n1,0,2,3,4\n",
                encoding="utf-8",
            )

            self.assertEqual(inspect_csv_wavelength_center(path), 760.0)

    def test_named_background_center_must_match_measurement(self) -> None:
        self.assertEqual(
            validate_named_wavelength_centers(
                ["sample_760nmc_rep1.csv"], ["old_group_760nmc_back.csv"]
            ),
            760.0,
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_named_wavelength_centers(
                ["sample_760nmc_rep1.csv"], ["old_group_700nmc_back.csv"]
            )

    def test_gate_inspection_distinguishes_sweep_from_constant_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            sweep = folder / "sweep.csv"
            constant = folder / "constant.csv"
            sweep.write_text(
                "Vbg,Vtg,700,701\n0,0,1,2\n1,0,2,3\n2,0,3,4\n",
                encoding="utf-8",
            )
            constant.write_text(
                "Vbg,Vtg,700,701\n0,0,1,2\n0,0,2,3\n",
                encoding="utf-8",
            )

            self.assertEqual(inspect_csv_gate(sweep), (True, 3))
            self.assertEqual(inspect_csv_gate(constant), (False, 2))

    def test_constant_gate_backgrounds_recommend_all_frames_only_when_gates_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            same_a = root / "back_a.csv"
            same_b = root / "back_b.csv"
            different = root / "back_different.csv"
            for path, gate in ((same_a, 0.0), (same_b, 0.0), (different, 1.0)):
                path.write_text(
                    f"Vbg,Vtg,740,760,780\n{gate},0,1,2,3\n{gate},0,2,3,4\n",
                    encoding="utf-8",
                )

            matching = assess_background_gate_files(
                root, [same_a.name, same_b.name]
            )
            mixed = assess_background_gate_files(
                root, [same_a.name, different.name]
            )

            self.assertTrue(matching.all_constant)
            self.assertTrue(matching.same_constant_values)
            self.assertTrue(mixed.all_constant)
            self.assertFalse(mixed.same_constant_values)

    def test_finds_newest_exact_saved_drr_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            processed = root / "Processed Data" / "DRR"
            initial.mkdir()
            processed.mkdir(parents=True)
            measurement = initial / "measurement.csv"
            background = initial / "background.csv"
            measurement.write_text("measurement", encoding="utf-8")
            background.write_text("background", encoding="utf-8")
            metadata = processed / "measurement.metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "operation": "DR/R",
                        "sources": [
                            {"role": "measurement", "source_path": str(measurement)},
                            {"role": "background", "source_path": str(background)},
                        ],
                        "processing": {
                            "baseline_selection": "External",
                            "baseline_which": "all",
                        },
                    }
                ),
                encoding="utf-8",
            )

            recipe = find_saved_drr_recipe(
                root, ["Initial Data/measurement.csv"]
            )

            self.assertIsNotNone(recipe)
            self.assertEqual(recipe.baseline_files, ("Initial Data/background.csv",))
            self.assertEqual(recipe.baseline_selection, "External")
            self.assertEqual(recipe.baseline_which, "all")
            self.assertIsNone(
                find_saved_drr_recipe(root, ["Initial Data/unprocessed.csv"])
            )

    def test_guesses_closest_earlier_compatible_background_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir()
            background_names = [
                "sample_760nmc_back_rep1_1.csv",
                "sample_760nmc_back_rep1_2.csv",
            ]
            measurement_name = "sample_760nmc_data_rep1_1.csv"
            content = "Vbg,Vtg,740,760,780\n0,0,1,2,3\n0,0,2,3,4\n"
            for index, name in enumerate(background_names):
                path = initial / name
                path.write_text(content, encoding="utf-8")
                os.utime(path, (1_700_000_000 + index, 1_700_000_000 + index))
            measurement = initial / measurement_name
            measurement.write_text(
                "Vbg,Vtg,740,760,780\n0,0,1,2,3\n1,0,2,3,4\n",
                encoding="utf-8",
            )
            os.utime(measurement, (1_700_000_100, 1_700_000_100))
            catalog = discover_drr_sources(root)

            guess = guess_drr_background(
                root,
                catalog,
                [f"Initial Data/{measurement_name}"],
            )

            self.assertIsNotNone(guess)
            self.assertEqual(
                set(guess.baseline_files),
                {f"Initial Data/{name}" for name in background_names},
            )
            self.assertEqual(guess.baseline_which, "all")
            self.assertIn("closest earlier background", guess.reason)

    def test_small_constant_gate_file_is_likely_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir()
            measurement_rows = [
                f"{gate},0," + ",".join(str(gate + index) for index in range(30))
                for gate in range(20)
            ]
            (initial / "measurement.csv").write_text(
                "Vbg,Vtg," + ",".join(str(700 + index) for index in range(30))
                + "\n" + "\n".join(measurement_rows) + "\n",
                encoding="utf-8",
            )
            (initial / "reference.csv").write_text(
                "Vbg,Vtg,700,701\n0,0,1,2\n0,0,2,3\n",
                encoding="utf-8",
            )

            sources = {source.filename: source for source in discover_drr_sources(root)}

            self.assertEqual(sources["measurement.csv"].classification, "measurement")
            # The filename is deliberately neutral enough for the content
            # classifier to be responsible for this result.
            self.assertEqual(sources["reference.csv"].classification, "background")
            self.assertTrue(sources["reference.csv"].is_background)

    def test_small_constant_gate_without_keyword_is_likely_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir()
            rows = [f"{gate},0," + ",".join(["1"] * 40) for gate in range(24)]
            (initial / "measurement.csv").write_text(
                "Vbg,Vtg," + ",".join(str(700 + index) for index in range(40))
                + "\n" + "\n".join(rows) + "\n",
                encoding="utf-8",
            )
            (initial / "zero.csv").write_text(
                "Vbg,Vtg,700,701\n0,0,1,2\n0,0,2,3\n",
                encoding="utf-8",
            )

            sources = {source.filename: source for source in discover_drr_sources(root)}

            self.assertEqual(sources["zero.csv"].classification, "likely_background")
            self.assertTrue(sources["zero.csv"].is_background)

    def test_run_suffixes_are_grouped_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir()
            for name in ("sample_5K_run1.csv", "sample_5K_run2.csv"):
                (initial / name).write_text(name, encoding="utf-8")

            groups = group_drr_sources(discover_drr_sources(root))

            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].title, "sample_5K")
            self.assertEqual(len(groups[0].files), 2)

    def test_newest_group_prefers_unprocessed_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            output = root / "Processed Data" / "DRR"
            initial.mkdir(parents=True)
            output.mkdir(parents=True)
            older = initial / "older_rep1.csv"
            newer = initial / "newer_rep1.csv"
            older.write_text("old", encoding="utf-8")
            newer.write_text("new", encoding="utf-8")
            os.utime(older, (1_700_000_000, 1_700_000_000))
            os.utime(newer, (1_700_100_000, 1_700_100_000))
            (output / "newer.metadata.json").write_text(
                json.dumps({"sources": [{"role": "measurement", "source_path": str(newer)}]}),
                encoding="utf-8",
            )

            newest = newest_measurement_group(group_drr_sources(discover_drr_sources(root)))

            self.assertIsNotNone(newest)
            self.assertEqual(newest.title, "older")

    def test_direct_initial_data_source_is_never_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Initial Data" / "old session" / "background.csv"
            source.parent.mkdir(parents=True)
            source.write_text("baseline", encoding="utf-8")

            record = verify_initial_data_working_file(
                portable_source_name(root, source), root, workflow="DRR", role="background"
            )

            self.assertFalse(record.provenance_verified)
            self.assertFalse(record.temporary_working_copy)
            self.assertEqual(record.verification_method, "direct_initial_data")
            self.assertEqual(Path(record.canonical_source_path), source.resolve())


if __name__ == "__main__":
    unittest.main()
