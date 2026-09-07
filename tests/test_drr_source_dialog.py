from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from core.drr_sources import (
    _read_drr_metadata,
    compatible_drr_repeats,
    discover_drr_sources,
    find_saved_drr_recipe,
    group_drr_sources,
    is_background_name,
)


def _csv(path: Path, gates: list[tuple[float, float]], *, axis=(740, 760, 780)) -> None:
    rows = ["Vbg,Vtg," + ",".join(str(value) for value in axis)]
    rows.extend(",".join((str(vbg), str(vtg), "1", "2", "3")) for vbg, vtg in gates)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _one_gate_csv(path: Path, gates: list[float]) -> None:
    rows = ["Vbg,740,760,780"]
    rows.extend(f"{gate},1,2,3" for gate in gates)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_drr_recipe(
    metadata_path: Path,
    measurements: list[Path],
    backgrounds: list[Path],
    *,
    baseline_selection: str = "External",
    baseline_which: str = "all",
) -> None:
    metadata_path.write_text(
        json.dumps(
            {
                "operation": "DR/R",
                "sources": [
                    *(
                        {"role": "measurement", "source_path": str(path)}
                        for path in measurements
                    ),
                    *(
                        {"role": "background", "source_path": str(path)}
                        for path in backgrounds
                    ),
                ],
                "processing": {
                    "baseline_selection": baseline_selection,
                    "baseline_which": baseline_which,
                },
            }
        ),
        encoding="utf-8",
    )


class DrrSourceMetadataAndGridTests(unittest.TestCase):
    def test_background_name_variants_keep_numbered_and_polarity_tokens(self) -> None:
        for name in (
            "YZ303_760_background1.csv",
            "YZ303_760_background2.csv",
            "YZ365_760_nBack(+11.4995,+12.5).csv",
            "YZ365_760_pBack(-9.1996,-10).csv",
        ):
            self.assertTrue(is_background_name(name), name)
        self.assertFalse(is_background_name("feedback_measurement.csv"))

    def test_saved_roles_override_names_and_expose_shared_two_background_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            processed = root / "Processed Data" / "DRR"
            initial.mkdir(parents=True)
            processed.mkdir(parents=True)
            measurement = initial / "sample_back_named.csv"
            background1 = initial / "background1.csv"
            background2 = initial / "background2.csv"
            for path in (measurement, background1, background2):
                _csv(path, [(0, 0), (1, 0)])
            metadata = {
                "operation": "DR/R",
                "sources": [
                    {"role": "measurement", "source_path": str(measurement)},
                    {"role": "background", "source_path": str(background1)},
                    {"role": "background", "source_path": str(background2)},
                ],
                "processing": {"baseline_selection": "External", "baseline_which": "all"},
            }
            (processed / "recipe.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            sources = {source.source: source for source in discover_drr_sources(root)}

            sample = sources["Initial Data/sample_back_named.csv"]
            self.assertFalse(sample.is_background)
            self.assertEqual(sample.classification, "measurement")
            self.assertEqual(sample.metadata_role, "measurement")
            self.assertEqual(sample.linked_backgrounds, (
                "Initial Data/background1.csv", "Initial Data/background2.csv"
            ))
            self.assertIn("External", sample.saved_baseline_modes)
            background = sources["Initial Data/background1.csv"]
            self.assertEqual(background.classification, "background")
            self.assertEqual(background.gate_labels, ("Vbg", "Vtg"))
            self.assertEqual(background.gate_ranges, ((0.0, 1.0), (0.0, 0.0)))

    def test_per_file_links_stay_independent_when_group_summary_is_union(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            processed = root / "Processed Data" / "DRR"
            initial.mkdir(parents=True)
            processed.mkdir(parents=True)
            measurement_a = initial / "sample_760nm_rep1_1.csv"
            measurement_b = initial / "sample_760nm_rep1_2.csv"
            background_1 = initial / "bg1.csv"
            background_2 = initial / "bg2.csv"
            for path in (measurement_a, measurement_b, background_1, background_2):
                _csv(path, [(0, 0), (1, 0)])
            _write_drr_recipe(processed / "a.metadata.json", [measurement_a], [background_1])
            _write_drr_recipe(processed / "b.metadata.json", [measurement_b], [background_2])

            catalog = discover_drr_sources(root)
            sources = {source.source: source for source in catalog}

            self.assertEqual(
                sources["Initial Data/sample_760nm_rep1_1.csv"].linked_backgrounds,
                ("Initial Data/bg1.csv",),
            )
            self.assertEqual(
                sources["Initial Data/sample_760nm_rep1_2.csv"].linked_backgrounds,
                ("Initial Data/bg2.csv",),
            )
            measurement_groups = [
                group for group in group_drr_sources(catalog) if not group.is_background
            ]
            self.assertEqual(len(measurement_groups), 1)
            self.assertEqual(
                set(measurement_groups[0].linked_backgrounds),
                {"Initial Data/bg1.csv", "Initial Data/bg2.csv"},
            )

    def test_cross_group_links_do_not_contaminate_different_or_unprocessed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            processed = root / "Processed Data" / "DRR"
            initial.mkdir(parents=True)
            processed.mkdir(parents=True)
            measurement_a = initial / "alpha_rep1_1.csv"
            measurement_b = initial / "beta_rep1_1.csv"
            measurement_c = initial / "charlie_rep1_1.csv"
            measurement_d = initial / "delta_rep1_1.csv"
            background_1 = initial / "bg1.csv"
            background_2 = initial / "bg2.csv"
            background_3 = initial / "bg3.csv"
            for path in (
                measurement_a,
                measurement_b,
                measurement_c,
                measurement_d,
                background_1,
                background_2,
                background_3,
            ):
                _csv(path, [(0, 0), (1, 0)])
            _write_drr_recipe(
                processed / "ac.metadata.json",
                [measurement_a, measurement_c],
                [background_1, background_2],
            )
            _write_drr_recipe(processed / "b.metadata.json", [measurement_b], [background_3])

            catalog = discover_drr_sources(root)
            sources = {source.source: source for source in catalog}
            shared = ("Initial Data/bg1.csv", "Initial Data/bg2.csv")

            self.assertEqual(sources["Initial Data/alpha_rep1_1.csv"].linked_backgrounds, shared)
            self.assertEqual(sources["Initial Data/charlie_rep1_1.csv"].linked_backgrounds, shared)
            self.assertEqual(
                sources["Initial Data/beta_rep1_1.csv"].linked_backgrounds,
                ("Initial Data/bg3.csv",),
            )
            self.assertEqual(sources["Initial Data/delta_rep1_1.csv"].linked_backgrounds, ())
            groups = {
                group.title: group
                for group in group_drr_sources(catalog)
                if not group.is_background
            }
            self.assertEqual(set(groups["alpha"].linked_backgrounds), set(shared))
            self.assertEqual(groups["beta"].linked_backgrounds, ("Initial Data/bg3.csv",))
            self.assertEqual(groups["delta"].linked_backgrounds, ())

    def test_exact_recipe_restore_is_per_selection_and_mixed_selection_is_unmatched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            processed = root / "Processed Data" / "DRR"
            initial.mkdir(parents=True)
            processed.mkdir(parents=True)
            measurement_a = initial / "sample_a.csv"
            measurement_b = initial / "sample_b.csv"
            background_1 = initial / "bg1.csv"
            background_2 = initial / "bg2.csv"
            for path in (measurement_a, measurement_b, background_1, background_2):
                _csv(path, [(0, 0), (1, 0)])
            _write_drr_recipe(processed / "a.metadata.json", [measurement_a], [background_1])
            _write_drr_recipe(processed / "b.metadata.json", [measurement_b], [background_2])

            recipe_a = find_saved_drr_recipe(root, ["Initial Data/sample_a.csv"])
            recipe_b = find_saved_drr_recipe(root, ["Initial Data/sample_b.csv"])

            self.assertIsNotNone(recipe_a)
            self.assertIsNotNone(recipe_b)
            self.assertEqual(recipe_a.measurement_files, ("Initial Data/sample_a.csv",))
            self.assertEqual(recipe_a.baseline_files, ("Initial Data/bg1.csv",))
            self.assertEqual(recipe_b.measurement_files, ("Initial Data/sample_b.csv",))
            self.assertEqual(recipe_b.baseline_files, ("Initial Data/bg2.csv",))
            self.assertIsNone(
                find_saved_drr_recipe(
                    root,
                    ["Initial Data/sample_a.csv", "Initial Data/sample_b.csv"],
                )
            )

    def test_multiple_histories_keep_exact_records_and_newest_restore_is_not_union(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            processed = root / "Processed Data" / "DRR"
            initial.mkdir(parents=True)
            processed.mkdir(parents=True)
            measurement = initial / "sample_rep1_1.csv"
            background_1 = initial / "bg1.csv"
            background_2 = initial / "bg2.csv"
            for path in (measurement, background_1, background_2):
                _csv(path, [(0, 0), (1, 0)])
            old_metadata = processed / "history-old.metadata.json"
            new_metadata = processed / "history-new.metadata.json"
            _write_drr_recipe(old_metadata, [measurement], [background_1])
            _write_drr_recipe(new_metadata, [measurement], [background_2])
            os.utime(old_metadata, (1_700_000_000, 1_700_000_000))
            os.utime(new_metadata, (1_700_000_010, 1_700_000_010))
            metadata_bytes = {
                path: path.read_bytes() for path in (old_metadata, new_metadata)
            }

            catalog = discover_drr_sources(root)
            source = next(item for item in catalog if item.source == "Initial Data/sample_rep1_1.csv")
            _roles, records = _read_drr_metadata(root, require_drr_operation=True)
            histories = {Path(record.metadata_path).name: record for record in records}

            self.assertEqual(
                histories[old_metadata.name].measurement_files,
                ("Initial Data/sample_rep1_1.csv",),
            )
            self.assertEqual(histories[old_metadata.name].baseline_files, ("Initial Data/bg1.csv",))
            self.assertEqual(
                histories[new_metadata.name].measurement_files,
                ("Initial Data/sample_rep1_1.csv",),
            )
            self.assertEqual(histories[new_metadata.name].baseline_files, ("Initial Data/bg2.csv",))
            self.assertEqual(
                set(source.linked_backgrounds),
                {"Initial Data/bg1.csv", "Initial Data/bg2.csv"},
            )
            newest = find_saved_drr_recipe(root, ["Initial Data/sample_rep1_1.csv"])
            self.assertIsNotNone(newest)
            self.assertEqual(newest.baseline_files, ("Initial Data/bg2.csv",))
            self.assertNotEqual(newest.baseline_files, source.linked_backgrounds)
            self.assertEqual(
                {path: path.read_bytes() for path in metadata_bytes},
                metadata_bytes,
            )

    def test_stale_metadata_does_not_match_unrelated_same_basename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data" / "session"
            initial.mkdir(parents=True)
            unrelated = root / "background.csv"
            _csv(unrelated, [(0, 0), (1, 0)])
            processed = root / "Processed Data" / "DRR"
            processed.mkdir(parents=True)
            (processed / "stale.metadata.json").write_text(json.dumps({
                "operation": "DR/R",
                "sources": [{
                    "role": "background",
                    "source_path": str(root / "old" / "background.csv"),
                    "name": "old/background.csv",
                }],
            }), encoding="utf-8")

            source = discover_drr_sources(root)[0]

            self.assertEqual(source.classification, "background")
            self.assertEqual(source.metadata_role, "")

    def test_unknown_operation_metadata_does_not_auto_restore_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir(parents=True)
            measurement = initial / "sample.csv"
            background = initial / "background.csv"
            _csv(measurement, [(0, 0), (1, 0)])
            _csv(background, [(0, 0), (0, 0)])
            processed = root / "Processed Data" / "DRR"
            processed.mkdir(parents=True)
            (processed / "other.metadata.json").write_text(json.dumps({
                "operation": "PL",
                "sources": [
                    {"role": "measurement", "source_path": str(measurement)},
                    {"role": "background", "source_path": str(background)},
                ],
            }), encoding="utf-8")

            self.assertIsNone(find_saved_drr_recipe(root, ["Initial Data/sample.csv"]))

    def test_discovery_reports_full_frame_count_and_partial_group_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir(parents=True)
            first = initial / "sample_rep1_1.csv"
            second = initial / "sample_rep1_2.csv"
            _csv(first, [(0, 0)] * 101)
            _csv(second, [(0, 0)] * 203)
            processed = root / "Processed Data" / "DRR"
            processed.mkdir(parents=True)
            (processed / "one.metadata.json").write_text(json.dumps({
                "operation": "DR/R",
                "sources": [{"role": "measurement", "source_path": str(first)}],
            }), encoding="utf-8")

            sources = discover_drr_sources(root)
            groups = [group for group in group_drr_sources(sources) if not group.is_background]

            by_name = {source.filename: source for source in sources}
            self.assertEqual(by_name[first.name].frame_count, 101)
            self.assertEqual(by_name[second.name].frame_count, 203)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].processed_count, 1)
            self.assertEqual(groups[0].frame_count_range, (101, 203))

    def test_seven_repeat_group_exposes_five_of_seven_processed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir(parents=True)
            repeats = [initial / f"pe_760_rep1_{index}.csv" for index in range(1, 8)]
            for index, path in enumerate(repeats):
                _csv(path, [(float(frame), 0.0) for frame in range(203 if index == 1 else 101)])
            processed = root / "Processed Data" / "DRR"
            processed.mkdir(parents=True)
            (processed / "pe.metadata.json").write_text(json.dumps({
                "operation": "DR/R",
                "sources": [
                    {"role": "measurement", "source_path": str(path)}
                    for path in repeats[:5]
                ],
            }), encoding="utf-8")

            groups = [
                group for group in group_drr_sources(discover_drr_sources(root))
                if not group.is_background
            ]

            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].processed_count, 5)
            self.assertFalse(groups[0].processed)
            self.assertEqual(len(groups[0].files), 7)

    def test_compatible_repeats_require_ordered_gate_and_spectral_grids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir(parents=True)
            reference = initial / "YZ365_p3_rep1_1.csv"
            same = initial / "YZ365_p3_rep1_2.csv"
            different_gate = initial / "YZ365_p3_rep1_3.csv"
            different_axis = initial / "YZ365_p3_rep1_4.csv"
            gates = [(float(index), 0.0) for index in range(40)]
            _csv(reference, gates)
            _csv(same, gates)
            _csv(different_gate, [gates[0], *reversed(gates[1:])])
            _csv(different_axis, gates, axis=(740, 761, 780))

            catalog = discover_drr_sources(root)
            compatible = compatible_drr_repeats(
                catalog, "Initial Data/YZ365_p3_rep1_1.csv"
            )

            self.assertEqual(compatible, (
                "Initial Data/YZ365_p3_rep1_1.csv",
                "Initial Data/YZ365_p3_rep1_2.csv",
            ))

    def test_moved_metadata_prefers_valid_relative_names_and_recipe_matches_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir(parents=True)
            first = initial / "moved_760nmc.csv"
            second = initial / "moved_760nmc_rep1_2.csv"
            extra = initial / "moved_760nmc_rep1_3.csv"
            background = initial / "background.csv"
            for path in (first, second, extra, background):
                _csv(path, [(0, 0), (1, 0)])
            processed = root / "Processed Data" / "DRR"
            processed.mkdir(parents=True)
            stale_first = root / "old-experiment" / first.name
            stale_background = root / "old-experiment" / background.name
            (processed / "moved.metadata.json").write_text(json.dumps({
                "operation": "DR/R",
                "sources": [
                    {
                        "role": "measurement",
                        "source_path": str(stale_first),
                        "name": "Initial Data/moved_760nmc.csv",
                    },
                    {
                        "role": "measurement",
                        "source_path": str(root / "old-experiment" / second.name),
                        "name": "Initial Data/moved_760nmc_rep1_2.csv",
                    },
                    {
                        "role": "background",
                        "source_path": str(stale_background),
                        "name": "Initial Data/background.csv",
                    },
                ],
                "processing": {"baseline_selection": "External", "baseline_which": "all"},
            }), encoding="utf-8")

            source = next(item for item in discover_drr_sources(root) if item.filename == first.name)
            self.assertEqual(source.metadata_role, "measurement")
            self.assertEqual(source.linked_backgrounds, ("Initial Data/background.csv",))
            self.assertIsNotNone(find_saved_drr_recipe(root, [
                "Initial Data/moved_760nmc.csv",
                "Initial Data/moved_760nmc_rep1_2.csv",
            ]))
            self.assertIsNone(find_saved_drr_recipe(root, ["Initial Data/moved_760nmc.csv"]))
            self.assertIsNone(find_saved_drr_recipe(root, [
                "Initial Data/moved_760nmc.csv",
                "Initial Data/moved_760nmc_rep1_2.csv",
                "Initial Data/moved_760nmc_rep1_3.csv",
            ]))

    def test_compatible_repeats_reject_gate_shape_or_identity_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir(parents=True)
            reference = initial / "sample_rep1_1.csv"
            one_gate = initial / "sample_rep1_2.csv"
            swapped = initial / "sample_rep1_3.csv"
            gates = [(0.0, 0.0), (1.0, 1.0)]
            _csv(reference, gates)
            _one_gate_csv(one_gate, [0.0, 1.0])
            swapped.write_text(
                "Vtg,Vbg,740,760,780\n0,0,1,2,3\n1,1,1,2,3\n",
                encoding="utf-8",
            )

            catalog = discover_drr_sources(root)

            self.assertEqual(
                compatible_drr_repeats(catalog, "Initial Data/sample_rep1_1.csv"),
                ("Initial Data/sample_rep1_1.csv",),
            )

    def test_malformed_gate_row_keeps_frame_count_but_blocks_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "Initial Data"
            initial.mkdir(parents=True)
            reference = initial / "sample_rep1_1.csv"
            malformed = initial / "sample_rep1_2.csv"
            gates = [(float(index), 0.0) for index in range(40)]
            _csv(reference, gates)
            malformed.write_text(
                "Vbg,Vtg,740,760,780\n"
                + "\n".join(
                    f"{vbg},0,1,2,3" if index != 20 else f"{vbg},,1,2,3"
                    for index, (vbg, _vtg) in enumerate(gates)
                )
                + "\n",
                encoding="utf-8",
            )

            catalog = discover_drr_sources(root)
            malformed_source = next(source for source in catalog if source.filename == malformed.name)

            self.assertEqual(malformed_source.frame_count, 40)
            self.assertFalse(malformed_source.grid_complete)
            self.assertEqual(
                compatible_drr_repeats(catalog, reference.relative_to(root).as_posix()),
                ("Initial Data/sample_rep1_1.csv",),
            )


if __name__ == "__main__":
    unittest.main()
