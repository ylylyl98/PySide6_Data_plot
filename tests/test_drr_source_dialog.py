from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from core.drr_sources import (
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
