"""Stable naming, history compatibility and rollback for native Peak Shift."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from core import mcd_peak_save as saving
from core.mcd_peak_export import (
    HISTORY_DIR, HISTORY_INDEX_NAME, mcd_peak_shift_export_worker,
    peak_shift_history_status, source_descriptor,
)
from core.mcd_peak_shift import PeakPoint, PeakShiftResult, PeakTrack


def snapshot(folder, source=None):
    source = source or folder/"sample_4K.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("original measurement", encoding="utf-8")
    branch = "B increasing"
    tracks = tuple(PeakTrack(index, branch,
        (PeakPoint(0., branch, center, 0., "tracked"), PeakPoint(1., branch, center+.001, .001, "tracked")),
        center, 0., "exact 0 T") for index, center in ((1, 1.7), (2, 1.75)))
    result = PeakShiftResult(np.array([0., 1.]), np.array([branch, branch]), (), tracks, "raw pos")
    return dict(result=result, method_results={method: {"pos": result, "neg": result} for method in ("Raw spectrum", "Second derivative")},
        tracker_method="Raw spectrum", selected=("pos", 1, branch, "peak"), selected_k=None, selected_kp=None,
        spectrum_source="Raw R", experiment_folder=str(folder), source_descriptor=source_descriptor(folder, source),
        history_settings=dict(resultMode="Valley splitting", peakDisplay="Delta E", peakProminence=.03))


class McdPeakSaveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name).resolve()
        self.snapshot = snapshot(self.folder)
        self.root = self.folder/HISTORY_DIR
        self.index = self.root/HISTORY_INDEX_NAME

    def save(self, value=None, root=None):
        return saving.export_peak_shift_analysis(value or self.snapshot, root or self.root)

    def preserved_bytes(self, output):
        return {path: Path(path).read_bytes() for path in (output["csv_path"], output["sidecar_path"], str(self.index))}

    def test_source_package_and_stable_analysis_name(self):
        first = self.save()
        second = self.save()
        self.assertEqual(first["package_dir"], str(self.root/"sample_4K_PeakShift"))
        self.assertEqual(first["csv_path"], second["csv_path"])
        self.assertFalse(first["updated"])
        self.assertTrue(second["updated"])
        self.assertIn("pos_P_E1.700000eV_inc_Raw_RawR_ValleySplitting", Path(first["csv_path"]).stem)
        self.assertNotIn("peak_shift_20", Path(first["csv_path"]).name)
        self.assertEqual(len(list(Path(first["package_dir"]).glob("*.csv"))), 1)
        self.assertEqual(len(json.loads(self.index.read_text())["exports"]), 1)
        metadata = json.loads(Path(first["sidecar_path"]).read_text())
        self.assertIn("created_utc", metadata)
        self.assertEqual(metadata["analysis_id"], first["analysis_id"])

    def test_parameter_edits_update_the_same_analysis(self):
        first = self.save()
        before = Path(first["csv_path"]).read_bytes()
        changed = deepcopy(self.snapshot)
        changed["history_settings"]["peakProminence"] = .05
        track = changed["result"].tracks[0]
        altered = replace(track, points=(track.points[0], replace(track.points[1], energy_ev=1.705)))
        changed["result"] = replace(changed["result"], tracks=(altered, *changed["result"].tracks[1:]))
        second = self.save(changed)
        self.assertEqual(first["csv_path"], second["csv_path"])
        self.assertNotEqual(before, Path(second["csv_path"]).read_bytes())
        self.assertEqual(json.loads(Path(second["sidecar_path"]).read_text())["processing"]["peakProminence"], .05)

    def test_peaks_methods_result_types_sources_and_pairs_remain_separate(self):
        variants = [deepcopy(self.snapshot) for _ in range(7)]
        variants[1]["selected"] = ("pos", 2, "B increasing", "peak")
        variants[2]["tracker_method"] = "Second derivative"
        variants[3]["history_settings"]["resultMode"] = "Single peak shift"
        variants[4]["history_settings"].update(resultMode="Single peak shift", peakDisplay="Absolute E")
        variants[5]["spectrum_source"] = "MCD-corrected R"
        variants[6].update(selected_k=1, selected_kp=2)
        results = [self.save(value) for value in variants]
        self.assertEqual(len({item["csv_path"] for item in results}), 7)
        self.assertEqual(len({item["package_dir"] for item in results}), 1)

    def test_same_basename_from_other_source_uses_separate_package(self):
        first = self.save()
        before = self.preserved_bytes(first)
        other = snapshot(self.folder, self.folder/"other"/"sample_4K.csv")
        second = self.save(other)
        self.assertNotEqual(first["package_dir"], second["package_dir"])
        self.assertRegex(Path(second["package_dir"]).name, r"sample_4K_[0-9a-f]{10}_PeakShift$")
        for path in (first["csv_path"], first["sidecar_path"]):
            self.assertEqual(Path(path).read_bytes(), before[path])

    def test_renumbered_peaks_keep_their_physical_save_identity(self):
        self.snapshot.update(selected_k=1, selected_kp=2)
        first = self.save()
        changed = deepcopy(self.snapshot)
        renumbered = replace(changed["result"], tracks=tuple(replace(track, peak_id=track.peak_id+1)
                                                          for track in changed["result"].tracks))
        changed.update(result=renumbered, selected=("pos", 2, "B increasing", "peak"), selected_k=2, selected_kp=3,
                       method_results={"Raw spectrum": {"pos": renumbered, "neg": renumbered}})
        second = self.save(changed)
        self.assertEqual(first["csv_path"], second["csv_path"])
        different = deepcopy(changed)
        different["selected"] = ("pos", 3, "B increasing", "peak")
        self.assertNotEqual(self.save(different)["csv_path"], first["csv_path"])

    def test_changed_locator_center_is_saved_separately(self):
        first = self.save()
        changed = deepcopy(self.snapshot)
        result = replace(changed["result"], tracks=(replace(changed["result"].tracks[0], reference_energy_ev=1.701),
                                                  *changed["result"].tracks[1:]))
        changed.update(result=result, method_results={"Raw spectrum": {"pos": result, "neg": result}})
        self.assertNotEqual(self.save(changed)["csv_path"], first["csv_path"])

    def test_local_refit_keeps_the_locator_identity_when_model_center_changes(self):
        local = deepcopy(self.snapshot)
        result = replace(local["result"], tracking_method="Local mixed fit",
            tracks=tuple(replace(track, locator_energy_ev=track.reference_energy_ev) for track in local["result"].tracks))
        local.update(result=result, tracker_method="Local mixed fit",
                     method_results={"Local mixed fit": {"pos": result, "neg": result}})
        first = self.save(local)
        changed = replace(result, tracks=(replace(result.tracks[0], reference_energy_ev=1.701), *result.tracks[1:]))
        local.update(result=changed, method_results={"Local mixed fit": {"pos": changed, "neg": changed}})
        self.assertEqual(self.save(local)["csv_path"], first["csv_path"])

    def test_absolute_energy_without_e0_uses_measured_track_location(self):
        value = deepcopy(self.snapshot)
        value["history_settings"].update(resultMode="Single peak shift", peakDisplay="Absolute E")
        result = replace(value["result"], tracks=(replace(value["result"].tracks[0], reference_energy_ev=None,
            reference_method="unavailable"), *value["result"].tracks[1:]))
        value.update(result=result, method_results={"Raw spectrum": {"pos": result, "neg": result}})
        self.assertIn("E1.700500eV", Path(self.save(value)["csv_path"]).name)

    def test_concurrent_analysis_saves_preserve_both_history_records(self):
        from concurrent.futures import ThreadPoolExecutor
        values = [deepcopy(self.snapshot), deepcopy(self.snapshot)]
        values[1]["selected"] = ("pos", 2, "B increasing", "peak")
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(self.save, values))
        self.assertEqual(len(json.loads(self.index.read_text())["exports"]), 2)
        self.assertTrue(all(Path(row["csv_path"]).is_file() for row in results))

    def test_custom_output_roots_match_mcd_structure(self):
        for root in (self.folder/"exports", self.folder/"exports"/"Processed Data",
                     self.folder/"exports"/HISTORY_DIR):
            result = self.save(root=root)
            self.assertEqual(Path(result["package_dir"]).parent, self.folder/"exports"/HISTORY_DIR)

    def test_long_source_names_are_bounded_and_keep_distinct_identities(self):
        self.snapshot["source_descriptor"]["filename"] = "long source "*35+".csv"
        first = self.save()
        self.assertLessEqual(len(Path(first["package_dir"]).name), max(24, 128-len(str(self.root)))+10)
        self.assertRegex(Path(first["package_dir"]).name, r"_[0-9a-f]{8}_PeakShift$")

    def test_legacy_timestamp_exports_and_history_are_kept(self):
        old = self.root/"sample_peak_shift_20261030_120000.csv"
        mcd_peak_shift_export_worker(self.snapshot, old)
        before = old.read_bytes()
        self.save()
        self.assertEqual(old.read_bytes(), before)
        self.assertEqual(len(json.loads(self.index.read_text())["exports"]), 2)
        self.assertEqual(peak_shift_history_status(self.folder, self.snapshot["source_descriptor"])[0], "processed")

    def test_csv_failure_leaves_previous_pair_and_history_untouched(self):
        first = self.save()
        before = self.preserved_bytes(first)
        with patch.object(saving, "export_peak_shift_csv", side_effect=ValueError("CSV failed")):
            with self.assertRaisesRegex(ValueError, "CSV failed"):
                self.save()
        self.assertEqual(self.preserved_bytes(first), before)
        self.assertFalse(list(Path(first["package_dir"]).glob(".staging-*")))

    def test_second_file_publication_failure_restores_the_previous_csv(self):
        first = self.save()
        before = self.preserved_bytes(first)
        original = Path.replace
        sidecar = Path(first["sidecar_path"])
        def replace_path(path, target):
            if Path(target) == sidecar and path.name == sidecar.name:
                raise OSError("publication failed")
            return original(path, target)
        with patch.object(Path, "replace", replace_path):
            with self.assertRaisesRegex(OSError, "publication failed"):
                self.save()
        self.assertEqual(self.preserved_bytes(first), before)

    def test_history_publication_failure_rolls_back_both_files(self):
        first = self.save()
        before = self.preserved_bytes(first)
        original = saving._atomic_json
        def write_json(path, payload):
            if path == self.index:
                raise OSError("index failed")
            return original(path, payload)
        with patch.object(saving, "_atomic_json", side_effect=write_json):
            with self.assertRaisesRegex(OSError, "index failed"):
                self.save()
        self.assertEqual(self.preserved_bytes(first), before)

    def test_corrupt_history_and_unverified_existing_files_are_preserved(self):
        first = self.save()
        self.index.write_text("not JSON", encoding="utf-8")
        before = self.preserved_bytes(first)
        with self.assertRaisesRegex(ValueError, "history could not be read"):
            self.save()
        self.assertEqual(self.preserved_bytes(first), before)
        self.index.unlink()
        Path(first["sidecar_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "unverified analysis identity"):
            self.save()
        self.assertEqual(Path(first["csv_path"]).read_bytes(), before[first["csv_path"]])


if __name__ == "__main__":
    unittest.main()
