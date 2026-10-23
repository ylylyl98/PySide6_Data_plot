import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core import drr_pn_assisted as A
from core.drr_composite_background import compose_background, read_composite_background
from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_store import DrrPnStore
from core.load_cancellation import LoadCancelled


class AssistedFileTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.energy = np.linspace(1.35, 2.15, 241)
        self.base = 1000*np.exp(.2*self.energy + .001*np.sin(180*self.energy))
        self.write("P.csv", np.stack([self.base]*2))
        self.write("N.csv", np.stack([self.base*1.01]*2))
        self.files = tuple(f"measurement_rep{i}.csv" for i in range(4))
        for i, name in enumerate(self.files):
            rows = np.array([self.base*(1 + .05*np.exp(-((self.energy-(1.65+.02*g))/.014)**2))
                             for g in range(5)])*(1+.001*i)
            self.write(name, rows)
        self.pair = DrrPnPair(("P.csv",), ("N.csv",))

    def write(self, name, values, *, shift=0., gate_shift=0.):
        with (self.folder/name).open("w", newline="", encoding="utf8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Vbg", "Vtg", "Vbias", *[format(x+shift, ".17g") for x in self.energy]])
            for index, row in enumerate(values):
                writer.writerow([index+gate_shift, 0, 0, *row])

    def prepare(self, files=None):
        return A.prepare_assisted_request(self.folder, self.pair, measurements=self.files if files is None else files)

    def generate(self, request=None):
        return A.generate_assisted_background(self.folder, self.pair, request=request or self.prepare())

    def test_roles_content_reuse_and_held_checks_are_separate(self):
        request = self.prepare()
        self.assertEqual(request.training_files, self.files[::2])
        self.assertEqual(request.evaluation_files, self.files[1::2])
        self.assertEqual(request.key.fingerprint, self.prepare().key.fingerprint)
        self.write(self.files[1], np.stack([self.base*2]*5))
        self.assertEqual(request.key.fingerprint, self.prepare().key.fingerprint)
        self.write(self.files[0], np.stack([self.base*1.2]*5))
        changed = self.prepare()
        self.assertNotEqual(request.key.fingerprint, changed.key.fingerprint)
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.generate(request)

    def test_actual_generation_store_and_roundtrip(self):
        request = self.prepare()
        result = self.generate(request)
        self.assertTrue(result.report["sample_used"])
        self.assertFalse(result.report["BN_used"])
        self.assertFalse(result.report["v66_equivalent"])
        self.assertEqual(result.report["settings"], request.settings)
        self.assertNotIn("evaluation_files", result.report)
        store = DrrPnStore(self.folder)
        version = store.save(result.document, request.key)
        self.assertTrue(version.fresh)
        self.assertTrue(A.matching_sample_context(version, self.files))
        self.assertFalse(A.matching_sample_context(version, self.files[::-1]))
        self.assertEqual(version.version_id, store.find_reusable(self.prepare().key).version_id)
        loaded = read_composite_background(version.csv_path)
        np.testing.assert_array_equal(compose_background(loaded)[1], result.arrays["background"])
        self.write(self.files[1], np.stack([self.base*1.03]*5))
        self.assertTrue(store.list_versions(self.pair)[0].fresh)
        self.write(self.files[0], np.stack([self.base*1.03]*5))
        self.assertFalse(store.list_versions(self.pair)[0].fresh)

    def test_assessment_does_not_mutate_generation_or_manifest(self):
        request = self.prepare()
        result = self.generate(request)
        background = result.arrays["background"].copy()
        store = DrrPnStore(self.folder)
        version = store.save(result.document, request.key)
        manifest = store.manifest_path.read_bytes()
        assessment = A.assess_assisted_result(self.folder, result, request)
        self.assertTrue(assessment["independent_holdout"])
        self.assertEqual(assessment["sample_shape"], [2, 5, len(self.energy)])
        self.assertFalse(assessment["acceptance"]["passed"])
        self.assertEqual(set(assessment["metrics"]), {"sample_assisted", "pn_only", "P", "N"})
        self.assertEqual(set(assessment["metrics"]["sample_assisted"]["d2E"]), {"11", "21"})
        path = A.save_assessment(self.folder, version, assessment)
        self.assertTrue(path.exists())
        self.assertEqual(path, A.save_assessment(self.folder, version, assessment))
        wrong = {**assessment, "assessed_background_sha256": "0"*64}
        with self.assertRaisesRegex(ValueError, "background values differ"):
            A.save_assessment(self.folder, version, wrong)
        self.assertEqual(json.loads(path.read_text())["background_version_id"], version.version_id)
        other = A.assess_assisted_result(self.folder, result, request, windows=(13,))
        self.assertNotEqual(path, A.save_assessment(self.folder, version, other))
        self.assertEqual(request.key.fingerprint, self.prepare().key.fingerprint)
        self.assertEqual(manifest, store.manifest_path.read_bytes())
        np.testing.assert_array_equal(background, result.arrays["background"])

    def test_small_selection_has_no_independent_holdout(self):
        request = self.prepare(self.files[:2])
        self.assertEqual(request.training_files, self.files[:2])
        self.assertFalse(request.evaluation_files)
        report = A.assess_assisted_result(self.folder, self.generate(request), request)
        self.assertFalse(report["independent_holdout"])
        self.assertEqual(report["role"], "training_data_only")

    def test_held_data_changes_do_not_change_generated_background(self):
        first = self.generate()
        self.write(self.files[1], np.stack([self.base*3]*5))
        second = self.generate()
        np.testing.assert_array_equal(first.arrays["background"], second.arrays["background"])
        self.assertEqual(first.report, second.report)

    def test_grid_gate_alias_and_role_errors_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "distinct files"):
            self.prepare((self.files[0], "./" + self.files[0]))
        with self.assertRaisesRegex(ValueError, "cannot also"):
            self.prepare(("P.csv",))
        self.write(self.files[2], np.stack([self.base]*5), shift=.001)
        with self.assertRaisesRegex(ValueError, "native spectral grid"):
            self.generate()
        self.write(self.files[2], np.stack([self.base]*5), gate_shift=.1)
        with self.assertRaisesRegex(ValueError, "gate coordinates"):
            self.generate()

    def test_cancellation_and_wrong_request_leave_no_store(self):
        request = self.prepare()
        with self.assertRaises(LoadCancelled):
            A.generate_assisted_background(self.folder, self.pair, request=request, cancelled=lambda: True)
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            A.generate_assisted_background(self.folder, self.pair, "first", request=request)
        self.assertFalse(DrrPnStore(self.folder).directory.exists())

    def test_content_rewrite_with_restored_mtime_does_not_reuse_sample_arrays(self):
        name = "rewrite_rep.csv"
        self.write(name, np.full((2, len(self.energy)), 1000.))
        path = self.folder/name
        first = self.prepare((name,))
        values, _, _ = A.load_sample_stack(self.folder, (name,), np.round(self.energy, 9), self.energy)
        np.testing.assert_array_equal(values, 1000.)
        stamp = path.stat()
        contents = path.read_bytes()
        self.assertIn(b"1000.0", contents)
        path.write_bytes(contents.replace(b"1000.0", b"2000.0"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertEqual(path.stat().st_size, stamp.st_size)
        second = self.prepare((name,))
        self.assertNotEqual(first.key.fingerprint, second.key.fingerprint)
        values, _, _ = A.load_sample_stack(self.folder, (name,), np.round(self.energy, 9), self.energy)
        np.testing.assert_array_equal(values, 2000.)

    def test_different_known_measurement_conditions_are_not_pooled_as_repeats(self):
        files = ("YZ100_9T_RotIn17deg_rep01.csv", "YZ100_9T_RotIn31deg_rep02.csv")
        for name in files:
            self.write(name, np.stack([self.base]*5))
        with self.assertRaisesRegex(ValueError, "different rotation"):
            self.generate(self.prepare(files))


if __name__ == "__main__":
    unittest.main()
