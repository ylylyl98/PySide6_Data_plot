import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_composite_background import compose_background, read_composite_background, save_composite_background
from core.drr_pn_reconstruction import io, pipeline


class RawReconstructionIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.columns = [str(x) for x in np.linspace(650, 780, 31)]
        self.write("P.csv", 10, 4)
        self.write("N.csv", 20, 4)
        self.write("measurement_rep01.csv", 30, 3, sample=True)
        self.write("measurement_rep03.csv", 40, 3, sample=True)
        self.training = ("measurement_rep01.csv", "measurement_rep03.csv")

    def write(self, name, level, count, *, sample=False):
        lines = ["Vbg,Vtg,Vbias," + ",".join(self.columns)]
        for i in range(count):
            gate = i - 1 if sample else 0
            values = level + i + np.arange(len(self.columns)) / 100
            lines.append(f"{gate},{gate},0," + ",".join(map(str, values)))
        (self.root / name).write_text("\n".join(lines) + "\n", encoding="utf8")

    def load(self):
        return io.load_raw_inputs(self.root, "P.csv", "N.csv", self.training)

    def test_raw_frames_native_order_and_explicit_training_only(self):
        self.write("not_selected_eval.csv", 100, 3, sample=True)
        result = self.load()
        self.assertEqual(result.pn_frames.shape, (2, 4, 31))
        self.assertEqual(result.training_samples.shape, (2, 3, 31))
        np.testing.assert_allclose(result.pn_frames.mean(1), result.pn_mean, rtol=1e-14)
        np.testing.assert_array_equal(result.energy, np.sort(1240 / np.round(np.asarray(self.columns, float), 9)))
        self.assertTrue(all(not a.flags.writeable for a in (
            result.pn_mean, result.pn_frames, result.training_samples, result.energy)))
        self.assertEqual(len(result.input_records), 4)
        self.assertNotIn("not_selected_eval.csv", json.dumps(result.input_records))

    def test_source_alias_and_sample_alias_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            io.load_raw_inputs(self.root, "P.csv", self.root / "P.csv", self.training)
        with self.assertRaisesRegex(ValueError, "distinct"):
            io.load_raw_inputs(self.root, "P.csv", "N.csv", ("P.csv", self.training[1]))

    def test_frame_policy_and_native_grid_are_validated(self):
        self.write("N.csv", 20, 5)
        with self.assertRaisesRegex(ValueError, "even count"):
            self.load()
        self.write("N.csv", 20, 6)
        with self.assertRaisesRegex(ValueError, "equal P/N"):
            self.load()
        self.write("N.csv", 20, 4)
        path = self.root / "N.csv"
        path.write_text(path.read_text().replace("650.0", "649.0"), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "same ordered"):
            self.load()

    @staticmethod
    def generated(energy, pn_mean, pn_frames, samples, **kwargs):
        return {"energy": energy.copy(), "background": pn_mean.mean(0)}, {
            "algorithm_version": "io-contract-test", "settings": {}, "BN_used": False}

    def test_file_generation_returns_native_unsaved_result_and_provenance(self):
        before = {p.name: p.read_bytes() for p in self.root.glob("*.csv")}
        with patch.object(pipeline, "reconstruct_background", side_effect=self.generated) as run:
            result = io.generate_from_files(self.root, "P.csv", "N.csv", self.training)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[3].shape, (2, 3, 31))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.glob("*.csv")})
        self.assertFalse((self.root / "Processed Data").exists())
        self.assertFalse(result.report["scientifically_validated"])
        self.assertEqual(len(result.report["input_records"]), 4)
        target = self.root / "generated.csv"
        save_composite_background(target, result.document)
        loaded = read_composite_background(target)
        np.testing.assert_array_equal(compose_background(loaded)[1], result.arrays["background"])
        with self.assertRaisesRegex(ValueError, "saved composite"):
            io.load_raw_inputs(self.root, target, "N.csv", self.training)

    def test_changed_raw_input_during_fit_discards_result(self):
        def change(*args, **kwargs):
            result = self.generated(*args, **kwargs)
            self.write("N.csv", 21, 4)
            return result
        with patch.object(pipeline, "reconstruct_background", side_effect=change):
            with self.assertRaisesRegex(ValueError, "input changed during fitting"):
                io.generate_from_files(self.root, "P.csv", "N.csv", self.training)
        self.assertFalse((self.root / "Processed Data").exists())

    def test_bounded_method_file_boundary_and_dependency_identity(self):
        dependencies = {path.name for path in io.dependency_paths()}
        self.assertTrue({"drr_pn_texture_compatibility.py", "drr_pn_texture_support.py"}.issubset(dependencies))
        with patch.object(pipeline, "reconstruct_background", side_effect=self.generated) as run:
            result = io.generate_from_files(self.root, "P.csv", "N.csv", self.training,
                                            method="conditioned_bounded")
        self.assertEqual(run.call_args.kwargs["method"], "conditioned_bounded")
        self.assertTrue(any(Path(p).name == "drr_pn_texture_support.py"
                            for p in result.report["implementation_sha256"]))
        self.assertFalse(result.report["scientifically_validated"])
        self.assertFalse((self.root / "Processed Data").exists())

    def test_texture_code_change_during_fit_discards_result(self):
        digest = io._file_digest
        state = {"changed": False}
        def checked_digest(path):
            if Path(path).name == "drr_pn_texture_support.py" and state["changed"]:
                return "changed-implementation"
            return digest(path)
        def generate(*args, **kwargs):
            state["changed"] = True
            return self.generated(*args, **kwargs)
        with patch.object(io, "_file_digest", side_effect=checked_digest), patch.object(
                pipeline, "reconstruct_background", side_effect=generate):
            with self.assertRaisesRegex(ValueError, "implementation changed during fitting"):
                io.generate_from_files(self.root, "P.csv", "N.csv", self.training,
                                       method="conditioned_bounded")
        self.assertFalse((self.root / "Processed Data").exists())

    def test_invalid_samples_rejected_before_fitting(self):
        with patch("core.drr_pn_reconstruction.source_pipeline.prepare_sources") as fit:
            with self.assertRaisesRegex(ValueError, "two positive raw training"):
                pipeline.reconstruct_background(np.arange(31), np.ones((2, 31)),
                    np.ones((2, 4, 31)), np.ones((1, 3, 31)))
        fit.assert_not_called()

    def test_source_reuse_identity_covers_pn_bytes_but_excludes_training(self):
        calls = []
        def generated(*args, **kwargs):
            calls.append((kwargs["source_cache_identity"], args[3].copy()))
            return self.generated(*args, **kwargs)
        with patch.object(pipeline, "reconstruct_background", side_effect=generated):
            io.generate_from_files(self.root, "P.csv", "N.csv", self.training)
            self.write(self.training[0], 55, 3, sample=True)
            io.generate_from_files(self.root, "P.csv", "N.csv", self.training)
            self.write("N.csv", 25, 4)
            io.generate_from_files(self.root, "P.csv", "N.csv", self.training)
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertFalse(np.array_equal(calls[0][1], calls[1][1]))
        self.assertNotEqual(calls[1][0], calls[2][0])
        self.assertFalse((self.root / "Processed Data").exists())


if __name__ == "__main__":
    unittest.main()
