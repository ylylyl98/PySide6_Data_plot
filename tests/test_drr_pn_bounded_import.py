"""Strict native-import proof for the optional conditioned bounded method.

The native loader/writer and identity validator are real. Only the expensive
scientific fitter is stubbed; these tests do not certify a reconstruction.
"""
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.drr_composite_background import save_composite_background
from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_reconstruction import io, pipeline
from core.drr_pn_store import DrrPnStore
from tests import test_drr_pn_reconstruction_io as io_fixture
from ui_qt import drr_pn_background_dialog as dialog_module


class ConditionedBoundedImportTests(unittest.TestCase):
    def setUp(self):
        fixture = io_fixture.RawReconstructionIOTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.training = fixture.root, fixture.training
        self.pair = DrrPnPair(("P.csv",), ("N.csv",))

        # Snapshot the real dependency graph into the temporary fixture. A
        # changed dependency can then be exercised without editing repo files.
        directory = self.root / "implementation"
        directory.mkdir()
        paths = tuple(io.dependency_paths())
        self.assertEqual(len(paths), len({path.name.casefold() for path in paths}))
        for path in paths:
            (directory / path.name).write_bytes(path.read_bytes())
        self.code = tuple(directory / path.name for path in paths)
        self.api = SimpleNamespace(dependency_paths=lambda: self.code)

        def fitted(energy, pn_mean, _pn_frames, _samples, **kwargs):
            self.assertEqual(kwargs["method"], "conditioned_bounded")
            return {"background": pn_mean.mean(axis=0)}, {
                "algorithm_version": pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION,
                "settings": dict(pipeline.CONDITIONED_BOUNDED_PROTOCOL),
                "source_models_refitted_from_raw": True,
                "scientifically_validated": False,
            }

        with patch.object(io, "dependency_paths", return_value=self.code), patch.object(
                pipeline, "reconstruct_background", side_effect=fitted):
            self.result = io.generate_from_files(
                self.root, "P.csv", "N.csv", self.training, method="conditioned_bounded")
        self.path = self.root / "conditioned_bounded.csv"
        save_composite_background(self.path, self.result.document)

    def validate(self, *, settings=None, path=None, pair=None, p_which="all"):
        return dialog_module._reconstruction_import_dependencies(
            self.root, path or self.path,
            self.result.report["settings"] if settings is None else settings, self.api,
            pair=self.pair if pair is None else pair, p_which=p_which, n_which="all")

    def save_report(self, report, filename):
        source = next(s for s in self.result.document.sources
                      if s.key == self.result.document.default_source)
        changed = replace(source, generation_json=json.dumps(report))
        document = replace(self.result.document, sources=tuple(
            changed if s.key == source.key else s for s in self.result.document.sources))
        target = self.root / filename
        save_composite_background(target, document)
        return target

    def snapshot(self):
        return {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.root.rglob("*") if path.is_file()}

    def test_known_protocol_validates_without_store_or_application_side_effects(self):
        before = self.snapshot()
        with ExitStack() as stack:
            writes = [stack.enter_context(patch.object(DrrPnStore, name))
                      for name in ("save", "import_saved", "mark_used")]
            dependencies = self.validate()
        self.assertEqual(dependencies, tuple(path.resolve() for path in
                         (*self.code, *(self.root / name for name in self.training))))
        for write in writes:
            write.assert_not_called()
        self.assertEqual(before, self.snapshot())
        self.assertFalse((self.root / "Processed Data").exists())
        self.assertFalse(self.result.report["scientifically_validated"])

    def test_tampered_settings_rejected_even_when_sidecar_agrees(self):
        changed = {**self.result.report["settings"], "texture_support": "unreviewed_support"}
        with self.assertRaisesRegex(ValueError, "settings.*protocol"):
            self.validate(settings=changed)
        path = self.save_report({**self.result.report, "settings": changed}, "changed_settings.csv")
        with self.assertRaisesRegex(ValueError, "settings.*protocol"):
            self.validate(settings=changed, path=path)

    def test_unknown_method_cannot_bypass_allowlist(self):
        report = {**self.result.report, "algorithm_version": "pn_raw_conditioned_bounded_texture_v999"}
        report.pop("source_models_refitted_from_raw")
        path = self.save_report(report, "unknown_method.csv")
        with self.assertRaisesRegex(ValueError, "Unsupported reconstruction algorithm"):
            self.validate(path=path)

    def test_changed_texture_dependencies_rejected_without_modifying_repository(self):
        for name in ("drr_pn_texture_compatibility.py", "drr_pn_texture_support.py"):
            with self.subTest(dependency=name):
                target = next(path for path in self.code if path.name == name)
                original = target.read_bytes()
                target.write_bytes(original + b"\n# changed dependency identity for this test\n")
                try:
                    with self.assertRaisesRegex(ValueError, "implementation.*changed"):
                        self.validate()
                finally:
                    target.write_bytes(original)

    def test_missing_texture_identity_is_not_accepted(self):
        report = {**self.result.report,
                  "implementation_sha256": {path: digest for path, digest
                      in self.result.report["implementation_sha256"].items()
                      if Path(path).name != "drr_pn_texture_support.py"}}
        path = self.save_report(report, "missing_texture_identity.csv")
        with self.assertRaisesRegex(ValueError, "incomplete implementation"):
            self.validate(path=path)

    def test_existing_raw_role_frame_and_training_guards_remain_in_force(self):
        with self.assertRaisesRegex(ValueError, "original P/N roles"):
            self.validate(pair=DrrPnPair(self.pair.n_files, self.pair.p_files))
        with self.assertRaisesRegex(ValueError, "All frames"):
            self.validate(p_which="first")
        training = self.root / self.training[0]
        training.write_text(training.read_text(encoding="utf8") + "\n", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "input.*changed"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
