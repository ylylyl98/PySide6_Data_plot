import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_reconstruction import app, io, pipeline
from core.drr_pn_store import DrrPnStore
from core.load_cancellation import LoadCancelled


class ReconstructionAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.columns = [str(x) for x in np.linspace(650, 780, 31)]
        self.write("P.csv", 10, 4)
        self.write("N.csv", 20, 4)
        for i in range(1, 5):
            self.write(f"measurement_rep0{i}.csv", 30 + i, 3, sample=True)
        self.files = tuple(f"measurement_rep0{i}.csv" for i in range(1, 5))
        self.training = self.files[::2]
        self.pair = DrrPnPair(("P.csv",), ("N.csv",))

    def write(self, name, level, count, sample=False):
        rows = ["Vbg,Vtg,Vbias," + ",".join(self.columns)]
        for i in range(count):
            g = i - 1 if sample else 0
            rows.append(f"{g},{g},0," + ",".join(map(str, level + i + np.arange(31)/100)))
        (self.root / name).write_text("\n".join(rows) + "\n", encoding="utf8")

    def prepare(self, **kwargs):
        return app.prepare_reconstruction_request(self.root, self.pair,
            kwargs.get("measurements", self.files), training_files=kwargs.get("training_files", self.training),
            method=kwargs.get("method", "generalized"))

    def test_only_training_content_defines_generation_identity(self):
        first = self.prepare()
        self.write(self.files[1], 99, 3, sample=True)
        changed_check = self.prepare()
        self.assertEqual(first.key.fingerprint, changed_check.key.fingerprint)
        self.write(self.training[0], 42, 3, sample=True)
        self.assertNotEqual(first.key.fingerprint, self.prepare().key.fingerprint)
        self.assertEqual(first.evaluation_files, self.files[1::2])
        self.assertFalse((self.root / "Processed Data").exists())

    def test_explicit_roles_and_aliases_are_checked(self):
        for train in ((self.training[0],), (self.training[0], str(self.root/self.training[0])), ("P.csv", self.training[1])):
            with self.subTest(train=train), self.assertRaises(ValueError):
                self.prepare(training_files=train)
        with self.assertRaisesRegex(ValueError, "one original file"):
            app.prepare_reconstruction_request(self.root, DrrPnPair(("P.csv", "extra_P.csv"), ("N.csv",)),
                self.files, training_files=self.training)

    def test_stale_request_rejected_before_generation(self):
        request = self.prepare()
        self.write("P.csv", 11, 4)
        with patch.object(app, "generate_from_files") as generate:
            with self.assertRaises(ValueError):
                app.generate_reconstruction_background(self.root, self.pair, request=request)
        generate.assert_not_called()

    def test_request_cannot_report_settings_the_generator_does_not_use(self):
        request = self.prepare()
        settings = json.loads(json.dumps(request.settings))
        settings["sample_analysis"]["subset"] = "cropped gate selection"
        key = DrrPnStore(self.root).prepare(self.pair, p_which="all", n_which="all",
            settings=settings, algorithm_version=pipeline.GENERALIZED_ALGORITHM_VERSION,
            dependencies=(*io.dependency_paths(), *(self.root / p for p in self.training)))
        altered = app.ReconstructionRequest(key, request.training_files, request.evaluation_files)
        with patch.object(app, "generate_from_files") as generate:
            with self.assertRaisesRegex(ValueError, "selected files or method"):
                app.generate_reconstruction_background(self.root, self.pair, request=altered)
        generate.assert_not_called()

    def test_cancelled_request_skips_preflight_and_generation(self):
        for method in ("generalized", "conditioned_bounded"):
            with self.subTest(method=method):
                request = self.prepare(method=method)
                with patch.object(app, "_verify_request") as verify, patch.object(app, "generate_from_files") as generate:
                    with self.assertRaises(LoadCancelled):
                        app.generate_reconstruction_background(self.root, self.pair, request=request,
                            cancelled=lambda: True)
                verify.assert_not_called()
                generate.assert_not_called()

    def test_cancellation_during_postflight_does_not_return_a_background(self):
        request = self.prepare()
        state = {"cancelled": False, "verifications": 0}
        verify_request = app._verify_request
        def verify(*args):
            verify_request(*args)
            state["verifications"] += 1
            if state["verifications"] == 2:
                state["cancelled"] = True
        def generated(e, pn, frames, samples, **kwargs):
            return {"energy": e.copy(), "background": pn.mean(0)}, {
                "algorithm_version": pipeline.GENERALIZED_ALGORITHM_VERSION,
                "settings": dict(pipeline.GENERALIZED_PROTOCOL), "BN_used": False}
        with patch.object(app, "_verify_request", side_effect=verify), patch.object(
                pipeline, "reconstruct_background", side_effect=generated):
            with self.assertRaises(LoadCancelled):
                app.generate_reconstruction_background(self.root, self.pair, request=request,
                    cancelled=lambda: state["cancelled"])
        self.assertEqual(state["verifications"], 2)
        self.assertFalse((self.root / "Processed Data").exists())

    def test_adaptive_method_provenance_and_store_round_trip(self):
        request = self.prepare()
        def generated(e, pn, frames, samples, **kwargs):
            self.assertEqual(kwargs["method"], "generalized")
            self.assertEqual(samples.shape, (2, 3, 31))
            return {"energy": e.copy(), "background": pn.mean(0)}, {
                "algorithm_version": pipeline.GENERALIZED_ALGORITHM_VERSION,
                "settings": dict(pipeline.GENERALIZED_PROTOCOL), "BN_used": False}
        with patch.object(pipeline, "reconstruct_background", side_effect=generated):
            result = app.generate_reconstruction_background(self.root, self.pair, request=request)
        self.assertEqual(result.report["settings"], request.settings)
        self.assertEqual(json.loads(result.document.sources[-1].generation_json), result.report)
        self.assertFalse((self.root / "Processed Data").exists())
        store = DrrPnStore(self.root)
        version = store.save(result.document, request.key)
        self.assertEqual(version.algorithm_version, pipeline.GENERALIZED_ALGORITHM_VERSION)
        self.assertTrue(version.fresh)
        self.assertTrue(version.is_draft)
        self.assertEqual(store._read()["applied"], {})

    def test_method_is_derived_from_frozen_algorithm_and_changes_identity(self):
        adaptive = app.prepare_reconstruction_request(self.root, self.pair, self.files,
                                                       training_files=self.training)
        bounded = self.prepare(method="conditioned_bounded")
        self.assertEqual(adaptive.method, "generalized")
        self.assertEqual(bounded.method, "conditioned_bounded")
        self.assertEqual(bounded.key.provenance["algorithm_version"], pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION)
        self.assertNotEqual(adaptive.key.fingerprint, bounded.key.fingerprint)
        self.assertNotEqual(adaptive.settings["rendering"], bounded.settings["rendering"])
        with self.assertRaises(FrozenInstanceError):
            bounded.method = "generalized"
        settings = bounded.settings
        settings["rendering"] = "changed"
        self.assertEqual(bounded.settings["rendering"], pipeline.CONDITIONED_BOUNDED_PROTOCOL["rendering"])
        self.write(self.files[1], 111, 3, sample=True)
        self.assertEqual(bounded.key.fingerprint, self.prepare(method="conditioned_bounded").key.fingerprint)

    def test_bounded_route_provenance_and_cache_are_separate_from_adaptive(self):
        request = self.prepare(method="conditioned_bounded")
        adaptive = self.prepare()
        events = []
        def progress(stage, event):
            events.append((stage, event))
        cancelled = lambda: False
        def generated(e, pn, frames, samples, **kwargs):
            self.assertEqual(kwargs["method"], "conditioned_bounded")
            self.assertIs(kwargs["cancelled"], cancelled)
            self.assertIs(kwargs["progress"], progress)
            kwargs["progress"]("bounded_texture", "complete")
            return {"energy": e.copy(), "background": pn.mean(0)}, {
                "algorithm_version": pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION,
                "settings": dict(pipeline.CONDITIONED_BOUNDED_PROTOCOL), "BN_used": False}
        with patch.object(pipeline, "reconstruct_background", side_effect=generated):
            result = app.generate_reconstruction_background(self.root, self.pair, request=request,
                                                              cancelled=cancelled, progress=progress)
        self.assertEqual(events, [("bounded_texture", "complete")])
        self.assertEqual(result.report["reconstruction_method"], "conditioned_bounded")
        self.assertEqual(result.report["settings"], request.settings)
        self.assertEqual(result.report["generation_fingerprint"], request.key.fingerprint)
        self.assertFalse(result.report["scientifically_validated"])
        self.assertFalse(result.report["automatic_application"])
        self.assertIn("bounded", result.document.sources[-1].label)
        self.assertEqual(json.loads(result.document.sources[-1].generation_json), result.report)
        self.assertFalse((self.root / "Processed Data").exists())
        store = DrrPnStore(self.root)
        version = store.save(result.document, request.key)
        self.assertTrue(version.fresh)
        self.assertTrue(version.is_draft)
        self.assertEqual(version.algorithm_version, pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION)
        self.assertIsNone(store.find_reusable(adaptive.key))
        self.assertEqual(store.find_reusable(request.key).version_id, version.version_id)
        self.assertEqual(store._read()["applied"], {})

    def test_unknown_app_method_rejected_before_io(self):
        for method in ("invalid", "conditioned", None, []):
            with self.subTest(method=method), patch.object(app, "DrrPnStore") as store:
                with self.assertRaisesRegex(ValueError, "Unknown P/N"):
                    self.prepare(method=method)
            store.assert_not_called()

    def test_forged_algorithm_or_protocol_key_cannot_choose_another_method(self):
        request = self.prepare(method="conditioned_bounded")
        for version in ("unknown-version", pipeline.GENERALIZED_ALGORITHM_VERSION):
            key = DrrPnStore(self.root).prepare(self.pair, p_which="all", n_which="all",
                settings=request.settings, algorithm_version=version,
                dependencies=(*io.dependency_paths(), *(self.root / p for p in self.training)))
            forged = replace(request, key=key)
            with self.subTest(version=version), patch.object(app, "generate_from_files") as generate:
                with self.assertRaises(ValueError):
                    app.generate_reconstruction_background(self.root, self.pair, request=forged)
            generate.assert_not_called()

    def test_self_consistent_key_cannot_omit_dependencies_or_rebind_sources(self):
        request = self.prepare(method="conditioned_bounded")
        store = DrrPnStore(self.root)
        missing_dependencies = store.prepare(self.pair, p_which="all", n_which="all",
            settings=request.settings, algorithm_version=pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION,
            dependencies=io.dependency_paths())
        reversed_sources = store.prepare(DrrPnPair(self.pair.n_files, self.pair.p_files),
            p_which="all", n_which="all", settings=request.settings,
            algorithm_version=pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION,
            dependencies=(*io.dependency_paths(), *(self.root / p for p in self.training)))
        reversed_sources = replace(reversed_sources, pair_id=request.key.pair_id)
        for key in (missing_dependencies, reversed_sources):
            with self.subTest(key=key.fingerprint), patch.object(app, "generate_from_files") as generate:
                with self.assertRaisesRegex(ValueError, "source or dependency identity"):
                    app.generate_reconstruction_background(self.root, self.pair,
                                                            request=replace(request, key=key))
            generate.assert_not_called()

    def test_wrong_generated_algorithm_or_protocol_is_not_relabeled(self):
        request = self.prepare(method="conditioned_bounded")
        cases = ((pipeline.GENERALIZED_ALGORITHM_VERSION, pipeline.CONDITIONED_BOUNDED_PROTOCOL),
                 (pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION, pipeline.GENERALIZED_PROTOCOL))
        for version, protocol in cases:
            def generated(e, pn, frames, samples, **kwargs):
                return {"energy": e.copy(), "background": pn.mean(0)}, {
                    "algorithm_version": version, "settings": dict(protocol), "BN_used": False}
            with self.subTest(version=version, protocol=protocol["rendering"]), patch.object(
                    pipeline, "reconstruct_background", side_effect=generated):
                with self.assertRaisesRegex(ValueError, "method/protocol"):
                    app.generate_reconstruction_background(self.root, self.pair, request=request)
        self.assertFalse((self.root / "Processed Data").exists())

    def test_unknown_method_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown P/N"):
            pipeline.reconstruct_background(None, None, None, None, method="invalid")

    def test_explicit_regeneration_bypasses_source_fit_reuse(self):
        request = self.prepare(method="conditioned_bounded")
        def generated(e, pn, frames, samples, **kwargs):
            self.assertTrue(kwargs["force_source_refit"])
            self.assertIsNotNone(kwargs["source_cache"])
            return {"energy": e.copy(), "background": pn.mean(0)}, {
                "algorithm_version": pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION,
                "settings": dict(pipeline.CONDITIONED_BOUNDED_PROTOCOL)}
        with patch.object(pipeline, "reconstruct_background", side_effect=generated):
            result = app.generate_reconstruction_background(self.root, self.pair, request=request,
                                                             force_source_refit=True)
        self.assertFalse(result.report["scientifically_validated"])


if __name__ == "__main__":
    unittest.main()
