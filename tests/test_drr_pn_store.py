from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_composite_background import (
    CompositeBackground, CompositeSource, composite_metadata_path, compose_background,
    load_composite_source, read_composite_background, save_composite_background,
)
from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_store import DrrPnStore
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


class PnStoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = DrrPnStore(self.root)
        self.pair = DrrPnPair(("Pbackground.csv",), ("Nbackground.csv",))
        self.columns = tuple(f"{value:.6f}" for value in np.linspace(700., 760., 31))
        for name, gate, value in (("Pbackground.csv", -20, 10), ("Nbackground.csv", 20, 20),
                                  ("sample1.csv", -6, 30), ("sample2.csv", 6, 40)):
            self.write_csv(name, gate, value)
        (self.root / "algorithm.py").write_text("version=1\n", encoding="utf-8")

    def write_csv(self, name, gate, value):
        header = "Vbg,Vtg,Vbias," + ",".join(self.columns)
        rows = [f"{gate},0,0," + ",".join([str(value + shift)] * len(self.columns)) for shift in (0, 1)]
        (self.root / name).write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    def key(self, **kwargs):
        options = dict(settings={"width": 4}, algorithm_version="test-v1", dependencies=("algorithm.py",))
        options.update(kwargs)
        return self.store.prepare(self.pair, **options)

    def document(self, key=None):
        key = key or self.key()
        provenance = key.provenance
        p = load_composite_source(self.root, self.pair.p_files, provenance["frame_choice"]["P"], key="P")
        n = load_composite_source(self.root, self.pair.n_files, provenance["frame_choice"]["N"], key="N")
        report = dict(algorithm_version=provenance["algorithm_version"],
                      settings=provenance["settings"], input_source_keys=["P", "N"])
        generated = CompositeSource(
            "generated", "Test generated result", p.files + n.files, "all", p.energy,
            p.spectral_columns, 0.7 * p.intensity + 0.3 * n.intensity,
            source_hashes=p.source_hashes + n.source_hashes, generation_json=json.dumps(report),
        )
        return CompositeBackground((p, n, generated), "generated")

    def test_native_roundtrip_reuse_and_explicit_last_used(self):
        key = self.key()
        document = self.document(key)
        before = {name: (self.root / name).read_bytes() for name in (*self.pair.p_files, *self.pair.n_files)}
        version = self.store.save(document, key)
        self.assertTrue(version.fresh)
        self.assertIsNone(self.store.last_used(self.pair))
        result = read_composite_background(version.csv_path)
        np.testing.assert_array_equal(compose_background(result)[1], compose_background(document)[1])
        self.assertEqual(result.sources[0].spectral_columns, self.columns)
        self.assertEqual(self.store.find_reusable(key).version_id, version.version_id)
        self.assertTrue(self.store.save(document, key).reused)
        self.assertEqual(len(self.store.list_versions(self.pair)), 1)
        self.store.mark_used(version.version_id)
        self.assertEqual(self.store.last_used(self.pair).version_id, version.version_id)
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in before})

    def test_fingerprint_covers_roles_order_frame_settings_algorithm_dependencies(self):
        key = self.key()
        variants = (
            self.key(p_which="first"), self.key(n_which="last"),
            self.key(settings={"width": 8}), self.key(algorithm_version="test-v2"),
            self.store.prepare(DrrPnPair(self.pair.n_files, self.pair.p_files),
                               settings={"width": 4}, algorithm_version="test-v1", dependencies=("algorithm.py",)),
        )
        self.assertTrue(all(other.fingerprint != key.fingerprint for other in variants))
        self.write_csv("Pbackground_repeat.csv", -20, 15)
        a = self.store.prepare(DrrPnPair(("Pbackground.csv", "Pbackground_repeat.csv"), self.pair.n_files),
                               algorithm_version="test-v1")
        b = self.store.prepare(DrrPnPair(("Pbackground_repeat.csv", "Pbackground.csv"), self.pair.n_files),
                               algorithm_version="test-v1")
        self.assertNotEqual(a.fingerprint, b.fingerprint)
        (self.root / "algorithm.py").write_text("version=2\n", encoding="utf-8")
        self.assertNotEqual(self.key().fingerprint, key.fingerprint)

    def test_versions_preserved_and_last_used_preferred_over_newest(self):
        first_key = self.key()
        first = self.store.save(self.document(first_key), first_key)
        self.store.mark_used(first.version_id)
        second_key = self.key(algorithm_version="test-v2")
        second = self.store.save(self.document(second_key), second_key)
        self.assertEqual([item.number for item in self.store.list_versions(self.pair)], [1, 2])
        self.assertEqual(self.store.preferred_version(self.pair).version_id, first.version_id)
        self.assertEqual(second.algorithm_version, "test-v2")
        self.assertTrue(first.csv_path.exists())
        self.store.mark_used(second.version_id)
        self.assertEqual(self.store.preferred_version(self.pair).version_id, second.version_id)

    def test_get_version_validates_only_target_and_detects_changed_target(self):
        key = self.key()
        document = self.document(key)
        first = self.store.save(document, key)
        second = self.store.save(document, key, force_new=True)
        second.csv_path.write_text("unrelated damaged version", encoding="utf8")
        with patch.object(self.store, "_version", wraps=self.store._version) as validate:
            current = self.store.get_version(first.version_id)
        self.assertTrue(current.fresh)
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(validate.call_args.args[0]["version_id"], first.version_id)
        self.assertIsNone(self.store.get_version("missing"))
        first.csv_path.write_text("changed target", encoding="utf8")
        self.assertEqual(self.store.get_version(first.version_id).status, "invalid")

    def test_explicit_regenerate_keeps_same_fingerprint_as_new_immutable_version(self):
        key = self.key()
        document = self.document(key)
        first = self.store.save(document, key)
        self.store.mark_used(first.version_id)
        before = first.csv_path.read_bytes(), first.metadata_path.read_bytes()
        second = self.store.save(document, key, force_new=True)
        self.assertNotEqual(first.version_id, second.version_id)
        self.assertEqual((first.number, second.number), (1, 2))
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertFalse(second.reused)
        self.assertEqual(before, (first.csv_path.read_bytes(), first.metadata_path.read_bytes()))
        self.assertEqual(self.store.last_used(self.pair).version_id, first.version_id)
        self.assertEqual(len(self.store.list_versions(self.pair)), 2)

    def test_new_drafts_do_not_displace_saved_result_without_explicit_use(self):
        saved_key = self.key(algorithm_version="prepared-regional-result")
        saved = self.store.save(self.document(saved_key), saved_key)
        for method in ("pn_regional_curvature_v1", "pn_sample_assisted_curvature_v1"):
            key = self.key(algorithm_version=method)
            draft = self.store.save(self.document(key), key)
            self.assertTrue(draft.is_draft)
            self.assertEqual(self.store.preferred_version(self.pair).version_id, saved.version_id)
            self.assertEqual(self.store.find_reusable(key).version_id, draft.version_id)
        self.store.mark_used(draft.version_id)
        self.assertEqual(self.store.preferred_version(self.pair).version_id, draft.version_id)
        self.assertTrue(self.store.preferred_version(self.pair).is_draft)

    def test_draft_only_history_requires_explicit_choice_but_remains_reusable(self):
        key = self.key(algorithm_version="pn_regional_curvature_v1")
        draft = self.store.save(self.document(key), key)
        self.assertIsNone(self.store.preferred_version(self.pair))
        self.assertEqual(self.store.list_versions(self.pair)[0].version_id, draft.version_id)
        self.store.mark_used(draft.version_id, measurements=("sample1.csv",))
        self.assertEqual(self.store.find_applied(("sample1.csv",)).version_id, draft.version_id)
        self.assertEqual(self.store.preferred_version(self.pair).version_id, draft.version_id)

    def test_explicit_unvalidated_metadata_is_not_automatically_preferred(self):
        key = self.key(algorithm_version="future-experimental-method")
        document = self.document(key)
        generated = document.sources[-1]
        report = json.loads(generated.generation_json)
        report["scientifically_validated"] = False
        document = replace(document, sources=(*document.sources[:-1], replace(
            generated, generation_json=json.dumps(report))))
        draft = self.store.save(document, key)
        self.assertTrue(draft.is_draft)
        self.assertIsNone(self.store.preferred_version(self.pair))
        self.assertEqual(self.store.find_reusable(key).version_id, draft.version_id)

    def test_raw_algorithm_drafts_do_not_displace_saved_for_any_validation_flag(self):
        saved_key = self.key(algorithm_version="prepared-regional-result")
        saved = self.store.save(self.document(saved_key), saved_key)
        saved_bytes = saved.csv_path.read_bytes(), saved.metadata_path.read_bytes()
        drafts = []
        for flag in (None, True, False):
            with self.subTest(scientifically_validated=flag):
                key = self.key(algorithm_version="pn_raw_conditioned_bounded_texture_v1",
                               settings={"validation_case": flag})
                document = self.document(key)
                generated = document.sources[-1]
                report = json.loads(generated.generation_json)
                if flag is not None:
                    report["scientifically_validated"] = flag
                document = replace(document, sources=(*document.sources[:-1], replace(
                    generated, generation_json=json.dumps(report))))
                draft = self.store.save(document, key)
                drafts.append(draft)
                self.assertTrue(draft.fresh)
                self.assertTrue(draft.is_draft)
                self.assertEqual(self.store.find_reusable(key).version_id, draft.version_id)
                self.assertEqual(self.store.preferred_version(self.pair).version_id, saved.version_id)
                self.assertIsNone(self.store.last_used(self.pair))
                self.assertIsNone(self.store.find_applied(("sample1.csv",)))
        self.assertEqual(saved_bytes, (saved.csv_path.read_bytes(), saved.metadata_path.read_bytes()))
        for draft in drafts:
            self.store.mark_used(draft.version_id, measurements=("sample1.csv",))
            self.assertEqual(self.store.preferred_version(self.pair).version_id, draft.version_id)
            self.assertTrue(self.store.preferred_version(self.pair).is_draft)
            self.assertEqual(self.store.find_applied(("sample1.csv",)).version_id, draft.version_id)

    def test_raw_namespace_only_history_never_chooses_untried_draft(self):
        for method in ("pn_raw_reconstruction_v1", "pn_raw_future_method_v999"):
            with self.subTest(method=method):
                key = self.key(algorithm_version=method)
                draft = self.store.save(self.document(key), key)
                self.assertTrue(draft.is_draft)
                self.assertIsNone(self.store.preferred_version(self.pair))
        self.store.mark_used(draft.version_id)
        self.assertEqual(self.store.preferred_version(self.pair).version_id, draft.version_id)

    def test_nonraw_saved_result_with_true_validation_flag_keeps_preference_behavior(self):
        key = self.key(algorithm_version="prepared-regional-result")
        document = self.document(key)
        generated = document.sources[-1]
        report = {**json.loads(generated.generation_json), "scientifically_validated": True}
        document = replace(document, sources=(*document.sources[:-1], replace(
            generated, generation_json=json.dumps(report))))
        saved = self.store.save(document, key)
        self.assertFalse(saved.is_draft)
        self.assertEqual(self.store.preferred_version(self.pair).version_id, saved.version_id)

    def test_same_size_restored_mtime_content_change_is_stale(self):
        key = self.key()
        first = self.store.save(self.document(key), key)
        path = self.root / self.pair.p_files[0]
        stamp = path.stat()
        header, body = path.read_bytes().split(b"\n", 1)
        path.write_bytes(header + b"\n" + body.replace(b"10", b"12"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        old = self.store.list_versions(self.pair)[0]
        self.assertEqual(old.status, "stale")
        self.assertIsNone(self.store.preferred_version(self.pair))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.store.find_reusable(key)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.store.mark_used(first.version_id)
        self.assertEqual(self.store.mark_used(first.version_id, allow_stale=True).status, "stale")
        new_key = self.key()
        self.assertIsNone(self.store.find_reusable(new_key))
        second = self.store.save(self.document(new_key), new_key)
        self.assertEqual(second.number, 2)

    def test_missing_sources_allow_explicit_independent_history_only(self):
        key = self.key()
        version = self.store.save(self.document(key), key)
        (self.root / "Nbackground.csv").unlink()
        self.assertEqual(self.store.list_versions(self.pair)[0].status, "missing_sources")
        self.assertIsNone(self.store.preferred_version(self.pair))
        with self.assertRaises(ValueError):
            self.store.mark_used(version.version_id)
        historical = self.store.mark_used(version.version_id, allow_stale=True)
        np.testing.assert_array_equal(compose_background(read_composite_background(historical.csv_path))[1],
                                      compose_background(read_composite_background(version.csv_path))[1])
        from core.loader import build_external_baseline
        loaded = build_external_baseline(str(self.root), [str(historical.csv_path)], which="all")
        np.testing.assert_array_equal(loaded["I0"], compose_background(
            read_composite_background(historical.csv_path))[1])

    def test_applied_association_is_order_independent_and_content_checked(self):
        key = self.key()
        version = self.store.save(self.document(key), key)
        self.store.mark_used(version.version_id, measurements=("sample1.csv", "sample2.csv"))
        self.assertEqual(self.store.find_applied(("sample2.csv", "sample1.csv")).version_id, version.version_id)
        self.assertIsNone(self.store.find_applied(("sample1.csv",)))
        # Reusing this same version on another selected group is explicit and
        # does not create a new numerical generation fingerprint/version.
        self.store.mark_used(version.version_id, measurements=("sample1.csv",))
        self.assertEqual(self.store.find_applied(("sample1.csv",)).version_id, version.version_id)
        path = self.root / "sample1.csv"
        stamp = path.stat()
        path.write_bytes(path.read_bytes().replace(b"30", b"32"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.store.find_applied(("sample1.csv", "sample2.csv"))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.store.find_applied(("sample1.csv",))
        self.assertEqual(len(self.store.list_versions()), 1)

    def test_stale_generation_dependency_prevents_automatic_restore(self):
        key = self.key()
        version = self.store.save(self.document(key), key)
        self.store.mark_used(version.version_id, measurements=("sample1.csv",))
        (self.root / "algorithm.py").write_text("changed", encoding="utf-8")
        self.assertEqual(self.store.list_versions()[0].status, "stale")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.store.find_applied(("sample1.csv",))

    def test_aliases_and_different_grid_rejected_before_generation(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.store.prepare(DrrPnPair(("Pbackground.csv",), (str(self.root / "Pbackground.csv"),)),
                               algorithm_version="v1")
        path = self.root / "Nbackground.csv"
        path.write_text(path.read_text().replace("700.000000", "699.000000"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "native spectral"):
            self.key()

    def test_save_rejects_wrong_pair_frame_algorithm_and_nonfinite_outputs(self):
        key = self.key()
        document = self.document(key)
        with self.assertRaisesRegex(ValueError, "frame choice"):
            self.store.save(document, self.key(p_which="first"))
        with self.assertRaisesRegex(ValueError, "algorithm"):
            self.store.save(document, self.key(algorithm_version="v2"))
        bad = replace(document.sources[-1], intensity=np.full(len(self.columns), np.nan))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.store.save(replace(document, sources=(*document.sources[:-1], bad)), key)

    def test_csv_or_metadata_tampering_never_reuses_even_as_historical(self):
        key = self.key()
        version = self.store.save(self.document(key), key)
        version.csv_path.write_bytes(version.csv_path.read_bytes() + b"\n")
        self.assertEqual(self.store.list_versions()[0].status, "invalid")
        self.assertIsNone(self.store.find_reusable(key))
        with self.assertRaises(ValueError):
            self.store.mark_used(version.version_id, allow_stale=True)

    def test_import_preserves_frozen_result_and_provenance_without_runtime_original(self):
        document = self.document()
        path = self.root / "external_generated.csv"
        save_composite_background(path, document)
        metadata_path = composite_metadata_path(path)
        metadata = json.loads(metadata_path.read_text())
        metadata["historical_note"] = "Previously evaluated, no new generation"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        original = path.read_bytes(), metadata_path.read_bytes()
        imported = self.store.import_saved(path, self.pair, algorithm_version="historical-v64",
                                          settings={"frozen": True}, provenance={"evaluation": "user selected"})
        self.assertEqual(original, (path.read_bytes(), metadata_path.read_bytes()))
        stored_metadata = json.loads(imported.metadata_path.read_text())
        self.assertEqual(stored_metadata["pn_generation"]["imported_from"]["metadata"]["historical_note"],
                         metadata["historical_note"])
        np.testing.assert_array_equal(compose_background(read_composite_background(imported.csv_path))[1],
                                      compose_background(document)[1])
        again = self.store.import_saved(path, self.pair, algorithm_version="historical-v64",
                                       settings={"frozen": True}, provenance={"evaluation": "user selected"})
        self.assertTrue(again.reused)
        path.unlink()
        metadata_path.unlink()
        self.assertTrue(self.store.list_versions()[0].fresh)
        self.assertEqual(self.store.preferred_version(self.pair).algorithm_version, "historical-v64")
        self.assertEqual(self.store.list_versions()[0].pair, self.pair)

    def test_mark_used_rejects_measurement_grid_conflict_without_creating_association(self):
        key = self.key()
        version = self.store.save(self.document(key), key)
        path = self.root / "sample1.csv"
        path.write_text(path.read_text().replace("700.000000", "698.000000"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different native grids"):
            self.store.mark_used(version.version_id, measurements=("sample1.csv",))
        self.assertIsNone(self.store.last_used(self.pair))
        self.assertIsNone(self.store.find_applied(("sample1.csv",)))

    def test_failed_manifest_publication_preserves_earlier_version_and_index(self):
        first_key = self.key()
        first = self.store.save(self.document(first_key), first_key)
        original = self.store.manifest_path.read_bytes()
        second_key = self.key(algorithm_version="v2")
        atomic = self.store._atomic_json

        def fail_index(path, payload):
            if path == self.store.manifest_path:
                raise OSError("disk full")
            return atomic(path, payload)

        with patch.object(self.store, "_atomic_json", side_effect=fail_index), self.assertRaises(OSError):
            self.store.save(self.document(second_key), second_key)
        self.assertEqual(self.store.manifest_path.read_bytes(), original)
        self.assertEqual([item.version_id for item in self.store.list_versions()], [first.version_id])
        self.assertFalse((self.store.directory / ".store.lock").exists())

    def test_source_change_during_generation_rejected(self):
        key = self.key()
        document = self.document(key)
        self.write_csv("Pbackground.csv", -20, 99)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.store.save(document, key)
        self.assertFalse(self.store.manifest_path.exists())

    def test_cancelled_apply_does_not_publish_last_used_or_measurement_association(self):
        from core import drr_pn_store
        key = self.key()
        version = self.store.save(self.document(key), key)
        before = self.store.manifest_path.read_bytes()
        cancel = LoadCancellation()
        original = drr_pn_store.inspect_csv_spectral_grid

        def cancel_after_grid(path):
            result = original(path)
            cancel.cancel()
            return result

        with load_scope(cancel), patch.object(drr_pn_store, "inspect_csv_spectral_grid", side_effect=cancel_after_grid):
            with self.assertRaises(LoadCancelled):
                self.store.mark_used(version.version_id, measurements=("sample1.csv",))
        self.assertEqual(self.store.manifest_path.read_bytes(), before)
        self.assertIsNone(self.store.last_used(self.pair))
        self.assertIsNone(self.store.find_applied(("sample1.csv",)))

    def test_nonfinite_settings_rejected(self):
        with self.assertRaises(ValueError):
            self.key(settings={"width": float("nan")})

    def test_real_generator_result_binds_to_request_and_reuses(self):
        from core.drr_pn_generator import (
            ALGORITHM_VERSION, default_settings, dependency_paths, generate_pn_background,
        )
        settings = default_settings()
        key = self.store.prepare(self.pair, settings=settings, algorithm_version=ALGORITHM_VERSION,
                                 dependencies=dependency_paths())
        result = generate_pn_background(str(self.root), self.pair, settings=settings)
        version = self.store.save(result.document, key)
        self.assertTrue(version.fresh)
        self.assertEqual(self.store.find_reusable(key).version_id, version.version_id)
        np.testing.assert_array_equal(compose_background(read_composite_background(version.csv_path))[1],
                                      compose_background(result.document)[1])


if __name__ == "__main__":
    unittest.main()
