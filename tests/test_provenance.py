from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch
from core import provenance

from core.provenance import (
    WorkingCopyRecord,
    can_cleanup,
    cleanup_working_copy,
    create_working_copy,
    verify_initial_data_working_file,
)


class ProvenanceTests(unittest.TestCase):
    def pair(self, root):
        canonical = root / 'Initial Data' / 'sample.csv'
        canonical.parent.mkdir()
        canonical.write_bytes(b'original')
        working = root / canonical.name
        working.write_bytes(b'original')
        return canonical, working

    def test_unchanged_load_verification_reuses_content_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, working = self.pair(root)
            verify_initial_data_working_file(working, root, workflow='DRR', role='measurement')
            with patch.object(provenance, 'sha256_file', wraps=provenance.sha256_file) as digest:
                record = verify_initial_data_working_file(working, root, workflow='DRR', role='measurement')
            self.assertTrue(record.provenance_verified)
            self.assertEqual(digest.call_count, 0)

    def test_same_size_and_restored_mtime_change_invalidates_load_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, working = self.pair(root)
            verify_initial_data_working_file(working, root, workflow='DRR', role='measurement')
            before = working.stat()
            working.write_bytes(b'modified')
            os.utime(working, ns=(before.st_atime_ns, before.st_mtime_ns))
            record = verify_initial_data_working_file(working, root, workflow='DRR', role='measurement')
            self.assertFalse(record.provenance_verified)

    def test_replacement_and_missing_file_invalidate_load_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, working = self.pair(root)
            verify_initial_data_working_file(working, root, workflow='DRR', role='measurement')
            before = working.stat()
            replacement = root / 'replacement.csv'
            replacement.write_bytes(b'modified')
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
            replacement.replace(working)
            self.assertFalse(verify_initial_data_working_file(working, root, workflow='DRR', role='measurement').provenance_verified)
            working.unlink()
            self.assertFalse(verify_initial_data_working_file(working, root, workflow='DRR', role='measurement').provenance_verified)

    def test_unstable_hash_is_unverified_and_never_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, working = self.pair(root)
            original_hash = provenance.sha256_file
            def mutate(path):
                value = original_hash(path)
                if Path(path).resolve() == working.resolve():
                    # Make the mutation observable even if rapid Windows
                    # writes share the same timestamp. Same-size/restored-
                    # mtime changes are covered by their own test above.
                    working.write_bytes(b'modified during hashing')
                return value
            with patch.object(provenance, 'sha256_file', side_effect=mutate):
                record = verify_initial_data_working_file(working, root, workflow='DRR', role='measurement')
            self.assertFalse(record.provenance_verified)
            self.assertFalse(verify_initial_data_working_file(working, root, workflow='DRR', role='measurement').provenance_verified)

    def test_cleanup_uses_fresh_hashes_after_warm_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, working = self.pair(root)
            record = verify_initial_data_working_file(working, root, workflow='DRR', role='measurement')
            with patch.object(provenance, 'sha256_file', wraps=provenance.sha256_file) as digest:
                self.assertTrue(can_cleanup(record, root))
            self.assertEqual(digest.call_count, 2)
            working.write_bytes(b'modified')
            self.assertFalse(can_cleanup(record, root))

    def test_copy_records_hashes_and_cleanup_preserves_canonical(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            root = Path(tmp)
            canonical = root / "Initial Data" / "sample with spaces.csv"
            working = root / "working folder"
            canonical.parent.mkdir()
            canonical.write_text("canonical", encoding="utf-8")
            record = create_working_copy(canonical, working, workflow="PL", role="measurement")
            self.assertTrue(record.provenance_verified)
            self.assertTrue(can_cleanup(record, working))
            self.assertTrue(cleanup_working_copy(record, working))
            self.assertTrue(canonical.exists())
            self.assertEqual(canonical.read_text(encoding="utf-8"), "canonical")
            self.assertFalse(Path(record.working_copy_path).exists())

    def test_modified_working_copy_fails_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "Initial Data" / "sample.csv"
            working = root / "processing"
            canonical.parent.mkdir()
            canonical.write_text("original", encoding="utf-8")
            record = create_working_copy(canonical, working, workflow="DRR", role="measurement")
            Path(record.working_copy_path).write_text("changed", encoding="utf-8")
            self.assertFalse(cleanup_working_copy(record, working))
            self.assertTrue(Path(record.working_copy_path).exists())

    def test_unverified_and_outside_root_records_are_never_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "Initial Data" / "sample.csv"
            working = root / "processing" / "sample.csv"
            canonical.parent.mkdir()
            working.parent.mkdir()
            canonical.write_text("original", encoding="utf-8")
            working.write_text("copy", encoding="utf-8")
            record = WorkingCopyRecord(str(canonical), str(working), None, None, "Compare", "source", False, False)
            self.assertFalse(cleanup_working_copy(record, root / "processing"))
            self.assertTrue(working.exists())

    def test_manual_initial_data_copy_is_verified_without_app_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "YZ327"
            canonical = root / "Initial Data" / "sample.csv"
            working = root / "sample.csv"
            canonical.parent.mkdir(parents=True)
            canonical.write_text("same", encoding="utf-8")
            working.write_text("same", encoding="utf-8")
            record = verify_initial_data_working_file(
                working, root, workflow="PL", role="measurement"
            )
            self.assertTrue(record.provenance_verified)
            self.assertTrue(record.temporary_working_copy)
            self.assertFalse(record.app_managed)
            self.assertEqual(record.verification_method, "initial_data_filename_sha256")
            self.assertTrue(cleanup_working_copy(record, root))
            self.assertTrue(canonical.exists())

    def test_direct_initial_data_file_is_never_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "YZ327"
            canonical = root / "Initial Data" / "sample.csv"
            canonical.parent.mkdir(parents=True)
            canonical.write_text("same", encoding="utf-8")
            record = verify_initial_data_working_file(
                canonical, root, workflow="DRR", role="background"
            )
            self.assertFalse(record.provenance_verified)
            self.assertFalse(record.temporary_working_copy)
            self.assertFalse(cleanup_working_copy(record, root))
            self.assertTrue(canonical.exists())

    def test_hash_mismatch_and_missing_canonical_are_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "YZ327"
            initial = root / "Initial Data"
            initial.mkdir(parents=True)
            working = root / "sample.csv"
            working.write_text("working", encoding="utf-8")
            canonical = initial / "sample.csv"
            canonical.write_text("canonical", encoding="utf-8")
            mismatch = verify_initial_data_working_file(working, root, workflow="Compare", role="source_KK")
            self.assertFalse(mismatch.provenance_verified)
            self.assertTrue(working.exists())
            canonical.unlink()
            missing = verify_initial_data_working_file(working, root, workflow="Compare", role="source_KKp")
            self.assertFalse(missing.provenance_verified)
            self.assertTrue(working.exists())

    def test_app_managed_copy_uses_collision_safe_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "Initial Data" / "sample.csv"
            working_root = root / "working"
            canonical.parent.mkdir(parents=True)
            working_root.mkdir()
            canonical.write_text("source", encoding="utf-8")
            (working_root / canonical.name).write_text("unrelated", encoding="utf-8")
            record = create_working_copy(canonical, working_root, workflow="PL", role="measurement")
            self.assertEqual(Path(record.working_copy_path).name, "sample__working_1.csv")
            self.assertEqual(Path(record.working_copy_path).read_text(encoding="utf-8"), "source")
