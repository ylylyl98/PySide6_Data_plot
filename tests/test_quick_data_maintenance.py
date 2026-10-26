"""Data maintenance checks touch temporary fixture directories exclusively."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_qt.quick.data_maintenance import (ARCHIVE_NAME, capture_export_sources, cleanup_verified_sources,
    finish_export, move_exported_sources, movable_sources, run_export_with_maintenance, validate_receipt)


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="quick-maintenance-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.working = self.root / "sample with spaces.csv"
        self.working.write_bytes(b"original scientific measurement")
        self.canonical = self.root / "Initial Data" / self.working.name
        self.canonical.parent.mkdir()
        self.canonical.write_bytes(self.working.read_bytes())
        self.output = self.root / "Processed Data" / "result.csv"
        self.output.parent.mkdir()

    def export(self):
        self.output.write_bytes(b"successfully exported result")
        return {"csv": self.output}

    def run_export(self, **extra):
        return run_export_with_maintenance(self.export, folder=self.root, sources=[self.working.name], workflow="PL", **extra)

    def test_success_does_not_move_or_clean_by_default_and_archival_is_explicit(self):
        result = self.run_export()
        self.assertTrue(self.working.exists())
        self.assertEqual(result.cleaned, 0)
        self.assertTrue(validate_receipt(result.receipt))
        self.assertEqual(move_exported_sources(result.receipt), 1)
        self.assertFalse(self.working.exists())
        self.assertEqual((self.root / ARCHIVE_NAME / self.working.name).read_bytes(), self.canonical.read_bytes())
        self.assertTrue(self.canonical.exists())

    def test_cleanup_is_opt_in_and_never_removes_originals_or_unverified_files(self):
        unverified = self.root / "unverified.csv"
        unverified.write_bytes(b"no identical canonical copy")
        external = self.root / "external" / "source.csv"
        external.parent.mkdir()
        external.write_bytes(b"separate source")
        result = run_export_with_maintenance(self.export, folder=self.root,
            sources=[self.working, unverified, self.canonical, external], workflow="DRR", cleanup_verified=True)
        self.assertEqual(result.cleaned, 1)
        self.assertFalse(self.working.exists())
        self.assertTrue(self.canonical.exists())
        self.assertTrue(unverified.exists())
        self.assertTrue(external.exists())
        self.assertEqual([Path(item.path).name for item in movable_sources(result.receipt)], ["unverified.csv"])
        self.assertTrue(validate_receipt(result.receipt))

    def test_cleanup_not_available_for_other_workflows(self):
        result = run_export_with_maintenance(self.export, folder=self.root, sources=[self.working], workflow="Power", cleanup_verified=True)
        self.assertEqual(result.cleaned, 0)
        self.assertTrue(self.working.exists())

    def test_export_failure_and_cancellation_never_delete_sources(self):
        def failing():
            raise ValueError("Simulated export failure")
        with self.assertRaisesRegex(ValueError, "Simulated"):
            run_export_with_maintenance(failing, folder=self.root, sources=[self.working], workflow="PL", cleanup_verified=True)
        self.assertTrue(self.working.exists())
        with self.assertRaises(InterruptedError):
            self.run_export(cleanup_verified=True, cancelled=lambda: True)
        self.assertTrue(self.working.exists())
        cancelled = False
        def finish_then_cancel():
            nonlocal cancelled
            result = self.export()
            cancelled = True
            return result
        with self.assertRaises(InterruptedError):
            run_export_with_maintenance(finish_then_cancel, folder=self.root, sources=[self.working], workflow="PL",
                cleanup_verified=True, cancelled=lambda: cancelled)
        self.assertTrue(self.working.exists())
        self.assertFalse((self.root / ARCHIVE_NAME).exists())

    def test_changed_loaded_version_prevents_export_and_cleanup(self):
        callback = Mock(side_effect=self.export)
        with self.assertRaisesRegex(ValueError, "changed since loading"):
            run_export_with_maintenance(callback, folder=self.root, sources=[self.working], workflow="PL",
                cleanup_verified=True, expected_hashes={self.working.name: "0" * 64})
        callback.assert_not_called()
        self.assertTrue(self.working.exists())

    def test_changed_source_during_export_or_after_success_cannot_be_moved(self):
        def changed_export():
            result = self.export()
            before = self.working.stat()
            self.working.write_bytes(b"modified scientific measurement")
            os.utime(self.working, ns=(before.st_atime_ns, before.st_mtime_ns))
            return result
        with self.assertRaisesRegex(ValueError, "changed during saving"):
            run_export_with_maintenance(changed_export, folder=self.root, sources=[self.working], workflow="PL", cleanup_verified=True)
        self.assertTrue(self.working.exists())
        receipt = self.run_export().receipt
        self.working.write_bytes(b"modified after export")
        with self.assertRaisesRegex(ValueError, "changed after saving"):
            move_exported_sources(receipt)
        self.assertTrue(self.working.exists())
        self.assertFalse((self.root / ARCHIVE_NAME).exists())

    def test_changed_or_missing_export_output_prevents_source_archival(self):
        result = self.run_export()
        self.output.write_bytes(b"replaced export")
        with self.assertRaises(ValueError):
            move_exported_sources(result.receipt)
        self.assertTrue(self.working.exists())
        result = self.run_export()
        self.output.unlink()
        with self.assertRaises(ValueError):
            move_exported_sources(result.receipt)
        self.assertTrue(self.working.exists())

    def test_missing_or_incomplete_outputs_never_trigger_cleanup(self):
        with self.assertRaisesRegex(ValueError, "No successful exported artifacts"):
            run_export_with_maintenance(lambda: None, folder=self.root, sources=[self.working], workflow="PL", cleanup_verified=True)
        self.assertTrue(self.working.exists())
        with self.assertRaisesRegex(ValueError, "artifact is missing"):
            run_export_with_maintenance(lambda: {"csv": self.root / "missing.csv"}, folder=self.root,
                sources=[self.working], workflow="PL", cleanup_verified=True)
        self.assertTrue(self.working.exists())

    def test_changed_canonical_copy_and_cancelled_cleanup_keep_working_copy(self):
        request = capture_export_sources(self.root, [self.working], "PL", cleanup_verified=True)
        receipt = finish_export(request, self.export())
        with self.assertRaises(InterruptedError):
            cleanup_verified_sources(receipt, cancelled=lambda: True)
        self.assertTrue(self.working.exists())
        self.canonical.write_bytes(b"canonical changed after export")
        result = cleanup_verified_sources(receipt)
        self.assertEqual(result.cleaned, ())
        self.assertTrue(self.working.exists())
        self.assertTrue(self.canonical.exists())

    def test_archive_scope_is_root_csv_and_collisions_do_not_overwrite(self):
        archive = self.root / ARCHIVE_NAME
        archive.mkdir()
        original_archive = archive / self.working.name
        original_archive.write_bytes(b"previously archived data")
        data = self.root / "map.dat"
        data.write_bytes(b"map")
        nested = self.root / "nested" / "source.csv"
        nested.parent.mkdir()
        nested.write_bytes(b"nested data")
        result = run_export_with_maintenance(self.export, folder=self.root, sources=[self.working, self.canonical, data, nested], workflow="PL")
        self.assertEqual(move_exported_sources(result.receipt), 1)
        self.assertEqual(original_archive.read_bytes(), b"previously archived data")
        self.assertTrue((archive / (self.working.stem + "_01.csv")).exists())
        self.assertTrue(data.exists())
        self.assertTrue(nested.exists())
        self.assertTrue(self.canonical.exists())

    def test_native_bridge_async_receipts_context_and_changed_file_guards(self):
        environment = dict(os.environ, PYTHONIOENCODING="utf-8")
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
            cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("MAINTENANCE_BRIDGE_OK", result.stdout)


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import time
    class NoWidgets(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.main_window", "ui_qt.common"}:
                raise AssertionError("Native maintenance imported Widgets: " + fullname)
    sys.meta_path.insert(0, NoWidgets())
    from PySide6.QtCore import QCoreApplication, QSettings
    from ui_qt.quick.data_maintenance_bridge import DataMaintenanceBridge
    app = QCoreApplication([])
    def wait(predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(.003)
        raise AssertionError("Timed out waiting for maintenance worker")
    with TemporaryDirectory(prefix="quick-maintenance-bridge-") as temporary:
        root = Path(temporary)
        settings = QSettings(str(root / "isolated.ini"), QSettings.Format.IniFormat)
        settings.setValue("unrelated/preserved", "keep")
        source, output = root / "sample.csv", root / "result.png"
        source.write_bytes(b"source")
        def export():
            output.write_bytes(b"output")
            return output
        bridge = DataMaintenanceBridge()
        refresh = []
        bridge.catalogRefreshRequested.connect(refresh.append)
        assert not bridge.snapshot["canMove"]
        assert not bridge.snapshot["cleanupVerifiedSources"]
        bridge.setContext((str(root), "PL", 1))
        receipt = run_export_with_maintenance(export, folder=root, sources=[source], workflow="PL")
        bridge.acceptExport(receipt, context=(str(root), "PL", 0))
        assert not bridge.snapshot["busy"]
        assert not bridge.snapshot["canMove"]
        bridge.acceptExport(receipt, context=(str(root), "PL", 1))
        wait(lambda: not bridge.snapshot["busy"])
        assert bridge.snapshot["canMove"]
        assert refresh == [str(root)]
        source.write_bytes(b"changed")
        bridge.invoke("move")
        wait(lambda: not bridge.snapshot["busy"])
        assert source.exists()
        assert not bridge.snapshot["canMove"]
        assert "changed" in bridge.snapshot["error"]
        receipt = run_export_with_maintenance(export, folder=root, sources=[source], workflow="PL")
        bridge.acceptExport(receipt)
        wait(lambda: not bridge.snapshot["busy"])
        bridge.setContext((str(root), "PL", 2))
        assert not bridge.snapshot["canMove"]
        bridge.acceptExport(receipt)
        wait(lambda: not bridge.snapshot["busy"])
        bridge.invoke("move")
        wait(lambda: not bridge.snapshot["busy"])
        assert not source.exists()
        assert (root / ARCHIVE_NAME / "sample.csv").read_bytes() == b"changed"
        assert settings.value("unrelated/preserved") == "keep"
        bridge.close()
    print("MAINTENANCE_BRIDGE_OK")
    return 0


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
