"""Captured source versions for explicit archival and verified-copy cleanup.

Pure functions belong in an export/maintenance worker. No application startup
scans files or mutates data. Archival retains the legacy root-CSV scope.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from core.provenance import (_digest_signature, cleanup_working_copy, sha256_file,
                             verify_initial_data_working_file)

ARCHIVE_NAME = "Initial data after processing"


@dataclass(frozen=True)
class FileVersion:
    path: str
    sha256: str
    signature: tuple


@dataclass(frozen=True)
class ExportSources:
    folder: str
    workflow: str
    sources: tuple[FileVersion, ...]
    provenance: tuple
    cleanup_verified: bool = False


@dataclass(frozen=True)
class ExportReceipt:
    request: ExportSources
    outputs: tuple[FileVersion, ...]
    cleaned: tuple[str, ...] = ()
    identity: str = ""

    @property
    def remaining_sources(self):
        return tuple(source for source in self.request.sources if source.path not in self.cleaned)


@dataclass(frozen=True)
class MaintainedExport:
    result: object
    receipt: ExportReceipt
    cleaned: int = 0


def _check_cancelled(cancelled):
    if cancelled and cancelled():
        raise InterruptedError("Export/maintenance cancelled; source files were kept.")


def file_version(path):
    path = Path(path).resolve()
    before = _digest_signature(path)
    digest = sha256_file(path)
    after = _digest_signature(path)
    if before is None or before != after or digest is None or not path.is_file():
        raise ValueError("A source or export changed while it was being verified: " + path.name)
    return FileVersion(str(path), digest, after)


def version_matches(version):
    try:
        current = file_version(version.path)
        return current == version
    except (OSError, ValueError):
        return False


def capture_export_sources(folder, sources, workflow, *, cleanup_verified=False, expected_hashes=None, cancelled=None):
    """Capture exactly the selected source files before the successful export."""
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError("The experiment folder no longer exists.")
    hashes = {}
    for name, value in (expected_hashes or {}).items():
        candidate = Path(name)
        hashes[str((candidate if candidate.is_absolute() else root / candidate).resolve()).casefold()] = value
    versions, records, seen = [], [], set()
    for name in sources:
        _check_cancelled(cancelled)
        candidate = Path(name)
        path = (candidate if candidate.is_absolute() else root / candidate).resolve()
        identity = str(path).casefold()
        if identity in seen:
            continue
        seen.add(identity)
        version = file_version(path)
        expected = hashes.get(identity)
        if expected is not None and str(expected).lower() != version.sha256.lower():
            raise ValueError("A source changed since loading. Reload before exporting: " + path.name)
        versions.append(version)
        records.append(verify_initial_data_working_file(path, root, workflow=workflow, role="export_source"))
    if not versions:
        raise ValueError("No exported source files were captured.")
    return ExportSources(str(root), str(workflow), tuple(versions), tuple(records), bool(cleanup_verified))


def export_files(result):
    """Extract explicit returned artifact paths; never scan the experiment."""
    ignored = {"folder", "out_folder", "source_files", "sources", "mode", "status", "save_status", "message"}
    paths = []
    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key not in ignored:
                    collect(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                collect(item)
        elif isinstance(value, Path):
            if not value.is_file():
                raise ValueError("An exported artifact is missing: " + value.name)
            paths.append(value.resolve())
        elif isinstance(value, str):
            path = Path(value)
            try:
                if path.is_file():
                    paths.append(path.resolve())
                elif path.suffix.lower() in {".csv", ".dat", ".png", ".json", ".xlsx", ".npz", ".pdf", ".svg"}:
                    raise ValueError("An exported artifact is missing: " + path.name)
            except OSError as error:
                raise ValueError("An exported artifact could not be verified.") from error
    collect(result)
    return tuple(dict.fromkeys(paths))


def finish_export(request, result, *, cancelled=None):
    """Produce a receipt only after all selected inputs and outputs verify."""
    _check_cancelled(cancelled)
    if not all(version_matches(source) for source in request.sources):
        raise ValueError("An exported source changed during saving. Source files were kept; reload before maintenance.")
    paths = export_files(result)
    if not paths:
        raise ValueError("No successful exported artifacts were returned. Source files were kept.")
    outputs = tuple(file_version(path) for path in paths)
    _check_cancelled(cancelled)
    return ExportReceipt(request, outputs, identity=uuid4().hex)


def cleanup_verified_sources(receipt, *, cancelled=None):
    """Clean only matching temporary copies after the export's success receipt.

    The final preflight is cancellable. Once the small cleanup commit begins,
    it completes without turning a successful export into a cancellation.
    Canonical originals are always preserved by core.provenance.
    """
    if not receipt.request.cleanup_verified or receipt.request.workflow.casefold() not in {"pl", "drr", "compare", "cmp"}:
        return receipt
    _check_cancelled(cancelled)
    if not all(version_matches(item) for item in (*receipt.remaining_sources, *receipt.outputs)):
        raise ValueError("Source/export versions changed; verified-copy cleanup was skipped.")
    versions = {source.path: source for source in receipt.remaining_sources}
    eligible = []
    for record in receipt.request.provenance:
        path = str(Path(record.working_copy_path).resolve())
        version = versions.get(path)
        if (version and record.provenance_verified and record.temporary_working_copy
                and record.canonical_sha256 == record.working_sha256_at_creation == version.sha256):
            eligible.append(record)
    _check_cancelled(cancelled)
    cleaned = list(receipt.cleaned)
    for record in eligible:
        if cleanup_working_copy(record, receipt.request.folder):
            cleaned.append(str(Path(record.working_copy_path).resolve()))
    return replace(receipt, cleaned=tuple(dict.fromkeys(cleaned)))


def run_export_with_maintenance(export_callable, *, folder, sources, workflow, cleanup_verified=False,
                                expected_hashes=None, cancelled=None, output_files=None):
    request = capture_export_sources(folder, sources, workflow, cleanup_verified=cleanup_verified,
        expected_hashes=expected_hashes, cancelled=cancelled)
    _check_cancelled(cancelled)
    result = export_callable()
    receipt = finish_export(request, output_files(result) if callable(output_files) else result, cancelled=cancelled)
    receipt = cleanup_verified_sources(receipt, cancelled=cancelled)
    return MaintainedExport(result, receipt, len(receipt.cleaned))


def movable_sources(receipt):
    root = Path(receipt.request.folder).resolve()
    return tuple(source for source in receipt.remaining_sources
        if Path(source.path).parent == root and Path(source.path).suffix.lower() == ".csv")


def validate_receipt(receipt):
    return isinstance(receipt, ExportReceipt) and bool(receipt.outputs) and all(
        version_matches(item) for item in (*receipt.remaining_sources, *receipt.outputs))


def move_exported_sources(receipt, *, cancelled=None):
    """Explicit archive of verified root CSV versions; no overwrites/recursion."""
    from core.file_ops import archive_selected
    _check_cancelled(cancelled)
    if not validate_receipt(receipt):
        raise ValueError("Source or export files changed after saving. Save the current data before moving sources.")
    candidates = movable_sources(receipt)
    if not candidates:
        return 0
    root = Path(receipt.request.folder).resolve()
    target = (root / ARCHIVE_NAME).resolve()
    if target.parent != root:
        raise ValueError("The archive folder resolves outside this experiment; sources were kept.")
    _check_cancelled(cancelled)
    moved = 0
    for source in candidates:
        # A fresh content check immediately precedes each collision-safe move.
        if not version_matches(source):
            raise ValueError("A source changed before archival; remaining files were kept.")
        moved += archive_selected(str(root), [Path(source.path).name], ARCHIVE_NAME)
    return moved


__all__ = ["FileVersion", "ExportSources", "ExportReceipt", "MaintainedExport", "file_version", "version_matches",
    "capture_export_sources", "finish_export", "run_export_with_maintenance", "cleanup_verified_sources",
    "export_files", "movable_sources", "validate_receipt", "move_exported_sources"]
