from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import shutil


@dataclass(frozen=True)
class WorkingCopyRecord:
    canonical_source_path: str
    working_copy_path: str
    canonical_sha256: str | None
    working_sha256_at_creation: str | None
    workflow: str
    role: str
    app_managed: bool = True
    provenance_verified: bool = True
    temporary_working_copy: bool = True
    verification_method: str = "app_managed_copy"

    @property
    def working_copy_sha256(self) -> str | None:
        return self.working_sha256_at_creation

    def to_dict(self) -> dict:
        value = asdict(self)
        value["working_copy_sha256"] = self.working_sha256_at_creation
        return value


def sha256_file(path: str | Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, PermissionError):
        return None


def create_working_copy(
    canonical_path: str | Path,
    working_folder: str | Path,
    *,
    workflow: str,
    role: str,
) -> WorkingCopyRecord:
    canonical = Path(canonical_path).resolve()
    working_root = Path(working_folder).resolve()
    if not canonical.is_file():
        raise FileNotFoundError(f"Canonical source does not exist: {canonical}")
    working_root.mkdir(parents=True, exist_ok=True)
    working = working_root / canonical.name
    if working == canonical:
        raise ValueError("Working copy must be different from the canonical source.")
    if working.exists():
        stem = canonical.stem
        suffix = canonical.suffix
        index = 1
        while working.exists():
            working = working_root / f"{stem}__working_{index}{suffix}"
            index += 1
    shutil.copy2(canonical, working)
    canonical_hash = sha256_file(canonical)
    working_hash = sha256_file(working)
    verified = canonical_hash is not None and working_hash == canonical_hash
    return WorkingCopyRecord(
        canonical_source_path=str(canonical),
        working_copy_path=str(working),
        canonical_sha256=canonical_hash,
        working_sha256_at_creation=working_hash,
        workflow=str(workflow),
        role=str(role),
        app_managed=True,
        provenance_verified=verified,
        temporary_working_copy=verified,
        verification_method="app_managed_copy",
    )


def verify_initial_data_working_file(
    working_path: str | Path,
    experiment_folder: str | Path,
    *,
    workflow: str,
    role: str,
) -> WorkingCopyRecord:
    """Describe a direct source or verify a root copy against Initial Data."""
    raw_working = Path(working_path)
    root = Path(experiment_folder).resolve()
    working = raw_working.resolve() if raw_working.is_absolute() else (root / raw_working).resolve()
    initial_root = (root / "Initial Data").resolve()
    try:
        working.relative_to(initial_root)
        is_canonical_file = True
    except ValueError:
        is_canonical_file = False

    if is_canonical_file or working.parent != root:
        digest = sha256_file(working)
        return WorkingCopyRecord(
            canonical_source_path=str(working),
            working_copy_path=str(working),
            canonical_sha256=digest,
            working_sha256_at_creation=digest,
            workflow=str(workflow),
            role=str(role),
            app_managed=False,
            # A canonical source is not a verified temporary copy. Its hash is
            # still recorded for reproducibility, but it must never be cleaned.
            provenance_verified=False,
            temporary_working_copy=False,
            verification_method="direct_initial_data" if is_canonical_file else "direct_external_source",
        )

    canonical = initial_root / working.name
    working_hash = sha256_file(working)
    canonical_hash = sha256_file(canonical)
    is_root_file = working.parent == root
    verified = (
        is_root_file
        and working.is_file()
        and canonical.is_file()
        and canonical.name == working.name
        and canonical != working
        and canonical_hash is not None
        and working_hash == canonical_hash
    )
    return WorkingCopyRecord(
        canonical_source_path=str(canonical),
        working_copy_path=str(working),
        canonical_sha256=canonical_hash,
        working_sha256_at_creation=working_hash,
        workflow=str(workflow),
        role=str(role),
        app_managed=False,
        provenance_verified=verified,
        temporary_working_copy=verified,
        verification_method="initial_data_filename_sha256",
    )


def can_cleanup(record: WorkingCopyRecord, working_root: str | Path) -> bool:
    canonical = Path(record.canonical_source_path).resolve()
    working = Path(record.working_copy_path).resolve()
    root = Path(working_root).resolve()
    try:
        working.relative_to(root)
    except ValueError:
        return False
    if not record.provenance_verified or not record.temporary_working_copy or working == canonical:
        return False
    if not canonical.is_file() or not working.is_file():
        return False
    if record.canonical_sha256 is None or record.working_sha256_at_creation is None:
        return False
    return (
        sha256_file(canonical) == record.canonical_sha256
        and sha256_file(working) == record.working_sha256_at_creation
    )


def cleanup_working_copy(record: WorkingCopyRecord, working_root: str | Path) -> bool:
    if not can_cleanup(record, working_root):
        return False
    try:
        Path(record.working_copy_path).unlink()
    except OSError:
        return False
    return True
