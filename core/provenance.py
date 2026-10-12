from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import shutil
from threading import RLock

_DIGEST_CACHE = OrderedDict()
_DIGEST_LOCK = RLock()

if os.name == 'nt':
    import ctypes
    from ctypes import wintypes

    class _FileBasicInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_int64) for name in
                    ('creation', 'access', 'write', 'change')] + [('attributes', wintypes.DWORD)]

    _kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                    ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _kernel.CreateFileW.restype = wintypes.HANDLE
    _kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                    ctypes.c_void_p, wintypes.DWORD]
    _kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel.CloseHandle.restype = wintypes.BOOL


def _digest_signature(path):
    try:
        stat = path.stat()
    except OSError:
        return None
    change = stat.st_ctime_ns
    if os.name == 'nt':
        # Windows st_ctime is creation time. FILE_BASIC_INFO.ChangeTime also
        # catches a same-size rewrite whose LastWriteTime was restored.
        handle = _kernel.CreateFileW(str(path), 0x80, 7, None, 3, 0x02000000, None)
        if handle == wintypes.HANDLE(-1).value:
            return None
        try:
            info = _FileBasicInfo()
            if not _kernel.GetFileInformationByHandleEx(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
                return None
            change = info.change
        finally:
            _kernel.CloseHandle(handle)
    return stat.st_mtime_ns, stat.st_size, stat.st_dev, stat.st_ino, change


def _cached_load_digest(path):
    """Reuse stable digests for loading; cleanup always calls sha256_file."""
    path = Path(path).resolve()
    before = _digest_signature(path)
    with _DIGEST_LOCK:
        cached = _DIGEST_CACHE.get(path)
        if before is not None and cached is not None and cached[0] == before:
            _DIGEST_CACHE.move_to_end(path)
            return cached[1]
    digest = sha256_file(path)
    after = _digest_signature(path)
    if before != after:
        return None
    if digest is not None and before is not None:
        with _DIGEST_LOCK:
            _DIGEST_CACHE[path] = before, digest
            _DIGEST_CACHE.move_to_end(path)
            while len(_DIGEST_CACHE) > 512:
                _DIGEST_CACHE.popitem(last=False)
    return digest


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
        digest = _cached_load_digest(working)
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
    working_hash = _cached_load_digest(working)
    canonical_hash = _cached_load_digest(canonical)
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
