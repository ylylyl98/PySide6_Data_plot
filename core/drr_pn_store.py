"""Versioned, content-addressed storage for explicitly generated P/N backgrounds.

The index is atomically replaced and published only after the native composite
CSV/sidecar pair has passed its existing reader.  A version is never overwritten.
Neither saving nor finding a result applies it to a measurement.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np

from core.drr_composite_background import (
    CompositeBackground, composite_metadata_path, compose_background,
    is_composite_background, read_composite_background, save_composite_background,
)
from core.drr_pn_pairing import DrrPnPair
from core.drr_sources import inspect_csv_spectral_grid, portable_source_name, resolve_source_path
from core.load_cancellation import check_load_cancelled
from core.provenance import _digest_signature


STORE_SCHEMA_VERSION = 1
_PREPARED_USE = object()


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    """Do not use mtime-only caches: source bytes define freshness."""
    path.stat()  # Preserve FileNotFoundError for the missing-source status.
    before = _digest_signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            check_load_cancelled()
            digest.update(block)
    after = _digest_signature(path)
    if before is None or after is None or before != after:
        raise ValueError(f"Input changed while being hashed: {path.name}")
    return digest.hexdigest()


def _fingerprint(provenance) -> str:
    def contents(items):
        return [{key: value for key, value in item.items() if key != "path"} for item in items]
    return _digest({
        **{key: provenance[key] for key in (
            "schema_version", "frame_choice", "settings", "algorithm_version", "spectral_columns")},
        "pair": {role: contents(provenance["pair"][role]) for role in ("P", "N")},
        "dependencies": contents(provenance["dependencies"]),
        "imported_identity": provenance.get("imported_identity"),
    })


@dataclass(frozen=True)
class PnGenerationKey:
    pair_id: str
    fingerprint: str
    provenance_json: str
    root: str

    @property
    def provenance(self) -> dict:
        return json.loads(self.provenance_json)


@dataclass(frozen=True)
class DrrPnVersion:
    version_id: str
    pair_id: str
    number: int
    fingerprint: str
    csv_path: Path
    metadata_path: Path
    created_at: str
    algorithm_version: str
    status: str
    status_reasons: tuple[str, ...] = ()
    reused: bool = False
    p_which: str = "all"
    n_which: str = "all"
    settings_json: str = "{}"
    pair: DrrPnPair | None = None
    is_draft: bool = False

    @property
    def fresh(self) -> bool:
        return self.status == "fresh"

    @property
    def settings(self) -> dict:
        return json.loads(self.settings_json)


@dataclass(frozen=True)
class PreparedPnUse:
    """Read-only verified use draft; only the accepting UI commits its link."""
    root: str
    version: DrrPnVersion
    measurement_records: tuple[tuple[str, str], ...]
    version_record_json: str
    signatures: tuple[tuple[str, tuple[int, ...] | None], ...]
    allow_stale: bool = False
    _verified: object = field(default=None, repr=False, compare=False)


class DrrPnStore:
    def __init__(self, folder: str | Path):
        self.root = Path(folder).resolve()
        self.directory = self.root / "Processed Data" / "DRR" / "PN Backgrounds"
        self.manifest_path = self.directory / "manifest.json"

    def pair_id(self, pair: DrrPnPair) -> str:
        return _digest({role: [os.path.normcase(portable_source_name(self.root, resolve_source_path(self.root, path)))
                               for path in files]
                        for role, files in (("P", pair.p_files), ("N", pair.n_files))})

    def prepare(
        self, pair: DrrPnPair, *, p_which: str = "all", n_which: str = "all",
        settings: Mapping | None = None, algorithm_version: str,
        dependencies: Sequence[str | Path] = (),
    ) -> PnGenerationKey:
        """Snapshot every input before generation, including non-P/N dependencies.

        Callers must include sample references, calibration files or other
        external inputs used by their algorithm.  Measurement selection itself
        is intentionally absent, so the same result can serve other groups.
        """
        if p_which not in ("first", "last", "all") or n_which not in ("first", "last", "all"):
            raise ValueError("Frame choice must be first, last or all.")
        if not isinstance(algorithm_version, str) or not algorithm_version.strip():
            raise ValueError("An explicit generation algorithm version is required.")
        canonical_settings = json.loads(_json(dict(settings or {})))

        p_paths = tuple(resolve_source_path(self.root, name) for name in pair.p_files)
        n_paths = tuple(resolve_source_path(self.root, name) for name in pair.n_files)
        if len(set((*p_paths, *n_paths))) != len(p_paths) + len(n_paths):
            raise ValueError("P and N require distinct source files, including resolved path aliases.")
        grid = inspect_csv_spectral_grid(p_paths[0])
        for path in (*p_paths, *n_paths):
            current = inspect_csv_spectral_grid(path)
            if len(current) != len(grid) or not np.allclose(current, grid, rtol=1e-9, atol=1e-10):
                raise ValueError("P/N files must share the same ordered native spectral columns.")

        def records(paths, *, composite_sources=False):
            result = []
            for source in paths:
                check_load_cancelled()
                path = resolve_source_path(self.root, source)
                item = {"path": portable_source_name(self.root, path), "sha256": _file_digest(path)}
                if composite_sources and is_composite_background(path):
                    item["metadata_sha256"] = _file_digest(composite_metadata_path(path))
                result.append(item)
            return result

        p, n = records(pair.p_files, composite_sources=True), records(pair.n_files, composite_sources=True)
        deps = records(dependencies)
        provenance = {
            "schema_version": STORE_SCHEMA_VERSION,
            "pair": {"P": p, "N": n},
            "frame_choice": {"P": p_which, "N": n_which},
            "settings": canonical_settings, "algorithm_version": algorithm_version,
            "dependencies": deps, "spectral_columns": list(grid),
        }
        # Paths are provenance/identity; numerical reuse depends on ordered
        # role contents.  A changed repeat order remains a different request.
        fingerprint = _fingerprint(provenance)
        return PnGenerationKey(self.pair_id(pair), fingerprint, _json(provenance), str(self.root))

    def _read(self) -> dict:
        if not self.manifest_path.exists():
            return {"schema_version": STORE_SCHEMA_VERSION, "versions": [], "last_used": {}, "applied": {}}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if (payload["schema_version"] != STORE_SCHEMA_VERSION
                    or not isinstance(payload["versions"], list)
                    or not isinstance(payload["last_used"], dict)):
                raise ValueError("Unsupported manifest schema.")
            ids = [item["version_id"] for item in payload["versions"]]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate version identities.")
            payload.setdefault("applied", {})
            if not isinstance(payload["applied"], dict):
                raise ValueError("Invalid applied measurement associations.")
            return payload
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"The P/N version manifest is unreadable; existing versions were preserved: {exc}") from exc

    @contextmanager
    def _locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / ".store.lock"
        try:
            stream = path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise RuntimeError("The P/N background store is busy; retry after the other save finishes.") from exc
        try:
            with stream:
                stream.write(f"{os.getpid()}\n")
            yield
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_json(path: Path, payload):
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            check_load_cancelled()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _artifact_path(self, name: str) -> Path:
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("Version artifact path escapes the P/N store.")
        return path

    def _input_status(self, provenance) -> tuple[str, tuple[str, ...]]:
        missing, changed = [], []
        for item in (*provenance["pair"]["P"], *provenance["pair"]["N"], *provenance["dependencies"]):
            path = resolve_source_path(self.root, item["path"])
            try:
                digest = _file_digest(path)
            except FileNotFoundError:
                missing.append(item["path"])
                continue
            except (OSError, ValueError) as exc:
                changed.append(f"{item['path']}: {exc}")
                continue
            if digest != item["sha256"]:
                changed.append(item["path"])
            if "metadata_sha256" in item:
                try:
                    if _file_digest(composite_metadata_path(path)) != item["metadata_sha256"]:
                        changed.append(item["path"] + " metadata")
                except FileNotFoundError:
                    missing.append(item["path"] + " metadata")
                except (OSError, ValueError) as exc:
                    changed.append(f"{item['path']} metadata: {exc}")
        if missing:
            return "missing_sources", tuple(f"missing: {name}" for name in missing) + tuple(
                f"changed: {name}" for name in changed)
        if changed:
            return "stale", tuple(f"changed: {name}" for name in changed)
        return "fresh", ()

    def _version(self, record) -> DrrPnVersion:
        csv_path = self._artifact_path(record["csv_path"])
        metadata_path = self._artifact_path(record["metadata_path"])
        status, reasons = self._input_status(record["provenance"])
        algorithm = record["provenance"]["algorithm_version"]
        # Raw reconstruction remains an opt-in draft regardless of an absent
        # or optimistic saved validation flag. Explicit use is still reusable.
        is_draft = str(algorithm).startswith("pn_raw_") or algorithm in {
            "pn_regional_curvature_v1", "pn_sample_assisted_curvature_v1",
        }
        try:
            if (_file_digest(csv_path) != record["csv_sha256"]
                    or _file_digest(metadata_path) != record["metadata_sha256"]):
                raise ValueError("Saved CSV or metadata hash changed.")
            document = read_composite_background(csv_path)
            source = next(item for item in document.sources if item.key == document.default_source)
            report = json.loads(source.generation_json or "{}")
            is_draft = is_draft or report.get("scientifically_validated") is False
        except (OSError, ValueError) as exc:
            status, reasons = "invalid", (*reasons, str(exc))
        return DrrPnVersion(
            record["version_id"], record["pair_id"], record["number"], record["fingerprint"],
            csv_path, metadata_path, record["created_at"],
            record["provenance"]["algorithm_version"], status, tuple(reasons),
            p_which=record["provenance"]["frame_choice"]["P"],
            n_which=record["provenance"]["frame_choice"]["N"],
            settings_json=_json(record["provenance"]["settings"]),
            pair=DrrPnPair(tuple(item["path"] for item in record["provenance"]["pair"]["P"]),
                           tuple(item["path"] for item in record["provenance"]["pair"]["N"])),
            is_draft=is_draft,
        )

    def list_versions(self, pair: DrrPnPair | None = None) -> tuple[DrrPnVersion, ...]:
        pair_id = self.pair_id(pair) if pair is not None else None
        records = self._read()["versions"]
        return tuple(self._version(record) for record in records
                     if pair_id is None or record["pair_id"] == pair_id)

    def get_version(self, version_id: str) -> DrrPnVersion | None:
        """Fully validate one requested version without loading unrelated files."""
        check_load_cancelled()
        record = next((item for item in self._read()["versions"]
                       if item["version_id"] == version_id), None)
        return self._version(record) if record is not None else None

    def find_reusable(self, key: PnGenerationKey) -> DrrPnVersion | None:
        self._validate_key(key)
        for record in reversed(self._read()["versions"]):
            if record["pair_id"] == key.pair_id and record["fingerprint"] == key.fingerprint:
                version = self._version(record)
                if version.fresh:
                    return replace(version, reused=True)
        return None

    def _validate_key(self, key: PnGenerationKey):
        if key.root != str(self.root):
            raise ValueError("Generation key belongs to another experiment folder.")
        if _fingerprint(key.provenance) != key.fingerprint:
            raise ValueError("Generation key fingerprint does not match its provenance.")
        status, reasons = self._input_status(key.provenance)
        if status != "fresh":
            raise ValueError("Generation inputs changed; prepare and generate again. " + "; ".join(reasons))

    def _validate_document_binding(self, document, key):
        provenance = key.provenance
        if "imported_identity" in provenance:
            return  # An explicit historical association, never a fresh generation claim.
        source_keys = []
        for role in ("P", "N"):
            records = provenance["pair"][role]
            expected_files = tuple(resolve_source_path(self.root, item["path"]) for item in records)
            expected_hashes = tuple((path, item["sha256"]) for path, item in zip(expected_files, records))
            matched = next((source for source in document.sources
                            if tuple(Path(path).resolve() for path in source.files) == expected_files
                            and source.which == provenance["frame_choice"][role]
                            and tuple((Path(path).resolve(), digest) for path, digest in source.source_hashes)
                            == expected_hashes), None)
            if matched is None:
                raise ValueError(f"The generated document has no matching {role} input snapshot/hash/frame choice.")
            source_keys.append(matched.key)
        generated = next(source for source in document.sources if source.key == document.default_source)
        report = json.loads(generated.generation_json or "{}")
        if (report.get("algorithm_version") != provenance["algorithm_version"]
                or report.get("settings") != provenance["settings"]
                or report.get("input_source_keys") != source_keys):
            raise ValueError("Generated algorithm/settings/ordered P/N source keys do not match the prepared request.")

    def save(
        self, document: CompositeBackground, key: PnGenerationKey, *, force_new: bool = False,
    ) -> DrrPnVersion:
        """Publish a version, reusing identical input unless explicitly regenerated."""
        self._validate_key(key)
        self._validate_document_binding(document, key)
        _, intensity, _ = compose_background(document)
        if not np.isfinite(intensity).all() or np.any(intensity <= 0):
            raise ValueError("A generated P/N background must be finite and positive on the complete grid.")
        columns = np.asarray(document.sources[0].spectral_columns, dtype=float)
        expected = np.asarray(key.provenance["spectral_columns"], dtype=float)
        if columns.shape != expected.shape or not np.allclose(columns, expected, rtol=1e-9, atol=1e-10):
            raise ValueError("The generated background must preserve the P/N native spectral columns.")
        with self._locked():
            existing = None if force_new else self.find_reusable(key)
            if existing is not None:
                return existing
            payload = self._read()
            number = 1 + max((item["number"] for item in payload["versions"]
                              if item["pair_id"] == key.pair_id), default=0)
            version_id = uuid4().hex
            csv_path = self.directory / f"PN_{key.pair_id[:12]}_v{number:04d}_{version_id[:8]}.csv"
            metadata_path = composite_metadata_path(csv_path)
            created_at = datetime.now(timezone.utc).isoformat()
            save_composite_background(csv_path, document)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["pn_generation"] = {
                "version_id": version_id, "pair_id": key.pair_id,
                "fingerprint": key.fingerprint, "number": number,
                "created_at": created_at, **key.provenance,
            }
            self._atomic_json(metadata_path, metadata)
            read_composite_background(csv_path)
            # No index entry is published if a source changed during generation
            # or saving.  An unindexed result is not considered reusable.
            self._validate_key(key)
            record = {
                "version_id": version_id, "pair_id": key.pair_id, "number": number,
                "fingerprint": key.fingerprint, "created_at": created_at,
                "csv_path": csv_path.name, "metadata_path": metadata_path.name,
                "csv_sha256": _file_digest(csv_path), "metadata_sha256": _file_digest(metadata_path),
                "provenance": key.provenance,
            }
            payload["versions"].append(record)
            check_load_cancelled()
            self._atomic_json(self.manifest_path, payload)
            return self._version(record)

    def import_saved(
        self, csv_path: str | Path, pair: DrrPnPair, *, algorithm_version: str,
        settings: Mapping | None = None, dependencies: Sequence[str | Path] = (),
        provenance: Mapping | None = None, p_which: str = "all", n_which: str = "all",
    ) -> DrrPnVersion:
        """Import a verified native composite, preserving its frozen provenance.

        The source CSV/sidecar are not runtime dependencies of the copy.  Their
        content hashes identify the imported result, while the selected P/N
        files and explicit generation dependencies still determine freshness.
        Association with this pair is an explicit caller/user action.
        """
        path = resolve_source_path(self.root, csv_path)
        csv_digest, metadata_digest = _file_digest(path), _file_digest(composite_metadata_path(path))
        document = read_composite_background(path)
        metadata = json.loads(composite_metadata_path(path).read_text(encoding="utf-8"))
        key = self.prepare(pair, p_which=p_which, n_which=n_which, settings=settings,
                           algorithm_version=algorithm_version, dependencies=dependencies)
        payload = key.provenance
        payload["imported_identity"] = {
            "csv_sha256": csv_digest, "metadata_sha256": metadata_digest,
            "user_provenance": json.loads(_json(dict(provenance or {}))),
        }
        payload["imported_from"] = {
            "path": portable_source_name(self.root, path),
            "metadata": metadata,
            "association": "Explicitly associated with the selected P/N pair; no regeneration performed.",
        }
        if (csv_digest != _file_digest(path)
                or metadata_digest != _file_digest(composite_metadata_path(path))):
            raise ValueError("Imported background changed while being read; import it again.")
        return self.save(document, replace(key, fingerprint=_fingerprint(payload), provenance_json=_json(payload)))

    def _measurement_records(self, measurements):
        paths = sorted({resolve_source_path(self.root, source) for source in measurements}, key=str)
        return [{"path": portable_source_name(self.root, path), "sha256": _file_digest(path)} for path in paths]

    @staticmethod
    def _measurement_identity(records):
        return _digest(sorted(os.path.normcase(item["path"]) for item in records))

    def mark_used(
        self, version_id: str, *, measurements: Sequence[str | Path] = (),
        allow_stale: bool = False,
    ) -> DrrPnVersion:
        """Explicit user use; stale/missing-source history needs explicit consent."""
        return self.commit_prepared_use(self.prepare_use(
            version_id, measurements=measurements, allow_stale=allow_stale))

    @staticmethod
    def _use_signatures(paths):
        result = []
        for path in paths:
            check_load_cancelled()
            result.append((str(path), _digest_signature(path)))
        return tuple(result)

    def prepare_use(
        self, version_id: str, *, measurements: Sequence[str | Path] = (),
        allow_stale: bool = False,
    ) -> PreparedPnUse:
        """Verify bytes in a worker without recording a background choice.

        Snapshot before validation so changes between individual hashes cannot
        produce a mixed draft. The accepting UI subsequently checks these cheap
        signatures and the original version record before publishing its link.
        """
        check_load_cancelled()
        record = next((item for item in self._read()["versions"] if item["version_id"] == version_id), None)
        if record is None:
            raise KeyError(f"Unknown P/N background version: {version_id}")
        record_json = _json(record)
        paths = {self._artifact_path(record["csv_path"]), self._artifact_path(record["metadata_path"])}
        provenance = record["provenance"]
        for item in (*provenance["pair"]["P"], *provenance["pair"]["N"], *provenance["dependencies"]):
            check_load_cancelled()
            path = resolve_source_path(self.root, item["path"])
            paths.add(path)
            if "metadata_sha256" in item:
                paths.add(composite_metadata_path(path))
        measurement_paths = tuple(sorted({resolve_source_path(self.root, source) for source in measurements}, key=str))
        paths.update(measurement_paths)
        paths = tuple(sorted(paths, key=str))
        signatures = self._use_signatures(paths)
        version = self._version(record)
        if version.status == "invalid" or (not version.fresh and not allow_stale):
            raise ValueError(f"Version is {version.status}; use a verified version"
                             " or explicitly choose an independent historical background.")
        records = self._measurement_records(measurement_paths)
        expected = np.asarray(provenance["spectral_columns"], dtype=float)
        for item in records:
            check_load_cancelled()
            current = np.asarray(inspect_csv_spectral_grid(
                resolve_source_path(self.root, item["path"])), dtype=float)
            if current.shape != expected.shape or not np.allclose(current, expected, rtol=1e-9, atol=1e-10):
                raise ValueError("The selected measurement and saved P/N background have different native grids.")
        current_record = next((item for item in self._read()["versions"] if item["version_id"] == version_id), None)
        if current_record is None or _json(current_record) != record_json or self._use_signatures(paths) != signatures:
            raise ValueError("P/N background or inputs changed during verification; review the selection again.")
        check_load_cancelled()
        return PreparedPnUse(str(self.root), version,
            tuple((item["path"], item["sha256"]) for item in records), record_json,
            signatures, bool(allow_stale), _PREPARED_USE)

    def commit_prepared_use(self, prepared: PreparedPnUse) -> DrrPnVersion:
        """Publish a verified draft with only manifest reads and fast file stats.

        Call after the UI context/cancellation guard. No CSV bytes or spectral
        headers are read here; loading still independently verifies the recipe.
        """
        check_load_cancelled()
        if (not isinstance(prepared, PreparedPnUse) or prepared._verified is not _PREPARED_USE
                or prepared.root != str(self.root)):
            raise ValueError("Prepared P/N use belongs to another experiment or was not verified.")
        with self._locked():
            payload = self._read()
            version = prepared.version
            record = next((item for item in payload["versions"] if item["version_id"] == version.version_id), None)
            if record is None or _json(record) != prepared.version_record_json:
                raise ValueError("P/N version history changed after verification; review the selection again.")
            if self._use_signatures(tuple(Path(path) for path, _ in prepared.signatures)) != prepared.signatures:
                raise ValueError("P/N background or inputs changed after verification; review the selection again.")
            payload["last_used"][version.pair_id] = version.version_id
            if prepared.measurement_records:
                records = [{"path": path, "sha256": digest} for path, digest in prepared.measurement_records]
                payload["applied"][self._measurement_identity(records)] = {
                    "version_id": version.version_id, "measurements": records,
                }
            check_load_cancelled()
            self._atomic_json(self.manifest_path, payload)
            return version

    def last_used(self, pair: DrrPnPair) -> DrrPnVersion | None:
        payload = self._read()
        version_id = payload["last_used"].get(self.pair_id(pair))
        record = next((item for item in payload["versions"] if item["version_id"] == version_id), None)
        return self._version(record) if record is not None else None

    def preferred_version(
        self, pair: DrrPnPair, *, p_which: str = "all", n_which: str = "all",
    ) -> DrrPnVersion | None:
        """Restore an explicit choice; otherwise skip untried experimental drafts.

        This is a reuse preference, not scientific approval. Freshness and
        method identity remain visible. Explicitly used drafts can be restored,
        but merely generating a newer draft must not displace an older result.
        """
        versions = self.list_versions(pair)
        last = self._read()["last_used"].get(self.pair_id(pair))
        eligible = [item for item in versions if item.fresh
                    and (item.p_which, item.n_which) == (p_which, n_which)]
        explicitly_used = next((item for item in eligible if item.version_id == last), None)
        if explicitly_used is not None:
            return explicitly_used
        saved = [item for item in eligible if not item.is_draft]
        return saved[-1] if saved else None

    def find_applied(self, measurements: Sequence[str | Path]) -> DrrPnVersion | None:
        """Restore an unchanged explicit association, never silently replace it.

        None means no previous association.  A broken explicit association
        raises ValueError, so External matching cannot fall back to a different
        numerical background without the user's choice.
        """
        if not measurements:
            return None
        paths = sorted({resolve_source_path(self.root, source) for source in measurements}, key=str)
        identities = [{"path": portable_source_name(self.root, path)} for path in paths]
        payload = self._read()
        link = payload["applied"].get(self._measurement_identity(identities))
        if link is None:
            return None
        try:
            records = self._measurement_records(measurements)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Previously applied P/N background cannot be restored: {exc}") from exc
        if ({os.path.normcase(item["path"]): item["sha256"] for item in link["measurements"]}
                != {os.path.normcase(item["path"]): item["sha256"] for item in records}):
            raise ValueError("Previously associated measurement files changed; explicitly select the background again.")
        record = next((item for item in payload["versions"] if item["version_id"] == link["version_id"]), None)
        if record is None:
            raise ValueError("The previously applied P/N background version is missing from history.")
        version = self._version(record)
        if not version.fresh:
            raise ValueError(f"The previously applied P/N background is {version.status}; explicitly choose a version again.")
        return version
