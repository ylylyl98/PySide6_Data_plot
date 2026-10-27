"""Lightweight PL source discovery and saved-processing metadata lookup."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import List, Sequence

from core.file_ops import _nat_key
from core.source_identity import match_source_identity


def list_pl_source_files(folder: str) -> List[str]:
    """Return raw PL inputs plus saved PL DAT results as portable paths."""
    root = Path(folder)
    if not root.is_dir():
        return []
    candidates: list[Path] = []
    try:
        candidates.extend(
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".dat"}
        )
        initial = root / "Initial Data"
        if initial.is_dir():
            candidates.extend(
                path
                for path in initial.rglob("*")
                if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".dat"}
            )
        processed = root / "Processed Data" / "PL"
        if processed.is_dir():
            candidates.extend(path for path in processed.rglob("*.dat") if path.is_file())
    except OSError:
        pass
    names = {path.relative_to(root).as_posix() for path in candidates}
    return sorted(names, key=_nat_key)


_PL_SOURCE_MARKER_RE = re.compile(
    r"(?:^|[_\-\s])(?P<kind>PL|REF)(?=$|[_\-\s])"
    r"|(?P<temperature>\d+(?:\.\d+)?)K(?P<attached>PL|REF)(?=$|[_\-\s])",
    re.IGNORECASE,
)


def classify_pl_source(source: str | Path) -> str:
    """Classify a raw PL filename conservatively as PL, REF, or Unknown."""
    suffix = Path(source).suffix.lower()
    if suffix == ".dat":
        return "DAT"
    if suffix not in {".csv", ".xlsx"}:
        return "Unknown"
    kinds = {
        (match.group("kind") or match.group("attached")).upper()
        for match in _PL_SOURCE_MARKER_RE.finditer(Path(source).stem)
    }
    return next(iter(kinds)) if len(kinds) == 1 else "Unknown"


def discover_pl_processing_status(
    experiment_root: str | Path, sources: Sequence[str], *,
    ambiguous_sources: set[str] | None = None,
) -> dict[str, str]:
    """Map raw PL sources to the newest matching export-metadata timestamp."""
    root = Path(experiment_root)
    metadata_root = root / "Processed Data" / "PL"
    if not metadata_root.is_dir():
        return {}
    raw_sources = [
        str(source)
        for source in sources
        if not str(source).replace("\\", "/").casefold().startswith("processed data/pl/")
    ]
    status: dict[str, str] = {}
    ambiguous: set[str] = set()
    try:
        metadata_files = metadata_root.rglob("*.metadata.json")
        for metadata_path in metadata_files:
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            workflow = str(payload.get("workflow", payload.get("operation", ""))).casefold()
            if workflow != "pl":
                continue
            created = str(payload.get("created_utc", "")).strip()
            if not created:
                try:
                    created = datetime.fromtimestamp(
                        metadata_path.stat().st_mtime, timezone.utc
                    ).isoformat()
                except OSError:
                    created = "processed"
            descriptors = payload.get("sources", payload.get("inputs", []))
            if not isinstance(descriptors, list):
                continue
            for descriptor in descriptors:
                if not isinstance(descriptor, dict):
                    continue
                matched, uncertain = match_source_identity(
                    root, raw_sources,
                    relative_path=str(descriptor.get("source_relative_path", "")),
                    path=str(descriptor.get("source_path", descriptor.get("path", ""))),
                    legacy_name=str(descriptor.get("name", descriptor.get("source_file", descriptor.get("filename", "")))),
                )
                ambiguous.update(uncertain)
                if matched is not None and created > status.get(matched, ""):
                    status[matched] = created
    except OSError:
        return status
    ambiguous.difference_update(status)
    if ambiguous_sources is not None:
        ambiguous_sources.update(ambiguous)
    return status
