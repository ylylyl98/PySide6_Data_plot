"""Small, immutable SHG export history snapshots.

The export settings files are the source of truth for SHG history.  This
module only reads them and resolves their source descriptors against the
current experiment catalog; it has no Qt or widget dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from core.source_identity import match_source_identity


@dataclass(frozen=True)
class ShgHistorySnapshot:
    """Result of one history scan, safe to pass across a worker boundary."""

    processed_at: dict[str, str]
    roles: dict[str, tuple[str, ...]]


def _created_at(metadata_path: Path, payload: dict) -> str:
    value = str(payload.get("created_utc", "")).strip()
    if value:
        return value
    try:
        return datetime.fromtimestamp(metadata_path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return "processed"


def scan_shg_history(
    experiment_root: str | Path,
    sources: Sequence[str],
    *,
    ambiguous_sources: set[str] | None = None,
) -> ShgHistorySnapshot:
    """Scan SHG metadata and match sources by exact identity where possible.

    Current exports carry absolute and experiment-relative paths.  Older
    basename-only metadata is accepted only when that basename is unique;
    collisions are surfaced as history-unknown through ``ambiguous_sources``.
    Roles are retained so callers can distinguish measurement/reference/sample
    sources from backgrounds without inventing current-parameter semantics.
    """
    root = Path(experiment_root)
    metadata_root = root / "Processed Data" / "SHG"
    if not metadata_root.is_dir():
        return ShgHistorySnapshot({}, {})
    raw_sources = tuple(str(source) for source in sources)
    status: dict[str, str] = {}
    role_sets: dict[str, set[str]] = {}
    ambiguous: set[str] = set()
    try:
        metadata_files = sorted(
            set(metadata_root.rglob("*.metadata.json"))
            | set(metadata_root.rglob("*_settings.json")),
            key=lambda path: str(path).casefold(),
        )
        for metadata_path in metadata_files:
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            workflow = str(payload.get("workflow", payload.get("operation", ""))).casefold()
            if workflow != "shg":
                continue
            descriptors = payload.get("inputs", payload.get("sources", []))
            if isinstance(descriptors, dict):
                descriptors = [descriptors]
            if not isinstance(descriptors, list):
                continue
            created = _created_at(metadata_path, payload)
            for descriptor in descriptors:
                if not isinstance(descriptor, dict):
                    continue
                role = str(descriptor.get("role", "source")).strip().casefold() or "source"
                matched, uncertain = match_source_identity(
                    root,
                    raw_sources,
                    relative_path=str(descriptor.get("source_relative_path", "")),
                    path=str(descriptor.get("source_path", descriptor.get("path", ""))),
                    legacy_name=str(descriptor.get("name", descriptor.get("source_file", descriptor.get("filename", "")))),
                )
                ambiguous.update(uncertain)
                if matched is None:
                    continue
                if created > status.get(matched, ""):
                    status[matched] = created
                role_sets.setdefault(matched, set()).add(role)
    except OSError:
        pass
    ambiguous.difference_update(status)
    if ambiguous_sources is not None:
        ambiguous_sources.update(ambiguous)
    return ShgHistorySnapshot(
        dict(status),
        {source: tuple(sorted(values)) for source, values in role_sets.items() if source in status},
    )


def discover_shg_processing_status(
    experiment_root: str | Path,
    sources: Sequence[str],
    *,
    ambiguous_sources: set[str] | None = None,
) -> dict[str, str]:
    """Compatibility helper matching PL/MCD status discovery APIs."""
    return scan_shg_history(
        experiment_root, sources, ambiguous_sources=ambiguous_sources
    ).processed_at


__all__ = [
    "ShgHistorySnapshot",
    "scan_shg_history",
    "discover_shg_processing_status",
]
