"""MCD source paths and saved-processing metadata without scientific imports."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import List

from core.file_ops import _nat_key, list_root_csvs
from core.source_identity import match_source_identity


def list_mcd_csv_files(folder: str) -> List[str]:
    """Return MCD acquisition CSVs, preferring the dedicated ``mcd`` folder.

    Acquisition writes MCD sweeps below ``<experiment>/mcd``.  Relative paths
    are returned so the existing loaders and export metadata remain portable.
    Root-level CSVs are retained as a fallback for older experiment folders.
    """
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []
    try:
        mcd_roots = [
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.casefold() == "mcd"
        ]
        candidates = [
            path
            for mcd_root in mcd_roots
            for path in mcd_root.rglob("*.csv")
            if path.is_file()
        ]
    except OSError:
        candidates = []
    if not candidates:
        return list_root_csvs(folder)
    names = {path.relative_to(root).as_posix() for path in candidates}
    return sorted(names, key=_nat_key)


def discover_mcd_processing_status(
    experiment_root: str | Path,
    sources: list[str] | tuple[str, ...],
    *, ambiguous_sources: set[str] | None = None,
) -> dict[str, str]:
    """Map raw MCD sources to the newest matching saved-analysis timestamp."""
    root = Path(experiment_root)
    metadata_root = root / "Processed Data" / "MCD"
    if not metadata_root.is_dir():
        return {}
    raw_sources = [str(source) for source in sources]
    status: dict[str, str] = {}
    ambiguous: set[str] = set()
    try:
        settings_files = metadata_root.rglob("*_MCD_settings*.json")
        for settings_path in settings_files:
            try:
                payload = json.loads(settings_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("workflow", "")).casefold() != "mcd":
                continue
            descriptor = payload.get("sources", [])
            if isinstance(descriptor, list) and descriptor:
                descriptor = next((item for item in descriptor if isinstance(item, dict) and str(item.get("role", "")).casefold() == "measurement"), {})
            elif not isinstance(descriptor, dict):
                descriptor = {}
            matched, uncertain = match_source_identity(
                root, raw_sources,
                relative_path=str(payload.get("source_relative_path", descriptor.get("source_relative_path", "") if descriptor else "")),
                path=str(payload.get("source_path", descriptor.get("source_path", "") if descriptor else "")),
                legacy_name=str(payload.get("source_file", descriptor.get("source_file", descriptor.get("filename", "")) if descriptor else payload.get("filename", ""))),
            )
            ambiguous.update(uncertain)
            if matched is None:
                continue
            created = str(payload.get("created_utc", "")).strip()
            if not created:
                try:
                    created = datetime.fromtimestamp(
                        settings_path.stat().st_mtime, timezone.utc
                    ).isoformat()
                except OSError:
                    created = "processed"
            if created > status.get(matched, ""):
                status[matched] = created
    except OSError:
        return status
    ambiguous.difference_update(status)
    if ambiguous_sources is not None:
        ambiguous_sources.update(ambiguous)
    return status
