"""Separate immutable DRR source inspections from saved-result history overlays."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from core.drr_sources import (
    DrrSource, DrrSourceCache, discover_drr_sources, drr_source_paths,
    refresh_drr_source_history,
)
from core.source_catalog_cache import SourceCatalogCache


def drr_raw_inventory(folder: str | Path, *, include_all: bool = False) -> list[list[object]]:
    root = Path(folder).resolve()
    if not root.is_dir():
        raise OSError(f"Source directory is unavailable: {root}")
    records = []
    for path in drr_source_paths(root, include_all=include_all):
        stat = path.stat()
        records.append([path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns])
    return sorted(records)


def drr_history_inventory(folder: str | Path, *, include_all: bool = False) -> list[list[object]]:
    """Metadata plus raw identity; PNG/DAT writes do not invalidate this layer."""
    root = Path(folder).resolve()
    records = drr_raw_inventory(root, include_all=include_all)
    history = root / "Processed Data" / "DRR"
    if history.is_dir():
        for path in history.rglob("*.metadata.json"):
            if path.is_file():
                stat = path.stat()
                records.append([path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns])
    return sorted(records)


def load_drr_catalog(
    folder: str | Path,
    cache: DrrSourceCache,
    *,
    force: bool = False,
    include_all: bool = False,
    publish_cached: Callable[[list[DrrSource]], None] | None = None,
) -> list[DrrSource]:
    """Preview persisted history, then update only the layer that changed."""
    suffix = "-all" if include_all else ""
    raw = SourceCatalogCache(str(folder), "DRR-raw-v2" + suffix,
                             inventory_provider=lambda: drr_raw_inventory(folder, include_all=include_all))
    history = SourceCatalogCache(str(folder), "DRR-history-v2" + suffix,
                                 inventory_provider=lambda: drr_history_inventory(folder, include_all=include_all))

    def build_raw():
        # Avoid loading the much larger per-file inspection cache on a hit in
        # either catalog layer. A forced refresh still reuses valid inspections.
        cache.load()
        sources = discover_drr_sources(folder, cache=cache, include_history=False, include_all=include_all)
        cache.save()
        return sources

    def build_history():
        sources = raw.refresh(build_raw, force=force)
        return refresh_drr_source_history(folder, sources)

    return history.refresh(build_history, force=force, publish_cached=publish_cached)
