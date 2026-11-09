"""Lightweight Megasweep inventory; numerical validation happens on load."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path


_EXCLUDED = {"processed data", ".git", ".venv", "build", "dist", "archive", "__pycache__"}


def _allowed_directory(name):
    return name.casefold() not in _EXCLUDED and not name.startswith(".") and not name.casefold().endswith("_outputs")


def source_paths(folder, recursive=True):
    root = Path(folder).resolve()
    for directory, directories, files in os.walk(root, followlinks=False):
        if not recursive:
            directories[:] = []
        directories[:] = [name for name in directories if _allowed_directory(name)]
        for name in sorted(files):
            if name.lower().endswith(".csv"):
                yield Path(directory) / name


def inventory(folder, cancellation=None, *, recursive=True):
    root = Path(folder).resolve()
    rows = []
    for path in source_paths(root, recursive=recursive):
        if cancellation:
            cancellation.check()
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append([path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns])
    return sorted(rows)


def scan(folder, cancellation=None, *, recursive=True):
    root = Path(folder).resolve()
    rows = []
    for relative, size, modified_ns in inventory(root, cancellation, recursive=recursive):
        if cancellation:
            cancellation.check()
        path = root / relative
        kind, detail = "CSV", "CSV candidate; compatibility is validated when loaded."
        try:
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                # Header discovery is bounded and never reads the numerical matrix.
                for _ in range(100):
                    line = handle.readline(1024 * 1024)
                    if not line:
                        break
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    columns = next(csv.reader([line]))
                    spectral = 0
                    for column in columns:
                        try:
                            spectral += int(math.isfinite(float(column)) and float(column) > 0)
                        except ValueError:
                            pass
                    kind = "Optical" if spectral >= 2 else "Transport"
                    detail = (f"{spectral} spectral channels" if spectral >= 2 else "Scalar CSV candidate")
                    detail += " · " + ", ".join(columns[:6])
                    break
        except OSError:
            detail = "File unavailable; retry after acquisition releases it."
        rows.append(dict(id=str(path.resolve()), source=relative, name=path.name, kind=kind,
                         status="unknown", saved=False, modified=modified_ns / 1e9, size=size,
                         processedAt="", background=False, detail=detail))
    watch_dirs=[]
    if recursive:
        for directory, directories, _ in os.walk(root, followlinks=False):
            if cancellation: cancellation.check()
            directories[:]=[name for name in directories if _allowed_directory(name)]
            watch_dirs.append(str(Path(directory)))
    else: watch_dirs=[str(root)]
    return dict(files=sorted(rows, key=lambda row: (-row["modified"], row["source"].casefold())),
                groups=[], sources=(), watch_dirs=watch_dirs)
