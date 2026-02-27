"""File-system helpers for root-level CSV workflows.

These helpers intentionally avoid raising for routine UI actions so pages can
show user-friendly messages instead of stack traces.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List


def _nat_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_root_csvs(user_folder: str) -> List[str]:
    """Return CSV filenames in the folder root, naturally sorted."""
    p = Path(user_folder)
    if not p.exists() or not p.is_dir():
        return []
    try:
        return sorted([f.name for f in p.glob("*.csv")], key=_nat_key)
    except OSError:
        return []


def archive_all(user_folder: str, archive_name: str) -> int:
    """Move all root CSV files into `<user_folder>/<archive_name>`."""
    p = Path(user_folder)
    if not p.exists() or not p.is_dir():
        return 0
    dst_root = p / archive_name
    try:
        dst_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return 0

    n = 0
    try:
        root_csvs = list(p.glob("*.csv"))
    except OSError:
        return 0

    for f in root_csvs:
        try:
            f.replace(dst_root / f.name)
            n += 1
        except OSError:
            pass
    return n


def restore_all(user_folder: str, archive_name: str) -> int:
    """Move archived CSV files back to the folder root."""
    p = Path(user_folder)
    if not p.exists() or not p.is_dir():
        return 0

    src_root = p / archive_name
    if not src_root.exists() or not src_root.is_dir():
        return 0

    n = 0
    try:
        archived_csvs = list(src_root.glob("*.csv"))
    except OSError:
        return 0

    for f in archived_csvs:
        try:
            f.replace(p / f.name)
            n += 1
        except OSError:
            pass
    return n
