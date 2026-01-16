from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List


def _nat_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_root_csvs(user_folder: str) -> List[str]:
    p = Path(user_folder)
    if not p.exists():
        return []
    files = sorted([f.name for f in p.glob("*.csv")], key=_nat_key)
    return files


def archive_all(user_folder: str, archive_name: str) -> int:
    p = Path(user_folder)
    dst_root = p / archive_name
    dst_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in p.glob("*.csv"):
        try:
            f.replace(dst_root / f.name)
            n += 1
        except Exception:
            pass
    return n


def restore_all(user_folder: str, archive_name: str) -> int:
    p = Path(user_folder)
    src_root = p / archive_name
    if not src_root.exists():
        return 0
    n = 0
    for f in src_root.glob("*.csv"):
        try:
            f.replace(p / f.name)
            n += 1
        except Exception:
            pass
    return n
