"""Measurement repeat naming shared by catalogs and numerical workflows."""
from pathlib import Path
import re


REP_SUB_SUFFIX_RE = re.compile(r"_rep(?P<rep>\d+)_(?P<sub>\d+)$", re.IGNORECASE)
REP_ONLY_SUFFIX_RE = re.compile(r"(?:\$_|_)rep(?P<rep>\d{1,3})$", re.IGNORECASE)
RUN_SUFFIX_RE = re.compile(r"(?:\$_|_)(?P<run>\d{3,})$")


def split_group_and_sort_key(filename: str) -> tuple[str, tuple[int, int, int]]:
    stem = Path(filename).stem

    m = REP_SUB_SUFFIX_RE.search(stem)
    if m:
        condition_key = re.sub(r"_rep\d+_\d+$", "", stem, flags=re.IGNORECASE)
        return condition_key, (0, int(m.group("rep")), int(m.group("sub")))

    m = REP_ONLY_SUFFIX_RE.search(stem)
    if m:
        prefix = stem[: m.start()]
        return prefix, (1, int(m.group("rep")), 0)

    m = RUN_SUFFIX_RE.search(stem)
    if m:
        prefix = stem[: m.start()]
        return prefix, (1, int(m.group("run")), 0)

    return stem, (2, 0, 0)
