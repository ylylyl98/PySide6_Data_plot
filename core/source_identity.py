"""Portable source identity matching for saved processing metadata.

This module deliberately knows nothing about metadata formats or UI policy.  It
only resolves a descriptor's source fields against the current experiment's
candidate names, keeping exact identities separate from legacy basenames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def _norm(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.casefold()


def _is_absolute(value: str) -> bool:
    # Path.is_absolute() on POSIX does not recognise Windows drive paths,
    # while metadata can be produced on either platform.
    text = str(value or "").replace("\\", "/")
    return Path(text).is_absolute() or (len(text) >= 3 and text[1] == ":" and text[2] == "/")


def _relative_absolute(root: Path, value: str) -> str:
    """Return an experiment-relative path for an absolute path inside root."""
    try:
        candidate = Path(value).resolve(strict=False)
        relative = candidate.relative_to(root.resolve(strict=False))
    except (OSError, ValueError, RuntimeError):
        return ""
    return relative.as_posix()


def match_source_identity(
    experiment_root: str | Path,
    sources: Sequence[str],
    *,
    relative_path: str = "",
    path: str = "",
    legacy_name: str = "",
) -> tuple[str | None, tuple[str, ...]]:
    """Resolve one metadata descriptor against current source names.

    Explicit experiment-relative paths and absolute paths inside the current
    experiment are exact.  A directory-bearing legacy path is also exact;
    basename-only evidence may resolve only when the basename is unique.
    Missing exact identities never fall back to another same-name candidate.
    The second return value contains candidates made ambiguous by basename
    evidence.
    """
    raw_sources = [str(source) for source in sources]
    by_relative = {_norm(source): source for source in raw_sources}
    by_name: dict[str, list[str]] = {}
    for source in raw_sources:
        by_name.setdefault(Path(source.replace("\\", "/")).name.casefold(), []).append(source)
    root = Path(experiment_root)

    strong_values: list[str] = []
    if str(relative_path or "").strip():
        strong_values.append(_norm(relative_path))
    if str(path or "").strip():
        path_text = str(path).strip()
        if _is_absolute(path_text):
            relative = _relative_absolute(root, path_text)
            # An absolute path outside this experiment is an unusable exact
            # identity; it must not degrade to a basename guess.
            strong_values.append(_norm(relative) if relative else "__missing_absolute__")
        elif "/" in path_text.replace("\\", "/"):
            strong_values.append(_norm(path_text))
    if str(legacy_name or "").strip() and "/" in str(legacy_name).replace("\\", "/"):
        strong_values.append(_norm(legacy_name))

    if strong_values:
        usable = [value for value in strong_values if value and value != "__missing_absolute__"]
        if len(set(usable)) > 1:
            affected: list[str] = []
            for value in usable:
                if value in by_relative:
                    affected.append(by_relative[value])
            return None, tuple(dict.fromkeys(affected))
        exact = usable[0] if usable else "__missing__"
        return by_relative.get(exact), ()

    # A path without a directory is legacy basename evidence.  If no path was
    # supplied, use the legacy name/filename field in the same way.
    raw_legacy = str(legacy_name or path or "").strip()
    normalized = _norm(raw_legacy)
    if not normalized:
        return None, ()
    basename = Path(normalized).name
    matches = by_name.get(basename, [])
    if len(matches) == 1:
        return matches[0], ()
    if len(matches) > 1:
        return None, tuple(matches)
    return None, ()
