"""Compact SHG output names with portable source identities and path budgets."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re


# Keep exports usable by ordinary Windows applications without long-path support.
# The budget includes the absolute path and counts supplementary Unicode characters
# as two UTF-16 code units, as Windows does.
SHG_PATH_LIMIT = 240
_COLLISION_RESERVE = 4  # _999
_STAGING_DIRECTORY = ".shg-export-00000000"


def windows_path_length(value: str | Path) -> int:
    return len(str(value).encode("utf-16-le")) // 2


def _shorten(value: str, limit: int) -> str:
    if windows_path_length(value) <= limit:
        return value
    # Retain a readable source prefix; the identity hash disambiguates truncation.
    result = []
    used = 0
    for character in value:
        size = windows_path_length(character)
        if used + size > limit:
            break
        result.append(character)
        used += size
    return "".join(result).rstrip(" ._") or "s"


def _source_label(source: str) -> str:
    name = Path(source).stem
    return re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', "_", name).strip(" ._") or "source"


def _center_tag(center_nm: float) -> str:
    if not math.isfinite(center_nm):
        raise ValueError("The SHG integration wavelength must be finite before exporting.")
    return f"{center_nm:g}".replace(".", "p").replace("+", "") + "nm"


def _identity_hash(root: Path, sources: tuple[str, ...], center_nm: float, compare: bool) -> str:
    identities = []
    root = root.resolve()
    for source in sources:
        path = Path(source)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            identity = "relative:" + path.relative_to(root).as_posix()
        except ValueError:
            identity = "absolute:" + path.as_posix()
        identities.append(os.path.normcase(identity))
    payload = json.dumps(["compare" if compare else "single", float(center_nm), identities], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _path_error(root: Path) -> ValueError:
    return ValueError(
        "The SHG export path is too long for Windows-compatible output. "
        "Choose a shorter experiment folder or move it closer to the drive root, then export again. "
        f"Folder: {root} (maximum output path: {SHG_PATH_LIMIT} UTF-16 characters)."
    )


def validate_shg_output_paths(paths) -> None:
    """Reject an unusable destination before writing any part of an export."""
    for path in paths:
        path = Path(path).absolute()
        if windows_path_length(path) > SHG_PATH_LIMIT or windows_path_length(path.name) > 255:
            raise _path_error(path.parent)


def create_shg_package_dir(
    experiment_root: str | Path,
    sources: tuple[str, ...],
    center_nm: float,
    *,
    compare: bool,
    reserve_staging: bool = False,
) -> Path:
    """Reserve a short analysis directory without replacing an earlier export."""
    root = Path(experiment_root).absolute()
    if len(sources) != (2 if compare else 1):
        raise ValueError("SHG export needs one source, or a reference and sample for comparison.")
    package_root = root / "Processed Data" / "SHG"
    parents = [package_root]
    if reserve_staging:
        parents.append(root / _STAGING_DIRECTORY)
    longest_file = "comparison_999_settings.json" if compare else "curve_999_settings.json"
    name_limit = min(64, *(SHG_PATH_LIMIT - windows_path_length(parent) - 2
                          - windows_path_length(longest_file) - _COLLISION_RESERVE
                          for parent in parents))
    suffix = f"_{_center_tag(center_nm)}_{_identity_hash(root, sources, center_nm, compare)}"
    prefix = "compare_" if compare else "single_"
    separator = "_vs_" if compare else ""
    available = name_limit - windows_path_length(prefix + separator + suffix)
    if available < (6 if compare else 3):
        raise _path_error(root)
    if compare:
        first_limit = min(16, available // 2)
        second_limit = min(16, available - first_limit)
        label = _shorten(_source_label(sources[0]), first_limit) + separator + _shorten(_source_label(sources[1]), second_limit)
    else:
        label = _shorten(_source_label(sources[0]), min(32, available))
    base = prefix + label + suffix
    # Validate first so a too-long root does not leave a partial history tree.
    for parent in parents:
        validate_shg_output_paths([parent / (base + "_999") / longest_file])
    package_root.mkdir(parents=True, exist_ok=True)
    for index in range(1000):
        candidate = package_root / (base if index == 0 else f"{base}_{index:02d}")
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"Could not allocate an SHG analysis package for {base!r}.")


def shg_curve_stem(folder: str | Path, output_dir: Path, source: str, center_nm: float) -> str:
    """Short, identifiable names for callers exporting into a shared directory."""
    suffix = f"_{_center_tag(center_nm)}_{_identity_hash(Path(folder), (source,), center_nm, False)}_curve"
    available = min(32, SHG_PATH_LIMIT - windows_path_length(output_dir.absolute()) - 1
                    - windows_path_length("shg_" + suffix + "_999_settings.json"))
    if available < 3:
        raise _path_error(output_dir)
    return "shg_" + _shorten(_source_label(source), available) + suffix
