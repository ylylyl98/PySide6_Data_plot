"""Conservative readers for existing Compare export metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from core.source_identity import match_source_identity


def _same_source(root: Path, candidates: Sequence[str], expected: str, recorded: str) -> bool:
    normalize = lambda value: str(value).replace("\\", "/").casefold()
    if normalize(expected) == normalize(recorded):
        return True
    path_text = str(recorded)
    matched, _uncertain = match_source_identity(
        root,
        candidates,
        relative_path=path_text if "/" in path_text.replace("\\", "/") and not Path(path_text).is_absolute() else "",
        path=path_text if Path(path_text).is_absolute() else "",
        legacy_name=path_text,
    )
    return matched is not None and normalize(matched) == normalize(expected)


def _source_from_descriptor(root: Path, candidates: Sequence[str], descriptor: dict) -> str | None:
    matched, _uncertain = match_source_identity(
        root,
        candidates,
        relative_path=str(descriptor.get("source_relative_path", "")),
        path=str(descriptor.get("source_path", descriptor.get("path", ""))),
        legacy_name=str(descriptor.get("name", descriptor.get("filename", ""))),
    )
    return matched


def _record_sources(record: dict, root: Path, candidates: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    descriptors = record.get("sources", record.get("inputs", []))
    if not isinstance(descriptors, list):
        return result
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            continue
        role = str(descriptor.get("role", ""))
        source = _source_from_descriptor(root, candidates, descriptor)
        if source is not None:
            result[role] = source
    return result


def compare_history_for_selection(
    records: Sequence[dict],
    experiment_root: str | Path,
    mapping: dict[str, str],
    *,
    view: str,
    sources: Sequence[str] = (),
) -> list[dict]:
    """Return saved Compare records relevant to the active mapping/view.

    Records with future ``processing.compare_source_mapping`` prove a matching
    combination. Legacy per-panel records are retained with an explicit
    ``individual_panel`` scope and never prove a combined selection.
    """
    root = Path(experiment_root)
    active = {str(key): str(value) for key, value in mapping.items() if value}
    candidates = list(dict.fromkeys([*map(str, sources), *active.values()]))
    wanted_view = "VP" if str(view).casefold() in {"vp", "valley polarization"} else "Intensity"
    matches: list[dict] = []
    for raw in records:
        if not isinstance(raw, dict):
            continue
        operation = str(raw.get("operation", raw.get("workflow", "")))
        if operation.casefold() == "pl" or not operation.casefold().startswith("compare/"):
            continue
        processing = raw.get("processing", {})
        if not isinstance(processing, dict):
            processing = {}
        record_view = "VP" if operation.casefold() == "compare/vp" else "Intensity"
        if record_view != wanted_view:
            continue
        record_sources = _record_sources(raw, root, candidates)
        combo = processing.get("compare_source_mapping")
        if isinstance(combo, dict):
            combo = {str(key): str(value) for key, value in combo.items() if value}
            if wanted_view == "VP":
                if active.get("KK") and active.get("KKp") and all(
                    key in combo and _same_source(root, candidates, active[key], combo[key])
                    for key in ("KK", "KKp")
                ):
                    result = dict(raw); result["history_scope"] = "combination"; matches.append(result)
            elif active and all(
                key in combo and _same_source(root, candidates, active[key], combo[key])
                for key in active
            ):
                result = dict(raw); result["history_scope"] = "combination"; matches.append(result)
            continue
        # Legacy VP records have role-specific source descriptors and are safe
        # to match only as an exact same-record pair.
        if wanted_view == "VP":
            if active.get("KK") and active.get("KKp") and record_sources.get("source_KK") == active.get("KK") and record_sources.get("source_KKp") == active.get("KKp"):
                result = dict(raw); result["history_scope"] = "combination"; matches.append(result)
            continue
        channel = operation.split("/", 1)[1]
        expected = active.get(channel)
        if expected and record_sources.get("source") == expected:
            result = dict(raw); result["history_scope"] = "individual_panel"; result["channel"] = channel; matches.append(result)
    return matches
