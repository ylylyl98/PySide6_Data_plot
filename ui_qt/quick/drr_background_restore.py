"""Worker-side restoration of the latest exact raw CSV group's background.

Reading history does not activate managed versions or record another use. The
normal DRR loader still verifies input identity and managed BN certificates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re

from core.drr_sources import _read_drr_metadata, resolve_source_path
from core.load_cancellation import check_load_cancelled
from ui_qt.quick.drr_background_workflow import BackgroundDraft, BackgroundRecipe


def _kind(assignments):
    """Preserve recorded P/N types without opening a widget or source catalog."""
    from core.drr_measured_bn import is_measured_bn_name
    from core.drr_pn_pairing import detect_pn_role
    kinds = set()
    for item in assignments:
        if item.baseline_mode != "External":
            kinds.add("self")
            continue
        matched = next((kind for kind in ("p", "n", "pn")
            if item.selection_reason.startswith(f"P/N match ({kind}): ")), None)
        if matched is not None:
            kinds.add(matched)
            continue
        for name in item.baseline_source_files:
            parts = str(name).replace("\\", "/").casefold().split("/")
            if ".drr_bn_calibration" in parts or is_measured_bn_name(name):
                kinds.add("bn")
            elif "pn backgrounds" in parts or re.search(
                    r"(?<![a-z0-9])pn[ _-]*back(?:ground)?(?=$|[^a-z])", parts[-1]):
                kinds.add("pn")
            else:
                role = detect_pn_role(Path(name).name)
                kinds.add(role.lower() if role in ("P", "N") else "custom")
    return next(iter(kinds)) if len(kinds) == 1 else "custom"


def _latest_exact_records(context):
    identities = {}

    def identity(name):
        if name not in identities:
            identities[name] = str(resolve_source_path(context.folder, name)).casefold()
        return identities[name]

    wanted = {identity(name) for name in context.measurements}
    _, records = _read_drr_metadata(Path(context.folder), require_drr_operation=True)
    matches = tuple(record for record in records
        if {identity(name) for name in record.measurement_files} == wanted)
    if not matches:
        return ()
    newest = max(record.saved_time for record in matches)
    return tuple(record for record in matches if record.saved_time == newest)


def _fitted_settings(records, context):
    """Fitted exports store one group recipe, outside member assignments."""
    from core.drr_sources import _history_file_signature, _history_payload

    fitted = []
    fitted_markers = []
    for record in records:
        check_load_cancelled()
        path = Path(record.metadata_path)
        try:
            payload = _history_payload(path, _history_file_signature(path))
            processing = payload.get("processing", {})
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError("The latest saved DRR background metadata cannot be read; "
                             "choose the background explicitly again.") from exc
        if not isinstance(processing, dict):
            processing = {}
        is_fitted = ("fitted" in record.baseline_selection.casefold()
            or processing.get("background_mode") == "fitted"
            or processing.get("numerical_path") == "fitted_raw_average"
            or "fitted_background" in processing)
        fitted_markers.append(is_fitted)
        fitted.append(processing.get("fitted_background") if is_fitted else None)
    if not any(value is not None for value in fitted):
        if any(fitted_markers):
            raise ValueError("The latest saved fitted background has no fitting settings; "
                             "choose Fitted and review its settings again.")
        return None
    if (any(not isinstance(value, dict) for value in fitted)
            or len({json.dumps(value, sort_keys=True) for value in fitted}) != 1):
        raise ValueError("The latest saved DRR exports have conflicting or incomplete fitted "
                         "background settings; choose the background explicitly again.")
    settings = deepcopy(fitted[0])
    try:
        from core.drr_fitted_background import validate_recipe
        validate_recipe(settings, context.measurements)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"The latest saved fitted background cannot be restored: {exc}") from exc
    return settings


def restore_saved_background(request, *, auto_bn_compare=False):
    """Restore only a newly opened exact CSV group, before its first load.

    Owners call this after checking explicit Keep-background intent. They must
    not call it for ordinary reloads or explicit background edits. No matching
    guess, older export, or default Self recipe replaces an unusable latest
    saved recipe. Non-CSV imports bypass this operation entirely.
    """
    if not request.files or any(Path(name).suffix.casefold() != ".csv" for name in request.files):
        return request
    check_load_cancelled()
    from ui_qt.quick.drr_workflow import background_context
    context = background_context(request)
    latest = _latest_exact_records(context)
    if not latest:
        return request
    fitted = _fitted_settings(latest, context)
    if fitted is not None:
        draft = BackgroundDraft(kind="fitted", fitted_result=fitted,
            auto_bn_compare=bool(auto_bn_compare))
        recipe = BackgroundRecipe("Fitted (sample)", (), "last", fitted_settings=fitted,
            auto_bn_compare=bool(auto_bn_compare), draft=draft, context=context)
        return replace(request, backgrounds=(), background_mode="fitted",
            background_frame="last", background_recipe=recipe)

    from core.drr_saved_background import resolve_saved_group_background
    resolved = resolve_saved_group_background(context.folder, (), context.measurements, saved_recipes=latest)
    if resolved is None or not resolved.resolved:
        reason = getattr(resolved, "reason", "") or "The saved exact-group export is no longer available."
        raise ValueError("Saved DRR background could not be restored: " + reason)
    assignments = tuple(resolved.assignments)
    first = assignments[0]
    common = resolved.numerical_path == "common"
    kind = _kind(assignments)
    # Differing Self methods also need an exact member map; represent them as
    # a custom recipe rather than flattening them into one Self frame choice.
    if not common and kind == "self":
        kind = "custom"
    mode = first.baseline_mode if common else "External"
    files = tuple(dict.fromkeys(name for item in assignments for name in item.baseline_source_files))
    regions = first.regions if common else ()
    calibration = None
    if common and mode == "External":
        from core.drr_bn_store import DrrBnStore, is_bn_calibration_source
        managed = tuple(name for name in files if is_bn_calibration_source(context.folder, name))
        if managed:
            if len(files) != 1 or regions:
                raise ValueError("A group-calibrated BN must be used directly as one common reference.")
            store = DrrBnStore(context.folder)
            # Verify managed ownership even when the marker was copied to an
            # arbitrary filename. The normal loader still checks freshness,
            # exact group/axis, SG settings and the active selection.
            version = store.version_for_csv(managed[0])
            payload = {**store.load_result(version), "csv_path": str(version.csv_path)}
            calibration = dict(payload=payload, original=False)
            kind = "bn"
    draft = BackgroundDraft(kind=kind, files=first.baseline_files if common else files,
        which=first.baseline_which, regions=regions,
        saved_assignments=assignments if not common else (),
        auto_bn_compare=bool(auto_bn_compare), calibration=calibration)
    recipe = BackgroundRecipe(mode, files, first.baseline_which, regions, assignments,
        auto_bn_compare=bool(auto_bn_compare), draft=draft, context=context)
    background_mode = ("self-first" if mode == "Self (first frame)" else
        "self-last" if mode == "Self (last frame)" else "external")
    check_load_cancelled()
    return replace(request, backgrounds=files, background_mode=background_mode,
        background_frame=recipe.which, background_recipe=recipe)
