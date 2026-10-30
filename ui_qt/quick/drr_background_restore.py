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


def _recorded_kind(records):
    """New exports retain the explicit role even for anonymously named sources."""
    from core.drr_sources import _history_file_signature, _history_payload
    values = []
    for record in records:
        path = Path(record.metadata_path)
        payload = _history_payload(path, _history_file_signature(path))
        processing = payload.get("processing", {})
        value = processing.get("background_kind") if isinstance(processing, dict) else None
        if value is not None and (not isinstance(value, str)
                or value not in {"self", "bn", "p", "n", "pn", "custom", "fitted"}):
            raise ValueError("Saved background source kind is invalid; choose the background again.")
        values.append(value)
    if not values or any(value is None for value in values):
        return None  # Older exports continue through their existing role inference.
    if len(set(values)) != 1:
        raise ValueError("Saved exports disagree on the background source kind; review the background again.")
    return values[0]


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


def _pn_batch_provenance(value):
    """Validate historical display data without recreating a live plan."""
    import math
    invalid = ValueError('Saved P/N batch provenance is invalid; choose backgrounds again.')
    if not isinstance(value, dict) or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
        raise invalid
    for key in ('p_candidates', 'n_candidates'):
        entries = value.get(key)
        if not isinstance(entries, list) or any(not isinstance(x, str) or not x for x in entries):
            raise invalid
    rows, segments = value.get('rows'), value.get('segments')
    if not isinstance(rows, list) or not rows or not isinstance(segments, list) or not segments:
        raise invalid
    names = []
    for row in rows:
        if (not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k]
                for k in ('measurement_file', 'p_file', 'n_file', 'timing_basis'))
                or type(row.get('manual')) is not bool or not isinstance(row.get('warnings'), list)
                or any(not isinstance(w, str) for w in row['warnings'])):
            raise invalid
        for key in ('measurement_p_gap', 'measurement_n_gap', 'pair_gap'):
            number = row.get(key)
            if number is not None and (type(number) not in (int, float) or not math.isfinite(number) or number < 0):
                raise invalid
        if row['p_file'] not in value['p_candidates'] or row['n_file'] not in value['n_candidates']:
            raise invalid
        names.append(row['measurement_file'])
    membership = []
    for segment in segments:
        if (not isinstance(segment, dict) or any(not isinstance(segment.get(k), str) or not segment[k]
                for k in ('p_file', 'n_file', 'method', 'version_id'))
                or segment['method'] not in ('conditioned_single', 'conditioned_bounded')):
            raise invalid
        for key in ('measurement_files', 'training_files'):
            if not isinstance(segment.get(key), list) or not segment[key] or any(not isinstance(x, str) or not x for x in segment[key]):
                raise invalid
        training = segment['training_files']
        expected_count = 1 if segment['method'] == 'conditioned_single' else 2
        if (len(training) != expected_count or len(set(training)) != expected_count
                or not set(training).issubset(segment['measurement_files'])):
            raise invalid
        matching_rows = [row for row in rows if row['measurement_file'] in segment['measurement_files']]
        if any((row['p_file'], row['n_file']) != (segment['p_file'], segment['n_file']) for row in matching_rows):
            raise invalid
        membership.extend(segment['measurement_files'])
    if len(set(names)) != len(names) or len(set(membership)) != len(membership) or set(names) != set(membership):
        raise invalid
    return deepcopy(value)


def _matching_policy(records):
    """Recover the selected policy without re-running or approving its matches."""
    from core.drr_match_conditions import normalize_relaxed_conditions
    from core.drr_sources import _history_file_signature, _history_payload
    policies = []
    for record in records:
        check_load_cancelled()
        path = Path(record.metadata_path)
        payload = _history_payload(path, _history_file_signature(path))
        processing = payload.get("processing", {})
        policy = processing.get("background_matching_policy") if isinstance(processing, dict) else None
        if policy is None:
            policies.append(None)
            continue
        if (not isinstance(policy, dict) or type(policy.get("schema_version")) is not int
                or policy["schema_version"] != 1 or policy.get("method") != "matched"
                or policy.get("kind") not in {"bn", "p", "n", "pn"}
                or policy.get("ranking") != "strict_before_relaxed_v1"):
            raise ValueError("Saved background matching conditions are invalid; choose them explicitly again.")
        relaxed = normalize_relaxed_conditions(policy.get("relaxed_conditions"))
        pool = policy.get("candidate_files")
        if pool is not None and (not isinstance(pool, list) or any(not isinstance(name, str) or not name for name in pool)):
            raise ValueError("Saved background candidate pool is invalid; select the backgrounds again.")
        warnings = policy.get("warnings")
        acknowledged = policy.get("warnings_acknowledged")
        if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings) or type(acknowledged) is not bool:
            raise ValueError("Saved background matching review is invalid; review the backgrounds again.")
        review = dict(warnings=warnings, warnings_acknowledged=acknowledged)
        if 'pn_batch' in policy:
            if policy['kind'] != 'pn':
                raise ValueError('Saved P/N batch provenance has an incompatible source kind.')
            review['pn_batch'] = _pn_batch_provenance(policy['pn_batch'])
        policies.append(dict(kind=policy["kind"], relaxed_conditions=relaxed,
            candidate_files=tuple(dict.fromkeys(pool)) if pool is not None else None,
            review=review))
    if not any(policy is not None for policy in policies):
        return None
    if any(policy != policies[0] for policy in policies):
        raise ValueError("Latest saved exports have conflicting background matching conditions; review them again.")
    return policies[0]


def _validate_batch_assignments(context, assignments, provenance):
    """Bind historical segment version IDs to the actual restored CSV sidecars."""
    from core.drr_composite_background import composite_metadata_path
    rows = {row['measurement_file'] for row in provenance['rows']}
    if rows != set(context.measurements):
        raise ValueError('Saved P/N batch provenance does not describe this measurement group.')
    by_name = {a.measurement_file: a for a in assignments}
    for segment in provenance['segments']:
        for name in segment['measurement_files']:
            assignment = by_name.get(name)
            if assignment is None or len(assignment.baseline_files) != 1:
                raise ValueError('Saved P/N batch provenance conflicts with restored assignments.')
            sidecar = composite_metadata_path(resolve_source_path(context.folder, assignment.baseline_files[0]))
            try:
                payload = json.loads(sidecar.read_text(encoding='utf-8'))
                version_id = payload['pn_generation']['version_id']
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError('Saved P/N batch version identity is missing.') from exc
            if version_id != segment['version_id']:
                raise ValueError('Saved P/N batch version identity conflicts with its restored assignment.')


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
    matching_policy = _matching_policy(latest)
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
    if matching_policy is not None and matching_policy['review'].get('pn_batch') is not None:
        _validate_batch_assignments(context, assignments, matching_policy['review']['pn_batch'])
    first = assignments[0]
    common = resolved.numerical_path == "common"
    kind = _recorded_kind(latest) or _kind(assignments)
    if (kind in {"p", "n", "pn", "bn"} and any(item.baseline_mode != "External" for item in assignments)
            or kind == "self" and any(item.baseline_mode == "External" for item in assignments)
            or kind == "fitted"):
        raise ValueError("Saved background source kind conflicts with its numerical recipe; review it again.")
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
    if matching_policy is not None:
        if matching_policy["kind"] != kind or calibration is not None or mode != "External":
            raise ValueError("Saved matching conditions conflict with the background recipe; review them again.")
        pool_field = "bn_candidates" if kind == "bn" else "pn_candidates"
        draft = replace(draft, assignment_method="matched", assignment_method_explicit=True,
            relaxed_conditions=matching_policy["relaxed_conditions"], saved_assignments=assignments,
            saved_matching_review=deepcopy(matching_policy["review"]),
            **{pool_field: matching_policy["candidate_files"]})
    recipe = BackgroundRecipe(mode, files, first.baseline_which, regions, assignments,
        auto_bn_compare=bool(auto_bn_compare), draft=draft, context=context)
    background_mode = ("self-first" if mode == "Self (first frame)" else
        "self-last" if mode == "Self (last frame)" else "external")
    check_load_cancelled()
    return replace(request, backgrounds=files, background_mode=background_mode,
        background_frame=recipe.which, background_recipe=recipe)
