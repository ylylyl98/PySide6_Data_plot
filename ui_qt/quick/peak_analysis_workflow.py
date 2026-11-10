"""Pure captured transactions for the standalone native Peak Analysis tool."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def dataset_snapshot(dataset, *, share_candidates=False):
    shared = {"cube", "candidates"} if share_candidates else {"cube"}
    return {key: value if key in shared else deepcopy(value) for key, value in dataset.items()}


def capture_request(dataset):
    """Capture controls cheaply; large committed result/cache values are read-only.

    The bridge disables result edits while a request is active. Algorithms copy
    the structures they mutate on the worker, so posting a job never deep-copies
    an entire candidate map on the GUI thread.
    """
    committed = {"cube", "candidates", "result", "branch_catalog", "candidate_assignments", "segment_assignments"}
    return {key: value if key in committed else deepcopy(value) for key, value in dataset.items()}


def prepare_sources(sources, *, reply_server="", cancelled=None, progress=None):
    from core.peak_workspace import create_dataset, fingerprint
    from core.peak_profiles import drr_initial_filter_settings, filter_presets, recommended_detection_method
    output = []
    for index, source in enumerate(sources):
        if cancelled and cancelled():
            raise InterruptedError("Peak Analysis snapshot cancelled.")
        if source["channel"] == "VP":
            continue
        dataset = create_dataset(source["cube"], source["kind"], source["channel"], source["source"],
            source["name"], source.get("view"))
        dataset["row"] = min(len(dataset["cube"].gate) - 1, max(0, int(source.get("row", 0))))
        dataset["analysis_scope"] = source.get("analysis_scope", "map")
        dataset["analysis_scope_explicit"] = "analysis_scope" in source
        dataset["analysis_workflow_version"] = 3
        dataset["scope"] = 0 if dataset["analysis_scope"] == "map" else 2
        bounds = {key: dataset["settings"][key] for key in ("x_min", "x_max", "y_min", "y_max")}
        view = source.get("view") or {}
        for axis in ("x", "y"):
            limits = view.get(axis + "lim")
            if limits is not None and len(limits) == 2:
                bounds[axis + "_min"], bounds[axis + "_max"] = sorted(map(float, limits))
        dataset["custom_range"] = bounds
        if dataset["scope"] == 2:
            dataset["settings"].update(bounds)
        dataset["reply_server"] = reply_server
        for field in ("group_id", "product_label", "source_name", "source_processing", "source_export", "analysis_scope", "processed_path", "processed_path_inferred", "processed_path_hint", "history_root"):
            if field in source:
                dataset[field] = deepcopy(source[field])
        label = dataset.get("product_label", "")
        dataset["settings"].update(filter_presets(dataset["cube"], dataset["kind"], dataset["channel"], label)["Balanced"])
        if dataset["kind"] == "DRR":
            dataset["settings"].update(drr_initial_filter_settings(dataset["cube"]))
            dataset["settings"]["candidate_floor"] = 0.
            dataset["auto_filters"] = True
        else:
            from core.pl_peak_profiles import pl_initial_filter_settings
            dataset['settings'].update(pl_initial_filter_settings(dataset['cube']))
            dataset['auto_filters'] = True
        dataset["settings"]["detection_method"] = recommended_detection_method(dataset["kind"], dataset["channel"], label)
        if dataset["kind"] == "DRR" and "group_id" not in dataset:
            dataset["group_id"] = fingerprint(source.get("group_cube", source["cube"]), "DRR", "source", source["source"])
        output.append(dataset)
        if progress:
            progress(int((index + 1) * 100 / max(1, len(sources))))
    if not output:
        raise ValueError("Plot a PL, DRR, Compare intensity or Power intensity dataset first. VP is not peak-fitted.")
    return output


def accept_filtered(dataset, pool, result):
    if dataset.get("kind") == "PL" and dataset.get("manual_peak_edits"):
        from core.pl_manual_peaks import apply_pl_peak_edits
        result = apply_pl_peak_edits(dataset, result, dataset["settings"])
    manual = {branch["id"]: branch for branch in dataset["branches"] if branch.get("manual")}
    for branch in result["branches"]:
        if branch["id"] in manual:
            branch.update(manual[branch["id"]])
    present = {branch["id"] for branch in result["branches"]}
    result["branches"].extend(branch for key, branch in manual.items() if key not in present)
    dataset.update(candidates=pool, result=result, branches=result["branches"], branch_catalog=result["branch_catalog"],
        manual_seeds=bool(manual), candidate_assignments=result["assignments"],
        segment_assignments=result["segment_assignments"], overlay_position="detected")
    return dataset


def analyze_dataset(dataset, action, *, cancelled=None, progress=None):
    from core.peak_candidates import detect_heatmap, filter_heatmap
    from core.peak_quality import prepare_quality_cache
    # Committed histories and candidate pools are read-only inputs: filtering
    # copies branch metadata and fitting builds a separate result. Sharing them
    # avoids cloning large maps on every transaction, including cancellation.
    dataset = capture_request(dataset)
    settings = deepcopy(dataset["settings"])
    scope = dataset.get("analysis_scope", "map")
    if scope not in {"map", "spectrum"}:
        raise ValueError("Unknown analysis scope.")
    if scope == "spectrum":
        row = int(dataset["row"])
        if not 0 <= row < len(dataset["cube"].gate):
            raise ValueError("Select a measured spectrum first.")
        settings.update(y_min=float(dataset["cube"].gate[row]), y_max=float(dataset["cube"].gate[row]), neighbor_support=0, remove_scatter=False, neighbor_adaptive=False)
    report = progress or (lambda value: None)
    if action in {"filter", "auto-filter", "fit"} and dataset.get("detection_dirty"):
        raise ValueError("Detection settings changed. Find peaks again first.")
    if action == "fit" and dataset.get("filters_pending"):
        raise ValueError("Apply filters before fitting.")
    if action in {"detect", "filter", "auto-filter"}:
        if action == "detect":
            pool = detect_heatmap(dataset, settings, cancelled, lambda n: report(n * 3 // 5))
        else:
            if scope == "spectrum" and dataset.get("candidates") and int(dataset["row"]) not in dataset["candidates"].get("detected_rows", list(range(dataset["candidates"].get("rows", 0)))):
                raise ValueError("Find peaks for the current spectrum before filtering.")
            if not dataset.get("candidates"):
                raise ValueError("Find all peaks before filtering cached candidates.")
            pool = prepare_quality_cache(dataset, dataset["candidates"], cancelled, lambda n: report(n // 3))
        dataset["candidates"] = pool
        if action == "auto-filter" or action == "detect" and dataset.get("auto_filters", False):
            if dataset['kind'] == 'PL':
                from core.pl_peak_profiles import recommend_pl_filters as recommend_filters
            else:
                from core.peak_profiles import recommend_drr_filters as recommend_filters
            recommended, dataset["auto_filter_summary"] = recommend_filters(dataset, pool, cancelled,
                lambda n: report(60 + n // 4) if action == "detect" else report(33 + n // 2))
            dataset["settings"].update(recommended)
            settings.update(recommended)
            if scope == "spectrum":
                settings["neighbor_support"] = 0
                settings['remove_scatter'] = False
                settings['neighbor_adaptive'] = False
            dataset["auto_filters"] = True
        result = filter_heatmap(dataset, settings, cancelled,
            lambda n: report(85 + n * 15 // 100) if action == "detect" else
                report(83 + n * 17 // 100) if action == "auto-filter" else report(33 + n * 2 // 3))
        if action == "detect":
            dataset["show_overlay"] = True
            dataset["detection_dirty"] = False
        dataset["filters_pending"] = False
        return accept_filtered(dataset, pool, result)
    if action == "find-fit":
        if dataset["kind"] != "PL":
            raise ValueError("Lorentzian/Gaussian fitting is available for PL intensity.")
        from core.peak_candidates import detection_settings
        from core import pl_peak_analysis as adapter
        previous_result = dataset.get("result") or {}
        pool = dataset.get("candidates")
        expected_rows = [int(dataset["row"])] if scope == "spectrum" else list(range(len(dataset["cube"].gate)))
        cached_rows = pool.get("detected_rows", list(range(pool.get("rows", 0)))) if pool else []
        if not pool or pool.get("dataset_key") != dataset["key"] or pool.get("settings") != detection_settings(settings) or cached_rows != expected_rows:
            pool = detect_heatmap(dataset, settings, cancelled, lambda n: report(n // 2))
        dataset["candidates"] = pool
        dataset["detection_dirty"] = False
        if dataset.get('auto_filters', False):
            from core.pl_peak_profiles import recommend_pl_filters
            recommended, dataset['auto_filter_summary'] = recommend_pl_filters(dataset, pool, cancelled)
            dataset['settings'].update(recommended)
            settings.update(recommended)
            if scope == 'spectrum':
                settings['neighbor_support'] = 0
        result = filter_heatmap(dataset, settings, cancelled, lambda n: report(50 + n // 4))
        dataset = accept_filtered(dataset, pool, result)
        result = dataset["result"]
        selected = {branch["id"] for branch in dataset["branches"] if branch.get("enabled", True)}
        result["points"] = [point for point in result["points"] if point["branch_id"] in selected]
        manual = [b for b in dataset["branches"] if b.get("manual") and not b.get("manual_point_only") and b.get("enabled", True)]
        if manual:
            tracked = adapter.track(dataset, manual, settings, cancelled)
            existing = {p["branch_id"] for p in result["points"]}
            manual_points = [p for p in tracked["points"] if p["branch_id"] not in existing]
            # A manually picked seed may precede detection and therefore have a
            # different identity for the very same measured extremum. Prefer
            # the explicit seed instead of fitting two coincident components.
            import numpy as np
            tolerance = float(np.median(np.diff(dataset["cube"].energy))) * .5
            duplicate_ids = set()
            kept = []
            for point in result["points"]:
                duplicate = any(point["row_index"] == other["row_index"]
                    and point["polarity"] == other["polarity"]
                    and abs(point["energy"] - other["energy"]) <= tolerance
                    and other.get("status") == "accepted" for other in manual_points)
                if duplicate:
                    duplicate_ids.add(point["branch_id"])
                else:
                    kept.append(point)
            result["points"] = kept + manual_points
            remaining = {point["branch_id"] for point in result["points"]}
            redundant = duplicate_ids - remaining
            result["branches"] = [branch for branch in result["branches"] if branch["id"] not in redundant]
            dataset["branches"] = result["branches"]
            result["notices"].extend(tracked.get("notices", []))
        result["candidate_mode"] = False
        if not any(p.get("status") == "accepted" for p in result["points"]):
            result["notices"].append("No selected peaks available to fit. Detection results are retained.")
            dataset["result"] = result
            dataset["show_overlay"] = True
            dataset['filters_pending'] = False
            return dataset
        from collections import Counter, defaultdict
        budget_deferred = []
        if dataset.get('auto_filters') and settings.get('pl_weak_features'):
            by_row = defaultdict(list)
            for point in result['points']:
                if point.get('status') == 'accepted':
                    by_row[point['row_index']].append(point)
            manual_ids = {b['id'] for b in dataset['branches'] if b.get('manual')}
            for points in by_row.values():
                if len(points) <= 8:
                    continue
                protected = [p for p in points if p.get('manual') or p['branch_id'] in manual_ids or p.get('prominence_fraction', 0) >= .1]
                if len(protected) > 8:
                    continue  # Never silently discard explicit/prominent seeds.
                protected_ids = {id(p) for p in protected}
                others = sorted((p for p in points if id(p) not in protected_ids),
                                key=lambda p: (-(p.get('snr') or 0), -p['width_mev'], p['energy']))
                for point in others[8 - len(protected):]:
                    point['fit_status'] = 'not fitted: automatic seed budget'
                    budget_deferred.append(point)
            if budget_deferred:
                deferred_ids = {id(p) for p in budget_deferred}
                result['points'] = [p for p in result['points'] if id(p) not in deferred_ids]
                result['notices'].append(f'{len(budget_deferred)} lower-evidence candidates retained without fitting: automatic fits use at most 8 seeds per spectrum. Narrow the range or select branches to review them.')
                result['auto_fit_seed_budget'] = dict(limit=8, deferred=len(budget_deferred))
        counts = Counter(p["row_index"] for p in result["points"] if p.get("status") == "accepted")
        overflow = {row for row, count in counts.items() if count > 8}
        skipped = [p for p in result["points"] if p["row_index"] in overflow]
        if overflow:
            result["notices"].append(
                "More than 8 selected peaks in " + str(len(overflow)) +
                " spectrum(s); narrow the range or deselect branches to fit those spectra.")
            result["points"] = [p for p in result["points"] if p["row_index"] not in overflow]
        if any(p.get("status") == "accepted" for p in result["points"]):
            # Fit depends only on the measured data, effective settings and
            # selected accepted peak seeds. Reuse a completed fit only when
            # each of these inputs is identical; still rebuild filtered points
            # and branch choices above so UI edits remain authoritative.
            def fit_key(point):
                return (point["row_index"], point["branch_id"], point["energy"],
                        point["width_mev"], point.get("status"), point.get("fit_center_min"),
                        point.get("fit_center_max"), point.get("fit_center_fixed"))
            old_points = [p for p in previous_result.get("points", [])
                          if p.get("status") == "accepted" and p["row_index"] not in overflow
                          and p.get('fit_status') != 'not fitted: automatic seed budget']
            new_points = [p for p in result["points"] if p.get("status") == "accepted"]
            old_by_key = {fit_key(p): p for p in old_points}
            reuse = (previous_result.get("dataset_key") == dataset["key"]
                     and previous_result.get('pl_fit_version') == adapter.PL_FIT_VERSION
                     and bool(previous_result.get("fits"))
                     and not any(fit["row_index"] in overflow for fit in previous_result["fits"])
                     and previous_result.get("settings") == result["settings"]
                     and len(old_by_key) == len(old_points) == len(new_points)
                     and all(fit_key(p) in old_by_key for p in new_points)
                     and all("fit_status" in old_by_key[fit_key(p)] for p in new_points))
            if reuse:
                if cancelled and cancelled():
                    raise RuntimeError("Peak fitting cancelled.")
                fit_fields = ("fit_status", "fit_center", "center_error_ev", "fwhm_mev", "fwhm_error_mev",
                    "height", "height_error", "area", "area_error", "fit_area_full", "fit_area_full_error", "uncertainty_status", "rms", "fit_quality_status", "fit_local_residual_sigma", "fit_height_to_local_noise")
                for point in new_points:
                    old = old_by_key[fit_key(point)]
                    point.update({field: old[field] for field in fit_fields if field in old})
                result["fits"] = deepcopy(previous_result["fits"])
                result["method"] = previous_result["method"]
                result['pl_fit_version'] = adapter.PL_FIT_VERSION
                report(100)
            else:
                result = adapter.fit(dataset, result, settings, cancelled, lambda n: report(75 + n // 4))
            dataset["overlay_position"] = "fitted"
        result["points"].extend(skipped)
        result['points'].extend(budget_deferred)
        dataset.update(result=result, show_overlay=True, filters_pending=False)
        return dataset
    if action not in {"track", "fit"}:
        raise ValueError("Unknown Peak Analysis operation.")
    if dataset["kind"] == "PL":
        from core import pl_peak_analysis as adapter
    else:
        from core import drr_peak_adapter as adapter
    if action == "fit" and dataset["kind"] != "PL":
        raise ValueError("Lorentzian/Gaussian fitting is available for PL intensity; DRR uses supplied extrema.")
    if action == "fit" and dataset.get("result") is not None:
        # Fit exactly the reviewed, enabled seeds. Never refilter or retrack here.
        from core.peak_analysis_state import invalidate_fit
        result = deepcopy(dataset["result"])
        selected = {b["id"] for b in dataset["branches"] if b.get("enabled", True)}
        reviewed_points = result.get("points", [])
        result["points"] = [p for p in reviewed_points
                            if p.get("branch_id") in selected and p.get("status") == "accepted"
                            and (scope != "spectrum" or p.get("row_index") == int(dataset["row"]))]
        selected_ids = {id(p) for p in result["points"]}
        untouched = [deepcopy(p) for p in reviewed_points if id(p) not in selected_ids]
        if not result["points"]:
            raise ValueError("Select at least one accepted peak to fit.")
        dataset["result"] = result
        invalidate_fit(dataset)
        result = dataset["result"]
        result["candidate_mode"] = False
        from collections import Counter, defaultdict
        by_row = defaultdict(list)
        for point in result["points"]:
            by_row[point["row_index"]].append(point)
        deferred = []
        manual_ids = {b["id"] for b in dataset["branches"] if b.get("manual")}
        if dataset.get("auto_filters") and settings.get("pl_weak_features"):
            for points in by_row.values():
                protected = [p for p in points if p.get("manual") or p["branch_id"] in manual_ids
                             or p.get("prominence_fraction", 0) >= .1]
                if len(points) > 8 and len(protected) <= 8:
                    protected_ids = {id(p) for p in protected}
                    others = sorted((p for p in points if id(p) not in protected_ids),
                                    key=lambda p: (-(p.get("snr") or 0), -p["width_mev"], p["energy"]))
                    deferred.extend(others[8 - len(protected):])
        deferred_ids = {id(p) for p in deferred}
        remaining = [p for p in result["points"] if id(p) not in deferred_ids]
        overflow = {row for row, count in Counter(p["row_index"] for p in remaining).items() if count > 8}
        skipped = [p for p in remaining if p["row_index"] in overflow]
        result["points"] = [p for p in remaining if p["row_index"] not in overflow]
        for point in deferred:
            point["fit_status"] = "not fitted: automatic seed budget"
        if deferred or skipped:
            result.setdefault("notices", []).append(
                f"{len(deferred) + len(skipped)} peaks were not fitted: at most 8 peaks per spectrum. Review the selection.")
        if result["points"]:
            result = adapter.fit(dataset, result, settings, cancelled, lambda n: report(n))
        from core.peak_analysis_state import FIT_FIELDS
        for point in untouched:
            for field in FIT_FIELDS:
                point.pop(field, None)
        result["points"].extend(skipped + deferred + untouched)
        dataset["result"] = result
        dataset["overlay_position"] = "fitted"
        return dataset
    if dataset["kind"] == "PL" and dataset.get("manual_peak_edits") and not dataset.get("candidates"):
        result = deepcopy(dataset["result"])
    elif dataset.get("candidates") and not dataset.get("manual_seeds"):
        result = filter_heatmap(dataset, settings, cancelled, lambda n: report(n // 2 if action == "fit" else n))
        selected = {branch["id"] for branch in dataset["branches"] if branch.get("enabled", True)}
        result["points"] = [point for point in result["points"] if point["branch_id"] in selected]
        result["candidate_mode"] = False
    else:
        result = adapter.track(dataset, dataset["branches"], settings, cancelled,
            lambda n: report(n // 2 if action == "fit" else n))
    if dataset["kind"] == "PL" and dataset.get("manual_peak_edits"):
        from core.pl_manual_peaks import apply_pl_peak_edits
        result = apply_pl_peak_edits(dataset, result, settings)
        selected = {branch['id'] for branch in dataset['branches'] if branch.get('enabled', True)}
        result['points'] = [point for point in result['points'] if point['branch_id'] in selected]
    if action == "fit":
        result = adapter.fit(dataset, result, settings, cancelled, lambda n: report(50 + n // 2))
        dataset["overlay_position"] = "fitted"
    dataset["result"] = result
    return dataset


def add_seed(dataset, row, energy):
    if dataset["kind"] == "PL":
        from core.pl_manual_peaks import edit_pl_peak
        return edit_pl_peak(dataset, row, energy)
    from uuid import uuid4
    from core import pl_peak_analysis, drr_peak_adapter
    if dataset["kind"] == "PL":
        adapter = pl_peak_analysis
    else:
        adapter = drr_peak_adapter
    candidates = adapter.detect(dataset, row, dataset["settings"])
    nearby = min(candidates, key=lambda branch: abs(branch["seed_energy"] - energy)) if candidates else None
    polarity = "dip" if dataset["kind"] == "DRR" and dataset["settings"]["polarity"] == "dips" else "peak"
    if nearby and abs(nearby["seed_energy"] - energy) <= dataset["settings"]["max_shift_mev"] / 1000:
        energy, polarity = nearby["seed_energy"], nearby["polarity"]
    cube = dataset["cube"]
    if not float(cube.energy.min()) <= energy <= float(cube.energy.max()):
        raise ValueError("Pick a seed inside the measured energy range.")
    existing = next((branch for branch in dataset["branches"] if branch["polarity"] == polarity
        and abs(branch["seed_energy"] - energy) < .0001), None)
    if existing:
        existing.update(seed_energy=energy, seed_y=float(cube.gate[row]), seed_row=row, manual=True)
    else:
        color = max((branch["color"] for branch in dataset["branches"]), default=-1) + 1
        dataset["branches"].append(dict(id=uuid4().hex, name=f"{'D' if polarity == 'dip' else 'P'}{color + 1}",
            color=color, enabled=True, manual=True, polarity=polarity, seed_energy=energy,
            seed_y=float(cube.gate[row]), seed_row=row, width_mev=5.))
    dataset.update(result=None, manual_seeds=True)
    return dataset


def refresh_source_export(dataset, session_path, *, cancelled=None):
    """Refresh from the live main plot, retaining captured settings offline."""
    import json
    import time
    from uuid import uuid4
    from ui_qt.peak_analysis_ipc import send_message
    reply = dataset.get('reply_server')
    if not reply:
        return dataset
    response = Path(session_path).parent / 'display_requests' / (uuid4().hex + '.json')
    response.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not send_message(reply, dict(export_display=dict(key=dataset['key'], response=str(response)))):
            return dataset
        deadline = time.monotonic() + 15.
        while not response.is_file():
            if cancelled and cancelled():
                raise InterruptedError('Source plot export cancelled.')
            if time.monotonic() >= deadline:
                raise ValueError('Main plot did not provide its display settings. Finish plotting and try saving again.')
            time.sleep(.025)
        payload = json.loads(response.read_text(encoding='utf-8'))
        if payload.get('error'):
            raise ValueError(payload['error'])
        return dict(dataset, source_export=payload['source_export'])
    finally:
        response.unlink(missing_ok=True)


def export_analysis(folder, dataset, *, cancelled=None, progress=None, include_figures=True):
    from core.peak_export import export_dataset
    from core.peak_analysis_state import analysis_state
    if not analysis_state(dataset)["can_export"]:
        raise ValueError("Find peaks and apply pending changes before exporting.")
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError("Select an existing export folder.")
    if cancelled and cancelled():
        raise InterruptedError("Peak Analysis export cancelled.")
    paths = export_dataset(root, dataset, cancelled=cancelled, include_figures=include_figures)
    if progress:
        progress(100)
    return paths


def send_overlay(dataset, session_path, reply_server, *, cancelled=None, progress=None):
    from uuid import uuid4
    from core.peak_plotting import overlay_payload
    from core.peak_workspace import atomic_json
    from ui_qt.peak_analysis_ipc import send_message
    from core.peak_analysis_state import analysis_state
    if not analysis_state(dataset)["can_export"]:
        raise ValueError("Find peaks and apply pending changes before applying the overlay.")
    if not reply_server:
        raise ValueError("Main app is unavailable. Reopen the dataset from the app to reconnect.")
    if cancelled and cancelled():
        raise InterruptedError("Peak overlay cancelled.")
    path = Path(session_path).parent / "overlays" / (uuid4().hex + ".json")
    atomic_json(path, overlay_payload(dataset))
    try:
        if cancelled and cancelled():
            raise InterruptedError("Peak overlay cancelled.")
        if not send_message(reply_server, {"overlay": str(path)}):
            raise ValueError("Main app is unavailable. Reopen the dataset from the app to reconnect.")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if progress:
        progress(100)
    return str(path)


__all__ = ["dataset_snapshot", "capture_request", "prepare_sources", "analyze_dataset", "accept_filtered", "add_seed",
           "export_analysis", "send_overlay"]
