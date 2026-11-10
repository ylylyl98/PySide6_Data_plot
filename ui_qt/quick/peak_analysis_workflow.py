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
    from core.peak_profiles import filter_presets, recommended_detection_method
    output = []
    for index, source in enumerate(sources):
        if cancelled and cancelled():
            raise InterruptedError("Peak Analysis snapshot cancelled.")
        if source["channel"] == "VP":
            continue
        dataset = create_dataset(source["cube"], source["kind"], source["channel"], source["source"],
            source["name"], source.get("view"))
        dataset["row"] = min(len(dataset["cube"].gate) - 1, max(0, int(source.get("row", 0))))
        dataset["analysis_scope"] = source.get("analysis_scope", "spectrum")
        dataset["scope"] = 2
        bounds = {key: dataset["settings"][key] for key in ("x_min", "x_max", "y_min", "y_max")}
        view = source.get("view") or {}
        for axis in ("x", "y"):
            limits = view.get(axis + "lim")
            if limits is not None and len(limits) == 2:
                bounds[axis + "_min"], bounds[axis + "_max"] = sorted(map(float, limits))
        dataset["custom_range"] = bounds
        dataset["settings"].update(bounds)
        dataset["reply_server"] = reply_server
        for field in ("group_id", "product_label", "source_name", "source_processing", "analysis_scope"):
            if field in source:
                dataset[field] = deepcopy(source[field])
        label = dataset.get("product_label", "")
        dataset["settings"].update(filter_presets(dataset["cube"], dataset["kind"], dataset["channel"], label)["Balanced"])
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
        settings.update(y_min=float(dataset["cube"].gate[row]), y_max=float(dataset["cube"].gate[row]), neighbor_support=0)
    report = progress or (lambda value: None)
    if action in {"detect", "filter"}:
        if action == "detect":
            pool = detect_heatmap(dataset, settings, cancelled, lambda n: report(n * 3 // 4))
        else:
            if scope == "spectrum" and dataset.get("candidates") and int(dataset["row"]) not in dataset["candidates"].get("detected_rows", list(range(dataset["candidates"].get("rows", 0)))):
                raise ValueError("Find peaks for the current spectrum before filtering.")
            if not dataset.get("candidates"):
                raise ValueError("Find all peaks before filtering cached candidates.")
            pool = prepare_quality_cache(dataset, dataset["candidates"], cancelled, lambda n: report(n // 3))
        dataset["candidates"] = pool
        result = filter_heatmap(dataset, settings, cancelled,
            lambda n: report(75 + n // 4) if action == "detect" else report(33 + n * 2 // 3))
        if action == "detect":
            dataset["show_overlay"] = True
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
        result = filter_heatmap(dataset, settings, cancelled, lambda n: report(50 + n // 4))
        dataset = accept_filtered(dataset, pool, result)
        selected = {branch["id"] for branch in dataset["branches"] if branch.get("enabled", True)}
        result["points"] = [point for point in result["points"] if point["branch_id"] in selected]
        manual = [b for b in dataset["branches"] if b.get("manual") and b.get("enabled", True)]
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
            return dataset
        from collections import Counter
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
                        point["width_mev"], point.get("status"))
            old_points = [p for p in previous_result.get("points", [])
                          if p.get("status") == "accepted" and p["row_index"] not in overflow]
            new_points = [p for p in result["points"] if p.get("status") == "accepted"]
            old_by_key = {fit_key(p): p for p in old_points}
            reuse = (previous_result.get("dataset_key") == dataset["key"]
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
                    "height", "height_error", "area", "area_error", "fit_area_full", "fit_area_full_error", "uncertainty_status", "rms")
                for point in new_points:
                    old = old_by_key[fit_key(point)]
                    point.update({field: old[field] for field in fit_fields if field in old})
                result["fits"] = deepcopy(previous_result["fits"])
                result["method"] = previous_result["method"]
                report(100)
            else:
                result = adapter.fit(dataset, result, settings, cancelled, lambda n: report(75 + n // 4))
            dataset["overlay_position"] = "fitted"
        result["points"].extend(skipped)
        dataset.update(result=result, show_overlay=True)
        return dataset
    if action not in {"track", "fit"}:
        raise ValueError("Unknown Peak Analysis operation.")
    if dataset["kind"] == "PL":
        from core import pl_peak_analysis as adapter
    else:
        from core import drr_peak_adapter as adapter
    if action == "fit" and dataset["kind"] != "PL":
        raise ValueError("Lorentzian/Gaussian fitting is available for PL intensity; DRR uses supplied extrema.")
    if dataset.get("candidates") and not dataset.get("manual_seeds"):
        result = filter_heatmap(dataset, settings, cancelled, lambda n: report(n // 2 if action == "fit" else n))
        selected = {branch["id"] for branch in dataset["branches"] if branch.get("enabled", True)}
        result["points"] = [point for point in result["points"] if point["branch_id"] in selected]
        result["candidate_mode"] = False
    else:
        result = adapter.track(dataset, dataset["branches"], settings, cancelled,
            lambda n: report(n // 2 if action == "fit" else n))
    if action == "fit":
        result = adapter.fit(dataset, result, settings, cancelled, lambda n: report(50 + n // 2))
        dataset["overlay_position"] = "fitted"
    dataset["result"] = result
    return dataset


def add_seed(dataset, row, energy):
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


def export_analysis(folder, dataset, *, cancelled=None, progress=None):
    import os
    from tempfile import TemporaryDirectory
    from core.peak_export import export_dataset
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError("Select an existing export folder.")
    if cancelled and cancelled():
        raise InterruptedError("Peak Analysis export cancelled.")
    # Atomic publication keeps cancelled/failed analysis packages out of history.
    with TemporaryDirectory(prefix=".peak-export-", dir=root) as temporary:
        staged = Path(export_dataset(temporary, dataset, cancelled=cancelled))
        if cancelled and cancelled():
            raise InterruptedError("Peak Analysis export cancelled.")
        target = root / staged.name
        if target.exists():
            raise FileExistsError("The generated Peak Analysis package already exists.")
        os.replace(staged, target)
    if progress:
        progress(100)
    return str(target)


def send_overlay(dataset, session_path, reply_server, *, cancelled=None, progress=None):
    from uuid import uuid4
    from core.peak_plotting import overlay_payload
    from core.peak_workspace import atomic_json
    from ui_qt.peak_analysis_ipc import send_message
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
