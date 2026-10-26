"""Captured DRR Analysis requests shared by native views and IPC.

The scientific kernels and schema-one workspace remain authoritative. This
module deliberately imports no Qt classes or legacy window/controller.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path


def catalog(folder, *, cancellation=None):
    from core.drr_processed_groups import discover_groups
    from core.drr_comparison_groups import build_catalog
    from core.load_cancellation import load_scope, check_load_cancelled
    with load_scope(cancellation):
        paths = discover_groups(folder)
        check_load_cancelled()
        groups, entries = build_catalog(paths)
        return dict(paths=[str(path) for path in paths], groups=groups, entries=entries)


def source_version(path):
    path = Path(path)
    dat = path.with_name(path.name.removesuffix(".metadata.json") + ".dat")
    try:
        return [[value.stat().st_size, value.stat().st_mtime_ns, value.stat().st_ctime_ns] for value in (path, dat)]
    except OSError:
        return None


def store_for(folder, storage_base):
    from core.drr_comparison_groups import ComparisonStore, digest
    return ComparisonStore(Path(storage_base) / "comparisons" / digest(str(Path(folder).resolve()).casefold()))


def copy_dataset(dataset):
    return replace(dataset, result=deepcopy(dataset.result), provenance=deepcopy(dataset.provenance))


def carry_p2p_ranges(previous, target, default_bounds=None):
    """Carry navigation locks, marking a changed scientific definition pending."""
    if not previous:
        return target
    calculation, display = previous.get("fixed_calculation", False), previous.get("fixed_display", False)
    if target is None:
        if not (calculation or display):
            return None
        target = {**previous, "records": [], "display_ranges": {}, "inspection_rows": {}, "current_key": None,
                  "pending": True, "_recalculate_on_open": True}
        if default_bounds is not None:
            target["bounds"] = list(default_bounds)
        target["limit_y"] = False
        target["smoothing"] = dict(method="Off", unit="points", window_points=21, polyorder=2)
    else:
        target = deepcopy(target)
    target["fixed_calculation"], target["fixed_display"] = calculation, display
    if calculation:
        bounds = list(target["bounds"])
        bounds[:2] = previous["bounds"][:2]
        if previous.get("limit_y"):
            bounds[2:] = previous["bounds"][2:]
        changed = (bounds[:2] != target["bounds"][:2] or previous.get("limit_y", False) != target.get("limit_y", False)
                   or previous.get("limit_y") and bounds[2:] != target["bounds"][2:])
        target["bounds"], target["limit_y"] = bounds, previous.get("limit_y", False)
        if changed:
            target["pending"] = target["_recalculate_on_open"] = True
    if display:
        target["display_bounds"] = list(previous["display_bounds"])
    return target


def history_path(store, key, p2p, datasets):
    from core.drr_comparison_groups import digest
    signature = digest([p2p["bounds"], p2p.get("limit_y"), p2p.get("smoothing"), sorted(dataset.key for dataset in datasets)])
    low, high = p2p["bounds"][:2]
    smoothing = p2p.get("smoothing", {}).get("method", "Off")
    return store.root / "history" / digest(key or "manual") / f"{low:.4f}-{high:.4f}eV_{smoothing}_{signature[:16]}.npz"


def persist_comparison(store, key, snapshot):
    from core.drr_workspace_session import save_session, load_session
    datasets, views, active, display = snapshot
    save_session(store.workspace(key or "manual"), datasets, views, active, display, compressed=False)
    for dataset in datasets:
        part = deepcopy({name: value for name, value in display.items() if name != "p2p"})
        p2p = display.get("p2p")
        if p2p:
            part["p2p"] = {**p2p, "records": [record for record in p2p.get("records", []) if record["dataset_key"] == dataset.key]}
        if (not part.get("p2p") or not part["p2p"]["records"]) and store.dataset(dataset.key).exists():
            previous = load_session(store.dataset(dataset.key), with_display=True)[3].get("p2p")
            if previous:
                part["p2p"] = previous
        part["result_filters"] = {dataset.key: display["result_filters"][dataset.key]} if dataset.key in display.get("result_filters", {}) else {}
        save_session(store.dataset(dataset.key), [dataset], {dataset.key: views[dataset.key]} if dataset.key in views else {}, dataset.key, part, compressed=False)
    p2p = display.get("p2p")
    if p2p and not p2p.get("pending"):
        save_session(history_path(store, key, p2p, datasets), datasets, views, active, display)


def load_comparison(store, key, group, entries, previous=None, old_key=None, *, cancellation=None, progress=lambda value: None):
    from core.drr_workspace_session import load_session
    from core.drr_processed_groups import load_group
    from core.drr_comparison_groups import product_id
    from core.load_cancellation import load_scope, check_load_cancelled
    with load_scope(cancellation):
        paths = store.members(key, group["members"])
        paths.sort(key=lambda path: ((entries.get(path, {}).get("condition") or {}).get("field", float("inf")), path))
        if not paths:
            raise ValueError("This comparison group has no included members.")
        versions = {product_id(path): source_version(entries.get(path, {}).get("path", path)) for path in paths}
        if previous and previous[0]:
            persist_comparison(store, old_key, previous)
        saved = load_session(store.workspace(key), with_display=True) if store.workspace(key).exists() else None
        by_path = {product_id(dataset.provenance.get("metadata_path", "")): dataset for dataset in saved[0]} if saved else {}
        old_versions = saved[3].get("source_versions", {}) if saved else {}
        datasets, views, filters, p2p_states = [], {}, {}, []
        for index, path in enumerate(paths):
            check_load_cancelled()
            pid = product_id(path)
            if pid in by_path and versions[pid] is not None and old_versions.get(pid) == versions[pid]:
                dataset = by_path[pid]
                datasets.append(dataset)
                if dataset.key in saved[3].get("result_filters", {}):
                    filters[dataset.key] = saved[3]["result_filters"][dataset.key]
            else:
                dataset = load_group(entries.get(path, {}).get("path", path))
                cache = store.dataset(dataset.key)
                if cache.exists():
                    prior, cached_views, _, display = load_session(cache, with_display=True)
                    dataset = prior[0]
                    views.update(cached_views)
                    filters.update(display.get("result_filters", {}))
                    if display.get("p2p"):
                        p2p_states.append(display["p2p"])
                datasets.append(dataset)
            progress(int(100 * (index + 1) / len(paths)))
        display = deepcopy(saved[3]) if saved else {}
        display["result_filters"], display["source_versions"] = filters, versions
        state = display.get("p2p")
        if state:
            def definition(value):
                return (value["bounds"][:2], value.get("limit_y"), value["bounds"][2:] if value.get("limit_y") else None, value.get("smoothing"))
            records = {record["dataset_key"]: record for record in state.get("records", [])}
            for cached in p2p_states:
                if definition(cached) == definition(state) and not cached.get("pending"):
                    for record in cached["records"]:
                        records.setdefault(record["dataset_key"], record)
            state["records"] = [records[dataset.key] for dataset in datasets if dataset.key in records]
        if saved:
            views.update({name: value for name, value in saved[1].items() if name in {dataset.key for dataset in datasets}})
        return datasets, views, saved[2] if saved else datasets[0].key, display


def analyze(datasets, *, automatic=True, cancellation=None, progress=lambda value: None):
    from core.drr_peak_analysis import analyze_drr_peaks
    output = []
    for index, dataset in enumerate(datasets):
        if cancellation is not None and cancellation.cancelled:
            break
        try:
            result = analyze_drr_peaks(dataset.cube, dataset.settings,
                cancelled=(lambda: cancellation.cancelled) if cancellation is not None else None,
                progress=lambda value, index=index: progress(int((index * 100 + value) / len(datasets))),
                retain_candidates=dataset.settings.mode == "all", auto_candidates=automatic and dataset.settings.mode == "all")
            result.update(dataset_id=dataset.key, dataset_name=dataset.name,
                dataset_revision=dataset.revision, provenance=deepcopy(dataset.provenance))
            output.append((dataset.key, dataset.revision, result, ""))
        except Exception as exc:
            output.append((dataset.key, dataset.revision, None, str(exc)))
    return output


def load_measurements(paths, recipe, *, cancellation=None, progress=lambda value: None):
    from core.drr_analysis_workspace import load_dataset
    from core.load_cancellation import load_scope
    output = []
    with load_scope(cancellation):
        for index, path in enumerate(paths):
            if cancellation is not None and cancellation.cancelled:
                break
            try:
                dataset = load_dataset(path, recipe["backgrounds"], recipe["mode"], recipe["frames"])
                output.append((str(path), dataset, ""))
            except Exception as exc:
                output.append((str(path), None, str(exc)))
            progress(int(100 * (index + 1) / len(paths)))
    return output


def export_peaks(folder, datasets, filters, *, summary=False):
    from core.drr_analysis_workspace import export_dataset, export_summary
    from core.drr_result_filter import filtered_result
    snapshots = [copy_dataset(dataset) for dataset in datasets]
    for dataset in snapshots:
        if dataset.result is None:
            raise ValueError("Analyze every selected dataset before export; stale results are not exported.")
        dataset.result = filtered_result(dataset.result, filters.get(dataset.key, {}))
    return [str(export_summary(folder, snapshots))] if summary else [str(path) for dataset in snapshots for path in export_dataset(folder, dataset).values()]


__all__ = ["catalog", "source_version", "store_for", "copy_dataset", "carry_p2p_ranges", "history_path",
           "persist_comparison", "load_comparison", "analyze", "load_measurements", "export_peaks"]
