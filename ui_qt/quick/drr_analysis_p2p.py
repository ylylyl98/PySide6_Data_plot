"""Captured P2P calculations and exports for the native analysis workspace.

This module has no Qt dependency; requests contain immutable cube snapshots.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import re

import numpy as np

DEFAULT_STYLE = dict(width=9., height=5., dpi=300, label_size=12, tick_size=10,
    legend_size=9, title_size=12, line_width=1.5, marker_size=3., markers=True,
    grid=True, legend_location="best", legend_columns=1, title=True, subtitle=True,
    field_min=0., field_max=7., offset=False, offset_step=.005)


def copy_state(value):
    """Copy mutable metadata while sharing immutable scientific arrays."""
    if isinstance(value,np.ndarray):
        value.setflags(write=False)
        return value
    if isinstance(value,dict): return {key:copy_state(item) for key,item in value.items()}
    if isinstance(value,list): return [copy_state(item) for item in value]
    if isinstance(value,tuple): return tuple(copy_state(item) for item in value)
    return value


def compute_records(datasets, bounds, limit_y=False, smoothing=None, *, cancellation=None, progress=lambda _: None):
    from core.drr_range_amplitude import range_amplitude, smooth_energy_cube
    if len({dataset.cube.gate_label for dataset in datasets}) > 1:
        raise ValueError("Compare datasets with the same Y axis.")
    records = []
    for index, dataset in enumerate(datasets):
        if cancellation is not None:
            cancellation.check()
        cube = dataset.cube
        used = (*bounds[:2], *(bounds[2:] if limit_y else (float(cube.gate.min()), float(cube.gate.max()))))
        record = dict(dataset_key=dataset.key, bounds=list(used), result=None, raw_result=None,
                      processed_Z=None, edge=None, error="", smoothing=dict(method="Off"))
        try:
            settings = smoothing or dict(method="Off")
            unit = settings.get("unit", "points")
            processed, metadata = (smooth_energy_cube(cube, settings["window_points"] if unit == "points"
                else settings["window_mev"], settings["polyorder"], unit=unit) if settings["method"] == "SG"
                else (cube, dict(method="Off")))
            raw = range_amplitude(cube, *used)
            result = range_amplitude(processed, *used)
            energies = np.sort(cube.energy[(cube.energy >= used[0]) & (cube.energy <= used[1])])
            spacing = float(np.median(np.diff(energies))) if len(energies) > 1 else 0.
            edge = np.isfinite(result[:, 1]) & ((np.minimum(result[:, 4], result[:, 5]) <= energies.min() + spacing)
                | (np.maximum(result[:, 4], result[:, 5]) >= energies.max() - spacing))
            record.update(result=result, raw_result=raw, smoothing=metadata, edge=edge,
                          processed_Z=processed.Z if settings["method"] == "SG" else None)
        except ValueError as error:
            record["error"] = str(error)
        records.append(record)
        progress(int(100 * (index + 1) / len(datasets)))
    return copy_state(records)


def hydrated(records, datasets):
    return [{**record, "dataset": datasets[record["dataset_key"]]} for record in records if record["dataset_key"] in datasets]


def metric_edge_flags(record, dataset, metric):
    from core.drr_p2p_metrics import metric_values
    result = record["result"]
    if metric not in {"maximum", "minimum"}:
        return np.asarray(record["edge"], dtype=bool)
    low, high = record["bounds"][:2]
    energies = np.sort(dataset.cube.energy[(dataset.cube.energy >= low) & (dataset.cube.energy <= high)])
    spacing = float(np.median(np.diff(energies))) if len(energies) > 1 else 0.
    values = metric_values(result, metric)
    return np.isfinite(values) & ((values <= energies.min() + spacing) | (values >= energies.max() - spacing))


def curve_styles(records, datasets, style):
    from matplotlib import colormaps
    from core.drr_p2p_export import field_value
    low, high = style["field_min"], style["field_max"]
    if high <= low:
        raise ValueError("B color maximum must exceed minimum.")
    fields = [field_value(datasets[record["dataset_key"]]) for record in records]
    order = sorted(range(len(records)), key=lambda i: (fields[i] if fields[i] is not None else float("inf"),
                                                      datasets[records[i]["dataset_key"]].name, i))
    seen, totals, styles = Counter(), Counter(fields), [None] * len(fields)
    for rank, index in enumerate(order):
        field = fields[index]
        repeat = seen[field]
        seen[field] += 1
        dataset = datasets[records[index]["dataset_key"]]
        label = f"{field:g} T" if field is not None else dataset.name
        if totals[field] > 1:
            label += f" · #{repeat + 1}"
            count = dataset.provenance.get("saved_processing", {}).get("average_count")
            if count is not None:
                label += f" · avg {count}"
        styles[index] = dict(color=tuple(colormaps["plasma"](.05 + .80 * np.clip((field-low)/(high-low), 0, 1)))
            if field is not None else (.4, .4, .4, 1.), marker=("o", "s", "^", "D", "v", "P", "X")[repeat % 7],
            linestyle=("-", "--", "-.", ":")[(repeat // 7 if repeat >= 7 else repeat) % 4],
            offset=rank * style["offset_step"] if style["offset"] else 0., rank=rank, label=label)
    return styles


def draw_curves(axes, records, datasets, state, *, compare=True):
    from core.drr_p2p_metrics import METRICS, metric_values, quantity_metrics
    style = {**DEFAULT_STYLE, **state.get("plot_style", {})}
    metrics = quantity_metrics(("amplitude", "positions", "separation")[state.get("comparison_metric", 0)] if compare else "amplitude")
    styles = curve_styles(records, datasets, style)
    for panel, (axis, metric) in enumerate(zip(axes, metrics)):
        for index in sorted(range(len(records)), key=lambda i: styles[i]["rank"]):
            record = records[index]
            if record["result"] is None or not compare and record["dataset_key"] != state.get("current_key"):
                continue
            result, curve = record["result"], styles[index]
            values = metric_values(result, metric) + (curve["offset"] if compare and metric == "amplitude" else 0.)
            order = np.argsort(result[:, 0], kind="stable")
            line, = axis.plot(result[order, 0], values[order], color=curve["color"], lw=style["line_width"],
                linestyle=curve["linestyle"], marker=curve["marker"] if style["markers"] else None,
                ms=style["marker_size"], label=curve["label"], picker=6)
            line._drr_p2p = (record["dataset_key"], order, values)
            if metric != "amplitude":
                marked = order[metric_edge_flags(record, datasets[record["dataset_key"]], metric)[order]]
                axis.plot(result[marked, 0], values[marked], linestyle="None", marker=curve["marker"],
                    ms=style["marker_size"]+1, markerfacecolor="white", markeredgecolor=curve["color"])
            if not compare and record.get("smoothing", {}).get("method") == "SG":
                axis.plot(result[order, 0], record["raw_result"][order, 1], "--", color="gray", label="Raw P2P")
        axis.set_ylabel(METRICS[metric][0], fontsize=style["label_size"])
        if metric == "amplitude" and compare and style["offset"]:
            axis.set_ylabel(f"P2P + offset ({style['offset_step']:g} / curve)")
        dataset = datasets.get(state.get("current_key")) or next(iter(datasets.values()))
        axis.set_xlabel(dataset.cube.gate_label, fontsize=style["label_size"])
        axis.tick_params(labelsize=style["tick_size"])
        if style["grid"]:
            axis.grid(True, alpha=.4, lw=.6, ls="--")
        if panel == 0:
            title = []
            if style["title"]:
                title.append(METRICS[metric][0] + " vs " + dataset.cube.gate_label)
            if style["subtitle"]:
                low, high = state["bounds"][:2]
                smoothing = state["smoothing"]
                method = "raw" if smoothing["method"] == "Off" else f"SG {smoothing.get('window_points') if smoothing.get('unit') == 'points' else smoothing.get('window_mev'):g} {smoothing.get('unit')} / {smoothing['polyorder']}"
                title.append(f"{low:.4f}–{high:.4f} eV · {method}")
            axis.set_title("\n".join(title), fontsize=style["title_size"])
            if axis.get_lines():
                axis.legend(fontsize=style["legend_size"], loc=style["legend_location"], ncol=style["legend_columns"], frameon=False)


def comparison_figure(records, datasets, state):
    from matplotlib.figure import Figure
    style = {**DEFAULT_STYLE, **state.get("plot_style", {})}
    figure = Figure(figsize=(style["width"], style["height"]), dpi=style["dpi"], facecolor="white", layout="constrained")
    axes = np.atleast_1d(figure.subplots(2 if state.get("comparison_metric") == 1 else 1, 1, sharex=True))
    draw_curves(axes, records, datasets, state)
    return figure


def write_csv(path, records, datasets):
    from core.drr_p2p_export import field_value
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Dataset", "Y_label", "Y", "P2P", "Min", "Max", "Energy_at_min_eV", "Energy_at_max_eV", "Finite_samples",
            "Energy_min_eV", "Energy_max_eV", "Y_min", "Y_max", "Edge_warning", "Status", "Raw_P2P", "Smoothing", "Window_meV",
            "Polynomial", "Window_points_min", "Window_points_max", "Actual_span_mev_min", "Actual_span_mev_max", "Window_unit", "Window_points",
            "B_T", "Separation_meV", "Emin_edge", "Emax_edge"])
        for record in records:
            dataset = datasets[record["dataset_key"]]
            if record["result"] is None:
                writer.writerow([dataset.name, dataset.cube.gate_label] + [""]*7 + list(record["bounds"]) + ["", record["error"]])
                continue
            low_edge = metric_edge_flags(record, dataset, "minimum")
            high_edge = metric_edge_flags(record, dataset, "maximum")
            for index, (row, edge) in enumerate(zip(record["result"], record["edge"])):
                meta = record["smoothing"]
                raw = record["raw_result"][index, 1]
                writer.writerow([dataset.name, dataset.cube.gate_label] + [float(v) if np.isfinite(v) else "" for v in row]
                    + list(record["bounds"]) + [bool(edge), "ok" if np.isfinite(row[1]) else "insufficient data", float(raw) if np.isfinite(raw) else ""]
                    + [meta.get(name, "") for name in ("method", "window_mev", "polyorder", "window_points_min", "window_points_max",
                        "actual_span_mev_min", "actual_span_mev_max", "unit", "window_points")]
                    + [field_value(dataset), float(abs(row[5]-row[4])*1000) if np.isfinite(row[4:6]).all() else "", bool(low_edge[index]), bool(high_edge[index])])


def export_path(folder, records, datasets, state, extension, comparison_key=None):
    from core.drr_comparison_groups import digest
    def safe(value, limit):
        return re.sub(r'[<>:"/\\|?*\x00-\x20]+', "_", str(value)).strip(" ._")[:limit] or "DRR"
    root = Path(folder)
    if root.name.casefold() == "drr" and root.parent.name.casefold() == "processed data":
        root = root.parent.parent
    elif root.name.casefold() == "processed data":
        root = root.parent
    label = safe(next(iter(datasets.values())).name if len(datasets) == 1 else "Manual datasets", 48)
    identity = comparison_key or digest(sorted(record["dataset_key"] for record in records))
    target = root / "Processed Data" / "DRR Analysis" / f"{label}_{safe(identity, 8)}"
    target.mkdir(parents=True, exist_ok=True)
    low, high = state["bounds"][:2]
    smoothing = state["smoothing"]
    sg = "raw" if smoothing["method"] == "Off" else f"SG{smoothing.get('window_points') if smoothing.get('unit') == 'points' else smoothing.get('window_mev'):g}{smoothing['unit']}_p{smoothing['polyorder']}"
    stem = safe(f"{label}_P2P_{low:.4f}-{high:.4f}eV_{sg}_{('amplitude','positions','separation')[state.get('comparison_metric',0)]}", 155)
    for version in range(1, 100000):
        path = target / f"{stem}{'' if version == 1 else f'_v{version:03d}'}.{extension}"
        try:
            with path.open("xb"):
                pass
            return path
        except FileExistsError:
            continue
    raise OSError("Cannot reserve another export filename.")
