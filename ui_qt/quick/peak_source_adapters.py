"""Capture only currently rendered scientific products for standalone analysis.

This module does not load data, transform intensity, or differentiate a DRR
product. Display normalization and source coordinates travel independently.
"""
from __future__ import annotations

import json
from pathlib import Path


def _pending(owner):
    timer = getattr(owner, "_plot_timer", None)
    return bool(getattr(owner, "_closed", False) or getattr(owner, "_busy", False)
        or timer is not None and timer.isActive())


def _view(axis):
    from core.peak_display import capture_display
    return dict(xlim=list(axis.get_xlim()), ylim=list(axis.get_ylim()),
                y_axis_log=axis.get_yscale() == "log", display=capture_display(axis))


def _row(cube, values, selected=None):
    import numpy as np
    if selected is not None and 0 <= int(selected) < len(cube.gate):
        return int(selected)
    return int(np.argmin(abs(np.asarray(cube.gate) - float(values.get("gate", cube.gate[0])))))


def _identity(mode, folder, files):
    return json.dumps([mode, str(folder), files], ensure_ascii=False, sort_keys=True, default=str)


def _root_sources(owner, mode):
    if _pending(owner) or getattr(owner, "_cube", None) is None:
        return []
    maps, cubes = getattr(owner, "_maps", []), getattr(owner, "_plot_cubes", [])
    if not maps or len(maps) != len(cubes):
        return []
    values = owner._values
    primary = str(getattr(owner, "_source", "") or cubes[0].title)
    files = tuple(getattr(owner, "_sources", ()))
    source = _identity("DRR" if mode == "drr" else "PL", primary, files)
    name = Path(primary).stem
    if mode == "pl":
        # A log color map still supplies the original linear intensity grid.
        # Both linear/log panels receive the same accepted source overlay.
        if any(cube is not owner._cube for cube in cubes):
            return []
        index = int(values.get("scale") == "log") if len(maps) > 1 else 0
        axis = maps[min(index, len(maps) - 1)][0]
        return [dict(cube=owner._cube, kind="PL", channel="PL", source=source, name="PL · " + name,
            axes=[entry[0] for entry in maps], view=_view(axis), row=_row(owner._cube, values), mode="PL")]
    products = getattr(owner, "_drr_products", None)
    if products is None:
        return []
    request = products.request
    result = []
    for (axis, _line), cube in zip(maps, cubes):
        channel = "second" if cube is products.second else "raw" if cube is products.raw else None
        if channel is None:
            return []
        derivative = request.second_derivative if channel == "second" else request.raw_derivative
        mixed = isinstance(derivative, tuple) and derivative and derivative[0] == "mixed"
        label = "Mixed derivative" if mixed else "2nd derivative" if derivative == 2 else "1st derivative" if derivative == 1 else "ΔR/R"
        descriptor = dict(cube=cube, group_cube=products.source, kind="DRR", channel=channel,
            source=source, name=f"DRR · {channel} · {name}", source_name="DRR · " + name,
            product_label=label, axes=[axis], view=_view(axis), row=_row(cube, values), mode="DRR")
        if derivative is not None:
            descriptor["source_processing"] = dict(method="Savitzky–Golay", derivative="mixed" if mixed else derivative,
                window_samples=int(products.second_window if channel == "second" else products.raw_window), polyorder=int(request.polyorder))
            if mixed:
                from core.drr_mixed_derivative import effective_window
                descriptor["source_processing"]["scan_window_samples"] = effective_window(derivative[1], len(cube.gate), request.polyorder)
        result.append(descriptor)
    return result


def _compare_sources(child):
    if _pending(child) or child._values.get("view") == "vp" or getattr(child, "_error", ""):
        return []
    derived, maps = child._derived, child._maps
    if not derived or set(derived) != set(maps):
        return []
    active = child._active_keys()
    loaded = dict(child._loaded_mapping)
    if len(active) < 2 or not all(key in child._cubes and loaded.get(key) == child._assignments.get(key) for key in active):
        return []
    source = _identity("Compare", child._folder, child._loaded_mapping)
    return [dict(cube=cube, kind="PL", channel=channel, source=source,
        name=f"Compare · {channel} · {Path(child._assignments.get(channel, cube.title)).stem}",
        axes=[maps[channel]], view=_view(maps[channel]), row=_row(cube, child._values), mode="Compare")
        for channel, cube in derived.items() if channel != "VP"]


def _power_sources(child):
    if (_pending(child) or getattr(child, "_stage", "") != "plot" or child._values.get("mode") == "vp"
            or getattr(child, "_running", set()) & {"load", "products", "review", "save", "export"}):
        return []
    products, maps = child._products, child._maps
    if products is None or products.get("mode", child._values["mode"]) != child._values["mode"]:
        return []
    cubes = products.get("cubes", {})
    if not cubes or set(cubes) != set(maps):
        return []
    source = _identity("Power Dependent", child._folder, child._active_selection)
    selected_rows = getattr(child, "_analysis_selected_rows", {})
    out = []
    for channel, cube in cubes.items():
        if channel == "VP":
            continue
        coordinates = getattr(child, "_display_coordinates", {}).get(channel)
        if coordinates is None:
            return []
        import numpy as np
        true, display, indices = [np.array(value, copy=True) for value in coordinates]
        order = np.argsort(true)
        def transform_y(values, original=true[order], shown=display[order]):
            return np.interp(values, original, shown, left=np.nan, right=np.nan)
        # Keep repeated physical powers unchanged. create_dataset explicitly
        # rejects them with the existing separate/aggregate-repeats message;
        # plotting jitter must never leak into scientific coordinates/exports.
        selected = selected_rows.get(channel, getattr(child, "_selected_row_index", None))
        out.append(dict(cube=cube, kind="PL", channel=channel, source=source,
            name=f"Power Dependent · {channel} · {cube.title}", axes=[maps[channel]],
            view=_view(maps[channel]), row=_row(cube, child._values, selected), mode="Power Dependent", transform_y=transform_y))
    return out


def capture_peak_sources(application_bridge):
    """Return descriptors only after the active product's axes were published."""
    owner = application_bridge
    mode = getattr(owner, "_workflow", "")
    if getattr(owner, "_closed", False) or getattr(owner, "_busy", False):
        return []
    if mode in {"pl", "drr"}:
        return _root_sources(owner, mode)
    child = getattr(owner, "_workflow_bridges", {}).get(mode)
    if child is None:
        return []
    if mode == "cmp":
        return _compare_sources(child)
    if mode == "power":
        return _power_sources(child)
    return []


__all__ = ["capture_peak_sources"]
