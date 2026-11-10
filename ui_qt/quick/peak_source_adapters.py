"""Capture only currently rendered scientific products for standalone analysis.

This module does not load data, transform intensity, or differentiate a DRR
product. Display normalization and source coordinates travel independently.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
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


def _processed_path(owner, mode, derivative=None, products=None, channel="raw", *, prefer_saved=True):
    """Predict the preferred export path without reading or writing files.

    Collision handling or DRR fingerprint reuse may choose a different name.
    """
    primary = getattr(owner, "_source", None)
    folder = getattr(owner, "_folder", None)
    if not primary or not folder:
        return None
    path = Path(primary)
    if path.suffix.casefold() == ".dat" and derivative is None:
        return str(path)
    cube = (getattr(owner, "_cube", None) if mode == "pl" else
            getattr(products, channel, None))
    saved = getattr(cube, "_peak_processed_path", None)
    if saved and prefer_saved:
        return str(saved)
    if mode == "pl":
        return str(Path(folder) / "Processed Data" / "PL" / f"{path.stem}_PL_linear.dat")
    request = products.request
    if not request.files:
        return None
    from core.export import build_drr_export_base
    from ui_qt.quick.drr_workflow import _export_mode_label
    base = build_drr_export_base(request.files[0], len(request.files), _export_mode_label(request),
        derivative if isinstance(derivative, int) else None,
        products.second_window if channel == "second" else products.raw_window,
        request.polyorder, "More correct (regrid)")
    if isinstance(derivative, tuple):
        base += "_dXdY"
    return str(Path(folder) / "Processed Data" / "DRR" / f"{base}.dat")


def _path_is_inferred(owner, mode, derivative=None, products=None, channel="raw"):
    path = getattr(owner, "_source", None)
    if path and Path(path).suffix.casefold() == ".dat" and derivative is None:
        return False
    cube = getattr(owner, "_cube", None) if mode == "pl" else getattr(products, channel, None)
    return not bool(getattr(cube, "_peak_processed_path", None))


def bind_exported_paths(owner, result):
    """Attach successful export paths to the exact products currently displayed."""
    if owner._workflow == "pl":
        pairs = ((owner._cube, result.get("dat")),)
    elif owner._workflow == "drr" and owner._drr_products is not None:
        products = owner._drr_products
        pairs = ((products.raw, result.get("raw_dat")), (products.second, result.get("second_dat")))
    else:
        return
    for cube, path in pairs:
        if cube is not None and path:
            cube._peak_processed_path = str(path)


def capture_source_export(owner, mode, cube, axis, channel="raw"):
    """Capture normal-export settings without rendering or reading data."""
    from core.drr_export_title import background_label, display_title, imported_background_label
    from core.peak_display import capture_display
    from core.plotting import SplitColorScale
    actual = capture_display(axis)
    if not all(key in actual and actual[key] is not None for key in ('cmap', 'vmin', 'vmax')):
        return None
    if mode == "pl":
        if not callable(getattr(owner, "_params", None)):
            return None
        scale = "log" if owner._values.get("scale") == "log" else "linear"
        params = owner._params(scale)
        png_title = None
    else:
        display = getattr(owner, "_drr_display", None)
        if display is None:
            return None
        params = display.heatmap_params(cube, channel)
        request = owner._drr_products.request
        processing = {"background_mode": request.background_mode}
        recipe = getattr(request, "background_recipe", None)
        if recipe is not None:
            processing["drr_background_assignments"] = [item.to_dict() for item in recipe.assignments]
            processing["background_kind"] = getattr(getattr(recipe, "draft", None), "kind", None)
            if getattr(owner._drr_products, "background_references", ()):
                processing["selection_method"] = "fitted_sample"
        background = (imported_background_label(getattr(owner._drr_products.source, "import_metadata", None))
                      if request.background_mode == "imported" else background_label(processing))
        png_title = display_title(params.title, background=background)
    colors = {key: actual[key] for key in ('cmap', 'vmin', 'vmax', 'log_scale', 'center_zero', 'clip_outliers') if key in actual}
    colors['split_scale'] = SplitColorScale(**actual['split_scale']) if actual.get('split_scale') else None
    params = replace(params, **colors)
    width = getattr(owner, "_png_export_width", None)
    params = replace(params, png_width_scale=width() if callable(width) else params.png_width_scale)
    if mode == "drr":
        params = replace(params, xlim=tuple(axis.get_xlim()), ylim=tuple(axis.get_ylim()))
    return dict(params=asdict(params), drr=mode == "drr", png_title=png_title)


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
            axes=[entry[0] for entry in maps], view=_view(axis), row=_row(owner._cube, values), mode="PL",
            processed_path=_processed_path(owner, "pl"),
            processed_path_hint=_processed_path(owner, "pl", prefer_saved=False),
            processed_path_inferred=_path_is_inferred(owner, "pl"),
            source_export=capture_source_export(owner, "pl", owner._cube, axis))]
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
            product_label=label, axes=[axis], view=_view(axis), row=_row(cube, values), mode="DRR",
            processed_path=_processed_path(owner, "drr", derivative, products, channel),
            processed_path_hint=_processed_path(owner, "drr", derivative, products, channel, prefer_saved=False),
            processed_path_inferred=_path_is_inferred(owner, "drr", derivative, products, channel),
            source_export=capture_source_export(owner, "drr", cube, axis, channel))
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
