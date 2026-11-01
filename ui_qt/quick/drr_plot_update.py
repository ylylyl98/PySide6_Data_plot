"""Update compatible DRR products without rebuilding axes and colorbars."""
from __future__ import annotations


def reuse_drr_plot(owner):
    import numpy as np
    from core.processing import nearest_gate_spectrum
    from ui_qt.quick.heatmap_style import update_heatmap_colors
    from ui_qt.quick.spectrum_limits import fit_spectrum_y

    if owner._workflow != "drr" or owner._drr_products is None or not owner._maps:
        return False
    view = owner._drr_processing["drrView"]
    names = ("raw", "second") if view == "pair" else (view,)
    if owner._drr_products.second is None:
        names = ("raw",)
    if (any(len(items) != len(names) for items in
            (owner._maps, owner._spectra, owner._captions, owner._plot_cubes, owner._manual_artists))
            or getattr(owner._canvas.figure, "_dptk_quick_colors", None) != owner.colors):
        return False

    def geometry(params):
        split = params.split_scale
        return (params.xlabel, params.ylabel, params.cbar_label, params.xlim, params.ylim,
                params.y_axis_log, params.right_axis,
                None if split is None else (split.split_x, split.split_x2, split.show_boundary))

    updates = []
    # Check every paired product before updating any visible artist.
    for index, name in enumerate(names):
        axis = owner._maps[index][0]
        record = getattr(axis, "_dptk_quick_render", None)
        if record is None or record[2] != name:
            return False
        render, previous, _ = record
        cube = owner._drr_products.second if name == "second" else owner._cube
        params = owner._drr_display.heatmap_params(cube, name)
        header = render.colorbar_host
        if (geometry(params) != geometry(axis._dptk_heatmap_params)
                or cube.Z.shape != previous.Z.shape
                or not np.array_equal(cube.energy, previous.energy)
                or not np.array_equal(cube.gate, previous.gate)
                or any(image.axes is not axis for image in render.images)
                or header is None or not hasattr(header, "_dptk_header_title")):
            return False
        updates.append((axis, render, previous, cube, params, name))
    if all(previous is cube for _, _, previous, cube, _, _ in updates):
        return False

    gate, _ = nearest_gate_spectrum(owner._cube, owner._values["gate"])
    owner._values["gate"] = gate
    for index, (axis, render, previous, cube, params, name) in enumerate(updates):
        update_heatmap_colors(render, cube, params)
        axis._dptk_quick_render = (render, cube, name)
        axis._dptk_heatmap_params = params
        title = ("ΔR/R" if name == "raw" else "Derivative") if len(names) > 1 else params.title
        render.colorbar_host._dptk_header_title = title
        axis.title.set_text(title)
        axis.set_xlim(params.xlim)
        axis.set_ylim(params.ylim)
        selected_gate, spectrum = nearest_gate_spectrum(cube, gate)
        owner._maps[index][1].set_ydata([selected_gate, selected_gate])
        spectrum_axis, line = owner._spectra[index]
        line.set_data(cube.energy, spectrum)
        spectrum_axis.relim()
        if previous is not cube:
            # A new product starts with automatic spectrum Y, as a rebuild does.
            spectrum_axis.set_autoscaley_on(True)
        fit_spectrum_y(spectrum_axis)
        owner._captions[index].set_text(f"Spectrum @ {gate:g}")
        owner._plot_cubes[index] = cube
        _, *artists = owner._manual_artists[index]
        owner._manual_artists[index] = (cube, *artists)
    owner._pl_draw_overlays(draw=False)
    owner._navigation.update()
    # Invalidate the retained blit backgrounds and schedule one complete frame.
    owner._apply_split()
    owner._plot_dirty = False
    owner._publish()
    return True
