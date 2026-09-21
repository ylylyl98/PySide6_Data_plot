"""Reuse DRR artists only when source, layout and mesh coordinates agree."""
from copy import deepcopy
import numpy as np
from core.plotting import HeatmapParams, _norm_from_params, _norm_from_bounds, resolve_split_boundary, split_color_layers


def products_for_display(owner):
    raw, _, win, poly = owner.drr_controller._drr_cube_with_metadata(None)
    view = 'both' if owner._drr_side_by_side else owner._drr_plot_view
    advanced = owner.drr_controller._drr_derivative_value()
    products = []
    if view in {'raw', 'both'}:
        cube, derivative = raw, None
        if view == 'raw' and advanced == 1:
            cube, _, win, poly = owner.drr_controller._drr_cube_with_metadata(1)
            derivative = 1
        params = owner._make_drr_params(cube, derivative)
        if derivative == 1:
            params = HeatmapParams(**{**params.__dict__,
                'title': f'{raw.title} (Advanced first derivative, dE)', 'cbar_label': 'd(DR/R)/dE'})
        products.append(('raw', cube, params, derivative, win, poly))
    if view in {'second', 'both'}:
        cube, _, win, poly = owner.drr_controller._drr_cube_with_metadata(2)
        owner._sync_drr_second_auto_scale(cube)
        products.append(('second', cube, owner._make_drr_params(cube, 2), 2, win, poly))
    if not products:
        products.append(('raw', raw, owner._make_drr_params(raw, None), None, win, poly))
    owner.loaded.drr_derivative_label = ('Advanced first derivative (dE)'
        if advanced == 1 and owner._drr_plot_view == 'raw'
        else {'raw': 'None', 'second': 'd2E'}.get(owner._drr_plot_view, 'None'))
    return products


def structure(params):
    return (params.title, params.xlabel, params.ylabel, params.cbar_label,
            params.xlim, params.ylim, params.log_scale, params.y_axis_log,
            None if params.split_scale is None else (params.split_scale.split_x, params.split_scale.split_x2, params.split_scale.show_boundary))


def remember(owner, products):
    records = {}
    for key, cube, params, *_ in products:
        preview = owner.drr_controller._drr_display_preview(cube)
        records[key] = (owner._drr_heatmap_renders[key],
                        np.asarray(preview.energy).copy(), np.asarray(preview.gate).copy(),
                        deepcopy(params))
    owner._drr_reuse_state = (owner.loaded, owner.loaded.cube,
        tuple(owner.figure.get_size_inches()), owner.figure.dpi, records)


def _layers(cube, params):
    z = np.asarray(cube.Z, float)
    split = params.split_scale
    if split is None:
        values = np.minimum(z, params.vmax) if params.clip_outliers else z
        return [(values, _norm_from_params(values, params))]
    return split_color_layers(cube, params)[0]


def try_update(owner):
    state = getattr(owner, '_drr_reuse_state', None)
    if state is None or owner.last_plotted_mode != 'DRR':
        return False
    loaded, source, size, dpi, records = state
    if (owner.loaded is not loaded or owner.loaded.cube is not source
            or tuple(owner.figure.get_size_inches()) != size or owner.figure.dpi != dpi):
        return False
    products = products_for_display(owner)
    if tuple(records) != tuple(product[0] for product in products):
        return False
    updates = []
    for key, cube, params, *_ in products:
        render, energy, gate, old_params = records[key]
        preview = owner.drr_controller._drr_display_preview(cube)
        meshes = list(render.images)
        if (structure(params) != structure(old_params)
                or not np.array_equal(energy, preview.energy)
                or not np.array_equal(gate, preview.gate)
                or any(mesh.axes is not owner._drr_heatmap_axes.get(key)
                       or mesh.axes not in owner.figure.axes or mesh not in mesh.axes.collections for mesh in meshes)):
            return False
        layers = _layers(preview, params)
        if len(layers) != len(meshes):
            return False
        updates.append((meshes, layers, params))
    # Validate all products before changing the first artist. Invalidate blit
    # backgrounds so a gate update cannot restore the old heatmap pixels.
    for helper in getattr(owner, '_drr_region_blitters', {}).values():
        helper.invalidate()
    for meshes, layers, params in updates:
        for mesh, (values, norm) in zip(meshes, layers):
            mesh.set_array(values)
            if type(mesh.norm) is type(norm):
                with mesh.norm.callbacks.blocked():
                    mesh.norm.vmin, mesh.norm.vmax = norm.vmin, norm.vmax
                mesh.changed()
            else:
                colorbar = mesh.colorbar
                formatter = colorbar.formatter if colorbar is not None else None
                locator = colorbar.locator if colorbar is not None else None
                mesh.set_norm(norm)
                if colorbar is not None:
                    colorbar.formatter = formatter
                    colorbar.locator = locator
                    colorbar.update_ticks()
            mesh.set_cmap(params.cmap)
    owner._drr_plot_cubes = {key: cube for key, cube, *_ in products}
    active = 'second' if owner._drr_plot_view == 'second' and 'second' in records else 'raw'
    owner._drr_heatmap_ax = owner._drr_heatmap_axes[active]
    owner._drr_spectrum_ax = owner._drr_spectrum_axes[active]
    owner._drr_spectrum_line = owner._drr_spectrum_lines[active]
    cube = owner._drr_plot_cubes[active]
    owner.drr_controller._update_drr_spectrum_and_gate_line(cube)
    owner._last_plot_cube = cube
    remember(owner, products)
    owner.canvas.draw_idle()
    return True
