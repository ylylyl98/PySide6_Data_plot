"""Reuse DRR artists only when source, layout and mesh coordinates agree."""
from copy import deepcopy
import numpy as np
from core.plotting import (HeatmapParams, _norm_from_params, _sanitize_limits,
                           _axis_edges_from_centers, split_region_slices, split_color_layers,
                           heatmap_meshes)


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
        else {'raw': 'None', 'second': ('dXdY' if owner.drr_second_kind_combo.currentData() == 'mixed' else 'd2E')}.get(owner._drr_plot_view, 'None'))
    return products


def structure(params):
    # Linear maps retain their full grids, including split-region meshes.
    # _layers validates that unchanged boundaries still fit the new X range
    # before any artists are touched. Log-axis layouts retain the fallback.
    limits = (params.xlim, params.ylim) if params.y_axis_log else None
    return (params.title, params.xlabel, params.ylabel, params.cbar_label,
            limits, params.log_scale, params.y_axis_log, params.right_axis)


def region_layout(params):
    split = params.split_scale
    return None if split is None else (split.split_x2 is not None, split.show_boundary)


def _color_geometry(params):
    split = params.split_scale
    return (params.xlim, params.ylim, region_layout(params),
            None if split is None else (split.split_x, split.split_x2))


def _replace_regions(owner, render, cube, params, *, paired, key):
    """Keep map/linecut axes and measured right axes when region count changes."""
    axis, host = render.primary.axes, render.colorbar_host
    inspection_header = bool(getattr(host, '_dptk_inspection_header', False))
    for mesh in render.images:
        if mesh.colorbar is not None:
            mesh.callbacks.disconnect(mesh.colorbar_cid)
            mesh.colorbar = None
        mesh.remove()
    for line in render.boundary_artists:
        line.remove()
    for child in tuple(host.child_axes):
        child.remove()
    host.clear()
    host.set_axis_on()
    # A single-region colorbar attaches itself to the host. Release that old
    # mappable as well when moving to inset colorbars.
    if hasattr(host, '_colorbar'):
        del host._colorbar
    updated = heatmap_meshes(axis, cube, params)
    for mesh in updated.images:
        mesh._dptk_drr_mesh = True
    if inspection_header:
        from ui_qt.inspection_colorbar import add_inspection_colorbar
        add_inspection_colorbar(owner.figure, updated, axis,
                                label=params.cbar_label, host=host)
    else:
        owner._add_heatmap_colorbar(updated, host, label='' if paired else params.cbar_label)
    if not inspection_header and paired and key == 'second':
        from matplotlib.ticker import ScalarFormatter
        if isinstance(host.yaxis.get_major_formatter(), ScalarFormatter):
            host.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useMathText=True)
    return updated


def _move_boundaries(render, cube, params, layers):
    """Replace only meshes whose cell boundaries moved; retain axes/colorbars."""
    slices, boundaries = split_region_slices(cube.energy, params.split_scale, params.xlim)
    if boundaries == render.boundaries:
        return
    x_edges = _axis_edges_from_centers(cube.energy)
    y_edges = _axis_edges_from_centers(cube.gate, log_scale=params.y_axis_log)
    meshes = []
    for old, region, (values, _norm) in zip(render.images, slices, layers):
        mesh = old.axes.pcolormesh(x_edges[region.start:region.stop + 1], y_edges, values,
                                  shading='flat', cmap=old.get_cmap(), norm=old.norm)
        mesh._dptk_drr_mesh = True
        mesh._format_cursor_data_override = old._format_cursor_data_override
        colorbar = old.colorbar
        if colorbar is not None:
            old.callbacks.disconnect(old.colorbar_cid)
            old.colorbar = None
            mesh.colorbar = colorbar
            mesh.colorbar_cid = mesh.callbacks.connect('changed', colorbar.update_normal)
            colorbar.update_normal(mesh)
        old.remove()
        meshes.append(mesh)
    render.primary, render.secondary = meshes[:2]
    render.tertiary = meshes[2] if len(meshes) == 3 else None
    render.split_x = boundaries[0]
    render.split_x2 = boundaries[1] if len(boundaries) == 2 else None
    for line, boundary in zip(render.boundary_artists, boundaries):
        line.set_xdata([boundary, boundary])
    if getattr(render.colorbar_host, '_dptk_inspection_header', False):
        # Header bars keep compact regional endpoint labels rather than the
        # vertical colorbars' boundary titles.
        return
    if len(meshes) == 2:
        titles = (f'x ≤ {boundaries[0]:.6g}', f'x ≥ {boundaries[0]:.6g}')
        location = 'center'
    else:
        x1, x2 = boundaries
        titles = (f'L: x ≤ {x1:.5g}', f'M: {x1:.5g} to {x2:.5g}', f'R: x ≥ {x2:.5g}')
        location = 'left'
    for mesh, title in zip(meshes, titles):
        if mesh.colorbar is not None:
            axis = mesh.colorbar.ax
            (axis.title if location == 'center' else axis._left_title).set_text(title)


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
    # A coalesced edit may change the gate or active product along with the
    # colors. Those edits must still refresh the linecuts and analysis.
    old_key = getattr(owner, '_last_plot_params_key', None)
    new_key = owner._current_plot_params_key('DRR')
    color_fields = {10, 11, 12, 18, 19, 20, 23, 24, 25, 26, 27, 28}
    only_colors = (old_key is not None and len(old_key) == len(new_key)
                   and all(old == new for index, (old, new) in enumerate(zip(old_key, new_key))
                           if index not in color_fields))
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
        changed_layout = region_layout(params) != region_layout(old_params)
        previous_cube = getattr(owner, '_drr_plot_cubes', {}).get(key)
        only_colors = (only_colors and previous_cube is not None
                       and cube.Z is previous_cube.Z
                       and _color_geometry(params) == _color_geometry(old_params))
        if changed_layout and (render.colorbar_host is None or render.colorbar_host not in owner.figure.axes):
            return False
        if not changed_layout and len(layers) != len(meshes):
            return False
        updates.append((key, render, layers, params, preview, old_params, changed_layout))
    # A color-only edit may reuse the last complete frame. Capture eligibility
    # before colorbar callbacks replace their internal artists.
    helpers = getattr(owner, '_drr_region_blitters', {})
    color_helper = helpers.get('gate_view') if only_colors else None
    color_token = color_helper.prepare_color_update() if color_helper is not None else None
    if color_token is None:
        for helper in helpers.values():
            helper.invalidate()
    changed_groups = []
    if (len(products) == 2 and any(update[-1] for update in updates)
            and not any(getattr(update[1].colorbar_host, '_dptk_inspection_header', False)
                        for update in updates)):
        grid = updates[0][1].primary.axes.get_subplotspec().get_topmost_subplotspec().get_gridspec()
        split = updates[0][3].split_scale
        right_margin = 125.0 if split is not None and split.split_x2 is not None else 55.0
        grid.update(right=1.0-right_margin/max(1.0, owner.figure.get_figwidth()*owner.figure.dpi))
        # Embedded Qt figures are not registered with pyplot's figure manager,
        # so GridSpec.update alone does not reposition their existing axes.
        for axis in owner.figure.axes:
            spec = axis.get_subplotspec()
            if spec is not None and spec.get_topmost_subplotspec().get_gridspec() is grid:
                axis.set_position(spec.get_position(owner.figure))
    for key, render, layers, params, preview, old_params, changed_layout in updates:
        if changed_layout:
            render = _replace_regions(owner, render, preview, params, paired=len(products) == 2, key=key)
            owner._drr_heatmap_renders[key] = render
        elif params.split_scale is not None:
            _move_boundaries(render, preview, params, layers)
        meshes = render.images
        axis = meshes[0].axes
        viewport = owner._drr_params_with_view_limits(params)
        axis._dptk_heatmap_params = viewport
        xlim = _sanitize_limits(viewport.xlim, float(np.nanmin(preview.energy)), float(np.nanmax(preview.energy)))
        ylim = _sanitize_limits(viewport.ylim, float(np.nanmin(preview.gate)), float(np.nanmax(preview.gate)))
        if not params.y_axis_log:
            if tuple(axis.get_xlim()) != xlim:
                axis.set_xlim(xlim)
            if tuple(axis.get_ylim()) != ylim:
                axis.set_ylim(ylim)
        color_changed = params != old_params
        if color_changed:
            changed_groups.append((axis, render.colorbar_host))
        for mesh, (values, norm) in zip(meshes, layers):
            mesh.set_array(values)
            if type(mesh.norm) is type(norm):
                if (mesh.norm.vmin, mesh.norm.vmax) != (norm.vmin, norm.vmax):
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
            if params.cmap != old_params.cmap:
                mesh.set_cmap(params.cmap)
    owner._drr_plot_cubes = {key: cube for key, cube, *_ in products}
    active = 'second' if owner._drr_plot_view == 'second' and 'second' in records else 'raw'
    owner._drr_heatmap_ax = owner._drr_heatmap_axes[active]
    owner._drr_spectrum_ax = owner._drr_spectrum_axes[active]
    owner._drr_spectrum_line = owner._drr_spectrum_lines[active]
    cube = owner._drr_plot_cubes[active]
    if color_token is not None and changed_groups and color_helper.draw_colors(changed_groups, color_token):
        owner._last_plot_cube = cube
        remember(owner, products)
        return True
    for helper in helpers.values():
        helper.invalidate()
    owner.drr_controller._update_drr_spectrum_and_gate_line(cube)
    owner._last_plot_cube = cube
    remember(owner, products)
    owner.canvas.draw_idle()
    return True
