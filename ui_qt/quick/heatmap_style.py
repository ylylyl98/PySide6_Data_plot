"""Update existing native heatmap colors without resetting their axes."""
from __future__ import annotations


def update_heatmap_colors(render, cube, params):
    """Apply colors to unchanged mesh geometry, using original scientific data.

    Clipped meshes must use the source array again when vmax increases. Reusing
    the previous clipped values would permanently hide the original intensity.
    """
    import numpy as np
    from core.plotting import _norm_from_params, split_color_layers

    if params.split_scale is None:
        values = np.asarray(cube.Z, float)
        if params.clip_outliers:
            values = np.minimum(values, params.vmax)
        layers = [(values, _norm_from_params(values, params))]
    else:
        layers, _ = split_color_layers(cube, params)
    if len(layers) != len(render.images):
        raise ValueError("Color update requires unchanged heatmap regions.")
    for image, (values, norm) in zip(render.images, layers):
        # Matplotlib copies and masks the entire array in set_array. Compare
        # actual values, not source identity: scientific arrays can be edited
        # in place, and clipped values must be restored when vmax increases.
        current = image.get_array()
        if (current is None or np.ma.is_masked(current)
                or not np.array_equal(np.ma.getdata(current), values)):
            image.set_array(values)
        if type(norm) is type(image.norm):
            image.set_clim(norm.vmin, norm.vmax)
        else:
            # Keep the compact inspection bar's endpoint formatter/locator
            # when a VP range starts or stops straddling zero.
            bar = image.colorbar
            locator, formatter = (bar.locator, bar.formatter) if bar else (None, None)
            image.set_norm(norm)
            if bar:
                bar.locator, bar.formatter = locator, formatter
                bar.update_ticks()
        # set_cmap emits a change even for an equivalent registry colormap;
        # on a colorbar this rebuilds its artists for every unchanged region.
        from matplotlib import colormaps
        cmap = colormaps.get_cmap(params.cmap)
        if image.get_cmap() != cmap:
            image.set_cmap(cmap)
