"""Normal source-export heatmaps with Peak Analysis overlays."""
from dataclasses import replace

import numpy as np

from core.plotting import HeatmapParams, RightAxis, SplitColorScale


def _params_from_dict(values):
    values = dict(values)
    if values.get('split_scale'):
        values['split_scale'] = SplitColorScale(**values['split_scale'])
    if values.get('right_axis'):
        values['right_axis'] = RightAxis(**values['right_axis'])
    return HeatmapParams(**values)


def _fallback_params(dataset):
    cube = dataset['cube']
    view = dataset.get('view') or {}
    display = dict(view.get('display') or {})
    if not all(key in display and display[key] is not None for key in ('cmap', 'vmin', 'vmax')):
        raise ValueError('The source heatmap color settings are unavailable. Reopen this dataset from the main window to capture its displayed colors before saving.')
    ymin, ymax = float(np.min(cube.gate)), float(np.max(cube.gate))
    if ymin == ymax:
        ymin, ymax = ymin - .5, ymax + .5
    params = HeatmapParams(cube.title, 'Photon Energy (eV)', cube.gate_label, cube.cbar_label,
                           float(display['vmin']), float(display['vmax']), tuple(view.get('xlim') or (float(np.min(cube.energy)), float(np.max(cube.energy)))),
                           tuple(view.get('ylim') or (ymin, ymax)),
                           cmap=display['cmap'],
                           y_axis_log=bool(view.get('y_axis_log')))
    if display.get('split_scale'):
        display['split_scale'] = SplitColorScale(**display['split_scale'])
    return replace(params, **{k: v for k, v in display.items() if k in {
        'cmap', 'vmin', 'vmax', 'log_scale', 'center_zero', 'clip_outliers', 'split_scale'}})


def source_export_width(dataset):
    """Return the selected main-window export width, or 1 for older data."""
    spec = dataset.get('source_export') or {}
    return float((spec.get('params') or {}).get('png_width_scale', 1.0))


def render_source_figure(dataset, *, width_scale=None):
    """Build a normal export figure with analysis markers; caller owns figure."""
    from core.export import _build_streamlit_style_heatmap_fig
    from core.peak_plotting import draw_overlay
    from core.colormaps import register_colormaps

    spec = dataset.get('source_export') or {}
    params = _params_from_dict(spec['params']) if spec.get('params') else _fallback_params(dataset)
    if width_scale is not None:
        params = replace(params, png_width_scale=float(width_scale))
    drr = bool(spec.get('drr', dataset.get('kind') == 'DRR'))
    cube = dataset['cube']
    if params.y_axis_log:
        positive = np.isfinite(cube.gate) & (np.asarray(cube.gate) > 0)
        if not positive.any():
            raise ValueError('Log power export needs positive power points.')
        if not positive.all():
            cube = replace(cube, gate=np.asarray(cube.gate)[positive], Z=np.asarray(cube.Z)[positive])
    register_colormaps()
    fig = _build_streamlit_style_heatmap_fig(cube, params, drr=drr, png_title=spec.get('png_title'))
    if dataset.get('show_overlay', True):
        # Rejected candidates are an analysis preview, not saved results.
        draw_overlay(fig.axes[0], dict(dataset, show_rejected=False))
    return fig
