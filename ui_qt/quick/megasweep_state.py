"""Small, dependency-free defaults for Megasweep's separate mode workspaces."""
from core.megasweep import DEFAULT_EXPORT_DPI, LEGACY_SCALE_WINDOWS

DEFAULTS = dict(view='intensity', energy_min=1.247, energy_max=1.428, energy_target=1.3375,
    baseline=595.0, auto_baseline=True, sg_window=31, sg_poly=3, ratio=1.0, tg_is_y=True,
    convention='TG+rBG', transformed=False, cut_type='doping', cut_coordinates='explicit', cut_axis='first', batch_axes='selected', cut_value=0.0,
    epsilon=0.03, min_points=2, exact_coordinates=False, derivative='spectrum',
    derivative_window=9, background_role='custom', background_average='all_frames', background_scale=1.0, auto_background_scale=False, background_scale_frame=None, background_scale_x=0.0, background_scale_y=0.0, auto_scale_windows=True, background_scale_windows=list(LEGACY_SCALE_WINDOWS),
    x_column='', y_column='', channel='', direction=None, pass_index=None,
    rc_feature='auto', auto_update=True, _display_states={}, export_dpi=DEFAULT_EXPORT_DPI, cmap='RdBu_r', color_auto=True, vmin=0.0, vmax=1.0, x_auto=True, x_reverse=False, xmin=0.0, xmax=1.0, y_auto=True, y_reverse=False, ymin=0.0, ymax=1.0, movie_options={}, movie_ranges={}, movie_colors={})

def restore_export_resolution(values):
    """Older desktop recipes persisted the old default for each map quantity."""
    for display in (values, *values['_display_states'].values()):
        if display.get('export_dpi') == 300:
            display['export_dpi'] = DEFAULT_EXPORT_DPI


def new_state(mode):
    from copy import deepcopy
    values = deepcopy(DEFAULTS)
    if mode == 'Transport': values['view'] = 'transport_map'
    if mode == 'Reflection': values.update(view='rc_p2p', auto_energy_window=True, auto_energy_target=True, last_map_view='rc_p2p')
    return dict(values=values, source='', backgrounds=[], background_groups={role: [] for role in ('custom', 'p', 'n', 'bn')}, picker_folder='', experiment_root='', recent_source='', background_folder='', background_folders={role: '' for role in ('custom', 'p', 'n', 'bn')})
