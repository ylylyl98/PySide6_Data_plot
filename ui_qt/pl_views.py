"""PL scale selection, independent color ranges and linked map layout."""
from dataclasses import replace

from PySide6.QtWidgets import QFrame, QHBoxLayout, QToolButton, QCheckBox, QLabel
from core.plotting import plot_pl, downsample_cube_for_display, resolve_split_boundary
from core.processing import compute_auto_limits


def build_bar(owner):
    bar = QFrame()
    row = QHBoxLayout(bar)
    row.setContentsMargins(8, 4, 8, 4)
    owner._pl_scale_bank = {}
    owner._pl_scale_source = None
    owner._pl_active_scale = False
    owner._pl_heatmap_axes = {}
    owner._pl_view_limits = None
    owner._pl_limits_from_controls = True
    for name, text in (('linear', 'Linear'), ('log', 'Log'), ('side', 'Side by side')):
        button = QToolButton()
        button.setText(text)
        button.setCheckable(True)
        setattr(owner, f'pl_view_{name}_btn', button)
        row.addWidget(button)
    owner.pl_view_linear_btn.clicked.connect(lambda: select_scale(owner, False))
    owner.pl_view_linear_btn.setChecked(True)
    owner.pl_view_log_btn.clicked.connect(lambda: select_scale(owner, True))
    owner.pl_view_side_btn.toggled.connect(lambda _: owner._schedule_plot_redraw('PL') if owner.loaded and owner.loaded.mode == 'PL' else None)
    owner.pl_scale_hint = QLabel('Color controls: Linear')
    row.addWidget(owner.pl_scale_hint)
    row.addStretch(1)
    owner.pl_export_pair_chk = QCheckBox('Include side-by-side PNG')
    row.addWidget(owner.pl_export_pair_chk)
    bar.setVisible(False)
    owner.pl_plot_view_bar = bar
    return bar


def _initialize(owner, cube):
    if owner._pl_scale_source is not cube:
        owner._pl_scale_bank = {}
        owner._pl_scale_source = cube
        owner._pl_view_limits = None
        owner._pl_limits_from_controls = True
    for log in (False, True):
        if log not in owner._pl_scale_bank:
            params = owner._make_params('PL', cube)
            try:
                limits = compute_auto_limits(cube, log_scale=log, xlim=params.xlim, ylim=params.ylim)
                params = replace(params, vmin=limits.vmin, vmax=limits.vmax)
            except ValueError:
                pass
            owner._pl_scale_bank[log] = dict(params=replace(params, log_scale=log, split_scale=None),
                fixed=(False, False), roi=(params.xlim, params.ylim), split_values=None,
                split_enabled=owner.pl_split_scale_chk.isChecked(),
                split_fixed={k: False for k in owner.pl_split_fix_checks})


def _visible_split(params, cube):
    """A viewport entirely within one region keeps that region's normalization."""
    split = params.split_scale
    if split is None:
        return params
    _, boundary = resolve_split_boundary(cube.energy, split.split_x)
    lo, hi = sorted(params.xlim)
    if hi <= boundary or lo >= boundary:
        side = 'left' if hi <= boundary else 'right'
        return replace(params, split_scale=None, vmin=getattr(split, side + '_vmin'),
                       vmax=getattr(split, side + '_vmax'))
    return replace(params, split_scale=replace(split, split_x=boundary))


def _refresh_saved_split(params, cube, fixed):
    split = params.split_scale
    if split is None:
        return params
    _, boundary = resolve_split_boundary(cube.energy, split.split_x)
    lo, hi = sorted(params.xlim)
    updates = {}
    for side, xlim in (('left', (lo, min(hi, boundary))), ('right', (max(lo, boundary), hi))):
        if xlim[0] >= xlim[1]:
            continue
        try:
            limits = compute_auto_limits(cube, log_scale=params.log_scale, xlim=xlim, ylim=params.ylim)
        except ValueError:
            continue
        for key in ('vmin', 'vmax'):
            name = side + '_' + key
            if not fixed.get(name, False):
                updates[name] = getattr(limits, key)
    candidate = replace(split, **updates)
    # Preserve a valid previous scale if this ROI contains only one value.
    for side in ('left', 'right'):
        if getattr(candidate, side + '_vmax') <= getattr(candidate, side + '_vmin'):
            updates.pop(side + '_vmin', None)
            updates.pop(side + '_vmax', None)
    return replace(params, split_scale=replace(split, **updates))


def _remember(owner, cube):
    _initialize(owner, cube)
    params = owner._make_params('PL', cube)
    log = owner._pl_active_scale
    owner._pl_scale_bank[log] = dict(params=replace(params, log_scale=log),
        fixed=tuple(owner.pl_fix_checks[k].isChecked() for k in ('vmin', 'vmax')),
        roi=(params.xlim, params.ylim),
        split_values={k: s.value() for k, s in owner.pl_split_spins.items()},
        split_enabled=owner.pl_split_scale_chk.isChecked(),
        split_fixed={k: s.isChecked() for k, s in owner.pl_split_fix_checks.items()})


def select_scale(owner, log):
    log = bool(log)
    loaded = getattr(owner, 'loaded', None)
    cube = loaded.cube if loaded and loaded.mode == 'PL' else None
    changed = log != owner._pl_active_scale
    if cube is not None and changed:
        # A legacy checkbox signal arrives after changing the widget. Snapshot
        # and validate the old scale before applying the requested log mode.
        blocked = owner.pl_log_chk.blockSignals(True)
        owner.pl_log_chk.setChecked(owner._pl_active_scale)
        owner.pl_log_chk.blockSignals(blocked)
        parameters(owner, cube)
    owner._pl_active_scale = log
    blocked = owner.pl_log_chk.blockSignals(True)
    owner.pl_log_chk.setChecked(log)
    owner.pl_log_chk.blockSignals(blocked)
    owner.pl_view_linear_btn.setChecked(not log)
    owner.pl_view_log_btn.setChecked(log)
    owner.pl_scale_hint.setText('Color controls: Log' if log else 'Color controls: Linear')
    if cube is not None and changed:
        entry = owner._pl_scale_bank[log]
        for widget, checked in [(owner.pl_split_scale_chk, entry['split_enabled']), *[
                (owner.pl_split_fix_checks[key], value) for key, value in entry['split_fixed'].items()]]:
            blocked = widget.blockSignals(True)
            widget.setChecked(checked)
            widget.blockSignals(blocked)
        for key in ('vmin', 'vmax'):
            owner._set_spin_value_silent(owner.pl_spins[key], getattr(entry['params'], key))
        for key, fixed in zip(('vmin', 'vmax'), entry['fixed']):
            widget = owner.pl_fix_checks[key]
            blocked = widget.blockSignals(True)
            widget.setChecked(fixed)
            widget.blockSignals(blocked)
        if entry['split_values'] is not None:
            for key, value in entry['split_values'].items():
                owner._set_spin_value_silent(owner.pl_split_spins[key], value)
            lo, hi = sorted(owner.pl_spins[k].value() for k in ('xmin', 'xmax'))
            if entry['split_enabled'] and not lo < owner.pl_split_spins['x0'].value() < hi:
                owner._refresh_automatic_ranges('PL', refresh_split=True)
        else:
            owner._refresh_automatic_ranges('PL', refresh_split=True)
        owner._update_action_states()
    if cube is not None:
        owner._invalidate_export_move_sources()
        owner._schedule_plot_redraw('PL')


def parameters(owner, cube):
    _remember(owner, cube)
    base = owner._make_params('PL', cube)
    result = {}
    for log, name in ((False, 'linear'), (True, 'log')):
        entry = owner._pl_scale_bank[log]
        saved = entry['params']
        if entry['roi'] != (base.xlim, base.ylim):
            try:
                limits = compute_auto_limits(cube, log_scale=log, xlim=base.xlim, ylim=base.ylim)
                saved = replace(saved, **{key: getattr(limits, key) for key, fixed in
                    zip(('vmin', 'vmax'), entry['fixed']) if not fixed})
            except ValueError:
                pass
            saved = _refresh_saved_split(replace(saved, xlim=base.xlim, ylim=base.ylim), cube, entry['split_fixed'])
            entry.update(params=saved, roi=(base.xlim, base.ylim))
            if saved.split_scale is not None and entry['split_values'] is not None:
                for side in ('left', 'right'):
                    for key in ('vmin', 'vmax'):
                        bound_key = side + '_' + key
                        entry['split_values'][bound_key] = getattr(saved.split_scale, bound_key)
        result[name] = _visible_split(replace(base, log_scale=log, vmin=saved.vmin, vmax=saved.vmax,
            split_scale=saved.split_scale), cube)
    return result


def draw(owner, cube):
    params = parameters(owner, cube)
    both = owner.pl_view_side_btn.isChecked()
    names = ('linear', 'log') if both else ('log' if owner.pl_log_chk.isChecked() else 'linear',)
    grid = owner.figure.add_gridspec(2, len(names), height_ratios=[1.25, 1], hspace=.4, wspace=.36)
    owner._pl_heatmap_axes = {}
    first = None
    preview = downsample_cube_for_display(cube)
    for column, name in enumerate(names):
        panel_grid = grid[0, column].subgridspec(1, 2, width_ratios=[1, .035], wspace=.12)
        ax = owner.figure.add_subplot(panel_grid[0, 0], sharex=first, sharey=first)
        cax = owner.figure.add_subplot(panel_grid[0, 1])
        if first is None:
            first = ax
        owner._pl_heatmap_axes[name] = ax
        panel = params[name]
        render = plot_pl(ax, preview, replace(panel, title=f'{panel.title} ({name.title()})'))
        owner._add_heatmap_colorbar(render, cax, label=panel.cbar_label)
        if column:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
    owner._pl_heatmap_ax = first
    owner._pl_spectrum_ax = owner.figure.add_subplot(grid[1, :], sharex=first)
    owner._pl_last_plot_cube = cube
    owner._pl_gate_line = None
    owner._pl_extra_gate_lines = []
    for helper in getattr(owner.pl_controller, '_pl_region_blitters', None) or ():
        helper.disconnect()
    owner.pl_controller._pl_region_blitters = None
    owner._pl_heatmap_peak_artist = None
    owner._pl_heatmap_fit_artist = None
    owner.pl_controller._update_pl_spectrum_and_gate_line(cube)
    if owner._pl_view_limits is not None:
        first.set_xlim(owner._pl_view_limits[0])
        first.set_ylim(owner._pl_view_limits[1])
    owner._pl_limits_from_controls = False


def capture_view(owner):
    if getattr(owner, '_pl_limits_from_controls', True):
        owner._pl_view_limits = None
        return
    ax = getattr(owner, '_pl_heatmap_ax', None)
    if owner.last_plotted_mode == 'PL' and ax in owner.figure.axes:
        owner._pl_view_limits = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))


def export_parameters(owner, cube):
    capture_view(owner)
    params = parameters(owner, cube)
    if owner._pl_view_limits is not None:
        params = {key: _visible_split(replace(value, xlim=owner._pl_view_limits[0], ylim=owner._pl_view_limits[1]), cube)
                  for key, value in params.items()}
    return params
