"""Apply remembered DRR displays without triggering intermediate plots."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import wraps
import json
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QCheckBox, QGroupBox, QLabel, QPushButton, QVBoxLayout

from core import drr_display_settings as history
from ui_qt.common import pending_numeric_input
from ui_qt.drr_regions import settings_state, sync_visibility

GLOBAL_KEY = 'drr/display_center_templates'
PREF_PREFIX = 'drr/display_memory/'
GEOMETRY_CONTEXT_KEY = PREF_PREFIX + 'geometry_context'
GLOBAL_GEOMETRY_KEY = PREF_PREFIX + 'locked_geometry'
GEOMETRY_FIXES = ('fix.xmin', 'fix.xmax', 'drr_fix.x0', 'drr_fix.x1',
                  'drr_second_fix.x0', 'drr_second_fix.x1')
GEOMETRY_LAYOUT = ('count', 'split', 'second_split', 'boundary')


def _preference(owner, name, default):
    value = owner.settings.value(PREF_PREFIX + name, default)
    return value if isinstance(value, bool) else str(value).lower() in ('true', '1', 'yes')


def build_controls(owner):
    box = QGroupBox('Remember display settings')
    layout = QVBoxLayout(box)
    layout.setContentsMargins(6, 6, 6, 6)
    for name, text, default in (
        ('files', 'Restore this file\'s saved display', True),
        ('centers', 'Reuse X range / boundaries by center', True),
        ('colors', 'Also reuse color limits by center', False),
        ('global', 'Share center templates across experiments', False),
    ):
        check = QCheckBox(text)
        check.setChecked(_preference(owner, name, default))
        check.toggled.connect(lambda on, key=name: owner.settings.setValue(PREF_PREFIX + key, on))
        setattr(owner, 'drr_memory_' + name + '_chk', check)
        layout.addWidget(check)
    lock = QCheckBox('Lock X range across files')
    lock.setObjectName('drr_memory_lock_x')
    lock.setChecked(_preference(owner, 'lock_x', False))
    lock.setEnabled(False)
    lock.setToolTip('Keep manual X limits and fixed boundaries across all files, overriding saved file ranges and wavelength presets. Off: reuse manual geometry only for the same wavelength; a different file\'s saved range takes priority.')
    lock.setAccessibleDescription(lock.toolTip())
    owner.drr_memory_lock_x_chk = lock
    lock.toggled.connect(lambda on: _set_global_x_lock(owner, on))
    layout.addWidget(lock)
    try:
        context = json.loads(owner.settings.value(GEOMETRY_CONTEXT_KEY, 'null'))
    except (ValueError, TypeError):
        context = None
    owner._drr_geometry_context = context if isinstance(context, dict) else None
    label = QLabel('Auto ranges: no saved display applied')
    label.setWordWrap(True)
    owner.drr_memory_status = label
    legend = QLabel('Fix M = manual · Fix λ = wavelength · Fix F = saved file')
    legend.setWordWrap(True)
    owner.drr_fix_source_legend = legend
    reset = QPushButton('Restore Auto for this load')
    reset.clicked.connect(lambda: restore_auto(owner))
    owner.drr_memory_auto_btn = reset
    layout.addWidget(reset)
    clear = QPushButton('Clear this center template')
    clear.clicked.connect(lambda: clear_template(owner))
    owner.drr_memory_clear_btn = clear
    layout.addWidget(clear)
    return box


def _controls(owner):
    values = {'axis.' + key: spin for key, spin in owner.drr_spins.items()}
    values.update({'fix.' + key: check for key, check in owner.drr_fix_checks.items()})
    for prefix in ('drr', 'drr_second'):
        values.update({prefix + '.' + key: spin for key, spin in getattr(owner, prefix + '_split_spins').items()})
        values.update({prefix + '_fix.' + key: check for key, check in getattr(owner, prefix + '_split_fix_checks').items()})
    values.update(count=owner.drr_region_count_combo, boundary=owner.drr_split_boundary_chk,
                  split=owner.drr_split_scale_chk, second_split=owner.drr_second_split_scale_chk,
                  cmap=owner.drr_cmap, second_cmap=owner.drr_second_cmap,
                  log=owner.drr_log_chk, clip=owner.drr_clip_chk, zero=owner.drr_center_zero_chk,
                  right_axis=owner.drr_right_yaxis_combo,
                  second_min=owner.drr_second_vmin_spin, second_max=owner.drr_second_vmax_spin)
    return values


def _value(widget):
    if hasattr(widget, 'value'):
        return widget.value()
    if hasattr(widget, 'currentIndex'):
        return widget.currentIndex()
    return widget.isChecked()


def _set(widget, value):
    blocker = QSignalBlocker(widget)
    if hasattr(widget, 'setValue'):
        widget.setValue(value)
    elif hasattr(widget, 'setCurrentIndex'):
        widget.setCurrentIndex(value)
    else:
        widget.setChecked(value)
    del blocker


@dataclass(frozen=True)
class _BackgroundSwitchDisplay:
    folder: str
    selected: tuple
    y_axis_spec: str
    energy: object
    gate: object
    values: tuple
    viewport: object
    limits_from_controls: bool
    second_auto_scale: bool
    borrowed: dict
    origins: dict
    protected: frozenset


def _background_switch_controls(owner):
    controls = _controls(owner)
    for name in ('drr_sg_window_spin', 'drr_sg_poly_spin', 'drr_sg_auto_chk',
                 'drr_sg_mode_combo', 'drr_derivative_combo', 'drr_second_kind_combo',
                 'drr_sg_y_window_spin', 'drr_sg_y_auto_chk'):
        widget = getattr(owner, name, None)
        if widget is not None:
            controls['processing.' + name] = widget
    return controls


def background_folder_key(folder):
    """Compare directory identity across Windows short names and path aliases."""
    return Path(folder).resolve()


def prepare_background_switch(owner):
    """Capture the current view immediately before accepting a BN-only result.

    This snapshot is local to the accepted load; it does not save a display or
    change processing controls. Capture at publication, so edits made while the
    reference was loading remain the current view.
    """
    loaded = getattr(owner, 'loaded', None)
    if loaded is None or loaded.mode != 'DRR' or loaded.cube is None:
        return None
    viewport = deepcopy(getattr(owner, '_drr_view_limits', None))
    figure_axes = set(getattr(getattr(owner, 'figure', None), 'axes', ()))
    axes = getattr(owner, '_drr_heatmap_axes', {}) or {}
    axis = getattr(owner, '_drr_heatmap_ax', None)
    if axis not in figure_axes:
        axis = next((item for item in axes.values() if item in figure_axes), None)
    if axis is not None:
        viewport = (tuple(float(value) for value in axis.get_xlim()),
                    tuple(float(value) for value in axis.get_ylim()))
    return _BackgroundSwitchDisplay(
        str(background_folder_key(loaded.folder)), tuple(loaded.selected_files), loaded.y_axis_spec,
        np.asarray(loaded.cube.energy).copy(), np.asarray(loaded.cube.gate).copy(),
        tuple((name, _value(widget)) for name, widget in _background_switch_controls(owner).items()),
        viewport, bool(getattr(owner, '_drr_limits_from_controls', False)),
        bool(owner.drr_second_auto_scale),
        deepcopy(getattr(owner, '_drr_memory_borrowed', {})),
        dict(getattr(owner, '_drr_fix_origins', {})),
        frozenset(getattr(owner, '_drr_memory_protected', ())))


def restore_background_switch(owner, loaded, snapshot):
    """Preserve all bounds for an accepted BN-only load of the exact same grid.

    Call after publishing ``loaded`` and clearing obsolete compute jobs, in
    place of normal history restoration and automatic load limits. Fix states
    stay unchanged. Explicit Auto actions release the temporary protections
    through the existing display-memory handlers.
    """
    if (not isinstance(snapshot, _BackgroundSwitchDisplay)
            or getattr(owner, 'loaded', None) is not loaded
            or loaded.mode != 'DRR' or loaded.cube is None
            or background_folder_key(loaded.folder) != Path(snapshot.folder)
            or tuple(loaded.selected_files) != snapshot.selected
            or background_folder_key(getattr(owner, 'current_folder', loaded.folder)) != Path(snapshot.folder)
            or tuple(getattr(owner, 'drr_selected_files', loaded.selected_files)) != snapshot.selected
            or loaded.y_axis_spec != snapshot.y_axis_spec
            or not np.array_equal(loaded.cube.energy, snapshot.energy, equal_nan=True)
            or not np.array_equal(loaded.cube.gate, snapshot.gate, equal_nan=True)):
        return False
    controls = _background_switch_controls(owner)
    if any(name not in controls for name, _ in snapshot.values):
        return False
    for name, value in snapshot.values:
        _set(controls[name], value)
    owner.drr_second_auto_scale = snapshot.second_auto_scale
    owner._drr_memory_borrowed = deepcopy(snapshot.borrowed)
    owner._drr_fix_origins = dict(snapshot.origins)
    owner._drr_memory_protected = set(snapshot.protected) | {
        name for name, _ in snapshot.values
        if name.startswith(('axis.', 'drr.', 'drr_second.'))
        or name in ('second_min', 'second_max')}
    owner._drr_memory_cube = loaded.cube
    owner._drr_memory_pending = None
    owner._drr_memory_pending_overrides = {}
    owner._drr_bn_group_processing_hold = (
        loaded.cube, int(owner.drr_sg_window_spin.value()), int(owner.drr_sg_poly_spin.value()),
        int(owner.drr_sg_y_window_spin.value()), bool(owner.drr_sg_auto_chk.isChecked()),
        bool(owner.drr_sg_y_auto_chk.isChecked()))
    owner._drr_view_limits = deepcopy(snapshot.viewport)
    owner._drr_limits_from_controls = snapshot.limits_from_controls if snapshot.viewport is None else False
    sync_controls(owner)
    refresh_fix_sources(owner)
    return True


def _fixed_value_key(name):
    if name.startswith('fix.'):
        return 'axis.' + name[4:]
    if '_fix.' in name:
        return name.replace('_fix.', '.')
    return None


def _set_global_x_lock(owner, enabled):
    if enabled:
        lo, hi = (owner.drr_spins[key].value() for key in ('xmin', 'xmax'))
        if (not owner.loaded or owner.loaded.mode != 'DRR' or owner.loaded.cube is None
                or not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi):
            _set(owner.drr_memory_lock_x_chk, False)
            owner.settings.setValue(PREF_PREFIX + 'lock_x', False)
            owner.drr_memory_status.setText('Load DRR data and choose a valid X range before locking across files.')
            return
    owner.settings.setValue(PREF_PREFIX + 'lock_x', enabled)
    if enabled:
        controls = _controls(owner)
        boundaries = ('x0', 'x1')[:owner.drr_region_count_combo.currentData() - 1]
        for name in GEOMETRY_FIXES:
            if name.startswith('fix.') or name.rsplit('.', 1)[1] in boundaries:
                _set(controls[name], True)
                release(owner, name)
    refresh_fix_sources(owner)


def save_global_geometry(owner):
    """Persist explicit global geometry before borrowed layout settings unwind."""
    if not owner.drr_memory_lock_x_chk.isChecked():
        owner.settings.remove(GLOBAL_GEOMETRY_KEY)
        return
    controls = _controls(owner)
    keys = (*GEOMETRY_LAYOUT, *GEOMETRY_FIXES, *(_fixed_value_key(name) for name in GEOMETRY_FIXES))
    owner.settings.setValue(GLOBAL_GEOMETRY_KEY, json.dumps({key: _value(controls[key]) for key in keys}))


def restore_global_geometry(owner):
    if not owner.drr_memory_lock_x_chk.isChecked():
        return
    try:
        state = json.loads(owner.settings.value(GLOBAL_GEOMETRY_KEY, '{}'))
        keys = (*GEOMETRY_LAYOUT, *GEOMETRY_FIXES, *(_fixed_value_key(name) for name in GEOMETRY_FIXES))
        if not isinstance(state, dict) or any(key not in state for key in keys):
            raise ValueError('Missing locked geometry')
        if (state['count'] not in (0, 1, 2)
                or any(not isinstance(state[key], bool) for key in (*GEOMETRY_FIXES, *GEOMETRY_LAYOUT[1:]))
                or any(not math.isfinite(float(state[_fixed_value_key(key)])) for key in GEOMETRY_FIXES)
                or float(state['axis.xmin']) >= float(state['axis.xmax'])):
            raise ValueError('Invalid locked geometry')
    except (ValueError, TypeError, KeyError):
        _set(owner.drr_memory_lock_x_chk, False)
        owner.settings.setValue(PREF_PREFIX + 'lock_x', False)
        return
    controls = _controls(owner)
    for key in keys:
        _set(controls[key], state[key])
    sync_controls(owner)
    refresh_fix_sources(owner)


def _manual_values(owner):
    controls = _controls(owner)
    borrowed = getattr(owner, '_drr_memory_borrowed', {})
    values = {}
    for name, check in controls.items():
        key = _fixed_value_key(name)
        if key is not None and check.isChecked() and name not in borrowed:
            values[name], values[key] = True, _value(controls[key])
    if not owner.drr_second_auto_scale and not all(key in borrowed for key in ('second_min', 'second_max')):
        values.update(second_min=owner.drr_second_vmin_spin.value(), second_max=owner.drr_second_vmax_spin.value())
    return values


def refresh_fix_sources(owner):
    origins = getattr(owner, '_drr_fix_origins', {})
    owner._drr_fix_origins = origins
    for name, check in _controls(owner).items():
        if _fixed_value_key(name) is None:
            continue
        origin = origins.get(name, 'manual') if check.isChecked() else 'auto'
        suffix = {'manual': ' M', 'center': ' λ', 'file': ' F', 'auto': ''}[origin]
        check.setProperty('fixOrigin', origin)
        check.setText('Fix' + suffix)
        check.setMinimumWidth(check.sizeHint().width())
        descriptions = {
            'manual': 'Manual Fix: keep this value when loading files. Uncheck to use Auto.',
            'center': f"Wavelength Fix ({getattr(owner, '_drr_memory_center', '')} nm): follows this center template. Edit the value, or uncheck and check again, to make it manual.",
            'file': 'Saved-file Fix: restored for this file. Edit the value, or uncheck and check again, to make it manual.',
            'auto': 'Auto: follows data or a saved display. Check to keep the current value as a manual Fix.',
        }
        if origin == 'manual' and name in GEOMETRY_FIXES:
            if owner.drr_memory_lock_x_chk.isChecked():
                descriptions['manual'] = 'Manual X lock across files: overrides saved file ranges and wavelength presets.'
            else:
                center = getattr(owner, '_drr_memory_center', None)
                scope = f'the same wavelength ({center} nm)' if center else 'this file'
                descriptions['manual'] = f'Manual Fix: kept for {scope}. A different file\'s saved X range takes priority. Enable Lock X range across files to override other wavelengths.'
        check.setToolTip(descriptions[origin])
        check.setAccessibleDescription(descriptions[origin])
    _refresh_status(owner)


def wire_controls(owner):
    # Register before normal plot handlers so an explicit edit releases its
    # remembered value before any dependent automatic range calculation.
    for name, widget in _controls(owner).items():
        signal = widget.valueChanged if hasattr(widget, 'valueChanged') else (
            widget.currentIndexChanged if hasattr(widget, 'currentIndexChanged') else widget.toggled)
        signal.connect(lambda _value, key=name: release(owner, key))
    for button, keys in (
        (owner.drr_auto_x_btn, ('axis.xmin', 'axis.xmax')),
        (owner.drr_auto_y_btn, ('axis.ymin', 'axis.ymax')),
        (owner.drr_auto_v_btn, ('axis.vmin', 'axis.vmax')),
        (owner.drr_second_auto_v_btn, ('second_min', 'second_max')),
    ):
        button.clicked.connect(lambda _on=False, names=keys: [release(owner, key, manual_edit=False) for key in names])
    for prefix in ('drr', 'drr_second'):
        for side in ('left', 'middle', 'right'):
            button = getattr(owner, prefix + '_split_auto_' + side + '_btn')
            button.clicked.connect(lambda _on=False, p=prefix, s=side: (
                release(owner, p + '.' + s + '_vmin', manual_edit=False),
                release(owner, p + '.' + s + '_vmax', manual_edit=False)))
    refresh_fix_sources(owner)


def release(owner, key, *, manual_edit=True):
    controls = _controls(owner)
    fixed_name = 'fix.' + key[5:] if key.startswith('axis.') else key.replace('.', '_fix.', 1)
    check = controls.get(fixed_name)
    # Auto respects checked limits. Leave their source and borrowed value intact;
    # only an actual numeric edit or checkbox toggle establishes a manual lock.
    if not manual_edit and check is not None and hasattr(check, 'isChecked') and check.isChecked():
        return
    if manual_edit and hasattr(owner, '_drr_color_compute'):
        owner._drr_color_compute.release(key)
    if manual_edit and key in {'axis.vmin', 'axis.vmax', 'fix.vmin', 'fix.vmax'}:
        owner._drr_pending_raw_auto = None
    deferred = getattr(owner, '_drr_second_split_deferred', None)
    if manual_edit and deferred is not None and key.startswith('drr_second.'):
        deferred[1].pop(key.split('.', 1)[1], None)
    if getattr(owner, '_drr_memory_pending', None) is not None:
        owner._drr_memory_pending_overrides[key] = _value(controls[key])
    getattr(owner, '_drr_memory_protected', set()).discard(key)
    getattr(owner, '_drr_memory_borrowed', {}).pop(key, None)
    origins = getattr(owner, '_drr_fix_origins', {})
    owner._drr_fix_origins = origins
    fixed_key = _fixed_value_key(key)
    origin_changed = False
    if fixed_key is not None:
        origins[key] = 'manual' if controls[key].isChecked() else 'auto'
        getattr(owner, '_drr_memory_borrowed', {}).pop(fixed_key, None)
        origin_changed = True
    else:
        if check is not None and hasattr(check, 'isChecked') and check.isChecked():
            origin_changed = origins.get(fixed_name) != 'manual'
            origins[fixed_name] = 'manual'
            getattr(owner, '_drr_memory_borrowed', {}).pop(fixed_name, None)
    if key.startswith('fix.'):
        getattr(owner, '_drr_memory_protected', set()).discard('axis.' + key[4:])
    elif '_fix.' in key:
        getattr(owner, '_drr_memory_protected', set()).discard(key.replace('_fix.', '.'))
    elif key == 'count':
        owner._drr_memory_protected = {key for key in getattr(owner, '_drr_memory_protected', set())
                                       if not key.startswith(('drr.', 'drr_second.'))}
    geometry_fix = key if fixed_key is not None else fixed_name
    if geometry_fix in GEOMETRY_FIXES and origins.get(geometry_fix) == 'manual':
        # A manually adopted X geometry owns its layout too. A temporary visit
        # to another mode must not unwind a borrowed count behind that lock.
        for name in GEOMETRY_LAYOUT:
            getattr(owner, '_drr_memory_borrowed', {}).pop(name, None)
    if origin_changed:
        refresh_fix_sources(owner)


def reset_memory(owner):
    controls = _controls(owner)
    for name, (old, applied) in getattr(owner, '_drr_memory_borrowed', {}).items():
        if _value(controls[name]) == applied:
            _set(controls[name], old)
            getattr(owner, '_drr_fix_origins', {}).pop(name, None)
    owner._drr_memory_borrowed = {}
    owner._drr_memory_protected = set()
    owner._drr_memory_cube = None
    sync_controls(owner)
    refresh_fix_sources(owner)


def sync_controls(owner):
    """Update layout state without the toggle handler's automatic scaling."""
    enabled = owner.drr_split_scale_chk.isChecked()
    for prefix in ('drr', 'drr_second'):
        getattr(owner, prefix + '_split_scale_panel').setVisible(enabled)
    for key in ('vmin', 'vmax'):
        owner.drr_spins[key].setEnabled(not enabled)
        owner.drr_fix_checks[key].setEnabled(not enabled)
    sync_visibility(owner)


def _x_range_tolerance(owner):
    # Saved limits pass through these spin boxes. Their rounding can put an
    # endpoint just outside the full-precision spectral extent of the same data.
    return max(0.5 * 10 ** -owner.drr_spins[key].decimals() for key in ('xmin', 'xmax'))


def _range(values, lo, hi, covered=True, tolerance=0.0):
    try:
        a, b = map(float, values)
        overlap = lo - tolerance <= a < b <= hi + tolerance if covered else a <= hi and b >= lo
        return math.isfinite(a) and math.isfinite(b) and a < b and overlap
    except (ValueError, TypeError):
        return False


def _valid(state, cube, exact, colors=False, x_tolerance=0.0):
    try:
        if not isinstance(state, dict) or state.get('version') != history.VERSION:
            return False
        deferred = state.get('deferred_second_regions', ())
        allowed = {side + '_v' + bound for side in ('left', 'middle', 'right') for bound in ('min', 'max')}
        if (not isinstance(deferred, (list, tuple)) or any(key not in allowed for key in deferred)
                or not isinstance(state.get('second_auto_pending', False), bool)):
            return False
        if any(not isinstance(state.get(key, {}), dict) for key in
               ('ranges', 'regions', 'fixes', 'raw', 'second', 'axis', 'viewport')):
            return False
        ranges = state['ranges']
        if not _range((ranges['xmin'], ranges['xmax']), np.nanmin(cube.energy), np.nanmax(cube.energy),
                      covered=not exact, tolerance=x_tolerance):
            return False
        regions = state['regions']
        if any(not isinstance(regions.get(key, {}), dict) for key in
               ('drr', 'drr_fix', 'drr_second', 'drr_second_fix')):
            return False
        count = regions['count']
        if count not in (1, 2, 3):
            return False
        boundaries = [ranges['xmin']]
        if count > 1:
            boundaries.append(regions['drr']['x0'])
        if count == 3:
            boundaries.append(regions['drr']['x1'])
        boundaries.append(ranges['xmax'])
        if not all(math.isfinite(float(x)) for x in boundaries) or not all(a < b for a, b in zip(boundaries, boundaries[1:])):
            return False
        if exact and not _range((ranges['ymin'], ranges['ymax']), np.nanmin(cube.gate), np.nanmax(cube.gate), covered=False):
            return False
        if exact and state.get('viewport'):
            viewport = state['viewport']
            if not _range(viewport['xlim'], np.nanmin(cube.energy), np.nanmax(cube.energy), covered=False) or not _range(
                    viewport['ylim'], np.nanmin(cube.gate), np.nanmax(cube.gate), covered=False):
                return False
        if exact or colors:
            pairs = [(ranges['vmin'], ranges['vmax'])]
            for prefix in ('drr', 'drr_second'):
                colors = regions.get(prefix, {})
                for side in ('left', 'middle', 'right'):
                    if side == 'middle' and count != 3 or count == 1:
                        continue
                    if side + '_vmin' in colors:
                        pairs.append((colors[side + '_vmin'], colors[side + '_vmax']))
            second = state.get('second', {})
            if second:
                pairs.append((second['vmin'], second['vmax']))
            if not all(math.isfinite(float(a)) and math.isfinite(float(b)) and a < b for a, b in pairs):
                return False
            if state.get('raw', {}).get('log_scale') and any(a <= 0 for a, _ in pairs[:1]):
                return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _scope_manual_geometry(owner, loaded, names, center, exact, template, manual, edits):
    """Expire outgoing manual geometry before borrowing any incoming snapshot."""
    context = {'folder': Path(loaded.folder).resolve().as_posix().casefold(),
               'source': history.source_key(loaded.folder, names), 'center': center}
    previous = owner._drr_geometry_context
    same_file = previous == context
    same_center = previous is not None and center is not None and previous.get('center') == center
    global_lock = owner.drr_memory_lock_x_chk.isChecked()
    keep = (global_lock or previous is None or same_file
            or same_center and exact is None)
    if keep and previous is not None and not same_file and not global_lock:
        lo, hi = float(np.nanmin(loaded.cube.energy)), float(np.nanmax(loaded.cube.energy))
        fallback = (exact or template or {}).get('ranges', {})
        limits = tuple(manual.get('axis.' + key, fallback.get(key, default))
                       for key, default in (('xmin', lo), ('xmax', hi)))
        keep = _range(limits, lo, hi, tolerance=_x_range_tolerance(owner))
    controls = _controls(owner)
    if not keep:
        for name in GEOMETRY_FIXES:
            key = _fixed_value_key(name)
            # An edit committed while loading belongs to the incoming display.
            if name in manual and name not in edits and key not in edits:
                manual.pop(name)
                manual.pop(key, None)
                _set(controls[name], False)
                owner._drr_fix_origins[name] = 'auto'
        for key in GEOMETRY_LAYOUT:
            manual.pop(key, None)
    owner._drr_geometry_context = context
    if previous != context:
        owner.settings.setValue(GEOMETRY_CONTEXT_KEY, json.dumps(context))


def _refresh_status(owner):
    base = getattr(owner, '_drr_memory_status_base', 'Auto ranges: no saved display applied')
    manual = _manual_values(owner)
    center = getattr(owner, '_drr_memory_center', None)
    if any(key in manual for key in GEOMETRY_FIXES):
        scope = ('locked across files' if owner.drr_memory_lock_x_chk.isChecked() else
                 f'manual for {center} nm' if center else 'manual for this file')
        base += f'; X geometry: {scope}'
    owner.drr_memory_status.setText(base)


def _restore_status(owner, exact, template, center):
    owner._drr_memory_status_base = ("Restored this file's saved display" if exact else
                                     f'Using {center} nm center template' if template else
                                     'Auto ranges: no compatible saved display')
    _refresh_status(owner)


def prepare_restore(owner, loaded, overrides=None):
    names = history.source_names(loaded.folder, loaded.selected_files or [loaded.primary_file], loaded.provenance_records) if loaded.mode == 'DRR' and loaded.folder else []
    owner._drr_memory_center = history.center_key(names) if names else None
    focused = pending_numeric_input(owner)
    if loaded.mode == 'DRR' and overrides is None and focused is not None:
        key = next((key for key, widget in _controls(owner).items() if widget is focused), None)
        if getattr(owner, '_drr_memory_pending', None) is None:
            owner._drr_memory_pending_overrides = {}
        owner._drr_memory_pending = (loaded, key)
        return
    owner._drr_memory_pending = None
    owner._drr_memory_pending_overrides = {}
    manual = _manual_values(owner)
    if owner._drr_geometry_context is not None and any(key in manual for key in GEOMETRY_FIXES):
        controls = _controls(owner)
        manual.update({key: _value(controls[key]) for key in GEOMETRY_LAYOUT})
    edits = overrides or {}
    reset_memory(owner)
    owner.drr_memory_lock_x_chk.setEnabled(loaded.mode == 'DRR' and loaded.cube is not None)
    if loaded.mode != 'DRR' or loaded.cube is None or not loaded.folder:
        return
    owner._update_drr_right_axis_options()
    requested = owner.drr_memory_files_chk.isChecked(), owner.drr_memory_centers_chk.isChecked()
    candidates = getattr(loaded, 'drr_display_candidates', None)
    if not any(requested):
        candidates = {'exact': None, 'template': None, 'center': owner._drr_memory_center}
    elif (candidates is None or candidates.get('requested', (True, True)) != requested
          or candidates.get('revision') != history.current_revision(loaded.folder)):
        candidates = history.find_candidates(loaded.folder, names,
                                            restore_files=requested[0], restore_centers=requested[1])
    center = candidates['center']
    owner._drr_memory_center = center
    exact = candidates['exact'] if owner.drr_memory_files_chk.isChecked() else None
    if isinstance(exact, dict):
        axis = exact.get('axis') or {}
        if not isinstance(axis, dict):
            axis = {'y_spec': 'incompatible'}
        if axis.get('y_spec') not in (None, loaded.y_axis_spec) or any(
            axis.get(key) is not None and axis[key] != value for key, value in (
                ('unit', loaded.cube.gate_unit), ('semantic', loaded.cube.y_axis_semantic),
                ('label', loaded.cube.gate_label))
        ):
            exact = None
    if not _valid(exact, loaded.cube, True):
        exact = None
    template = candidates['template'] if owner.drr_memory_centers_chk.isChecked() else None
    if template is None and center and owner.drr_memory_centers_chk.isChecked() and owner.drr_memory_global_chk.isChecked():
        try:
            template = json.loads(owner.settings.value(GLOBAL_KEY, '{}')).get(center)
        except (ValueError, TypeError, AttributeError):
            template = None
    if not _valid(template, loaded.cube, False, colors=owner.drr_memory_colors_chk.isChecked(),
                  x_tolerance=_x_range_tolerance(owner)):
        template = None
    _scope_manual_geometry(owner, loaded, names, center, exact, template, manual, edits)
    overrides = {**manual, **edits}
    state = exact or template
    if state is None:
        _restore_status(owner, exact, template, center)
        if overrides:
            controls = _controls(owner)
            for name, value in overrides.items():
                _set(controls[name], value)
            owner._drr_memory_protected.update(overrides)
            owner._drr_memory_cube = loaded.cube
            owner._drr_view_limits = None
            owner._drr_limits_from_controls = True
            sync_controls(owner)
        refresh_fix_sources(owner)
        return
    restore_colors = bool(exact) or owner.drr_memory_colors_chk.isChecked()
    controls = _controls(owner)
    assignments = {'count': state['regions']['count'] - 1,
                   'split': state['regions']['count'] > 1, 'second_split': state['regions']['count'] > 1,
                   'boundary': bool(state['regions'].get('boundary', True))}
    if not exact:
        if not restore_colors and 'second_min' not in manual:
            owner.drr_second_auto_scale = True
    ranges = state['ranges']
    for key in ('xmin', 'xmax', 'ymin', 'ymax', 'vmin', 'vmax'):
        if key.startswith('x') or exact or restore_colors and key.startswith('v'):
            assignments['axis.' + key] = float(ranges[key])
            assignments['fix.' + key] = bool(state.get('fixes', {}).get(key, False)) if exact else key.startswith('x')
    for prefix in ('drr', 'drr_second'):
        active_boundaries = ('x0', 'x1')[:state['regions']['count'] - 1]
        for key in ('x0', 'x1'):
            if key not in active_boundaries:
                assignments[prefix + '_fix.' + key] = False
        for key, value in state['regions'].get(prefix, {}).items():
            if prefix + '.' + key not in controls or not (key.startswith('x') or restore_colors):
                continue
            if prefix == 'drr_second' and key in state.get('deferred_second_regions', ()):
                continue
            if key.startswith('x') and key not in active_boundaries:
                continue
            assignments[prefix + '.' + key] = float(value)
            assignments[prefix + '_fix.' + key] = bool(state['regions'].get(prefix + '_fix', {}).get(key, False)) if exact else key.startswith('x')
    if restore_colors:
        raw = state.get('raw', {})
        for name, key in (('log', 'log_scale'), ('clip', 'clip_outliers'), ('zero', 'center_zero')):
            assignments[name] = bool(raw.get(key, False))
        for name, params in (('cmap', raw), ('second_cmap', state.get('second', {}))):
            cmap = str(params.get('cmap', ''))
            index = controls[name].findData(cmap)
            if index < 0:
                index = controls[name].findText(cmap)
            if index >= 0:
                assignments[name] = index
        second = state.get('second', {})
        if second and not state.get('second_auto_pending'):
            assignments['second_min'], assignments['second_max'] = float(second['vmin']), float(second['vmax'])
        elif state.get('second_auto_pending') and 'second_min' not in manual:
            owner.drr_second_auto_scale = True
    if exact and (state.get('right_axis') == 'Off' or state.get('right_axis') in loaded.drr_axis_coordinates):
        index = controls['right_axis'].findText(state['right_axis'])
        if index >= 0:
            assignments['right_axis'] = index
    assignments.update(overrides or {})
    if overrides and 'count' in overrides:
        assignments['split'] = assignments['second_split'] = assignments['count'] > 0
    for name, value in assignments.items():
        if name not in manual:
            owner._drr_memory_borrowed[name] = (_value(controls[name]), value)
        _set(controls[name], value)
        if _fixed_value_key(name) is not None:
            owner._drr_fix_origins[name] = 'manual' if name in manual else 'file' if exact else 'center'
        if name.startswith(('axis.', 'drr.', 'drr_second.')) or name in ('second_min', 'second_max'):
            owner._drr_memory_protected.add(name)
    owner._drr_memory_cube = loaded.cube
    changed_axes = {'axis.xmin', 'axis.xmax', 'axis.ymin', 'axis.ymax'} & set(overrides or {})
    viewport = state.get('viewport') if exact else None
    if viewport:
        limits = []
        for axis in ('x', 'y'):
            keys = ('axis.' + axis + 'min', 'axis.' + axis + 'max')
            limits.append(tuple(_value(controls[key]) for key in keys) if changed_axes.intersection(keys)
                          else tuple(viewport[axis + 'lim']))
        owner._drr_view_limits = tuple(limits)
    else:
        owner._drr_view_limits = None
    owner._drr_limits_from_controls = not bool(viewport)
    sync_controls(owner)
    refresh_fix_sources(owner)
    if overrides and 'count' in overrides and assignments['count'] == 2:
        from ui_qt.drr_regions import initialize_boundaries
        initialize_boundaries(owner)
    _restore_status(owner, exact, template, center)


def guarded_auto(function):
    @wraps(function)
    def guarded(owner, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0] != 'DRR':
            return function(owner, *args, **kwargs)
        if getattr(owner, '_drr_memory_pending', None) is not None:
            return False
        with protect(owner):
            return function(owner, *args, **kwargs)
    return guarded


@contextmanager
def protect(owner):
    protected = getattr(owner, '_drr_memory_protected', set())
    if not protected or getattr(owner.loaded, 'cube', None) is not getattr(owner, '_drr_memory_cube', None):
        yield
        return
    controls = _controls(owner)
    restored, blockers = [], []
    for name in protected:
        check_name = 'fix.' + name[5:] if name.startswith('axis.') else name.replace('.', '_fix.', 1)
        check = controls.get(check_name)
        if check is not None and hasattr(check, 'isChecked'):
            blockers.append(QSignalBlocker(check))
            restored.append((check, check.isChecked()))
            check.setChecked(True)
    try:
        yield
    finally:
        for check, value in restored:
            check.setChecked(value)
        blockers.clear()


def second_is_protected(owner):
    if getattr(owner, '_drr_memory_pending', None) is not None:
        return True
    return bool({'second_min', 'second_max'} & getattr(owner, '_drr_memory_protected', set())) and (
        getattr(owner.loaded, 'cube', None) is getattr(owner, '_drr_memory_cube', None))


def resume_pending(owner):
    pending = getattr(owner, '_drr_memory_pending', None)
    if pending is None:
        return
    loaded, key = pending
    if owner.loaded is not loaded:
        owner._drr_memory_pending = None
        owner._drr_memory_pending_overrides = {}
        return
    if pending_numeric_input(owner) is not None:
        return
    overrides = {key: _value(_controls(owner)[key])} if key is not None else {}
    overrides.update(owner._drr_memory_pending_overrides)
    if 'count' in overrides:
        controls = _controls(owner)
        overrides['split'] = overrides['second_split'] = overrides['count'] > 0
        for name in ('drr.x0', 'drr.x1', 'drr_fix.x0', 'drr_fix.x1'):
            overrides.setdefault(name, _value(controls[name]))
    prepare_restore(owner, loaded, overrides=overrides)
    owner._apply_auto_limits_for_loaded()


def restore_auto(owner):
    if not owner.loaded or owner.loaded.mode != 'DRR':
        return
    owner._drr_memory_pending = None
    owner._drr_memory_pending_overrides = {}
    reset_memory(owner)
    owner.drr_memory_lock_x_chk.setChecked(False)
    for name, control in _controls(owner).items():
        if name.startswith(('fix.', 'drr_fix.', 'drr_second_fix.')):
            _set(control, False)
    refresh_fix_sources(owner)
    owner.drr_second_auto_scale = True
    owner._drr_view_limits = None
    owner._drr_limits_from_controls = True
    owner._apply_auto_limits_for_loaded()
    owner._drr_memory_status_base = 'Auto ranges for this load'
    _refresh_status(owner)
    owner._schedule_plot_redraw('DRR', delay_ms=0)


def clear_template(owner):
    center = getattr(owner, '_drr_memory_center', None)
    if not center or not owner.loaded or owner.loaded.mode != 'DRR':
        owner.drr_memory_status.setText('No single wavelength center to clear')
        return
    try:
        history.clear_center(owner.loaded.folder, center)
        globals_ = json.loads(owner.settings.value(GLOBAL_KEY, '{}'))
        globals_.pop(center, None)
        owner.settings.setValue(GLOBAL_KEY, json.dumps(globals_))
        owner.drr_memory_status.setText(f'Cleared {center} nm template; file snapshots kept')
    except (OSError, ValueError, TypeError, AttributeError) as error:
        owner._status(f'Could not clear center template: {error}')


def capture(owner, raw_params, second_params):
    loaded = owner.loaded
    names = history.source_names(loaded.folder, loaded.selected_files or [loaded.primary_file], loaded.provenance_records)
    center = history.center_key(names)
    return {'version': history.VERSION, 'files': names, 'center': center,
            'axis': {'y_spec': loaded.y_axis_spec, 'unit': loaded.cube.gate_unit,
                     'semantic': loaded.cube.y_axis_semantic, 'label': loaded.cube.gate_label},
            'ranges': {key: spin.value() for key, spin in owner.drr_spins.items()},
            'fixes': {key: check.isChecked() for key, check in owner.drr_fix_checks.items()},
            'regions': settings_state(owner), 'raw': _plot_settings(raw_params), 'second': _plot_settings(second_params),
            'right_axis': owner.drr_right_yaxis_combo.currentText(),
            'share_global': owner.drr_memory_global_chk.isChecked()}


def capture_visible(owner):
    """Freeze the toolbar's rendered viewport without computing derivatives."""
    loaded = owner.loaded
    rendered = getattr(owner, '_drr_reuse_state', None)
    axes = getattr(owner, '_drr_heatmap_axes', {})
    displayed = {key: axis._dptk_heatmap_params for key, axis in axes.items()
                 if hasattr(axis, '_dptk_heatmap_params')}
    if (not loaded or loaded.mode != 'DRR' or not loaded.folder or not displayed
            or rendered is None or rendered[0] is not loaded or rendered[1] is not loaded.cube):
        return None
    visible_key = 'raw' if 'raw' in displayed else 'second'
    visible = displayed[visible_key]
    viewport = dict(xlim=tuple(axes[visible_key].get_xlim()), ylim=tuple(axes[visible_key].get_ylim()))
    params = {}
    for key, prefix in (('raw', 'drr'), ('second', 'drr_second')):
        source = displayed.get(key)
        if source is None:
            split = visible.split_scale
            if split is not None:
                spins = getattr(owner, prefix + '_split_spins')
                split = replace(split, **{side + '_v' + bound: spins[side + '_v' + bound].value()
                                          for side in ('left', 'middle', 'right') for bound in ('min', 'max')})
            source = replace(visible, cmap=owner._resolved_cmap(getattr(owner, prefix + '_cmap')),
                             vmin=owner.drr_spins['vmin'].value() if key == 'raw' else owner.drr_second_vmin_spin.value(),
                             vmax=owner.drr_spins['vmax'].value() if key == 'raw' else owner.drr_second_vmax_spin.value(),
                             split_scale=split)
        params[key] = replace(source, **viewport)
    state = finalize(capture(owner, params['raw'], params['second']), params['raw'], params['second'])
    domain = rendered[4][visible_key][3]
    state['ranges'].update(dict(zip(('xmin', 'xmax', 'ymin', 'ymax'), (*domain.xlim, *domain.ylim))))
    split = visible.split_scale
    state['regions']['count'] = 3 if split and split.split_x2 is not None else 2 if split else 1
    if split is not None:
        state['regions']['boundary'] = split.show_boundary
        for key, prefix in (('raw', 'drr'), ('second', 'drr_second')):
            product = params[key].split_scale
            region = state['regions'][prefix]
            region.update(x0=split.split_x)
            if split.split_x2 is not None:
                region['x1'] = split.split_x2
            if product is not None:
                for side in ('left', 'middle', 'right'):
                    for bound in ('min', 'max'):
                        name = side + '_v' + bound
                        value = getattr(product, name)
                        if value is not None:
                            region[name] = value
    state['right_axis'] = 'Off' if visible.right_axis is None else state['right_axis']
    if 'second' not in displayed:
        from ui_qt.drr_regions import deferred_auto_keys
        state['deferred_second_regions'] = list(deferred_auto_keys(owner))
        state['second_auto_pending'] = bool(getattr(owner, 'drr_second_auto_scale', True)) and not second_is_protected(owner)
    return loaded.folder, state


def remember_shared(owner, state):
    if not state or not state.get('share_global') or not state.get('center'):
        return
    try:
        templates = json.loads(owner.settings.value(GLOBAL_KEY, '{}'))
        templates[state['center']] = state
        owner.settings.setValue(GLOBAL_KEY, json.dumps(templates))
    except (ValueError, TypeError, AttributeError):
        owner._append_log('Could not update shared center templates.')


def finalize(state, raw_params, second_params):
    state = deepcopy(state)
    state['raw'], state['second'] = _plot_settings(raw_params), _plot_settings(second_params)
    # Split boundaries belong to the sidebar's full domain. A zoom can show
    # only one region, so retain its viewport separately from that geometry.
    state['viewport'] = {'xlim': list(raw_params.xlim), 'ylim': list(raw_params.ylim)}
    state['ranges'].update(vmin=raw_params.vmin, vmax=raw_params.vmax)
    for prefix, params in (('drr', raw_params), ('drr_second', second_params)):
        split = params.split_scale
        if split is not None:
            for side in ('left', 'middle', 'right'):
                for bound in ('vmin', 'vmax'):
                    key = side + '_' + bound
                    value = getattr(split, key)
                    if value is not None:
                        state['regions'][prefix][key] = float(value)
    return state


def _plot_settings(params):
    # Measured right-axis arrays belong to the data, not reusable preferences.
    return {'vmin': float(params.vmin), 'vmax': float(params.vmax), 'cmap': params.cmap,
            'log_scale': bool(params.log_scale), 'clip_outliers': bool(params.clip_outliers),
            'center_zero': bool(params.center_zero)}
