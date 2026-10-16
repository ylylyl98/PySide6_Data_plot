"""Optional DRR three-region controls, automatic limits, and saved state."""
import json
import numpy as np
from PySide6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel, QToolButton, QWidget
from core.plotting import SplitColorScale, split_region_slices

SETTINGS_KEY = 'drr/region_colors_json'


def deferred_auto_keys(owner):
    deferred = getattr(owner, '_drr_second_split_deferred', None)
    if deferred is None or deferred[0] is not getattr(owner.loaded, 'cube', None):
        return ()
    protected = getattr(owner, '_drr_memory_protected', set())
    return tuple(key for key, value in deferred[1].items()
                 if not owner.drr_second_split_fix_checks[key].isChecked()
                 and 'drr_second.' + key not in protected
                 and owner.drr_second_split_spins[key].value() == value)


def note_second_region_auto(owner, side):
    deferred = getattr(owner, '_drr_second_split_deferred', None)
    if deferred is not None and deferred[0] is getattr(owner.loaded, 'cube', None):
        for key in (side + '_vmin', side + '_vmax'):
            if not owner.drr_second_split_fix_checks[key].isChecked():
                deferred[1][key] = owner.drr_second_split_spins[key].value()


def is_triple(owner, prefix='drr'):
    return prefix in {'drr', 'drr_second'} and owner.drr_region_count_combo.currentData() == 3


def build_panel(owner, prefix, grid, spins, fixes, boundary, left_auto, right_auto, make_spin):
    if prefix == 'drr':
        combo = QComboBox()
        for n in (1, 2, 3):
            combo.addItem(f'{n} region' + ('s' if n > 1 else ''), n)
        owner.drr_region_count_combo = combo
    for key in ('x1', 'middle_vmin', 'middle_vmax'):
        spins[key] = make_spin()
        fixes[key] = QCheckBox('Fix')
    spins['middle_vmax'].setValue(1)
    middle_auto = QToolButton()
    middle_auto.setText('Auto')
    setattr(owner, f'{prefix}_split_auto_middle_btn', middle_auto)
    middle_auto.clicked.connect(lambda: owner._auto_split_vrange(prefix, 'middle'))
    extra = []
    # Shared X boundaries stay outside the product-specific color panels.
    boundaries = QWidget(grid.parentWidget())
    boundary_grid = QGridLayout(boundaries)
    boundary_grid.setContentsMargins(0, 0, 0, 0)
    boundary_grid.addWidget(QLabel('Boundary 1'), 0, 0)
    boundary_grid.addWidget(spins['x0'], 0, 1, 1, 2)
    boundary_grid.addWidget(fixes['x0'], 0, 3)
    boundary_grid.addWidget(boundary, 2, 0, 1, 4)
    for column, widget in ((0, QLabel('Boundary 2')), (1, spins['x1']), (3, fixes['x1'])):
        boundary_grid.addWidget(widget, 1, column, 1, 2 if column == 1 else 1)
        extra.append(widget)
    if prefix == 'drr_second':
        boundaries.hide()
        extra = []  # Boundaries are shared; no duplicate controls in d2E.
    else:
        owner.drr_region_boundaries = boundaries
    for row, side, auto in ((0, 'left', left_auto), (1, 'middle', middle_auto), (2, 'right', right_auto)):
        # Use the same two-column editor as axes, retaining independent locks.
        from ui_qt.drr_sidebar import range_row
        editor = range_row(
            spins[f'{side}_vmin'], spins[f'{side}_vmax'],
            fixes[f'{side}_vmin'], fixes[f'{side}_vmax'], auto, side.title(), adaptive_step=True)
        grid.addWidget(editor, row, 0)
        if side == 'middle':
            extra.append(editor)
    setattr(owner, f'{prefix}_three_region_widgets', extra)
    for widget in extra:
        widget.hide()


def sync_visibility(owner):
    triple = is_triple(owner)
    for prefix in ('drr', 'drr_second'):
        for widget in getattr(owner, f'{prefix}_three_region_widgets'):
            widget.setVisible(triple)
        getattr(owner, f'{prefix}_split_scale_panel').setTitle(
            'Three X-Region Color Limits' if triple else 'Two X-Region Color Limits')
    from ui_qt.drr_sidebar import sync_color_visibility
    sync_color_visibility(owner)


def select_count(owner):
    count = owner.drr_region_count_combo.currentData()
    if count == 3:
        initialize_boundaries(owner)
    sync_visibility(owner)
    checked = count != 1
    if owner.drr_split_scale_chk.isChecked() != checked:
        owner.drr_split_scale_chk.setChecked(checked)
    else:
        owner._on_split_scale_param_changed('drr')


def sync_legacy_toggle(owner, checked):
    combo = owner.drr_region_count_combo
    if not checked or combo.currentData() == 1:
        blocked = combo.blockSignals(True)
        combo.setCurrentIndex(1 if checked else 0)
        combo.blockSignals(blocked)
    sync_visibility(owner)


def initialize_boundaries(owner, *, center=False):
    spins = owner.drr_split_spins
    fixes = owner.drr_split_fix_checks
    lo, hi = sorted(owner.drr_spins[k].value() for k in ('xmin', 'xmax'))
    a, b = spins['x0'].value(), spins['x1'].value()
    if center or not lo < a < b < hi:
        first, second = lo + (hi-lo)/3, lo + 2*(hi-lo)/3
        if fixes['x0'].isChecked() and lo < a < hi:
            second = (a + hi) / 2
        if fixes['x1'].isChecked() and lo < b < hi:
            first = (lo + b) / 2
        for key, value in (('x0', first), ('x1', second)):
            if not fixes[key].isChecked() or not lo < spins[key].value() < hi:
                owner._set_spin_value_silent(spins[key], value)


def scale(owner, prefix):
    spins = getattr(owner, f'{prefix}_split_spins')
    shared = owner.drr_split_spins
    lo, hi = sorted(owner.drr_spins[k].value() for k in ('xmin', 'xmax'))
    a, b = shared['x0'].value(), shared['x1'].value()
    if not lo < a < b < hi:
        raise ValueError('Three regions require xmin < Boundary 1 < Boundary 2 < xmax.')
    bounds = [(spins[s+'_vmin'].value(), spins[s+'_vmax'].value()) for s in ('left', 'middle', 'right')]
    for vmin, vmax in bounds:
        if vmax <= vmin or (owner._mode_log('DRR') and vmin <= 0):
            raise ValueError('Each region requires min < max (and positive min in log mode).')
    return SplitColorScale(a, *bounds[0], *bounds[2], owner.drr_split_boundary_chk.isChecked(),
                           split_x2=b, middle_vmin=bounds[1][0], middle_vmax=bounds[1][1])


def refresh(owner, prefix, cubes, *, side=None, center=False):
    initialize_boundaries(owner, center=center)
    spins = getattr(owner, f'{prefix}_split_spins')
    fixes = getattr(owner, f'{prefix}_split_fix_checks')
    shared = owner.drr_split_spins
    xlim = tuple(owner.drr_spins[k].value() for k in ('xmin', 'xmax'))
    ylim = sorted(owner.drr_spins[k].value() for k in ('ymin', 'ymax'))
    shape = SplitColorScale(shared['x0'].value(), 0, 1, 0, 1,
                            split_x2=shared['x1'].value(), middle_vmin=0, middle_vmax=1)
    for cube in cubes:
        try:
            split_region_slices(cube.energy, shape, xlim)
        except ValueError as exc:
            owner._status(str(exc))
            return False
    from ui_qt.drr_color_limits import bounds, roi_request
    changed = False
    for index, s in enumerate(('left', 'middle', 'right')):
        if (side is not None and s != side):
            continue
        result = bounds(owner, cubes, roi_request(owner,
            boundaries=(shape.split_x, shape.split_x2), region=index), prefix=prefix, side=s)
        if result is None:
            continue
        lo, hi = result
        keys = (s+'_vmin', s+'_vmax')
        candidates = [spins[k].value() if fixes[k].isChecked() else value for k,value in zip(keys,(lo,hi))]
        if candidates[0] >= candidates[1] or (owner._mode_log('DRR') and candidates[0] <= 0):
            owner._status(f'Auto {s} conflicts with a fixed color limit.')
            continue
        for key, value in zip(keys, candidates):
            if not fixes[key].isChecked():
                owner._set_spin_value_silent(spins[key], value)
                changed = True
    return changed


def settings_state(owner):
    state = {'count': owner.drr_region_count_combo.currentData(),
             'boundary': owner.drr_split_boundary_chk.isChecked()}
    for prefix in ('drr', 'drr_second'):
        state[prefix] = {key: spin.value() for key, spin in getattr(owner, f'{prefix}_split_spins').items()}
        state[prefix+'_fix'] = {key: check.isChecked() for key, check in getattr(owner, f'{prefix}_split_fix_checks').items()}
    return state


def save_settings(owner):
    state = settings_state(owner)
    # File/center values have their own history. Do not turn their temporary
    # locks into app-wide manual preferences when the next window starts.
    for name, (old, applied) in getattr(owner, '_drr_memory_borrowed', {}).items():
        if name == 'count' and state['count'] == applied + 1:
            state['count'] = old + 1
        elif name == 'boundary' and state['boundary'] == applied:
            state['boundary'] = old
        elif '.' in name:
            prefix, key = name.split('.', 1)
            if prefix in state and isinstance(state[prefix], dict) and state[prefix].get(key) == applied:
                state[prefix][key] = old
    if state == getattr(owner, '_drr_region_settings_baseline', None):
        return
    owner.settings.setValue(SETTINGS_KEY, json.dumps(state))
    owner._drr_region_settings_baseline = state


def restore_settings(owner):
    try:
        state = json.loads(owner.settings.value(SETTINGS_KEY, '{}'))
        if not isinstance(state, dict) or state.get('count') not in (1,2,3):
            return
        # Toggle first, then restore exact saved numbers without Auto replacing them.
        owner.drr_region_count_combo.setCurrentIndex(state['count']-1)
        for prefix in ('drr', 'drr_second'):
            for key, spin in getattr(owner, f'{prefix}_split_spins').items():
                value = float(state.get(prefix, {}).get(key, spin.value()))
                if np.isfinite(value):
                    owner._set_spin_value_silent(spin, value)
            for key, check in getattr(owner, f'{prefix}_split_fix_checks').items():
                check.setChecked(bool(state.get(prefix+'_fix', {}).get(key, False)))
        blocked = owner.drr_split_boundary_chk.blockSignals(True)
        owner.drr_split_boundary_chk.setChecked(bool(state.get('boundary', True)))
        owner.drr_split_boundary_chk.blockSignals(blocked)
    except (ValueError, TypeError, AttributeError):
        return
