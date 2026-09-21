"""Optional DRR three-region controls, automatic limits, and saved state."""
import json
import numpy as np
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QToolButton
from core.plotting import SplitColorScale, split_region_slices

SETTINGS_KEY = 'drr/region_colors_json'


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
    grid.addWidget(QLabel('Boundary 1'), 0, 0)
    grid.addWidget(spins['x0'], 0, 1, 1, 2)
    grid.addWidget(fixes['x0'], 0, 3)
    grid.addWidget(boundary, 1, 0, 1, 4)
    for column, widget in ((0, QLabel('Boundary 2')), (1, spins['x1']), (3, fixes['x1'])):
        grid.addWidget(widget, 2, column, 1, 2 if column == 1 else 1)
        extra.append(widget)
    if prefix == 'drr_second':
        for widget in extra:
            widget.hide()
        extra = []  # Boundaries are shared; no duplicate controls in d2E.
    for row, side, auto in ((3, 'left', left_auto), (6, 'middle', middle_auto), (9, 'right', right_auto)):
        title = QLabel(side.title())
        grid.addWidget(title, row, 0, 1, 3)
        grid.addWidget(auto, row, 3)
        widgets = [title, auto]
        for offset, bound in ((1, 'min'), (2, 'max')):
            key = f'{side}_v{bound}'
            label = QLabel(bound)
            grid.addWidget(label, row + offset, 0)
            grid.addWidget(spins[key], row + offset, 1, 1, 2)
            grid.addWidget(fixes[key], row + offset, 3)
            widgets.extend((label, spins[key], fixes[key]))
        if side == 'middle':
            extra.extend(widgets)
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
            if not fixes[key].isChecked():
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
    if center:
        initialize_boundaries(owner, center=True)
    spins = getattr(owner, f'{prefix}_split_spins')
    fixes = getattr(owner, f'{prefix}_split_fix_checks')
    shared = owner.drr_split_spins
    xlim = tuple(owner.drr_spins[k].value() for k in ('xmin', 'xmax'))
    ylim = sorted(owner.drr_spins[k].value() for k in ('ymin', 'ymax'))
    shape = SplitColorScale(shared['x0'].value(), 0, 1, 0, 1,
                            split_x2=shared['x1'].value(), middle_vmin=0, middle_vmax=1)
    values = {s: [] for s in ('left', 'middle', 'right')}
    for cube in cubes:
        try:
            slices, _ = split_region_slices(cube.energy, shape, xlim)
        except ValueError as exc:
            owner._status(str(exc))
            return False
        x, y = np.asarray(cube.energy), np.asarray(cube.gate)
        for s, region in zip(values, slices):
            mask = np.zeros(x.size, bool)
            mask[region] = True
            mask &= (x >= min(xlim)) & (x <= max(xlim))
            z = np.asarray(cube.Z)[np.ix_((y >= ylim[0]) & (y <= ylim[1]), mask)]
            z = z[np.isfinite(z)]
            if owner._mode_log('DRR'):
                z = z[z > 0]
            if z.size:
                values[s].append(z)
    changed = False
    for s, chunks in values.items():
        if (side is not None and s != side) or not chunks:
            continue
        lo, hi = np.percentile(np.concatenate(chunks), [0.01, 99.99])
        if owner.drr_center_zero_chk.isChecked() and not owner._mode_log('DRR'):
            hi = max(abs(lo), abs(hi), 1e-12)
            lo = -hi
        if hi <= lo:
            pad = max(1e-12, abs(lo)*.01)
            lo, hi = lo-pad, hi+pad
        if owner._mode_log('DRR'):
            lo = max(lo, 1e-12)
            hi = max(hi, lo*1.01)
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


def save_settings(owner):
    state = {'count': owner.drr_region_count_combo.currentData(),
             'boundary': owner.drr_split_boundary_chk.isChecked()}
    for prefix in ('drr', 'drr_second'):
        state[prefix] = {key: spin.value() for key, spin in getattr(owner, f'{prefix}_split_spins').items()}
        state[prefix+'_fix'] = {key: check.isChecked() for key, check in getattr(owner, f'{prefix}_split_fix_checks').items()}
    owner.settings.setValue(SETTINGS_KEY, json.dumps(state))


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
