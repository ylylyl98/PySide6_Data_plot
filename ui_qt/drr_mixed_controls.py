"""DR/R mixed-derivative controls and independent X/Y window recommendations."""
import numpy as np
from PySide6.QtWidgets import QCheckBox, QLabel
from ui_qt.common import QComboBox, QSpinBox
from core.drr_mixed_derivative import recommend_window, effective_window, window_span


def enabled(owner):
    combo = getattr(owner, 'drr_second_kind_combo', None)
    return combo is not None and combo.currentData() == 'mixed'


def build(owner, form):
    owner.drr_second_kind_combo = QComboBox()
    owner.drr_second_kind_combo.addItem('X twice (d2E)', 'energy')
    owner.drr_second_kind_combo.addItem('X once + Y once (dXdY)', 'mixed')
    owner.drr_second_kind_combo.setAccessibleName('Second derivative direction')
    owner.drr_sg_y_window_spin = QSpinBox()
    owner.drr_sg_y_window_spin.setRange(5, 401)
    owner.drr_sg_y_window_spin.setSingleStep(2)
    owner.drr_sg_y_window_spin.setValue(7)
    owner.drr_sg_y_window_spin.setAccessibleName('Y derivative window points')
    owner.drr_sg_y_auto_chk = QCheckBox('Auto Y window')
    owner.drr_sg_y_auto_chk.setChecked(True)
    owner.drr_mixed_recommendation = QLabel('Select mixed derivative to see X/Y recommendations.')
    owner.drr_mixed_recommendation.setWordWrap(True)
    owner.drr_mixed_recommendation.setToolTip('Recommended starting points use noise estimates and detected feature widths. Inspect narrow peaks and adjust manually if needed. Spans are median measured window spans.')
    form.insertRow(0, 'Direction', owner.drr_second_kind_combo)
    form.addRow(owner.drr_sg_y_auto_chk)
    form.addRow('Y window points', owner.drr_sg_y_window_spin)
    form.addRow(owner.drr_mixed_recommendation)
    owner.drr_sg_y_auto_chk.setEnabled(False)
    owner.drr_sg_y_window_spin.setEnabled(False)


def update(owner):
    active = enabled(owner)
    owner.drr_sg_auto_chk.setToolTip(
        'Mixed mode: recommend an X window from noise and detected feature widths.' if active
        else 'Use 11 points for up to 512 energy samples, otherwise 21; limited to available samples.')
    owner.drr_sg_y_auto_chk.setEnabled(active)
    owner.drr_sg_y_window_spin.setEnabled(active)
    if not active:
        owner.drr_mixed_recommendation.setText('Mixed derivative uses the left Y axis; the right axis is display only.')
        return int(owner.drr_sg_window_spin.value())
    loaded = getattr(owner, 'loaded', None)
    cube = getattr(loaded, 'cube', None)
    if cube is None or loaded.mode != 'DRR':
        owner.drr_mixed_recommendation.setText('Load data for X/Y window recommendations.')
        return int(owner.drr_sg_window_spin.value())
    poly = int(owner.drr_sg_poly_spin.value())
    cached = getattr(owner, '_drr_mixed_recommendation_cache', None)
    if cached is None or cached[0] is not cube or cached[1] != poly:
        rx = recommend_window(cube.energy, cube.Z, poly, maximum=21)
        ry = recommend_window(cube.gate, np.asarray(cube.Z).T, poly, maximum=11)
        cached = (cube, poly, rx, ry)
        owner._drr_mixed_recommendation_cache = cached
    _, _, rx, ry = cached
    wx = effective_window(rx if owner.drr_sg_auto_chk.isChecked() else owner.drr_sg_window_spin.value(), len(cube.energy), poly)
    wy = effective_window(ry if owner.drr_sg_y_auto_chk.isChecked() else owner.drr_sg_y_window_spin.value(), len(cube.gate), poly)
    for spin, value in [(owner.drr_sg_window_spin, wx), (owner.drr_sg_y_window_spin, wy)]:
        blocked = spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(blocked)
    owner.drr_mixed_recommendation.setText(
        f'Recommended start: X {rx} points (~{window_span(cube.energy, rx)*1000:.3g} meV); '
        f'Y {ry} points (~{window_span(cube.gate, ry):.3g} {cube.gate_unit or "V"}).\n'
        f'Using X {wx}, Y {wy}. Y = {cube.gate_label}.')
    return wx


def derivative_key(owner):
    if not enabled(owner):
        return 2
    update(owner)
    return ('mixed', int(owner.drr_sg_y_window_spin.value()))


def changed(owner, *, manual_y=False):
    if manual_y:
        blocked = owner.drr_sg_y_auto_chk.blockSignals(True)
        owner.drr_sg_y_auto_chk.setChecked(False)
        owner.drr_sg_y_auto_chk.blockSignals(blocked)
    if hasattr(owner, 'drr_peak_analysis'):
        owner.drr_peak_analysis.changed()
    try:
        update(owner)
        owner.drr_controller._on_drr_derivative_changed()
    except ValueError as exc:
        owner.drr_mixed_recommendation.setText(str(exc))
        owner._status(str(exc))
