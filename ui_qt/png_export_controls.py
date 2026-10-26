"""Shared, remembered PNG layout and wide-export multiplier."""
from dataclasses import replace

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

from core.png_geometry import widened_figsize


def export_width(owner):
    mode = getattr(owner, 'png_layout_combo', None)
    if mode is None or mode.currentData() != 'wide':
        return 1.0
    spin = getattr(owner, 'png_width_spin', None)
    return spin.value() if spin is not None else 2.0


def install_png_width_control(owner, toolbar):
    mode = QComboBox(toolbar)
    mode.setObjectName('png_layout_combo')
    mode.setAccessibleName('PNG export layout')
    mode.addItem('PNG 1.0×', 'original')
    mode.addItem('Wide PNG', 'wide')
    layout = str(owner.settings.value('export/png_layout', 'original'))
    mode.setCurrentIndex(1 if layout == 'wide' else 0)
    mode.setToolTip('PNG 1.0× saves only the original width. For DRR, Wide PNG saves both 1.0× and the selected wider version.')
    mode.setAccessibleDescription(mode.toolTip())
    spin = QDoubleSpinBox(toolbar)
    spin.setObjectName('png_width_spin')
    spin.setAccessibleName('PNG width multiplier')
    spin.setRange(1.0, 3.0)
    spin.setDecimals(2)
    spin.setSingleStep(0.25)
    spin.setSuffix(' ×')
    spin.setKeyboardTracking(False)
    try:
        value = float(owner.settings.value('export/png_width_scale', 2.0))
        widened_figsize((1, 1), value)
        if value == 1.0:
            value = 2.0
    except (TypeError, ValueError):
        value = 2.0
    spin.setValue(value)
    spin.setEnabled(mode.currentData() == 'wide')
    tooltip = ('Wide PNG multiplier (default 2×). Select Wide PNG to enable it; '
               'height, font sizes and data ranges stay unchanged. Applies to main-plot '
               'PNG saves, PNG + DAT heatmaps and MCD Save Results. The on-screen plot stays unchanged.')
    spin.setToolTip(tooltip)
    spin.setAccessibleDescription(tooltip)
    # Keep this export control next to Save, before Matplotlib's expanding
    # cursor readout can push it into the overflow at smaller window widths.
    cursor = getattr(toolbar, 'locLabel', None)
    before = next((action for action in toolbar.actions()
                   if cursor is not None and toolbar.widgetForAction(action) is cursor), None)
    for widget in (mode, spin):
        if before is None:
            toolbar.addWidget(widget)
        else:
            toolbar.insertWidget(before, widget)
    owner.png_layout_combo = mode
    owner.png_width_spin = spin
    mode.currentIndexChanged.connect(lambda _index: spin.setEnabled(mode.currentData() == 'wide'))
    mode.currentIndexChanged.connect(lambda _index: owner.settings.setValue('export/png_layout', mode.currentData()))
    spin.valueChanged.connect(lambda value: owner.settings.setValue('export/png_width_scale', value))


def apply_export_width(options, width_scale):
    """Freeze the preference on every product before starting the worker."""
    widened_figsize((1, 1), width_scale)
    fields = ('params', 'params_linear', 'params_log', 'params_intensity',
              'drr_raw_params', 'drr_second_params')
    return replace(options, **{
        name: replace(value, png_width_scale=width_scale)
        for name in fields if (value := getattr(options, name)) is not None
    })
