"""Text-only mouse coordinates for the independent Peak Analysis canvas."""
import re
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QSizePolicy

from core.plotting import _axis_edges_from_centers


def _number(value):
    if not np.isfinite(value):
        return '—'
    return f'{value:.6g}' if value else '0'


def _scan_label(cube):
    label, unit = cube.gate_label.strip(), cube.gate_unit.strip()
    if unit and not re.search(rf'(?<!\w){re.escape(unit)}(?!\w)', label):
        label = f'{label} ({unit})'
    return label or 'Scan coordinate'


class PeakCursorReadout(QLabel):
    """Coalesce mouse events; never mutate analysis data or draw the canvas."""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.setTextFormat(Qt.PlainText)
        self.setAccessibleName('Plot cursor coordinates')
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setMinimumWidth(0)
        self.setMargin(3)
        self._cube = None
        self._axes = {}
        self._pending = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(34)  # At most about 30 text updates per second.
        self._timer.timeout.connect(self._flush)
        canvas.mpl_connect('motion_notify_event', self._motion)
        canvas.mpl_connect('axes_leave_event', self._leave)
        canvas.mpl_connect('figure_leave_event', self._leave)
        canvas.mpl_connect('draw_event', self.clear_reading)
        self.clear_reading()

    def bind(self, cube=None, heat_ax=None, spectrum_ax=None, residual_ax=None):
        """Refresh references after axes replacement, retaining original row IDs."""
        self.clear_reading()
        self._cube = cube
        self._axes = {ax: name for ax, name in ((heat_ax, 'Heatmap'),
                      (spectrum_ax, 'Spectrum'), (residual_ax, 'Residual')) if ax is not None}
        if cube is None or heat_ax is None:
            return
        self._scan_label = _scan_label(cube)
        log_y = heat_ax.get_yscale() == 'log'
        self._rows = np.argsort(cube.gate)
        if log_y:
            self._rows = self._rows[cube.gate[self._rows] > 0]
        # Use the same cell boundaries as the mesh, including geometric edges
        # on log scan axes. Coordinates and values remain in physical units.
        self._x_edges = _axis_edges_from_centers(cube.energy)
        self._y_edges = _axis_edges_from_centers(cube.gate[self._rows], log_scale=log_y)

    def clear_reading(self, *_, keep_details=False):
        self._timer.stop()
        self._pending = None
        self.setText('Cursor · X: — · Y: — · Z: —')
        if not keep_details:
            self._last_tip = ''
        if self._last_tip:
            self.setToolTip('Last cursor reading (pointer outside the plot)\n' + self._last_tip)
        else:
            self.setToolTip('Move over a plot to read mouse X/Y coordinates. '
                            'On the heatmap, Z is the nearest displayed grid cell’s input value. '
                            'Hovering does not select a spectrum or rerun analysis.')

    def _leave(self, *_):
        # Keep details explicitly marked as historical so they can be inspected
        # by moving onto this label. Data/row changes clear them entirely.
        self.clear_reading(keep_details=True)

    def _motion(self, event):
        if (self._cube is None or event.inaxes not in self._axes
                or event.xdata is None or event.ydata is None
                or not np.isfinite([event.xdata, event.ydata]).all()):
            self._leave()
            return
        self._pending = (event.inaxes, float(event.xdata), float(event.ydata))
        if not self._timer.isActive():
            self._timer.start()

    def _cell(self, x, y):
        if not (self._x_edges[0] <= x <= self._x_edges[-1]
                and self._y_edges[0] <= y <= self._y_edges[-1]):
            return None
        col = min(int(np.searchsorted(self._x_edges, x, side='right')) - 1, len(self._x_edges) - 2)
        row = min(int(np.searchsorted(self._y_edges, y, side='right')) - 1, len(self._rows) - 1)
        return int(self._rows[row]), col

    def _flush(self):
        pending, self._pending = self._pending, None
        if pending is None:
            return
        axes, x, y = pending
        name = self._axes.get(axes)
        if name is None or self._cube is None:
            self.clear_reading()
            return
        cube = self._cube
        text = f'{name} · X: {x:.5f} eV'
        tip = f'{name}\nMouse X — Energy (eV): {_number(x)}'
        if name == 'Heatmap':
            cell = self._cell(x, y)
            z = float(cube.Z[cell]) if cell is not None else np.nan
            text += f' · Y ({self._scan_label}): {_number(y)} · Z (nearest): {_number(z)}'
            tip += f'\nMouse Y — {self._scan_label}: {_number(y)}\nZ — {cube.cbar_label}: {_number(z)}'
            if cell is not None:
                row, col = cell
                tip += (f'\nNearest cell: Energy = {_number(cube.energy[col])} eV; '
                        f'{self._scan_label} = {_number(cube.gate[row])}')
                if not np.isfinite(z):
                    tip += '\nThis cell has no finite value.'
            else:
                tip += '\nOutside the heatmap grid.'
            tip += ('\nZ is the input value before color clipping/normalization and any extra detection smoothing. '
                    'On a log scan axis, the nearest cell follows the log spacing.')
        else:
            text += f' · Y ({cube.cbar_label}): {_number(y)}'
            tip += f'\nMouse Y — {cube.cbar_label}: {_number(y)}\nMouse coordinates; not snapped to a curve.'
        self.setText(text)
        self._last_tip = tip
        self.setToolTip(tip)
