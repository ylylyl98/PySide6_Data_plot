"""On-demand, non-modal source inspection; never modifies the DRR workflow."""
import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QDialogButtonBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from core.drr_raw_preview import load_preview
from core.plotting import HeatmapParams, plot_heatmap, downsample_cube_for_display
from core.processing import compute_auto_limits
from ui_qt.common import Worker, QComboBox, QDoubleSpinBox
from ui_qt.matplotlib_theme import QtFontFigureCanvasQTAgg


def _read_request(snapshot, token, role, scope, source, *, progress, log):
    try:
        return token, load_preview(snapshot, role, scope, source), None
    except Exception as exc:
        return token, None, str(exc)


class DrrRawDialog(QDialog):
    def __init__(self, snapshot, *, pool, initial_gate=0., parent=None):
        super().__init__(parent)
        self.setWindowTitle('DRR · Raw data inspection')
        self.resize(1050, 820)
        self.snapshot = snapshot
        self.pool = pool
        self.preview = None
        self._generation = 0
        self._busy = False
        self._pending = None
        self._closed = False
        self._preferred_gate = float(initial_gate)
        self.heatmap_ax = None
        layout = QVBoxLayout(self)
        self.info = QLabel('Sources from the loaded DRR result. Viewing does not change its processing or export settings.')
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        row = QHBoxLayout()
        self.role_combo = QComboBox()
        self.role_combo.addItem('Measurement', 'measurement')
        self.role_combo.addItem('Background', 'background')
        self.scope_combo = QComboBox()
        self.scope_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.scope_combo.setMinimumContentsLength(22)
        self.scope_combo.addItem('Single file', 'single')
        self.scope_combo.addItem('Average raw files', 'average')
        self.scope_combo.setCurrentIndex(1 if len(snapshot.measurements) > 1 else 0)
        self.file_combo = QComboBox()
        self.file_combo.setMinimumContentsLength(24)
        self.file_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.refresh_button = QPushButton('Reload')
        row.addWidget(self.role_combo)
        row.addWidget(self.scope_combo)
        row.addWidget(self.file_combo, 1)
        row.addWidget(self.refresh_button)
        layout.addLayout(row)
        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        self.source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.source_label)
        gate_row = QHBoxLayout()
        self.gate_label = QLabel('Gate / frame')
        self.gate_spin = QDoubleSpinBox()
        self.gate_spin.setDecimals(6)
        self.gate_spin.setRange(-1e12, 1e12)
        self.gate_spin.setValue(initial_gate)
        self.gate_spin.setKeyboardTracking(False)
        self.gate_spin.setEnabled(False)
        gate_row.addWidget(self.gate_label)
        gate_row.addWidget(self.gate_spin)
        gate_row.addStretch(1)
        layout.addLayout(gate_row)
        self.figure = Figure(constrained_layout=True)
        self.canvas = QtFontFigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.role_combo.currentIndexChanged.connect(self._role_changed)
        self.scope_combo.currentIndexChanged.connect(self.request_preview)
        self.file_combo.currentIndexChanged.connect(self.request_preview)
        self.refresh_button.clicked.connect(self.request_preview)
        self.gate_spin.valueChanged.connect(self._gate_changed)
        self.canvas.mpl_connect('button_press_event', self._canvas_click)
        self.finished.connect(self._stop)
        self._populate_files()
        QTimer.singleShot(0, self.request_preview)

    def _populate_files(self):
        sources = self.snapshot.measurements if self.role_combo.currentData() == 'measurement' else self.snapshot.backgrounds
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        for source in sources:
            self.file_combo.addItem(source, source)
            self.file_combo.setItemData(self.file_combo.count()-1, source, Qt.ToolTipRole)
        self.file_combo.blockSignals(False)

    def _role_changed(self):
        scope = self.scope_combo.currentData()
        self.scope_combo.blockSignals(True)
        self.scope_combo.clear()
        self.scope_combo.addItem('Single file', 'single')
        self.scope_combo.addItem('Average raw files', 'average')
        if self.role_combo.currentData() == 'background':
            self.scope_combo.addItem('Reference used (mean)', 'reference')
        self.scope_combo.setCurrentIndex(max(0, self.scope_combo.findData(scope)))
        self.scope_combo.blockSignals(False)
        self._populate_files()
        self.request_preview()

    @Slot()
    def request_preview(self):
        if self._closed:
            return
        self._generation += 1
        role, scope, source = self.role_combo.currentData(), self.scope_combo.currentData(), self.file_combo.currentData()
        self.file_combo.setEnabled(scope == 'single')
        sources = self.snapshot.measurements if role == 'measurement' else self.snapshot.backgrounds
        self.source_label.setText((source or 'No source file') if scope == 'single' else f'{len(sources)} source files from the loaded DRR result')
        self.source_label.setToolTip('\n'.join(sources))
        self.preview = None
        self.gate_spin.setEnabled(False)
        self.figure.clear()
        self.heatmap_ax = None
        self.canvas.draw_idle()
        self.status.setText('Loading raw data…')
        self._pending = (self._generation, role, scope, source)
        if not self._busy:
            self._start_pending()

    def _start_pending(self):
        if self._pending is None or self._closed:
            return
        request, self._pending = self._pending, None
        self._busy = True
        self._worker = Worker(_read_request, self.snapshot, *request)
        self._worker.signals.result.connect(self._loaded, Qt.QueuedConnection)
        self._worker.signals.finished.connect(self._finished, Qt.QueuedConnection)
        self.pool.start(self._worker)

    @Slot(object)
    def _loaded(self, payload):
        token, preview, error = payload
        if self._closed or token != self._generation:
            return
        if error:
            self.status.setText('Unable to inspect raw data: ' + error)
            return
        try:
            self.preview = preview
            self._draw()
            self.status.setText(preview.description)
        except (ValueError, TypeError) as exc:
            self.preview = None
            self.status.setText('Unable to display raw data: ' + str(exc))

    @Slot()
    def _finished(self):
        self._busy = False
        self._worker = None
        self._start_pending()

    def _draw(self):
        cube = self.preview.cube
        self.figure.clear()
        self.heatmap_ax = None
        self.gate_line = None
        if self.preview.spectrum_only:
            self.spectrum_ax = self.figure.add_subplot(111)
        else:
            grid = self.figure.add_gridspec(2, 2, width_ratios=[1, .035], height_ratios=[1.2, 1])
            self.heatmap_ax = self.figure.add_subplot(grid[0, 0])
            cax = self.figure.add_subplot(grid[0, 1])
            self.spectrum_ax = self.figure.add_subplot(grid[1, 0], sharex=self.heatmap_ax)
            bounds = compute_auto_limits(cube)
            params = HeatmapParams(cube.title, 'Photon energy (eV)', cube.gate_label, cube.cbar_label,
                bounds.vmin, bounds.vmax, (bounds.xmin, bounds.xmax), (bounds.ymin, bounds.ymax))
            render = plot_heatmap(self.heatmap_ax, downsample_cube_for_display(cube), params)
            self.figure.colorbar(render.primary, cax=cax, label=cube.cbar_label)
            self.gate_line = self.heatmap_ax.axhline(self._preferred_gate, color='#dd7733', linestyle='--')
        self.gate_label.setText(cube.gate_label)
        self.gate_spin.setEnabled(not self.preview.spectrum_only)
        self.spectrum_line, = self.spectrum_ax.plot([], [])
        self.spectrum_ax.set_xlabel('Photon energy (eV)')
        self.spectrum_ax.set_ylabel(cube.cbar_label)
        self.spectrum_ax.grid(alpha=.2)
        self._gate_changed(self._preferred_gate)
        self.toolbar.update()

    def _gate_changed(self, value):
        if self.preview is None:
            return
        cube = self.preview.cube
        index = int(np.argmin(np.abs(cube.gate - value)))
        gate = float(cube.gate[index])
        if not self.preview.spectrum_only:
            self._preferred_gate = gate
        blocked = self.gate_spin.blockSignals(True)
        self.gate_spin.setValue(gate)
        self.gate_spin.blockSignals(blocked)
        self.spectrum_line.set_data(cube.energy, cube.Z[index])
        self.spectrum_ax.relim()
        self.spectrum_ax.autoscale_view(scalex=self.heatmap_ax is None)
        self.spectrum_ax.set_title(cube.title if self.preview.spectrum_only else f'{cube.gate_label} = {gate:.6g}')
        if self.gate_line is not None:
            self.gate_line.set_ydata([gate, gate])
        self.canvas.draw_idle()

    def _canvas_click(self, event):
        if event.inaxes is self.heatmap_ax and event.ydata is not None and not self.toolbar.mode:
            self._gate_changed(float(event.ydata))

    @Slot()
    def _stop(self):
        self._closed = True
        self._generation += 1
        self._pending = None
        self.preview = None

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)
