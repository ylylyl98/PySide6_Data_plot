"""Range-scoped DRR analysis UI; scientific/export code lives in core."""
from __future__ import annotations

from copy import deepcopy
from threading import Event

import numpy as np
from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QPushButton, QComboBox,
    QDoubleSpinBox, QSpinBox, QLabel, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView,
)

from ui_qt.common import Worker
from ui_qt.compact_spinbox import CompactDoubleSpinBox as QDoubleSpinBox


class DrrPeakAnalysisController(QObject):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.result = None
        self.result_key = None
        self.generation = 0
        self.workers = []
        self.cancel_event = None
        self.artists = []
        self.loaded_cube = None
        self.results_widget = QWidget()
        results = QVBoxLayout(self.results_widget)
        self.summary = QLabel('Analyze a selected X/Y range to find peak positions.')
        self.summary.setWordWrap(True)
        results.addWidget(self.summary)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['Source', 'Y', 'Track', 'Type', 'Energy (eV)', 'Status'])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.inspect_selected)
        self.table.setMinimumHeight(120)
        results.addWidget(self.table, 1)
        delete = QPushButton('Exclude Selected Peak')
        delete.clicked.connect(self.exclude_selected)
        results.addWidget(delete, 0, Qt.AlignRight)
        self.points = []

    def build_controls(self):
        panel = QWidget()
        self.panel = panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        form = QFormLayout()
        self.range_source_form=form
        self.source = QComboBox()
        for text, data in [('Both: DRR + d²/dE²', 'both'), ('Original DRR', 'raw'), ('Second derivative', 'second')]:
            self.source.addItem(text, data)
        form.addRow('Analyze', self.source)
        self.mode = QComboBox()
        self.mode.addItem('Target branch (seed required)', 'seed')
        self.mode.addItem('All candidates (exploratory)', 'all')
        # Preserve existing initial source settings; picking a seed selects its product.
        self.mode.setCurrentIndex(1)
        form.addRow('Mode', self.mode)
        self.seed_energy = QDoubleSpinBox(); self.seed_energy.setDecimals(7); self.seed_energy.setRange(-1e9, 1e9)
        self.seed_energy.display_precision=4
        self.seed_y = QDoubleSpinBox(); self.seed_y.setDecimals(7); self.seed_y.setRange(-1e9, 1e9)
        form.addRow('Seed E (eV)', self.seed_energy)
        form.addRow('Seed Y', self.seed_y)
        self.seed_button = QPushButton('Pick target branch on colorplot')
        self.seed_button.setCheckable(True)
        self.seed_button.setToolTip('Click a clear peak or dip on a DRR heatmap. The source is selected from that panel. Choose Peaks or Dips explicitly.')
        form.addRow(self.seed_button)
        if hasattr(self.owner, 'canvas'):
            self.owner.canvas.mpl_connect('button_press_event', self.capture_seed)
        self.bounds = {}
        self.range_form=form;self.range_rows=[]
        for name, label in [('x', 'E (eV)'), ('y', 'Y range')]:
            row = QWidget(); row_layout = QHBoxLayout(row); row_layout.setContentsMargins(0, 0, 0, 0)
            for suffix in ('min', 'max'):
                spin = QDoubleSpinBox(); spin.setDecimals(9); spin.setRange(-1e9, 1e9)
                spin.display_precision=4 if name=='x' else 3
                spin.setAccessibleName(f'Analysis {name.upper()} {suffix}')
                spin.setMinimumWidth(0)
                self.bounds[f'{name}_{suffix}'] = spin
                if suffix == 'max': row_layout.addWidget(QLabel('to'))
                row_layout.addWidget(spin)
                spin.valueChanged.connect(self.changed)
            form.addRow(label, row)
            self.range_rows.append(row)
        layout.addLayout(form)
        row = QHBoxLayout()
        view = QPushButton('Use Current View'); view.clicked.connect(self.use_view)
        full = QPushButton('Full Range'); full.clicked.connect(self.full_range)
        self.range_buttons=(view,full)
        row.addWidget(view); row.addWidget(full); layout.addLayout(row)
        self.count = QLabel('Load DRR data first.'); self.count.setWordWrap(True); layout.addWidget(self.count)
        self.analyze_button = QPushButton('Analyze Selected Range')
        self.analyze_button.clicked.connect(self.analyze); layout.addWidget(self.analyze_button)
        row = QHBoxLayout()
        self.show = QCheckBox('Show peak positions'); self.show.setChecked(True)
        self.show.toggled.connect(self.redraw)
        clear = QPushButton('Clear'); clear.clicked.connect(self.clear)
        row.addWidget(self.show); row.addWidget(clear); layout.addLayout(row)
        advanced = QWidget(); adv = QFormLayout(advanced)
        self.detection_form=adv
        self.polarity = QComboBox()
        for text, value in [('Peaks and dips', 'both'), ('Peaks', 'peaks'), ('Dips', 'dips')]:
            self.polarity.addItem(text, value)
        form.addRow('Feature', self.polarity)
        self.prominence = QDoubleSpinBox(); self.prominence.setRange(0, 1); self.prominence.setDecimals(3); self.prominence.setValue(.05); self.prominence.setSingleStep(.01)
        self.distance = QDoubleSpinBox(); self.distance.setRange(.001, 1000); self.distance.setDecimals(3); self.distance.setValue(1)
        self.shift = QDoubleSpinBox(); self.shift.setRange(.001, 1000); self.shift.setDecimals(3); self.shift.setValue(3)
        self.maximum = QSpinBox(); self.maximum.setRange(1, 50); self.maximum.setValue(6)
        adv.addRow('Prominence fraction', self.prominence)
        adv.addRow('Min spacing (meV)', self.distance)
        adv.addRow('Max Y-step shift (meV)', self.shift)
        adv.addRow('Max peaks / row', self.maximum)
        self.noise = QDoubleSpinBox(); self.noise.setRange(1, 20); self.noise.setValue(4)
        self.width = QDoubleSpinBox(); self.width.setDecimals(2); self.width.setRange(.01, 100); self.width.setValue(1)
        self.gaps = QSpinBox(); self.gaps.setRange(0, 10); self.gaps.setValue(2)
        adv.addRow('Target noise multiplier', self.noise)
        adv.addRow('Target min width (meV)', self.width)
        adv.addRow('Target max missing rows', self.gaps)
        if hasattr(self.owner, '_make_expander'):
            self.detection_section=self.owner._make_expander('Detection settings', advanced, expanded=False)
            layout.addWidget(self.detection_section)
        else:
            layout.addWidget(advanced)
        for control in (self.source, self.polarity): control.currentIndexChanged.connect(self.changed)
        for control in (self.prominence, self.distance, self.shift, self.maximum): control.valueChanged.connect(self.changed)
        for control in (self.owner.drr_sg_window_spin, self.owner.drr_sg_poly_spin): control.valueChanged.connect(self.changed)
        for control in (self.owner.drr_baseline_combo, self.owner.drr_baseline_combine_combo): control.currentTextChanged.connect(self.changed)
        self.mode.currentIndexChanged.connect(self.changed)
        for control in (self.seed_energy, self.seed_y, self.noise, self.width, self.gaps): control.valueChanged.connect(self.changed)
        self.analyze_button.setEnabled(False)
        return panel

    def capture_seed(self, event):
        if not self.seed_button.isChecked() or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        axes = getattr(self.owner, '_drr_heatmap_axes', {})
        source = next((key for key, axis in axes.items() if axis is event.inaxes), None)
        if source not in ('raw', 'second'): return
        event._drr_seed_consumed = True
        self.seed_energy.setValue(float(event.xdata)); self.seed_y.setValue(float(event.ydata))
        self.source.setCurrentIndex(self.source.findData(source))
        self.mode.setCurrentIndex(self.mode.findData('seed'))
        self.seed_button.setChecked(False)
        self.summary.setText('Seed selected. Check Peaks/Dips and X/Y range, then Analyze Selected Range.')

    def cube(self):
        loaded = getattr(self.owner, 'loaded', None)
        return loaded.cube if loaded is not None and loaded.mode == 'DRR' else None

    def settings(self):
        from core.drr_peak_analysis import PeakAnalysisSettings
        return PeakAnalysisSettings(**{key: spin.value() for key, spin in self.bounds.items()},
            source=self.source.currentData(), polarity=self.polarity.currentData(),
            prominence=self.prominence.value(), min_distance_mev=self.distance.value(),
            max_peaks=self.maximum.value(), max_shift_mev=self.shift.value(),
            sg_window=int(self.owner.drr_sg_window_spin.value()),
            sg_polyorder=int(self.owner.drr_sg_poly_spin.value()),
            mode=self.mode.currentData(), seed_energy=self.seed_energy.value(), seed_y=self.seed_y.value(),
            noise_sigma=self.noise.value(), min_width_mev=self.width.value(), max_gap=self.gaps.value())

    def key(self):
        return (id(self.cube()), repr(self.settings()),
                tuple(getattr(self.owner, 'drr_selected_files', ())),
                tuple(getattr(self.owner, 'drr_baseline_files_manual', ())),
                self.owner.drr_baseline_combo.currentText(), self.owner.drr_baseline_combine_combo.currentText())

    def selected_count(self):
        cube = self.cube()
        if cube is None: return 0
        gate = np.asarray(cube.gate)
        return int(np.count_nonzero(np.isfinite(gate) & (gate >= self.bounds['y_min'].value()) & (gate <= self.bounds['y_max'].value())))

    def loaded_changed(self):
        cube = self.cube()
        if cube is not self.loaded_cube:
            self.loaded_cube = cube
            self.clear()
            if cube is not None: self.full_range()
        self.update_count()

    def set_bounds(self, x, y):
        for prefix, limits in [('x', x), ('y', y)]:
            low, high = sorted(limits)
            # Round outwards so Full Range includes the actual endpoint samples.
            for suffix, value in zip(('min', 'max'), (np.floor(low * 1e9) / 1e9, np.ceil(high * 1e9) / 1e9)):
                spin = self.bounds[f'{prefix}_{suffix}']; old = spin.blockSignals(True)
                spin.setValue(float(value)); spin.blockSignals(old)
        self.changed()

    def full_range(self):
        cube = self.cube()
        if cube is not None:
            self.set_bounds((np.nanmin(cube.energy), np.nanmax(cube.energy)), (np.nanmin(cube.gate), np.nanmax(cube.gate)))

    def use_view(self):
        axis = getattr(self.owner, '_drr_heatmap_ax', None)
        view = (axis.get_xlim(), axis.get_ylim()) if axis is not None else getattr(self.owner, '_drr_view_limits', None)
        if view is not None: self.set_bounds(*view)

    def update_count(self):
        count = self.selected_count()
        cube = self.cube()
        self.count.setText(f'{count} spectra · {cube.gate_label}' if cube is not None else 'Load DRR data first.')
        self.analyze_button.setEnabled(count > 0 and not self.workers)

    def changed(self, *_):
        self.generation += 1
        if self.cancel_event is not None: self.cancel_event.set()
        if self.result is not None:
            self.summary.setText('Analysis is out of date. Analyze again before exporting peak results.')
        self.update_count()
        self.redraw()

    def clear(self):
        self.generation += 1
        if self.cancel_event is not None: self.cancel_event.set()
        self.result = None; self.result_key = None; self.points = []
        self.table.setRowCount(0)
        self.summary.setText('No peak analysis. Original PNG/DAT exports remain unchanged.')
        self.redraw()

    def valid_result(self):
        if self.result is not None and self.result_key == self.key() and not getattr(self.owner, '_load_in_progress', False):
            return self.result
        return None

    def analyze(self):
        cube = self.cube()
        if cube is None or self.workers or getattr(self.owner, '_load_in_progress', False): return
        from core.drr_peak_analysis import analyze_drr_peaks
        settings = self.settings(); key = self.key()
        if settings.mode == 'seed' and (settings.source == 'both' or settings.polarity == 'both'):
            self.summary.setText('Target branch needs one source and an explicit Feature: Peaks or Dips.')
            if hasattr(self.owner, 'results_dock'): self.owner.results_dock.show()
            return
        self.generation += 1; generation = self.generation
        cancellation = Event(); self.cancel_event = cancellation
        def run(*, progress, log):
            return analyze_drr_peaks(cube, settings, cancelled=cancellation.is_set)
        worker = Worker(run)
        worker.signals.result.connect(lambda result: self.finished(result, key, generation))
        worker.signals.error.connect(lambda message: self.failed(message, generation))
        worker.signals.finished.connect(lambda: self.release(worker))
        self.workers.append(worker)
        self.summary.setText(f'Analyzing {self.selected_count()} spectra…')
        self.update_count()
        if hasattr(self.owner, 'results_dock'): self.owner.results_dock.show()
        self.owner.thread_pool.start(worker)

    def release(self, worker):
        if worker in self.workers: self.workers.remove(worker)
        if not getattr(self.owner, '_is_closing', False): self.update_count()

    def failed(self, message, generation):
        if generation == self.generation:
            self.summary.setText('Analysis failed: ' + str(message).split('Traceback')[0])

    def finished(self, result, key, generation):
        if getattr(self.owner, '_is_closing', False) or generation != self.generation or key != self.key(): return
        loaded = self.owner.loaded
        result = deepcopy(result)
        result['provenance'] = {
            'measurement_files': list(loaded.selected_files), 'background_files': list(loaded.baseline_files),
            'background_mode': loaded.drr_baseline_text, 'background_frame': loaded.drr_baseline_which,
            'y_axis': loaded.y_axis_spec,
            'assignments': [a.to_dict() for a in getattr(loaded, 'drr_assignments', ())],
        }
        self.result = result; self.result_key = key
        self.points = [(source, point) for source, product in result['products'].items() for point in product['points']]
        self.populate_table()
        self.summary.setText(f'{len(self.points)} peak positions in {len(result["y_values"])} spectra. Save PNG/DAT also saves separate XLSX, JSON and analysis PNG.')
        if result['settings'].get('mode') == 'seed':
            found = len({p['row_index'] for _, p in self.points})
            self.summary.setText(f'Target branch: {found}/{len(result["y_values"])} rows matched. Missing rows remain blank; no forced continuation. Separate analysis files are included on export.')
        self.redraw()

    def populate_table(self):
        old = self.table.blockSignals(True)
        self.table.setRowCount(len(self.points))
        for row, (source, p) in enumerate(self.points):
            for col, value in enumerate((source, f'{p["y"]:.6g}', str(p['track_id']), p['polarity'], f'{p["energy"]:.7f}', p['status'])):
                self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.blockSignals(old)

    def inspect_selected(self):
        row = self.table.currentRow()
        if self.valid_result() is None or not 0 <= row < len(self.points): return
        self.owner.drr_controller._set_drr_gate_spin_value(self.points[row][1]['y'])
        cube = getattr(self.owner, '_last_plot_cube', None)
        if cube is not None: self.owner.drr_controller._update_drr_spectrum_and_gate_line(cube)

    def exclude_selected(self):
        row = self.table.currentRow()
        if self.valid_result() is None or not 0 <= row < len(self.points): return
        source, point = self.points[row]
        self.result.setdefault('excluded_points', []).append({'source': source, **deepcopy(point)})
        self.result['products'][source]['points'].remove(point)
        self.points.pop(row)
        self.populate_table()
        self.summary.setText(f'{len(self.points)} peak positions. Selected point excluded; its track has a gap.')
        self.redraw()

    def snapshot(self):
        result = self.valid_result()
        if result is None and self.result is not None and hasattr(self.owner, '_status'):
            self.owner._status('Peak results are out of date; exporting clean PNG/DAT only. Analyze again to export peaks.')
        return deepcopy(result) if result is not None else None

    def redraw(self, *_):
        self.draw_overlays()
        if hasattr(self.owner, 'canvas'):
            for helper in getattr(self.owner, '_drr_region_blitters', {}).values(): helper._background = None
            self.owner.canvas.draw_idle()

    def draw_overlays(self):
        for artist in self.artists:
            try: artist.remove()
            except (ValueError, NotImplementedError): pass
        self.artists = []
        if self.mode.currentData() == 'seed' and self.show.isChecked():
            axis = getattr(self.owner, '_drr_heatmap_axes', {}).get(self.source.currentData())
            if axis is not None and hasattr(axis, 'scatter'):
                self.artists.append(axis.scatter([self.seed_energy.value()], [self.seed_y.value()],
                    marker='*', s=95, c='#ffe100', edgecolors='black', linewidths=.8, zorder=45))
        result = self.display_result()
        if result is None or not self.show.isChecked(): return
        for source, product in result['products'].items():
            axis = getattr(self.owner, '_drr_heatmap_axes', {}).get(source)
            if axis is None: continue
            points = self.visible_points(product['points'])
            for polarity, marker in [('peak', 'o'), ('dip', 'v')]:
                chosen = [p for p in points if p['polarity'] == polarity and p['status'] == 'accepted' and p.get('confidence')!='low']
                if chosen:
                    artist = axis.scatter([p['energy'] for p in chosen], [p['y'] for p in chosen],
                        s=12, marker=marker, facecolors='none', edgecolors='black', linewidths=.7, zorder=40, picker=5)
                    artist._drr_batch_points = chosen
                    artist._drr_batch_source = source
                    self.artists.append(artist)
            uncertain = [p for p in points if p['status'] == 'uncertain' or p.get('confidence')=='low']
            if uncertain:
                artist = axis.scatter([p['energy'] for p in uncertain], [p['y'] for p in uncertain],
                    marker='x', s=14, c='gray', linewidths=.7, zorder=40, picker=5)
                artist._drr_batch_points = uncertain; artist._drr_batch_source = source
                self.artists.append(artist)
            tracks={}
            for point in points:
                if point['status']=='accepted' and point.get('confidence')!='low':tracks.setdefault(point['track_id'],[]).append(point)
            for track,tracked in sorted(tracks.items()):
                if len(tracked) < 2: continue
                yy = sorted(result['y_values']); xx = np.full(len(yy), np.nan)
                mapping = {float(y): i for i, y in enumerate(yy)}
                for p in tracked: xx[mapping[float(p['y'])]] = p['energy']
                line, = axis.plot(xx, yy, color='black', linewidth=.6, alpha=.65, zorder=39)
                self.artists.append(line)
            spectrum = getattr(self.owner, '_drr_spectrum_axes', {}).get(source)
            if spectrum is not None:
                gate = self.owner.drr_controller._drr_gate_value()
                for p in points:
                    if abs(p['y']-gate)<=1e-9:
                        self.artists.append(spectrum.axvline(p['energy'], linewidth=.7, linestyle=':' if p['status'] == 'uncertain' else '--', color='black', alpha=.7))

    def visible_points(self,points):
        return points

    def display_result(self):
        return self.valid_result()

    def pick(self, event):
        if self.seed_button.isChecked(): return
        points = getattr(event.artist, '_drr_batch_points', None)
        if points and len(event.ind) and self.valid_result() is not None:
            point = points[int(event.ind[0])]
            source = event.artist._drr_batch_source
            for row, (candidate_source, candidate) in enumerate(self.points):
                if source == candidate_source and candidate is point:
                    self.table.selectRow(row)
                    self.inspect_selected()
                    break
