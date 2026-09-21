"""Independent range-amplitude view over a processed DRR cube."""
from pathlib import Path
import csv
import numpy as np
from PySide6.QtWidgets import QWidget,QVBoxLayout,QFormLayout,QHBoxLayout,QPushButton,QLabel,QFileDialog,QSpinBox
from matplotlib.figure import Figure
from ui_qt.compact_spinbox import CompactDoubleSpinBox
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
from core.drr_range_amplitude import range_amplitude


class RangeAmplitudeDialog(QWidget):
    def __init__(self,parent,cube,name,bounds,*,calculate_initial=True):
        super().__init__(parent)
        self.cube=cube;self.name=name;self.result=None;self.used_bounds=None
        self.setWindowTitle('DRR Peak-to-peak vs Y');self.resize(800,620)
        layout=QVBoxLayout(self)
        info=QLabel('For each Y: maximum DRR − minimum DRR within the Energy window.\nUses processed DRR directly; no peak detection or additional smoothing.')
        info.setWordWrap(True);layout.addWidget(info)
        form=QFormLayout();self.bounds=[]
        for index,(axis,values) in enumerate([('Energy (eV)',bounds[:2]),(cube.gate_label,bounds[2:])]):
            row=QHBoxLayout()
            for value in values:
                spin=CompactDoubleSpinBox();spin.setRange(-1e9,1e9);spin.setDecimals(5);spin.setValue(float(value));spin.setKeyboardTracking(False)
                if index==0:
                    spin.display_precision=4;spin.keep_trailing_zeros=True;spin.setSingleStep(.0001)
                    spin.lineEdit().setText(spin.textFromValue(spin.value()))
                spin.valueChanged.connect(self.invalidate);self.bounds.append(spin);row.addWidget(spin)
            form.addRow(axis,row)
        layout.addLayout(form)
        inspect=QHBoxLayout();inspect.addWidget(QLabel('Spectrum row'))
        self.row_spin=QSpinBox();self.row_spin.setRange(1,len(cube.gate));self.row_spin.valueChanged.connect(self.inspect_row)
        inspect.addWidget(self.row_spin);self.row_label=QLabel();inspect.addWidget(self.row_label,1);layout.addLayout(inspect)
        self.figure=Figure();self.canvas=ThemeAwareFigureCanvasQTAgg(self.figure);layout.addWidget(self.canvas,1)
        self.canvas.mpl_connect('button_press_event',self.curve_clicked)
        self.status=QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        buttons=QHBoxLayout();calculate=QPushButton('Calculate');calculate.clicked.connect(self.calculate);buttons.addWidget(calculate)
        self.csv_button=QPushButton('Save CSV');self.csv_button.clicked.connect(self.save_csv);buttons.addWidget(self.csv_button)
        self.png_button=QPushButton('Save PNG');self.png_button.clicked.connect(self.save_png);buttons.addWidget(self.png_button)
        layout.addLayout(buttons)
        if calculate_initial:self.calculate()

    def invalidate(self,*_):
        if not hasattr(self,'csv_button'):return
        self.csv_button.setEnabled(False);self.png_button.setEnabled(False)
        self.status.setText('Range changed. Click Calculate to update the curve.')

    def calculate(self):
        bounds=tuple(spin.value() for spin in self.bounds)
        try:result=range_amplitude(self.cube,*bounds)
        except ValueError as exc:
            self.invalidate();self.status.setText(str(exc));return
        self.result=result;self.used_bounds=bounds
        self.figure.clear();ax,self.spectrum_axis=self.figure.subplots(2,1)
        self.amplitude_axis=ax
        order=np.argsort(result[:,0],kind='stable')
        ax.plot(result[order,0],result[order,1],'.-',linewidth=1,markersize=3)
        ax.set_xlabel(self.cube.gate_label);ax.set_ylabel('DRR peak-to-peak (max − min)')
        ax.set_title(f'{self.name[:65]}\nEnergy {bounds[0]:.5g}–{bounds[1]:.5g} eV')
        self.cursor=ax.axvline(float(self.cube.gate[self.row_spin.value()-1]),color='tab:red',ls='--',lw=.8)
        self.spectrum_line,=self.spectrum_axis.plot(self.cube.energy,self.cube.Z[self.row_spin.value()-1],lw=1)
        self.spectrum_axis.axvspan(bounds[0],bounds[1],alpha=.12,color='tab:orange')
        self.extrema=self.spectrum_axis.scatter([np.nan,np.nan],[np.nan,np.nan],c=['tab:blue','tab:red'],s=35,zorder=4)
        self.spectrum_axis.set_xlabel('Energy (eV)');self.spectrum_axis.set_ylabel('DRR')
        self.inspect_row()
        self.figure.tight_layout();self.canvas.draw_idle()
        valid=int(np.isfinite(result[:,1]).sum())
        self.status.setText(f'{valid}/{len(result)} valid Y rows. Fewer than two finite samples → missing. Extrema can be noise-sensitive.')
        self.csv_button.setEnabled(True);self.png_button.setEnabled(True)

    def curve_clicked(self,event):
        if event.inaxes is self.amplitude_axis and event.xdata is not None:self.select_y(event.xdata)

    def select_y(self,value):
        row=int(np.argmin(abs(self.cube.gate-value)))
        self.row_spin.setValue(row+1);self.inspect_row()

    def inspect_row(self,*_):
        if not hasattr(self,'spectrum_line'):return
        row=self.row_spin.value()-1;y=float(self.cube.gate[row])
        self.spectrum_line.set_ydata(self.cube.Z[row]);self.cursor.set_xdata([y,y])
        # Compute from this exact row, including duplicate Y coordinates.
        from dataclasses import replace
        single=replace(self.cube,gate=self.cube.gate[row:row+1],Z=self.cube.Z[row:row+1])
        result=range_amplitude(single,*self.used_bounds[:2],y,y)[0]
        self.extrema.set_offsets([[result[4],result[2]],[result[5],result[3]]])
        self.spectrum_axis.relim();self.spectrum_axis.autoscale_view()
        self.spectrum_axis.set_title(f'{self.cube.gate_label} = {y:.6g} · peak-to-peak = {result[1]:.5g}',fontsize=9)
        self.row_label.setText(f'{self.cube.gate_label} = {y:.6g} · click the amplitude curve or step through rows')
        self.canvas.draw_idle()

    def write_csv(self,path):
        with open(path,'w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.writer(handle)
            writer.writerow(['Y','DRR_peak_to_peak','DRR_min','DRR_max','Energy_at_min_eV','Energy_at_max_eV','Finite_samples','Energy_min_eV','Energy_max_eV','Y_min','Y_max','Dataset','Y_label'])
            for row in self.result:
                writer.writerow([float(v) if np.isfinite(v) else '' for v in row]+list(self.used_bounds)+[self.name,self.cube.gate_label])

    def save_csv(self):
        path,_=QFileDialog.getSaveFileName(self,'Save peak-to-peak table','drr_peak_to_peak.csv','CSV (*.csv)')
        if path:
            try:self.write_csv(path)
            except OSError as exc:self.status.setText(str(exc))

    def save_png(self):
        path,_=QFileDialog.getSaveFileName(self,'Save peak-to-peak curve','drr_peak_to_peak.png','PNG (*.png)')
        if path:
            try:
                from ui_qt.async_figure_save import save_figure_async
                from shiboken6 import isValid
                if getattr(self, '_png_save_job', None) is not None and not self._png_save_job.done:
                    self.status.setText('PNG save already in progress.');return
                job = save_figure_async(self.figure, Path(path), dpi=200)
                self._png_save_job = job
                self.status.setText('Saving PNG…')
                def complete():
                    if isValid(self):
                        self.status.setText('PNG save failed: ' + job.error.splitlines()[0] if job.error else 'PNG saved: ' + path)
                job.finished.connect(complete)
            except Exception as exc:self.status.setText(str(exc))
