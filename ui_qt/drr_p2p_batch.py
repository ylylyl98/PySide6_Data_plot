"""Common-window P2P comparison with source-map and spectrum inspection."""
import csv
import numpy as np
from PySide6.QtWidgets import QCheckBox,QComboBox,QPushButton,QHBoxLayout,QWidget,QVBoxLayout,QLabel,QToolButton,QGroupBox,QFormLayout,QSizePolicy,QSpinBox
from PySide6.QtCore import QTimer
from ui_qt.compact_spinbox import CompactDoubleSpinBox
from matplotlib.widgets import SpanSelector
from matplotlib.patches import Rectangle
from core.drr_range_amplitude import range_amplitude,smooth_energy_cube
from ui_qt.drr_range_amplitude import RangeAmplitudeDialog
from ui_qt.drr_quick_export import QuickExportMixin
from ui_qt.drr_p2p_style import P2PStyleMixin
from core.drr_p2p_metrics import quantity_metrics


class BatchRangeAmplitudePage(P2PStyleMixin,QuickExportMixin,RangeAmplitudeDialog):
    def __init__(self,owner,dataset,bounds,*,state=None):
        self.owner=owner;self.records=[];self.selector=None;self.map_axis=None;self.pending=False;self.inspection_rows={}
        super().__init__(owner,dataset.cube,dataset.name,bounds,calculate_initial=False)
        self.limit_y=QCheckBox('Limit Y (otherwise all rows in each file)')
        self.limit_y.toggled.connect(self.invalidate)
        self.inspect_file=QComboBox();self.inspect_file.currentIndexChanged.connect(self.inspect_file_changed)
        self.view_mode=QComboBox();self.view_mode.addItems(['Single file','Compare all datasets'])
        self.view_mode.currentIndexChanged.connect(lambda _:self.render_comparison() if self.records else None)
        self.batch_button=QPushButton('Apply range to all datasets')
        self.batch_button.clicked.connect(self.run_batch)
        row=QHBoxLayout();row.addWidget(self.view_mode);row.addWidget(self.limit_y);row.addWidget(self.batch_button);row.addWidget(self.inspect_file,1)
        self.layout().insertLayout(2,row)
        self.display_ranges={};self.display_bounds=[]
        self.display_timer=QTimer(self);self.display_timer.setSingleShot(True);self.display_timer.timeout.connect(self.apply_display_range)
        self.display_panel=QWidget();display_layout=QVBoxLayout(self.display_panel)
        display_layout.setContentsMargins(0,0,0,0)
        display_row=QHBoxLayout()
        for index,label in enumerate(('Energy min','max','Y min','max')):
            display_row.addWidget(QLabel(label));spin=CompactDoubleSpinBox()
            spin.setRange(-1e9,1e9);spin.setDecimals(9);spin.setKeyboardTracking(False)
            spin.display_precision=4 if index<2 else 3;spin.keep_trailing_zeros=index<2
            spin.setSingleStep(.0001 if index<2 else .1);spin.setMaximumWidth(115)
            spin.valueChanged.connect(lambda _:self.display_timer.start(100))
            self.display_bounds.append(spin);display_row.addWidget(spin)
        display_layout.addLayout(display_row)
        options=QHBoxLayout();self.fixed_display=QCheckBox('Fix display range across files / groups');options.addWidget(self.fixed_display)
        full=QPushButton('Full display range');full.clicked.connect(self.full_display_range);options.addWidget(full);options.addStretch()
        display_layout.addLayout(options)
        toggle=QToolButton();toggle.setText('Colorplot display range');toggle.setCheckable(True)
        toggle.toggled.connect(self.display_panel.setVisible);self.display_panel.hide()
        self.layout().insertWidget(3,toggle);self.layout().insertWidget(4,self.display_panel)
        for button in self.findChildren(QPushButton):
            if button.text()=='Calculate':
                button.setText('Recalculate all datasets');button.clicked.disconnect();button.clicked.connect(self.run_batch)
        self.canvas.mpl_connect('pick_event',self.pick_curve)
        self.build_sidebar()
        if state is None:self.receive_records(self.compute_records(list(getattr(owner,'datasets',{dataset.key:dataset}).values()),tuple(spin.value() for spin in self.bounds),False))
        self.auto_timer=QTimer(self);self.auto_timer.setSingleShot(True)
        self.auto_timer.timeout.connect(self.run_batch)
        for spin in self.bounds:spin.valueChanged.connect(self.schedule_calculation)
        self.limit_y.toggled.connect(self.schedule_calculation)
        for control in (self.smoothing,self.sg_width,self.sg_order):
            signal=control.currentIndexChanged if control is self.smoothing else control.valueChanged
            signal.connect(self.smoothing_changed)
        self.sg_unit.currentTextChanged.connect(self.change_sg_unit)
        self.batch_button.setText('Refresh all datasets')
        self.batch_button.setToolTip('Calculation range edits update automatically; this button refreshes manually.')
        if state is not None:self.restore_state(state)
        if state is not None and state.get('_recalculate_on_open'):self.schedule_calculation()

    def snapshot_state(self):
        records=[]
        for record in self.records:
            data={key:record.get(key) for key in ('bounds','result','raw_result','edge','error','smoothing')}
            data['smoothing']=record.get('smoothing') or {'method':'Off'}
            data['dataset_key']=record['dataset'].key
            if record.get('smoothing',{}).get('method')=='SG' and record.get('processed') is not None:data['processed_Z']=record['processed'].Z
            records.append(data)
        index=self.inspect_file.currentIndex()
        return dict(schema=1,bounds=[spin.value() for spin in self.bounds],limit_y=self.limit_y.isChecked(),
            fixed_calculation=self.fixed_calculation.isChecked(),
            smoothing=self.smoothing_settings(),display_bounds=[spin.value() for spin in self.display_bounds],
            display_ranges=self.display_ranges,fixed_display=self.fixed_display.isChecked(),view_mode=self.view_mode.currentIndex(),
            inspection_rows=self.inspection_rows,current_key=self.records[index]['dataset'].key if 0<=index<len(self.records) else None,
            pending=self.pending,records=records,comparison_metric=self.comparison_metric.currentIndex(),plot_style=self.plot_style(),show_inspection=self.show_inspection.isChecked())

    def restore_state(self,state):
        from dataclasses import replace
        if state.get('schema')!=1:raise ValueError('Unsupported P2P workspace state.')
        self.restore_plot_style(state.get('plot_style',{}))
        blocked=self.show_inspection.blockSignals(True);self.show_inspection.setChecked(state.get('show_inspection',False));self.show_inspection.blockSignals(blocked)
        blocked=self.comparison_metric.blockSignals(True);self.comparison_metric.setCurrentIndex(state.get('comparison_metric',0));self.comparison_metric.blockSignals(blocked)
        controls=[*self.bounds,self.limit_y,self.smoothing,self.sg_unit,self.sg_width,self.sg_order,self.fixed_display,self.fixed_calculation,self.view_mode]
        previous=[control.blockSignals(True) for control in controls]
        try:
            for spin,value in zip(self.bounds,state['bounds']):spin.setValue(value)
            self.limit_y.setChecked(state.get('limit_y',False))
            self.fixed_calculation.setChecked(state.get('fixed_calculation',False))
            settings=state.get('smoothing',{'method':'Off'})
            unit=settings.get('unit','meV')
            self.sg_unit.setCurrentText(unit);self.configure_sg_unit(unit)
            self.smoothing.setCurrentText(settings['method']);self.sg_width.setValue(settings.get('window_points',21) if unit=='points' else settings.get('window_mev',2.));self.sg_order.setValue(settings.get('polyorder',2))
            self.sg_width.setEnabled(settings['method']=='SG');self.sg_order.setEnabled(settings['method']=='SG')
            self.fixed_display.setChecked(state.get('fixed_display',False));self.view_mode.setCurrentIndex(state.get('view_mode',0))
        finally:
            for control,blocked in zip(controls,previous):control.blockSignals(blocked)
        self.calc_y_row.setVisible(self.limit_y.isChecked())
        self.display_ranges=state.get('display_ranges',{});self.inspection_rows=state.get('inspection_rows',{})
        self.set_display_bounds(state['display_bounds'])
        records=[]
        for record in state.get('records',[]):
            dataset=self.owner.datasets.get(record['dataset_key'])
            if dataset is None:continue
            record=dict(record);record['dataset']=dataset
            record['processed']=replace(dataset.cube,Z=record['processed_Z']) if record.get('processed_Z') is not None else dataset.cube
            records.append(record)
        self.receive_records(records,cache_completed=not state.get('pending'))
        index=next((i for i,r in enumerate(records) if r['dataset'].key==state.get('current_key')),0)
        if records:self.inspect_file.setCurrentIndex(index)
        self.auto_timer.stop()
        if state.get('pending'):
            self._completed_state=None
            self.invalidate();self.status.setText('Restored previous results; saved calculation settings have unapplied changes. Refresh all datasets to apply.')

    def smoothing_settings(self):
        unit=self.sg_unit.currentText()
        return dict(method=self.smoothing.currentText(),unit=unit,**{('window_points' if unit=='points' else 'window_mev'):self.sg_width.value()},polyorder=self.sg_order.value())

    def configure_sg_unit(self,unit):
        points=unit=='points'
        self.sg_width.setDecimals(0 if points else 2);self.sg_width.setRange(3 if points else .01,1001 if points else 1000)
        self.sg_width.setSingleStep(2 if points else .1)

    def change_sg_unit(self,unit):
        blocked=self.sg_width.blockSignals(True)
        self.configure_sg_unit(unit);self.sg_width.setValue(21 if unit=='points' else 2.)
        self.sg_width.blockSignals(blocked);self.smoothing_changed()

    def smoothing_changed(self,*_):
        enabled=self.smoothing.currentText()=='SG'
        self.sg_width.setEnabled(enabled);self.sg_order.setEnabled(enabled)
        self.invalidate();self.schedule_calculation()

    def schedule_calculation(self,*_):
        if not hasattr(self.owner,'start_job') or getattr(self.owner,'_is_closing',False):return
        xmin,xmax,ymin,ymax=(spin.value() for spin in self.bounds)
        if xmin>=xmax or (self.limit_y.isChecked() and ymin>ymax):
            self.auto_timer.stop();self.status.setText('Invalid calculation range; previous results retained.');return
        self.status.setText('Range changed · automatic update pending…')
        self.auto_timer.start(250)

    def build_sidebar(self):
        """Move controls into a dedicated sidebar; center holds only view and plots."""
        keep={self.canvas,self.view_mode,self.limit_y,self.batch_button,self.inspect_file,
              self.row_spin,self.row_label,self.status,self.csv_button,self.png_button,
              self.fixed_display,*self.bounds,*self.display_bounds}
        def clear_layout(layout):
            while layout.count():
                item=layout.takeAt(0)
                if item.layout():clear_layout(item.layout());item.layout().deleteLater()
                elif item.widget():
                    widget=item.widget()
                    if widget is self.display_panel:clear_layout(widget.layout())
                    widget.hide();widget.setParent(None)
                    if widget not in keep:widget.deleteLater()
        clear_layout(self.layout())
        self.sidebar=QWidget();side=QVBoxLayout(self.sidebar);side.setContentsMargins(8,8,8,8);side.setSpacing(8)
        header=QHBoxLayout();header.addWidget(QLabel('View'));header.addWidget(self.view_mode)
        self.comparison_metric=QComboBox()
        for label,value in [('P2P amplitude','amplitude'),('Extrema positions','positions'),('Extrema separation','separation')]:self.comparison_metric.addItem(label,value)
        self.comparison_metric.setToolTip('Reuse the existing extrema; no peak detection or recalculation.')
        self.comparison_metric.currentIndexChanged.connect(lambda _:self.render_comparison() if self.records else None)
        header.addWidget(QLabel('Compare'));header.addWidget(self.comparison_metric);header.addStretch()
        self.layout().addLayout(header);self.layout().addWidget(self.canvas,1)
        def group(title):
            box=QGroupBox(title);layout=QVBoxLayout(box);side.addWidget(box);return layout
        def pair(spins):
            row=QWidget();layout=QHBoxLayout(row);layout.setContentsMargins(0,0,0,0)
            for i,spin in enumerate(spins):
                if i:layout.addWidget(QLabel('to'))
                spin.setMinimumWidth(70);spin.setMaximumWidth(115);spin.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
                layout.addWidget(spin)
            return row
        calc=group('1 · Calculation range')
        calc.addWidget(QLabel('Energy (eV)'));calc.addWidget(pair(self.bounds[:2]))
        self.fixed_calculation=QCheckBox('Fix calculation range across groups')
        self.fixed_calculation.setToolTip('Keep Energy and optional Y calculation bounds when opening another group. Smoothing remains group-specific.')
        calc.addWidget(self.fixed_calculation)
        form=QFormLayout();self.smoothing=QComboBox();self.smoothing.addItems(['Off','SG'])
        form.addRow('Smoothing',self.smoothing)
        self.sg_unit=QComboBox();self.sg_unit.addItems(['points','meV'])
        self.sg_width=CompactDoubleSpinBox();self.configure_sg_unit('points');self.sg_width.setValue(21);self.sg_width.setKeyboardTracking(False)
        self.sg_order=QSpinBox();self.sg_order.setRange(1,5);self.sg_order.setValue(2)
        self.sg_width.setEnabled(False);self.sg_order.setEnabled(False)
        form.addRow('Window unit',self.sg_unit);form.addRow('Window',self.sg_width);form.addRow('Polynomial',self.sg_order);calc.addLayout(form)
        self.edit_range=QCheckBox('Edit calculation window on map')
        self.edit_range.toggled.connect(lambda yes:self.selector.set_active(yes) if self.selector else None)
        calc.addWidget(self.edit_range)
        self.limit_y.setText('Limit calculation Y');calc.addWidget(self.limit_y)
        yrow=pair(self.bounds[2:]);self.calc_y_row=yrow;calc.addWidget(yrow);yrow.setVisible(self.limit_y.isChecked())
        self.limit_y.toggled.connect(yrow.setVisible)
        self.batch_button.setText('Calculate all datasets');calc.addWidget(self.batch_button)
        view=group('2 · Inspect')
        view.addWidget(QLabel('Current file'));self.inspect_file.setMinimumWidth(0)
        self.inspect_file.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon);self.inspect_file.setMinimumContentsLength(15)
        view.addWidget(self.inspect_file)
        self.show_inspection=QCheckBox('Show spectrum inspection in Compare')
        self.show_inspection.toggled.connect(lambda _:self.render_comparison() if self.records else None)
        view.addWidget(self.show_inspection)
        step=QHBoxLayout();previous=QPushButton('Previous Y');following=QPushButton('Next Y')
        previous.clicked.connect(lambda:self.row_spin.stepDown());following.clicked.connect(lambda:self.row_spin.stepUp())
        step.addWidget(previous);step.addWidget(following);view.addLayout(step)
        self.row_label.setWordWrap(True);view.addWidget(self.row_label)
        self.row_spin.setParent(self.sidebar);self.row_spin.hide()
        self.display_panel=QWidget();display=QVBoxLayout(self.display_panel);display.setContentsMargins(0,0,0,0)
        display.addWidget(QLabel('Display Energy (eV)'));display.addWidget(pair(self.display_bounds[:2]))
        display.addWidget(QLabel('Display Y'));display.addWidget(pair(self.display_bounds[2:]))
        display.addWidget(self.fixed_display)
        full=QPushButton('Full display range');full.clicked.connect(self.full_display_range);display.addWidget(full)
        toggle=QToolButton();toggle.setText('Display range');toggle.setCheckable(True)
        toggle.toggled.connect(self.display_panel.setVisible);view.addWidget(toggle);view.addWidget(self.display_panel)
        result=group('3 · Results');result.addWidget(self.status)
        export=QHBoxLayout();export.addWidget(self.csv_button);export.addWidget(self.png_button);result.addLayout(export)
        self.build_export_options(result)
        self.build_plot_style(side)
        side.addStretch()
        # Reparented controls were explicitly hidden while dismantling the old layout.
        for widget in keep:
            if widget is not self.row_spin:widget.show()
        yrow.setVisible(self.limit_y.isChecked());self.display_panel.hide()

    def invalidate(self,*_):
        self.pending=True
        super().invalidate()
        if hasattr(self,'status'):self.status.setText('Calculation range changed · results awaiting update.')
        if self.map_axis is not None and hasattr(self,'window_patch'):
            lo,hi,ylo,yhi=(spin.value() for spin in self.bounds)
            if not self.limit_y.isChecked():ylo,yhi=float(self.cube.gate.min()),float(self.cube.gate.max())
            self.window_patch.set_bounds(lo,ylo,hi-lo,yhi-ylo)
            for line,value in zip(self.boundary_lines,(lo,hi)):line.set_xdata([value,value])
            self.canvas.draw_idle()

    @staticmethod
    def compute_records(datasets,bounds,limit_y,progress=lambda _:None,smoothing=None):
        records=[]
        for index,dataset in enumerate(datasets):
            cube=dataset.cube
            used=(*bounds[:2],*(bounds[2:] if limit_y else (float(cube.gate.min()),float(cube.gate.max()))))
            try:
                settings=smoothing or dict(method='Off')
                unit=settings.get('unit','meV')
                processed,metadata=(smooth_energy_cube(cube,settings['window_points'] if unit=='points' else settings['window_mev'],settings['polyorder'],unit=unit) if settings['method']=='SG' else (cube,dict(method='Off')))
                raw_result=range_amplitude(cube,*used)
                result=range_amplitude(processed,*used)
                energies=cube.energy[(cube.energy>=used[0])&(cube.energy<=used[1])]
                spacing=float(np.median(abs(np.diff(np.sort(energies))))) if len(energies)>1 else 0.
                edge=np.isfinite(result[:,1]) & ((np.minimum(result[:,4],result[:,5])<=energies.min()+spacing) | (np.maximum(result[:,4],result[:,5])>=energies.max()-spacing))
                records.append(dict(dataset=dataset,bounds=used,result=result,raw_result=raw_result,processed=processed,smoothing=metadata,edge=edge,error=None))
            except ValueError as exc:records.append(dict(dataset=dataset,bounds=used,result=None,edge=None,error=str(exc)))
            progress(int(100*(index+1)/len(datasets)))
        return records

    def run_batch(self,*_,reuse=False):
        if getattr(self.owner,'_is_closing',False):return
        if self.owner.jobs:
            self.auto_timer.start(250);return
        self.auto_timer.stop()
        datasets=list(self.owner.datasets.values())
        if not datasets:self.status.setText('Add processed datasets first.');return
        if len({d.cube.gate_label for d in datasets})>1:
            self.status.setText('Select files with the same Y axis for comparison.');return
        bounds=tuple(spin.value() for spin in self.bounds);limit_y=self.limit_y.isChecked()
        smoothing=self.smoothing_settings()
        if bounds[0]>=bounds[1] or (limit_y and bounds[2]>bounds[3]):return
        archive=self.owner.prepare_p2p_archive(getattr(self,'_completed_state',None)) if hasattr(self.owner,'prepare_p2p_archive') else None
        self.invalidate();self.status.setText(f'Calculating all {len(datasets)} datasets…')
        def receive(records):
            # Range edits during calculation must not silently export old data.
            if bounds!=tuple(spin.value() for spin in self.bounds) or limit_y!=self.limit_y.isChecked() or smoothing!=self.smoothing_settings():
                self.schedule_calculation();return
            self.receive_records([r for r in records if r['dataset'].key in self.owner.datasets])
            self.workspace_changed()
        def run(progress,log):
            if archive:
                from core.drr_workspace_session import save_session
                save_session(*archive)
            return self.compute_records(datasets,bounds,limit_y,progress.emit,smoothing)
        self.owner.start_job(run,receive,cancellable=False)

    def workspace_changed(self):
        keys=set(self.owner.datasets)
        if keys=={r['dataset'].key for r in self.records}:return
        kept=[r for r in self.records if r['dataset'].key in keys]
        if kept:self.receive_records(kept,cache_completed=False)
        else:
            self.records=[];self.result=None;self.figure.clear();self.canvas.draw_idle()
            self.inspect_file.clear()
        self.invalidate()
        self.status.setText(f'Datasets changed ({len(keys)} files). Recalculate all datasets using the common Energy range.')

    def receive_records(self,records,*,cache_completed=True):
        current_key=self.records[self.inspect_file.currentIndex()]['dataset'].key if self.records and self.inspect_file.currentIndex()>=0 else None
        self.pending=False
        self.records=records
        self.inspect_file.blockSignals(True);self.inspect_file.clear()
        for index,record in enumerate(records):
            dataset=record['dataset']
            label=self.owner.comparison_dataset_label(dataset) if hasattr(self.owner,'comparison_dataset_label') else dataset.name
            self.inspect_file.addItem(f'{index+1}: {label}')
        index=next((i for i,r in enumerate(records) if r['dataset'].key==current_key),0)
        if records:self.inspect_file.setCurrentIndex(index)
        self.inspect_file.blockSignals(False)
        self.inspect_file_changed(index)
        if cache_completed:self._completed_state=self.snapshot_state()
        if hasattr(self.owner,'refresh_labels'):self.owner.refresh_labels()

    def inspect_file_changed(self,index):
        if not 0<=index<len(self.records):return
        record=self.records[index];dataset=record['dataset']
        self.cube=dataset.cube;self.name=dataset.name;self.result=record['result'];self.used_bounds=record['bounds']
        if not self.fixed_display.isChecked():
            limits=self.display_ranges.get(dataset.key,(self.cube.energy.min(),self.cube.energy.max(),self.cube.gate.min(),self.cube.gate.max()))
            self.set_display_bounds(limits)
        self.row_spin.blockSignals(True);self.row_spin.setRange(1,len(self.cube.gate));self.row_spin.setValue(self.inspection_rows.get(dataset.key,1));self.row_spin.blockSignals(False)
        self.render_comparison()

    def set_display_bounds(self,limits):
        for spin,value in zip(self.display_bounds,limits):
            old=spin.blockSignals(True);spin.setValue(float(value));spin.blockSignals(old)

    def full_display_range(self):
        self.set_display_bounds((self.cube.energy.min(),self.cube.energy.max(),self.cube.gate.min(),self.cube.gate.max()))
        self.apply_display_range()

    def apply_display_range(self):
        if not self.records:return
        xmin,xmax,ymin,ymax=(spin.value() for spin in self.display_bounds)
        if xmin>=xmax or ymin>ymax:return
        key=self.records[self.inspect_file.currentIndex()]['dataset'].key
        self.display_ranges[key]=(xmin,xmax,ymin,ymax)
        if self.map_axis is not None:
            self.map_axis.set_xlim(xmin,xmax)
            self.map_axis.set_ylim((ymin,ymax) if ymin<ymax else (ymin-.5,ymax+.5))
            maskx=(self.cube.energy>=xmin)&(self.cube.energy<=xmax)
            masky=(self.cube.gate>=ymin)&(self.cube.gate<=ymax)
            values=self.cube.Z[np.ix_(masky,maskx)];finite=values[np.isfinite(values)]
            if finite.size:
                lim=max(float(np.quantile(abs(finite),.98)),1e-12)
                self.map_axis.collections[0].set_clim(-lim,lim)
        self.sync_spectrum_view((xmin,xmax))
        self.canvas.draw_idle()

    def sync_spectrum_view(self,limits=None):
        if not hasattr(self,'spectrum_line'):return
        if limits is None:
            limits=self.map_axis.get_xlim() if self.map_axis is not None else tuple(spin.value() for spin in self.display_bounds[:2])
        if limits[0]==limits[1]:return
        self.spectrum_axis.set_xlim(*limits)
        low,high=sorted(limits)
        row=self.cube.Z[self.row_spin.value()-1]
        visible=row[(self.cube.energy>=low)&(self.cube.energy<=high)]
        finite=visible[np.isfinite(visible)]
        if finite.size:
            bottom,top=float(finite.min()),float(finite.max())
            pad=max((top-bottom)*.08,max(abs(bottom),abs(top))*.01,1e-12)
            self.spectrum_axis.set_ylim(bottom-pad,top+pad)
        self.canvas.draw_idle()

    def render_comparison(self):
        if self.selector:self.selector.disconnect_events()
        self.selector=None;self.map_axis=None
        compare=self.view_mode.currentIndex()==1
        style=self.plot_style();labels=self.comparison_labels()
        curve_styles=self.comparison_curve_styles()
        metrics=quantity_metrics(self.comparison_quantity());self.comparison_metric.setEnabled(compare)
        self.style_controls['offset'].setEnabled(compare and metrics==('amplitude',))
        self.style_controls['offset_step'].setEnabled(compare and metrics==('amplitude',))
        self.show_inspection.setEnabled(compare)
        self.edit_range.setEnabled(not compare)
        self.figure.clear()
        lo,hi,ylo,yhi=self.used_bounds
        if compare:
            panels=np.atleast_1d(self.figure.subplots(len(metrics)+1,1));self.metric_axes=list(panels[:-1]);self.amplitude_axis=self.metric_axes[0];self.spectrum_axis=panels[-1]
            for axis in self.metric_axes[1:]:axis.sharex(self.amplitude_axis)
        else:
            grid=self.figure.add_gridspec(2,2,width_ratios=[1,1.3])
            self.map_axis=self.figure.add_subplot(grid[:,0]);self.amplitude_axis=self.figure.add_subplot(grid[0,1]);self.spectrum_axis=self.figure.add_subplot(grid[1,1])
            self.metric_axes=[self.amplitude_axis]
            xorder=np.argsort(self.cube.energy);yorder=np.argsort(self.cube.gate)
            mesh=self.map_axis.pcolormesh(self.cube.energy[xorder],self.cube.gate[yorder],self.cube.Z[np.ix_(yorder,xorder)],shading='nearest',cmap='RdBu_r')
            values=self.cube.Z[np.isfinite(self.cube.Z)]
            if values.size:
                lim=max(float(np.quantile(abs(values),.98)),1e-12);mesh.set_clim(-lim,lim)
            self.map_axis.set_xlabel('Energy (eV)');self.map_axis.set_ylabel(self.cube.gate_label)
            self.map_axis.set_title(self.name[:35],fontsize=9)
            self.window_patch=Rectangle((lo,ylo),hi-lo,yhi-ylo,facecolor='orange',alpha=.2,edgecolor='black')
            self.map_axis.add_patch(self.window_patch)
            self.boundary_lines=[self.map_axis.axvline(value,color='tab:orange',lw=.8) for value in (lo,hi)]
            self.map_cursor=self.map_axis.axhline(self.cube.gate[0],ls='--',color='black',lw=.8)
        for i,record in enumerate(self.records):
            selected=i==self.inspect_file.currentIndex()
            if not compare and not selected:continue
            result=record['result']
            if result is None:continue
            order=np.argsort(result[:,0],kind='stable')
            dataset=record['dataset']
            label=self.owner.comparison_dataset_label(dataset) if hasattr(self.owner,'comparison_dataset_label') else dataset.name[:28]
            for axis,metric in zip(self.metric_axes,metrics):self.plot_metric_curve(axis,i,metric,compare=compare)
            if not compare and record.get('smoothing',{}).get('method')=='SG':
                raw=record['raw_result']
                self.amplitude_axis.plot(raw[order,0],raw[order,1],'--',color='gray',alpha=.65,lw=1,label='Raw P2P')
        self.amplitude_axis.set_xlabel(self.cube.gate_label);self.amplitude_axis.set_ylabel('P2P (max − min)')
        meta=self.records[self.inspect_file.currentIndex()].get('smoothing',{})
        unit=meta.get('unit','meV');window=meta.get('window_points') if unit=='points' else meta.get('window_mev')
        method=f" · SG {window:g} {unit} / {meta['polyorder']}" if meta.get('method')=='SG' else ' · raw'
        self.amplitude_axis.set_title(f'P2P · {lo:.4f}–{hi:.4f} eV{method}',fontsize=10)
        if any(r['result'] is not None for r in self.records):self.amplitude_axis.legend(fontsize=7)
        for panel,(axis,metric) in enumerate(zip(self.metric_axes,metrics)):
            axis.set_xlabel(self.cube.gate_label);self.style_comparison_axis(axis,metric,heading=panel==0)
            if panel and axis.get_legend():axis.get_legend().remove()
        self.cursor=self.amplitude_axis.axvline(self.cube.gate[0],ls='--',color='gray',lw=.8)
        self.cursor.set_visible(not compare or self.show_inspection.isChecked())
        self.spectrum_line,=self.spectrum_axis.plot(self.cube.energy,self.cube.Z[0],lw=1)
        self.spectrum_line.set_label('Raw DRR')
        self.smoothed_line,=self.spectrum_axis.plot([],[],lw=1.4,color='tab:orange',label='SG DRR')
        self.spectrum_axis.axvspan(lo,hi,alpha=.15,color='orange')
        self.extrema=self.spectrum_axis.scatter([np.nan,np.nan],[np.nan,np.nan],c=['tab:blue','tab:red'],s=35,zorder=4)
        self.spectrum_axis.set_xlabel('Energy (eV)');self.spectrum_axis.set_ylabel('DRR')
        if self.map_axis is not None:
            self.selector=SpanSelector(self.map_axis,self.select_window,'horizontal',useblit=False,props=dict(alpha=.2,facecolor='orange'))
            self.selector.set_active(self.edit_range.isChecked())
        self.inspect_row();self.apply_display_range()
        if self.map_axis is not None:
            self.map_axis.callbacks.connect('xlim_changed',lambda ax:self.sync_spectrum_view(ax.get_xlim()))
        if compare and not self.show_inspection.isChecked():
            self.spectrum_axis.set_visible(False)
            for panel,axis in enumerate(self.metric_axes):
                axis.set_position([.12,.14+(len(metrics)-1-panel)*.43,.84,.72 if len(metrics)==1 else .30])
        else:self.figure.tight_layout()
        self.canvas.draw_idle()
        edges=sum(int(r['edge'].sum()) for r in self.records if r['edge'] is not None)
        missing=sum(int((~np.isfinite(r['result'][:,1])).sum()) for r in self.records if r['result'] is not None)
        failures=[f'{r["dataset"].name[:25]}: {r["error"]}' for r in self.records if r['error']]
        hint='Click a curve point to inspect its file and spectrum.' if compare else 'Drag Energy window on map; click a curve point to inspect.'
        if metrics!=('amplitude',):hint+=' Hollow markers: extrema near the calculation window edge.'
        self.status.setText(f'{len(self.records)} files · {edges} rows with extrema near window edges · {missing} missing rows. {hint} '+ '; '.join(failures))
        if any(c['field'] is not None and not style['field_min']<=c['field']<=style['field_max'] for c in curve_styles):
            self.status.setText(self.status.text()+' B outside fixed color limits; adjust Fixed B color min / max in Plot style.')
        valid=any(r['result'] is not None for r in self.records)
        self.csv_button.setEnabled(valid);self.png_button.setEnabled(valid)
        if self.pending:self.invalidate()

    def select_window(self,low,high):
        if low==high:return
        self.bounds[0].setValue(min(low,high));self.bounds[1].setValue(max(low,high))
        for line,value in zip(self.boundary_lines,sorted((low,high))):line.set_xdata([value,value])
        self.window_patch.set_x(min(low,high));self.window_patch.set_width(abs(high-low));self.canvas.draw_idle()

    def pick_curve(self,event):
        if not hasattr(event.artist,'_p2p_record') or not len(event.ind):return
        index=event.artist._p2p_record;result=self.records[index]['result']
        # Choose the closest picked point in display coordinates.
        indices=event.artist._p2p_order[np.asarray(event.ind,dtype=int)]
        plotted=result[indices,:2].copy()
        values=getattr(event.artist,'_p2p_yvalues',None)
        if values is not None:plotted[:,1]=values[indices]
        else:plotted[:,1]+=getattr(event.artist,'_p2p_offset',0.)
        coords=event.artist.axes.transData.transform(plotted)
        mouse=event.mouseevent
        row=indices[np.argmin(np.sum((coords-[mouse.x,mouse.y])**2,axis=1))]
        record=self.records[index];cube=record['dataset'].cube
        source_rows=np.flatnonzero((cube.gate>=record['bounds'][2])&(cube.gate<=record['bounds'][3]))
        self.inspect_file.setCurrentIndex(index)
        self.row_spin.setValue(int(source_rows[row])+1);self.inspect_row()

    def curve_clicked(self,event):
        # Curve picking also identifies the file; plain X clicks cannot.
        if self.map_axis is not None and event.inaxes is self.map_axis and event.ydata is not None:self.select_y(event.ydata)

    def inspect_row(self,*_):
        if not self.records:return super().inspect_row()
        if 0<=self.inspect_file.currentIndex()<len(self.records):
            self.inspection_rows[self.records[self.inspect_file.currentIndex()]['dataset'].key]=self.row_spin.value()
        if self.result is None:
            self.spectrum_line.set_ydata(self.cube.Z[self.row_spin.value()-1]);self.extrema.set_offsets(np.empty((0,2)));self.sync_spectrum_view();return
        super().inspect_row()
        record=self.records[self.inspect_file.currentIndex()]
        if record.get('smoothing',{}).get('method')=='SG':
            from dataclasses import replace
            row=self.row_spin.value()-1;processed=record['processed'];y=float(self.cube.gate[row])
            self.smoothed_line.set_data(processed.energy,processed.Z[row]);self.smoothed_line.set_visible(True)
            single=replace(processed,gate=processed.gate[row:row+1],Z=processed.Z[row:row+1])
            metrics=range_amplitude(single,*self.used_bounds[:2],y,y)[0]
            self.extrema.set_offsets([[metrics[4],metrics[2]],[metrics[5],metrics[3]]])
            self.spectrum_axis.set_title(f'{self.cube.gate_label} = {y:.6g} · SG P2P = {metrics[1]:.5g}',fontsize=9)
            self.spectrum_line.set_alpha(.5);self.spectrum_axis.legend(fontsize=8)
        else:
            self.smoothed_line.set_visible(False);self.spectrum_line.set_alpha(1.)
        self.row_label.setText(f'{self.cube.gate_label} = {self.cube.gate[self.row_spin.value()-1]:.6g} · row {self.row_spin.value()}/{len(self.cube.gate)}')
        if self.map_axis is not None:self.map_cursor.set_ydata([self.cube.gate[self.row_spin.value()-1]]*2)
        self.sync_spectrum_view()

    def write_csv(self,path):
        with open(path,'w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.writer(handle)
            writer.writerow(['Dataset','Y_label','Y','P2P','Min','Max','Energy_at_min_eV','Energy_at_max_eV','Finite_samples','Energy_min_eV','Energy_max_eV','Y_min','Y_max','Edge_warning','Status','Raw_P2P','Smoothing','Window_meV','Polynomial','Window_points_min','Window_points_max','Actual_span_meV_min','Actual_span_meV_max','Window_unit','Window_points','B_T','Separation_meV','Emin_edge','Emax_edge'])
            for record in self.records:
                dataset=record['dataset']
                from core.drr_p2p_export import field_value
                field=field_value(dataset)
                if record['result'] is None:
                    writer.writerow([dataset.name,dataset.cube.gate_label]+['']*7+list(record['bounds'])+['',record['error']]+['']*10+[field,'','','']);continue
                emin_edge=self.metric_edge_flags(record,'minimum');emax_edge=self.metric_edge_flags(record,'maximum')
                for index,(row,edge) in enumerate(zip(record['result'],record['edge'])):
                    meta=record.get('smoothing',dict(method='Off'));raw=record.get('raw_result',record['result'])[index,1]
                    writer.writerow([dataset.name,dataset.cube.gate_label]+[float(v) if np.isfinite(v) else '' for v in row]+list(record['bounds'])+[bool(edge),'ok' if np.isfinite(row[1]) else 'insufficient data',float(raw) if np.isfinite(raw) else '']+[meta.get(k,'') for k in ('method','window_mev','polyorder','window_points_min','window_points_max','actual_span_mev_min','actual_span_mev_max','unit','window_points')]+[field,float(abs(row[5]-row[4])*1000) if np.isfinite(row[4:6]).all() else '',bool(emin_edge[index]),bool(emax_edge[index])])
