"""Explicit per-series group mapping and common-range CW comparisons."""
import copy
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QTableWidget,
    QTableWidgetItem,QComboBox,QDoubleSpinBox,QCheckBox,QPushButton,QTabWidget,
    QWidget,QPlainTextEdit,QMessageBox,QFileDialog,QAbstractItemView,QToolTip)
from PySide6.QtGui import QCursor
from matplotlib.figure import Figure
from core.theta_comparison import compare_theta
from core.mcd_extract import dependency_export_folder
from ui_qt.common import Worker
from ui_qt.export_workers import OwnedWorkerPool
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg


def _condition_label(value):
    """Remove machine roundoff at precision finer than condition tolerances."""
    if value is None:
        return 'unknown'
    number = round(float(value), 8)
    if number == 0:
        return '0'
    return f'{number:.8f}'.rstrip('0').rstrip('.')


def comparison_point_details(row):
    def number(value):
        return 'N/A' if value is None else f'{value:.5g}'
    low, high = row.get('ci95_low_k'), row.get('ci95_high_k')
    ci = 'unavailable / unreliable' if low is None or high is None else f'[{number(low)}, {number(high)}] K'
    fit_detail=(f"CW R² (1/s vs T) = {number(row.get('r_squared'))}" if row.get('method','inverse')=='inverse'
                else f"Method: {row.get('method')} · reduced χ² = {number(row.get('reduced_chi2'))}")
    return (f"{row.get('group', '')} · {row['branch']}\n"
            f"Doping={number(row.get('doping'))} · E-field={number(row.get('efield'))}\n"
            f"B fit: ±{number(row.get('halfwidth_t'))} T\n"
            f"θCW = {number(row.get('theta_k'))} K\n95% CI: {ci}\n"
            f"{fit_detail}\n"
            f"N = {row.get('n', 'N/A')} temperature points\n{row.get('diagnostics','')}")


def draw_comparison(figure, payload, axis_name='efield', visible_groups=None, y_limits=None, reliable_y=False, method='inverse', _ax=None):
    if _ax is None and method=='all':
        figure.clear();artists={};axes=[]
        for i,key in enumerate(('inverse','slope','weighted')):
            ax=figure.add_subplot(1,3,i+1);axes.append(ax)
            artists.update(draw_comparison(figure,payload,axis_name,visible_groups,y_limits,reliable_y,key,ax))
        if y_limits is None:
            bounds=[ax.get_ylim() for ax in axes];common=(min(v[0] for v in bounds),max(v[1] for v in bounds))
            for ax in axes:ax.set_ylim(*common)
            for ax in axes:
                for text in list(ax.texts):
                    if 'outside Y range' in text.get_text():text.remove()
                outside=sum(r['theta_k']<common[0] or r['theta_k']>common[1] for artist,r in artists.items() if artist.axes is ax)
                if outside:ax.text(.99,.02,f'{outside} point(s) outside shared Y range',transform=ax.transAxes,ha='right',fontsize=8)
        bounds=[ax.get_xlim() for ax in axes]
        for ax in axes:ax.set_xlim(min(v[0] for v in bounds),max(v[1] for v in bounds))
        return artists
    if _ax is None:figure.clear()
    ax=_ax if _ax is not None else figure.add_subplot(111)
    point_artists = {}
    use_methods=bool(payload.get('method_rows'))
    all_rows = [dict(r) for r in payload['method_rows'] if r['method']==method] if use_methods else (payload['rows'] if method=='inverse' else [])
    for row in all_rows:
        if row['status']=='diagnostic only' and row.get('theta_k') is not None:
            row.update(status='ok',ci95_low_k=None,ci95_high_k=None)
    rows_to_show = [r for r in all_rows if visible_groups is None or r.get('group') in visible_groups]
    groups = sorted({r.get('group', '') for r in all_rows})
    from core.mcd_energy_groups import energy_group_colors
    group_colors = energy_group_colors(dict(enumerate(groups)))
    plotted_values = []
    reliable_bounds = []
    def mark_uncertainty(x, row, color):
        plotted_values.append(row['theta_k'])
        bounds = (row['ci95_low_k'], row['ci95_high_k'])
        if all(value is not None and np.isfinite(value) for value in bounds):
            if bounds[0] <= row['theta_k'] <= bounds[1]:
                reliable_bounds.extend(bounds)
        if row['ci95_low_k'] is None:
            from matplotlib import patheffects
            ax.annotate('×', (x, row['theta_k']), xytext=(0, 0),
                        textcoords='offset points', color='#222222', fontsize=9,
                        ha='center', va='center', zorder=6,
                        path_effects=[patheffects.withStroke(linewidth=1.3, foreground='white')],
                        annotation_clip=True)
    if axis_name=='sensitivity':
        keys=list(dict.fromkeys((r['series_id'],r['branch'],r.get('group','')) for r in rows_to_show))
        for i,key in enumerate(keys):
            rows=sorted([r for r in rows_to_show if (r['series_id'],r['branch'],r.get('group',''))==key],key=lambda r:r['halfwidth_t'])
            color = group_colors[key[2]] if len(groups)>1 else f'C{i%10}'
            label=f"{key[2]} · D={_condition_label(rows[0]['doping'])}, F={_condition_label(rows[0]['efield'])} · {key[1]}"
            ax.plot([r['halfwidth_t'] for r in rows],
                    [r['theta_k'] if r['status']=='ok' else np.nan for r in rows],
                    '-' if key[1]=='B increasing' else '--',label=label,color=color)
            for r in rows:
                if r['status']!='ok': continue
                ci=r['ci95_low_k']
                error=None if ci is None else [[r['theta_k']-ci],[r['ci95_high_k']-r['theta_k']]]
                artist = ax.errorbar([r['halfwidth_t']],[r['theta_k']],yerr=error,
                    fmt='o' if key[1]=='B increasing' else 's',capsize=2,color=color,
                    markerfacecolor=color if key[1]=='B increasing' else 'none')
                point_artists[artist.lines[0]] = r
                mark_uncertainty(r['halfwidth_t'], r, color)
        ax.set_xlabel('Common symmetric B half-width (T)')
        ax.set_title('Range sensitivity · bars: 95% CI; ×: CI unreliable; gaps: failed\n'
                     'Increasing: filled circles · Decreasing: open squares')
    else:
        other='doping' if axis_name=='efield' else 'efield'
        primary=[r for r in rows_to_show if r['is_primary'] and r['status']=='ok' and r[axis_name] is not None]
        labels=[]
        for row in primary:
            extra=' · '.join(f'{k}={_condition_label(v)}' for k,v in sorted(row['fixed_conditions'].items())
                             if k not in ('Doping','E-field') and v is not None)
            label=f"{row.get('group', '')} · {other}={_condition_label(row[other])} · {row['branch']}" + (f' · {extra}' if extra else '')
            if label not in labels: labels.append(label)
            color=group_colors[row.get('group','')] if len(groups)>1 else f'C{labels.index(label)%10}'
            ci=row['ci95_low_k']
            error=None if ci is None else [[row['theta_k']-ci],[row['ci95_high_k']-row['theta_k']]]
            artist = ax.errorbar([row[axis_name]],[row['theta_k']],yerr=error,
                        fmt='o' if row['branch']=='B increasing' else 's',
                        markerfacecolor=color if row['branch']=='B increasing' else 'none',color=color,capsize=3,
                        label=label if not any(line.get_label()==label for line in ax.containers) else '_nolegend_')
            point_artists[artist.lines[0]] = row
            mark_uncertainty(row[axis_name], row, color)
        ax.set_xlabel('E-field (V)' if axis_name=='efield' else 'Doping (V)')
        widths = sorted({r['halfwidth_t'] for r in payload['rows'] if r['is_primary']})
        ranges = '; '.join(f'−{width:g} to +{width:g} T' for width in widths) or 'unavailable'
        ax.set_title(f'θCW comparison · B fit range: {ranges}\n'
                     'Increasing: filled circles · Decreasing: open squares · Bars: 95% CI; ×: CI unreliable')
    range_note = ''
    if reliable_y and y_limits is None:
        if reliable_bounds:
            low, high = min(reliable_bounds), max(reliable_bounds)
            padding = max((high-low)*.08, max(abs(low),abs(high))*1e-6, .05)
            y_limits = (low-padding, high+padding)
            range_note = 'Auto Y: reliable 95% CI only · '
        else:
            ax.text(.99, .02, 'No reliable CI: showing full Y range', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='0.8', alpha=.9))
    if y_limits is not None:
        low, high = y_limits
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError('Y min must be smaller than Y max.')
        ax.set_ylim(low, high)
        outside = sum(value < low or value > high for value in plotted_values)
        ax.text(.99, .02, range_note + f'{outside} point(s) outside Y range', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='0.8', alpha=.9))
    ax.set_ylabel('θCW (K)')
    if use_methods or method!='inverse':
        names={'inverse':'Inverse / equal','slope':'Slope / equal','weighted':'Slope / SE weighted'}
        failed=sum(r.get('theta_k') is None for r in rows_to_show if axis_name=='sensitivity' or r['is_primary'])
        ax.set_title(names[method]+f" · ±{payload.get('parameters',{}).get('halfwidth_t','?')} T\nFilled: inc · open: dec · ×: diagnostic / CI unavailable",fontsize=9)
        if failed:ax.text(.02,.02,f'{failed} fit(s) unavailable',transform=ax.transAxes,fontsize=8)
    ax.axhline(0,color='0.5',linewidth=.8)
    ax.grid(alpha=.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7,loc='best')
    return point_artists


class ThetaComparisonDialog(QDialog):
    def __init__(self, entries, branches, output_root, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Compare θCW · temperature series')
        self.resize(1150,900)
        self.entries=copy.deepcopy(entries)
        self.branches=tuple(branches)
        self.output_root=Path(output_root)
        self.result=None
        self._point_artists = {}
        self.worker=None
        self.pool=OwnedWorkerPool(self)
        outer=QVBoxLayout(self)
        note=QLabel('Choose the same optical resonance for each condition; Group numbers are independent. '
                    'All curves are refitted with one B range and one temperature interval. '
                    'Zero background. Default: inverse slope / equal weights. Optional slope-space methods '
                    'use the same points; method comparison is a sensitivity check, not automatic selection.')
        note.setWordWrap(True)
        outer.addWidget(note)
        self.controls=QWidget()
        controls=QVBoxLayout(self.controls)
        controls.setContentsMargins(0,0,0,0)
        self.mapping=QTableWidget(len(entries),3)
        self.mapping.setHorizontalHeaderLabels(['Condition series','Optical energy group','Included records'])
        self.mapping.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mapping.setMaximumHeight(230)
        self.group_combos=[]
        for i,e in enumerate(self.entries):
            self.mapping.setItem(i,0,QTableWidgetItem(e['label']))
            combo=QComboBox()
            combo.addItem('Select matching resonance…',None)
            names=sorted({e['groups'][r.record_id] for r in e['records']})
            for name in names:
                records=[r for r in e['records'] if e['groups'][r.record_id]==name]
                combo.addItem(f"{name} · {min(r.center_ev for r in records):.6f}–{max(r.center_ev for r in records):.6f} eV · N={len(records)}",name)
            if len(names)==1: combo.setCurrentIndex(1)
            combo.currentIndexChanged.connect(self._invalidate)
            self.mapping.setCellWidget(i,1,combo)
            self.mapping.setItem(i,2,QTableWidgetItem(str(len(e['records']))))
            self.group_combos.append(combo)
        self.mapping.setColumnWidth(0,500)
        self.mapping.setColumnWidth(1,420)
        bulk_row = QHBoxLayout()
        bulk_row.addWidget(QLabel('Select group for all conditions:'))
        self.bulk_group_combo = QComboBox()
        names = {combo.itemData(i) for combo in self.group_combos for i in range(1, combo.count())}
        def group_order(name):
            number = name[6:] if name.startswith('Group ') else ''
            return (0, int(number)) if number.isdigit() else (1, name)
        for name in sorted(names, key=group_order):
            count = sum(combo.findData(name) >= 0 for combo in self.group_combos)
            self.bulk_group_combo.addItem(f'{name} ({count}/{len(self.entries)} conditions)', name)
        self.bulk_group_btn = QPushButton('Apply to all')
        self.bulk_group_btn.setEnabled(bool(names))
        self.bulk_group_btn.clicked.connect(self._apply_bulk_group)
        bulk_row.addWidget(self.bulk_group_combo)
        bulk_row.addWidget(self.bulk_group_btn)
        bulk_row.addStretch()
        controls.addLayout(bulk_row)
        self.all_groups_chk = QCheckBox('Compare all groups (same group names across conditions)')
        self.all_groups_chk.toggled.connect(self._group_mode_changed)
        controls.addWidget(self.all_groups_chk)
        controls.addWidget(self.mapping)
        row=QHBoxLayout()
        self.halfwidth=QDoubleSpinBox();self.halfwidth.setRange(.0001,1000);self.halfwidth.setDecimals(4);self.halfwidth.setValue(.2)
        self.t_min=QDoubleSpinBox();self.t_max=QDoubleSpinBox()
        for spin in (self.t_min,self.t_max): spin.setRange(0,100000);spin.setDecimals(4)
        self.t_max.setValue(1000)
        for label,spin in [('Common ±B (T)',self.halfwidth),('T min (K)',self.t_min),('T max (K)',self.t_max)]:
            row.addWidget(QLabel(label));row.addWidget(spin)
            spin.setKeyboardTracking(False);spin.valueChanged.connect(self._invalidate)
        self.temperature=QComboBox();self.temperature.addItem('Setpoint / catalog T','setpoint');self.temperature.addItem('Measured T only','measured')
        self.temperature.currentIndexChanged.connect(self._invalidate)
        row.addWidget(self.temperature)
        controls.addLayout(row)
        self.sensitivity=QCheckBox('Also compare ±0.1 / ±0.15 / ±0.2 / ±0.3 T')
        self.sensitivity.setChecked(True);self.sensitivity.toggled.connect(self._invalidate)
        controls.addWidget(self.sensitivity)
        method_row=QHBoxLayout()
        self.fit_method=QComboBox()
        for label,key in [('Inverse slope / equal weights','inverse'),('Slope / equal weights','slope'),('Slope / SE weighted','weighted')]:
            self.fit_method.addItem(label,key)
        self.fit_method.currentIndexChanged.connect(self._invalidate)
        self.methods_chk=QCheckBox('Compare three methods (shared axes)')
        self.methods_chk.toggled.connect(self._invalidate)
        method_row.addWidget(QLabel('Fit method:'));method_row.addWidget(self.fit_method);method_row.addWidget(self.methods_chk)
        controls.addLayout(method_row)
        outer.addWidget(self.controls)
        actions=QHBoxLayout()
        self.run_btn=QPushButton('Compare θCW');self.save_btn=QPushButton('Save comparison…');self.save_btn.setEnabled(False)
        self.axis_combo=QComboBox();self.axis_combo.addItem('θCW vs E-field','efield');self.axis_combo.addItem('θCW vs doping','doping');self.axis_combo.addItem('B range sensitivity','sensitivity')
        for widget in (self.run_btn,self.save_btn,self.axis_combo):actions.addWidget(widget)
        outer.addLayout(actions)
        self.status=QLabel('Select groups, then run the comparison.');self.status.setWordWrap(True);outer.addWidget(self.status)
        self.group_visibility_row = QHBoxLayout()
        self.group_visibility_row.addWidget(QLabel('Show groups:'))
        self.group_visibility_checks = {}
        outer.addLayout(self.group_visibility_row)
        y_row = QHBoxLayout()
        self.auto_y_chk = QCheckBox('Auto Y range')
        self.auto_y_chk.setChecked(True)
        y_row.addWidget(self.auto_y_chk)
        self.reliable_y_chk = QCheckBox('Use reliable CI only')
        self.reliable_y_chk.setToolTip('Auto Y uses finite 95% CI bounds of visible groups. Other points remain plotted. No R² filter.')
        self.reliable_y_chk.toggled.connect(self._draw)
        y_row.addWidget(self.reliable_y_chk)
        self.y_min = QDoubleSpinBox(); self.y_max = QDoubleSpinBox()
        for label, spin, value in [('Y min (K)', self.y_min, -25.), ('Y max (K)', self.y_max, 5.)]:
            spin.setRange(-1e9, 1e9); spin.setDecimals(3); spin.setValue(value)
            spin.setKeyboardTracking(False); spin.setEnabled(False)
            y_row.addWidget(QLabel(label)); y_row.addWidget(spin)
            spin.valueChanged.connect(self._draw)
        self.auto_y_chk.toggled.connect(self._y_mode_changed)
        self.y_range_note = QLabel()
        y_row.addWidget(self.y_range_note)
        outer.addLayout(y_row)
        self.tabs=QTabWidget()
        self.table=QTableWidget();self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.table,'Results')
        self.figure=Figure(figsize=(8,5),layout='constrained')
        self.canvas=ThemeAwareFigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self._hover_point)
        self.canvas.mpl_connect('figure_leave_event', lambda event: QToolTip.hideText())
        self.tabs.addTab(self.canvas,'Comparison plot')
        self.details=QPlainTextEdit();self.details.setReadOnly(True);self.tabs.addTab(self.details,'Diagnostics')
        outer.addWidget(self.tabs,1)
        self.run_btn.clicked.connect(self._run);self.save_btn.clicked.connect(self._save)
        self.axis_combo.currentIndexChanged.connect(self._draw)

    def _group_mode_changed(self, checked):
        self.mapping.setEnabled(not checked)
        self.bulk_group_combo.setEnabled(not checked)
        self.bulk_group_btn.setEnabled(not checked and self.bulk_group_combo.count()>0)
        self._invalidate()

    def _visible_groups(self):
        return {name for name, check in self.group_visibility_checks.items() if check.isChecked()}

    def _apply_bulk_group(self):
        name = self.bulk_group_combo.currentData()
        if name is None or self.worker is not None:
            return
        missing = []
        for row, combo in enumerate(self.group_combos):
            index = combo.findData(name)
            combo.blockSignals(True)
            combo.setItemText(0, f'{name} unavailable — select another group' if index < 0 else 'Select matching resonance…')
            combo.setCurrentIndex(max(0, index))
            combo.blockSignals(False)
            if index < 0:
                missing.append(str(row+1))
        self._invalidate()
        self.status.setText(
            f'{name}: applied to {len(self.entries)-len(missing)}/{len(self.entries)} conditions. '
            + (f'Missing in row(s) {", ".join(missing)}; choose those groups individually.' if missing
               else 'Check resonance energies, then click Compare θCW.'))

    def _invalidate(self,*_):
        self.result=None
        self._point_artists = {}
        QToolTip.hideText()
        if hasattr(self,'save_btn'):
            self.save_btn.setEnabled(False);self.table.setRowCount(0);self.figure.clear();self.canvas.draw_idle()
            self.details.clear();self.status.setText('Settings changed; run comparison again.')

    def _run(self):
        entries=[]
        for e,combo in zip(self.entries,self.group_combos):
            if self.all_groups_chk.isChecked():
                for group in sorted({e['groups'][r.record_id] for r in e['records']}):
                    if group.startswith('Unassigned '):
                        continue
                    entries.append(dict(series_id=e['series_id'],group=group,fixed_conditions=e['fixed_conditions'],
                                        records=[r for r in e['records'] if e['groups'][r.record_id]==group]))
                continue
            group=combo.currentData()
            if group is None:
                self.status.setText('Choose an optical group for every series.');return
            entries.append(dict(series_id=e['series_id'],group=group,fixed_conditions=e['fixed_conditions'],
                                records=[r for r in e['records'] if e['groups'][r.record_id]==group]))
        self._invalidate()
        self.controls.setEnabled(False);self.run_btn.setEnabled(False)
        self.worker=Worker(compare_theta,entries,halfwidth=self.halfwidth.value(),t_min=self.t_min.value(),t_max=self.t_max.value(),
            temperature_mode=self.temperature.currentData(),branches=self.branches,
            sensitivity_widths=[.1,.15,.2,.3] if self.sensitivity.isChecked() else [],
            compare_methods=self.methods_chk.isChecked() or self.fit_method.currentData()!='inverse')
        self.worker.signals.result.connect(self._accept_result)
        self.worker.signals.error.connect(lambda message:self.status.setText(str(message)))
        self.worker.signals.finished.connect(self._finished)
        self.status.setText('Fitting selected temperature series…')
        self.pool.start_worker(self.worker)

    def _finished(self):
        self.worker=None;self.controls.setEnabled(True);self.run_btn.setEnabled(True)

    def _accept_result(self,result):
        self.result=result;self.save_btn.setEnabled(True)
        for check in self.group_visibility_checks.values():
            self.group_visibility_row.removeWidget(check)
            check.deleteLater()
        self.group_visibility_checks = {}
        for name in sorted({r['group'] for r in result['rows']}):
            check = QCheckBox(name)
            check.setChecked(True)
            check.toggled.connect(self._draw)
            self.group_visibility_row.addWidget(check)
            self.group_visibility_checks[name] = check
        display_rows=(result.get('method_rows') or result['rows'])
        if not self.methods_chk.isChecked() and result.get('method_rows'):
            display_rows=[r for r in display_rows if r['method']==self.fit_method.currentData()]
        fields=['series_id','group','method','doping','efield','branch','halfwidth_t','is_primary','n',
                'temperature_min_k','temperature_max_k','theta_k','ci95_low_k','ci95_high_k','r_squared','status','diagnostics']
        self.table.setColumnCount(len(fields));self.table.setHorizontalHeaderLabels(fields);self.table.setRowCount(len(display_rows))
        for i,row in enumerate(display_rows):
            for j,key in enumerate(fields):
                value=row.get(key)
                self.table.setItem(i,j,QTableWidgetItem('—' if value is None else f'{value:.5g}' if isinstance(value,float) else str(value)))
        self.table.resizeColumnsToContents()
        good=sum(r['status']=='ok' and r['is_primary'] for r in display_rows)
        self.status.setText(f'{good} primary branch fits succeeded. '+' '.join(result['warnings']))
        self.details.setPlainText(result['assumptions']+'\n\n'+'\n'.join(
            f"{r['series_id']} / {r['group']} / {r['branch']} / {r.get('method','inverse')} / ±{r['halfwidth_t']:g} T: {r['status']}; {r['diagnostics']}" for r in display_rows))
        self._draw()

    def _y_limits(self):
        return None if self.auto_y_chk.isChecked() else (self.y_min.value(), self.y_max.value())

    def _y_mode_changed(self, automatic):
        self.reliable_y_chk.setEnabled(automatic)
        self.y_min.setEnabled(not automatic)
        self.y_max.setEnabled(not automatic)
        self._draw()

    def _hover_point(self, event):
        hits = []
        for artist, row in self._point_artists.items():
            if event.inaxes is not artist.axes:
                continue
            x, y = artist.get_xdata()[0], artist.get_ydata()[0]
            if not (min(artist.axes.get_ylim()) <= y <= max(artist.axes.get_ylim())
                    and min(artist.axes.get_xlim()) <= x <= max(artist.axes.get_xlim())):
                continue
            if artist.contains(event)[0]:
                hits.append(comparison_point_details(row))
        if hits:
            QToolTip.showText(QCursor.pos(), '\n\n'.join(hits), self.canvas)
        else:
            QToolTip.hideText()

    def _draw(self,*_):
        QToolTip.hideText()
        limits = self._y_limits()
        if limits is not None and limits[0] >= limits[1]:
            self.y_range_note.setText('Y min must be less than Y max')
            return
        self.y_range_note.clear()
        if self.result:
            self._point_artists = draw_comparison(self.figure,self.result,self.axis_combo.currentData(),self._visible_groups(),limits,
                            self.auto_y_chk.isChecked() and self.reliable_y_chk.isChecked(),
                            method='all' if self.methods_chk.isChecked() else self.fit_method.currentData());self.canvas.draw_idle()

    def export_to(self,folder):
        if self.result is None: raise ValueError('Run the comparison first.')
        limits = self._y_limits()
        if limits is not None and limits[0] >= limits[1]:
            raise ValueError('Y min must be smaller than Y max.')
        root=dependency_export_folder(folder,'Temperature')/'Theta_CW_comparison'
        target=root/datetime.now().strftime('%Y%m%d_%H%M%S_%f');target.mkdir(parents=True,exist_ok=False)
        reliable_y = self.auto_y_chk.isChecked() and self.reliable_y_chk.isChecked()
        method='all' if self.methods_chk.isChecked() else self.fit_method.currentData()
        (target/'comparison.json').write_text(json.dumps(dict(self.result, visible_method=method,visible_groups=sorted(self._visible_groups()), y_limits_k=limits, reliable_ci_auto_y=reliable_y),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        for filename,key in [('summary.csv','rows'),('points.csv','points'),('methods.csv','method_rows')]:
            records=self.result.get(key,[])
            fields=list(dict.fromkeys(k for r in records for k in r))
            with (target/filename).open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                writer.writerows({k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()} for r in records)
        for axis in ('efield','doping','sensitivity'):
            fig=Figure(figsize=(16,6) if method=='all' else (9,6),layout='constrained')
            draw_comparison(fig,self.result,axis,self._visible_groups(),limits,reliable_y,method=method);fig.savefig(target/f'theta_{axis}.png',dpi=180)
        (target/'diagnostics.txt').write_text(self.details.toPlainText(),encoding='utf-8')
        return target

    def _save(self):
        folder=QFileDialog.getExistingDirectory(self,'Save θCW comparison',str(self.output_root))
        if folder:
            try:
                target=self.export_to(folder)
                self.status.setText(f'Saved: {target}')
            except (OSError,ValueError) as exc: QMessageBox.warning(self,'Export failed',str(exc))

    def reject(self):
        if self.worker is not None:
            self.status.setText('Wait for the running comparison to finish before closing.');return
        super().reject()

    def closeEvent(self,event):
        if self.worker is not None: event.ignore()
        else: super().closeEvent(event)
