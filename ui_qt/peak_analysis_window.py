"""Independent English Peak Analysis window; no main-app window imports."""
from copy import deepcopy
from pathlib import Path
from threading import Event
from uuid import uuid4

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import Qt, QEvent, QThreadPool, QTimer
from PySide6.QtGui import QColor, QAction
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QScrollArea, QPushButton, QCheckBox, QSlider, QLabel, QProgressBar,
    QListWidgetItem, QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMenu, QTabBar)

from core.peak_workspace import save_workspace, load_workspace, atomic_json, session_directory, recommended_detection_method
from core.peak_plotting import build_figure, draw_spectrum, branch_color, overlay_payload, resize_overlay
from core.peak_candidates import detect_heatmap, filter_heatmap, detection_settings
from core.peak_quality import prepare_quality_cache
from ui_qt.common import Worker, QSpinBox
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
from ui_qt.peak_analysis_controls import PeakControls
from ui_qt.peak_cursor_readout import PeakCursorReadout
from ui_qt.peak_analysis_ipc import send_message


def snapshot(dataset):
    return {k: v if k=='cube' else deepcopy(v) for k,v in dataset.items()}


class AnalysisToolbar(NavigationToolbar2QT):
    def changeEvent(self,event):
        super().changeEvent(event)
        if event.type()==QEvent.PaletteChange and hasattr(self,'_actions'):
            for _,_,icon,callback in self.toolitems:
                if icon and callback in self._actions:
                    self._actions[callback].setIcon(self._icon(icon+'.png'))

    def save_figure(self,*args):
        with self.canvas.publication_context():
            super().save_figure(*args)


class PeakAnalysisWindow(QMainWindow):
    def __init__(self, session_path, reply_server=''):
        super().__init__()
        self.setWindowTitle('DPTK · Peak Analysis'); self.resize(1180,820)
        self.session_path=Path(session_path);self.reply_server=reply_server
        self.datasets={};self.dataset_groups={};self.active_key=None;self.busy=None;self.cancel_event=Event();self.generation=0
        self.restart_requested=False;self._closing=False;self._close_ready=False;self._syncing=False
        self._view_range_timer=QTimer(self);self._view_range_timer.setSingleShot(True)
        self._view_range_timer.setInterval(200);self._view_range_timer.timeout.connect(self.apply_view_range)
        self.thread_pool=QThreadPool(self);self.thread_pool.setMaxThreadCount(1);self.pending=[]
        self.figure=Figure(figsize=(8,7));self.canvas=ThemeAwareFigureCanvasQTAgg(self.figure)
        self.heat_ax=self.spectrum_ax=self.residual_ax=self.gate_line=None
        central=QWidget();layout=QVBoxLayout(central);layout.setContentsMargins(8,6,8,0)
        bar=QHBoxLayout();self.dataset_label=QLabel();self.dataset_label.setMinimumWidth(0)
        self.dataset_label.setTextInteractionFlags(Qt.TextSelectableByMouse);bar.addWidget(self.dataset_label,1)
        self.dataset_tabs=QTabBar();self.dataset_tabs.setExpanding(False);self.dataset_tabs.setUsesScrollButtons(True)
        self.dataset_tabs.setElideMode(Qt.ElideMiddle);self.dataset_tabs.setAccessibleName('Analysis files');bar.addWidget(self.dataset_tabs,1)
        self.results_button=QPushButton('Results');self.export_button=QPushButton('Export...');self.apply_button=QPushButton('Apply overlay')
        for button in (self.results_button,self.export_button,self.apply_button):bar.addWidget(button)
        self.more_button=QPushButton('More');menu=QMenu(self.more_button)
        restart=QAction('Restart Analysis',self);restart.triggered.connect(self.request_restart);menu.addAction(restart)
        save_as=QAction('Save workspace copy...',self);save_as.triggered.connect(self.save_as);menu.addAction(save_as)
        details=QAction('Task details',self);details.triggered.connect(self.show_task_details);menu.addAction(details)
        self.more_button.setMenu(menu);bar.addWidget(self.more_button);layout.addLayout(bar)
        splitter=QSplitter();plot=QWidget();pl=QVBoxLayout(plot);pl.setContentsMargins(0,0,0,0)
        self.product_tabs=QTabBar();self.product_tabs.setExpanding(False)
        self.product_tabs.setAccessibleName('DRR product')
        self.product_tabs.addTab('ΔR/R');self.product_tabs.addTab('2nd derivative')
        self.product_tabs.hide();pl.addWidget(self.product_tabs)
        self.product_tabs.currentChanged.connect(self.select_product)
        toolrow=QHBoxLayout();self.navigation=AnalysisToolbar(self.canvas,self,coordinates=False);toolrow.addWidget(self.navigation,1)
        self.show_overlay=QCheckBox('Show peaks');self.show_overlay.setChecked(True);toolrow.addWidget(self.show_overlay)
        self.marker_size=QSpinBox();self.marker_size.setRange(1,64);self.marker_size.setValue(9)
        self.marker_size.setKeyboardTracking(False);self.marker_size.setAccessibleName('Heatmap marker size')
        self.marker_size.setToolTip('Heatmap marker area in pt². Display only; detection and filtering are unchanged.')
        toolrow.addWidget(QLabel('Size'));toolrow.addWidget(self.marker_size)
        pl.addLayout(toolrow);pl.addWidget(self.canvas,1)
        self.cursor_readout=PeakCursorReadout(self.canvas,plot);pl.addWidget(self.cursor_readout)
        row=QHBoxLayout();row.addWidget(QLabel('Spectrum'));self.row_slider=QSlider(Qt.Horizontal);row.addWidget(self.row_slider,1)
        self.row_label=QLabel();row.addWidget(self.row_label);pl.addLayout(row);splitter.addWidget(plot)
        self.controls=PeakControls();scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(self.controls)
        scroll.setMinimumWidth(270);scroll.setMaximumWidth(420);splitter.addWidget(scroll);splitter.setSizes([850,300])
        splitter.setStretchFactor(0,1);layout.addWidget(splitter,1);self.setCentralWidget(central)
        self.status_label=QLabel('Open a plotted dataset from the app to begin.');self.status_label.setWordWrap(False)
        self.statusBar().addWidget(self.status_label,1);self.progress=QProgressBar();self.progress.setMaximumWidth(130);self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress);self.cancel_button=QPushButton('Cancel');self.cancel_button.hide();self.statusBar().addPermanentWidget(self.cancel_button)
        self.last_error='';self.dataset_tabs.currentChanged.connect(self.select_dataset)
        self.row_slider.valueChanged.connect(self.select_row)
        self.controls.preview.clicked.connect(self.preview);self.controls.track.clicked.connect(lambda:self.analyze('track'))
        self.controls.fit.clicked.connect(lambda:self.analyze('fit'));self.controls.settings_changed.connect(self.settings_changed)
        self.controls.branches.itemChanged.connect(self.branch_changed)
        self.controls.branches.itemClicked.connect(self.branch_clicked)
        self.controls.remove.clicked.connect(self.remove_branch)
        self.controls.residual.toggled.connect(self.redraw);self.controls.position.currentIndexChanged.connect(self.position_changed)
        self.show_overlay.toggled.connect(self.overlay_changed)
        self.marker_size.valueChanged.connect(self.marker_size_changed)
        self.controls.show_rejected.toggled.connect(self.rejected_changed)
        self.controls.reset_filters.clicked.connect(self.reset_filters)
        self.results_button.clicked.connect(self.show_results);self.export_button.clicked.connect(self.export)
        self.apply_button.clicked.connect(self.apply_overlay);self.cancel_button.clicked.connect(lambda:self.cancel_event.set())
        self.canvas.mpl_connect('button_press_event',self.plot_clicked)
        self.update_actions()

    @property
    def dataset(self):
        return self.datasets.get(self.active_key)

    def add_datasets(self, datasets):
        if self.busy:
            self.pending.extend(datasets);return
        for d in datasets:
            # Keep prior analyses accessible without offering two versions of
            # the same product behind a single tab.
            if d['kind']=='DRR':
                for other in self.datasets.values():
                    if (other['key']!=d['key'] and other['kind']=='DRR' and other['source']==d['source']
                            and other['channel']==d['channel'] and other.get('group_id','legacy')==d.get('group_id','legacy')):
                        other['group_id']='previous:'+other['key']
            previous=self.datasets.get(d['key'])
            if previous is not None:
                previous['view']=d.get('view');previous['reply_server']=d.get('reply_server',self.reply_server)
                previous['row']=d.get('row',previous.get('row',0))
                for field in ('group_id','source_name','product_label','source_processing'):
                    if field in d:previous[field]=d[field]
            else:
                self.datasets[d['key']]=d
            self.active_key=d['key']
        self.rebuild_dataset_choices()
        self.select_dataset()
        self.status_label.setText('Ready. Find all peaks detects the full heatmap; filters reuse the candidate cache.')

    def rebuild_dataset_choices(self):
        groups={}
        for d in self.datasets.values():
            group=('DRR',d['source'],d.get('group_id','legacy')) if d['kind']=='DRR' and d['channel'] in ('raw','second') else ('single',d['key'])
            groups.setdefault(group,[]).append(d['key'])
        blocked=self.dataset_tabs.blockSignals(True)
        while self.dataset_tabs.count():self.dataset_tabs.removeTab(0)
        self.dataset_groups={}
        counts={};selected=0
        for keys in groups.values():
            first=self.datasets[keys[0]]
            label=first.get('source_name',first['name'])
            if first['kind']=='DRR':label=label.replace(' · raw · ',' · ').replace(' · second · ',' · ')
            counts[label]=counts.get(label,0)+1
            suffix=f' ({counts[label]})' if counts[label]>1 else ''
            index=self.dataset_tabs.addTab(label+suffix);self.dataset_tabs.setTabData(index,keys[0]);self.dataset_tabs.setTabToolTip(index,label+suffix)
            self.dataset_groups[keys[0]]=keys
            if self.active_key in keys:selected=self.dataset_tabs.count()-1
        self.dataset_tabs.setCurrentIndex(selected);self.dataset_tabs.blockSignals(blocked)
        self.dataset_tabs.setVisible(self.dataset_tabs.count()>1);self.dataset_label.setVisible(self.dataset_tabs.count()<=1)
        self.dataset_label.setText(self.dataset_tabs.tabText(selected));self.dataset_label.setToolTip(self.dataset_label.text())

    def current_group(self):
        return self.dataset_groups.get(self.dataset_tabs.tabData(self.dataset_tabs.currentIndex()),[])

    def select_product(self,index):
        if self._syncing or self.busy or not self.dataset:return
        channel=('raw','second')[index]
        target=next((self.datasets[key] for key in self.current_group() if self.datasets[key]['channel']==channel),None)
        if target is None:return
        current=self.dataset
        if target is not current:
            gate=float(current['cube'].gate[self.row_slider.value()])
            target['row']=int(np.argmin(abs(target['cube'].gate-gate)))
            target['view']={**(target.get('view') or {}),**{key:value for key,value in (current.get('view') or {}).items() if key in ('xlim','ylim')}}
        self.active_key=target['key'];self.select_dataset()

    def select_dataset(self, *_):
        if self.busy:return
        keys=self.current_group()
        if self.active_key not in keys:
            channel=('raw','second')[max(0,self.product_tabs.currentIndex())]
            self.active_key=next((key for key in keys if self.datasets[key]['channel']==channel),keys[0] if keys else None)
        d=self.dataset
        if not d:return
        self._syncing=True;self.controls.load(d)
        self.product_tabs.setVisible(d['kind']=='DRR' and d['channel'] in ('raw','second'))
        blocked=self.product_tabs.blockSignals(True)
        for index,(channel,label) in enumerate((('raw','ΔR/R'),('second','2nd derivative'))):
            product=next((self.datasets[key] for key in keys if self.datasets[key]['channel']==channel),None)
            self.product_tabs.setTabEnabled(index,product is not None)
            self.product_tabs.setTabText(index,product.get('product_label',label) if product else label)
            self.product_tabs.setTabToolTip(index,label if product else 'Display this product in the app and open Peak Analysis to add it.')
        if d['kind']=='DRR':self.product_tabs.setCurrentIndex(1 if d['channel']=='second' else 0)
        self.product_tabs.blockSignals(blocked)
        self.row_slider.setRange(0,len(d['cube'].gate)-1)
        self.row_slider.setValue(min(max(0,int(d.get('row',0))),len(d['cube'].gate)-1))
        self.show_overlay.setChecked(d.get('show_overlay',True))
        self.marker_size.setValue(int(d.get('marker_size',9)))
        self.controls.position.setCurrentIndex(1 if d.get('overlay_position')=='fitted' else 0)
        self.populate_branches();self._syncing=False;self.update_cache_status();self.redraw();self.update_actions()

    def populate_branches(self):
        old=self.controls.branches.blockSignals(True);self.controls.branches.clear()
        for b in (self.dataset or {}).get('branches',[]):
            item=QListWidgetItem(b['name']);item.setData(Qt.UserRole,b['id'])
            item.setFlags(item.flags()|Qt.ItemIsUserCheckable|Qt.ItemIsEditable)
            item.setCheckState(Qt.Checked if b.get('enabled',True) else Qt.Unchecked)
            item.setForeground(QColor(branch_color(b)))
            item.setToolTip(f"{b['polarity'].title()} seed: {b['seed_energy']:.6g} eV at {b['seed_y']:.6g}")
            self.controls.branches.addItem(item)
        self.controls.branches.blockSignals(old)

    def update_actions(self):
        d=self.dataset;idle=self.busy is None;ready=bool(d and d.get('result'))
        self.dataset_tabs.setEnabled(idle)
        self.product_tabs.setEnabled(bool(d) and idle)
        self.controls.setEnabled(bool(d) and idle)
        self.controls.track.setEnabled(bool(d and d['branches']) and idle)
        self.controls.fit.setEnabled(bool(d and d['branches']) and idle)
        manual=bool(d and (not d.get('candidates') or d.get('manual_seeds')))
        self.controls.numeric['noise_sigma'].setEnabled(manual);self.controls.max_gap.setEnabled(manual)
        for b in (self.results_button,self.export_button,self.apply_button):b.setEnabled(ready and idle)

    def _start(self, title, work, accept):
        if self.busy:return
        self.cancel_event=Event();cancel=self.cancel_event;self.generation+=1;generation=self.generation
        def run(*,progress,log):return work(cancel.is_set,progress.emit)
        worker=Worker(run);self.busy=worker;self.progress.setRange(0,100);self.progress.setValue(0);self.progress.show()
        self.status_label.setText(title);self.cancel_button.show();self.update_actions()
        def done(result):
            if generation==self.generation and not cancel.is_set():accept(result)
        def failed(message):
            self.last_error=message
            self.status_label.setText('Cancelled; previous results kept.' if cancel.is_set() else message.split('\n')[0])
        def finished():
            self.busy=None;self.progress.hide();self.cancel_button.hide();self.update_actions()
            if cancel.is_set():self.status_label.setText('Cancelled; previous results kept.')
            if self._closing:
                QTimer.singleShot(0,self.close)
            elif self.pending:
                pending,self.pending=self.pending,[];self.add_datasets(pending)
        worker.signals.result.connect(done);worker.signals.error.connect(failed)
        worker.signals.progress.connect(self.progress.setValue);worker.signals.finished.connect(finished)
        self.thread_pool.start(worker)

    def settings_changed(self):
        if self._syncing or not self.dataset:return
        d=self.dataset;settings=self.controls.settings(d)
        d['scope']=self.controls.scope.currentIndex()
        if d['scope']==2:d['custom_range']={key:settings[key] for key in self.controls.bounds}
        else:self.controls.sync_range_values(settings)
        if settings!=d['settings']:
            previous=d['settings'];d['settings']=settings
            self.update_cache_status()
            if d.get('candidates'):
                fields=set(settings)-set(detection_settings(settings))
                if any(settings[k]!=previous.get(k) for k in fields):self.refilter()
            else:
                self.status_label.setText('Settings changed. Find all peaks to build the candidate cache.')

    def update_cache_status(self):
        d=self.dataset
        if not d:return
        pool=d.get('candidates');result=d.get('result') or {}
        if not pool:
            self.controls.cache_status.setText('No candidates yet.');self.controls.cache_status.setToolTip('');return
        text=f'{len(result.get("points",[])):,} kept / {len(pool["points"]):,} candidates · {pool["rows"]} spectra'
        sg=pool['settings']['detection_method']=='sg'
        derivative=recommended_detection_method(d['kind'],d['channel'],d.get('product_label',''))=='local'
        method=(('Extra SG' if derivative else 'SG')+f' ({pool["settings"]["smoothing_window"]} samples) + extrema' if sg else 'Local extrema')
        text+='\nCached: '+method
        details=pool['method']
        counts=result.get('filter_counts')
        if counts:
            details+=f'\nShape/SNR: {counts["shape_snr"]:,} · Supported: {counts["supported"]:,}'
        self.controls.cache_status.setToolTip(details)
        if not pool.get('quality'):
            text+='\nLegacy cache: choose a preset to add noise filtering.'
        if detection_settings(d['settings'])!=pool['settings']:
            text+='\nDetection settings changed. Find all peaks to rebuild; current filters use the existing cache.'
        if d['settings']['prominence']<pool['settings']['candidate_floor']:
            text+='\nBelow cached floor: lower Candidate floor and find again for weaker peaks.'
        self.controls.cache_status.setText(text)

    def accept_filtered(self,d,result):
        manual={b['id']:b for b in d['branches'] if b.get('manual')}
        for b in result['branches']:
            if b['id'] in manual:b.update(manual[b['id']])
        present={b['id'] for b in result['branches']}
        result['branches'].extend(b for key,b in manual.items() if key not in present)
        d['result']=result;d['branches']=result['branches'];d['branch_catalog']=result['branch_catalog']
        d['manual_seeds']=bool(manual)
        d['candidate_assignments']=result['assignments'];d['overlay_position']='detected'
        d['segment_assignments']=result['segment_assignments']
        old=self.controls.position.blockSignals(True);self.controls.position.setCurrentIndex(0);self.controls.position.blockSignals(old)
        self.populate_branches();self.update_cache_status();self.redraw()
        self.status_label.setText(f'{len(result["points"]):,} kept / {len(d["candidates"]["points"]):,} candidates. Filters reuse cached measurements.')

    def refilter(self):
        if self.busy or not self.dataset.get('candidates'):return
        d=self.dataset;copy=snapshot(d);copy['candidates']=d['candidates']
        def work(cancel,progress):
            pool=prepare_quality_cache(copy,copy['candidates'],cancel,lambda n:progress(n//3))
            copy['candidates']=pool
            return pool,filter_heatmap(copy,copy['settings'],cancel,lambda n:progress(33+n*2//3))
        def accept(payload):
            d['candidates'],result=payload
            self.accept_filtered(d,result)
        self._start('Filtering cached candidates...',work,accept)

    def reset_filters(self):
        if not self.dataset or self.busy:return
        self._syncing=True
        self.controls.scope.setCurrentIndex(0);self.controls.polarity.setCurrentIndex(0)
        for name in ('prominence','min_width_mev','min_distance_mev','min_snr'):self.controls.numeric[name].setValue(0)
        self.controls.support.setValue(0)
        self.controls.max_width.setValue(0);self.controls.max_count.setValue(0)
        self._syncing=False;self.settings_changed()

    def rejected_changed(self,checked):
        if self.dataset and not self._syncing:
            self.dataset['show_rejected']=checked;self.redraw()

    def adapter(self):
        if self.dataset['kind']=='PL':
            from core import pl_peak_analysis
            return pl_peak_analysis
        from core import drr_peak_adapter
        return drr_peak_adapter

    def preview(self):
        if not self.dataset or self.busy:return
        d=self.dataset;d['settings']=self.controls.settings(d);copy=snapshot(d)
        self.controls.sync_range_values(d['settings'])
        def work(cancel,progress):
            pool=detect_heatmap(copy,copy['settings'],cancel,lambda n:progress(n*3//4))
            copy['candidates']=pool
            return pool,filter_heatmap(copy,copy['settings'],cancel,lambda n:progress(75+n//4))
        def accept(payload):
            d['candidates'],result=payload
            d['show_overlay']=True
            blocked=self.show_overlay.blockSignals(True);self.show_overlay.setChecked(True);self.show_overlay.blockSignals(blocked)
            self.accept_filtered(d,result)
        self._start('Finding candidates across the full heatmap...',work,accept)

    def analyze(self, mode):
        if not self.dataset or self.busy:return
        d=self.dataset;d['settings']=self.controls.settings(d);copy=snapshot(d);adapter=self.adapter()
        def work(cancel,progress):
            if copy.get('candidates') and not copy.get('manual_seeds'):
                result=filter_heatmap(copy,copy['settings'],cancel,lambda n:progress(n//2 if mode=='fit' else n))
                selected={b['id'] for b in copy['branches'] if b.get('enabled',True)}
                result['points']=[p for p in result['points'] if p['branch_id'] in selected]
                result['candidate_mode']=False
            else:
                result=adapter.track(copy,copy['branches'],copy['settings'],cancel,lambda n:progress(n//2 if mode=='fit' else n))
            if mode=='fit':result=adapter.fit(copy,result,copy['settings'],cancel,lambda n:progress(50+n//2))
            return result
        def accept(result):
            d['result']=result
            if mode=='fit':
                d['overlay_position']='fitted'
                blocked=self.controls.position.blockSignals(True);self.controls.position.setCurrentIndex(1);self.controls.position.blockSignals(blocked)
            good=sum(p['status']=='accepted' for p in result['points'])
            message=f'{good} positions across {len(result["branches"])} branches.'
            if mode=='fit':message+=f' {sum(p.get("fit_status")=="ok" for p in result["points"])} valid fits.'
            if result['notices']:message+=' '+result['notices'][0]
            self.status_label.setText(message);self.last_error='\n'.join(result['notices']);self.update_cache_status();self.redraw()
        self._start('Fitting selected branches...' if mode=='fit' else 'Tracking selected branches...',work,accept)

    def branch_changed(self, item):
        if self._syncing or not self.dataset:return
        b=next(b for b in self.dataset['branches'] if b['id']==item.data(Qt.UserRole))
        b['name']=item.text().strip() or b['name'];b['enabled']=item.checkState()==Qt.Checked
        self.redraw()

    def branch_clicked(self,item):
        if self.dataset:
            b=next(b for b in self.dataset['branches'] if b['id']==item.data(Qt.UserRole))
            self.row_slider.setValue(b['seed_row'])

    def remove_branch(self):
        if not self.dataset:return
        item=self.controls.branches.currentItem()
        if item is None:return
        identity=item.data(Qt.UserRole)
        for b in self.dataset['branches']:
            if b['id']==identity:b['enabled']=False
        self.populate_branches();self.redraw();self.update_actions()

    def plot_clicked(self,event):
        if not self.dataset or event.xdata is None or event.ydata is None or self.navigation.mode:return
        d=self.dataset;cube=d['cube']
        if event.inaxes is self.heat_ax:
            self.row_slider.setValue(int(np.argmin(abs(cube.gate-event.ydata))))
        elif event.inaxes is self.spectrum_ax and self.controls.seed.isChecked() and not self.busy:
            row=self.row_slider.value();color=max((b['color'] for b in d['branches']),default=-1)+1
            polarity='dip' if d['kind']=='DRR' and self.controls.polarity.currentIndex()==2 else 'peak'
            # In Both mode infer the nearest local extremum using the same detector.
            candidates=self.adapter().detect(d,row,d['settings'])
            near=min(candidates,key=lambda b:abs(b['seed_energy']-event.xdata)) if candidates else None
            if near and abs(near['seed_energy']-event.xdata)<=d['settings']['max_shift_mev']/1000:
                energy=near['seed_energy'];polarity=near['polarity']
            else:energy=float(event.xdata)
            existing=next((b for b in d['branches'] if b['polarity']==polarity and abs(b['seed_energy']-energy)<.0001),None)
            if existing:existing.update(seed_energy=energy,seed_y=float(cube.gate[row]),seed_row=row,manual=True)
            else:d['branches'].append(dict(id=uuid4().hex,name=f"{'D' if polarity=='dip' else 'P'}{color+1}",
                color=color,enabled=True,manual=True,polarity=polarity,seed_energy=energy,seed_y=float(cube.gate[row]),seed_row=row,width_mev=5.))
            d['result']=None;d['manual_seeds']=True;self.controls.seed.setChecked(False);self.populate_branches();self.redraw();self.update_actions()

    def position_changed(self):
        if not self._syncing and self.dataset:
            self.dataset['overlay_position']='fitted' if self.controls.position.currentIndex() else 'detected';self.redraw()

    def overlay_changed(self):
        if not self._syncing and self.dataset:
            self.dataset['show_overlay']=self.show_overlay.isChecked();self.redraw()

    def marker_size_changed(self,value):
        if self._syncing or not self.dataset:return
        self.dataset['marker_size']=int(value)
        if self.heat_ax is not None:
            resize_overlay(self.heat_ax,value);self.canvas.draw_idle()

    def apply_view_range(self):
        if self._closing or not self.dataset or self.controls.scope.currentIndex()!=1:return
        # Wait until panning/zooming and any previous filter have completed.
        if self.busy or QApplication.mouseButtons()!=Qt.NoButton:
            self._view_range_timer.start();return
        self.settings_changed()

    def redraw(self,*_):
        if self._syncing or not self.dataset:return
        self.cursor_readout.bind()
        d=self.dataset
        try:
            self.heat_ax,self.spectrum_ax,self.residual_ax,self.gate_line=build_figure(d,self.row_slider.value(),
                self.show_overlay.isChecked(),self.controls.residual.isChecked() and d['kind']=='PL',self.figure)
            self.cursor_readout.bind(d['cube'],self.heat_ax,self.spectrum_ax,self.residual_ax)
            heat=self.heat_ax
            def remember_view(_):
                d['view']={**(d.get('view') or {}),'xlim':list(heat.get_xlim()),
                           'ylim':list(heat.get_ylim()),'y_axis_log':heat.get_yscale()=='log'}
                if d is self.dataset and self.controls.scope.currentIndex()==1:
                    settings=self.controls.settings(d)
                    self.controls.sync_range_values(settings)
                    if any(settings[key]!=d['settings'][key] for key in self.controls.bounds):
                        self._view_range_timer.start()
            heat.callbacks.connect('xlim_changed',remember_view)
            heat.callbacks.connect('ylim_changed',remember_view)
            remember_view(heat)
            self.row_label.setText(f'{d["cube"].gate[self.row_slider.value()]:.6g}')
            self.canvas.draw_idle()
        except ValueError as exc:self.status_label.setText(str(exc))

    def select_row(self,row):
        if self._syncing or not self.dataset or self.spectrum_ax is None:return
        self.cursor_readout.clear_reading()
        d=self.dataset;d['row']=row;self.row_label.setText(f'{d["cube"].gate[row]:.6g}')
        xlim=self.spectrum_ax.get_xlim();draw_spectrum(d,row,self.spectrum_ax,self.residual_ax);self.spectrum_ax.set_xlim(xlim)
        self.gate_line.set_ydata([float(d['cube'].gate[row])]*2);self.canvas.draw_idle()

    def show_results(self):
        if not self.dataset or not self.dataset['result']:return
        dialog=QDialog(self);dialog.setWindowTitle('Peak results');dialog.resize(800,420);layout=QVBoxLayout(dialog)
        method=QLabel(self.dataset['result']['method']);method.setWordWrap(True);layout.addWidget(method)
        table=QTableWidget();fields=['branch','y','energy','snr','neighbor_support','neighbor_available','prominence_fraction','width_mev','fit_center','fwhm_mev','status','fit_status']
        table.setColumnCount(len(fields));table.setHorizontalHeaderLabels(['Branch','Scan coordinate','Detected (eV)','SNR','Supporting rows','Available rows','Prominence fraction','Width at half prominence (meV)','Fitted (eV)','Fitted FWHM (meV)','Association','Fit status'])
        points=self.dataset['result']['points'];names={b['id']:b['name'] for b in self.dataset['branches']}
        table.setRowCount(len(points));table.setEditTriggers(QTableWidget.NoEditTriggers)
        for row,p in enumerate(points):
            for col,key in enumerate(fields):
                value=names.get(p['branch_id'],p['branch_id']) if key=='branch' else p.get(key)
                table.setItem(row,col,QTableWidgetItem('' if value is None else f'{value:.7g}' if isinstance(value,float) else str(value)))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents);layout.addWidget(table)
        table.cellClicked.connect(lambda row,col:self.row_slider.setValue(points[row]['row_index']))
        close=QPushButton('Close');close.clicked.connect(dialog.accept);layout.addWidget(close);dialog.exec()

    def show_task_details(self):
        from PySide6.QtWidgets import QPlainTextEdit
        dialog=QDialog(self);dialog.setWindowTitle('Task details');layout=QVBoxLayout(dialog)
        text=QPlainTextEdit(self.last_error or 'No task errors.');text.setReadOnly(True);layout.addWidget(text)
        close=QPushButton('Close');close.clicked.connect(dialog.accept);layout.addWidget(close);dialog.resize(650,330);dialog.exec()

    def export(self):
        if not self.dataset or not self.dataset['result']:return
        folder=QFileDialog.getExistingDirectory(self,'Export peak analysis')
        if not folder:return
        from core.peak_export import export_dataset
        d=snapshot(self.dataset)
        self._start('Exporting CSV, settings and PNG...',lambda cancel,progress:export_dataset(folder,d),
                    lambda path:self.status_label.setText('Saved '+path))

    def apply_overlay(self):
        if not self.dataset or not self.dataset['result']:return
        d=overlay_payload(self.dataset);server=self.dataset.get('reply_server',self.reply_server)
        def work(cancel,progress):
            path=session_directory()/'overlays'/(uuid4().hex+'.json')
            atomic_json(path,{k:v for k,v in d.items() if k!='cube'})
            if not send_message(server,{'overlay':str(path)}):
                raise ValueError('Main app is unavailable. Reopen this dataset from the app to reconnect.')
            return path
        self._start('Sending overlay...',work,lambda path:self.status_label.setText('Overlay sent to the app.'))

    def save_as(self):
        if self.busy:return
        path,_=QFileDialog.getSaveFileName(self,'Save Peak Analysis workspace copy',str(self.session_path),'Peak workspace (*.npz)')
        if not path:return
        datasets=[snapshot(d) for d in self.datasets.values()];active=self.active_key
        self._start('Saving workspace...',lambda cancel,progress:save_workspace(path,datasets,active),
                    lambda _:self.saved_as(path))

    def saved_as(self,path):
        self.status_label.setText('Workspace copy saved: '+str(path))

    def request_restart(self):
        self.restart_requested=True;self.close()

    def closeEvent(self,event):
        self.cursor_readout.clear_reading()
        if self._close_ready or (not self.datasets and not self.busy):
            event.accept();return
        event.ignore();self._closing=True
        if self.busy:
            self.cancel_event.set();return
        datasets=[snapshot(d) for d in self.datasets.values()];active=self.active_key
        def work(cancel,progress):
            try:save_workspace(self.session_path,datasets,active)
            except Exception:
                self._closing=False
                raise
        def saved(_):self._close_ready=True
        self._start('Saving workspace before closing...',work,saved)
