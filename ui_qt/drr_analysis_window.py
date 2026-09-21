"""Independent multi-dataset DRR analysis workspace."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from time import monotonic

import numpy as np
from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QAbstractItemView, QSplitter,
    QScrollArea, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFileDialog, QDialog, QDialogButtonBox, QFormLayout, QToolButton, QMessageBox, QSizePolicy, QProgressBar, QLineEdit, QTableWidgetItem, QHeaderView, QTabWidget, QStackedWidget)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg as FigureCanvasQTAgg
from matplotlib.patches import Rectangle
from ui_qt.compact_spinbox import CompactDoubleSpinBox as QDoubleSpinBox

from core.drr_analysis_workspace import (create_dataset, load_dataset,
    apply_settings, apply_common_settings, export_dataset, export_summary)
from core.drr_peak_analysis import analyze_drr_peaks
from core.processing import apply_sg_derivative_energy
from ui_qt.common import Worker
from ui_qt.drr_peak_analysis import DrrPeakAnalysisController


class WorkspacePeakController(DrrPeakAnalysisController):
    def display_result(self):
        return self.owner.filtered_dataset_result(self.owner.active_key,self.valid_result())

    def populate_table(self):
        self.all_points=[(source,point) for source,product in (self.display_result() or {}).get('products',{}).items()
                     for point in product['points']]
        self.page=min(getattr(self,'page',0),max(0,(len(self.all_points)-1)//500))
        self.points=self.all_points[self.page*500:(self.page+1)*500]
        header=self.table.horizontalHeader()
        modes=[header.sectionResizeMode(i) for i in range(self.table.columnCount())]
        updates=self.table.updatesEnabled();blocked=self.table.blockSignals(True)
        self.table.setUpdatesEnabled(False)
        # Disabling painting does not stop ResizeToContents from scanning every
        # cell on each setItem. Suspend sizing for the entire nine-column batch.
        header.setSectionResizeMode(QHeaderView.Fixed)
        try:
            super().populate_table()
            for row,(_,point) in enumerate(self.points):
                for column,name in ((6,'prominence_fraction'),(7,'width_mev')):
                    value=point.get('auto_'+name,point.get(name))
                    self.table.setItem(row,column,QTableWidgetItem('—' if value is None else f'{value:.4g}'))
                self.table.setItem(row,8,QTableWidgetItem(point.get('confidence','—')))
        finally:
            for column,mode in enumerate(modes):header.setSectionResizeMode(column,mode)
            self.table.blockSignals(blocked)
            self.table.setUpdatesEnabled(updates)
        if hasattr(self.owner,'result_page_label'):
            self.owner.result_page_label.setText(f'Page {self.page+1}/{max(1,(len(self.all_points)+499)//500)} · {len(self.all_points)} results')

    def change_page(self,delta):
        self.page=max(0,getattr(self,'page',0)+delta);self.populate_table()

    def pick(self,event):
        points=getattr(event.artist,'_drr_batch_points',None)
        if points and len(event.ind):
            point=points[int(event.ind[0])];source=event.artist._drr_batch_source
            for index,(src,p) in enumerate(getattr(self,'all_points',[])):
                if src==source and p is point:
                    self.page=index//500;self.populate_table();break
        super().pick(event)

    def update_count(self):
        super().update_count()
        self.analyze_button.setEnabled(bool(self.owner.active_key) and not self.owner.jobs)

    def changed(self, *_):
        if not self.owner.ready: return
        self.generation += 1
        self.update_count()
        self.owner.settings_changed()

    def analyze(self):
        self.owner.find_peaks_keys([self.owner.active_key] if self.owner.active_key else [])

    def clear(self):
        super().clear()
        if self.owner.ready and self.owner.active_key:
            item=self.owner.datasets[self.owner.active_key]
            item.result=None;item.revision+=1
            self.owner.refresh_labels()
            self.owner.update_result_filter_count()

    def exclude_selected(self):
        if self.owner.jobs:self.owner._status('Wait for the current task to finish.');return
        row=self.table.currentRow()
        if self.valid_result() is None or not 0<=row<len(self.points):return
        source,selected=self.points[row]
        point=next((p for p in self.result['products'][source]['points'] if
                    (p['row_index'],p['energy'],p['polarity'])==(selected['row_index'],selected['energy'],selected['polarity'])),None)
        if point is None:return
        self.result.setdefault('excluded_points',[]).append({'source':source,**deepcopy(point)})
        self.result['products'][source]['points'].remove(point)
        self.owner.filtered_cache.clear()
        if self.owner.active_key and self.valid_result() is not None:
            item=self.owner.datasets[self.owner.active_key]
            item.result=deepcopy(self.result);item.revision+=1
            item.result['dataset_revision']=item.revision
            self.result['dataset_revision']=item.revision
            self.owner.refresh_labels()
            self.populate_table();self.redraw();self.owner.update_result_filter_count()


from ui_qt.drr_comparison_workspace import ComparisonWorkspaceMixin


class DrrAnalysisWindow(ComparisonWorkspaceMixin, QMainWindow):
    def __init__(self, parent=None, *, current_provider=None, start_folder=None, session_path=None):
        super().__init__(parent, Qt.Window)
        self.session_path=Path(session_path) if session_path else None
        self._comparison_storage_base=self.session_path.parent if self.session_path else None
        self.persistence_required=session_path is not None
        self.setWindowTitle('DRR Analysis');self.resize(1500,940)
        self.current_provider=current_provider;self.start_folder=str(start_folder or Path.cwd())
        self.comparison_key=None;self.comparison_catalog={};self.comparison_entries={}
        self.datasets={};self.active_key=None;self.views={};self.derivative_cache={};self.preview_sg={}
        self.ready=False;self._is_closing=False;self._load_in_progress=False
        self._processed_refresh_pending=False
        self.plot_range=None
        self.color_maps=[];self.locked_color_limits={}
        self.color_timer=QTimer(self);self.color_timer.setSingleShot(True)
        self.color_timer.timeout.connect(self.update_visible_color_scales)
        self.result_filters={};self.result_selectors=[]
        self.filtered_cache={}
        self.filter_timer=QTimer(self);self.filter_timer.setSingleShot(True);self.filter_timer.timeout.connect(self.apply_result_filter_async)
        self.loaded=None;self.drr_selected_files=[];self.drr_baseline_files_manual=[]
        self._drr_heatmap_axes={};self._drr_spectrum_axes={};self._drr_heatmap_ax=None
        self._last_plot_cube=None;self._gate=0.;self.drr_controller=self
        self.jobs=[];self.cancel=Event();self.thread_pool=QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self.drr_sg_window_spin=QSpinBox();self.drr_sg_window_spin.setRange(5,1001);self.drr_sg_window_spin.setSingleStep(2);self.drr_sg_window_spin.setValue(31)
        self.drr_sg_poly_spin=QSpinBox();self.drr_sg_poly_spin.setRange(2,10);self.drr_sg_poly_spin.setValue(2)
        self.drr_baseline_combo=QComboBox();self.drr_baseline_combo.addItem('Snapshot')
        self.drr_baseline_combine_combo=QComboBox();self.drr_baseline_combine_combo.addItem('all')
        self.figure=Figure();self.canvas=FigureCanvasQTAgg(self.figure)
        from ui_qt.matplotlib_theme import bind_theme_canvas
        self._canvas_theme=bind_theme_canvas(self.canvas)
        self.analysis=WorkspacePeakController(self)
        controls=self.analysis.build_controls()
        for name in ('x_min','x_max'):
            spin=self.analysis.bounds[name]
            spin.display_precision=4;spin.keep_trailing_zeros=True;spin.setSingleStep(.0001)
        self.analysis.table.setColumnCount(9)
        self.analysis.table.setHorizontalHeaderLabels(['Source','Y','Track','Type','Energy (eV)','Status','Detection prom.','Detection width (meV)','Auto support'])
        pager=QHBoxLayout();previous=QPushButton('Previous');following=QPushButton('Next');self.result_page_label=QLabel()
        previous.clicked.connect(lambda:self.analysis.change_page(-1));following.clicked.connect(lambda:self.analysis.change_page(1))
        pager.addWidget(previous);pager.addWidget(self.result_page_label,1);pager.addWidget(following)
        self.analysis.results_widget.layout().insertLayout(1,pager)
        controls.layout().setAlignment(Qt.AlignTop)
        for field in controls.findChildren(QWidget):
            if not isinstance(field,(QDoubleSpinBox,QSpinBox,QComboBox)):continue
            field.setMinimumWidth(0)
            field.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.canvas.mpl_connect('pick_event',self.analysis.pick)
        self.canvas.mpl_connect('button_press_event',self.map_clicked)
        self.timer=QTimer(self);self.timer.setSingleShot(True);self.timer.timeout.connect(self.redraw_plot)
        self.overlay_timer=QTimer(self);self.overlay_timer.setSingleShot(True);self.overlay_timer.timeout.connect(self.analysis.redraw)
        self.gate_timer=QTimer(self);self.gate_timer.setSingleShot(True);self.gate_timer.timeout.connect(self.refresh_linecuts)
        self.linecuts=[]
        root=QWidget();self.setCentralWidget(root);layout=QVBoxLayout(root)
        commands=QHBoxLayout();layout.addLayout(commands)
        for label,slot in [('Add files…',self.add_files),('Add current DRR',self.add_current),
                ('Update from current DRR',self.update_current),('Background / reload selected…',self.reload_selected),
                ('Remove selected',self.remove_selected),('Save workspace…',lambda:self.save_workspace())]:
            button=QPushButton(label);button.clicked.connect(slot);commands.addWidget(button)
            if 'current DRR' in label:button.setEnabled(current_provider is not None)
        self.notice=QLabel('Add measurement files or a processed snapshot from DRR.');self.notice.setWordWrap(True);layout.addWidget(self.notice)
        jobrow=QHBoxLayout();self.progress_bar=QProgressBar();self.progress_bar.setRange(0,100);self.progress_bar.setValue(0)
        self.job_status=QLabel('Idle');self.cancel_button=QPushButton('Cancel');self.cancel_button.setEnabled(False);self.cancel_button.clicked.connect(self.cancel_job)
        jobrow.addWidget(self.progress_bar,1);jobrow.addWidget(self.job_status);jobrow.addWidget(self.cancel_button);layout.addLayout(jobrow)
        self.elapsed_timer=QTimer(self);self.elapsed_timer.setInterval(250);self.elapsed_timer.timeout.connect(self.update_elapsed)
        self.job_started=0.;self.job_failed=False
        split=QSplitter();layout.addWidget(split,1)
        left=QWidget();ll=QVBoxLayout(left);ll.setContentsMargins(0,0,0,0)
        ll.addWidget(QLabel('Magnetic-field comparison groups'))
        self.comparison_search=QLineEdit();self.comparison_search.setPlaceholderText('Sample / spot / gate condition…');ll.addWidget(self.comparison_search)
        self.comparison_search.textChanged.connect(self.filter_comparisons)
        self.comparison_groups=QListWidget();self.comparison_groups.setMaximumHeight(200);ll.addWidget(self.comparison_groups)
        self.comparison_groups.itemDoubleClicked.connect(self.open_selected_comparison)
        self.comparison_groups.currentItemChanged.connect(lambda item,*_:self.comparison_title.setText('Selected group:\n'+item.text() if item else 'Choose a group above, then Open group.'))
        self.comparison_title=QLabel('Double-click a group to load all its members and restore analysis.');self.comparison_title.setWordWrap(True);ll.addWidget(self.comparison_title)
        group_actions=QHBoxLayout()
        self.open_group_button=QPushButton('Open group');self.open_group_button.clicked.connect(self.open_selected_comparison);group_actions.addWidget(self.open_group_button)
        self.manage_group_button=QPushButton('Manage members…');self.manage_group_button.clicked.connect(self.preview_comparison);group_actions.addWidget(self.manage_group_button)
        ll.addLayout(group_actions)
        history=QPushButton('Restore saved P2P range…');history.clicked.connect(self.restore_comparison_history);ll.addWidget(history)
        browse=QPushButton('Add individual processed files…');browse.clicked.connect(self.choose_processed_groups);ll.addWidget(browse)
        self.processed_filter=QLineEdit();self.processed_filter.setPlaceholderText('Filter saved groups…');ll.addWidget(self.processed_filter)
        self.processed_filter.textChanged.connect(self.filter_processed_groups);self.processed_filter.hide()
        self.processed_groups=QListWidget();self.processed_groups.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.processed_groups.setMaximumHeight(180);ll.addWidget(self.processed_groups);self.processed_groups.hide()
        refresh=QPushButton('Refresh processed groups');refresh.clicked.connect(self.refresh_processed_groups);ll.addWidget(refresh)
        reuse=QPushButton('Use selected groups');reuse.clicked.connect(self.use_processed_groups);ll.addWidget(reuse);reuse.hide()
        self.processed_groups.itemDoubleClicked.connect(lambda *_:self.use_processed_groups())
        ll.addWidget(QLabel('Loaded datasets — click a file to inspect'))
        self.list=QListWidget();self.list.setSelectionMode(QAbstractItemView.ExtendedSelection);ll.addWidget(self.list)
        self.peak_batch_buttons=[]
        for label,slot in [('Apply parameters to selected',self.apply_selected),('Analyze selected',self.analyze_selected),
                ('Export selected…',self.export_selected),('Export summary workbook…',self.export_combined)]:
            button=QPushButton(label);button.clicked.connect(slot);ll.addWidget(button)
            self.peak_batch_buttons.append(button)
            if label=='Analyze selected':self.batch_analyze_button=button
        split.addWidget(left)
        self.analysis_tabs=QTabWidget();split.addWidget(self.analysis_tabs)
        center=QSplitter(Qt.Vertical);self.analysis_tabs.addTab(center,'Peak positions')
        self.amplitude_page=None
        self.amplitude_placeholder=QWidget();placeholder_layout=QVBoxLayout(self.amplitude_placeholder)
        placeholder_layout.addWidget(QLabel('Select a processed dataset on the left to start range analysis.'))
        self.analysis_tabs.addTab(self.amplitude_placeholder,'Peak-to-peak')
        plot=QWidget();pl=QVBoxLayout(plot);pl.setContentsMargins(0,0,0,0)
        viewrow=QHBoxLayout();self.display=QComboBox()
        self.display.addItems(['DRR + second derivative','DRR','Second derivative','Compare selected: DRR','Compare selected: second derivative'])
        self.display.currentIndexChanged.connect(lambda *_:self.timer.start(80))
        viewrow.addWidget(self.display)
        self.follow=QCheckBox('Link view to detection (invalidates results)');self.follow.setChecked(False)
        pl.addLayout(viewrow);pl.addWidget(NavigationToolbar2QT(self.canvas,self));pl.addWidget(self.canvas,1)
        center.addWidget(plot);center.addWidget(self.analysis.results_widget);center.setSizes([670,190])
        right=QWidget();rl=QVBoxLayout(right);rl.setContentsMargins(6,4,6,4);rl.setSpacing(6)
        first=QWidget();first_layout=QVBoxLayout(first);first_layout.setContentsMargins(2,2,2,2)
        self.detection_scope=QComboBox();self.detection_scope.addItems(['Full data (default)','Current display range','Custom calculation range'])
        self.detection_scope.setToolTip('This determines where candidates are calculated. Result filters are applied afterward.')
        first_layout.addWidget(self.detection_scope)
        first_layout.addWidget(self.analysis.analyze_button)
        rl.addWidget(self._make_expander('1 · Find peaks',first,expanded=True))
        calculation=QWidget();calculation_form=QFormLayout(calculation)
        ranges=QWidget();range_layout=QFormLayout(ranges);self.display_bounds={}
        for axis,label in [('x','Energy (eV)'),('y','Y range')]:
            row=QWidget();row_layout=QHBoxLayout(row);row_layout.setContentsMargins(0,0,0,0)
            for suffix in ('min','max'):
                spin=QDoubleSpinBox();spin.setDecimals(9);spin.setRange(-1e9,1e9);spin.setKeyboardTracking(False)
                spin.setAccessibleName(f'Display {axis} {suffix}');self.display_bounds[f'{axis}_{suffix}']=spin
                spin.display_precision=4 if axis=='x' else 3
                spin.keep_trailing_zeros=axis=='x'
                spin.setSingleStep(.0001 if axis=='x' else .1)
                if suffix=='max':row_layout.addWidget(QLabel('to'))
                row_layout.addWidget(spin);spin.valueChanged.connect(self.display_range_changed)
            range_layout.addRow(label,row)
        self.follow.hide()
        self.lock_color_scale=QCheckBox('Lock color scale')
        self.lock_color_scale.setToolTip('Keep current limits while changing ranges or datasets. DRR and derivative have separate limits.')
        self.lock_color_scale.toggled.connect(self.color_lock_changed)
        range_layout.addRow(self.lock_color_scale)
        copy_view=QPushButton('Use displayed range for detection');copy_view.clicked.connect(self.copy_view_to_detection)
        calculation_form.addRow(copy_view)
        for label,row in zip(('Peak Energy (eV)','Peak Y range'),self.analysis.range_rows):
            taken=self.analysis.range_form.takeRow(row)
            if taken.labelItem and taken.labelItem.widget():taken.labelItem.widget().deleteLater()
            calculation_form.addRow(label,row)
        self.analysis.range_form=calculation_form
        reset_calculation=QPushButton('Reset calculation to full data');reset_calculation.clicked.connect(self.analysis.full_range);calculation_form.addRow(reset_calculation)
        self.fixed_range=QCheckBox('Keep display range across datasets');range_layout.addRow(self.fixed_range)
        full=QPushButton('Full data range');full.clicked.connect(self.full_display_range);range_layout.addRow(full)
        display_section=self._make_expander('Display only',ranges,expanded=False)
        result_panel=QWidget();result_form=QFormLayout(result_panel)
        self.result_filter_enabled=QCheckBox('Limit results to Energy range');result_form.addRow(self.result_filter_enabled)
        self.result_filter_bounds={}
        for axis,label in [('x','Energy (eV)'),('y','Y range')]:
            row=QWidget();row_layout=QHBoxLayout(row);row_layout.setContentsMargins(0,0,0,0)
            for suffix in ('min','max'):
                spin=QDoubleSpinBox();spin.setDecimals(9);spin.setRange(-1e9,1e9);spin.setKeyboardTracking(False)
                self.result_filter_bounds[f'{axis}_{suffix}']=spin;row_layout.addWidget(spin)
                spin.display_precision=4 if axis=='x' else 3
                spin.keep_trailing_zeros=axis=='x'
                spin.setSingleStep(.0001 if axis=='x' else .1)
                spin.valueChanged.connect(self.result_filter_changed)
            result_form.addRow(label,row)
        self.result_filter_y=QCheckBox('Also limit Y');result_form.addRow(self.result_filter_y)
        self.result_drag=QPushButton('Drag Energy interval on map');self.result_drag.setCheckable(True)
        self.result_drag.toggled.connect(self.result_drag_changed);result_form.addRow(self.result_drag)
        result_form.addRow(self.analysis.show)
        self.result_metrics={}
        for name,label,value,maximum,step in [('min_prominence','Min prominence fraction',0,1,.01),
                ('min_width_mev','Min width (meV)',0,1000,.1),('max_width_mev','Max width (0 = off)',0,1000,.1),
                ('min_distance_mev','Min spacing (meV)',0,1000,.1),('max_peaks','Max / row (0 = all)',0,1000,1),
                ('max_shift_mev','Track step (meV)',3,1000,.1)]:
            spin=QDoubleSpinBox();spin.setDecimals(3 if name!='max_peaks' else 0);spin.setRange(0,maximum);spin.setSingleStep(step);spin.setValue(value)
            spin.setKeyboardTracking(False);spin.valueChanged.connect(self.result_filter_changed)
            self.result_metrics[name]=spin;result_form.addRow(label,spin)
        self.result_link_tracks=QCheckBox('Connect branches (optional)');self.result_link_tracks.toggled.connect(self.result_filter_changed);result_form.addRow(self.result_link_tracks)
        self.result_low_confidence=QCheckBox('Show low-confidence candidates')
        self.result_low_confidence.setToolTip('Auto quality is heuristic evidence, not proof that a peak is real.')
        self.result_low_confidence.toggled.connect(self.result_filter_changed);result_form.addRow(self.result_low_confidence)
        hint=QLabel('Defaults retain all candidates. Thresholds are not validated for this dataset.');hint.setWordWrap(True);result_form.addRow(hint)
        reset=QPushButton('Restore all candidates');reset.clicked.connect(self.reset_result_filter);result_form.addRow(reset)
        self.result_filter_count=QLabel('No candidates');self.result_filter_count.setWordWrap(True);result_form.addRow(self.result_filter_count)
        self.result_filter_enabled.toggled.connect(self.result_filter_changed);self.result_filter_y.toggled.connect(self.result_filter_changed)
        self.result_filter_section=self._make_expander('2 · Filter candidates',result_panel,expanded=False)
        rl.addWidget(self.result_filter_section)
        self.result_filter_section.setEnabled(False)
        rl.addWidget(display_section)
        self.follow.toggled.connect(self.range_link_changed)
        self.range_link_changed()
        preview=QWidget();sg=QFormLayout(preview);sg.addRow('SG window',self.drr_sg_window_spin);sg.addRow('SG polynomial',self.drr_sg_poly_spin)
        self.auto_detection=QCheckBox('Auto noise / scale assessment');self.auto_detection.setChecked(True);sg.addRow(self.auto_detection)
        self.preview_button=QPushButton('Update preview');self.preview_button.clicked.connect(self.update_preview);sg.addRow(self.preview_button)
        self.gate_spin=QDoubleSpinBox();self.gate_spin.setDecimals(6);self.gate_spin.setRange(-1e9,1e9)
        sg.addRow('Inspect Y',self.gate_spin);self.gate_spin.valueChanged.connect(self.gate_changed)
        self.recipe=QLabel();self.recipe.setWordWrap(True);sg.addRow(self.recipe)
        advanced=QWidget();advanced_layout=QVBoxLayout(advanced);advanced_layout.setContentsMargins(0,0,0,0)
        advanced_layout.addWidget(calculation);advanced_layout.addWidget(preview);advanced_layout.addWidget(controls)
        self.calculation_section=self._make_expander('Advanced · calculation / SG / seed',advanced,expanded=False)
        rl.addWidget(self.calculation_section);rl.addStretch(1)
        self.detection_scope.currentIndexChanged.connect(lambda i:self.calculation_section.findChild(QToolButton).setChecked(True) if i==2 else None)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(right)
        self.peak_sidebar=scroll
        self.sidebar_stack=QStackedWidget();self.sidebar_stack.addWidget(scroll)
        self.p2p_sidebar=QScrollArea();self.p2p_sidebar.setWidgetResizable(True)
        self.p2p_sidebar.setWidget(QLabel('Select a dataset to start P2P analysis.'))
        self.sidebar_stack.addWidget(self.p2p_sidebar)
        self.sidebar_stack.setMinimumWidth(310);split.addWidget(self.sidebar_stack);split.setSizes([250,870,330])
        for form in right.findChildren(QFormLayout):
            form.setContentsMargins(2,2,2,2);form.setVerticalSpacing(4);form.setHorizontalSpacing(6)
        for field in right.findChildren(QWidget):
            if isinstance(field,(QDoubleSpinBox,QSpinBox)):
                field.setMinimumWidth(65);field.setMaximumWidth(115)
                field.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)
            elif isinstance(field,QPushButton):field.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)
        def update_seed_visibility(*_):
            enabled=self.analysis.mode.currentData()=='seed'
            for field in (self.analysis.seed_energy,self.analysis.seed_y,self.analysis.seed_button):
                self.analysis.range_source_form.setRowVisible(field,enabled)
            for field in (self.analysis.prominence,self.analysis.distance,self.analysis.maximum,self.analysis.shift,
                          self.analysis.noise,self.analysis.width,self.analysis.gaps):
                self.analysis.detection_form.setRowVisible(field,enabled)
            self.analysis.analyze_button.setText('Track seed branch' if enabled else 'Auto Find peaks' if self.auto_detection.isChecked() else 'Find all extrema')
            self.analysis.detection_section.setVisible(enabled)
        self.analysis.mode.currentIndexChanged.connect(update_seed_visibility);update_seed_visibility()
        self.auto_detection.toggled.connect(update_seed_visibility)
        self.list.currentItemChanged.connect(self.selection_changed)
        self.list.itemSelectionChanged.connect(lambda:self.timer.start(80) if self.display.currentIndex()>=3 else None)
        self.analysis_tabs.currentChanged.connect(self.analysis_tab_changed)
        self.ready=True;self.redraw_plot()
        self.analysis.show.toggled.connect(lambda *_:self.timer.start(50))
        for field in self.findChildren(QWidget):
            if isinstance(field,(QDoubleSpinBox,QSpinBox)):field.setKeyboardTracking(False)
        if start_folder is not None:QTimer.singleShot(0,self.refresh_processed_groups)

    def refresh_processed_groups(self):
        if self.jobs:
            self._processed_refresh_pending=True
            self._status('Processed groups will refresh after the current task.');return
        self._processed_refresh_pending=False
        from core.drr_processed_groups import discover_groups
        folder=self.start_folder
        def received(result):
            paths,catalog=result
            if self._is_closing or folder!=self.start_folder:return
            self.processed_groups.clear()
            for path in paths:
                item=QListWidgetItem(path.name.removesuffix('.metadata.json'))
                item.setData(Qt.UserRole,str(path));item.setToolTip(str(path));self.processed_groups.addItem(item)
            self.filter_processed_groups()
            self.update_comparison_catalog(paths,catalog)
            self.notice.setText(f'{len(paths)} saved DRR groups available in {folder}. Select groups to reuse their processed data.' if paths else f'No saved DRR groups with raw DAT + metadata found in {folder}. Use Add current DRR from the main window for unsaved data.')
        def discover(**_):
            from core.drr_comparison_groups import build_catalog
            paths=discover_groups(folder)
            return paths,build_catalog(paths)
        self.start_job(discover,received)

    def filter_processed_groups(self,*_):
        query=self.processed_filter.text().strip().casefold()
        for index in range(self.processed_groups.count()):
            item=self.processed_groups.item(index)
            hidden=query not in item.text().casefold()
            item.setHidden(hidden)
            if hidden:item.setSelected(False)

    def use_processed_groups(self):
        self.load_processed_paths([item.data(Qt.UserRole) for item in self.processed_groups.selectedItems()])

    def choose_processed_groups(self):
        if self.jobs:self._status('Wait for the current task to finish.');return
        from ui_qt.drr_group_picker import DrrGroupPicker
        paths=[self.processed_groups.item(i).data(Qt.UserRole) for i in range(self.processed_groups.count())]
        dialog=DrrGroupPicker(paths,self,selected=[item.data(Qt.UserRole) for item in self.processed_groups.selectedItems()])
        if dialog.exec()==QDialog.Accepted:self.load_processed_paths(dialog.selected_paths())

    def load_processed_paths(self,paths):
        if self.jobs:self._status('Wait for the current task to finish.');return
        if not paths:return
        from core.drr_processed_groups import load_group
        def run(*,progress,log):
            output=[]
            for index,path in enumerate(paths):
                if self.cancel.is_set():break
                log.emit(f'Reusing saved group {index+1}/{len(paths)}')
                try:output.append((load_group(path),None))
                except Exception as exc:output.append((None,f'{Path(path).name}: {exc}'))
                progress.emit(int(100*(index+1)/len(paths)))
            return output
        def received(output):
            if self._is_closing:return
            errors=[]
            for dataset,error in output:
                if error:errors.append(error)
                else:
                    if self.comparison_key:
                        from core.drr_comparison_groups import product_id
                        path=dataset.provenance.get('metadata_path')
                        if path:self.comparison_store().set_member(self.comparison_key,product_id(path),True)
                    self.add_dataset(dataset)
            self.job_failed=bool(errors)
            self.notice.setText('\n'.join(errors) if errors else f'Reused {len(output)} saved groups. Background processing was not repeated.')
        self.start_job(run,received)

    def _make_expander(self,title,content,*,expanded=True):
        widget=QWidget();layout=QVBoxLayout(widget);layout.setContentsMargins(0,0,0,0);layout.setSpacing(3)
        toggle=QToolButton();toggle.setText(title);toggle.setCheckable(True);toggle.setChecked(expanded);toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        toggle.toggled.connect(lambda yes:(content.setVisible(yes),toggle.setArrowType(Qt.DownArrow if yes else Qt.RightArrow)))
        layout.addWidget(toggle);layout.addWidget(content);content.setVisible(expanded);return widget

    def _status(self,text):self.statusBar().showMessage(str(text),12000)

    def selected_keys(self):return [i.data(Qt.UserRole) for i in self.list.selectedItems()]

    def add_dataset(self,dataset):
        if dataset.key not in self.datasets:
            self.datasets[dataset.key]=dataset
            item=QListWidgetItem();item.setData(Qt.UserRole,dataset.key);self.list.addItem(item)
        self.refresh_labels();self.activate_dataset(dataset.key)
        if self.amplitude_page is not None:self.amplitude_page.workspace_changed()

    def refresh_labels(self):
        for i in range(self.list.count()):
            item=self.list.item(i);d=self.datasets[item.data(Qt.UserRole)]
            state='Analyzed' if d.result is not None else ('Parameters changed' if d.revision else 'Not analyzed')
            from core.drr_comparison_groups import product_id,parse_condition
            entry=self.comparison_entries.get(product_id(d.provenance.get('metadata_path','')),{})
            c=entry.get('condition') or parse_condition(Path(d.provenance.get('measurement_files',[''])[0]).name if d.provenance.get('measurement_files') else '')
            p2p=next((r for r in (self.amplitude_page.records if self.amplitude_page else []) if r['dataset'].key==d.key),None)
            if c:
                recipe=d.provenance.get('saved_processing',{})
                item.setText(f"{c['field']:g} T · {c['spot']} · {c['gate']} · avg {recipe.get('average_count','?')}\nY {np.nanmin(d.cube.gate):g} … {np.nanmax(d.cube.gate):g}\nPeaks: {state} · P2P: {'pending' if self.amplitude_page and self.amplitude_page.pending else 'saved' if p2p else '—'}")
            else:item.setText(f'{d.name}\n{state}')
            item.setToolTip(d.name+'\n'+str(d.provenance))

    def save_workspace(self,*,quiet=False):
        from core.drr_workspace_session import save_session
        path=self.session_path
        if not quiet:
            filename,_=QFileDialog.getSaveFileName(self,'Save DRR workspace',str(path or Path(self.start_folder)/'drr_workspace.npz'),'DRR workspace (*.npz)')
            if not filename:return False
            path=Path(filename)
        if path is None:
            if self.comparison_key:
                self.persist_comparison(self.comparison_store(),self.comparison_key,self.comparison_snapshot())
                return True
            return False
        try:
            self.save_view()
            display=self.workspace_display_state()
            save_session(path,list(self.datasets.values()),self.views,self.active_key,display=display)
            if self.comparison_key:
                self.persist_comparison(self.comparison_store(),self.comparison_key,(list(self.datasets.values()),self.views,self.active_key,display))
            self.session_path=path
            if not quiet:self._status(f'Workspace saved: {path}')
            return True
        except Exception as exc:
            if quiet:raise
            QMessageBox.warning(self,'Save workspace',str(exc))
            return False

    def restore_session(self,path):
        from core.drr_workspace_session import load_session
        datasets,views,key,display=load_session(path,with_display=True)
        self.comparison_key=display.get('comparison_key')
        self.p2p_plot_style=display.get('p2p_plot_style',{})
        if display.get('comparison_folder'):self.start_folder=display['comparison_folder']
        for dataset in datasets:self.add_dataset(dataset)
        self.views=views;self.active_key=None
        self.result_filters=display.get('result_filters',{})
        self.auto_detection.setChecked(bool(display.get('auto_detection',True)))
        self.follow.setChecked(False)
        self.fixed_range.setChecked(bool(display.get('fixed',False)))
        if display.get('range') is not None:self.set_display_range(display['range'])
        if datasets:self.activate_dataset(key if key in self.datasets else datasets[0].key)
        if datasets and display.get('p2p') is not None:self.open_range_amplitude(state=display['p2p'])
        self.analysis_tabs.setCurrentIndex(display.get('analysis_tab',0))
        self.session_path=Path(path)

    def save_view(self):
        if self.active_key and self._drr_heatmap_ax is not None:
            self.views[self.active_key]=(self._drr_heatmap_ax.get_xlim(),self._drr_heatmap_ax.get_ylim(),self._gate,self.follow.isChecked(),self.display.currentIndex())

    def selection_changed(self,current,previous):
        if self.ready and current:self.activate_dataset(current.data(Qt.UserRole))

    def activate_dataset(self,key):
        if key not in self.datasets:return
        if key==self.active_key:return
        self.save_view();self.ready=False;self.active_key=key;d=self.datasets[key]
        self.preview_sg.setdefault(key,(d.settings.sg_window,d.settings.sg_polyorder))
        self.loaded=SimpleNamespace(mode='DRR',cube=d.cube)
        self.drr_selected_files=list(d.provenance.get('measurement_files',[]));self.drr_baseline_files_manual=list(d.provenance.get('background_files',[]))
        self.restore_controls(d.settings)
        state=self.views.get(key);self._gate=state[2] if state else (d.settings.seed_y if d.settings.mode=='seed' else float(d.cube.gate[0]))
        self.display.setCurrentIndex(state[4] if state else 0)
        if self.plot_range is None or not self.fixed_range.isChecked():
            self.set_display_range((d.settings.x_min,d.settings.x_max,d.settings.y_min,d.settings.y_max) if self.follow.isChecked() or not state else (*state[0],*state[1]))
        if self.follow.isChecked() and self.plot_range is not None:
            self.analysis.set_bounds(self.plot_range[:2],self.plot_range[2:])
            apply_settings(d,self.analysis.settings())
        self.gate_spin.setValue(self._gate)
        self.analysis.loaded_cube=d.cube;self._last_plot_cube=d.cube
        self.restore_result_filter()
        self.recipe.setText(f"{d.cube.gate_label}\nBackground: {d.provenance.get('background_mode','snapshot')}\n{len(self.drr_baseline_files_manual)} reference files")
        self.ready=True
        if self.analysis_tabs.currentIndex()==0:self.hydrate_result()
        self.analysis.update_count()
        self.preview_button.setText('Update preview (SG changed)' if self.preview_sg[key]!=(d.settings.sg_window,d.settings.sg_polyorder) else 'Update preview')
        if self.list.currentItem() is None or self.list.currentItem().data(Qt.UserRole)!=key:
            old=self.list.blockSignals(True)
            for i in range(self.list.count()):
                if self.list.item(i).data(Qt.UserRole)==key:self.list.setCurrentItem(self.list.item(i));break
            self.list.blockSignals(old)
        self._drr_heatmap_ax=None
        self._peak_view_dirty=True
        if self.analysis_tabs.currentIndex()==0:self.redraw_plot()
        if self.analysis_tabs.currentIndex()==1 and self.amplitude_page is None:self.open_range_amplitude()
        elif self.amplitude_page is not None:
            for index,record in enumerate(self.amplitude_page.records):
                if record['dataset'].key==key:
                    self.amplitude_page.inspect_file.setCurrentIndex(index);break

    def restore_controls(self,s):
        c=self.analysis
        for key,spin in c.bounds.items():
            value=getattr(s,key)
            spin.setValue((np.floor(value*1e9) if key.endswith('min') else np.ceil(value*1e9))/1e9)
        for name in ('source','mode','polarity'):
            combo=getattr(c,name);combo.setCurrentIndex(combo.findData(getattr(s,name)))
        for control,name in [(c.prominence,'prominence'),(c.distance,'min_distance_mev'),(c.shift,'max_shift_mev'),(c.maximum,'max_peaks'),(c.seed_energy,'seed_energy'),(c.seed_y,'seed_y'),(c.noise,'noise_sigma'),(c.width,'min_width_mev'),(c.gaps,'max_gap'),(self.drr_sg_window_spin,'sg_window'),(self.drr_sg_poly_spin,'sg_polyorder')]:control.setValue(getattr(s,name))

    def set_display_range(self,values):
        self.plot_range=tuple(float(v) for v in values)
        for spin,value in zip(self.display_bounds.values(),self.plot_range):
            old=spin.blockSignals(True);spin.setValue(value);spin.blockSignals(old)

    def range_link_changed(self,*_):
        linked=self.follow.isChecked()
        for row in self.analysis.range_rows:self.analysis.range_form.setRowVisible(row,not linked)
        for button in self.analysis.range_buttons:button.setVisible(False)
        if self.ready:self.display_range_changed()

    def display_range_changed(self,*_):
        if not self.ready or not self.active_key:return
        values=tuple(spin.value() for spin in self.display_bounds.values())
        if values[0]>=values[1] or values[2]>=values[3]:
            self._status('Display minimum must be less than maximum; previous range retained.');return
        self.plot_range=values
        if self.follow.isChecked():self.analysis.set_bounds(values[:2],values[2:])
        self.timer.start(150)

    def copy_view_to_detection(self):
        if not self.active_key or self.plot_range is None:return
        self.detection_scope.setCurrentIndex(2)
        self.analysis.set_bounds(self.plot_range[:2],self.plot_range[2:])
        self._status('Detection range updated. Existing candidates are stale; click Detect candidates when ready.')

    def open_range_amplitude(self,*,state=None):
        if not self.active_key:
            self._status('Select a processed DRR dataset first.');return
        if self.amplitude_page is not None:
            if state is not None:self.amplitude_page.restore_state(state)
            self.analysis_tabs.setCurrentWidget(self.amplitude_page);return
        from ui_qt.drr_p2p_batch import BatchRangeAmplitudePage
        if state is None:state=getattr(self,'_pending_p2p_state',None)
        self._pending_p2p_state=None
        d=self.datasets[self.active_key];cube=d.cube
        bounds=self.plot_range or (cube.energy.min(),cube.energy.max(),cube.gate.min(),cube.gate.max())
        self.amplitude_page=BatchRangeAmplitudePage(self,d,bounds,state=state)
        self.p2p_sidebar.setWidget(self.amplitude_page.sidebar)
        blocked=self.analysis_tabs.blockSignals(True)
        self.analysis_tabs.removeTab(self.analysis_tabs.indexOf(self.amplitude_placeholder))
        self.analysis_tabs.addTab(self.amplitude_page,'Peak-to-peak')
        self.analysis_tabs.setCurrentWidget(self.amplitude_page)
        self.analysis_tabs.blockSignals(blocked)
        if state is not None:self.amplitude_page.workspace_changed()
        self.analysis_tab_changed(1)

    def analysis_tab_changed(self,index):
        peaks=index==0
        self.sidebar_stack.setCurrentIndex(0 if peaks else 1)
        for button in self.peak_batch_buttons:button.setVisible(peaks)
        if peaks and self.ready and getattr(self,'_peak_view_dirty',False):
            self.hydrate_result();self.redraw_plot()
        if not peaks and self.amplitude_page is None and self.active_key:self.open_range_amplitude()

    def full_display_range(self):
        if not self.active_key:return
        cube=self.datasets[self.active_key].cube
        self.set_display_range((cube.energy.min(),cube.energy.max(),cube.gate.min(),cube.gate.max()))
        self.display_range_changed()

    def hydrate_result(self):
        self.filtered_cache.pop(self.active_key,None)
        self.restore_result_filter()
        c=self.analysis;d=self.datasets[self.active_key]
        c.result=d.result;c.result_key=c.key() if d.result is not None else None
        c.points=[(src,p) for src,product in (c.result or {}).get('products',{}).items() for p in product['points']]
        c.populate_table();c.summary.setText(f'{len(c.all_points)} positions. Table is paginated; export includes all retained points.' if c.result else 'No current result. Set detection range and parameters, then detect candidates.')
        if len(c.all_points)>20000 and getattr(self,'_last_overlay_result',None) is not c.result:
            old=c.show.blockSignals(True);c.show.setChecked(False);c.show.blockSignals(old)
            c.summary.setText(c.summary.text()+' Large-pool overlay is off; filter first or enable Show peak positions.')
        self._last_overlay_result=c.result
        self.update_result_filter_count()

    def restore_result_filter(self):
        bounds=self.result_filters.get(self.active_key)
        s=self.datasets[self.active_key].settings
        for key,spin in self.result_filter_bounds.items():
            old=spin.blockSignals(True);spin.setValue(bounds[key] if bounds else getattr(s,key));spin.blockSignals(old)
        for name,spin in self.result_metrics.items():
            default={'max_shift_mev':3}.get(name,0)
            old=spin.blockSignals(True);spin.setValue(bounds.get(name,default) if bounds else default);spin.blockSignals(old)
        for control,value in [(self.result_filter_enabled,bool(bounds and bounds.get('limit_x',True))),(self.result_filter_y,bool(bounds and bounds.get('limit_y'))),(self.result_link_tracks,bool(bounds and bounds.get('link_tracks',False)))]:
            old=control.blockSignals(True);control.setChecked(value);control.blockSignals(old)
        old=self.result_low_confidence.blockSignals(True);self.result_low_confidence.setChecked(bool(bounds and bounds.get('show_low_confidence')));self.result_low_confidence.blockSignals(old)
        self.result_drag.setChecked(False)

    def result_filter_changed(self,*_):
        if not self.ready or not self.active_key:return
        bounds={key:spin.value() for key,spin in self.result_filter_bounds.items()}
        bounds.update({key:spin.value() for key,spin in self.result_metrics.items()})
        bounds['limit_x']=self.result_filter_enabled.isChecked();bounds['limit_y']=self.result_filter_y.isChecked()
        bounds['link_tracks']=self.result_link_tracks.isChecked()
        bounds['show_low_confidence']=self.result_low_confidence.isChecked()
        if (bounds['limit_x'] and bounds['x_min']>=bounds['x_max']) or (bounds['limit_y'] and bounds['y_min']>bounds['y_max']) or (bounds['max_width_mev']>0 and bounds['min_width_mev']>bounds['max_width_mev']):
            self._status('Invalid result filter range; previous filter retained.');return
        self.result_filters[self.active_key]=bounds
        self.result_filter_count.setText('Filter pending…');self.filter_timer.start(180)

    def apply_result_filter_async(self):
        if self._is_closing or not self.active_key:return
        if self.jobs:self.filter_timer.start(180);return
        key=self.active_key;result=self.analysis.valid_result()
        if result is None:self.update_result_filter_count();return
        from core.drr_result_filter import filtered_result
        bounds=deepcopy(self.effective_result_filter(key,result));token=(id(result),repr(bounds))
        if self.filtered_cache.get(key,(None,))[0]==token:self.update_result_filter_count();return
        def received(filtered):
            if self._is_closing or key!=self.active_key or self.analysis.valid_result() is not result:return
            if bounds!=self.effective_result_filter(key,result):return
            self.filtered_cache[key]=(token,filtered);self.analysis.page=0
            self.analysis.populate_table();self.analysis.redraw();self.update_result_filter_count()
        self.start_job(lambda **_:filtered_result(result,bounds,for_display=True),received,cancellable=False)
        self.job_status.setText('Filtering candidates…')

    def reset_result_filter(self):
        for control in (self.result_filter_enabled,self.result_filter_y,self.result_link_tracks):
            old=control.blockSignals(True);control.setChecked(False);control.blockSignals(old)
        old=self.result_low_confidence.blockSignals(True);self.result_low_confidence.setChecked(True);self.result_low_confidence.blockSignals(old)
        for name,spin in self.result_metrics.items():
            old=spin.blockSignals(True);spin.setValue(3 if name=='max_shift_mev' else 0);spin.blockSignals(old)
        self.result_filter_changed()

    def effective_result_filter(self,key,result):
        s=self.datasets[key].settings
        bounds=dict(self.result_filters.get(key,dict(x_min=s.x_min,x_max=s.x_max,y_min=s.y_min,y_max=s.y_max,
            limit_x=False,limit_y=False,min_prominence=0,min_distance_mev=0,max_shift_mev=3,link_tracks=False)))
        bounds.setdefault('show_low_confidence',not result.get('auto_candidates',False))
        if not result.get('candidate_pool'):
            bounds={k:v for k,v in bounds.items() if k in ('x_min','x_max','y_min','y_max','limit_x','limit_y')}
        return bounds

    def filtered_dataset_result(self,key,result):
        if result is None or key not in self.datasets:return None
        from core.drr_result_filter import filtered_result
        bounds=self.effective_result_filter(key,result)
        if (not result.get('auto_candidates') or bounds.get('show_low_confidence',True)) and not bounds.get('limit_x',True) and not bounds.get('limit_y') and not bounds.get('link_tracks',False) and not any(bounds.get(k,0)>0 for k in ('min_prominence','min_width_mev','max_width_mev','min_distance_mev','max_peaks')):
            return result
        token=(id(result),repr(bounds))
        cached=self.filtered_cache.get(key)
        if cached is not None and cached[0][0]==id(result) and (self.filter_timer.isActive() or self.jobs):return cached[1]
        if cached is None or cached[0]!=token:
            cached=(token,filtered_result(result,bounds,for_display=True));self.filtered_cache[key]=cached
        return cached[1]

    def update_result_filter_count(self):
        result=self.analysis.valid_result()
        self.result_filter_section.setEnabled(result is not None)
        self.result_low_confidence.setEnabled(bool(result and result.get('auto_candidates')))
        total=sum(len(product['points']) for product in (result or {}).get('products',{}).values())
        self.result_filter_count.setText(f'{total} candidates → {len(getattr(self.analysis,"all_points",self.analysis.points))} retained'+
            (' · Redetect for metric filters' if result and not result.get('candidate_pool') else ''))
        if result:
            s=result['settings']
            self.result_filter_count.setText(self.result_filter_count.text()+f"\nCandidate pool: E {s['x_min']:.4g}–{s['x_max']:.4g}; Y {s['y_min']:.4g}–{s['y_max']:.4g}. Filters cannot add peaks outside this range.")
            if result.get('auto_candidates'):self.result_filter_count.setText(self.result_filter_count.text()+'\nAuto support is heuristic, not a physical validation.')
        for spin in self.result_metrics.values():spin.setEnabled(bool(result and result.get('candidate_pool')))

    def result_drag_changed(self,enabled):
        if enabled:self.analysis.seed_button.setChecked(False)
        for selector in self.result_selectors:selector.set_active(enabled)

    def result_interval_selected(self,lo,hi):
        if lo>=hi:return
        for key,value in [('x_min',lo),('x_max',hi)]:
            spin=self.result_filter_bounds[key];old=spin.blockSignals(True);spin.setValue(value);spin.blockSignals(old)
        old=self.result_filter_enabled.blockSignals(True);self.result_filter_enabled.setChecked(True);self.result_filter_enabled.blockSignals(old)
        self.result_drag.setChecked(False);self.result_filter_changed()

    def settings_changed(self):
        if not self.ready or not self.active_key:return
        d=self.datasets[self.active_key];old=d.settings;new=self.analysis.settings()
        if not apply_settings(d,new):return
        self.refresh_labels()
        self.analysis.summary.setText('Parameters changed — click Analyze to compute new results.')
        if (old.sg_window,old.sg_polyorder)!=(new.sg_window,new.sg_polyorder):
            self.preview_button.setText('Update preview (SG changed)')
        if (old.x_min,old.x_max,old.y_min,old.y_max)!=(new.x_min,new.x_max,new.y_min,new.y_max):
            if self.follow.isChecked():self.set_display_range((new.x_min,new.x_max,new.y_min,new.y_max))
            self.timer.start(300)
        self.overlay_timer.start(300)

    def update_preview(self):
        if not self.active_key:return
        d=self.datasets[self.active_key];self.preview_sg[d.key]=(d.settings.sg_window,d.settings.sg_polyorder)
        self.preview_button.setText('Update preview');self.timer.stop();self.redraw_plot()

    def zoom_range(self):
        if not self.active_key:return
        s=self.datasets[self.active_key].settings
        self.set_display_range((s.x_min,s.x_max,s.y_min,s.y_max));self.display_range_changed()

    def product(self,d,source):
        if source=='raw':return d.cube
        sg=self.preview_sg.setdefault(d.key,(d.settings.sg_window,d.settings.sg_polyorder))
        key=(d.key,*sg)
        if key not in self.derivative_cache:
            for old in list(self.derivative_cache):
                if old[0]==d.key:self.derivative_cache.pop(old)
            self.derivative_cache[key]=apply_sg_derivative_energy(d.cube,derivative=2,window_length=sg[0],polyorder=sg[1])[0]
        return self.derivative_cache[key]

    def redraw_plot(self):
        if not self.ready or self._is_closing:return
        if self.analysis_tabs.currentIndex()!=0:
            self._peak_view_dirty=True;return
        self._peak_view_dirty=False
        self.color_timer.stop();self.color_maps=[]
        for selector in self.result_selectors:selector.disconnect_events()
        self.result_selectors=[]
        old_view=(self._drr_heatmap_ax.get_xlim(),self._drr_heatmap_ax.get_ylim()) if self._drr_heatmap_ax is not None else None
        self.analysis.artists=[];self.figure.clear();self._drr_heatmap_axes={};self._drr_spectrum_axes={};self._drr_heatmap_ax=None
        self.linecuts=[]
        if not self.active_key:
            ax=self.figure.add_subplot();ax.text(.5,.5,'Add DRR datasets to begin',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();self.canvas.draw_idle();return
        d=self.datasets[self.active_key];mode=self.display.currentIndex()
        if mode>=3:
            keys=self.selected_keys()
            if len(keys)!=2:
                self._status('Select exactly two datasets to compare.');keys=[self.active_key]
            pairs=[(self.datasets[key],'raw' if mode==3 else 'second') for key in keys]
        else:pairs=[(d,src) for src in (('raw','second') if mode==0 else ('raw',) if mode==1 else ('second',))]
        grid=self.figure.subplots(2,len(pairs),squeeze=False,gridspec_kw={'height_ratios':[3,1]})
        for col,(item,source) in enumerate(pairs):
            ax,spec=grid[:,col]
            try:cube=self.product(item,source)
            except Exception as exc:ax.text(.1,.5,str(exc),transform=ax.transAxes,wrap=True);continue
            order=np.argsort(cube.gate);xo=np.argsort(cube.energy);s=item.settings
            mesh=ax.pcolormesh(cube.energy[xo],cube.gate[order],cube.Z[np.ix_(order,xo)],shading='nearest',cmap='RdBu_r',rasterized=True)
            mesh.set_clim(-1.,1.)
            self.figure.colorbar(mesh,ax=ax,pad=.02)
            used_sg=self.preview_sg.get(item.key,(s.sg_window,s.sg_polyorder))
            ax.set_title(f'{item.name[:42]}\n'+('DRR' if source=='raw' else f'd²/dE² · SG {used_sg[0]}/{used_sg[1]}'),fontsize=9)
            if s.x_min<s.x_max and s.y_min<=s.y_max:
                if self.follow.isChecked():
                    ax.set_xlim(s.x_min,s.x_max)
                    if s.y_min<s.y_max:ax.set_ylim(s.y_min,s.y_max)
                else:
                    saved=self.views.get(item.key)
                    view=old_view if item.key==self.active_key and old_view else (saved[:2] if saved else None)
                    if view:ax.set_xlim(*view[0]);ax.set_ylim(*view[1])
                    ax.add_patch(Rectangle((s.x_min,s.y_min),s.x_max-s.x_min,s.y_max-s.y_min,fill=False,edgecolor='green',lw=1))
            idx=int(np.argmin(abs(cube.gate-self._gate)));yy=float(cube.gate[idx]);gate_line=ax.axhline(yy,color='.3',ls='--',lw=.7)
            if self.plot_range is not None and (item.key==self.active_key or self.fixed_range.isChecked()):
                ax.set_xlim(*self.plot_range[:2])
                if self.plot_range[2]<self.plot_range[3]:ax.set_ylim(*self.plot_range[2:])
            self.color_maps.append((source,cube,ax,mesh))
            ax.callbacks.connect('xlim_changed',lambda _:self.color_timer.start(100))
            ax.callbacks.connect('ylim_changed',lambda _:self.color_timer.start(100))
            spectrum_line,=spec.plot(cube.energy,cube.Z[idx],lw=1)
            self.linecuts.append((cube,ax,spec,gate_line,spectrum_line))
            spec.set_xlim(ax.get_xlim());spec.set_title(f'Y = {yy:.6g}',fontsize=9);spec.set_xlabel('Energy (eV)');ax.set_ylabel(cube.gate_label)
            visible=(cube.energy>=min(ax.get_xlim()))&(cube.energy<=max(ax.get_xlim()))
            local=cube.Z[idx,visible];local=local[np.isfinite(local)]
            if local.size:
                lo,hi=float(local.min()),float(local.max());pad=max((hi-lo)*.08,abs(hi)*.01,1e-10);spec.set_ylim(lo-pad,hi+pad)
            if item.key==self.active_key:
                from matplotlib.widgets import SpanSelector
                selector=SpanSelector(ax,self.result_interval_selected,'horizontal',useblit=False,button=1,
                                      props={'alpha':.2,'facecolor':'tab:orange'})
                selector.set_active(self.result_drag.isChecked());self.result_selectors.append(selector)
                self._drr_heatmap_axes[source]=ax;self._drr_spectrum_axes[source]=spec;self._drr_heatmap_ax=ax
            elif item.result and self.analysis.show.isChecked():
                points=self.filtered_dataset_result(item.key,item.result)['products'].get(source,{}).get('points',[])
                for status,polarity,marker in [('accepted','peak','o'),('accepted','dip','v'),('uncertain','peak','x'),('uncertain','dip','x')]:
                    group=[p for p in points if ('uncertain' if p.get('confidence')=='low' else p['status'])==status and p['polarity']==polarity]
                    style={'color':'gray'} if marker=='x' else {'facecolors':'none','edgecolors':'black'}
                    ax.scatter([p['energy'] for p in group],[p['y'] for p in group],s=12,marker=marker,**style)
        self.update_visible_color_scales()
        self.figure.tight_layout();self.analysis.draw_overlays();self.canvas.draw_idle()

    def color_lock_changed(self,locked):
        self.locked_color_limits={}
        if locked:
            for source,cube,ax,mesh in self.color_maps:
                self.locked_color_limits.setdefault(source,mesh.get_clim())
        self.update_visible_color_scales()

    def update_visible_color_scales(self):
        if self._is_closing:return
        for source,cube,ax,mesh in self.color_maps:
            limits=self.locked_color_limits.get(source) if self.lock_color_scale.isChecked() else None
            if limits is None:
                xmin,xmax=sorted(ax.get_xlim());ymin,ymax=sorted(ax.get_ylim())
                ix=(cube.energy>=xmin)&(cube.energy<=xmax)
                iy=(cube.gate>=ymin)&(cube.gate<=ymax)
                values=cube.Z[np.ix_(iy,ix)];finite=values[np.isfinite(values)]
                if not finite.size:continue
                lim=max(float(np.quantile(abs(finite),.98)),1e-12);limits=(-lim,lim)
                if self.lock_color_scale.isChecked():self.locked_color_limits[source]=limits
            # Set both bounds atomically: colorbar callbacks can otherwise
            # expand an intermediate inverted interval while narrowing limits.
            with mesh.norm.callbacks.blocked():mesh.set_clim(*limits)
            mesh.changed()
        self.canvas.draw_idle()

    def _drr_gate_value(self):return self._gate
    def _set_drr_gate_spin_value(self,value):self.gate_spin.setValue(value)
    def _update_drr_spectrum_and_gate_line(self,cube):self.gate_timer.start(60)
    def refresh_linecuts(self):
        if not self.ready or self._is_closing:return
        for cube,ax,spec,gate_line,spectrum_line in self.linecuts:
            idx=int(np.argmin(abs(cube.gate-self._gate)));yy=float(cube.gate[idx])
            gate_line.set_ydata([yy,yy]);spectrum_line.set_ydata(cube.Z[idx])
            spec.set_xlim(ax.get_xlim());spec.set_title(f'Y = {yy:.6g}',fontsize=9)
            visible=(cube.energy>=min(ax.get_xlim()))&(cube.energy<=max(ax.get_xlim()))
            local=cube.Z[idx,visible];local=local[np.isfinite(local)]
            if local.size:
                lo,hi=float(local.min()),float(local.max());pad=max((hi-lo)*.08,abs(hi)*.01,1e-10);spec.set_ylim(lo-pad,hi+pad)
        self.analysis.redraw()
    def gate_changed(self,value):
        if self.loaded is not None:
            yy=self.loaded.cube.gate;value=float(yy[np.argmin(abs(yy-value))])
        self._gate=value
        old=self.gate_spin.blockSignals(True);self.gate_spin.setValue(value);self.gate_spin.blockSignals(old)
        if self.ready:self.gate_timer.start(60)
    def map_clicked(self,event):
        if self.result_drag.isChecked():return
        if getattr(event,'_drr_seed_consumed',False) or event.button!=1 or event.ydata is None:return
        if event.inaxes in self._drr_heatmap_axes.values():self.gate_spin.setValue(float(event.ydata))

    def apply_selected(self):
        if not self.active_key:return
        targets=[self.datasets[k] for k in self.selected_keys() if k!=self.active_key]
        apply_common_settings(self.datasets[self.active_key],targets);self.refresh_labels()
        if self.display.currentIndex()>=3:self.timer.start(300)
        self._status(f'Parameters applied to {len(targets)} datasets; their seed coordinates were preserved.')

    def analyze_selected(self):self.find_peaks_keys(self.selected_keys())

    def find_peaks_keys(self,keys):
        if self.jobs:self._status('Wait for the current task or cancel it first.');return
        for key in keys:
            dataset=self.datasets[key];scope=self.detection_scope.currentIndex()
            if scope==0:
                cube=dataset.cube
                values=(float(cube.energy.min()),float(cube.energy.max()),float(cube.gate.min()),float(cube.gate.max()))
            elif scope==1 and self.plot_range is not None:values=self.plot_range
            else:continue
            apply_settings(dataset,replace(dataset.settings,**dict(zip(('x_min','x_max','y_min','y_max'),values))))
        if self.active_key in keys:
            self.ready=False;self.restore_controls(self.datasets[self.active_key].settings);self.ready=True
            self.hydrate_result()
        self.refresh_labels();self.analyze_keys(keys)

    def analyze_keys(self,keys):
        if self.jobs:self._status('A task is already running. Wait or cancel it first.');return
        if not keys:return
        if self.fixed_range.isChecked() and self.follow.isChecked() and self.plot_range is not None:
            bounds=dict(zip(('x_min','x_max','y_min','y_max'),self.plot_range))
            for key in keys:
                dataset=self.datasets[key]
                apply_settings(dataset,replace(dataset.settings,**bounds))
            self.refresh_labels()
        snapshots=[(k,self.datasets[k].revision,self.datasets[k].cube,self.datasets[k].settings,deepcopy(self.datasets[k].provenance),self.datasets[k].name,self.datasets[k]) for k in keys]
        use_auto=self.auto_detection.isChecked()
        def run(*,progress,log):
            output=[]
            last_percent=-1
            def report(value,index):
                nonlocal last_percent
                percent=int((index*100+value)/len(snapshots))
                if percent!=last_percent:progress.emit(percent);last_percent=percent
            for index,(k,revision,cube,settings,provenance,name,instance) in enumerate(snapshots):
                if self.cancel.is_set():break
                log.emit(f'Analyzing {index+1}/{len(snapshots)}: {name}')
                try:
                    result=analyze_drr_peaks(cube,settings,cancelled=self.cancel.is_set,
                        progress=lambda value,i=index:report(value,i),retain_candidates=settings.mode=='all',auto_candidates=use_auto and settings.mode=='all')
                    result.update(provenance=provenance,dataset_id=k,dataset_name=name,dataset_revision=revision)
                    output.append((k,revision,result,None,instance))
                except Exception as exc:output.append((k,revision,None,str(exc),instance))
            return output
        self.notice.setText(f'Analyzing {len(keys)} datasets…');self.start_job(run,self.accept_results)

    def accept_results(self,output):
        if self._is_closing:return
        errors=[];accepted=0;stale=0
        for key,revision,result,error,instance in output:
            d=self.datasets.get(key)
            if d is not instance or d.revision!=revision:stale+=1;continue
            if error:errors.append(f'{d.name}: {error}');continue
            d.result=result;accepted+=1
            self.result_filters.pop(key,None);self.filtered_cache.pop(key,None)
            self.preview_sg[key]=(d.settings.sg_window,d.settings.sg_polyorder)
        self.refresh_labels()
        large_pool=False
        if self.active_key and self.datasets[self.active_key].result:
            current=self.datasets[self.active_key].result
            large_pool=sum(1 for product in current['products'].values() for point in product['points']
                           if not current.get('auto_candidates') or point.get('confidence')!='low')>20000
            if large_pool:self.analysis.show.setChecked(False)
        if self.active_key:self.hydrate_result();self.redraw_plot()
        if self.active_key:
            s=self.datasets[self.active_key].settings
            self.preview_button.setText('Update preview (SG changed)' if self.preview_sg[self.active_key]!=(s.sg_window,s.sg_polyorder) else 'Update preview')
        self.notice.setText(f'{accepted} analyses completed; {stale} outdated results discarded.'+ ('\n'+'\n'.join(errors) if errors else ''))
        if large_pool:self.notice.setText(self.notice.text()+' All candidates retained; overlay is off for this large pool. Filter first, then enable Show peak positions.')
        if accepted:self.result_filter_section.findChild(QToolButton).setChecked(True)
        self.job_failed=bool(errors)


    def start_job(self,fn,callback,*,cancellable=True):
        self.cancel.clear();self.job_failed=False;self.job_started=monotonic()
        self.progress_bar.setRange(0,0);self.job_status.setText('Running · 0.0 s');self.cancel_button.setEnabled(cancellable);self.elapsed_timer.start()
        self.analysis.analyze_button.setEnabled(False)
        self.batch_analyze_button.setEnabled(False)
        worker=Worker(fn);self.jobs.append(worker)
        worker.signals.result.connect(callback);worker.signals.error.connect(self.job_error)
        worker.signals.progress.connect(self.job_progress);worker.signals.log.connect(self.notice.setText)
        worker.signals.finished.connect(lambda:self.job_finished(worker))
        self.thread_pool.start(worker)

    def job_progress(self,value):
        self.progress_bar.setRange(0,100);self.progress_bar.setValue(min(100,max(0,value)))

    def update_elapsed(self):self.job_status.setText(f'{"Cancelling" if self.cancel.is_set() else "Running"} · {monotonic()-self.job_started:.1f} s')

    def cancel_job(self):self.cancel.set();self.cancel_button.setEnabled(False);self.update_elapsed()

    def job_error(self,message):self.job_failed=True;self.notice.setText(str(message))

    def job_finished(self,worker):
        if worker in self.jobs:self.jobs.remove(worker)
        self.elapsed_timer.stop();self.cancel_button.setEnabled(False)
        state='Cancelled' if self.cancel.is_set() else 'Finished with errors' if self.job_failed else 'Completed'
        self.job_status.setText(f'{state} · {monotonic()-self.job_started:.1f} s')
        self.progress_bar.setRange(0,100)
        if state=='Completed':self.progress_bar.setValue(100)
        self.analysis.update_count()
        self.batch_analyze_button.setEnabled(True)
        if self._processed_refresh_pending and not self._is_closing:
            QTimer.singleShot(0,self.refresh_processed_groups)

    def choose_recipe(self):
        dlg=QDialog(self);dlg.setWindowTitle('Background for selected measurements');layout=QFormLayout(dlg)
        mode=QComboBox();mode.addItem('External reference files','external');mode.addItem('Self: first frame','self_first');mode.addItem('Self: last frame','self_last')
        layout.addRow('Background',mode);files=[];label=QLabel('No external references selected');label.setWordWrap(True)
        def pick():
            selected,_=QFileDialog.getOpenFileNames(dlg,'Select background CSVs',self.start_folder,'CSV (*.csv)')
            if selected:files[:]=selected;label.setText('\n'.join(Path(f).name for f in files))
        choose=QPushButton('Select background files…');choose.clicked.connect(pick);layout.addRow(choose);layout.addRow(label)
        frames=QComboBox();frames.addItems(['all','first','last']);layout.addRow('Frames per reference',frames)
        info=QLabel('All: average frames within each file, then average files equally. This recipe applies to every selected measurement.');info.setWordWrap(True);layout.addRow(info)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);layout.addRow(buttons)
        def accept():
            if mode.currentData()=='external' and not files:label.setText('Select external reference files first.');return
            dlg.accept()
        buttons.accepted.connect(accept);buttons.rejected.connect(dlg.reject)
        return (files if mode.currentData()=='external' else [],mode.currentData(),frames.currentText()) if dlg.exec()==QDialog.Accepted else None

    def add_files(self):
        files,_=QFileDialog.getOpenFileNames(self,'Add measurements',self.start_folder,'CSV (*.csv)')
        if not files:return
        recipe=self.choose_recipe()
        if recipe:self.load_files(files,recipe)

    def load_files(self,files,recipe,replacements=None):
        if self.jobs:self._status('Wait for the current task to finish.');return
        def run(*,progress,log):
            output=[]
            for index,path in enumerate(files):
                if self.cancel.is_set():break
                log.emit(f'Loading {index+1}/{len(files)}: {Path(path).name}')
                try:output.append((path,load_dataset(path,recipe[0],background_mode=recipe[1],background_which=recipe[2]),None))
                except Exception as exc:output.append((path,None,str(exc)))
                progress.emit(int(100*(index+1)/len(files)))
            return output
        def received(output):
            if self._is_closing:return
            errors=[]
            for path,d,error in output:
                if error:errors.append(f'{Path(path).name}: {error}');continue
                if replacements and path in replacements:
                    key,revision,instance=replacements[path]
                    previous=self.datasets.get(key)
                    if previous is not instance or previous.revision!=revision:
                        errors.append(f'{Path(path).name}: reload discarded because the dataset changed.');continue
                    d.settings=previous.settings;d.revision=previous.revision+1
                    view=self.views.get(key)
                    self.remove_keys([key])
                    if view:self.views[d.key]=view
                self.add_dataset(d)
            self.notice.setText('\n'.join(errors) if errors else f'Loaded {len(output)} datasets.')
            self.job_failed=bool(errors)
        self.notice.setText('Loading measurements and background…');self.start_job(run,received)

    def reload_selected(self):
        mapping={}
        for key in self.selected_keys():
            files=self.datasets[key].provenance.get('measurement_files',[])
            if len(files)!=1:self._status('Reload supports individual measurement files. Re-add combined snapshots from DRR.');return
            if files[0] in mapping:
                self._status('Select one snapshot per measurement when reloading backgrounds.');return
            mapping[files[0]]=(key,self.datasets[key].revision,self.datasets[key])
        if not mapping:return
        recipe=self.choose_recipe()
        if recipe:self.load_files(list(mapping),recipe,mapping)

    def snapshot_current(self):
        loaded=self.current_provider() if self.current_provider else None
        if loaded is None or loaded.mode!='DRR' or loaded.cube is None:
            self._status('Load DRR data in the main window first.');return None
        provenance={'measurement_files':list(loaded.selected_files),'background_files':list(loaded.baseline_files),
            'background_mode':loaded.drr_baseline_text,'background_frame':loaded.drr_baseline_which,'y_axis':loaded.y_axis_spec,
            'assignments':[a.to_dict() for a in getattr(loaded,'drr_assignments',())]}
        return create_dataset(loaded.cube,Path(loaded.primary_file or (loaded.selected_files or ['Current DRR'])[0]).stem,provenance)

    def add_current(self):
        d=self.snapshot_current()
        if d is not None:self.add_dataset(d)

    def update_current(self):
        d=self.snapshot_current()
        if d is None:return
        key=self.active_key
        if key==d.key:self._status('Current DRR data already matches this snapshot.');return
        if key:self.remove_keys([key])
        self.add_dataset(d)

    def notify_current_changed(self):
        if self.datasets:self.notice.setText('Main DRR data changed. Existing analysis snapshots are retained; use Add current DRR or Update from current DRR explicitly.')

    def remove_selected(self):
        if self.jobs:self._status('Wait for the current task to finish.');return
        keys=self.selected_keys()
        if self.comparison_key and keys:
            store=self.comparison_store();snapshot=self.comparison_snapshot();group_key=self.comparison_key
            paths=[self.datasets[key].provenance.get('metadata_path') for key in keys]
            def persist(*,progress,log):
                self.persist_comparison(store,group_key,snapshot)
                from core.drr_comparison_groups import product_id
                for path in paths:
                    if path:store.set_member(group_key,product_id(path),False)
                return keys
            self.start_job(persist,self.remove_keys,cancellable=False)
            return
        self.remove_keys(keys)
    def remove_keys(self,keys):
        self.ready=False
        for key in keys:
            self.datasets.pop(key,None);self.views.pop(key,None);self.preview_sg.pop(key,None)
            for cachekey in list(self.derivative_cache):
                if cachekey[0]==key:self.derivative_cache.pop(cachekey)
            for i in range(self.list.count()-1,-1,-1):
                if self.list.item(i).data(Qt.UserRole)==key:self.list.takeItem(i)
        if self.active_key in keys:
            self.active_key=None;self.loaded=None;self.analysis.result=None;self.analysis.result_key=None;self.analysis.points=[];self.analysis.populate_table()
        self.ready=True
        if not self.active_key and self.datasets:self.activate_dataset(next(iter(self.datasets)))
        else:self.redraw_plot()
        if self.amplitude_page is not None:self.amplitude_page.workspace_changed()

    def export_selected(self):self.export(False)
    def export_combined(self):self.export(True)
    def export(self,combined):
        if self.jobs:self._status('Wait for the current task to finish.');return
        selected=[self.datasets[k] for k in self.selected_keys()]
        if not selected:return
        if any(d.result is None for d in selected):self.notice.setText('Analyze every selected dataset before export; stale results are not exported.');return
        folder=QFileDialog.getExistingDirectory(self,'Export analysis',self.start_folder)
        if not folder:return
        snapshots=deepcopy(selected)
        from core.drr_result_filter import filtered_result
        for dataset in snapshots:dataset.result=filtered_result(dataset.result,self.effective_result_filter(dataset.key,dataset.result))
        def run(*,progress,log):
            return export_summary(folder,snapshots) if combined else [export_dataset(folder,d) for d in snapshots]
        self.start_job(run,lambda paths:self.notice.setText(f'Exported: {paths}'),cancellable=False)

    def closeEvent(self,event):
        # Keep a main-window-owned workspace alive for reopening with its state.
        if self.parent() is not None and not getattr(self.parent(),'_is_closing',False):
            self.hide();event.ignore();return
        self._is_closing=True;self.cancel.set();self.timer.stop();self.overlay_timer.stop();self.gate_timer.stop();self.elapsed_timer.stop();self.thread_pool.waitForDone()
        self.filter_timer.stop()
        self.color_timer.stop()
        try:
            saved=self.save_workspace(quiet=True)
            if not saved and self.persistence_required and self.datasets:
                saved=self.save_workspace()
                if not saved:
                    self._is_closing=False;self.cancel.clear();event.ignore();return
        except Exception as exc:
            self._is_closing=False;self.cancel.clear();event.ignore()
            QMessageBox.warning(self,'Workspace not saved',f'{exc}\nThe window remains open so you can save your work.')
            return
        event.accept()
