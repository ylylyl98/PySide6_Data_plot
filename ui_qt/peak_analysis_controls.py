"""Compact English Qt controls shared by PL and DRR Peak Analysis."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QSizePolicy, QLabel,
    QPushButton, QListWidget, QGroupBox, QCheckBox, QToolButton, QTabWidget, QTabBar)
from ui_qt.common import QComboBox, QDoubleSpinBox, QSpinBox
from ui_qt.fluent_ui.style import set_fluent_property
from core.peak_workspace import recommended_detection_method
from core.peak_profiles import PRESET_NAMES, filter_presets, profile_name, sampling_step_mev


class PeakControls(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recommended_method='sg'
        self._custom_bounds={}
        self._presets={}
        self._sg_step_mev=None
        set_fluent_property(self,'fluentSize','compact')
        root = QVBoxLayout(self); root.setContentsMargins(4,4,4,4)
        self.pages=QTabWidget();root.addWidget(self.pages)
        detection=QWidget();self.pages.addTab(detection,'Find && filter')
        layout=QVBoxLayout(detection);layout.setSpacing(8)
        self.method=QTabBar();self.method.setExpanding(True)
        self.method.addTab('Local extrema');self.method.addTab('SG + extrema')
        self.method.setAccessibleName('Peak detection method');layout.addWidget(self.method)
        self.method_description=QLabel();self.method_description.setWordWrap(True);layout.addWidget(self.method_description)
        self.input_processing=QLabel();self.input_processing.setWordWrap(True);self.input_processing.hide()
        layout.addWidget(self.input_processing)
        self.detection_toggle=QToolButton();self.detection_toggle.setText('Detection options');self.detection_toggle.setCheckable(True)
        self.detection_toggle.setArrowType(Qt.RightArrow);self.detection_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.detection_options=QWidget();detection_form=QFormLayout(self.detection_options);detection_form.setContentsMargins(0,0,0,0)
        self.floor=QDoubleSpinBox();self.floor.setRange(0,1);self.floor.setDecimals(4);self.floor.setSingleStep(.005)
        self.floor.setKeyboardTracking(False)
        self.floor.setToolTip('Fraction of each full spectrum range. 0 caches all local extrema; no peak-count cap.')
        detection_form.addRow('Candidate floor',self.floor)
        self.smoothing=QSpinBox();self.smoothing.setRange(5,501);self.smoothing.setSingleStep(2);self.smoothing.setValue(11)
        self.smoothing.setKeyboardTracking(False);self.smoothing_label=QLabel('SG window (samples)')
        self.smoothing.setToolTip('Odd number of energy samples, not meV. Smaller windows preserve narrower or closely spaced features.')
        self.smoothing_options=QWidget();smoothing_form=QFormLayout(self.smoothing_options)
        smoothing_form.setContentsMargins(0,0,0,0);smoothing_form.addRow(self.smoothing_label,self.smoothing)
        layout.addWidget(self.smoothing_options);self.detection_options.hide()
        self.detection_toggle.toggled.connect(self.detection_options.setVisible)
        self.detection_toggle.toggled.connect(lambda checked:self.detection_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow))
        self.preview=QPushButton('Find all peaks');set_fluent_property(self.preview,'fluentAppearance','primary');layout.addWidget(self.preview)
        self.cache_status=QLabel('No candidates yet.');self.cache_status.setWordWrap(True);layout.addWidget(self.cache_status)
        self.filter_heading=QLabel('2. Filter cached candidates');layout.addWidget(self.filter_heading)
        presets=QHBoxLayout();presets.setSpacing(2);self.preset_buttons={}
        for name in PRESET_NAMES:
            button=QPushButton(name);button.setCheckable(True);button.setMinimumWidth(0)
            button.setAccessibleName(name+' peak filters')
            button.setToolTip('Apply '+name.lower()+' shape, SNR and neighbor filters to the existing cache.')
            button.clicked.connect(lambda checked=False, name=name:self.apply_preset(name))
            self.preset_buttons[name]=button;presets.addWidget(button)
        layout.addLayout(presets)
        self.preset_status=QLabel();layout.addWidget(self.preset_status)
        form = QFormLayout(); form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.scope = QTabBar();self.scope.setExpanding(True);self.scope.setAccessibleName('Filter range')
        for label,tip in [('Full','Filter the full dataset'),('View','Filter the current heatmap view'),
                          ('Custom','Filter the X and Y bounds below')]:
            i=self.scope.addTab(label);self.scope.setTabToolTip(i,tip)
        form.addRow('Range', self.scope)
        self.polarity = QTabBar();self.polarity.setExpanding(True);self.polarity.setAccessibleName('Peak or dip filter')
        for label in ('Both','Peaks','Dips'):self.polarity.addTab(label)
        self.polarity.setTabToolTip(0,'Peaks and dips')
        self.polarity_label = QLabel('Feature'); form.addRow(self.polarity_label, self.polarity)
        layout.addLayout(form)
        self.ranges_widget = QWidget(); ranges = QFormLayout(self.ranges_widget); ranges.setContentsMargins(0,0,0,0)
        self.bounds = {};self.range_labels={}
        for axis, label in [('x', 'X (eV)'), ('y', 'Y')]:
            row = QWidget(); line = QHBoxLayout(row); line.setContentsMargins(0,0,0,0)
            line.setSpacing(3)
            for suffix in ('min', 'max'):
                spin = QDoubleSpinBox(); spin.setRange(-1e12,1e12); spin.setDecimals(5); spin.setKeyboardTracking(False)
                spin.setMinimumWidth(0); spin.setAccessibleName(f'{axis.upper()} range {suffix}')
                # The broad scientific range must not reserve enough width
                # for its 13-digit maximum in every side-by-side field.
                spin.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
                if suffix=='max':line.addWidget(QLabel('to'))
                self.bounds[f'{axis}_{suffix}'] = spin; line.addWidget(spin,1)
                spin.valueChanged.connect(self.bounds_changed)
            self.range_labels[axis]=QLabel(label);ranges.addRow(self.range_labels[axis],row)
        self.ranges_widget.setToolTip('Inclusive X and Y filters. Editing a bound selects Custom; candidates stay cached.')
        layout.addWidget(self.ranges_widget)
        self.numeric={}
        filters=QGridLayout();filters.setHorizontalSpacing(6);filters.setVerticalSpacing(3)
        for name,label,minimum,maximum,default in [
            ('min_snr','Min SNR',0,1000,5),
            ('prominence','Min prominence',0,1,.05),
            ('min_width_mev','Min width (meV)',0,1000,.5),
            ('min_distance_mev','Min spacing (meV)',0,1000,2)]:
            spin=QDoubleSpinBox();spin.setRange(minimum,maximum);spin.setDecimals(4);spin.setValue(default);spin.setKeyboardTracking(False)
            spin.setMinimumWidth(0);spin.setAccessibleName(label)
            spin.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
            self.numeric[name]=spin
            if name!='min_distance_mev':
                i=('min_snr','prominence','min_width_mev').index(name)
                filters.addWidget(QLabel(label),(i//2)*2,i%2);filters.addWidget(spin,(i//2)*2+1,i%2)
            spin.valueChanged.connect(self.settings_changed)
        self.numeric['min_snr'].setDecimals(1);self.numeric['min_snr'].setSingleStep(.5);self.numeric['min_snr'].setSpecialValueText('Off')
        self.numeric['min_snr'].setToolTip('Prominence / estimated detection-signal noise (MAD after broad baseline removal). A heuristic ratio, not a statistical confidence. 0 disables this filter.')
        self.support=QSpinBox();self.support.setRange(0,5);self.support.setSpecialValueText('Off');self.support.setKeyboardTracking(False)
        self.support.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        self.support.setAccessibleName('Neighbor support out of 5 rows')
        self.support.setToolTip('Required same-polarity support among up to 5 neighboring spectra, including this row. Scaled at map edges; skipped for a single spectrum. Support uses the full map before range cropping.')
        filters.addWidget(QLabel('Support (of 5)'),2,1);filters.addWidget(self.support,3,1)
        self.support.valueChanged.connect(self.settings_changed)
        self.numeric['prominence'].setSingleStep(.01)
        self.numeric['prominence'].setToolTip('Fraction of the full spectrum range, measured before range filtering. 0.05 means 5%.')
        self.numeric['min_width_mev'].setToolTip('Full width at half prominence, measured on the detection signal.')
        layout.addLayout(filters)
        self.filter_toggle=QToolButton();self.filter_toggle.setText('More filters');self.filter_toggle.setCheckable(True)
        self.filter_toggle.setArrowType(Qt.RightArrow);self.filter_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon);layout.addWidget(self.filter_toggle)
        self.more_filters=QWidget();extra=QFormLayout(self.more_filters);extra.setContentsMargins(0,0,0,0)
        extra.setRowWrapPolicy(QFormLayout.WrapLongRows)
        extra.addRow('Min spacing (meV)',self.numeric['min_distance_mev'])
        self.neighbor_tolerance=QDoubleSpinBox();self.neighbor_tolerance.setRange(0,1000);self.neighbor_tolerance.setDecimals(3)
        self.neighbor_tolerance.setKeyboardTracking(False);self.neighbor_tolerance.setValue(1.5)
        self.neighbor_tolerance.setToolTip('Energy matching tolerance per neighboring scan row; twice this value for a spectrum two rows away. This checks support, not branch identity.')
        extra.addRow('Neighbor drift (meV/row)',self.neighbor_tolerance);self.neighbor_tolerance.valueChanged.connect(self.settings_changed)
        self.max_width=QDoubleSpinBox();self.max_width.setRange(0,1000);self.max_width.setDecimals(3);self.max_width.setSpecialValueText('No limit');self.max_width.setKeyboardTracking(False)
        self.max_count=QSpinBox();self.max_count.setRange(0,10000);self.max_count.setSpecialValueText('No limit');self.max_count.setKeyboardTracking(False)
        extra.addRow('Max width (meV)',self.max_width);extra.addRow('Max peaks per row',self.max_count)
        layout.addWidget(self.more_filters);self.more_filters.hide()
        layout.addWidget(self.detection_toggle);layout.addWidget(self.detection_options)
        self.filter_toggle.toggled.connect(self.more_filters.setVisible)
        self.filter_toggle.toggled.connect(lambda checked:self.filter_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow))
        self.max_width.valueChanged.connect(self.settings_changed);self.max_count.valueChanged.connect(self.settings_changed)
        self.show_rejected=QCheckBox('Show rejected candidates');layout.addWidget(self.show_rejected)
        self.reset_filters=QPushButton('Reset filters');layout.addWidget(self.reset_filters)
        hint=QLabel('Filters reuse cached measurements. Width is measured at half prominence, before fitting.')
        hint.setWordWrap(True);layout.addWidget(hint);layout.addStretch(1)
        branch_page=QWidget();self.pages.addTab(branch_page,'Branches && fit');layout=QVBoxLayout(branch_page)
        self.track = QPushButton('Connect selected branches')
        set_fluent_property(self.track, 'fluentAppearance', 'primary');layout.addWidget(self.track)
        hint=QLabel('Names and colors persist across filters. Ambiguous matches remain separate points.');hint.setWordWrap(True);layout.addWidget(hint)
        layout.addWidget(QLabel('Branches'))
        self.branches = QListWidget(); self.branches.setMinimumHeight(115); self.branches.setMaximumHeight(210)
        self.branches.setAccessibleName('Peak branches; check to include, double-click to rename')
        self.branches.setToolTip('Check branches to include. Double-click a name to rename it.')
        layout.addWidget(self.branches, 1)
        row = QHBoxLayout(); self.seed = QPushButton('Pick seed'); self.seed.setCheckable(True)
        self.seed.setToolTip('Click a peak or dip in the spectrum to add or move a seed.')
        self.remove = QPushButton('Exclude branch'); row.addWidget(self.seed); row.addWidget(self.remove); layout.addLayout(row)
        self.fit_group = QGroupBox('Fit (optional)'); fit_layout = QFormLayout(self.fit_group)
        self.model = QComboBox(); self.model.addItems(['Lorentzian', 'Gaussian']); fit_layout.addRow('Model', self.model)
        self.fit = QPushButton('Fit selected branches'); fit_layout.addRow(self.fit)
        self.residual = QCheckBox('Show residual'); fit_layout.addRow(self.residual); layout.addWidget(self.fit_group)
        self.position = QComboBox(); self.position.addItems(['Detected positions', 'Fitted centers'])
        self.position.setToolTip('Fitted centers include valid fits only.'); layout.addWidget(self.position)
        self.advanced_toggle = QToolButton(); self.advanced_toggle.setText('Branch settings'); self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon); self.advanced_toggle.setArrowType(Qt.RightArrow)
        layout.addWidget(self.advanced_toggle)
        self.advanced = QWidget(); advanced = QFormLayout(self.advanced); advanced.setContentsMargins(0,0,0,0)
        for name, label, minimum, maximum, default in [
            ('max_shift_mev','Max step (meV)',.01,1000,5),
            ('noise_sigma','Noise multiplier',1,20,3)]:
            spin=QDoubleSpinBox();spin.setRange(minimum,maximum);spin.setDecimals(3);spin.setValue(default);spin.setKeyboardTracking(False)
            self.numeric[name]=spin;advanced.addRow(label,spin);spin.valueChanged.connect(self.settings_changed)
        self.max_gap = QSpinBox(); self.max_gap.setRange(0,10); self.max_gap.setValue(2)
        advanced.addRow('Max missing rows',self.max_gap);self.max_gap.valueChanged.connect(self.settings_changed)
        self.numeric['noise_sigma'].setToolTip('Used only for manual seed tracking. Cached candidate connections use Max step.')
        self.max_gap.setToolTip('Used only for manual seed tracking. Cached candidate connections stop at a missing row.')
        layout.addWidget(self.advanced);self.advanced.hide();layout.addStretch(1)
        self.advanced_toggle.toggled.connect(self.toggle_advanced)
        self.scope.currentChanged.connect(self.scope_changed)
        self.polarity.currentChanged.connect(self.settings_changed)
        self.model.currentIndexChanged.connect(self.settings_changed)
        self.method.currentChanged.connect(self.method_changed)
        self.floor.valueChanged.connect(self.settings_changed);self.smoothing.valueChanged.connect(self.settings_changed)
        self.smoothing.valueChanged.connect(self.update_smoothing_hint)
        self.settings_changed.connect(self.update_preset_state)
        self.method_changed()

    def method_changed(self):
        sg=self.method.currentIndex()==1
        derivative=self._recommended_method=='local'
        self.method.setTabText(1,'Extra SG' if derivative else 'SG + extrema')
        self.method.setTabToolTip(1,'Additional SG smoothing of the supplied derivative; no derivative is recomputed.' if derivative else
                                   'Savitzky–Golay smoothing, followed by local peak finding.')
        self.smoothing_label.setText('Extra SG window' if derivative else 'SG window (samples)')
        self.smoothing.setAccessibleName('Extra SG window (samples)' if derivative else 'SG window (samples)')
        self.smoothing.setVisible(sg);self.smoothing_label.setVisible(sg)
        self.smoothing_options.setVisible(sg)
        recommendation=('Recommended: Local extrema for derivatives.' if self._recommended_method=='local' else
                        'Recommended: SG + extrema for intensity or ΔR/R.')
        description=('Savitzky–Golay (order 2), then SciPy find_peaks. Smoothing only; no extra derivative.' if sg else
                     'Local extrema: SciPy find_peaks, without additional smoothing. DRR dips use the inverted spectrum.')
        detail=('Extra smoothing before find_peaks; may weaken narrow features.' if sg and derivative else
                'Savitzky–Golay + SciPy find_peaks.' if sg else 'SciPy find_peaks; no extra smoothing.')
        self.method_description.setText(recommendation+'\n'+detail)
        self.method_description.setToolTip(description)
        self.settings_changed.emit()

    def toggle_advanced(self, checked):
        self.advanced.setVisible(checked)
        self.advanced_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def update_smoothing_hint(self, *_):
        hint='Odd number of energy samples, not meV. Smaller windows preserve narrower or closely spaced features.'
        if self._sg_step_mev is not None:
            span=(self.smoothing.value()-1)*self._sg_step_mev
            hint+=f'\nApproximate full-grid span: {span:.4g} meV. Finite segments may use shorter windows.'
        self.smoothing.setToolTip(hint)

    def scope_changed(self):
        if self.scope.currentIndex()==2 and self._custom_bounds:self.sync_range_values(self._custom_bounds)
        self.settings_changed.emit()

    def sync_range_values(self,settings):
        for key,control in self.bounds.items():
            old=control.blockSignals(True);control.setValue(settings[key]);control.blockSignals(old)

    def bounds_changed(self):
        self._custom_bounds={key:control.value() for key,control in self.bounds.items()}
        old=self.scope.blockSignals(True);self.scope.setCurrentIndex(2);self.scope.blockSignals(old)
        self.settings_changed.emit()

    def apply_preset(self,name):
        widgets={**self.numeric,'neighbor_support':self.support,'neighbor_tolerance_mev':self.neighbor_tolerance,
                 'max_width_mev':self.max_width,'max_per_row':self.max_count}
        for key,value in self._presets[name].items():
            control=widgets[key];old=control.blockSignals(True);control.setValue(value);control.blockSignals(old)
        self.settings_changed.emit()

    def update_preset_state(self):
        widgets={**self.numeric,'neighbor_support':self.support,'neighbor_tolerance_mev':self.neighbor_tolerance,
                 'max_width_mev':self.max_width,'max_per_row':self.max_count}
        selected=None
        for name,values in self._presets.items():
            checked=all(abs(widgets[key].value()-value)<1e-9 for key,value in values.items())
            self.preset_buttons[name].setChecked(checked)
            set_fluent_property(self.preset_buttons[name],'fluentAppearance','primary' if checked else 'secondary')
            if checked:selected=name
        self.preset_status.setText(selected+' filters' if selected else 'Custom filters')
        self.preset_status.setVisible(selected is None)

    def load(self, dataset):
        settings = dataset['settings']
        cube=dataset['cube'];label=profile_name(dataset['kind'],dataset['channel'],dataset.get('product_label',''))
        self._presets=filter_presets(cube,dataset['kind'],dataset['channel'],dataset.get('product_label',''))
        self.filter_heading.setText(f'2. Filter · {label} profile')
        step=sampling_step_mev(cube)
        self.filter_heading.setToolTip(f'Starting filters for {label}; not a guarantee that every point is physical. '
            f'Median energy step: {step:.4g} meV. Presets reuse the current candidate cache.')
        for name,values in self._presets.items():
            self.preset_buttons[name].setToolTip(f'{label} · {name}\nMin SNR: {values["min_snr"]:g}; '
                f'prominence: {values["prominence"]:g}; support: {values["neighbor_support"]}/5\n'
                f'Min width: {values["min_width_mev"]:g} meV; spacing: {values["min_distance_mev"]:g} meV.\n'
                'Filter cached candidates; range, polarity and detection settings stay as selected.')
        self.numeric['min_width_mev'].setToolTip('Full width at half prominence on the detection signal, not fitted FWHM. '
            f'Median step: {step:.4g} meV; Balanced starts at 3 steps. Adjust for the narrowest feature to retain.')
        self.numeric['min_distance_mev'].setToolTip('Minimum spacing between same-polarity features. '
            'Presets start at 2 median energy steps; keep below the closest branch separation you want to resolve.')
        self._sg_step_mev=float(cube.energy[-1]-cube.energy[0])*1000/(len(cube.energy)-1)
        self._recommended_method=recommended_detection_method(dataset['kind'],dataset['channel'],dataset.get('product_label',''))
        derivative=self._recommended_method=='local'
        info=dataset.get('source_processing') or {}
        if info.get('method')=='Savitzky–Golay' and info.get('derivative') in (1,2):
            label='2nd' if info['derivative']==2 else '1st'
            self.input_processing.setText(f'Input: SG {label} derivative · {info["window_samples"]} samples · order {info["polyorder"]}')
            self.input_processing.setToolTip('SG smoothing is already part of the derivative calculation. Adjust its window in the main DRR tab and open Peak Analysis again to import the new product. Compare peak positions across nearby window sizes.')
        else:
            self.input_processing.setText('Input: derivative · SG settings unavailable')
            self.input_processing.setToolTip('This snapshot does not record its derivative settings. Check the source DRR controls; do not assume this input is unsmoothed.')
        self.input_processing.setVisible(derivative)
        self._custom_bounds=dict(dataset.get('custom_range') or {})
        if dataset.get('scope')==2 and not self._custom_bounds:
            self._custom_bounds={key:settings[key] for key in self.bounds}
        self.range_labels['y'].setToolTip(dataset['cube'].gate_label)
        self.range_labels['y'].setText('Y'+(f' ({dataset["cube"].gate_unit})' if dataset['cube'].gate_unit else ''))
        for key in ('y_min','y_max'):self.bounds[key].setToolTip(dataset['cube'].gate_label+' '+key[2:])
        self.pages.setTabText(1,'Branches && fit' if dataset['kind']=='PL' else 'Branches')
        self.polarity.setVisible(dataset['kind']=='DRR'); self.polarity_label.setVisible(dataset['kind']=='DRR')
        self.fit_group.setVisible(dataset['kind']=='PL'); self.position.setVisible(dataset['kind']=='PL')
        for key, control in {**self.numeric, **self.bounds, 'max_gap':self.max_gap}.items():
            old=control.blockSignals(True);control.setValue(settings[key]);control.blockSignals(old)
        for control,value in [(self.floor,settings.get('candidate_floor',0)),(self.smoothing,settings.get('smoothing_window',11)),
                              (self.max_width,settings.get('max_width_mev',0)),(self.max_count,settings.get('max_per_row',0)),
                              (self.support,settings.get('neighbor_support',0)),(self.neighbor_tolerance,settings.get('neighbor_tolerance_mev',1.5))]:
            old=control.blockSignals(True);control.setValue(value);control.blockSignals(old)
        old=self.method.blockSignals(True);self.method.setCurrentIndex(1 if settings.get('detection_method')=='sg' else 0);self.method.blockSignals(old)
        self.method_changed()
        self.update_smoothing_hint()
        old=self.show_rejected.blockSignals(True);self.show_rejected.setChecked(dataset.get('show_rejected',False));self.show_rejected.blockSignals(old)
        old=self.model.blockSignals(True);self.model.setCurrentText(settings['model']);self.model.blockSignals(old)
        old=self.polarity.blockSignals(True);self.polarity.setCurrentIndex(['both','peaks','dips'].index(settings['polarity']));self.polarity.blockSignals(old)
        old=self.scope.blockSignals(True);self.scope.setCurrentIndex(dataset.get('scope',0));self.scope.blockSignals(old)
        self.update_preset_state()

    def settings(self, dataset):
        result = dict(dataset['settings']); cube=dataset['cube']
        result.update({key:control.value() for key,control in self.numeric.items()})
        result.update(max_gap=self.max_gap.value(), model=self.model.currentText(),
                      detection_method='sg' if self.method.currentIndex() else 'local',smoothing_window=self.smoothing.value(),
                      candidate_floor=self.floor.value(),max_width_mev=self.max_width.value(),max_per_row=self.max_count.value(),
                      neighbor_support=self.support.value(),neighbor_tolerance_mev=self.neighbor_tolerance.value(),
                      polarity=['both','peaks','dips'][self.polarity.currentIndex()] if dataset['kind']=='DRR' else 'peaks')
        if self.scope.currentIndex()==2:
            result.update({k:w.value() for k,w in self.bounds.items()})
        else:
            view = dataset.get('view') or {}
            xlim = view.get('xlim', (float(cube.energy.min()),float(cube.energy.max()))) if self.scope.currentIndex()==1 else (float(cube.energy.min()),float(cube.energy.max()))
            ylim = view.get('ylim', (float(cube.gate.min()),float(cube.gate.max()))) if self.scope.currentIndex()==1 else (float(cube.gate.min()),float(cube.gate.max()))
            result.update(x_min=min(xlim),x_max=max(xlim),y_min=min(ylim),y_max=max(ylim))
        return result
