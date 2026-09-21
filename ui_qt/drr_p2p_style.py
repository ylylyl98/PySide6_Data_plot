"""Independent publication figure and persistent P2P presentation controls."""
from collections import Counter
import numpy as np
from matplotlib.figure import Figure
from matplotlib import colormaps
from PySide6.QtWidgets import QWidget,QVBoxLayout,QFormLayout,QToolButton,QDoubleSpinBox,QSpinBox,QComboBox,QCheckBox
from core.drr_comparison_groups import parse_condition
from core.drr_p2p_metrics import METRICS,metric_values,quantity_metrics


DEFAULT_STYLE=dict(width=9.,height=5.,dpi=300,label_size=12,tick_size=10,legend_size=9,
                   title_size=12,line_width=1.5,marker_size=3.,markers=True,grid=True,
                   legend_location='best',legend_columns=1,title=True,subtitle=True,
                   field_min=0.,field_max=7.,offset=False,offset_step=.005)


class P2PStyleMixin:
    def comparison_quantity(self):
        return self.comparison_metric.currentData() if hasattr(self,'comparison_metric') and self.view_mode.currentIndex()==1 else 'amplitude'

    def metric_edge_flags(self,record,metric):
        result=record['result']
        if metric not in ('maximum','minimum'):return np.asarray(record.get('edge',np.zeros(len(result))),dtype=bool)
        energy=record['dataset'].cube.energy;low,high=record['bounds'][:2]
        energy=np.sort(energy[(energy>=low)&(energy<=high)])
        if not len(energy):return np.zeros(len(result),dtype=bool)
        spacing=float(np.median(np.diff(energy))) if len(energy)>1 else 0.
        values=metric_values(result,metric)
        return np.isfinite(values)&((values<=energy.min()+spacing)|(values>=energy.max()-spacing))

    def plot_metric_curve(self,axis,index,metric,*,compare=True):
        record=self.records[index];result=record['result']
        if result is None:return
        s=self.plot_style();curve=self.comparison_curve_styles()[index]
        offset=curve['offset'] if compare and metric=='amplitude' else 0.
        values=metric_values(result,metric)+offset;order=np.argsort(result[:,0],kind='stable')
        line,=axis.plot(result[order,0],values[order],color=curve['color'],lw=s['line_width'],alpha=1.,linestyle=curve['linestyle'],
                       marker=curve['marker'] if s['markers'] else None,ms=s['marker_size'],label=self.comparison_labels()[index],picker=6)
        line._p2p_record=index;line._p2p_order=order;line._p2p_yvalues=values;line._p2p_offset=offset
        if metric!='amplitude':
            flagged=order[self.metric_edge_flags(record,metric)[order]]
            if len(flagged):
                boundary,=axis.plot(result[flagged,0],values[flagged],linestyle='None',marker=curve['marker'],ms=s['marker_size']+1,
                                    markerfacecolor='white',markeredgecolor=curve['color'],label='_nolegend_',picker=6)
                boundary._p2p_record=index;boundary._p2p_order=flagged;boundary._p2p_yvalues=values
        return line
    def build_plot_style(self,layout):
        self.style_controls={};panel=QWidget();form=QFormLayout(panel)
        toggle=QToolButton();toggle.setText('Plot style / Export');toggle.setCheckable(True)
        toggle.toggled.connect(panel.setVisible);layout.addWidget(toggle);layout.addWidget(panel);panel.hide()
        specs=[('width','Width (in)',2,20,.5),('height','Height (in)',2,16,.5),('dpi','DPI',72,600,50),
               ('label_size','Axis labels',6,30,1),('tick_size','Tick labels',6,24,1),('legend_size','Legend font',6,24,1),
               ('title_size','Title font',6,30,1),('line_width','Line width',.3,5,.1),('marker_size','Marker size',1,10,.5),('legend_columns','Legend columns',1,8,1),
               ('field_min','Fixed B color min (T)',-100,100,1.),('field_max','Fixed B color max (T)',-100,100,1.),('offset_step','Offset per curve',0,100,.001)]
        saved={**DEFAULT_STYLE,**getattr(self.owner,'p2p_plot_style',{})}
        for key,label,low,high,step in specs:
            control=QDoubleSpinBox() if key in ('width','height','line_width','marker_size','field_min','field_max','offset_step') else QSpinBox()
            if isinstance(control,QDoubleSpinBox):control.setDecimals(4 if key=='offset_step' else 1)
            control.setRange(low,high);control.setSingleStep(step);control.setValue(saved[key]);control.setKeyboardTracking(False)
            self.style_controls[key]=control;form.addRow(label,control);control.valueChanged.connect(self.plot_style_changed)
        location=QComboBox();location.addItems(['best','upper right','upper left','lower right','lower left','upper center','lower center','center right'])
        location.setCurrentText(saved['legend_location']);self.style_controls['legend_location']=location;form.addRow('Legend position',location)
        location.currentTextChanged.connect(self.plot_style_changed)
        for key,label in [('markers','Show markers'),('grid','Show grid'),('title','Show title'),('subtitle','Show Energy / SG subtitle'),('offset','Offset curves (Compare only)')]:
            check=QCheckBox(label);check.setChecked(saved[key]);self.style_controls[key]=check;form.addRow(check);check.toggled.connect(self.plot_style_changed)

    def plot_style(self):
        return {key:(c.isChecked() if isinstance(c,QCheckBox) else c.currentText() if isinstance(c,QComboBox) else c.value()) for key,c in self.style_controls.items()}

    def restore_plot_style(self,style):
        for key,value in style.items():
            if key not in self.style_controls:continue
            c=self.style_controls[key];old=c.blockSignals(True)
            try:
                if isinstance(c,QCheckBox):c.setChecked(bool(value))
                elif isinstance(c,QComboBox):c.setCurrentText(str(value))
                else:c.setValue(value)
            finally:c.blockSignals(old)
        self.owner.p2p_plot_style=self.plot_style()

    def plot_style_changed(self,*_):
        s=self.plot_style()
        if s['field_min']>=s['field_max']:
            self.status.setText('B color maximum must be greater than minimum.');return
        self.owner.p2p_plot_style=self.plot_style()
        if self.records:self.render_comparison()

    def comparison_curve_styles(self):
        s=self.plot_style();low=s['field_min'];high=s['field_max']
        if high<=low:raise ValueError('B color maximum must be greater than minimum.')
        fields=[]
        for record in self.records:
            files=record['dataset'].provenance.get('measurement_files',[])
            c=parse_condition(files[0]) if files else None
            fields.append(c['field'] if c else None)
        order=sorted(range(len(fields)),key=lambda i:(fields[i] if fields[i] is not None else float('inf'),self.records[i]['dataset'].name,i))
        seen=Counter();styles=[None]*len(fields)
        for rank,i in enumerate(order):
            field=fields[i];repeat=seen[field];seen[field]+=1
            color=tuple(colormaps['plasma'](.05+.80*np.clip((field-low)/(high-low),0.,1.))) if field is not None else (.4,.4,.4,1.)
            styles[i]=dict(color=color,marker=('o','s','^','D','v','P','X')[repeat%7],
                           linestyle=('-','--','-.',':')[(repeat//7 if repeat>=7 else repeat)%4],
                           offset=rank*s['offset_step'] if s['offset'] else 0.,rank=rank,field=field)
        return styles

    def comparison_labels(self):
        fields=[]
        for record in self.records:
            files=record['dataset'].provenance.get('measurement_files',[])
            c=parse_condition(files[0]) if files else None
            fields.append(f"{c['field']:g} T" if c else record['dataset'].name)
        counts=Counter(fields);occurrences=Counter();labels=[]
        for base,record in zip(fields,self.records):
            occurrences[base]+=1
            if counts[base]>1:
                avg=record['dataset'].provenance.get('saved_processing',{}).get('average_count')
                base+=f" · #{occurrences[base]}"+(f' · avg {avg}' if avg is not None else '')
            labels.append(base)
        return labels

    def style_comparison_axis(self,axis,metric='amplitude',*,heading=True):
        s=self.plot_style()
        axis.set_axisbelow(True)
        axis.grid(False,which='both')
        if s['grid']:axis.grid(True,which='major',color='#aaaaaa',alpha=.4,linewidth=.6,linestyle='--')
        axis.xaxis.label.set_fontsize(s['label_size']);axis.yaxis.label.set_fontsize(s['label_size'])
        axis.tick_params(labelsize=s['tick_size'])
        title=[]
        axis.set_ylabel(METRICS[metric][0],fontsize=s['label_size'])
        if s['title'] and heading:title.append(('Peak-to-peak' if metric=='amplitude' else 'Extrema positions' if metric in ('maximum','minimum') else 'Extrema separation')+' vs '+self.cube.gate_label)
        if s['subtitle'] and heading:
            lo,hi=self.used_bounds[:2];meta=self.records[self.inspect_file.currentIndex()].get('smoothing',{})
            method='raw'
            if meta.get('method')=='SG':
                unit=meta.get('unit','meV');width=meta.get('window_points') if unit=='points' else meta.get('window_mev')
                method=f"SG {width:g} {unit} / {meta['polyorder']}"
            title.append(f'{lo:.4f}–{hi:.4f} eV · {method}')
        axis.set_title('\n'.join(title),fontsize=s['title_size'])
        if metric=='amplitude' and s['offset'] and (axis.figure is not self.figure or self.view_mode.currentIndex()==1):
            axis.set_ylabel(f"P2P + offset ({s['offset_step']:g} / curve)",fontsize=s['label_size'])
        handles,labels=axis.get_legend_handles_labels()
        if metric!='amplitude':
            from matplotlib.lines import Line2D
            handles.append(Line2D([],[],linestyle='None',marker='o',markerfacecolor='white',markeredgecolor='gray'))
            labels.append('Window edge')
        curve_styles=self.comparison_curve_styles()
        rank={label:curve_styles[i]['rank'] for i,label in enumerate(self.comparison_labels())}
        if handles:
            ordered=sorted(zip(handles,labels),key=lambda pair:rank.get(pair[1],float('inf')))
            handles,labels=zip(*ordered)
        if handles:axis.legend(handles,labels,fontsize=s['legend_size'],loc=s['legend_location'],ncol=s['legend_columns'],frameon=False)

    def comparison_export_figure(self):
        s=self.plot_style();figure=Figure(figsize=(s['width'],s['height']),dpi=s['dpi'],facecolor='white',layout='constrained')
        metrics=quantity_metrics(self.comparison_quantity())
        axes=np.atleast_1d(figure.subplots(len(metrics),1,sharex=True))
        styles=self.comparison_curve_styles()
        for panel,(axis,metric) in enumerate(zip(axes,metrics)):
            axis.set_facecolor('white')
            for i in sorted(range(len(self.records)),key=lambda i:styles[i]['rank']):
                self.plot_metric_curve(axis,i,metric)
            axis.set_xlabel(self.cube.gate_label)
            self.style_comparison_axis(axis,metric,heading=panel==0)
            if panel and axis.get_legend():axis.get_legend().remove()
            axis.xaxis.label.set_color('black');axis.yaxis.label.set_color('black');axis.title.set_color('black');axis.tick_params(colors='black')
            for spine in axis.spines.values():spine.set_color('black')
            if axis.get_legend():
                for text in axis.get_legend().get_texts():text.set_color('black')
        return figure

    def png_export_snapshot(self, *, inspection=False):
        s=self.plot_style()
        if self.view_mode.currentIndex()==1 and not inspection:
            return self.comparison_export_figure(), dict(dpi=s['dpi'],facecolor='white')
        import pickle
        was_checked=self.show_inspection.isChecked();size=self.figure.get_size_inches().copy()
        try:
            if inspection:self.show_inspection.setChecked(True)
            self.figure.set_size_inches(s['width'],s['height'],forward=False)
            return pickle.loads(pickle.dumps(self.figure)), dict(dpi=s['dpi'])
        finally:
            self.figure.set_size_inches(*size,forward=False)
            self.show_inspection.setChecked(was_checked);self.canvas.draw_idle()

    def write_png(self,path,*,inspection=False):
        figure, options = self.png_export_snapshot(inspection=inspection)
        try:figure.savefig(path, **options)
        finally:figure.clear()
