"""Lightweight app-side toolbar and snapshot/overlay bridge.

Never import the analysis window here: reopening the child must import fresh UI
and algorithm modules without restarting the main app.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QObject, QProcess, QThreadPool, QTimer
from PySide6.QtWidgets import QToolButton, QCheckBox
from shiboken6 import isValid

from core.peak_workspace import create_dataset, fingerprint, save_workspace, session_directory, validate_result
from core.peak_plotting import draw_overlay
from core.peak_display import capture_display
from ui_qt.common import Worker
from ui_qt.peak_analysis_ipc import MessageServer


def launch_arguments(snapshot_path=None, session_path=None, reply_server=''):
    executable=sys.executable
    root=Path(__file__).resolve().parents[1]
    if getattr(sys,'frozen',False):
        arguments=['--peak-analysis']
    else:
        if sys.platform=='win32' and Path(executable).with_name('pythonw.exe').exists():
            executable=str(Path(executable).with_name('pythonw.exe'))
        arguments=[str(root/'run_peak_analysis.py')]
    if snapshot_path:arguments+=['--snapshot',str(snapshot_path)]
    if session_path:arguments+=['--session',str(session_path)]
    if reply_server:arguments+=['--reply-server',reply_server]
    return executable,arguments,str(root)


class PeakAnalysisBridge(QObject):
    def __init__(self,owner,toolbar):
        super().__init__(owner)
        self.owner=owner;self.job=None;self.records={};self.artists=[];self.current_keys={}
        self.signature=None;self.paint_signature=None;self.queued=False;self.open_pending=False
        self.content_check_pending=False
        self.pending_packets=[];self._captured=[]
        self.pool=QThreadPool(self);self.pool.setMaxThreadCount(1)
        self.server_name='DPTK-Peak-Return-'+uuid4().hex
        self.server=MessageServer(self.server_name,self);self.server.received.connect(self.receive)
        self.show=QCheckBox('Show peaks');self.show.setChecked(True);self.show.hide()
        self.show.setAccessibleName('Show Peak Analysis overlay');toolbar.addWidget(self.show)
        self.button=QToolButton();self.button.setText('Peak Analysis...');self.button.setEnabled(False)
        self.button.setObjectName('peak_analysis_button');self.button.setAccessibleName('Open Peak Analysis')
        self.button.setToolTip('Open the current plotted data in the independent Peak Analysis window.')
        toolbar.addWidget(self.button);self.button.clicked.connect(self.open)
        self.show.toggled.connect(self.paint);owner.canvas.mpl_connect('draw_event',self.schedule_refresh)

    def sources(self):
        w=self.owner;loaded=getattr(w,'loaded',None);mode=getattr(w,'last_plotted_mode',None)
        active=getattr(w,'_active_mode',lambda:mode)()
        if loaded is None or active!=mode or loaded.mode!=mode or getattr(w,'_load_in_progress',False):return []
        if mode=='PL':cubes={'PL':getattr(w,'_pl_last_plot_cube',None)};axes=getattr(w,'_pl_heatmap_axes',{})
        elif mode=='DRR':cubes=getattr(w,'_drr_plot_cubes',{});axes=getattr(w,'_drr_heatmap_axes',{})
        elif mode=='Compare':cubes=getattr(w,'_cmp_active_cubes',{});axes=getattr(w,'_cmp_heatmap_axes',{})
        elif mode=='Power Dependent':cubes=getattr(w,'_power_active_cubes',{});axes=getattr(w,'_power_heatmap_axes',{})
        else:return []
        source=json.dumps([mode,str(getattr(loaded,'primary_file','')),list(getattr(loaded,'selected_files',[]))],sort_keys=True)
        out=[]
        for channel,cube in cubes.items():
            if cube is None or channel=='VP':continue
            target=list(axes.values()) if mode=='PL' else [axes[channel]] if channel in axes else []
            view={}
            if target:
                display_axis=axes.get('log' if getattr(w,'_pl_active_scale',False) else 'linear',target[0]) if mode=='PL' else target[0]
                view=dict(xlim=list(display_axis.get_xlim()),ylim=list(display_axis.get_ylim()),
                          y_axis_log=display_axis.get_yscale()=='log',display=capture_display(display_axis))
            prefix={'PL':'pl','DRR':'drr','Compare':'cmp','Power Dependent':'power'}[mode]
            spins=getattr(w,prefix+'_spins',{})
            gate=float(spins['gate'].value()) if 'gate' in spins else float(cube.gate[0])
            if mode=='DRR' and hasattr(w,'_drr_gate_value'):gate=float(w._drr_gate_value())
            row=int(np.argmin(abs(np.asarray(cube.gate)-gate)))
            out.append(dict(cube=cube,kind='DRR' if mode=='DRR' else 'PL',channel=channel,source=source,
                name=f'{mode} · {channel} · {Path(getattr(loaded,"primary_file","") or cube.title).stem}',
                axes=target,view=view,row=row,mode=mode))
            if mode=='DRR':
                out[-1]['group_cube']=getattr(loaded,'cube',cube)
                label='2nd derivative' if channel=='second' else 'ΔR/R'
                second_kind=getattr(w,'drr_second_kind_combo',None)
                if channel=='second' and second_kind is not None and second_kind.currentData()=='mixed':label='Mixed derivative'
                if channel=='raw' and not getattr(w,'_drr_side_by_side',False) and 'Advanced first' in str(getattr(loaded,'drr_derivative_label','')):label='1st derivative'
                out[-1]['product_label']=label
                out[-1]['source_name']=f'DRR · {Path(getattr(loaded,"primary_file","") or cube.title).stem}'
                # Read the provenance of the displayed object, not controls
                # that may already describe the next asynchronous transform.
                for cache_key,(cached,used_window) in getattr(w,'_drr_derivative_cache',{}).items():
                    if cached is not cube or cache_key[1] not in (1,2):continue
                    out[-1]['source_processing']=dict(method='Savitzky–Golay',derivative=cache_key[1],
                        window_samples=int(used_window),polyorder=int(cache_key[3]))
                    break
        if mode=='DRR':out.sort(key=lambda s:s['channel']==getattr(w,'_drr_plot_view','raw'))
        return out

    @staticmethod
    def source_signature(sources):
        return tuple((s['kind'],s['channel'],s['source'],id(s['cube']),id(s['cube'].Z),id(s['cube'].energy),id(s['cube'].gate)) for s in sources)

    def schedule_refresh(self,*_):
        if isValid(self) and isValid(self.owner) and not self.queued:
            self.queued=True;QTimer.singleShot(0,self,self.refresh)

    def status(self,message):
        callback=getattr(self.owner,'_status',None)
        if callback:callback(message)

    def refresh(self,check_content=True):
        self.queued=False;sources=self.sources();signature=self.source_signature(sources)
        self.button.setEnabled(bool(sources) and not self.job)
        self.button.setToolTip('Analyze plotted intensity/DRR data in a separate process.' if sources else 'Plot PL or DRR data first. For Compare/Power, select Intensity instead of VP.')
        if not sources:
            self.current_keys={};self.signature=None;self.clear_artists();self.show.hide();return
        if (self.records or self.open_pending) and (signature!=self.signature or check_content):
            if signature!=self.signature:self.clear_artists()
            if self.job:self.content_check_pending=True
            else:self.prepare(sources,self.open_pending)
        else:self.paint()

    def start(self,work,accept):
        if self.job:return
        worker=Worker(lambda *,progress,log:work());self.job=worker;self.button.setEnabled(False)
        # Python closures can remain queued after their Qt owner is destroyed.
        def guarded(callback):
            def deliver(*args):
                if isValid(self) and isValid(self.owner):callback(*args)
            return deliver
        worker.signals.result.connect(guarded(accept))
        worker.signals.error.connect(guarded(lambda message:self.status(message.split('\n')[0])))
        def finished():
            self.job=None
            if self.pending_packets:
                self.receive(self.pending_packets.pop(0))
            elif self.open_pending:
                self.open_pending=False;self.open()
            elif self.content_check_pending:
                self.content_check_pending=False;self.refresh()
            else:QTimer.singleShot(0,self,lambda:self.refresh(check_content=False))
        worker.signals.finished.connect(guarded(finished));self.pool.start(worker)

    def prepare(self,sources,open_window=False):
        signature=self.source_signature(sources);self.open_pending=False
        self.content_check_pending=False
        def work():
            keys={};datasets=[]
            for s in sources:
                key=fingerprint(s['cube'],s['kind'],s['channel'],s['source']);keys[s['channel']]=key
                if open_window:
                    d=create_dataset(s['cube'],s['kind'],s['channel'],s['source'],s['name'],s['view'])
                    d['row']=s['row'];d['reply_server']=self.server_name;datasets.append(d)
                    for field in ('product_label','source_name','source_processing'):
                        if field in s:d[field]=s[field]
            path=None
            if open_window:
                if datasets[0]['kind']=='DRR':
                    source=sources[0]
                    group_id=fingerprint(source['group_cube'],'DRR','source',source['source'])
                    for d in datasets:d['group_id']=group_id
                path=session_directory()/'inbox'/(uuid4().hex+'.npz')
                save_workspace(path,datasets,datasets[-1]['key'])
            return keys,path
        def accept(result):
            keys,path=result
            if self.source_signature(self.sources())==signature and not self.content_check_pending:
                self.signature=signature;self.current_keys=keys;self._captured=sources;self.paint()
            if path:
                executable,args,cwd=launch_arguments(path,reply_server=self.server_name)
                success,_=QProcess.startDetached(executable,args,cwd)
                self.status('Peak Analysis opened with a data snapshot.' if success else 'Could not launch Peak Analysis.')
        self.start(work,accept)

    def open(self):
        sources=self.sources()
        if not sources:return
        if self.job:self.open_pending=True;return
        self.status('Preparing Peak Analysis snapshot...');self.prepare(sources,True)

    def receive(self,message):
        path=message.get('overlay')
        if not path:return
        if self.job:self.pending_packets.append(message);return
        path=Path(path)
        try:path.resolve().relative_to((session_directory()/'overlays').resolve())
        except ValueError:return
        def read():
            with path.open(encoding='utf-8') as stream:return json.load(stream)
        def accept(packet):
            self.accept_overlay(packet);path.unlink(missing_ok=True)
        self.start(read,accept)

    def accept_overlay(self,packet):
        # Validate coordinates against the current numeric snapshot before painting.
        key=packet.get('key')
        if not key or not packet.get('result') or packet['result'].get('dataset_key')!=key:
            self.status('Rejected an invalid Peak Analysis overlay.');return
        self.records[key]=packet
        if len(self.records)>50:self.records.pop(next(iter(self.records)))
        self.paint_signature=None;self.signature=None
        self.show.setChecked(True)
        if not self.job:self.refresh()

    def clear_artists(self):
        had=bool(self.artists)
        for artist in self.artists:
            try:artist.remove()
            except (ValueError,AttributeError):pass
        self.artists=[];self.paint_signature=None
        if had:self.owner.canvas.draw_idle()

    def paint(self,*_):
        sources=self.sources()
        if self.source_signature(sources)!=self.signature:return
        selected=[(s,self.records[self.current_keys[s['channel']]]) for s in sources
                  if self.current_keys.get(s['channel']) in self.records]
        self.show.setVisible(bool(selected))
        signature=(self.show.isChecked(),tuple((p['key'],id(p),tuple(map(id,s['axes']))) for s,p in selected))
        if signature==self.paint_signature:return
        self.clear_artists();self.paint_signature=signature
        if not self.show.isChecked():return
        for s,packet in selected:
            dataset={**packet,'cube':s['cube']}
            try:validate_result(dataset,packet['result'])
            except (ValueError,KeyError,TypeError):
                self.status('Rejected stale or invalid peak coordinates.');continue
            transform=None
            if s['mode']=='Power Dependent' and hasattr(self.owner,'power_controller'):
                _,original,display=self.owner.power_controller._display_power_cube(s['cube'])
                def transform(values,original=original,display=display):
                    return np.interp(values,original,display,left=np.nan,right=np.nan)
            for axis in s['axes']:
                self.artists.extend(draw_overlay(axis,dataset,transform))
        if self.artists:self.owner.canvas.draw_idle()
