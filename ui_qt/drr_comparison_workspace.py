"""Comparison catalog integration, separate membership and scientific snapshots."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem,QDialog,QWidget,QVBoxLayout,QLabel,QFileDialog
from core.drr_comparison_groups import ComparisonStore,build_catalog,digest,product_id
from core.drr_workspace_session import session_directory,save_session,load_session


def source_version(path):
    """Cheap source identity; missing sources must never validate a cache."""
    path=Path(path)
    dat=path.with_name(path.name.removesuffix('.metadata.json')+'.dat')
    try:
        return [[p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns] for p in (path,dat)]
    except OSError:
        return None


def carry_p2p_ranges(previous,target,default_bounds=None):
    """Apply navigation locks without presenting cached values as recalculated."""
    if not previous:return target
    calculation=previous.get('fixed_calculation',False)
    display=previous.get('fixed_display',False)
    if target is None:
        if not (calculation or display):return None
        target={**previous,'records':[],'display_ranges':{},'inspection_rows':{},'current_key':None,
                'pending':True,'_recalculate_on_open':True}
        if default_bounds is not None:target['bounds']=list(default_bounds)
        target['limit_y']=False
        target['smoothing']={'method':'Off','unit':'points','window_points':21,'polyorder':2}
    else:target=dict(target)
    # Locks belong to the current navigation session, not the destination group.
    target['fixed_calculation']=calculation;target['fixed_display']=display
    if calculation:
        bounds=list(target['bounds']);bounds[:2]=previous['bounds'][:2]
        if previous.get('limit_y'):bounds[2:]=previous['bounds'][2:]
        changed=(bounds[:2]!=target['bounds'][:2] or previous.get('limit_y',False)!=target.get('limit_y',False)
                 or (previous.get('limit_y') and bounds[2:]!=target['bounds'][2:]))
        target['bounds']=bounds;target['limit_y']=previous.get('limit_y',False)
        if changed:target['pending']=True;target['_recalculate_on_open']=True
    if display:target['display_bounds']=list(previous['display_bounds'])
    return target


class ComparisonWorkspaceMixin:
    def comparison_store(self):
        root=(self._comparison_storage_base or session_directory())/'comparisons'/digest(str(Path(self.start_folder).resolve()).casefold())
        return ComparisonStore(root)

    def update_comparison_catalog(self,paths,catalog=None):
        current=self.comparison_groups.currentItem()
        selected=current.data(Qt.UserRole) if current else getattr(self,'comparison_key',None)
        self.comparison_catalog,self.comparison_entries=catalog if catalog is not None else build_catalog(paths)
        self.comparison_groups.clear()
        store=self.comparison_store()
        for key,g in sorted(self.comparison_catalog.items(),key=lambda x:(x[1]['pending'],-len(x[1]['members']),x[1]['label'])):
            members=store.members(key,g['members']);fields={(self.comparison_entries[p].get('condition') or {}).get('field') for p in members if p in self.comparison_entries}
            item=QListWidgetItem(f"{g['label']}\n{len(members)} files · {len(fields-{None})} fields")
            item.setData(Qt.UserRole,key);item.setToolTip(g['label']);self.comparison_groups.addItem(item)
            if key==selected:self.comparison_groups.setCurrentItem(item)
        self.filter_comparisons(self.comparison_search.text())

    def comparison_dataset_label(self,dataset):
        entry=self.comparison_entries.get(product_id(dataset.provenance.get('metadata_path','')),{})
        from core.drr_comparison_groups import parse_condition
        files=dataset.provenance.get('measurement_files',[])
        c=entry.get('condition') or (parse_condition(files[0]) if files else None)
        if not c:return dataset.name[:40]
        recipe=dataset.provenance.get('saved_processing',{})
        return f"{c['field']:g} T · avg {recipe.get('average_count','?')}"

    def filter_comparisons(self,text):
        terms=text.casefold().replace('−','-').split()
        for i in range(self.comparison_groups.count()):
            item=self.comparison_groups.item(i);item.setHidden(not all(t in item.text().casefold().replace('−','-') for t in terms))

    def preview_comparison(self,*_):
        if self.jobs:self._status('Wait for the current task to finish.');return
        item=self.comparison_groups.currentItem()
        if item is None:return
        from ui_qt.drr_comparison_browser import ComparisonPreview
        key=item.data(Qt.UserRole)
        dialog=ComparisonPreview(self.comparison_catalog[key],self.comparison_entries,self.comparison_store(),self)
        accepted=dialog.exec()==QDialog.Accepted
        self.update_comparison_catalog([], (self.comparison_catalog,self.comparison_entries))
        if accepted and self.comparison_key==key:self.open_comparison(key)

    def open_selected_comparison(self,*_):
        if self.jobs:self._status('Wait for the current task to finish.');return
        item=self.comparison_groups.currentItem()
        if item is None:
            self._status('Choose a comparison group in the list above first.');return
        self.open_comparison(item.data(Qt.UserRole))

    def workspace_display_state(self):
        return {'range':self.plot_range,'linked':False,'fixed':self.fixed_range.isChecked(),
                'result_filters':deepcopy(self.result_filters),'auto_detection':self.auto_detection.isChecked(),
                'analysis_tab':self.analysis_tabs.currentIndex(),
                'comparison_key':getattr(self,'comparison_key',None),'comparison_folder':self.start_folder,
                'source_versions':getattr(self,'_comparison_source_versions',{}),
                'p2p_plot_style':getattr(self,'p2p_plot_style',{}),
                'p2p':self.amplitude_page.snapshot_state() if self.amplitude_page else getattr(self,'_pending_p2p_state',None)}

    def comparison_snapshot(self):
        self.save_view()
        datasets=[replace(d,result=deepcopy(d.result),provenance=deepcopy(d.provenance)) for d in self.datasets.values()]
        return (datasets,deepcopy(self.views),self.active_key,self.workspace_display_state())

    @staticmethod
    def persist_comparison(store,key,snapshot):
        datasets,views,active,display=snapshot
        save_session(store.workspace(key or 'manual'),datasets,views,active,display,compressed=False)
        # Independent per-file snapshots survive exclusion and group changes.
        for d in datasets:
            part=deepcopy({k:v for k,v in display.items() if k!='p2p'})
            p2p=display.get('p2p')
            if p2p:
                part['p2p']={**p2p,'records':[r for r in p2p.get('records',[]) if r['dataset_key']==d.key]}
            if (not part.get('p2p') or not part['p2p']['records']) and store.dataset(d.key).exists():
                previous=load_session(store.dataset(d.key),with_display=True)[3].get('p2p')
                if previous:part['p2p']=previous
            part['result_filters']={d.key:display['result_filters'][d.key]} if d.key in display['result_filters'] else {}
            save_session(store.dataset(d.key),[d],{d.key:views[d.key]} if d.key in views else {},d.key,part,compressed=False)
        p2p=display.get('p2p')
        if p2p and not p2p.get('pending'):
            save_session(ComparisonWorkspaceMixin.history_path(store,key,p2p,datasets),datasets,views,active,display)

    @staticmethod
    def history_path(store,key,p2p,datasets):
        signature=digest([p2p['bounds'],p2p.get('limit_y'),p2p.get('smoothing'),sorted(d.key for d in datasets)])
        lo,hi=p2p['bounds'][:2];sg=p2p.get('smoothing',{}).get('method','Off')
        return store.root/'history'/digest(key or 'manual')/(f'{lo:.4f}-{hi:.4f}eV_{sg}_{signature[:16]}.npz')

    def prepare_p2p_archive(self,state):
        if not self.comparison_key or not state or state.get('pending'):return None
        datasets,views,active,display=self.comparison_snapshot()
        display['p2p']=state
        return self.history_path(self.comparison_store(),self.comparison_key,state,datasets),datasets,views,active,display

    def open_comparison(self,key):
        if self.jobs:return
        store=self.comparison_store();g=self.comparison_catalog[key]
        paths=store.members(key,g['members'])
        paths.sort(key=lambda p:((self.comparison_entries.get(p,{}).get('condition') or {}).get('field',float('inf')),p))
        if not paths:self._status('This group has no included members.');return
        versions={product_id(p):source_version(self.comparison_entries.get(p,{}).get('path',p)) for p in paths}
        current_paths={product_id(d.provenance.get('metadata_path','')) for d in self.datasets.values()}
        if (key==getattr(self,'comparison_key',None) and current_paths==set(versions)
                and all(v is not None for v in versions.values())
                and versions==getattr(self,'_comparison_source_versions',{})):
            self._status('This group is already loaded; current analysis and display are retained.')
            return
        snapshot=self.comparison_snapshot();oldkey=getattr(self,'comparison_key',None)
        def run(*,progress,log):
            from core.drr_processed_groups import load_group
            if snapshot[0]:
                log.emit('Saving previous comparison…')
                self.persist_comparison(store,oldkey,snapshot)
            log.emit('Restoring comparison cache…')
            saved=load_session(store.workspace(key),with_display=True) if store.workspace(key).exists() else None
            saved_by_path={product_id(d.provenance.get('metadata_path','')):d for d in saved[0]} if saved else {}
            saved_versions=saved[3].get('source_versions',{}) if saved else {}
            datasets=[];views={};filters={};p2p_states=[]
            for index,p in enumerate(paths):
                if self.cancel.is_set():return None
                log.emit(f'Loading comparison {index+1}/{len(paths)}')
                pid=product_id(p)
                if pid in saved_by_path and versions[pid] is not None and saved_versions.get(pid)==versions[pid]:
                    d=saved_by_path[pid];datasets.append(d)
                    filters.update({d.key:saved[3]['result_filters'][d.key]} if d.key in saved[3].get('result_filters',{}) else {})
                    progress.emit(int(100*(index+1)/len(paths)))
                    continue
                d=load_group(self.comparison_entries.get(p,{}).get('path',p))
                cache=store.dataset(d.key)
                if cache.exists():
                    prior,pviews,_,pdisplay=load_session(cache,with_display=True)
                    d=prior[0];views.update(pviews);filters.update(pdisplay.get('result_filters',{}))
                    if pdisplay.get('p2p'):p2p_states.append(pdisplay['p2p'])
                datasets.append(d);progress.emit(int(100*(index+1)/len(paths)))
            display=saved[3] if saved else {};display['result_filters']=filters
            display['source_versions']=versions
            # Restore cached per-file P2P only when its common calculation
            # definition agrees. Never mix results from different windows.
            state=display.get('p2p')
            if state:
                def definition(s):return (s['bounds'][:2],s.get('limit_y'),s['bounds'][2:] if s.get('limit_y') else None,s.get('smoothing'))
                records={r['dataset_key']:r for r in state.get('records',[])}
                for cached in p2p_states:
                    if definition(cached)==definition(state) and not cached.get('pending'):
                        for r in cached['records']:records.setdefault(r['dataset_key'],r)
                state['records']=[records[d.key] for d in datasets if d.key in records]
            if saved:views.update({k:v for k,v in saved[1].items() if k in {d.key for d in datasets}})
            return datasets,views,saved[2] if saved else datasets[0].key,display
        def received(result):
            if result is None or self._is_closing:return
            self.replace_comparison(result);self.comparison_key=key
            for i in range(self.comparison_groups.count()):
                item=self.comparison_groups.item(i)
                if item.data(Qt.UserRole)==key:
                    self.comparison_groups.setCurrentItem(item);self.comparison_groups.scrollToItem(item);break
            self.comparison_title.setText(g['label'])
            self.refresh_labels()
            self.notice.setText(f'Opened {len(self.datasets)} members. Comparison uses all members; selection only changes inspection. Repeated fields remain separate.')
        self.start_job(run,received)

    def replace_comparison(self,result):
        target_tab=self.analysis_tabs.currentIndex()
        previous_p2p=self.amplitude_page.snapshot_state() if self.amplitude_page else getattr(self,'_pending_p2p_state',None)
        self.ready=False;self.timer.stop();self.filter_timer.stop();self.overlay_timer.stop();self.gate_timer.stop()
        self.analysis_tabs.blockSignals(True)
        if self.amplitude_page:
            page=self.amplitude_page;page.auto_timer.stop();self.p2p_sidebar.takeWidget()
            self.analysis_tabs.removeTab(1);page.sidebar.deleteLater();page.deleteLater();self.amplitude_page=None
            self.amplitude_placeholder=QWidget();QVBoxLayout(self.amplitude_placeholder).addWidget(QLabel('Select Peak-to-peak to inspect this group.'))
            self.analysis_tabs.addTab(self.amplitude_placeholder,'Peak-to-peak')
        self.datasets={};self.list.clear();self.active_key=None;self.loaded=None;self.views={};self.derivative_cache.clear();self.filtered_cache.clear();self.preview_sg={};self.result_filters={};self.plot_range=None
        self.analysis.result=None;self.analysis.result_key=None;self.analysis.points=[]
        datasets,views,active,display=result
        self._comparison_source_versions=display.get('source_versions',{})
        cube=datasets[0].cube if datasets else None
        default_bounds=(float(cube.energy.min()),float(cube.energy.max()),float(cube.gate.min()),float(cube.gate.max())) if cube is not None else None
        self._pending_p2p_state=carry_p2p_ranges(previous_p2p,display.get('p2p'),default_bounds)
        # Populate in bulk. add_dataset activates and plots every member.
        for d in datasets:
            self.datasets[d.key]=d
            item=QListWidgetItem();item.setData(Qt.UserRole,d.key);self.list.addItem(item)
        self.refresh_labels()
        self.analysis_tabs.setCurrentIndex(target_tab)
        self.analysis_tabs.blockSignals(False)
        self.views=views;self.result_filters=display.get('result_filters',{})
        self.fixed_range.setChecked(display.get('fixed',False));self.auto_detection.setChecked(display.get('auto_detection',True))
        if display.get('range') is not None:self.set_display_range(display['range'])
        self.active_key=None
        if datasets:self.activate_dataset(active if active in self.datasets else datasets[0].key)
        self.analysis_tab_changed(target_tab)

    def restore_comparison_history(self):
        if self.jobs:return
        folder=self.comparison_store().root/'history'/digest(getattr(self,'comparison_key',None) or 'manual')
        filename,_=QFileDialog.getOpenFileName(self,'Restore saved analysis range',str(folder),'Analysis snapshot (*.npz)')
        if not filename:return
        keys=set(self.datasets)
        def run(**_):
            datasets,_,_,display=load_session(filename,with_display=True)
            if {d.key for d in datasets}!=keys:raise ValueError('Snapshot belongs to different members or data versions.')
            return display['p2p']
        self.start_job(run,lambda state:self.open_range_amplitude(state=state),cancellable=False)
