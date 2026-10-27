"""Native PL/DRR source picker: cached inventory, history and explicit selection."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import perf_counter

from core.load_cancellation import LoadCancellation

from PySide6.QtCore import QAbstractListModel,QModelIndex,QObject,Property,Qt,Signal,Slot,QUrl,QFileSystemWatcher,QTimer


class _CatalogCancellation(LoadCancellation):
    """Keep partial inspection work while an explicitly chosen file loads.

    Pause between files instead of discarding the inspection or holding a CSV
    open during a foreground load. Ordinary cancellation checks never wait for a resume.
    A process-pool batch already dispatched may finish, but no further files
    are dispatched until the load finishes. Cancellation always releases it.
    """
    def __init__(self):
        super().__init__()
        self._resumed=Event();self._resumed.set()

    def pause(self):
        if not self.cancelled:self._resumed.clear()

    def resume(self):
        self._resumed.set()

    def cancel(self):
        super().cancel();self.resume()

    def checkpoint(self):
        super().check()
        while not self._resumed.is_set():
            self._resumed.wait(.1)
            super().check()


class CatalogModel(QAbstractListModel):
    EntryRole=int(Qt.ItemDataRole.UserRole)+1
    def __init__(self,parent=None):
        super().__init__(parent);self.rows=[]
    def roleNames(self):return {self.EntryRole:b"entry"}
    def rowCount(self,parent=QModelIndex()):return 0 if parent.isValid() else len(self.rows)
    def data(self,index,role=Qt.ItemDataRole.DisplayRole):
        return self.rows[index.row()] if role==self.EntryRole and index.isValid() and 0<=index.row()<len(self.rows) else None
    @Slot(str,result=int)
    def indexOfId(self,identity):
        return next((index for index,row in enumerate(self.rows) if row["id"]==identity),-1)
    def replace(self,rows):
        if rows==self.rows:return
        if [row["id"] for row in rows]==[row["id"] for row in self.rows]:
            changed=[index for index,(old,new) in enumerate(zip(self.rows,rows)) if old!=new]
            self.rows=rows
            if changed:
                first=last=changed[0]
                for index in changed[1:]:
                    if index==last+1:last=index;continue
                    self.dataChanged.emit(self.index(first),self.index(last),[self.EntryRole]);first=last=index
                self.dataChanged.emit(self.index(first),self.index(last),[self.EntryRole])
        else:
            self.beginResetModel();self.rows=rows;self.endResetModel()


def _saved_drr_inventory(root):
    from core.load_cancellation import check_load_cancelled
    history=(root if root.name.casefold()=="drr" and root.parent.name.casefold()=="processed data"
             else root/"DRR" if root.name.casefold()=="processed data" else root/"Processed Data"/"DRR")
    rows=[]
    for metadata in sorted(history.rglob("*.metadata.json")):
        check_load_cancelled()
        for path in (metadata,metadata.with_name(metadata.name.removesuffix(".metadata.json")+".dat")):
            try:stat=path.stat()
            except FileNotFoundError:continue
            rows.append([path.relative_to(root).as_posix(),stat.st_size,stat.st_mtime_ns])
    return rows


def _catalog_inventory(folder,workflow,include_all=False):
    """Track only inputs which can change this picker, excluding its own cache."""
    from core.load_cancellation import check_load_cancelled
    root=Path(folder).resolve()
    if workflow=="drr":
        from core.drr_sources import drr_source_paths
        paths=drr_source_paths(root,include_all=include_all)
        rows=_saved_drr_inventory(root)
    else:
        from core.pl_sources import list_pl_source_files
        paths=[root/source for source in list_pl_source_files(str(root))]
        paths.extend((root/"Processed Data"/"PL").rglob("*.metadata.json"))
        rows=[]
    for path in paths:
        check_load_cancelled()
        try:stat=path.stat()
        except FileNotFoundError:continue
        rows.append([path.relative_to(root).as_posix(),stat.st_size,stat.st_mtime_ns])
    return sorted(rows)


def _saved_drr_rows(root,cancellation=None):
    from core.drr_processed_groups import discover_groups
    rows=[]
    for metadata in discover_groups(root):
        if cancellation:cancellation.check()
        path=metadata.with_name(metadata.name.removesuffix(".metadata.json")+".dat")
        try:stat=path.stat()
        except OSError:continue
        rows.append(dict(id=str(path.resolve()),source=path.relative_to(root).as_posix(),name=path.name,kind="DAT",status="saved",saved=True,
            modified=stat.st_mtime,size=stat.st_size,processedAt="",background=False,detail="Saved processed DRR matrix"))
    return rows


def _drr_catalog_rows(root,sources,saved=()):
    from core.drr_sources import group_drr_sources
    from ui_qt.quick.catalog_summary import source_catalog_detail,group_catalog_detail
    paths={source.source:str(root/source.source) for source in sources}
    rows=[]
    for source in sources:
        rows.append(dict(id=paths[source.source],source=source.source,name=source.filename,
            kind=source.classification,status="processed" if source.processed else "new",saved=False,
            modified=source.modified_time,size=source.size_bytes,processedAt="",background=source.is_background,
            detail=source_catalog_detail(source),
            linkedBackgrounds=list(source.linked_backgrounds),savedModes=list(source.saved_baseline_modes)))
    groups=[]
    for group in group_drr_sources(sources):
        groups.append(dict(id=group.key,name=group.title,session=group.session_date,
            members=[paths[source.source] for source in group.files],modified=group.modified_time,
            background=group.is_background,kind=group.classification,processedCount=group.processed_count,
            count=len(group.files),complete=group.grid_complete,detail=group_catalog_detail(group)))
    return dict(files=sorted([*rows,*saved],key=lambda row:(-row["modified"],row["source"].casefold())),groups=groups,sources=tuple(sources))


def scan_catalog(folder,workflow,*,include_all=False,cache_path=None,cancellation=None,force=False,_uncached=False,
                 publish_cached=None,publish_inventory=None):
    from core.load_cancellation import load_scope
    root=Path(folder).resolve()
    with load_scope(cancellation):
        if workflow=="pl" and not _uncached and cache_path is not None:
            from core.source_catalog_cache import SourceCatalogCache
            cache=SourceCatalogCache(str(root),workflow+"-"+str(int(include_all)),cache_root=Path(cache_path).parent/"inventory")
            return cache.refresh(lambda:scan_catalog(root,workflow,include_all=include_all,cache_path=cache_path,cancellation=cancellation,_uncached=True),force=force,publish_cached=publish_cached)
        if workflow=="pl":
            from core.pl_sources import list_pl_source_files,classify_pl_source,discover_pl_processing_status
            paths=list_pl_source_files(str(root));ambiguous=set()
            status=discover_pl_processing_status(root,paths,ambiguous_sources=ambiguous)
            rows=[]
            for source in paths:
                if cancellation:cancellation.check()
                path=root/source
                try:stat=path.stat()
                except OSError:continue
                saved=path.suffix.casefold()==".dat"
                kind=classify_pl_source(source)
                state="saved" if saved else "processed" if source in status else "unknown" if source in ambiguous else "new"
                rows.append(dict(id=str(path),source=source,name=path.name,kind=kind,status=state,saved=saved,
                    modified=stat.st_mtime,size=stat.st_size,processedAt=status.get(source,""),background=kind=="REF",
                    detail=("REF raw preview · no background subtraction" if kind=="REF" else "History unknown · legacy metadata matches multiple files" if state=="unknown" else "")))
            return dict(files=sorted(rows,key=lambda row:(-row["modified"],row["source"].casefold())),groups=[],sources=())
        from core.drr_sources import DrrSourceCache
        from core.drr_catalog import load_drr_catalog
        from core.source_catalog_cache import SourceCatalogCache
        cache_root=Path(cache_path).parent/"inventory" if cache_path is not None else None
        saved_cache=SourceCatalogCache(str(root),"quick-DRR-saved-v1",cache_root=cache_root,inventory_provider=lambda:_saved_drr_inventory(root))
        saved=saved_cache.refresh(lambda:_saved_drr_rows(root,cancellation),force=force)
        prepared_sources=prepared_rows=None
        def prepare(sources):
            nonlocal prepared_sources,prepared_rows
            if sources is not prepared_sources:
                prepared_sources,prepared_rows=sources,_drr_catalog_rows(root,sources,saved)
            return prepared_rows
        if saved and publish_cached is not None:publish_cached(prepare(()))
        cache=DrrSourceCache(cache_path,load_on_init=False)
        sources=load_drr_catalog(root,cache,force=force,include_all=include_all,parallel_inspection=True,cache_root=cache_root,
            publish_cached=(lambda sources:publish_cached(prepare(sources))) if publish_cached is not None else None,
            publish_inventory=(lambda sources:publish_inventory(prepare(sources))) if publish_inventory is not None else None)
        if cancellation:cancellation.check()
        result=prepare(sources)
        # Saved exports can change while a cold raw inspection is still busy.
        # Revalidate their small inventory before certifying the final catalog.
        current_saved=saved_cache.refresh(lambda:_saved_drr_rows(root,cancellation))
        if cancellation:cancellation.check()
        if current_saved!=saved:
            result={**result,"files":sorted([row for row in result["files"] if not row["saved"]]+current_saved,
                key=lambda row:(-row["modified"],row["source"].casefold()))}
        return result


class SourceCatalogBridge(QObject):
    snapshotChanged=Signal()
    selectionAccepted=Signal(str,object)
    dialogRequested=Signal(str,bool)
    statusMessage=Signal(str)
    operationCompleted=Signal(str,float)
    _completed=Signal(int,object,str,float,object)
    _previewed=Signal(int,object)
    _watch_checked=Signal(int,object)

    def __init__(self,*,settings,parent=None):
        super().__init__(parent)
        self.settings=settings
        self._workflow,self._folder,self._query="pl","",""
        self._type=str(settings.value("quick/catalog/plType","PL"))
        self._status_filter=str(settings.value("pl/source_filter","all"))
        self._saved_only=self._include_all=self._include_backgrounds=False
        self._only_new=str(settings.value("drr/source_unprocessed_only",True)).lower() in {"true","1","yes","on"}
        self._selected=[];self._explicit_selected=set();self._selected_group="";self._focused=""
        self._reopen_group_pending=False
        self._loaded_positions={}
        self._pending_load_position=None
        self._catalog=dict(files=[],groups=[],sources=())
        self._catalog_validated=False
        self._preserve_catalog=False
        self._indexed_catalog=None;self._lookup={};self._counts={status:0 for status in ("new","processed","unknown")}
        self._busy,self._closed,self._generation=False,False,0
        self._dialog_open=self._foreground_busy=self._prepare_pending=False
        self._status,self._error="Choose an experiment folder.",""
        self._cancel=None;self._future=None
        self._scan_scope=None
        self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="dptk-source-catalog")
        self._groups,self._files,self._chosen=CatalogModel(self),CatalogModel(self),CatalogModel(self)
        self._completed.connect(self._finish)
        self._previewed.connect(self._preview)
        self._watch_checked.connect(self._finish_watch_check)
        self._watcher=QFileSystemWatcher(self)
        self._watch_timer=QTimer(self);self._watch_timer.setSingleShot(True);self._watch_timer.setInterval(350)
        self._watch_timer.timeout.connect(self._watch_refresh)
        self._watcher.directoryChanged.connect(self._directory_changed)
        self._refresh_pending=self._watch_active=self._watch_probe_running=False
        self._watch_serial=0;self._watch_cancel=None
        self._watch_signature=None
        self._watch_candidate=None;self._watch_candidate_since=0.
        self._watch_settle_ms=3000
        self._prepare_timer=QTimer(self);self._prepare_timer.setSingleShot(True);self._prepare_timer.setInterval(150)
        self._prepare_timer.timeout.connect(self._prepare_refresh)

    @Property(QObject,constant=True)
    def groupsModel(self):return self._groups
    @Property(QObject,constant=True)
    def filesModel(self):return self._files
    @Property(QObject,constant=True)
    def chosenModel(self):return self._chosen
    @Property("QVariantMap",notify=snapshotChanged)
    def snapshot(self):
        return dict(workflow=self._workflow,folder=self._folder,query=self._query,type=self._type,statusFilter=self._status_filter,
            savedOnly=self._saved_only,includeAll=self._include_all,includeBackgrounds=self._include_backgrounds,onlyNew=self._only_new,
            selected=list(self._selected),selectedGroup=self._selected_group,focused=self._focused,reopenPositionPending=self._reopen_group_pending,busy=self._busy,status=self._status,error=self._error,
            counts=dict(self._counts),canAccept=bool(self._selected) and (self._catalog_validated or set(self._selected)<=self._explicit_selected),
            canAddGroup=not self._busy and self._catalog_validated,detail=self._lookup.get(self._focused,{}).get("detail",""))

    @Slot(str,str,"QVariantList",bool)
    def open(self,workflow,folder,selected,saved_only=False):
        if self._configure(workflow,folder,selected,saved_only,restore_position=True):
            self._dialog_open=True
            self._prepare_pending=False;self._prepare_timer.stop()
            self._watch_active=True
            scope=(self._workflow,self._folder,self._include_all)
            # Opening a picker changes selection, not its source inventory.
            # Reuse an ongoing inspection or the last validated list; a small
            # asynchronous inventory probe catches changes while it was hidden.
            if self._busy and self._scan_scope==scope:
                if not self._foreground_busy:self._cancel.resume()
                return
            if self._catalog_validated:
                self._sync_watches()
                if not self._watch_probe_running and not self._watch_timer.isActive():
                    self._watch_timer.start(0)
            else:self.refresh()

    @Slot(str,str)
    def prepare(self,workflow,folder):
        """Warm the active folder without opening or changing a picker draft."""
        if self._closed or workflow not in {"pl","drr"} or not folder:return
        folder=str(Path(folder).resolve())
        if (workflow,folder)!=(self._workflow,self._folder):
            self._dialog_open=False
            if not self._configure(workflow,folder,[],False):return
        self._watch_active=True
        self._sync_watches()
        if self._busy:return
        if self._catalog_validated:
            if not self._watch_probe_running and not self._watch_timer.isActive():self._watch_timer.start(350)
        elif not self._prepare_pending:
            self._prepare_pending=True
            self._prepare_timer.start()

    def _prepare_refresh(self):
        if self._closed or not self._prepare_pending or self._foreground_busy:return
        self._prepare_pending=False
        if not self._busy:self.refresh()

    @Slot(bool)
    def setForegroundBusy(self,busy):
        """Yield source inspection while an explicitly requested load runs."""
        busy=bool(busy)
        if self._closed or busy==self._foreground_busy:return
        self._foreground_busy=busy
        if self._foreground_busy:
            self._watch_timer.stop();self._cancel_watch_check()
            self._refresh_pending=True
        if self._cancel:
            if self._foreground_busy:self._cancel.pause()
            else:self._cancel.resume()
        if not self._foreground_busy and self._prepare_pending:self._prepare_timer.start(0)
        elif not self._foreground_busy and self._refresh_pending and self._watch_active:self._watch_timer.start(0)

    def _configure(self,workflow,folder,selected,saved_only=False,*,restore_position=False):
        if self._closed or workflow not in {"pl","drr"}:return
        folder=str(Path(folder or Path.cwd()).resolve())
        changed=(workflow,folder)!=(self._workflow,self._folder)
        if changed or bool(saved_only)!=self._saved_only:
            self._focused="";self._selected_group=""
        self._workflow,self._folder,self._saved_only=workflow,folder,bool(saved_only)
        self._selected=[str(Path(path).resolve()) for path in selected]
        self._explicit_selected=set(self._selected)
        # Incoming sources are a fallback until a successful load is recorded.
        # Clear, canceled drafts and failed selections cannot replace that load.
        if self._selected and self._focused not in self._selected:self._focused=self._selected[-1]
        if restore_position:
            loaded=self._loaded_positions.get((workflow,folder,bool(saved_only)))
            if loaded is not None:
                self._focused=loaded[1]
                self._selected_group=""
        self._reopen_group_pending=bool(restore_position and workflow=="drr" and not saved_only and self._focused)
        if changed:
            self._prepare_timer.stop();self._prepare_pending=False
            if self._cancel:self._cancel.cancel()
            self._generation+=1;self._busy=False;self._scan_scope=None
            self._stop_watches()
            self._selected_group="";self._catalog=dict(files=[],groups=[],sources=());self._catalog_validated=False
            self._watch_signature=None
        self._refresh_models()
        return True

    @Slot(str,str,"QVariantList")
    def recordLoaded(self,workflow,folder,files):
        """Bookmark a successful measurement load, separately from picker drafts."""
        if self._closed or workflow not in {"pl","drr"}:return
        paths=tuple(str(Path(path).resolve()) for path in files)
        if not paths:return
        scope=(workflow,str(Path(folder or Path.cwd()).resolve()),all(Path(path).suffix.casefold()==".dat" for path in paths))
        previous=self._loaded_positions.get(scope)
        pending=self._pending_load_position
        if pending is not None and pending[:2]==(scope,paths):
            target=pending[2]
            self._pending_load_position=None
        elif previous is not None and previous[0]==paths:
            # Reprocessing the same loaded selection must not move its bookmark.
            target=previous[1]
        else:target=paths[-1]
        self._loaded_positions[scope]=(paths,target)

    @Slot(str,str,"QVariantList",bool)
    def refreshCurrent(self,workflow,folder,selected,force=False):
        if self._configure(workflow,folder,selected,False):self.refresh(force)

    def _scan_catalog(self,folder,workflow,**options):
        return scan_catalog(folder,workflow,**options)

    def _source_inventory(self,folder,workflow,include_all=False):
        return _catalog_inventory(folder,workflow,include_all)

    def _display_cache(self,folder,workflow,include_all,cache_path):
        from core.source_catalog_cache import SourceCatalogCache
        # This snapshot is a display aid, never a validated inventory or source
        # inspection. A live acquisition can prevent the certified raw/history
        # caches from being written for hours; its last list is still useful.
        namespace=f"quick-picker-display-v1-{type(self).__name__}-{workflow}-{int(include_all)}"
        return SourceCatalogCache(folder,namespace,cache_root=Path(cache_path).parent/"inventory",
            inventory_provider=lambda:[])

    @staticmethod
    def _display_snapshot(result):
        if not isinstance(result,dict) or not isinstance(result.get("files"),list) or not isinstance(result.get("groups"),list):return None
        required={"id","source","name","kind","status","saved","modified","processedAt","background","detail"}
        if not all(isinstance(row,dict) and required<=row.keys() for row in result["files"]):return None
        required={"id","name","session","members","modified","background","kind","processedCount","count","complete","detail"}
        if not all(isinstance(row,dict) and required<=row.keys() for row in result["groups"]):return None
        return dict(files=result["files"],groups=result["groups"],sources=())

    @Slot()
    @Slot(bool)
    def refresh(self,force=False):
        if self._closed:return
        from hashlib import sha256
        if self._cancel:self._cancel.cancel()
        cancellation=self._cancel=_CatalogCancellation()
        if self._foreground_busy:cancellation.pause()
        self._prepare_timer.stop();self._prepare_pending=False
        self._generation+=1;generation=self._generation
        self._watch_timer.stop();self._refresh_pending=False
        self._cancel_watch_check()
        self._watch_candidate=None;self._watch_candidate_since=0.
        folder,workflow,include_all=self._folder,self._workflow,self._include_all
        self._scan_scope=(workflow,folder,include_all)
        from PySide6.QtCore import QSettings,QStandardPaths
        base=Path(self.settings.fileName()).parent if self.settings.format()==QSettings.IniFormat else Path(QStandardPaths.writableLocation(QStandardPaths.CacheLocation))
        cache=base/"quick-source-catalog"/(sha256(folder.casefold().encode()).hexdigest()+".json")
        self._preserve_catalog=self._catalog_validated
        self._busy=True;self._catalog_validated=False;self._error="";self._status="Inspecting sources and saved processing history…";self.snapshotChanged.emit()
        def run():
            start=perf_counter()
            inventory=None
            try:
                from core.load_cancellation import load_scope,check_load_checkpoint
                preview=lambda value:self._previewed.emit(generation,value) if not self._closed and generation==self._generation else None
                with load_scope(cancellation):
                    check_load_checkpoint()
                    display_cache=self._display_cache(folder,workflow,include_all,cache)
                    cached=self._display_snapshot(display_cache.read())
                    if cached is not None:preview(cached)
                    # Capture the pre-scan signature after the optimistic list
                    # was published. A change during inspection must still be
                    # found by the post-scan watch probe.
                    inventory=self._source_inventory(folder,workflow,include_all)
                    check_load_checkpoint()
                result=self._scan_catalog(folder,workflow,include_all=include_all,cache_path=cache,cancellation=cancellation,force=bool(force),
                    publish_cached=preview,publish_inventory=preview)
                cancellation.checkpoint()
                snapshot=self._display_snapshot(result)
                if snapshot is not None:display_cache.refresh(lambda:snapshot,force=True)
                error=""
            except Exception as failure:result,error=None,"" if cancellation.cancelled else str(failure)
            if not self._closed and generation==self._generation:
                self._completed.emit(generation,result,error,(perf_counter()-start)*1000,inventory)
        self._future=self._executor.submit(run)

    @Slot(int,object)
    def _preview(self,generation,result):
        if self._closed or generation!=self._generation or not self._busy:return
        self._status="Source inventory is available; checking acquisition grids and processing history…"
        # A saved-results-only or filename-only preview is useful on a cold
        # open. It must not replace a validated list during a warm refresh.
        saved_preview=result["files"] and all(row["saved"] for row in result["files"])
        have_raw=any(not row["saved"] for row in self._catalog["files"])
        if self._preserve_catalog or saved_preview and have_raw:self.snapshotChanged.emit()
        else:
            self._catalog=result;self._catalog_validated=False
            self._refresh_models()

    @Slot(str)
    def refreshFolder(self,folder):
        if self._folder and Path(folder).resolve()==Path(self._folder).resolve():self.refresh()

    @Slot(int,object,str,float,object)
    def _finish(self,generation,result,error,elapsed,inventory):
        if self._closed or generation!=self._generation:return
        self._busy=False;self._error=error
        if result is not None:self._catalog=result;self._catalog_validated=True;self._watch_signature=inventory
        if result is not None:self._sync_watches()
        self._status=error or f"{len(self._catalog['files'])} files · {len(self._catalog['groups'])} acquisition groups"
        self._refresh_models()
        if self._dialog_open:self.statusMessage.emit(self._status)
        self.operationCompleted.emit("catalog",elapsed)
        # A file may have changed while the first scan had no directory watches
        # yet. Recheck its inventory without re-entering the visible busy state.
        self._refresh_pending=False
        if self._watch_active and result is not None:self._watch_timer.start(350)

    def _sync_watches(self):
        if not self._watch_active:return
        # Watch only the active experiment and discovered source directories.
        paths={self._folder,*(str(Path(row["id"]).parent) for row in self._catalog["files"])}
        root=Path(self._folder)
        paths.update(str(path) for path in (root/"Initial Data",root/"Initial Data"/"REF",root/"Processed Data",root/"Processed Data"/self._workflow.upper()) if path.is_dir())
        wanted={path for path in paths if Path(path).is_dir()}
        current=set(self._watcher.directories())
        if current-wanted:self._watcher.removePaths(list(current-wanted))
        if wanted-current:self._watcher.addPaths(list(wanted-current))

    def _stop_watches(self):
        self._watch_active=self._refresh_pending=False
        self._cancel_watch_check()
        self._watch_candidate=None;self._watch_candidate_since=0.
        self._watch_timer.stop()
        if self._watcher.directories():self._watcher.removePaths(self._watcher.directories())

    def _cancel_watch_check(self):
        self._watch_serial+=1
        if self._watch_cancel:self._watch_cancel.cancel()
        self._watch_cancel=None;self._watch_probe_running=False

    @Slot(str)
    def _directory_changed(self,path):
        if self._watch_active and not self._closed:self._watch_timer.start(350)

    def _watch_refresh(self):
        if self._closed or not self._watch_active:return
        if self._busy or self._foreground_busy or self._watch_probe_running:
            self._refresh_pending=True
            return
        from core.load_cancellation import LoadCancellation
        self._watch_probe_running=True
        self._watch_serial+=1;serial=self._watch_serial
        folder,workflow,include_all=self._folder,self._workflow,self._include_all
        cancellation=self._watch_cancel=LoadCancellation()
        def check():
            try:
                from core.load_cancellation import load_scope
                with load_scope(cancellation):inventory=self._source_inventory(folder,workflow,include_all)
            except Exception:inventory=None
            if not self._closed and serial==self._watch_serial:self._watch_checked.emit(serial,inventory)
        self._executor.submit(check)

    @Slot(int,object)
    def _finish_watch_check(self,serial,inventory):
        if self._closed or not self._watch_active or serial!=self._watch_serial:return
        self._watch_probe_running=False;self._watch_cancel=None
        pending,self._refresh_pending=self._refresh_pending,False
        if inventory is None or inventory==self._watch_signature:
            # A failed probe is not evidence of changed sources. Do not create
            # a scan -> failed post-scan probe -> scan feedback loop.
            self._watch_candidate=None;self._watch_candidate_since=0.
            if pending:self._watch_timer.start(350)
            return
        now=perf_counter()
        if inventory!=self._watch_candidate:
            self._watch_candidate=inventory;self._watch_candidate_since=now
        remaining=self._watch_settle_ms-(now-self._watch_candidate_since)*1000
        if remaining>0:
            # Acquisition appends can arrive seconds apart. Keep the validated
            # list usable until the inventory settles, using metadata probes
            # only; a manual Refresh still inspects immediately.
            self._watch_timer.start(max(1,int(remaining)+1))
        else:self.refresh()

    def _refresh_models(self):
        if self._indexed_catalog is not self._catalog:
            self._indexed_catalog=self._catalog
            self._lookup={row["id"]:row for row in self._catalog["files"]}
            self._counts={status:sum(row["status"]==status for row in self._catalog["files"] if not row["saved"]) for status in ("new","processed","unknown")}
        terms=self._query.casefold().replace("−","-").split()
        groups=[]
        visible_ids=set()
        for row in self._catalog["files"]:
            if row["saved"]!=self._saved_only:continue
            if any(term not in (row["source"]+" "+row["detail"]).casefold().replace("−","-") for term in terms):continue
            if not self._saved_only:
                if self._workflow=="pl":
                    if self._type!="All" and row["kind"]!=self._type:continue
                    if self._status_filter!="all" and row["status"]!={"unprocessed":"new"}.get(self._status_filter,self._status_filter):continue
                else:
                    if row["background"] and not self._include_backgrounds:continue
                    if self._only_new and row["status"]!="new":continue
            visible_ids.add(row["id"])
        for group in self._catalog["groups"]:
            if any(path in visible_ids for path in group["members"]):groups.append(group)
        if self._reopen_group_pending:
            target=next((group for group in groups if self._focused in group["members"] and self._focused in visible_ids),None)
            if target is not None:
                self._selected_group=target["id"];self._reopen_group_pending=False
            elif self._catalog_validated:self._reopen_group_pending=False
        self._groups.replace(groups)
        active=next((group for group in groups if group["id"]==self._selected_group),None)
        members=set(active["members"]) if active and not self._saved_only else None
        selected=set(self._selected)
        self._files.replace([{**row,"selected":row["id"] in selected,"focused":row["id"]==self._focused} for row in self._catalog["files"]
            if row["id"] in visible_ids and (members is None or row["id"] in members)])
        self._refresh_selection()

    def _refresh_selection(self):
        selected=set(self._selected)
        self._files.replace([row if (row["selected"]==(row["id"] in selected) and row["focused"]==(row["id"]==self._focused))
            else {**row,"selected":row["id"] in selected,"focused":row["id"]==self._focused} for row in self._files.rows])
        self._chosen.replace([{**self._lookup.get(path,dict(id=path,name=Path(path).name,source=path,status="external",detail="Selected external file")),"selected":True} for path in self._selected])
        self.snapshotChanged.emit()

    @Slot(str,"QVariant")
    def setParameter(self,key,value):
        self._reopen_group_pending=False
        if key=="query":self._query=str(value)
        elif key=="type":self._type=str(value);self.settings.setValue("quick/catalog/plType",self._type)
        elif key=="statusFilter":self._status_filter=str(value);self.settings.setValue("pl/source_filter",self._status_filter)
        elif key=="selectedGroup":self._selected_group=str(value)
        elif key=="focused":self._focused=str(value);self._refresh_selection();return
        elif key=="savedOnly":self._saved_only=bool(value);self._selected_group=""
        elif key=="includeBackgrounds":self._include_backgrounds=bool(value)
        elif key=="onlyNew":self._only_new=bool(value);self.settings.setValue("drr/source_unprocessed_only",self._only_new)
        elif key=="includeAll":self._include_all=bool(value);self.refresh()
        else:return
        self._refresh_models()

    @Slot()
    def cancelReopenPosition(self):
        self._reopen_group_pending=False

    @Slot(str,bool)
    def selectFile(self,path,selected):
        self._reopen_group_pending=False
        path=str(Path(path).resolve())
        if selected:
            if self._workflow=="pl" or self._saved_only:self._selected=[path]
            elif path not in self._selected:self._selected.append(path)
            self._focused=path
            self._explicit_selected.add(path)
        elif path in self._selected:self._selected.remove(path);self._explicit_selected.discard(path)
        self._refresh_selection()

    @Slot("QVariantList")
    def addExternal(self,urls):
        paths=[]
        for value in urls:
            url=value if isinstance(value,QUrl) else QUrl(str(value))
            paths.append(str(Path(url.toLocalFile() if url.isLocalFile() else str(value)).resolve()))
        if not paths:return
        self._selected=paths[-1:] if self._workflow=="pl" or self._saved_only else list(dict.fromkeys([*self._selected,*paths]))
        self._focused=paths[-1];self._explicit_selected.update(paths)
        self._refresh_selection()

    @Slot(str)
    def invoke(self,action):
        if action=="refresh":self.refresh();return
        if action=="browse":self.dialogRequested.emit("catalog",self._workflow=="drr" and not self._saved_only);return
        if action=="clear":self._selected=[];self._explicit_selected.clear()
        elif action=="add-group":
            if not self.snapshot["canAddGroup"]:return
            group=next((group for group in self._catalog["groups"] if group["id"]==self._selected_group),None)
            if group:
                self._selected=list(dict.fromkeys([*self._selected,*group["members"]]))
                if group["members"] and self._focused not in group["members"]:self._focused=group["members"][0]
        elif action=="compatible" and self._selected:
            if not self.snapshot["canAddGroup"]:return
            from core.drr_sources import compatible_drr_repeats
            root=Path(self._folder)
            source=next((source.source for source in self._catalog["sources"] if str((root/source.source).resolve())==self._selected[0]),"")
            self._selected=list(dict.fromkeys([*self._selected,*(str((root/path).resolve()) for path in compatible_drr_repeats(self._catalog["sources"],source))]))
        elif action=="accept":
            if self.snapshot["canAccept"]:
                paths=tuple(self._selected)
                scope=(self._workflow,self._folder,all(Path(path).suffix.casefold()==".dat" for path in paths))
                target=self._focused if self._focused in paths else paths[-1]
                self._pending_load_position=(scope,paths,target)
                self._dialog_open=False
                self.setForegroundBusy(True)
                self.selectionAccepted.emit(self._workflow,list(self._selected))
            return
        elif action=="cancel":
            self._dialog_open=False
        self._refresh_models()

    def next_unprocessed(self,completed):
        from core.pl_sources import classify_pl_source
        completed=Path(completed).resolve();kind=classify_pl_source(completed)
        if completed.suffix.casefold()==".dat":return None
        return next((row["id"] for row in self._catalog["files"] if row["kind"]==kind and row["status"]=="new" and not row["saved"] and Path(row["id"])!=completed),None)

    @Slot()
    def close(self):
        if self._closed:return
        self._closed=True;self._generation+=1;self._busy=False;self._refresh_pending=False
        self._prepare_timer.stop();self._prepare_pending=False
        self._stop_watches()
        if self._cancel:self._cancel.cancel()
        # A cancelled inventory can still be unwinding an atomic cache write.
        # Wait for its handles to close before the owner releases its folder.
        self._executor.shutdown(wait=True,cancel_futures=True)
