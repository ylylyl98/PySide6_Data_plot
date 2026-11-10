"""Metadata-only reference selection using the native measurement catalog UI."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from threading import Lock

from PySide6.QtCore import Property, Signal, Slot, QUrl

from ui_qt.quick.source_catalog import SourceCatalogBridge


_LABELS = {"bn": "BN", "p": "P", "n": "N", "pn": "P + N", "custom": "Custom"}


class BackgroundSourceCatalogBridge(SourceCatalogBridge):
    """Choose identities; application validation remains in the background worker."""

    selectionCancelled = Signal()
    selectionEdited = Signal("QVariantList")

    def __init__(self, *, settings, parent=None, inline=False, inventory_kind='drr'):
        super().__init__(settings=settings, parent=parent)
        if inventory_kind not in {'drr', 'megasweep'}:
            raise ValueError('Unsupported background inventory kind.')
        self._inventory_kind = inventory_kind
        self._inline_selection = bool(inline)
        self._editing_enabled = True
        self._source_kind = "bn"
        self._role = "background"
        self._multiple = True
        self._show_all_files = False
        self._selection_open = False
        self._selection_touched = False
        self._selection_token = 0
        self._only_new = False
        self._include_backgrounds = True
        self._bn_header_cache = OrderedDict()
        self._bn_header_lock = Lock()

    def _protect_measured_bn(self):
        return self._inventory_kind == 'drr' and self._source_kind == 'bn' and self._role == 'background'

    def _is_generated_bn(self, value):
        """Inspect only the CSV header; cache by file identity, not filename."""
        path = Path(value)
        if any(part.casefold() == '.drr_bn_calibration' for part in path.parts):
            return True
        if path.suffix.casefold() != '.csv':
            return False
        try:
            stat = path.stat()
            identity = (str(path.resolve()).casefold(), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            with self._bn_header_lock:
                if identity in self._bn_header_cache:
                    self._bn_header_cache.move_to_end(identity)
                    return self._bn_header_cache[identity]
            with path.open('r', encoding='utf-8-sig', errors='replace') as stream:
                generated = 'BNCalibrationVersion' in stream.readline()
        except OSError:
            # An unreadable selection cannot be certified as measured BN.
            return True
        with self._bn_header_lock:
            self._bn_header_cache[identity] = generated
            self._bn_header_cache.move_to_end(identity)
            while len(self._bn_header_cache) > 4096:
                self._bn_header_cache.popitem(last=False)
        return generated

    def _reject_generated_bn(self, paths):
        if self._protect_measured_bn() and any(self._is_generated_bn(path) for path in paths):
            self._error = ('Choose an original measured BN file; calibrated BN belongs to its measurement. '
                           'Use Auto trend to generate or reuse its correction.')
            self._refresh_models()
            return True
        self._error = ''
        return False

    @Property("QVariantMap", notify=SourceCatalogBridge.snapshotChanged)
    def snapshot(self):
        state = SourceCatalogBridge.snapshot.fget(self)
        label = self._role if self._role in {"P", "N"} else _LABELS[self._source_kind]
        can_accept = bool(self._selected) if self._multiple else len(self._selected) == 1
        can_accept = can_accept or not self._selected and self._selection_touched
        hidden_selected = self._protect_measured_bn() and any(self._is_generated_bn(path) for path in self._selected)
        if hidden_selected:
            can_accept = False
        state.update(backgroundSelection=True, selectionTitle=f"Choose {label} references",
                     sourceKindLabel=label, acceptLabel="Use selected", showAllFiles=self._show_all_files,
                     multipleSelection=self._multiple,
                     inlineSelection=self._inline_selection,
                     editEnabled=self._selection_open and self._editing_enabled,
                     selectionOpen=self._selection_open, selectionToken=self._selection_token,
                     canAccept=self._selection_open and self._editing_enabled and can_accept,
                     canAddGroup=False,
                     status=("Listing reference files…" if self._busy else
                         "A calibrated BN is hidden; choose an original measured BN." if hidden_selected else
                         f"{len(self._files.rows)} files · {len(self._selected)} chosen"))
        return state

    def open_selection(self, folder, kind, selected, *, role="background", multiple=False):
        if kind not in _LABELS or role not in {"background", "P", "N"}:
            raise ValueError("Choose a supported background source kind and role.")
        root = Path(folder).resolve()
        reuse_catalog = self._can_reuse_catalog(root)
        self._stop_selection()
        self._selection_token += 1
        self._selection_open = True
        self._editing_enabled = True
        self._selection_touched = False
        self._source_kind, self._role = kind, role
        self._multiple = bool(multiple) or role == "background"
        self._query, self._show_all_files, self._include_all = "", False, False
        self._selected_group, self._focused = "", ""
        selected = [self._local_path(value, root) for value in selected]
        if not self._multiple:
            selected = selected[-1:]
        if self._configure("drr", str(root), selected, False):
            self._resume_selection_catalog(reuse_catalog)

    def open_inline_selection(self, folder, kind, selected):
        """Rebind the draft editor, reusing validated same-folder metadata."""
        if not self._inline_selection or kind not in _LABELS:
            raise ValueError("Choose a supported inline background source kind.")
        root = Path(folder).resolve()
        reuse_catalog = self._can_reuse_catalog(root)
        self._stop_selection()
        self._selection_open, self._editing_enabled = True, True
        self._selection_touched = False
        self._source_kind, self._role = kind, "background"
        self._multiple = True
        self._query, self._show_all_files, self._include_all = "", False, False
        self._selected_group, self._focused = "", ""
        selected = [self._local_path(value, root) for value in selected]
        self._configure("drr", str(root), selected, False)
        self._resume_selection_catalog(reuse_catalog)

    def _can_reuse_catalog(self, root):
        # Older in-memory snapshots have no verified header marker. Re-scan
        # them rather than keeping a permanently empty filtered catalog.
        marked = (self._inventory_kind != 'drr' or
                  all('generatedBN' in row for row in self._catalog['files']))
        return self._folder == str(root) and self._catalog_validated and not self._include_all and marked

    def _resume_selection_catalog(self, reuse_catalog):
        self._watch_active = True
        if reuse_catalog:
            # The cancelled scan token belongs to the previous editor session.
            # A lightweight watch check should use the retained inventory.
            self._cancel = None
            self._sync_watches()
            self._watch_timer.start(0)
        else:
            self.refresh()

    def rebind_inline_selection(self, selected):
        """Reflect parent draft changes without reporting a user edit or scan."""
        if not self._inline_selection or not self._selection_open:
            return
        paths = list(dict.fromkeys(self._local_path(value, Path(self._folder)) for value in selected))
        if paths != self._selected:
            self._selected, self._explicit_selected = paths, set(paths)
            self._refresh_models()

    def set_edit_enabled(self, enabled):
        enabled = bool(enabled)
        if self._editing_enabled != enabled:
            self._editing_enabled = enabled
            if not enabled:
                self._selection_token += 1
            self.snapshotChanged.emit()

    @staticmethod
    def _local_path(value, root):
        url = value if isinstance(value, QUrl) else QUrl(str(value))
        path = Path(url.toLocalFile() if url.isLocalFile() else str(value))
        return str((path if path.is_absolute() else root / path).resolve())

    def _source_paths(self, root, include_all):
        if self._inventory_kind == 'megasweep':
            from core.megasweep_catalog import source_paths
            return source_paths(root)
        from core.drr_sources import drr_source_paths
        return drr_source_paths(root, include_all=include_all)

    def _scan_catalog(self, folder, workflow, *, include_all=False, cancellation=None, **options):
        from core.drr_measured_bn import is_measured_bn_name
        from core.drr_pn_pairing import detect_pn_role
        from core.load_cancellation import load_scope
        root = Path(folder).resolve()
        with load_scope(cancellation):
            rows = []
            for path in self._source_paths(root, include_all):
                if cancellation:
                    cancellation.check()
                if path.suffix.casefold() != ".csv":
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                # The inventory already supplies absolute paths below a
                # resolved root. Resolve each file once for aliases, then use
                # that identity for both its id and its portable name.
                identity = path.resolve()
                try:
                    name = identity.relative_to(root).as_posix()
                except ValueError:
                    name = str(identity)
                label = "BN" if is_measured_bn_name(path.name) else detect_pn_role(path.name) or "CSV"
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                relative_parent = str(Path(name).parent).replace("\\", "/")
                rows.append(dict(id=str(identity), source=name,
                                 name=path.name, kind=label, status="", saved=False,
                                 generatedBN=self._is_generated_bn(identity) if self._inventory_kind == 'drr' else False,
                                 modified=stat.st_mtime, size=stat.st_size, processedAt="",
                                 background=label != "CSV",
                                 summary=f"{label} · {modified} · {relative_parent}",
                                 detail=f"File modified: {modified}\n{stat.st_size:,} bytes · Filename selection; validate when applying"))
            return dict(files=sorted(rows, key=lambda row: (-row["modified"], row["source"].casefold())),
                        groups=[], sources=())

    def _source_inventory(self, folder, workflow, include_all=False):
        """Reference choices depend on CSV metadata, not saved DRR history."""
        from core.load_cancellation import check_load_cancelled
        root = Path(folder).resolve()
        rows = []
        for path in self._source_paths(root, include_all):
            check_load_cancelled()
            if path.suffix.casefold() != ".csv":
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            rows.append([path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns])
        return sorted(rows)

    def _matches_kind(self, row):
        kind = self._role.lower() if self._role in {"P", "N"} else self._source_kind
        return kind == "custom" or row["kind"] in ({"P", "N"} if kind == "pn" else {_LABELS[kind]})

    def _refresh_models(self):
        # Filename hints only narrow the list. They never relabel the draft or
        # certify acquisition compatibility, gates, or an eligible background.
        self._indexed_catalog = self._catalog
        self._lookup = {row["id"]: row for row in self._catalog["files"]}
        self._counts = {status: 0 for status in ("new", "processed", "unknown")}
        terms = self._query.casefold().replace("−", "-").split()
        selected = set(self._selected)
        rows = []
        for row in self._catalog["files"]:
            # Old optimistic caches predate this marker; hide unknown entries
            # until the worker rechecks their headers, even when selected.
            if self._protect_measured_bn() and row.get('generatedBN') is not False:
                continue
            if not self._show_all_files and not self._matches_kind(row) and row["id"] not in selected:
                continue
            searchable = (row["source"] + " " + row["detail"]).casefold().replace("−", "-")
            if any(term not in searchable for term in terms):
                continue
            rows.append({**row, "selected": row["id"] in selected, "focused": row["id"] == self._focused})
        self._groups.replace([])
        self._files.replace(rows)
        self._refresh_selection()

    def _refresh_selection(self):
        """Keep previous draft identities, but never show a managed BN as raw."""
        selected = set(self._selected)
        self._files.replace([row if (row['selected'] == (row['id'] in selected)
            and row['focused'] == (row['id'] == self._focused)) else
            {**row, 'selected': row['id'] in selected, 'focused': row['id'] == self._focused}
            for row in self._files.rows])
        visible = [path for path in self._selected
                   if not self._protect_measured_bn() or not self._is_generated_bn(path)]
        self._chosen.replace([{**self._lookup.get(path, dict(id=path, name=Path(path).name,
            source=path, status='external', detail='Selected external file')), 'selected': True}
            for path in visible])
        self.snapshotChanged.emit()

    @Slot(str, "QVariant")
    def setParameter(self, key, value):
        if not self._selection_open:
            return
        if key == "showAllFiles":
            self._show_all_files = bool(value)
            self._refresh_models()
        elif key in {"query", "focused", "includeAll"}:
            super().setParameter(key, value)

    @Slot(str, bool)
    def selectFile(self, path, selected):
        if not self._selection_open or not self._editing_enabled:
            return
        path = self._local_path(path, Path(self._folder))
        if selected and self._reject_generated_bn((path,)):
            return
        self._selection_touched = True
        if selected:
            if not self._multiple:
                self._selected = [path]
                self._explicit_selected = {path}
            elif path not in self._selected:
                self._selected.append(path)
                self._explicit_selected.add(path)
            self._focused = path
        elif path in self._selected:
            self._selected.remove(path)
            self._explicit_selected.discard(path)
        self._refresh_models()
        self.selectionEdited.emit(list(self._selected))

    @Slot("QVariantList")
    def addExternal(self, values):
        if not self._selection_open or not self._editing_enabled or not values:
            return False
        paths = [self._local_path(value, Path(self._folder)) for value in values]
        if self._reject_generated_bn(paths):
            return False
        self._selected = list(dict.fromkeys([*self._selected, *paths])) if self._multiple else paths[-1:]
        self._explicit_selected = set(self._selected)
        self._focused, self._selection_touched = paths[-1], True
        self._refresh_models()
        self.selectionEdited.emit(list(self._selected))
        return True

    def accept_external(self, token, values):
        if not self._selection_open or not self._editing_enabled or token != self._selection_token:
            return False
        return self.addExternal(values) is True

    def _stop_selection(self):
        self._selection_open = False
        self._selection_token += 1
        if self._cancel:
            self._cancel.cancel()
        self._generation += 1
        self._busy = False
        self._stop_watches()

    def _watch_refresh(self):
        if self._selection_open:
            super()._watch_refresh()

    @Slot()
    @Slot(bool)
    def refresh(self, force=False):
        if self._selection_open:
            super().refresh(force)

    @Slot(str)
    def invoke(self, action):
        if not self._selection_open:
            return
        if not self._editing_enabled and action in {"browse", "clear", "accept"}:
            return
        if action == "browse":
            namespace = "inline-catalog" if self._inline_selection else "catalog"
            self.dialogRequested.emit(f"{namespace}:{self._selection_token}", self._multiple)
        elif action == "clear":
            self._selection_touched = True
            self._selected, self._explicit_selected = [], set()
            self._refresh_models()
            self.selectionEdited.emit([])
        elif action == "accept":
            if self.snapshot["canAccept"]:
                selected = list(self._selected)
                self._stop_selection()
                self.snapshotChanged.emit()
                self.selectionAccepted.emit("drr", selected)
        elif action == "cancel":
            self._stop_selection()
            self.snapshotChanged.emit()
            self.selectionCancelled.emit()
        elif action == "refresh":
            self.refresh()

    @Slot()
    def close(self):
        self._stop_selection()
        super().close()


class MegasweepBackgroundSourceCatalogBridge(BackgroundSourceCatalogBridge):
    """Recursive Megasweep inventory with a separate optimistic display cache."""

    def __init__(self, *, settings, parent=None, inline=False):
        super().__init__(settings=settings, parent=parent, inline=inline, inventory_kind='megasweep')
