"""Metadata-only reference selection using the native measurement catalog UI."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Property, Signal, Slot, QUrl

from ui_qt.quick.source_catalog import SourceCatalogBridge


_LABELS = {"bn": "BN", "p": "P", "n": "N", "pn": "P + N", "custom": "Custom"}


class BackgroundSourceCatalogBridge(SourceCatalogBridge):
    """Choose identities; application validation remains in the background worker."""

    selectionCancelled = Signal()
    selectionEdited = Signal("QVariantList")

    def __init__(self, *, settings, parent=None, inline=False):
        super().__init__(settings=settings, parent=parent)
        self._inline_selection = bool(inline)
        self._editing_enabled = True
        self._source_kind = "bn"
        self._role = "background"
        self._show_all_files = False
        self._selection_open = False
        self._selection_touched = False
        self._selection_token = 0
        self._only_new = False
        self._include_backgrounds = True

    @Property("QVariantMap", notify=SourceCatalogBridge.snapshotChanged)
    def snapshot(self):
        state = SourceCatalogBridge.snapshot.fget(self)
        label = self._role if self._role in {"P", "N"} else _LABELS[self._source_kind]
        can_accept = len(self._selected) == 1 if self._role in {"P", "N"} else bool(self._selected)
        can_accept = can_accept or not self._selected and self._selection_touched
        state.update(backgroundSelection=True, selectionTitle=f"Choose {label} references",
                     sourceKindLabel=label, acceptLabel="Use selected", showAllFiles=self._show_all_files,
                     inlineSelection=self._inline_selection,
                     editEnabled=self._selection_open and self._editing_enabled,
                     selectionOpen=self._selection_open, selectionToken=self._selection_token,
                     canAccept=self._selection_open and self._editing_enabled and can_accept,
                     canAddGroup=False,
                     status="Listing reference files…" if self._busy else f"{len(self._files.rows)} files · {len(self._selected)} chosen")
        return state

    def open_selection(self, folder, kind, selected, *, role="background"):
        if kind not in _LABELS or role not in {"background", "P", "N"}:
            raise ValueError("Choose a supported background source kind and role.")
        self._stop_selection()
        self._selection_token += 1
        self._selection_open = True
        self._editing_enabled = True
        self._selection_touched = False
        self._source_kind, self._role = kind, role
        self._query, self._show_all_files, self._include_all = "", False, False
        self._selected_group, self._focused = "", ""
        root = Path(folder).resolve()
        selected = [self._local_path(value, root) for value in selected]
        if self._configure("drr", str(root), selected, False):
            self._watch_active = True
            self.refresh()

    def open_inline_selection(self, folder, kind, selected):
        """Rebind the draft editor, reusing validated same-folder metadata."""
        if not self._inline_selection or kind not in _LABELS:
            raise ValueError("Choose a supported inline background source kind.")
        root = Path(folder).resolve()
        reuse_catalog = self._folder == str(root) and self._catalog_validated
        self._stop_selection()
        self._selection_open, self._editing_enabled = True, True
        self._selection_touched = False
        self._source_kind, self._role = kind, "background"
        self._query, self._show_all_files, self._include_all = "", False, False
        self._selected_group, self._focused = "", ""
        selected = [self._local_path(value, root) for value in selected]
        self._configure("drr", str(root), selected, False)
        self._watch_active = True
        if reuse_catalog:
            # The cancelled scan token belongs to the previous editor session.
            # A lightweight watch check should use the retained inventory.
            self._cancel = None
            self._sync_watches()
            self._watch_timer.start()
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

    def _scan_catalog(self, folder, workflow, *, include_all=False, cancellation=None, **options):
        from core.drr_sources import drr_source_paths, portable_source_name
        from core.drr_measured_bn import is_measured_bn_name
        from core.drr_pn_pairing import detect_pn_role
        from core.load_cancellation import load_scope
        root = Path(folder).resolve()
        with load_scope(cancellation):
            rows = []
            for path in drr_source_paths(root, include_all=include_all):
                if cancellation:
                    cancellation.check()
                if path.suffix.casefold() != ".csv":
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                name = portable_source_name(root, path)
                label = "BN" if is_measured_bn_name(path.name) else detect_pn_role(path.name) or "CSV"
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                relative_parent = str(Path(name).parent).replace("\\", "/")
                rows.append(dict(id=str(path.resolve()), source=name,
                                 name=path.name, kind=label, status="", saved=False,
                                 modified=stat.st_mtime, size=stat.st_size, processedAt="",
                                 background=label != "CSV",
                                 summary=f"{label} · {modified} · {relative_parent}",
                                 detail=f"File modified: {modified}\n{stat.st_size:,} bytes · Filename selection; validate when applying"))
            return dict(files=sorted(rows, key=lambda row: (-row["modified"], row["source"].casefold())),
                        groups=[], sources=())

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
            if not self._show_all_files and not self._matches_kind(row) and row["id"] not in selected:
                continue
            searchable = (row["source"] + " " + row["detail"]).casefold().replace("−", "-")
            if any(term not in searchable for term in terms):
                continue
            rows.append({**row, "selected": row["id"] in selected, "focused": row["id"] == self._focused})
        self._groups.replace([])
        self._files.replace(rows)
        self._refresh_selection()

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
        self._selection_touched = True
        if selected:
            if self._role in {"P", "N"}:
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
            return
        paths = [self._local_path(value, Path(self._folder)) for value in values]
        self._selected = paths[-1:] if self._role in {"P", "N"} else list(dict.fromkeys([*self._selected, *paths]))
        self._explicit_selected.update(paths)
        self._focused, self._selection_touched = paths[-1], True
        self._refresh_models()
        self.selectionEdited.emit(list(self._selected))

    def accept_external(self, token, values):
        if not self._selection_open or not self._editing_enabled or token != self._selection_token:
            return False
        self.addExternal(values)
        return True

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
            self.dialogRequested.emit(f"{namespace}:{self._selection_token}", self._role not in {"P", "N"})
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
