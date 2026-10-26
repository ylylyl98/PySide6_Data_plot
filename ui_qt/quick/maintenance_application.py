"""Main-window routing for explicit, version-bound source maintenance."""
from __future__ import annotations


class MaintenanceApplicationMixin:
    def _maintenance_init(self):
        self._maintenance = None
        self._maintenance_export_contexts = {}

    def _maintenance_context(self):
        child = self._workflow_bridges.get(self._workflow)
        if self._workflow == "cmp" and child is not None:
            return ("cmp", child.maintenanceContext)
        ignored = {"gate", "heatmapFraction", "spectrumVisible", "sideBySide"}
        values = tuple((key, repr(value)) for key, value in sorted(self._values.items()) if key not in ignored)
        return (self._workflow, self._folder, self._sources, id(self._cube),
                id(self._drr_products) if self._workflow == "drr" else None, values, repr(self._scales))

    def _sync_maintenance_context(self):
        if self._maintenance is not None and not self._closed:
            self._maintenance.setContext(self._maintenance_context())

    def _ensure_maintenance(self):
        if self._maintenance is None:
            from ui_qt.quick.data_maintenance_bridge import DataMaintenanceBridge
            self._maintenance = DataMaintenanceBridge(parent=self)
            self._maintenance.snapshotChanged.connect(self._publish)
            self._maintenance.statusMessage.connect(self._message)
            self._maintenance.catalogRefreshRequested.connect(self._maintenance_refresh_catalog)
            self._sync_maintenance_context()
        return self._maintenance

    def _maintenance_snapshot(self):
        state = self._maintenance.snapshot if self._maintenance is not None else {}
        supported = self._workflow in {"pl", "drr", "cmp"}
        return dict(cleanVerifiedSources=state.get("cleanupVerifiedSources", False),
                    canMoveExportedSources=supported and state.get("canMove", False),
                    maintenanceBusy=state.get("busy", False), maintenanceSupported=supported)

    def _set_cleanup(self, enabled):
        self._ensure_maintenance().setCleanupVerifiedSources(enabled)
        compare = self._workflow_bridges.get("cmp")
        if compare is not None:
            compare.setMaintenanceCleanup(enabled)

    def _maintenance_refresh_catalog(self, folder):
        # Each workflow preserves its active file selection when refreshing.
        catalog = getattr(self, "_source_catalog", None)
        if catalog is not None:
            catalog.refreshFolder(folder)
        compare = self._workflow_bridges.get("cmp")
        if compare is not None and compare.snapshot.get("sourcePath") == folder:
            compare.invoke("refresh-catalog")

    def _accept_compare_export(self, result, context):
        manager = self._ensure_maintenance()
        self._sync_maintenance_context()
        manager.acceptExport(result, context=("cmp", context))

    def _submit_export(self, operation):
        from ui_qt.quick.data_maintenance import run_export_with_maintenance
        manager = self._ensure_maintenance()
        manager.invalidate()
        self._sync_maintenance_context()
        context = self._maintenance_context()
        sequence = self._sequence + 1
        self._maintenance_export_contexts[sequence] = context
        folder, sources, workflow = self._folder, self._sources, self._workflow
        cleanup = manager.snapshot["cleanupVerifiedSources"]
        expected = self._pl_source_hashes() if workflow == "pl" and hasattr(self, "_pl_source_hashes") else None
        self._submit("export", lambda: run_export_with_maintenance(operation, folder=folder, sources=sources,
            workflow=workflow, cleanup_verified=cleanup, expected_hashes=expected,
            cancelled=lambda: self._closed or self._sequence != sequence))

    def _unwrap_export(self, sequence, result):
        from ui_qt.quick.data_maintenance import MaintainedExport
        context = self._maintenance_export_contexts.pop(sequence, None)
        if isinstance(result, MaintainedExport):
            self._sync_maintenance_context()
            self._ensure_maintenance().acceptExport(result, context=context)
            return result.result
        return result

    def _close_maintenance(self):
        if self._maintenance is not None:
            self._maintenance.close()
