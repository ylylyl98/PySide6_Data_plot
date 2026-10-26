"""Native Data Maintenance controls bound to the latest successful export."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot


class DataMaintenanceBridge(QObject):
    snapshotChanged = Signal()
    statusMessage = Signal(str)
    logMessage = Signal(str)
    catalogRefreshRequested = Signal(str)
    _completed = Signal(int, str, object, str)

    def __init__(self, *, parent=None):
        super().__init__(parent)
        self._closed = self._busy = self._cleanup = False
        self._receipt, self._context = None, None
        self._generation = 0
        self._status, self._error = "Save first to enable moving exported source files.", ""
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dptk-data-maintenance")
        self._publish_timer = QTimer(self)
        self._publish_timer.setSingleShot(True)
        self._publish_timer.timeout.connect(self.snapshotChanged)
        self._completed.connect(self._finish)

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        from ui_qt.quick.data_maintenance import movable_sources
        sources = movable_sources(self._receipt) if self._receipt else ()
        return dict(cleanupVerifiedSources=self._cleanup, busy=self._busy, canMove=bool(sources) and not self._busy,
            sourceCount=len(sources), sourceNames=[item.path for item in sources], status=self._status, error=self._error,
            folder=self._receipt.request.folder if self._receipt else "", exportedWorkflow=self._receipt.request.workflow if self._receipt else "")

    def _publish(self):
        if not self._closed and not self._publish_timer.isActive():
            self._publish_timer.start(0)

    def _message(self, message, *, error=False):
        self._status, self._error = message, message if error else ""
        self.statusMessage.emit(message)
        self.logMessage.emit(message)
        self._publish()

    @Slot(bool)
    def setCleanupVerifiedSources(self, value):
        if self._closed:
            return
        self._cleanup = bool(value)
        self._publish()

    def setContext(self, context):
        """Invalidate when folder/source/processing generation changes."""
        if not self._closed and context != self._context:
            self._context = context
            self.invalidate()

    @Slot()
    def invalidate(self):
        if self._closed:
            return
        self._receipt = None
        self._message("Save first to enable moving exported source files.")

    def _start(self, kind, operation):
        self._generation += 1
        token = self._generation
        self._busy = True
        future = self._executor.submit(operation)
        def done(task):
            try:
                result, error = task.result(), ""
            except Exception as exception:
                result, error = None, str(exception).splitlines()[0]
            if not self._closed:
                self._completed.emit(token, kind, result, error)
        future.add_done_callback(done)
        self._publish()

    def acceptExport(self, result, context=None):
        from ui_qt.quick.data_maintenance import ExportReceipt, MaintainedExport, validate_receipt
        receipt = result.receipt if isinstance(result, MaintainedExport) else result
        if self._closed or self._busy or not isinstance(receipt, ExportReceipt) or context is not None and context != self._context:
            return
        self._receipt = None
        captured_context = self._context
        self._start("accept", lambda: (receipt, captured_context, validate_receipt(receipt)))

    @Slot(int, str, object, str)
    def _finish(self, token, kind, result, error):
        if self._closed or token != self._generation:
            return
        self._busy = False
        if error:
            self._receipt = None
            self._message(error, error=True)
        elif kind == "accept":
            receipt, context, valid = result
            if context != self._context:
                self._publish()
                return
            if not valid:
                self._message("Source/export versions changed after saving; moving sources is disabled.", error=True)
                return
            self._receipt = receipt
            self._message(f"Export verified. {len(receipt.cleaned)} verified temporary source copy(s) cleaned.")
            self.catalogRefreshRequested.emit(receipt.request.folder)
        elif kind == "move":
            folder, moved = result
            self._receipt = None
            self._message(f"Moved {moved} source CSV file(s) to 'Initial data after processing'.")
            self.catalogRefreshRequested.emit(folder)
        self._publish()

    @Slot(str)
    def invoke(self, action):
        if self._closed or self._busy:
            return
        if action == "move" and self._receipt:
            from ui_qt.quick.data_maintenance import move_exported_sources
            receipt = self._receipt
            self._receipt = None
            self._message("Verifying exported source versions before moving…")
            self._start("move", lambda: (receipt.request.folder, move_exported_sources(receipt)))
        elif action == "invalidate":
            self.invalidate()

    @Slot()
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._publish_timer.stop()
        self._executor.shutdown(wait=True, cancel_futures=True)


__all__ = ["DataMaintenanceBridge"]
