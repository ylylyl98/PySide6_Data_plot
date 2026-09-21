"""Deliver background matching results on the GUI thread."""
from threading import Event
from PySide6.QtCore import QObject, Slot
from core.drr_auto_external import resolve_auto_external
from ui_qt.common import Worker


class ExternalMatchRequest(QObject):
    def __init__(self, owner, controller):
        super().__init__(owner)
        self.controller = controller
        self.folder = owner.current_folder
        self.selected = tuple(owner.drr_selected_files)
        self.cancelled = Event()
        self.worker = Worker(resolve_auto_external, self.folder,
                             tuple(owner.drr_available_sources), self.selected,
                             cancelled=self.cancelled)
        self.worker.signals.result.connect(self.complete)
        self.worker.signals.error.connect(self.failed)
        self.worker.signals.finished.connect(self.deleteLater)

    def current(self):
        c = self.controller
        return (not self.cancelled.is_set()
                and not c._is_closing
                and getattr(c, '_drr_external_request', None) is self
                and c.current_folder == self.folder
                and tuple(c.drr_selected_files) == self.selected
                and c.drr_baseline_combo.currentText() == 'External'
                and not c.drr_baseline_files_manual)

    @Slot(object)
    def complete(self, result):
        if self.current():
            self.controller._finish_auto_external(result)
        elif getattr(self.controller, '_drr_external_request', None) is self:
            self.controller._drr_external_request = None

    @Slot(str)
    def failed(self, message):
        if self.current():
            from core.drr_sources import DrrBackgroundResolution
            self.controller._finish_auto_external(DrrBackgroundResolution(
                reason='Background matching failed: ' + message.splitlines()[0]))
        elif getattr(self.controller, '_drr_external_request', None) is self:
            self.controller._drr_external_request = None
