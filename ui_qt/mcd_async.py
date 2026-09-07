"""Small Qt worker used by the MCD catalog browsers."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal

from core.mcd_extract import discover_processed_mcd


class McdScanSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class McdScanWorker(QRunnable):
    def __init__(self, root: str | Path, rebuild_catalog: bool = False) -> None:
        super().__init__()
        self.root = Path(root)
        self.rebuild_catalog = bool(rebuild_catalog)
        self.signals = McdScanSignals()

    def run(self) -> None:
        try:
            records = discover_processed_mcd(
                self.root, rebuild_catalog=self.rebuild_catalog
            )
            self.signals.result.emit((str(self.root.resolve()), records))
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()
