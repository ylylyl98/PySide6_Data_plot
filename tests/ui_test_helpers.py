from __future__ import annotations

from PySide6.QtTest import QTest


def wait_for_file_catalog(window, timeout_ms: int = 3000) -> None:
    """Pump Qt until MainWindow's asynchronous file catalog is ready."""
    remaining = max(1, int(timeout_ms // 10))
    for _ in range(remaining):
        QTest.qWait(10)
        if not getattr(window, "_file_refresh_running", False):
            return
    raise AssertionError("Timed out waiting for the asynchronous file catalog")


def dispose_owned_window(window) -> None:
    """Close and destroy a window explicitly owned by the calling test.

    processEvents alone does not deliver DeferredDelete outside Qt's event
    loop. Flush it so subsequent tests do not retain this window's widgets.
    """
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import isValid

    if isValid(window):
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QCoreApplication.processEvents()
