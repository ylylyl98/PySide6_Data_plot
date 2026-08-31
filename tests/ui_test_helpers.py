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
