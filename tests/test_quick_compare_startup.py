"""Cold Compare assignment leaves scientific initialization on its worker."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickCompareStartupTests(unittest.TestCase):
    def test_first_assignment_keeps_gui_responsive_and_clear_wins_during_import(self):
        with TemporaryDirectory(prefix="quick-compare-startup-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary,
                XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True,
                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_COMPARE_STARTUP_OK", result.stdout)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import threading
    import time
    from PySide6.QtCore import QSettings, QTimer, QUrl
    from PySide6.QtGui import QGuiApplication
    from ui_qt.quick.compare_bridge import CompareBridge
    app = QGuiApplication([])
    main_thread = threading.get_ident()
    started, release = threading.Event(), threading.Event()
    imports, heartbeats = [], []

    class ImportRecorder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "core.processing":
                imports.append((fullname, threading.get_ident()))
                assert threading.get_ident() != main_thread, "Compare grouping imported science on the GUI"
                started.set()
                assert release.wait(30), "Grouping import was not released"
            elif fullname in {"numpy", "pandas", "scipy"}:
                imports.append((fullname, threading.get_ident()))

    recorder = ImportRecorder()
    sys.meta_path.insert(0, recorder)
    with TemporaryDirectory(prefix="compare-startup-data-") as temporary:
        root = Path(temporary)
        paths = []
        for channel in ("KK", "KKp"):
            path = root / ("sample_PL_" + channel + ".csv")
            path.write_text("Vbg,Vtg,Vbias,600,610,620\n0,0,0,10,15,10\n1,1,0,20,25,20\n", encoding="utf-8")
            paths.append(QUrl.fromLocalFile(str(path)))
        bridge = CompareBridge(settings=QSettings(str(root / "test.ini"), QSettings.IniFormat))
        heartbeat = QTimer()
        heartbeat.setInterval(1)
        heartbeat.timeout.connect(lambda: heartbeats.append(True))
        try:
            assert not {"numpy", "pandas", "scipy", "matplotlib"}.intersection(sys.modules)
            heartbeat.start()
            bridge.openFiles("sources", paths)
            assert started.wait(20), "No grouping worker started"
            deadline = time.monotonic() + 5
            while not heartbeats and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.001)
            assert heartbeats, "GUI heartbeat stopped while grouping imported"
            bridge.invoke("clear-files")
            release.set()
            deadline = time.monotonic() + 60
            while bridge._pending and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.002)
            assert not bridge._pending, bridge.snapshot
            assert not bridge._assignments, "Late import completion replaced an explicit clear"
            assert not bridge.snapshot["canExport"]
            bridge.openFiles("sources", paths)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                app.processEvents()
                if bridge._cubes and not bridge._pending:
                    break
                time.sleep(.002)
            assert set(bridge._cubes) == {"KK", "KKp"}, bridge.snapshot
            assert not bridge.snapshot["error"], bridge.snapshot
            assert all(thread != main_thread for _, thread in imports), imports
        finally:
            release.set()
            heartbeat.stop()
            bridge.close()
            sys.meta_path.remove(recorder)
    print("QUICK_COMPARE_STARTUP_OK")


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        _isolated()
    else:
        unittest.main()
