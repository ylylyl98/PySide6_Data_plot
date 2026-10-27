"""Measure native file-picker readiness and GUI stalls using isolated caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, sleep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow", choices=("pl", "drr"), default="drr")
    parser.add_argument("--baseline", type=Path, help="Use a saved baseline core/source_catalog snapshot")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    if args.baseline:
        sys.path.insert(0, str(args.baseline.resolve()))
    with TemporaryDirectory(prefix="quick-catalog-profile-") as temporary:
        os.environ.update(QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
            QT_SCALE_FACTOR="1", MPLBACKEND="agg", MPLCONFIGDIR=temporary,
            LOCALAPPDATA=temporary, XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary,
            QML_DISK_CACHE_PATH=temporary)
        from PySide6.QtCore import QEvent, QSettings, QTimer, QUrl
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine
        from PySide6.QtQuickControls2 import QQuickStyle
        from ui_qt.quick.appearance import quick_colors
        if args.baseline:
            import importlib.util
            spec = importlib.util.spec_from_file_location("ui_qt.quick.source_catalog",
                args.baseline.resolve()/"source_catalog.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        from ui_qt.quick.source_catalog import SourceCatalogBridge

        QQuickStyle.setStyle("Basic")
        app = QGuiApplication([])
        calls = []

        class MeasuredCatalog(SourceCatalogBridge):
            def _refresh_models(self):
                start = perf_counter()
                super()._refresh_models()
                calls.append(("models", (perf_counter()-start)*1000))

            def _sync_watches(self):
                start = perf_counter()
                super()._sync_watches()
                calls.append(("watches", (perf_counter()-start)*1000))

        settings = QSettings(str(Path(temporary)/"profile.ini"), QSettings.IniFormat)
        settings.setValue("drr/source_unprocessed_only", False)
        settings.setValue("quick/catalog/plType", "All")
        bridge = MeasuredCatalog(settings=settings)
        engine = QQmlApplicationEngine()
        warnings = []
        engine.warnings.connect(lambda values: warnings.extend(value.toString() for value in values))
        engine.setInitialProperties(dict(bridge=bridge, colors=quick_colors("light")))
        qml = repo/"ui_qt/quick/qml/profile-catalog.qml"
        engine.loadData(b'''import QtQuick
import QtQuick.Controls.Basic as Basic
Basic.ApplicationWindow {
    id: window
    required property var bridge
    required property var colors
    width: 1280; height: 900; visible: true
    SourceCatalog { id: picker; objectName: "profile_picker"; bridge: window.bridge; colors: window.colors }
    function showPicker() { picker.open() }
}''', QUrl.fromLocalFile(str(qml)))
        if not engine.rootObjects():
            raise RuntimeError("\n".join(warnings))
        window = engine.rootObjects()[0]
        last = perf_counter()
        gaps = []
        clock = QTimer()
        clock.setInterval(10)

        def tick():
            nonlocal last
            now = perf_counter()
            gaps.append((now-last)*1000)
            last = now

        clock.timeout.connect(tick)
        clock.start()
        reports = []
        active = None

        def changed():
            if active is None or "start" not in active:
                return
            elapsed = (perf_counter()-active["start"])*1000
            if bridge.filesModel.rows and "first_rows_ms" not in active:
                active["first_rows_ms"] = elapsed
            if any(not row["saved"] for row in bridge.filesModel.rows) and "first_sources_ms" not in active:
                active["first_sources_ms"] = elapsed
            if bridge.groupsModel.rows and "first_groups_ms" not in active:
                active["first_groups_ms"] = elapsed

        bridge.snapshotChanged.connect(changed)
        for policy in ("cold", "warm"):
            for _ in range(5):
                app.processEvents()
            gaps.clear()
            calls.clear()
            last = perf_counter()
            active = dict(policy=policy, start=last)
            bridge.open(args.workflow, str(args.folder.resolve()), [], False)
            active["open_call_ms"] = (perf_counter()-last)*1000
            window.showPicker()
            deadline = perf_counter()+240
            while bridge.snapshot["busy"] and perf_counter() < deadline:
                app.processEvents()
                sleep(.001)
            app.processEvents()
            active["ready_ms"] = (perf_counter()-active.pop("start"))*1000
            if bridge.snapshot["busy"] or bridge.snapshot["error"]:
                raise RuntimeError(str(bridge.snapshot))
            active.update(files=len(bridge._catalog["files"]), groups=len(bridge._catalog["groups"]),
                max_gui_gap_ms=max(gaps, default=0), gui_calls=list(calls))
            searches = []
            for query in ("pwn1", "pwn1 9T", "pwn1 9T 730", ""):
                start = perf_counter()
                bridge.setParameter("query", query)
                app.processEvents()
                searches.append((query, (perf_counter()-start)*1000))
            active["search_ms"] = searches
            reports.append(active)
            active = None
            bridge.invoke("cancel")
        clock.stop()
        inventory_digest = hashlib.sha256(json.dumps(bridge._watch_signature,
            sort_keys=True).encode()).hexdigest()
        bridge.close()
        window.setVisible(False)
        engine.deleteLater()
        app.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
        report = dict(folder=str(args.folder.resolve()), workflow=args.workflow,
            inventory_sha256=inventory_digest,
            scope="Fresh process and temporary caches; cold then warm native picker; read-only sources",
            runs=reports, qml_warnings=warnings)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if warnings:
            raise RuntimeError("QML warnings invalidate the picker profile: " + "\n".join(warnings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
