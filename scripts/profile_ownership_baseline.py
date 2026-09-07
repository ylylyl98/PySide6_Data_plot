"""Bounded, non-mutating Qt ownership probe for MainWindow lifecycle diagnosis.

This script is diagnostic-only.  It uses an isolated QApplication/QSettings
process, disables update/file-restore side effects, and writes only scalar
snapshots under the requested output directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cpp_identity(obj: object) -> int | None:
    try:
        import shiboken6

        ptr = shiboken6.getCppPointer(obj)
        return int(ptr[0]) if ptr else None
    except Exception:
        return None


def _safe_text(value: object) -> str:
    try:
        return str(value)
    except Exception:
        return "<unavailable>"


def _widget_record(widget: object, *, first_seen: dict[int, int], cycle: int, depth: int = 0) -> dict[str, Any]:
    cpp = _cpp_identity(widget)
    key = cpp if cpp is not None else id(widget)
    if key not in first_seen:
        first_seen[key] = cycle
    parent = None
    try:
        parent = widget.parent()
    except Exception:
        pass
    record: dict[str, Any] = {
        "class": type(widget).__name__,
        "object_name": _safe_text(widget.objectName()),
        "identity": int(id(widget)),
        "cpp_identity": cpp,
        "first_seen_cycle": first_seen[key],
        "creation_cycle": cycle,
        "visible": bool(widget.isVisible()),
        "enabled": bool(widget.isEnabled()),
        "parent_class": type(parent).__name__ if parent is not None else None,
        "parent_object_name": _safe_text(parent.objectName()) if parent is not None else None,
        "parent_cpp_identity": _cpp_identity(parent) if parent is not None else None,
        "depth": depth,
    }
    try:
        record["window_title"] = _safe_text(widget.windowTitle())
    except Exception:
        record["window_title"] = None
    try:
        text_method = getattr(widget, "text", None)
        record["text"] = _safe_text(text_method()) if callable(text_method) else None
    except Exception:
        record["text"] = None
    return record


def _owner_attrs(window: object, widget_records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return only attribute names, never attribute values or Qt wrappers."""
    by_id = {r["identity"]: r for r in widget_records}
    owners: dict[str, list[str]] = {}
    try:
        attrs = vars(window)
    except Exception:
        attrs = {}
    for name, value in list(attrs.items()):
        # Avoid shiboken calls on arbitrary Python attributes: some wrappers
        # expose native identity helpers that can abort the process when given
        # non-QObject values.  Direct QWidget wrapper identity is sufficient
        # to identify MainWindow-owned attributes.
        try:
            from PySide6.QtWidgets import QWidget

            if not isinstance(value, QWidget):
                continue
        except Exception:
            continue
        target = by_id.get(id(value))
        if target is not None:
            owners.setdefault(str(target["cpp_identity"] or target["identity"]), []).append(str(name))
    return owners


def _descendant_records(root: object, *, first_seen: dict[int, int], cycle: int, max_depth: int = 2) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(widget: object, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = list(widget.children())
        except Exception:
            children = []
        for child in children:
            # QObject children include timers, layouts, and helpers.  Keep
            # QWidget descendants plus direct QObject helper summaries.
            try:
                is_widget = hasattr(child, "isVisible") and hasattr(child, "objectName")
            except Exception:
                is_widget = False
            if not is_widget:
                continue
            records.append(_widget_record(child, first_seen=first_seen, cycle=cycle, depth=depth))
            visit(child, depth + 1)

    visit(root, 1)
    return records


def _snapshot(label: str, app: object, *, cycle: int, first_seen: dict[int, int], window: object | None = None) -> dict[str, Any]:
    from PySide6.QtCore import QThreadPool, QTimer
    from PySide6.QtWidgets import QMainWindow

    widgets = list(app.allWidgets())
    top_levels = list(app.topLevelWidgets())
    records = [_widget_record(w, first_seen=first_seen, cycle=cycle) for w in widgets]
    top_records = [_widget_record(w, first_seen=first_seen, cycle=cycle) for w in top_levels]
    descendant_groups = []
    if window is not None:
        root_record = _widget_record(window, first_seen=first_seen, cycle=cycle)
        descendants = _descendant_records(window, first_seen=first_seen, cycle=cycle)
        descendant_groups.append({"root": root_record, "descendants": descendants})
    active_timers = 0
    for widget in widgets:
        try:
            active_timers += sum(1 for timer in widget.findChildren(QTimer) if timer.isActive())
        except Exception:
            pass
    main_windows = [w for w in widgets if isinstance(w, QMainWindow)]
    histogram = dict(sorted(Counter(r["class"] for r in records).items()))
    owners = _owner_attrs(window, records) if window is not None else {}
    all_widget_records = records if label.startswith("baseline") else []
    snapshot = {
        "label": label,
        "cycle": cycle,
        "timestamp": time.time(),
        "all_widgets": len(widgets),
        "top_level_widgets": len(top_levels),
        "visible_top_level_widgets": sum(bool(w.isVisible()) for w in top_levels),
        "main_window_count": len(main_windows),
        "main_window_identities": [_cpp_identity(w) or id(w) for w in main_windows],
        "widget_class_histogram": histogram,
        "active_timer_count": active_timers,
        "threadpool_active_count": int(QThreadPool.globalInstance().activeThreadCount()),
        "top_levels": top_records,
        "all_widget_records": all_widget_records,
        "descendant_groups": descendant_groups,
        "direct_window_attribute_owners": owners,
    }
    # Ensure no wrapper list survives the diagnostic boundary.  The records
    # retained in ``snapshot`` are scalar dictionaries only; never retain the
    # Qt wrapper lists themselves.
    main_windows.clear(); top_levels.clear(); widgets.clear()
    if not label.startswith("baseline"):
        records.clear()
    return snapshot


def _force_ini_qsettings(base_class: object) -> type:
    class IsolatedQSettings(base_class):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            if len(args) >= 2 and all(isinstance(value, str) for value in args[:2]):
                super().__init__(base_class.IniFormat, base_class.UserScope, args[0], args[1])
            else:
                super().__init__(*args, **kwargs)

    return IsolatedQSettings


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".venv", "__pycache__", "build", "dist"} for part in path.parts):
            continue
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def run(output_dir: Path, cycles: int = 5) -> dict[str, Any]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_SCALE_FACTOR", "1")
    mpl_cache = tempfile.TemporaryDirectory(prefix="dptk-ownership-mpl-")
    settings_dir = tempfile.TemporaryDirectory(prefix="dptk-ownership-settings-")
    os.environ["MPLCONFIGDIR"] = mpl_cache.name
    from PySide6.QtCore import QCoreApplication, QEvent, QSettings
    from PySide6.QtWidgets import QApplication

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, settings_dir.name)
    settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "DPTK", "PySide6_Data_Plot")
    settings.setValue("updates/check_automatically", False)
    settings.sync()
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    from ui_qt.theme import install_theme

    install_theme(app, mode="light")
    import ui_qt.main_window as main_window_module
    from ui_qt.main_window import MainWindow
    original_settings = main_window_module.QSettings
    original_restore = MainWindow._restore_last_folder
    original_schedule = MainWindow._schedule_automatic_update_check
    main_window_module.QSettings = _force_ini_qsettings(original_settings)
    MainWindow._restore_last_folder = lambda self: None
    MainWindow._schedule_automatic_update_check = lambda self: None
    first_seen: dict[int, int] = {}
    snapshots = [_snapshot("baseline0", app, cycle=0, first_seen=first_seen)]
    construction_seconds: list[float] = []
    try:
        for cycle in range(1, cycles + 1):
            started = time.perf_counter()
            window = MainWindow()
            construction_seconds.append(time.perf_counter() - started)
            snapshots.append(_snapshot(f"cycle{cycle}_after_construct", app, cycle=cycle, first_seen=first_seen, window=window))
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
            del window
            gc.collect()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
            snapshots.append(_snapshot(f"baseline{cycle}", app, cycle=cycle, first_seen=first_seen))
    finally:
        main_window_module.QSettings = original_settings
        MainWindow._restore_last_folder = original_restore
        MainWindow._schedule_automatic_update_check = original_schedule
        app.processEvents()
        mpl_cache.cleanup()
        settings_dir.cleanup()
    return {
        "schema_version": 1,
        "python": os.sys.executable,
        "cycles": cycles,
        "source_hashes": _source_hashes(),
        "construction_seconds": construction_seconds,
        "snapshots": snapshots,
        "stable_residual_counts": [
            {
                "baseline": s["label"],
                "all_widgets": s["all_widgets"],
                "top_level_widgets": s["top_level_widgets"],
                "main_window_count": s["main_window_count"],
                "active_timer_count": s["active_timer_count"],
                "classes": s["widget_class_histogram"],
            }
            for s in snapshots
            if s["label"].startswith("baseline")
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run(args.output_dir, cycles=args.cycles)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "report.json"), "baselines": report["stable_residual_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
