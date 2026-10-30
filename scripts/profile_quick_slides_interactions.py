"""Measure large synthetic Slides catalogs without reading images or user INIs."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=8000)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtGui import QGuiApplication
    from core.presentation import PlotImage
    from ui_qt.quick.presentation_bridge import PresentationBridge
    app = QGuiApplication([])
    with TemporaryDirectory() as directory, patch("ui_qt.quick.presentation_bridge.powerpoint_integration_available", return_value=(False, "profile")):
        root = Path(directory)
        bridge = PresentationBridge(settings=QSettings(str(root/"profile.ini"), QSettings.IniFormat))
        try:
            bridge._records = [PlotImage(root/("keep" if index < args.records//2 else "hide")/f"measurement_{index:06}.png",
                                        ("keep" if index < args.records//2 else "hide")+f"/measurement_{index:06}.png", "PL", float(index))
                               for index in range(args.records)]
            bridge._selected_available = [str(record.path) for record in bridge._records]
            bridge._options["query"] = "keep"
            bridge._apply_filters()
            rows = {}
            for name, action in (("filtered_snapshot", lambda: bridge.snapshot),
                                 ("control_selection", lambda: bridge.selectAvailableRow(0, Qt.ControlModifier.value))):
                samples = []
                for _ in range(5):
                    start = perf_counter()
                    action()
                    samples.append((perf_counter()-start)*1000)
                rows[name] = round(statistics.median(samples), 3)
            report = json.loads(args.output.read_text()) if args.output.exists() else {}
            report.update(scope="Synthetic catalog; synchronous GUI bridge work only, no PNG/deck I/O or input-to-photon timing.", records=args.records)
            report[args.phase] = rows
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(rows))
        finally:
            bridge.close()
            app.processEvents()


if __name__ == "__main__":
    main()
