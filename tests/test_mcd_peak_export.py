from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.mcd_peak_export import (
    mcd_peak_shift_export_worker,
    peak_shift_history_status,
    source_descriptor,
)
from core.mcd_peak_shift import PeakCandidate, PeakPoint, PeakShiftResult, PeakTrack
from ui_qt.common import Worker


def _snapshot(folder: Path) -> dict:
    source = folder / "source.csv"
    source.write_text("source\n", encoding="utf-8")
    result = PeakShiftResult(
        fields_t=np.asarray([0.0]),
        branches=np.asarray(["B increasing"]),
        candidates=((PeakCandidate(1.7, 1.0, 3.0),),),
        tracks=(PeakTrack(
            peak_id=1,
            branch="B increasing",
            points=(PeakPoint(0.0, "B increasing", 1.7, None, "tracked"),),
            reference_energy_ev=1.7,
            reference_field_t=0.0,
            reference_method="exact 0 T",
        ),),
        source="raw pos",
    )
    return {
        "result": result,
        "method_results": {},
        "tracker_method": "Raw spectrum",
        "selected_k": None,
        "selected_kp": None,
        "experiment_folder": str(folder),
        "source_descriptor": source_descriptor(folder, source.name),
        "history_settings": {"tracker_method": "Raw spectrum"},
    }


class McdPeakExportTests(unittest.TestCase):
    def test_worker_writes_equivalent_csv_sidecar_and_experiment_history(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            snapshot = _snapshot(folder)
            output = folder / "arbitrary" / "peak.csv"
            events: list[str] = []
            worker = Worker(mcd_peak_shift_export_worker, snapshot, str(output))
            worker.signals.result.connect(lambda _value: events.append("result"))
            worker.signals.error.connect(lambda _value: events.append("error"))
            worker.signals.finished.connect(lambda: events.append("finished"))
            worker.run()
            self.assertEqual(events, ["result", "finished"])
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_name("peak.metadata.json").is_file())
            index = folder / "Processed Data" / "MCD Peak Shift" / ".peak_shift_history.json"
            self.assertTrue(index.is_file())
            self.assertEqual(json.loads(index.read_text(encoding="utf-8"))["workflow"], "MCD Peak Shift")
            self.assertEqual(
                peak_shift_history_status(folder, snapshot["source_descriptor"])[0],
                "processed",
            )

    def test_incomplete_or_corrupt_history_stays_unknown(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            snapshot = _snapshot(folder)
            index = folder / "Processed Data" / "MCD Peak Shift" / ".peak_shift_history.json"
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(json.dumps({
                "workflow": "MCD Peak Shift",
                "exports": [{"source_file": str(folder / "source.csv")}],
            }), encoding="utf-8")
            self.assertEqual(
                peak_shift_history_status(folder, snapshot["source_descriptor"])[0],
                "unknown",
            )
            index.write_text('{"workflow":"MCD Peak Shift","exports":null}', encoding="utf-8")
            self.assertEqual(
                peak_shift_history_status(folder, snapshot["source_descriptor"])[0],
                "unknown",
            )

    def test_complete_different_path_with_same_basename_is_new(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            snapshot = _snapshot(folder)
            other = folder / "other"
            other.mkdir()
            index = folder / "Processed Data" / "MCD Peak Shift" / ".peak_shift_history.json"
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(json.dumps({
                "workflow": "MCD Peak Shift",
                "exports": [{
                    "source_descriptor": {
                        "path": str(other / "source.csv"),
                        "filename": "source.csv",
                        "name": "source.csv",
                    },
                    "created_utc": "2026-09-11T00:00:00Z",
                }],
            }), encoding="utf-8")
            self.assertEqual(
                peak_shift_history_status(folder, snapshot["source_descriptor"])[0],
                "new",
            )
if __name__ == "__main__":
    unittest.main()
