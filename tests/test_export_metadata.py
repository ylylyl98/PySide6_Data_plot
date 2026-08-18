from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.export import export_drr_png_and_dat
from core.loader import DataCube
from core.plotting import HeatmapParams


class ExportMetadataTests(unittest.TestCase):
    def test_drr_export_writes_reproducibility_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            measurement = folder / "measurement.csv"
            background = folder / "background.csv"
            measurement.write_text("measurement", encoding="utf-8")
            background.write_text("background", encoding="utf-8")
            params = HeatmapParams(
                title="DR/R",
                xlabel="Photon energy",
                ylabel="Gate",
                cbar_label="DR/R",
                vmin=-1.0,
                vmax=1.0,
                xlim=(0.0, 2.0),
                ylim=(0.0, 1.0),
                center_zero=True,
            )
            cube = DataCube(
                energy=np.asarray([0.0, 1.0, 2.0]),
                gate=np.asarray([0.0]),
                Z=np.asarray([[0.1, 0.2, 0.3]]),
                gate_label="Gate",
                title="DR/R",
                cbar_label="DR/R",
            )

            paths = export_drr_png_and_dat(
                str(folder),
                cube=cube,
                params=params,
                export_base="sample_avg2_DR_R_External",
                metadata_input_files=(
                    ("measurement", measurement.name),
                    ("background", background.name),
                ),
                metadata_processing={
                    "mode": "DR/R External",
                    "average_count": 2,
                    "baseline_which": "all",
                },
            )

            metadata_path = paths["dat"].with_suffix(".metadata.json")
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["operation"], "DR/R")
            self.assertEqual(payload["processing"]["average_count"], 2)
            self.assertEqual(
                [item["role"] for item in payload["inputs"]],
                ["measurement", "background"],
            )
            self.assertTrue(all(item["exists"] for item in payload["inputs"]))
            self.assertIn(paths["dat"].name, payload["outputs"])
            self.assertIn(paths["png"].name, payload["outputs"])


if __name__ == "__main__":
    unittest.main()
