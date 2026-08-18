from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.export import export_compare_panels, export_drr_png_and_dat, export_pl_pngs_and_dat
from core.loader import DataCube, load_dat
from core.plotting import HeatmapParams


class ExportOrganizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cube = DataCube(
            energy=np.asarray([1.0, 2.0]),
            gate=np.asarray([0.0, 1.0]),
            Z=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            gate_label="Gate",
            title="sample",
            cbar_label="Value",
        )
        self.params = HeatmapParams(
            title="sample",
            xlabel="Energy",
            ylabel="Gate",
            cbar_label="Value",
            vmin=0.0,
            vmax=4.0,
            xlim=(1.0, 2.0),
            ylim=(0.0, 1.0),
        )

    def test_workflow_exports_create_separate_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("core.export._save_heatmap_png"):
            pl = export_pl_pngs_and_dat(
                tmp,
                "sample.csv",
                cube_linear=self.cube,
                cube_log=self.cube,
                params_linear=self.params,
                params_log=self.params,
                processed_name="Processed Data/PL",
            )
            drr = export_drr_png_and_dat(
                tmp,
                cube=self.cube,
                params=self.params,
                export_base="sample_avg2_DR_R_External",
                processed_name="Processed Data/DRR",
            )
            compare = export_compare_panels(
                tmp,
                cubes={"KK": self.cube},
                source_files={"KK": "sample.csv"},
                params=self.params,
                scale_tag="linear",
                clip_outliers=False,
                export_vp=False,
                processed_name="Processed Data/Compare",
            )
            self.assertEqual(pl["dat"].parent, Path(tmp) / "Processed Data" / "PL")
            self.assertEqual(drr["dat"].parent, Path(tmp) / "Processed Data" / "DRR")
            self.assertTrue(compare)
            self.assertTrue(all(path.parent == Path(tmp) / "Processed Data" / "Compare" for path in compare))
            self.assertTrue(pl["dat"].with_suffix(".metadata.json").is_file())
            self.assertTrue(drr["dat"].with_suffix(".metadata.json").is_file())

    def test_collision_namespaces_are_independent_per_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("core.export._save_heatmap_png"):
            pl = export_pl_pngs_and_dat(
                tmp, "sample.csv", cube_linear=self.cube, cube_log=self.cube,
                params_linear=self.params, params_log=self.params,
                processed_name="Processed Data/PL",
            )
            drr = export_pl_pngs_and_dat(
                tmp, "sample.csv", cube_linear=self.cube, cube_log=self.cube,
                params_linear=self.params, params_log=self.params,
                processed_name="Processed Data/DRR",
            )
            self.assertEqual(pl["dat"].stem, "sample_PL_linear")
            self.assertEqual(drr["dat"].stem, "sample_PL_linear")

    def test_dat_import_works_from_workflow_folder_without_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Processed Data" / "Compare" / "sample.dat"
            path.parent.mkdir(parents=True)
            path.write_text(
                "Photon energy\t0\t1\n1\t2\t3\n2\t4\t5\n",
                encoding="utf-8",
            )
            cube = load_dat(path)
            self.assertEqual(cube.Z.shape, (2, 2))
            self.assertEqual(cube.title, "sample")


if __name__ == "__main__":
    unittest.main()
