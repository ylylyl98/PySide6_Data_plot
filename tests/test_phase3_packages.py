from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.export import create_unique_package_dir, export_drr_png_and_dat
from core.loader import DataCube
from core.mcd import McdSettings, export_mcd_analysis_bundle, process_mcd
from core.mcd_extract import mcd_catalog_database_path
from core.plotting import HeatmapParams
from core.shg import ShgSettings, ShgSweepData, process_shg_sweep
from core.export import export_shg_results


def _synthetic_data() -> ShgSweepData:
    wavelength = np.linspace(507.0, 523.0, 21)
    spectra = np.vstack([
        800.0 + amplitude * np.exp(-0.5 * ((wavelength - 515.0) / 0.35) ** 2)
        for amplitude in (80.0, 140.0, 40.0)
    ])
    return ShgSweepData(
        source_file="synthetic.csv", wavelength_nm=wavelength, spectra=spectra,
        sweep_axis=("rot1", "rot1", "rot1"),
        target_angle_deg=np.array([1.0, 0.0, 2.0]),
        measured_angle_deg=np.array([1.01, -0.01, 2.02]),
        move_error_deg=np.array([0.01, 0.01, 0.02]),
        move_ok=np.array([True, True, False]),
        acquisition_ok=np.array([True, True, True]),
        source_rows=np.array([2, 3, 4]),
        detected_columns={"measured_angle": "measured position"},
    )


class Phase3PackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cube = DataCube(
            energy=np.asarray([1.0, 2.0]), gate=np.asarray([0.0, 1.0]),
            Z=np.asarray([[1.0, 2.0], [3.0, 4.0]]), gate_label="Gate",
            title="sample", cbar_label="Value",
        )
        self.params = HeatmapParams(
            title="sample", xlabel="Energy", ylabel="Gate", cbar_label="Value",
            vmin=0.0, vmax=4.0, xlim=(1.0, 2.0), ylim=(0.0, 1.0),
        )

    def test_package_names_are_safe_and_collision_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Processed Data" / "SHG"
            first = create_unique_package_dir(root, "sample: test*analysis?")
            second = create_unique_package_dir(root, "sample: test*analysis?")
            self.assertEqual(first.name, "sample__test_analysis")
            self.assertEqual(second.name, "sample__test_analysis_01")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_mcd_map_and_analysis_bundle_can_share_one_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("core.export._save_heatmap_png"):
            root = Path(tmp)
            source = root / "branch.csv"
            rows = []
            for field in (-1.0, 0.0, 1.0, 1.0, 0.0, -1.0):
                for angle, scale in ((10.0, 1.0), (50.0, 2.0)):
                    rows.append({"B_T": field, "angle_deg": angle, "700": scale * (10 + field), "710": scale * (8 + field)})
            pd.DataFrame(rows).to_csv(source, index=False)
            result = process_mcd(str(source), McdSettings(max_sequence_gap=1, max_delta_b=0.01))
            package = create_unique_package_dir(root / "Processed Data" / "MCD", "branch_MCD_E1.7eV_W5meV")
            relative = str(package.relative_to(root))
            map_paths = export_drr_png_and_dat(
                str(root), cube=result.cube("Combo"), params=self.params,
                export_base="branch_MCD_Combo", drr_style=False,
                processed_name=relative,
            )
            bundle = export_mcd_analysis_bundle(
                result, str(package), trace_map="Combo", center_ev=1.7,
                width_mev=5.0, metric="mean", settings=McdSettings(),
            )
            self.assertTrue(all(path.parent == package for path in map_paths.values()))
            self.assertTrue(all(path.parent == package for path in bundle.values()))
            payload = json.loads(bundle["settings"].read_text(encoding="utf-8"))
            self.assertEqual(payload["package"], package.name)
            catalog_path = mcd_catalog_database_path(root)
            self.assertTrue(catalog_path.is_file())
            connection = sqlite3.connect(catalog_path)
            try:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM processed_mcd").fetchone()[0],
                    1,
                )
            finally:
                connection.close()

    def test_shg_single_result_and_settings_stay_in_package(self) -> None:
        data = _synthetic_data()
        result = process_shg_sweep(data, ShgSettings())
        with tempfile.TemporaryDirectory() as tmp:
            package = create_unique_package_dir(Path(tmp) / "Processed Data" / "SHG", "sample_SHG_750nm")
            paths = export_shg_results(
                tmp, data=data, result=result, settings=ShgSettings(),
                processed_name=str(package.relative_to(Path(tmp))),
            )
            self.assertEqual(paths["csv"].parent, package)
            self.assertEqual(paths["settings"].parent, package)
            payload = json.loads(paths["settings"].read_text(encoding="utf-8"))
            self.assertEqual(payload["package"], package.name)


if __name__ == "__main__":
    unittest.main()
