"""Cold catalog work must not pull in unrelated numerical plotting modules."""
from pathlib import Path
import subprocess
import sys
import unittest

from core.measurement_names import split_group_and_sort_key


class MeasurementNamingTests(unittest.TestCase):
    def test_repeat_and_run_suffixes_preserve_group_and_sort_order(self):
        cases = {
            "folder/sample_rep02_003.csv": ("sample", (0, 2, 3)),
            "sample_REP2_4.csv": ("sample", (0, 2, 4)),
            "sample_rep2.csv": ("sample", (1, 2, 0)),
            "sample$_rep010.csv": ("sample", (1, 10, 0)),
            "sample_003.csv": ("sample", (1, 3, 0)),
            "sample$_1000.csv": ("sample", (1, 1000, 0)),
            "sample_03.csv": ("sample_03", (2, 0, 0)),
            "sample_rep1000.csv": ("sample_rep1000", (2, 0, 0)),
            "测量.760nmc.csv": ("测量.760nmc", (2, 0, 0)),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(split_group_and_sort_key(name), expected)

    def test_processing_keeps_the_existing_public_naming_api(self):
        from core.processing import split_group_and_sort_key as processing_split
        self.assertIs(processing_split, split_group_and_sort_key)


class LightweightDrrCatalogTests(unittest.TestCase):
    def test_cold_scan_history_cache_write_and_restart_avoid_plotting_imports(self):
        code = r'''
import importlib.abc
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

class RejectPlottingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in {"core.processing", "core.processing_run", "core.loader", "core.data_io"} or fullname.split(".")[0] in {"scipy", "pandas", "matplotlib"}:
            raise AssertionError("DRR catalog imported " + fullname)

sys.meta_path.insert(0, RejectPlottingImports())
from core.drr_catalog import load_drr_catalog
from core.drr_sources import DrrSourceCache, group_drr_sources
from core.source_catalog_cache import _DRR_SNAPSHOTS

with TemporaryDirectory() as folder:
    base = Path(folder)
    experiment = base / "experiment"
    raw = experiment / "Initial Data" / "REF"
    raw.mkdir(parents=True)
    measurement = raw / "sample_700nmc_rep01.csv"
    measurement.write_text("Vbg,700,701\n0,1,2\n1,2,3\n", encoding="utf-8")
    history = experiment / "Processed Data" / "DRR"
    history.mkdir(parents=True)
    (history / "saved.metadata.json").write_text(json.dumps({
        "operation": "DR/R", "sources": [{"role": "measurement", "source_path": str(measurement)}],
        "processing": {"baseline_selection": "Self (last frame)"}}), encoding="utf-8")
    preview = []
    cache_path = base / "inspection.json"
    options = {"cache_root": base / "catalog"}
    sources = load_drr_catalog(experiment, DrrSourceCache(cache_path, load_on_init=False),
        publish_inventory=preview.extend, **options)
    assert len(preview) == 1 and not preview[0].grid_complete
    assert len(sources) == 1 and sources[0].processed and sources[0].grid_complete
    assert sources[0].gate_grid == ((0.,), (1.,))
    assert len(group_drr_sources(sources)) == 1
    _DRR_SNAPSHOTS.clear()
    restored = load_drr_catalog(experiment, DrrSourceCache(cache_path, load_on_init=False), **options)
    assert restored == sources
    assert not any(name in sys.modules for name in ("core.data_io", "scipy", "pandas", "matplotlib"))
print("LIGHTWEIGHT_DRR_CATALOG_OK")
'''
        result = subprocess.run([sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1], capture_output=True,
            text=True, encoding="utf-8", timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("LIGHTWEIGHT_DRR_CATALOG_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
