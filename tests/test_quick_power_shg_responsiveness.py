"""Large catalogs and SHG tables should not be rebuilt on every cursor update."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QCoreApplication, QSettings

from core.power_workflow import PowerMeasurementGroup
from ui_qt.quick.power_bridge import PowerBridge
from ui_qt.quick.shg_bridge import ShgBridge
from ui_qt.quick.compare_bridge import CompareBridge


class ResponsivenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = QSettings(str(Path(self.directory.name)/"settings.ini"), QSettings.IniFormat)

    def test_power_serializes_visible_groups_and_selected_hidden_group(self):
        bridge = PowerBridge(settings=self.settings)
        self.addCleanup(bridge.close)
        groups = [PowerMeasurementGroup(str(i), f"group {i}", ("a.csv",), {}, {}) for i in range(100)]
        bridge._catalog = dict(groups=groups, sources={}, combined=set(), processed=set(), warnings=[])
        bridge._selected_group = "80"
        from dataclasses import asdict
        with patch("ui_qt.quick.power_bridge.asdict", wraps=asdict) as convert:
            state = bridge._selection_snapshot()
        self.assertEqual(convert.call_count, 21)
        self.assertEqual(state["totalGroups"], 100)
        self.assertEqual(state["olderCount"], 80)
        self.assertEqual(state["selectedMeasurement"]["key"], "80")
        state["measurementGroups"][0]["sources"].append("mutated")
        self.assertEqual(bridge._selection_snapshot()["measurementGroups"][0]["sources"], ["a.csv"])
        bridge._values["query"] = "group 80"
        self.assertEqual(bridge._selection_snapshot()["totalGroups"], 1)

    def test_compare_group_snapshot_is_independent(self):
        bridge = CompareBridge(settings=self.settings)
        self.addCleanup(bridge.close)
        bridge._groups = [dict(key="g", label="group", sources=["a", "b"], mapping={"KK": "a"},
                               duplicates={"KK": ["a", "b"]}, status="new", modifiedText="date unavailable")]
        group = bridge.snapshot["sourceGroups"][0]
        group["sources"].append("c")
        group["mapping"]["KK"] = "changed"
        group["duplicates"]["KK"].clear()
        self.assertEqual(bridge._groups[0]["sources"], ["a", "b"])
        self.assertEqual(bridge._groups[0]["mapping"]["KK"], "a")
        self.assertEqual(bridge._groups[0]["duplicates"]["KK"], ["a", "b"])

    def test_shg_rows_cache_tracks_new_products_and_is_independent(self):
        bridge = ShgBridge(settings=self.settings)
        self.addCleanup(bridge.close)
        array = np.array([1., 2.])
        result = SimpleNamespace(integrated_area=array, measured_angle_deg=array, area_uncertainty=array,
            included=np.ones(2, dtype=bool), cosmic_pixels_removed=np.zeros(2), quality_flags=["ok", "ok"],
            data=SimpleNamespace(source_file="source.csv", source_rows=array))
        products = SimpleNamespace(results=(result,), fits=(None,))
        first = bridge._rows(products)
        first[0]["area"] = -1
        self.assertEqual(bridge._rows(products)[0]["area"], 1.)
        changed = SimpleNamespace(results=(result,), fits=(SimpleNamespace(fit_mask=(False, True)),))
        self.assertFalse(bridge._rows(changed)[0]["fitIncluded"])
        self.assertTrue(bridge._rows(changed)[1]["fitIncluded"])
        self.assertEqual(bridge._rows(None), [])
        self.assertIsNone(bridge._rows_products)


if __name__ == "__main__":
    unittest.main()
