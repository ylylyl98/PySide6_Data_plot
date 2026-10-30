"""Avoid constructing the potentially large pair table on every UI update."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication, QSettings
from ui_qt.quick.mcd_bridge import McdBridge


class McdSnapshotResponsivenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_diagnostics_are_built_when_the_sheet_is_open(self):
        bridge = McdBridge(settings=QSettings())
        self.addCleanup(bridge.close)
        rows = [{"index": 0, "field": 1.0}]
        with patch.object(bridge, "_diagnostics", return_value=rows) as diagnostics:
            self.assertEqual(bridge.snapshot["diagnostics"], [])
            diagnostics.assert_not_called()
            bridge._sheet = "diagnostics"
            self.assertEqual(bridge.snapshot["diagnostics"], rows)
            diagnostics.assert_called_once()
            bridge._sheet = "slope-details"
            self.assertEqual(bridge.snapshot["diagnostics"], [])

    def test_recommended_centers_ignore_feature_filter_and_select_window(self):
        bridge = McdBridge(settings=QSettings())
        self.addCleanup(bridge.close)
        request = object()
        bridge._requested = request
        bridge._products = SimpleNamespace(request=request, candidates=(
            dict(id="one", kind="window", recommended=True, energy_ev=1.7, width_mev=5., label="1"),
            dict(id="two", kind="window", recommended=False, energy_ev=1.8, width_mev=5., label="2"),
        ))
        bridge._values["candidateFilter"] = "Spectrum"
        bridge._selected = "spectrum-selection"
        self.assertEqual([row["id"] for row in bridge._recommended_centers()], ["one"])
        bridge.selectRecommendedCenter("one")
        self.assertEqual(bridge._values["center"], 1.7)
        self.assertEqual(bridge._values["width"], 5.)
        self.assertEqual(bridge._values["candidateFilter"], "Spectrum")
        self.assertEqual(bridge._selected, "spectrum-selection")
        self.assertTrue(bridge._recommended_centers()[0]["selected"])
        self.assertTrue(bridge._window_timer.isActive())

    def test_view_state_avoids_copying_tracks_but_public_snapshot_is_independent(self):
        bridge = McdBridge(settings=QSettings())
        self.addCleanup(bridge.close)
        bridge._feature_payload = {"tracks": [{"points": [{"energy_ev": 1.7}]}]}
        self.assertEqual(bridge.viewState["featurePayload"], {})
        snapshot = bridge.snapshot
        snapshot["featurePayload"]["tracks"][0]["points"][0]["energy_ev"] = 0
        self.assertEqual(bridge.snapshot["featurePayload"]["tracks"][0]["points"][0]["energy_ev"], 1.7)

    def test_shared_auto_limits_are_reused_and_locks_are_respected(self):
        first = McdBridge(settings=QSettings())
        second = McdBridge(settings=QSettings())
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        result = SimpleNamespace(cube=lambda name: name)
        first._products = second._products = SimpleNamespace(result=result)
        bounds = SimpleNamespace(xmin=1, xmax=2, ymin=-3, ymax=3, vmin=-.2, vmax=.2)
        second._values.update(fixX=True, xmin=4, xmax=5)
        with patch("core.processing.compute_auto_limits", return_value=bounds) as compute:
            first._auto_ranges()
            second._auto_ranges()
            compute.assert_called_once_with("Combo")
        self.assertEqual((second._values["xmin"], second._values["xmax"]), (4, 5))
        self.assertEqual((second._values["vmin"], second._values["vmax"]), (-.2, .2))

    def test_worker_limits_allow_a_valid_combo_with_unavailable_auxiliary_map(self):
        import numpy as np
        from ui_qt.quick.mcd_workflow import prepare_auto_limits
        energy = np.linspace(1.5, 1.8, 7)
        fields = np.linspace(-1, 1, 3)
        result = SimpleNamespace(maps={
            "Combo": SimpleNamespace(energy=energy, gate=fields, Z=np.ones((3, 7))),
            "Normalized": SimpleNamespace(energy=energy, gate=fields, Z=np.full((3, 7), np.nan)),
        })
        prepare_auto_limits(result)
        self.assertEqual(set(result._quick_mcd_auto_limits), {"Combo"})
        self.assertEqual(result._quick_mcd_auto_limits["Combo"].xmin, 1.5)
        result.maps["Combo"].Z[:] = np.nan
        result._quick_mcd_auto_limits.clear()
        with self.assertRaisesRegex(ValueError, "no finite"):
            prepare_auto_limits(result)


if __name__ == "__main__":
    unittest.main()
