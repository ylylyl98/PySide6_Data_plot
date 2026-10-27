"""Native DRR manual ranges survive restart without adopting borrowed history."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickDrrRestartRangeTests(unittest.TestCase):
    def test_restart_ranges_in_isolated_process(self):
        with TemporaryDirectory(prefix="quick-drr-restart-process-") as temporary:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                       MPLBACKEND="agg", MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary,
                       XDG_CACHE_HOME=temporary)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                                    cwd=Path(__file__).resolve().parents[1], env=env,
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import json
    from copy import deepcopy
    from dataclasses import replace
    import numpy as np
    from PySide6.QtCore import QCoreApplication, QLocale, QSettings
    from core.loader import DataCube
    from ui_qt.quick.drr_display_state import DrrDisplayState

    app = QCoreApplication([])
    QLocale.setDefault(QLocale.c())

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="quick-drr-restart-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name)
            self.ini = self.root / "native.ini"
            self.states = []
            self.addCleanup(self.close_states)
            x, gate = np.linspace(1., 3., 61), np.linspace(-1., 1., 9)
            z = 100 * gate[:, None] + 2 * x[None, :]
            self.raw = DataCube(x, gate, z, "Gate", "Synthetic DRR", "ΔR/R")
            self.second = replace(self.raw, Z=z / 1000, title="Synthetic D2E",
                                  cbar_label="d²(ΔR/R)/dE²")
            self.state = self.new_state()
            self.load(self.state)

        def close_states(self):
            for state in self.states:
                state.close()
            app.processEvents()

        def new_state(self):
            # A fresh QSettings object reads the same disk-backed temporary INI.
            state = DrrDisplayState(settings=QSettings(str(self.ini), QSettings.Format.IniFormat))
            self.states.append(state)
            return state

        def restart(self, state=None):
            (state or self.state).close()
            return self.new_state()

        def load(self, state, *, candidates=None, name="a_720nmc.csv", second=True):
            state.sync_cube(self.raw, self.second if second else None)
            state.restore_candidates(candidates or {}, folder=str(self.root), files=[name])

        def edit(self, key, value, *, fixed=False, state=None):
            state = state or self.state
            self.assertTrue(state.setNumber(key, str(value)), (key, state.error))
            if fixed:
                self.assertTrue(state.setFlag("fix." + key, True), key)

        def assert_manual(self, state, expected):
            for key, value in expected.items():
                with self.subTest(key=key):
                    self.assertEqual(state.snapshot["numbers"][key]["value"], value)
                    flag = state.snapshot["flags"]["fix." + key]
                    self.assertTrue(flag["checked"])
                    self.assertEqual(flag["text"], "Fix M")

        def saved_view(self):
            saved = deepcopy(self.state.capture_state(["a_720nmc.csv"]))
            saved["ranges"].update(xmin=1.2, xmax=2.8, ymin=-.4, ymax=.4,
                                    vmin=-20., vmax=30.)
            saved["fixes"].update(xmin=True, xmax=True, ymin=True, ymax=True,
                                   vmin=True, vmax=True)
            saved["raw"].update(vmin=-20., vmax=30.)
            saved["second"].update(vmin=-.2, vmax=.3)
            return saved

        def test_fixed_y_and_raw_endpoints_survive_restart_and_first_history_restore(self):
            saved = self.saved_view()
            expected = {"axis.ymin": -.7, "axis.ymax": .8,
                        "raw.vmin": -50., "raw.vmax": 60.}
            for key, value in expected.items():
                self.edit(key, value, fixed=True)
            restored = self.restart()
            self.load(restored, candidates={"exact": saved, "center": "720"})
            self.assert_manual(restored, expected)
            self.assertEqual(restored.heatmap_params(self.raw).ylim, (-.7, .8))

        def test_second_manual_scale_survives_raw_only_first_load_and_delayed_product(self):
            saved = self.saved_view()
            self.edit("second.vmin", -.12345678)
            self.edit("second.vmax", .23456789)
            restored = self.restart()
            self.load(restored, candidates={"exact": saved, "center": "720"}, second=False)
            self.assertFalse(restored.second_auto_scale)
            restored.sync_cube(self.raw, self.second)
            params = restored.heatmap_params(self.second, "second")
            self.assertEqual((params.vmin, params.vmax), (-.12345678, .23456789))
            self.assertFalse(restored.second_auto_scale)

        def test_stepping_fixed_range_persists_last_value_without_waiting_for_debounce(self):
            self.edit("raw.vmin", -50., fixed=True)
            self.assertTrue(self.state.stepNumber("raw.vmin", 1))
            value = self.state.snapshot["numbers"]["raw.vmin"]["value"]
            self.assertNotEqual(value, -50.)
            restored = self.restart()
            self.load(restored)
            self.assert_manual(restored, {"raw.vmin": value})

        def test_fixed_split_color_bounds_survive_restart_independently(self):
            self.assertTrue(self.state.setChoice("regionCount", 2))
            self.assertTrue(self.state.setFlag("memory.lock_x", True))
            expected = {}
            for product, factor in (("raw", 1.), ("second", .001)):
                for index, side in enumerate(("left", "middle", "right"), 1):
                    for bound, value in (("vmin", -200. - index), ("vmax", 200. + index)):
                        key = product + "." + side + "_" + bound
                        expected[key] = round(value * factor, 6)
                        self.edit(key, expected[key], fixed=True)
            restored = self.restart()
            self.load(restored)
            self.assertEqual(restored.snapshot["regionCount"], 3)
            self.assert_manual(restored, expected)
            raw = restored.heatmap_params(self.raw).split_scale
            second = restored.heatmap_params(self.second, "second").split_scale
            self.assertEqual(raw.middle_vmin, -202.)
            self.assertEqual(second.right_vmax, .203)

        def test_partial_fix_release_and_auto_survive_another_restart(self):
            self.assertTrue(self.state.setChoice("regionCount", 2))
            self.assertTrue(self.state.setFlag("memory.lock_x", True))
            pairs = (("axis.ymin", -.7, "axis.ymax", .8, "auto.y"),
                     ("raw.vmin", -50., "raw.vmax", 60., "auto.raw"),
                     ("second.left_vmin", -.5, "second.left_vmax", .6, "auto.second.left"))
            for low, minimum, high, maximum, _action in pairs:
                self.edit(low, minimum, fixed=True)
                self.edit(high, maximum, fixed=True)
            restored = self.restart()
            self.load(restored)
            for low, _minimum, _high, _maximum, action in pairs:
                self.assertTrue(restored.setFlag("fix." + low, False))
                self.assertTrue(restored.invoke(action))
            again = self.restart(restored)
            self.load(again)
            for low, minimum, high, maximum, _action in pairs:
                with self.subTest(low=low):
                    self.assertFalse(again.snapshot["flags"]["fix." + low]["checked"])
                    self.assertNotEqual(again.snapshot["numbers"][low]["value"], minimum)
                    self.assert_manual(again, {high: maximum})

        def test_explicit_second_auto_persists_after_restart(self):
            self.edit("second.vmin", -.5)
            self.edit("second.vmax", .6)
            self.assertTrue(self.state.invoke("auto.second"))
            restored = self.restart()
            self.load(restored)
            self.assertTrue(restored.second_auto_scale)
            params = restored.heatmap_params(self.second, "second")
            self.assertNotEqual((params.vmin, params.vmax), (-.5, .6))

        def test_restore_auto_clears_durable_manual_ranges(self):
            self.edit("axis.ymin", -.7, fixed=True)
            self.edit("raw.vmin", -50., fixed=True)
            self.edit("second.vmin", -.5)
            self.edit("second.vmax", .6)
            self.assertTrue(self.state.invoke("memory.auto"))
            restored = self.restart()
            self.load(restored)
            self.assertEqual(restored.heatmap_params(self.raw).ylim, (-1., 1.))
            self.assertTrue(restored.second_auto_scale)
            for key in ("axis.ymin", "raw.vmin"):
                self.assertFalse(restored.snapshot["flags"]["fix." + key]["checked"])

        def test_fixed_y_restores_its_custom_coordinate_expression_before_loading(self):
            self.assertTrue(self.state.setChoice("yaxis.left", 6))
            for key, value in (("yaxis.a", 2.), ("yaxis.b", -.5), ("yaxis.c", .25)):
                self.edit(key, value)
            self.edit("axis.ymin", -.7, fixed=True)
            self.edit("axis.ymax", .8, fixed=True)
            expected_spec = self.state.y_axis_spec()
            restored = self.restart()
            self.assertEqual(restored.y_axis_spec(), expected_spec)
            self.load(restored)
            self.assertEqual(restored.y_axis_spec(), "linear:2,-0.5,0.25")
            self.assert_manual(restored, {"axis.ymin": -.7, "axis.ymax": .8})

        def test_corrupt_manual_entry_rejects_the_entire_payload_without_partial_fixes(self):
            invalid_entries = ({"limits": [-20., 30.], "fixed": ["yes", True]},
                               {"limits": [float("nan"), 30.], "fixed": [True, True]},
                               {"limits": [30., -20.], "fixed": [True, True]})
            for index, invalid in enumerate(invalid_entries):
                with self.subTest(invalid=invalid):
                    self.ini = self.root / ("corrupt-" + str(index) + ".ini")
                    settings = QSettings(str(self.ini), QSettings.Format.IniFormat)
                    settings.setValue("drr/display_memory/manual_ranges", json.dumps({
                        "version": 1, "yAxis": {"choice": 0, "coefficients": [1., -1., 0.]}, "ranges": {
                            "axis.y": {"limits": [-.7, .8], "fixed": [True, True]},
                            "raw.v": invalid}}))
                    settings.sync()
                    restored = self.new_state()
                    self.load(restored)
                    self.assertEqual(restored.heatmap_params(self.raw).ylim, (-1., 1.))
                    for key in ("axis.ymin", "axis.ymax", "raw.vmin", "raw.vmax"):
                        self.assertFalse(restored.snapshot["flags"]["fix." + key]["checked"])

        def test_borrowed_file_and_center_ranges_do_not_become_global_manual_overrides(self):
            saved = self.saved_view()
            for kind in ("exact", "template"):
                with self.subTest(kind=kind):
                    # Separate temporary settings per history origin.
                    self.ini = self.root / (kind + ".ini")
                    borrowed = self.new_state()
                    borrowed.set_memory_status("Synthetic history", available=True)
                    self.assertTrue(borrowed.setFlag("memory.colors", True))
                    self.load(borrowed, candidates={kind: saved, "center": "720"})
                    self.assertEqual(borrowed.heatmap_params(self.raw).vmin, -20.)
                    self.assertEqual(borrowed.snapshot["flags"]["fix.axis.xmin"]["text"],
                                     "Fix F" if kind == "exact" else "Fix λ")
                    # Saving a genuine manual Y range must not adopt borrowed V.
                    self.edit("axis.ymin", -.7, fixed=True, state=borrowed)
                    restored = self.restart(borrowed)
                    self.load(restored, name="different_730nmc.csv")
                    self.assertEqual(restored.heatmap_params(self.raw).ylim, (-.7, 1.))
                    self.assert_manual(restored, {"axis.ymin": -.7})
                    self.assertNotEqual(restored.heatmap_params(self.raw).vmin, -20.)
                    self.assertTrue(restored.second_auto_scale)
                    for key in ("axis.ymax", "raw.vmin", "raw.vmax"):
                        self.assertFalse(restored.snapshot["flags"]["fix." + key]["checked"])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_isolated()) if "--isolated" in sys.argv else unittest.main()
