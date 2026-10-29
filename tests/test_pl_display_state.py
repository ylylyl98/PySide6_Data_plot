"""Canonical PL display state, automatic limits and persistence precedence."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class PlDisplayStateTests(unittest.TestCase):
    def test_native_pl_display_state(self):
        with TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, QT_QPA_PLATFORM="offscreen",
                    MPLBACKEND="agg", MPLCONFIGDIR=folder, PYTHONIOENCODING="utf-8"),
                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ui_qt.quick.pl_display_state import PlDisplayState
    # Empty display construction must not pull the plotting stack or Widgets.
    assert not any(key in sys.modules for key in ("numpy", "scipy", "matplotlib", "PySide6.QtWidgets"))
    from copy import deepcopy
    import time
    import numpy as np
    from PySide6.QtCore import QCoreApplication, QLocale, QSettings
    from core.loader import DataCube
    from core.plotting import HeatmapParams, split_region_slices

    app = QCoreApplication([])
    QLocale.setDefault(QLocale.c())

    def cube(offset=0., multiplier=1., columns=9):
        x = np.arange(1., columns+1.)+offset
        y = np.array([-1., 0., 1.])
        z = multiplier*(np.arange(1., columns+1.)[None,:]+np.arange(3.)[:,None])
        return DataCube(x, y, z, "Gate", "PL", "Counts")

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory()
            self.addCleanup(self.temporary.cleanup)
            self.settings = QSettings(str(Path(self.temporary.name)/"state.ini"), QSettings.IniFormat)
            self.state = PlDisplayState(settings=self.settings)
            self.addCleanup(self.state.close)
            self.cube = cube()
            self.state.sync_cube(self.cube)

        def params(self, scale="linear"):
            return self.state.heatmap_params(HeatmapParams("PL", "Energy", "Gate", "Counts", 0., 1., (1.,9.), (-1.,1.)), scale)

        def test_default_and_protocol_are_independent_copies(self):
            self.assertEqual(self.state.values["cmap"], "turbo")
            self.assertEqual(self.state.snapshot["scale"], "linear")
            descriptor = self.state.snapshot["numbers"]["linear.vmin"]
            self.assertTrue(all(key in descriptor for key in ("value","text","editText","step","error","enabled","visible")))
            snapshot = self.state.snapshot
            snapshot["numbers"]["axis.xmin"]["value"] = 200
            self.assertEqual(self.state.values["xmin"], 1.)
            self.assertNotIn("PySide6.QtWidgets", sys.modules)

        def test_auto_respects_each_lock_and_numeric_edits_do_not_lock(self):
            self.assertTrue(self.state.setNumber("axis.xmin", "1.25"))
            self.assertFalse(self.state.axis_fixed["xmin"])
            self.state.setFlag("fix.axis.xmin", True)
            self.state.setNumber("axis.xmax", "8.5")
            self.assertTrue(self.state.invoke("auto.x"))
            self.assertEqual((self.state.values["xmin"],self.state.values["xmax"]), (1.25,9.))
            self.state.setNumber("linear.vmin", ".5")
            self.state.setFlag("fix.linear.vmin", True)
            self.state.setNumber("linear.vmax", "50")
            self.state.invoke("auto.linear")
            self.assertEqual(self.state.banks["linear"]["vmin"], .5)
            self.assertLess(self.state.banks["linear"]["vmax"], 12.)
            self.assertFalse(self.state.banks["linear"]["fixed"]["vmax"])

        def test_tabs_share_geometry_but_have_independent_color_values(self):
            self.state.setChoice("regionCount", 2)
            self.state.setNumber("boundary.x0", "3.5")
            self.state.setNumber("boundary.x1", "6.5")
            self.state.setNumber("linear.middle_vmin", "2")
            self.state.setFlag("fix.linear.middle_vmin", True)
            geometry = deepcopy(self.state.geometry)
            log_before = deepcopy(self.state.banks["log"])
            self.state.setChoice("scale", 1)
            self.assertEqual(self.state.values["scale"], "log")
            self.assertEqual(self.state.geometry, geometry)
            self.assertEqual(self.state.banks["log"], log_before)
            self.assertFalse(self.state.banks["log"]["fixed"]["middleMin"])
            params = self.params("log")
            self.assertEqual((params.split_scale.split_x,params.split_scale.split_x2), (3.5,6.5))
            self.assertTrue(params.log_scale)
            self.assertEqual(params.cmap, "turbo")

        def test_nonuniform_grid_initializes_distinct_regions_and_keeps_fixed_boundary(self):
            nonuniform = cube(columns=4)
            nonuniform.energy = np.array([1.,2.,100.,101.])
            self.state.sync_cube(nonuniform)
            self.assertTrue(self.state.setChoice("regionCount",2))
            params = self.params()
            slices, boundaries = split_region_slices(nonuniform.energy,params.split_scale,params.xlim)
            self.assertLess(boundaries[0],boundaries[1])
            self.assertTrue(all(part.stop>part.start for part in slices))
            self.state.setChoice("regionCount",0)
            self.state.setChoice("regionCount",1)
            self.state.setNumber("boundary.x0","51")
            self.state.setFlag("fix.boundary.x0",True)
            self.assertTrue(self.state.setChoice("regionCount",2))
            self.assertEqual(self.state.geometry["x0"],51.)
            self.assertEqual(self.state.geometry["x1"],100.5)

        def test_public_manual_axis_context_is_copied_and_requires_explicit_y_lock(self):
            self.assertIsNone(self.state.manual_axis_context)
            self.state.sync_cube(cube(),axis_context=dict(y_axis="bg"))
            self.state.setFlag("fix.axis.ymin",True)
            context = self.state.manual_axis_context
            self.assertEqual(context["y_axis"],"bg")
            context["y_axis"] = "tg"
            self.assertEqual(self.state.manual_axis_context["y_axis"],"bg")
            self.state.setFlag("fix.axis.ymin",False)
            self.assertIsNone(self.state.manual_axis_context)

        def test_offscreen_regions_project_colors_without_moving_boundaries(self):
            self.state.setChoice("regionCount", 2)
            self.state.setNumber("boundary.x0", "3.5")
            self.state.setNumber("boundary.x1", "6.5")
            for side, lower, upper in (("left",1,2),("middle",3,4),("right",5,6)):
                self.state.setNumber("linear."+side+"_vmin", str(lower))
                self.state.setNumber("linear."+side+"_vmax", str(upper))
                self.state.setFlag("fix.linear."+side+"_vmin", True)
                self.state.setFlag("fix.linear."+side+"_vmax", True)
            geometry = deepcopy(self.state.geometry)
            self.state.view_changed((4.,9.), (-1.,1.))
            params = self.params()
            self.assertEqual(self.state.geometry, geometry)
            self.assertIsNone(params.split_scale.split_x2)
            self.assertEqual(params.split_scale.bounds, ((3.,4.),(5.,6.)))
            split_region_slices(self.cube.energy,params.split_scale,params.xlim)
            self.state.view_changed((7.,9.), (-1.,1.))
            params = self.params()
            self.assertIsNone(params.split_scale)
            self.assertEqual((params.vmin,params.vmax),(5.,6.))

        def test_empty_region_auto_retains_values(self):
            self.state.setChoice("regionCount", 2)
            self.state.setNumber("boundary.x0", "3.5")
            self.state.setNumber("boundary.x1", "6.5")
            self.state.view_changed((7.,9.),(-1.,1.))
            before = self.state.banks["linear"]["middleMin"], self.state.banks["linear"]["middleMax"]
            self.assertFalse(self.state.snapshot["actions"]["auto.linear.middle"]["enabled"])
            self.assertFalse(self.state.invoke("auto.linear.middle"))
            self.assertIn("No visible",self.state.error)
            self.assertEqual(before,(self.state.banks["linear"]["middleMin"],self.state.banks["linear"]["middleMax"]))

        def test_bad_drafts_and_colliding_boundaries_do_not_mutate_values(self):
            before = self.state.capture_state()
            for text in ("nan","inf","broken","1e13","99"):
                self.assertFalse(self.state.setNumber("axis.xmin",text))
            self.assertEqual(before["values"],self.state.values)
            self.state.setChoice("regionCount",2)
            self.state.setNumber("boundary.x0","3.5")
            self.state.setNumber("boundary.x1","6.5")
            self.assertFalse(self.state.setNumber("boundary.x1","3.6"))
            self.assertEqual(self.state.geometry["x1"],6.5)
            self.assertFalse(self.state.setNumber("log.vmin","0"))
            self.assertFalse(self.state.setNumber("linear.vmin","10000"))

        def test_auto_disabled_when_both_fixed_and_no_positive_log_samples(self):
            self.state.setFlag("fix.axis.xmin",True)
            self.state.setFlag("fix.axis.xmax",True)
            self.assertFalse(self.state.snapshot["actions"]["auto.x"]["enabled"])
            self.assertFalse(self.state.invoke("auto.x"))
            self.state.sync_cube(cube(multiplier=-1.))
            self.assertFalse(self.state.snapshot["actions"]["auto.log"]["enabled"])
            self.assertFalse(self.state.invoke("auto.log"))
            self.assertIn("positive",self.state.error)

        def test_restart_manual_locks_override_saved_borrowed_values(self):
            self.state.setNumber("axis.xmin","1.5")
            self.state.setFlag("fix.axis.xmin",True)
            self.state.setNumber("log.vmin",".5")
            self.state.setFlag("fix.log.vmin",True)
            self.settings.sync()
            saved = self.state.capture_state()
            saved["values"].update(xmin=2.,cmap="plasma",scale="log")
            saved["banks"]["log"]["vmin"] = .8
            other = PlDisplayState(settings=self.settings)
            self.addCleanup(other.close)
            other.sync_cube(cube(),saved=saved)
            self.assertEqual(other.values["xmin"],1.5)
            self.assertEqual(other.banks["log"]["vmin"],.5)
            self.assertEqual(other.values["cmap"],"plasma")
            self.assertEqual(other.values["scale"],"log")
            other.sync_cube(cube(offset=10.))
            self.assertEqual(other.values["xmin"],1.5)
            self.assertTrue(other.axis_fixed["xmin"])

        def test_manual_lock_releases_only_a_conflicting_borrowed_peer(self):
            saved = self.state.capture_state()
            saved["values"].update(xmin=2.,xmax=3.)
            saved["axis_fixed"].update(xmin=True,xmax=True)
            self.state.setNumber("axis.xmin","4")
            self.state.setFlag("fix.axis.xmin",True)
            self.state.sync_cube(cube(),saved=saved)
            self.assertEqual(self.state.values["xmin"],4.)
            self.assertGreater(self.state.values["xmax"],4.)
            self.assertTrue(self.state.axis_fixed["xmin"])
            self.assertFalse(self.state.axis_fixed["xmax"])

        def test_saved_file_fix_is_not_promoted_to_global_lock(self):
            saved = self.state.capture_state()
            saved["values"]["xmin"] = 1.5
            saved["axis_fixed"]["xmin"] = True
            self.state.sync_cube(cube(),saved=saved)
            self.assertEqual(self.state.values["xmin"],1.5)
            self.state.sync_cube(cube(offset=10.))
            self.assertEqual(self.state.values["xmin"],11.)
            self.assertFalse(self.state.axis_fixed["xmin"])

        def test_borrowed_colormap_does_not_become_new_file_default(self):
            saved = self.state.capture_state()
            saved["values"]["cmap"] = "plasma"
            self.state.sync_cube(cube(),saved=saved)
            self.assertEqual(self.state.values["cmap"],"plasma")
            self.state.sync_cube(cube(offset=10.))
            self.assertEqual(self.state.values["cmap"],"turbo")
            self.state.set_parameter("cmap","inferno")
            self.state.sync_cube(cube(),saved=saved)
            self.assertEqual(self.state.values["cmap"],"plasma")
            self.state.sync_cube(cube(offset=10.))
            self.assertEqual(self.state.values["cmap"],"inferno")
            self.state.sync_cube(cube(),saved=saved)
            self.state.set_parameter("cmap","plasma")
            self.state.sync_cube(cube(offset=10.))
            self.assertEqual(self.state.values["cmap"],"plasma")

        def test_manual_y_locks_follow_axis_context(self):
            context = dict(y_axis="tg")
            self.state.sync_cube(cube(),axis_context=context)
            self.state.setNumber("axis.ymin","-.5")
            self.state.setFlag("fix.axis.ymin",True)
            self.state.sync_cube(cube(),axis_context=dict(y_axis="bg"))
            self.assertEqual(self.state.values["ymin"],-1.)
            self.assertFalse(self.state.axis_fixed["ymin"])
            self.state.setNumber("linear.vmin",".1")
            self.state.sync_cube(cube(),axis_context=context)
            self.assertEqual(self.state.values["ymin"],-.5)
            self.assertTrue(self.state.axis_fixed["ymin"])

        def test_saved_and_manual_y_locks_suspend_when_only_label_changes(self):
            self.state.setNumber("axis.ymin","-.5")
            self.state.setFlag("fix.axis.ymin",True)
            saved = self.state.capture_state()
            changed = cube()
            changed.gate_label = "Doping"
            self.state.sync_cube(changed,saved=saved)
            self.assertEqual(self.state.values["ymin"],-1.)
            self.assertFalse(self.state.axis_fixed["ymin"])
            self.state.setNumber("linear.vmin",".1")
            self.state.sync_cube(cube())
            self.assertEqual(self.state.values["ymin"],-.5)

        def test_exact_saved_unlocked_colors_survive_until_auto_or_roi_change(self):
            saved = self.state.capture_state()
            saved["banks"]["linear"].update(vmin=.2,vmax=99.)
            self.state.sync_cube(cube(),saved=saved)
            self.state.refresh_auto()
            self.assertEqual((self.state.banks["linear"]["vmin"],self.state.banks["linear"]["vmax"]),(.2,99.))
            self.state.invoke("auto.linear")
            self.assertLess(self.state.banks["linear"]["vmax"],12.)
            self.state.sync_cube(cube(),saved=saved)
            self.state.view_changed((2.,8.),(-1.,1.))
            self.assertLess(self.state.banks["linear"]["vmax"],12.)

        def test_changed_revision_recalculates_only_unfixed_saved_colors(self):
            saved = self.state.capture_state()
            saved["compatibility"] = dict(x=True,y=True,axis=True,revision=False)
            saved["banks"]["linear"].update(vmin=.2,vmax=99.)
            saved["banks"]["linear"]["fixed"]["vmin"] = True
            self.state.sync_cube(cube(multiplier=2.),saved=saved)
            self.assertEqual(self.state.banks["linear"]["vmin"],.2)
            self.assertGreater(self.state.banks["linear"]["vmax"],20.)
            self.assertLess(self.state.banks["linear"]["vmax"],23.)

        def test_load_does_not_emit_mutation_and_steps_debounce(self):
            changes = []
            self.state.parameterChanged.connect(lambda key,value: changes.append((key,value)))
            self.state.sync_cube(cube())
            self.assertEqual(changes,[])
            self.state.setNumber("axis.xmin","1.25")
            self.assertEqual(changes[-1],("axis.xmin",1.25))
            changes.clear()
            for _ in range(4):self.state.stepNumber("axis.xmin",1)
            self.assertEqual(changes,[])
            deadline = time.monotonic()+.25
            while time.monotonic()<deadline:
                app.processEvents()
                time.sleep(.001)
            self.assertEqual(len(changes),1)
            changes.clear()
            self.state.stepNumber("axis.xmin",1)
            self.state.sync_cube(cube(offset=10.))
            deadline = time.monotonic()+.25
            while time.monotonic()<deadline:
                app.processEvents()
                time.sleep(.001)
            self.assertEqual(changes,[])

        def test_legacy_routes_and_clear_retains_explicit_locks(self):
            self.assertTrue(self.state.set_parameter("pl.bank.regionCount",3))
            self.assertTrue(self.state.set_parameter("pl.bank.splitX",3.5))
            self.assertTrue(self.state.set_parameter("pl.bank.middleMin",.25))
            self.assertTrue(self.state.set_parameter("pl.fixed.middleMin",True))
            self.assertTrue(self.state.set_parameter("scale","log"))
            self.assertTrue(self.state.set_parameter("cmap","inferno"))
            self.state.clear_cube()
            self.assertFalse(self.state.snapshot["numbers"]["linear.vmin"]["enabled"])
            self.state.sync_cube(cube())
            self.assertEqual(self.state.banks["linear"]["middleMin"],.25)
            self.assertTrue(self.state.banks["linear"]["fixed"]["middleMin"])

        def test_new_file_with_too_few_columns_projects_empty_region(self):
            self.state.setChoice("regionCount",2)
            self.state.setNumber("boundary.x0","3.5")
            self.state.setNumber("boundary.x1","6.5")
            self.state.setFlag("fix.boundary.x0",True)
            self.state.setFlag("fix.boundary.x1",True)
            self.state.sync_cube(cube(columns=2))
            self.assertEqual(self.state.geometry["count"],3)
            self.assertIsNone(self.params().split_scale)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        _isolated()
    else:
        unittest.main()
