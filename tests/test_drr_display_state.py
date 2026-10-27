"""Native DRR display protocol and approved QML in an isolated GUI process."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class NativeDrrDisplayStateTests(unittest.TestCase):
    def test_state_and_existing_display_qml_without_widgets(self):
        with TemporaryDirectory() as temporary:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                       QT_SCALE_FACTOR="1", MPLBACKEND="agg", MPLCONFIGDIR=temporary,
                       XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                                    cwd=Path(__file__).resolve().parents[1], env=env,
                                    capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import time
    import json
    from copy import deepcopy
    from dataclasses import replace
    import numpy as np
    from PySide6.QtCore import QCoreApplication, QEvent, QLocale, QObject, Property, QSettings, Qt, QTimer, QUrl, Signal, Slot
    from PySide6.QtGui import QGuiApplication, QKeyEvent, QMouseEvent
    from PySide6.QtQml import QQmlComponent, QQmlEngine
    from PySide6.QtQuick import QQuickItem, QQuickWindow
    from core.loader import DataCube
    from ui_qt.quick.drr_display_state import DrrDisplayState

    app = QGuiApplication([])
    QLocale.setDefault(QLocale.c())
    colors = dict(surface="#f3f5f8", control="#ffffff", text="#20242b", muted="#606978",
                  border="#c6cbd3", accent="#0067c0", hover="#edf1f6", pressed="#e0e7ef",
                  error="#b42318", disabled="#939aa5", selectedText="#ffffff", inset="#e9edf2",
                  divider="#dce1e8", borderHover="#aeb7c4", borderDisabled="#dce1e8",
                  controlDisabled="#edf0f4", focus="#20242b", accentText="#0067c0",
                  selected="#e6f0fa", selectedBorder="#b4d6fa", errorSurface="#fdf1f0")

    def settle(milliseconds=15):
        deadline = time.monotonic() + milliseconds / 1000
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.001)
        app.processEvents()

    def wait_until(predicate, description, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError('Timed out waiting for ' + description)
            app.processEvents()
            time.sleep(.001)

    class RootBridge(QObject):
        snapshotChanged = Signal()
        def __init__(self, state):
            super().__init__()
            self.state = state
            self.calls = []
        @Property(QObject, constant=True)
        def drrDisplayBridge(self):
            return self.state
        @Property("QVariantMap", notify=snapshotChanged)
        def snapshot(self):
            return dict(busy=False, canConfigureBackground=True, sourceSummary="Selected synthetic measurement", backgroundSummary="Self (last frame)",
                        processingGroups=[dict(title="Derivative processing", fields=[dict(key="sgWindow", label="X window points", type="number", value=21, decimals=0, step=2)])])
        @Slot(str)
        def invoke(self, action):
            self.calls.append(action)
        @Slot(str, "QVariant")
        def setParameter(self, key, value):
            self.calls.append((key, value))

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory()
            self.addCleanup(self.temporary.cleanup)
            self.settings = QSettings(self.temporary.name + "/native.ini", QSettings.Format.IniFormat)
            self.state = DrrDisplayState(settings=self.settings)
            self.addCleanup(self.state.close)
            x = np.linspace(1., 3., 31)
            gate = np.linspace(-1., 1., 9)
            z = 100 * gate[:, None] + 2 * x[None, :]
            self.raw = DataCube(x, gate, z, "Gate", "Raw DRR", "ΔR/R")
            self.second = replace(self.raw, Z=z / 1000, title="Second derivative", cbar_label="d²(ΔR/R)/dE²")
            self.state.sync_cube(self.raw, self.second)

        def test_snapshot_has_every_existing_display_control_and_independent_maps(self):
            snapshot = self.state.snapshot
            self.assertEqual(set(snapshot), {"numbers", "flags", "choices", "actions", "texts", "regionCount", "colorProduct", "advancedYAxis"})
            for product in ("raw", "second"):
                self.assertIn(product + ".cmap", snapshot["choices"])
                for region in ("", "left_", "middle_", "right_"):
                    for bound in ("vmin", "vmax"):
                        self.assertIn(product + "." + region + bound, snapshot["numbers"])
            self.assertNotEqual(self.state.heatmap_params(self.raw).cmap, self.state.heatmap_params(self.second, "second").cmap)
            self.assertNotEqual(snapshot["numbers"]["raw.vmax"]["value"], snapshot["numbers"]["second.vmax"]["value"])
            for key in ("memory.files", "memory.centers", "memory.colors", "memory.global", "memory.lock_x"):
                self.assertIn(key, snapshot["flags"])
            snapshot["numbers"]["raw.vmin"]["value"] = 999
            self.assertNotEqual(self.state.snapshot["numbers"]["raw.vmin"]["value"], 999)

        def test_precision_validation_and_color_tab_does_not_redraw(self):
            events = []
            self.state.parameterChanged.connect(lambda key, value: events.append((key, value)))
            self.state.setEditing("axis.xmin", True)
            self.assertTrue(self.state.setNumber("axis.xmin", "1.234567"))
            self.assertEqual(events, [("axis.xmin", 1.2346)])
            self.assertFalse(self.state.editing)
            before = self.state.snapshot["numbers"]["axis.xmin"]["value"]
            for text in ("nan", "inf", "2000000000000", "4"):
                self.assertFalse(self.state.setNumber("axis.xmin", text))
                self.assertEqual(self.state.snapshot["numbers"]["axis.xmin"]["value"], before)
            self.assertTrue(self.state.setNumber("axis.xmin", "1.2346"))
            self.assertEqual(self.state.error, "")
            self.assertTrue(self.state.setNumber("second.vmin", "-0.123456789"))
            self.assertEqual(self.state.snapshot["numbers"]["second.vmin"]["value"], -.12345679)
            previous_events = list(events)
            self.assertTrue(self.state.setChoice("colorProduct", 1))
            self.assertEqual(events, previous_events)

        def test_adaptive_steps_coalesce_and_wait_for_pending_edit(self):
            self.state.setNumber("raw.vmin", "-100")
            self.state.setNumber("raw.vmax", "100")
            self.assertEqual(self.state.snapshot["numbers"]["raw.vmin"]["step"], 1)
            events = []
            self.state.parameterChanged.connect(lambda key, value: events.append((key, value)))
            debounce = self.state._debounce
            timeouts = []
            debounce.timeout.connect(lambda: timeouts.append(True))
            for _ in range(3):
                self.assertTrue(self.state.stepNumber("raw.vmin", 1))
                self.assertTrue(debounce.isActive())
            self.assertEqual(events, [])
            self.assertEqual(self.state.snapshot["numbers"]["raw.vmin"]["value"], -97.)
            self.state.setEditing("axis.xmax", True)
            self.assertFalse(debounce.isActive())
            # Pump a full debounce interval while editing, using a real timer
            # as the boundary instead of assuming when Qt will dispatch it.
            edit_interval = []
            guard = QTimer(self.state)
            guard.setSingleShot(True)
            guard.timeout.connect(lambda: edit_interval.append(True))
            guard.start(debounce.interval())
            wait_until(lambda: edit_interval, "debounce interval during pending edit")
            self.assertEqual(timeouts, [])
            self.assertEqual(events, [])
            self.state.setEditing("axis.xmax", False)
            self.assertTrue(debounce.isActive())
            wait_until(lambda: events, "coalesced change after editing ends")
            self.assertEqual(events, [("raw.vmin", -97.)])
            self.assertEqual(timeouts, [True])
            self.assertFalse(debounce.isActive())

        def test_auto_retains_fixed_bounds_and_derivative_manual_range(self):
            self.state.setNumber("raw.vmin", "-50")
            self.state.setFlag("fix.raw.vmin", True)
            self.assertEqual(self.state.snapshot["flags"]["fix.raw.vmin"]["text"], "Fix M")
            self.state.setNumber("second.vmax", "0.5")
            self.assertFalse(self.state.second_auto_scale)
            self.state.sync_cube(replace(self.raw, Z=self.raw.Z * 2), replace(self.second, Z=self.second.Z * 2))
            self.assertEqual(self.state.snapshot["numbers"]["raw.vmin"]["value"], -50)
            self.assertEqual(self.state.snapshot["numbers"]["second.vmax"]["value"], .5)
            self.state.invoke("auto.second")
            self.assertTrue(self.state.second_auto_scale)
            self.assertNotEqual(self.state.snapshot["numbers"]["second.vmax"]["value"], .5)

        def test_single_region_reuses_statistics_and_next_auto_reads_current_data(self):
            from unittest.mock import patch
            from core.color_limits import ColorRequest, calculate_bounds

            self.state.setNumber("raw.left_vmin", "-200")
            self.state.setFlag("fix.raw.left_vmin", True)
            with patch("core.color_limits.calculate_bounds", wraps=calculate_bounds) as bounds:
                self.state._auto_product("raw", all_regions=True)
                self.assertEqual(bounds.call_count, 1)
            expected = calculate_bounds((self.raw,), ColorRequest((1., 3.), (-1., 1.), fallback_full=True))
            for prefix in ("raw.", "raw.right_"):
                self.assertEqual(self.state._numbers[prefix + "vmin"], self.state._quantize(prefix + "vmin", expected[0]))
                self.assertEqual(self.state._numbers[prefix + "vmax"], self.state._quantize(prefix + "vmax", expected[1]))
            self.assertEqual(self.state._numbers["raw.left_vmin"], -200)
            self.assertEqual(self.state._numbers["raw.left_vmax"], round(expected[1], 6))
            self.raw.Z[:] *= 2
            with patch("core.color_limits.calculate_bounds", wraps=calculate_bounds) as bounds:
                self.state._auto_product("raw", all_regions=True)
                self.assertEqual(bounds.call_count, 1)
            self.assertEqual(self.state._numbers["raw.vmax"], round(expected[1] * 2, 4))

        def test_split_statistics_and_saved_colors_keep_independent_limits(self):
            from unittest.mock import patch
            from core.color_limits import ColorRequest, calculate_bounds

            for count in (2, 3):
                with self.subTest(count=count):
                    self.state.setChoice("regionCount", count - 1)
                    with patch("core.color_limits.calculate_bounds", wraps=calculate_bounds) as bounds:
                        self.state._auto_product("raw", all_regions=True)
                        self.assertEqual(bounds.call_count, count + 1)
                    cuts = tuple(self.state._numbers["boundary." + key] for key in ("x0", "x1")[:count - 1])
                    sides = ("left", "right") if count == 2 else ("left", "middle", "right")
                    for index, side in enumerate(sides):
                        expected = calculate_bounds((self.raw,), ColorRequest((1., 3.), (-1., 1.), cuts, index, fallback_full=True))
                        for bound, value in zip(("vmin", "vmax"), expected):
                            self.assertEqual(self.state._numbers["raw." + side + "_" + bound], round(value, 6))
            saved = self.saved_view()
            saved["regions"]["count"] = 1
            self.assertTrue(self.state.restore_candidates(dict(exact=saved, center="720"),
                folder=self.temporary.name, files=["a_720nmc.csv"]))
            self.assertEqual(self.state._numbers["raw.vmin"], -20.)
            self.assertEqual(self.state._numbers["raw.vmax"], 30.)
            self.assertEqual(self.state._numbers["second.vmin"], -.2)
            self.assertEqual(self.state._numbers["second.vmax"], .3)

        def test_three_regions_keep_product_limits_and_shared_valid_boundaries(self):
            self.assertTrue(self.state.setChoice("regionCount", 2))
            self.state.setNumber("raw.left_vmin", "-200")
            self.state.setNumber("second.left_vmin", "-0.5")
            self.state.setFlag("fix.raw.left_vmin", True)
            self.state.invoke("auto.raw.left")
            raw = self.state.heatmap_params(self.raw)
            second = self.state.heatmap_params(self.second, "second")
            self.assertIsNotNone(raw.split_scale.split_x2)
            self.assertEqual(raw.split_scale.left_vmin, -200)
            self.assertEqual(second.split_scale.left_vmin, -.5)
            self.assertEqual(raw.split_scale.split_x, second.split_scale.split_x)
            self.assertFalse(self.state.setNumber("boundary.x0", "4"))
            self.assertTrue(self.state.setChoice("colorProduct", 1))
            self.assertTrue(self.state.snapshot["numbers"]["second.middle_vmin"]["visible"])
            self.assertFalse(self.state.snapshot["numbers"]["raw.middle_vmin"]["visible"])

        def test_native_axis_requests_and_valid_measured_right_mapping(self):
            self.assertTrue(self.state.setChoice("yaxis.left", 2))
            self.assertEqual(self.state.y_axis_spec(), "linear:1,-1,0")
            self.state.set_axis_coordinates({"TG": self.raw.gate * 2 + .3,
                                             "BG": np.zeros(len(self.raw.gate))})
            self.assertTrue(self.state.setChoice("yaxis.right", 3))
            self.assertFalse(self.state.setChoice("yaxis.right", 4))
            axis = self.state.heatmap_params(self.raw).right_axis
            np.testing.assert_allclose(axis.forward(self.raw.gate), self.raw.gate * 2 + .3)
            np.testing.assert_allclose(axis.inverse(axis.forward(self.raw.gate)), self.raw.gate)
            self.state.setChoice("yaxis.left", 6)
            self.state.setNumber("yaxis.a", "2.123456789")
            self.assertEqual(self.state.y_axis_spec(), "linear:2.123457,-1,0")
            self.assertTrue(self.state.snapshot["advancedYAxis"])

        def test_unsupported_history_is_explicit_and_lock_x_is_effective(self):
            self.assertFalse(self.state.snapshot["flags"]["memory.files"]["enabled"])
            self.assertFalse(self.state.snapshot["actions"]["memory.clear"]["enabled"])
            self.assertIn("not connected", self.state.snapshot["texts"]["memoryStatus"]["text"])
            self.assertFalse(self.state.invoke("memory.clear"))
            self.state.setNumber("axis.xmin", "1.5")
            self.state.setFlag("memory.lock_x", True)
            self.state.sync_cube(replace(self.raw, energy=self.raw.energy + 1), self.second)
            self.assertEqual(self.state.snapshot["numbers"]["axis.xmin"]["value"], 1.5)
            self.assertTrue(self.settings.value("drr/display_memory/lock_x", False, type=bool))

        def test_existing_lock_preference_cannot_lock_uninitialized_default_geometry(self):
            self.settings.setValue("drr/display_memory/lock_x", True)
            state = DrrDisplayState(settings=self.settings)
            self.addCleanup(state.close)
            state.sync_cube(self.raw, self.second)
            self.assertEqual(state.heatmap_params(self.raw).xlim, (1., 3.))

        def test_log_rejects_fixed_negative_limits_with_explicit_error(self):
            self.state.setNumber("raw.vmin", "-100")
            self.state.setFlag("fix.raw.vmin", True)
            self.assertFalse(self.state.setFlag("log", True))
            self.assertFalse(self.state.heatmap_params(self.raw).log_scale)
            self.assertIn("positive", self.state.error)

        def saved_view(self, name="a_720nmc.csv"):
            state = self.state.capture_state([name])
            state["ranges"].update(xmin=1.2, xmax=2.8, ymin=-.5, ymax=.5, vmin=-20., vmax=30.)
            state["fixes"].update(xmin=True, ymin=True, vmin=True)
            state["raw"].update(vmin=-20., vmax=30., cmap="plasma")
            state["second"].update(vmin=-.2, vmax=.3)
            return state

        def test_capture_roundtrips_portable_history_and_preserves_saved_fix_origins(self):
            from core.drr_display_settings import find_candidates, save_snapshot
            self.state.setChoice("regionCount", 2)
            self.state.setNumber("raw.left_vmin", "-150")
            self.state.setFlag("fix.raw.left_vmin", True)
            frozen = self.state.capture_state(["a_720nmc.csv"])
            self.state.setNumber("raw.left_vmin", "-140")
            self.assertEqual(frozen["regions"]["drr"]["left_vmin"], -150)
            save_snapshot(self.temporary.name, frozen)
            candidates = find_candidates(self.temporary.name, ["a_720nmc.csv"])
            restored = DrrDisplayState(settings=QSettings(self.temporary.name + "/restored.ini", QSettings.Format.IniFormat))
            self.addCleanup(restored.close)
            restored.sync_cube(self.raw, self.second)
            self.assertTrue(restored.restore_candidates(candidates, folder=self.temporary.name, files=["a_720nmc.csv"]))
            self.assertEqual(restored.heatmap_params(self.raw).split_scale.left_vmin, -150)
            self.assertEqual(restored.snapshot["flags"]["fix.raw.left_vmin"]["text"], "Fix F")
            restored.setNumber("raw.left_vmin", "-130")
            self.assertEqual(restored.snapshot["flags"]["fix.raw.left_vmin"]["text"], "Fix M")
            restored.sync_cube(self.raw, replace(self.second, Z=self.second.Z * 2))
            self.assertEqual(restored.heatmap_params(self.raw).split_scale.left_vmin, -130)

        def test_center_template_restores_only_geometry_until_colors_are_enabled(self):
            saved = self.saved_view()
            original_y = self.state.heatmap_params(self.raw).ylim
            original_colors = (self.state.heatmap_params(self.raw).vmin, self.state.heatmap_params(self.raw).vmax)
            candidates = dict(exact=None, template=saved, center="720")
            self.assertTrue(self.state.restore_candidates(candidates, folder=self.temporary.name, files=["b_720nmc.csv"]))
            params = self.state.heatmap_params(self.raw)
            self.assertEqual(params.xlim, (1.2, 2.8))
            self.assertEqual(params.ylim, original_y)
            from core.color_limits import ColorRequest, calculate_bounds
            expected = calculate_bounds((self.raw,), ColorRequest((1.2, 2.8), original_y))
            self.assertEqual((params.vmin, params.vmax), tuple(round(value, 4) for value in expected))
            self.assertNotEqual((params.vmin, params.vmax), (-20., 30.))
            self.assertEqual(self.state.snapshot["flags"]["fix.axis.xmin"]["text"], "Fix λ")
            self.state.set_memory_status("History connected", available=True)
            self.state.setFlag("memory.colors", True)
            self.assertTrue(self.state.restore_candidates(candidates, folder=self.temporary.name, files=["c_720nmc.csv"]))
            params = self.state.heatmap_params(self.raw)
            self.assertEqual((params.vmin, params.vmax), (-20., 30.))
            self.assertEqual(params.cmap, "plasma")

        def test_exact_source_and_axis_validation_reject_wrong_or_malformed_history(self):
            saved = self.saved_view()
            for payload in (saved, {**saved, "axis": {**saved["axis"], "y_spec": "tg"}},
                            {**saved, "deferred_second_regions": "left_vmin"},
                            {**saved, "second_auto_pending": "yes"},
                            {**saved, "ranges": {**saved["ranges"], "xmin": float("nan")}}):
                files = ["b_720nmc.csv"] if payload is saved else ["a_720nmc.csv"]
                self.assertFalse(self.state.restore_candidates(dict(exact=payload, template=None, center="720"), folder=self.temporary.name, files=files))
            self.assertEqual(self.state.heatmap_params(self.raw).xlim, (1., 3.))

        def test_manual_geometry_scope_and_exact_file_priority_preserve_genuine_color_fixes(self):
            folder = self.temporary.name
            self.state.restore_candidates({}, folder=folder, files=["a_720nmc.csv"])
            self.state.setNumber("axis.xmin", "1.4")
            self.state.setFlag("fix.axis.xmin", True)
            self.state.setNumber("axis.ymin", "-0.8")
            self.state.setFlag("fix.axis.ymin", True)
            self.state.setNumber("raw.vmin", "-80")
            self.state.setFlag("fix.raw.vmin", True)
            self.state.sync_cube(replace(self.raw, Z=self.raw.Z + 1), self.second)
            self.state.restore_candidates({}, folder=folder, files=["b_720nmc.csv"])
            self.assertEqual(self.state.heatmap_params(self.raw).xlim[0], 1.4)
            saved = self.saved_view("c_720nmc.csv")
            self.state.sync_cube(replace(self.raw, Z=self.raw.Z + 2), self.second)
            self.assertTrue(self.state.restore_candidates(dict(exact=saved, center="720"), folder=folder, files=["c_720nmc.csv"]))
            params = self.state.heatmap_params(self.raw)
            self.assertEqual(params.xlim[0], 1.2)
            self.assertEqual(params.ylim[0], -.8)
            self.assertEqual(params.vmin, -80)
            self.assertEqual(self.state.snapshot["flags"]["fix.axis.xmin"]["text"], "Fix F")
            self.state.setNumber("axis.xmin", "1.35")
            self.state.sync_cube(replace(self.raw, Z=self.raw.Z + 3), self.second)
            self.state.restore_candidates({}, folder=folder, files=["d_730nmc.csv"])
            self.assertEqual(self.state.heatmap_params(self.raw).xlim[0], 1)
            self.assertFalse(self.state.snapshot["flags"]["fix.axis.xmin"]["checked"])
            self.assertEqual(self.state.heatmap_params(self.raw).vmin, -80)

        def test_global_lock_persists_real_geometry_and_overrides_other_file_history(self):
            self.state.setChoice("regionCount", 1)
            self.state.setNumber("axis.xmin", "1.3")
            self.state.setFlag("memory.lock_x", True)
            restored = DrrDisplayState(settings=self.settings)
            self.addCleanup(restored.close)
            cube = replace(self.raw, energy=self.raw.energy + 1)
            restored.sync_cube(cube, replace(self.second, energy=cube.energy))
            self.assertEqual(restored.heatmap_params(cube).xlim, (1.3, 3.))
            saved = self.saved_view("b_730nmc.csv")
            saved["ranges"].update(xmin=2.2, xmax=3.8)
            saved["regions"]["drr"]["x0"] = 3.
            saved["regions"]["drr_second"]["x0"] = 3.
            self.assertTrue(restored.restore_candidates(dict(exact=saved, center="730"), folder=self.temporary.name, files=["b_730nmc.csv"]))
            self.assertEqual(restored.heatmap_params(cube).xlim, (1.3, 3.))
            self.assertEqual(restored.snapshot["flags"]["fix.axis.xmin"]["text"], "Fix M")

        def test_invalid_global_geometry_does_not_partially_apply_any_fix(self):
            self.state.setFlag("memory.lock_x", True)
            geometry = json.loads(self.settings.value("drr/display_memory/locked_geometry"))
            geometry["drr_second_fix.x0"] = "yes"
            self.settings.setValue("drr/display_memory/locked_geometry", json.dumps(geometry))
            restored = DrrDisplayState(settings=self.settings)
            self.addCleanup(restored.close)
            restored.sync_cube(self.raw, self.second)
            self.assertFalse(restored.memory_preferences["lock_x"])
            self.assertFalse(restored.snapshot["flags"]["fix.axis.xmin"]["checked"])

        def test_pending_restore_defers_until_valid_draft_and_retains_incoming_edit(self):
            saved = self.saved_view()
            self.state.setEditing("axis.xmin", True)
            self.assertFalse(self.state.restore_candidates(dict(exact=saved, center="720"), folder=self.temporary.name, files=["a_720nmc.csv"]))
            self.assertEqual(self.state.heatmap_params(self.raw).xlim[0], 1)
            self.state.setNumber("axis.xmin", "1.4")
            settle()
            self.assertEqual(self.state.heatmap_params(self.raw).xlim[0], 1.4)
            self.assertEqual(self.state.heatmap_params(self.raw).xlim[1], 2.8)

        def test_global_templates_write_only_through_explicit_success_hook(self):
            self.state.set_memory_status("History connected", available=True)
            self.state.setFlag("memory.global", True)
            saved = self.state.capture_state(["a_720nmc.csv"])
            self.assertIsNone(self.settings.value("drr/display_center_templates"))
            self.state.setFlag("memory.global", False)
            self.state.record_saved_state(saved)
            globals_ = json.loads(self.settings.value("drr/display_center_templates"))
            self.assertEqual(globals_["720"]["files"], ["a_720nmc.csv"])

        def test_borrowed_derivative_limits_unwind_on_a_different_source(self):
            saved = self.saved_view()
            self.assertTrue(self.state.restore_candidates(dict(exact=saved, center="720"), folder=self.temporary.name, files=["a_720nmc.csv"]))
            self.assertEqual(self.state.heatmap_params(self.second, "second").vmax, .3)
            changed_second = replace(self.second, Z=self.second.Z * 4)
            self.state.sync_cube(replace(self.raw, Z=self.raw.Z * 2), changed_second)
            self.state.restore_candidates({}, folder=self.temporary.name, files=["b_730nmc.csv"])
            self.assertTrue(self.state.second_auto_scale)
            self.assertNotEqual(self.state.heatmap_params(changed_second, "second").vmax, .3)

        def test_genuine_axis_fix_overrides_saved_viewport_only_on_its_axis(self):
            self.state.restore_candidates({}, folder=self.temporary.name, files=["a_720nmc.csv"])
            self.state.setNumber("axis.ymin", "-0.8")
            self.state.setFlag("fix.axis.ymin", True)
            saved = self.saved_view()
            saved["viewport"] = dict(xlim=[1.4, 2.6], ylim=[-.2, .2])
            self.assertTrue(self.state.restore_candidates(dict(exact=saved, center="720"), folder=self.temporary.name, files=["a_720nmc.csv"]))
            params = self.state.heatmap_params(self.raw)
            self.assertEqual(params.xlim, (1.4, 2.6))
            self.assertEqual(params.ylim, (-.8, .5))

        def test_valid_pending_derivative_limits_are_computed_when_product_arrives(self):
            self.state.sync_cube(self.raw, None)
            self.assertTrue(self.state.setFlag("log", True))
            saved = self.state.capture_state(["a_720nmc.csv"])
            self.assertTrue(saved["second_auto_pending"])
            restored = DrrDisplayState(settings=QSettings(self.temporary.name + "/pending.ini", QSettings.Format.IniFormat))
            self.addCleanup(restored.close)
            restored.sync_cube(self.raw, None)
            self.assertTrue(restored.restore_candidates(dict(exact=saved, center="720"), folder=self.temporary.name, files=["a_720nmc.csv"]))
            restored.sync_cube(self.raw, self.second)
            self.assertTrue(restored.second_auto_scale)
            self.assertGreater(restored.heatmap_params(self.second, "second").vmin, 0)

        def test_approved_qml_and_pinned_data_actions_at_compact_width(self):
            owner = RootBridge(self.state)
            engine = QQmlEngine()
            warnings = []
            engine.warnings.connect(lambda messages: warnings.extend(message.toString() for message in messages))
            component = QQmlComponent(engine, QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / "ui_qt/quick/qml/DrrParameters.qml")))
            self.assertTrue(component.isReady(), str(component.errors()))
            panel = component.createWithInitialProperties({"bridge": owner, "colors": colors})
            self.assertIsNotNone(panel, str(component.errors()))
            window = QQuickWindow()
            window.resize(320, 720)
            panel.setWidth(320)
            panel.setHeight(720)
            panel.setParentItem(window.contentItem())
            window.show()
            settle(60)
            def find(name):
                def visit(parent):
                    if parent.objectName() == name:
                        return parent
                    for child in parent.childItems():
                        result = visit(child)
                        if result is not None:
                            return result
                    return None
                return visit(panel)
            self.assertIsNotNone(find("drr_quick_display"))
            self.assertIsNotNone(find("drr_input_raw.vmin"))
            self.assertIsNotNone(find("drr_input_second.vmin"))
            def send_key(code, text="", modifier=Qt.KeyboardModifier.NoModifier):
                for kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                    QCoreApplication.sendEvent(window, QKeyEvent(kind, code, modifier, text))
            editor = find("drr_input_axis.xmin")
            editor.forceActiveFocus()
            send_key(Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
            for character in "1.234567":
                send_key(ord(character), character)
            self.assertTrue(self.state.editing)
            self.assertEqual(self.state.snapshot["numbers"]["axis.xmin"]["value"], 1)
            send_key(Qt.Key.Key_Return, "\r")
            settle()
            self.assertEqual(self.state.snapshot["numbers"]["axis.xmin"]["value"], 1.2346)
            self.assertIs(find("drr_input_axis.xmin"), editor)
            tabs = find("drr_native_parameter_tabs")
            tabs.setProperty("currentIndex", 1)
            settle()
            self.assertTrue(find("drr_native_processing").isVisible())
            self.assertTrue(find("drr_select_measurements").isVisible())
            self.assertIsNone(find("drr_select_background"))
            self.assertEqual(find("drr_background_settings").property("text"), "Background…")
            for name, action in (("drr_select_measurements", "select-files"), ("drr_clear_measurements", "clear-files"), ("drr_saved_results", "saved-results"), ("drr_background_settings", "background-settings")):
                button = find(name)
                point = button.mapToScene(button.boundingRect().center())
                global_point = window.mapToGlobal(point.toPoint())
                for kind, buttons in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton), (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
                    QCoreApplication.sendEvent(window, QMouseEvent(kind, point, global_point, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier))
                settle()
                self.assertIn(action, owner.calls)
            self.assertEqual(warnings, [])
            self.assertFalse(window.grabWindow().isNull())
            panel.setWidth(220)
            window.resize(220, 720)
            tabs.setProperty("currentIndex", 0)
            settle(30)
            for name in ("drr_select_measurements", "drr_clear_measurements", "drr_saved_results", "drr_background_settings", "drr_native_parameter_tabs", "drr_native_display_tab", "drr_native_processing_tab", "drr_input_axis.xmin", "drr_input_axis.xmax"):
                item = find(name)
                left = item.mapToItem(panel, item.boundingRect().topLeft()).x()
                right = item.mapToItem(panel, item.boundingRect().topRight()).x()
                self.assertGreaterEqual(left, -.5, name)
                self.assertLessEqual(right, 220.5, name)
            self.assertEqual(find("drr_background_settings").property("text"), "Background…")
            panel.setParentItem(None)
            panel.deleteLater()
            window.close()
            window.deleteLater()
            engine.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            settle()

        def test_close_cancels_delayed_changes_without_widget_imports(self):
            events = []
            self.state.parameterChanged.connect(lambda *args: events.append(args))
            self.state.stepNumber("axis.xmin", 1)
            self.state.close()
            settle(200)
            self.assertEqual(events, [])
            self.assertNotIn("PySide6.QtWidgets", sys.modules)
            self.assertNotIn("ui_qt.main_window", sys.modules)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_isolated()) if "--isolated" in sys.argv else unittest.main()
