from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QToolButton

from core.loader import DataCube
from ui_qt.common import LoadedState
from ui_qt.controllers_compare import CompareController
from ui_qt.controllers_mcd import McdController
from ui_qt.main_window import MainWindow


def _cube(title: str, offset: float) -> DataCube:
    energy = np.linspace(1.0, 3.0, 5)
    gate = np.array([0.0, 1.0, 2.0])
    z = np.vstack([energy + offset + gate_value for gate_value in gate])
    return DataCube(energy, gate, z, "Gate", title, "a.u.")


class _CompareOwner:
    def __init__(self, *, vp: bool = False) -> None:
        self.last_plotted_mode = "Compare"
        self._load_in_progress = False
        self._plot_redraw_pending = set()
        self._cmp_active_cubes = {"KK": _cube("KK", 0.0), "KKp": _cube("KKp", 1.0)}
        self._cmp_gate_lines = {}
        self._shown_draw_identity = ("source", 1)
        self.loaded = SimpleNamespace(mode="Compare")
        self._shown_source_identity_for_loaded = lambda _loaded: ("source", 1)
        self._last_plot_params_key = self._key(1.0)
        self._canvas_draws = 0
        self.canvas = SimpleNamespace(draw_idle=self._draw_idle)
        self.cmp_spins = {key: QDoubleSpinBox() for key in ("xmin", "xmax", "ymin", "ymax", "gate", "vmin", "vmax")}
        self.cmp_spins["gate"].setRange(-10.0, 10.0)
        self.cmp_spins["gate"].setValue(1.0)
        self._cmp_heatmap_axes = {}
        self._cmp_linecut_ax = None
        self.cmp_view_vp_btn = QToolButton()
        self.cmp_view_vp_btn.setCheckable(True)
        self.cmp_view_vp_btn.setChecked(vp)
        self._pending_range_refresh = {}
        self.cmp_log_chk = QCheckBox()
        self.cmp_vp_background_spin = QDoubleSpinBox()
        self.cmp_vp_auto_background_chk = QCheckBox()
        self.cmp_channel_combos = {}
        self.cmp_show_checks = {}
        self._invalidate_export_move_sources = Mock()
        self._cmp_update_assignment_summary = Mock()
        self._schedule_plot_redraw = Mock()

    def _draw_idle(self) -> None:
        self._canvas_draws += 1

    def _key(self, gate: float) -> tuple:
        key = list(range(19))
        key[0] = "Compare"
        key[15] = float(gate)
        return tuple(key)

    def _current_plot_params_key(self, _mode: str) -> tuple:
        return self._key(self.cmp_spins["gate"].value())


class CompareGateLatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _owner_with_artists(self, *, vp: bool = False):
        owner = _CompareOwner(vp=vp)
        figure = Figure()
        heat = figure.add_subplot(211)
        line = figure.add_subplot(212)
        cubes = owner._cmp_active_cubes
        if vp:
            owner._cmp_active_cubes = {"VP": _cube("VP", 2.0)}
            cubes = owner._cmp_active_cubes
            line.plot(cubes["VP"].energy, cubes["VP"].Z[1])
        else:
            for key, cube in cubes.items():
                line.plot(cube.energy, cube.Z[1], label=key)
        owner._cmp_heatmap_axes = {key: heat for key in cubes}
        owner._cmp_linecut_ax = line
        owner._last_plot_params_key = owner._key(1.0)
        return owner, heat, line

    def test_intensity_gate_update_reuses_axes_and_line_artists(self) -> None:
        owner, heat, line = self._owner_with_artists()
        controller = CompareController(owner)
        owner.cmp_spins["gate"].setValue(1.8)
        self.assertTrue(controller._update_cmp_gate_only())
        self.assertIs(owner._cmp_heatmap_axes["KK"], heat)
        self.assertIs(owner._cmp_linecut_ax, line)
        np.testing.assert_allclose(line.lines[0].get_ydata(), owner._cmp_active_cubes["KK"].Z[2])
        self.assertGreater(owner._canvas_draws, 0)

    def test_vp_gate_update_reuses_linecut_and_gate_artist(self) -> None:
        owner, heat, line = self._owner_with_artists(vp=True)
        controller = CompareController(owner)
        owner.cmp_spins["gate"].setValue(0.2)
        self.assertTrue(controller._update_cmp_gate_only())
        self.assertIs(owner._cmp_heatmap_axes["VP"], heat)
        np.testing.assert_allclose(line.lines[0].get_ydata(), owner._cmp_active_cubes["VP"].Z[0])
        self.assertEqual(line.get_title(), "VP Linecut @ 0 V")

    def test_gate_fast_path_rejects_pending_full_redraw(self) -> None:
        owner, _heat, _line = self._owner_with_artists()
        owner._plot_redraw_pending.add("Compare")
        controller = CompareController(owner)
        self.assertFalse(controller._update_cmp_gate_only())

    def test_gate_spin_callback_uses_fast_path_without_scheduling_full_redraw(self) -> None:
        owner, _heat, _line = self._owner_with_artists()
        controller = CompareController(owner)
        owner.cmp_spins["gate"].setValue(1.8)
        with patch.object(CompareController, "_cmp_update_assignment_summary", return_value=None):
            controller._on_cmp_plot_param_changed(owner.cmp_spins["gate"])
        owner._schedule_plot_redraw.assert_not_called()

    def test_auto_background_cache_rejects_cube_and_z_replacements(self) -> None:
        owner, _heat, _line = self._owner_with_artists()
        owner.cmp_vp_auto_background_chk.setChecked(True)
        controller = CompareController(owner)
        with patch("ui_qt.controllers_compare.estimate_constant_background",
                   side_effect=(10.0, 20.0, 30.0)) as estimate:
            self.assertEqual(controller._cmp_background_value(owner._cmp_active_cubes), 10.0)
            owner._cmp_active_cubes["KK"].Z = owner._cmp_active_cubes["KK"].Z.copy()
            self.assertEqual(controller._cmp_background_value(owner._cmp_active_cubes), 20.0)
            owner._cmp_active_cubes["KK"] = _cube("KK replacement", 9.0)
            self.assertEqual(controller._cmp_background_value(owner._cmp_active_cubes), 30.0)
        self.assertEqual(estimate.call_count, 3)


class McdRedrawLatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_range_color_burst_coalesces_through_shared_scheduler(self) -> None:
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            window = MainWindow()
        window.loaded = LoadedState(mode="MCD", folder="")
        try:
            with patch.object(window, "_plot_mode") as plot_mode, patch.object(
                window, "_refresh_automatic_ranges", return_value=None
            ):
                for value in (0.1, 0.2, 0.3):
                    window.mcd_spins["xmin"].setValue(value)
                window.mcd_cmap.setCurrentIndex((window.mcd_cmap.currentIndex() + 1) % max(1, window.mcd_cmap.count()))
                timer = window._plot_redraw_timers.get("MCD")
                self.assertIsNotNone(timer)
                self.assertTrue(timer.isActive())
                self.app.processEvents()
                from PySide6.QtTest import QTest
                QTest.qWait(120)
                self.app.processEvents()
                plot_mode.assert_called_once_with("MCD")
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
