from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from ui_qt.main_window import LoadedState, MainWindow
from core.loader import DataCube
from core.plotting import downsample_cube_for_display
from ui_qt.controllers_shg import ShgController
from ui_qt.controllers_pl import PlController


class _PowerOwner:
    current_folder = "experiment"

    def __init__(self) -> None:
        self._power_sources_cache = None
        self._power_sources_cache_files = ()
        self.files = ["one.csv"]

    def _power_candidate_files(self):
        return list(self.files)


class _ShgOwner:
    def __init__(self) -> None:
        self.loaded = LoadedState(mode="SHG Processing", folder="")
        self._shg_reprocess_generation = 2
        self._shg_reprocess_key = (2, object(), object(), False)
        self.summary_updates = 0
        self.redraws = []

    def _shg_update_summary(self) -> None:
        self.summary_updates += 1

    def _schedule_plot_redraw(self, mode: str, delay_ms: int = 90) -> None:
        self.redraws.append((mode, delay_ms))


class _Status:
    def __init__(self, value: str) -> None:
        self.value = value

    def text(self) -> str:
        return self.value

    def setText(self, value: str) -> None:
        self.value = value


class _PlFitOwner:
    def __init__(self) -> None:
        self.loaded = LoadedState(mode="PL", folder="", primary_file="new.csv")
        self.last_plotted_mode = "PL"
        self._pl_last_plot_cube = object()
        self.pl_fit_status = _Status("Fitting Lorentz peaks…")
        self._pl_fit_generation = 4


class ResponsivenessTests(unittest.TestCase):
    def test_power_catalog_is_reused_until_candidate_list_changes(self) -> None:
        owner = _PowerOwner()
        sentinel = {"group": object()}
        with patch("ui_qt.main_window.data_io.get_power_series_sources", return_value=sentinel) as discover:
            self.assertIs(MainWindow._power_current_sources(owner), sentinel)
            self.assertIs(MainWindow._power_current_sources(owner), sentinel)
            self.assertEqual(discover.call_count, 1)
            owner.files.append("two.csv")
            self.assertIs(MainWindow._power_current_sources(owner), sentinel)
            self.assertEqual(discover.call_count, 2)

    def test_stale_shg_result_is_discarded(self) -> None:
        owner = _ShgOwner()
        controller = ShgController(owner)
        controller._on_shg_reprocessed(1, ("stale", None, None, None, None))
        self.assertIsNone(owner.loaded.shg_result)
        self.assertEqual(owner.summary_updates, 0)
        self.assertEqual(owner.redraws, [])

    def test_current_generation_pl_fit_completion_clears_stale_status(self) -> None:
        owner = _PlFitOwner()
        controller = PlController(owner)
        controller._on_pl_fit_finished(
            4, object(), "old.csv", 0.0, 0.0, 1, np.array([1.0]), np.array([1.0, 0.0, 1.0, 1.0, 0.1])
        )
        self.assertEqual(owner.pl_fit_status.text(), "Fit discarded: PL source changed.")

    def test_drr_derivative_cube_is_cached_by_cube_and_settings(self) -> None:
        owner = MainWindow.__new__(MainWindow)
        base = DataCube(np.arange(5.0), np.arange(2.0), np.ones((2, 5)), "Gate", "DRR", "DR/R")
        owner.loaded = LoadedState(mode="DRR", folder="", cube=base)
        owner._drr_derivative_cache = {}
        owner._drr_derivative_value = lambda: 1
        owner.drr_sg_poly_spin = _Value(2)
        owner.drr_sg_window_spin = _Value(3)
        owner._enforce_drr_sg_constraints = lambda show_status: 3
        with patch("ui_qt.main_window.apply_sg_derivative_energy", return_value=(base, 3)) as apply:
            first = MainWindow._drr_cube_with_metadata(owner)
            second = MainWindow._drr_cube_with_metadata(owner)
        self.assertIs(first[0], second[0])
        self.assertEqual(apply.call_count, 1)

    def test_display_downsampling_is_bounded_and_preserves_source_cube(self) -> None:
        energy = np.arange(1000.0)
        gate = np.arange(800.0)
        z = np.arange(800000.0).reshape(800, 1000)
        cube = DataCube(energy, gate, z, "Gate", "DRR", "DR/R")
        display = downsample_cube_for_display(cube, max_points=10_000)
        self.assertLessEqual(display.Z.size, 10_000)
        self.assertEqual(display.energy[[0, -1]].tolist(), [0.0, 999.0])
        self.assertEqual(display.gate[[0, -1]].tolist(), [0.0, 799.0])
        self.assertIs(cube.Z, z)
        self.assertEqual(cube.Z.shape, (800, 1000))


class _Value:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


if __name__ == "__main__":
    unittest.main()
