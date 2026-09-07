from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject

from ui_qt.main_window import LoadedState, MainWindow
from core.loader import DataCube
from core.plotting import downsample_cube_for_display
from ui_qt.controllers_drr import DrrController
from ui_qt.controllers_shg import ShgController
from ui_qt.controllers_pl import PlController
from ui_qt.controllers_power import PowerController
from ui_qt.controllers_mcd import McdController


class _PowerOwner:
    current_folder = "experiment"

    def __init__(self) -> None:
        self._power_sources_cache = None
        self._power_sources_cache_files = ()
        self.available_files = ["one.csv"]


class _PowerCheck:
    def __init__(self, checked: bool) -> None:
        self.checked = checked

    def isChecked(self) -> bool:
        return self.checked


class _PowerSpin:
    def __init__(self) -> None:
        self.enabled = True

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled


class _PowerBackgroundOwner:
    loaded = None

    def __init__(self) -> None:
        self.power_background_auto_chk = _PowerCheck(True)
        self.power_background_spin = _PowerSpin()

    def _invalidate_export_move_sources(self) -> None:
        pass


class _RecordingPowerController(PowerController):
    def __init__(self, owner, seen) -> None:
        super().__init__(owner)
        object.__setattr__(self, "_seen", seen)

    def _on_power_plot_param_changed(self, sender=None) -> None:
        self._seen.append(sender)


class _ShgOwner:
    def __init__(self) -> None:
        self.loaded = LoadedState(mode="SHG Processing", folder="")
        self.summary_updates = 0
        self.redraws = []

    def _shg_update_summary(self) -> None:
        self.summary_updates += 1

    def _schedule_plot_redraw(self, mode: str, delay_ms: int = 90) -> None:
        self.redraws.append((mode, delay_ms))


class _ShgQtOwner(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = None


class _McdLifecycleOwner(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.current_folder = "folder-a"
        self._is_closing = False


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
    def test_mcd_lifecycle_state_is_controller_local_and_timers_are_parented(self) -> None:
        QCoreApplication.instance() or QCoreApplication([])
        owner = _McdLifecycleOwner()
        controller = McdController(owner)

        controller._mcd_reapply_pending = True
        controller._mcd_angle_generation = 7
        controller._mcd_source_observations = {"source": (1, 2)}

        self.assertTrue(controller._mcd_reapply_pending)
        self.assertEqual(controller._mcd_angle_generation, 7)
        self.assertIs(controller._mcd_auto_apply_timer.parent(), owner)
        self.assertIs(controller._mcd_center_refresh_timer.parent(), owner)
        self.assertIs(controller._mcd_source_stability_timer.parent(), owner)
        for name in (
            "_mcd_reapply_pending", "_mcd_angle_generation",
            "_mcd_source_observations", "_mcd_auto_apply_timer",
            "_mcd_center_refresh_timer", "_mcd_source_stability_timer",
        ):
            self.assertNotIn(name, owner.__dict__)
        controller._shutdown_mcd_lifecycle()
        self.assertFalse(controller._mcd_auto_apply_timer.isActive())
        self.assertFalse(controller._mcd_center_refresh_timer.isActive())
        self.assertFalse(controller._mcd_source_stability_timer.isActive())
        owner.deleteLater()

    def test_mcd_source_stability_timeout_rejects_stale_source_folder_and_close(self) -> None:
        owner = _McdLifecycleOwner()
        controller = McdController(owner)
        started = []
        selected = ["source.csv"]
        owner._load_in_progress = False
        owner.mcd_files = object()
        owner._selected = lambda _widget: list(selected)
        owner._start_load = lambda mode: started.append(mode)
        owner._status = lambda _text: None
        controller._mcd_source_stability_source = "source.csv"
        controller._mcd_source_stability_folder = "folder-a"
        controller._mcd_source_stability_generation = 4
        controller._on_mcd_source_stability_timeout(4)
        self.assertEqual(started, ["MCD"])

        selected[:] = ["other.csv"]
        controller._mcd_source_stability_source = "source.csv"
        controller._mcd_source_stability_folder = "folder-a"
        controller._mcd_source_stability_generation = 5
        controller._on_mcd_source_stability_timeout(5)
        owner.current_folder = "folder-b"
        selected[:] = ["source.csv"]
        controller._mcd_source_stability_source = "source.csv"
        controller._mcd_source_stability_folder = "folder-a"
        controller._mcd_source_stability_generation = 6
        controller._on_mcd_source_stability_timeout(6)
        owner._is_closing = True
        controller._mcd_source_stability_source = "source.csv"
        controller._mcd_source_stability_folder = "folder-b"
        controller._mcd_source_stability_generation = 7
        controller._on_mcd_source_stability_timeout(7)
        self.assertEqual(started, ["MCD"])
        owner.deleteLater()

    def test_stale_mcd_angle_result_is_discarded_by_controller_generation(self) -> None:
        owner = _McdLifecycleOwner()
        owner.mcd_files = object()
        owner._selected = lambda _widget: ["source.csv"]
        controller = McdController(owner)
        controller._mcd_angle_generation = 3
        controller._on_mcd_angles_result(
            ("source.csv", (1, 2), (10.0, 50.0)),
            2,
            "folder-a",
            "source-key",
        )
        self.assertEqual(controller._mcd_angle_cache, {})
        owner.deleteLater()

    def test_phase_36a_processing_helpers_are_controller_owned(self) -> None:
        expected = {
            "_update_mcd_selection_summary",
            "_open_mcd_source_dialog",
            "_edit_mcd_source",
            "_clear_mcd_source",
            "_mcd_refresh_sources",
            "_mcd_detect_available_angles",
            "_apply_mcd_detected_angles",
            "_mcd_settings_from_ui",
            "_format_mcd_background_ranges",
            "_update_mcd_background_preview",
            "_suggest_mcd_background_ranges",
            "_show_mcd_background_suggestion_dialog",
            "_mcd_result_cache_key",
            "_cached_mcd_result",
            "_store_cached_mcd_result",
        }
        self.assertTrue(expected.issubset(vars(McdController)))

    def test_power_catalog_is_reused_until_candidate_list_changes(self) -> None:
        owner = _PowerOwner()
        controller = PowerController(owner)
        sentinel = {"group": object()}
        with patch("core.data_io.get_power_series_sources", return_value=sentinel) as discover:
            self.assertIs(controller._power_current_sources(), sentinel)
            self.assertIs(controller._power_current_sources(), sentinel)
            self.assertEqual(discover.call_count, 1)
            owner.available_files.append("two.csv")
            self.assertIs(controller._power_current_sources(), sentinel)
            self.assertEqual(discover.call_count, 2)

    def test_power_background_toggle_propagates_signal_source(self) -> None:
        owner = _PowerBackgroundOwner()
        seen = []
        controller = _RecordingPowerController(owner, seen)
        controller._on_power_background_mode_changed(True)
        self.assertEqual(seen, [owner.power_background_auto_chk])

    def test_stale_shg_result_is_discarded(self) -> None:
        owner = _ShgOwner()
        controller = ShgController(owner)
        controller._shg_reprocess_generation = 2
        controller._shg_reprocess_key = (2, object(), object(), False)
        controller._on_shg_reprocessed(1, ("stale", None, None, None, None))
        self.assertIsNone(owner.loaded.shg_result)
        self.assertEqual(owner.summary_updates, 0)
        self.assertEqual(owner.redraws, [])

    def test_shg_reprocess_state_is_controller_local(self) -> None:
        owner = _ShgOwner()
        controller = ShgController(owner)

        controller._shg_reprocess_generation = 3
        controller._shg_reprocess_key = (3, object(), object(), False)
        controller._shg_loaded_processing_key = controller._shg_reprocess_key

        self.assertEqual(controller._shg_reprocess_generation, 3)
        self.assertEqual(controller._shg_reprocess_key[0], 3)
        self.assertNotIn("_shg_reprocess_generation", owner.__dict__)
        self.assertNotIn("_shg_reprocess_key", owner.__dict__)
        self.assertNotIn("_shg_loaded_processing_key", owner.__dict__)

    def test_shg_reprocess_timer_is_parented_to_main_window(self) -> None:
        QCoreApplication.instance() or QCoreApplication([])
        owner = _ShgQtOwner()
        controller = ShgController(owner)

        controller._request_shg_reprocess()

        self.assertIs(controller._shg_reprocess_timer.parent(), owner)
        controller._shg_reprocess_timer.stop()
        owner.deleteLater()

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
        owner.drr_sg_poly_spin = _Value(2)
        owner.drr_sg_window_spin = _Value(3)
        owner.drr_derivative_combo = _TextValue("dE")
        controller = DrrController(owner)
        with patch("ui_qt.controllers_drr.apply_sg_derivative_energy", return_value=(base, 3)) as apply:
            first = controller._drr_cube_with_metadata()
            second = controller._drr_cube_with_metadata()
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


class _TextValue:
    def __init__(self, value):
        self._value = value

    def currentText(self):
        return self._value


if __name__ == "__main__":
    unittest.main()
