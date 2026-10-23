"""Guard redundant reads/reductions on page load and export paths."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from core import data_io
from core.loader import DataCube
from ui_qt.common import ExportOptions, LoadedState, LoadOptions
from ui_qt.controllers_power import PowerController
from ui_qt.main_window import MainWindow


class PagePerformanceTests(unittest.TestCase):
    def test_power_roles_share_validated_catalog_and_refresh_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = [f"sweep_{index}.csv" for index in range(4)]
            for name in names:
                Path(tmp, name).write_text("Power_uW,600,620,640\n1,10,20,30\n2,20,40,60\n")
            sources = data_io.get_power_series_sources(tmp, names)
            signatures = tuple((name, Path(tmp, name).stat().st_size,
                                Path(tmp, name).stat().st_mtime_ns) for name in names)
            options = LoadOptions("Power Dependent", tmp, names, [], False, "", "", False,
                power_group_key=data_io.power_sweep_source_key(names[0]),
                power_role_group_keys=(data_io.power_sweep_source_key(names[1]),),
                power_source_catalog=tuple(sources.values()), power_source_signatures=signatures)
            owner = SimpleNamespace(_provenance_records_for_load=lambda _: ())
            with patch.object(data_io.pd, "read_csv", wraps=data_io.pd.read_csv) as read:
                loaded = MainWindow._load_task_data(owner, options, progress=Mock(), log=Mock())
                self.assertEqual(read.call_count, 2)  # selected spectral data only
                self.assertTrue(all("usecols" not in call.kwargs for call in read.call_args_list))
                read.reset_mock()
                again = MainWindow._load_task_data(owner, options, progress=Mock(), log=Mock())
                read.assert_not_called()
                np.testing.assert_array_equal(again.cube.Z, loaded.cube.Z)
            changed = Path(tmp, names[1])
            changed.write_text("Power_uW,600,620,640\n3,30,60,90\n4,40,80,120\n")
            stamp = changed.stat().st_mtime_ns + 1_000_000_000
            os.utime(changed, ns=(stamp, stamp))
            with patch.object(data_io.pd, "read_csv", wraps=data_io.pd.read_csv) as read:
                refreshed = MainWindow._load_task_data(owner, options, progress=Mock(), log=Mock())
                self.assertEqual(read.call_count, 2)  # one changed catalog + its data
            result = refreshed.power_results[data_io.power_sweep_source_key(names[1])]
            self.assertEqual(result.sources[result.group_key].power_values, (3., 4.))
            np.testing.assert_array_equal(result.cube.gate, [3., 4.])

    def test_power_background_reuses_reduction_until_data_or_revision_changes(self):
        owner = SimpleNamespace(power_background_spin=Mock(), power_background_auto_chk=Mock())
        owner.power_background_auto_chk.isChecked.return_value = True
        controller = PowerController(owner)
        cube = DataCube(np.array([1., 2.]), np.array([0., 1.]), np.arange(4.).reshape(2, 2), "Power", "", "")
        from ui_qt.controllers_power import estimate_constant_background
        with patch("ui_qt.controllers_power.estimate_constant_background", wraps=estimate_constant_background) as estimate:
            first = controller._power_background_value({"KK": cube})
            self.assertEqual(controller._power_background_value({"KK": cube}), first)
            self.assertEqual(estimate.call_count, 1)
            cube.Z = cube.Z + 10
            self.assertAlmostEqual(controller._power_background_value([cube]), first + 10)
            cube.Z += 10
            cube.revision = 1
            self.assertAlmostEqual(controller._power_background_value([cube]), first + 20)
            self.assertEqual(estimate.call_count, 3)
            owner.power_background_auto_chk.isChecked.return_value = False
            owner.power_background_spin.value.return_value = 7
            self.assertEqual(controller._power_background_value([cube]), 7)
            self.assertEqual(estimate.call_count, 3)

    def test_pl_dat_export_reads_once_and_preserves_display_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = DataCube(np.array([1., 2.]), np.array([0., 1.]), np.ones((2, 2)), "Custom field", "", "")
            cube.gate_unit, cube.y_axis_semantic = "T", "field"
            loaded = LoadedState(mode="PL", folder=tmp, primary_file="sample.dat", cube=cube)
            options = ExportOptions(mode="PL", params=None, params_linear=object(), params_log=object())
            paths = {key: Path(tmp, key + suffix) for key, suffix in
                     (("png_linear", ".png"), ("png_log", ".png"), ("dat", ".dat"))}
            parsed = DataCube(cube.energy, cube.gate, cube.Z.copy(), "Gate", "", "")
            with patch.object(data_io, "load_pl_cube", return_value=parsed) as load, patch(
                    "ui_qt.main_window.export_pl_pngs_and_dat", return_value=paths) as export:
                MainWindow._export_task(SimpleNamespace(), loaded, options, progress=Mock(), log=Mock())
            load.assert_called_once()
            args = export.call_args.kwargs
            self.assertIs(args["cube_linear"], args["cube_log"])
            self.assertEqual(args["cube_linear"].gate_label, "Custom field")
            self.assertEqual(args["cube_log"].gate_unit, "T")
            np.testing.assert_array_equal(parsed.Z, cube.Z)

    def test_power_compare_plot_key_and_render_share_background_reduction(self):
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        from tests.ui_test_helpers import dispose_owned_window
        from ui_qt.controllers_power import estimate_constant_background
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp, patch("ui_qt.main_window.QSettings",
                return_value=QSettings(str(Path(tmp, "settings.ini")), QSettings.IniFormat)), patch.object(
                MainWindow, "_restore_last_folder", lambda _: None), patch.object(
                MainWindow, "_schedule_automatic_update_check", lambda _: None):
            window = MainWindow()
            try:
                cubes = [DataCube(np.array([1., 2.]), np.array([0., 1.]),
                                  np.arange(4.).reshape(2, 2) + offset, "Power", "", "")
                         for offset in (10., -10.)]
                window.loaded = LoadedState(mode="Power Dependent", folder=tmp, cube=cubes[0])
                window._power_result_cache = {key: SimpleNamespace(cube=cube)
                                              for key, cube in zip(("KK", "KKp"), cubes)}
                for combo, key in ((window.power_kk_group_combo, "KK"), (window.power_kkp_group_combo, "KKp")):
                    blocked = combo.blockSignals(True)
                    combo.clear(); combo.addItem(key, key)
                    combo.blockSignals(blocked)
                for check in (window.power_compare_chk, window.power_background_auto_chk):
                    blocked = check.blockSignals(True)
                    check.setChecked(True)
                    check.blockSignals(blocked)
                with patch("ui_qt.controllers_power.estimate_constant_background", wraps=estimate_constant_background) as estimate:
                    for _ in range(2):
                        window._current_plot_params_key("Power Dependent")
                        value = window.power_controller._power_background_value(cubes)
                    self.assertEqual(estimate.call_count, 1)
                    self.assertEqual(value, estimate_constant_background(cubes, percentile=1.0))
            finally:
                dispose_owned_window(window)
