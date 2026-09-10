import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.data_io import PowerSeriesResult, load_power_sweep_csv
from core.export import _power_record_header_lines
from core.loader import DataCube
from core.power_combine import combine_power_sweeps, save_combined_power_sweep
from core.processing import PowerSweepPoint


def sweep(name, powers, values, energy=(1., 2.), stages=None):
    records = tuple(PowerSweepPoint(name, p, stages[i] if stages else None, i + 2)
                    for i, p in enumerate(powers))
    return PowerSeriesResult(DataCube(np.array(energy), np.array(powers), np.array(values),
                                     "Power (uW)", name, "PL"), name, records, {})


class PowerCombineTests(unittest.TestCase):
    def setUp(self):
        self.a = sweep("low.csv", [1., 2.], [[1., 2.], [3., 4.]], stages=[1., 2.])
        self.b = sweep("high.csv", [2., 2.00001, 4.], [[5., 6.], [7., 8.], [9., 10.]],
                       stages=[3., 4., 5.])

    def test_duplicate_choices_and_nearby_powers(self):
        for policy, expected in [("first", [3., 4.]), ("second", [5., 6.]),
                                 ("average", [4., 5.])]:
            result = combine_power_sweeps(self.a, self.b, duplicate_policy=policy)
            np.testing.assert_array_equal(result.cube.gate, [1., 2., 2.00001, 4.])
            np.testing.assert_array_equal(result.cube.Z[1], expected)
            if policy == "average":
                self.assertIsNone(result.records[1].stage)
                self.assertEqual(len(json.loads(result.records[1].source_provenance)["sources"]), 2)

    def test_saved_table_reloads_and_exports_provenance(self):
        result = combine_power_sweeps(self.a, self.b, duplicate_policy="average")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "combined.csv"
            save_combined_power_sweep(path, result)
            cube, records = load_power_sweep_csv(folder, path.name)
            np.testing.assert_allclose(cube.Z, result.cube.Z)
            np.testing.assert_array_equal(cube.gate, result.cube.gate)
            self.assertEqual(records[1].source_provenance, result.records[1].source_provenance)
            headers = "\n".join(_power_record_header_lines(records))
            self.assertIn("low.csv", headers)
            self.assertIn("high.csv", headers)
            from core.export import export_power_series_png_and_dat
            from core.plotting import HeatmapParams
            params = HeatmapParams(title="Combined", xlabel="Energy", ylabel="Power", cbar_label="PL",
                                   vmin=0., vmax=10., xlim=(1., 2.), ylim=(1., 4.))
            paths = export_power_series_png_and_dat(
                folder, cube=cube, params=params, records=records,
                group_key="csv::combined.csv", y_axis_log=False)
            self.assertTrue(paths["png"].is_file())
            metadata = json.loads(paths["dat"].with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["processing"]["source_provenance"][1], records[1].source_provenance)
            with self.assertRaises(FileExistsError):
                save_combined_power_sweep(path, result)

    def test_energy_interpolation_stays_in_shared_range(self):
        a = sweep("a", [1.], [[0., 10., 20., 30.]], energy=(1., 2., 3., 4.))
        b = sweep("b", [2.], [[5., 15., 25.]], energy=(1.5, 2.5, 3.5))
        result = combine_power_sweeps(a, b)
        np.testing.assert_array_equal(result.cube.energy, [2., 3.])
        np.testing.assert_allclose(result.cube.Z, [[10., 20.], [10., 20.]])

    def test_invalid_sources_and_disjoint_energy(self):
        with self.assertRaises(ValueError):
            combine_power_sweeps(self.a, self.a)
        b = sweep("b", [3.], [[1., 2.]], energy=(5., 6.))
        with self.assertRaises(ValueError):
            combine_power_sweeps(self.a, b)


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class PowerWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui_qt.main_window import MainWindow
        with patch.object(MainWindow, "_restore_last_folder", lambda self: None):
            self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_filename_series_are_opt_in_and_hidden_selections_removed(self):
        from PySide6.QtWidgets import QCheckBox
        from PySide6.QtCore import Qt
        from ui_qt.source_picker_dialog import SourcePickerDialog
        with tempfile.TemporaryDirectory() as folder:
            self.window.current_folder = folder
            root = Path(folder)
            (root / 'sweep.csv').write_text('Power_uW,1.4,1.5\n1,2,3\n2,4,5\n')
            for power in (1, 2):
                (root / f'PL_{power}uW.csv').write_text('Gate,1.4,1.5\n0,2,3\n')
            def inspect(dialog):
                checkbox = next(c for c in dialog.findChildren(QCheckBox) if c.text() == 'Include filename-based series')
                self.assertFalse(checkbox.isChecked())
                self.assertEqual(dialog.source_list.count(), 1)
                checkbox.setChecked(True)
                self.assertEqual(dialog.source_list.count(), 2)
                for i in range(dialog.source_list.count()):
                    item = dialog.source_list.item(i)
                    if not item.data(Qt.UserRole).startswith('csv::'):
                        item.setCheckState(Qt.Checked)
                self.assertTrue(dialog.ok_button.isEnabled())
                checkbox.setChecked(False)
                self.assertEqual(dialog.source_list.count(), 1)
                self.assertFalse(dialog.ok_button.isEnabled())
                return 0
            with patch.object(SourcePickerDialog, 'exec', inspect):
                self.window.power_controller._power_choose_combination()

    def test_add_multiple_to_chosen_panel_and_keep_across_filter(self):
        from PySide6.QtWidgets import QListWidget, QPushButton, QComboBox, QAbstractItemView
        from core.data_io import PowerSeriesSource
        from ui_qt.source_picker_dialog import SourcePickerDialog
        c = self.window.power_controller
        def inspect(dialog):
            chosen = dialog.findChild(QListWidget, 'power_chosen_files')
            add = dialog.findChild(QPushButton, 'power_add_selected')
            remove = dialog.findChild(QPushButton, 'power_remove_selected')
            clear = dialog.findChild(QPushButton, 'power_clear_selected')
            self.assertEqual(dialog.source_list.selectionMode(), QAbstractItemView.ExtendedSelection)
            dialog.source_list.selectAll()
            add.click()
            self.assertEqual(chosen.count(), 2)
            add.click()
            self.assertEqual(chosen.count(), 2)
            self.assertEqual(dialog.ok_button.text(), 'Review combination…')
            status = next(combo for combo in dialog.findChildren(QComboBox) if combo.findText('Processed') >= 0)
            self.assertEqual(status.currentText(), 'All')
            status.setCurrentText('Processed')
            self.assertEqual(dialog.source_list.count(), 0)
            self.assertEqual(chosen.count(), 2)
            self.assertTrue(dialog.ok_button.isEnabled())
            chosen.item(0).setSelected(True)
            remove.click()
            self.assertEqual(chosen.count(), 1)
            self.assertEqual(dialog.ok_button.text(), 'Open sweep')
            clear.click()
            self.assertEqual(chosen.count(), 0)
            self.assertFalse(dialog.ok_button.isEnabled())
            return 0
        with patch.object(type(c), '_power_current_sources', return_value={k: PowerSeriesSource(k, k, 'table') for k in ('a', 'b')}), \
             patch.object(SourcePickerDialog, 'exec', inspect):
            c._power_choose_combination()

    def test_picker_uses_current_combo_over_stale_picker_memory(self):
        from PySide6.QtCore import Qt
        from core.data_io import PowerSeriesSource
        from ui_qt.source_picker_dialog import SourcePickerDialog
        c = self.window.power_controller
        c._power_picker_selection = ('csv::old.csv',)
        c.power_group_combo.clear()
        c.power_group_combo.addItem('— Select power sweep —', '')
        c.power_group_combo.addItem('Current', 'csv::current.csv')
        c.power_group_combo.setCurrentIndex(c.power_group_combo.findData('csv::current.csv'))
        seen = []
        def inspect(dialog):
            seen.append([dialog.source_list.item(i).checkState() == Qt.Checked
                         for i in range(dialog.source_list.count())])
            return 0
        with patch.object(type(c), '_power_current_sources', return_value={
                'csv::current.csv': PowerSeriesSource('csv::current.csv', 'Current', 'table')}), \
             patch.object(SourcePickerDialog, 'exec', inspect):
            c._power_choose_combination()
        self.assertEqual(seen, [[True]])

    def test_role_pickers_are_single_sweep_and_load_only_when_pair_complete(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QAbstractItemView
        from core.data_io import PowerSeriesSource
        from ui_qt.source_picker_dialog import SourcePickerDialog
        c = self.window.power_controller
        self.window.power_compare_chk.setChecked(True)
        sources = {key: PowerSeriesSource(key, key, 'table') for key in ('a', 'b')}
        results = {key: sweep(key, [1., 2.], [[1., 2.], [3., 4.]]) for key in sources}
        calls = []
        def pick(dialog):
            self.assertEqual(dialog.source_list.selectionMode(), QAbstractItemView.SingleSelection)
            self.assertEqual(dialog.source_list.count(), 2 if not calls else 1)
            dialog.source_list.setCurrentRow(0)
            dialog.findChild(type(dialog.ok_button), 'power_add_selected').click()
            calls.append(dialog.windowTitle())
            return 1
        with patch.object(type(c), '_power_current_sources', return_value=sources), \
             patch.object(type(c), '_power_load_group_result', side_effect=lambda key: results[key]), \
             patch.object(type(self.window), '_start_load') as load, \
             patch.object(SourcePickerDialog, 'exec', pick):
            c._power_choose_dataset('KK')
            self.assertEqual(load.call_count, 0)
            c._power_choose_dataset('KKp')
        self.assertEqual(load.call_count, 1)
        self.assertNotEqual(c._power_role_group_key('KK'), c._power_role_group_key('KKp'))

    def test_cancelled_role_picker_leaves_existing_assignment_untouched(self):
        from core.data_io import PowerSeriesSource
        from ui_qt.source_picker_dialog import SourcePickerDialog
        c = self.window.power_controller
        sources = {key: PowerSeriesSource(key, key, 'table') for key in ('a', 'b')}
        with patch.object(type(c), '_power_current_sources', return_value=sources):
            c._power_refresh_groups()
            c.power_kk_group_combo.setCurrentIndex(c.power_kk_group_combo.findData('a'))
            before = c.power_kk_group_combo.currentData()
            with patch.object(SourcePickerDialog, 'exec', lambda _dialog: 0):
                c._power_choose_dataset('KK')
        self.assertEqual(c.power_kk_group_combo.currentData(), before)

    def test_multiselect_back_preserves_settings_and_selection(self):
        from PySide6.QtCore import Qt
        from core.data_io import PowerSeriesSource
        from ui_qt.source_picker_dialog import SourcePickerDialog
        from ui_qt.power_combine_review import PowerCombineReview
        c = self.window.power_controller
        energy = np.linspace(1.3, 1.6, 101)
        powers = np.array([1., 1.1, 1.2, 1.3])
        shape = np.exp(-((energy - 1.44) / .02) ** 2)
        data = {key: sweep(key, powers, 640 + height * powers[:, None] * shape, energy=energy)
                for key, height in [('a', 100), ('b', 400), ('c', 800)]}
        picker_calls, review_calls = [], []
        def picker(dialog):
            picker_calls.append(1)
            if len(picker_calls) == 1:
                dialog.source_list.item(0).setCheckState(Qt.Checked)
                self.assertEqual(dialog.ok_button.text(), 'Open sweep')
                dialog.source_list.item(1).setCheckState(Qt.Checked)
            else:
                self.assertEqual(dialog.source_list.item(0).checkState(), Qt.Checked)
                self.assertEqual(dialog.source_list.item(1).checkState(), Qt.Checked)
                dialog.source_list.item(2).setCheckState(Qt.Checked)
            self.assertEqual(dialog.ok_button.text(), 'Review combination…')
            return 1
        def review(dialog):
            review_calls.append(dialog.keys)
            if len(review_calls) == 1:
                dialog.controls['b'][0].setCurrentIndex(2)
                dialog.filename.setText('my_combination.csv')
                dialog.go_back()
            else:
                self.assertEqual(dialog.controls['b'][0].currentIndex(), 2)
                self.assertEqual(dialog.filename.text(), 'my_combination.csv')
                self.assertAlmostEqual(dialog.controls['b'][2].value(), .25, places=3)
                dialog.reject()
            return 0
        with patch.object(type(c), '_power_current_sources', return_value={k: PowerSeriesSource(k, k, 'table') for k in data}), \
             patch.object(type(c), '_power_load_group_result', side_effect=lambda k: data[k]), \
             patch.object(SourcePickerDialog, 'exec', picker), patch.object(PowerCombineReview, 'exec', review):
            c._power_choose_combination()
        self.assertEqual(review_calls, [('a', 'b'), ('a', 'b', 'c')])

    def test_combination_rejects_cross_channel_sources(self):
        from core.data_io import PowerSeriesSource
        from ui_qt.power_combine_review import PowerCombineReview
        c = self.window.power_controller
        sources = {
            'kk': PowerSeriesSource('kk', 'sample_KK.csv', 'table', 'sample_KK.csv'),
            'kkp': PowerSeriesSource('kkp', 'sample_KKp.csv', 'table', 'sample_KKp.csv'),
        }
        data = {'kk': sweep('sample_KK.csv', [1., 2.], [[1., 2.], [2., 3.]]),
                'kkp': sweep('sample_KKp.csv', [1., 2.], [[1., 2.], [2., 3.]])}
        with patch.object(type(c), '_power_current_sources', return_value=sources), \
             patch.object(type(c), '_power_load_group_result', side_effect=lambda key: data[key]):
            with self.assertRaises(ValueError) as error:
                PowerCombineReview(self.window, c, ('kk', 'kkp'))
        self.assertIn('same-channel', str(error.exception))

    def test_three_sweeps_correct_then_save_and_plot(self):
        from core.data_io import PowerSeriesSource
        from ui_qt.power_combine_review import PowerCombineReview
        c = self.window.power_controller
        energy = np.linspace(1.3, 1.6, 101)
        powers = np.array([1., 1.1, 1.2, 1.3])
        shape = np.exp(-((energy - 1.44) / .02) ** 2)
        data = {key: sweep(key, powers, 640 + height * powers[:, None] * shape, energy=energy)
                for key, height in [('a', 100), ('b', 400), ('c', 800)]}
        with tempfile.TemporaryDirectory() as folder:
            self.window.current_folder = folder
            with patch.object(type(c), '_power_current_sources', return_value={k: PowerSeriesSource(k, k, 'table') for k in data}), \
                 patch.object(type(c), '_power_load_group_result', side_effect=lambda k: data[k]):
                dialog = PowerCombineReview(self.window, c, tuple(data))
                self.assertTrue(dialog.table.isColumnHidden(2))
                self.assertTrue(dialog.table.isColumnHidden(4))
                dialog.advanced.setChecked(True)
                self.assertFalse(dialog.table.isColumnHidden(2))
                self.assertFalse(dialog.table.isColumnHidden(4))
                dialog.advanced.setChecked(False)
                for key in ('b', 'c'): dialog.controls[key][0].setCurrentIndex(2)
                self.assertIsNotNone(dialog.result, dialog.summary.text())
                self.assertAlmostEqual(dialog.controls['b'][2].value(), .25, places=3)
                self.assertAlmostEqual(dialog.controls['c'][2].value(), .125, places=3)
                dialog.compatible.setChecked(True)
                dialog._save()
                self.assertTrue(dialog.open_requested)
                self.assertTrue(dialog.saved_path.is_file())
                dialog.deleteLater()

    def test_automatic_reference_and_scaling_with_disjoint_outer_ranges(self):
        from core.data_io import PowerSeriesSource
        from ui_qt.power_combine_review import PowerCombineReview
        c = self.window.power_controller
        energy = np.linspace(1.3, 1.6, 101)
        shape = np.exp(-((energy - 1.44) / .02) ** 2)
        data = {key: sweep(key, np.array(powers), 640 + height * np.array(powers)[:, None] * shape, energy=energy)
                for key, powers, height in [('high', [490., 510., 530.], 100),
                    ('low', [1., 1.1, 1.2], 400),
                    ('bridge', [1., 1.1, 1.2, 490., 510., 530.], 100)]}
        with patch.object(type(c), '_power_current_sources', return_value={k: PowerSeriesSource(k, k, 'table') for k in data}), \
             patch.object(type(c), '_power_load_group_result', side_effect=lambda k: data[k]):
            dialog = PowerCombineReview(self.window, c, tuple(data))
            self.assertEqual(dialog.reference.currentData(), 'bridge')
            self.assertEqual(dialog.controls['low'][0].currentIndex(), 2)
            self.assertAlmostEqual(dialog.controls['low'][2].value(), .25, places=3)
            self.assertIsNotNone(dialog.result, dialog.summary.text())
            dialog.automatic.setChecked(False)
            dialog.reference.setCurrentIndex(0)
            self.assertIsNone(dialog.result)
            self.assertIsNotNone(dialog.preview_result)
            self.assertEqual(len(dialog.figure.axes), 3)
            self.assertIn('Sweep 2:', dialog.summary.text())
            dialog.deleteLater()

    def test_single_mode_ignores_stale_roles_and_vp_requires_comparison(self):
        w = self.window
        c = w.power_controller
        for combo in (w.power_group_combo, w.power_kk_group_combo, w.power_kkp_group_combo):
            combo.blockSignals(True)
            combo.addItem("Low", "low")
            combo.addItem("High", "high")
        w.power_group_combo.setCurrentIndex(w.power_group_combo.findData("high"))
        w.power_kk_group_combo.setCurrentIndex(w.power_kk_group_combo.findData("low"))
        w.power_kkp_group_combo.setCurrentIndex(w.power_kkp_group_combo.findData("high"))
        self.assertFalse(c._power_has_distinct_role_groups())
        self.assertEqual(c._power_selected_group_key(), "high")
        c._power_update_vp_availability()
        self.assertFalse(w.power_view_vp_btn.isEnabled())
        with patch.object(type(c), "_power_load_group_result", side_effect=ValueError("fixture")):
            w.power_compare_chk.setChecked(True)
        self.assertTrue(c._power_has_distinct_role_groups())
        self.assertEqual(c._power_selected_group_key(), "low")
        self.assertFalse(w.power_group_combo.isEnabled())
        self.assertTrue(w.power_view_vp_btn.isEnabled())
        c._power_set_view_mode("VP")
        w.power_compare_chk.setChecked(False)
        self.assertEqual(c._power_view(), "Intensity")
        self.assertTrue(w.power_group_combo.isEnabled())
        self.assertEqual(c._power_selected_group_key(), "high")

    def test_combine_dialog_preview_and_save(self):
        from core.data_io import PowerSeriesSource
        from ui_qt.power_combine_dialog import PowerCombineDialog
        c = self.window.power_controller
        a = sweep("low", [1., 2.], [[1., 2.], [3., 4.]])
        b = sweep("high", [2., 3.], [[5., 6.], [7., 8.]])
        with tempfile.TemporaryDirectory() as folder:
            self.window.current_folder = folder
            original = Path(folder) / "combined_power.csv"
            previous = Path(folder) / "combined_power_01.csv"
            original.write_text("preserve original")
            previous.write_text("preserve previous")
            with patch.object(type(c), "_power_current_sources", return_value={
                key: PowerSeriesSource(key, key, "table") for key in ("low", "high")
            }), patch.object(type(c), "_power_load_group_result", side_effect=lambda key: {"low": a, "high": b}[key]):
                dialog = PowerCombineDialog(self.window, c)
                self.assertIsNotNone(dialog.result)
                self.assertEqual(dialog.filename.text(), "low__high__combined.csv")
                dialog.filename.setText("combined_power.csv")
                self.assertEqual(dialog.pair.count(), 1)
                dialog.canvas.draw()
                dialog.compatible.setChecked(True)
                dialog._save()
                self.assertTrue(dialog.saved_path.is_file())
                self.assertEqual(dialog.saved_path.name, "combined_power.csv")
                self.assertIn("Processed Data", str(dialog.saved_path))
                self.assertFalse(dialog.open_requested)
                self.assertEqual(original.read_text(), "preserve original")
                self.assertEqual(previous.read_text(), "preserve previous")
                self.assertEqual(load_power_sweep_csv(str(dialog.saved_path.parent), dialog.saved_path.name)[0].gate.size, 3)
                dialog.deleteLater()

    def test_background_bias_and_rejected_estimate(self):
        from core.data_io import PowerSeriesSource
        from ui_qt.power_combine_dialog import PowerCombineDialog
        from PySide6.QtWidgets import QDialogButtonBox
        energy = np.linspace(1.3, 1.6, 201)
        shape = np.exp(-((energy - 1.44) / .02) ** 2)
        powers = np.array([1., 1.1, 1.2, 1.3])
        a = sweep('low.csv', powers, 640 + 100 * powers[:, None] * shape, energy=energy)
        b = sweep('high.csv', powers, 670 + 400 * powers[:, None] * shape, energy=energy)
        c = self.window.power_controller
        with tempfile.TemporaryDirectory() as folder:
            self.window.current_folder = folder
            with patch.object(type(c), '_power_current_sources', return_value={
                key: PowerSeriesSource(key, key, 'table') for key in ('low', 'high')
            }), patch.object(type(c), '_power_load_group_result', side_effect=lambda key: {'low': a, 'high': b}[key]):
                dialog = PowerCombineDialog(self.window, c)
                dialog.correction.setCurrentText('Estimate from overlap')
                self.assertIsNotNone(dialog.result, dialog.summary.text())
                self.assertAlmostEqual(dialog.factor.value(), .25, places=3)
                np.testing.assert_allclose([v.value() for v in dialog.backgrounds], [640, 670], atol=.01)
                dialog.manual_background.setChecked(True)
                for control in dialog.backgrounds:
                    control.setValue(0)
                self.assertIsNone(dialog.result)
                self.assertIn('CORRECTION REJECTED', dialog.summary.text())
                dialog.compatible.setChecked(True)
                self.assertFalse(dialog.buttons.button(QDialogButtonBox.Save).isEnabled())
                np.testing.assert_allclose(dialog.inputs[1].cube.Z, b.cube.Z)
                dialog.deleteLater()

    def test_render_single_comparison_vp_and_back(self):
        from core.data_io import PowerSeriesSource
        from ui_qt.main_window import LoadedState
        w = self.window
        c = w.power_controller
        a = sweep("low", [1., 2., 3.], [[1., 2.], [3., 4.], [5., 6.]], stages=[1., 1., 2.])
        b = sweep("high", [1., 1.5, 2., 3.], [[7., 8.], [9., 10.], [11., 12.], [13., 14.]],
                  stages=[1., 2., 3., 4.])
        for combo, key in [(w.power_group_combo, "low"), (w.power_kk_group_combo, "low"),
                           (w.power_kkp_group_combo, "high")]:
            combo.blockSignals(True)
            combo.addItem(key, key)
            combo.setCurrentIndex(combo.findData(key))
        w.available_files = ["low", "high"]
        w.loaded = LoadedState(mode="Power Dependent", folder="", primary_file="low",
                               selected_files=["low"], cube=a.cube, power_records=a.records,
                               power_groups={}, power_group_key="low", y_axis_spec="auto")
        w.power_axis_scale_combo.setCurrentText("Linear")
        w.power_background_auto_chk.setChecked(False)
        w.power_background_spin.setValue(0)
        with patch.object(type(c), "_power_current_sources", return_value={
            key: PowerSeriesSource(key, key, "table") for key in ("low", "high")
        }), patch.object(type(c), "_power_load_group_result", side_effect=lambda key: {"low": a, "high": b}[key]), \
                patch.object(w, "_show_error") as error:
            w._apply_auto_limits_for_loaded()
            w._plot_mode("Power Dependent")
            self.assertEqual(set(w._power_active_cubes), {"Power"})
            single_key = w._current_plot_params_key("Power Dependent")
            w.power_compare_chk.setChecked(True)
            self.assertFalse(w.power_pair_mode_combo.model().item(0).isEnabled())
            self.assertEqual(c._power_pairing_mode(), "power")
            self.assertNotEqual(w._current_plot_params_key("Power Dependent"), single_key)
            w._plot_mode("Power Dependent")
            self.assertEqual(set(w._power_active_cubes), {"KK", "KKp"})
            w.power_spins["gate"].setValue(2.)
            c._update_power_compare_spectrum_and_lines(w._power_active_cubes)
            np.testing.assert_allclose(w._power_spectrum_ax.lines[1].get_ydata(), [11., 12.])
            c._power_set_view_mode("VP")
            w._plot_mode("Power Dependent")
            self.assertEqual(set(w._power_active_cubes), {"VP"})
            w.power_compare_chk.setChecked(False)
            w._plot_mode("Power Dependent")
            self.assertEqual(set(w._power_active_cubes), {"Power"})
            error.assert_not_called()
