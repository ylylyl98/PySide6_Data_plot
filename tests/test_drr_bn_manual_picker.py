"""Manual BN history follows the selected measurement without reading CSVs."""
from dataclasses import replace
import builtins
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem, QPushButton,
)

from core.drr_sources import DrrSource
from tests.ui_test_helpers import dispose_owned_window
from ui_qt.controllers_drr import DrrController
from ui_qt import drr_background_settings_workflow as workflow
from ui_qt.main_window import MainWindow


class BnManualPickerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = QSettings(str(self.root/'settings.ini'), QSettings.IniFormat)
        with patch('ui_qt.main_window.QSettings', return_value=self.store), patch.object(
                MainWindow, '_restore_last_folder', lambda _: None), patch.object(
                MainWindow, '_schedule_automatic_update_check', lambda _: None):
            self.window = MainWindow()
        self.addCleanup(dispose_owned_window, self.window)
        self.window.current_folder = str(self.root)
        self.window._suspend_drr_autoplot = True
        self.measurement = self.source('TG', 49, 100)
        self.bn49 = self.source('BNbackground', 49, 500, '001')
        self.bn49_newer = self.source('BNbackground', 49, 700, '002')
        self.bn94 = self.source('BNbackground', 94, 1200, '001')
        self.unknown = self.source('BNbackground', None, 1300, '001')
        self.wrong_grid = replace(self.source('BNbackground', 49, 1400, '003'),
                                  spectral_grid=(741., 761., 781.))
        self.p = self.source('Pbackground', 49, 1500, '001')
        self.n = self.source('Nbackground', 49, 1600, '001')
        self.catalog = [self.measurement, self.bn94, self.p, self.n, self.unknown,
                        self.wrong_grid, self.bn49, self.bn49_newer]
        self.window.drr_available_sources = self.catalog
        self.window.drr_selected_files = [self.measurement.source]

    @staticmethod
    def source(role, rotation, stamp, suffix=''):
        angle = f'RotIn{rotation}deg_' if rotation is not None else ''
        name = f'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_{angle}polH_1mW_{role}'
        name += ('_'+suffix if suffix else '')+'.csv'
        measurement = role == 'TG'
        return DrrSource(name, name, f'{role}-{rotation}', '2026-10-21', stamp, not measurement,
                         gate_varies=measurement, frame_count=2, wavelength_center_nm=730,
                         spectral_grid=(740., 760., 780.), gate_labels=('Vbg', 'Vtg'),
                         gate_grid=((0., 0.), (1., 1.)) if measurement else ((0., 0.), (0., 0.)),
                         grid_complete=True, classification='measurement' if measurement else 'background')

    @staticmethod
    def paths(dialog):
        groups = dialog.findChild(QListWidget, 'drr_source_group_list')
        return [name for i in range(groups.count()) for name in groups.item(i).data(Qt.UserRole+4)]

    @staticmethod
    def click(dialog, text):
        next(button for button in dialog.findChildren(QPushButton) if button.text() == text).click()

    def choose(self, execute, *, kind='bn', selected=()):
        original_open, original_io_open = builtins.open, io.open
        def guarded(original):
            def read(path, *args, **kwargs):
                if isinstance(path, (str, Path)) and str(path).casefold().endswith(('.csv', '.jsonl')):
                    raise AssertionError('The manual picker must use cached metadata, not read CSVs/logs.')
                return original(path, *args, **kwargs)
            return read
        with patch.object(QDialog, 'exec', execute), patch.object(
                self.window, '_queue_drr_catalog_refresh') as refresh, patch.object(
                DrrController, '_drr_missing_sources', return_value=[]), patch.object(
                self.window, '_drr_catalog_is_inventory_preview', return_value=False), patch(
                'ui_qt.controllers_drr.inspect_csv_spectral_grid', side_effect=AssertionError('No GUI header scan')), patch(
                'builtins.open', side_effect=guarded(original_open)), patch(
                'io.open', side_effect=guarded(original_io_open)):
            result = workflow._select_files(self.window, None, kind, selected, 'Choose measured BN')
        return result, refresh

    def test_same_angle_first_in_groups_and_files_without_choosing_it_for_the_user(self):
        def execute(dialog):
            self.assertEqual(self.paths(dialog), [self.bn49_newer.source, self.bn49.source,
                                                  self.unknown.source, self.bn94.source, self.wrong_grid.source])
            files = dialog.findChild(QListWidget, 'drr_source_file_list')
            self.assertEqual([files.item(i).data(Qt.UserRole) for i in range(files.count())],
                             [self.bn49_newer.source, self.bn49.source])
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            self.assertEqual(chosen.count(), 0)
            return QDialog.Rejected
        result, _ = self.choose(execute)
        self.assertIsNone(result)

    def test_manual_choice_requires_add_and_accept_and_keeps_the_applied_recipe_untouched(self):
        before = (tuple(self.window.drr_baseline_files_manual), tuple(self.window._drr_assignments),
                  workflow.source_intent(self.window), self.store.allKeys())
        def execute(dialog):
            self.click(dialog, 'Add Entire Group')
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            self.assertEqual(chosen.count(), 2)
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
            return dialog.result()
        result, _ = self.choose(execute)
        self.assertEqual(result, (self.bn49_newer.source, self.bn49.source))
        self.assertEqual((tuple(self.window.drr_baseline_files_manual), tuple(self.window._drr_assignments),
                          workflow.source_intent(self.window), self.store.allKeys()), before)

    def test_changing_measurement_rotation_invalidates_cached_picker_order(self):
        self.choose(lambda _: QDialog.Rejected)
        measurement94 = self.source('TG', 94, 200)
        self.window.drr_available_sources = [*self.catalog, measurement94]
        self.window.drr_selected_files = [measurement94.source]
        def execute(dialog):
            paths = self.paths(dialog)
            self.assertEqual(paths[0], self.bn94.source)
            self.assertLess(paths.index(self.unknown.source), paths.index(self.bn49.source))
            self.assertEqual(dialog.findChild(QListWidget, 'drr_source_chosen_list').count(), 0)
            return QDialog.Rejected
        self.choose(execute)

    def test_both_bn_entry_points_and_refresh_filters_keep_bn_role(self):
        p_extra = replace(self.p, source='Initial Data/other_Pbackground.csv', filename='other_Pbackground.csv')
        for kind in ('bn', 'bn_candidates'):
            with self.subTest(kind=kind):
                def execute(dialog):
                    combo = dialog.findChild(QComboBox, 'drr_source_type_combo')
                    combo.setCurrentIndex(0)
                    self.assertNotIn(self.p.source, self.paths(dialog))
                    self.assertNotIn(self.n.source, self.paths(dialog))
                    self.window.drr_available_sources = [*self.catalog, p_extra]
                    self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
                    combo.setCurrentIndex(1)
                    self.assertNotIn(p_extra.source, self.paths(dialog))
                    self.assertNotIn(self.measurement.source, self.paths(dialog))
                    self.assertEqual(self.paths(dialog)[0], self.bn49_newer.source)
                    return QDialog.Rejected
                self.choose(execute, kind=kind)

    def test_known_wrong_grid_and_non_bn_browse_aliases_cannot_enter_chosen_files(self):
        wrong_abs = str((self.root/self.wrong_grid.source).resolve())
        p_abs = str((self.root/self.p.source).resolve())
        compatible_abs = str((self.root/self.bn49.source).resolve())
        outside = str((self.root.parent/'elsewhere'/'manual_RotIn49deg_BNbackground_009.csv').resolve())
        def execute(dialog):
            with patch('ui_qt.controllers_drr.QFileDialog.getOpenFileNames',
                       return_value=([wrong_abs, p_abs, compatible_abs, outside], '')):
                self.click(dialog, 'Browse File Anywhere...')
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            paths = [chosen.item(i).data(Qt.UserRole) for i in range(chosen.count())]
            self.assertEqual(paths, [compatible_abs, outside])
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
            return dialog.result()
        result, _ = self.choose(execute)
        self.assertEqual(result, (compatible_abs, outside))

    def test_directory_aliases_keep_cached_grids_for_selection_browse_and_accept(self):
        canonical_root = self.root.resolve()
        alias_root = canonical_root/'directory-alias'
        original_resolve = Path.resolve
        resolved_paths = []

        def resolve(path, *args, **kwargs):
            # Emulate a Windows short-directory alias on every platform.
            # The native short-name case is the same mapping at the OS boundary.
            resolved_paths.append(path)
            try:
                path = canonical_root/path.relative_to(alias_root)
            except ValueError:
                pass
            return original_resolve(path, *args, **kwargs)

        self.window.current_folder = str(alias_root)
        self.window.drr_selected_files = [str(alias_root/self.measurement.source)]
        wrong = str(canonical_root/self.wrong_grid.source)
        compatible = str(canonical_root/self.bn49.source)
        outside = str(canonical_root.parent/'elsewhere'/'manual_RotIn49deg_BNbackground_009.csv')

        def execute(dialog):
            self.assertEqual(self.paths(dialog)[0], self.bn49_newer.source)
            with patch('ui_qt.controllers_drr.QFileDialog.getOpenFileNames',
                       return_value=([wrong, str(canonical_root/self.p.source), compatible, outside], '')):
                self.click(dialog, 'Browse File Anywhere...')
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            self.assertEqual([chosen.item(i).data(Qt.UserRole) for i in range(chosen.count())],
                             [compatible, outside])
            # Final validation must also reject a row inserted under the alias.
            item = QListWidgetItem(self.wrong_grid.filename)
            item.setData(Qt.UserRole, str(alias_root/self.wrong_grid.source))
            chosen.addItem(item)
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
            self.assertEqual(dialog.result(), QDialog.Rejected)
            chosen.takeItem(chosen.count()-1)
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
            return dialog.result()

        with patch.object(Path, 'resolve', resolve):
            result, _ = self.choose(execute)
        self.assertEqual(result, (compatible, outside))
        self.assertEqual(resolved_paths.count(alias_root), 1)
        # Building/sorting the relative catalog must not resolve every CSV.
        self.assertNotIn(canonical_root/self.bn49_newer.source, resolved_paths)

    def test_wrong_angle_remains_explicit_choice_and_is_not_an_automatic_fallback(self):
        self.window.drr_available_sources = [self.measurement, self.bn94, self.p, self.n]
        def execute(dialog):
            self.assertEqual(self.paths(dialog), [self.bn94.source])
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            self.assertEqual(chosen.count(), 0)
            self.click(dialog, 'Add Entire Group')
            self.assertEqual(chosen.count(), 1)
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
            return dialog.result()
        result, _ = self.choose(execute)
        self.assertEqual(result, (self.bn94.source,))

    def test_malformed_preselection_and_final_mutated_rows_fail_role_and_grid_checks(self):
        def execute(dialog):
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            self.assertEqual([chosen.item(i).data(Qt.UserRole) for i in range(chosen.count())],
                             [self.wrong_grid.source, self.bn49.source])
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
            self.assertEqual(dialog.result(), QDialog.Rejected)
            chosen.takeItem(0)
            for source in (self.p, self.wrong_grid):
                item = QListWidgetItem(source.filename)
                item.setData(Qt.UserRole, source.source)
                chosen.addItem(item)
                dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
                self.assertEqual(dialog.result(), QDialog.Rejected)
                chosen.takeItem(chosen.count()-1)
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
            return dialog.result()
        result, _ = self.choose(execute, selected=(self.p.source, self.wrong_grid.source, self.bn49.source))
        self.assertEqual(result, (self.bn49.source,))

    def test_filename_context_still_orders_uncached_selected_measurement_without_gui_reads(self):
        uncached = self.measurement.source.replace('_TG.csv', '_TG_rep99.csv')
        self.window.drr_selected_files = [uncached]
        def execute(dialog):
            # Without the measurement header there is no basis to reject
            # BN003's otherwise known grid. Its newer same-angle timestamp
            # can rank first, while suitability still needs worker validation.
            self.assertEqual(self.paths(dialog)[0], self.wrong_grid.source)
            files = dialog.findChild(QListWidget, 'drr_source_file_list')
            self.assertIn('grid needs validation', files.item(0).text())
            return QDialog.Rejected
        _, refresh = self.choose(execute)
        self.assertTrue(refresh.called)

    def test_workflow_propagates_bn_role_and_cancel_does_not_turn_into_an_empty_choice(self):
        with patch.object(DrrController, '_open_drr_source_dialog', return_value=[self.bn49.source]) as picker:
            self.assertEqual(workflow._select_files(self.window, None, 'bn', (), 'Choose BN'), (self.bn49.source,))
        self.assertEqual(picker.call_args.kwargs['background_role'], 'BN')
        def cancelled(**kwargs):
            self.window._drr_source_dialog_accepted = False
            return [self.bn49.source]
        with patch.object(DrrController, '_open_drr_source_dialog', side_effect=cancelled) as picker:
            self.assertIsNone(workflow._select_files(self.window, None, 'bn_candidates',
                                                   (self.bn49.source,), 'Choose BN candidates'))
        self.assertEqual(picker.call_args.kwargs['background_role'], 'BN')


if __name__ == '__main__':
    unittest.main()
