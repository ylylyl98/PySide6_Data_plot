"""P-only/N-only history stays role scoped and follows the selected measurement."""
import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem, QPushButton,
)

from core.drr_sources import DrrSource
from ui_qt.controllers_drr import DrrController
from ui_qt import drr_background_settings_workflow as workflow
from ui_qt.main_window import MainWindow
from tests.ui_test_helpers import dispose_owned_window


class DrrPnDirectPickerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        settings = QSettings(str(Path(self.tmp.name) / 'settings.ini'), QSettings.IniFormat)
        with (patch.object(MainWindow, '_restore_last_folder', lambda self: None),
              patch.object(MainWindow, '_schedule_automatic_update_check', lambda self: None),
              patch('ui_qt.main_window.QSettings', return_value=settings)):
            self.window = MainWindow()
        self.addCleanup(dispose_owned_window, self.window)
        self.window.current_folder = self.tmp.name
        self.window._suspend_drr_autoplot = True
        prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn'

        def source(role, rotation, stamp, suffix=''):
            name = f'{prefix}{rotation}deg_{role}{suffix}.csv'
            measurement = role == 'TG'
            return DrrSource(
                name, name, f'{role}-{rotation}', '2026-10-21', stamp, not measurement,
                gate_varies=measurement, frame_count=2, wavelength_center_nm=730,
                spectral_grid=(740., 760., 780.), gate_labels=('Vbg', 'Vtg'),
                gate_grid=((0., 0.), (1., 1.)) if measurement else
                          ((-1., -1.), (-1., -1.)) if role == 'Pbackground' else
                          ((1., 1.), (1., 1.)),
                grid_complete=True, classification='measurement' if measurement else 'background',
            )

        self.measurement = source('TG', 49, 100)
        self.p49 = source('Pbackground', 49, 200, '_001')
        self.p49_older = source('Pbackground', 49, 150, '_002')
        self.p94 = source('Pbackground', 94, 900)
        self.n49 = source('Nbackground', 49, 1000)
        self.n94 = source('Nbackground', 94, 1100)
        self.bn = source('BNbackground', 49, 2000)
        self.catalog = [self.measurement, self.bn, self.p94, self.n94,
                        self.n49, self.p49, self.p49_older]
        self.window.drr_available_sources = self.catalog
        self.window.drr_selected_files = [self.measurement.source]

    def open_picker(self, role, execute, selected=()):
        with (patch.object(QDialog, 'exec', execute),
              patch.object(self.window, '_queue_drr_catalog_refresh') as refresh,
              patch.object(DrrController, '_drr_missing_sources', return_value=[]),
              patch.object(self.window, '_drr_catalog_is_inventory_preview', return_value=False)):
            result = self.window.drr_controller._open_drr_source_dialog(
                title=f'Choose {role}', selected=list(selected), baseline_mode=True,
                independent_background=True, background_role=role)
        return result, refresh

    @staticmethod
    def visible_paths(dialog):
        groups = dialog.findChild(QListWidget, 'drr_source_group_list')
        return [path for row in range(groups.count())
                for path in groups.item(row).data(Qt.UserRole + 4)]

    @staticmethod
    def click(dialog, text):
        next(button for button in dialog.findChildren(QPushButton) if button.text() == text).click()

    def test_p_only_filters_bn_and_n_then_prioritizes_current_rotation_before_newest(self):
        def execute(dialog):
            self.assertEqual(self.visible_paths(dialog),
                             [self.p49.source, self.p49_older.source, self.p94.source])
            files = dialog.findChild(QListWidget, 'drr_source_file_list')
            self.assertEqual(files.item(0).data(Qt.UserRole), self.p49.source)
            # Direct averaging still permits two-frame files and multiple repeats.
            self.click(dialog, 'Add Entire Group')
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            self.assertEqual(chosen.count(), 2)
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
            return dialog.result()

        result, _ = self.open_picker('P', execute)
        self.assertEqual(result, [self.p49.source, self.p49_older.source])

    def test_role_and_measurement_context_invalidate_reopened_presentation(self):
        self.open_picker('P', lambda dialog: QDialog.Rejected)

        def n49(dialog):
            self.assertEqual(self.visible_paths(dialog), [self.n49.source, self.n94.source])
            return QDialog.Rejected

        self.open_picker('N', n49)
        measurement94 = replace(self.measurement,
                                source=self.measurement.source.replace('49deg', '94deg'),
                                filename=self.measurement.filename.replace('49deg', '94deg'))
        self.window.drr_available_sources = [*self.catalog, measurement94]
        self.window.drr_selected_files = [measurement94.source]

        def n94(dialog):
            self.assertEqual(self.visible_paths(dialog), [self.n94.source, self.n49.source])
            return QDialog.Rejected

        self.open_picker('N', n94)

    def test_ref_all_filter_and_catalog_refresh_keep_role_scope(self):
        p_pl = replace(self.p49, source=self.p49.source.replace('KREF', 'KPL'),
                       filename=self.p49.filename.replace('KREF', 'KPL'))

        def execute(dialog):
            type_combo = dialog.findChild(QComboBox, 'drr_source_type_combo')
            type_combo.setCurrentIndex(0)
            self.assertEqual(set(self.visible_paths(dialog)),
                             {self.p49.source, self.p49_older.source, self.p94.source})
            self.window.drr_available_sources = [*self.catalog, p_pl]
            self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
            self.assertNotIn(p_pl.source, self.visible_paths(dialog))
            type_combo.setCurrentIndex(1)
            self.assertEqual(set(self.visible_paths(dialog)),
                             {self.p49.source, self.p49_older.source, self.p94.source, p_pl.source})
            return QDialog.Rejected

        self.open_picker('P', execute)

    def test_malformed_preselection_cannot_retain_bn_or_opposite_role(self):
        def execute(dialog):
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            self.assertEqual(chosen.count(), 1)
            self.assertEqual(chosen.item(0).data(Qt.UserRole), self.p49.source)
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
            return dialog.result()

        result, _ = self.open_picker('P', execute, [self.bn.source, self.n49.source, self.p49.source])
        self.assertEqual(result, [self.p49.source])

    def test_manual_browse_rejects_named_wrong_role_and_preserves_unlabelled_multi_select(self):
        manual = [str(Path(self.tmp.name) / 'unlabelled_first.csv'),
                  str(Path(self.tmp.name) / 'unlabelled_second.csv')]

        def execute(dialog):
            with patch('ui_qt.controllers_drr.QFileDialog.getOpenFileNames',
                       return_value=([self.bn.source, self.n49.source, self.p49.source, *manual], '')):
                self.click(dialog, 'Browse File Anywhere...')
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            self.assertEqual(chosen.count(), 3)
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
            return dialog.result()

        result, _ = self.open_picker('P', execute)
        self.assertEqual([Path(path).name for path in result], [self.p49.filename, *map(lambda x: Path(x).name, manual)])

    def test_absolute_manual_paths_keep_inferred_catalog_roles(self):
        inferred_p = replace(self.p49, source='Initial Data/REF/background_001.csv',
                             filename='background_001.csv', group_key='inferred-p')
        inferred_n = replace(self.n49, source='Initial Data/REF/background_002.csv',
                             filename='background_002.csv', group_key='inferred-n')
        self.window.drr_available_sources = [*self.catalog, inferred_p, inferred_n]
        absolute_p = str(Path(self.tmp.name, inferred_p.source).resolve())
        absolute_n = str(Path(self.tmp.name, inferred_n.source).resolve())

        def execute(dialog):
            self.assertIn(inferred_p.source, self.visible_paths(dialog))
            self.assertNotIn(inferred_n.source, self.visible_paths(dialog))
            with patch('ui_qt.controllers_drr.QFileDialog.getOpenFileNames',
                       return_value=([absolute_n, absolute_p], '')):
                self.click(dialog, 'Browse File Anywhere...')
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            self.assertEqual(chosen.count(), 1)
            self.assertEqual(chosen.item(0).data(Qt.UserRole), absolute_p)
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
            return dialog.result()

        result, _ = self.open_picker('P', execute, [absolute_n])
        self.assertEqual(result, [absolute_p])

    def test_known_conditions_precede_name_preference_for_role_classified_sources(self):
        # A generic back name can have an inferred P role without an explicit
        # Pback token; known matching rotation must still outrank a newer P94.
        generic_p = replace(self.p49,
                            source=self.p49.source.replace('Pbackground_001', 'bg_001'),
                            filename=self.p49.filename.replace('Pbackground_001', 'bg_001'),
                            group_key='generic-p49')
        self.window.drr_available_sources = [self.measurement, self.p94, generic_p]

        def execute(dialog):
            self.assertEqual(self.visible_paths(dialog), [generic_p.source, self.p94.source])
            return QDialog.Rejected

        self.open_picker('P', execute)

    def test_ambiguous_role_names_are_rejected_at_import_and_final_acceptance(self):
        ambiguous = 'Pbackground_Nbackground.csv'

        def execute(dialog):
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            self.assertEqual(chosen.count(), 1)
            with patch('ui_qt.controllers_drr.QFileDialog.getOpenFileNames',
                       return_value=([str(Path(self.tmp.name, ambiguous))], '')):
                self.click(dialog, 'Browse File Anywhere...')
            self.assertEqual(chosen.count(), 1)
            # A retained/mutated row must also fail the final eligibility check.
            wrong = QListWidgetItem(ambiguous)
            wrong.setData(Qt.UserRole, ambiguous)
            chosen.addItem(wrong)
            buttons = dialog.findChild(QDialogButtonBox)
            buttons.button(QDialogButtonBox.Ok).click()
            self.assertEqual(dialog.result(), QDialog.Rejected)
            chosen.takeItem(1)
            buttons.button(QDialogButtonBox.Ok).click()
            return dialog.result()

        result, _ = self.open_picker('P', execute, [ambiguous, self.p49.source])
        self.assertEqual(result, [self.p49.source])
        with patch.object(DrrController, '_open_drr_source_dialog', return_value=[ambiguous]):
            with self.assertRaisesRegex(ValueError, 'not a P reference'):
                workflow._select_files(self.window, None, 'p', (), 'Choose P')

    def test_missing_selected_measurement_requests_refresh_without_stale_ranking(self):
        self.window.drr_selected_files = ['uncached_measurement.csv']
        _, refresh = self.open_picker('P', lambda dialog: QDialog.Rejected)
        self.assertTrue(refresh.called)
        self.assertEqual(refresh.call_args.kwargs['selected_sources'], ('uncached_measurement.csv',))

    def test_settings_propagates_role_and_cancellation_keeps_draft(self):
        owner = self.window
        with patch.object(DrrController, '_open_drr_source_dialog', return_value=[self.p49.source]) as choose:
            self.assertEqual(workflow._select_files(owner, None, 'p', (), 'Choose P'), (self.p49.source,))
        self.assertEqual(choose.call_args.kwargs['background_role'], 'P')

        def cancelled(**kwargs):
            owner._drr_source_dialog_accepted = False
            return [self.p49.source]

        with patch.object(DrrController, '_open_drr_source_dialog', side_effect=cancelled) as choose:
            self.assertIsNone(workflow._select_files(owner, None, 'n', (self.n49.source,), 'Choose N'))
        self.assertEqual(choose.call_args.kwargs['background_role'], 'N')


if __name__ == '__main__':
    unittest.main()
