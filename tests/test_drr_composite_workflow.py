"""Composite references through save/select, worker loading and source changes."""
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QPushButton
import numpy as np

from core.drr_composite_background import (
    CompositeBackground, assign_region, load_composite_source, save_composite_background,
)
from core.drr_background_regions import DrrBackgroundRegion
from core.drr_sources import DrrMeasurementAssignment
from core.provenance import can_cleanup
from ui_qt.common import LoadOptions
from ui_qt.drr_composite_workflow import background_dependencies, edit_composite_background
from ui_qt.main_window import MainWindow


class CompositeWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, value in [('sample.csv', 100.), ('p_back.csv', 50.), ('n_back.csv', 25.)]:
            (self.root / name).write_text('Vbg,Vtg,' + ','.join(map(str, np.linspace(740., 780., 81)))
                + '\n0,0,' + ','.join([str(value)] * 81) + '\n', encoding='utf-8')
        self.owner = SimpleNamespace(
            current_folder=str(self.root), drr_selected_files=['sample.csv'],
            drr_baseline_files_manual=['p_back.csv'], _drr_background_regions=(),
            drr_baseline_files_found=[], drr_edit_baselines_btn=QPushButton(),
            _drr_assignments=(), _drr_assignments_automatic=False,
            _show_error=MagicMock(), _status=MagicMock(), _start_load=MagicMock(),
            _queue_drr_catalog_refresh=MagicMock(), drr_available_sources=[], thread_pool=None,
            drr_composite_background_btn=QPushButton(), drr_baseline_combo=QComboBox(),
            drr_baseline_combine_combo=QComboBox(), drr_pin_baseline_chk=QCheckBox(),
            drr_controller=SimpleNamespace(_cancel_auto_external=MagicMock(),
                _update_drr_selection_labels=MagicMock(), _open_drr_source_dialog=MagicMock(return_value=['n_back.csv']),
                _reject_mixed_xlsx_selection=lambda _: None),
        )
        self.owner.drr_baseline_combo.addItems(['Self (last frame)', 'External'])
        self.owner.drr_baseline_combo.setCurrentText('External')
        self.owner.drr_baseline_combine_combo.addItems(['Last frame from each file, then average'])
        self.owner._provenance_records_for_load = lambda _: ()

    def composite(self):
        p = load_composite_source(self.root, ('p_back.csv',), key='p')
        n = load_composite_source(self.root, ('n_back.csv',), key='n')
        document = assign_region(CompositeBackground((p, n), 'p'), 1.61, 1.64, 'n')
        return save_composite_background(self.root / 'clean_reference.csv', document)

    def test_save_selects_one_file_and_pins_it_without_measurement(self):
        self.owner.drr_selected_files = []
        path = self.composite()
        with patch('ui_qt.drr_composite_background_dialog.DrrCompositeBackgroundDialog') as factory:
            factory.return_value.exec.return_value = QDialog.Accepted
            factory.return_value.saved_path = path
            edit_composite_background(self.owner)
        self.assertEqual(self.owner.drr_baseline_files_manual, ['clean_reference.csv'])
        self.assertTrue(self.owner.drr_pin_baseline_chk.isChecked())
        self.assertTrue(self.owner._drr_baseline_user_selected)
        self.owner._start_load.assert_not_called()
        self.owner._queue_drr_catalog_refresh.assert_called_once()

    def test_save_applies_to_measurement_and_preserves_initial_region_draft(self):
        region = DrrBackgroundRegion(1.61, 1.64, ('n_back.csv',), 'first')
        self.owner._drr_background_regions = (region,)
        path = self.composite()
        with patch('ui_qt.drr_composite_background_dialog.DrrCompositeBackgroundDialog') as factory:
            factory.return_value.exec.return_value = QDialog.Accepted
            factory.return_value.saved_path = path
            edit_composite_background(self.owner)
        self.assertEqual(factory.call_args.kwargs['initial_regions'], (region,))
        self.assertEqual(self.owner._drr_background_regions, ())
        self.owner._start_load.assert_called_once_with('DRR')
        factory.call_args.kwargs['select_files']([])
        self.assertTrue(self.owner.drr_controller._open_drr_source_dialog.call_args.kwargs['independent_background'])

    def test_cancel_preserves_main_recipe_and_does_not_refresh(self):
        before = list(self.owner.drr_baseline_files_manual)
        with patch('ui_qt.drr_composite_background_dialog.DrrCompositeBackgroundDialog') as factory:
            factory.return_value.exec.return_value = QDialog.Rejected
            edit_composite_background(self.owner)
        self.assertEqual(before, self.owner.drr_baseline_files_manual)
        self.owner._start_load.assert_not_called()
        self.owner._queue_drr_catalog_refresh.assert_not_called()

    def test_workspace_use_applies_native_region_recipe_and_enables_keep(self):
        from ui_qt.drr_background_workspace import BackgroundSelection
        from ui_qt.drr_background_workspace_workflow import apply_background_selection
        region = DrrBackgroundRegion(1.61, 1.64, ('n_back.csv',), 'first')
        apply_background_selection(self.owner, BackgroundSelection(files=('p_back.csv',), regions=(region,)))
        self.assertEqual(self.owner._drr_background_regions, (region,))
        self.assertEqual(self.owner._drr_assignments[0].regions, (region,))
        self.assertTrue(self.owner.drr_pin_baseline_chk.isChecked())
        self.assertTrue(self.owner._drr_baseline_user_selected)
        self.owner._start_load.assert_called_once_with('DRR')

    def test_workspace_cancel_or_changed_group_keeps_existing_recipe(self):
        from ui_qt.drr_background_workspace import BackgroundSelection
        from ui_qt.drr_background_workspace_workflow import edit_background_workspace
        with patch('ui_qt.drr_background_workspace.DrrBackgroundWorkspace') as factory:
            factory.return_value.exec.return_value = QDialog.Rejected
            edit_background_workspace(self.owner)
            factory.return_value.selection = BackgroundSelection(files=('n_back.csv',))
            def finish():
                self.owner.drr_selected_files = []
                return QDialog.Accepted
            factory.return_value.exec.side_effect = finish
            edit_background_workspace(self.owner)
        self.assertEqual(self.owner.drr_baseline_files_manual, ['p_back.csv'])
        self.assertFalse(self.owner.drr_pin_baseline_chk.isChecked())
        self.owner._start_load.assert_not_called()
        self.owner._queue_drr_catalog_refresh.assert_not_called()

    def test_selection_changed_while_editing_does_not_apply_to_new_measurement(self):
        with patch('ui_qt.drr_composite_background_dialog.DrrCompositeBackgroundDialog') as factory:
            def finish():
                self.owner.drr_selected_files = []
                return QDialog.Accepted
            factory.return_value.exec.side_effect = finish
            factory.return_value.saved_path = self.composite()
            edit_composite_background(self.owner)
        self.assertEqual(self.owner.drr_baseline_files_manual, ['p_back.csv'])
        self.owner._start_load.assert_not_called()

    def load(self):
        self.composite()
        for name in ('p_back.csv', 'n_back.csv'):
            (self.root / name).unlink()
        assignment = DrrMeasurementAssignment('sample.csv', 'External', ('clean_reference.csv',), 'last')
        options = LoadOptions(mode='DRR', folder=str(self.root), selected_files=['sample.csv'],
            baseline_files=['clean_reference.csv'], pl_log_scale=False, drr_baseline_text='External',
            drr_baseline_which='last', compare_log_scale=False, drr_assignments=(assignment,))
        signal = SimpleNamespace(emit=lambda *_: None)
        loaded = MainWindow._load_task_data(self.owner, options, progress=signal, log=signal)
        self.owner.loaded = loaded
        params = {'selected_files': loaded.selected_files, 'baseline_files': loaded.baseline_files,
            'baseline_mode': 'External', 'baseline_which': 'Last frame from each file, then average',
            'y_axis_spec': 'auto', 'drr_assignments': (assignment.to_dict(),)}
        self.owner.drr_controller._read_drr_params = lambda: params
        return loaded

    def test_worker_loads_frozen_composite_and_metadata_change_reloads(self):
        loaded = self.load()
        self.assertEqual(loaded.drr_mode_label, 'DR/R External')
        mask = (loaded.cube.energy >= 1.61) & (loaded.cube.energy < 1.64)
        np.testing.assert_allclose(loaded.cube.Z[:, mask], 3.)
        np.testing.assert_allclose(loaded.cube.Z[:, ~mask], 1.)
        self.assertEqual(loaded.cube.drr_background_boundaries, (1.61, 1.64))
        MainWindow._ensure_loaded_matches_drr_params(self.owner)
        self.owner._start_load.assert_not_called()
        metadata = self.root / 'clean_reference.metadata.json'
        metadata.write_bytes(metadata.read_bytes() + b'\n')
        MainWindow._ensure_loaded_matches_drr_params(self.owner)
        self.owner._start_load.assert_called_once_with('DRR')

    def test_metadata_deletion_invalidates_loaded_dependencies(self):
        loaded = self.load()
        (self.root / 'clean_reference.metadata.json').unlink()
        self.assertNotEqual(loaded.drr_background_dependencies,
            background_dependencies(str(self.root), loaded.baseline_files))
        MainWindow._ensure_loaded_matches_drr_params(self.owner)
        self.owner._start_load.assert_called_once_with('DRR')

    def test_reference_pair_is_provenance_tracked_and_never_temporary_cleanup(self):
        path = self.composite()
        initial = self.root / 'Initial Data'
        initial.mkdir()
        (initial / path.name).write_bytes(path.read_bytes())
        records = MainWindow._drr_provenance_records(self.owner, str(self.root), [], [path.name])
        self.assertEqual([r.role for r in records], ['background', 'background_recipe'])
        self.assertEqual({Path(r.canonical_source_path).name for r in records},
            {'clean_reference.csv', 'clean_reference.metadata.json'})
        self.assertTrue(all(not can_cleanup(r, self.root) for r in records))


if __name__ == '__main__':
    unittest.main()
