"""Do not relabel legacy per-file fits as one fit of the raw group average."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from core.drr_fitted_background import estimate_background
from tests.test_drr_fitted_background import SETTINGS, synthetic


class FittedGroupExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        raw, _, _ = synthetic()
        self.cube, self.first = estimate_background(raw, SETTINGS, source='a.csv')
        self.second = replace(self.first, source='b.csv')
        self.loaded = SimpleNamespace(mode='DRR', folder=str(self.folder), cube=self.cube,
            selected_files=['a.csv', 'b.csv'], primary_file='a.csv', baseline_files=[],
            drr_baseline_text='Fitted (sample)', drr_mode_label='DR/R Self',
            drr_baseline_which='last', y_axis_spec='bg', drr_assignments=(),
            drr_background_selection={}, provenance_records=(),
            drr_fitted_references=(self.first, self.second))

    def test_export_rejects_legacy_or_mismatched_group_before_writing(self):
        from ui_qt.common import ExportOptions
        from ui_qt.main_window import MainWindow
        signal = SimpleNamespace(emit=lambda *_: None)
        for references in ((self.first, self.second), (self.first,), ()):
            self.loaded.drr_fitted_references = references
            with self.subTest(count=len(references)), patch(
                    'core.drr_fitted_background.export_references') as export:
                with self.assertRaisesRegex(ValueError, 'Reload DRR.*raw intensity average'):
                    MainWindow._export_task(None, self.loaded,
                        ExportOptions(mode='DRR', params=None, drr_cube=self.cube), progress=signal, log=signal)
                export.assert_not_called()
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_analysis_rejects_legacy_or_mismatched_group_before_writing(self):
        from core import drr_workspace_session as session
        for references in ((self.first, self.second), (self.first,), ()):
            self.loaded.drr_fitted_references = references
            with self.subTest(count=len(references)), patch.object(session, 'save_session') as save:
                with self.assertRaisesRegex(ValueError, 'Reload DRR.*raw intensity average'):
                    session.snapshot_loaded(self.loaded)
                save.assert_not_called()

    def test_valid_group_and_legacy_single_reference_preserve_provenance(self):
        from core import drr_workspace_session as session
        group = replace(self.first, input_sources=(
            dict(source='a.csv', path='a.csv', signature=[1, 2]),
            dict(source='b.csv', path='b.csv', signature=[3, 4])))
        for files, reference in ((['a.csv', 'b.csv'], group), (['a.csv'], self.first)):
            self.loaded.selected_files = files
            self.loaded.drr_fitted_references = (reference,)
            with self.subTest(files=files), patch.object(session, 'session_directory', return_value=self.folder):
                path = session.snapshot_loaded(self.loaded)
                datasets, _, _ = session.load_session(path)
            self.assertEqual(datasets[0].provenance['measurement_files'], files)
            self.assertIn('one fitted background', datasets[0].provenance['average_method'])
            self.assertEqual(datasets[0].provenance['fitted_background'], json.loads(json.dumps([reference.metadata()])))


if __name__ == '__main__':
    unittest.main()
