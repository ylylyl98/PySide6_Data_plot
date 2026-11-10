import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ui_qt.quick.drr_workflow import DrrRequest
from ui_qt.quick.peak_source_adapters import _processed_path, _path_is_inferred, bind_exported_paths


class ProcessedPathTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path('experiment')
        self.owner = SimpleNamespace(_folder=str(self.folder), _source=str(self.folder / 'Initial Data' / 'sample.csv'))

    def test_pl_and_imported_dat(self):
        with patch.object(Path, 'exists', side_effect=AssertionError('unexpected file access')):
            self.assertEqual(_processed_path(self.owner, 'pl'), str(self.folder / 'Processed Data' / 'PL' / 'sample_PL_linear.dat'))
            self.owner._source = str(self.folder / 'Processed Data' / 'PL' / 'saved.dat')
            self.assertEqual(_processed_path(self.owner, 'pl'), self.owner._source)

    def test_drr_group_modes_and_derivatives(self):
        files = (str(self.folder / 'Initial Data' / 'sample.csv'), 'other.csv')
        for mode, label in [('self-first', 'DR_R_Self_First'), ('self-last', 'DR_R_Self'),
                            ('external', 'DR_R_External'), ('fitted', 'DR_R_Fitted')]:
            request = DrrRequest(files=files, background_mode=mode)
            products = SimpleNamespace(request=request, raw_window=21)
            base = self.folder / 'Processed Data' / 'DRR'
            with self.subTest(mode=mode), patch.object(Path, 'exists', side_effect=AssertionError('unexpected file access')):
                self.assertEqual(_processed_path(self.owner, 'drr', None, products), str(base / f'sample_avg2_{label}.dat'))
                self.assertEqual(_processed_path(self.owner, 'drr', 2, products), str(base / f'sample_avg2_{label}_d2E.dat'))
                self.assertEqual(_processed_path(self.owner, 'drr', ('mixed', 11), products), str(base / f'sample_avg2_{label}_dXdY.dat'))

    def test_imported_drr_derivative_gets_new_path(self):
        self.owner._source = str(self.folder / 'Processed Data' / 'DRR' / 'old.dat')
        request = DrrRequest(files=(self.owner._source,), background_mode='imported')
        products = SimpleNamespace(request=request, raw_window=21)
        self.assertEqual(_processed_path(self.owner, 'drr', None, products), self.owner._source)
        self.assertEqual(_processed_path(self.owner, 'drr', 1, products),
                         str(self.folder / 'Processed Data' / 'DRR' / 'old_avg1_DR_R_Imported_dE.dat'))

    def test_successful_export_binds_each_product_to_actual_collision_path(self):
        raw, second = SimpleNamespace(), SimpleNamespace()
        products = SimpleNamespace(request=DrrRequest(files=(self.owner._source,)), raw=raw, second=second,
                                   raw_window=21, second_window=15)
        self.owner._workflow = 'drr'
        self.owner._drr_products = products
        self.assertTrue(_path_is_inferred(self.owner, 'drr', 2, products, 'second'))
        bind_exported_paths(self.owner, {'raw_dat': 'raw_01.dat', 'second_dat': 'second_02.dat'})
        self.assertEqual(_processed_path(self.owner, 'drr', None, products), 'raw_01.dat')
        self.assertEqual(_processed_path(self.owner, 'drr', 2, products, 'second'), 'second_02.dat')
        self.assertFalse(_path_is_inferred(self.owner, 'drr', 2, products, 'second'))

    def test_imported_dat_retains_identity_even_after_export(self):
        self.owner._source = 'imported.dat'
        self.owner._workflow = 'pl'
        self.owner._cube = SimpleNamespace()
        bind_exported_paths(self.owner, {'dat': 'exported_01.dat'})
        self.assertEqual(_processed_path(self.owner, 'pl'), 'imported.dat')
        self.assertFalse(_path_is_inferred(self.owner, 'pl'))


if __name__ == '__main__':
    unittest.main()
