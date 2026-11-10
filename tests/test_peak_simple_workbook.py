import tempfile
import unittest
from openpyxl import load_workbook
from core.peak_workspace import create_dataset
from core.peak_workbook import export_peak_workbook
from tests.test_peak_workspace import sample
from ui_qt.quick.peak_analysis_workflow import analyze_dataset


class SimplePeakWorkbookTests(unittest.TestCase):
    def dataset(self, kind):
        dataset = create_dataset(sample(kind), kind, 'PL' if kind == 'PL' else 'second', 'sample.dat', 'sample')
        dataset['settings'].update(detection_method='local', min_snr=0, neighbor_support=0)
        return analyze_dataset(dataset, 'find-fit' if kind == 'PL' else 'detect')

    def test_branch_labels_and_empty_branch_exclusion(self):
        dataset = self.dataset('PL')
        first, second = dataset['result']['branches'][:2]
        first['name'] = 'P9'
        second['name'] = 'Custom feature'
        with tempfile.TemporaryDirectory() as folder:
            book = load_workbook(export_peak_workbook(folder, dataset), read_only=True)
            try:
                header = next(book['Detected_peaks'].values)
                self.assertIn('P1 (eV)', header)
                self.assertIn('P2 - Custom feature (eV)', header)
                self.assertFalse(any('P9' in value for value in header))
            finally:
                book.close()
            for point in dataset['result']['points']:
                if point['branch_id'] == first['id']:
                    point['status'] = 'uncertain'
            book = load_workbook(export_peak_workbook(folder, dataset), read_only=True)
            try:
                for sheet in book:
                    header = next(sheet.values)
                    self.assertFalse(any(value.startswith('P1 ') for value in header[1:]))
                    self.assertTrue(any(value.startswith('P2 - Custom feature ') for value in header[1:]))
            finally:
                book.close()

    def test_disabled_branch_is_excluded_without_renumbering(self):
        dataset = self.dataset('PL')
        self.assertGreaterEqual(len(dataset['result']['branches']), 2)
        first, second = dataset['result']['branches'][:2]
        dataset['branches'] = [dict(branch) for branch in dataset['branches']]
        next(branch for branch in dataset['branches'] if branch['id'] == first['id'])['enabled'] = False
        self.assertTrue(any(p['branch_id'] == first['id'] for p in dataset['result']['points']))
        with tempfile.TemporaryDirectory() as folder:
            book = load_workbook(export_peak_workbook(folder, dataset), read_only=True)
            try:
                headers = next(book['Detected_peaks'].values)
                self.assertFalse(any(h.startswith('P1 ') for h in headers[1:]))
                self.assertTrue(any(h.startswith('P2 ') for h in headers[1:]))
                self.assertTrue(any(p['branch_id'] == second['id'] for p in dataset['result']['points']))
            finally:
                book.close()

    def test_drr_peaks_and_dips_are_numeric_plot_tables(self):
        dataset = self.dataset('DRR')
        with tempfile.TemporaryDirectory() as folder:
            book = load_workbook(export_peak_workbook(folder, dataset, filename='result.xlsx'), read_only=True)
            try:
                self.assertEqual(book.sheetnames, ['Peaks', 'Dips'])
                for sheet in book:
                    rows = list(sheet.values)
                    self.assertTrue(all('(eV)' in h for h in rows[0][1:]))
                    self.assertTrue(all(v is None or isinstance(v, (float, int)) for row in rows[1:] for v in row))
                    self.assertFalse(any('status' in h.casefold() for h in rows[0]))
            finally:
                book.close()

    def test_pl_fit_tables_and_explicit_detailed_export(self):
        dataset = self.dataset('PL')
        self.assertTrue(dataset['result']['fits'])
        first = dataset['result']['branches'][0]
        first['name'] = 'Manual branch'
        point = next(p for p in dataset['result']['points'] if p['branch_id'] == first['id'] and p.get('fit_status') == 'ok')
        point['fit_area_full'] = 123.25
        point['fit_area_full_error'] = 0.125
        point['area'] = 999.
        with tempfile.TemporaryDirectory() as folder:
            book = load_workbook(export_peak_workbook(folder, dataset, filename='compact.xlsx'), read_only=True)
            try:
                self.assertEqual(book.sheetnames, ['Detected_peaks', 'Fitted_peaks', 'FWHM', 'Height', 'Area'])
                rows = list(book['Area'].values)
                col = next(i for i,h in enumerate(rows[0]) if 'Manual branch' in h and ' SE ' not in h)
                self.assertEqual(rows[point['row_index'] + 1][col:col+2], (123.25, .125))
                self.assertTrue(all('(eV)' in h for h in next(book['Detected_peaks'].values)[1:]))
            finally:
                book.close()
            detailed = load_workbook(export_peak_workbook(folder, dataset, filename='full.xlsx', compact=False), read_only=True)
            try:
                self.assertEqual(detailed.sheetnames, ['Results', 'Details', 'Curves', 'Components', 'Settings'])
            finally:
                detailed.close()

    def test_failed_fit_is_blank(self):
        dataset = self.dataset('PL')
        point = next(p for p in dataset['result']['points'] if p.get('fit_status') == 'ok')
        point.update(fit_status='failed', fit_center=7.5, fit_area_full=8.)
        dataset['result']['points'] = [point]
        with tempfile.TemporaryDirectory() as folder:
            book = load_workbook(export_peak_workbook(folder, dataset), read_only=True)
            try:
                rows = list(book['Fitted_peaks'].values)
                self.assertTrue(all(v is None for v in rows[point['row_index']+1][1:]))
            finally:
                book.close()

    def test_pl_without_fit_only_exports_detected_peaks(self):
        dataset = self.dataset('PL')
        dataset['result']['fits'] = []
        for point in dataset['result']['points']:
            point.pop('fit_status', None)
        with tempfile.TemporaryDirectory() as folder:
            book = load_workbook(export_peak_workbook(folder, dataset), read_only=True)
            try:
                self.assertEqual(book.sheetnames, ['Detected_peaks'])
            finally:
                book.close()

    def test_uncertain_detection_is_blank(self):
        dataset = self.dataset('DRR')
        point = dataset['result']['points'][0]
        point['status'] = 'uncertain'
        dataset['result']['points'] = [point]
        with tempfile.TemporaryDirectory() as folder:
            book = load_workbook(export_peak_workbook(folder, dataset), read_only=True)
            try:
                sheet = book['Peaks' if point['polarity'] == 'peak' else 'Dips']
                self.assertTrue(all(v is None for v in list(sheet.values)[point['row_index']+1][1:]))
            finally:
                book.close()
