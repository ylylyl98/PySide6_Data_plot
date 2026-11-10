import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from openpyxl import load_workbook
from core.peak_workspace import create_dataset
from core.peak_workbook import export_peak_workbook
from tests.test_peak_workspace import sample
from ui_qt.quick.peak_analysis_workflow import analyze_dataset


class WorkbookTests(unittest.TestCase):
    def test_small_uncertainties_are_not_displayed_as_zero(self):
        from core.analysis_workbook import AnalysisTable, write_analysis_workbook
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'errors.xlsx'
            write_analysis_workbook(path, [AnalysisTable('Fits', ['Center_SE_eV'], [[1.2e-12]])])
            book = self.read(path)
            try:
                self.assertEqual(book['Fits']['A2'].value, 1.2e-12)
                self.assertIn('E+', book['Fits']['A2'].number_format)
            finally:
                book.close()

    def test_mid_stream_cancel_releases_xml_and_preserves_existing_file(self):
        from core.analysis_workbook import AnalysisTable, write_analysis_workbook
        from openpyxl.worksheet._writer import ALL_TEMP_FILES
        before = set(ALL_TEMP_FILES)
        calls = []
        def cancel():
            calls.append(True)
            return len(calls) >= 3
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'existing.xlsx'
            path.write_bytes(b'original')
            with self.assertRaises(InterruptedError):
                write_analysis_workbook(path, [AnalysisTable('Rows', ['X'], ([i] for i in range(1000)))], cancelled=cancel)
            self.assertEqual(path.read_bytes(), b'original')
            self.assertEqual(set(ALL_TEMP_FILES), before)

    def dataset(self, kind):
        d = create_dataset(sample(kind), kind, 'PL' if kind == 'PL' else 'second', '=SUM(1,2).csv', kind)
        d['settings'].update(detection_method='local', min_snr=0, neighbor_support=0)
        return analyze_dataset(d, 'find-fit' if kind == 'PL' else 'detect')

    def read(self, path):
        return load_workbook(path, read_only=True, data_only=False)

    def test_pl_roundtrip_missing_fit_and_formula_source(self):
        d = self.dataset('PL')
        d['result']['fits'] = [f for f in d['result']['fits'] if f['row_index'] != 1]
        d['result']['points'] = [p for p in d['result']['points'] if p['row_index'] != 1]
        with tempfile.TemporaryDirectory() as tmp:
            path = export_peak_workbook(tmp, d, compact=False)
            book = self.read(path)
            rows = list(book['Results'].values); head = rows[0]
            self.assertIn('P1_fit_area_full', head); self.assertIn('P1_center_err_eV', head)
            self.assertEqual(rows[2][head.index('P1_detection_status')], 'not detected')
            self.assertIsNone(rows[2][head.index('Spectrum_area_roi')])
            settings = list(book['Settings'].iter_rows())
            source = next(row[1] for row in settings if row[0].value == 'Source filename')
            self.assertEqual(source.value, '=SUM(1,2).csv'); self.assertEqual(source.data_type, 's')
            branch = next(row[1].value for row in settings if row[0].value == 'Branch: P1')
            self.assertIn(d['result']['branches'][0]['id'], branch)
            book.close()

    def test_drr_separate_aliases_and_no_false_fit_for_current_unanalyzed(self):
        d = self.dataset('DRR')
        with tempfile.TemporaryDirectory() as tmp:
            book = self.read(export_peak_workbook(tmp, d, compact=False))
            self.assertIn('P1_detected_eV', next(book['Peaks'].values))
            self.assertIn('D1_detected_eV', next(book['Dips'].values))
            book.close()
        d = self.dataset('PL'); d['row'] = 1
        d['result']['fits'] = [f for f in d['result']['fits'] if f['row_index'] != 1]
        with tempfile.TemporaryDirectory() as tmp:
            book = self.read(export_peak_workbook(tmp, d, compact=False))
            rows = list(book['Curves'].values); head=rows[0]
            self.assertTrue(all(row[head.index('Total_fit')] is None and row[head.index('Fit_status')] == 'not fitted' for row in rows[1:]))
            self.assertEqual(len(list(book['Components'].values)), 1)
            book.close()

    def test_aliases_remain_stable_across_filter(self):
        d = self.dataset('PL')
        self.assertGreaterEqual(len(d['result']['branches']), 2)
        first, second = d['result']['branches'][:2]
        with tempfile.TemporaryDirectory() as tmp:
            book = self.read(export_peak_workbook(tmp, d, compact=False))
            initial = next(book['Results'].values)
            self.assertIn('P2_detected_eV', initial); book.close()
            reduced = deepcopy(d)
            reduced['result']['branches'] = [second]
            reduced['branches'] = [second]
            reduced['result']['points'] = [p for p in reduced['result']['points'] if p['branch_id'] == second['id']]
            book = self.read(export_peak_workbook(tmp, reduced, compact=False))
            header = next(book['Results'].values)
            self.assertIn('P2_detected_eV', header); self.assertNotIn('P1_detected_eV', header)
            self.assertNotEqual(first['id'], second['id']); book.close()

    def test_cancel_no_publish_and_long_source(self):
        d = self.dataset('PL'); d['source'] = 'A' * 240 + '.csv'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InterruptedError): export_peak_workbook(tmp, d, cancelled=lambda: True)
            self.assertEqual(list(Path(tmp).iterdir()), [])
            path = export_peak_workbook(tmp, d, compact=False)
            self.assertLessEqual(len(path.name), 180)
            book = self.read(path); self.assertEqual(book['Settings'].title, 'Settings'); book.close()

if __name__ == '__main__': unittest.main()
