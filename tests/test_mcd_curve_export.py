from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from core.mcd_curve_export import export_mcd_curve_csvs
from core.mcd_extract import ProcessedMcdRecord


class McdCurveExportTests(unittest.TestCase):
    def record(self, root):
        trace = root/'trace.csv'
        # Independent native grids, an unequal branch length, and repeated B.
        pd.DataFrame({
            'B_increasing_T': [-.6, -.2, 0., 0., .2],
            'corrected_signed_mean_increasing': [-6., -2., 0., .1, 2.],
            'B_decreasing_T': [.6, .1, -.3, None, None],
            'corrected_signed_mean_decreasing': [6., 1., -3., None, None],
        }).to_csv(trace, index=False)
        return ProcessedMcdRecord(
            'one', root/'settings.json', trace, 'raw.csv', 'pkg', '',
            1.62, 5., 'mean', .2,
            {'T': (1.67, 1.67), 'Doping': (6.3, 6.3), 'E-field': (24., 24.), 'Vbias': (0., 0.)},
            {'T': 'setpoint'}, 1., 1., 1.67, 1.72,
        )

    def test_full_curves_preserve_native_grids_repeats_and_group_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = self.record(root)
            second = replace(first, record_id='two', center_ev=1.63)
            repeat = replace(first, record_id='repeat', source_file='repeat.csv')
            groups = {'one': 'Series 1 / Group 2', 'two': 'Series 1_Group 2',
                      'repeat': 'Series 1 / Group 2'}
            index = export_mcd_curve_csvs([first, second, repeat], root/'curves', energy_groups=groups)
            rows = pd.read_csv(index)
            self.assertEqual(list(rows['record_id']), ['one', 'two', 'repeat'])
            self.assertEqual(list(rows['energy_group']), list(groups.values()))
            self.assertEqual(rows['csv_file'].nunique(), 3)
            self.assertEqual(len(list(index.parent.rglob('T*.csv'))), 3)
            self.assertEqual(set(rows['temperature_K']), {1.67})
            self.assertEqual(set(rows['temperature_measured_K']), {1.72})
            self.assertEqual(set(rows['efield_V']), {24.})
            self.assertEqual(set(rows['doping_V']), {6.3})
            for row in rows.to_dict('records'):
                curve = pd.read_csv(index.parent/row['csv_file'])
                self.assertEqual(list(curve['B_increasing_T']), [-.6, -.2, 0., 0., .2])
                self.assertEqual(list(curve['MCD_increasing']), [-6., -2., 0., .1, 2.])
                self.assertEqual(list(curve['B_decreasing_T'].dropna()), [.6, .1, -.3])
                self.assertEqual(list(curve['MCD_decreasing'].dropna()), [6., 1., -3.])
                self.assertEqual((row['increasing_points'], row['decreasing_points']), (5, 3))
            # Reusing a destination cannot overwrite a previous export.
            with self.assertRaises(FileExistsError):
                export_mcd_curve_csvs([first], index.parent)

    def test_selected_branch_and_measured_temperature_use_exact_workbook_window(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = self.record(root)
            workbook = root/'windows.xlsx'
            pd.DataFrame({'B_dec': [.6, .1, -.3], 'window1': [60., 10., -30.],
                          'window2': [6., 1., -3.]}).to_excel(workbook, sheet_name='MCD', index=False)
            record = replace(record, trace_path=workbook, workbook_columns={
                'B_decreasing_T': 'B_dec', 'corrected_signed_mean_decreasing': 'window2',
                'corrected_integral_decreasing': 'unused_legacy_column',
                'B_increasing_T': 'unselected_branch',
            })
            index = export_mcd_curve_csvs([record], root/'curves', branches=('B decreasing',),
                                         energy_groups={'one': 'Group 2'}, temperature_mode='measured')
            row = pd.read_csv(index).iloc[0]
            self.assertEqual(row['temperature_K'], 1.72)
            self.assertEqual(row['temperature_source'], 'measured')
            self.assertIn('T1.72K', row['csv_file'])
            table = pd.read_csv(index.parent/row['csv_file'])
            self.assertEqual(list(table), ['B_decreasing_T', 'MCD_decreasing'])
            self.assertEqual(list(table['MCD_decreasing']), [6., 1., -3.])

    def test_missing_trace_is_reported_instead_of_synthesizing_curve_from_slope(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = replace(self.record(root), trace_path=root/'missing.csv')
            with self.assertRaisesRegex(OSError, 'missing.csv'):
                export_mcd_curve_csvs([record], root/'curves')
            self.assertFalse((root/'curves'/'index.csv').exists())


if __name__ == '__main__':
    unittest.main()
