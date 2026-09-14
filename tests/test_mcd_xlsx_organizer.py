import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.mcd_unified_export import export_mcd_analysis
from core.mcd_extract import discover_processed_mcd, load_branch_traces, index_processed_mcd_settings, organize_mcd_series
from tests.test_mcd_unified_export import _snapshot


class XlsxOrganizerTests(unittest.TestCase):
    def test_catalog_reads_selected_window_low_slopes_from_unified_json(self):
        with tempfile.TemporaryDirectory() as raw:
            snapshot = _snapshot()
            snapshot['slopes'] = [
                {'window_id': wid, 'center_ev': center, 'width_mev': 20,
                 'region': region, 'branch': branch, 'slope': value, 'status': 'ok', 'metric': 'Signed mean'}
                for wid, center, region, branch, value in (
                    ('w2', 1., 'low', 'B increasing', 99.),
                    ('w1', 1.5, 'high_positive', 'B increasing', 88.),
                    ('w1', 1.5, 'low', 'B increasing', .012),
                    ('w1', 1.5, 'low', 'B decreasing', .034),
                )]
            root = Path(raw) / 'Processed Data' / 'MCD'
            export_mcd_analysis(snapshot, root)
            records = discover_processed_mcd(root)
            self.assertEqual(records[0].increasing_slope_per_t, .012)
            self.assertEqual(records[0].decreasing_slope_per_t, .034)
            self.assertAlmostEqual(records[0].slope('B increasing', 'low_minus_positive'), .012 - 88.)
            self.assertIsNone(records[0].slope('B decreasing', 'low_minus_positive'))
            self.assertIsNone(records[0].slope('B increasing', 'low_minus_negative'))

    def test_missing_export_conditions_are_recovered_for_efield_series(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / 'Processed Data' / 'MCD'
            for field in (10, 20):
                source = Path(raw) / f'sample_1p67K_D6p3_F{field}_Vb0.csv'
                source.write_text(f'Doping_V,Efield_V,Vbias_V\n6.3,{field},0\n')
                snapshot = _snapshot()
                snapshot['source_descriptor'] = {'filename': source.name, 'path': str(source)}
                export_mcd_analysis(snapshot, root)
            records = discover_processed_mcd(root, rebuild_catalog=True)
            self.assertEqual({r.condition_value('E-field') for r in records}, {10, 20})
            self.assertEqual({r.condition_value('Doping') for r in records}, {6.3})
            self.assertEqual(len(organize_mcd_series(records, 'E-field', include_singletons=False)), 1)

    def test_unified_export_preserves_acquisition_conditions(self):
        with tempfile.TemporaryDirectory() as raw:
            snapshot = _snapshot()
            snapshot['acquisition_conditions'] = {'Doping': [6.3, 6.3], 'E-field': [10, 10]}
            result = export_mcd_analysis(snapshot, raw)
            payload = json.loads(Path(result['metadata_json']).read_text())
            self.assertEqual(payload.get('acquisition_conditions'), snapshot['acquisition_conditions'])

    def test_save_without_csv_round_trips_selected_window_and_branches(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / 'Processed Data' / 'MCD'
            root.mkdir(parents=True)
            self.assertEqual(discover_processed_mcd(root), [])
            snapshot = _snapshot()
            # The current window is second, so choosing the first column is wrong.
            snapshot['settings']['mcd_center_ev'] = 1.0
            result = export_mcd_analysis(snapshot, root)
            self.assertFalse(list(root.rglob('*.csv')))
            self.assertNotIn('trace_csv', result)
            records = discover_processed_mcd(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].trace_path.suffix, '.xlsx')
            traces = load_branch_traces(records[0])
            inc = traces[traces.branch == 'B increasing']
            dec = traces[traces.branch == 'B decreasing']
            np.testing.assert_allclose(inc.B_T, [0, 1, 2])
            np.testing.assert_allclose(inc.corrected_signed_mean, [0, 1, 2])
            np.testing.assert_allclose(dec.B_T, [-2, -1])
            np.testing.assert_allclose(dec.corrected_signed_mean, [-2, -1])
            self.assertEqual(len(discover_processed_mcd(root, rebuild_catalog=True)), 1)
            again = export_mcd_analysis(snapshot, root)
            self.assertEqual(again['xlsx'], result['xlsx'])

    def test_incomplete_branch_is_not_catalogued(self):
        with tempfile.TemporaryDirectory() as raw:
            snapshot = _snapshot()
            snapshot['plot_state']['visible_branches'] = ['B increasing']
            root = Path(raw) / 'Processed Data' / 'MCD'
            result = export_mcd_analysis(snapshot, root)
            self.assertFalse(index_processed_mcd_settings(result['metadata_json']))
            self.assertEqual(discover_processed_mcd(root, rebuild_catalog=True), [])

    def test_integral_window_does_not_get_read_as_signed_mean(self):
        with tempfile.TemporaryDirectory() as raw:
            snapshot = _snapshot()
            snapshot['windows'] = [dict(snapshot['windows'][0], metric='integral')]
            snapshot['settings']['metric'] = 'integral'
            root = Path(raw) / 'Processed Data' / 'MCD'
            export_mcd_analysis(snapshot, root)
            traces = load_branch_traces(discover_processed_mcd(root)[0])
            inc = traces[traces.branch == 'B increasing']
            np.testing.assert_allclose(inc.corrected_signed_mean, [0, .5, 1])
            np.testing.assert_allclose(inc.corrected_integral, [0, 0, 0])

    def test_invalid_workbook_or_mapping_does_not_break_catalog_rebuild(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / 'Processed Data' / 'MCD'
            result = export_mcd_analysis(_snapshot(), root)
            settings = Path(result['metadata_json'])
            original = json.loads(settings.read_text())
            for traces in (None, [{'center_ev': 1.5, 'width_mev': 20,
                                   'branch': 'B increasing', 'columns': None}]):
                with self.subTest(traces=traces):
                    payload = json.loads(json.dumps(original))
                    payload['trace_workbook']['traces'] = traces
                    settings.write_text(json.dumps(payload))
                    self.assertEqual(discover_processed_mcd(root, rebuild_catalog=True), [])
            settings.write_text(json.dumps(original))
            Path(result['xlsx']).write_bytes(b'not a zip file')
            self.assertEqual(discover_processed_mcd(root, rebuild_catalog=True), [])
