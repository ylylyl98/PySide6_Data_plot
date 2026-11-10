"""Tests for frozen-reference feature injection and input validation."""
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import numpy as np
from tests.bn_injection_audit import audit_injection


class BnInjectionAuditTests(unittest.TestCase):
    def setUp(self):
        self.e = np.linspace(1.5, 1.95, 901)
        self.gates = np.array([-2., 0., 2.])
        self.bn = np.full(len(self.e), 2.)
        self.raw = np.tile(self.bn*(1.5 + .1*(self.e-1.7)), (len(self.gates), 1))

    def evaluator(self, absorption=0., switch=False):
        calls = []
        def evaluate(e, z, b, **kwargs):
            calls.append(z.copy())
            reference = self.raw if e[0] < e[-1] else self.raw[:, ::-1]
            change = np.mean(z/reference-1, axis=0)
            applied = b*1.1*(1+absorption*change)
            model = 'changed' if switch and len(calls) > 1 else 'baseline'
            return SimpleNamespace(applied_background=applied, eligible=True, selected_method=model, report={})
        return evaluate, calls

    def audit(self, evaluator, *, reverse=False):
        return audit_injection(self.e[::-1] if reverse else self.e, self.raw[:, ::-1] if reverse else self.raw,
            self.bn[::-1] if reverse else self.bn, self.gates, center_ev=1.8, sigma_mev=2.,
            amplitude=.001, evaluator=evaluator)

    def test_frozen_calibrated_baseline_and_original_bn_are_distinct(self):
        evaluator, calls = self.evaluator()
        result = self.audit(evaluator)
        self.assertEqual(len(calls), 2)
        self.assertFalse(result['model_switched'])
        for row in result['gates']:
            self.assertAlmostEqual(row['peak_ratio'], 1.)
            self.assertAlmostEqual(row['area_ratio'], 1.)
            self.assertAlmostEqual(row['peak_ratio_original_bn'], 1/1.1)
            self.assertAlmostEqual(row['area_ratio_original_bn'], 1/1.1)

    def test_absorption_and_model_switch_are_visible(self):
        evaluator, _ = self.evaluator(absorption=.5, switch=True)
        result = self.audit(evaluator)
        self.assertTrue(result['model_switched'])
        self.assertEqual(result['injected_model'], 'changed')
        self.assertTrue(.49 < result['gates'][0]['peak_ratio'] < .51)
        self.assertGreater(result['background_log_change_max'], 0)

    def test_ineligible_preview_audit_detects_absorption(self):
        calls = []
        def evaluator(e, z, b, **kwargs):
            calls.append(z.copy())
            change = np.mean(z/self.raw-1, axis=0)
            preview = b*1.1*(1+.5*change)
            return SimpleNamespace(applied_background=b, candidate_backgrounds={'gap_preview': preview},
                                   eligible=False, selected_method='gap_preview', report={})
        result = audit_injection(self.e, self.raw, self.bn, self.gates, center_ev=1.8,
            sigma_mev=2., amplitude=.001, evaluator=evaluator, audit_selected_candidate=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['audit_target'], 'selected_preview_candidate')
        self.assertFalse(result['baseline_eligible'])
        self.assertFalse(result['injected_eligible'])
        self.assertTrue(.49 < result['gates'][0]['peak_ratio'] < .51)
        self.assertGreater(result['background_log_change_max'], 0)

    def test_shared_trend_order_change_is_not_hidden_by_common_model_label(self):
        calls = []
        def evaluator(e, z, b, **kwargs):
            calls.append(z.copy())
            name = 'trend_linear' if len(calls) == 1 else 'trend_quadratic'
            return SimpleNamespace(applied_background=b, eligible=True, selected_method=name,
                report={'optimization_policy': 'shared_bn_trend_v1',
                        'single_pair_scope': {'model': 'shared_trend_normalization'}})
        result = self.audit(evaluator)
        self.assertTrue(result['model_switched'])
        self.assertEqual(result['injected_model'], 'shared_trend_normalization:trend_quadratic')

    def test_gap_benchmark_uses_preview_for_plots_and_injections(self):
        from tests import benchmark_bn_calibration as benchmark
        e, g, z, b = self.e, self.gates, self.raw, self.bn
        frames = np.stack((b, b))
        def parse(path):
            return (e, g, g, None, z) if Path(path).name == 'sample.csv' else (e, np.zeros(2), np.zeros(2), None, frames)
        candidate = b*1.2
        preview_result = SimpleNamespace(eligible=False, selected_method='gap_preview', applied_background=b,
            candidate_backgrounds={'original': b, 'gap_preview': candidate}, status='preview_only', reasons=[],
            report={'single_pair_scope': {'model': 'gap_preview'}})
        captured = []
        def audit(*args, **kwargs):
            captured.append(kwargs)
            return dict(center_ev=kwargs['center_ev'], model_switched=False)
        with TemporaryDirectory() as tmp:
            root = Path(tmp); sample = root/'sample.csv'; bn = root/'bn.csv'
            sample.write_text('sample'); bn.write_text('bn')
            with patch.object(benchmark, '_parse_raw_spectrum', side_effect=parse), \
                 patch.object(benchmark, 'inspect_csv_spectral_grid', return_value=1240/e), \
                 patch.object(benchmark, '_draw'), \
                 patch.object(benchmark, '_algorithm_identity', return_value={}), \
                 patch.object(benchmark, 'audit_injection', side_effect=audit), \
                 patch('core.drr_bn_gap_prediction.evaluate_bn_gap', return_value=preview_result) as evaluate:
                summary = benchmark.run(sample, bn, root/'out', method='gap', injections=True)
                preview_result.report['candidate_generated'] = False
                preview_result.candidate_backgrounds['gap_preview'] = b
                empty_summary = benchmark.run(sample, bn, root/'flat', method='gap')
            self.assertEqual(evaluate.call_count, 2)
            self.assertFalse(empty_summary['candidate_generated'])
            self.assertEqual(len(captured), 12)
            self.assertTrue(all(item['audit_selected_candidate'] and item['evaluator'] is evaluate for item in captured))
            self.assertTrue(summary['preview_only'])
            self.assertTrue(summary['candidate_generated'])
            self.assertEqual(summary['report_file'], 'report.json')
            self.assertEqual(json.loads((root/'out'/'report.json').read_text())['single_pair_scope']['model'], 'gap_preview')
            self.assertFalse(summary['eligible'])
            self.assertEqual(summary['audit_target'], 'selected_preview_candidate')
            self.assertTrue(summary['gap_module_sha256'])
            with np.load(root/'out'/'arrays.npz') as data:
                np.testing.assert_allclose(data['candidate_background'], candidate)
                np.testing.assert_allclose(data['candidate_drr'], z/candidate-1)
            self.assertIn('gap_preview_intensity', (root/'out'/'gap_preview_background.csv').read_text())
            self.assertTrue((root/'out'/'gap_preview_d2.dat').is_file())
            self.assertEqual(json.loads((root/'out'/'summary.json').read_text())['benchmark_method'], 'gap')

    def test_reversed_axis_retains_response_and_invalid_ranges_fail(self):
        evaluator, _ = self.evaluator()
        forward = self.audit(evaluator)
        evaluator, _ = self.evaluator()
        reverse = self.audit(evaluator, reverse=True)
        np.testing.assert_allclose([r['peak_ratio'] for r in forward['gates']],
                                   [r['peak_ratio'] for r in reverse['gates']], rtol=1e-9)
        evaluator, _ = self.evaluator()
        with self.assertRaises(ValueError):
            audit_injection(self.e, self.raw, self.bn, self.gates, center_ev=1.5,
                            sigma_mev=5., amplitude=.001, evaluator=evaluator)
        with self.assertRaises(ValueError):
            audit_injection(self.e, self.raw, self.bn, self.gates, center_ev=1.8,
                            sigma_mev=2., amplitude=float('nan'), evaluator=evaluator)


if __name__ == '__main__':
    unittest.main()
