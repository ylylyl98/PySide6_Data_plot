"""Shared BN trend normalization through the real CSV/store/application path."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_bn_service import compare_bn_group
from core.drr_bn_store import DrrBnStore, calibration_usable, joint_validation_passed, _validate_trend_payload
from core.drr_bn_single_exports import save_single_pair_products
from ui_qt.quick.drr_background_workflow import (
    BackgroundContext, BackgroundDraft, prepare_application, load_with_recipe,
    commit_application,
)


class TrendLifecycleTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.energy = np.linspace(1.4, 2.1, 601)
        self.gates = np.linspace(-1, 1, 7)
        self.bn = 'YZ382_pa2_9T_1.67KREF_730nmc_0p1sx20_RotIn39deg_BNbackground_PIXIS.csv'
        self.measurement = 'YZ382_pa2_9T_1.67KREF_730nmc_0p1sx20_RotIn39deg_TG-1BG=0_PIXIS.csv'
        background = 10000*np.exp(.015*np.sin(np.arange(601)*.3))
        gain = np.exp(.18+.22*(self.energy-1.75))
        raw = background*gain*(1+.01*self.gates[:, None])
        raw *= 1+.2*np.exp(-((self.energy[None, :]-1.72-.01*self.gates[:, None])/.008)**2)
        for name, rows, gates in ((self.measurement, raw, self.gates),
                                   (self.bn, np.tile(background, (4, 1)), np.zeros(4))):
            with (self.root/name).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Vbg', 'Vtg', *(1240/self.energy)])
                for gate, row in zip(gates, rows):
                    writer.writerow([gate, gate, *row])
        self.config = dict(single_measurement=True, single_trend_enabled=True)
        self.context = BackgroundContext(str(self.root), (self.measurement,), y_axis='BG')
        self.store = DrrBnStore(self.root)

    def compare(self):
        return compare_bn_group(self.root, (self.measurement,), (self.bn,),
                                settings=self.config, y_axis='BG')

    def test_explicit_apply_reload_cache_and_shared_gate_relationship(self):
        payload = self.compare()
        self.assertTrue(payload['eligible'], payload['report'])
        self.assertEqual(payload['report']['optimization_policy'], 'shared_bn_trend_v2')
        self.assertEqual(payload['report']['schema_version'], 2)
        self.assertEqual(payload['selected_candidate'], 'trend_linear')
        self.assertTrue(calibration_usable(payload['report'], payload['settings']))
        self.assertFalse(joint_validation_passed(payload['report'], payload['settings']))
        version = self.store.save_result(payload)
        cached = self.compare()
        self.assertTrue(cached['cached'])
        self.assertEqual(cached['version_id'], version.version_id)
        draft = BackgroundDraft(kind='bn', files=(self.bn,), which='all',
                                calibration={'payload': cached, 'original': False})
        prepared = prepare_application(self.context, draft)
        cube = load_with_recipe(self.context, prepared.recipe, prepared=prepared).cube
        expected = payload['mean_raw']/payload['candidate_background']-1
        np.testing.assert_allclose(cube.Z, expected, atol=1e-12)
        np.testing.assert_allclose((cube.Z+1)/(cube.Z[0]+1),
                                   payload['mean_raw']/payload['mean_raw'][0], atol=1e-12)
        from PySide6.QtCore import QSettings
        settings = QSettings(str(self.root/'settings.ini'), QSettings.Format.IniFormat)
        commit_application(prepared, current_context=self.context, settings=settings)
        self.assertEqual(self.store.find_applied((self.measurement,)).version_id, version.version_id)

    def test_exports_preserve_both_models_and_numerical_matrices(self):
        payload = self.compare()
        version = self.store.save_result(payload)
        output = save_single_pair_products(payload, version)
        paths = output['paths']
        for key in ('trend_models_png', 'trend_gain_png', 'trend_backgrounds_csv',
                    'trend_linear_drr_dat', 'trend_quadratic_d2_dat'):
            self.assertTrue(Path(paths[key]).is_file())
        matrix = np.loadtxt(paths['trend_linear_drr_dat'], skiprows=1)
        np.testing.assert_allclose(matrix[:, 1:].T,
            payload['mean_raw']/payload['candidate_backgrounds']['trend_linear']-1,
            rtol=1e-9, atol=1e-11)
        backgrounds = np.loadtxt(paths['trend_backgrounds_csv'], delimiter=',', skiprows=1)
        np.testing.assert_array_equal(backgrounds[:, 2], payload['candidate_backgrounds']['trend_linear'])
        metadata = json.loads(Path(paths['metadata_json']).read_text(encoding='utf-8'))
        self.assertEqual(metadata['normalization_kind'], 'shared_spectral_trend')
        self.assertFalse(metadata['independent_validation'])
        self.assertTrue(save_single_pair_products(payload, version)['cached'])

    def test_store_rejects_changed_array_polynomial_scope_and_settings(self):
        payload = self.compare()
        wrong = deepcopy(payload)
        wrong['candidate_backgrounds']['trend_linear'] *= 1.01
        wrong['candidate_background'] = wrong['candidate_backgrounds']['trend_linear'].copy()
        with self.assertRaisesRegex(ValueError, 'gain curve'):
            self.store.save_result(wrong)
        wrong = deepcopy(payload)
        wrong['report']['candidates']['trend_linear']['parameters']['strength'] = .125
        if wrong['report']['models']['trend_linear']['strength'] == .125:
            wrong['report']['candidates']['trend_linear']['parameters']['strength'] = .25
        with self.assertRaisesRegex(ValueError, 'strength'):
            self.store.save_result(wrong)
        wrong = deepcopy(payload)
        wrong['settings']['single_trend_enabled'] = False
        with self.assertRaises(ValueError):
            self.store.save_result(wrong)
        with self.assertRaisesRegex(ValueError, 'single-measurement'):
            compare_bn_group(self.root, (self.measurement,), (self.bn,),
                             settings=dict(single_trend_enabled=True))

    def test_v2_gain_and_strength_are_bound_to_every_candidate(self):
        energy = np.linspace(1.4, 2.1, 51)
        original = np.full_like(energy, 1000.)
        center, scale = 1.75, .35
        models, candidates, descriptions = {}, {'original': original}, {}
        for name, base, strength in (
                ('trend_linear', np.array([.1, .03]), .546875),
                ('trend_quadratic', np.array([.12, .02, -.01]), .25)):
            final = base * strength
            coeff = [value / scale**i for i, value in enumerate(final)]
            models[name] = dict(base_coefficients=base.tolist(), coefficients=final.tolist(), strength=strength)
            descriptions[name] = dict(parameters=dict(coefficients=coeff, energy_center_ev=center, strength=strength))
            candidates[name] = original * np.exp(np.polynomial.polynomial.polyval(energy-center, coeff))
        payload = dict(energy=energy, original_background=original, candidate_backgrounds=candidates,
            selected_candidate='trend_linear', candidate_background=candidates['trend_linear'].copy(),
            report=dict(optimization_policy='shared_bn_trend_v2', selected_candidate='trend_linear',
                coordinate=dict(center_ev=center, scale_ev=scale), models=models, candidates=descriptions))
        _validate_trend_payload(payload)
        for path in ('strength', 'off_grid_strength', 'base', 'array', 'selection'):
            with self.subTest(tamper=path):
                bad = deepcopy(payload)
                if path == 'strength': bad['report']['candidates']['trend_linear']['parameters']['strength'] = .75
                elif path == 'off_grid_strength':
                    strength = .6
                    model = bad['report']['models']['trend_linear']
                    model['strength'] = strength
                    model['coefficients'] = (np.asarray(model['base_coefficients']) * strength).tolist()
                    params = bad['report']['candidates']['trend_linear']['parameters']
                    params['strength'] = strength
                    params['coefficients'] = [value / scale**i for i, value in enumerate(model['coefficients'])]
                    altered = original * np.exp(np.polynomial.polynomial.polyval(energy-center, params['coefficients']))
                    bad['candidate_backgrounds']['trend_linear'] = altered
                    bad['candidate_background'] = altered.copy()
                elif path == 'base': bad['report']['models']['trend_linear']['base_coefficients'][0] += .1
                elif path == 'array': bad['candidate_backgrounds']['trend_quadratic'][10] *= 1.001
                else: bad['candidate_background'][10] *= 1.001
                with self.assertRaises(ValueError): _validate_trend_payload(bad)

    def test_batch_individual_adaptive_then_mean_drr(self):
        from ui_qt.quick.drr_bn_batch_workflow import compare_bn_batch, verify_batch
        second = 'YZ382_pa3_9T_1.67KREF_730nmc_0p1sx20_RotIn39deg_TG-1BG=0_PIXIS.csv'
        first_payload = self.compare()
        raw = first_payload['mean_raw'] * 2.1 * np.exp(.38*(self.energy-1.75))[None, :]
        with (self.root / second).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Vbg', 'Vtg', *(1240/self.energy)])
            for gate, row in zip(self.gates, raw):
                writer.writerow([gate, gate, *row])
        context = BackgroundContext(str(self.root), (self.measurement, second), y_axis='BG')
        with patch('core.drr_bn_single_exports.save_single_pair_products') as single_export, \
             patch('ui_qt.quick.drr_bn_batch_workflow._export_batch') as batch_export:
            batch = compare_bn_batch(context, (self.bn,), 'all',
                                     settings={'single_trend_enabled': True}, export_products=False)
            single_export.assert_not_called()
            batch_export.assert_not_called()
        self.assertIsNone(batch['exports'])
        self.assertTrue(all(item['exports'] is None for item in batch['items']))
        versions = verify_batch(context, batch)
        self.assertEqual(len(versions), 2)
        self.assertNotEqual(versions[0].version_id, versions[1].version_id)
        saved = [self.store.load_result(version) for version in versions]
        for name, version, item in zip(context.measurements, versions, saved):
            self.assertEqual(version.report['optimization_policy'], 'shared_bn_trend_v2')
            self.assertEqual(tuple(x['path'] for x in version.source_identity['measurements']), (name,))
            model = version.report['models'][item['selected_candidate']]
            self.assertIn('strength', model)
            self.assertTrue(np.isfinite(model['strength']))
            self.assertFalse(joint_validation_passed(version.report, version.settings))
        self.assertFalse(np.allclose(saved[0]['candidate_background'], saved[1]['candidate_background']))
        denominators = [item['candidate_background'] if version.eligible else item['original_background']
                        for item, version in zip(saved, versions)]
        expected = np.mean([item['mean_raw'] / denominator - 1
                            for item, denominator in zip(saved, denominators)], axis=0)
        np.testing.assert_allclose(batch['candidate_drr'], expected, rtol=1e-11, atol=1e-11)
        ratio_of_means = np.mean([item['mean_raw'] for item in saved], axis=0) / np.mean(denominators, axis=0) - 1
        self.assertGreater(np.max(np.abs(expected-ratio_of_means)), 1e-5)
        self.assertEqual(batch['report']['processing_order'],
                         'native per-file DR/R -> align without extrapolation -> equal-file finite mean -> D2')



if __name__ == '__main__':
    unittest.main()
