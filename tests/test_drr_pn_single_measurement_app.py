"""Single measurement P/N app and store boundaries; reconstruction is stubbed."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from core.drr_composite_background import compose_background, read_composite_background
from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_reconstruction import app, io, pipeline
from core.drr_pn_store import DrrPnStore


class SingleMeasurementAppTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(prefix='pn-single-app-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.columns = np.linspace(650., 780., 41)
        self.write('P.csv', 20, 4)
        self.write('N.csv', 22, 4)
        self.write('measurement_rep01.csv', 30, 7, sample=True)
        self.write('measurement_rep02.csv', 31, 7, sample=True)
        self.pair = DrrPnPair(('P.csv',), ('N.csv',))
        self.files = ('measurement_rep01.csv', 'measurement_rep02.csv')

    def write(self, name, level, count, *, sample=False):
        rows = ['Vbg,Vtg,Vbias,' + ','.join(map(str, self.columns))]
        for i in range(count):
            gate = i-3 if sample else 0
            values = level + i*.01 + np.linspace(0, 1, len(self.columns))
            rows.append(f'{gate},{gate},0,' + ','.join(map(str, values)))
        (self.root/name).write_text('\n'.join(rows) + '\n', encoding='utf-8')

    def request(self):
        return app.prepare_reconstruction_request(self.root, self.pair, (self.files[0],),
            training_files=(self.files[0],), method='conditioned_single')

    @staticmethod
    def stub(energy, pn_mean, frames, samples, **kwargs):
        if samples.shape[0] != 1:
            raise AssertionError('Single measurement must not be duplicated')
        if kwargs['method'] != 'conditioned_single':
            raise AssertionError('Incorrect reconstruction route')
        return {'energy': energy.copy(), 'background': pn_mean.mean(axis=0)}, {
            'algorithm_version': pipeline.CONDITIONED_SINGLE_ALGORITHM_VERSION,
            'settings': dict(pipeline.CONDITIONED_SINGLE_PROTOCOL), 'BN_used': False,
            'independent_sample_repeats': 1,
            'uncertainty_basis': 'gate_group_dispersion_and_conditional_PN_half_difference',
            'cross_measurement_drift': 'not evaluable from one measurement',
            'bounded_texture': 'not evaluated; independent sample repeats unavailable'}

    def test_single_raw_input_is_one_sample_and_distinct_roles(self):
        raw = io.load_raw_inputs(self.root, 'P.csv', 'N.csv', (self.files[0],), single_measurement=True)
        self.assertEqual(raw.training_samples.shape, (1, 7, len(self.columns)))
        self.assertEqual(len(raw.input_records), 3)
        self.assertEqual([row['role'] for row in raw.input_records], ['P', 'N', 'training'])
        with self.assertRaises(ValueError):
            io.load_raw_inputs(self.root, 'P.csv', 'N.csv', (self.files[0], str(self.root/self.files[0])))
        with self.assertRaises(ValueError):
            io.load_raw_inputs(self.root, 'P.csv', 'N.csv', (self.files[0],))

    def test_single_version_roundtrip_and_distinct_double_identity(self):
        request = self.request()
        double = app.prepare_reconstruction_request(self.root, self.pair, self.files,
            training_files=self.files, method='conditioned_bounded')
        self.assertNotEqual(request.key.fingerprint, double.key.fingerprint)
        with patch.object(pipeline, 'reconstruct_background', side_effect=self.stub):
            result = app.generate_reconstruction_background(self.root, self.pair, request=request)
        self.assertFalse(result.report['scientifically_validated'])
        self.assertEqual(result.report['training_shape'][0], 1)
        self.assertEqual(result.report['reconstruction_method'], 'conditioned_single')
        store = DrrPnStore(self.root)
        version = store.save(result.document, request.key)
        self.assertTrue(version.fresh)
        self.assertTrue(version.is_draft)
        self.assertEqual(version.algorithm_version, pipeline.CONDITIONED_SINGLE_ALGORITHM_VERSION)
        self.assertIsNone(store.find_applied((self.files[0],)))
        loaded = read_composite_background(version.csv_path)
        np.testing.assert_array_equal(compose_background(loaded)[1], result.arrays['background'])
        report = json.loads(loaded.sources[-1].generation_json)
        self.assertEqual(report['training_shape'][0], 1)
        self.assertEqual(report['independent_sample_repeats'], 1)
        self.assertEqual(report['uncertainty_basis'], 'gate_group_dispersion_and_conditional_PN_half_difference')
        self.assertIn('not evaluable', report['cross_measurement_drift'])
        self.assertIn('not evaluated', report['bounded_texture'])
        from ui_qt.quick.drr_background_workflow import BackgroundContext, import_pn
        context = BackgroundContext(str(self.root), (self.files[0],))
        imported = import_pn(context, self.pair, version.csv_path)
        self.assertTrue(imported.fresh)
        self.assertEqual(imported.algorithm_version, pipeline.CONDITIONED_SINGLE_ALGORITHM_VERSION)
        self.write(self.files[0], 99, 7, sample=True)
        with self.assertRaises(ValueError):
            import_pn(context, self.pair, version.csv_path)
        self.assertNotEqual(request.key.fingerprint, double.key.fingerprint)

    def test_stale_and_fake_double_aliases_are_rejected(self):
        request = self.request()
        with self.assertRaises(ValueError):
            app.prepare_reconstruction_request(self.root, self.pair,
                (self.files[0], str(self.root/self.files[0])),
                training_files=(self.files[0], str(self.root/self.files[0])), method='conditioned_bounded')
        self.write(self.files[0], 99, 7, sample=True)
        with patch.object(pipeline, 'reconstruct_background', side_effect=self.stub) as run:
            with self.assertRaises(ValueError):
                app.generate_reconstruction_background(self.root, self.pair, request=request)
        run.assert_not_called()

    def test_real_csv_reconstruction_preserves_single_sample_and_protocol(self):
        from tests.test_drr_pn_texture_compatibility import synthetic_inputs
        energy, frames, samples, _ = synthetic_inputs(n=301)
        columns = ','.join(map(str, 1240 / energy))
        for name, values in (('P.csv', frames[0]), ('N.csv', frames[1]),
                             (self.files[0], samples[0])):
            gates = np.arange(len(values)) if name == self.files[0] else np.zeros(len(values))
            np.savetxt(self.root/name, np.column_stack((gates, gates, values)), delimiter=',',
                       header='Vbg,Vtg,' + columns, comments='')
        request = self.request()
        result = app.generate_reconstruction_background(self.root, self.pair, request=request)
        self.assertEqual(result.report['training_shape'], [1, 9, 301])
        self.assertEqual(result.report['independent_sample_repeats'], 1)
        self.assertEqual(result.report['settings'], request.settings)
        self.assertIn('not evaluated', result.report['bounded_texture'])
        self.assertEqual(result.arrays['background'].shape, energy.shape)
        self.assertTrue(np.all(np.isfinite(result.arrays['background']) & (result.arrays['background'] > 0)))
        store = DrrPnStore(self.root)
        version = store.save(result.document, request.key)
        self.assertTrue(version.fresh)
        np.testing.assert_array_equal(compose_background(read_composite_background(version.csv_path))[1],
                                      result.arrays['background'])


if __name__ == '__main__':
    unittest.main()
