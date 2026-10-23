from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_fitted_background import (BackgroundFitSettings, estimate_background,
    export_references, load_fitted_group, read_reference_raw)
from core.loader import DataCube, load_dat
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


SETTINGS = BackgroundFitSettings(((1.56, 1.64), (1.76, 1.84)), 8, .001)


def synthetic():
    energy = np.linspace(1.4, 2., 401)
    gate = np.array([-1., 0., 1.])
    x = (energy-1.7)/.3
    background = np.exp(8 + .4*x + .12*x*x + gate[:, None]*(.2+.03*x))
    contrast = (.12*np.exp(-.5*((energy-1.6)/.006)**2)
                - .08*np.exp(-.5*((energy-1.8)/.008)**2)) * (1+.2*gate[:, None])
    return DataCube(energy, gate, background*(1+contrast), 'BG (V)', 'sample', 'counts'), background, contrast


def write_csv(path, cube):
    header = 'Vbg,Vtg,' + ','.join(f'{1240/e:.15g}' for e in cube.energy)
    np.savetxt(path, np.column_stack([cube.gate, np.zeros(len(cube.gate)), cube.Z]),
               delimiter=',', header=header, comments='', fmt='%.15g')


class FittedBackgroundTests(unittest.TestCase):
    def test_recovers_known_background_with_positive_and_negative_peaks(self):
        raw, background, contrast = synthetic()
        original = raw.Z.copy()
        result, reference = estimate_background(raw, SETTINGS)
        np.testing.assert_allclose(reference.background()/background, 1., atol=8e-4)
        np.testing.assert_allclose(result.Z, contrast, atol=8e-4)
        np.testing.assert_array_equal(raw.Z, original)
        self.assertFalse(reference.metadata()['bn_reference_used'])

    def test_descending_energy_and_outlier_do_not_mutate_input(self):
        raw, background, _ = synthetic()
        raw.Z[1, 30] *= 2
        descending = replace(raw, energy=raw.energy[::-1], Z=raw.Z[:, ::-1])
        result, reference = estimate_background(descending, SETTINGS)
        np.testing.assert_array_equal(result.energy, raw.energy)
        np.testing.assert_allclose(reference.background()/background, 1., atol=.001)
        self.assertGreater(result.Z[1, 30], .9)

    def test_invalid_support_and_intensity_are_rejected(self):
        raw, _, _ = synthetic()
        for settings in (BackgroundFitSettings(), BackgroundFitSettings(((1.4, 1.6),)),
                         BackgroundFitSettings(((1.41, 1.99),)), BackgroundFitSettings(((1.6, 1.5),))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                estimate_background(raw, settings)
        for value in (0., -1., np.nan, np.inf):
            bad = raw.Z.copy()
            bad[0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive raw'):
                estimate_background(replace(raw, Z=bad), SETTINGS)

    def test_cancellation(self):
        cancellation = LoadCancellation()
        with load_scope(cancellation), self.assertRaises(LoadCancelled):
            cancellation.cancel()
            estimate_background(synthetic()[0], SETTINGS)

    def test_group_fits_mean_raw_once_and_exports_its_actual_denominator(self):
        raw, _, _ = synthetic()
        altered = replace(raw, Z=3*raw.Z*(1+.04*np.exp(-.5*((raw.energy-1.6)/.005)**2)))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_csv(root/'a.csv', raw)
            write_csv(root/'b.csv', altered)
            result, references = load_fitted_group(folder, ['a.csv', 'b.csv'], asdict(SETTINGS), y_axis='bg')
            mean_raw = replace(raw, Z=(raw.Z+altered.Z)/2)
            expected = estimate_background(mean_raw, SETTINGS)[0].Z
            old_per_file_average = np.mean([estimate_background(cube, SETTINGS)[0].Z
                                           for cube in (raw, altered)], axis=0)
            np.testing.assert_allclose(result.Z, expected, atol=1e-10)
            self.assertGreater(float(np.max(abs(result.Z-old_per_file_average))), .01)
            self.assertEqual(len(references), 1)
            self.assertIn('BG', result.gate_coordinates)
            files = export_references(folder, references)
            self.assertEqual(len(files), 1)
            for name, reference in zip(files, references):
                exported = load_dat(root/name)
                np.testing.assert_allclose(exported.Z, reference.background(), rtol=1e-11)
                self.assertEqual(exported.gate_label, 'BG (V)')
                metadata = json.loads((root/name).with_suffix('.metadata.json').read_text())
                self.assertEqual(metadata['processing']['settings']['spline_intervals'], 8)
                original = read_reference_raw(folder, reference, y_axis='bg')
                self.assertEqual(original.Z.shape, exported.Z.shape)
                np.testing.assert_allclose(original.Z, mean_raw.Z, rtol=1e-12)
            self.assertEqual(files, export_references(folder, references))
            (root/'a.csv').write_text('changed source', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed after fitting'):
                read_reference_raw(folder, references[0], y_axis='bg')

    def test_alignment_uses_overlap_only(self):
        raw, _, _ = synthetic()
        second = replace(raw, gate=np.array([0., 1., 2.]), Z=raw.Z*2)
        with tempfile.TemporaryDirectory() as folder:
            write_csv(Path(folder)/'a.csv', raw)
            write_csv(Path(folder)/'b.csv', second)
            combined, _ = load_fitted_group(folder, ['a.csv', 'b.csv'], SETTINGS, y_axis='bg')
            mean_raw = replace(raw, gate=np.array([0., 1.]), Z=(raw.Z[1:]+second.Z[:2])/2)
            expected = estimate_background(mean_raw, SETTINGS)[0]
            np.testing.assert_array_equal(combined.gate, [0., 1.])
            np.testing.assert_allclose(combined.Z, expected.Z, atol=1e-10)

    def test_group_reports_raw_averaging_before_one_fit_and_cache_reuse_afterwards(self):
        raw, _, _ = synthetic()
        with tempfile.TemporaryDirectory() as folder, patch.dict(
                'os.environ', DPTK_FITTED_CACHE_DIR=folder+'/cache'):
            write_csv(Path(folder)/'a.csv', raw)
            write_csv(Path(folder)/'b.csv', raw)
            for cached in (False, True):
                events = []
                progress = lambda value: events.append(('progress', value))
                log = lambda message: events.append(('log', message))
                _, references = load_fitted_group(folder, ['a.csv', 'b.csv'], SETTINGS,
                    y_axis='bg', progress=progress, log=log)
                self.assertEqual(events[0], ('progress', 0))
                averaging = [i for i, (kind, message) in enumerate(events)
                             if kind == 'log' and message.startswith('Averaging raw intensity ')]
                fitting = [i for i, (kind, message) in enumerate(events)
                           if kind == 'log' and message.startswith('Fitting one background to average of 2 files')]
                self.assertEqual(len(averaging), 2)
                self.assertEqual(len(fitting), 1)
                self.assertLess(max(averaging), fitting[0])
                self.assertEqual(events[-1], ('progress', 90))
                completed = [message for kind, message in events
                             if kind == 'log' and ('cache reused' if cached else 'calculated') in message]
                self.assertTrue(completed)
                self.assertEqual([ref.cache_hit for ref in references], [cached])

    def test_single_file_reports_stages_before_expensive_fit_operations(self):
        raw, _, _ = synthetic()
        with tempfile.TemporaryDirectory() as folder, patch.dict(
                'os.environ', DPTK_FITTED_CACHE_DIR=folder+'/cache'):
            write_csv(Path(folder)/'a.csv', raw)
            from core import drr_background_structure as structure
            from core import drr_background_quality as quality
            messages = []
            def guarded_stage(function, expected):
                def call(*args, **kwargs):
                    self.assertIn(expected, messages[-1])
                    return function(*args, **kwargs)
                return call
            with patch.object(structure, 'gate_anchor_masks', guarded_stage(
                    structure.gate_anchor_masks, 'Identifying protected peaks')), patch.object(
                    quality, 'inspect_background', guarded_stage(
                        quality.inspect_background, 'Checking background quality')):
                _, references = load_fitted_group(folder, ['a.csv'],
                    replace(SETTINGS, automatic_protection=True), y_axis='bg', log=messages.append)
            self.assertFalse(references[0].cache_hit)
            stages = [message for message in messages if message.startswith('DRR load: ')]
            self.assertGreaterEqual(len(stages), 6)
            self.assertTrue(any('Checking fitted background cache' in message for message in stages))

    def test_cancelled_fitted_group_does_not_start_next_file_or_publish_completion(self):
        raw, _, _ = synthetic()
        cancellation = LoadCancellation()
        events = []
        def log(message):
            events.append(message)
            if message.startswith('Averaging raw intensity 1/2'):
                cancellation.cancel()
        with tempfile.TemporaryDirectory() as folder, patch.dict(
                'os.environ', DPTK_FITTED_CACHE_DIR=folder+'/cache'):
            write_csv(Path(folder)/'a.csv', raw)
            write_csv(Path(folder)/'b.csv', raw)
            with load_scope(cancellation), self.assertRaises(LoadCancelled):
                load_fitted_group(folder, ['a.csv', 'b.csv'], SETTINGS, y_axis='bg', log=log)
        self.assertFalse(any(message.startswith('Averaging raw intensity 2/2') for message in events))
        self.assertFalse(any(message.startswith('Fitting one background') for message in events))


if __name__ == '__main__':
    unittest.main()
