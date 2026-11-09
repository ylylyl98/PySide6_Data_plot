from pathlib import Path
import tempfile
import json
import unittest
from types import SimpleNamespace
import numpy as np

from core.power_analysis_store import save_analysis, load_analysis, analysis_fingerprint
from core.power_multi_peaks import PeakSeed, MultiPeakSettings, fit_multi_spectrum


def test_round_trip_and_stale_data(tmp_path):
    x = np.linspace(1.3, 1.5, 100)
    y = 1 + 4 / (1 + (2 * (x - 1.4) / .02) ** 2)
    cube = SimpleNamespace(energy=x, gate=np.array([1., 2.]), Z=np.array([y, np.ones_like(y)]))
    cubes = {'Power': cube}
    rows = [dict(center=1.4, width=20., height=None, lower=None, upper=None)]
    settings = MultiPeakSettings((PeakSeed(1.4, .02),), 'Gaussian')
    first = fit_multi_spectrum(x, y, 1., settings, manual=True)
    failed = fit_multi_spectrum(x, cube.Z[1], 2., settings)
    path = tmp_path / 'analysis.npz'
    interval = dict(channel='Power', peak=1, lower=1., upper=2., suggested=False)
    law = dict(interval, observed_lower=1., observed_upper=2., alpha=1.2, alpha_error=.1, amplitude=2., n=4, excluded=0, log_rms=.1, method='OLS log-log', name='Peak 1', crosses_sources=False)
    save_analysis(path, cubes, 0., rows, 'Lorentzian', {'Power': (first, failed)}, laws=[law], ranges=[interval], preferences={'metric': 'Peak height'})
    result = load_analysis(path, cubes, 0.)
    assert result['rows'] == rows and result['settings'].model == 'Lorentzian'
    assert result['ranges'] == [interval]
    assert result['fingerprint'] == analysis_fingerprint(cubes, 0.)
    assert result['preferences'] == {'metric': 'Peak height'}
    loaded, loaded_failed = result['results']['Power']
    assert loaded.manual and loaded.settings == first.settings and loaded.parameters == first.parameters
    np.testing.assert_array_equal(loaded.fitted, first.fitted)
    np.testing.assert_array_equal(loaded.component_curves[0], first.component_curves[0])
    assert loaded_failed.status == 'flat spectrum'
    assert result['laws'] == [law]
    assert load_analysis(path, cubes, 1.) is None
    cube.Z[0, 0] += 1
    assert load_analysis(path, cubes, 0.) is None


def test_corrupt_and_object_payload_rejected(tmp_path):
    x = np.linspace(1., 2., 10)
    cubes = {'p': SimpleNamespace(energy=x, gate=np.array([1.]), Z=np.array([x]))}
    path = tmp_path / 'cache.npz'
    path.write_bytes(b'PK\x03\x04broken')
    assert load_analysis(path, cubes, 0.) is None
    np.savez(path, metadata=np.array([object()]))
    assert load_analysis(path, cubes, 0.) is None
    np.savez(path, metadata=np.frombuffer(b'{bad', dtype=np.uint8))
    assert load_analysis(path, cubes, 0.) is None


class PowerAnalysisStoreTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            test_round_trip_and_stale_data(Path(folder))

    def test_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            test_corrupt_and_object_payload_rejected(Path(folder))

    def test_tampered_fit_shape_and_power(self):
        x = np.linspace(1.3, 1.5, 100)
        y = 1 + 4 / (1 + (2 * (x - 1.4) / .02) ** 2)
        cube = SimpleNamespace(energy=x, gate=np.array([1.]), Z=np.array([y]))
        rows = [dict(center=1.4, width=20., height=None, lower=None, upper=None)]
        settings = MultiPeakSettings((PeakSeed(1.4, .02),))
        fit = fit_multi_spectrum(x, y, 1., settings, manual=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fit.npz'
            save_analysis(path, {'P': cube}, 0., rows, 'Lorentzian', {'P': (fit,)})
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            metadata = json.loads(arrays['metadata'].tobytes())
            metadata['results']['P'][0]['fitted'][1] -= 1
            arrays['metadata'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
            np.savez(path, **arrays)
            self.assertIsNone(load_analysis(path, {'P': cube}, 0.))
            metadata['results']['P'][0]['fitted'][1] += 1
            metadata['results']['P'][0]['power_uw'] = 999.
            arrays['metadata'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
            np.savez(path, **arrays)
            self.assertIsNone(load_analysis(path, {'P': cube}, 0.))

    def test_invalid_law_metadata_is_rejected(self):
        x = np.linspace(1.3, 1.5, 100)
        y = 1 + 4 / (1 + (2 * (x - 1.4) / .02) ** 2)
        cube = SimpleNamespace(energy=x, gate=np.array([1.]), Z=np.array([y]))
        rows = [dict(center=1.4, width=20., height=None, lower=None, upper=None)]
        fit = fit_multi_spectrum(x, y, 1., MultiPeakSettings((PeakSeed(1.4, .02),)))
        interval = dict(channel='P', peak=1, lower=1., upper=2., suggested=False)
        law = dict(interval, observed_lower=1., observed_upper=2., alpha=1., alpha_error=.1,
                   amplitude=2., n=4, excluded=0, log_rms=.1, method='OLS log-log',
                   name='Peak 1', crosses_sources=False)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fit.npz'
            save_analysis(path, {'P': cube}, 0., rows, 'Lorentzian', {'P': (fit,)}, laws=[law], ranges=[interval])
            with np.load(path, allow_pickle=False) as archive:
                original = {key: archive[key] for key in archive.files}
            for key, value in [('laws', [{}]), ('ranges', [{}]),
                               ('ranges', [dict(interval, channel='missing')]),
                               ('ranges', [dict(interval, peak=2)]),
                               ('ranges', [dict(interval, lower=-1.)]),
                               ('laws', [dict(law, alpha=float('nan'))])]:
                arrays = dict(original)
                metadata = json.loads(arrays['metadata'].tobytes())
                metadata[key] = value
                arrays['metadata'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
                np.savez(path, **arrays)
                self.assertIsNone(load_analysis(path, {'P': cube}, 0.), (key, value))

    def test_legacy_archive_restores(self):
        x = np.linspace(1.3, 1.5, 100)
        y = 1 + 4 / (1 + (2 * (x - 1.4) / .02) ** 2)
        cube = SimpleNamespace(energy=x, gate=np.array([1.]), Z=np.array([y]))
        rows = [dict(center=1.4, width=20., height=None, lower=None, upper=None)]
        fit = fit_multi_spectrum(x, y, 1., MultiPeakSettings((PeakSeed(1.4, .02),)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fit.npz'
            save_analysis(path, {'P': cube}, 0., rows, 'Lorentzian', {'P': (fit,)})
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(archive['metadata'].tobytes())
                data = archive['data']
            item = metadata['results']['P'][0]
            old = {}
            for name in ('x', 'fitted', 'residual', 'baseline'):
                start, length = item[name]
                old[name] = data[start:start + length]
                item[name] = name
            curves = []
            for index, (start, length) in enumerate(item['component_curves']):
                key = 'curve' + str(index)
                old[key] = data[start:start + length]
                curves.append(key)
            item['component_curves'] = curves
            metadata['version'] = 1
            old['metadata'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
            np.savez_compressed(path, **old)
            result = load_analysis(path, {'P': cube}, 0.)
            self.assertIsNotNone(result)
            np.testing.assert_array_equal(result['results']['P'][0].fitted, fit.fitted)
