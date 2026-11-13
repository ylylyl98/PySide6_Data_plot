import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path


class PowerPointLazyLoadingTests(unittest.TestCase):
    def test_dat_without_processing_avoids_heavy_modules_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            dat = Path(directory) / 'sample.dat'
            dat.write_text('Photon energy(eV)\t-1\t2\n1\t10\t20\n2\t30\t40\n', encoding='utf-8')
            dat.with_suffix('.metadata.json').write_text(json.dumps({
                'operation': 'DR/R',
                'plot': {'title': 'Measured map', 'ylabel': 'Magnetic field (T)',
                         'cbar_label': 'dR/R', 'y_unit': 'T', 'cmap': 'viridis'},
            }), encoding='utf-8')
            script = '''
import sys
import numpy as np
from ui_qt.quick.pl_workflow import PlRequest, load_pl_source
from core.processing import nearest_gate_spectrum
cube = load_pl_source(PlRequest(sys.argv[1], dat_choice='original', processing='none'))
np.testing.assert_allclose(cube.energy, [1., 2.])
np.testing.assert_allclose(cube.gate, [-1., 2.])
np.testing.assert_allclose(cube.Z, [[10., 30.], [20., 40.]])
assert cube.title == 'Measured map'
assert cube.gate_label == 'Magnetic field (T)'
assert cube.gate_unit == 'T'
assert cube.plot_metadata['cmap'] == 'viridis'
assert not cube.Z.flags.writeable
selected, values = nearest_gate_spectrum(cube, 2.)
assert selected == 2.
np.testing.assert_allclose(values, [20., 40.])
from pathlib import Path
from core.plotting import HeatmapParams
from core.export import _save_heatmap_png
params = HeatmapParams('Measured map', 'Energy', 'Magnetic field (T)', 'dR/R',
                       10., 40., (1., 2.), (-1., 2.))
output = Path(sys.argv[1]).with_suffix('.png')
_save_heatmap_png(output, cube, params, drr=True)
assert output.is_file() and output.stat().st_size > 0
for name in ('core.processing_run', 'core.data_io', 'scipy.signal'):
    assert name not in sys.modules, name
'''
            result = subprocess.run([sys.executable, '-c', script, str(dat)], text=True,
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_loader_backend_attribute_replacement_is_respected(self):
        from core import loader
        backend = SimpleNamespace(process_pl=Mock(return_value={'ok': True}))
        loader._load_pl_cached.cache_clear()
        try:
            with patch.object(loader, 'P', backend):
                result = loader._load_pl_cached('folder', 'file.csv', False, 'auto', (0, 0))
            self.assertEqual(result, {'ok': True})
            backend.process_pl.assert_called_once()
        finally:
            loader._load_pl_cached.cache_clear()
