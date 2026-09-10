import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.data_io import inspect_power_sweep_csv


class PowerHeaderCacheTests(unittest.TestCase):
    def test_wide_header_avoids_pandas_and_invalidates_on_file_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'sweep.csv')
            path.write_text('\ufeff"Power_uW",' + ','.join(str(700+i/10) for i in range(2000)) + '\n', encoding='utf-8')
            with patch('core.data_io.pd.read_csv', side_effect=AssertionError('Header inspection must not build a DataFrame')):
                self.assertTrue(inspect_power_sweep_csv(folder, path.name))
                self.assertTrue(inspect_power_sweep_csv(folder, path.name))
                stamp = path.stat().st_mtime_ns
                path.write_text('NotPower,1.4,1.5\n')
                os.utime(path, ns=(stamp + 1000000000, stamp + 1000000000))
                self.assertFalse(inspect_power_sweep_csv(folder, path.name))
                path.unlink()
                self.assertFalse(inspect_power_sweep_csv(folder, path.name))

    def test_tab_delimited_power_header(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'sweep.csv').write_text('Power_uW\t1.4\t1.5\n')
            self.assertTrue(inspect_power_sweep_csv(folder, 'sweep.csv'))
