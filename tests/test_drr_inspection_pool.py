import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.drr_inspection_pool import inspect_csv_profiles
from core.drr_sources import discover_drr_sources, DrrSourceCache, inspect_csv_gate_profile


class DrrInspectionPoolTests(unittest.TestCase):
    def test_spawned_workers_match_serial_inspection_and_release_children(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(3):
                (root / f'measurement_{index}.csv').write_text(
                    'Vbg,Vtg,700,710\n0,1,3,4\n1,2,5,nan\n2,3,7,8\n', encoding='utf-8')
            # -c does not re-import an unguarded custom unittest runner on
            # Windows. This still exercises the real spawn/pickle protocol.
            code = """
from pathlib import Path
import multiprocessing, sys, threading
from core.drr_inspection_pool import inspect_csv_profiles
from core.drr_sources import inspect_csv_gate_profile
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope
paths = list(Path(sys.argv[1]).glob('*.csv'))
expected = {p: inspect_csv_gate_profile(p, max_rows=None) for p in paths}
actual = inspect_csv_profiles(paths)
assert actual == expected, (actual, expected)
assert not multiprocessing.active_children()
token = LoadCancellation()
timer = threading.Timer(.2, token.cancel)
timer.start()
try:
    with load_scope(token):
        inspect_csv_profiles(paths)
except LoadCancelled:
    pass
else:
    raise AssertionError('Scan did not cancel during worker startup')
finally:
    timer.join()
assert not multiprocessing.active_children()
"""
            result = subprocess.run([sys.executable, '-c', code, folder], capture_output=True,
                                    text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_unavailable_processes_fall_back_without_losing_sources(self):
        with patch('core.drr_inspection_pool.ProcessPoolExecutor', side_effect=OSError('unavailable')):
            self.assertEqual(inspect_csv_profiles([Path('measurement.csv')]), {})
        with patch('core.drr_inspection_pool.get_context') as context:
            context.return_value.Event.side_effect = OSError('unavailable')
            self.assertEqual(inspect_csv_profiles([Path('measurement.csv')]), {})

    def test_changed_file_discards_profile_computed_for_old_signature(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / f'repeat_{i}.csv' for i in range(32)]
            for path in paths:
                path.write_text('Vbg,700\n0,2\n', encoding='utf-8')
            # Avoid 32 MiB of synthetic data; only the acceleration threshold
            # sees inflated sizes, while per-file signatures remain real.
            original_stat = Path.stat
            class Stat:
                def __init__(self, value):
                    self._value = value
                    self.st_size = value.st_size + 2*1024**2

                def __getattr__(self, name):
                    return getattr(self._value, name)
            def stat(path, *args, **kwargs):
                result = original_stat(path, *args, **kwargs)
                return Stat(result) if path.suffix == '.csv' else result
            def inspect(selected):
                result = {p: inspect_csv_gate_profile(p, max_rows=None) for p in selected}
                paths[0].write_text('Vbg,700\n0,2\n1,3\n', encoding='utf-8')
                return result
            with patch.object(Path, 'stat', stat), patch('core.drr_inspection_pool.inspect_csv_profiles', side_effect=inspect):
                sources = discover_drr_sources(root, cache=DrrSourceCache(root/'cache.json', load_on_init=False),
                                               include_history=False, parallel_inspection=True)
            source = next(s for s in sources if s.filename == paths[0].name)
            self.assertEqual(source.frame_count, 2)
