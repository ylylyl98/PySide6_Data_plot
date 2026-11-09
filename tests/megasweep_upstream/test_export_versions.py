# Adapted from Megasweep Analysis commit 2612f6f.
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from core.megasweep.exporting import export_file, save_dataframe


class ExportVersionTests(unittest.TestCase):
    def test_images_always_get_new_names_and_preserve_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.png'
            results = [export_file(path, lambda p, n=n: Path(p).write_bytes(bytes([n]))) for n in range(3)]
            self.assertEqual([r.path.name for r in results], ['map.png', 'map_002.png', 'map_003.png'])
            self.assertEqual(path.read_bytes(), b'\x00')

    def test_csv_reuses_identical_content_in_any_existing_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.csv'
            original = pd.DataFrame({'x': [0, 1], 'signal': [1e-12, 2e-12]})
            first = save_dataframe(original, path)
            timestamp = path.stat().st_mtime_ns
            self.assertFalse(save_dataframe(original, path).created)
            self.assertEqual(path.stat().st_mtime_ns, timestamp)
            changed = original.copy(); changed.loc[0, 'signal'] = 1.01e-12
            second = save_dataframe(changed, path)
            self.assertEqual(second.path.name, 'map_002.csv')
            self.assertFalse(save_dataframe(changed, path).created)
            self.assertEqual(save_dataframe(original, path).path, first.path)
            self.assertEqual(len(list(Path(directory).glob('*.csv'))), 2)

    def test_writer_failure_leaves_original_and_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.png'; path.write_bytes(b'old')
            def fail(target):
                Path(target).write_bytes(b'partial')
                raise ValueError('render failed')
            with self.assertRaises(ValueError):
                export_file(path, fail)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertEqual(path.read_bytes(), b'old')

    def test_concurrent_exports_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.png'
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda n: export_file(path, lambda p: Path(p).write_bytes(bytes([n]))), range(4)))
            self.assertEqual(len({r.path for r in results}), 4)
            self.assertEqual({r.path.read_bytes() for r in results}, {bytes([n]) for n in range(4)})
