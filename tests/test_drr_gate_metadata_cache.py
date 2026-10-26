"""Interrupted DRR inventories retain completed, signature-validated metadata."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from core import drr_sources as sources
from core.drr_catalog import load_drr_catalog
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


class DrrGateMetadataCacheTests(unittest.TestCase):
    CONTENT = 'Vtg,Vbg,Vbias,700,701\n-1,0,0,1,2\n1,2,0,2,3\n'

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        # Catalog IDs and inspection paths resolve Windows 8.3 temp aliases.
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / 'experiment'
        self.root.mkdir()
        self.cache_path = self.base / 'inspection.json'

    def write(self, name):
        path = self.root / name
        path.write_text(self.CONTENT, encoding='utf-8')
        return path

    def load(self, **options):
        # Every call models a new worker or application process; in-memory
        # entries from the interrupted worker cannot satisfy these assertions.
        cache = sources.DrrSourceCache(self.cache_path, load_on_init=False)
        return load_drr_catalog(self.root, cache, cache_root=self.base / 'snapshots', **options)

    def assert_gate_metadata(self, source):
        self.assertEqual(source.gate_labels, ('Vtg', 'Vbg', 'Vbias'))
        self.assertEqual(source.gate_ranges, ((-1.0, 1.0), (0.0, 2.0), (0.0, 0.0)))
        self.assertEqual(source.frame_count, 2)
        self.assertTrue(source.grid_complete)

    def test_cancelled_serial_scan_reuses_completed_files_after_restart(self):
        paths = [self.write(f'{index}.csv') for index in range(3)]
        cancellation = LoadCancellation()
        original = sources.inspect_csv_gate
        attempted = []

        def inspect(path, **options):
            attempted.append(path)
            if path == paths[1]:
                cancellation.cancel()
            return original(path, **options)

        with load_scope(cancellation), patch.object(sources, 'inspect_csv_gate', side_effect=inspect):
            with self.assertRaises(LoadCancelled):
                self.load()
        self.assertEqual(attempted, paths[:2])

        with patch.object(sources, 'inspect_csv_gate', wraps=original) as inspect:
            result = self.load()
        self.assertEqual([call.args[0] for call in inspect.call_args_list], paths[1:])
        self.assert_gate_metadata(next(source for source in result if source.filename == paths[0].name))

    def test_cancelled_parallel_scan_persists_completed_worker_profiles(self):
        paths = [self.write(f'{index:02d}.csv') for index in range(32)]
        original_stat = Path.stat

        class LargeStat:
            def __init__(self, stat):
                self.original = stat
                self.st_size = stat.st_size + 2 * 1024 ** 2

            def __getattr__(self, name):
                return getattr(self.original, name)

        def stat(path, *args, **options):
            value = original_stat(path, *args, **options)
            return LargeStat(value) if path.suffix == '.csv' else value

        completed = []

        def interrupted_pool(selected, *, on_profile=None):
            path = next(iter(selected))
            profile = sources.inspect_csv_gate_profile(path, max_rows=None)
            completed.append(path)
            if on_profile is not None:
                on_profile(path, profile)
            raise LoadCancelled()

        # Inflate only synthetic CSV stat sizes, preserving the same signature
        # for the interrupted worker and its restart without creating 32 MiB.
        with patch.object(Path, 'stat', stat):
            with patch('core.drr_inspection_pool.inspect_csv_profiles', side_effect=interrupted_pool):
                with self.assertRaises(LoadCancelled):
                    self.load(parallel_inspection=True)
            self.assertEqual(completed, paths[:1])
            with patch.object(sources, 'inspect_csv_gate', wraps=sources.inspect_csv_gate) as inspect, \
                    patch('core.drr_inspection_pool.inspect_csv_profiles',
                          side_effect=AssertionError('completed profile was lost before restart')):
                result = self.load(parallel_inspection=True)
        self.assertEqual([call.args[0] for call in inspect.call_args_list], paths[1:])
        self.assert_gate_metadata(next(source for source in result if source.filename == paths[0].name))

    def test_inventory_preview_retains_only_unchanged_gate_metadata(self):
        unchanged = self.write('unchanged.csv')
        changed = self.write('changed.csv')
        self.load()
        changed.write_text(self.CONTENT + '2,4,0,3,4\n', encoding='utf-8')
        added = self.write('added.csv')
        previews = []
        with patch.object(sources, 'inspect_csv_gate', wraps=sources.inspect_csv_gate) as inspect:
            result = self.load(publish_inventory=previews.append)
        self.assertEqual({call.args[0] for call in inspect.call_args_list}, {changed, added})
        self.assertEqual(len(previews), 1)
        preview = {source.filename: source for source in previews[0]}
        self.assert_gate_metadata(preview[unchanged.name])
        for path in (changed, added):
            self.assertIsNone(preview[path.name].frame_count)
            self.assertEqual(preview[path.name].gate_ranges, ())
            self.assertEqual(preview[path.name].gate_labels, ())
            self.assertFalse(preview[path.name].grid_complete)
        self.assertEqual(next(source for source in result if source.filename == changed.name).frame_count, 3)

        # Both ordinary validation and explicit Refresh reuse the completed
        # full profile; neither needs to read unchanged CSV contents again.
        with patch.object(sources, 'inspect_csv_gate', side_effect=AssertionError('re-read unchanged CSV')):
            self.assertEqual(self.load(), result)
            self.assertEqual(self.load(force=True), result)

    def test_unchanged_forced_refresh_does_not_rewrite_inspection_cache(self):
        self.write('measurement.csv')
        expected = self.load()
        before = self.cache_path.stat()
        content = self.cache_path.read_bytes()
        # Catalog snapshots use os.replace; only inspection persistence uses
        # Path.replace. Force may refresh snapshots without rewriting profiles.
        with patch.object(Path, 'replace', side_effect=AssertionError('unchanged inspection cache rewritten')):
            self.assertEqual(self.load(force=True), expected)
        after = self.cache_path.stat()
        self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(self.cache_path.read_bytes(), content)

    def test_cold_bridge_preserves_gate_details_and_group_during_inventory_preview(self):
        # Keep QCoreApplication ownership separate from GUI test processes.
        code = '''
import json
from pathlib import Path
import sys
from PySide6.QtCore import QCoreApplication, QSettings
from ui_qt.quick.source_catalog import SourceCatalogBridge, scan_catalog

app = QCoreApplication([])
base = Path(sys.argv[1])
root = base / 'experiment'
content = 'Vtg,Vbg,Vbias,700,701\\n-1,0,0,1,2\\n1,2,0,2,3\\n'
measurement = root / 'sample_700nmc_rep01.csv'
measurement.write_text(content, encoding='utf-8')
history = root / 'Processed Data' / 'DRR'
history.mkdir(parents=True)
(history / 'saved.metadata.json').write_text(json.dumps({
    'sources': [{'source_path': str(measurement), 'role': 'measurement'}]
}), encoding='utf-8')
cache = base / 'inspection.json'
initial = scan_catalog(root, 'drr', cache_path=cache)
row = next(row for row in initial['files'] if row['id'] == str(measurement))
metadata_line = row['detail'].splitlines()[-1]
group = next(group for group in initial['groups'] if str(measurement) in group['members'])
(root / 'new_700nmc_rep01.csv').write_text(content, encoding='utf-8')
settings = QSettings(str(base / 'settings.ini'), QSettings.IniFormat)
bridge = SourceCatalogBridge(settings=settings)
bridge._workflow = 'drr'
bridge._folder = str(root)
bridge._only_new = False
bridge._busy = True
previews = []
def publish(payload):
    bridge._preview(bridge._generation, payload)
    visible = next(row for row in bridge.filesModel.rows if row['id'] == str(measurement))
    shown_group = next(item for item in bridge.groupsModel.rows if str(measurement) in item['members'])
    assert visible['detail'].splitlines()[-1] == metadata_line, visible['detail']
    assert all(label in visible['detail'] for label in ('Vtg:', 'Vbg:', 'Vbias:', '2 frames'))
    assert shown_group['id'] == group['id'], (shown_group['id'], group['id'])
    assert shown_group['detail'] == group['detail']
    previews.append(payload)
try:
    scan_catalog(root, 'drr', cache_path=cache, publish_cached=publish, publish_inventory=publish)
    assert len(previews) == 2, len(previews)
finally:
    bridge.close()
'''
        environment = dict(os.environ, QT_QPA_PLATFORM='offscreen',
                           LOCALAPPDATA=str(self.base), XDG_CACHE_HOME=str(self.base),
                           XDG_CONFIG_HOME=str(self.base), PYTHONIOENCODING='utf-8')
        result = subprocess.run([sys.executable, '-c', code, str(self.base)],
                                cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
