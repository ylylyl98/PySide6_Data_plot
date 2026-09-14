import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from core.source_catalog_cache import SourceCatalogCache
from ui_qt.main_window import _cached_folder_sources_worker


class CatalogRefreshOptimizationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / 'experiment'
        self.root.mkdir()
        env = patch.dict(os.environ, {'LOCALAPPDATA': str(Path(tmp.name) / 'cache')})
        env.start()
        self.addCleanup(env.stop)

    def refresh(self, force=False, mode='Power Dependent'):
        return _cached_folder_sources_worker(
            str(self.root), mode=mode, power_include_legacy=False, force=force,
            publish_cached=lambda _: None, progress=None, log=None)

    def write(self, name, power):
        (self.root / name).write_text(f'Power_uW,1.4,1.5\n{power},2,3\n')

    def powers(self, result):
        snapshot = next(item for item in result if isinstance(item, dict)
                        and item.get('tag') == 'power_catalog')
        return {source.file_name: source.power_values for source in snapshot['sources'].values()}

    def test_changed_power_file_reuses_other_persisted_inspections(self):
        import pandas as pd
        self.write('a.csv', 1)
        self.write('b.csv', 2)
        self.refresh()
        self.write('b.csv', 999)
        with patch('core.data_io.pd.read_csv', wraps=pd.read_csv) as read:
            result = self.refresh()
        self.assertEqual(self.powers(result), {'a.csv': (1.0,), 'b.csv': (999.0,)})
        self.assertEqual([Path(call.args[0]).name for call in read.call_args_list
                          if call.kwargs.get('usecols') == ['Power_uW']], ['b.csv'])

    def test_added_and_deleted_power_files_keep_unchanged_values(self):
        import pandas as pd
        self.write('a.csv', 1)
        self.write('b.csv', 2)
        self.refresh()
        (self.root / 'a.csv').unlink()
        self.write('c.csv', 3)
        with patch('core.data_io.pd.read_csv', wraps=pd.read_csv) as read:
            result = self.refresh()
        self.assertEqual(self.powers(result), {'b.csv': (2.0,), 'c.csv': (3.0,)})
        self.assertEqual([Path(call.args[0]).name for call in read.call_args_list
                          if call.kwargs.get('usecols') == ['Power_uW']], ['c.csv'])

    def test_rebuild_reparses_power_values(self):
        import pandas as pd
        self.write('a.csv', 1)
        self.refresh()
        with patch('core.data_io.pd.read_csv', wraps=pd.read_csv) as read:
            result = self.refresh(force=True)
        self.assertEqual(self.powers(result), {'a.csv': (1.0,)})
        self.assertEqual(read.call_count, 1)

    def test_same_size_edit_invalidates_by_modification_time(self):
        self.write('a.csv', 1)
        self.refresh()
        path = self.root / 'a.csv'
        before = path.stat()
        self.write('a.csv', 9)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10_000_000))
        self.assertEqual(path.stat().st_size, before.st_size)
        self.assertEqual(self.powers(self.refresh()), {'a.csv': (9.0,)})

    def test_rebuild_repairs_same_signature_external_edit(self):
        self.write('a.csv', 1)
        self.refresh()
        path = self.root / 'a.csv'
        before = path.stat()
        self.write('a.csv', 9)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(self.powers(self.refresh(force=True)), {'a.csv': (9.0,)})

    def test_file_no_longer_a_power_table_is_removed(self):
        self.write('a.csv', 1)
        self.refresh()
        (self.root / 'a.csv').write_text('Power_uW,FWHM_meV\n1,2\n')
        self.assertEqual(self.powers(self.refresh()), {})

    def test_worker_decodes_snapshot_only_once(self):
        self.refresh(mode='PL')
        original = SourceCatalogCache._read_record
        reads = []
        def read(cache):
            reads.append(cache.path)
            return original(cache)
        with patch.object(SourceCatalogCache, '_read_record', read):
            self.refresh(mode='PL')
        self.assertEqual(len(reads), 1)

    def test_inventory_reuses_directory_entry_metadata(self):
        self.write('a.csv', 1)
        nested = self.root / 'nested'
        nested.mkdir()
        (nested / 'b.csv').write_text('data')
        cache = SourceCatalogCache(str(self.root), 'PL')
        original = Path.stat
        calls = []
        def stat(path, *args, **kwargs):
            if path.suffix == '.csv':
                calls.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'stat', stat):
            inventory = cache.inventory()
        self.assertEqual([row[0] for row in inventory], ['a.csv', 'nested/b.csv'])
        self.assertEqual(calls, [])
