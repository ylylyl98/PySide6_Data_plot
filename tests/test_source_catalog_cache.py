import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from core.source_catalog_cache import SourceCatalogCache
from core.data_io import PowerSeriesSource
from core.processing import PowerSeriesFile
from core.drr_sources import DrrSource


class SourceCatalogCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / 'source'
        self.folder.mkdir()
        self.cache_root = self.root / 'cache'

    def cache(self, namespace='PL', folder=None):
        return SourceCatalogCache(str(folder or self.folder), namespace, self.cache_root)

    def test_refresh_decodes_once_and_publishes_before_validation(self):
        self.cache().refresh(lambda: ['saved'])
        cache = self.cache()
        events = []
        original_inventory = cache.inventory
        def inventory():
            events.append('validate')
            return original_inventory()
        with patch.object(cache, '_read_record', wraps=cache._read_record) as read, patch.object(cache, 'inventory', inventory):
            result = cache.refresh(lambda: self.fail('unchanged'), publish_cached=lambda value: events.append(value))
        self.assertEqual(result, ['saved'])
        self.assertEqual(events, [['saved'], 'validate'])
        self.assertEqual(read.call_count, 1)

    def test_custom_inventory_ignores_unrelated_outputs(self):
        cache = SourceCatalogCache(str(self.folder), 'custom', self.cache_root, inventory_provider=lambda: [['raw.csv', 1, 2]])
        cache.refresh(lambda: ['raw'])
        (self.folder / 'output.dat').write_text('new')
        self.assertEqual(cache.refresh(lambda: self.fail('unchanged raw files')), ['raw'])

    def test_restart_reuses_payload_without_builder(self):
        (self.folder / 'a.csv').write_text('data')
        self.assertEqual(self.cache().refresh(lambda: {'files': ['a.csv']}), {'files': ['a.csv']})
        restarted = self.cache()
        self.assertEqual(restarted.read(), {'files': ['a.csv']})
        self.assertEqual(restarted.refresh(lambda: self.fail('unnecessary rebuild')), {'files': ['a.csv']})

    def test_changed_added_deleted_file_invalidates(self):
        path = self.folder / 'a.csv'
        path.write_text('a')
        self.cache().refresh(lambda: 1)
        path.write_text('longer')
        self.assertEqual(self.cache().refresh(lambda: 2), 2)
        extra = self.folder / 'nested'
        extra.mkdir()
        (extra / 'metadata.json').write_text('{}')
        self.assertEqual(self.cache().refresh(lambda: 3), 3)
        path.unlink()
        self.assertEqual(self.cache().refresh(lambda: 4), 4)

    def test_malformed_cache_recovers(self):
        self.cache().refresh(lambda: 'old')
        next(self.cache_root.glob('*.json')).write_text('{broken')
        self.assertIsNone(self.cache().read())
        self.assertEqual(self.cache().refresh(lambda: 'new'), 'new')

    def test_force_and_folder_namespace_isolation(self):
        self.cache().refresh(lambda: 'PL')
        self.assertIsNone(self.cache('MCD').read())
        other = self.root / 'other'
        other.mkdir()
        self.assertIsNone(self.cache(folder=other).read())
        self.assertEqual(self.cache().refresh(lambda: 'forced', force=True), 'forced')

    def test_cache_inside_source_does_not_invalidate_itself(self):
        local = self.folder / 'cache'
        cache = SourceCatalogCache(str(self.folder), 'test', local)
        cache.refresh(lambda: ['value'])
        self.assertEqual(cache.refresh(lambda: self.fail('cache counted as source')), ['value'])

    def test_serialization_preserves_containers_and_allowed_dataclasses(self):
        power = PowerSeriesSource('key', 'title', 'csv', records=(
            PowerSeriesFile('p.csv', 10.0, None, 'group', 'power'),))
        drr = DrrSource('source', 'a.csv', 'group', 'date', 1.0, False,
                        gate_grid=((1.0, 2.0),))
        payload = {'power': power, 'drr': [drr], 'nested': ({'a', 'b'}, (1, 2))}
        self.cache().refresh(lambda: payload)
        restored = self.cache().read()
        self.assertEqual(restored, payload)
        self.assertIsInstance(restored['power'], PowerSeriesSource)
        self.assertIsInstance(restored['drr'][0], DrrSource)
        self.assertIsInstance(restored['power'].records[0], PowerSeriesFile)

    def test_unknown_cached_type_is_rejected(self):
        self.cache().refresh(lambda: 'old')
        path = self.cache().path
        record = json.loads(path.read_text())
        record['payload'] = {'type': 'os.system', 'value': 'anything'}
        path.write_text(json.dumps(record))
        self.assertIsNone(self.cache().read())
        self.assertEqual(self.cache().refresh(lambda: 'safe'), 'safe')

    def test_write_failure_keeps_previous_snapshot_and_returns_fresh_result(self):
        self.cache().refresh(lambda: 'old')
        with patch('core.source_catalog_cache.os.replace', side_effect=OSError('locked')):
            self.assertEqual(self.cache().refresh(lambda: 'new', force=True), 'new')
        self.assertEqual(self.cache().read(), 'old')
        self.assertEqual(list(self.cache_root.glob('*.tmp')), [])

    def test_inventory_error_never_reuses_or_certifies_old_cache(self):
        self.cache().refresh(lambda: 'old')
        with patch.object(SourceCatalogCache, 'inventory', side_effect=OSError('offline')):
            self.assertEqual(self.cache().refresh(lambda: 'fresh'), 'fresh')
        self.assertEqual(self.cache().read(), 'old')

    def test_file_changing_during_builder_is_not_certified(self):
        path = self.folder / 'a.csv'
        path.write_text('a')
        def build():
            path.write_text('new content')
            return 'possibly stale'
        self.assertEqual(self.cache().refresh(build), 'possibly stale')
        self.assertIsNone(self.cache().read())

    def test_data_modes_ignore_images_but_invalidate_on_metadata(self):
        for mode in ('PL', 'MCD', 'Compare', 'SHG', 'Power', 'Power Dependent', 'DRR'):
            with self.subTest(mode=mode):
                cache = self.cache(f'{mode}-0')
                cache.refresh(lambda: 'catalog')
                image = self.folder / 'export.png'
                image.write_bytes(b'new figure')
                self.assertEqual(cache.refresh(lambda: 'unnecessary'), 'catalog')
                metadata = self.folder / 'Processed Data' / mode / 'result.metadata.json'
                metadata.parent.mkdir(parents=True, exist_ok=True)
                metadata.write_text('{"saved": true}')
                self.assertEqual(cache.refresh(lambda: 'updated'), 'updated')
                image.unlink()

    def test_unknown_namespace_keeps_image_invalidation(self):
        cache = self.cache('Slides')
        cache.refresh(lambda: [])
        (self.folder / 'new.png').write_bytes(b'figure')
        self.assertEqual(cache.refresh(lambda: ['new.png']), ['new.png'])


if __name__ == '__main__':
    unittest.main()
