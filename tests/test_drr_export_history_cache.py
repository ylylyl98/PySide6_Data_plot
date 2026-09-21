import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.export import _existing_drr_result, _cached_drr_export_record


class ExportHistoryCacheTests(unittest.TestCase):
    def test_pair_snapshot_reuses_index_json_but_validates_external_edits(self):
        from core.export import _drr_export_index
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'a.metadata.json'
            path.write_text(json.dumps(dict(operation='DR/R', analysis_fingerprint='old')))
            _drr_export_index(root)
            snapshot = {}
            with patch('core.export.json.load', wraps=json.load) as read:
                first = _drr_export_index(root, snapshot=snapshot)
                path.write_text(json.dumps(dict(operation='DR/R', analysis_fingerprint='new-value')))
                second = _drr_export_index(root, snapshot=snapshot)
                self.assertEqual(read.call_count, 1)
            self.assertEqual(first['a.metadata.json']['fingerprint'], 'old')
            self.assertEqual(second['a.metadata.json']['fingerprint'], 'new-value')

    def test_corrupt_index_is_rebuilt_and_readonly_cache_is_optional(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'a.metadata.json').write_text(json.dumps(
                dict(operation='DR/R', analysis_fingerprint='a')))
            for broken in ('{', '[]', '{"version":1,"entries":[]}'):
                (root / '.drr-export-index.json').write_text(broken)
                _cached_drr_export_record.cache_clear()
                with patch.object(Path, 'replace', side_effect=PermissionError):
                    self.assertEqual(_existing_drr_result(root, analysis_fingerprint='a')[0], 'a')

    def test_persistent_index_skips_unrelated_json_after_memory_cache_clear(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(12):
                (root / f'{index}.metadata.json').write_text(json.dumps(
                    dict(operation='DR/R', analysis_fingerprint=str(index))))
            self.assertIsNone(_existing_drr_result(root, analysis_fingerprint='missing'))
            _cached_drr_export_record.cache_clear()
            original = Path.read_text
            reads = []
            def read(path, *args, **kwargs):
                if path.name.endswith('.metadata.json'):
                    reads.append(path.name)
                return original(path, *args, **kwargs)
            with patch.object(Path, 'read_text', read):
                self.assertEqual(_existing_drr_result(root, analysis_fingerprint='7')[0], '7')
            self.assertEqual(reads, ['7.metadata.json'])

    def test_unchanged_history_is_not_reopened_for_second_product(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, fingerprint in [('a', 'raw'), ('b', 'second')]:
                (root / f'{name}.metadata.json').write_text(json.dumps(
                    dict(operation='DR/R', analysis_fingerprint=fingerprint)))
            for index in range(80):
                (root / f'z{index}.metadata.json').write_text(json.dumps(
                    dict(operation='DR/R', analysis_fingerprint=f'unrelated-{index}')))
            self.assertIsNone(_existing_drr_result(root, analysis_fingerprint='absent'))
            with patch.object(Path, 'read_text', side_effect=AssertionError('reopened unchanged history')):
                self.assertEqual(_existing_drr_result(root, analysis_fingerprint='second')[0], 'b')

    def test_changed_added_and_removed_records_are_seen(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'a.metadata.json'
            path.write_text(json.dumps(dict(operation='DR/R', analysis_fingerprint='old')))
            self.assertEqual(_existing_drr_result(root, analysis_fingerprint='old')[0], 'a')
            path.write_text(json.dumps(dict(operation='DR/R', analysis_fingerprint='changed-value')))
            self.assertIsNone(_existing_drr_result(root, analysis_fingerprint='old'))
            self.assertEqual(_existing_drr_result(root, analysis_fingerprint='changed-value')[0], 'a')
            path.unlink()
            self.assertIsNone(_existing_drr_result(root, analysis_fingerprint='changed-value'))
            (root/'new.metadata.json').write_text(json.dumps(dict(operation='DR/R', analysis_fingerprint='new')))
            self.assertEqual(_existing_drr_result(root, analysis_fingerprint='new')[0], 'new')

    def test_unrelated_non_drr_source_schema_is_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'other.metadata.json').write_text(json.dumps(
                dict(operation='Other', sources=['raw.csv'], processing={})))
            self.assertIsNone(_existing_drr_result(root, analysis_fingerprint='absent'))
