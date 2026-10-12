import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core import drr_display_settings as history


class DisplayHistoryCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'Processed Data' / 'DRR'
        self.directory.mkdir(parents=True)

    def write(self, name='old.metadata.json', source='old_720nmc.csv', xmin=1.55, **extra):
        (self.root / source).touch()
        payload = dict(operation='DR/R', sources=[dict(role='measurement', name=source)],
                       plot=dict(title=source, xlim=[xmin, 1.74], ylim=[-.5, .5],
                                 vmin=-.03, vmax=.07, cmap='coolwarm'))
        payload.update(extra)
        path = self.directory / name
        path.write_text(json.dumps(payload), encoding='utf-8')
        return path

    def test_missing_files_and_centers_reuse_the_parsed_catalog(self):
        for i in range(8):
            self.write(f'{i}.metadata.json', f'old_{i}_720nmc.csv')
        first = history.find_candidates(str(self.root), ['new_720nmc.csv'])
        self.assertIsNone(first['exact'])
        self.assertIsNotNone(first['template'])
        with patch.object(history, '_read', wraps=history._read) as read:
            second = history.find_candidates(str(self.root), ['another_720nmc.csv'])
            third = history.find_candidates(str(self.root), ['new_760nmc.csv'])
        self.assertIsNone(second['exact'])
        self.assertIsNotNone(second['template'])
        self.assertIsNone(third['template'])
        sidecar_reads = [call for call in read.call_args_list if str(call.args[0]).endswith('.metadata.json')]
        self.assertEqual(sidecar_reads, [])

    def test_changed_added_removed_and_completed_metadata_are_visible(self):
        path = self.write()
        self.assertEqual(history.find_candidates(str(self.root), ['old_720nmc.csv'])['exact']['ranges']['xmin'], 1.55)
        self.write(xmin=1.56)
        self.assertEqual(history.find_candidates(str(self.root), ['old_720nmc.csv'])['exact']['ranges']['xmin'], 1.56)
        path.unlink()
        self.assertIsNone(history.find_candidates(str(self.root), ['old_720nmc.csv'])['exact'])
        path = self.write(display_save_pending=True)
        self.assertIsNone(history.find_candidates(str(self.root), ['old_720nmc.csv'])['exact'])
        history.complete_metadata_snapshots([path.with_suffix('').with_suffix('.dat')])
        self.assertIsNotNone(history.find_candidates(str(self.root), ['old_720nmc.csv'])['exact'])

    def test_results_cannot_mutate_a_warm_catalog(self):
        self.write()
        first = history.find_candidates(str(self.root), ['new_720nmc.csv'])
        first['template']['ranges']['xmin'] = 99
        second = history.find_candidates(str(self.root), ['new_720nmc.csv'])
        self.assertEqual(second['template']['ranges']['xmin'], 1.55)

    def test_added_export_reads_only_the_new_sidecar(self):
        self.write()
        (self.root / 'another_720nmc.csv').touch()
        history.find_candidates(str(self.root), ['new_720nmc.csv'])
        self.write('another.metadata.json', 'another_720nmc.csv', xmin=1.56)
        with patch.object(history, '_read', wraps=history._read) as read:
            result = history.find_candidates(str(self.root), ['new_720nmc.csv'])
        self.assertEqual(result['template']['ranges']['xmin'], 1.56)
        sidecars = [Path(call.args[0]).name for call in read.call_args_list
                    if str(call.args[0]).endswith('.metadata.json')]
        self.assertEqual(sidecars, ['another.metadata.json'])

    def test_source_path_reappearance_invalidates_cached_identity(self):
        initial = self.root / 'Initial Data'
        initial.mkdir()
        source = initial / 'old_720nmc.csv'
        path = self.write()
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['sources'][0] = dict(role='measurement', source_path=str(source), name='old_720nmc.csv')
        path.write_text(json.dumps(payload), encoding='utf-8')
        self.assertIsNotNone(history.find_candidates(self.root, [str(source)])['exact'])
        source.touch()
        self.assertIsNotNone(history.find_candidates(self.root, [str(source)])['exact'])
        source.unlink()
        # A recorded portable subdirectory may take over only when it exists.
        alternate = self.root / 'moved' / 'old_720nmc.csv'
        payload['sources'][0]['name'] = 'moved/old_720nmc.csv'
        path.write_text(json.dumps(payload), encoding='utf-8')
        history.find_candidates(self.root, [str(source)])
        alternate.parent.mkdir()
        alternate.touch()
        self.assertIsNone(history.find_candidates(self.root, [str(source)])['exact'])
        self.assertIsNotNone(history.find_candidates(self.root, ['moved/old_720nmc.csv'])['exact'])

    def test_removing_second_sidecar_does_not_leave_a_cached_merged_product(self):
        self.write()
        second = self.write('second.metadata.json', processing={'derivative_order': 2})
        self.assertIn('second', history.find_candidates(self.root, ['old_720nmc.csv'])['exact'])
        second.unlink()
        self.assertNotIn('second', history.find_candidates(self.root, ['old_720nmc.csv'])['exact'])

    def test_file_disabled_uses_an_indexed_center_without_scanning_legacy(self):
        self.write()
        state = history.find_candidates(str(self.root), ['old_720nmc.csv'])['exact']
        history.save_snapshot(str(self.root), state)
        with patch.object(history, '_read', wraps=history._read) as read:
            result = history.find_candidates(str(self.root), ['new_720nmc.csv'], restore_files=False)
        self.assertIsNone(result['exact'])
        self.assertIsNotNone(result['template'])
        self.assertFalse(any(str(call.args[0]).endswith('.metadata.json') for call in read.call_args_list))

    def test_both_disabled_do_not_read_history(self):
        self.write()
        with patch.object(history, '_read', side_effect=AssertionError('disabled history was read')):
            result = history.find_candidates(str(self.root), ['new_720nmc.csv'], restore_files=False, restore_centers=False)
        self.assertIsNone(result['exact'])
        self.assertIsNone(result['template'])

    def test_unavailable_change_time_keeps_history_readable_without_caching(self):
        self.write()
        with patch.object(history, '_digest_signature', return_value=None):
            self.assertIsNotNone(history.find_candidates(self.root, ['old_720nmc.csv'])['exact'])
            with patch.object(history, '_read', wraps=history._read) as read:
                self.assertIsNotNone(history.find_candidates(self.root, ['old_720nmc.csv'])['exact'])
            self.assertTrue(any(str(call.args[0]).endswith('.metadata.json') for call in read.call_args_list))

    def test_external_index_replacement_updates_a_warm_result(self):
        self.write()
        state = history.find_candidates(str(self.root), ['old_720nmc.csv'])['exact']
        history.save_snapshot(str(self.root), state)
        history.find_candidates(str(self.root), ['old_720nmc.csv'])
        path = self.directory / history.INDEX_NAME
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['files'][history.source_key(str(self.root), ['old_720nmc.csv'])]['ranges']['xmin'] = 1.57
        path.write_text(json.dumps(payload), encoding='utf-8')
        self.assertEqual(history.find_candidates(str(self.root), ['old_720nmc.csv'])['exact']['ranges']['xmin'], 1.57)

    def test_same_size_metadata_and_index_edits_with_restored_mtime_are_visible(self):
        path = self.write()
        state = history.find_candidates(self.root, ['old_720nmc.csv'])['exact']
        before = path.stat()
        path.write_text(path.read_text(encoding='utf-8').replace('1.55', '1.56'), encoding='utf-8')
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(history.find_candidates(self.root, ['old_720nmc.csv'])['exact']['ranges']['xmin'], 1.56)
        history.save_snapshot(self.root, state)
        history.find_candidates(self.root, ['old_720nmc.csv'])
        path = self.directory / history.INDEX_NAME
        before = path.stat()
        path.write_text(path.read_text(encoding='utf-8').replace('1.55', '1.57'), encoding='utf-8')
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(history.find_candidates(self.root, ['old_720nmc.csv'])['exact']['ranges']['xmin'], 1.57)
