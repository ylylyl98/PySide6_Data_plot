import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import drr_sources as sources


class DrrHistoryCacheTests(unittest.TestCase):
    def test_edit_reparses_only_changed_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'Processed Data/DRR'
            history.mkdir(parents=True)
            for name in ('a', 'b'):
                (history / f'{name}.metadata.json').write_text(json.dumps({
                    'operation': 'DR/R', 'sources': [{'role': 'measurement', 'name': name + '.csv'}]}))
            sources._read_drr_metadata(root)
            (history / 'a.metadata.json').write_text(json.dumps({
                'operation': 'DR/R', 'sources': [{'role': 'measurement', 'name': 'changed.csv'}]}))
            with patch.object(sources.json, 'loads', wraps=json.loads) as parse:
                records = sources._read_drr_metadata(root)[1]
            self.assertEqual(parse.call_count, 1)
            self.assertEqual({r.measurement_files for r in records}, {('changed.csv',), ('b.csv',)})

    def test_catalog_worker_prepares_selection_history_before_return(self):
        from ui_qt.main_window import _scan_drr_catalog_worker
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'Processed Data' / 'DRR'
            history.mkdir(parents=True)
            (history / 'one.metadata.json').write_text(json.dumps({
                'operation': 'DR/R', 'sources': [
                    {'role': 'measurement', 'name': 'measurement.csv'}]}))
            with patch('core.drr_catalog.load_drr_catalog', return_value=[]):
                _scan_drr_catalog_worker(tmp, None, progress=None, log=None)
            with patch.object(sources.json, 'loads', wraps=json.loads) as parse:
                sources._read_drr_metadata(root, require_drr_operation=True)
                self.assertEqual(parse.call_count, 0)

    def test_unchanged_history_is_not_reparsed_and_results_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'Processed Data' / 'DRR'
            history.mkdir(parents=True)
            metadata = history / 'one.metadata.json'
            metadata.write_text(json.dumps({'operation': 'DR/R', 'sources': [
                {'role': 'measurement', 'name': 'measurement.csv'}]}))
            with patch.object(sources.json, 'loads', wraps=json.loads) as parse:
                roles, records = sources._read_drr_metadata(root)
                roles.clear()
                records.clear()
                roles, records = sources._read_drr_metadata(root)
                self.assertEqual(parse.call_count, 1)
                self.assertEqual(len(records), 1)
                self.assertTrue(roles)

    def test_history_edits_additions_and_deletions_invalidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'Processed Data' / 'DRR'
            history.mkdir(parents=True)
            metadata = history / 'one.metadata.json'
            def write(path, name):
                path.write_text(json.dumps({'operation': 'DR/R', 'sources': [
                    {'role': 'measurement', 'name': name}]}))
            write(metadata, 'a.csv')
            self.assertEqual(sources._read_drr_metadata(root)[1][0].measurement_files, ('a.csv',))
            write(metadata, 'longer.csv')
            self.assertEqual(sources._read_drr_metadata(root)[1][0].measurement_files, ('longer.csv',))
            second = history / 'two.metadata.json'
            write(second, 'b.csv')
            self.assertEqual(len(sources._read_drr_metadata(root)[1]), 2)
            second.unlink()
            self.assertEqual(len(sources._read_drr_metadata(root)[1]), 1)

    def test_source_move_reselects_existing_portable_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'Processed Data' / 'DRR'
            history.mkdir(parents=True)
            old = root / 'old.csv'
            new = root / 'Initial Data' / 'new.csv'
            new.parent.mkdir()
            old.write_text('old')
            new.write_text('new')
            (history / 'one.metadata.json').write_text(json.dumps({
                'operation': 'DR/R', 'sources': [{'role': 'measurement',
                    'source_path': str(old), 'name': 'Initial Data/new.csv'}]}))
            self.assertEqual(sources._read_drr_metadata(root)[1][0].measurement_files, ('old.csv',))
            old.unlink()
            self.assertEqual(sources._read_drr_metadata(root)[1][0].measurement_files, ('Initial Data/new.csv',))

    def test_operation_filter_has_separate_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'Processed Data' / 'DRR'
            history.mkdir(parents=True)
            (history / 'one.metadata.json').write_text(json.dumps({'sources': [
                {'role': 'measurement', 'name': 'legacy.csv'}]}))
            self.assertEqual(len(sources._read_drr_metadata(root)[1]), 1)
            self.assertEqual(sources._read_drr_metadata(root, require_drr_operation=True)[1], [])

    def test_move_during_read_does_not_publish_stale_dependency_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'Processed Data' / 'DRR'
            history.mkdir(parents=True)
            old = root / 'old.csv'
            new = root / 'Initial Data' / 'new.csv'
            new.parent.mkdir()
            old.write_text('old')
            new.write_text('new')
            (history / 'one.metadata.json').write_text(json.dumps({
                'operation': 'DR/R', 'sources': [{'role': 'measurement',
                    'source_path': str(old), 'name': 'Initial Data/new.csv'}]}))
            parse = sources._parse_drr_metadata
            def moving_parse(*args, **kwargs):
                result = parse(*args, **kwargs)
                old.unlink()
                return result
            with patch.object(sources, '_parse_drr_metadata', side_effect=moving_parse):
                sources._read_drr_metadata(root)
            self.assertEqual(sources._read_drr_metadata(root)[1][0].measurement_files,
                             ('Initial Data/new.csv',))
