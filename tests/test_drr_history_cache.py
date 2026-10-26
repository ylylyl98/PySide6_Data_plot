import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import drr_sources as sources


class DrrHistoryCacheTests(unittest.TestCase):
    def test_common_member_references_resolve_each_descriptor_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'Processed Data' / 'DRR'
            history.mkdir(parents=True)
            measurements = ('a.csv', 'b.csv', 'c.csv')
            backgrounds = ('first.csv', 'second.csv')
            for name in (*measurements, *backgrounds):
                (root / name).write_text(name)
            (history / 'one.metadata.json').write_text(json.dumps({
                'operation': 'DR/R',
                'sources': [
                    {'role': role, 'source_path': str(root / name), 'name': name}
                    for role, names in (('measurement', measurements), ('background', backgrounds))
                    for name in names],
                'processing': {
                    'baseline_selection': 'External', 'baseline_which': 'all',
                    'drr_background_assignments': [
                        {'measurement': str(root / name), 'baseline_files': list(backgrounds),
                         'baseline_mode': 'External', 'baseline_which': 'all'}
                        for name in measurements]}}))
            with patch.object(sources, '_metadata_source_name',
                              wraps=sources._metadata_source_name) as resolve:
                records = sources._read_drr_metadata(root, require_drr_operation=True)[1]
            self.assertEqual(resolve.call_count, len(measurements) + len(backgrounds))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].measurement_files, measurements)
            self.assertEqual(tuple(item.baseline_files for item in records[0].member_assignments),
                             (backgrounds,) * len(measurements))

    def test_member_aliases_rebind_after_sources_move_between_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'Processed Data' / 'DRR'
            history.mkdir(parents=True)
            old = root / 'old'
            new = root / 'Initial Data'
            old.mkdir()
            new.mkdir()
            measurements = ('a.csv', 'b.csv')
            backgrounds = ('background.csv',)
            for name in (*measurements, *backgrounds):
                (old / name).write_text('old ' + name)
                (new / name).write_text('new ' + name)
            # The recorded assignment spelling differs from its descriptor;
            # only canonical path identity connects them.
            def alias(name):
                return str(old / '..' / 'old' / name)
            (history / 'one.metadata.json').write_text(json.dumps({
                'operation': 'DR/R',
                'sources': [
                    {'role': role, 'source_path': str(old / name), 'name': 'Initial Data/' + name}
                    for role, names in (('measurement', measurements), ('background', backgrounds))
                    for name in names],
                'processing': {'baseline_selection': 'External',
                    'drr_background_assignments': [
                        {'measurement': alias(name), 'baseline_files': [alias(backgrounds[0])]}
                        for name in measurements]}}))
            first = sources._read_drr_metadata(root, require_drr_operation=True)[1][0]
            self.assertEqual(first.measurement_files, tuple('old/' + name for name in measurements))
            self.assertEqual(tuple(item.baseline_files for item in first.member_assignments),
                             (('old/background.csv',),) * len(measurements))
            for name in (*measurements, *backgrounds):
                (old / name).unlink()
            second = sources._read_drr_metadata(root, require_drr_operation=True)[1][0]
            self.assertEqual(second.measurement_files, tuple('Initial Data/' + name for name in measurements))
            self.assertEqual(tuple(item.baseline_files for item in second.member_assignments),
                             (('Initial Data/background.csv',),) * len(measurements))

    def test_symlink_target_change_invalidates_member_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'Processed Data' / 'DRR'
            history.mkdir(parents=True)
            first, second = root / 'first.csv', root / 'second.csv'
            first.write_text('first')
            second.write_text('different second')
            alias = root / 'alias.csv'
            try:
                alias.symlink_to(first)
            except OSError as exc:
                self.skipTest(f'Test filesystem cannot create symbolic links: {exc}')
            (history / 'one.metadata.json').write_text(json.dumps({
                'operation': 'DR/R',
                'sources': [{'role': 'measurement', 'source_path': str(alias)}],
                'processing': {'drr_background_assignments': [
                    {'measurement': str(alias), 'baseline_mode': 'Self (last frame)'}]}}))
            record = sources._read_drr_metadata(root, require_drr_operation=True)[1][0]
            self.assertEqual(record.measurement_files, ('first.csv',))
            self.assertEqual(record.member_assignments[0].measurement_file, 'first.csv')
            alias.unlink()
            alias.symlink_to(second)
            record = sources._read_drr_metadata(root, require_drr_operation=True)[1][0]
            self.assertEqual(record.measurement_files, ('second.csv',))
            self.assertEqual(record.member_assignments[0].measurement_file, 'second.csv')

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
