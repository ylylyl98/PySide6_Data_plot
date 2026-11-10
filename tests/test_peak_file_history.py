import json
import numpy as np
from core.loader import DataCube
from core.peak_workspace import create_dataset, load_workspace, save_workspace
from core.peak_file_history import history_path, load_histories, load_last_histories, save_histories


def make(path, channel='raw', value=1):
    cube = DataCube(np.array([1.,2.,3.]), np.array([0.]), np.array([[value,2.,3.]]), 'Gate','test','DRR')
    return create_dataset(cube, 'DRR', channel, json.dumps(['DRR',str(path),['input1']]), 'sample')


def check_paths_are_stable_and_separate(tmp_path):
    root = tmp_path/'experiment'
    a = make(root/'Initial Data'/'one.dat')
    b = make(root/'Initial Data'/'two.dat')
    first = history_path(a, tmp_path)
    assert first == history_path(dict(a), tmp_path)
    assert first != history_path(b, tmp_path)
    assert first == root/'Processed Data'/'DRR'/'one_analysis_history.npz'
    changed = dict(a, source=json.dumps(['DRR',str(root/'Initial Data'/'one.dat'),['input2']]))
    assert history_path(changed, tmp_path) != first
    processed = dict(a, processed_path=str(root/'Processed Data'/'DRR'/'result.dat'))
    assert history_path(processed, tmp_path) == root/'Processed Data'/'DRR'/'result_analysis_history.npz'
    assert history_path(dict(processed, key='different'),tmp_path) == history_path(processed,tmp_path)


def check_save_merge_and_last_session(tmp_path):
    a = make(tmp_path/'Initial Data'/'one.dat')
    b = make(tmp_path/'Initial Data'/'one.dat', value=9)
    other = make(tmp_path/'Initial Data'/'two.dat')
    save_histories([a,other], tmp_path, a['key'])
    save_histories([b], tmp_path, b['key'])
    data, report = load_histories([b], tmp_path)
    assert {d['key'] for d in data} == {a['key'], b['key']}
    assert report['migrated_paths'] == []
    last, active = load_last_histories(tmp_path)
    assert active == b['key'] and {d['key'] for d in last} == {a['key'], b['key']}
    assert len(load_workspace(history_path(other,tmp_path))[0]) == 1


def check_legacy_migration_target_wins_and_source_unchanged(tmp_path):
    a = make(tmp_path/'Initial Data'/'one.dat')
    legacy = tmp_path/'legacy.npz'
    a['name'] = 'legacy'
    save_workspace(legacy, [a], a['key'])
    original = legacy.read_bytes()
    processed = str(tmp_path/'Processed Data'/'DRR'/'one.dat')
    current = dict(a, name='current', processed_path=processed, history_root=str(tmp_path))
    save_histories([current], tmp_path, current['key'])
    loaded, report = load_histories([current], tmp_path, legacy)
    assert len(loaded) == 1 and loaded[0]['name'] == 'current'
    assert report['migrated_paths'] and legacy.read_bytes() == original
    _, second = load_histories([current], tmp_path, legacy)
    assert second['migrated_paths'] == []


def check_migration_maps_old_versions_to_current_file(tmp_path):
    a = make(tmp_path/'Initial Data'/'one.dat')
    b = make(tmp_path/'Initial Data'/'one.dat', value=9)
    legacy = tmp_path/'legacy.npz'
    save_workspace(legacy,[a,b],a['key'])
    current = dict(a, processed_path=str(tmp_path/'Processed Data'/'DRR'/'one.dat'))
    loaded, _ = load_histories([current],tmp_path,legacy)
    assert {d['key'] for d in loaded} == {a['key'],b['key']}
    assert all(d['processed_path'] == current['processed_path'] and d['history_root'] == str(tmp_path) for d in loaded)


def check_legacy_split_once_across_sources_and_unmatched(tmp_path, monkeypatch):
    import core.peak_file_history as history
    a = make(tmp_path/'Initial Data'/'one.dat')
    b = make(tmp_path/'Initial Data'/'two.dat')
    c = make(tmp_path/'Initial Data'/'missing.dat')
    legacy = tmp_path/'legacy.npz'
    save_workspace(legacy, [a,b], a['key'])
    original = legacy.read_bytes()
    real_load = history.load_workspace
    calls = []
    def tracked(path, *args, **kwargs):
        calls.append(path)
        return real_load(path, *args, **kwargs)
    monkeypatch.setattr(history, 'load_workspace', tracked)
    for item in (a,b,c):
        result, _ = load_histories([item],tmp_path,legacy)
        assert bool(result) == (item is not c)
    assert sum(str(p) == str(legacy) for p in calls) == 1
    assert legacy.read_bytes() == original


def check_relative_path_stays_under_fallback_without_stat(tmp_path, monkeypatch):
    from pathlib import Path
    d = make('unknown.dat')
    def fail(*args, **kwargs):
        raise AssertionError('path lookup must not stat')
    monkeypatch.setattr(Path, 'is_dir', fail)
    monkeypatch.setattr(Path, 'resolve', fail)
    from core.peak_output_names import output_folder
    assert output_folder(d,tmp_path).is_relative_to(tmp_path)


import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


class Tests(unittest.TestCase):
    def test_existing_flat_history_rebinds_across_index_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'Initial Data'/'one.dat'
            a = make(source)
            a.update(processed_path=str(root/'out'/'one.dat'), processed_path_inferred=True)
            first_root, second_root = root/'settings1', root/'settings2'
            path = save_histories([a], first_root)[0]
            self.assertEqual(history_path(a, second_root), path)
            other = dict(a, source=json.dumps(['DRR', str(source), ['different']]))
            self.assertNotEqual(history_path(other, second_root), path)

    def test_nested_history_is_read_only_once_after_migration(self):
        import core.peak_file_history as history
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = make(root/'Initial Data'/'one.dat')
            old = history._legacy_history_path(a, root)
            save_workspace(old, [a], a['key'])
            real = history.load_workspace
            with patch.object(history, 'load_workspace', wraps=real) as tracked:
                save_histories([a], root)
                save_histories([a], root)
                load_histories([a], root)
                self.assertEqual(sum(call.args[0] == old for call in tracked.call_args_list), 1)

    def test_flat_collision_binding_and_old_nested_migration(self):
        from core.peak_file_history import _legacy_history_path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = make(root/'Initial Data'/'one.dat')
            a.update(processed_path=str(root/'out'/'long.dat'), processed_path_inferred=True)
            b = dict(a, source=json.dumps(['DRR', str(root/'Initial Data'/'one.dat'), ['different']]))
            old = _legacy_history_path(a, root)
            save_workspace(old, [a], a['key'])
            first = history_path(a, root)
            second = history_path(b, root)
            self.assertEqual(first.name, 'long_analysis_history.npz')
            self.assertEqual(second.name, 'long_analysis_history_02.npz')
            self.assertEqual(first, history_path(dict(a, key='new fingerprint'), root))
            loaded, _ = load_histories([a], root)
            self.assertEqual([d['key'] for d in loaded], [a['key']])
            self.assertTrue(old.is_file())
            self.assertTrue(first.is_file())

    def test_unbound_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = make(root/'Initial Data'/'one.dat')
            folder = root/'Processed Data'/'DRR'
            folder.mkdir(parents=True)
            foreign = folder/'one_analysis_history.npz'
            foreign.write_bytes(b'untouched')
            self.assertEqual(history_path(a, root).name, 'one_analysis_history_02.npz')
            save_histories([a], root)
            self.assertEqual(foreign.read_bytes(), b'untouched')

    def test_inferred_to_exact_preserves_only_matching_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = make(root/'Initial Data'/'one.dat')
            a.update(processed_path=str(root/'Processed Data'/'DRR'/'result.dat'), processed_path_inferred=True)
            b = make(root/'Initial Data'/'two.dat')
            b.update(processed_path=a['processed_path'], processed_path_inferred=True)
            self.assertNotEqual(history_path(a, root), history_path(b, root))
            save_histories([a, b], root)
            exact = dict(a, processed_path_inferred=False)
            results, _ = load_histories([exact], root)
            self.assertEqual({d['key'] for d in results}, {a['key']})
            self.assertFalse(results[0]['processed_path_inferred'])
            collision = dict(exact, processed_path=str(root/'Processed Data'/'DRR'/'result_01.dat'),
                             processed_path_hint=a['processed_path'])
            moved, _ = load_histories([collision], root)
            self.assertEqual({d['key'] for d in moved}, {a['key']})
            self.assertEqual(moved[0]['processed_path'], collision['processed_path'])
            other = dict(exact, processed_path=str(root/'Processed Data'/'DRR'/'other.dat'))
            self.assertEqual(load_histories([other], root)[0], [])

    def test_paths_are_stable_and_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            check_paths_are_stable_and_separate(Path(directory))

    def test_save_merge_and_last_session(self):
        with tempfile.TemporaryDirectory() as directory:
            check_save_merge_and_last_session(Path(directory))

    def test_legacy_migration_target_wins_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            check_legacy_migration_target_wins_and_source_unchanged(Path(directory))

    def test_migration_maps_old_versions_to_current_file(self):
        with tempfile.TemporaryDirectory() as directory:
            check_migration_maps_old_versions_to_current_file(Path(directory))

    def test_legacy_split_once_across_sources_and_unmatched(self):
        with tempfile.TemporaryDirectory() as directory:
            with _MonkeyPatch() as monkeypatch:
                check_legacy_split_once_across_sources_and_unmatched(Path(directory), monkeypatch)

    def test_relative_path_stays_under_fallback_without_stat(self):
        with tempfile.TemporaryDirectory() as directory:
            with _MonkeyPatch() as monkeypatch:
                check_relative_path_stays_under_fallback_without_stat(Path(directory), monkeypatch)


class _MonkeyPatch:
    def __init__(self):
        self.patches = []

    def __enter__(self):
        return self

    def setattr(self, obj, name, value):
        p = patch.object(obj, name, value)
        p.start()
        self.patches.append(p)

    def __exit__(self, *args):
        for p in reversed(self.patches):
            p.stop()
