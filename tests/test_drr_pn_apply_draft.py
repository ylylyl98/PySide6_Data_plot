"""Prepared P/N usage has no side effects before its guarded UI commit."""
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_composite_background import CompositeBackground, CompositeSource, load_composite_source
from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_store import DrrPnStore
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


class PreparedPnUseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = DrrPnStore(self.root)
        self.pair = DrrPnPair(('Pbackground.csv',), ('Nbackground.csv',))
        columns = tuple(f'{value:.6f}' for value in np.linspace(700.,760.,31))
        for name,gate,value in (('Pbackground.csv',-20,10),('Nbackground.csv',20,20),('sample.csv',-6,30)):
            (self.root/name).write_text('Vbg,Vtg,Vbias,'+','.join(columns)+'\n'+
                f'{gate},0,0,'+','.join([str(value)]*len(columns))+'\n',encoding='utf-8')
        (self.root/'algorithm.py').write_text('version=1\n',encoding='utf-8')
        self.key = self.store.prepare(self.pair,settings={'width':4},algorithm_version='test-v1',dependencies=('algorithm.py',))
        p = load_composite_source(self.root,self.pair.p_files,'all',key='P')
        n = load_composite_source(self.root,self.pair.n_files,'all',key='N')
        report = dict(algorithm_version='test-v1',settings={'width':4},input_source_keys=['P','N'])
        generated = CompositeSource('generated','Prepared result',p.files+n.files,'all',p.energy,
            p.spectral_columns,.7*p.intensity+.3*n.intensity,
            source_hashes=p.source_hashes+n.source_hashes,generation_json=json.dumps(report))
        self.document = CompositeBackground((p,n,generated),'generated')
        self.version = self.store.save(self.document,self.key)

    def prepare(self,**kwargs):
        return self.store.prepare_use(self.version.version_id,measurements=('sample.csv',),**kwargs)

    def manifest(self):
        return self.store.manifest_path.read_bytes()

    def test_prepare_is_immutable_and_does_not_write_or_create_a_lock(self):
        before = self.manifest()
        paths = tuple(sorted(self.store.directory.iterdir()))
        with patch.object(self.store,'_locked',side_effect=AssertionError('prepare wrote a lock')), \
                patch.object(self.store,'_atomic_json',side_effect=AssertionError('prepare wrote history')):
            prepared = self.prepare()
        self.assertEqual(prepared.version.version_id,self.version.version_id)
        self.assertEqual(prepared.measurement_records[0][0],'sample.csv')
        self.assertEqual(self.manifest(),before)
        self.assertEqual(tuple(sorted(self.store.directory.iterdir())),paths)
        with self.assertRaises(FrozenInstanceError):
            prepared.root = 'elsewhere'

    def test_commit_reads_no_spectra_or_hashes_and_preserves_measurement_identity(self):
        prepared = self.prepare()
        with patch('core.drr_pn_store._file_digest',side_effect=AssertionError('commit hashed bytes')), \
                patch('core.drr_pn_store.inspect_csv_spectral_grid',side_effect=AssertionError('commit read header')), \
                patch.object(self.store,'_version',side_effect=AssertionError('commit revalidated spectra')):
            result = self.store.commit_prepared_use(prepared)
        self.assertEqual(result.version_id,self.version.version_id)
        self.assertEqual(self.store.find_applied(('sample.csv',)).version_id,self.version.version_id)
        payload = json.loads(self.manifest())
        association = next(iter(payload['applied'].values()))
        self.assertEqual(association['measurements'],[
            {'path':name,'sha256':digest} for name,digest in prepared.measurement_records])

    def test_each_input_change_between_prepare_and_commit_rejects_without_association(self):
        for path in (self.version.csv_path,self.version.metadata_path,self.root/'Pbackground.csv',
                     self.root/'Nbackground.csv',self.root/'algorithm.py',self.root/'sample.csv'):
            with self.subTest(path=path.name):
                prepared = self.prepare()
                manifest = self.manifest()
                original = path.read_bytes()
                path.write_bytes(original+b'\n')
                try:
                    with self.assertRaisesRegex(ValueError,'changed after verification'):
                        self.store.commit_prepared_use(prepared)
                    self.assertEqual(self.manifest(),manifest)
                finally:
                    path.write_bytes(original)

    def test_same_size_measurement_rewrite_with_restored_mtime_rejects(self):
        prepared = self.prepare()
        path = self.root/'sample.csv'
        before = path.stat()
        path.write_bytes(path.read_bytes().replace(b'30',b'31'))
        self.assertEqual(path.stat().st_size,before.st_size)
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        manifest = self.manifest()
        with self.assertRaisesRegex(ValueError,'changed after verification'):
            self.store.commit_prepared_use(prepared)
        self.assertEqual(self.manifest(),manifest)

    def test_changed_version_record_rejects_but_unrelated_history_is_preserved(self):
        prepared = self.prepare()
        payload = json.loads(self.manifest())
        payload['versions'][0]['number'] += 1
        self.store._atomic_json(self.store.manifest_path,payload)
        changed = self.manifest()
        with self.assertRaisesRegex(ValueError,'history changed'):
            self.store.commit_prepared_use(prepared)
        self.assertEqual(self.manifest(),changed)
        prepared = self.prepare()
        payload = json.loads(self.manifest())
        payload['last_used']['unrelated-pair'] = 'unrelated-version'
        self.store._atomic_json(self.store.manifest_path,payload)
        self.store.commit_prepared_use(prepared)
        self.assertEqual(json.loads(self.manifest())['last_used']['unrelated-pair'],'unrelated-version')

    def test_unrelated_new_version_does_not_expire_prepared_use(self):
        prepared = self.prepare()
        another = self.store.save(self.document,self.key,force_new=True)
        self.store.commit_prepared_use(prepared)
        self.assertEqual({item.version_id for item in self.store.list_versions()},
                         {self.version.version_id,another.version_id})

    def test_cancelled_prepare_and_commit_write_nothing(self):
        prepared = self.prepare()
        manifest = self.manifest()
        cancellation = LoadCancellation()
        with load_scope(cancellation):
            cancellation.cancel()
            with self.assertRaises(LoadCancelled):
                self.prepare()
        cancellation = LoadCancellation()
        with load_scope(cancellation):
            cancellation.cancel()
            with self.assertRaises(LoadCancelled):
                self.store.commit_prepared_use(prepared)
        self.assertEqual(self.manifest(),manifest)
        self.assertFalse((self.store.directory/'.store.lock').exists())

    def test_changes_during_prepare_are_rejected_without_side_effects(self):
        original = self.store._measurement_records
        def change_after_hash(names):
            records = original(names)
            with (self.root/'Pbackground.csv').open('ab') as stream:
                stream.write(b'\n')
            return records
        manifest = self.manifest()
        with patch.object(self.store,'_measurement_records',change_after_hash), \
                self.assertRaisesRegex(ValueError,'changed during verification'):
            self.prepare()
        self.assertEqual(self.manifest(),manifest)

    def test_stale_or_missing_sources_require_consent_and_cannot_change_after_consent(self):
        path = self.root/'Pbackground.csv'
        path.unlink()
        with self.assertRaisesRegex(ValueError,'missing_sources'):
            self.prepare()
        prepared = self.prepare(allow_stale=True)
        self.assertEqual(prepared.version.status,'missing_sources')
        self.assertTrue(any(stamp is None for _,stamp in prepared.signatures))
        self.store.commit_prepared_use(prepared)
        self.assertEqual(self.store.last_used(self.pair).version_id,self.version.version_id)
        prepared = self.prepare(allow_stale=True)
        path.write_text('restored but different',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'changed after verification'):
            self.store.commit_prepared_use(prepared)

    def test_wrong_root_and_unverified_draft_reject(self):
        prepared = self.prepare()
        other = DrrPnStore(self.root/'another')
        with self.assertRaisesRegex(ValueError,'another experiment'):
            other.commit_prepared_use(prepared)
        self.assertFalse(other.directory.exists())
        with self.assertRaisesRegex(ValueError,'not verified'):
            self.store.commit_prepared_use(replace(prepared,_verified=None))


if __name__ == '__main__':
    unittest.main()
