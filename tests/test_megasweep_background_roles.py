"""Role grouping must never silently mix physically different references."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
from core.megasweep.background_roles import validate_background_role
from core.megasweep.workflow import (load_dataset,analyze,reflection_preview,export_result,export_batch,export_video,MoviePreviewSession)


class BackgroundRoleTests(unittest.TestCase):
    def test_role_evidence_and_conflicts(self):
        self.assertEqual(validate_background_role('p',['unknown.csv'])['manual_paths'],[str(Path('unknown.csv').resolve())])
        self.assertEqual(validate_background_role('p',['Pback.csv'])['sources'][0]['filename_role'],'p')
        self.assertEqual(validate_background_role('custom',['Pback.csv','unknown.csv'])['named_roles'],['p'])
        for role,paths in [('p',['Nback.csv']),('n',['BNback.csv']),('bn',['Pback.csv']),('custom',['Pback.csv','Nback.csv']),('custom',['BNback.csv','Pback.csv']),('custom',['Pback_Nback.csv'])]:
            with self.subTest(role=role,paths=paths), self.assertRaises(ValueError): validate_background_role(role,paths)

    def test_consistent_calculation_previews_and_exports(self):
        with TemporaryDirectory() as folder:
            root=Path(folder).resolve(); wavelengths=np.linspace(700,720,7)
            columns=['Vbg','Vtg',*map(str,wavelengths)]
            source=root/'sample.csv'
            pd.DataFrame([[x,y,*([20.]*7)] for x in (0,1) for y in (0,1)],columns=columns).to_csv(source,index=False)
            p=root/'Pback.csv'; n=root/'Nback.csv'; bn=root/'BNback.csv'; unknown=root/'manual.csv'
            for path,value in ((p,5),(n,10),(bn,4),(unknown,8)):
                pd.DataFrame([[0,0,*([value]*7)]],columns=columns).to_csv(path,index=False)
            dataset=load_dataset(source,'Reflection')
            groups={'p':[str(p)],'n':[str(n)],'bn':[str(bn)],'custom':[str(unknown)]}
            params={'view':'rc_energy','energy_target':1240/710,'auto_background_scale':False,'background_groups':groups,'background_role':'p','cut_type':'x','movie_options':{'width':320,'height':240,'fps':4,'hold':.25}}
            for role,path,expected in [('p',p,3),('n',n,1),('bn',bn,4),('custom',unknown,1.5)]:
                config=dict(params,background_role=role)
                result=analyze(dataset,config,[path]); np.testing.assert_allclose(result['z'],expected)
                self.assertEqual(result['background_role'],role)
                preview=reflection_preview(dataset,config,[path]); np.testing.assert_allclose(preview['rc'],expected)
                self.assertEqual(preview['background_sources'][0]['filename_role'], 'unknown' if role=='custom' else role)
            result=analyze(dataset,params,[p]); outputs=export_result(result,root/'single')
            meta=json.loads(Path(outputs[2]).read_text(encoding='utf8'))
            self.assertEqual(meta['background_role'],'p'); self.assertEqual(meta['background_sources'][0]['path'],str(p))
            self.assertEqual(meta['parameters']['background_groups'],groups)
            batch=export_batch(dataset,params,[p],root/'batch'); self.assertTrue(batch)
            for file in batch:
                if file.endswith('.json'): self.assertEqual(json.loads(Path(file).read_text(encoding='utf8'))['background_role'],'p')
            preview=MoviePreviewSession(dataset,params,[p])
            try: self.assertTrue(preview.frame(0)['png'].startswith(b'\x89PNG'))
            finally: preview.close()
            video=export_video(dataset,params,[p],root/'movie')
            movie=json.loads(Path(video[1]).read_text(encoding='utf8'))
            self.assertEqual(movie['metadata']['background_role'],'p'); self.assertEqual(movie['metadata']['background_sources'][0]['path'],str(p))
            self.assertEqual(movie['validation'],'FFmpeg full decode passed')
            for fn in (lambda: analyze(dataset,params,[n]),lambda: reflection_preview(dataset,params,[n]),lambda: export_batch(dataset,params,[n],root/'badbatch'),lambda: MoviePreviewSession(dataset,params,[n]),lambda: export_video(dataset,params,[n],root/'badmovie')):
                with self.assertRaises(ValueError): fn()

    def test_ibias_and_resistance_do_not_require_or_validate_spectral_reference(self):
        with TemporaryDirectory() as folder:
            root=Path(folder).resolve(); source=root/'sample.csv'
            pd.DataFrame({'Vbg':[0,0,1,1], 'Vtg':[0,1,0,1], 'Ibias':[1,2,4,8],
                          'Vbias':[2,4,8,16], '700':[20]*4, '710':[20]*4}).to_csv(source,index=False)
            dataset=load_dataset(source,'Reflection')
            mixed=[root/'Pback.csv',root/'Nback.csv']
            for paths in ([],mixed):
                current=analyze(dataset,{'view':'ibias','background_role':'custom'},paths)
                resistance=analyze(dataset,{'view':'resistance','background_role':'custom'},paths)
                np.testing.assert_array_equal(current['z'],[[1,2],[4,8]])
                np.testing.assert_array_equal(resistance['z'],[[2,2],[2,2]])
            with self.assertRaises(ValueError):
                export_batch(dataset,{'background_role':'custom','cut_type':'x'},mixed,root/'badbatch')

if __name__ == '__main__': unittest.main()
