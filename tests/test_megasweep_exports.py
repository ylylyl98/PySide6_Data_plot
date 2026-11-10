import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from core.megasweep.workflow import load_dataset, analyze, export_result, export_batch

class MegasweepExportTests(unittest.TestCase):
    def test_map_and_batch_exports(self):
        with TemporaryDirectory() as d:
            p=Path(d); src=p/'sample.csv'
            pd.DataFrame({'x':[0,0,1,1], 'y':[0,1,0,1], '700':[10,11,12,13], '710':[9,10,11,12]}).to_csv(src,index=False)
            ds=load_dataset(src); params={'view':'intensity','energy_min':1.7,'energy_max':1.8,'baseline':0,'auto_baseline':False,'cut_type':'x'}
            paths=export_result(analyze(ds,params),p/'out')
            self.assertTrue(all(Path(x).is_file() for x in paths))
            original=Path(paths[1]).read_bytes()
            repeated=export_result(analyze(ds,params),p/'out')
            self.assertNotEqual(paths[1],repeated[1]); self.assertEqual(original,Path(paths[1]).read_bytes())
            paths=export_batch(ds,params,(),p/'out')
            self.assertEqual(len(paths),7); self.assertTrue(all(Path(x).is_file() for x in paths))
            index = pd.read_csv(paths[0])
            self.assertEqual(index['status'].tolist(), ['completed', 'completed'])
            self.assertTrue(all(len(__import__('json').loads(value)) == 3 for value in index['exported_files']))

    def test_batch_index_replace_failure_retains_previous_snapshot(self):
        import os
        from unittest.mock import patch
        with TemporaryDirectory() as d:
            p=Path(d); src=p/'sample.csv'
            pd.DataFrame({'x':[0,0,1,1], 'y':[0,1,0,1], '700':[10,11,12,13], '710':[9,10,11,12]}).to_csv(src,index=False)
            real_replace=os.replace
            calls=[]
            def fail_second(source, destination):
                calls.append(destination)
                if len(calls) == 2:
                    raise OSError('simulated index replace failure')
                return real_replace(source, destination)
            with patch('os.replace', side_effect=fail_second):
                with self.assertRaisesRegex(OSError, 'simulated index'):
                    export_batch(load_dataset(src), {'cut_type':'x'}, (), p/'out')
            index=next((p/'out').glob('*_batch_index.csv'))
            self.assertEqual(pd.read_csv(index)['status'].tolist(), ['pending', 'pending'])
            self.assertFalse(list((p/'out').glob('.batch-index-*.tmp')))

    def test_preview_cache_and_cancellation(self):
        from core.megasweep.workflow import MoviePreviewSession
        with TemporaryDirectory() as d:
            p=Path(d); src=p/'sample.csv'
            pd.DataFrame({'x':[0,0,1,1], 'y':[0,1,0,1], '700':[10,11,12,13], '710':[9,10,11,12]}).to_csv(src,index=False)
            session=MoviePreviewSession(load_dataset(src),{'cut_type':'x','movie_options':{'width':320,'height':240,'fps':4,'hold':.5,'direction':'reverse'}})
            try:
                first=session.frame(0); again=session.frame(1); last=session.frame(3)
                self.assertEqual(first['value'],1); self.assertEqual(last['value'],0)
                self.assertEqual(first['png'],again['png']); self.assertEqual(first['frame_count'],4)
                self.assertEqual(len(session._cache.frame_cache),2)
                with self.assertRaises(InterruptedError): session.frame(0,cancelled=lambda:True)
            finally: session.close()

    def test_invalid_color_does_not_publish_and_metadata(self):
        import json
        with TemporaryDirectory() as d:
            p=Path(d); src=p/'sample.csv'
            pd.DataFrame({'x':[0,0,1,1], 'y':[0,1,0,1], '700':[10,11,12,13], '710':[9,10,11,12]}).to_csv(src,index=False)
            ds=load_dataset(src); result=analyze(ds,{'view':'intensity','energy_min':1.7,'energy_max':1.8,'auto_baseline':False,'baseline':0})
            with self.assertRaises(ValueError): export_result(result,p/'out',parameters={'color_auto':False,'vmin':3,'vmax':2})
            self.assertFalse((p/'out').exists())
            paths=export_result(result,p/'out',parameters={'color_auto':False,'vmin':0,'vmax':20})
            self.assertEqual(len(paths),3)
            info=json.loads(Path(paths[2]).read_text(encoding='utf8'))
            self.assertEqual(info['source_path'],str(src.resolve())); self.assertEqual(info['parameters']['vmax'],20)

    def test_batch_cancellation_preserves_completed_group(self):
        from core.megasweep.workflow import BatchCancelled
        with TemporaryDirectory() as d:
            p=Path(d); src=p/'sample.csv'
            pd.DataFrame({'x':[0,0,1,1], 'y':[0,1,0,1], '700':[10,11,12,13], '710':[9,10,11,12]}).to_csv(src,index=False)
            calls=[0]
            def cancel():
                calls[0]+=1
                return calls[0]>=4
            with self.assertRaises(BatchCancelled) as cm:
                export_batch(load_dataset(src),{'cut_type':'x'},(),p/'out',cancelled=cancel)
            self.assertEqual(len(cm.exception.completed_paths),4)
            index = pd.read_csv(cm.exception.completed_paths[0])
            self.assertEqual(index['status'].tolist(), ['completed', 'cancelled'])
            self.assertTrue(all(Path(path).is_file() for path in cm.exception.completed_paths))
            self.assertFalse(any(item.name.startswith('.megasweep-export-') for item in (p/'out').iterdir()))

    def test_empty_slice_and_batch_are_not_success(self):
        with TemporaryDirectory() as d:
            p=Path(d); src=p/'sample.csv'
            pd.DataFrame({'x':[0,0], 'y':[0,1], '700':[10,11], '710':[9,10]}).to_csv(src,index=False)
            ds=load_dataset(src)
            with self.assertRaisesRegex(ValueError,'No measured points'):
                analyze(ds,{'view':'slice','cut_type':'x','cut_value':99})
            with self.assertRaisesRegex(ValueError,'No measured slices'):
                export_batch(ds,{'cut_type':'x','min_points':3},(),p/'out')

    def test_video_wrapper_writes_source_metadata(self):
        import json
        from core.megasweep.workflow import export_video
        with TemporaryDirectory() as d:
            p=Path(d); src=p/'sample.csv'
            pd.DataFrame({'x':[0,0,1,1], 'y':[0,1,0,1], '700':[10,11,12,13], '710':[9,10,11,12]}).to_csv(src,index=False)
            paths=export_video(load_dataset(src),{'cut_type':'x','movie_options':{'width':320,'height':240,'fps':4,'hold':.25}},(),p/'out')
            self.assertEqual(len(paths),2); self.assertTrue(all(Path(path).is_file() for path in paths))
            info=json.loads(Path(paths[1]).read_text(encoding='utf8'))
            self.assertEqual(info['metadata']['source_path'],str(src.resolve()))
            self.assertEqual(info['validation'],'FFmpeg full decode passed')

    def test_transport_export_keeps_raw_names_and_units(self):
        import json
        with TemporaryDirectory() as d:
            p=Path(d); src=p/'transport.csv'
            src.write_text('Vbg,Current\nV,nA\n0,2\n1,3\n',encoding='utf8')
            ds=load_dataset(src,mode='Transport')
            result=analyze(ds,{'view':'transport_curve','x_column':'Vbg','channel':'Current'})
            paths=export_result(result,p/'out')
            self.assertEqual(list(pd.read_csv(paths[0]).columns),['Vbg','Current'])
            info=json.loads(Path(paths[2]).read_text(encoding='utf8'))
            self.assertEqual(info['units'],{'Vbg':'V','Current':'nA'})

    def test_derivative_movie_preview_all_frames_and_video(self):
        import json
        import numpy as np
        from core.megasweep.workflow import MoviePreviewSession, export_video
        with TemporaryDirectory() as d:
            p=Path(d); src=p/'sample.csv'; wavelength=np.linspace(660,820,41)
            rows=[[x,y,*(1000+(200+20*x+10*y)*np.exp(-((wavelength-745-x*3-y)/12)**2))] for x in (-1.,0.,1.) for y in (-2.,0.,2.)]
            pd.DataFrame(rows,columns=['Vbg','Vtg',*map(str,wavelength)]).to_csv(src,index=False)
            ds=load_dataset(src); params={'cut_type':'x','derivative':'second_derivative','derivative_window':9,'movie_options':{'width':800,'height':600,'fps':5,'hold':.2}}
            preview=MoviePreviewSession(ds,params)
            try:
                frames=[preview.frame(i) for i in range(preview.frame_count)]
                self.assertEqual(len(frames),3)
                self.assertEqual([item['value'] for item in frames],[-1,0,1])
                self.assertTrue(all(item['png'].startswith(b'\x89PNG') for item in frames))
            finally: preview.close()
            paths=export_video(ds,params,(),p/'out')
            info=json.loads(Path(paths[1]).read_text(encoding='utf8'))
            self.assertEqual(info['frame_count'],3)
            self.assertEqual(info['validation'],'FFmpeg full decode passed')

if __name__=='__main__': unittest.main()
