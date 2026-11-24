import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from core.megasweep.workflow import load_dataset, analyze, export_result, export_batch

class MegasweepExportTests(unittest.TestCase):
    def test_plot_axis_labels_keep_exported_coordinate_headers(self):
        from unittest.mock import patch
        from core.megasweep import workflow
        from core.megasweep.megasweep_analysis import display_axis_name, spectral_slice_title

        self.assertEqual(display_axis_name('Bias_set (V)'), 'Bias (V)')
        self.assertEqual(display_axis_name('Offset'), 'Offset')
        self.assertEqual(display_axis_name('Bias_set_meas'), 'Bias_set_meas')

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'scan.csv'
            x_column, y_column = 'Doping_axis_b_set', 'Vbias_axis_a_set'
            pd.DataFrame({y_column: [0, 1, 0, 1], x_column: [0, 0, 1, 1],
                          '700': [10, 11, 12, 13], '710': [9, 10, 11, 12]}).to_csv(source, index=False)
            data = load_dataset(source)
            self.assertEqual((data['x_name'], data['y_name']), (x_column, y_column))
            params = dict(view='intensity', energy_min=1.7, energy_max=1.8,
                          auto_baseline=False, baseline=0., export_dpi=100)
            plot_labels = []
            save_png = workflow._save_result_png

            def capture_labels(fig, ax, path, dpi):
                plot_labels.append((ax.get_xlabel(), ax.get_ylabel()))
                save_png(fig, ax, path, dpi)

            with patch.object(workflow, '_save_result_png', side_effect=capture_labels):
                scan = analyze(data, params)
                self.assertEqual((scan['x_label'], scan['y_label']), ('Doping', 'Vbias'))
                map_csv, _, _ = export_result(scan, root / 'out')
                cut = analyze(data, dict(params, view='slice', cut_type='x', cut_value=0.))
                self.assertEqual(cut['y_label'], 'Vbias')
                self.assertEqual(spectral_slice_title(cut['data']), 'Doping = 0')
                cut_csv, _, _ = export_result(cut, root / 'out')

            self.assertEqual(plot_labels, [('Doping', 'Vbias'), ('Energy (eV)', 'Vbias')])
            self.assertEqual(pd.read_csv(map_csv).columns.tolist(), [x_column, y_column, 'PL intensity'])
            cut_columns = pd.read_csv(cut_csv).columns.tolist()
            self.assertEqual((cut_columns[0], cut_columns[-1]), (y_column, f'fixed_{x_column}'))

    def test_png_canvas_and_data_rectangle_match_across_maps_and_cuts(self):
        import matplotlib as mpl
        import numpy as np
        from PIL import Image

        with TemporaryDirectory() as directory, mpl.rc_context({'savefig.bbox': 'tight'}):
            folder = Path(directory)
            for view, ymax in [('intensity', 1.), ('intensity', 12345.), ('slice', 12345.)]:
                with self.subTest(view=view, ymax=ymax):
                    source = folder / 'scan.csv'
                    pd.DataFrame({'x': [0, 0, 1, 1], 'y': [0, ymax, 0, ymax],
                                  '700': [1]*4, '710': [1]*4}).to_csv(source, index=False)
                    params = dict(view=view, energy_min=1.7, energy_max=1.8, auto_baseline=False,
                                  baseline=0., cut_type='x', cut_value=0., export_dpi=100, cmap='viridis')
                    result = analyze(load_dataset(source), params)
                    png = export_result(result, folder)[1]
                    with Image.open(png) as image:
                        self.assertEqual(image.size, (450, 305))
                        pixels = np.asarray(image.convert('RGB'))
                    # Inspect the filled data area, excluding spines and the separate colorbar.
                    middle = pixels[143, 219]
                    self.assertGreater(int(np.ptp(middle)), 20)
                    xs = np.flatnonzero((pixels[143, :375] == middle).all(axis=1))
                    ys = np.flatnonzero((pixels[:, 219] == middle).all(axis=1))
                    np.testing.assert_allclose([xs[0], xs[-1], ys[0], ys[-1]],
                                               [69, 369, 31, 256], atol=2)
                    # The color scale starts 0.05 inches from the data frame.
                    nonwhite = np.any(pixels[143, 371:389] < 180, axis=1)
                    bar = np.flatnonzero(nonwhite) + 371
                    np.testing.assert_allclose([bar[0], bar[-1]], [374, 383], atol=1)

    def test_map_and_batch_exports(self):
        with TemporaryDirectory() as d:
            p=Path(d); src=p/'sample.csv'
            pd.DataFrame({'x':[0,0,1,1], 'y':[0,1,0,1], '700':[10,11,12,13], '710':[9,10,11,12]}).to_csv(src,index=False)
            ds=load_dataset(src); params={'view':'intensity','energy_min':1.7,'energy_max':1.8,'baseline':0,'auto_baseline':False,'cut_type':'x'}
            paths=export_result(analyze(ds,params),p/'out')
            self.assertTrue(all(Path(x).is_file() for x in paths))
            from PIL import Image
            with Image.open(paths[1]) as image:
                self.assertEqual(image.width, 432)
                self.assertAlmostEqual(image.info['dpi'][0], 96, delta=0.1)
            scan_folder = p / src.stem
            compact = export_result(analyze(ds, params), scan_folder)
            self.assertEqual([Path(path).name for path in compact], ['intensity.csv', 'intensity.png', 'intensity.json'])
            repeated_compact = export_result(analyze(ds, params), scan_folder)
            self.assertEqual(compact, repeated_compact)
            changed_params = dict(params, cmap='RdBu')
            changed = export_result(analyze(ds, changed_params), scan_folder)
            self.assertTrue(all(Path(path).stem == 'intensity_002' for path in changed))
            self.assertEqual(changed, export_result(analyze(ds, changed_params), scan_folder))
            Path(compact[1]).unlink()
            self.assertEqual(changed, export_result(analyze(ds, changed_params), scan_folder))
            self.assertTrue(all(Path(f).stem == 'intensity_003' for f in export_result(analyze(ds, params), scan_folder)))
            original=Path(paths[1]).read_bytes()
            repeated=export_result(analyze(ds,params),p/'out')
            self.assertEqual(paths,repeated); self.assertEqual(original,Path(paths[1]).read_bytes())
            paths=export_batch(ds,params,(),p/'out')
            self.assertEqual(len(paths),7); self.assertTrue(all(Path(x).is_file() for x in paths))
            index = pd.read_csv(paths[0])
            self.assertEqual(index['status'].tolist(), ['completed', 'completed'])
            self.assertTrue(all(len(__import__('json').loads(value)) == 3 for value in index['exported_files']))

    def test_reflection_export_names_and_titles_describe_the_calculation(self):
        import json
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'sample.csv'
            reference = root / 'background.csv'
            source.write_text('Vbg,Vtg,700,710,720,730,740\n0,0,9,8,1,8,9\n')
            reference.write_text('700,710,720,730,740\n1,1,1,1,1\n')
            data = load_dataset(source, 'Reflection')
            for view, feature, title, stem in (
                ('rc_energy', 'auto', 'RC at 1.75 eV', 'rc_at_1.75eV'),
                ('rc_p2p', 'auto', 'RC peak-to-peak (1.67–1.78 eV)', 'rc_p2p_1.67-1.78eV'),
                ('rc_peak', 'dip', 'RC dip position, 1.67–1.78 eV', 'rc_position_dip_1.67-1.78eV'),
            ):
                with self.subTest(view=view):
                    params = dict(view=view, rc_feature=feature, energy_target=1.75, energy_min=1.67,
                                  energy_max=1.78, auto_background_scale=False, sg_window=3, sg_poly=1)
                    result = analyze(data, params, [reference])
                    self.assertEqual(result['title'], title)
                    paths = export_result(result, root / source.stem)
                    self.assertEqual([Path(path).name for path in paths],
                                     [stem + ext for ext in ('.csv', '.png', '.json')])
                    self.assertEqual(json.loads(Path(paths[2]).read_text(encoding='utf8'))['title'], title)

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
            src.write_text('Offset_SET,Current_set\nV,nA\n0,2\n1,3\n',encoding='utf8')
            ds=load_dataset(src,mode='Transport')
            result=analyze(ds,{'view':'transport_curve','x_column':'Offset_SET','channel':'Current_set'})
            self.assertEqual((result['x_label'], result['y_label']), ('Offset (V)', 'Current (nA)'))
            paths=export_result(result,p/'out')
            self.assertEqual(list(pd.read_csv(paths[0]).columns),['Offset_SET','Current_set'])
            info=json.loads(Path(paths[2]).read_text(encoding='utf8'))
            self.assertEqual(info['units'],{'Offset_SET':'V','Current_set':'nA'})

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
