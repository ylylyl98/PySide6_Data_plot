import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from core.megasweep_store import load_session, save_session, session_path

class MegasweepStoreTests(unittest.TestCase):
    def test_roundtrip_and_mode_isolation(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.csv'
            save_session(directory, source, 'PL', {'baseline': 3}, [])
            save_session(directory, source, 'Reflection', {'baseline': 7}, ['bg.csv'])
            self.assertEqual(load_session(directory, source, 'PL')['values']['baseline'], 3)
            self.assertEqual(load_session(directory, source, 'Reflection')['background_paths'], ['bg.csv'])
    def test_legacy_import_and_default_frame_weight(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.csv'
            path = session_path(directory, source)
            path.write_text(json.dumps({'version':1, 'csv_path':str(source), 'controls':{'mode_combo':'Reflection','baseline_spin':12}, 'background_paths':['bg.csv']}))
            result = load_session(directory, source, 'Reflection')
            self.assertEqual(result['values']['background_average'], 'all_frames')
            self.assertEqual(result['values']['baseline'], 12)
            self.assertEqual(load_session(directory, source, 'PL'), {})
            self.assertEqual(json.loads(path.read_text())['version'], 1)

    def test_legacy_frame_modes_and_output(self):
        for label, expected in [('Average first frame per CSV', 'first_frame'), ('Average last frame per CSV', 'last_frame')]:
            with self.subTest(label=label), TemporaryDirectory() as directory:
                source = Path(directory) / 'source.csv'
                legacy = {'version':1, 'csv_path':str(source), 'controls':{'mode_combo':'Reflection', 'bg_average_mode_combo':label, 'background_scale_check':True, 'convention_combo':'D = r·TG + BG, E = r·TG − BG'}, 'output_dir':str(Path(directory)/'output')}
                session_path(directory, source).write_text(json.dumps(legacy))
                data = load_session(directory, source, 'Reflection')
                self.assertEqual(data['values']['background_average'], expected)
                self.assertIs(data['values']['auto_background_scale'], True)
                self.assertNotIn('background_scale', data['values'])
                self.assertEqual(data['values']['convention'], 'rTG+BG')
                self.assertEqual(data['output_directory'], legacy['output_dir'])
    def test_versioned_output_root_roundtrip(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.csv'
            save_session(directory, source, 'PL', {}, [], source_root=directory, output_directory='custom')
            data = load_session(directory, source, 'PL')
            self.assertEqual(data['source_root'], directory)
            self.assertEqual(data['output_directory'], 'custom')

    def test_legacy_full_spectral_migration_and_notices(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.csv'
            legacy = {'version':1,'csv_path':str(source),'controls':{'mode_combo':'Reflection','map_type_combo':'RC at Energy','map_axes_combo':'Transformed','fixed_energy_spin':1.55,'scale_left_min_spin':1.3,'scale_left_max_spin':1.4,'scale_right_min_spin':1.7,'scale_right_max_spin':1.8},'tab':4,
                      'axis_ranges':{'Transformed':{'x':{'auto':False,'min':-1,'max':1}}},
                      'spectral_slices':{'coordinates':'Original X/Y','fixed':'y','value':'2.25','epsilon':.02,'processing':'second_derivative','derivative_window':7,'ranges':{'x':{'auto':False,'min':1.2,'max':1.8}},'colors':{'auto':False,'min':-2,'max':2},'movie':{'first':1,'last':4,'stride':2,'options':{'fps':24},'colors':{'min':-1,'max':1}}},
                      'line_specs':[{'cut_type':'doping','c_value':0,'epsilon':.1}], 'views':['fixed_original']}
            session_path(directory,source,'Reflection').write_text(json.dumps(legacy))
            result=load_session(directory,source,'Reflection'); values=result['values']
            self.assertEqual(values['view'],'slice'); self.assertEqual(values['cut_coordinates'],'original'); self.assertEqual(values['cut_axis'],'second')
            self.assertEqual(values['cut_value'],2.25); self.assertEqual(values['derivative'],'second_derivative')
            self.assertEqual(values['background_scale_windows'],[[1.3,1.4],[1.7,1.8]])
            self.assertEqual(values['movie_options']['fps'],24); self.assertEqual(values['movie_stride'],2)
            self.assertEqual((values['xmin'],values['xmax']),(1.2,1.8)); self.assertFalse(values['color_auto'])
            self.assertTrue(any('line-cut' in item for item in result['migration_notices']))
            self.assertEqual(json.loads(session_path(directory,source,'Reflection').read_text())['version'],1)

    def test_legacy_transport_migration(self):
        with TemporaryDirectory() as directory:
            source=Path(directory)/'transport.csv'
            legacy={'version':1,'csv_path':str(source),'controls':{'mode_combo':'Transport'}, 'transport':{'x':'Vbg','y':'Vtg','channel':'Current','kind':'map','direction':'forward','pass_index':2,'cut':True,'fixed_axis':'Vbg','cut_value':1.5,'map_ranges':{},'cut_ranges':{'x':{'auto':False,'min':0,'max':3}},'auto_scale':False,'min':'-1','max':'2'}}
            session_path(directory,source,'Transport').write_text(json.dumps(legacy))
            v=load_session(directory,source,'Transport')['values']
            self.assertEqual((v['x_column'],v['y_column'],v['channel']),('Vbg','Vtg','Current'))
            self.assertEqual((v['view'],v['cut_type'],v['cut_value']),('slice','Vbg',1.5))
            self.assertEqual((v['direction'],v['pass_index']),('forward',2)); self.assertEqual((v['vmin'],v['vmax']),(-1,2))
            self.assertFalse(v['x_auto']); self.assertEqual((v['xmin'],v['xmax']),(0,3))
