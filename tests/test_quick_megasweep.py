import os
import subprocess
import sys
import unittest

class MegasweepQuickTests(unittest.TestCase):
    def test_modes_session_and_async_plot(self):
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
        completed = subprocess.run([sys.executable, '-c', RUNNER], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

RUNNER = r'''
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtCore import QSettings, QUrl
from PySide6.QtGui import QGuiApplication
from ui_qt.quick.megasweep_bridge import MegasweepBridge
app = QGuiApplication([])
def wait_for(predicate, seconds=15):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        if predicate(): return True
        time.sleep(.01)
    return False
with TemporaryDirectory() as directory:
    root = Path(directory).resolve(); source = root / 'sample.csv'
    source.write_text('x_set,y_set,700,710\n0,0,10,9\n0,1,11,10\n1,0,12,11\n1,1,13,12\n')
    settings = QSettings(str(root / 'test.ini'), QSettings.IniFormat)
    bridge = MegasweepBridge(settings=settings, session_directory=root / 'sessions')
    try:
        bridge.setExperimentFolder(str(root)); bridge.selectFiles([QUrl.fromLocalFile(str(source))])
        bridge.setParameter('auto_baseline', False); bridge.setParameter('baseline', 0)
        bridge.setParameter('energy_min', 1.7); bridge.setParameter('energy_max', 1.8)
        assert wait_for(lambda: bool(bridge.snapshot['canExport'])), bridge.snapshot['error']
        assert bridge.plotBackend is not None
        bridge.setParameter('mode', 'Reflection'); assert bridge.snapshot['sourceFile'] == ''
        # Reflection initializes the measured range before requiring a reference.
        import numpy as np
        bridge.selectFiles([str(source)])
        assert wait_for(lambda: bool(bridge.snapshot['error'])), bridge.snapshot
        measured = (1240 / 710, 1240 / 700)
        def energy_window():
            values = bridge.snapshot['values']
            return (values['energy_min'], values['energy_max'])
        assert np.allclose(energy_window(), measured)
        def axis_label(key):
            fields = bridge.snapshot['parameterGroups'][0]['fields']
            field = next(f for f in fields if f['key'] == key)
            return next(o['label'] for o in field['options'] if o['value'] == field['value'])
        assert axis_label('x_column') == 'x_set'
        assert axis_label('y_column') == 'y_set'
        groups = {g['key']: g for g in bridge.snapshot['parameterGroups']}
        def gate_fields():
            return {f['key']: f for group in bridge.snapshot['parameterGroups']
                    if group['key'] == 'coordinates' for f in group['fields']}
        assert set(gate_fields()) == {'transformed'}
        bridge.setParameter('transformed', True)
        assert [o['label'] for o in gate_fields()['tg_is_y']['options']] == ['Y (y)', 'X (x)']
        bridge.setParameter('convention', 'rTG+BG')
        definitions = gate_fields()['convention']
        assert definitions['value'] == 'rTG+BG' and len(definitions['options']) == 2
        assert [item['label'].splitlines() for item in definitions['options']] == [
            ['D = TG + r × BG', 'E = TG − r × BG'], ['D = r × TG + BG', 'E = r × TG − BG']]
        assert set(gate_fields()) == {'transformed', 'tg_is_y', 'convention', 'ratio'}
        bridge.setParameter('transformed', False)
        bridge.setParameter('convention', 'TG+rBG')
        assert groups['processing']['title'] == 'Energy window'
        assert {f['key'] for f in groups['processing']['fields']} == {'auto_energy_window', 'energy_min', 'energy_max'}
        assert list(groups) == ['data', 'coordinates', 'processing']
        reference_groups = {g['key']: g for g in bridge.snapshot['referenceParameterGroups']}
        assert {f['key'] for f in reference_groups['background_normalization']['fields']} == {'auto_background_scale', 'background_average', 'background_scale'}
        for view, expected in [('rc_energy', {'auto_energy_target', 'energy_target'}), ('rc_peak', {'auto_energy_window', 'energy_min', 'energy_max'}), ('ibias', set())]:
            bridge.setParameter('view', view)
            current = {g['key']: g for g in bridge.snapshot['parameterGroups']}
            assert {f['key'] for f in current.get('processing', {}).get('fields', [])} == expected
            assert ('peak_detection' in current) == (view == 'rc_peak')
        bridge.setParameter('view', 'rc_p2p')
        for field in bridge.snapshot['parameterGroups'][0]['fields']:
            if field['key'] in ('x_column', 'y_column'):
                assert [item['value'] for item in field['options']] == ['', 'x_set', 'y_set']
        bridge.setParameter('x_column', 'x_set')
        assert axis_label('x_column') == 'x_set'
        assert bridge.snapshot['values']['x_column'] == 'x_set'
        bridge.setParameter('x_column', '')
        assert bridge.snapshot['values']['x_column'] == '' and axis_label('x_column') == 'x_set'
        bridge.setParameter('auto_background_scale', False)
        assert not bridge.snapshot['values']['auto_background_scale']
        bridge.openFiles('background', [str(source)])
        assert wait_for(lambda: bridge.snapshot['reflectionPreviewVisible']), bridge.snapshot['error']
        assert bridge._reflection_preview['scale'] == 1.0
        bridge.invoke('return-to-map')
        assert wait_for(lambda: bridge.snapshot['canExport']), bridge.snapshot['error']
        bridge.setParameter('view', 'rc_energy')
        assert wait_for(lambda: bridge.snapshot['canExport']), bridge.snapshot['error']
        assert bridge._result['parameters']['energy_target'] in measured
        bridge.setParameter('energy_target', 1.76)
        assert wait_for(lambda: bridge.snapshot['canExport']), bridge.snapshot['error']
        assert not bridge.snapshot['values']['auto_energy_target']
        assert bridge.snapshot['values']['cmap'] == 'RdBu_r'
        limits = bridge.mapLimits()
        assert len(limits) == 3 and limits[-1]['colormaps'] == ['RdBu_r', 'RdBu']
        bridge.setParameter('cmap', 'RdBu')
        assert bridge.mapLimits()[-1]['cmap'] == 'RdBu'
        assert bridge._canvas.figure.axes[0].collections[0].get_cmap().name == 'RdBu'
        bridge.setParameter('cmap', 'RdBu_r')
        for row, bounds in zip(limits, [(-1, 3), (-2, 2), (-0.5, 0.5)]):
            row.update(automatic=False, low=bounds[0], high=bounds[1])
        assert bridge.applyMapLimits(limits) == ''
        assert tuple(bridge._canvas.figure.axes[0].collections[0].get_clim()) == (-0.5, 0.5)
        assert tuple(bridge._canvas.figure.axes[0].get_ylim()) == (-2, 2)
        assert bridge.applyMapLimits([dict(automatic=False, low=1, high=0)] * 3)
        from core.megasweep.megasweep_analysis import resolve_color_limits
        axis = bridge._canvas.figure.axes[0]
        assert (axis.get_xlabel(), axis.get_ylabel()) == ('x', 'y')
        axis.set_xlim(-.25, .75)
        bridge.setParameter('color_auto', True)
        assert axis.collections[0].get_clim() == resolve_color_limits(bridge._result['z'])
        assert axis.get_xlim() == (-.25, .75) and bridge.mapLimits()[2]['automatic']
        bridge.setParameter('color_auto', False)
        assert (bridge.snapshot['values']['vmin'], bridge.snapshot['values']['vmax']) == axis.collections[0].get_clim()
        axis.set_xlim(-1, 3)
        bridge.setParameter('x_reverse', True)
        assert bridge._canvas.figure.axes[0].get_xlim() == (3, -1)
        assert bridge.mapLimits()[0]['reversed'] and not bridge.mapLimits()[1]['reversed']
        bridge.invoke('reflection-preview')
        assert wait_for(lambda: bridge.snapshot['reflectionPreviewVisible']), bridge.snapshot['error']
        scale = bridge._reflection_preview['scale']
        point = bridge.snapshot['referencePoint']
        assert (point['xName'], point['yName']) == ('x', 'y')
        bridge.setReferenceCoordinate('x', 2.5)
        assert bridge.snapshot['referenceFrame'] == 2
        bridge.setReferenceCoordinate('y', .7)
        bridge.setReferenceCoordinate('x', float('nan'))
        assert (bridge.snapshot['referencePoint']['x'], bridge.snapshot['referencePoint']['y']) == (1., 1.)
        assert bridge._canvas.figure._suptitle.get_text() == 'RC at 1.76 eV · Sample spectrum · x = 1, y = 1'
        assert bridge.snapshot['referenceFrame'] == 3 and bridge._reflection_preview['scale'] == scale
        assert any('1.76' in line.get_label() for line in bridge._canvas.figure.axes[1].lines)
        bridge.setParameter('view', 'rc_p2p')
        assert wait_for(lambda: bridge.snapshot['reflectionPreviewVisible']), bridge.snapshot['error']
        assert bridge.snapshot['referenceFrame'] == 3
        assert 'Peak-to-peak RC' in bridge._canvas.figure.axes[1].get_title()
        bridge.invoke('show-spectral-cut')
        assert wait_for(lambda: bridge.snapshot['canExport']), bridge.snapshot['error']
        assert bridge._result['kind'] == 'slice'
        bridge.invoke('return-to-map')
        assert wait_for(lambda: bridge.snapshot['canExport']), bridge.snapshot['error']
        assert bridge.snapshot['values']['view'] == 'rc_p2p'
        bridge.setParameter('view', 'rc_p2p')
        bridge.setParameter('energy_min', 1.76)
        assert wait_for(lambda: bridge.snapshot['canExport']), bridge.snapshot['error']
        assert not bridge.snapshot['values']['auto_energy_window']
        assert bridge._result['parameters']['energy_min'] == 1.76
        other_scan = root / 'other-band.csv'
        other_scan.write_text(source.read_text().replace('700,710', '910,920'))
        bridge.selectFiles([str(other_scan)])
        assert wait_for(lambda: bool(bridge.snapshot['error'])), bridge.snapshot
        assert np.allclose(energy_window(), (1240 / 920, 1240 / 910))
        bridge.selectFiles([str(source)])
        assert wait_for(lambda: bridge.snapshot['canExport']), bridge.snapshot['error']
        assert energy_window()[0] == 1.76
        bridge.setParameter('auto_energy_window', True)
        assert wait_for(lambda: bridge.snapshot['canExport']), bridge.snapshot['error']
        assert np.allclose(energy_window(), measured)
        bridge.setParameter('mode', 'PL'); assert bridge.snapshot['values']['baseline'] == 0
        assert wait_for(lambda: bool(bridge.snapshot['canExport'])), bridge.snapshot['error']
        bridge.setParameter('movie.fps', 24); assert bridge.snapshot['movieOptions']['fps'] == 24
        bridge.setParameter('movie.fps', 24.5); assert bridge.snapshot['movieOptions']['fps'] == 24
        assert bridge.snapshot['movieError']
        bridge._busy = True; bridge._operation = 'export'
        bridge.setParameter('movie.fps', 30); assert bridge.snapshot['movieOptions']['fps'] == 24
        bridge._busy = False; bridge._operation = ''
        bridge.setParameter('auto_update', False); bridge.setParameter('baseline', 5)
        assert not bridge.snapshot['canExport']
        bridge.invoke('update'); assert wait_for(lambda: bool(bridge.snapshot['canExport'])), bridge.snapshot['error']
        transport = root / 'transport.csv'
        transport.write_text('Vbg,Vtg,current\n2,4,10\n2,5,11\n3,4,12\n3,5,13\n')
        bridge.setParameter('mode', 'Transport'); bridge.selectFiles([str(transport)])
        assert wait_for(lambda: bool(bridge.snapshot['canExport'])), bridge.snapshot['error']
        assert bridge.snapshot['values']['cut_type'] in ('Vbg','Vtg')
        assert bridge.snapshot['values']['cut_value'] in (2.,3.,4.,5.)
        bridge.setParameter('pass_index', 1); bridge.setParameter('pass_index', ''); assert bridge.snapshot['values']['pass_index'] is None
        bridge.setParameter('mode', 'PL'); bridge.setParameter('auto_update', True)
        assert wait_for(lambda: bool(bridge.snapshot['canExport'])), bridge.snapshot['error']
        import core.megasweep.workflow as workflow
        original = workflow.analyze
        def slow(dataset, parameters, background_paths=(), cancelled=None):
            if parameters['baseline'] == 6: time.sleep(.5)
            return original(dataset, parameters, background_paths, cancelled)
        workflow.analyze = slow
        bridge.setParameter('baseline', 6); assert wait_for(lambda: bridge.snapshot['busy'])
        bridge.setParameter('baseline', 7)
        assert wait_for(lambda: bool(bridge.snapshot['canExport']) and bridge._result['parameters']['baseline'] == 7), bridge.snapshot['error']
        time.sleep(.6); app.processEvents()
        assert bridge._result['parameters']['baseline'] == 7
        workflow.analyze = original
        # Reference spectra are a distinct inspection state: never export the hidden map.
        import numpy as np
        preview = dict(energy=np.array([1.,2.]), raw=np.array([2.,3.]), background=np.array([1.,1.]), scaled_background=np.array([1.,1.]), rc=np.array([1.,2.]), frame=0)
        bridge._finish(bridge._sequence, 'reflection-preview', preview, '', 0.)
        assert bridge.snapshot['reflectionPreviewVisible'] and not bridge.snapshot['canExport']
        bridge.invoke('return-to-map')
        assert not bridge.snapshot['reflectionPreviewVisible'] and bridge.snapshot['canExport']
        # Keep the export receipt and the background chooser alive through shortcuts.
        import threading
        gate = threading.Event(); original_export = workflow.export_result; exported_parameters = {}
        bridge._canvas.figure.axes[0].set_xlim(-.25, .75)
        bridge._canvas.figure.axes[0].set_ylim(-.5, .5)
        bridge.setParameter('y_reverse', True)
        assert bridge._canvas.figure.axes[0].get_ylim() == (.5, -.5)
        def held_export(result, directory, *, parameters=None):
            exported_parameters.update(parameters)
            assert gate.wait(5)
            output = root / 'receipt.csv'; output.write_text('saved')
            return [str(output)]
        workflow.export_result = held_export
        bridge.invoke('export'); assert bridge.snapshot['busy']
        seq = bridge._sequence; before_source = bridge.snapshot['sourceFile']; before_mode = bridge.snapshot['mode']
        for action in ('update','reload','choose-background','clear-background','slice-next','reflection-preview','movie-preview'):
            bridge.invoke(action)
        bridge.setParameter('mode', 'Reflection'); bridge.openFiles('background', [str(source)]); bridge.selectFiles([str(transport)])
        assert bridge._sequence == seq and bridge.snapshot['busy'] and bridge.snapshot['sourceFile'] == before_source and bridge.snapshot['mode'] == before_mode
        assert not bridge._background._editing_enabled
        gate.set(); assert wait_for(lambda: not bridge.snapshot['busy'])
        assert bridge._background._editing_enabled and bridge.snapshot['resultPaths'] == [str(root / 'receipt.csv')]
        assert not exported_parameters['x_auto'] and not exported_parameters['y_auto']
        assert (exported_parameters['xmin'], exported_parameters['xmax']) == (-.25, .75)
        assert (exported_parameters['ymin'], exported_parameters['ymax']) == (-.5, .5)
        workflow.export_result = original_export
        bridge.invoke('export'); assert wait_for(lambda: not bridge.snapshot['busy'])
        import json
        paths = [Path(p) for p in bridge.snapshot['resultPaths']]
        assert any(p.suffix == '.png' and p.is_file() for p in paths)
        metadata = json.loads(next(p for p in paths if p.suffix == '.json').read_text(encoding='utf8'))
        exported = metadata['parameters']
        assert exported['y_reverse'] and not exported['x_reverse']
        assert not exported['x_auto'] and (exported['xmin'], exported['xmax']) == (-.25, .75)
        assert not exported['y_auto'] and (exported['ymin'], exported['ymax']) == (-.5, .5)
        # A slide edit restores the exported recipe and operates on the original scan.
        from core.pptx_figure_index import FigureRecord
        png = next(p for p in paths if p.suffix == '.png')
        record = FigureRecord(png, png.with_suffix('.csv'), 'megasweep', 'map', {}, metadata, 'example')
        before_edit = bridge.snapshot['values']
        bridge.open_saved_figure(record)
        assert wait_for(lambda: bridge.snapshot['canExport']), bridge.snapshot['error']
        assert bridge.snapshot['sourceFile'] == metadata['source_path']
        assert bridge.snapshot['values']['baseline'] == exported['baseline']
        bridge.setParameter('baseline', 3.)
        assert wait_for(lambda: bridge.snapshot['canExport']), bridge.snapshot['error']
        assert not np.array_equal(bridge._result['z'], np.zeros_like(bridge._result['z']))
        revised = [Path(p) for p in bridge.saved_figure_export_request()()]
        revised_meta = json.loads(revised[2].read_text(encoding='utf8'))
        assert revised[1].parent == png.parent and revised[1] != png
        assert revised_meta['parameters']['baseline'] == 3.
        assert revised[0].read_bytes() != record.dat_path.read_bytes()
        assert metadata == json.loads(record.png_path.with_suffix('.json').read_text(encoding='utf8'))
        bridge.close_saved_figure()
        assert wait_for(lambda: bridge.snapshot['canExport']), bridge.snapshot['error']
        assert bridge.snapshot['values'] == before_edit
        named = root / 'gates.csv'
        named.write_text(source.read_text().replace('x_set,y_set,', 'Vbg_set,Vtg_set,'))
        bridge.selectFiles([str(named)])
        bridge.setParameter('tg_is_y', False)
        bridge.setParameter('transformed', True)
        assert wait_for(lambda: 'convention' in gate_fields()), bridge.snapshot['error']
        assert 'tg_is_y' not in gate_fields()
        # A source outside the browser root exports beside its own experiment.
        browse = root / 'browse'; external = root / 'external'; browse.mkdir(); external.mkdir()
        other = external / 'other.csv'; other.write_text(source.read_text())
        bridge.setExperimentFolder(str(browse)); bridge.selectFiles([str(other)])
        assert bridge._state['source_root'] == str(external)
        from core.megasweep_store import load_session
        assert load_session(root / 'sessions', other, 'PL')['source_root'] == str(external)
    finally: bridge.close()
'''
