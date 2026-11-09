"""Versioned, per-source Megasweep recipes, including legacy session migration."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

VERSION = 2

def identity(path):
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))

def session_path(directory, path, mode=None):
    key = identity(path) + (('::' + str(mode)) if mode else '')
    return Path(directory) / (hashlib.sha256(key.encode('utf-8')).hexdigest() + '.json')

_LEGACY = {'auto_baseline_check':'auto_baseline', 'int_min_spin':'energy_min', 'int_max_spin':'energy_max', 'baseline_spin':'baseline',
           'ratio_spin':'ratio', 'convention_combo':'convention', 'sg_window_spin':'sg_window',
           'sg_poly_spin':'sg_poly', 'bg_average_mode_combo':'background_average',
           'background_scale_check':'auto_background_scale', 'vbg_combo':'x_column', 'vtg_combo':'y_column',
           'batch_epsilon_spin':'epsilon', 'fixed_energy_spin':'energy_target', 'reflection_preview_vbg_spin':'background_scale_x', 'reflection_preview_vtg_spin':'background_scale_y', 'map_cmap_combo':'cmap', 'map_auto_scale_check':'color_auto', 'map_vmin_spin':'vmin', 'map_vmax_spin':'vmax', 'map_auto_update_check':'auto_update'}

def load_session(directory, path, mode):
    target = session_path(directory, path, mode)
    legacy = session_path(directory, path)
    if not target.exists():
        target = legacy
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or payload.get('version') not in (1, VERSION):
        raise ValueError('Unsupported Megasweep session format')
    if payload.get('csv_path') and identity(payload['csv_path']) != identity(path):
        raise ValueError('Megasweep session belongs to another source')
    if payload['version'] == 1:
        controls = payload.get('controls', {})
        if not isinstance(controls, dict):
            raise ValueError('Invalid legacy controls')
        if controls.get('mode_combo', mode).casefold() != mode.casefold():
            return {}
        values = {_LEGACY[k]: v for k, v in controls.items() if k in _LEGACY}
        average = str(values.get('background_average', 'all_frames')).casefold()
        values['background_average'] = ('first_frame' if 'first' in average else 'last_frame' if 'last' in average else 'file_equal' if 'file_equal' in average else 'all_frames')
        convention = str(values.get('convention', ''))
        if convention: values['convention'] = 'rTG+BG' if convention.startswith('D = r') or convention == 'rTG+BG' else 'TG+rBG'
        if 'rc_feature_combo' in controls: values['rc_feature'] = str(controls['rc_feature_combo']).split(':', 1)[0].strip().lower()
        notices=[]
        windows=['scale_left_min_spin','scale_left_max_spin','scale_right_min_spin','scale_right_max_spin']
        if all(key in controls for key in windows):
            values['background_scale_windows']=[[controls[windows[0]],controls[windows[1]]],[controls[windows[2]],controls[windows[3]]]]
        elif any(key in controls for key in windows): notices.append('Incomplete background scale windows were not restored.')
        label=str(controls.get('map_type_combo',''))
        mapping={'Intensity':'intensity','Peak Energy':'peak','Ibias':'ibias','Resistance':'resistance',
                 'RC Peak-to-Peak':'rc_p2p','RC at Energy':'rc_energy','RC Peak Position':'rc_peak'}
        if label in mapping: values['view']=mapping[label]
        elif label: notices.append('Unsupported legacy map type: '+label)
        values['transformed']=controls.get('map_axes_combo')=='Transformed'
        def ranges(spec):
            if not isinstance(spec,dict): return
            for axis in ('x','y'):
                item=spec.get(axis)
                if not isinstance(item,dict): continue
                if isinstance(item.get('auto'),bool): values[axis+'_auto']=item['auto']
                for bound in ('min','max'):
                    if isinstance(item.get(bound),(int,float)) and not isinstance(item[bound],bool): values[axis+bound]=item[bound]
        axis_ranges=payload.get('axis_ranges',{})
        if isinstance(axis_ranges,dict): ranges(axis_ranges.get('Transformed' if values['transformed'] else 'Original'))
        recipe=payload.get('spectral_slices')
        if isinstance(recipe,dict):
            coordinate={'Follow map':'follow_map','Original X/Y':'original','Transformed D/E':'transformed'}.get(recipe.get('coordinates'))
            if coordinate: values['cut_coordinates']=coordinate
            fixed=recipe.get('fixed')
            if fixed in ('x','y','doping','efield'):
                values['cut_type']=fixed; values['cut_axis']='first' if fixed in ('x','doping') else 'second'
            for source,dest in (('epsilon','epsilon'),('processing','derivative'),('derivative_window','derivative_window')):
                if source in recipe: values[dest]=recipe[source]
            try:
                if str(recipe.get('value','')).strip(): values['cut_value']=float(recipe['value'])
            except (TypeError,ValueError): notices.append('Invalid saved spectral slice value was not restored.')
            if payload.get('tab') == 4:
                values['view']='slice'; ranges(recipe.get('ranges'))
                colors=recipe.get('colors')
                if isinstance(colors,dict):
                    if isinstance(colors.get('auto'),bool): values['color_auto']=colors['auto']
                    for source,dest in (('min','vmin'),('max','vmax')):
                        if isinstance(colors.get(source),(int,float)) and not isinstance(colors[source],bool): values[dest]=colors[source]
            movie=recipe.get('movie')
            if isinstance(movie,dict):
                for key in ('first','last','stride'):
                    if key in movie: values['movie_'+key]=movie[key]
                for key in ('options','ranges','colors'):
                    if isinstance(movie.get(key),dict): values['movie_'+key]=movie[key]
            if payload.get('tab') != 4 and (recipe.get('colors') or recipe.get('ranges')): notices.append('Inactive spectral slice display settings were not restored.')
        transport=payload.get('transport')
        if mode=='Transport' and isinstance(transport,dict):
            for source,dest in (('x','x_column'),('y','y_column'),('channel','channel'),('direction','direction'),('pass_index','pass_index'),('cmap','cmap'),('auto_scale','color_auto'),('auto_update','auto_update')):
                if source in transport and transport[source] is not None: values[dest]=transport[source]
            values['view']='transport_curve' if transport.get('kind')=='curve' else 'transport_map'
            if transport.get('cut') and transport.get('fixed_axis'):
                values['view']='slice'; values['cut_type']=transport['fixed_axis']
                if transport.get('cut_value') is not None: values['cut_value']=transport['cut_value']
            for key,dest in (('min','vmin'),('max','vmax')):
                try: values[dest]=float(transport[key])
                except (KeyError,TypeError,ValueError): pass
            ranges(transport.get('cut_ranges') if values['view']=='slice' else transport.get('map_ranges'))
        if payload.get('line_specs'): notices.append('Legacy line-cut table was not restored; review individual cuts.')
        if payload.get('views') or payload.get('preview') or payload.get('lines'): notices.append('Saved plot views will be recomputed; multiple prior views are not restored.')
        payload = {'version': VERSION, 'values': values, 'background_paths': payload.get('background_paths', []), 'output_directory': payload.get('output_dir', ''), 'migration_notices': notices}
    elif str(payload.get('mode', mode)).casefold() != mode.casefold():
        return {}
    if not isinstance(payload.get('values', {}), dict) or not isinstance(payload.get('background_paths', []), list):
        raise ValueError('Invalid Megasweep session')
    return payload

def save_session(directory, path, mode, values, background_paths, source_root='', output_directory=''):
    target = session_path(directory, path, mode)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {'version': VERSION, 'csv_path': identity(path), 'mode': mode,
               'values': values, 'background_paths': list(background_paths), 'source_root': str(source_root), 'output_directory': str(output_directory)}
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    os.replace(temporary, target)
