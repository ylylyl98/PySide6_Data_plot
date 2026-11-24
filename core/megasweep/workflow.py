"""UI-independent Megasweep workflow; numerical routines originate at 2612f6f."""
from pathlib import Path
import numpy as np
from . import DEFAULT_EXPORT_DPI, LEGACY_SCALE_WINDOWS, has_legacy_scale_windows
from . import megasweep_analysis as a
from . import transport_analysis as t
from .spectral_processing import process_spectral_slice


def default_parameters():
    return dict(view='intensity', energy_min=1.247, energy_max=1.428, energy_target=1.3375,
                baseline=595.0, auto_baseline=True, sg_window=31, sg_poly=3, ratio=1.0, tg_is_y=True,
                convention='TG+rBG', transformed=False, cut_type='doping', cut_coordinates='explicit', cut_axis='first', batch_axes='selected', cut_value=0.0,
                epsilon=0.03, min_points=2, exact_coordinates=False, derivative='spectrum',
                derivative_window=9, background_role='custom', background_groups={}, background_average='all_frames', background_scale=1.0, auto_background_scale=False, background_scale_frame=None, background_scale_x=0.0, background_scale_y=0.0, auto_scale_windows=None, background_scale_windows=list(LEGACY_SCALE_WINDOWS),
                x_column='', y_column='', channel='', direction=None, pass_index=None,
                rc_feature='auto', cmap='RdBu_r', color_auto=True, vmin=0.0, vmax=1.0, x_auto=True, x_reverse=False, xmin=0.0, xmax=1.0, y_auto=True, y_reverse=False, ymin=0.0, ymax=1.0, export_dpi=DEFAULT_EXPORT_DPI, movie_options={}, movie_ranges={}, movie_colors={})


def _top_gate_axis(x_name, y_name):
    """Resolve the orientation for a requested gate transform of these scan axes."""
    x_role, y_role = a.classify_axis_role(x_name), a.classify_axis_role(y_name)
    if x_role == y_role == 'unknown':
        return 'manual'
    if x_role in ('bg', 'unknown') and y_role in ('tg', 'unknown'):
        return 'y'
    if x_role in ('tg', 'unknown') and y_role in ('bg', 'unknown'):
        return 'x'
    return None


def _params(parameters, dataset=None):
    p = default_parameters(); p.update(parameters or {})
    if dataset is not None and dataset.get('top_gate_axis') in ('x', 'y'):
        p['tg_is_y'] = dataset['top_gate_axis'] == 'y'
    if p['auto_scale_windows'] is None:
        p['auto_scale_windows'] = has_legacy_scale_windows(p['background_scale_windows'])
    coordinates=p['cut_coordinates']
    if coordinates != 'explicit':
        if coordinates not in ('follow_map','original','transformed'):
            raise ValueError('Unknown cut coordinates: '+str(coordinates))
        transformed=p['transformed'] if coordinates=='follow_map' else coordinates=='transformed'
        if p['cut_axis'] not in ('first','second'): raise ValueError('Unknown cut axis')
        p['cut_type']=('doping' if p['cut_axis']=='first' else 'efield') if transformed else ('x' if p['cut_axis']=='first' else 'y')
    return p


def _check(cancelled):
    if cancelled and cancelled():
        raise InterruptedError('Megasweep operation cancelled')


def load_dataset(path, mode='PL', x_col=None, y_col=None):
    if mode not in ('PL', 'Reflection', 'Transport'):
        raise ValueError('Unknown measurement mode: ' + str(mode))
    if mode == 'Transport':
        data=t.load_transport_csv(path)
    else:
        if x_col is None or y_col is None:
            import pandas as pd
            columns=list(pd.read_csv(path,nrows=0).columns)
            metadata,_=a._split_numeric_spectral_columns(columns)
            gx,gy=a.guess_sweep_axis_columns(metadata or columns[:2])
            x_col=x_col or gx; y_col=y_col or gy
        status=a.validate_axis_selection(x_col,y_col)
        if not status['ok']: raise ValueError(status['message'])
        data=a.load_megasweep_csv(path,x_col,y_col)
        data['axis_validation']=status
        data['top_gate_axis'] = _top_gate_axis(x_col, y_col)
    data['mode'] = mode
    data['source_path'] = str(Path(path).resolve())
    return data


def load_background(paths, wavelength, average='all_frames', cancelled=None, *, role='custom'):
    from .background_roles import inspect_background_roles
    inspect_background_roles(paths,role)
    if not paths: raise ValueError('No background CSV files were selected.')
    if average not in ('all_frames','first_frame','last_frame','file_equal'):
        raise ValueError('Unknown background averaging mode: ' + str(average))
    selected=[]
    for path in paths:
        _check(cancelled)
        data=a.load_spectral_csv(str(path))
        if not a.spectral_axes_match(data['wavelength'], wavelength):
            raise ValueError(f'Background spectral channels do not match: {path}')
        frames=np.asarray(data['Intensity'],float)
        if not len(frames): raise ValueError(f'No usable spectra in background: {path}')
        if average == 'first_frame': frames=frames[:1]
        elif average == 'last_frame': frames=frames[-1:]
        elif average == 'file_equal': frames=np.mean(frames,axis=0,keepdims=True)
        selected.append(frames)
    return np.mean(np.vstack(selected),axis=0)


def _cut_axis_space(data):
    if data.get('top_gate_axis') is not None:
        return 'gate'
    return data.get('axis_space', 'gate')


def _cut(data,p,spectra):
    return a.extract_line_cut(None,data['x_data'],data['y_data'],data['energy'],
        cut_type=p['cut_type'],c_value=float(p['cut_value']),ratio=float(p['ratio']),
        epsilon=float(p['epsilon']),tg_is_y=bool(p['tg_is_y']),convention=p['convention'],
        axis_space=_cut_axis_space(data),x_axis_name=data['x_name'],y_axis_name=data['y_name'],
        exact_coordinates=bool(p['exact_coordinates']),spectra=spectra)


def slice_values(dataset,parameters):
    p=_params(parameters, dataset)
    if dataset['mode']=='Transport':
        col=p['cut_type'] if p['cut_type'] in dataset['columns'] else p['y_column']
        return sorted(float(x) for x in dataset['df'][col].dropna().unique())
    return a.find_all_cut_values(dataset['x_data'],dataset['y_data'],p['cut_type'],float(p['ratio']),float(p['epsilon']),
        min_points=int(p['min_points']),tg_is_y=bool(p['tg_is_y']),convention=p['convention'],
        axis_space=_cut_axis_space(dataset),x_axis_name=dataset['x_name'],y_axis_name=dataset['y_name'],
        exact_coordinates=bool(p['exact_coordinates']))


def rc_map_labels(parameters):
    """Describe the actual RC calculation for plot titles and compact filenames."""
    def energy(key):
        return f"{float(parameters[key]):.6f}".rstrip('0').rstrip('.')

    view = parameters['view']
    if view == 'rc_energy':
        target = energy('energy_target')
        return f'RC at {target} eV', f'rc_at_{target}eV'
    low, high = energy('energy_min'), energy('energy_max')
    if view == 'rc_p2p':
        return f'RC peak-to-peak ({low}–{high} eV)', f'rc_p2p_{low}-{high}eV'
    feature = str(parameters['rc_feature']).strip().lower()
    description = {'peak': 'peak position', 'dip': 'dip position', 'auto': 'feature position (auto)'}[feature]
    return f'RC {description}, {low}–{high} eV', f'rc_position_{feature}_{low}-{high}eV'


def _estimate_reference_scale(p, sample, background, energy):
    windows = None if p['auto_scale_windows'] else p['background_scale_windows']
    scale, info = a.estimate_background_scale(sample, background, energy, windows)
    p['background_scale_windows'] = info['windows']
    return scale, info


def analyze(dataset,parameters,background_paths=(),cancelled=None):
    p=_params(parameters, dataset); _check(cancelled); mode=dataset['mode']; view=p['view']
    base=dict(source_path=dataset['source_path'],mode=mode,background_paths=[str(Path(v).resolve()) for v in background_paths],diagnostics={},title=view,
              x_label='',y_label='',signal_label='',parameters=p)
    if mode=='Transport':
        x=p['x_column']; y=p['y_column']; channel=p['channel']
        units=dataset.get('units',{})
        def label(name, *, axis=False):
            unit=str(units.get(name,'')).strip()
            text=a.display_axis_name(name) if axis else name
            return f'{text} ({unit})' if unit else text
        base['units']=dict(units)
        signal_label=label(channel,axis=view in ('transport_curve','slice'))
        kw=dict(direction=p['direction'] or None,pass_index=p['pass_index'])
        if view=='transport_curve':
            raw=t.transport_curve(dataset,x,channel,**kw); rows=raw['samples']
            return dict(base,kind='curve',x=rows[x].to_numpy(),values=rows[channel].to_numpy(),x_label=label(x,axis=True),y_label=signal_label,signal_label=signal_label,data=raw)
        raw=t.transport_map(dataset,x,y,channel,**kw)
        if view=='slice':
            cut=t.transport_cut(raw,p['cut_type'],p['cut_value'])
            return dict(base,kind='curve',x=cut['coordinates'],values=cut['values'],x_label=label(cut['axis'],axis=True),y_label=signal_label,signal_label=signal_label,data=cut)
        return dict(base,kind='map',x=raw['X2D'],y=raw['Y2D'],z=raw['Z2D'],x_label=label(x,axis=True),y_label=label(y,axis=True),signal_label=signal_label,data=raw)
    spectra=dataset['Intensity']; diagnostics={'background_average':p['background_average']} if mode=='Reflection' else {}
    if mode=='Reflection' and view not in ('ibias','resistance'):
        from .background_roles import inspect_background_roles
        provenance=inspect_background_roles(background_paths,p['background_role'])
        diagnostics['background_role']=provenance['role']; diagnostics['background_sources']=provenance['sources']
        base['background_role']=provenance['role']; base['background_sources']=provenance['sources']
        bg=load_background(background_paths,dataset['wavelength'],p['background_average'],cancelled,role=p['background_role'])
        scale=p['background_scale']
        if p['auto_background_scale']:
            if p['background_scale_frame'] is None:
                xs=np.asarray(dataset['x_data'],float); ys=np.asarray(dataset['y_data'],float)
                valid=np.isfinite(xs)&np.isfinite(ys)
                if not np.any(valid): raise ValueError('No finite reflection sweep coordinates')
                indices=np.flatnonzero(valid)
                frame=int(indices[np.argmin((xs[valid]-float(p['background_scale_x']))**2+(ys[valid]-float(p['background_scale_y']))**2)])
            else: frame=int(p['background_scale_frame'])
            if frame < 0 or frame >= len(spectra): raise ValueError('Background scale frame is out of range')
            scale,info=_estimate_reference_scale(p, spectra[frame], bg, dataset['energy'])
            diagnostics.update(background_scale=scale,background_scale_info=info,background_scale_frame=frame)
        diagnostics['background_scale']=float(scale)
        spectra=a.compute_rc_spectra(spectra,bg,scale)
    _check(cancelled)
    if view=='slice':
        cut=_cut(dataset,p,spectra)
        if not len(cut['axis_values']): raise ValueError('No measured points match this slice coordinate.')
        cut=process_spectral_slice(cut,p['derivative'],int(p['derivative_window']),is_rc=mode=='Reflection')
        x,y=np.meshgrid(cut['energy'],cut['axis_values'])
        return dict(base,kind='slice',x=x,y=y,z=cut['spectra'],x_label='Energy (eV)',y_label=a.display_axis_name(cut['axis_label']),signal_label=cut['signal_label'],diagnostics=diagnostics,data=cut)
    lo=float(p['energy_min']); hi=float(p['energy_max']); energy=dataset['energy']
    if view=='intensity':
        baseline=p['baseline']
        if p['auto_baseline']:
            baseline,info=a.estimate_global_baseline(dataset['Intensity'],energy,lo,hi,return_info=True)
            diagnostics.update(baseline=baseline,baseline_info=info)
        diagnostics['baseline']=float(baseline)
        values=a.compute_intensity_map(dataset,lo,hi,baseline); label='PL intensity'
    elif view=='peak': values=a.compute_peak_energy_map(dataset,int(p['sg_window']),int(p['sg_poly'])); label='Peak energy (eV)'
    elif view=='rc_p2p': values=a.compute_rc_peak_to_peak_map(spectra,energy,lo,hi); label='RC peak-to-peak'
    elif view=='rc_energy': values=a.compute_rc_at_energy_map(spectra,energy,float(p['energy_target'])); label='RC'
    elif view=='rc_peak': values=a.compute_rc_peak_position_map(spectra,energy,lo,hi,int(p['sg_window']),int(p['sg_poly']),p['rc_feature']); label='RC peak position (eV)'
    elif view in ('ibias','resistance'):
        current=dataset.get('ibias_data'); voltage=dataset.get('vbias_data')
        if current is None: raise ValueError('Dataset has no Ibias column')
        if view=='resistance':
            if voltage is None: raise ValueError('Dataset has no Vbias column')
            with np.errstate(divide='ignore',invalid='ignore'): values=np.where(current!=0.0,voltage/current,np.nan)
            label='Resistance'
        else: values=current; label='Ibias'
    else: raise ValueError('Unknown analysis view: '+str(view))
    if view in ('rc_p2p', 'rc_energy', 'rc_peak'):
        base['title'], base['export_label'] = rc_map_labels(p)
    x,y=dataset['x_data'],dataset['y_data']; xl=dataset['x_name']; yl=dataset['y_name']
    raw=a.build_map_payload(x,y,values)
    if p['transformed']:
        if dataset.get('axis_space') == 'transformed':
            raise ValueError('Input axes are already transformed.')
        if 'top_gate_axis' in dataset and dataset['top_gate_axis'] is None:
            raise ValueError('Select two gate-voltage columns to transform the map.')
        tx,ty=a.compute_transformed_coords(x,y,p['ratio'],p['tg_is_y'],p['convention'])
        raw['x_flat']=np.asarray(tx)[raw['idx_flat']]
        raw['y_flat']=np.asarray(ty)[raw['idx_flat']]
        if np.asarray(raw['X2D']).ndim == 2:
            raw['X2D'],raw['Y2D']=a.compute_transformed_coords(raw['X2D'],raw['Y2D'],p['ratio'],p['tg_is_y'],p['convention'])
        else:
            raw['X2D'],raw['Y2D']=raw['x_flat'],raw['y_flat']
        xl='Doping'; yl='Electric field'
    raw.update(x_name=xl,y_name=yl)
    return dict(base,kind='map',x=raw['X2D'],y=raw['Y2D'],z=raw['Z2D'],x_label=a.display_axis_name(xl),y_label=a.display_axis_name(yl),signal_label=label,diagnostics=dict(diagnostics,grid_kind=raw['grid_kind'],missing_count=raw['missing_count']),data=raw)


_export_lock = __import__('threading').RLock()


def _json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(f'Cannot encode {type(value).__name__}')


def _export_stem(directory, source_path, label):
    source_stem = Path(source_path).stem
    # The desktop workflow already groups artifacts in a folder named for the scan.
    if Path(directory).name.casefold() == source_stem.casefold():
        return label
    return f'{source_stem}_{label}'


def _save_result_png(fig, ax, path, dpi):
    """Keep the data rectangle aligned when exported pictures have the same width."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas_width, canvas_height = 4.5, 3.05
    left, bottom, plot_width, plot_height = 0.69, 0.49, 3.0, 2.25

    def rectangle(x, width):
        return (x / canvas_width, bottom / canvas_height,
                width / canvas_width, plot_height / canvas_height)

    fig.set_layout_engine('none')
    fig.set_size_inches(canvas_width, canvas_height, forward=False)
    ax.set_position(rectangle(left, plot_width))
    ax.set_title(ax.get_title(), fontsize=9, pad=4)
    for item in fig.axes:
        item.tick_params(axis='both', labelsize=8, pad=2, length=3)
        for axis in (item.xaxis, item.yaxis):
            axis.label.set_fontsize(9)
            axis.labelpad = 3
            axis.get_offset_text().set_fontsize(8)
    if len(fig.axes) > 1:
        colorbar = fig.axes[1]
        colorbar.set_box_aspect(None)
        colorbar.set_aspect('auto')
        colorbar.set_position(rectangle(left + plot_width + 0.05, 0.09))
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=dpi, bbox_inches=fig.bbox_inches, pad_inches=0)


def export_result(result,directory,*,parameters=None):
    """Stage and publish CSV, PNG and provenance as a non-overwriting group."""
    import filecmp, json, os, re, shutil, tempfile
    import pandas as pd
    from matplotlib.figure import Figure
    p=_params(parameters if parameters is not None else result.get('parameters'))
    # Export can override display settings; record the transform actually used for this result.
    for key in ('transformed', 'tg_is_y', 'ratio', 'convention', 'auto_scale_windows', 'background_scale_windows'):
        if key in result.get('parameters', {}):
            p[key] = result['parameters'][key]
    def limits(auto,lo,hi,label):
        if auto: return None
        lo=float(lo); hi=float(hi)
        if not np.isfinite([lo,hi]).all() or lo >= hi: raise ValueError(f'{label} range requires finite min < max')
        return (lo,hi)
    clim=limits(p['color_auto'],p['vmin'],p['vmax'],'Color')
    xlim=limits(p['x_auto'],p['xmin'],p['xmax'],'X')
    ylim=limits(p['y_auto'],p['ymin'],p['ymax'],'Y')
    dpi=int(p['export_dpi'])
    if dpi <= 0 or dpi > 2400: raise ValueError('Export DPI must be between 1 and 2400')
    folder=Path(directory); folder.mkdir(parents=True,exist_ok=True)
    label = str(result.get('export_label', result['title'])).replace('/', '_').replace('\\', '_')
    stem = _export_stem(folder, result['source_path'], label)
    with tempfile.TemporaryDirectory(prefix='.megasweep-export-',dir=folder) as temporary:
        stage=Path(temporary); csv=stage/'output.csv'; png=stage/'output.png'; metadata=stage/'output.json'
        if result['kind']=='slice':
            a.save_line_csv(result['data'],str(csv))
            fig,ax=a.plot_line_cut_spectrogram(result['data'],title=result['title'],z_label=result['signal_label'],cmap=p['cmap'],vmin=clim[0] if clim else None,vmax=clim[1] if clim else None,ylim=ylim)
        elif result['kind']=='curve':
            pd.DataFrame({result['data'].get('x_name',result['data'].get('axis',result['x_label'])):result['x'],result['data'].get('channel',result['signal_label']):result['values']}).to_csv(csv,index=False)
            fig=Figure(); ax=fig.add_subplot(111); ax.plot(result['x'],result['values']); ax.set_xlabel(result['x_label']); ax.set_ylabel(result['y_label'])
        else:
            raw=result['data']
            if 'samples' in raw: raw['samples'].to_csv(csv,index=False)
            else: pd.DataFrame({raw.get('x_name',result['x_label']):raw['x_flat'],raw.get('y_name',result['y_label']):raw['y_flat'],result['signal_label']:raw['z_flat']}).to_csv(csv,index=False)
            fig,ax=a.plot_map(result['x'],result['y'],result['z'],x_label=result['x_label'],y_label=result['y_label'],z_label=result['signal_label'],title=result['title'],cmap=p['cmap'],vmin=clim[0] if clim else None,vmax=clim[1] if clim else None)
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        ax.xaxis.set_inverted(p['x_reverse'])
        ax.yaxis.set_inverted(p['y_reverse'])
        _save_result_png(fig, ax, png, dpi)
        fig.clear()
        metadata.write_text(json.dumps(dict(source_path=result['source_path'],mode=result['mode'],kind=result['kind'],title=result['title'],background_paths=result.get('background_paths',[]),background_role=result.get('background_role'),background_sources=result.get('background_sources',[]),units=result.get('units',{}),parameters=p,diagnostics=result['diagnostics']),default=_json_default,indent=2,ensure_ascii=False),encoding='utf8')
        with _export_lock:
            version_pattern = re.compile(re.escape(os.path.normcase(stem)) + r'(?:_(\d{3,}))?\.(csv|png|json)')
            versions = set()
            for entry in folder.iterdir():
                match = version_pattern.fullmatch(os.path.normcase(entry.name))
                if match:
                    versions.add(int(match[1] or 1))

            def group_paths(version):
                name = stem if version == 1 else f'{stem}_{version:03d}'
                return [folder / (name + ext) for ext in ('.csv', '.png', '.json')]

            for version in sorted(versions):
                targets = group_paths(version)
                # Check the small recipe first, and reuse only a complete, byte-identical group.
                pairs = ((metadata, targets[2]), (csv, targets[0]), (png, targets[1]))
                if all(target.is_file() and filecmp.cmp(source, target, shallow=False) for source, target in pairs):
                    return [str(target) for target in targets]
            version = 1
            while version in versions:
                version += 1
            targets = group_paths(version)
            created=[]
            try:
                for source,target in zip((csv,png,metadata),targets):
                    with target.open('xb') as output, source.open('rb') as stream:
                        created.append(target); shutil.copyfileobj(stream,output)
            except BaseException:
                for target in created: target.unlink(missing_ok=True)
                raise
        return [str(target) for target in targets]


class BatchCancelled(InterruptedError):
    def __init__(self, completed_paths):
        super().__init__('Batch cancelled; completed exports were retained.')
        self.completed_paths=list(completed_paths)


def export_batch(dataset,parameters,background_paths,directory,cancelled=None):
    import csv
    import json
    import os
    import tempfile
    p=_params(parameters, dataset); paths=[]
    if dataset['mode']=='Reflection':
        from .background_roles import validate_background_role
        validate_background_role(p['background_role'],background_paths)
    axes=(p['cut_axis'],) if p['batch_axes']=='selected' else ('first','second')
    if p['batch_axes'] not in ('selected','both'): raise ValueError('Unknown batch axes')
    planned=[]
    for axis in axes:
        q=dict(p,cut_axis=axis)
        if p['batch_axes']=='both' and p['cut_coordinates']=='explicit':
            pairs={'x':'y','y':'x','doping':'efield','efield':'doping'}
            q['cut_type']=p['cut_type'] if axis=='first' else pairs[p['cut_type']]
        q=_params(q)
        for value in slice_values(dataset,q):
            planned.append((dict(q,view='slice',cut_value=value), value))
    if not planned: raise ValueError('No measured slices are available for batch export.')
    out=Path(directory); out.mkdir(parents=True,exist_ok=True)
    base=_export_stem(out, dataset['source_path'], 'batch_index')
    index=out/(base+'.csv'); suffix=2
    while index.exists():
        index=out/(f'{base}_{suffix}.csv'); suffix+=1
    rows=[dict(source=str(dataset['source_path']),cut_axis=q['cut_axis'],cut_type=q['cut_type'],
               cut_value=value,cut_unit='',status='pending',exported_files='[]') for q,value in planned]
    def write_index():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8-sig', newline='',
                                             dir=out, prefix='.batch-index-', suffix='.tmp',
                                             delete=False) as stream:
                temporary = Path(stream.name)
                writer=csv.DictWriter(stream,fieldnames=['source','cut_axis','cut_type','cut_value','cut_unit','status','exported_files'])
                writer.writeheader(); writer.writerows(rows)
            os.replace(temporary, index)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    write_index()
    paths.append(str(index))
    for i,(q,value) in enumerate(planned):
        try:
            if cancelled and cancelled(): raise InterruptedError()
            result=analyze(dataset,q,background_paths,cancelled)
            result['title']=f"slice_{q['cut_type']}_{value:.9g}"
            exported=export_result(result,directory,parameters=q)
        except InterruptedError as exc:
            rows[i]['status']='cancelled'; write_index()
            raise BatchCancelled(paths) from exc
        except Exception:
            rows[i]['status']='failed'; write_index()
            raise
        paths.extend(exported)
        if dataset['mode'] == 'Transport':
            column = q['cut_type'] if q['cut_type'] in dataset['columns'] else q['y_column']
            rows[i]['cut_unit'] = str(dataset.get('units', {}).get(column, ''))
        else:
            import re
            label = str(result.get('data', {}).get('fixed_axis_label', ''))
            match = re.search(r'\(([^()]*)\)\s*$', label)
            rows[i]['cut_unit'] = match.group(1) if match else ''
        rows[i]['status']='completed'
        rows[i]['exported_files']=json.dumps([Path(path).name for path in exported],ensure_ascii=False)
        write_index()
    return paths


def export_video(dataset,parameters,background_paths,directory,cancelled=None):
    from .spectral_movie import MovieOptions, export_movie, select_values
    p=_params(parameters, dataset)
    if dataset['mode']=='Transport': raise ValueError('Spectral movies require PL or Reflection data.')
    values=select_values(slice_values(dataset,p),p.get('movie_first'),p.get('movie_last'),p.get('movie_stride',1))
    options=MovieOptions(**p['movie_options'])
    class CancelAdapter:
        def is_set(self): return bool(cancelled and cancelled())
    destination=Path(directory)/(Path(dataset['source_path']).stem+'_slices.mp4')
    metadata={'source_path':dataset['source_path'],'mode':dataset['mode'],
              'background_paths':[str(Path(v).resolve()) for v in background_paths], 'parameters':p}
    spectra=np.asarray(dataset['Intensity'])
    if dataset['mode']=='Reflection':
        reference=reflection_preview(dataset,p,background_paths,cancelled)
        if reference['scale_info']:
            p['background_scale_windows'] = reference['scale_info']['windows']
        metadata.update(background_role=reference['background_role'],background_sources=reference['background_sources'],background_scale=reference['scale'],background_scale_info=reference['scale_info'],background_scale_frame=reference['frame'])
        spectra=a.compute_rc_spectra(spectra,reference['background'],reference['scale'])
    def make_cut(value):
        _check(cancelled)
        cut=_cut(dataset,dict(p,cut_value=value),spectra)
        return process_spectral_slice(cut,p['derivative'],int(p['derivative_window']),is_rc=dataset['mode']=='Reflection')
    result=export_movie(make_cut,values,destination,options=options,ranges=p['movie_ranges'],colors=p['movie_colors'],metadata=metadata,cancel=CancelAdapter())
    return [result['path'],result['sidecar']]


def preview_video(dataset,parameters,background_paths=(),cancelled=None,frame_index=0):
    """Return first selected movie frame as PNG bytes; never writes a movie."""
    from io import BytesIO
    from PIL import Image
    from .spectral_movie import MovieOptions,select_values,resolve_sequence_view,render_frame,timeline
    from .spectral_preview import preview_dimensions
    from dataclasses import replace
    p=_params(parameters, dataset)
    if dataset['mode']=='Transport': raise ValueError('Spectral movies require PL or Reflection data.')
    values=select_values(slice_values(dataset,p),p.get('movie_first'),p.get('movie_last'),p.get('movie_stride',1))
    options=MovieOptions(**p['movie_options']); options.validate()
    class CancelAdapter:
        def is_set(self): return bool(cancelled and cancelled())
    def cut(value): return analyze(dataset,dict(p,view='slice',cut_value=value),background_paths,cancelled)['data']
    view=resolve_sequence_view(cut,values,p['movie_ranges'],options.color_mode,p['movie_colors'],cancel=CancelAdapter())
    segments=timeline(len(view['values']),options)
    frame_count=sum(count for _,count in segments)
    frame_index=int(frame_index)
    if frame_index < 0 or frame_index >= frame_count: raise ValueError('Movie preview frame is out of range')
    remaining=frame_index
    for index,count in segments:
        if remaining < count: break
        remaining-=count
    value=view['values'][index]
    width,height=preview_dimensions(options)
    image=render_frame(cut(value),view,replace(options,width=width,height=height)); _check(cancelled)
    out=BytesIO(); Image.fromarray(image).save(out,format='PNG')
    return {'png':out.getvalue(),'value':value,'values':view['values'],'frame_index':frame_index,'frame_count':frame_count,'duration':frame_count/options.fps}


def reflection_preview(dataset,parameters,background_paths,cancelled=None):
    """Return raw, reference and RC spectra for the nearest measured sweep point."""
    if dataset['mode'] != 'Reflection': raise ValueError('Reflection data required.')
    p=_params(parameters, dataset); _check(cancelled)
    from .background_roles import inspect_background_roles
    provenance=inspect_background_roles(background_paths,p['background_role'])
    bg=load_background(background_paths,dataset['wavelength'],p['background_average'],cancelled,role=p['background_role'])
    xs=np.asarray(dataset['x_data'],float); ys=np.asarray(dataset['y_data'],float)
    valid=np.isfinite(xs)&np.isfinite(ys)
    if not valid.any(): raise ValueError('No finite reflection sweep coordinates')
    indices=np.flatnonzero(valid)
    frame=p['background_scale_frame']
    if frame is None:
        frame=int(indices[np.argmin((xs[valid]-float(p['background_scale_x']))**2+(ys[valid]-float(p['background_scale_y']))**2)])
    frame=int(frame)
    if frame < 0 or frame >= len(xs): raise ValueError('Background scale frame is out of range')
    scale=float(p['background_scale']); info={}
    if p['auto_background_scale']:
        scale,info=_estimate_reference_scale(p, dataset['Intensity'][frame], bg, dataset['energy'])
    raw=dataset['Intensity'][frame]
    return dict(background_role=provenance['role'],background_sources=provenance['sources'],frame=frame,x=float(xs[frame]),y=float(ys[frame]),energy=dataset['energy'],wavelength=dataset['wavelength'],raw=raw,background=bg,scaled_background=bg*scale,rc=a.compute_rc_spectra(raw,bg,scale),scale=scale,scale_info=info)


class MoviePreviewSession:
    """A generation-scoped, bounded movie cache; close when inputs change."""
    def __init__(self,dataset,parameters,background_paths=(),cancelled=None):
        import threading
        from .spectral_movie import MovieOptions,select_values,timeline
        from .spectral_preview import SpectralPreviewSession
        self._lock=threading.RLock(); self._cancelled=cancelled; self._closed=False
        self._dataset=dataset; self._params=_params(parameters, dataset); self._background_paths=tuple(background_paths)
        p=self._params
        if dataset['mode']=='Transport': raise ValueError('Spectral movies require PL or Reflection data.')
        self._values=select_values(slice_values(dataset,p),p.get('movie_first'),p.get('movie_last'),p.get('movie_stride',1))
        self._options=MovieOptions(**p['movie_options']); self._options.validate()
        class CancelAdapter:
            def is_set(inner): return bool(self._cancelled and self._cancelled())
        self._cancel=CancelAdapter()
        spectra=np.asarray(dataset['Intensity'])
        if dataset['mode']=='Reflection':
            preview=reflection_preview(dataset,p,self._background_paths,cancelled)
            spectra=a.compute_rc_spectra(spectra,preview['background'],preview['scale'])
        def cut(value):
            _check(self._cancelled)
            result=_cut(self._dataset,dict(p,cut_value=value),spectra)
            return process_spectral_slice(result,p['derivative'],int(p['derivative_window']),is_rc=dataset['mode']=='Reflection')
        self._cache=SpectralPreviewSession(cut)
        try:
            prepared=self._cache.prepare(self._values,self._options,p['movie_ranges'],p['movie_colors'],cancel=self._cancel)
            self._view=prepared['view']; self._segments=timeline(len(self._view['values']),self._options)
            self.frame_count=sum(count for _,count in self._segments)
            self.duration=self.frame_count/self._options.fps
        except BaseException:
            self._cache.close(); raise

    def frame(self,frame_index=0,cancelled=None):
        from io import BytesIO
        from PIL import Image
        with self._lock:
            if self._closed: raise RuntimeError('Movie preview session is closed')
            self._cancelled=cancelled; _check(cancelled)
            frame_index=int(frame_index)
            if frame_index < 0 or frame_index >= self.frame_count: raise ValueError('Movie preview frame is out of range')
            remaining=frame_index
            for index,count in self._segments:
                if remaining < count: break
                remaining-=count
            value=self._view['values'][index]
            image=self._cache.frame(value,self._view,self._options,cancel=self._cancel)
            _check(cancelled); out=BytesIO(); Image.fromarray(image).save(out,format='PNG')
            return {'png':out.getvalue(),'value':value,'values':self._view['values'],
                    'frame_index':frame_index,'frame_count':self.frame_count,'duration':self.duration}

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed=True; self._cache.close()
