"""Calibrate each measurement, average its DR/R, then differentiate the mean."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import hashlib
import json
import shutil

import numpy as np

from core.drr_bn_store import DrrBnStore
from core.drr_sources import DrrMeasurementAssignment, portable_source_name
from core.load_cancellation import check_load_cancelled


def verify_batch(context, payload):
    if not payload.get('batch') or Path(payload['source_identity']['root']).resolve() != Path(context.folder):
        raise ValueError('BN batch belongs to another experiment.')
    items = payload['items']
    if tuple(item['measurement'] for item in items) != context.measurements:
        raise ValueError('BN batch belongs to another measurement selection.')
    store = DrrBnStore(context.folder)
    versions = []
    for item in items:
        check_load_cancelled()
        version = store.get(item['version_id'])
        if (not version.fresh or
            tuple(v['path'] for v in version.source_identity['measurements']) != (item['measurement'],) or
            version.source_identity['y_axis'] != context.y_axis or
            (version.settings['derivative_window'], version.settings['derivative_polyorder']) !=
                (context.derivative_window, context.derivative_polyorder)):
            raise ValueError('BN batch sources, measurement scope or SG settings changed; regenerate the batch.')
        versions.append(version)
    return tuple(versions)


def batch_assignments(context, versions, *, original=False):
    values = []
    for name, version in zip(context.measurements, versions):
        use = version.eligible and not original
        source = version.source_identity
        files = ((portable_source_name(context.folder, version.csv_path),) if use else
                 tuple(v['path'] for v in source['baseline_files']))
        values.append(DrrMeasurementAssignment(name, 'External', files,
            'last' if use else source['baseline_which'], source_recipe='bn_per_measurement_v1',
            selection_reason='Individual BN calibration' if use else 'Original BN retained'))
    return tuple(values)


def _average(cubes):
    from core.loader import _interp_rows_no_extrapolation
    from core.streaming_average import StreamingNanMean
    from core.drr_axis_coordinates import CoordinateAccumulator
    first = cubes[0]
    accumulator = StreamingNanMean(first.Z.shape)
    coordinates = CoordinateAccumulator(first.gate)
    count = np.zeros(first.Z.shape, dtype=int)
    for cube in cubes:
        if cube.gate_label != first.gate_label:
            raise ValueError('Measurements use incompatible gate axes.')
        values = cube.Z
        coordinates.add(cube.gate, cube.gate_coordinates)
        if not np.array_equal(cube.energy, first.energy):
            values = _interp_rows_no_extrapolation(values, cube.energy, first.energy, context='BN batch energy')
        if not np.array_equal(cube.gate, first.gate):
            values = _interp_rows_no_extrapolation(values.T, cube.gate, first.gate, context='BN batch gate').T
        accumulator.add(values)
        count += np.isfinite(values)
    cube = replace(first, Z=accumulator.mean(), gate_coordinates=coordinates.result(),
                   title=f'Mean DR/R of {len(cubes)} individually referenced measurements')
    cube.drr_numerical_path = 'heterogeneous'
    return cube, count


def batch_cubes(context, versions, *, original=False, trial=False):
    """Load verified snapshots for trial processing without activating versions."""
    from core.loader import load_drr_avg
    store = DrrBnStore(context.folder)
    cubes = []
    for name, version in zip(context.measurements, versions):
        check_load_cancelled()
        saved = store.load_result(version)
        use_candidate = not original and (version.eligible or trial)
        background = (saved['candidate_backgrounds'].get('single_pair', saved['candidate_background'])
                      if trial and not original else saved['candidate_background']) if use_candidate else saved['original_background']
        cubes.append(load_drr_avg(context.folder, (name,), bg_mode='external', y_axis=context.y_axis,
            external_vector=background, external_energy=saved['energy']))
    return _average(cubes)


def load_batch_trial(context, payload, *, original=False):
    versions = verify_batch(context, payload)
    cube, _ = batch_cubes(context, versions, original=original)
    cube.drr_assignments = tuple(v.to_dict() for v in batch_assignments(context, versions, original=original))
    return cube


def compare_bn_batch(context, files, which, *, settings=None, assignments=(), cancellation=None, progress=None, log=None,
                     export_products=True):
    from ui_qt.quick.drr_background_workflow import compare_bn
    from core.processing import apply_sg_derivative_energy
    started = perf_counter()
    config = dict(settings or {})
    config.pop('per_measurement', None)
    mapping = {v.measurement_file: v for v in assignments}
    if assignments and (len(assignments) != len(context.measurements) or set(mapping) != set(context.measurements)):
        raise ValueError('BN mapping needs exactly one entry for each measurement.')
    items, baseline_records = [], {}
    for index, name in enumerate(context.measurements):
        check_load_cancelled()
        assignment = mapping.get(name)
        if assignment and (assignment.baseline_mode != 'External' or assignment.regions):
            raise ValueError('Individual BN calibration needs direct measured BN references, without spliced regions.')
        selected = assignment.baseline_files if assignment else files
        frame = assignment.baseline_which if assignment else which
        one = compare_bn(replace(context, measurements=(name,)), selected, frame,
            settings=config, cancellation=cancellation, log=log, export_products=export_products)
        for record in one['source_identity']['baseline_files']:
            baseline_records[record['path']] = record
        items.append(dict(measurement=name, version_id=one['version_id'], eligible=one['eligible'],
            status='calibrated' if one['eligible'] else 'original_retained',
            summary=one['summary'], exports=one.get('exports')))
        if progress is not None:
            value = int(80*(index+1)/len(context.measurements))
            progress.emit(value) if hasattr(progress, 'emit') else progress(value)
    payload = dict(batch=True, items=items, source_identity=dict(root=context.folder,
        measurements=[dict(path=name) for name in context.measurements],
        baseline_files=list(baseline_records.values()), y_axis=context.y_axis),
        status='per_measurement', eligible=any(v['eligible'] for v in items))
    versions = verify_batch(context, payload)
    payload['settings'] = dict(versions[0].settings)
    payload['original_assignments'] = [v.to_dict() for v in batch_assignments(context, versions, original=True)]
    payload['candidate_assignments'] = [v.to_dict() for v in batch_assignments(context, versions)]
    old, count = batch_cubes(context, versions, original=True)
    effective, _ = batch_cubes(context, versions)
    trial, _ = batch_cubes(context, versions, trial=True)
    derivatives = []
    for cube in (old, effective, trial):
        value, used = apply_sg_derivative_energy(cube, derivative=2,
            window_length=context.derivative_window, polyorder=context.derivative_polyorder)
        derivatives.append(value.Z)
    calibrated = sum(v['eligible'] for v in items)
    payload.update(energy=old.energy, gates=old.gate, original_drr=old.Z, candidate_drr=effective.Z,
        trial_drr=trial.Z, original_d2=derivatives[0], candidate_d2=derivatives[1], trial_d2=derivatives[2],
        contribution_count=count, used_window=used,
        summary=f'{calibrated}/{len(items)} individually calibrated; {len(items)-calibrated} retain original BN. Equal-file DR/R mean, then D2.',
        report=dict(optimization_policy='per_measurement_bn_v1', independent_validation=False,
            processing_order='native per-file DR/R -> align without extrapolation -> equal-file finite mean -> D2',
            items=items, measurement_count=len(items), calibrated_count=calibrated),
        candidates=[dict(name='original', eligible=False, reason='All original measured BN references.'),
                    dict(name='per_measurement', eligible=bool(calibrated), reason='Eligible per-file calibrations with explicit original fallback.')])
    payload['exports'] = _export_batch(context, payload, versions) if export_products else None
    verify_batch(context, payload)
    payload['elapsed_seconds'] = perf_counter()-started
    return payload


def _export_batch(context, payload, versions):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from core.typography import matplotlib_font_context
    from core.drr_bn_store import file_digest
    identity = dict(versions=[(v.version_id,v.report_sha256) for v in versions],
        measurements=list(context.measurements), averaging='mean_individual_drr_v1',
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    parent = Path(context.folder)/'Processed Data'/'DRR'/'BN Calibration'
    directory = parent/('batch_'+key)
    def existing():
        doc = json.loads((directory/'results.json').read_text(encoding='utf-8'))
        paths = {}
        if doc['identity'] != json.loads(json.dumps(identity)):
            raise ValueError('BN batch export identity changed.')
        for role, item in doc['outputs'].items():
            path = directory/item['file']
            if path.resolve().parent != directory.resolve() or file_digest(path) != item['sha256']:
                raise ValueError('BN batch output changed; it will not be overwritten.')
            paths[role] = str(path)
        paths['metadata_json'] = str(directory/'results.json')
        return dict(directory=str(directory), paths=paths, status='per_measurement', cached=True)
    if directory.exists():
        return existing()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent/('.batch_'+uuid4().hex)
    temporary.mkdir()
    paths = {}
    try:
        header = 'Photon energy(eV)\t'+'\t'.join(format(v,'.12g') for v in payload['gates'])
        for role in ('original_drr','candidate_drr','trial_drr','original_d2','candidate_d2','trial_d2','contribution_count'):
            path=temporary/(role+'.dat')
            np.savetxt(path,np.column_stack((payload['energy'],payload[role].T)),delimiter='\t',header=header,comments='',fmt='%.12g')
            paths[role+'_dat']=path
        with matplotlib_font_context():
            fig=Figure(figsize=(16,8),layout='constrained'); FigureCanvasAgg(fig)
            axes=fig.subplots(2,3,squeeze=False)
            order=np.argsort(payload['gates'])
            for row,product in enumerate(('drr','d2')):
                data=[payload[name+'_'+product] for name in ('original','candidate','trial')]
                finite=np.concatenate([v[np.isfinite(v)] for v in data])
                if product == 'drr' and finite.size:
                    lo,hi=map(float,np.percentile(finite,[1,99]))
                    if hi <= lo:
                        padding=max(abs(lo)*1e-6,1e-12);lo,hi=lo-padding,hi+padding
                else:
                    limit=max(float(np.percentile(np.abs(finite),99)) if finite.size else 1.,1e-12)
                    lo,hi=-limit,limit
                for col,(values,label) in enumerate(zip(data,('Original BN mean','Eligible corrections + original fallbacks','All candidates - exploratory only'))):
                    mesh=axes[row,col].pcolormesh(payload['energy'],payload['gates'][order],values[order],shading='auto',cmap='RdBu_r',vmin=lo,vmax=hi)
                    axes[row,col].set(title=label,xlabel='Energy (eV)',ylabel='Gate (V)')
                fig.colorbar(mesh,ax=list(axes[row]),label='DR/R' if product=='drr' else 'D2 (eV^-2)')
            fig.suptitle('Per-file BN correction -> equal-file mean DR/R -> D2\nSame color scale in each row; all selected files remain in the average.')
            path=temporary/'comparison.png'; fig.savefig(path,dpi=140,bbox_inches='tight');fig.clear()
            paths['comparison_png']=path
        verify_batch(context,payload)
        doc=dict(schema_version=1,identity=identity,report=payload['report'],
            outputs={k:dict(file=p.name,sha256=file_digest(p)) for k,p in paths.items()})
        (temporary/'results.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.rename(directory)
        result=existing();result['cached']=False
        return result
    finally:
        if temporary.exists() and temporary.resolve().parent==parent.resolve() and temporary.name.startswith('.batch_'):
            shutil.rmtree(temporary)
