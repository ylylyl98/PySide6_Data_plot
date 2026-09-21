"""Batch CW comparisons using explicit energy groups and shared fit ranges."""
import numpy as np
from core.curie_weiss import fit_inverse_curie_weiss
from core.mcd_slope_refit import refit_record_slopes


def compare_theta(entries, *, halfwidth, t_min, t_max, branches,
                  temperature_mode='setpoint', sensitivity_widths=(), compare_methods=False, progress=None, log=None):
    widths = sorted(set([float(halfwidth), *map(float, sensitivity_widths)]))
    if not widths or any(not np.isfinite(w) or w <= 0 for w in widths):
        raise ValueError('Field half-widths must be positive and finite.')
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min > t_max:
        raise ValueError('Invalid temperature interval.')
    if not entries or not branches:
        raise ValueError('Select temperature series and sweep branches.')
    if temperature_mode not in ('setpoint', 'measured'):
        raise ValueError('Unknown temperature source.')
    rows, points, fits, warnings = [], [], [], []
    method_rows=[]
    grids = []
    for entry in entries:
        selected, errors = [], []
        for r in entry['records']:
            t = (r.temperature_measured_k if temperature_mode == 'measured' else
                 None if 'assumed' in r.condition_sources.get('T','').lower() else r.condition_value('T'))
            if t is None or not np.isfinite(t) or t <= 0:
                errors.append(f'Missing/invalid {temperature_mode} temperature: {r.source_file}')
            elif t_min <= t <= t_max:
                selected.append((float(t), r))
        selected.sort(key=lambda p:p[0])
        grids.append(tuple(t for t,r in selected))
        if len({round(r.width_mev,6) for t,r in selected}) > 1:
            errors.append('Energy-window widths differ within this series.')
        for width in widths:
            branch_points = {b:[] for b in branches}
            branch_errors = {b:list(errors) for b in branches}
            for t, r in selected:
                try:
                    field_fits = refit_record_slopes(r,-width,width,branches)
                except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as exc:
                    field_fits = []
                    for b in branches:
                        branch_errors[b].append(f'{t:g} K: {exc}')
                for b in branches:
                    f = next((f for f in field_fits if f['branch']==b), None)
                    point = dict(series_id=entry['series_id'], group=entry['group'], branch=b,
                        temperature_k=t, halfwidth_t=width, record_id=r.record_id,
                        source=r.source_file, settings_path=str(r.settings_path), trace_path=str(r.trace_path),
                        center_ev=r.center_ev, width_mev=r.width_mev, slope=None, slope_se=None,
                        field_n=0, field_min_t=None, field_max_t=None, field_r_squared=None,
                        curvature_flag=None, jump_flag=None, status='load failed')
                    if f:
                        point.update(slope=f['slope'], slope_se=f['slope_se'], field_n=f['n'],
                            field_min_t=f['field_min_t'], field_max_t=f['field_max_t'],
                            field_r_squared=f['r_squared'], curvature_flag=f['curvature_flag'],
                            jump_flag=f['jump_flag'], status=f['status'])
                        if f['status']!='ok':
                            branch_errors[b].append(f"{t:g} K: {f['status']} ({f['n']} B points)")
                    branch_points[b].append(point)
                    points.append(point)
            for b in branches:
                block = branch_points[b]
                row = dict(series_id=entry['series_id'], group=entry['group'],
                    doping=entry['fixed_conditions'].get('Doping'), efield=entry['fixed_conditions'].get('E-field'),
                    fixed_conditions=entry['fixed_conditions'], branch=b, halfwidth_t=width,
                    is_primary=width==halfwidth, n=len(block), temperature_min_k=selected[0][0] if selected else None,
                    temperature_max_k=selected[-1][0] if selected else None,
                    theta_k=None, theta_se_k=None, ci95_low_k=None, ci95_high_k=None,
                    r_squared=None, status='failed', diagnostics='')
                try:
                    if branch_errors[b]:
                        raise ValueError('; '.join(branch_errors[b]))
                    result = fit_inverse_curie_weiss([p['temperature_k'] for p in block],
                        [p['slope'] for p in block],slope_se=[p['slope_se'] for p in block])
                    ci = result['theta_ci95_k']
                    diagnostics = list(result['warnings'])
                    if any(p['curvature_flag'] or p['jump_flag'] for p in block):
                        diagnostics.append('Inspect MCD–B curvature/jump flags before interpreting theta.')
                    if any(p['field_n']<5 for p in block):
                        diagnostics.append('Some field fits contain fewer than 5 points.')
                    row.update(theta_k=result['theta_k'],theta_se_k=result['theta_se_k'],
                        ci95_low_k=ci[0] if ci else None,ci95_high_k=ci[1] if ci else None,
                        r_squared=result['r_squared'],status='ok',diagnostics='; '.join(diagnostics),
                        interpretation=result['interpretation'])
                    fits.append(dict(series_id=entry['series_id'],group=entry['group'],branch=b,
                                     halfwidth_t=width,fit=result))
                    for i,p in enumerate(block):
                        p.update(inverse_slope=result['inverse_slope'][i],inverse_slope_se=result['inverse_slope_se'][i],
                                 predicted_inverse=result['predicted_inverse'][i],inverse_residual=result['inverse_residuals'][i])
                except ValueError as exc:
                    row['diagnostics'] = str(exc)
                rows.append(row)
                if compare_methods:
                    from core.cw_method_comparison import compare_cw_methods
                    compared=compare_cw_methods([p['temperature_k'] for p in block],
                        [p['slope'] if p['slope'] is not None else float('nan') for p in block],
                        [p['slope_se'] for p in block])
                    for result in compared:
                        item=dict(row)
                        item.update(result)
                        item['r_squared']=None
                        item['theta_se_k']=None
                        item.pop('interpretation',None)
                        item['status']='ok' if result['status']=='estimated' else result['status']
                        if branch_errors[b]:
                            item.update(theta_k=None,ci95_low_k=None,ci95_high_k=None,status='failed',diagnostics='; '.join(branch_errors[b]))
                        elif any(p['curvature_flag'] or p['jump_flag'] for p in block):
                            item['diagnostics']+=' Inspect MCD–B curvature/jump flags.'
                        method_rows.append(item)
        if progress:
            progress.emit(round(100*(len(grids))/len(entries)))
    if len(set(grids))>1:
        warnings.append('Actual temperature grids differ across series; compare the reported ranges and points.')
    if len({round(r.width_mev,6) for e in entries for r in e['records']})>1:
        warnings.append('Energy-window widths differ across series; verify comparable optical extraction.')
    return dict(schema_version=2,rows=rows,method_rows=method_rows,points=points,fits=fits,warnings=warnings,
        parameters=dict(halfwidth_t=halfwidth,temperature_range_k=[t_min,t_max],temperature_mode=temperature_mode,
                        branches=list(branches),sensitivity_widths_t=widths,background=0,
                        method='Unweighted inverse-slope linear CW fit'),
        assumptions='Same optical resonance; paramagnetic temperature range; temperature-independent optical conversion. Theta is not proof of magnetic order.')
