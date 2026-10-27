"""Widget-free shared numerical theta-comparison plots."""
import numpy as np

def _condition_label(value):
    """Remove machine roundoff at precision finer than condition tolerances."""
    if value is None:
        return 'unknown'
    number = round(float(value), 8)
    if number == 0:
        return '0'
    return f'{number:.8f}'.rstrip('0').rstrip('.')

def comparison_point_details(row):
    def number(value):
        return 'N/A' if value is None else f'{value:.5g}'
    low, high = row.get('ci95_low_k'), row.get('ci95_high_k')
    ci = 'unavailable / unreliable' if low is None or high is None else f'[{number(low)}, {number(high)}] K'
    fit_detail=(f"CW R² (1/s vs T) = {number(row.get('r_squared'))}" if row.get('method','inverse')=='inverse'
                else f"Method: {row.get('method')} · reduced χ² = {number(row.get('reduced_chi2'))}")
    return (f"{row.get('group', '')} · {row['branch']}\n"
            f"Doping={number(row.get('doping'))} · E-field={number(row.get('efield'))}\n"
            f"B fit: ±{number(row.get('halfwidth_t'))} T\n"
            f"θCW = {number(row.get('theta_k'))} K\n95% CI: {ci}\n"
            f"{fit_detail}\n"
            f"N = {row.get('n', 'N/A')} temperature points\n{row.get('diagnostics','')}")

def draw_comparison(figure, payload, axis_name='efield', visible_groups=None, y_limits=None, reliable_y=False, method='inverse', _ax=None):
    if _ax is None and method=='all':
        figure.clear();artists={};axes=[]
        for i,key in enumerate(('inverse','slope','weighted')):
            ax=figure.add_subplot(1,3,i+1);axes.append(ax)
            artists.update(draw_comparison(figure,payload,axis_name,visible_groups,y_limits,reliable_y,key,ax))
        if y_limits is None:
            bounds=[ax.get_ylim() for ax in axes];common=(min(v[0] for v in bounds),max(v[1] for v in bounds))
            for ax in axes:ax.set_ylim(*common)
            for ax in axes:
                for text in list(ax.texts):
                    if 'outside Y range' in text.get_text():text.remove()
                outside=sum(r['theta_k']<common[0] or r['theta_k']>common[1] for artist,r in artists.items() if artist.axes is ax)
                if outside:ax.text(.99,.02,f'{outside} point(s) outside shared Y range',transform=ax.transAxes,ha='right',fontsize=8)
        bounds=[ax.get_xlim() for ax in axes]
        for ax in axes:ax.set_xlim(min(v[0] for v in bounds),max(v[1] for v in bounds))
        return artists
    if _ax is None:figure.clear()
    ax=_ax if _ax is not None else figure.add_subplot(111)
    point_artists = {}
    use_methods=bool(payload.get('method_rows'))
    all_rows = [dict(r) for r in payload['method_rows'] if r['method']==method] if use_methods else (payload['rows'] if method=='inverse' else [])
    for row in all_rows:
        if row['status']=='diagnostic only' and row.get('theta_k') is not None:
            row.update(status='ok',ci95_low_k=None,ci95_high_k=None)
    rows_to_show = [r for r in all_rows if visible_groups is None or r.get('group') in visible_groups]
    groups = sorted({r.get('group', '') for r in all_rows})
    from core.mcd_energy_groups import energy_group_colors
    group_colors = energy_group_colors(dict(enumerate(groups)))
    plotted_values = []
    reliable_bounds = []
    def mark_uncertainty(x, row, color):
        plotted_values.append(row['theta_k'])
        bounds = (row['ci95_low_k'], row['ci95_high_k'])
        if all(value is not None and np.isfinite(value) for value in bounds):
            if bounds[0] <= row['theta_k'] <= bounds[1]:
                reliable_bounds.extend(bounds)
        if row['ci95_low_k'] is None:
            from matplotlib import patheffects
            ax.annotate('×', (x, row['theta_k']), xytext=(0, 0),
                        textcoords='offset points', color='#222222', fontsize=9,
                        ha='center', va='center', zorder=6,
                        path_effects=[patheffects.withStroke(linewidth=1.3, foreground='white')],
                        annotation_clip=True)
    if axis_name=='sensitivity':
        keys=list(dict.fromkeys((r['series_id'],r['branch'],r.get('group','')) for r in rows_to_show))
        for i,key in enumerate(keys):
            rows=sorted([r for r in rows_to_show if (r['series_id'],r['branch'],r.get('group',''))==key],key=lambda r:r['halfwidth_t'])
            color = group_colors[key[2]] if len(groups)>1 else f'C{i%10}'
            label=f"{key[2]} · D={_condition_label(rows[0]['doping'])}, F={_condition_label(rows[0]['efield'])} · {key[1]}"
            ax.plot([r['halfwidth_t'] for r in rows],
                    [r['theta_k'] if r['status']=='ok' else np.nan for r in rows],
                    '-' if key[1]=='B increasing' else '--',label=label,color=color)
            for r in rows:
                if r['status']!='ok': continue
                ci=r['ci95_low_k']
                error=None if ci is None else [[r['theta_k']-ci],[r['ci95_high_k']-r['theta_k']]]
                artist = ax.errorbar([r['halfwidth_t']],[r['theta_k']],yerr=error,
                    fmt='o' if key[1]=='B increasing' else 's',capsize=2,color=color,
                    markerfacecolor=color if key[1]=='B increasing' else 'none')
                point_artists[artist.lines[0]] = r
                mark_uncertainty(r['halfwidth_t'], r, color)
        ax.set_xlabel('Common symmetric B half-width (T)')
        ax.set_title('Range sensitivity · bars: 95% CI; ×: CI unreliable; gaps: failed\n'
                     'Increasing: filled circles · Decreasing: open squares')
    else:
        other='doping' if axis_name=='efield' else 'efield'
        primary=[r for r in rows_to_show if r['is_primary'] and r['status']=='ok' and r[axis_name] is not None]
        labels=[]
        for row in primary:
            extra=' · '.join(f'{k}={_condition_label(v)}' for k,v in sorted(row['fixed_conditions'].items())
                             if k not in ('Doping','E-field') and v is not None)
            label=f"{row.get('group', '')} · {other}={_condition_label(row[other])} · {row['branch']}" + (f' · {extra}' if extra else '')
            if label not in labels: labels.append(label)
            color=group_colors[row.get('group','')] if len(groups)>1 else f'C{labels.index(label)%10}'
            ci=row['ci95_low_k']
            error=None if ci is None else [[row['theta_k']-ci],[row['ci95_high_k']-row['theta_k']]]
            artist = ax.errorbar([row[axis_name]],[row['theta_k']],yerr=error,
                        fmt='o' if row['branch']=='B increasing' else 's',
                        markerfacecolor=color if row['branch']=='B increasing' else 'none',color=color,capsize=3,
                        label=label if not any(line.get_label()==label for line in ax.containers) else '_nolegend_')
            point_artists[artist.lines[0]] = row
            mark_uncertainty(row[axis_name], row, color)
        ax.set_xlabel('E-field (V)' if axis_name=='efield' else 'Doping (V)')
        widths = sorted({r['halfwidth_t'] for r in payload['rows'] if r['is_primary']})
        ranges = '; '.join(f'−{width:g} to +{width:g} T' for width in widths) or 'unavailable'
        ax.set_title(f'θCW comparison · B fit range: {ranges}\n'
                     'Increasing: filled circles · Decreasing: open squares · Bars: 95% CI; ×: CI unreliable')
    range_note = ''
    if reliable_y and y_limits is None:
        if reliable_bounds:
            low, high = min(reliable_bounds), max(reliable_bounds)
            padding = max((high-low)*.08, max(abs(low),abs(high))*1e-6, .05)
            y_limits = (low-padding, high+padding)
            range_note = 'Auto Y: reliable 95% CI only · '
        else:
            ax.text(.99, .02, 'No reliable CI: showing full Y range', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='0.8', alpha=.9))
    if y_limits is not None:
        low, high = y_limits
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError('Y min must be smaller than Y max.')
        ax.set_ylim(low, high)
        outside = sum(value < low or value > high for value in plotted_values)
        ax.text(.99, .02, range_note + f'{outside} point(s) outside Y range', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='0.8', alpha=.9))
    ax.set_ylabel('θCW (K)')
    if use_methods or method!='inverse':
        names={'inverse':'Inverse / equal','slope':'Slope / equal','weighted':'Slope / SE weighted'}
        failed=sum(r.get('theta_k') is None for r in rows_to_show if axis_name=='sensitivity' or r['is_primary'])
        ax.set_title(names[method]+f" · ±{payload.get('parameters',{}).get('halfwidth_t','?')} T\nFilled: inc · open: dec · ×: diagnostic / CI unavailable",fontsize=9)
        if failed:ax.text(.02,.02,f'{failed} fit(s) unavailable',transform=ax.transAxes,fontsize=8)
    ax.axhline(0,color='0.5',linewidth=.8)
    ax.grid(alpha=.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7,loc='best')
    return point_artists
