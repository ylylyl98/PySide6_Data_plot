"""Shared branch overlays and spectra for the desktop window and PNG export."""
from dataclasses import replace
from copy import deepcopy
from collections import defaultdict
import numpy as np
from matplotlib import rcParams
from matplotlib.figure import Figure

from core.plotting import HeatmapParams, SplitColorScale, plot_heatmap
from core.colormaps import register_colormaps
from core.pl_fit_models import evaluate_fit, evaluate_component


def branch_color(branch):
    colors = rcParams['axes.prop_cycle'].by_key()['color']
    return colors[int(branch['color']) % len(colors)]


def overlay_payload(dataset):
    """Send visible positions to the app without copying the analysis cache."""
    packet={k:deepcopy(dataset[k]) for k in ('key','kind','channel','source','name','branches','overlay_position','show_overlay','marker_size') if k in dataset}
    packet['result']={k:deepcopy(v) for k,v in dataset['result'].items()
                      if k not in ('assignments','segment_assignments','branch_catalog','fits')}
    if dataset.get('show_rejected') and dataset.get('candidates'):
        packet['show_rejected']=True
        from core.peak_filter_preview import excluded_candidates
        packet['candidates']={'points':[{k:p[k] for k in ('candidate_id','energy','y','polarity') if k in p} for p in excluded_candidates(dataset)]}
    return packet


def overlay_series(dataset):
    result = dataset.get('result')
    if not result:
        return []
    fitted = dataset.get('overlay_position') == 'fitted'
    by_branch = defaultdict(list)
    for point in result['points']:
        by_branch[point['branch_id']].append(point)
    output = []
    for branch in dataset['branches']:
        if not branch.get('enabled', True):
            continue
        points = sorted(by_branch.get(branch['id'], ()), key=lambda p: p['y'])
        x, y, previous = [], [], None
        for p in points:
            value = p.get('fit_center') if fitted else p['energy']
            good = p['status'] == 'accepted' and (not fitted or p.get('fit_status') == 'ok') and value is not None
            if previous is not None and p.get('segment') != previous:
                x.append(np.nan); y.append(np.nan)
            x.append(value if good else np.nan); y.append(p['y'])
            previous = p.get('segment')
        output.append((branch, np.asarray(x, float), np.asarray(y, float)))
    return output


def draw_overlay(ax, dataset, transform_y=None):
    artists = []
    size=float(dataset.get('marker_size',4))
    result=dataset.get('result') or {}
    branches={b['id']:b for b in dataset['branches']}
    if dataset.get('show_rejected') and dataset.get('candidates'):
        from core.peak_filter_preview import excluded_candidates
        rejected=excluded_candidates(dataset)
        if rejected:
            y=np.asarray([p['y'] for p in rejected])
            artist=ax.scatter([p['energy'] for p in rejected],transform_y(y) if transform_y else y,
                marker='x',s=max(1., size*.6),c='0.5',alpha=.55,linewidths=.4,zorder=13,label='Rejected candidates')
            artist.set_gid('peak-rejected');artists.append(artist)
    if result.get('candidate_mode'):
        for polarity,marker in (('peak','^'),('dip','v')):
            points=[p for p in result.get('points',[]) if p['polarity']==polarity and p.get('candidate_origin') != 'curvature' and branches.get(p['branch_id'],{}).get('enabled',True)]
            if not points:continue
            y=np.asarray([p['y'] for p in points])
            artist=ax.scatter([p['energy'] for p in points],transform_y(y) if transform_y else y,
                marker=marker,s=size,c=[branch_color(branches[p['branch_id']]) for p in points],
                edgecolors='white',linewidths=.25,zorder=15,label='Peaks' if polarity=='peak' else 'Dips')
            artist.set_gid('peak-candidates');artists.append(artist)
        shoulders = [p for p in result.get('points', []) if p.get('candidate_origin') == 'curvature'
                     and branches.get(p['branch_id'], {}).get('enabled', True)]
        if shoulders:
            y = np.asarray([p['y'] for p in shoulders])
            artist = ax.scatter([p['energy'] for p in shoulders], transform_y(y) if transform_y else y,
                                marker='D', s=size, facecolors='none', edgecolors='#ed9b28', linewidths=.8,
                                zorder=15, label='Provisional shoulders')
            artist.set_gid('peak-shoulders');artists.append(artist)
        return artists
    if dataset.get('kind') == 'PL' and result.get('pl_fit_version', 0) >= 2 and dataset.get('overlay_position') == 'fitted':
        visible = [p for p in result.get('points', []) if branches.get(p['branch_id'], {}).get('enabled', True)]
        for points, color, marker, label, gid in (
                (visible, '#00bcd4', 'o', 'Detected positions', 'peak-detected-reference'),
                ([p for p in visible if p.get('fit_status') != 'ok' or p.get('status') != 'accepted'],
                 '#ed9b28', 'x', 'Unconfirmed: detected position', 'peak-fit-review')):
            if points:
                y = np.asarray([p['y'] for p in points])
                style = dict(facecolors='none', edgecolors=color) if marker == 'o' else dict(c=color)
                artist = ax.scatter([p['energy'] for p in points], transform_y(y) if transform_y else y,
                                    marker=marker, s=size, linewidths=.65, zorder=14, label=label, **style)
                artist.set_gid(gid)
                artists.append(artist)
    for branch, x, y in overlay_series(dataset):
        if transform_y:
            y = transform_y(y)
        lines=ax.plot(x, y, '.-', color=branch_color(branch), ms=np.sqrt(size), lw=1,
                      label=branch['name'], zorder=15)
        for line in lines:line.set_gid('peak-track')
        artists.extend(lines)
    if dataset.get('overlay_position')!='fitted':
        uncertain=[p for p in result.get('points',[]) if p['status']=='uncertain' and branches.get(p['branch_id'],{}).get('enabled',True)]
        if uncertain:
            y=np.asarray([p['y'] for p in uncertain])
            artist=ax.scatter([p['energy'] for p in uncertain],transform_y(y) if transform_y else y,
                marker='x',s=size,c=[branch_color(branches[p['branch_id']]) for p in uncertain],
                linewidths=.6,zorder=15,label='Ambiguous association')
            artist.set_gid('peak-uncertain');artists.append(artist)
    return artists


def resize_overlay(ax, size):
    """Resize existing peak artists without touching data or heatmap meshes."""
    for artist in ax.collections:
        if artist.get_gid() in ('peak-candidates','peak-uncertain','peak-rejected', 'peak-detected-reference', 'peak-fit-review', 'peak-shoulders'):
            artist.set_sizes([max(1., size*.6) if artist.get_gid()=='peak-rejected' else size])
    for line in ax.lines:
        if line.get_gid()=='peak-track':line.set_markersize(np.sqrt(size))


def draw_spectrum(dataset, row, ax, residual_ax=None):
    cube = dataset['cube']; x = cube.energy; spectrum = cube.Z[row]
    ax.clear(); ax.plot(x, spectrum, lw=1, color='0.5', label='Measured')
    result = dataset.get('result') or {}
    pool=dataset.get('candidates')
    if pool and pool['settings']['detection_method']=='sg':
        from core.peak_candidates import detection_spectrum
        ax.plot(x,detection_spectrum(x,spectrum,pool['settings']),lw=1,label='SG detection signal')
    if dataset.get('show_rejected'):
        from core.peak_filter_preview import excluded_candidates
        rejected = [p for p in excluded_candidates(dataset) if p.get('row_index') == row]
        if rejected:
            energies = [p['energy'] for p in rejected]
            artist = ax.scatter(energies, np.interp(energies, x, spectrum), marker='x',
                s=18, color='0.5', alpha=.65, linewidths=.7, zorder=8)
            artist.set_gid('peak-rejected')
    fit = next((f for f in result.get('fits', []) if f['row_index'] == row), None)
    branches = {b['id']: b for b in dataset['branches']}
    if residual_ax is not None:
        residual_ax.clear(); residual_ax.axhline(0, color='0.5', lw=.6)
        residual_ax.set_ylabel('Residual'); residual_ax.set_xlabel('Energy (eV)')
    if fit:
        mask = (x >= fit['x_min']) & (x <= fit['x_max'])
        xf = x[mask]; p = fit['parameters']
        yf = evaluate_fit(xf, fit)
        ax.plot(xf, yf, '--', lw=1.3, label='Total fit')
        baseline = p[0]+p[1]*(xf-fit['origin'])
        for i, branch_id in enumerate(fit['branch_ids']):
            b = branches.get(branch_id)
            if b and b.get('enabled', True):
                c = evaluate_component(xf, fit, i)
                ax.plot(xf, baseline+c, lw=1, color=branch_color(b), label=b['name'])
        if residual_ax is not None:
            residual_ax.plot(xf, spectrum[mask]-yf, lw=.8)
    # Index the selected row once. Large candidate maps can contain thousands
    # of short branches; rescanning the full map for each one blocks the UI.
    row_points = {}
    for point in result.get('points', []):
        if point['row_index'] == row:
            row_points.setdefault(point['branch_id'], point)
    for b in branches.values():
        if not b.get('enabled', True):
            continue
        point = row_points.get(b['id'])
        energy = point['energy'] if point else b['seed_energy'] if b['seed_row'] == row else None
        if energy is not None:
            value = float(np.interp(energy, x, spectrum))
            ax.scatter([energy], [value], color=branch_color(b), s=20, zorder=10)
            label=ax.annotate(b['name'], (energy, value), xytext=(3, 8), textcoords='offset points', fontsize=8)
            label._use_theme_text=True
    ax.set_xlabel('Energy (eV)'); ax.set_ylabel(cube.cbar_label)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=False)
    ax.set_title(f'{cube.gate_label} = {cube.gate[row]:.6g}', fontsize=9)
    ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=.15)


def build_fit_trends(dataset, figure=None):
    """Fitted position and linewidth versus the original scan, with valid SEs."""
    fig = figure if figure is not None else Figure(figsize=(8, 7))
    fig.clear()
    axes = fig.subplots(2, 1, sharex=True)
    fig.subplots_adjust(left=.14, right=.96, bottom=.11, top=.93, hspace=.18)
    result = dataset.get('result') or {}
    settings = result.get('settings', dataset['settings'])
    rows = np.flatnonzero((dataset['cube'].gate >= settings['y_min']) & (dataset['cube'].gate <= settings['y_max']))
    rows = rows[np.argsort(dataset['cube'].gate[rows])]
    grouped = {}
    for point in result.get('points', []):
        grouped.setdefault(point['branch_id'], {})[point['row_index']] = point
    count = 0
    for branch in dataset.get('branches', []):
        if not branch.get('enabled', True):
            continue
        points = grouped.get(branch['id'], {})
        for axis, field, error_field in zip(axes, ('fit_center', 'fwhm_mev'), ('center_error_ev', 'fwhm_error_mev')):
            values, errors = [], []
            for row in rows:
                point = points.get(int(row), {})
                valid = point.get('fit_status') == 'ok'
                values.append(point.get(field) if valid and point.get(field) is not None else np.nan)
                errors.append(point.get(error_field) if valid and point.get(error_field) is not None else np.nan)
            values, errors = np.asarray(values, float), np.asarray(errors, float)
            if not np.isfinite(values).any():
                continue
            x = dataset['cube'].gate[rows]
            axis.plot(x, values, 'o-', markersize=3, linewidth=.8, color=branch_color(branch), label=branch['name'])
            valid = np.isfinite(values) & np.isfinite(errors) & (errors >= 0)
            if valid.any():
                axis.errorbar(x[valid], values[valid], yerr=errors[valid], fmt='none', capsize=2, color=branch_color(branch))
            count += 1
    axes[0].set_ylabel('Fit center (eV)'); axes[1].set_ylabel('FWHM (meV)')
    axes[1].set_xlabel(dataset['cube'].gate_label)
    axes[0].set_title('Fitted peaks · error bars: ±1 fit standard error', fontsize=10)
    for axis in axes:
        axis.grid(alpha=.15)
        if axis.lines:
            axis.legend(fontsize=7)
    if not count:
        text = axes[0].text(.5, .5, 'No valid fitted peaks in this analysis range', transform=axes[0].transAxes, ha='center')
        text._use_theme_text = True
    return axes


def build_figure(dataset, row=0, show_overlay=True, show_residual=False, figure=None):
    fig = figure if figure is not None else Figure(figsize=(8, 7))
    fig.clear()
    height_ratios = (([1.25, 3, .8] if show_residual else [1.25, 3])
        if dataset.get('analysis_scope') == 'spectrum' else ([3, 1.35, .55] if show_residual else [3, 1.35]))
    if 'map_height_percent' in dataset:
        share = max(20, min(80, float(dataset['map_height_percent'])))
        height_ratios = [share, (100 - share) * .8, (100 - share) * .2] if show_residual else [share, 100 - share]
    gs = fig.add_gridspec(3 if show_residual else 2, 2, width_ratios=[1, .025],
                         height_ratios=height_ratios,
                         left=.11, right=.9, top=.95, bottom=.09, hspace=.5, wspace=.05)
    heat = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1])
    spectrum = fig.add_subplot(gs[1, 0]); residual = fig.add_subplot(gs[2, 0], sharex=spectrum) if show_residual else None
    cube = dataset['cube']; order = np.argsort(cube.gate)
    view = dataset.get('view') or {}; log_y = bool(view.get('y_axis_log', False))
    if log_y:
        order = order[cube.gate[order] > 0]
    if not len(order):
        raise ValueError('No positive scan values for a log axis.')
    shown = replace(cube, gate=cube.gate[order], Z=cube.Z[order])
    finite = shown.Z[np.isfinite(shown.Z)]
    lo, hi = (np.percentile(finite, [.5, 99.5]) if finite.size else (0, 1))
    if not hi > lo:
        hi = lo + max(1, abs(lo))*.01
    ymin, ymax = float(shown.gate.min()), float(shown.gate.max())
    if ymin == ymax:
        ymin, ymax = ymin-.5, ymax+.5
    title=dataset['name']
    if dataset['kind']=='DRR':
        product=dataset.get('product_label',{'raw':'ΔR/R','second':'2nd derivative'}.get(dataset['channel'],dataset['channel']))
        title=title.replace(f" · {dataset['channel']} · ",f' · {product} · ')
    params = HeatmapParams(title, 'Energy (eV)', cube.gate_label, cube.cbar_label,
        float(lo), float(hi), (float(cube.energy.min()), float(cube.energy.max())), (ymin, ymax),
        cmap='RdBu_r' if dataset['kind']=='DRR' else 'viridis', y_axis_log=log_y)
    display=view.get('display') or {}
    if display:
        fields={key:display[key] for key in ('cmap','vmin','vmax','log_scale','center_zero','clip_outliers') if key in display}
        if display.get('split_scale'):
            fields['split_scale']=SplitColorScale(**display['split_scale'])
        params=replace(params,**fields)
    register_colormaps()
    render = plot_heatmap(heat, shown, params)
    if render.tertiary is not None:
        from core.region_colorbars import three_preview_axes,add_three_colorbars
        add_three_colorbars(fig,render,three_preview_axes(cax),label=cube.cbar_label,fontsize=7)
    elif render.is_split:
        cax.set_axis_off()
        for mesh,bottom,title in zip(render.images,(.60,.05),(f'x ≤ {render.split_x:.5g}',f'x ≥ {render.split_x:.5g}')):
            bar_axis=cax.inset_axes([0,bottom,1,.35])
            bar=fig.colorbar(mesh,cax=bar_axis)
            bar.ax.set_title(title,fontsize=7,pad=4)
            bar.ax.tick_params(labelsize=7)
    else:
        fig.colorbar(render.primary, cax=cax, label=cube.cbar_label)
    if show_overlay:
        draw_overlay(heat, dataset)
    gate_line = heat.axhline(float(cube.gate[row]), color='0.8', ls='--', lw=.8)
    if view.get('xlim'):heat.set_xlim(view['xlim'])
    if view.get('ylim'):heat.set_ylim(view['ylim'])
    draw_spectrum(dataset, row, spectrum, residual)
    spectrum.set_xlim(heat.get_xlim())
    return heat, spectrum, residual, gate_line
