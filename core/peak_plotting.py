"""Shared branch overlays and spectra for the desktop window and PNG export."""
from dataclasses import replace
from copy import deepcopy
import numpy as np
from matplotlib import rcParams
from matplotlib.figure import Figure

from core.plotting import HeatmapParams, SplitColorScale, plot_heatmap
from core.colormaps import register_colormaps
from core.power_multi_peaks import multi_peak_model
from core.power_peaks import peak_model


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
        packet['candidates']={'points':[{k:p[k] for k in ('candidate_id','energy','y')} for p in dataset['candidates']['points']]}
    return packet


def overlay_series(dataset):
    result = dataset.get('result')
    if not result:
        return []
    fitted = dataset.get('overlay_position') == 'fitted'
    output = []
    for branch in dataset['branches']:
        if not branch.get('enabled', True):
            continue
        points = sorted([p for p in result['points'] if p['branch_id'] == branch['id']], key=lambda p: p['y'])
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
    size=float(dataset.get('marker_size',9))
    result=dataset.get('result') or {}
    branches={b['id']:b for b in dataset['branches']}
    if dataset.get('show_rejected') and dataset.get('candidates'):
        kept={p.get('candidate_id') for p in result.get('points',[])}
        rejected=[p for p in dataset['candidates']['points'] if p['candidate_id'] not in kept]
        if rejected:
            y=np.asarray([p['y'] for p in rejected])
            artist=ax.scatter([p['energy'] for p in rejected],transform_y(y) if transform_y else y,
                marker='x',s=size*.32,c='0.5',alpha=.4,linewidths=.35,zorder=13,label='Rejected candidates')
            artist.set_gid('peak-rejected');artists.append(artist)
    if result.get('candidate_mode'):
        for polarity,marker in (('peak','^'),('dip','v')):
            points=[p for p in result.get('points',[]) if p['polarity']==polarity and branches.get(p['branch_id'],{}).get('enabled',True)]
            if not points:continue
            y=np.asarray([p['y'] for p in points])
            artist=ax.scatter([p['energy'] for p in points],transform_y(y) if transform_y else y,
                marker=marker,s=size,c=[branch_color(branches[p['branch_id']]) for p in points],
                edgecolors='white',linewidths=.25,zorder=15,label='Peaks' if polarity=='peak' else 'Dips')
            artist.set_gid('peak-candidates');artists.append(artist)
        return artists
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
        if artist.get_gid() in ('peak-candidates','peak-uncertain','peak-rejected'):
            artist.set_sizes([size*(.32 if artist.get_gid()=='peak-rejected' else 1)])
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
    fit = next((f for f in result.get('fits', []) if f['row_index'] == row), None)
    branches = {b['id']: b for b in dataset['branches']}
    if residual_ax is not None:
        residual_ax.clear(); residual_ax.axhline(0, color='0.5', lw=.6)
        residual_ax.set_ylabel('Residual'); residual_ax.set_xlabel('Energy (eV)')
    if fit:
        mask = (x >= fit['x_min']) & (x <= fit['x_max'])
        xf = x[mask]; p = fit['parameters']
        yf = multi_peak_model(xf, p, model=fit['model'], origin=fit['origin'])
        ax.plot(xf, yf, '--', lw=1.3, label='Total fit')
        baseline = p[0]+p[1]*(xf-fit['origin'])
        for i, branch_id in enumerate(fit['branch_ids']):
            b = branches.get(branch_id)
            if b and b.get('enabled', True):
                c = peak_model(xf, 0, 0, *p[2+3*i:5+3*i], model=fit['model'], origin=fit['origin'])
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


def build_figure(dataset, row=0, show_overlay=True, show_residual=False, figure=None):
    fig = figure if figure is not None else Figure(figsize=(8, 7))
    fig.clear()
    gs = fig.add_gridspec(3 if show_residual else 2, 2, width_ratios=[1, .025],
                         height_ratios=[3, 1.35, .55] if show_residual else [3, 1.35],
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
