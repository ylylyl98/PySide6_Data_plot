"""Conservative grouping and linked plots of saved integration windows.

Centers are saved window coordinates, never fitted resonance energies.
"""
from collections import Counter
import numpy as np


def initial_energy_groups(records, tolerance_mev=5.0):
    tolerance = max(0., float(tolerance_mev)) * .001
    groups = {}; anchor = None; number = 0
    for record in sorted(records, key=lambda r: (r.center_ev, r.record_id)):
        if anchor is None or record.center_ev - anchor > tolerance + 1e-12:
            anchor = record.center_ev; number += 1
        groups[record.record_id] = f"Group {number}"
    return groups


def compact_group_numbers(groups):
    """Close gaps in numbered groups without changing membership or custom names."""
    import re
    labels = sorted({name for name in groups.values() if re.fullmatch(r'Group [1-9][0-9]*',name)},
                    key=lambda name: int(name.split()[1]))
    replacements = {name: f'Group {index}' for index,name in enumerate(labels,1)}
    return {key: replacements.get(name,name) for key,name in groups.items()}


def visible_group_numbers(groups, included_ids):
    """Number visible groups first, retaining distinct restorable hidden groups."""
    import re
    numbered = {name for name in groups.values() if re.fullmatch(r'Group [1-9][0-9]*',name)}
    visible = {name for key,name in groups.items() if key in included_ids}
    ordered = sorted(numbered,key=lambda name:(name not in visible,int(name.split()[1])))
    mapping = {name:f'Group {index}' for index,name in enumerate(ordered,1)}
    return {key:mapping.get(name,name) for key,name in groups.items()}


def safe_energy_edges(records, tolerance_mev=5.0, manual_record_ids=()):
    """Connect only unique adjacent fields with small energy and field steps."""
    manual_record_ids = set(manual_record_ids)
    fields = [r.condition_value('E-field') for r in records]
    valid = [i for i, f in enumerate(fields) if f is not None and np.isfinite(f)]
    counts = Counter(round(fields[i], 8) for i in valid)
    ordered = sorted(valid, key=lambda i: fields[i])
    edges = []
    for a, b in zip(ordered, ordered[1:]):
        # An export may contain several fixed-condition series. Group colors
        # alone are never permission to connect different physical conditions.
        same_conditions = True
        for key in ('Doping','T','Vbias'):
            left,right = records[a].condition_value(key),records[b].condition_value(key)
            if left is None or right is None:
                same_conditions = left is None and right is None
            else:
                same_conditions = bool(np.isclose(left,right,atol=.1 if key=='T' else .01,rtol=0))
            if not same_conditions:
                break
        if not same_conditions:
            continue
        if abs(records[a].width_mev-records[b].width_mev) > .001:
            continue
        if counts[round(fields[a], 8)] != 1 or counts[round(fields[b], 8)] != 1:
            continue
        confirmed = all(records[i].record_id in manual_record_ids for i in (a,b))
        if confirmed or abs(records[a].center_ev - records[b].center_ev) <= tolerance_mev * .001 + 1e-12:
            edges.append((a, b))
    return edges


def energy_group_colors(groups, palette='tab10'):
    from matplotlib import colormaps
    import hashlib
    import re
    labels = sorted(set(groups.values()))
    cmap = colormaps[palette]
    result = {}
    for name in labels:
        match = re.search(r'(?:^| / )Group ([1-9][0-9]*)$', name)
        index = int(match[1])-1 if match else int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big')
        result[name] = cmap(index % 10) if palette == 'tab10' else cmap(.15 + .7*((index * .61803398875) % 1))
    return result


def energy_record_colors(records, groups, group_colors, variable='E-field'):
    """Light-to-dark shades computed from the full comparison membership."""
    from matplotlib.colors import to_rgba
    result = {}
    for group in set(groups.values()):
        members = [r for r in records if groups.get(r.record_id) == group]
        fields = [r.condition_value('T' if variable == 'Temperature' else variable) for r in members]
        fields = [round(f, 8) if f is not None and np.isfinite(f) else None for f in fields]
        finite = [f for f in fields if f is not None and np.isfinite(f)]
        levels = sorted(set(finite))
        ranks = {value:index/max(1,len(levels)-1) for index,value in enumerate(levels)}
        base = np.asarray(to_rgba(group_colors[group])[:3])
        for record, field in zip(members, fields):
            t = ranks.get(field,.5) if len(levels)>1 else .5
            # Avoid nearly white points; preserve the group's hue.
            light = .55*base + .45
            dark = .65*base
            rgb = light*(1-t) + dark*t
            # Floating-point blending can produce 1.0000000000000002 for
            # saturated tab10 channels; Matplotlib rejects that color.
            result[record.record_id] = tuple(float(v) for v in np.clip(rgb, 0., 1.)) + (1.,)
    return result


def draw_energy_slope_panels(figure, records, groups, metrics, branches,
                             tolerance_mev=5., palette='tab10', group_colors=None, manual_record_ids=(), record_colors=None, panel="both"):
    """Return point artists with exact record identities for hover and picking."""
    from core.mcd_extract import SLOPE_METRICS
    from matplotlib.lines import Line2D
    figure.clear()
    show_energy = panel != "slopes"
    show_slopes = bool(metrics) and panel != "energy"
    axes = np.atleast_1d(figure.subplots(int(show_energy)+int(show_slopes),1,sharex=True)).tolist()
    slope_axis = axes[-1]
    colors = energy_group_colors(groups,palette)
    colors.update(group_colors or {})
    shades = energy_record_colors(records, groups, colors)
    shades.update(record_colors or {})
    point_artists = {}
    visible_groups = set()
    line_styles = {'near_zero':'-', 'low_minus_negative':'--', 'low_minus_positive':':'}
    for metric in ((('window_energy',) if show_energy else ()) + (tuple(metrics) if show_slopes else ())):
        axis = axes[0] if metric=='window_energy' else slope_axis
        for group in sorted(set(groups.values())):
            members = sorted([r for r in records if groups.get(r.record_id)==group],
                             key=lambda r:(r.condition_value('E-field') if r.condition_value('E-field') is not None else np.inf,r.record_id))
            if not members:
                continue
            x = np.asarray([r.condition_value('E-field') for r in members],float)
            edges = safe_energy_edges(members,tolerance_mev,manual_record_ids)
            for branch in branches:
                y = np.asarray([r.center_ev if metric=='window_energy' else r.slope(branch,metric) for r in members],float)
                if np.any(np.isfinite(x) & np.isfinite(y)):
                    visible_groups.add(group)
                inc = branch=='B increasing'
                for a,b in edges:
                    if np.isfinite(y[a]) and np.isfinite(y[b]):
                        axis.plot(x[[a,b]],y[[a,b]],color=shades[members[b].record_id],alpha=.65,
                                  linestyle=line_styles.get(metric,'-'),linewidth=1,zorder=1)
                point_colors = [shades[r.record_id] for r in members]
                artist = axis.scatter(x,y,marker='o' if inc else 's',s=25,
                    facecolors=point_colors if inc else 'none',edgecolors=point_colors,
                    picker=7,zorder=3)
                point_artists[artist] = (tuple(members),metric,branch)
        axis.set_title('Window energy vs E-field (saved integration center)' if metric=='window_energy' else 'MCD slopes vs E-field',fontsize=10,loc='left')
        axis.set_ylabel('Energy (eV)' if metric=='window_energy' else 'Slope / slope difference\n(MCD/T)',fontsize=9)
        axis.grid(alpha=.2)
        if metrics and metric == metrics[0]:
            axis.axhline(0,color='#666',linewidth=.6)
    handles = [Line2D([],[],color=color,marker='o',linestyle='none',label=name) for name,color in colors.items() if name in visible_groups]
    handles += [Line2D([],[],color='#555',marker='o' if b=='B increasing' else 's',
                      markerfacecolor='#555' if b=='B increasing' else 'none',linestyle='none',
                      label='Inc' if b=='B increasing' else 'Dec') for b in branches]
    if handles:
        axes[0].legend(handles=handles,loc='upper right',fontsize=7,ncol=min(4,len(handles)))
    if show_slopes:
        if not show_energy and handles:
            axes[0].add_artist(axes[0].get_legend())
        slope_axis.legend(handles=[Line2D([],[],color='#555',linestyle=line_styles[m],label=SLOPE_METRICS[m]) for m in metrics],loc='upper left',fontsize=7)
        if not any(r.slope(b,m) is not None for r in records for b in branches for m in metrics):
            slope_axis.text(.5,.5,'N/A: no matching fits',ha='center',transform=slope_axis.transAxes)
    for axis in axes:
        low, high = axis.get_ylim()
        axis.set_ylim(low, high + .18*(high-low))
    axes[0].text(0.01, .02, 'Within group: light to dark = increasing E-field', transform=axes[0].transAxes, fontsize=7, color='#555')
    axes[-1].set_xlabel('E-field (V)')
    figure.align_ylabels(axes)
    figure.tight_layout(pad=1.2,h_pad=.6)
    return point_artists


def temperature_curve_legend(figure, records, groups, colors):
    """Exact temperature swatches per group, shared by preview and export."""
    from matplotlib.lines import Line2D
    handles = []
    for group in sorted({groups[r.record_id] for r in records}):
        members = sorted([r for r in records if groups[r.record_id] == group],
                         key=lambda r: r.condition_value('T') if r.condition_value('T') is not None else np.inf)
        seen = set()
        for record in members:
            t = record.condition_value('T')
            label = f'{group} · {t:g} K' if t is not None else f'{group} · T unknown'
            if label in seen:
                continue
            seen.add(label)
            handles.append(Line2D([], [], color=colors[record.record_id], linewidth=2, label=label))
    # Measure the legend rather than assuming it fits a fixed right margin.
    legend = figure.legend(handles=handles, loc='center left', bbox_to_anchor=(1, .5),
                           fontsize=7, title='Group / Temperature', title_fontsize=8,
                           ncol=max(1, (len(handles)+19)//20))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    box = legend.get_window_extent(renderer)
    width = box.width / figure.bbox.width + .025
    if width <= .38 and box.height <= figure.bbox.height * .85:
        legend.set_bbox_to_anchor((1-width, .5))
        figure.tight_layout(rect=(0, .04, 1-width, .92))
    else:
        legend.remove()
        columns = max(1, int(figure.get_figwidth() / 2.4))
        legend = figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01),
                               fontsize=7, title='Group / Temperature', title_fontsize=8, ncol=columns)
        figure.canvas.draw()
        box = legend.get_window_extent(figure.canvas.get_renderer())
        # Large selections stay inside the canvas; group filtering remains available.
        if box.height > figure.bbox.height * .45 or box.width > figure.bbox.width * .96:
            scale = min(.45*figure.bbox.height/box.height, .96*figure.bbox.width/box.width)
            for text in legend.get_texts():
                text.set_fontsize(max(3, 7*scale))
            figure.canvas.draw()
            box = legend.get_window_extent(figure.canvas.get_renderer())
        figure.tight_layout(rect=(0, box.height/figure.bbox.height+.035, 1, .92))


def draw_temperature_slope_panels(figure, records, groups, metrics, branches,
                                  palette='tab10', group_colors=None):
    """Separate metric panels; fixed group hue and branch style, never cross groups."""
    from core.mcd_extract import SLOPE_METRICS
    from matplotlib.lines import Line2D
    figure.clear()
    axes = np.atleast_1d(figure.subplots(max(1, len(metrics)), 1, sharex=True))
    colors = energy_group_colors(groups, palette)
    colors.update(group_colors or {})
    artists = {}
    visible = sorted({groups[r.record_id] for r in records})
    for axis, metric in zip(axes, metrics):
        for group in visible:
            members = sorted([r for r in records if groups[r.record_id] == group],
                             key=lambda r: r.condition_value('T') if r.condition_value('T') is not None else np.inf)
            x = np.asarray([r.condition_value('T') for r in members], float)
            for branch in branches:
                inc = branch == 'B increasing'
                y = np.asarray([r.slope(branch, metric) for r in members], float)
                # Duplicate temperatures are ambiguous; show points without connecting.
                unique = len(set(np.round(x, 8))) == len(x)
                axis.plot(x, y, color=colors[group], linestyle=('-' if inc else '--') if unique else 'none', linewidth=1)
                artist = axis.scatter(x, y, color=colors[group], marker='o' if inc else 's',
                                      facecolors=colors[group] if inc else 'none', picker=7)
                artists[artist] = (tuple(members), metric, branch)
        axis.set_ylabel('Slope (MCD/T)')
        axis.set_title(SLOPE_METRICS[metric], fontsize=10, loc='left')
        axis.axhline(0, color='#777', linewidth=.6)
        axis.grid(alpha=.2)
    handles = [Line2D([], [], color=colors[g], linewidth=2, label=g) for g in visible]
    handles += [Line2D([], [], color='#555', marker='o' if b == 'B increasing' else 's',
                       linestyle='-' if b == 'B increasing' else '--', label=b) for b in branches]
    axes[0].legend(handles=handles, fontsize=7)
    if not metrics:
        axes[0].text(.5, .5, 'Select a slope metric.', ha='center', transform=axes[0].transAxes)
    axes[-1].set_xlabel('Temperature (K)')
    figure.tight_layout()
    return artists


def point_description(record, metric, branch, group):
    from core.mcd_extract import SLOPE_METRICS
    label = 'Window center' if metric=='window_energy' else SLOPE_METRICS[metric]
    value = record.center_ev if metric=='window_energy' else record.slope(branch,metric)
    number = 'N/A' if value is None else f'{value:.12g}'
    extra = ''
    if metric == 'window_energy':
        for key,name in SLOPE_METRICS.items():
            slope = record.slope(branch,key)
            extra += f'\n{name}: ' + ('N/A' if slope is None else f'{slope:.12g} MCD/T')
    return (f'{group} | {branch}\nWindow center: {record.center_ev:.12g} eV\n'
            f'E-field: {record.condition_value("E-field")} V | T: {record.condition_value("T")} K | Width: {record.width_mev:.12g} meV\n'
            f'{label}: {number} {"eV" if metric=="window_energy" else "MCD/T"}{extra}')
