"""Connect power-law analysis to fitted spectral components and their provenance."""
import numpy as np

from core.power_law import fit_power_law
from core.power_workflow import record_sources


def peak_series(results, records=None):
    series = []
    for channel, fits in results.items():
        count = max((len(f.components) for f in fits), default=0)
        for index in range(count):
            power = np.array([f.power_uw for f in fits])
            area = np.array([f.components[index].area if index < len(f.components) else np.nan for f in fits])
            valid = np.array([index < len(f.components) and f.components[index].status == 'ok' for f in fits])
            origins = [tuple(record_sources(r)) for r in (records or {}).get(channel, ())]
            boundaries = []
            if len(origins) == len(power):
                order = np.argsort(power, kind='stable')
                for a, b in zip(order[:-1], order[1:]):
                    if origins[a] != origins[b] and power[a] > 0 and power[b] > power[a]:
                        boundaries.append(float(np.sqrt(power[a] * power[b])))
            series.append(dict(channel=channel, peak=index + 1,
                name=f'{channel} Peak {index + 1}' if channel != 'Power' else f'Peak {index + 1}',
                power=power, intensity=area, valid=valid, boundaries=boundaries))
    return series


def evaluate_ranges(results, records, ranges):
    available = {(s['channel'], s['peak']): s for s in peak_series(results, records)}
    output = []
    for setting in ranges:
        source = available.get((setting['channel'], setting['peak']))
        if source is None:
            continue
        fit = fit_power_law(source['power'], source['intensity'], setting['lower'], setting['upper'], valid=source['valid'])
        fit.update(channel=source['channel'], peak=source['peak'], name=source['name'],
            suggested=setting.get('suggested', False),
            crosses_sources=any(fit['lower'] < b < fit['upper'] for b in source['boundaries']))
        output.append(fit)
    return output


def draw_power_laws(axis, fits):
    colors = {line.get_label(): line.get_color() for line in axis.lines}
    limits = axis.get_xlim()
    for index, fit in enumerate(fits):
        x = np.geomspace(fit['lower'], fit['upper'], 100)
        color = colors.get(fit['name'], f'C{index % 10}')
        axis.axvspan(fit['lower'], fit['upper'], color=color, alpha=.07)
        axis.plot(x, fit['amplitude'] * x ** fit['alpha'], '--', color=color,
            label=f"{fit['name']}: alpha={fit['alpha']:.3f} +/- {fit['alpha_error']:.3f}")
    axis.set_xlim(limits)
    if fits:
        axis.legend(fontsize=7)
