"""Plot-ready PL and DRR analysis tables with stable branch/scan identities."""
from pathlib import Path
import numpy as np

from core.analysis_workbook import AnalysisTable, analysis_filename, write_analysis_workbook
from core.power_peaks import peak_model
from core.power_multi_peaks import multi_peak_model


FIT_COLUMNS = [('center_eV', 'fit_center'), ('center_err_eV', 'center_error_ev'),
    ('FWHM_meV', 'fwhm_mev'), ('FWHM_err_meV', 'fwhm_error_mev'),
    ('height', 'height'), ('height_err', 'height_error'),
    ('fit_area_full', 'fit_area_full'), ('fit_area_full_err', 'fit_area_full_error'),
    ('fit_area_roi', 'area'), ('fit_area_roi_err', 'area_error')]


def peak_tables(dataset):
    result = dataset['result']; cube = dataset['cube']
    settings = result['settings']
    points = result['points']
    branches = list(result.get('branches', dataset.get('branches', [])))
    known = {b['id'] for b in branches}
    for b in dataset.get('branches', []):
        if b['id'] not in known and any(p['branch_id'] == b['id'] for p in points):
            branches.append(b); known.add(b['id'])
    catalog = list(dataset.get('branch_catalog', result.get('branch_catalog', [])))
    catalog_ids = {b['id'] for b in catalog}
    catalog.extend(b for b in branches if b['id'] not in catalog_ids)
    aliases = {}; counter = {'peak': 0, 'dip': 0}
    for b in catalog:
        pol = b['polarity']; counter[pol] += 1
        aliases[b['id']] = ('P' if pol == 'peak' else 'D') + str(counter[pol])
    rows = [i for i, y in enumerate(cube.gate) if settings['y_min'] <= y <= settings['y_max']]
    lookup = {}
    for p in points:
        lookup.setdefault((p['row_index'], p['branch_id']), []).append(p)
    fits = {f['row_index']: f for f in result.get('fits', [])}
    scan = cube.gate_label or 'Scan'
    source = str(dataset.get('source', ''))
    pl = dataset['kind'] == 'PL'

    def results_table(name, selected):
        headers = [scan, 'Source_row_index']
        if pl:
            headers += ['Spectrum_area_roi']
        for b in selected:
            prefix = aliases[b['id']]
            labels = ['detected_eV', 'detection_status']
            if pl:
                labels += [c[0] for c in FIT_COLUMNS] + ['fit_status', 'uncertainty_status', 'RMS']
            else:
                labels += ['signal', 'half_prominence_width_meV', 'SNR']
            headers.extend(prefix + '_' + s for s in labels)
        def values():
            for row in rows:
                out = [float(cube.gate[row]), row]
                if pl:
                    out.append(fits.get(row, {}).get('spectrum_area_roi'))
                for b in selected:
                    candidates = lookup.get((row, b['id']), [])
                    p = candidates[0] if len(candidates) == 1 else {}
                    status = p.get('status', 'ambiguous' if candidates else 'not detected')
                    accepted = status == 'accepted'
                    out += [p.get('energy') if accepted else None, status]
                    if pl:
                        good = p.get('fit_status') == 'ok'
                        out += [p.get(k) if good else None for _, k in FIT_COLUMNS]
                        out += [p.get('fit_status', 'not fitted'), p.get('uncertainty_status', 'unavailable'), p.get('rms')]
                    else:
                        signal = float(np.interp(p['energy'], cube.energy, cube.Z[row])) if accepted else None
                        out += [signal, p.get('width_mev') if accepted else None, p.get('snr') if accepted else None]
                yield out
        return AnalysisTable(name, headers, values())

    tables = [results_table('Results', branches)] if pl else [
        results_table('Peaks', [b for b in branches if b['polarity'] == 'peak']),
        results_table('Dips', [b for b in branches if b['polarity'] == 'dip'])]
    detail_fields = ['row_index', 'y', 'branch_id', 'polarity', 'energy', 'status', 'segment',
        'snr', 'prominence', 'width_mev', 'fit_center', 'center_error_ev', 'fwhm_mev', 'fwhm_error_mev',
        'height', 'height_error', 'fit_area_full', 'fit_area_full_error', 'area', 'area_error',
        'fit_status', 'uncertainty_status', 'rms']
    tables.append(AnalysisTable('Details', ['Feature', *detail_fields],
        ([aliases.get(p['branch_id'], p['branch_id']), *[p.get(k) for k in detail_fields]] for p in points)))

    scope = dataset.get('export_curve_scope', 'current')
    if scope not in {'current', 'all'}:
        raise ValueError('Choose current or all analyzed spectra for curve export.')
    curve_rows = rows if scope == 'all' else [int(dataset.get('row', 0))]
    curve_rows = [r for r in curve_rows if 0 <= r < len(cube.gate)]
    def curves():
        for row in curve_rows:
            fit = fits.get(row)
            lo, hi = (fit['x_min'], fit['x_max']) if fit else (settings['x_min'], settings['x_max'])
            mask = (cube.energy >= lo) & (cube.energy <= hi)
            x, y = cube.energy[mask], cube.Z[row, mask]
            baseline = total = None
            if fit:
                p = fit['parameters']; baseline = p[0] + p[1] * (x - fit['origin'])
                total = multi_peak_model(x, p, model=fit['model'], origin=fit['origin'])
            for j, e in enumerate(x):
                yield [row, float(cube.gate[row]), float(e), float(y[j]),
                    baseline[j] if fit else None, total[j] if fit else None,
                    y[j] - total[j] if fit else None, fit.get('status') if fit else 'not fitted']
    tables.append(AnalysisTable('Curves', ['Source_row_index', scan, 'Energy_eV', 'Measured_signal',
        'Baseline', 'Total_fit', 'Residual', 'Fit_status'], curves()))
    if pl:
        def components():
            for row in curve_rows:
                fit = fits.get(row)
                if not fit:
                    continue
                x = cube.energy[(cube.energy >= fit['x_min']) & (cube.energy <= fit['x_max'])]
                for i, identity in enumerate(fit['branch_ids']):
                    y = peak_model(x, 0., 0., *fit['parameters'][2+3*i:5+3*i], model=fit['model'], origin=fit['origin'])
                    for e, v in zip(x, y):
                        yield [row, aliases.get(identity, identity), identity, float(e), float(v)]
        tables.append(AnalysisTable('Components', ['Source_row_index', 'Feature', 'branch_id', 'Energy_eV', 'Fitted_signal_above_baseline'], components()))
    metadata = [
        ['Schema', 'peak-analysis-xlsx/1'], ['Source filename', Path(source).name], ['Source path', source],
        ['Dataset key', dataset['key']], ['Kind / channel', dataset['kind'] + ' / ' + dataset['channel']],
        ['Signal label', cube.cbar_label], ['Scan label', scan], ['Scan unit', getattr(cube, 'gate_unit', '')],
        ['Method', result['method']], ['Analysis scope', dataset.get('analysis_scope', 'map')],
        ['Curve scope', scope], ['Curve source rows (zero-based)', curve_rows],
        ['Missing values', 'Blank numeric cells mean unavailable, not zero. Detection and fit status are separate.'],
        ['Uncertainties', 'Approximate 1-sigma least-squares fit standard errors; excludes calibration/systematic uncertainty.'],
        ['Spectrum_area_roi', 'Once per spectrum: measured signal minus fitted common linear baseline, integrated over finite sampled fit ROI.'],
        ['fit_area_full', 'Analytic whole-domain area of an individual fitted component; excludes baseline; extrapolates model tails.'],
        ['fit_area_roi / legacy area', 'Fitted component numerically integrated on the sampled fit domain; legacy values are unchanged.'],
        ['Area units', 'Signal units times eV. No photon-count, exposure or wavelength-density correction is implied.'],
        ['Finite sampling', 'Trapezoid integration connects retained finite samples, including across internal missing samples.'],
        ['Detection width', 'Full width at half prominence; distinct from fitted FWHM.'],
        ['Identity', 'Stable branch_id maps to each feature column. P1 and D1 do not imply pairing.'],
    ]
    metadata.extend(['Setting: ' + k, v] for k, v in settings.items())
    metadata.extend(['Branch: ' + aliases[b['id']], {'id': b['id'], 'name': b.get('name'), 'polarity': b['polarity']}] for b in branches)
    metadata.extend(['Notice', text] for text in result.get('notices', []))
    tables.append(AnalysisTable('Settings', ['Setting', 'Value'], metadata))
    return tables


def export_peak_workbook(folder, dataset, *, cancelled=None):
    suffix = '_PLfit' if dataset['kind'] == 'PL' else '_DRRpeaks'
    name = analysis_filename(dataset.get('source') or dataset['name'], suffix, folder)
    return write_analysis_workbook(Path(folder) / name, peak_tables(dataset), cancelled=cancelled)
