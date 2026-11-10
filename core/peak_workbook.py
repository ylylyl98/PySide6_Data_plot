"""Plot-ready PL and DRR analysis tables with stable branch/scan identities."""
from pathlib import Path
import numpy as np

from core.analysis_workbook import AnalysisTable, analysis_filename, write_analysis_workbook
from core.pl_fit_models import evaluate_fit, evaluate_component


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
        'fit_status', 'uncertainty_status', 'rms', 'fit_quality_status', 'candidate_origin', 'provisional',
        'detection_contrast', 'curvature_width_mev', 'curvature_score', 'scale_support',
        'fit_local_residual_sigma', 'fit_height_to_local_noise',
        'manual', 'fit_center_fixed', 'fit_center_min', 'fit_center_max']
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
                total = evaluate_fit(x, fit)
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
                    y = evaluate_component(x, fit, i)
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
        ['Fixed centers', 'A user-fixed center is a constraint, not a measured value with zero uncertainty. See fit_center_fixed and uncertainty_status in Details.'],
        ['Asymmetric Lorentzian', 'Only the strongest seeded component has separate left/right widths. Reported width is total FWHM; asymmetry is bounded to [-0.8, 0.8].'],
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
    metadata.extend(['Manual edit', edit] for edit in dataset.get('manual_peak_edits', []))
    metadata.extend([f"Fit row {row}", {key: fit.get(key) for key in
        ('model', 'asymmetric_branch_id', 'parameters', 'n_vary', 'fixed_center_branch_ids', 'status', 'fit_quality')}]
        for row, fit in fits.items())
    tables.append(AnalysisTable('Settings', ['Setting', 'Value'], metadata))
    return tables


def compact_peak_tables(dataset):
    """Build numeric, scan-aligned tables for plotting filtered features."""
    result, cube = dataset['result'], dataset['cube']
    points = result.get('points', [])
    branches = list(result.get('branches', dataset.get('branches', [])))
    known = {b['id'] for b in branches}
    for branch in dataset.get('branches', []):
        if branch['id'] not in known and any(p['branch_id'] == branch['id'] for p in points):
            branches.append(branch)
            known.add(branch['id'])
    catalog = list(dataset.get('branch_catalog', result.get('branch_catalog', [])))
    catalog_ids = {b['id'] for b in catalog}
    catalog.extend(b for b in branches if b['id'] not in catalog_ids)
    aliases, counts = {}, {'peak': 0, 'dip': 0}
    for branch in catalog:
        polarity = branch['polarity']
        counts[polarity] += 1
        aliases[branch['id']] = ('P' if polarity == 'peak' else 'D') + str(counts[polarity])
    enabled = {branch['id']: branch.get('enabled', True) for branch in dataset.get('branches', [])}
    accepted_ids = {point['branch_id'] for point in points if point.get('status') == 'accepted'}
    branches = [branch for branch in branches if enabled.get(branch['id'], branch.get('enabled', True))
                and branch['id'] in accepted_ids]
    lookup = {}
    error_available = {name: set() for name in ('center_error_ev', 'fwhm_error_mev', 'height_error', 'fit_area_full_error')}
    for point in points:
        lookup.setdefault((point['row_index'], point['branch_id']), []).append(point)
        if point.get('status') == 'accepted' and point.get('fit_status') == 'ok':
            for field, identities in error_available.items():
                if point.get(field) is not None:
                    identities.add(point['branch_id'])
    settings = result['settings']
    rows = [i for i, y in enumerate(cube.gate) if settings['y_min'] <= y <= settings['y_max']]
    scan = cube.gate_label or 'Scan'
    unit = getattr(cube, 'gate_unit', '') or ''
    if unit and f'({unit})'.casefold() not in scan.casefold() and f'[{unit}]'.casefold() not in scan.casefold():
        scan += ' (' + unit + ')'

    def table(name, selected, field, unit_label, error=None):
        headers = [scan]
        error_columns = {}
        for branch in selected:
            alias = aliases[branch['id']]
            name_value = str(branch.get('name') or '')
            import re
            label = alias if not name_value or re.fullmatch(r'[PD]\d+', name_value) else alias + ' - ' + name_value
            headers.append(f'{label} ({unit_label})')
            has_error = bool(error and branch['id'] in error_available[error])
            error_columns[branch['id']] = has_error
            if has_error:
                headers.append(f'{label} SE ({unit_label})')
        def values():
            for row in rows:
                output = [float(cube.gate[row])]
                for branch in selected:
                    matches = lookup.get((row, branch['id']), [])
                    point = matches[0] if len(matches) == 1 else {}
                    accepted = point.get('status') == 'accepted'
                    good = accepted and (field == 'energy' or point.get('fit_status') == 'ok')
                    output.append(point.get(field) if good else None)
                    if error_columns[branch['id']]:
                        output.append(point.get(error) if good else None)
                yield output
        return AnalysisTable(name, headers, values())

    if dataset['kind'] == 'DRR':
        return [table('Peaks', [b for b in branches if b['polarity'] == 'peak'], 'energy', 'eV'),
                table('Dips', [b for b in branches if b['polarity'] == 'dip'], 'energy', 'eV')]
    tables = [table('Detected_peaks', branches, 'energy', 'eV')]
    if result.get('fits') or any(p.get('fit_status') not in (None, '', 'not fitted') for p in points):
        tables.extend([table('Fitted_peaks', branches, 'fit_center', 'eV', 'center_error_ev'),
                       table('FWHM', branches, 'fwhm_mev', 'meV', 'fwhm_error_mev'),
                       table('Height', branches, 'height', cube.cbar_label or 'signal', 'height_error'),
                       table('Area', branches, 'fit_area_full', 'eV*signal', 'fit_area_full_error')])
    return tables


def export_peak_workbook(folder, dataset, *, cancelled=None, filename=None, compact=True):
    suffix = '_PLfit' if dataset['kind'] == 'PL' else '_DRRpeaks'
    name = filename or analysis_filename(dataset.get('source') or dataset['name'], suffix, folder)
    tables = compact_peak_tables(dataset) if compact else peak_tables(dataset)
    return write_analysis_workbook(Path(folder) / name, tables, cancelled=cancelled)
