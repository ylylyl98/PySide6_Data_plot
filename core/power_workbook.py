"""Reproducible power-sweep workbook; never changes the intensity definition."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import math
import numpy as np
from core.analysis_workbook import AnalysisTable, analysis_filename, write_analysis_workbook


PARAMETERS = [('fit_area_full', 'fit_area_full'), ('fit_area_full_err', 'fit_area_full_error'),
    ('area_legacy', 'area'), ('area_legacy_err', 'area_error'), ('center_eV', 'center_ev'),
    ('center_err_eV', 'center_error_ev'), ('FWHM_meV', 'fwhm_mev'), ('FWHM_err_meV', 'fwhm_error_mev'),
    ('height', 'height'), ('height_err', 'height_error'), ('RMS', 'rms'),
    ('fit_status', 'status'), ('uncertainty_status', 'uncertainty_status')]


def _components(fit):
    return getattr(fit, 'components', (fit,))


def _finite(value):
    return value is not None and math.isfinite(float(value))


def power_tables(payload):
    results = payload['results']; records = payload.get('records', {})
    count = max((len(_components(f)) for fits in results.values() for f in fits), default=0)

    def provenance(channel, index):
        entries = records.get(channel, ())
        record = entries[index] if index < len(entries) else None
        return [getattr(record, 'file_name', None), getattr(record, 'row_index', None),
                getattr(record, 'source_provenance', None)]

    headers = ['Channel', 'Spectrum_index', 'Power_uW', 'Source_file', 'Source_row', 'Source_provenance',
               'Spectrum_area_roi', 'Fit_min_eV', 'Fit_max_eV', 'Model', 'Manual_refit']
    headers += [f'P{i+1}_{name}' for i in range(count) for name, _ in PARAMETERS]
    def rows():
        for channel, fits in results.items():
            for index, fit in enumerate(fits):
                components = _components(fit); x = getattr(fit, 'x', None)
                settings = getattr(fit, 'settings', payload['settings'])
                row = [channel, index, fit.power_uw, *provenance(channel, index),
                    getattr(fit, 'spectrum_area_roi', None), x[0] if x is not None and len(x) else None,
                    x[-1] if x is not None and len(x) else None, settings.model, getattr(fit, 'manual', False)]
                for i in range(count):
                    peak = components[i] if i < len(components) else None
                    for _, field in PARAMETERS:
                        row.append(getattr(peak, field, None))
                yield row
    tables = [AnalysisTable('Results', headers, rows())]
    law_columns = ['Record_type', 'Record_id', 'channel', 'peak', 'name', 'intensity_metric', 'area_method',
        'lower', 'upper', 'observed_lower', 'observed_upper', 'alpha', 'alpha_error', 'amplitude',
        'n', 'excluded', 'log_rms', 'method', 'suggested', 'crosses_sources', 'captured_utc', 'input_fingerprint']
    law_records = [('applied', i, law) for i, law in enumerate(payload.get('power_laws', []))]
    law_records += [('retained', i, law) for i, law in enumerate(payload.get('power_law_history', []))]
    def laws():
        for kind, identity, law in law_records:
            data = dict(intensity_metric='area', area_method='legacy_measured_axis_component', **{k:v for k,v in law.items() if k not in {'intensity_metric','area_method'}})
            data.update({k:law[k] for k in ('intensity_metric', 'area_method') if k in law})
            yield [kind, identity, *[data.get(k) for k in law_columns[2:]]]
    tables.append(AnalysisTable('PowerLaw', law_columns, laws()))
    def fit_data():
        for kind, identity, law in law_records:
            metric = law.get('intensity_metric', 'area')
            if metric not in {'area', 'fit_area_full'}:
                raise ValueError('Unsupported saved power-law intensity metric: ' + str(metric))
            channel = law['channel']; peak_index = int(law['peak']) - 1
            if kind == 'retained':
                captured = law.get('input_rows')
                if captured is None:
                    yield [kind, identity, channel, None, law['peak'], None, None, None, metric,
                           False, 'historical input snapshot unavailable; not reconstructed from current spectra', None, None, None]
                else:
                    for row in captured:
                        reason = row['inclusion_reason']
                        yield [kind, identity, channel, row['spectrum_index'], law['peak'], row['power'], row['intensity'],
                               row.get('intensity_error'), metric, reason == 'included', reason, None, None, row.get('source')]
                continue
            for index, fit in enumerate(results.get(channel, [])):
                components = _components(fit)
                peak = components[peak_index] if 0 <= peak_index < len(components) else None
                intensity = getattr(peak, metric, None)
                uncertainty = getattr(peak, 'area_error' if metric == 'area' else 'fit_area_full_error', None)
                power = fit.power_uw
                reason = ('missing component' if peak is None else 'spectrum fit invalid' if peak.status != 'ok'
                    else 'nonfinite or nonpositive power' if not _finite(power) or power <= 0
                    else 'nonfinite or nonpositive intensity' if not _finite(intensity) or intensity <= 0
                    else 'outside selected power range' if not law['lower'] <= power <= law['upper'] else 'included')
                yield [kind, identity, channel, index, law['peak'], power, intensity, uncertainty, metric,
                       reason == 'included', reason, *provenance(channel, index)]
    tables.append(AnalysisTable('FitData', ['Record_type', 'Record_id', 'Channel', 'Spectrum_index', 'Peak',
        'Power_uW', 'Intensity', 'Intensity_fit_SE', 'Intensity_metric', 'Included', 'Reason',
        'Source_file', 'Source_row', 'Source_provenance'], fit_data()))
    scope = payload.get('export_curve_scope', 'current')
    if scope not in {'current', 'all'}:
        raise ValueError('Choose current or all analyzed spectra for curve export.')
    selected_channel = payload.get('export_curve_channel') or next(iter(results), '')
    selected_index = int(payload.get('export_curve_row', 0))
    selected = [(channel, i, fit) for channel, fits in results.items() for i, fit in enumerate(fits)
        if scope == 'all' or (channel == selected_channel and i == selected_index)]
    def curves():
        for channel, index, fit in selected:
            x, y, residual = (getattr(fit, k, None) for k in ('x', 'fitted', 'residual'))
            if x is None or y is None or residual is None:
                continue
            baseline = getattr(fit, 'baseline', None)
            for j, energy in enumerate(x):
                yield [channel, index, fit.power_uw, energy, y[j]+residual[j],
                    baseline[j] if baseline is not None else None, y[j], residual[j], fit.status]
    tables.append(AnalysisTable('Curves', ['Channel', 'Spectrum_index', 'Power_uW', 'Energy_eV',
        'Measured_signal', 'Baseline', 'Total_fit', 'Residual', 'Fit_status'], curves()))
    def components():
        for channel, index, fit in selected:
            x = getattr(fit, 'x', None)
            if x is None:
                continue
            for i, values in enumerate(getattr(fit, 'component_curves', ())):
                for energy, value in zip(x, values):
                    yield [channel, index, fit.power_uw, i+1, energy, value]
    tables.append(AnalysisTable('Components', ['Channel', 'Spectrum_index', 'Power_uW', 'Peak',
        'Energy_eV', 'Fitted_signal_above_baseline'], components()))
    def sources():
        for channel, fits in results.items():
            for index, fit in enumerate(fits):
                entries = records.get(channel, ())
                record = entries[index] if index < len(entries) else None
                data = asdict(record) if is_dataclass(record) else str(record) if record else None
                yield [channel, index, fit.power_uw, *provenance(channel, index), data]
    tables.append(AnalysisTable('Sources', ['Channel', 'Spectrum_index', 'Power_uW', 'Source_file',
        'Source_row', 'Source_provenance', 'Acquisition_record'], sources()))
    settings = [
        ['Schema', 'power-analysis-xlsx/1'], ['Curve scope', scope], ['Curve channel', selected_channel],
        ['Curve spectrum index', selected_index], ['Fit settings', asdict(payload['settings'])],
        ['Applied ranges', payload.get('power_law_ranges', [])], ['Background constant', payload.get('background')],
        ['Intensity displayed', payload.get('metric')], ['Original areas preserved', True],
        ['area_legacy', 'Multi-peak: fitted component trapezoid integral over finite measured fit axis. Legacy single-peak: measured signal minus fitted baseline over sampled fit window.'],
        ['fit_area_full', 'Analytic whole-domain fitted peak area, excludes baseline, extrapolates model tails; independent of legacy area.'],
        ['Spectrum_area_roi', 'Measured spectrum minus shared fitted linear baseline. Finite sampled fit domain, once per spectrum.'],
        ['Area units', 'Signal units times eV. This export adds no exposure normalization or spectral-density Jacobian correction.'],
        ['Normalization', 'Use acquisition/source metadata; unknown values remain unknown. No inferred exposure correction.'],
        ['Fit uncertainty', 'Approximate 1-sigma covariance standard errors, including parameter correlations; excludes calibration/systematic uncertainty.'],
        ['Legacy single-peak area error', 'Baseline covariance contribution only; excludes measurement-noise integration uncertainty.'],
        ['Power-law uncertainty', 'OLS log-log slope standard error; intensity SE is not used as regression weight.'],
        ['Retained intervals', 'Explicitly retained fits use their captured input rows. Old histories without snapshots are marked unavailable, never reconstructed.'],
        ['Finite sampling', 'Trapezoid integration connects retained finite samples, including across internal missing samples.'],
        ['Missing values', 'Unavailable numeric values are blank. Repeated powers and source rows remain distinct.'],
        ['Peak identity', 'Peak number follows the fitted component/seed identity; not sorted by height per power.'],
    ]
    tables.append(AnalysisTable('Settings', ['Setting', 'Value'], settings))
    return tables


def export_power_workbook(folder, payload, *, name=None, cancelled=None):
    if name is None:
        files = list(dict.fromkeys(str(getattr(record, 'file_name', '')) for rows in payload.get('records', {}).values() for record in rows))
        source = files[0] if len(files) == 1 and files[0] else Path(folder).name
        name = analysis_filename(source, '_PowerAnalysis', folder)
    return write_analysis_workbook(Path(folder)/name, power_tables(payload), cancelled=cancelled)
