"""Reproducible plots and numerical outputs for explicit single-scan calibration."""
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import csv
import hashlib
import json
import shutil

import numpy as np

from core.drr_bn_joint import production_d2
from core.drr_bn_settings import BnCalibrationSettings
from core.load_cancellation import check_load_cancelled


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_existing(directory, version):
    manifest = directory / 'results.json'
    doc = json.loads(manifest.read_text(encoding='utf-8'))
    if doc['version_id'] != version.version_id or doc['report_sha256'] != version.report_sha256:
        raise ValueError('Saved BN products do not match this calibration version.')
    paths = {}
    for key, record in doc['outputs'].items():
        path = (directory / record['file']).resolve()
        if path.parent != directory.resolve() or _digest(path) != record['sha256']:
            raise ValueError('Saved BN product changed; choose a new comparison version.')
        paths[key] = str(path)
    paths['metadata_json'] = str(manifest)
    return dict(paths=paths, elapsed_seconds=0., status=doc['status'], directory=str(directory), cached=True)


def save_single_pair_products(payload, version):
    """Export a saved result without activating it or changing any source file.

    Numerical matrices use the application's DAT convention: first column
    energy, one remaining column per acquisition-order gate. Diagnostic maps
    compare original and candidate with the same scale, never smoother SG.
    """
    start = perf_counter()
    if payload.get('report', {}).get('optimization_policy') not in {'single_pair_v1', 'multi_bn_single_v1', 'shared_bn_trend_v1', 'shared_bn_trend_v2'}:
        return dict(paths={}, elapsed_seconds=0., status='not_single_pair', directory='')
    candidate_name = payload.get('selected_candidate')
    candidate = payload.get('candidate_backgrounds', {}).get(candidate_name) if candidate_name != 'original' else None
    if candidate is None:
        return dict(paths={}, elapsed_seconds=0., status='original_retained', directory='')
    if not version.fresh or version.source_identity != payload['source_identity']:
        raise ValueError('BN outputs need a fresh saved comparison for these exact sources.')
    # Export immutable saved bytes, not a caller-modified array with a valid ID.
    from core.drr_bn_store import DrrBnStore
    store = DrrBnStore(payload['source_identity']['root'])
    version = store.get(version.version_id)
    if not version.fresh:
        raise ValueError('BN sources changed before results export.')
    payload = store.load_result(version)
    candidate_name = payload['selected_candidate']
    candidate = payload['candidate_backgrounds'][candidate_name]
    root = Path(payload['source_identity']['root']).resolve()
    parent = root / 'Processed Data' / 'DRR' / 'BN Calibration'
    directory = parent / version.version_id
    if directory.exists():
        return _read_existing(directory, version)
    check_load_cancelled()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / ('.' + version.version_id + '_' + uuid4().hex)
    temporary.mkdir()
    paths = {}
    try:
        energy = np.asarray(payload['energy'], float)
        gates = np.asarray(payload['gates'], float)
        raw = np.asarray(payload['mean_raw'], float)
        backgrounds = [np.asarray(payload['original_background'], float), np.asarray(candidate, float)]
        config = BnCalibrationSettings(**payload['settings'])
        ratios = np.stack([raw / b - 1 for b in backgrounds])
        derivatives, used = production_d2(energy, ratios, config)
        status = 'single_pair' if version.eligible else 'preview_only'
        trend = payload['report'].get('optimization_policy') in {'shared_bn_trend_v1', 'shared_bn_trend_v2'}
        candidate_label = ('Shared trend - numerical checks passed' if version.eligible else 'Shared trend preview') if trend else (
            'Calibrated BN' if version.eligible else 'Candidate only - not applied')
        header = 'Photon energy(eV)\t' + '\t'.join(format(v, '.12g') for v in gates)
        for role, values in [('original_drr', ratios[0]), ('calibrated_drr', ratios[1]),
                             ('original_d2', derivatives[0]), ('calibrated_d2', derivatives[1])]:
            path = temporary / (role + '.dat')
            np.savetxt(path, np.column_stack([energy, values.T]), delimiter='\t',
                       header=header, comments='', fmt='%.12g')
            paths[role + '_dat'] = path
            check_load_cancelled()
        # Keep the full measured wavelength grid so this background can be inspected independently.
        columns = np.asarray(payload['source_identity']['spectral_columns'], float)
        source_energy = 1240 / columns if np.median(columns) > 20 else columns
        order = np.argsort(source_energy, kind='stable')
        values = np.empty_like(candidate); values[order] = candidate
        path = temporary / 'calibrated_background.csv'
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            from core.drr_bn_store import BN_MARKER
            writer.writerow(['Vbg', 'Vtg', 'Vbias', BN_MARKER, *columns])
            writer.writerow([0, 0, 0, version.version_id, *values])
        paths['background_csv'] = path
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from core.typography import matplotlib_font_context

        def draw(path, regions, title):
            check_load_cancelled()
            with matplotlib_font_context():
                fig = Figure(figsize=(12, max(4., 3.3*len(regions))), layout='constrained')
                FigureCanvasAgg(fig)
                axes = fig.subplots(len(regions), 2, squeeze=False)
                sort = np.argsort(gates, kind='stable')
                for row, (lo, hi, product) in enumerate(regions):
                    mask = (energy >= lo) & (energy <= hi)
                    data = ratios if product == 'DR/R' else derivatives
                    clipped = data[:, :, mask]
                    finite = clipped[np.isfinite(clipped)]
                    if product == 'DR/R':
                        lower, upper = map(float, np.percentile(finite, [1, 99]))
                        if upper-lower < 1e-12:
                            lower, upper = lower-1e-12, upper+1e-12
                    else:
                        limit = max(float(np.percentile(np.abs(finite), 98)), 1e-12)
                        lower, upper = -limit, limit
                    for col, label in enumerate(('Original BN', candidate_label)):
                        ax = axes[row, col]
                        mesh = ax.pcolormesh(energy[mask], gates[sort], clipped[col, sort], shading='auto',
                                             cmap='RdBu_r', vmin=lower, vmax=upper, rasterized=True)
                        ax.set(title=f'{label} | {product} | {lo:.3g}-{hi:.3g} eV',
                               xlabel='Energy (eV)', ylabel=f'{payload["source_identity"]["y_axis"]} (V)')
                    fig.colorbar(mesh, ax=list(axes[row]), label='DR/R' if product == 'DR/R' else 'eV$^{-2}$')
                fig.suptitle(title + '\nSame-scan fit; independent repeats not used. Same color scale within each row.', fontsize=11)
                fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=.12)
                fig.clear()
            check_load_cancelled()

        regions = [(float(energy.min()), float(energy.max()), 'DR/R'),
                   (float(energy.min()), float(energy.max()), 'D2')]
        path = temporary / 'comparison.png'
        draw(path, regions, f'BN calibration | SG {used}, polynomial {config.derivative_polyorder}')
        paths['comparison_png'] = path
        path = temporary / 'background_adjustment.png'
        with matplotlib_font_context():
            fig = Figure(figsize=(11, 6), layout='constrained')
            FigureCanvasAgg(fig)
            axes = fig.subplots(2, 1, sharex=True)
            axes[0].plot(energy, backgrounds[0], label='Original BN', linewidth=1)
            axes[0].plot(energy, backgrounds[1], label=candidate_label, linewidth=1)
            axes[0].set(ylabel='Measured intensity', title='Background spectra')
            axes[0].legend()
            axes[1].plot(energy, 100*(backgrounds[1]/backgrounds[0]-1), linewidth=1)
            axes[1].axhline(0, color='0.5', linewidth=.7)
            axes[1].set(xlabel='Energy (eV)', ylabel='Background change (%)',
                        title='Correction at every measured energy point')
            fig.suptitle(candidate_label + ' | same-scan calibration')
            fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=.12)
            fig.clear()
        paths['background_adjustment_png'] = path
        if trend:
            paths.update(_save_trend_comparison(temporary, payload, config))
        recorded_regions = [record['region'] for record in payload['report'].get('candidates', {}).get(
            candidate_name, {}).get('parameters', {}).get('regions', []) if 'region' in record]
        if not recorded_regions:
            recorded_regions = (payload['report'].get('background_regions_ev', []) if trend else
                                payload['report'].get('single_pair_scope', {}).get('effective_regions', []))
        regions = [(max(float(lo), float(energy.min())), min(float(hi), float(energy.max())), product)
                   for lo, hi in recorded_regions if np.count_nonzero((energy >= lo)&(energy <= hi)) >= 2
                   for product in ('DR/R', 'D2')]
        if regions:
            path = temporary / 'diagnostics.png'
            draw(path, regions, 'Local DR/R and D2 comparison - recorded calibration regions')
            paths['diagnostics_png'] = path
        check_load_cancelled()
        doc = dict(schema_version=1, version_id=version.version_id, report_sha256=version.report_sha256,
                   status=status, independent_validation=False,
                   normalization_kind='shared_spectral_trend' if trend else 'bn_texture',
                   interpretation=('Background regions are assumptions. Trend normalization changes absolute DR/R and is not a D2 de-fringing certificate.' if trend else
                                   'Within-scan texture checks do not establish independent physical validation.'),
                   source_identity=payload['source_identity'], settings=payload['settings'],
                   report_path=str(version.report_path),
                   data_layout='First column: photon energy in eV. Other columns: acquisition-order gate values from header.',
                   d2_units='eV^-2', unchanged_raw_inputs=True,
                   comparison_color_limits='Shared across original/candidate: DRR joint percentiles 1/99; D2 symmetric joint absolute percentile 98',
                   outputs={key: dict(file=p.name, sha256=_digest(p)) for key, p in paths.items()})
        (temporary/'results.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
        if not store.get(version.version_id).fresh:
            raise ValueError('BN sources changed during results export.')
        temporary.rename(directory)
        result = _read_existing(directory, version)
        result.update(elapsed_seconds=perf_counter()-start, cached=False)
        return result
    finally:
        # Only this call's new staging directory, verified inside the output parent.
        if temporary.exists() and temporary.resolve().parent == parent.resolve() and temporary.name.startswith('.'+version.version_id+'_'):
            shutil.rmtree(temporary)


def _save_trend_comparison(directory, payload, config):
    """Export both low-order alternatives with their original reference intact."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from core.typography import matplotlib_font_context
    energy, gates = np.asarray(payload['energy']), np.asarray(payload['gates'])
    raw = np.asarray(payload['mean_raw'])
    candidates = payload['candidate_backgrounds']
    names = [name for name in ('original', 'trend_linear', 'trend_quadratic') if name in candidates]
    selected = payload['selected_candidate']
    paths = {}
    path = directory / 'trend_backgrounds.csv'
    np.savetxt(path, np.column_stack([energy, *(candidates[name] for name in names)]),
               delimiter=',', header='energy_eV,' + ','.join(names), comments='', fmt='%.17g')
    paths['trend_backgrounds_csv'] = path
    header = 'Photon energy(eV)\t' + '\t'.join(format(v, '.12g') for v in gates)
    maps = np.stack([raw / candidates[name] - 1 for name in names])
    d2, _ = production_d2(energy, maps, config)
    for i, name in enumerate(names):
        if name == 'original':
            continue
        for product, data in (('drr', maps[i]), ('d2', d2[i])):
            check_load_cancelled()
            path = directory / (name + '_' + product + '.dat')
            np.savetxt(path, np.column_stack([energy, data.T]), delimiter='\t',
                       header=header, comments='', fmt='%.12g')
            paths[name + '_' + product + '_dat'] = path
    with matplotlib_font_context():
        fig = Figure(figsize=(5*len(names), 7), layout='constrained')
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, len(names), squeeze=False)
        order = np.argsort(gates, kind='stable')
        for row, (product, values) in enumerate((('DR/R', maps), ('D2', d2))):
            if row == 0:
                lo, hi = np.percentile(values[np.isfinite(values)], [1, 99])
                hi = max(hi, lo+1e-12)
            else:
                hi = max(float(np.percentile(np.abs(values[np.isfinite(values)]), 98)), 1e-12)
                lo = -hi
            for col, name in enumerate(names):
                mesh = axes[row, col].pcolormesh(energy, gates[order], values[col, order],
                    shading='auto', cmap='RdBu_r', vmin=lo, vmax=hi, rasterized=True)
                axes[row, col].set(title=name.replace('_', ' ') + (' (selected)' if name == selected else ''),
                                  xlabel='Energy (eV)', ylabel=payload['source_identity']['y_axis']+' (V)')
            fig.colorbar(mesh, ax=list(axes[row]), label=product)
        fig.suptitle('Shared BN trend alternatives | same color scale within each row\n'
                     'Background-region assumptions; absolute DR/R changes; no independent validation')
        path = directory / 'trend_models.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=.12)
        fig.clear()
        paths['trend_models_png'] = path
        fig = Figure(figsize=(11, 6), layout='constrained')
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 1, sharex=True)
        axes[0].plot(energy, np.median(raw, axis=0), color='black', label='Median measured spectrum', linewidth=1)
        for name in names:
            axes[0].plot(energy, candidates[name], label=name.replace('_', ' '), linewidth=1)
            axes[1].plot(energy, candidates[name]/candidates['original'], label=name.replace('_', ' '))
        for ax in axes:
            for lo, hi in payload['report'].get('background_regions_ev', []):
                ax.axvspan(lo, hi, color='gray', alpha=.12)
            ax.legend()
        axes[0].set(ylabel='Measured intensity', title='Gray bands: assumed background fit regions')
        axes[1].set(xlabel='Energy (eV)', ylabel='Shared gain relative to measured BN')
        path = directory / 'trend_gain.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=.12)
        fig.clear()
        paths['trend_gain_png'] = path
    return paths
