"""Explicit, collision-safe XLSX/CSV/JSON/PNG export of an analysis snapshot."""

from core.typography import save_figure
import csv
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import copyfileobj

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from core.peak_plotting import build_figure
from core.peak_workspace import atomic_json, validate_result


def export_dataset(folder, dataset, *, cancelled=None, include_figures=True, include_sidecars=False):
    """Publish flat, full-source filenames; return only the files just saved."""
    from core.peak_output_names import output_stem
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    stem = output_stem(dataset)
    roles = dict(workbook='_peak_results.xlsx', settings='_analysis_settings.json')
    if include_sidecars:
        roles['details'] = '_peak_details.csv'
    if include_figures:
        from core.peak_source_figure import source_export_width
        roles['overlay'] = '_peak_overlay.png'
        width = source_export_width(dataset)
        if width > 1:
            roles['overlay_wide'] = f'_peak_overlay_{float(width)}x.png'
        if include_sidecars and dataset['kind'] == 'PL' and (dataset.get('result') or {}).get('fits'):
            roles['trends'] = '_fit_trends.png'
    def check():
        if cancelled and cancelled():
            raise InterruptedError('Peak Analysis export cancelled.')
    check()
    # Staging is temporary and removed on success/failure. Persistent outputs
    # live directly beside the processed source, with readable sequence numbers.
    with TemporaryDirectory(prefix='.peak-export-', dir=root) as temporary:
        staged = Path(temporary)
        names = {role: stem + tail for role, tail in roles.items()}
        _write_files(staged, dataset, names, cancelled=cancelled, include_figures=include_figures)
        number = 1
        while True:
            suffix = '' if number == 1 else f'_{number:02d}'
            all_tails = ('_peak_details.csv', '_analysis_settings.json', '_peak_results.xlsx', '_peak_overlay.png', '_fit_trends.png')
            targets = {role: root / (Path(name).stem + suffix + Path(name).suffix) for role, name in names.items()}
            wide_pattern = re.compile(re.escape(stem + '_peak_overlay_') + r'\d+(?:\.\d+)?x' + re.escape(suffix) + r'\.png\Z')
            if (any((root / (Path(stem + tail).stem + suffix + Path(tail).suffix)).exists() for tail in all_tails)
                    or any(wide_pattern.fullmatch(path.name) for path in root.iterdir())):
                number += 1
                continue
            created = []
            try:
                for role, target in targets.items():
                    check()
                    with target.open('xb') as destination:
                        created.append(target)
                        with (staged / names[role]).open('rb') as source:
                            copyfileobj(source, destination)
                check()
                return [str(path) for path in created]
            except FileExistsError:
                for path in created:
                    path.unlink(missing_ok=True)
                number += 1
            except BaseException:
                for path in created:
                    path.unlink(missing_ok=True)
                raise


def _write_files(target, dataset, names, *, cancelled=None, include_figures=True):
    result = dataset['result']
    if result is None:
        raise ValueError('Find peaks before exporting results.')
    validate_result(dataset, result)
    fields = ['branch', 'branch_id', 'row_index', 'y', 'energy', 'polarity', 'status', 'segment',
              'candidate_id', 'prominence', 'prominence_fraction', 'width_mev',
              'noise_sigma', 'snr', 'neighbor_support', 'neighbor_available', 'neighbor_required',
              'fit_center', 'center_error_ev', 'fit_status', 'fwhm_mev', 'fwhm_error_mev',
              'height', 'height_error', 'area', 'area_error', 'fit_area_full', 'fit_area_full_error',
              'uncertainty_status', 'rms', 'fit_quality_status', 'candidate_origin', 'provisional',
              'detection_contrast', 'curvature_width_mev', 'curvature_score', 'scale_support',
              'fit_local_residual_sigma', 'fit_height_to_local_noise',
              'manual', 'fit_center_fixed', 'fit_center_min', 'fit_center_max']
    branch_names = {b['id']: b['name'] for b in dataset['branches']}
    if 'details' in names:
        with (target/names['details']).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader()
            for p in result['points']:
                writer.writerow({**p, 'branch': branch_names.get(p['branch_id'], p['branch_id'])})
    if 'settings' in names:
        atomic_json(target/names['settings'], {k: v for k, v in dataset.items() if k != 'cube'})
    from core.peak_workbook import export_peak_workbook
    export_peak_workbook(target, dataset, cancelled=cancelled, filename=names['workbook'])
    if not include_figures:
        return str(target)
    from core.peak_source_figure import render_source_figure
    from matplotlib import pyplot as plt
    for role, width in (('overlay', 1.0), ('overlay_wide', None)):
        if role not in names:
            continue
        if cancelled and cancelled():
            raise InterruptedError('Peak Analysis export cancelled.')
        fig = render_source_figure(dataset, width_scale=width)
        try:
            save_figure(fig, target/names[role], dpi=fig.dpi, facecolor=fig.get_facecolor(),
                        edgecolor='none', bbox_inches=fig.bbox_inches, pad_inches=0)
        finally:
            plt.close(fig)
    if 'trends' in names:
        fig = Figure(figsize=(8, 7)); FigureCanvasAgg(fig)
        from core.peak_plotting import build_fit_trends
        build_fit_trends(dataset, fig)
        save_figure(fig, target/names['trends'], dpi=180); fig.clear()
    return str(target)
