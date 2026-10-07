"""Explicit, collision-safe CSV/JSON/PNG export of a Peak Analysis snapshot."""
import csv
from datetime import datetime
from pathlib import Path
import re
from uuid import uuid4

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from core.peak_plotting import build_figure
from core.peak_workspace import atomic_json, validate_result


def export_dataset(folder, dataset):
    result = dataset['result']
    if result is None:
        raise ValueError('Find peaks before exporting results.')
    validate_result(dataset, result)
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset['name']).strip('._')[:65] or 'peaks'
    target = Path(folder) / f'{name}_peaks_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}'
    target.mkdir(parents=True, exist_ok=False)
    fields = ['branch', 'branch_id', 'row_index', 'y', 'energy', 'polarity', 'status', 'segment',
              'candidate_id', 'prominence', 'prominence_fraction', 'width_mev',
              'noise_sigma', 'snr', 'neighbor_support', 'neighbor_available', 'neighbor_required',
              'fit_center', 'fit_status', 'fwhm_mev', 'fwhm_error_mev', 'height', 'area', 'rms']
    names = {b['id']: b['name'] for b in dataset['branches']}
    with (target/'peaks.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader()
        for p in result['points']:
            writer.writerow({**p, 'branch': names.get(p['branch_id'], p['branch_id'])})
    atomic_json(target/'analysis.json', {k: v for k, v in dataset.items() if k != 'cube'})
    fig = Figure(figsize=(8, 7)); FigureCanvasAgg(fig)
    build_figure(dataset, dataset['row'], True, bool(result.get('fits')), fig)
    fig.savefig(target/'peaks.png', dpi=180); fig.clear()
    return str(target)
