"""Benchmark one measurement against one BN; run as python -m tests.benchmark_bn_calibration."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from core.drr_bn_calibration import _json_safe
from core.drr_bn_joint import production_d2
from core.drr_bn_settings import BnCalibrationSettings
from core.drr_bn_single import evaluate_bn_single
from core.drr_bn_service import _algorithm_identity
from core.processing_run import _parse_raw_spectrum
from core.drr_sources import inspect_csv_spectral_grid
from tests.bn_injection_audit import audit_injection


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _draw(path, energy, gates, original, candidate, derivative_original, derivative_candidate, *, title, eligible, axis, d2_limit=None):
    figure = Figure(figsize=(12, 8), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharex=True, sharey=True)
    order = np.argsort(gates)
    for row, (left, right, label) in enumerate(((original, candidate, 'DR/R'),
                                                (derivative_original, derivative_candidate, 'D2 (eV⁻²)'))):
        finite = np.concatenate((left[np.isfinite(left)], right[np.isfinite(right)]))
        if not finite.size:
            raise ValueError('Comparison contains no finite values.')
        if row == 0:
            lo, hi = map(float, np.percentile(finite, [1, 99]))
            if hi <= lo:
                pad = max(abs(lo)*1e-6, 1e-12); lo, hi = lo-pad, hi+pad
        else:
            hi = float(d2_limit) if d2_limit is not None else max(float(np.percentile(np.abs(finite), 99)), 1e-12)
            lo = -hi
        for col, (values, kind) in enumerate(((left, 'Original BN'),
                                               (right, 'Eligible candidate' if eligible else 'Exploratory candidate; not eligible'))):
            mesh = axes[row, col].pcolormesh(energy, gates[order], values[order], shading='auto',
                                            cmap='RdBu_r', vmin=lo, vmax=hi)
            axes[row, col].set(title=f'{kind} · {label}', xlabel='Energy (eV)', ylabel=f'{axis.upper()} (V)')
        figure.colorbar(mesh, ax=list(axes[row]), label=label)
    figure.suptitle(title + '\nWithin-scan comparison; not independent validation')
    figure.savefig(path, dpi=140, bbox_inches='tight')
    figure.clear()


def run(measurement, bn, output, *, gate_axis='bg', injections=False, method='default'):
    measurement, bn, output = (Path(p).expanduser().resolve() for p in (measurement, bn, output))
    if measurement == bn or not measurement.is_file() or not bn.is_file():
        raise ValueError('Select two different existing measurement and BN files.')
    if gate_axis not in ('bg', 'tg'):
        raise ValueError('Gate axis must be bg or tg.')
    if method not in ('default', 'gap'):
        raise ValueError('Unknown benchmark method.')
    evaluator = evaluate_bn_single
    gap_module = None
    if method == 'gap':
        from core import drr_bn_gap_prediction as gap_module
        evaluator = gap_module.evaluate_bn_gap
    source_sha = {'measurement': _sha256(measurement), 'bn': _sha256(bn)}
    energy, bg, tg, _, raw = _parse_raw_spectrum(measurement)
    bn_energy, _, _, _, frames = _parse_raw_spectrum(bn)
    if not np.array_equal(energy, bn_energy):
        raise ValueError('Measurement and BN must share the exact native energy grid.')
    if np.count_nonzero((energy >= 1.6) & (energy <= 1.9)) < 2:
        raise ValueError('The measurement must include at least two points in 1.6–1.9 eV.')
    background = np.mean(frames, axis=0)
    halves = (np.stack((frames[:len(frames)//2].mean(axis=0), frames[len(frames)//2:].mean(axis=0)))
              if len(frames) >= 2 and len(frames) % 2 == 0 else None)
    gates = np.asarray(bg if gate_axis == 'bg' else tg, float)
    settings = BnCalibrationSettings(single_measurement=True, single_full_range=True)
    # A new directory prevents silently overwriting a previous benchmark.
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    result = evaluator(energy, raw, background, gates=gates, settings=settings, background_repeats=halves)
    elapsed = perf_counter()-start
    if method == 'gap' and (result.eligible or result.selected_method != 'gap_preview'):
        raise ValueError('Gap benchmark must remain an explicitly selected ineligible preview.')
    if result.selected_method != 'original':
        candidate = result.candidate_backgrounds.get(result.selected_method)
        if candidate is None:
            raise ValueError('Selected calibration candidate is missing.')
    else:
        candidate = result.applied_background
    candidate = np.asarray(candidate, float)
    if candidate.shape != energy.shape or not np.isfinite(candidate).all() or np.any(candidate <= 0):
        raise ValueError('Preview candidate must be positive, finite and match the native grid.')
    original_drr, candidate_drr = raw/background-1, raw/candidate-1
    original_d2, used = production_d2(energy, original_drr, settings)
    candidate_d2, _ = production_d2(energy, candidate_drr, settings)
    if source_sha != {'measurement': _sha256(measurement), 'bn': _sha256(bn)}:
        raise ValueError('A source changed during calibration; discard this benchmark.')
    mask = (energy >= 1.6) & (energy <= 1.9)
    for name, selected in (('full', np.ones(len(energy), dtype=bool)), ('1p6_to_1p9', mask)):
        _draw(output/(name+'.png'), energy[selected], gates, original_drr[:, selected], candidate_drr[:, selected],
              original_d2[:, selected], candidate_d2[:, selected], title=measurement.name,
              eligible=result.eligible, axis=gate_axis)
    _draw(output/'1p6_to_1p9_d2_detail.png', energy[mask], gates, original_drr[:, mask], candidate_drr[:, mask],
          original_d2[:, mask], candidate_d2[:, mask],
          title='Residual detail: D2 uses fixed +/-6000 eV^-2; stronger peaks saturate',
          eligible=result.eligible, axis=gate_axis, d2_limit=6000.)
    np.savez_compressed(output/'arrays.npz', energy=energy, gates=gates, original_background=background,
                        candidate_background=candidate, original_drr=original_drr, candidate_drr=candidate_drr,
                        original_d2=original_d2, candidate_d2=candidate_d2)
    if method == 'gap':
        # A labelled audit table, never an instrument-format BN or managed source.
        columns = np.asarray(inspect_csv_spectral_grid(bn), float)
        native_wavelengths = np.full(len(energy), np.nan)
        if np.median(columns) > 20:
            parsed_energy = 1240/np.round(columns, 9)
            order = np.argsort(parsed_energy, kind='stable')
            if len(order) != len(energy) or not np.allclose(parsed_energy[order], energy, rtol=1e-10, atol=1e-10):
                raise ValueError('BN native wavelength headers do not match the parsed energy grid.')
            native_wavelengths = columns[order]
        with (output/'gap_preview_background.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['energy_eV', 'native_wavelength_nm', 'original_intensity', 'gap_preview_intensity'])
            for energy_value, wavelength, original, predicted in zip(energy, native_wavelengths, background, candidate):
                writer.writerow([format(energy_value, '.17g'), format(wavelength, '.17g') if np.isfinite(wavelength) else '',
                                 format(original, '.17g'), format(predicted, '.17g')])
        header = 'Photon energy(eV)\t'+'\t'.join(format(value, '.12g') for value in gates)
        for name, values in (('original_drr', original_drr), ('gap_preview_drr', candidate_drr),
                             ('original_d2', original_d2), ('gap_preview_d2', candidate_d2)):
            np.savetxt(output/(name+'.dat'), np.column_stack((energy, values.T)), delimiter='\t',
                       header=header, comments='', fmt='%.12g')
    cases = []
    if injections:
        for moving in (False, True):
            for sigma in (1., 2., 5.):
                for amplitude in (2e-4, 1e-3):
                    audit = audit_injection(energy, raw, background, gates, center_ev=1.75 if moving else 1.80,
                        sigma_mev=sigma, amplitude=amplitude, moving=moving, baseline_result=result,
                        settings=settings, background_repeats=halves, evaluator=evaluator,
                        audit_selected_candidate=(method == 'gap'))
                    name = f"{'moving' if moving else 'fixed'}_s{sigma:g}_a{amplitude:.0e}.json"
                    (output/name).write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
                    cases.append(dict(file=name, center_ev=audit['center_ev'], sigma_mev=sigma,
                                      amplitude=amplitude, moving=moving, model_switched=audit['model_switched']))
        if source_sha != {'measurement': _sha256(measurement), 'bn': _sha256(bn)}:
            raise ValueError('A source changed during the injection audit; discard this benchmark.')
    (output/'report.json').write_text(json.dumps(_json_safe(result.report), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    summary = dict(measurement=str(measurement), bn=str(bn), source_sha256=source_sha, gate_axis=gate_axis,
        report_file='report.json', candidate_generated=bool(result.report.get('candidate_generated', not np.array_equal(candidate, background))),
        algorithm_identity=_algorithm_identity(),
        gap_module_sha256=_sha256(Path(gap_module.__file__).resolve()) if gap_module is not None else None,
        benchmark_method=method, audit_target='selected_preview_candidate' if method == 'gap' else 'applied_background',
        preview_only=method == 'gap',
        benchmark_sha256=_sha256(Path(__file__).resolve()),
        injection_audit_sha256=_sha256(Path(__file__).with_name('bn_injection_audit.py').resolve()),
        bn_frame_policy='all frames averaged', bn_frame_count=len(frames), settings=asdict(settings),
        status=result.status, model=result.report.get('single_pair_scope', {}).get('model', result.selected_method),
        eligible=bool(result.eligible), reasons=list(result.reasons), elapsed_seconds=elapsed,
        derivative_window_used=int(used), d2_detail_color_limit=6000., independent_validation=False,
        caveat='Same-scan matching and synthetic injections do not establish independent validation or real-peak preservation.',
        injections=cases)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', required=True, help='Original measurement CSV')
    parser.add_argument('--bn', required=True, help='Original measured BN CSV')
    parser.add_argument('--output', required=True, help='New output directory; existing paths are rejected')
    parser.add_argument('--gate-axis', choices=('bg', 'tg'), default='bg')
    parser.add_argument('--injections', action='store_true', help='Run twelve additional complete refits')
    parser.add_argument('--method', choices=('default', 'gap'), default='default',
                        help='gap is an experimental, never-applied preview')
    args = parser.parse_args()
    summary = run(args.measurement, args.bn, args.output, gate_axis=args.gate_axis, injections=args.injections, method=args.method)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
