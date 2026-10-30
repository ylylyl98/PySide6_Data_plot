"""Conservative single-measurement gate-consistency evidence; not repeat noise."""
import numpy as np
from .regions_classifier import bands


def classify_single(e, pn, train_samples, *, baseline, control, half_aligned_sources):
    e, pn, samples = np.asarray(e, float), np.asarray(pn, float), np.asarray(train_samples, float)
    n = len(e)
    if e.ndim != 1 or n < 21 or not np.all(np.isfinite(e)) or not np.all(np.diff(e) > 0):
        raise ValueError("Original ascending native energy required")
    if samples.ndim != 3 or samples.shape[0] != 1 or samples.shape[1] < 3 or samples.shape[2] != n:
        raise ValueError("One complete measurement with at least three gate rows required")
    if pn.shape != (2, n) or not np.all(np.isfinite(pn) & (pn > 0)) or not np.all(np.isfinite(samples) & (samples > 0)):
        raise ValueError("Positive finite P/N and measurement data required")
    for item in (baseline, control):
        if not np.array_equal(e, item['energy']):
            raise ValueError("Frozen source grid mismatch")
    c, q = np.asarray(baseline['v32_effective_corrected_source_log']), np.asarray(baseline['v32_alignment_q_log'])
    aligned = c + np.array([-.5, .5])[:, None] * q
    halves = np.asarray(half_aligned_sources)
    if halves.shape != (3, 2, n) or not np.array_equal(halves[0], aligned):
        raise ValueError("PN half-frame provenance mismatch")
    weights = np.asarray(control['source_weights'])
    if not np.allclose(np.log(control['background']), np.sum(weights * aligned, axis=0), rtol=0, atol=1e-13):
        raise ValueError("Control uses different q-aligned sources")
    gaussian = np.exp(-.5*((e[:, None]-e[None, :])/.004)**2)*np.gradient(e)[None, :]
    gaussian /= gaussian.sum(1, keepdims=True)
    def rms(values):
        shape = values.shape
        return np.sqrt((gaussian @ (values*values).reshape(-1, n).T).T.reshape(shape))
    source = bands(e, halves)
    sample = bands(e, np.log(samples[0]))
    witness = np.median(sample, axis=0)
    groups = np.array_split(np.arange(samples.shape[1]), 3)
    group_witness = np.array([np.median(sample[g], axis=0) for g in groups])
    group_error = rms(group_witness-witness[None])
    gate_mad = 1.4826*np.median(np.abs(sample-witness[None]), axis=0)
    dispersion = rms(gate_mad)
    source_noise = rms((source[1]-source[2])/2)
    # Gate dispersion is a conservative stability bound, not independent noise.
    # An empty source-half estimate cannot certify a perfectly noiseless sample.
    scale = np.maximum(np.maximum(dispersion, np.max(source_noise, axis=0)), 1e-8)
    witness_level = rms(witness)
    evidence = np.any(witness_level > 3*scale, axis=0)
    consistent = np.all(group_error <= 2*scale[None], axis=(0, 1))
    complete = (e-e[0] >= .0175) & (e[-1]-e >= .0175)
    eligible = evidence & consistent & complete
    residual = source[0]-witness[None]
    residual_rms = rms(residual)
    bound = np.sqrt(source_noise**2 + scale[None]**2)
    score = residual_rms / bound
    within = np.all(score <= 3, axis=1)
    usable = within & eligible[None]
    states = usable[0].astype(np.int8)+2*usable[1].astype(np.int8)
    arrays = dict(energy=e.copy(), v45_state=states, state=states.copy(), source_usable_masks=usable,
        v45_control_source_weights=weights.copy(), v45_source_bands=source[0],
        v45_sample_witness_bands=witness, v45_sample_witness_level=witness_level,
        v45_PN_repeat_noise=source_noise, v45_matching_residual_bands=residual,
        v45_matching_residual_RMS=residual_rms, v45_matching_score=score,
        v45_sample_gate_MAD=gate_mad, v45_sample_gate_local_dispersion=dispersion,
        v45_sample_gate_group_witness=group_witness, v45_sample_gate_group_discrepancy=group_error,
        single_gate_stability_bound=scale, v45_witness_has_evidence=evidence,
        v45_witness_gate_consistent=consistent, v45_complete_Q35_support=complete,
        v45_witness_eligible=eligible)
    diagnostics = dict(training_repeats=1, independent_sample_repeats=1,
        training_gates=samples.shape[1], uncertainty_basis='gate_group_dispersion_and_conditional_PN_half_difference',
        state_counts={str(i): int(np.sum(states == i)) for i in range(4)},
        state_meanings={'0': 'unknown', '1': 'P-only conditional', '2': 'N-only conditional', '3': 'both conditional'},
        sample_used=True, BN_used=False, scientifically_validated=False,
        cross_measurement_drift='not evaluable from one measurement',
        limitation='Gate groups are not independent sample repeats; common fixed sample resonances can imitate background texture. PN halves use the same full-fit model.')
    return arrays, diagnostics
