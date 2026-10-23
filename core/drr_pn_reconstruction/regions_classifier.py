"""Training-sample conditional source consistency; no background optimization."""
import numpy as np
from .regions_helpers import local_quadratic

def bands(e, values):
    q4 = local_quadratic(e, values, .004)
    q35 = local_quadratic(e, values, .035)
    return np.stack((values-q4, q4-q35), axis=-2)


def classify(e, pn, train_samples, *, baseline, control, half_aligned_sources):
    e = np.asarray(e, float)
    pn = np.asarray(pn, float)
    samples = np.asarray(train_samples, float)
    n = len(e)
    if e.ndim != 1 or n < 21 or not np.all(np.diff(e) > 0):
        raise ValueError("Original ascending native energy required")
    if samples.ndim != 3 or samples.shape[0] != 2 or samples.shape[1] < 3 or samples.shape[2] != n:
        raise ValueError("Two complete training repeats with at least three gate rows required")
    gate_count = samples.shape[1]
    if pn.shape != (2, n) or not np.all(np.isfinite(pn)&(pn > 0)):
        raise ValueError("Positive finite PN required")
    if not np.all(np.isfinite(samples)&(samples > 0)):
        raise ValueError("Positive finite training samples required")
    for item in (baseline, control):
        if not np.array_equal(e, item["energy"]):
            raise ValueError("Frozen source grid mismatch")
    C = np.asarray(baseline["v32_effective_corrected_source_log"])
    q = np.asarray(baseline["v32_alignment_q_log"])
    A = C + np.array([-.5, .5])[:, None]*q
    halves = np.asarray(half_aligned_sources)
    if halves.shape != (3, 2, n) or not np.array_equal(halves[0], A):
        raise ValueError("PN half-frame provenance mismatch")
    oldw = np.asarray(control["source_weights"])
    oldB = np.asarray(control["background"])
    if not np.allclose(np.log(oldB), np.sum(oldw*A, axis=0), rtol=0, atol=1e-13):
        raise ValueError("Control uses different q-aligned sources")

    G = np.exp(-.5*((e[:, None]-e[None, :])/.004)**2)*np.gradient(e)[None, :]
    G /= G.sum(1, keepdims=True)
    def rms(values):
        shape = values.shape
        return np.sqrt((G@(values*values).reshape(-1, n).T).T.reshape(shape))

    source_bands = bands(e, halves)
    sample_mean_bands = bands(e, np.log(samples.mean(axis=0)))
    sample_repeat_bands = bands(e, np.log(samples))
    witness = np.median(sample_mean_bands, axis=0)
    source_noise = rms((source_bands[1]-source_bands[2])/2)
    sample_repeat_noise = rms((sample_repeat_bands[0]-sample_repeat_bands[1])/2)
    # Conservative per-gate uncertainty; do not treat101 gates as independent.
    sample_noise = np.sqrt(np.median(sample_repeat_noise**2, axis=0))
    gate_mad = 1.4826*np.median(abs(sample_mean_bands-witness[None]), axis=0)
    gate_dispersion = rms(gate_mad)
    groups = np.array_split(np.arange(gate_count), 3)
    group_witness = np.array([np.median(sample_mean_bands[g], axis=0) for g in groups])
    group_discrepancy = rms(group_witness-witness[None])
    witness_level = rms(witness)
    sample_scale = np.maximum(sample_noise, 1e-12)
    evidence = np.any(witness_level > 3*sample_scale, axis=0)
    gate_consistent = np.all(gate_dispersion <= 3*sample_scale, axis=0)
    complete = (e-e[0] >= .0175)&(e[-1]-e >= .0175)
    eligible = evidence & gate_consistent & complete
    residual = source_bands[0]-witness[None]
    residual_rms = rms(residual)
    combined_noise = np.sqrt(source_noise**2+sample_noise[None]**2)
    match_score = residual_rms/np.maximum(combined_noise, 1e-12)
    within_noise = np.all(match_score <= 3, axis=1)
    usable = within_noise & eligible[None]
    states = usable[0].astype(np.int8)+2*usable[1].astype(np.int8)
    arrays = dict(energy=e.copy(), v45_state=states, state=states.copy(),
        source_usable_masks=usable, v45_control_source_weights=oldw.copy(),
        v45_source_bands=source_bands[0], v45_sample_witness_bands=witness,
        v45_sample_witness_level=witness_level, v45_sample_repeat_noise=sample_noise,
        v45_PN_repeat_noise=source_noise, v45_combined_repeat_noise=combined_noise,
        v45_matching_residual_bands=residual, v45_matching_residual_RMS=residual_rms,
        v45_matching_score=match_score, v45_within_repeat_uncertainty=within_noise,
        v45_sample_gate_MAD=gate_mad, v45_sample_gate_local_dispersion=gate_dispersion,
        v45_sample_gate_group_witness=group_witness,
        v45_sample_gate_group_discrepancy=group_discrepancy,
        v45_witness_has_evidence=evidence, v45_witness_gate_consistent=gate_consistent,
        v45_complete_Q35_support=complete, v45_witness_eligible=eligible)
    diagnostics = dict(training_repeats=2, training_gates=gate_count,
        state_counts={str(i): int(np.sum(states == i)) for i in range(4)},
        state_meanings={"0": "unknown", "1": "P-only conditional",
                        "2": "N-only conditional", "3": "both conditional"},
        sample_used=True, BN_used=False, scientifically_validated=False,
        historical_LP_not_run="Final V50 constructor consumes these classifications, not V45 projected weights.",
        limitation="A majority-gate texture witness may contain common sample structure; it is not proof of featurelessness.")
    return arrays, diagnostics
