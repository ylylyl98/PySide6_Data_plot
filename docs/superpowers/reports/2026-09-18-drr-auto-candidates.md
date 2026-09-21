# DRR Auto candidate assessment

Auto Find peaks now enables automatic noise and smoothing-scale assessment by default. Original extrema and their measurements remain intact; the default view retains supported candidates, with a reversible Show low-confidence candidates option. Manual prominence and width filters use the separate smoothed detection measurements when available.

Evidence uses SG windows 5/9/15, robust residual noise, agreement across at least two scales, resolved width and edge exclusion. Strong candidates require a prominence/noise ratio of 5. Weak candidates at 2.5 require support from both adjacent Y rows. These are heuristic thresholds, not optimized physical truth or probabilities.

For second derivatives, each finite segment additionally uses a raw first-difference MAD noise estimate propagated through the SG derivative coefficient norm. This assumes independent sample noise. Segment-local estimation prevents a quiet section from suppressing the noise floor in a separate noisy section. The policy and candidate evidence are exported.

Validation: 67 unit/integration tests passed, including broad-peak recovery, flat/gapped input, reversible filtering, pure noise, and heteroskedastic NaN-separated noise. Before correction, the latter regression promoted 71 noisy derivative candidates and failed; it passes with segment-local propagation. A representative 157 x 1340 real dataset took about 8.5 seconds before the segment-local correction and retained 5,824 supported candidates from 162,167 extrema. Those counts are not a measurement of detection accuracy.

Reopen the standalone DRR Analysis window to load updated source code. Existing results must be recomputed with Auto Find peaks to acquire quality evidence.
