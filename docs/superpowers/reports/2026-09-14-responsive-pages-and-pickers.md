# Responsive pages and file pickers — current evidence

Date: 2026-09-14. Astra planned; Luna implemented and corrected review findings. Independent final Astra re-review completed: **READY WITH DOCUMENTED LIMITATIONS**; see the authoritative Astra review report. Earlier claims are retained in the interim-history and review-history reports. This report does not claim every latency target was reached.

## Current implementation

| Area | Implemented behavior |
|---|---|
| Pickers | Worker metadata, memory-only mtime lookup, content/appearance descriptors and unchanged-row reuse for PL/MCD/SHG/Compare |
| R1 | Local drawing with full-draw/export restoration; Power artists sharing an axes share a helper |
| R2 | Fresh owned-worker Power validation, decoded-result handoff, stale acceptance callbacks rejected before state mutation |
| R3 | Scoped PL/MCD publication preserves other mode caches; old disk payloads rebuild |
| R4 | All acceptance paths block while filtering is pending |
| R5 | Compare history readiness follows accepted publication; coalesced force waits for its result |
| R6 | DRR known/relative/external identities survive inspection; missing results are selection-scoped |
| R7 | Hidden pages retain dirty work, including Peak Shift ownership |
| MCD | Selected-column calculation, energy-value revision invalidation, numerical equivalence including Integral equal-NaN semantics |
| SHG | View-only angle changes and latest-angle display after processing result publication |

Final independent Astra review found no remaining reproducible P1/P2 issue within the agreed scope. Astra independently reran 49 callback tests (all passed), verified all 93 rev6 source hashes, and reproduced both the queued Compare final completion and old-prefetch-to-current-Power-acceptance completion. Root independently reran astra-final-review-repro.py after correction: Power pixel delta 0; old Power result does not cancel the new request; queued Compare force stays pending; known DRR missing tuple is empty; relative selection matches returned identity.

## Current validation

Artifact paths below are under artifacts/responsive-pages-and-pickers/ unless stated otherwise.

- rev5-affected-tests-clean.txt: 20 modules in fresh processes, **204 tests, 0 failures**, exit 0. rev5-picker-tests.txt separately reports 57/57; these overlap and must not be added.
- Final R2/R5 callback corrections were independently verified by root: rev6-final-callback-tests.txt, **49 tests passed**, exit 0. These overlap earlier suites and are not additive. The full followup reproduction confirms both intermediate pending and final ready states, and that old Power prefetch cannot cancel the new acceptance.
- rev6-source-manifest.json identifies the final 93-file source snapshot; rev5 is historical. Changes are compared to artifacts/responsive-implementation-baseline-20260914/source.zip, not unrelated dirty git changes. No commit or reset was performed. git diff --check exited 0.
- rev5-pages-r1/r2/r3.json and .log: three completed exit-0 runs, respectively 25.75, 27.33 and 25.12 seconds. Raw render_ms includes full-draw fallbacks; slot time and first next event are not completed-render latency. Prior rev4 page measurements are historical.
- render-acceptance-rev8/results.json and PNGs: exit 0; DRR and Power each contain two spectrum lines; MCD contains six real ok fits across two branches; dark 150% contains loaded PL heatmap and spectrum. Root inspected the MCD fit export. Earlier blank dark and zero-fit fixtures are superseded and cannot certify these cases.
- rev5-picker-final.json/.txt: **6/6 runs, exit 0**, 100/1000 files times three rounds, PL/MCD cold/warm; wall 56.79 seconds. Real accepted snapshots have asserted nonzero mtimes and correct date sorting. Missing-cache GUI stat count is zero. A middle selection and nonzero scrollbar are compared across equivalent-query refresh; selection, scroll and row identity survive, with zero actual population callbacks.

## File selection measurements

For 1000 files, three cold and three warm samples per mode, synthetic local/offscreen data:

| Measurement (ms) | PL median / p95 | MCD median / p95 |
|---|---:|---:|
| Opener to shown dialog | 595.10 / 807.17 | 399.08 / 552.43 |
| Six-character synchronous input | 0.400 / 0.690 | 0.590 / 0.750 |
| Subsequent debounce/settling | 222.79 / 236.80 | 164.41 / 198.13 |

Opening and completed filtering are not sub-25-ms operations. The improvement is avoiding GUI metadata reads and repeated population. A 1000-file probe took 141.68 seconds using QTest.qWait versus 14.44 seconds using processEvents plus short sleep; old harness timeouts/native failures cannot be attributed to production scanning from that evidence alone.

## Remaining risks and limits

- Full Matplotlib redraws and initial layout remain costly. Keep completed-render/fallback data visible; do not advertise only fast local updates.
- MCD analysis-only on valid 256x512 increasing/decreasing data: median 4.19 ms, below the 75-ms worker threshold. No extra analysis worker is justified by this measurement.
- Slides full-list filtering at 1000 records measured 259–287 ms; decoded thumbnail cache reached 1000 entries. These remain unmodified risks. Cache entry counts do not measure all Qt native memory.
- Real 1200x800 to 1400x900 resize measured 325–332 ms end to end. Theme/dense-layout cost is not isolated, so no standalone cache optimization is claimed.
- Complete Peak Shift interaction/analysis fixture remains unmeasured; existing workers were retained. An independent 0.3-second pool wait is not an application-close latency test. Ownership-safe shutdown redesign is deferred.
- Export label crowding is not attributed to this round without a baseline image. Slope-fit export does not claim feature-shift completeness.
- Confirmed baseline failures: DRR split-scale (drr-split-baseline.txt); four tests.test_mcd_auto_refresh failures (matching current/baseline mcd_auto_refresh logs); eleven failures out of 62 tests.test_mcd tests (matching mcd-current-last.log / mcd-baseline-last.log). These are outside the 204 passing affected tests; the full suite is not claimed green.

## Delivery status

The listed implementation and validation are complete. The independent acceptance decision belongs to docs/superpowers/reports/2026-09-14-responsive-pages-and-pickers-astra-review.md. Delivery remains awaiting its update for this snapshot.
