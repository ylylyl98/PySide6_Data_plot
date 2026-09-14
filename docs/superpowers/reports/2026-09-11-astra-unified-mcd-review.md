# Astra review of unified MCD workflow

## Initial core review — 2026-09-11

Scope: `core/mcd_analysis.py` and `tests/test_mcd_analysis.py`, against the approved unified workflow plan. UI/export were still being integrated and are excluded from this initial verdict. Production and test files were not modified. All reproductions used `.venv/Scripts/python.exe` and synthetic inputs; the focused suite passed 12 tests, but does not cover the failures below.

### Important findings

1. **P1 — Missing samples move detected energies.** In `_detect_row_extrema`, finite-value filtering changes the index domain, but the returned index is applied to the original energy and row arrays in `detect_analysis_features` (initial lines 336–338, 365, 413, 440). Reproduction: 401 energy points from 1.5 to 1.9 eV; Gaussian centered at 1.680 eV; first 50 row values NaN; two positive-field rows. Both spectral peak and MCD maximum are reported at **1.630 eV**. Return the actual energy/value or map filtered indices back to the source. Cover internal holes, leading missing values, and descending energy order.

2. **P1 — Associations cannot retain one MCD feature linked to multiple spectral features.** `associate_features` stores manual corrections as a dictionary from MCD ID to one spectral ID (initial lines 524–535), and the automatic path emits only its best candidate (559–567). Reproduction: two explicitly supplied manual links with the same MCD ID retain only the second; the first spectral feature is reported unmatched. This violates the agreed one-to-many relation and loses user corrections. Preserve all explicit links; ensure the automatic representation distinguishes several supported associations from mutually ambiguous alternatives without silently collapsing the graph.

3. **P1 — Unstable channel ordering is claimed to be stable.** `infer_valley_mapping` checks only median energy difference. Positive-field records at 0.5, 1.0, and 1.5 T with channel differences −0.020, −0.020, +0.020 eV return `inferred`, K=`pos`, reason `stable positive-field ordering`. These samples reverse ordering. Require the intended stability criterion and distinct positive-field samples; ambiguous evidence must stay unknown. The array-mapping input also has no actual field validation.

4. **P2 — Reported paired slope-difference uncertainty is numerically incorrect.** In `fit_mcd_slopes`, branch covariance uses residual `np.cov(..., ddof=1)` with different normalization than the OLS variances, and assumes a common-grid denominator even when fits use different grids. Reproduction: five fields evenly spaced from −0.2 to 0.2 T; identical residual vector `[0, .01, 0, -.01, 0]`; curves `2B+residual` and `3B+residual`. Their paired difference is exactly linear, but the function returns SE **0.0163299316**, `covariance_supported=True`. Compute valid OLS paired covariance with the actual fit weights/valid samples, or explicitly return unsupported. Tests should assert the uncertainty value, not just that a number exists.

### Checked behavior

- Detection separates MCD signs/branches rather than signed averaging them.
- The focused tests cover finite unique-field threshold and basic unequal-grid same-branch splitting with the explicit `E_K-E_Kp` convention.
- Existing association and covariance tests are insufficient to detect the reproduced issues.

Status: **Changes requested for core; integrated UI/export review pending.**

## Bounded export review — 2026-09-11

Scope expanded to `core/mcd_unified_export.py`, its six focused tests, and the export implementation report. The six tests passed in 2.865 s. The tests mostly check file presence/schema and a single retained window; the issues below remain despite that result. UI integration remains excluded pending readiness.

1. **P1 — DAT silently discards a roundtrip branch.** `_write_dat` (lines 432–447) deduplicates by numeric B alone, retaining the first row. Reproduction with fields `[-1, 1, 1, -1]`, branches `[inc, inc, dec, dec]`, and distinguishable maps `[10..12], [20..22], [30..32], [40..42]` exports only the first two rows as columns; all decreasing data disappear. Preserve branch plus measured-row identity in the DAT, with explicit headers and actual grids.

2. **P1 — Independent feature/channel analyses collapse into one curve.** `_feature_rows` (256 onward) drops source/method/analysis identity when flattening tracks; `_feature_table` (348–365) groups only feature kind/ID/branch. Two raw channels with the same normal `peak-1` ID at B=1, energies 1.6 and 1.7 eV, become successive rows in one B–E pair. `_splitting_table` (380 onward) groups only by branch: two retained feature pairs at B=1 with 10 and 20 meV also become one curve. Preserve snapshot, feature/pair, source/channel, method, and branch identities through flattening, table grouping, and headers. Do not join unrelated numerical points.

3. **P1 — PNGs do not honor the captured visual/numerical state.** `_plot_pngs` (449–489) never reads `plot_state`. The map uses `imshow` with min/max B extent across acquisition rows, which misplaces roundtrip and nonuniform grids. Spectra always use `z[0]` regardless of selected B (and if absent substitute MCD under a spectrum title). MCD(B) uses only the first retained window; the feature PNG joins all branches sharing a feature ID and draws only absolute energy, omitting requested shift/splitting and slope-fit overlays. Render the selected branches on actual grids, each requested retained product with its identity, actual selected spectra B, and selected energy/shift/splitting/fit state from the immutable snapshot. Include numerical assertions on plotted artists, rather than PNG count alone.

4. **P2 — Slopes lose retained-window identity and detailed diagnostics.** `_slope_table` (368–377) emits only branch, region, method, numerical values, n, and status. Two windows with different centers and slopes therefore produce indistinguishable rows. Its headers contain no window ID, center, width, or metric; JSON also does not serialize the `slopes` input itself, only an optional separate `fit_diagnostics` key. Preserve window/fit/difference identifiers, the requested ranges and actual ranges, covariance status, residual diagnostics, units, and curve metadata in their appropriate XLSX/JSON outputs.

5. **P2 — Square fields×energy grids are transposed silently.** `_mcd_data` transposes whenever shape equals `(energy_count, field_count)`, before checking the documented `(field_count, energy_count)` orientation. For two fields and two energies, input `[[1,2],[3,4]]` becomes `[[1,3],[2,4]]`. Prefer the documented orientation and transpose only an unambiguous alternate shape, or require an explicit orientation flag. The analogous shape detection in `_series_table` has the same ambiguity.

Additional implementation concern: `_plot_pngs` calls global `matplotlib.use('Agg', force=True)` in a worker despite constructing `FigureCanvasAgg` explicitly. Remove that global backend mutation so exporting cannot switch the application's active Matplotlib backend.

Observed positives: the exporter contains no feature-fitting call or direct Qt-widget access; snapshots copy numerical arrays; sequential revisions are distinct; the focused discovery compatibility test passes. The failure test only rejects malformed input before staging, so it does not establish cleanup after a late rendering/write failure.

Status: **Changes requested for export; complete UI/integration review still pending.**

## Core fix re-review — 2026-09-11

Re-ran the focused suite: **15 tests passed in 0.175 s**. Independent reproductions confirm the missing-value index case now returns 1.680 eV, two manual associations are both retained, and the reversing-order valley example returns unknown. The identical-order paired-residual slope test now has approximately zero SE (4.66e-10).

Two important core findings remain:

1. **P1 — Automatic one-to-many remains unimplemented.** The automatic path still emits only its best spectral candidate, while other supported candidates are labeled unmatched. One MCD feature with two spectrum candidates separated by 0.003 eV and identical branch/field support produces one automatic edge. The manual fix did not change this path.

2. **P1 — Covariance fix crashes on unequal grids and depends on acquisition order.** The new `cross_design = np.dot(x_a - mean(x_a), x_b - mean(x_b))` pairs raw acquisition arrays by row rather than matching valid measured B. With five increasing and six decreasing samples in the same fit range, the entire `fit_mcd_slopes` call raises `ValueError: shapes (6,) and (5,) not aligned`. With the original identical-residual example and decreasing rows reversed, SE becomes **0.0461880215** although the paired difference remains exactly linear and should have zero residual uncertainty. Require aligned finite matched samples and actual OLS weights, or explicitly mark uncertainty unsupported for unhandled grids. Include reversed acquisition order and unequal-grid tests.

Status: **Core still requires changes; three of the original core failure examples are resolved.** UI/export were not re-reviewed during this bounded pass.

## Export fix re-review and second core fix re-review — 2026-09-11

**Core:** 18 focused tests passed in 0.134 s. Independent checks now return two automatic edges for supported pos/neg channel features, ambiguous alternatives within one channel, zero paired-difference SE with reversed acquisition order, and explicit unsupported uncertainty without crashing for unequal grids. The previously reported core blockers are resolved for these checked cases.

**Export:** 9 focused tests passed in 4.775 s. DAT preserves duplicate-field branch rows; explicit feature rows with channel/method identity separate correctly; corrected square-grid orientation is fixed; global backend mutation is removed; selected spectra B and multiple retained-window traces are now consumed; a late-render failure cleanup test has been added. However the normal integration contract exposes additional unresolved cases:

1. **P1 — Actual slope results prevent export.** Feeding `fit_mcd_slopes(...).to_dict()` through `_slope_table` and `_write_xlsx` raises `ValueError: Cannot convert [-0.2, 0.2] to Excel`. The adapter turns actual field range into a Python list and passes it directly to an Excel cell. Use separate min/max scalar columns or a readable serialized value for range fields; test with the real `SlopeAnalysis.to_dict` structure, not simplified hand-written rows.

2. **P1 — Nested production analyses still merge channels.** Given `analysis_results={'Raw spectrum': {'raw pos': PeakShiftResult(...), 'raw neg': PeakShiftResult(...)}}`, `_feature_rows` emits both rows with empty `channel` and `analysis_id='Raw spectrum'`. It ignores `analysis.source`, while nested `raw pos`/`raw neg` keys do not set channel. Both `peak-1` tracks again merge into one energy curve (verified at B=1 with 1.6/1.7 eV values). Read method/source from the result object's actual attributes, and preserve retained analysis identity through the nested schema.

3. **P1 — Candidate inventory becomes fabricated tracked curves.** The supplied UI contract passes candidates in `features`. `_feature_rows` blindly treats them as tracked numerical points: two `AnalysisFeature`-shaped records with IDs `mcd-abc`/`mcd-def`, representative B=1, energies 1.6/1.7 become one `Energy_eV_feature_unknown_MCD_corrected___inc` curve. Inventory representative energies and support-field medians are not trajectories. Store candidates only in JSON; build numerical curves only from explicit tracked point records/results.

4. **P1 — PNG visibility remains incomplete.** Actual selected B is honored, but `show_corrected=False, show_raw=True` still draws the corrected spectra plus raw data. Feature energy, shift, and splitting do not apply selected branch/feature visibility; with `visible_features=[]` the energy plot still has two curves. Slope-fit overlays are still absent. Complete the captured state contract across all four products; assert actual artist values and visibility. Missing spectra currently produce a blank spectra PNG and legend warning rather than an explicitly omitted/unavailable product.

5. **P2 — Raw square-grid orientation remains wrong.** `_mcd_data` now protects corrected square arrays, but the separate `raw_values` transpose branch still transposes whenever dimensions match `(energy_count, field_count)`. Apply the same documented-orientation guard to raw data.

Status: **Checked core blockers resolved; export still requires changes. Complete UI/integration review remains pending.**

## Selected-feature helper review — 2026-09-11

Reviewed `core/mcd_feature_analysis.py` and ran its six focused tests: **6 passed in 0.169 s**. The helper uses actual channel B arrays, keeps channel names rather than inventing valley labels, and only calls the local fitter for an explicit local method. Three important correctness issues need changes:

1. **P1 — Canonical energy plus wavelength-ordered columns breaks the catalog and selected tracking.** This newly checked `McdResult` contract exposes a core detector issue: `_energy_axis` sorts `result.energy_ev` and uses that permutation on raw/MCD columns, even when the coordinate is already canonical ascending energy and the columns remain in wavelength acquisition order. Reproduction: canonical 401-point energy 1.5–1.9 eV; true Gaussian at 1.680 eV; wavelength and numerical columns reversed while canonical energy remains ascending. Catalog detection reports **1.720 eV**, then `analyze_selected_feature` seeds that wrong energy and returns unmatched with zero tracks. Apply the established spectrum column ordering independently of canonical coordinate sorting. Cover this actual result contract, not only fixtures whose energy and raw columns share order.

2. **P1 — Unselected/rejected tracks leak through the export payload.** `analyze_selected_feature` filters `analysis.tracks` into the top-level `tracks`, but `_analysis_dict(analysis, ...)` serializes the original full analysis into `analysis_results`. Reproduction with a Gaussian present on inc/dec branches and selecting an inc candidate: top-level contains two inc channel tracks, while nested export analysis contains four tracks including both dec tracks. Export consumers use `analysis_results`, so saved results disagree with the selected/accepted tracks. Serialize the filtered analysis consistently, including kind/branch rejection, and test actual nested output.

3. **P1 — Multiple same-channel targets overwrite each other in the export representation.** In the target loop, `analysis_results[channel_source] = ...` replaces earlier analysis whenever valid manual one-to-many links select several spectral features in the same channel. Top-level tracks may contain all targets, but exported nested analysis retains only the last. Preserve a separate feature/analysis identity inside each channel or provide an unambiguous list schema and adapt exporter/UI together.

Status: **Core energy-order fix and selected-helper payload fixes required.** UI and latest export edits remain excluded from this bounded pass.

## Third export fix re-review — 2026-09-11

The 13 export tests passed in 6.016 s. Real `SlopeAnalysis.to_dict` values now serialize using scalar range columns. Nested `PeakShiftResult.source` separates channels; candidate inventory is metadata-only; raw square-grid orientation is protected. Raw-only selected-B spectra and omission of unavailable spectra/empty features are covered. Three important export issues still need correction:

1. **P1 — The snapshot builder mislabels real processed data energy.** Verified with a temporary CSV passed through actual `process_mcd`, wavelengths `[500,600,700]` nm. The processed canonical energy axis is `[1.7712028,2.0664033,2.4796840]` eV. `build_mcd_export_snapshot` changes it to `[2.4796840,2.0664033,1.7712028]` while independently reversing source data columns into ascending energy order. The original final MCD row `[-0.047619,-0.090909,-0.130435]` becomes `[-0.130435,-0.090909,-0.047619]` with the wrong coordinate attached. Sort the canonical coordinate independently and apply acquisition-column ordering only to raw/corrected matrices. Existing real-process test checks file existence rather than coordinate/value correctness.

2. **P1 — Branch visibility does not consistently control exported curves.** Artist inspection with `visible_branches=['B increasing']` still returns `slope B decreasing`, a B-decreasing feature energy/shift curve, and B-decreasing splitting. Feature branch filtering currently happens only inside the optional `visible_features` filter; slopes/splitting have no branch filter. Apply branch visibility unconditionally and feature/pair selection to splitting. Assert each product's actual plotted labels/points after filtering.

3. **P1 — Invalid retained windows silently become a full-spectrum mean.** `_window_values` was changed from rejecting missing center/width to returning `nanmean(data['values'], axis=1)`. A malformed/stale retained record `{'window_id':'stale','metric':'integral'}` therefore exports a mean over the entire energy axis, irrespective of the requested integral metric. Restore explicit invalid/uncomputed handling; valid current-item fallback must capture its actual window before invoking export.

Status: **Export still requires changes.** Current helper/core fixes and UI were excluded from this bounded pass.

## Core coordinate and helper payload fix verification — 2026-09-11

Ran `tests.test_mcd_analysis` and `tests.test_mcd_feature_analysis`: **28 tests passed in 0.480 s**. Independent reproductions verify all three findings from the selected-helper review are resolved:

- The canonical ascending energy/reversed wavelength-column Gaussian now detects at **1.680 eV** and produces status `ok` with two measured channel tracks. The payload serializes with `allow_nan=False`.
- Selecting an inc candidate produces two inc tracks in both the top-level display list and nested export analysis; dec tracks no longer leak through.
- Two manual links to different same-channel feature IDs remain separate analysis records, with keys containing source, feature ID, and method.

Status: **No unresolved important findings from the bounded core/helper review.** This is not an integrated UI/export approval; those components require their own final review.

## UI readiness-gap audit — work in progress, 2026-09-11

At root's request, inspected the current unified page and relevant MainWindow/controller adapters against the full plan. These are implementation gaps to converge on before final review, **not a final verdict on unfinished code**. Sent directly to the UI owner and root.

1. Complete independent retained feature/window numerical snapshots, inclusion checkboxes, inspection by clicking, explicit Update, and source/staleness validation. Current retention stores window settings only; Save uses all retained windows and the current feature payload.
2. Use the same requested MCD metric for displayed per-branch traces and slopes. Current BL draws a mean as one joined branch curve while MainWindow computes slopes from the chosen metric. Slope control changes currently call only `draw_idle`; provide recomputation and readable slope/SE/status output.
3. Connect fixed calibrated/optional inferred valley mapping and explicit same-branch `E_K-E_Kp` computation/export. The mapping control currently changes labels; BR splitting expects a point key the tracker does not produce.
4. Select TR spectra using each channel's actual measured fields and report each nearest actual B. Current four-line view uses the paired field grid for both channels. Turning both branch controls off currently falls back to all rows.
5. Wire advanced reference/prominence controls into request arguments/cache keys. Make manual seeds valid analysis records (`energy_ev`, kind/source/branch) and selectable from the default MCD filter path.
6. Reject completion of an earlier selected-feature request when a newer selection is pending, and block snapshot export of stale/processing payloads. Current pending-key scheduling leaves a window where an old result can display/save under a newer candidate.
7. Finish the agreed feature presentation: shift default, absolute energy on a labeled secondary axis, meaningful per-curve identities/visibility, and a computed splitting product. Search controls now derive a midpoint/window, but must validate actual ordered bounds and preserve the exact computation settings.

Final UI testing should cover these interactions from normal source selection; no preview-only injection or fixed YZ365 data is sufficient.

## Fourth export and integrated UI review — 2026-09-11

Review scope expanded after UI readiness was declared. Re-ran baseline `delta.py`: eleven intended implementation/test files differ from the task baseline. Export suite: **16 tests passed in 7.617 s**. The actual processed coordinate/value reproduction now pairs ascending energy with the correct MCD columns. The UI is **not ready for approval**; the normal workflow still fails and several agreed numerical/retention contracts are absent.

### Verified blocking issues

1. **P1 — Normal Save Results fails at the UI/export map contract.** Ran `artifacts/unified-mcd-baseline/check_load_workflow.py --export`, which copies a real YZ365 CSV into a temporary folder and uses normal Open File/loading. Load took 1.616 s; catalog completion 10.712 s (155 candidates). Selected peak was 1.635985 eV, but this run produced no completed track payload. Export raised `TypeError: unhashable type: 'dict'` because UI puts a numerical map dictionary in `plot_state.selected_map`, while `_selected_map_data` uses this value as a map-name key. No XLSX, PNG, or Processed result was created. Pass a defined selected-map name and captured map collection through the agreed API; ensure successful tracked current-item export from the actual entry point.

2. **P1 — Retained features are metadata, not retained numerical results.** `retain_selected_feature` copies the candidate dictionary only; neither retained features nor retained windows contain the accepted numerical payload/trace and fit diagnostics. Save always uses the current track payload. The global Include selected feature control only decides whether candidate dictionaries are appended to `features`, which the exporter intentionally treats as metadata; it does not include/exclude numerical tracks. There are no independent per-item inclusion checkboxes or click-to-inspect bindings. Implement immutable numerical snapshots with independent feature/window selection, inspection, explicit Update, and stale blocking before freezing the export.

3. **P1 — Displayed MCD trace differs from its selected metric and fitted/exported result.** `_draw_mcd_vs_b` always uses `nanmean` and draws one joined acquisition-order curve across both branches. MainWindow separately computes the selected metric's slopes. A standalone reproduction with `window_metric='integral'` still draws mean values `[0.016259,0.016521,0.015736]`. Branch toggles do not filter this trace. Display the exact per-branch metric values used by slope analysis and export, and show their slope/SE/status, not only a count of valid fits.

4. **P1 — Default shift reference is silently overridden.** The updated selected-worker call always supplies `reference_energy_ev=selected.center_ev`, even while the control says `E(0) per channel / branch`. The helper treats this as an explicit manual reference, so shifts become `E(B)-catalog_center`, replacing each track's measured/reference E(0). Pass no override for the default mode; provide an actual editable manual reference value for explicit manual mode.

5. **P1 — Valley mapping/splitting is not connected to production results.** The combo relabels measured channels, but the production path does not call the new mapping/splitting enrichment or pass a splitting product into export. BR looks for `splitting_ev` on ordinary tracking points, where none exists. Wire calibrated/explicit optional inferred fixed mapping, preserve unknown/ambiguous outcomes, compute same-branch `E_K-E_Kp`, and display/export the same result.

6. **P1 — Exported Combo map PNG is blank.** With a real `process_mcd` result and valid window, builder-default Combo map captures the cube, DAT has values, but the first saved figure has **zero collections**. Neutral selected-cube rows carry branch `''`, then `_plot_pngs` excludes them against inc/dec. Preserve the selected neutral map through plotting while keeping branch-specific cubes filtered appropriately.

7. **P1 — Branch visibility replaces captured retained trace values.** Export filters the numerical MCD grid before reducing retained windows. A captured window with direct trace `[10,20,30,40,50]` and only increasing rows visible should retain `[30,40,50]`; the exported workbook instead contains `[0,.5,1]`, recomputed from the current map because the original direct-trace length no longer matches the filtered grid. Apply the same original branch mask to captured direct values; never replace them with fresh calculations.

8. **P2 — Cursor movement leaves stale displayed B labels and axes.** The native screenshot shows B control **1.32865 T** while both top plot titles remain **+0.12343 T**. `set_selected_b` changes titles and may change Y limits, but the blit path redraws only in-axis dynamic artists, leaving static labels/ticks stale. Rebuild the relevant background when static text/limits change or provide an independently updated readout. Per-channel actual nearest B should also appear in spectrum labels.

9. **P1 — Manual seed remains unselectable in the normal default filter.** Standalone reproduction with an existing MCD candidate: adding a 1.65 eV manual spectrum seed leaves the old MCD candidate selected, combo count one, and one unused manual record. The manual record is appended separately without switching to/showing the new candidate. Select the newly created valid seed immediately and preserve it across same-source refresh.

### Additional acceptance gaps to resolve in the same implementation pass

- Map interaction still requires legacy Ctrl-click rather than a direct horizontal B-cursor interaction, and plot-local bounds/color state are not comprehensively applied to the unified renderer.
- Fixed source labels and feature/curve identities must remain readable. Mapping and manual association controls need real state and worker arguments, not presentation-only changes.
- Successful export callback currently only shows a status string; refresh the source's Processed/history state after a coherent revision is finalized.
- BR needs the agreed shift view with an absolute-energy secondary axis and usable curve visibility. Current shift selection plots shift again on a second meV axis instead of preserving the absolute-energy overlay.

Native smoke completed with spectrum changes after B movement; measured updates were **684.6, 17.0, 13.5 ms** (three observations, not a distribution). Screenshot: `artifacts/unified-mcd-baseline/native-current.png`; inspected visually. The initial full-redraw recovery remains much slower than subsequent blits. Normal workflow and saved numerical contracts take priority over further timing/polish work.

Status: **Changes required; integrated acceptance has not passed.** Findings were sent directly to root and relevant owners. No production or test files were changed by review.

## Fifth export fix verification — bounded, 2026-09-11

The 17 focused export tests passed in **8.636 s**. Independent checks confirm the two newest exporter blockers are resolved:

- Actual `process_mcd` default Combo export now creates one map artist whose numerical color array matches the selected Combo cube exactly.
- An immutable direct trace `[10,20,30,40,50]` exported with increasing rows only retains **[30,40,50]** in the workbook, rather than being recomputed. A mismatched two-value direct trace is explicitly rejected with `ValueError`.

The selected-map API now rejects non-string selection with a clear contract error; the UI must supply the map name and captured maps correctly. **The two bounded export fixes are approved.** This does not close the remaining integrated UI/retention/reference/splitting findings, which were excluded while their owners continue implementation.

## Selected-helper extension review — bounded, 2026-09-11

Combined core/helper suite: **33 tests passed in 0.520 s**. Independently verified default helper calls preserve their existing E(0) reference when no override is provided; an explicit 1.67 eV override changes both displayed and nested-export point deltas consistently. Fixed/inferred/unknown APIs are explicit, and exact unmatched fields remain unavailable. Two new extension issues require correction:

1. **P1 — MCD association is promoted into an invalid valley pair.** `_linked_track_pairs` groups every successful association sharing an MCD root and chooses one pos/neg track without checking feature kind or analysis method. Reproduction: pos **peak** energies `[1.60,1.61,1.62]` and neg **dip** energies `[1.63,1.64,1.65]`, at B `[.5,1,1.5]`, both automatically associated with the same MCD extremum. `enrich_valley_analysis(..., mapping_mode='energy_based')` returns inferred mapping and `splitting_status='ok'`, with three −0.03 eV splitting points. Sharing an MCD extremum does not establish that a peak and dip are the same resonance in opposite valleys. Require compatible feature/method pairing evidence and keep MCD-to-spectrum association distinct from K/Kp pairing. Unknown or incompatible pairing must not produce splitting.

2. **P2 — Exact search bounds discard the entire valid trajectory.** The non-local path does not constrain candidate detection/tracking to `search_window_ev`; it widens a symmetric seed range, then rejects a whole track if any accepted point is outside the exact window. Actual synthetic Gaussian centers `[1.675,1.6775,1.6825,1.685]` eV track successfully without bounds. Requested `[1.676,1.686]` returns `unmatched`, zero tracks, and no reason, losing three valid inside-window measurements because one point is outside. Apply bounds during candidate/tracking selection and preserve valid points with explicit unavailable/out-of-window gaps. Ensure status/reason explains the retained/missing data.

Status: **Helper extension requires changes.** UI, retention, and their in-progress integration were not reviewed in this bounded pass.

## Helper extension final fix verification — bounded, 2026-09-11

Combined core/helper suite: **35 tests passed in 0.504 s**. Independent repetitions close the two extension findings:

- A shared MCD association root now returns `unpaired` with the explicit reason that MCD associations do not establish a valley pair. An explicit peak/dip channel relation is also rejected with a feature-kind reason; an explicit compatible peak/peak relation yields the expected `E_K-E_Kp` value.
- The shifted-Gaussian window case now retains exactly three inside-window measurements and marks the fourth `window gap` with null energy/delta. The E(0) reference is recomputed from remaining valid evidence; top-level and nested export tracks are identical and strict JSON serialization passes.

**The two bounded helper fixes are approved.** Remaining UI and retention integration work was not reviewed during this pass.

## Astra bounded retention review — controller/store (2026-09-11)

Scope: `ui_qt/mcd_result_retention.py` and its focused tests, before ongoing unified-page integration. No production or test edits. Independent command `.venv/Scripts/python.exe -m unittest tests.test_mcd_result_retention -q`: **9/9 passed**. Separate offscreen Qt reproductions exposed these blockers:

- **P1 — stale current fallback bypasses validation**, `get_included`, approximately lines 405–418. An empty store with `source_generation=5, current_window=_window(4)` returns a generation-4 window. The combined selection branch must apply the same generation/status checks as explicitly included and kind-specific fallback items.
- **P1 — all-unchecked selection exports an unselected current item**, same combined fallback block. Add one unchecked retained window, leave retained features empty, and provide a completed current feature. `get_included` returns `{'window': (), 'feature': (current_feature,)}` despite zero checkboxes selected. Current-item fallback should be available only when no retained results exist; a retained selection with everything unchecked must remain empty or reject save.
- **P1 — nonempty metadata is accepted as completed analysis**, `_coerce_feature`, approximately lines 275–303. A snapshot containing only `track_payload={'status':'processing', 'tracks': []}` with no outer status becomes `computed=True` and passes included-result save validation. Check the helper payload's completion status and actual accepted numerical trajectories, rather than mapping nonemptiness.
- **P1 — explicit Update replaces the wrong kind**, `update_selected`, approximately lines 527–536. With a selected row in each list, click the feature last and invoke the shared Update action. It replaces the selected window because the presence of any current window row wins; the inspected feature remains unchanged. Record the inspected kind, or make the Update target explicit and unambiguous. Existing update test has no current snapshot, so it tests a no-op rather than replacement.

Additional integration boundary: `bind(save_results_btn=...)` calls `save_selection()` without a current generation and discards its returned payload; it does not trigger export. The page should own a generation-aware export action, or the controller needs a corresponding signal/callback.

Existing focused tests do verify copied/read-only window arrays and independent checkbox/inspect signals. Full helper-payload-to-export preservation still requires integration coverage after the above guards are fixed. Verdict: **changes required**; findings sent directly to root and Luna retention owner.

### Astra retention follow-up — four reported triggers closed

Independent rerun: **13/13 focused retention tests passed**, plus all four original offscreen reproductions now pass: stale combined fallback raises `StaleRetainedResultError`; an explicit all-unchecked store raises `NoRetainedSelectionError`; the processing/empty-track feature raises `UncomputedRetainedResultError`; shared Update replaces the last-inspected feature while leaving the selected window unchanged. The generationless save-button auto-binding was removed; integration uses the explicit selection API.

Verdict: the four reported reproduction cases are closed. This was intentionally bounded; ongoing UI integration and full helper/export payload validation were not re-reviewed. No production edits.

## Astra UI numerical follow-up — changes required (2026-09-11)

Scope: current unified view, selected-feature worker adapter and main-window plotting wiring; retention integration still in progress. Existing core/export verification reused. No redundant full suite or native timing rerun, and no production edits. Four independent offscreen reproductions establish remaining numerical blockers:

1. **P1 — default slope metric disagrees with the displayed trace**, `MainWindow._compute_unified_mcd_slopes`. Field-signed absolute mean takes the unsigned absolute-mean branch without multiplying by sign(B). For B=[-1,0,1], the fitter received [0.017312,0.018873,0.017312]; displayed branch reduction requires [-0.017312,0,0.017312]. Reuse the same metric reduction for display, retained values, fits and export.
2. **P1 — map click confuses map and acquired-row indices**, `McdUnifiedView._on_map_click`. With selected-map fields [-2,0,2] and acquired pair_b=[2,1,0,-1,-2], clicking -2 selects +2. Convert the clicked map field to the nearest acquired field before setting selection.
3. **P1 — primary Splitting view ignores valid splitting records**, `_draw_feature_vs_b`. It looks for splitting_ev inside ordinary track points. The helper puts these values in the separate splitting payload. With a valid 0.003 eV splitting record, selecting Splitting (meV) and leaving the optional ΔK checkbox at its default false plots only NaN.
4. **P1 — feature overlays mix energy and shift units**, same method. Energy primary plus the optional secondary overlay plots delta_energy_ev=0.002 under a Shift (meV) label instead of 2.0. The ΔK overlay also plots meV directly on the main Energy (eV) axis. All artists must use the units stated on their corresponding axis.

Additional important wiring gaps found by inspection:

- `_queue_unified_selected_feature_analysis` silently converts equal/reversed explicit search bounds to None. Invalid bounds must show a reason instead of using a wider implicit search.
- `_draw_map` hardcodes colormap and percentile limits and does not apply the existing manual XY/color-limit controls or expose a colorbar.
- `_draw_mcd_vs_b` truncates fit output with details[:5] and never displays differences. Six valid fits hide later results; required slope differences, SE, sample counts/ranges and invalid-fit reasons need an accessible result presentation.

Exact reproduction outputs and actionable locations sent directly to the UI owner and root. Four original retention cases remain closed from the preceding bounded review; final combined UI/export/retention verdict is pending these fixes and completed integration.

### Astra frozen UI + retention handoff re-review — changes still required

Bounded independent offscreen checks (no full-suite/native-smoke duplication): map-click -2 T now selects -2 T; primary splitting 0.003 eV renders 3 meV; the secondary shift 0.002 eV renders 2 meV. The live slope function now applies sign(B). These view-specific triggers are closed.

Two remaining **P1** handoff failures were independently reproduced by invoking the actual MainWindow methods on an isolated harness and capturing exporter-builder arguments (no files exported):

- `_current_unified_window_snapshot` still uses unsigned absolute mean for the default field-signed absolute metric. B=[-1,0,1] retained trace=[0.017312,0.018873,0.017312]. It disagrees with the now-correct live slopes and displayed trace, and the exporter deliberately preserves the wrong retained arrays.
- `_queue_unified_mcd_export` adds current_payload to numerical_results unconditionally. With one checked retained window and one unchecked retained feature, validated selection contains zero features, yet the builder receives the unchecked current feature's track. All numerical feature/splitting payloads must derive from the validated retained/current-fallback selection only; inspection must not change export inclusion.

Additional frozen-state findings remain open: selected retained windows' frozen slope payloads are not passed as their fit results (the builder receives the current global slopes); invalid equal/reversed search bounds still become None; map manual XY/color limits and colorbar remain unwired; slope differences and later fits remain inaccessible because the display still truncates details[:5]. These were reported previously and not closed by the latest files.

Verdict: **changes required**, limited to the stated functional issues. Existing core/export tests and root's consolidated/real-source smoke remain the broader verification evidence. Findings sent to root and the UI owner; no production/test edits.

## Astra final combined functional verdict — approved within reviewed scope

The previously reported numerical and retention blockers are closed. This final pass reused prior core/export suite evidence and root's broader regression/native checks; it did not repeat those suites or operate live experiment files.

Independent focused checks on the final code:

- Selected colormap renders `viridis`; manual X/Y limits and color limits match supplied values; the map has a colorbar. The nonzero reversed/equal search-bound guard rejects before worker dispatch by inspection.
- All six valid slope fits remain in the display, and a high-minus-low difference is correctly labeled `high_positive−low / inc` with its SE. Previously rerun map-click, primary splitting and secondary-unit cases remain closed.
- Actual `_current_unified_window_snapshot` retains the correct negative/zero/positive values for field-signed absolute mean.
- Actual `_queue_unified_mcd_export` passes no feature/splitting payload when only a window is checked; the unchecked current feature does not enter the builder.
- Selected frozen window fits/differences survive the builder handoff and `_slope_table`: retained slope 2 and difference 3 are exported with `window-a`, despite current live slope 99. The nested-list blank-cell defect is closed by flat records with window metadata.
- Explicit all-unchecked retained selection never reaches the builder. A completely empty retention store still supplies the completed current window/feature fallback.

These final handoff checks used an isolated Qt harness and intercepted the builder (no output files or production edits). The initial harness missed the newly required controller adapter; after supplying the real method contract, all five retention handoff checks passed. Reported focused suite counts remain owner/root evidence, not rerun by this reviewer.

Nonblocking presentation limitation: the compact slope plot still omits invalid-fit reasons, n and actual-range details; those remain available in exported diagnostics/tables. Root owns the final real-load/export/status QA and consolidated affected tests. No remaining P1 finding in the reviewed functional paths.

Final root-reported QA received after this verdict: affected **34 tests passed (19.55 s)**; real normal-load/export smoke produced one accepted peak track at 1.635985 eV, four PNGs, MCD/Slopes/Energy/Shift sheets, and **Processed=true**. Load 2.104 s, catalog 13.3 s (155 candidates), export 2.535 s. Earlier 64 legacy regressions passed in 32.3 s. These complement the independent bounded checks above.

## Astra bounded follow-up for 2026-09-12 user regressions

Functional changes pass this bounded review; native layout remains pending root/owner correction.

Independent verification: three focused tests passed (catalog completion queues selected tracking, center refresh retains existing artists, top-five prominence selection then energy ordering; 5.407 s). Additional isolated Qt checks confirmed a filter change emits one semantic selected-ID event, a stable catalog rebuild emits none (no recursive selection loop), smaller MCD catalogs are energy sorted, an empty feature panel displays its explicit unmatched reason, moving the MCD window keeps the same axes/trace lines while expanding the Y range, and the center refresh updates both the owner slope cache and view payload. The stale retained-fit risk identified during implementation is closed by updating that owner cache. No prior approved scientific helpers/export code was re-reviewed.

Root's real native QA remains authoritative for first-load analysis and presentation: default five candidates and selected ID are present, automatic analysis can correctly report unmatched with a reason, and map artists survive center updates. Root reported center updates of 200.9/281.2/601.5 ms; this is improved model reuse, not evidence of uniformly fast rendering.

**Remaining layout blocker (root-observed):** at 1280×720 and also the larger native capture, first-row X labels overlap second-row titles, the long BR status title crosses into BL, slope text overwrites plotted curves, and bottom labels clip. These are concrete fit/visibility defects within the requested 2×2 layout, not a redesign request. Sent targeted correction to the UI owner. Functional approval is conditional on root closing this native-layout QA.
