# DRR background and scientific reference

[Project home](../../README.md) · [简体中文](../zh-CN/background-reference.md) · [User guide](user-guide.md) · [Data and exports](data-and-exports.md) · [Development](development.md)

This reference describes background construction, matching, processing order, validation limits, and provenance. For loading data and ordinary plot controls, start with the [user guide](user-guide.md). A valid file, fresh cache, successful numerical reconstruction, or lower residual is not by itself scientific validation of a reference.

Historical evaluation statements about V64/V66, YZ383, 49-degree/94-degree scans and original/repeat acquisitions describe those particular data. They are not universal defaults or acceptance of new P/N pairs.

## Contents

- [Background workflow and processing order](#workflow)
- [External references and assignment](#external)
- [Measured BN matching and group comparison](#bn)
- [BN calibration and its validation contract](#bn-calibration)
- [P/N selection, versions, and saved backgrounds](#pn-workflow)
- [Experimental general P/N draft generators](#pn-general-drafts)
- [Raw reconstruction methods](#pn-raw)
- [Common-texture diagnostics and bounded reconstruction](#pn-texture)
- [Explicit feature protection and local comparison](#pn-features)
- [Scientific review and regional reports](#pn-review)
- [Raw reconstruction in the app and caching](#pn-app)
- [Manual interval backgrounds and suggestions](#manual)
- [Sample-fitted background and peak protection](#fitted)
- [Shared fine texture, periodic prediction, and fringe reference](#fitted-texture)
- [Fitted-background exports, templates, diagnostics, and caches](#fitted-records)

<a id="workflow"></a>
## Background workflow and processing order

The default native interface opens **Background…**. Its source tabs are **Self, BN, P, N, P + N, Fitted**; External is the processing category for selected reference files, not an additional source tab. **Advanced tools…** opens source-specific tools, and **Back to background** returns to the source draft. **Apply background** validates and loads the draft before publishing the result; **Close** retains the applied reference. In the Qt Widgets compatibility dialogs, the corresponding labels include **Background settings…**, **Advanced**, **Apply**, **Fixed current background**, **Use background**, and **Use in settings**. The protocol descriptions below retain Widgets labels where relevant; controls such as **Group BN**, **Assign to selected**, and the detailed advanced inspectors belong to their documented Widgets workflow and are not a promise that the native panel exposes every control. In native settings, **Use files** offers **Same files for all measurements** and **Match files to each measurement**. Saving a draft never substitutes for applying it.

Self offers **First frame / Last frame**, with Last selected for a new draft. Changing the choice edits only the draft. New BN/P/N/P + N references default to averaging all frames within each file, then averaging files with equal weight. **Advanced tools… → Reference frames** retains explicit first/last-frame overrides. Different gate conditions do not make interchangeable repeated frames; background validation still applies. A saved P/N composite is a **Saved composite · single-frame background**; its output frame selector does not reselect original P/N source frames, whose choices are recorded separately.

The calculation order matters:

| Mode | Order |
| --- | --- |
| External, including per-measurement BN/P/N assignments | Correct each measurement using its assigned reference, then average measurements equally; differentiate the resulting group DR/R. |
| Fitted (sample) | Align raw measurements to common measured energy/gate support, average intensities equally, fit one group background, then calculate `I_mean / B_fit - 1` and differentiate. |
| Multiple source files in one external reference | Reduce frames independently within each file, then average reference files equally. This is separate from repeated-measurement averaging. |

For example, nine measurements corrected by one BN and six by another contribute 9/15 and 6/15. Within a batch using one BN, averaging raw measurements before division is equivalent to averaging their individually corrected DR/R.

Opening new CSV measurements first checks for a saved background. Without one, background settings open and no provisional Self plot is calculated. Cancel leaves the selection waiting; **Choose background…** resumes. Reload, returning to DRR, and clearing the background do not silently select Self or repeatedly reopen settings. Missing/conflicting saved references are reported; Reload can retry restoration. Valid saved references, explicitly kept backgrounds, and precomputed DAT/XLSX maps load directly. Basic settings defer preview canvases and fitting imports until needed.

Loading processed raw CSVs through **Load CSV** or the measurement picker restores the latest export recipe for that exact selection: frame choices, member assignments, regional references, and fitted settings. An explicitly kept common background has priority; ordinary reloads retain later manual edits. Restoration runs on the load worker. Missing/incompatible saved references stop loading for review instead of falling back to Self. An explicit P/N candidate pool retains its selected files on group changes and prepares new matches; saved exports do not replace that pool. A frozen saved composite is reused after integrity/native-grid checks without regeneration or requiring unchanged original P/N inputs.

**Keep background across groups** retains a common reference across failed/superseded CSV loads and validates it against each new group. Automatic BN comparison can still produce a review while Keep is enabled but cannot automatically replace that reference. Heterogeneous BN assignments and group-specific calibrated BN cannot be fixed. Without an explicit Keep choice or exact saved recipe, the chosen source-type policy governs group changes; new BN mappings await review.

<a id="external"></a>
## External references and assignment

Native settings show available files and selected references side by side for BN/P/N/P + N. **Use files** chooses a common reference or per-measurement matching. Source-tab changes retain each source's files and assignment method. Search edits only that source draft; **Show all CSV files** and **Browse files…** permit other files without adding another source type. Clicking/double-clicking a candidate adds it and its checkbox toggles selection. Double-click, Delete, or **Remove** removes a selected entry; **Clear all** also removes hidden/out-of-view references. P/N generation has independent single-role pickers with **Use selected** and **Close**.

File lists enumerate filenames/timestamps without inspecting CSV grids or processing history; selection changes do not rescan the folder. **Load catalog** in the P/N tool explicitly loads acquisition details and saved versions. File pickers start in the requesting workflow's folder and discard stale filenames from previous dialogs.

P-only and N-only references are used directly. Candidate ordering prioritizes matching known acquisition conditions, including rotation, with newest-file-first ties. Same-angle candidates without known conflicts are expanded first; other conditions and unknown angles remain labelled and available. Full filenames wrap, and search/asynchronous catalog refresh preserve selected paths. Browse imports are validated only on Apply. The older Widgets inline selector uses one searchable list without the measurement picker's group/files/chosen panes or Add/OK steps: common BN is single-select, direct P/N allows repeated-file checkboxes.

### Matching a selected P/N pool

P/N starts with **Match selected background files**. Choose **P only**,
**N only** or **P+N composite**, then choose the background files to make
available for this group. Matching uses only that selected candidate pool;
an empty pool waits for file selection instead of searching the experiment.
Add or remove candidate files to update the suggestions automatically, or
clear the list to start again. **Rematch selected files** reruns matching
within the selected pool. **One background for all** keeps the common reference option,
and the composite generation and saved-history entry remains available.

Inspect the mapping, select several measurement rows, and use **Assign to
selected** to give them one compatible candidate from the pool. The choice
shows the rows' current assignment, or indicates multiple backgrounds when
their assignments differ. Missing or ambiguous matches must be resolved,
and flagged matches require review before **Apply**. Matching uses acquisition conditions, native
spectral grids and available timing evidence; it does not generate a new
P/N composite or establish which spectrum is physically cleanest. Each
measurement is corrected with its assigned background before measurements
are averaged equally. DRR export metadata saves these assignments so the
exact measurement group's saved recipe can be restored later.

<a id="bn"></a>
## Measured BN matching and group comparison

In BN, **Match BN by batch** automatically matches acquisition batches;
**One BN for all** selects a common reference explicitly.
**Choose BN manually…** beside the match summary opens that selection directly,
including when matching finds no result. Its inline list contains measured BN files
only, putting the same angle first, followed by unknown and different angles.
Other known conditions, cached grid compatibility and modification time
break ties; known incompatible grids appear last and cannot be accepted.
Visible labels explain the angle and grid status. Ordering uses filenames
and cached metadata without reading measurement bodies or acquisition logs.
A manual selection remains a draft until **Apply**; cancelling keeps the
match result and current background.
**Match BN** searches named measured BN CSVs in the current experiment;
**Advanced** can restrict that search to candidate files you choose. This
pool only limits automatic matching; checking several BN candidates does not
average them or apply them as the current background. Matching
checks known sample/position, acquisition settings, constant gate and the
native spectral grid. Verified capture run/experiment IDs and matching
settings/grid define acquisition batches. Files without a verified run
remain separate and are marked unverified; sharing a date, filename suffix
or suggested BN does not establish a batch. A shared candidate is assessed
using the sum of acquisition-time gaps across the batch and checked against
each measurement's individual nearest candidate. Disagreement, ambiguous
timing or excessive age is flagged for review instead of forcing a majority
choice onto the batch. This is a timing/consistency heuristic, not spectral
SNR optimization or evidence that the optical state stayed unchanged.
If every file has a resolved suggestion, acknowledging a batch conflict
applies those reviewed per-file choices without forcing a common BN.
Any unresolved measurement blocks Apply; it is never dropped from the average.

The main result summarizes measurement counts and verified capture runs per
BN, with complete filenames wrapped across lines. Separate verified batches
may share a compact summary row when their agreed BN is the same; their
run count remains visible. Conflicts and unverified batches remain
explicit. **Match details** (or **Review details** when flagged) expands individual file pairs, timing evidence,
gaps and warnings; a compact review summary remains visible when collapsed.
Missing conditions, incomplete logs using modification-time
proxies, gaps over 24 hours or runner-up margins under 60 seconds require
acknowledgement; these are review thresholds, not physical-confidence
estimates. Equal-distance or incompatible matches remain unresolved.
No BN request falls back to P/N. Each matched BN uses all its frames;
separate BN acquisitions are not pooled.

In the Qt Widgets workflow, after applying an automatic BN mapping, **Group BN** in the main sidebar
switches between that mapping and one compatible mapped BN for the whole
selected group. The main DRR/d2E view updates when calculation succeeds;
no additional Apply step is needed. Every measurement keeps equal weight,
and current axes, color limits, gate and SG settings are retained for comparison.
Failed or superseded requests keep the previous valid result. Overrides
apply only to the current measurement selection; **Auto** restores its
original mapping. Original measurement and BN CSVs are unchanged.
Verified results and their energy derivatives are cached privately for the
active group, bounded to three choices and 96 MiB of array storage. Revisiting
a cached choice avoids loading and differentiation; fixed-signature checks
still guard publication. Source/plan, ordered group, axis or processing changes
invalidate reuse. The cache keeps private copies so displayed arrays cannot
alter a saved result; oversized results remain usable without being cached.

Confirmed unchanged mappings are cached in application settings, including
review acknowledgement. Changed measurements, BN files, timing logs or new
automatic candidates invalidate reuse. Approvals from the older per-file-only
matching policy require a new batch review. Repeated per-point log entries
cache their resolved output paths during the request; all point/run and
chronology checks remain in place. A bounded in-memory event index also
reuses unchanged acquisition logs across different measurement selections.
File identity, size, modification and change timestamps are checked before
and after reading; changed or partially read logs are not reused. Output-path
ambiguity, frame counts, run identity and point chronology are validated for
each selection, including when its events come from the index.
Rebuilt match plans also reuse complete gate-profile inspections in a bounded
process-local cache after checking file identity and change signatures.
Missing, incomplete, changed or cancelled inspections are never cached.
In **Advanced**, **Follow the current background type**
keeps the chosen type on group changes; new BN mappings await review.
**Fixed current background** explicitly keeps a common external reference;
heterogeneous BN assignments and group-specific calibrations cannot be fixed.
All selected measurement files still contribute equally to **one** DR/R
average, followed by the requested energy derivative. The summary shows
the actual background type (BN, calibrated BN, P/N or other external) and
measurement count per background; hover to inspect the complete mapping.
Saving the result preserves those assignments for reuse. Raw CSVs are unchanged.

<a id="bn-calibration"></a>
## BN calibration and its validation contract

**BN → Advanced tools… → Compare BN references…** (Widgets: **Background settings → Advanced → BN calibration**) compares the selected measured BN reference
separately for the current group. It evaluates the original BN, a bounded
local fringe shift, and a bounded local fringe shift with gain adjustment.
Each of the two fitted families produces fixed 25%, 50%, 75% and 100%
correction candidates. The unshrunk fit is computed once per family, then
the log background correction is scaled: `B_alpha = B0 * exp(alpha * log(Bfit / B0))`.
Alpha is not optimized inside the fit, so the fitter cannot compensate for
a weaker correction by increasing its raw parameters. Zero correction is
the original measured BN.
It reads raw intensities and fits constant shift/gain parameters against
the actual production D2E, using a fixed response basis derived only from
the original BN. Each training repeat and selected gate is processed before
combining the objective; sample spectra are not averaged before fitting.
It does not repeatedly correct an already calibrated background. The preview
shows candidate backgrounds, DR/R and D2E curves, D2E improvement, the DR/R
constraint result, and the full report. The same candidate BN produces both
DR/R and `d²(DR/R)/dE²` through the application's actual energy-derivative
pipeline with the recorded SG window and polynomial order.

### Repeats, objective, and noninferiority

The explicit Advanced automatic BN comparison/application setting enables
comparison after loading a group. Manual comparison defaults to draft-only.
Unchanged inputs and settings reuse a saved comparison. A new comparison
can take substantially longer than a fit alone because it also refits weak
test features and reruns candidate selection; it runs in the cancellable
background worker. Only the candidate
selected before final validation can be applied automatically: fitting,
candidate selection and final confirmation use separate repeated files,
with at least four repeats, including at least two final confirmation files.
D2E is the primary objective: its held-out residual must improve, while DR/R
must remain within the declared noninferiority margins. DR/R improvement is
displayed but is not itself required. The default per-repeat DR/R residual
allowance is the larger of 2% of its original value and 0.00001 in DR/R units;
the same constraint is applied to individual training repeat/gate rows.
These fixed operational tolerances are not measured noise floors. All gates are checked for
degradation, including overlapping windows over the complete energy axis
and the correction's transition edges. Temporal stability and disjoint
BN-frame-half sensitivity checks must also pass. BN halves
require averaging all frames with an even frame count of at least two in
each reference file; they are not independent acquisition sessions.
Insufficient evidence leaves the original BN in use. Only the constant
shift and shift/gain families are compared; phase slopes, flexible background
splines and automatic SG tuning are not part of this version.

### Training bands and SG bandwidth

Fitting reads complete spectra from training repeats so production SG and
regridding retain their actual context. Only the configured training bands
contribute to the loss, but other energies in a training repeat are not
independent holdouts. Selection and final confirmation use separate whole
measurement files. Reports record both the read channels and scored channels.
**Fit energy ranges (optional)** explicitly replaces the legacy training band
with multiple disjoint intervals. Each interval is detrended independently;
gaps never connect their quadratic trends. Empty input keeps the legacy band.
The report records per-band response rank, bandwidth and parameter-consistency
diagnostics. These diagnostics do not declare a region physically trustworthy.
Additional bands must be chosen from experimental evidence, not automatically
added because their curves look flat.
The training loss is BN-projected D2E residual energy; the displayed held-out
D2E metric is quadratic-detrended production D2E residual RMS. They are
different quantities. Local D2E safeguards also inspect undetrended RMS so a
broad artificial curvature cannot be hidden by the scoring detrend.
Automatic use also requires a contiguous training scoring interval with
enough channels and energy span to contain one actual SG window. A longer
SG footprint can suppress the BN response while spreading a weak sample
feature throughout the fit band. If that bandwidth check fails, candidates
remain available for preview and the original BN stays in use. The program
does not silently shorten the SG window or change the declared intervals.

### Feature influence and candidate-selection sensitivity

Each candidate that passes the selection-set checks also needs a
training-only feature-influence check before it can compete for selection.
A fixed protocol uses 17 positions across the correction and SG context,
both signs of a 0.0002 log-intensity Gaussian perturbation, and widths of
1.5 and 3 native energy spacings. Each perturbation is common to all training
repeats and gates. Linear sensitivity first screens all 68 probes. Candidates
that pass also undergo a finer check around the three highest-risk coarse
centers: offsets of -1, -0.5, 0, 0.5 and 1 native spacing and Gaussian widths
of 1, 1.25, 1.5, 2, 2.5 and 3 native spacings, with both signs. After clipping
centers to the measured axis and removing duplicates, this adds up to 168
probes. Every generated probe must pass. This refinement covers local
neighborhoods of three centers; it does not cover every gap in the coarse grid.
Three probes selected from the combined coarse and refined checks, including
the worst DR/R and D2E responses, are checked by actually refitting the same
unshrunk candidate family on perturbed training data and then applying the
same fixed correction strength. Each training gate is checked
separately against the signal transferred through the frozen background.
More than 5% maximum shape error in DR/R or actual D2E vetoes application.
Screening is a local approximation, not a guaranteed error bound; these
finite probes do not prove preservation of every unknown feature. Only
candidates passing both checks compete on D2E benefit, with deterministic
family/strength ordering and a 0.01 improvement margin for replacing the
current choice. This can retain a weaker valid correction when the full
correction absorbs a weak feature.

Before final confirmation, three declared high-risk Gaussian training
probes additionally rerun both raw fits, every strength, candidate
qualification and the complete selection rule. Selection spectra stay
unchanged; final files are physically excluded from this rerun. The audit
measures output changes from switching family, strength or back to original
BN, rather than hiding such jumps behind a later fallback. A shape error
above 5% vetoes automatic use. This is a finite selection-sensitivity check,
using the combined coarse and refined probe risks. Older saved results that
lack refinement evidence must be compared again before automatic application.
It is not a guarantee for all possible peaks or selection boundaries. Final
confirmation can only confirm or veto the already frozen candidate.

### Trusted support and final confirmation

Automatic application additionally requires an explicitly declared trusted
background interval covering the correction support. The default trusted
interval is empty, and **1.94–2.02 eV** is marked uncertain. A lower ripple
score near 1.98 eV alone therefore cannot replace the original BN. Declaring
an interval trusted is an experimental judgment, not a conclusion inferred
from smoothness. The default D2E improvement thresholds are 10% and an
absolute 0.1 (DR/R)/eV²; both must pass for every confirmation repeat. These
are operational cutoffs, not statistical confidence levels. A failed final check retains
the original BN instead of choosing another candidate on the final data.

Protected intervals are checked in both raw DR/R and the actual full-axis
derivative, since interpolation and Savitzky–Golay processing can spread a
correction beyond its local support. The checks use operational tolerances;
lower residuals and passing these checks do not establish physical truth or
prove preservation of unknown features. Changing SG settings invalidates
the current joint validation claim and requires comparison at the new
settings before derivative plotting or paired export. Failed revalidation
automatically restores the original BN without recording a manual choice
that would disable later automatic matching. A saved certificate can be
verified in the background after reopening, even with automatic fitting
disabled. Old DR/R-only, dual-improvement and previous D2E-priority saved comparisons
remain available for preview and original-BN restoration, but cannot be
activated under the current shrinkage and selection-audit policy without a new comparison.

### Storage, reuse, and original-BN restoration

Comparisons, source snapshots and immutable versions are stored under
`.drr_bn_calibration` in the experiment folder. Reopening the comparison
restores its settings and report. Activated versions are bound to the exact
measurement group, original BN, frame choice, SG settings and recorded
inputs; changed bytes invalidate reuse. Exact comparisons can be reused
from history when switching back to previous settings, while numerical
pipeline changes require a new comparison. Keeping the background fixed is disabled for these versions,
and loader checks prevent copying them to another group or wrapping them
into a composite reference. **Use original BN** restores the original
selection and records that choice so automatic checks do not replace it
again. Original measurement and BN CSVs are never overwritten.

<a id="pn-workflow"></a>
## P/N selection, versions, and saved backgrounds

**P + N → Advanced tools… → P / N drafts and saved versions…** (Widgets: **Background settings → P/N → P+N composite**) opens the P/N workspace
for generation and saved versions. Combination and X-range editing are
available separately under **Advanced tools… → Build composite…** (Widgets: **Background settings → Advanced → Edit background**).
The P/N page has **P back** and **N back** tabs. Each lists raw CSV
files from newest to oldest by **file modification time**, not acquisition
time. Click a row to select or replace that side; the selected P and N remain
visible below the tabs. The reconstruction protocol takes one original CSV
per side. Each source's frames are reduced independently, never mixed into
ordinary baseline-file averaging.

The catalog uses known acquisition conditions and the native spectral grid
to identify files for the current measurement group, including arbitrary
angles. Select P and N yourself in their respective tabs. After choosing one
side, the other tab shows each file's time difference from that selection to
help you compare nearby files; its chronological order stays unchanged.
File suffixes such as `_001` and `_002` do not bind pairs. Changing one side
preserves your selection on the other side. Time proximity is a selection
aid, not confirmation of acquisition completion or scientific suitability.
Unknown conditions and conflicts remain available in the selection details.
**Advanced** contains manual file browsing, frame reduction, generation mode,
training-file overrides, import, full history, and forced regeneration.

The default follows the current measurement group. An unchanged group with
an explicitly applied, fresh version restores that version in **Use saved
background** mode. A new group starts in **P/N reconstruction draft** mode;
sharing a P/N pair with an earlier group does not select its old V66. Saved
versions remain available through explicit history selection. Saved-version
times are displayed in the computer's system local time; stored timestamps
retain their original timezone information. An explicitly
used draft remains restorable as that same draft; freshness and restoration
do not mean scientific acceptance. To reuse a reviewed V66, explicitly
select its saved version and click **Use background**; generation is unnecessary.

If no saved result exists, **Import saved…** accepts an existing composite
CSV with its metadata. Automatic generation remains experimental and starts
only when **Generate** is clicked; it is not a general V66 generator.
It saves the CSV, immutable source snapshots,
generation metadata, and version index under
`Processed Data/DRR/PN Backgrounds`. **Regenerate new version** keeps earlier
versions and forces fresh P/N source fitting. Source content, frame reduction, parameters, algorithm, and any
declared generation dependencies determine exact reuse. Previously generated
backgrounds can still be explicitly reused for another measurement group;
this does not certify suitability or silently generate a replacement.

This window shows the P and N source curves and the saved background, plus
generation and version controls. It does not duplicate the main view's
ΔR/R and d²E maps or compute measurement maps while browsing backgrounds.
Choose **Use in settings**, then **Apply**, and inspect both complete maps in the main DRR
view using its processing and display controls.
An incompatible native spectral grid blocks application. Applying records the
measurement association and selects one generated CSV in External. The
chosen group-change strategy determines whether that common reference is
kept. Validation prepares a read-only usage draft; the current-group guard
precedes committing its association. Merely generating, previewing, or cancelling does
not apply a background.

**All saved pairs** opens history even when original files are missing.
Changed or missing inputs require explicitly choosing the saved snapshot;
regeneration needs the original P/N files. **Import saved…** registers an
existing composite CSV plus metadata under the selected pair while retaining
its original provenance. Copy the whole background-store directory to
preserve version history and associations.

<a id="pn-general-drafts"></a>
## Experimental general P/N draft generators

**Experimental sample-assisted draft** and **Experimental P/N only draft**
are optional generation modes. Neither is selected by default. The
sample-assisted method (`pn_sample_assisted_curvature_v1`) compares repeated
sample spectra with P/N at multiple energy scales to inform regional source
choice. P/N supply the intensity and envelope; sample spectra and BN never
replace them. Moving or inconsistent sample structure limits the evidence
used for selection. Source protection is heuristic, and ambiguous shared
resonances remain unresolved. This is a new general draft, **not a replay
of the prepared V66 method or proof of equal scientific quality**.

With at least four selected measurement files, their displayed selection
order is split before analysis: files 1, 3, ... inform generation and files
2, 4, ... provide separate checks. Fewer files all inform generation, with no
independent holdout; one repeat cannot support repeat-based source choice.
The mode shows these roles. Samples must have matching native energy and
gate coordinates and compatible known acquisition conditions. It never
assumes a particular angle, sample name or number of gate points. This split
is withheld from the current generation call, not a claim that previously
examined experimental data constitute blind validation.

Actual training-file content, order, complete-range subset and generation
parameters are included in the background fingerprint. Sample-context
matching does not override the saved-version preference. A change to
training data requires a new explicit generation; changing only withheld
check data or derivative settings does not change the background. Separate
full-range diagnostics (raw intensity mean before division, SG 11/21 points,
order 3) are saved under `PN Backgrounds/Evaluations` for new sample-assisted
drafts. Their statistics are descriptive, not a quality ranking or an
automatic pass; both complete ΔR/R and d²E still require review in the main
view at fixed processing parameters and display scales. Loading an existing
version does not rerun these checks. BN may be used there as an optional
final-effect comparator, never as a template for generation.

The P/N-only generator is labelled `pn_regional_curvature_v1` and
saves **drafts requiring review**, not scientifically accepted results.
It screens residual structures at multiple scales, selects P/N regions,
and joins their log-intensity curvature with smooth transitions; uncertain
regions remain reported. It does not use BN or sample spectra. Shared
resonances can remain ambiguous and regional levels can change, so both
complete maps must be assessed in the main view. Previously assessed prepared V64/V66 results
retain their own method identity and batch-specific provenance when imported;
the new general generators do not reproduce the full V64/V66 research chain.
The historical real-data checks of the two general draft generators did not
meet the joint full-range ΔR/R and d²E criteria. They remain separate from
the raw reconstruction backend described below.

<a id="pn-raw"></a>
## Raw reconstruction methods

### V66 reconstruction backend

**Raw V66 reconstruction backend:**
`core.drr_pn_reconstruction.generate_from_files(folder, p_file, n_file, training_files)` recomputes the adopted source-resonance fits, energy-dependent
alignment, sample-conditioned classifications, regional reference and final
V66 construction from original CSVs. It accepts one raw file per P/N role,
all frames, and exactly two explicitly selected training measurement files.
It never loads the old fitted curves, old source masks, saved V66 or BN as
generation inputs. Evaluation measurements remain outside this entry point.
The returned `PnGenerationResult` can be exported through
`save_composite_background(output_csv, result.document)` and loaded using
**Import saved…**; generation itself does not save, apply or overwrite a
background. Keep the exported CSV and its metadata together.

The backend preserves the adopted regional recipe, including its three
pure-P regions and repair of the third region. Unsupported geometry raises
an explicit error; it never falls back to the unsuccessful raw-PN draft.
Gate counts are derived from the input, while the two-repeat comparison is
part of the declared protocol. P/N currently require equal, even frame counts
of at least four. First/second-half checks reuse this run's full-data source
fits and classifications, recomputing their references and bridge constants;
they are conditional repeat checks, not independent validation.

In the historical reconstruction evaluation, both original measured groups were reconstructed from their raw files,
with stage comparisons and complete-range ΔR/R and SG 11/21-point checks.
Numerical reproduction of V66 establishes this migration's fidelity, not
scientific acceptance of every new P/N pair.

### Adaptive and local methods

**Adaptive reconstruction under validation:** the same entry point accepts
`method="generalized"` to select P or N repair regions from the current
source states and spectral resolution. It has no fixed angle, energy band,
number of P regions or ordinal region to repair. It reports unsupported or
contradictory regions instead of marking them clean. Original-batch full
and conditional-half outputs reproduce V66 exactly. However, a second P/N
acquisition at the same position did not pass joint full-range dR/R and
d2E review: a long selected-source tail changed the baseline, and persistent
fringe differences produced fixed vertical structure.

A separate experimental `method="local_evidence"` confines shape changes
to exclusive-source intervals and compact smooth joins, keeping the current
parent background exact outside each support. This removes the extended
tail change, but the tested original data gained a fixed join structure and
a confirmed weak shoulder became shallower. Neither successful numerical
execution nor a better texture-fit score is scientific acceptance. These
methods do not use BN or a saved background as generation inputs.

Experimental `method="conditioned"` keeps the existing regional construction
for bounded source regions without opposing exclusive evidence. Boundary
regions or regions containing contradictory evidence can receive only the
compact local repair, provided its support does not contact a retained
regional repair. Decisions use the current source states and native
resolution, without angle-specific or fixed-energy rules. Short unresolved
source evidence stays explicitly marked. Conditional halves use the same
decisions and their own coefficients. This is a separate experimental method;
neither the backend default nor the app's default generation is changed.
The recorded real-data review preserved the original V66 outputs exactly and
reduces the long-tail baseline change in one repeat acquisition. A residual
fixed band and the other repeat's dense fringe mismatch remain, so the joint
full-range dR/R and d2E acceptance has not passed. Inspection of raw common
P/N texture places substantial repeat mismatch before bridge construction;
changing the joins alone does not resolve that mismatch.

<a id="pn-texture"></a>
## Common-texture diagnostics and bounded reconstruction

### Raw common-texture diagnostic

**Experimental raw P/N common-texture diagnostic:**
`core.drr_pn_texture_compatibility.analyze_raw_common_texture` accepts raw
P/N frames, two training measurements and an explicit eligibility mask. It
returns diagnostics and a proposed log-background increment without saving
or applying it. Only a bounded native-pixel shift and gain of the raw P/N
common fine component are fitted. Six energy-block checks with a 35 meV
guard report both the complete block and the eligible subset; the same
transform is checked against P, N, each frame half, three gate groups and
each training repeat. Q4 and medium-band leakage are reported explicitly.
Eligibility and shared training information make these conditional checks,
not independent scientific validation. `apply_common_fine_delta` constructs
an explicitly requested candidate and preserves the parent exactly for a
zero increment. It does not carry over obsolete curvature/PPoly metadata.
This diagnostic is separate from the app's generation modes; optimization
or block-prediction success does not enable automatic reconstruction or
certify complete dR/R, d2E or weak-feature preservation.
The historical frozen real-data probe rejected calibration for both original
acquisitions, retaining their backgrounds exactly. Both repeat acquisitions
improved eligible-block predictions, but residual fixed bands and dense
full-map derivative fringes still prevent joint quality acceptance. These
proposals have not replaced the saved V66 or entered the default generator.

### Optional bounded fine correction

**Optional bounded fine-correction diagnostic:**
`core.drr_pn_texture_support.bound_raw_common_texture` consumes a frozen
common-texture analysis without fitting new parameters. It retains separate
continuous eligibility intervals, guards the Q4/shift input windows and the
complete SG21 endpoint span, and uses C2 transitions inside supported runs.
Short fragments are left unchanged. The original six-block proposal rule
remains necessary; bounded predictions and actual coverage are checked again
for every block. Rejection returns zero applied increments while preserving
the proposal for diagnosis. Source and derivative splines have global
coefficients, so these guards do not prove exact derivative isolation.
Saved regional records describe correction coverage only; they do not certify
the underlying background, and endpoint preservation is not proof of a
useful correction. This remains separate from automatic app generation.
The historical frozen follow-up applied a nonzero correction only to the 49-degree
repeat acquisition (763 of 1340 points). Its previous added endpoint
derivative excursion is removed in the complete-product check, while the
original fixed band remains. The 94-degree repeat has no continuous support
in one required test block and therefore retains its parent exactly; both
original acquisitions also retain their parents. These safe fallbacks are
not improvements, and complete-background scientific acceptance is still
not established.

### Conditioned bounded reconstruction and version 2

The raw reconstruction API also exposes experimental
`method="conditioned_bounded"`. It rebuilds the conditioned parent, source
eligibility and common-texture registration from the current P/N frames and
two training measurements, then applies the supported fine increment to the
full background and conditional halves. Checks, BN and historical saved
backgrounds do not enter construction. Rejected increments retain the
current parent exactly. The final sampled background and its parent have
separate provenance; parent polynomial coefficients must not be interpreted
as coefficients of the fine-corrected curve. The dialog's **P/N
reconstruction draft** action uses this method. It is offered for a new
measurement group; a fresh, explicitly applied association opens in saved mode.
Its native CSV plus metadata can be imported through **Import saved** when
the original P/N, training inputs, protocol and implementation fingerprints
all match. Importing creates a draft; applying it remains a separate action.

The current `conditioned_bounded` version 2 retains unapplied fitted source
components in the native differential residual. A component rejected for
source subtraction can no longer enter the other source through the shared
alignment term. Fit/source acceptance thresholds are unchanged. The source
alignment policy is included in generation provenance and cache identity;
old saved versions remain readable and are not overwritten or relabelled.
The historical `v66`, `generalized`, `local_evidence` and `conditioned`
methods retain their original alignment policy for reproducibility. Thus
the exact parent retained after a rejected fine increment is the v2 run's
own conditioned parent, not the independently generated historical method.

<a id="pn-features"></a>
## Explicit feature protection and local comparison

### Protecting real features

**Explicit protection of real features:** in reconstruction mode, enable
**Protect real features**, enter the energy interval in eV and explicitly
choose the raw **P** or **N** source whose local shape should be retained.
**Add range** supports more than one interval. This is off for a new
measurement group; no angle or energy interval is built into the default.
The complete choices are part of the saved version and generation key,
and are restored when that version is selected. Editing a choice invalidates
the old preview; it does not silently relabel an existing background.

Protection is applied after conditioned reconstruction and optional fine
correction. It extends the source plateau by one complete nominal SG21 span
on each side, blends the selected raw source's log curvature into the
parent's over another span, then integrates the curvature exactly.
A full-axis affine log alignment minimizes the overall log-intensity change.
The selected source's log line shape is preserved up to gain/tilt on the
plateau; exterior parent log curvature is preserved. **Exterior intensity
is not unchanged:** gain, tilt and sample DR/R amplitudes can change across
the complete spectrum. This avoids forcing a short intensity closure that
can manufacture derivative peaks. Invalid intervals, supports extending
to or beyond the axis endpoints and overlapping supports are rejected. These margins target
SG21; different derivative settings need their own review. Half checks use
the full-fit geometry and global integration constants with their own
source frames and parent.

Fixed weak sample resonances must not be dismissed as background merely
because they recur at most gates. Protection is an explicit source choice,
not proof that the chosen source is featureless. Review weak-peak signs,
line shapes and both transition neighborhoods on full-axis derivatives and
held-out measurements. Samples still inform the parent reconstruction and
therefore the integrated result and its alignment; their spectral intensity
and BN are not inserted into the protected source. Neither smooth curvature
blending nor a restored peak establishes accurate absolute amplitude or
absence of derivative artifacts. Compare full backgrounds, joins and
independent repeats; old saved versions remain available.

### Offline local feature comparison

**Experimental local feature comparison (offline API):**
`core.drr_pn_reconstruction.auto_features.analyze_local_feature_sources`
accepts the native energy axis, original P/N frames, exactly two training
repeats, and an explicitly matched current parent background. It returns
an unsaved candidate and an evidence report; it is not wired into the
dialog's default generation or saved-background selection.
Local extrema retain their own centers, anchors, source observations and
repeat contrasts, even when their SG21 windows overlap. Close extrema are
not certified independent physical peaks; unresolved processed peaks are
not artificially split. Source comparison ranks the parent against actual
locally constructed P and N edits, including each full transition support.
Where geometry permits, disputed stationary regions still receive all
three numerical comparisons and signed local contrasts; those diagnostics
do not authorize automatic adoption. The parent can win. Overlapping
construction proposals are excluded without
merging feature identities; remaining proposals undergo a combined-product
check and revert together if any declared guard fails. Stationary source
ambiguity remains a review requirement. The numerical transport score and
repeat-spread tolerance are not physical truth or a confidence interval.
Review the full-axis products, weak stationary features, joins and withheld
repeats before using a result. This API does not save, apply, or certify a
background, and array shape validation cannot establish the supplied
parent's source/frame/measurement provenance.

The optional offline `construction_mode="compact_bounded"` compares a
constrained local edit while retaining the same source-selection guards.
It fits the selected source's log-curvature difference using a compact
degree-five spline: native background values outside the support remain
exactly equal to the parent, the log correction closes through its second
derivative, and its continuous intensity ratio is bounded by
`[1 / 1.05, 1.05]`. These engineering constraints can prevent matching the
requested source. A candidate must meet a predeclared relative curvature
RMS error budget of 0.50 in the requested interval, plateau and complete
support; near-zero targets use a roundoff-level absolute check. Numerically
constructed candidates remain visible even when this fit check fails,
but cannot be proposed for adoption. Neither the gain bound nor this fit
budget certifies physical accuracy. Full-axis derivative processing can
still carry changes beyond the native support, and transition artifacts
need separate review. The default remains `"integrated_curvature"`;
existing Generate behavior and saved backgrounds are unchanged.

<a id="pn-review"></a>
## Scientific review and regional reports

**Feature-scoped review:** residual structure outside the features being
analyzed is not by itself a reason to reject local use. Review the main
branches, weak features, line shapes and repeatability inside the intended
analysis range, including their baseline and derivative neighborhoods.
Compute derivatives on the complete native axis before examining a region.
Stationary structure is not automatically a background artifact. The
retrospective YZ383 review supports conditional qualitative use of the
observed main features in 1.58–1.89 eV, with the uncertain 1.98 eV structure
retained separately. Those sample-specific ranges are not generator
parameters or defaults for other data. Absolute weak-feature amplitudes and
small derivative ripples still require measurement-specific interpretation;
unchanged feature signs alone do not establish quantitative accuracy.

The P/N dialog's **Region details** view reads saved source segments,
transitions, regional/compact repairs, unresolved evidence and optional
correction coverage. For bounded reconstructions, base construction and the
optional fine correction are shown separately. It explicitly marks missing records. File freshness,
native-grid checks, saving and selecting a baseline are separate from
scientific review of the current measurement. Changing the selected version
clears the previous regional report before loading the next one.

<a id="pn-app"></a>
## Raw reconstruction in the app and caching

The app adapter freezes the raw P/N files, two resolved training
measurements, complete-grid settings and implementation identity. Other
selected measurements are checks and do not enter fitting. In **P+N composite**,
select one matched original CSV in each P/N tab, then click **Generate**.
Sample repeats are selected automatically from the current measurement
group: numbered repeats use the first and third in repeat-number order;
a group of two uses both. Unnumbered filenames retain the selected order.
The summary shows the chosen files, with full paths in its tooltip.
**Advanced → Choose sample repeats manually** enables an optional override.
This selection does not scan other groups, rank peaks or change fitting;
identical inputs reuse the same saved result. Fewer than two distinct raw
repeats require selecting a measurement group with at least two repeats.
The current raw protocol requires **All frames** for P and N, matching native grids,
matching sample gate rows, and equal even P/N frame counts of at least four.
Exclusive-source corrections whose fixed supports conflict with opposite
evidence, another requested support or a deferred short island are left
unresolved instead of aborting generation. Their control weights and
background are retained at the source-selection stage; subsequent rendering
and optional texture correction still follow their separate rules. The
report records the deferred regions and reasons; generation remains a draft.
Successful generation saves and previews a draft. It does not apply the
background; **Use background** is the separate explicit action. The
progress area below the workspace shows completed stages, the current
operation and elapsed time, with **Cancel generation** available while busy.
Stage durations vary; this is not a remaining-time estimate. Completion is
shown only after saving and loading the preview succeed. Cancellation keeps
the dialog open; any draft already saved remains in history.
Review the intended features, baseline neighborhoods and repeatability in both products
for each new measurement group. Availability is not validation for arbitrary
P/N pairs. Existing versions, including V66, remain in history. Full dR/R and d2E are inspected in
the main view; the background dialog keeps only its background preview.

Raw P/N source preparation uses a bounded, process-local cache keyed by the
complete P/N arrays and frame order, original file content, spectral columns,
and implementation identity. Changing only the two training repeats reuses
those source fits but recomputes sample-dependent regions, full/half rendering
and bounded texture correction. Changing P/N or implementation invalidates
the cache; restarting the app clears it. Exact complete requests may reuse a
saved result. **Regenerate new version** bypasses both kinds of reuse.
Progress and the saved execution report identify reuse and stage timings.
Fixed source-attribution filters are computed once per stage instead of once
per candidate; frame counts, grids, fitting choices and checks are unchanged.

<a id="manual"></a>
## Manual interval backgrounds and suggestions

**Advanced tools… → Build composite…** (Widgets: **Background settings → Advanced → Edit background**) creates a reusable background from the clean
regions of references such as p back and n back. It can be opened before
selecting measurements. Add sources, choose first/last/all frames for each,
and compare their overlaid spectra with the bold combined result. Drag a
range on the plot or enter its bounds, select a source, and use it for that
interval. Other energies use the default source; Undo restores the prior
edit. Existing X range recipes can also be opened in this editor.
**Save as…** writes a single-frame background CSV and its matching
`.metadata.json` recipe without changing the main view. **Use in settings**
returns the preview; **Apply** applies it with the chosen group strategy. An ordinary
raw-source X-range recipe can be applied directly; generated or frozen source
snapshots must be saved first. Open a saved composite to edit it again.
**Edit selected P/N background** copies a version into a separate manual
draft, retaining its parent provenance without claiming the original
reconstruction method or scientific validation.
Frozen source spectra in the recipe allow reselecting regions even after the
original files move; changing their frame selection requires those originals.
Keep the CSV and recipe together when copying or moving a saved background.
Both files are protected from temporary-source cleanup. Source spectra must
share their original ordered grid. Manual assignments retain direct switches;
the editor reports junction jumps without silently scaling the sources.
Reloaded composites retain their derivative boundaries.

### Auto suggest and short-gap reconstruction

**Auto suggest** screens the selected original references for local and broad
peaks and dips, then proposes intervals from their measured samples. Select
references acquired under compatible conditions, including angle and exposure;
matching grids alone do not establish compatibility. **Keep manual intervals**
preserves existing assignments. **Reconstruct short gaps** is optional and off
by default: it estimates only the slow envelope and retains measured fine
interference texture when clean source windows and both surrounding comparison
bands support it. A smooth bridge alone is not accepted as a background.
Missing or inconsistent texture support leaves the interval unresolved.
Estimated, unresolved and unsupported ranges have distinct plot hatches and
a review summary; a low feature score does not prove a featureless reference.
Review the proposal before **Save as…**; Auto suggest does not apply or save it.
Undo restores the complete previous recipe. The saved recipe includes the
original snapshots, generated result, source choices and diagnostic ranges;
estimates remain marked after reopening and derivatives respect their edges.

### Interval and derivative rules

The same **Edit background** page assigns source files to selected energy
intervals in eV. The current External selection is the default for all
unspecified energies. Add sources, choose their first/last/all frame reduction,
then drag or enter an interval and assign its source. Multiple files in a
source are averaged equally after frame selection. Inspect the combined
preview, then click **Use background**; Cancel leaves the main recipe intact.
Regions cannot overlap or extend outside the measured spectrum. Intervals
include their lower endpoint and exclude their upper endpoint, except that
an interval ending at the spectral maximum also includes that final sample.
Every background must have the same original ordered wavelength grid as
the measurements. Backgrounds switch directly at the specified boundaries;
no intensity rescaling or cross-fade is applied. First, second and mixed
derivatives are computed separately within each background interval and
leave approximately half an effective SG window blank beside a switch;
intervals too short for a derivative remain blank in that product.
Raw ΔR/R retains every measured point. Region recipes and all background
sources are included in export metadata and restored with saved recipes.
**Fixed current background** retains a common manual recipe when changing measurements,
subject to validation against the new spectrum. Editing a group with different
per-measurement recipes requires selecting a common external default first.

<a id="fitted"></a>
## Sample-fitted background and peak protection

**Fitted (sample)**: select original CSV measurements, choose Fitted in Background
settings and Apply. Configure fit previews and peak protection there before
applying if desired. On first use, peak protection
is estimated at each gate of the averaged raw intensity, combining 35 and
70 meV feature screens. A moving peak only excludes that gate's data, rather
than removing its entire swept energy band from every gate. The shared
envelope and per-gate intensity/tilt are fitted alternately using these masks.
**Fit / view background…** shows the actual applied background, its smooth
envelope, and the current gate's protection, with raw-spectrum overlay.
To adjust protection, uncheck **Automatically protect peaks at each gate**;
this preserves the union of all detected gate regions as manual protection.
Then edit resonance intervals in eV (for example,
`1.56 1.64; 1.69 1.77`) or enable **Drag to protect a region**. Choose
**Fit preview**, inspect the result, then **Use in DRR** to apply the edit.
Automatic protection adapts when source data changes; manual recipes retain
their explicit intervals. Broad or weak features may still need manual
correction. If no narrow features are detected, the smooth fit is applied
with a review note; insufficient background support stops the fit.
This mode fits a robust shared cubic spline in log intensity outside the
protected regions, with a gain and linear energy tilt for each gate. It
first aligns selected raw measurements to their common measured energy/gate
support and takes their equal-weight intensity average. It fits one background
to that average and computes `I_mean / B_fit - 1`; existing SG smoothing and
derivative controls then operate on that result. A missing or changed member
fails the group rather than silently averaging fewer files. No BN file
or additional dark subtraction is used. Original intensities must be finite
and positive, and protected regions must leave sufficient background support.
A good residual outside peaks does not establish background accuracy under
the protected peaks; inspect changes with different protection/smoothing.

<a id="fitted-texture"></a>
## Shared fine texture, periodic prediction, and fringe reference

### Correct shared fine texture (learned outside peaks)

**Correct shared fine texture (learned outside peaks)** estimates a
  weak multiplicative spectral response from unprotected data. It requires
  at least 12 gates. At each energy, the available unprotected gates are
  split into lower and upper groups in gate order; each needs at least six
  gates and 15% of the scan, with local agreement between the groups.
  Each spectrum is detrended on its measured energy coordinates over
  35 meV before pooling, using only its unprotected samples. Protected
  values enter neither the local trend nor the pooled texture. Partial
  windows require at least ten anchors and a stable quadratic fit;
  insufficient local support is excluded instead of interpolated.
  This retains support when a moving peak blocks one fixed half of a scan.
  The template
  is learned outside each gate's protection, then applied at supported
  energies to every gate, including gates with a protected peak there. Unsupported
  regions receive no correction; transitions taper smoothly. This does not
  change the SG derivative window. Uncheck it and choose **Fit preview** to
  compare with the smooth envelope alone. **Shared texture correction (%)**
  displays the applied response. **Texture coverage** shows the applied
  weight (0% means no correction, 100% means the full template) alongside
  the fraction of unprotected gates; grey regions receive no correction.
  Coverage is not a confidence score. Persistent stripes in unsupported
  bands cannot be identified reliably from this sample alone.
  Weak stationary sample features can be
  indistinguishable from instrument response: protect them manually or leave
  this correction disabled. Strong automatically detected stationary peaks
  remain protected. With periodic prediction disabled, no fitted texture is
  extrapolated through missing support.

### Predict fringes across protected gaps (validated only)

**Predict fringes across protected gaps (validated only)** is an optional
  extension of shared texture and is off by default. **Max components**
  limits each local fit to 1–5 components (initially 3); passing validation
  may select fewer components. Local models test energy, wavelength and
  pixel coordinates with fixed or gently varying frequency. Frequency
  limits follow the native spectral spacing, including when samples are
  masked or held out. The general model must predict two continuous bands
  at least 34 meV wide. If it fails, a fixed single-frequency energy model
  must pass all four predefined tests: two central bands at least 12 meV
  wide and two outer bands at least 24 meV wide. Each band must also cover
  the requested gap and at least two phase cycles. All models learn only
  from unprotected training samples; frequencies are reselected in every
  test, and predictions must agree between separate available-gate groups.
  Reports retain failed general-model tests when the single-frequency
  fallback passes. Gaps beyond the validated support remain unchanged. Fully
  supported shared texture is retained; accepted predictions supplement
  missing or tapered coverage before normalization and the existing SG
  derivative. These corrections are learned from and applied to the averaged
  intensity, using the same single background as the envelope fit.
  The inspector reports whether predictions added any correction, their
  added coverage, and rejection reasons. Coverage is an applied weight,
  not uncertainty or proof that a feature is instrumental. Compare with
  prediction disabled and protect known sample features; a smaller d2E
  residual alone does not establish peak fidelity. **Envelope held-out RMS**
  in the spline/polynomial comparison evaluates only the smooth envelope,
  separately from periodic prediction validation. Templates, the fitted
  cache and exports retain the option, validation report and actual applied
  denominator. No original measurements are overwritten.

### Fringe reference

**Fringe reference** optionally uses an explicitly selected raw CSV from
  the same sample position and optical path. **Browse** selects it and
  **Clear** disables it; the default is empty. Constant-gate repeated
  frames are retained individually. This supplies only weak fine texture:
  the sample still determines the smooth envelope, and no additional dark
  subtraction is performed. Shared texture must be enabled.
  Sample and reference energy grids must match exactly. The reference's
  own peak protection and independent frame halves must support the
  correction. A shared-texture gap must be at most 25 meV wide, with
  contiguous validation bands on both sides at least as wide as the gap.
  Both bands and their one-pixel shifts in each direction must pass all six
  training-only calibrations (oscillatory residual ratio at most 0.4 and
  correlation at least 0.9). Protected sample values never enter those
  calibrations. Failed regions retain shared texture and any separately
  validated periodic prediction; selecting a file does not imply success.
  Accepted reference texture takes priority over periodic prediction only
  where shared coverage is incomplete. The inspector reports actual
  application, while saved metadata retains validation, the selected path
  and its SHA-256 fingerprint. Editing or removing the reference invalidates
  reuse; select its new path after moving files to another computer.

<a id="fitted-records"></a>
## Fitted-background exports, templates, diagnostics, and caches

**Save both** also writes the single actual group background matrix and fit parameters
to `Processed Data/DRR/Fitted Backgrounds`, including the applied texture and
protection at each gate, all input member identities, and the average-before-fit
processing order, with links in the DRR metadata.

- **Save template / Load template** reuses a named recipe for similar
  measurement conditions. Templates contain protection and model settings,
  never the fitted intensity. Protection applies to the averaged intensity.
  Legacy manual per-file protection is combined as a union for the selected
  group; automatic mode rebuilds gate-local masks from the average. Loading
  a template replaces the group protection.
- The averaged raw intensity is screened across all gates for positive and
  negative narrow features. **Check average** fits/checks the selected
  group in the background and lists residuals, review notes, and the worst
  gate in one row. **Add suggested regions** merges suggestions into the
  average's protection without shrinking existing regions. Refit and inspect before applying. Screening
  is heuristic (35 meV local quadratic, an energy-dependent noise threshold,
  repeated-gate or strong single-gate features, and 5 meV margins),
  so broad or weak resonances can be missed. `Checked` is not proof of the
  background under peaks. Fit diagnostics are retained in exported metadata.
- **Compare methods** overlays the averaged intensity's robust spline and a
  degree 2–4 polynomial in log intensity, using identical protection. The
  plot can switch to both second derivatives with the same SG settings
  captured when the dialog opens. It reports background differences and
  prediction errors on up to three held-out anchor bands using up to nine
  representative gates. This comparison does not automatically change the
  group's chosen method or establish uncertainty inside protected regions.
- Compact fits and quality checks are cached locally across app restarts,
  keyed by source identity, intensity/coordinate contents, axis, complete
  fitting settings, and algorithm version. Gate, color, and derivative-view
  changes do not refit backgrounds. The disposable cache is bounded to
  256 entries / 128 MiB in the user's application cache directory; missing,
  corrupt, or unwritable cache entries fall back to fitting. Original CSVs
  are unchanged; reading/parsing files can still take time on a cache hit.
- Per-gate gain/tilt fits run in bounded batches, retaining independent
  robust weights and convergence for every gate. Masked medians are
  evaluated across the batch without per-gate Python
  calls. Group loads show fitting progress and the current file; the Run Log
  records each file's elapsed time and whether its fitted cache was reused.
  During the current session,
  protection-independent feature screening and held-out validation scores
  are also reused by data content and relevant parameters. Changing only
  protection rechecks existing feature candidates against the new regions;
  changing SG settings recalculates derivatives while reusing valid scores.
  Each review cache holds at most 128 small results / 2 MiB, without retaining
  raw spectra. Temporary held-out fits calculate predictions; full model
  fits retain all-gate quality checks.

## Interpreting historical evidence

The V64/V66, YZ383, 49-degree/94-degree, original-acquisition and repeat-acquisition statements above describe the specific historical evaluations recorded for those data. They are not universal defaults, a validation dataset for an arbitrary new pair, or proof of physical accuracy. In particular, the 1.58–1.89 eV conditional qualitative range and uncertain structure near 1.98 eV are sample-specific review conclusions. The BN calibration defaults in code are a separate, explicit operational policy and must not be inferred from those retrospective findings.

## Implementation map

Use these sources when checking current behavior:

- Native background UI: [`DrrBackgroundSettings.qml`](../../ui_qt/quick/qml/DrrBackgroundSettings.qml), [`drr_background_bridge.py`](../../ui_qt/quick/drr_background_bridge.py), and [`drr_background_restore.py`](../../ui_qt/quick/drr_background_restore.py).
- BN matching/calibration: [`drr_bn_matching.py`](../../core/drr_bn_matching.py), [`drr_bn_batches.py`](../../core/drr_bn_batches.py), [`drr_bn_settings.py`](../../core/drr_bn_settings.py), [`drr_bn_calibration.py`](../../core/drr_bn_calibration.py), and [`drr_bn_store.py`](../../core/drr_bn_store.py).
- P/N protocols: [`pipeline.py`](../../core/drr_pn_reconstruction/pipeline.py), [`app.py`](../../core/drr_pn_reconstruction/app.py), [`feature_protection.py`](../../core/drr_pn_reconstruction/feature_protection.py), [`auto_features.py`](../../core/drr_pn_reconstruction/auto_features.py), [`drr_pn_texture_compatibility.py`](../../core/drr_pn_texture_compatibility.py), and [`drr_pn_texture_support.py`](../../core/drr_pn_texture_support.py).
- Fitted processing: [`drr_fitted_background.py`](../../core/drr_fitted_background.py), [`drr_periodic_texture.py`](../../core/drr_periodic_texture.py), [`drr_reference_texture.py`](../../core/drr_reference_texture.py), [`drr_background_cache.py`](../../core/drr_background_cache.py), and [`drr_background_review.py`](../../core/drr_background_review.py).
