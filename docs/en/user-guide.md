# User guide

[Home](../../README.md) · [简体中文](../zh-CN/user-guide.md)

Use this guide for the desktop controls and scientific workflows. See [Data and exports](data-and-exports.md) for input tables, experiment folders, output packages and Origin import; [Background reference](background-reference.md) for detailed DRR background, P/N, fitting and BN comparison procedures; [Development](development.md) for source launch options and checks; and [Troubleshooting](troubleshooting.md) for problems.

## Contents

- [Workspace and common controls](#workspace-and-common-controls)
- [Finding and loading measurements](#finding-and-loading-measurements)
- [PL](#pl)
- [DRR](#drr)
- [Peak Analysis](#peak-analysis)
- [Compare](#compare)
- [Power Dependent](#power-dependent)
- [SHG Processing](#shg-processing)
- [MCD and MCD Peak Shift](#mcd-and-mcd-peak-shift)
- [Slides](#slides)
- [Tools, logs and saving](#tools-logs-and-saving)

## Workspace and common controls

Normal startup opens the native Qt Quick application. Independent Peak Analysis, DRR Analysis and MCD Organizer windows also use native Quick controls. The older Widgets application remains a compatibility option; launch instructions are in [Development](development.md).

Choose a workflow in the persistent navigation rail or workspace header. **Controls** or **Ctrl+B** hides the resizable parameter sidebar. Side-by-side plots hide the far-left rail to make room while retaining the parameter controls; the header still provides workflow selection and a navigation toggle. Switching workflows preserves their separate loaded data. Revisiting PL/DRR retains the figure, zoom/navigation state and spectrum overlays.

Use **View → Appearance** for system, light, dark or high-contrast appearance; the choice is remembered. Windows, dialogs and plots use Arial, including exported titles, labels, ticks, legends and annotations. Mathematical glyphs may use a symbol fallback. Appearance applies to independent analysis and background windows too.

PL/DRR start with a draggable **70% heatmap / 30% spectrum** split. **Hide spectrum / Show spectrum** is in the bottom-right footer. Spectrum captions are inside the axes; colorbars are horizontal above the maps. Spectrum Y limits follow the visible curves and X range in PL/DRR, Compare, Power, MCD and MCD Peak. Manual Y zoom persists until **Home** restores automatic fitting. The common plot toolbar acts on the active workflow, including Compare.

### Editing numbers

Typed plot/processing values commit on **Enter** or focus change. Label and unit edits also wait until editing finishes. Arrow buttons still adjust values; repeated display changes are combined after you stop. Pending plot refreshes or MCD recalculations wait while another number is being typed. Gate/cursor interaction remains immediate.

Decimal and scientific notation are accepted, for example `1e3`, `1e-6`, `-2.5e4`, within each field's range and storage precision. Integer fields accept scientific notation only if the result is an integer. Idle Quick controls omit trailing zeros:

| Input | Idle precision |
| --- | --- |
| Energy ranges and boundaries (eV) | 4 decimal places |
| Peak fitting/manual centers (eV) | 6 decimal places |
| Voltage and magnetic field | 3 decimal places; small tolerances 4 |
| Temperature, wavelength (nm), angles | 2 decimal places |
| Widths/distances (meV) | 3 decimal places |
| Intensity, color limits, power, ratios, coefficients | 5 significant digits |
| Counts, orders, indices | Integers |

Small nonzero values below `1e-3` and magnitudes from `1e6` use scientific notation. Focus reveals full configured precision, and more precise input remains supported. Formatting does not alter calculations, limits or exports. Legacy controls also compact idle text while retaining their storage precision and specialized DRR formats.

Color-limit arrows use approximately 0.5% of that minimum/maximum span, rounded to a 1/2/5 step: −100…100 steps by 1; −0.01…0.01 by 0.0001. Each product/region adapts independently after Auto or view restoration. Steps respect numeric resolution; equal bounds use their magnitude, or 0.01 when both are zero. Typed input commits at the editor's configured precision.

Mouse-wheel scrolling moves panels and lists without changing dropdowns, numbers, sliders, dials or active tabs. Click or use the keyboard to edit.

### Color range sliders

PL and DRR keep **C limits** in a fixed bar below the plot. One, two or three
regions share the row, with independent minimum/maximum handles, synchronized
readouts and **Auto**. Exact **Min / Max** inputs and **Fix** are in the left
sidebar. Dragging updates those numbers immediately, independently of the image
refresh rate. Type a value and press **Enter** or leave the field to apply it;
unfinished text is retained during refresh, and **Escape** cancels a draft.
Explicit numeric edits remain allowed for fixed endpoints. Narrow windows scroll
the slider row horizontally. The sidebar retains colormaps, region boundaries,
axes and other display options. DRR **Overall / Weak features** remain beside
the sliders; **Weak strength** is in the sidebar. In paired views, the product
selector chooses the colors being edited.
Compare uses its intensity regions or VP range; MCD uses one map range and its
existing shared color lock. Pure spectrum/curve panels have no color slider.

The nonlinear track retains the full editable data range when limits are
narrowed: drag outward for a wider view without a mode switch. Numeric entry can
extend the track. Positions are not proportional to values; read the numbers and
endpoint ticks. Fixed endpoints stay put during dragging. With zero centering
and both endpoints unlocked, dragging adjusts a symmetric range; exact numeric
edits retain their existing semantics. Previews are rate-limited from a 50 ms
target and adapt to rendering cost; release applies the latest range. Large
rectilinear maps use a temporary lower-resolution screen preview while dragging;
release and export use the full data. Color
edits do not recalculate Spectrum data or change its selected scan. Show/Hide
spectrum and the draggable heatmap/spectrum divider remain separate controls.

## Finding and loading measurements

Choose the experiment folder and then the sources for the desired workflow. PL and DRR have separate selections and a source/history catalog; recent folders and local file drops are supported. Dialogs start from the requesting workflow's folder and do not reuse stale filenames from a previous open/save request.

The PL/DRR measurement picker groups acquisitions, shows **Unprocessed / Partial / Processed** badges for groups and files, and retains chosen files while filtering or browsing. Search accepts space-separated keywords in any order, including history and full filenames, for example `YZ365 760`. DRR group summaries show sample/position, conditions, file count and status; differing conditions appear first. Full filenames wrap, with tooltips and Copy filename for original paths.

Click or double-click an available filename to add it; its checkbox toggles it. In the chosen-file list, double-click, press **Delete**, or use **Remove** to remove a file. **Clear all** removes all chosen entries, including hidden or out-of-view files. **Selection actions** contains optional group/repeat additions. **Close**, the title-bar close button and **Escape** leave the loaded measurement unchanged.

The picker can show cached or filename-only rows while inspection continues. Group/repeat actions wait for compatibility checks. Explicit file choices can be submitted and are validated by the loader. Gate summaries combine set/measured channels only when their complete traces match; real differences remain separate. Equal frame-count endpoints are shown once, such as **541 frames**.

Reopening centers the most recently loaded file and acquisition group where scrolling permits. Successful loads update that position; browsing, cancellation and failed loads do not. Main-window **Clear** keeps the browsing position for the same folder and file type. Search/status filters still apply; manual browsing cancels pending automatic positioning.

Folder selection prepares PL/DRR catalogs in the background. During acquisition, lightweight rows remain usable while automatic checks wait for the source inventory to stay unchanged for three seconds; **Refresh** checks immediately. Hidden pickers keep watches and pending inspection active, without replacing main-window load/export status. Source/result changes trigger refreshes; logs and cache writes do not continuously restart scans. A failed directory probe preserves the validated list.

A newer **Load** supersedes the queued selection and cancels obsolete work cooperatively. The current file read or numerical operation may finish before cancellation takes effect. **Clear** and changing experiment folders also invalidate loads. Cancelled loads do not restore an old plot or show a load-failed dialog; a failed newest selection does not fall back to an older queued selection. Exports retain their captured data independently.

## PL

PL provides a one-file heatmap and spectrum workflow with source/history search, saved DAT selection, axis derivation/custom DAT labels, peak detection and single-spectrum Lorentz fitting. Use the toolbar's Gate and previous/next controls to inspect measured spectra. See [Data and exports](data-and-exports.md) for supported inputs and Y-axis choices.

### Display, regions and locks

The native Display page keeps **X/Y Min/Max**, each **Fix**, and **Auto** together. **Auto adjusts only limits without Fix.** Editing a number does not enable Fix. To let a locked endpoint follow data again, release its Fix before Auto. Color Auto uses the visible X/Y range and the selected energy region; Log uses positive finite samples only. If a region has no finite visible samples, or no positive samples for Log, its limits stay unchanged and Auto explains why it is unavailable.

**Linear / Log** tabs select both the plotted scale and the color controls. With **Side by side** enabled, both plots stay visible and the tab chooses which color bank to edit. Linear and Log have independent limits, including every regional endpoint. New PL displays default to **turbo**; a saved custom colormap takes precedence.

**Regions** selects one, two or three energy regions. Boundaries are shared between Linear and Log, while regional color limits are independent. Each region needs measured energy columns. Boundaries have individual Fix controls and an optional boundary display. Changing the region count preserves usable existing/shared boundaries and initializes new ones; three-region boundaries must remain distinct and ordered. Zoom retains their physical energies and shows only the visible intersections. **Clip outliers** remains a separate color option.

Invalid numeric drafts show an error and prevent switching color tabs. Correct the value or press **Escape** to cancel the draft. This avoids silently applying a draft to the wrong Linear/Log bank.

Successful saves retain exact-file display settings in the experiment folder and export metadata. Loading starts from defaults, restores compatible exact-file values, then overlays explicit manual locks; remaining automatic endpoints follow the current data/visible region. A fully compatible saved view retains its saved colors until Auto or an actual visible-range change. Explicit manual locks survive file changes and restarts; restored file locks do not become global manual preferences. Saved Y locks retain their compatible axis derivation/label context. A first save retains the existing processing-history/auto-next behavior. PL exports separate Linear/Log PNGs and one full-grid DAT; there is no extra combined side-by-side PNG option. Shared image-width behavior is described in [Data and exports](data-and-exports.md).

## DRR

DRR supports native CSV self/external references, precomputed DAT/XLSX maps, energy SG derivatives and mixed derivatives. The native Display page uses the common approved `Display.qml` controls. Keep **Data**, **Display** and **Processing** roles distinct:

- **Data** stays above the parameter pages, showing measurement selection, current background summary and **Background…**.
- **Display** controls axes and colors; **More display settings** contains the right Y axis, scale/clip options and display memory.
- **Processing** contains derivative/SG controls and single-spectrum fits. Y-window controls appear only for the mixed derivative.

Use the toolbar's **Gate / Previous / Next** to inspect adjacent measured scan coordinates. **Analysis…** can add the current committed source to DRR Analysis or open its workspace. A derivative view sends the original committed source, avoiding differentiation of an already differentiated input.

### Recommended display ranges

The native DRR bottom **C limits** bar offers **Overall** and **Weak features** beside
each region's color limits, in addition to the existing **Auto**. Overall uses
robust percentiles; Weak features estimates variation within quieter energy
windows, separately from differences between their background levels. It also
keeps a central portion of the visible data within the color scale, so a quiet
patch cannot saturate most of a map with a changing background. Select
**Weak strength** (Gentle, Standard, Strong or Maximum), then click
**Weak features** in the region to apply it. The result reports the applied
limits and saturation percentage. Before numeric rounding and fixed endpoints,
Gentle/Standard/Strong/Maximum limit saturation to approximately 20/40/60/75%,
respectively. Weak limits stay within the Overall range, and stronger contrast
never widens them. These are display heuristics, not noise measurements or proof
that an enhanced structure is a real signal. Both actions retain Fix endpoints and the
selected colormap, including **RdBu_r p=0.60**. Linear weak contrast turns
**Center Zero** off; Log uses positive-data percentiles instead. These actions
change display values only, without smoothing, differentiating or discarding data.

**Trim empty X edges** proposes a range from the full measured maps, retaining
scan rows, color limits and region boundaries. It respects fixed X endpoints;
ambiguous edges are retained. Use **Auto** beside X to restore the full extent
(release Fix first for a fixed endpoint). Trimming does not change analysis or
exported numerical data, and a proposed edge is not proof of absent weak signals.

### Choosing a background

New CSV measurements first look for a saved background. Without one, background settings open automatically and no provisional Self plot is calculated. Cancel leaves the measurements waiting; use **Choose background…** to continue. Reload, returning to DRR and clearing a background do not silently choose Self or repeatedly reopen settings. Valid saved references, explicit Keep backgrounds and precomputed DAT/XLSX maps can load directly. Missing/conflicting references require review; **Reload** can retry saved-reference restoration.

The background window has one source-tab row: **Self, BN, P, N, P + N, Fitted**. Each tab owns its files and assignment method. Self has **First frame / Last frame**, with Last selected for a new draft. Frame changes edit the draft; Apply calculates the result. New BN/P/N/P + N drafts average all frames within a file, then average files equally. **Advanced tools… → Reference frames** retains explicit first/last overrides. Different gate conditions are not interchangeable repeats.

BN/P/N/P + N show available files on the left and selected references on the right. **Use files** chooses shared references or measurement-specific assignments. Search edits the current source's draft; the add/remove gestures match the measurement picker. **Show all CSV files** and **Browse files…** expose other files. Switching source tabs retains each draft. P and N generation use independent single-role pickers with **Use selected / Close**.

**Advanced tools…** provides source-relevant P/N drafts/versions, fitting, composite editing, BN comparison and raw previews. **Back to background** returns without changing the selected type; existing custom references keep their type/files until you explicitly select another source. **Diagnostics** opens the last preview report. Detailed procedures and scientific validation are in [Background reference](background-reference.md).

**Apply background** validates and trial-loads the complete result before changing the plotted association. Failure or cancellation preserves the previous result. **Close** retains the applied reference. **Keep background across groups** keeps an applied common reference even after failed/superseded loads and revalidates it for every new group. Automatic BN comparison remains reviewable but does not automatically replace a kept reference.

Opening processed raw CSVs through **Load CSV** or the measurement picker restores the latest export's background for that exact selection: frame choices, member assignments, regional references and fitted settings. An explicitly kept reference takes priority. Incompatible/missing saved references stop loading with a review message. Ordinary reloads preserve subsequent manual background edits. **Load catalog** on the P/N page explicitly loads acquisition details and saved versions when needed.

Optional automatic BN comparison runs after loading in a cancellable worker. It preserves explicit original-BN choices and activates only a validated candidate after a successful trial load. Changed SG settings revalidate managed candidates; without a current joint certificate, the original measured BN is restored. Reports matching the current group, frame and SG settings can be reopened without repeating comparison. Exports retain measurement provenance and the actual fitted denominators.

### Product tabs, Auto and regions

In **Display → Colors**, **ΔR/R / Derivative** tabs choose the color controls to edit; they do **not** switch the plotted product. Selecting ΔR/R or Derivative in the plot controls also selects its matching color tab. **Side by side** keeps the current color tab. Each product has its own colormap and limits; region count and X boundaries are shared. Hidden values still apply to plotting and paired exports.

Minimum/maximum fields sit together with their Fix controls. Auto changes only the selected product's unfixed limits. **Auto X/Y/V** and regional Auto retain fixed endpoints; uncheck Fix or use **Restore Auto for this load** to release locks. One/two/three-region maps retain every measured point with independent color scales and original cell boundaries.

### Remembering a view

Successful native **Save PNG + DAT** exports remember X/Y/color ranges, one/two/three regions, boundaries, raw/derivative colormaps and limits, Fix choices and zoomed viewport. The legacy Widgets frontend also remembers these settings through its paired **Save both: PNG + DAT** action and PNG-only plot-toolbar save; the native main plot toolbar has no separate PNG-only save. Reloading the same file/group restores its latest saved view, including after restart. The captured view is saved only after successful export, under `Processed Data/DRR/.display_settings.json` and metadata sidecars; these travel with the experiment folder.

For a new file with the same filename wavelength center (such as `720nmc`), X range/boundaries can reuse a center template. Y/color remain automatic unless manually fixed. On a file change, X geometry follows this priority:

1. Explicit **Lock X range across files**.
2. The target file's own snapshot.
3. Compatible manual geometry for the same center.
4. The target center's template.
5. Auto.

X limits, region count and fixed boundaries travel together. Manual X geometry expires when the center changes; unknown centers retain it only on same-file reload. Reloading the current file preserves unsaved manual edits. Incompatible spectral coverage falls back to Auto. Fixing one axis retains saved zoom on the other.

**Display → More display settings** can disable same-file/center memory, enable optional color reuse, or share center templates across experiment folders. Color reuse and sharing start off. **Lock X range across files** also starts off; enabling it locks current X geometry and active boundaries, including preset-restored values, across restarts. Status text and Fix tooltips explain scope.

| Fix label | Meaning |
| --- | --- |
| **Fix M** | Manual lock; X geometry normally follows the same wavelength center |
| **Fix λ** | Inherited wavelength-center template |
| **Fix F** | Current file's saved display |

Editing a locked value or toggling Fix off/on makes an inherited lock manual. File/center locks expire outside their scope and do not become app-wide manual defaults. Valid boundaries stay; out-of-range ones are recentered while retaining Fix source, with three-region boundaries ordered around valid fixed boundaries.

Manual **Fix M** Y and DR/D2E regional endpoints persist across restarts without export. Fixed Y retains axis choice and custom coefficients; whole-map D2E retains its manual color mode until Auto. Unfixed endpoints follow the next measurement. Releasing Fix updates restart memory. **Restore Auto for this load** releases remembered/manual ranges and disables the cross-file X lock. **Clear this center template** keeps individual file snapshots.

Raw view can defer hidden derivative computation and automatic colors until showing that product, requesting its Auto, or saving both products. Large-map Auto calculations run in the background; initial plotting can wait while controls remain usable. Obsolete source/SG/ROI results are discarded and current manual locks/edits are respected. Paired saves capture the data, ROI and pending colors together. Raw-only PNG-toolbar saves in the legacy Widgets frontend do not turn placeholder derivative limits into saved limits.

## Peak Analysis

Plot PL, DRR, Compare intensity or Power intensity, then use **Analyze…** at the right of the plot toolbar or the Workflow menu. The English analysis window runs in a separate process using a numeric snapshot. PL supports detection and optional multi-peak Lorentzian/Gaussian fitting. DRR supports peaks/dips for each supplied product; Compare/Power intensity use the PL adapter. VP is excluded. Repeated scan coordinates require explicit separation or aggregation; measured power values are not modified to hide repeats. Power's own peak workflow remains available for those datasets.

Snapshots inherit colormaps, limits, log/zero-centered normalization and regional scales. DRR has **ΔR/R / 2nd derivative** tabs above the plot; files have a separate tab strip (a single file shows its name). Products retain independent branches/results and share the selected scan coordinate. To enable a missing product, display and send it from the main app. Reopening the same numeric data refreshes display settings while preserving analysis; older saved snapshots need a new handoff for updated display metadata.

### DRR: find candidates, then filter

A new DRR handoff defaults to **All spectra in dataset** and the full energy and scan ranges. In **1 Find candidates**, click **Find candidates** to scan the full energy range of every spectrum and cache both polarities. In **2 Filter results → Keep**, choose **Peaks only**, **Dips only**, or **Both**, then click **Apply filters**; switching types reuses the cache without redetection. Choose **Current spectrum** under Spectra when you want to inspect just one row. Changing scope requires finding candidates again.

New DRR data start in **Auto from data** mode with candidate floor `0`. Find builds a broad candidate cache, then generates and applies initial filters from measured candidate metrics. The analysis panel shows candidate and retained counts. The recommendation is a heuristic starting point, not a physical classification; inspect and adjust it for your measurement. Detection runs in the background with progress and cancellation. **Detection settings** offers Off (SciPy `find_peaks`) or Savitzky–Golay smoothing, its window, and the candidate floor. Raw ΔR/R defaults to SG with an 11-sample window; derivative products default to local extrema, with optional smoothing labeled **Extra SG**. SG does not differentiate the supplied product. Changing the detector, SG window or floor requires **Find candidates** again; filters cannot recover features excluded by the floor or smoothing.

In **2 Filter results**, set the **Keep** type, **Keep range**, preset and quality thresholds, then click **Apply filters**. Draft range and threshold changes do not alter the result until applied. Filtering reuses the candidate cache: widen a range, change polarity or clear a threshold and apply again to restore matching cached candidates without redetection. **Full range**, **Follow plot view**, and **Custom range** choose the filter bounds; Adjust filters include spacing, maximum width/count and neighbor support. A preset such as Balanced, Sensitive or Strict deliberately excludes some candidates; **All candidates** and **Clear quality filters** remove quality cutoffs. Detection width is the width at half prominence, separate from fitted FWHM.

For map data with enough neighboring-spectrum evidence, the automatic recommendation compares seven filter proposals against the same continuity reference. Its heuristic score weights retention of weaker reference candidates at 45%, retention of all reference candidates at 25%, and removal of candidates lacking reference continuity at 30%; available terms are renormalized when evidence is missing. The bounded reference follows sustained, smooth same-polarity trajectories, allows one missing row at a time, and breaks at large scan gaps. It is used for scoring, not physical identification or ground truth: noise can appear continuous and real features can be isolated or move rapidly. When neighboring evidence is insufficient, including a single-spectrum analysis, the statistical starting filters are used instead.

After detection, **Suggest filters** recomputes and applies a recommendation from the cached candidates. Manually editing filter parameters or choosing a preset disables automatic overwriting; click **Apply filters** to use manual changes, or generate a new recommendation when desired.

After choosing the numerical filters, automatic DRR analysis enables **Remove isolated clusters** when sustained trajectories cover at least seven spectra and 5% of the candidates passing quality and neighboring-support filters. Cleanup preserves those trajectories, SNR ≥ 8 or unavailable SNR, and locally repeated same-type features: at least three distinct spectra within a seven-row window, with a width-based energy tolerance bounded to one–three energy samples. Large scan gaps break neighborhoods. These protections favor weak-feature retention and may also retain correlated noise. Single-spectrum analysis and insufficient evidence skip cleanup. **Details** reports the actual outcome. To restore removed points, disable **Adjust filters → Remove isolated clusters** and click **Apply filters**; **All candidates** also disables it. Saved manual choices are preserved; PL fitting is unchanged.

Enable **Preview excluded points** to show faded × markers on the map and selected spectrum. Only exclusions within the last applied X/Y range and feature type are shown. The panel counts the first failing condition (SNR, prominence, width, neighboring support, isolated cluster, spacing, or per-spectrum limit); hover near a marker to read its reason in the cursor bar and hover over that bar for details. Draft edits do not alter the preview until **Apply filters**. Hiding the preview does not change results. Older saved results need filters reapplied to generate explanations.

The sidebar separates **1 Find candidates** from **2 Filter results**. Detection always caches both peaks and dips; **Keep** controls only the retained result. Find also applies suggested or current filters to produce a starting preview. X/Y editors appear only for **Custom range**; single-spectrum analysis uses the selected scan row. Expand **Adjust filters** for numeric quality thresholds, **Detection settings** for smoothing and candidate floor, or **Details** for recommendation and exclusion explanations. **Find candidates** sits in the detection section; **Apply filters** remains fixed at the bottom.

The optional **Balanced** filter preset uses these starting thresholds. Prominence is normalized by the range of the full spectrum, not the selected X range.

| Product | Minimum SNR | Prominence fraction | Support out of 5 |
| --- | ---: | ---: | ---: |
| PL, Compare/Power intensity | 5 | 0.05 | 3 |
| DRR ΔR/R | 5 | 0.08 | 3 |
| DRR 2nd derivative / other labeled derivatives | 7 | 0.10 | 4 |

**Sensitive** lowers SNR by 2, halves prominence and lowers support by 1; **Strict** raises SNR by 2, multiplies prominence by 1.5 and raises support by 1. Their width thresholds use 2, 3 and 4 median energy-sample intervals for Sensitive, Balanced and Strict respectively; same-polarity spacing starts at 2 intervals. These sampling-based values are starting points, not instrument resolution.

### PL and shared controls

PL uses **1 Find peaks → 2 Check peaks → 3 Fit → 4 Results**. Choose all spectra or the current spectrum and an analysis range, then **Find peaks** to detect candidates and apply the default filters. Check them on the plot; changes under **Adjust filters** require **Apply filters**. Choose a shape and **Fit checked peaks**, then inspect residuals and save results. Fitting uses the reviewed candidates without finding peaks or choosing filters again. Return to any step to revise the analysis.

New handoffs default to the full map and full energy/scan ranges. Fits use supplied linear intensities without detection smoothing. Detection changes require finding again; pending filters block fitting, export and sending overlays. Shape or peak-selection changes invalidate fits while preserving candidates and filter values. Saved workspaces retain their choices.

New PL handoffs enable **Recover weak PL features** and **Auto from data**. Candidates have no relative-prominence floor; local raw-minus-SG noise and neighboring spectra provide editable screening. **Suggest filters** compares seven settings while protecting prominent, supported extrema, so a stricter drift setting cannot win by deleting the main branch. Manual presets and edits stop automatic replacement. Changing weak-feature mode requires **Find** again; existing saved analyses retain their original mode.

Weak-feature mode also proposes shoulders from three distinct curvature scales derived from the dominant peak width (a 5–9 meV central scale, evaluated at 0.8/1/1.2 times that scale). Orange hollow diamonds are **provisional shoulder landmarks**, not independently resolved peaks. Their measured prominence is zero; curvature contrast and seed width are separate heuristics, not peak height or fitted FWHM. A positive minimum-prominence filter excludes these shoulders. In map mode they require at least four-of-five neighboring-row support when support filtering is enabled. Raw data and its displayed intensities are unchanged. Close branches use mutual-nearest prediction with up to the configured missing-row gap; no missing measured points are invented, and large scan-coordinate gaps break tracking.

PL fitting limits automatic center drift, tries a second initialization for weak companions, and compares at most three one-component deletions using global and local BIC evidence on the same samples. Manual seeds are not pruned. This bounded comparison is not an exhaustive peak-count search or physical identification. Automatic analysis fits at most eight prioritized seeds per spectrum, retaining excess candidates as explicitly unfitted; more than eight manual/prominent seeds require user selection. Boundary hits, shoulder shifts and structured residuals remain review states. In fitted overlays, cyan circles retain detected positions, orange crosses show unconfirmed detections, and branch lines show valid fitted centers. **Show residual** and the current-fit summary expose model limitations; CSV/JSON and XLSX **Details** retain origin, quality and provisional-candidate fields.

In **2 Check peaks**, choose a spectrum on the map, then click a peak marker on the spectrum to select it. An orange vertical line identifies the selected peak. Drag horizontally to move it, double-click empty space to add a peak, or press **Delete** with the plot focused to exclude the selected peak. **Esc** cancels dragging or picking. Editing is disabled during pan/zoom, computation or pending result changes. **Edit current spectrum** also allows exact numeric positions followed by **Update selected peak**.

The review sidebar keeps only peak selection, Add, Remove and Next open. Under **Exact position / constraints**, set an exact position, width, allowed movement or **Fix center at this position**, then apply with **Update selected peak**. Edits affect this spectrum only and survive Find, filtering and workspace reload; **Reset edits in this spectrum** restores its automatic results. **Whole-map branch selection → Show all branches** reveals checkboxes for entire branches. Edits invalidate fits; continue to **3 Fit** to refit. Model comparison does not prune manually assigned peaks. A fixed center's zero standard error describes a constraint, not measurement precision. The review step gives the spectrum more height by default; manual Map size settings take precedence.

**4 Results** distinguishes detected positions, valid fitted centers and other/unconfirmed positions. **Save results** saves plotting tables in XLSX, a source-export heatmap overlay and complete analysis JSON together; CSV is no longer generated by default. **Save editable session** preserves the workspace for continued editing. **Show positions in main app** explicitly sends the overlay to the main application.

The optional **Asymmetric Lorentzian** adds unequal left/right widths only to the strongest seeded component. Other components remain symmetric, and the default Lorentzian is unchanged. Asymmetry is bounded to ±0.8; exported width is total FWHM. A smaller residual does not establish that weak components are reliably resolved: inspect boundaries, uncertainty and residual structure. JSON retains manual constraints, models, quality metrics and edit history; the default workbook contains numeric tables for plotting.

The plot is on the left and the controls on the right. **Map size** changes the map's share of plot height from 20% to 80%; **Auto** restores the scope-dependent layout. The choice is remembered per dataset. Results appear beside the plot, with **View all results** for the detailed table. Export is explicit; **Apply to main app** sends an overlay only when requested.

With **Zoom** active, drag a visible black-and-white rectangle over the desired plot area and release to zoom. Press **Esc** during the gesture to cancel it.

A fresh handoff initially shows the incoming datasets. Earlier workspace datasets and results are retained: choose **More → Show previous datasets** to display and select them. Closing the option returns to the current handoff. Reopening the same numerical dataset preserves its explicitly chosen analysis scope; legacy implicit DRR defaults are updated for the new workflow.

### Interpreting detection metrics

For ordinary extrema, candidate width is full width at half prominence, not fitted FWHM. DRR and legacy PL SNR is prominence divided by robust noise on the cached detection signal, per finite segment. Noise is MAD / 0.67448975 of residuals after an order-2 SG baseline of up to 51 samples; this baseline only estimates noise and never shifts/refits extrema. New PL weak-feature mode instead uses a local raw-minus-SG residual MAD window of up to 101 samples, with shoulder contrast separately identified. Segments under seven samples lack a noise estimate and are rejected when SNR filtering is enabled. SNR is a heuristic ratio, not confidence probability.

Same-polarity support is counted after shape/SNR filtering and before range cropping/branch association in sorted scan coordinates. Fixed presets use a base tolerance of `1.5 meV` per neighboring row; automatic filters estimate it from the data. The second neighbor has twice the reach. New DRR map analysis and **Suggest filters** enable **Adjust filters → Adapt matching to peak width**: when a base match fails, broad candidates may match within one quarter of the smaller width, capped at twice the base tolerance per row. Extra matches must be mutual nearest neighbors, with the second-nearest alternative farther by more than one quarter of the base reach. This allows limited peak-position jitter without declaring a branch connection. Narrow peaks, unavailable widths and zero base tolerance do not gain extra reach. Turn the option off and click **Apply filters** to restore fixed-tolerance matching without redetection. Saved choices are retained; older saved analyses, fixed presets, PL and single-spectrum analysis retain the previous behavior. **All candidates** disables the option. Required support scales at map edges; scan gaps above 1.5 median steps break neighborhoods. Counts do not recursively promote points. Strong support can shorten branch endpoints: lower support to inspect them instead of increasing smoothing. Applying a preset to an old cache computes missing noise metrics once, then reuses them. Exports/workspaces retain the original candidate pool and metrics.

### Branches, spectra, fits and overlays

In **Edit peaks…**, edit a name to rename, check inclusion or use **Exclude branch**. Names/IDs/colors survive reversible filter splits. **Connect selected branches** conservatively connects cached positions in adjacent scan rows; ambiguous matches stay isolated crosses and missing rows break connections. **Pick seed** opens separate manual tracking with its gap/noise settings.

Click the map or move **Spectrum** to inspect a measured spectrum. The mouse readout shows X/Y and labels; heatmaps also show **Z (nearest)** from the input grid before color clipping/normalization or extra smoothing. Hover for signal label/cell coordinates. Leaving the axes marks the last reading; data/view changes clear it. Log axes report physical scan values using displayed cell boundaries. Spectrum/residual Y is the actual mouse coordinate, not curve-snapped; missing coordinates/Z show `—`. Mouse motion only changes text and does not select a spectrum or run analysis.

For PL, **3 Fit → Fit checked peaks** fits the reviewed selection; **Show residual** displays residuals when fits exist. Choose detected positions or valid fitted centers for the overlay. **Apply to main app** sends positions to matching main-app heatmaps. Numeric source changes invalidate overlays; zoom/color changes do not rerun analysis. Toolbar **Size** sets marker area (default `4 pt²`) without detection/filtering or heatmap rebuilding, and is retained in workspace saves, exports and overlay packets.

**Save results** saves `_peak_results.xlsx`, `_peak_overlay.png` and `_analysis_settings.json` together. The adjacent **Save figure** button saves the overlay alone. PNG uses the main application's normal heatmap export renderer, preserving its layout, colors, title, colorbars and dimensions, with retained peak/dip markers overlaid. It is not the Analyze window's map-and-spectrum layout. Source export settings are captured when opening analysis and refreshed from the live main plot when saving, with numeric source identity checked. Offline exports retain the captured settings. Old histories without verifiable color limits require reopening from the main plot; automatic color guesses are not used.

For DRR, choose **Peaks only**, **Dips only** or **Both** under **2 Filter results → Keep**, then click **Apply filters** before saving. The saved markers use the applied result, as does XLSX; there is no separate image-only feature filter. Rejected-candidate previews are excluded from saved PNGs. Adjust **Size** above the map to change marker area (default 4 pt²).

When the main plot selects a wide PNG (for example 2.0×), **Save results** includes both the original 1.0× `_peak_overlay.png` and `_peak_overlay_2.0x.png`; **Save figure** also saves both sizes. They share the same data, colors and peak/dip markers. Only canvas width increases; height, font sizes and DPI remain unchanged. An analysis of d2E exports the d2E map and detections, not the original DRR signal. Repeated saves and save-as preserve existing files with matching sequence numbers.

The default XLSX contains numeric plotting sheets: **Peaks** and **Dips** for DRR; **Detected_peaks** for PL, plus **Fitted_peaks**, **FWHM**, **Height**, and **Area** after fitting. Each sheet begins with the scan coordinate and units, followed by stable branch columns and adjacent available standard errors (SE). Unreliable or missing values remain blank rather than zero. Area is the fitted component's full analytic area. Thresholds, candidate points, quality flags, failure reasons and edit history remain in JSON/session history. Details, Curves, Components and Settings sheets are not included by default.

Outputs live directly beside the processed file, preserving its complete experimental filename with readable ASCII normalization. Existing outputs are preserved using matching sequential `_02`, `_03` suffixes for XLSX, PNG and JSON; names are not shortened or given random/hash IDs. Raw inputs not yet exported use `Processed Data/PL` or `Processed Data/DRR`. **More** contains save-as and open-folder actions.

Analysis history is saved beside each processed file as `<full filename>_analysis_history.npz` when switching files or closing. Reopening loads only that file's history. Old combined workspaces and nested histories remain untouched while matching records are migrated in the background. `%LOCALAPPDATA%/DPTK/Peak Analysis` holds settings, handoffs and history path bindings/migration indexes/cache instead of an ever-growing combined workspace. **More → Restart** saves before restarting. Source-code/bundled-code restart requirements are in [Development](development.md).

Use the **mouse wheel** to zoom around the pointer, **middle-button drag** to pan, and **Home** (button or key) to restore the full range. **Zoom** rectangle selection remains available for precise inspection. Left-click continues to select spectra and edit peak positions.

## Compare

Compare provides its own native controls and plot workspace for coherent channel groups, manual channel assignment, intensity and VP, shared background correction, independent color limits, spectrum inspection and domain PNG/DAT export. It can render 2–4 selected files in a compare grid. Dialogs retain the requesting channel/workflow; switching pages preserves loaded data, and the common toolbar acts on the active Compare workspace. Its angle references carry into Power, while the two pages keep independent data.

1. Use **Select files…**, choose the **Coherent group**, and review **Auto Detect** assignments. **Manual channel assignments** allows explicit sources for KK, KKp, KpK and KpKp; **Swap KK / KKp** exchanges the first pair.
2. Choose **View → Intensity Compare / Valley Polarization**. Intensity **Channels** offers **KK + KKp**, **KpK + KpKp**, **All four** or **Custom**. VP requires distinct assigned KK and KKp sources.
3. Under **Background and VP**, use **Auto constant background** or enter **Constant background** after disabling Auto. **Also export VP from KK / KKp** adds VP when that pair is available.
4. Adjust X/Y, **Cursor gate**, spectrum visibility and color limits; Intensity and VP retain separate limits. Advanced **Keep bounds on Auto** locks individual endpoints. **Regional intensity colors** enables independent left/right colors and an X boundary for intensity; it does not apply to VP.

Advanced **Angle rules** selects rotation input/output mapping, K/Kp reference angles and tolerance; **Infer Angles** helps derive references. **Source and history filters** searches groups/filenames and filters export status; **Measured Y axis** selects TG/BG/Bias or a custom linear combination. **Export titles and filenames** previews the planned names before saving.

## Power Dependent

Power supports one header-table sweep (`Power_uW`, optional `stage_pos`, numeric wavelength/energy headers; one row per power) and legacy filename power series such as `37.96uW`. Numeric wavelength headers such as `752.5829` convert to energy, with rows sorted by power. Input details are in [Data and exports](data-and-exports.md).

### Choose a sweep or pair

Opening files enters the explicit selection workspace before plotting. **Choose Power Group…** shows newest-modified measurements first, initially 20; **Show older measurements** shows the rest. Rows show identity, date, channels and power range. Modification time orders groups and does not determine pairing. **Refresh** discovers acquisitions while preserving the view; search/status filters retain the selection draft.

Use **Open sweep** for one sweep, **Open comparison** for a unique pair, or **Resolve pairing** for ambiguous channels. **Details / Adjust pairing** shows full filenames and manual assignments. **Swap KK ↔ KKp** exchanges assignments and shows angles. **Open comparison** saves the group assignment in portable `.power-selection.json`; Cancel preserves the previous saved choice, and other groups retain theirs.

A single sweep needs no KK/KKp assignment. Intensity comparison needs two distinct channels. VP additionally needs spectral overlap and valid stage/power pairing. Different power grids are supported. A compatible pairing is suggested with a detailed override; repeated stage positions disable stage pairing, so use power interpolation. Switch **Intensity / VP** on the page after opening the pair.

Grouping prefers matching `*.experiment.metadata.json` sidecars (`files[].path`, `session_id`, `measurement_id`, and optional sample/position/temperature/gates). `channel`, `compare_channel` or `polarization` may identify KK/KKp. Without explicit identity, filename context remains part of the group. Whole-sweep polarization `deg` fields distinguish channels without splitting matching measurement settings; run suffixes and other conditions retain group identity. Bare `deg24`/`deg69` tokens are output/analyzer angles in Compare and Power. Unique legacy pairs can infer two output references within the group; the one nearest configured output K remains K. Suggestions use Compare's angle references.

Discovery includes measurement subfolders such as `power_sweep`; it excludes `Initial Data`, old/archive folders and analysis CSVs without spectra. Nested combinations remain under Processed. **Include filename-based series** is off initially; enabling it finds series with at least two distinct powers, separately per containing folder, avoiding unrelated PL names in the ordinary picker. Channel readiness and processing status are separate.

### Combine segments of one channel

**Combine sweeps…** combines sweep segments of the same channel. Combine KK segments and KKp segments separately, then assign saved sweeps for comparison. Review acquisition compatibility before saving.

Automatic setup selects a reference favoring valid direct calibrations, with coverage breaking ties. Correction subtracts separate backgrounds and scales added sweeps to that reference. Default review shows sweeps, factors and previews. **Advanced settings** can disable automatic setup to choose a reference, no scaling or known factors, plus background bands, exclusions, duplicate policy and metadata.

Scaling removes a separate background from every input. Backgrounds default to separate medians in a shared editable energy band, initially the highest 20% of the common spectrum; choose a signal-free band or use manual backgrounds. Known factors are supported. Overlap estimates do not extrapolate: power interpolation bridges only neighboring powers within 25%, and preview spectra must agree within 2% in measured power. Reports show pair count, factor scatter and spectral mismatch for the applied common factor. Estimated factor scatter or spectral mismatch above 10% rejects correction and blocks saving while keeping the affected sweep/preview visible. Limited coverage is marked provisional. Review saturation and exclude questionable spectra by their displayed 1-based power-order numbers.

Exact equal powers can keep the first, keep the second or average all contributions together; nearby powers remain distinct. Spectra use the first sweep's energy grid within shared coverage, linearly interpolated without extrapolation. Provenance preserves source rows and merge policy in CSV/DAT/JSON. Reusing the same original rows is blocked, even for renamed inputs.

Saved combinations live in `Processed Data/Power Dependence/Combined Sweeps` and appear under Processed. Editable descriptive names and source/settings fingerprints support **Open existing / Create another**. **Back to selection** keeps checked sweeps/review settings, including when adding another sweep. **Save and plot** opens the saved result; **Plot Setup** then controls it. Existing CSVs are preserved using numbered names on collision.

### Fit intensity and linewidth

Click **Analyze peaks**, or enable **Peak analysis → Intensity + linewidth vs power**, for heatmap, selected-spectrum fit/residual, integrated intensity and FWHM trends. The first analysis automatically suggests up to eight peaks and fits all measured powers. Detection normalizes sampled spectra separately across powers and channels, so bright high-power spectra do not dominate weak lower-power peaks. Suggestions still need review, especially for unresolved, noisy, crossing or short-lived peaks. Heatmap/spectrum occupy the left column; intensity sits above linewidth on the right. Default **Lorentzian** and optional **Gaussian** fits fit all peaks simultaneously across the full measured spectrum with a shared linear baseline. Plot zoom does not limit the fitted energy range.

For adjustments, use **Detect peak guesses**, **Add / Remove**, or **Spectrum clicks → Add peak on click**. With **Move selected peak**, click or drag a guess marker to place it; dragging stays between neighboring peaks. Center/FWHM fields are initial guesses, not measured results. These changes form a draft and do not repeatedly fit the sweep. Confirm the markers and click **Fit all powers** once; export stays unavailable until the applied analysis is ready. Peak 1 is the lowest-energy peak, with stable trend labels. **Advanced fit and display settings** contains the model, intensity metric and independent Y-axis scales. Integrated area uses the fitted component over the measured range (`a.u. eV`); **Peak height** uses amplitude.

Use **Issues only** to filter the results table, then **Next issue** to inspect each affected measured spectrum. **Manual refit selected power** edits guesses, center bounds and model, previews, then applies only to that channel/power. Other fits remain. Applied fits, manual corrections, power-law intervals and analysis display choices save automatically in local application storage. After reopening the same data, **Analyze peaks** restores the saved results without refitting. Data/background changes invalidate the saved results; unapplied drafts are not saved. Keep exporting the Power package for portable numerical results and provenance.

Noisy, unresolved or failed fits are flagged and omitted from trends. Click a trend point to inspect its spectrum. **Power Axis: Linear / Log** changes heatmap and both trend X axes without refitting; nonpositive powers disappear only from log views and remain exported. **Color Log** is separate. Intensity/linewidth have independent Y Log/Linear controls, default Log, including exported plots. Nonpositive Y is hidden with a note. Sparse powers retain normal cell boundaries without artificial white bands. Marker shapes identify source sweeps; colors identify peaks.

Fits run in the background. Once ready, the Power package includes `Power_peak_analysis` CSV, trend PNG and JSON with status, FWHM in meV, covariance uncertainty, centers, intensity and provenance. KK/KKp fit independently; VP is not peak-fitted.

### Fit a power law

**Fit power law…** in peak analysis or above the intensity plot fits integrated peak area to `I = A P^alpha` without recalculating spectra. Peak 1 starts selected; any peak combination can be fitted separately. Enter/drag limits, shared by default or independent per peak. Fit intervals are independent of plot axis limits.

The dialog shows exponent, regression standard error, counts, source-boundary warnings and an optional seven-point local-slope plot. **Suggest ranges near alpha = 1** searches for broad stable intervals with at least six points and half a decade, reporting when none is reliable. Suggestions remain exploratory and editable. Apply adds integrated-intensity overlays. Exports retain limits/results in analysis JSON, `_power_law.csv` and trend PNG. Regression errors exclude gain-calibration uncertainty. Use the Power workflow's export action to save its analysis package.

## SHG Processing

Load a wide sweep table with `measured position`, motion/acquisition status and numeric wavelength headers. SHG subtracts a local background at each angle and integrates around configurable center wavelength ± half-range; symmetric background sidebands use configurable gap/width. Optional conservative cosmic-ray removal replaces narrow positive spikes before fitting. Raw/cleaned spectra remain inspectable; export records removal counts and settings.

The two persistent tabs, **Single file** and **Compare / Twist**, fit `I(theta) = I0 + A cos^2[2(theta-xc)]`. Comparison independently processes both sources, overlays the angular curves and reports `twist = (2/3) delta_xc`, with explicit 90-degree wrapping/branch selection. Tabs above the controls support left/right keyboard navigation. Background, exclusions, peak integration, angular fits, residuals and result tables stay in the workflow. Wide layouts show both spectra; compact layouts offer **Raw / Corrected** while retaining angular result/residual.

**Save XLSX + settings** writes one Origin-friendly workbook and settings JSON. All measured-angle spectra export as wavelength rows/angle columns; choose angles after importing into Origin. Optional CSV copies and full-band cosmic-cleaned spectra start off and do not trigger reprocessing. After export, hover over the compact relative results path for the full path, or use **Open folder / Copy path**. Receipts list the folder once plus filenames. Workbook sheets/units and Origin steps are in [Data and exports](data-and-exports.md).

## MCD and MCD Peak Shift

MCD loads a B-sweep CSV with numeric `B_T`, angle and wavelength columns. It pairs opposite-angle frames, subtracts optional angle-specific dark/offset spectra and normalizes each angle to its own near-zero-field reference. Inspect raw/corrected paired spectra and MCD linecuts at a chosen actual B field. Source, dark-reference and output-folder dialogs have separate roles; changing output folder preserves source selection.

Choose **Data → Select file…** in MCD or MCD Peak Shift to open the wide measurement list. Long names wrap in full; search by filename or folder and filter processing status. Review or copy the selected full path, then choose **Open selected**. **Browse files…** stages an external CSV in the same selector; cancelling preserves the current measurement.

Drift options include global gain, per-pair scale/offset and robust linear/quadratic spectral-background correction constrained to non-resonant intervals. New sessions default to global gain plus quadratic per-pair spectral baseline. **Select protected regions** opens a reflection plot: drag across each resonance to protect it. Only the active MCD(B) window and enabled protected regions are excluded; sufficiently wide unprotected intervals become blue background bands. Adding/editing/enabling/removing regions recalculates immediately before comparing linear/quadratic models on held-out points. Manual regions survive recalculation and reopening.

The full B–energy map shares the linecuts' energy window. Processing edits recalculate after a short debounce; **Recalculate now** retries. MCD(B) separates increasing/decreasing branches and supports signed mean, field-signed absolute mean and signed integral. Initially only signed mean is shown; it is the primary PNG/settings metric. Magnitude/integral remain optional diagnostics, and CSV retains all metric columns. Compact export includes map PNG/DAT, MCD(B) PNG/CSV, pair diagnostics and reproducible JSON settings.

**Peak center (eV)** above the ordinary MCD plot shows up to six recommended integration centers. Click an energy to update the MCD(B) window. These recommendations remain visible when the Feature catalog is filtered to Spectrum or History. If no reliable center is found, enter **Center energy (eV)** manually.

### Saved MCD Extract / Compare

**Open processed MCD Extract / Compare** catalogs saved MCD(B) without raw reprocessing. Filter by doping/E-field tolerances, integration-energy ranges, integration widths and nearby-energy groups, with individual inclusion/exclusion. Choose automatic/explicit ordering, ascending/descending direction and palette.

Increasing traces use solid lines/filled markers; decreasing traces use dashed lines/open markers. Preview and PNG place branches on separate axes with shared scaling and an external legend. Extraction creates a descriptive Origin-ready XLSX with **Increasing, Decreasing, Slope Summary**, a PNG and settings JSON. Branch sheets contain X/Y pairs, fitted curves, slopes/intercepts, counts and R². Optional branch CSV starts off. Missing temperature resolves from measurement JSON or original CSV before filename fallback, with provenance retained.

### MCD Peak Shift and Organizer

MCD Peak Shift reuses accepted processed MCD products without reloading. It supports raw/derivative feature tracking, actual-field selection, field fits, manual feature merging and CSV/metadata export. MCD Organizer opens separately for saved-series comparison, energy grouping, slopes, Curie-Weiss fitting, field/temperature range scans, batch theta analysis and scientific export. Tools exposes the independent Organizer entry.

Native Peak Shift saves under `Processed Data/MCD Peak Shift/<source>_PeakShift/`. Repeating the same peak-location/method/result identity updates its CSV and metadata; other analyses remain alongside it. Creation times stay in metadata, and existing timestamp-named exports remain available.

## Slides

Open **Slides**, choose the existing `.pptx` to edit, then build the source/queue/preview workspace. **Insert live** changes that exact deck while open in desktop PowerPoint without saving; **Insert and save** changes the same file whether open or closed. A separate copy is optional.

Offline file generation uses the bundled `python-pptx` library and does not require PowerPoint. Live insertion requires Windows and desktop PowerPoint; the Windows release includes the Python integration bridge.

1. Search PNGs recursively under the experiment's `Processed Data`; sort by modified time or name. Full filenames wrap.
2. Filter MCD Combo maps and MCD(B) separately, or add the newest matching pair from the same subfolder. Folder badges/queue warnings expose mixed MCD folders.
3. Select in order, drag the queue to refine it and preview planned slides.
4. Choose 1–12 images per slide, including 2×4, 3×3 and 3×4 layouts for 8/9/12.
5. Add optional short captions as editable PowerPoint text. A/B/C panel labels start off. PNG sources are never altered or cropped.
6. Insert. A recovery backup and sidecar manifest protect the deck and prevent duplicate insertion of the same saved plot.

Launcher/PowerPoint setup and platform requirements are in [Development](development.md) and [Troubleshooting](troubleshooting.md).

## Tools, logs and saving

**Tools** contains the run log and independent DRR Analysis/MCD Organizer entries. The older standalone DRR workspace remains at **Tools → Data Organization → Open DRR Analysis** for candidate inspection, history/comparison and peak-to-peak analysis; use the plot toolbar for the current map's Peak Analysis.

Successful saves report completion in the status bar and paths in **Log**, preserving bottom-panel visibility and scientific text. **Analysis results** remains for peak finding/fitting results. Log retains the latest 300 plain-text messages, including multiline entries. Scroll to the bottom to follow new messages; reading older retained text keeps position. Clear removes displayed and queued messages. Load/export timing details are documented in [Development](development.md).

**Data Maintenance** moves exported source CSVs only against a verified successful export. Optional cleanup removes verified temporary copies while keeping canonical originals; source/output changes invalidate its receipt. Source catalog refresh/rebuild and recent-folder controls remain available. See [Data and exports](data-and-exports.md) before maintaining experiment files.

For PNG/DAT widths, paired DRR products, collision handling, full numerical grids and metadata portability, follow [Data and exports](data-and-exports.md). **Help → Check for Updates...** checks for stable updates; installation remains under your control. [Home](../../README.md) links to Windows releases.
