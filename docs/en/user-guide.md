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

Plot PL, DRR, Compare intensity or Power intensity, then use **Peak Analysis...** at the right of the plot toolbar or the Workflow menu. The English analysis window runs in a separate process using a numeric snapshot. PL supports detection and optional multi-peak Lorentzian/Gaussian fitting. DRR supports peaks/dips for each supplied product; Compare/Power intensity use the PL adapter. VP is excluded. Repeated scan coordinates require explicit separation or aggregation; measured power values are not modified to hide repeats. Power's own peak workflow remains available for those datasets.

Snapshots inherit colormaps, limits, log/zero-centered normalization and regional scales. DRR has **ΔR/R / 2nd derivative** tabs above the plot; files have a separate tab strip (a single file shows its name). Products retain independent branches/results and share the selected scan coordinate. To enable a missing product, display and send it from the main app. Reopening the same numeric data refreshes display settings while preserving analysis; older saved snapshots need a new handoff for updated display metadata.

### Detect once, filter the cache

1. In **Find & filter**, choose **Local extrema** (SciPy `find_peaks`, no smoothing) or **SG + extrema** (order-2 SG smoothing followed by that detector). Dips invert the supplied signal. SG does not differentiate an already processed product. Uneven energy coordinates are regridded per finite segment; windows shorten to fit and segments under five samples remain unsmoothed.
2. **Find all peaks** scans spectra and overlays candidates. **Candidate floor** is a fraction of each full spectrum's range. PL/raw ΔR/R default to **SG + extrema**, window **11**; derivative products default to **Local extrema**. Derivative smoothing is labeled **Extra SG**, and recorded upstream window/order are shown when available. Provenance comes from the displayed product, not pending main-window settings. Change derivative generation in DRR, then send the updated product to compare SG windows.
3. Both defaults use floor **0.01** and **Balanced** filters. Floor `0` deliberately retains all local extrema; there is no peak-count cap. Detection runs with progress/cancellation in the analysis worker. Changing detector, SG window or floor rebuilds the candidate pool; filters below the cached floor cannot recover omitted candidates.

New datasets use these product-specific Balanced thresholds:

| Product | Minimum SNR | Prominence fraction | Support out of 5 |
| --- | ---: | ---: | ---: |
| PL, Compare/Power intensity | 5 | 0.05 | 3 |
| DRR ΔR/R | 5 | 0.08 | 3 |
| DRR 2nd derivative / other labeled derivatives | 7 | 0.10 | 4 |

**Sensitive** lowers SNR by 2, halves prominence and lowers support by 1. **Strict** raises SNR by 2, multiplies prominence by 1.5 and raises support by 1. Width thresholds use 2/3/4 median energy-sample intervals for Sensitive/Balanced/Strict; same-polarity spacing starts at 2 intervals. They convert to meV, round to 4 decimal places and stay in 0.0001–1000 meV. At 0.2 meV/sample, Balanced width is 0.6 meV and spacing 0.4 meV. These sampling-based values are starting points, not instrument resolution. Tooltips list thresholds and the SG window's approximate span.

Presets clear maximum width/count caps and retain range, polarity and detector choices. Manual edits show **Custom filters**. Existing workspaces keep settings/caches; choose a preset explicitly to update them. New handoffs apply current defaults in the child process, even if the main app is running; known datasets retain their choices. **More → Restart Analysis** refreshes the analysis process.

**X/Y bounds**, **Both/Peaks/Dips**, **Min SNR**, **Min prominence**, **Min width** and **Support (of 5)** are directly visible. **Full / View / Custom** selects filter range; editing bounds selects Custom and reuses the cache. View follows heatmap pan/zoom with one refilter after the gesture; saved custom bounds survive range switching. **More filters** contains spacing, neighbor drift, maximum width and per-row count (`0` means off/no cap as labeled). **Reset filters** restores cached candidates; **Show rejected candidates** draws gray crosses. **Detection options** exposes the acquisition floor. The page reports kept/total counts and the actual cached method.

### Interpreting detection metrics

Candidate width is full width at half prominence, not fitted FWHM. SNR is prominence divided by robust noise on the cached detection signal, per finite segment. Noise is MAD / 0.67448975 of residuals after an order-2 SG baseline of up to 51 samples; this baseline only estimates noise and never shifts/refits extrema. Segments under seven samples lack a noise estimate and are rejected when SNR filtering is enabled. SNR is a heuristic ratio, not confidence probability.

Same-polarity support is counted after shape/SNR filtering and before range cropping/branch association in sorted scan coordinates. Default tolerance is `1.5 meV` per neighboring row, doubled for the second neighbor. Required support scales at map edges; scan gaps above 1.5 median steps break neighborhoods. Single spectra do not require absent neighbors; counts do not recursively promote points. Strong support can shorten branch endpoints: lower support to inspect them instead of increasing smoothing. Applying a preset to an old cache computes missing noise metrics once, then reuses them. Exports/workspaces retain the original candidate pool and metrics.

### Branches, spectra, fits and overlays

In **Branches**, double-click a name to rename, check inclusion or use **Exclude branch**. Names/IDs/colors survive reversible filter splits. **Connect selected branches** conservatively connects cached positions in adjacent scan rows; ambiguous matches stay isolated crosses and missing rows break connections. **Pick seed** opens separate manual tracking with its gap/noise settings.

Click the map or move **Spectrum** to inspect a measured spectrum. The mouse readout shows X/Y and labels; heatmaps also show **Z (nearest)** from the input grid before color clipping/normalization or extra smoothing. Hover for signal label/cell coordinates. Leaving the axes marks the last reading; data/view changes clear it. Log axes report physical scan values using displayed cell boundaries. Spectrum/residual Y is the actual mouse coordinate, not curve-snapped; missing coordinates/Z show `—`. Mouse motion only changes text and does not select a spectrum or run analysis.

For PL, **Fit selected branches** optionally fits peaks; **Show residual** exposes residuals. Choose detected positions or valid fitted centers for the overlay. **Apply overlay** sends positions to matching main-app heatmaps. Numeric source changes invalidate overlays; zoom/color changes do not rerun analysis. Toolbar **Size** sets marker area (default `9 pt²`) without detection/filtering or heatmap rebuilding, and is retained in workspace saves, exports and overlay packets.

**Results** opens the table on demand with SNR, supporting/available rows, prominence and measured width. **Export...** creates a new folder containing CSV/JSON/PNG; CSV also includes noise/support metrics. Progress/cancellation remains in the status bar without automatically opening the main app's bottom docks. On Windows the workspace saves on close under `%LOCALAPPDATA%/DPTK/Peak Analysis`. **More → Restart Analysis** saves before restarting. Source-code/bundled-code restart requirements are in [Development](development.md).

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

Enable **Peak Analysis → Intensity + linewidth vs power** for heatmap, selected-spectrum fit/residual, integrated intensity and FWHM trends. Heatmap/spectrum occupy the left column; intensity sits above linewidth on the right. Default **Lorentzian** and optional **Gaussian** fits fit all peaks simultaneously across the full measured spectrum with a shared linear baseline.

Detect/add/remove peaks and edit center/width guesses in the table. Peak 1 is the lowest-energy peak, with stable trend labels. Integrated area uses the fitted component over the measured range (`a.u. eV`); **Peak height** uses amplitude. **Manual refit selected power** edits guesses, center bounds and model, previews, then applies only to that channel/power. Other fits remain. Fits are cached during the session; analysis files record manual settings but do not automatically restore fits after restart.

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
