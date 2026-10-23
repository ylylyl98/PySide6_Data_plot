# PySide6 Data Plot App

Desktop data plotting app built with PySide6. It loads measurement CSV files, renders PL/DRR/Compare views with Matplotlib, and exports CSV/PNG outputs.

## Download DPTK for Windows

**[Download the latest Windows release](https://github.com/ylylyl98/PySide6_Data_plot/releases/latest)**

The easiest way to get DPTK is the installer:

1. Download `DPTK-Setup-vX.Y.Z-Windows-x64.exe`.
2. Run the installer.
3. Launch DPTK from the Start Menu.

The installer is the recommended option for normal users. Python and other
development tools are not required.

Download DPTK from GitHub **Releases**, not from GitHub Actions artifacts.
Actions artifacts are build outputs for development and testing, not user
downloads.

Unsigned installers may initially show a Microsoft Defender SmartScreen warning.
DPTK does not bypass or disable Windows security checks.

### Updating

Use **Help → Check for Updates...** at any time. DPTK may notify you when a
newer stable version is available, but DPTK never installs updates
automatically. Downloading and installing an update always stays under your
control.

### Portable version

If you prefer not to install, download `DPTK-vX.Y.Z-Windows-x64.zip` from the
latest release:

1. Download the ZIP.
2. Extract it.
3. Run `DPTK.exe`.

The portable ZIP remains available for restricted systems and users who prefer
no installation.

## Installation
```bash
git clone https://github.com/ylylyl98/PySide6_Data_plot.git
cd PySide6_Data_plot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python run_qt.py
```

## Editing plot controls

Numeric plot and processing controls commit typed values when you press
**Enter** or move focus to another control. Label and unit edits also wait
until editing finishes. Arrow buttons still adjust numbers; repeated adjustments
to display parameters are combined into one refresh after you stop. A pending
plot refresh or MCD recalculation waits while the next number is being typed.
Gate and cursor controls retain their existing interactive updates. Ordinary
DRR range changes reuse the current heatmap instead of rebuilding its axes,
including two- and three-region linear maps whose split boundaries stay valid.
Unchanged colorbars are retained during these updates.

### DRR sidebar

The sidebar keeps a standalone **Data** section above the **Display** and
**Processing** pages. Measurement selection and background settings remain
available while switching or scrolling either parameter page. The Data section
shows the current background summary and one **Background settings…** entry.
Choose BN, P/N, self reference or fitted background inside that window;
P/N explicitly offers P only, N only and P+N composite. **Display** contains shared axis ranges and
colors. **Processing** contains derivative/SG settings and single-spectrum fits;
Y-window controls appear only for the mixed derivative.

In Display → Colors, the **ΔR/R** and **Derivative** tabs select which color
controls to edit without changing the plotted view. Each product retains its own color map and limits;
region count and X boundaries are shared. Auto changes the selected product's
unfixed limits. Minimum and maximum editors are side by side, with their Fix
controls below them. **More display settings** contains the right Y axis,
scale/clip options and display-memory preferences. Hidden controls retain their
values for plotting and paired exports.

DRR range editors show up to four significant digits when idle, using scientific
notation for small values. Focusing an editor reveals its full configured
precision. Formatting does not change the stored limits, calculations or exports.
Color-limit arrows use approximately 0.5% of their own min/max span, rounded to
a 1/2/5 step (for example, −100 to 100 steps by 1; −0.01 to 0.01 by 0.0001).
Each product and each color region adapts independently, including after Auto
or restoring a saved view. Steps never fall below the input's numeric resolution;
equal bounds use their magnitude, or 0.01 when both are zero. Typing a number
still commits that exact value at the editor's configured precision.

Use the plot toolbar for Gate and previous/next steps. The **Analysis…** menu
offers both adding the current DRR data and opening the analysis workspace.

The DRR file chooser keeps measurement groups in a flat list. Each group uses
a compact summary of sample/position, acquisition conditions, file count and
processing status; conditions that differ between groups of the same sample
and position appear first. Summaries wrap to the pane width and rows grow to
show all conditions. The middle pane wraps full filenames, and tooltips
and Copy filename retain the original paths. Search accepts space-separated
keywords in any order (for example, `YZ365 760`), including older history and
filenames hidden from the group summary. Filtering and browsing groups retain
the chosen files in the right pane.

### Remembering DRR display settings

Successful DRR image saves remember the displayed X/Y and color ranges,
one/two/three-region layout, split boundaries, raw and second-derivative color
limits, colormaps, and Fix choices. Reloading the same file or selected group
restores its latest saved display and zoomed viewport, including after restarting
the app. Both **Save both: PNG + DAT** and the plot toolbar's PNG save remember
settings; toolbar saves update the experiment's display index.

A new file with the same filename wavelength center (for example, `720nmc`)
reuses the saved X range and boundaries; Y and color limits remain automatic
unless you have manually fixed them. On a file change, X geometry follows this
priority: explicit **Lock X range across files**, the target file's own snapshot,
compatible manual geometry from the same wavelength center, the target center's
template, then Auto. Manual X limits and split boundaries expire when the center
changes; files without a known center keep manual geometry only on a reload of
that same file. Reloading the current file preserves its unsaved manual edits.
X limits, region count, and fixed boundaries travel together. Manual Y/color
locks keep their existing priority. Fixing one axis keeps the saved zoom on the
other axis. Templates with
incompatible spectral coverage fall back to Auto. Open **Display → More display settings**
in the DRR sidebar to disable either behavior, optionally reuse color
limits, or share center templates across experiment folders. Color reuse and
sharing are off by default. **Lock X range across files** is also off by default;
enabling it locks the current X range and active boundaries, including values
restored from a preset. Its range and region layout survive app restarts. The
status line and Fix tooltips explain the active scope.
**Restore Auto for this load** releases remembered ranges and manual fixes and
turns off the cross-file X lock; **Clear this center template** keeps individual
file snapshots.

The DRR sidebar identifies each checked Fix by its source: **Fix M** means a
manual lock (X geometry normally follows the same wavelength), **Fix λ** means a wavelength template,
and **Fix F** means the current file's saved display. Hover over a Fix for details
and the wavelength center. Editing a locked value, or unchecking and checking
its Fix again, turns an inherited lock into a manual lock. File and wavelength
locks expire outside their scope and are not stored as app-wide manual defaults
when closing the app. Valid fixed boundaries stay in place. A boundary outside
a new X range is recentered while keeping its Fix source; three-region boundaries
are placed inside the range in order, respecting valid fixed boundaries.
**Auto X/Y/V** and regional **Auto** update unfixed
limits; uncheck Fix or use **Restore Auto for this load** to release locks.

Snapshots are stored in `Processed Data/DRR/.display_settings.json` and the
export metadata sidecars, so they travel with the experiment folder. Existing
DRR export metadata can also supply saved ranges. Settings update only after a
successful save and use the view captured when saving starts.

DRR loading reuses parsed display history, including missing-file/center lookups,
and reads only changed sidecars when new exports appear. External edits, source
path changes, and deleted records invalidate the cache. Disabling both memory
options skips history queries. Loading also reuses unchanged source-file hashes;
source cleanup always checks fresh hashes.

In raw view, hidden second-derivative computation and regional Auto colors are
deferred until switching to second/both, pressing a second-product Auto button,
or saving both products. Paired export computes deferred colors on its worker
and remembers the final values while retaining manual and restored limits.
Raw-only toolbar PNG saves retain pending Auto status for hidden derivative
colors so their initial placeholders are not restored as saved limits.

Split heatmaps draw nonoverlapping column grids using the original cell edges.
Two/three-region plots retain every measured point, independent color scales,
cursor readings and export resolution without duplicating the full matrix in
each mesh. CSV parsing uses change-sensitive file fingerprints, including
Windows ChangeTime, so same-size rewrites with restored modification times
invalidate cached data. Files without a reliable fingerprint are parsed afresh.

For DRR maps with at least 250,000 cells, ROI and regional color percentiles
run on a background worker. Repeated data/ROI/log/zero/region requests reuse a
bounded cache, and rapid changes retain the newest queued request per region.
An initial plot waits for its automatic colors while the interface stays usable.
Results respect edits, Fix choices and restored settings made during calculation;
loading another source or changing SG/ROI settings discards obsolete results.
Paired saves freeze pending requests together with their numerical product and
complete those colors on the export worker using the saved ROI. Smaller maps
keep their immediate automatic color updates.

CSV DRR loads collect consistent measured right-axis coordinates during the
numerical load, avoiding a second spectrum read solely for the right axis.
Multi-file averages accumulate per-pixel sums and non-NaN counts instead of
retaining and stacking every raw/corrected matrix. The common baseline path
keeps its existing align-before-correct order; heterogeneous assignments keep
native-grid correction before alignment. Raw averages, missing coverage,
infinities, derivative processing and right-axis consistency checks are preserved.
The bounded raw CSV parse cache remains separate from these working buffers.

DRR source discovery streams CSV rows and retains only gate metadata while
checking every spectral value for completeness. It avoids retaining the full
text and parsed spectral table solely for grouping. Empty, incomplete and
nonfinite channels retain the same compatibility rules. Unchanged sources still
use the existing inspection cache.
Large uncached DRR inventories (at least 32 CSV files and 32 MiB) use at most
two background processes in the app, with four files in flight. Small refreshes
stay in the existing worker. Cancellation joins the child workers; unavailable
processes fall back to serial inspection. File signatures are rechecked before
using a prefetched profile.
On a cold catalog, file identities appear before spectrum inspection finishes;
selection confirmation and loading wait for grid/background validation. Refreshing
an already cataloged selection reuses that catalog. An external selection inspects
only the requested file, without rescanning all of its sibling measurements.

DRR gate changes retain the heatmap image and redraw the complete linecut axes
and cursors over it. Spectrum titles, automatic Y ticks and fit overlays update
without redrawing the heatmaps or colorbars. Source/layout changes, zoom,
resize and exports invalidate the retained background. PNG/PDF/SVG saves still
include every spectrum at full resolution. SG changes with no peak overlays no
longer repaint the old product before the new derivative is ready.
Moving a split within the same two/three-region layout replaces only the region
meshes and retains the axes, colorbars and their formatting. Interactive linear
views skip drawing cells outside the current viewport, with guard cells along
each edge. All model arrays and exported data keep their original resolution;
no visible cells are downsampled. Non-affine axes retain the normal draw path.
Switching between one, two and three regions retains the map axes, measured
right axes and linecut artists, rebuilding only the region meshes and colorbars.
Paired views adjust the existing layout margins for the larger three-region
legends. Theme updates style current and future ticks without requesting tick
labels and gridlines repeatedly before drawing.
Color-only edits repaint the affected map and colorbar while retaining the other
product and linecuts. Updated colorbar labels are included in the cleared area;
overlapping panels/annotations fall back to a complete redraw. Combined color and
gate edits still refresh both linecuts. Confirming a numeric edit with Enter or
leaving the field flushes its pending redraw; held spin arrows remain debounced.

Nonuniform energy derivatives batch complete rows on their shared grid for
cubic interpolation and Savitzky–Golay filtering. Each batch is limited to 32
rows and a 65,536-cell target; gappy rows keep the original per-row calculation.
The measured grid, filter settings, output resolution and DAT precision are
preserved. Export history validation reuses directory enumeration attributes
while still checking for changed, added and deleted metadata on every request.

DRR derivative history uses a 128 MiB buffer budget and at most 12 entries,
evicting older parameter results first. Current full-resolution dE/second
products stay available together. If those current products alone exceed the
budget, history is discarded and the current products are retained to avoid
repeated calculation. This budget covers cached derivative buffers, not total
application memory.

A newer Load selection replaces the queued request and cooperatively cancels
the obsolete one. DRR loads check between files and numerical/history stages;
the current file read or NumPy operation completes before cancellation takes
effect. Clear and changing experiment folders also invalidate active loads.
Cancelled loads do not restore old plots or display a loading-failed dialog.
If the newest selection fails validation, an older queued selection is dropped
rather than loaded afterwards. Export workers keep their captured data and
are independent of Load cancellation.

Display derivative jobs also cancel obsolete source/SG/direction/window
requests between batches, gappy rows, polynomial windows and mixed-derivative passes. Parameter
edits invalidate the old job even when the second product is hidden or new
settings fail validation. Useful dE/second jobs with the same source and SG
settings still complete together. A single uniform-grid SG call completes
before checking cancellation. Completed derivatives adopt the filter's own
independent output buffer, avoiding another full-matrix copy.

Logs append plain text in short batches and retain the latest 300 messages,
including multiline entries. Reading older surviving text keeps that text in
place as earlier entries expire; scrolling to the bottom follows new messages.
Clear removes both displayed and queued log messages. Successful Load workers
log inclusive stage times for data loading, canonical CSV/cache reads, source
verification, display-memory lookup and coordinate metadata, plus worker total.
CSV/cache and verification times are included in the data-load total; these
times exclude UI preparation, derivative display jobs and plotting.

DRR PNG exports fit titles with whole-word measurements and a bounded,
per-font cache, retaining character wrapping for long identifiers and complete
math tokens. Export meshes retain original cell boundaries and full resolution
inside the X/Y view, with adjacent guard cells for clipping. Offscreen cells are
omitted from the PNG mesh; DAT files and interactive heatmaps retain all data.
Singular/nearly singular ranges and invalid log bounds retain the full extent
so Matplotlib's automatic range handling stays consistent. PNG dimensions,
color normalization, measured right axes and saved-image reuse are preserved.

The export status follows PNG build, title fitting, draw and encoding/writing.
Each newly rendered Raw/second product logs `build`, `title`, `draw`,
`encode_write` and `total` elapsed times. Title time is included in build;
these worker measurements exclude queue delay and GUI publication. Agg still
draws each PNG once. Timing collection is thread-scoped and reset on failures.

## Peak Analysis

Plot a PL, DRR, Compare intensity, or Power Dependent intensity map, then select
**Peak Analysis...** at the right of the plot toolbar. The English analysis
window runs in a separate process with its own data snapshot. PL supports peaks
and optional multi-peak Lorentzian/Gaussian fits; DRR supports peaks and dips for
each displayed product. Compare/Power intensity channels use the PL adapter.
VP is excluded. Repeated scan coordinates currently need to be separated or
aggregated before tracking.

The older DRR workspace remains available under **Tools → Data Organization →
Open DRR Analysis**. Its shortcut has been removed from the source bar beside
**Open File**; use the plot toolbar for analysis of the currently plotted data.

Snapshots inherit the main map's colormap, color limits, log/zero-centered
normalization, and two- or three-region color scales. DRR files have visible
**ΔR/R / 2nd derivative** tabs above the plot. Files use a separate tab strip;
with one file, only its name is shown (no file dropdown).
Tabs retain separate branches/results and share the selected scan coordinate.
Display and send a missing product from the main app to enable its tab. Opening
the same data again refreshes its display settings while preserving analysis;
older saved snapshots need this handoff to receive the new display metadata.

1. In **Find & filter**, choose **Local extrema** (SciPy `find_peaks`, no smoothing)
   or **SG + extrema** (Savitzky–Golay order-2 smoothing followed by the same
   detector). DRR dips use the inverted supplied signal. SG does not differentiate
   an already processed product. It regrids uneven energy coordinates per finite
   segment; windows shorten to fit, and segments below five samples stay unsmoothed.
2. **Find all peaks** scans every spectrum and immediately overlays candidates on
   the heatmap. **Candidate floor** is a fraction of each full spectrum's range;
   new PL and raw ΔR/R datasets default to **SG + extrema**, with an **11-sample
   window**. DRR derivative products default to **Local extrema**, avoiding an
   additional smoothing pass. The method description explains the recommendation;
   its optional smoothing tab is explicitly labeled **Extra SG**. Recorded upstream
   SG window/order are shown for derivative snapshots; older snapshots report that
   these settings are unavailable instead of assuming unsmoothed input. Provenance
   comes from the displayed transform cache, never pending main-window controls.
   Adjust derivative generation in the main DRR tab and import the updated product
   to compare window sizes. Either method can still be selected manually. Both defaults use a **0.01
   candidate floor** and the **Balanced** quality preset below. Set the floor to `0` only
   when intentionally retaining all local extrema; there is no peak-count cap.
   Existing saved parameters and per-product manual choices are preserved.
   New handoff defaults are applied by the analysis process, so this recommendation
   update requires only **More → Restart Analysis**, even if the main app is already
   running. Previously saved datasets retain their chosen method. All work runs
   in the analysis process's background worker with progress and cancellation.
3. Select **Balanced**, **Sensitive**, or **Strict** to filter the cache without
   repeating detection. The filter heading identifies the current product profile.
   New datasets start with product-specific **Balanced** filters:

   | Product | Minimum SNR | Prominence fraction | Support out of 5 |
   | --- | ---: | ---: | ---: |
   | PL intensity (including Compare/Power intensity) | 5 | 0.05 | 3 |
   | DRR ΔR/R | 5 | 0.08 | 3 |
   | DRR 2nd derivative | 7 | 0.10 | 4 |

   Other explicitly labeled DRR derivatives use the derivative profile too.
   **Sensitive** lowers SNR by 2, halves prominence, and lowers support by 1;
   **Strict** raises SNR by 2, multiplies prominence by 1.5, and raises support by 1.
   Width thresholds adapt to the median energy sample interval: **Sensitive** uses
   2 intervals, **Balanced** 3, and **Strict** 4. Same-polarity spacing starts at
   2 intervals for all presets. Both are converted to meV, rounded to the controls'
   4 decimal places, and bounded to their supported range (0.0001–1000 meV).
   For example, at 0.2 meV/sample, Balanced uses a 0.6 meV width and 0.4 meV spacing.
   Check narrow/nearby features and adjust these values rather than treating the
   sampling-based threshold as an instrument-resolution measurement.
   Button tooltips list all preset thresholds. Width/spacing tips explain their
   sampling basis; the SG window tooltip shows its approximate full-grid span in meV.
   These are starting points, not assurances that every kept point is physical.
   Stronger support can shorten a branch near where it appears or disappears;
   lower support to inspect these endpoints instead of increasing smoothing.
   Presets clear maximum-width/count caps; range, polarity and detector settings
   remain as selected. Manual adjustments display **Custom filters**. Existing
   workspaces retain their settings and cached results; choose a preset explicitly
   to apply the new profile. New app handoffs get current defaults in the child
   process even while the main app is running; reimporting a known dataset keeps
   its existing analysis choices.
   **X/Y bounds**, **Both/Peaks/Dips**, **Min SNR**, **Min prominence**, **Min width**,
   and **Support (of 5)** are directly visible. **Full / View / Custom** selects the
   filter range; editing a bound selects Custom and reuses the cache. View follows
   the heatmap limits, with one refilter after a pan/zoom gesture. Custom bounds
   are preserved when switching ranges, including existing saved custom sessions.
   **More filters** holds spacing, neighbor drift, maximum width
   and per-row count; `0` means no cap/off as indicated. **Reset filters** restores all cached
   candidates; **Show rejected candidates** draws excluded points as gray crosses.
   The page reports kept/total counts and the actual cached method. Rebuild only
   when changing method, SG window or detection floor; lowering a filter below the
   cached floor cannot recover candidates omitted during detection. Width is full
   width at half prominence, not fitted FWHM. The original candidate pool is saved.
   SNR is prominence divided by robust noise estimated on the **cached detection
   signal**, separately per finite energy segment. Noise is MAD / 0.67448975 of
   the residual after removing an order-2 SG baseline of up to 51 samples; this
   baseline is used only for noise estimation, never to move or refit extrema.
   Segments shorter than 7 samples have no noise estimate and are rejected when
   the SNR filter is enabled. This heuristic ratio is not a confidence probability.
   Same-polarity support is counted after shape/SNR filtering, before range cropping
   and branch association, in sorted scan coordinates. The default matching
   tolerance is `1.5 meV` per neighboring row, doubled for the second neighbor.
   Required support scales with available rows at map edges; a scan gap above
   1.5 times the median step breaks the neighborhood. Single spectra are retained
   without requiring unavailable neighbors. Counts do not recursively promote points.
   Saved analyses keep their prior settings/results. Choosing a preset on an old
   cache computes the missing noise metrics once in the worker; subsequent filters
   reuse them. The SG window is directly visible when smoothing is selected;
   **Detection options** exposes the less frequently adjusted acquisition floor.
   Plot-toolbar **Size** controls heatmap marker area (default `9 pt²`, previously
   `25 pt²`). It updates existing artists without detection, filtering or rebuilding
   the heatmap, and is retained in workspace saves, exports and overlay packets.
4. In **Branches**, double-click to rename, check to include, or **Exclude branch**.
   Names/IDs/colors survive reversible filter splits. **Connect selected branches**
   conservatively connects cached positions in adjacent scan rows; ambiguous
   matches remain isolated crosses and missing rows break connections. **Pick seed**
   enables the separate manual tracking workflow; gap/noise settings apply there.
   Click the heatmap or move **Spectrum** to inspect a measured spectrum.
   The cursor readout above **Spectrum** shows mouse X/Y coordinates and axis
   labels. On a heatmap it also shows **Z (nearest)** from the input grid, before
   color clipping/normalization or extra detection smoothing. Hover over the
   readout for the signal label and sampled cell coordinates, marked as the last
   reading after leaving the plot and cleared when the data/view changes. Log scan axes use
   their displayed cell boundaries while reporting physical scan values.
   Spectrum and residual plots show the mouse Y coordinate without snapping to
   a curve. Missing/out-of-grid Z values and inactive coordinates show `—`.
   Moving the mouse only updates this text (at most about 30 times per second);
   it does not change the selected spectrum, run analysis, or redraw the heatmap.
5. For PL, optionally use **Fit selected branches** and **Show residual**.
   Choose detected positions or valid fitted centers for the overlay.
6. Use **Apply overlay** to send positions to matching main-app heatmaps. Changed
   numeric data invalidates the overlay; zoom and color limits do not rerun analysis.
7. **Results** opens a table on demand, including SNR, supporting/available rows,
   prominence and measured width. CSV exports also include noise and support metrics.
   **Export...** creates a new folder with
   CSV, JSON and PNG files. Progress and cancellation stay in the status bar;
   this workflow does not automatically open the app's bottom log/results docks.

The workspace is saved on close under `%LOCALAPPDATA%/DPTK/Peak Analysis` on
Windows. **More → Restart Analysis** saves it and starts a fresh analysis process.
The initial toolbar installation requires one main-app restart. During source
development, edits to the analysis window, controls, and algorithms take effect
after restarting only that window. Changes to the main-app bridge or shared
main-window overlay rendering still require
an app restart. Packaged EXE changes require a rebuild; restarting the page does
not replace bundled Python code.

PL-specific detection/fitting lives in `core/pl_peak_analysis.py`; DRR-specific
behavior lives in `core/drr_peak_adapter.py`. They share tracking, plotting,
persistence and `ui_qt/peak_analysis_window.py`, rather than maintaining duplicate
pages for each tab. `ui_qt/peak_analysis_bridge.py` owns the toolbar integration
and `run_peak_analysis.py` starts the independent process.

## Building PowerPoint slides

Open the **Slides** workflow to assemble processed PNG plots into a PowerPoint
presentation. Choose the existing `.pptx` you want to edit. **Insert live**
updates that exact deck while it is open in desktop PowerPoint without saving;
**Insert and save** updates the same file whether it is open or closed. A
separate copy remains available as an optional action.

- Search PNGs recursively below the experiment's `Processed Data` folder;
  filenames wrap in full and can be sorted by modified time or name.
- Filter MCD Combo maps and MCD(B) traces separately, or add the newest matching
  same-subfolder pair in one click. Folder badges and a queue warning make
  mixed MCD folders visible before insertion.
- Select plots in the desired order, drag the queue to refine it, and preview
  each planned slide.
- Choose any layout from 1 through 12 images per slide, including 2×4, 3×3,
  and 3×4 layouts for 8, 9, and 12 images.
- Add optional short captions as editable PowerPoint text. A/B/C panel labels
  are available but off by default. Source PNG files are never changed or cropped.
- Insert again safely: a recovery backup and sidecar manifest protect the deck
  and prevent the same saved plot from being appended twice.

On Windows, run `Data_Plot_App.bat` once. It creates `DPTK Desktop.lnk` beside
the launcher with the application icon; use that shortcut for normal launches
and taskbar pinning. To create a Desktop shortcut instead, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_windows_shortcut.ps1 -Desktop
```

On macOS, install Python 3 and double-click `Data_Plot_App.command`. The
launcher uses a project-local `.venv`, installs the declared requirements when
needed, and starts the same `run_qt.py` entry point. Finder may require
right-clicking the file and choosing **Open** the first time; the file must be
executable in a checkout (`chmod +x Data_Plot_App.command`).

To build an unsigned macOS application on a Mac, install the build requirements
and run:

```bash
python -m pip install -r requirements.txt -r requirements-build.txt
python -m PyInstaller packaging/PySide6_Data_Plot_macos.spec
```

This produces `dist/DPTK Desktop.app`. It must be built on macOS for the target
CPU architecture. Unsigned or unnotarized builds may trigger Gatekeeper; use
right-click **Open** for a local build. Developer ID signing and notarization
are future distribution work. The existing Windows BAT launcher and Windows
PyInstaller spec remain unchanged.

## Build and Release

For a local Windows build, install the build dependency and run:

```powershell
python -m pip install -r requirements.txt -r requirements-build.txt
.\build_windows.bat
```

The portable application is created at
`dist\PySide6_Data_Plot\PySide6_Data_Plot.exe`. PyInstaller uses an `onedir`
layout, so distribute the entire `PySide6_Data_Plot` directory rather than the
EXE alone. The build stops automatically if the Windows PowerPoint bridge was
not included; installed users do not need to install Python packages.

To build the installer locally, install NSIS and run:

```powershell
.\scripts\build_installer.ps1 -Version 1.0.0 -OutputFile DPTK-Setup-Windows-x64.exe
```

To test packaging on GitHub without publishing a release, open **Actions**,
select **Windows Build**, choose **Run workflow**, and download the
`DPTK-Windows-x64` artifact after the run completes. It contains the portable
ZIP, installer, and `SHA256SUMS.txt`.

To publish a version, push a tag:

```powershell
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions first verifies that the tag matches `app_version.py`, then runs
the tests, builds the Windows application and installer, and attaches
`DPTK-v1.0.0-Windows-x64.zip`, `DPTK-Setup-v1.0.0-Windows-x64.exe`, and
`SHA256SUMS.txt` to the corresponding GitHub Release.
Generated `build/` and `dist/` content stays local and is not committed.

## Architecture
- `core/`: UI-agnostic data loading, processing, plotting primitives, and export.
- `ui_qt/`: PySide6 desktop UI (`QMainWindow`, controls, Matplotlib embed, log/progress).
- `run_qt.py`: PySide6 entrypoint.

## Expected Folder Structure
Choose the experiment folder, such as `YZ327`, as the app's data folder. Canonical
acquisition files belong in `Initial Data`. The app writes new processed results
under workflow-specific directories without moving historical results.

```text
YZ327/
├── Initial Data/                         # canonical acquisition/raw files
│   ├── raw_001.csv
│   └── raw_002.csv
├── Processed Data/                       # generated results
│   ├── PL/
│   ├── DRR/
│   ├── Compare/
│   ├── MCD/<analysis package>/
│   ├── MCD Extracts/                       # Origin XLSX, comparison PNG, settings JSON
│   ├── SHG/<analysis package>/
│   └── Power Dependence/<group or comparison package>/
├── temporary working CSV copies/          # normally directly under YZ327
└── Initial data after processing/         # manual archive only
```

For normal CSV processing, selected working files remain directly under the
experiment folder because the file browser is more convenient for selecting
many raw files. Existing historical files directly under `Processed Data/`
remain valid; the application does not migrate or reorganize them.

### Raw-data lifecycle and cleanup

For PL, DRR, and Compare:

1. The acquisition application writes canonical files into `Initial Data/`.
2. Use Windows Explorer or macOS Finder to copy only the files needed for the
   analysis into the experiment folder.
3. Load and process those root-level working copies in the app.
4. Results are written to `Processed Data/<workflow>/`.
5. If **Clean verified source copies after successful export** is enabled, the
   app may remove a root-level duplicate only after its matching
   `Initial Data/<filename>` file exists and the SHA-256 hashes match.

Cleanup never deletes canonical files. A missing canonical file, invalid path
relationship, hash mismatch, failed export, disabled option, or unverified/manual
source leaves the working file in place. A file selected directly from
`Initial Data/` is never treated as a disposable temporary copy.

The **Move Exported Sources** button is separate and manual. It moves explicitly
selected source files to the legacy `Initial data after processing/` archive.
That folder is not canonical raw storage, automatic cleanup storage, or normal
processed-result storage.

### Export locations and contents

| Workflow | New export location | Typical contents |
|---|---|---|
| PL | `Processed Data/PL/` | DAT, metadata sidecar, linear/log PNG figures |
| DRR | `Processed Data/DRR/` | averaged or derivative DAT, metadata, PNG |
| Compare | `Processed Data/Compare/` | channel DAT/PNG/metadata and VP results |
| MCD | `Processed Data/MCD/<analysis package>/` | map DAT/PNG/metadata, MCD(B) CSV/PNG, diagnostics, settings |
| SHG | `Processed Data/SHG/<analysis package>/` | area-vs-angle CSV, settings, fit/twist summaries |
| Power Dependence | `Processed Data/Power Dependence/<package>/` | KK/KKp/VP DAT, PNG, metadata |

MCD package names use a readable source, energy, and window identity such as
`<source>_MCD_E1.650000eV_W5meV`. SHG single analyses use
`<source>_SHG_<center>nm`; twist analyses use
`<reference>_vs_<sample>_SHG_twist`. Power packages use one of:
`<group>_PowerDep`, `<KK-group>_vs_<KKp-group>_IntensityCompare`, or
`<KK-group>_vs_<KKp-group>_VP`. Package names are sanitized for Windows and
macOS. Related numerical files, metadata, figures, and diagnostics stay
together.

### Metadata and collision handling

Metadata sidecars use the DAT file's own directory and are optional for reading
the numerical data. Where emitted, the shared metadata core records schema and
app versions, workflow, dataset type, creation time, sources/provenance,
processing, plot settings, outputs, and an output manifest. Workflow-specific
fields record details such as DRR backgrounds, MCD windows, SHG fits, or Power
pairing and alignment.

Repeated exports do not silently overwrite earlier results. Simple result stems
use collision-safe suffixes such as `_01`; complex MCD, SHG, and Power analyses
use collision-safe package directories. Associated DAT, PNG, and JSON files
retain the same logical stem or package.

### DAT re-import and Origin use

Exported DAT files remain tab-delimited, Origin-friendly numerical matrices and
can be opened without JSON. The app can load DAT files from old root-level
`Processed Data`, the new workflow folders, or other supported user-selected
locations. Sidecar metadata is searched beside the DAT file, so moving a DAT
does not require the experiment root to be known. When present, metadata can
restore labels, units, plot context, and processing information.

For imported DAT colormaps, the PL Y-axis selector supports `Y`, `Doping`,
`Electric field`, `Gate voltage`, and `Custom` labels, with an optional unit.
Without metadata, neutral generic labels are used.

The same layout works with Windows Explorer and macOS Finder. Documentation
uses portable folder names and the application uses cross-platform path
handling.

Compatibility note: the active PySide GUI routes new exports to the workflow
locations above. Some lower-level legacy Python helper functions retain their
old configurable/default `Processed Data` destination for existing scripts;
calling those helpers directly is separate from the normal GUI workflow.

## Usage Notes
- Mouse-wheel scrolling moves panels and lists without changing dropdown
  selections, numeric values, sliders/dials, or active tabs. Click or use the
  keyboard to edit controls; this applies to dialogs as well as the main window.
- `PL`: one-file plotting workflow for heatmap + spectrum.
- `Power Dependent`: select a single header-table CSV containing `Power_uW`, optional
  `stage_pos`, and numeric wavelength/energy column names. Each row is one power
  point. Numeric wavelength headers (for example `752.5829`) are converted to
  photon energy and the rows are sorted by power. Legacy multi-file series with a
  power token such as `37.96uW` in each filename remain supported.
  **Choose Power Group…** shows recent measurements, newest modification first.
  The first 20 groups are shown initially; **Show older measurements** opens the
  rest. Each row shows its modification date, measurement identity, available
  channels and power range. Modification time controls ordering, never pairing.
  Whole-sweep KK/KKp filenames share measurement settings; polarization `deg`
  fields identify the channels and do not split the group.
  Bare legacy tokens such as `deg24` and `deg69` are treated as output/analyzer
  angles by both Compare and Power. A unique legacy Power pair can infer its
  two output references using Compare's group-local inference; the reference
  nearest the configured output K angle remains K. Run suffixes and other
  measurement settings remain part of the group identity.
  Different power grids inside the two CSVs are supported. A unique pair uses **Open comparison**;
  a single sweep uses **Open sweep**. Ambiguous matches use **Resolve pairing**.
  Full filenames and manual assignments live under **Details / Adjust pairing**.
  Confirmed manual assignments are remembered in `.power-selection.json` in the
  experiment folder and remain portable when that folder is moved.
  **Swap KK ↔ KKp** exchanges the assignments without opening Details and shows
  the assigned angles. **Open comparison** saves that choice for this measurement
  group; Cancel leaves the saved choice unchanged. Other groups retain their assignments.
  Single-sweep plotting and export need one sweep and no KK/KKp assignment.
  Intensity comparison needs two distinct assigned channels. VP additionally
  requires compatible spectral overlap and valid stage or power pairing.
  Switch between Intensity and VP on the Power page after opening a pair.
  A compatible stage/power pairing is suggested automatically, with an override
  available in the detailed selection controls.
  Grouping prefers matching `*.experiment.metadata.json` sidecars with
  `files[].path`, `session_id` and `measurement_id`; optional sample, position,
  temperature and gate fields further identify the measurement. A per-file
  `channel`, `compare_channel` or `polarization` value can identify KK/KKp.
  Without an explicit measurement identity, filename context remains part of
  the group. Angle-based suggestions use the references configured in Compare.
  Search and status filters retain the current draft and use the cached catalog;
  **Refresh** rescans files. Cancel leaves the active selection unchanged.
  **Combine sweeps…** is a separate operation for segments of the same channel;
  selecting KK and KKp together does not combine their intensities.
  Power datasets are discovered automatically beneath the selected main folder,
  including measurement subfolders such as `power_sweep`. `Initial Data`, old/archive folders
  and analysis CSVs without spectral columns are excluded. Nested saved combinations
  remain available under Processed. Refresh discovers new acquisitions while
  preserving the current view. Filename-based legacy grouping is off by default,
  so ordinary PL filenames containing power values do not clutter the picker.
  Enable **Include filename-based series** to access these groups; they need at
  least two distinct powers and are grouped separately by containing folder.
  Review combines selected segments of the same channel and saves a new
  CSV under Processed. Check acquisition compatibility in the preview;
  reference selection and overlap-based scaling are automatic by default. The
  reference favors valid direct calibrations to the other sweeps, with power
  coverage breaking ties. Correction subtracts separate backgrounds and scales
  additional sweeps to the reference. Unsupported or rejected estimates keep the
  preview visible, identify the affected sweep, and block saving.
  The default review shows sweeps, multipliers and previews. **Advanced
  settings** allows disabling automatic setup to choose a reference, no scaling
  or known factors, and reveals background bands, exclusions, duplicate policy and metadata;
  rejected corrections and flagged measurements remain visible in the main view.
  Enabling scaling removes a separate background from every input. Multi-sweep
  duplicate averaging uses all contributions at once, without pairwise averaging bias.
  A known factor can be entered, or estimated from overlap without extrapolation.
  Calibration interpolation only bridges neighboring powers within 25%; spectra
  in the overlap preview must be within 2% in measured power. Calibration reports
  pair count, factor scatter, and spectral mismatch using the applied common factor.
  Backgrounds default to separate medians in an editable shared energy band
  (initially the highest 20% of the shared spectrum). Choose a signal-free band;
  manual backgrounds are also available. Estimated corrections with >10% factor
  scatter or spectral mismatch are rejected, left unapplied, and cannot be saved.
  Limited power coverage remains explicitly provisional. Check saturation and exclude
  questionable spectra using their displayed 1-based power-order numbers.
  **Fit power law…** in peak analysis (also available above the intensity plot)
  fits integrated peak areas to `I = A P^alpha` without recalculating spectra.
  Peak 1 is selected initially; any combination of peaks can be fitted separately.
  Enter or drag power limits, shared by default or independent per peak. These
  intervals are independent of plot axis limits. The dialog shows exponent and
  regression standard error, point counts, source-boundary warnings, and an
  optional seven-point local-slope plot. **Suggest ranges near alpha = 1** searches
  the full sweep for broad stable intervals (at least six points and half a
  decade). Suggestions are exploratory and can be adjusted; no reliable interval
  produces an explicit message. Applying displays the integrated-intensity plot
  with fit overlays. Export saves the selected limits and results in the analysis
  JSON, a companion `_power_law.csv`, and overlays on the integrated trend PNG.
  Regression errors do not include gain-calibration uncertainty.
  Exact matching powers can keep the first sweep,
  keep the second, or average; nearby powers remain separate. Spectra use the
  first sweep's energy grid within the shared energy range, with linear energy
  interpolation and no extrapolation. Source rows and the merge policy remain in
  the CSV and DAT/JSON exports. Combine KK segments and KKp segments separately,
  then assign the saved sweeps for comparison. Repeated stage positions disable
  stage pairing; use power interpolation in that case.
  The group picker separates channel readiness from processing status. Combined sweeps are
  processed results, discovered in `Processed Data/Power Dependence/Combined Sweeps`.
  Saved combined files appear under Processed and can be selected directly.
  Reusing the same original rows is blocked, including renamed inputs identified
  from provenance.
  Descriptive filenames remain editable; source/settings fingerprints recognize
  existing combinations and offer Open existing or Create another.
  **Back to selection** preserves checked sweeps and review settings, including
  when another sweep is added. **Save and plot** saves and opens the result;
  **Plot Setup** then
  controls the loaded result. Existing CSVs are preserved; saving automatically
  uses a numbered filename when the chosen name is already taken.
  Enable **Peak Analysis → Intensity + linewidth vs
  power** for a heatmap, selected spectrum with fit and residual, integrated PL
  intensity versus power, and FWHM versus power. The default fit is **Lorentzian**;
  the heatmap and spectrum/residuals occupy the left column, with intensity above
  linewidth in the right column.
  **Gaussian** is also available. All peaks are fitted simultaneously across the
  full measured spectrum with a shared linear baseline. Detect, add, or remove
  peaks and edit their center and width guesses in the Peak Analysis table.
  Peak 1 is the lowest energy peak; separate intensity and linewidth curves
  retain these labels across power. Integrated area uses each fitted component
  over the measured energy range (a.u. eV); **Peak height** uses its amplitude.
  **Manual refit selected power** lets you edit guesses, center bounds, and model,
  preview the fit, and Apply it to just that channel and power. Other fits are
  preserved. Fits are cached in memory during the session; saved analysis files
  include manual refit settings but are not automatically restored on restart.
  Unresolved, noisy, or
  failed fits are flagged and omitted from trend curves. Click a trend point to
  inspect its spectrum. **Power Axis: Linear / Log** changes the heatmap's power
  axis and both trend X axes together without refitting; nonpositive powers are
  omitted in log views but retained in data exports. **Color Log** is independent.
  Intensity and linewidth each have an independent Y-axis Log/Linear control,
  defaulting to Log, also respected in exported plots. Nonpositive Y values are
  hidden with a note. Colormaps use normal cell boundaries between measured
  powers; sparse sampling does not insert artificial white bands or alter data.
  Source marker shapes identify contributing sweeps; colors identify peaks.
  Fits run in the background. Once the fit status is ready, Save includes a
  `Power_peak_analysis` CSV, trend PNG, and JSON settings in the power package,
  including failed-fit status, FWHM in meV, covariance uncertainty, peak center,
  intensity, and source provenance. KK and KKp are fitted independently in
  intensity comparison; VP itself is not peak-fitted.
- `SHG Processing`: load a wide sweep table containing `measured position`, motion
  and acquisition status columns, and numeric wavelength headers. The tab removes
  a per-angle local background, integrates the background-subtracted spectrum over
  a configurable center wavelength and ± half-range, and automatically positions
  symmetric background sidebands using a configurable gap and width. It plots area
  versus the measured angle and can conservatively replace narrow positive cosmic-ray
  spikes before background fitting. Raw and cleaned spectra remain available for
  inspection, and exports include removal counts plus JSON settings. Nested `Single
  File` and `Compare / Twist Angle` workflows fit
  `I(theta) = I0 + A cos^2[2(theta-xc)]`; comparison mode overlays two independently
  processed curves and reports `twist = (2/3) delta_xc` with explicit 90-degree phase
  wrapping and branch selection.
- `DRR`:
  - `Self (last/first frame)` baseline modes.
  - `External` baseline mode from selected baseline files.
    **Background settings…** is the shared entry point. Select the source first:
    BN; P/N with **P only**, **N only** or **P+N composite**; self reference; or
    fitted background. P-only and N-only sources are used directly. Their inline
    file lists show the corresponding P or N references and prioritize acquisition
    conditions matching the current measurement group, including rotation;
    equally matching references retain newest-file-first order. Existing
    files and saved versions are selected within their source type, rather than
    a separate competing "existing background" category. Custom imports,
    interval editing and BN calibration are under **Advanced**.

    BN and direct P/N file choices appear in one searchable list inside settings,
    without the measurement browser's group/files/chosen panes or Add/OK steps.
    Common BN uses single selection; P-only and N-only use checkboxes for repeated
    background files. Same-angle candidates without known condition conflicts
    are expanded first; other conditions and unknown angles remain available in
    labelled, expandable groups. Filenames wrap in full. Search and asynchronous
    catalog refresh preserve selected paths. **Browse…** imports an external
    path for validation on Apply; it does not change the main result immediately.

    Changes inside Background settings remain drafts until **Apply**. Nested editors return their
    selection to settings; generating/saving a version does not apply it.
    Cancel preserves the applied background and main view. Background-file
    frame averaging is separate from averaging repeated measurements.
    A saved P/N composite is displayed as **Saved composite · single-frame
    background**. Its P and N source frame choices are shown separately from
    the saved version's metadata; the internal output frame selector does not
    select frames from the original P/N measurements.

    When reopening a measurement group through automatic background restoration
    or the External source-type workflow, the newest saved DRR PNG/DAT export
    recipe for that exact selection restores its background files,
    frame choices and per-measurement assignments before applying the current
    source-type preference. **Fixed current background** keeps its explicit
    priority. Missing or incompatible saved backgrounds require review; they do
    not silently fall back to another file. A saved composite is reused as its
    frozen spectrum after integrity and native-grid checks, without regenerating
    it or requiring the original P/N inputs to remain unchanged. Groups without
    an exact export retain the existing source-type matching/review behavior.

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

    After applying an automatic BN mapping, **Group BN** in the main sidebar
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
    For example, nine measurements corrected by one BN and six by another
    contribute with weights 9/15 and 6/15, not equal weights for the two batches.
    Within a batch using one BN, averaging the raw measurements before division
    gives the same result as averaging their individually corrected DR/R values.

    **Background settings → Advanced → BN calibration** compares the selected measured BN reference
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

    **Background settings → P/N → P+N composite** opens the P/N workspace
    for generation and saved versions. Combination and X-range editing are
    available separately under **Background settings → Advanced → Edit background**.
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
    The current real-data checks of the two general draft generators did not
    meet the joint full-range ΔR/R and d²E criteria. They remain separate from
    the raw reconstruction backend described below.

    **Raw V66 reconstruction backend:**
    `core.drr_pn_reconstruction.generate_from_files(folder, p_file, n_file,
    training_files)` recomputes the adopted source-resonance fits, energy-dependent
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

    Both original measured groups have been reconstructed from their raw files,
    with stage comparisons and complete-range ΔR/R and SG 11/21-point checks.
    Numerical reproduction of V66 establishes this migration's fidelity, not
    scientific acceptance of every new P/N pair.

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
    Current real-data review preserves the original V66 outputs exactly and
    reduces the long-tail baseline change in one repeat acquisition. A residual
    fixed band and the other repeat's dense fringe mismatch remain, so the joint
    full-range dR/R and d2E acceptance has not passed. Inspection of raw common
    P/N texture places substantial repeat mismatch before bridge construction;
    changing the joins alone does not resolve that mismatch.

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
    The frozen real-data probe rejected calibration for both original
    acquisitions, retaining their backgrounds exactly. Both repeat acquisitions
    improved eligible-block predictions, but residual fixed bands and dense
    full-map derivative fringes still prevent joint quality acceptance. These
    proposals have not replaced the saved V66 or entered the default generator.

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
    The frozen follow-up applies a nonzero correction only to the 49-degree
    repeat acquisition (763 of 1340 points). Its previous added endpoint
    derivative excursion is removed in the complete-product check, while the
    original fixed band remains. The 94-degree repeat has no continuous support
    in one required test block and therefore retains its parent exactly; both
    original acquisitions also retain their parents. These safe fallbacks are
    not improvements, and complete-background scientific acceptance is still
    not established.

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

    **Background settings → Advanced → Edit background** creates a reusable background from the clean
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
  - `Fitted (sample)`: select original CSV measurements, choose Fitted in Background
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
    - **Correct shared fine texture (learned outside peaks)** estimates a
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
    - **Predict fringes across protected gaps (validated only)** is an optional
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
    - **Fringe reference** optionally uses an explicitly selected raw CSV from
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
- `MCD`: load one B-sweep CSV with numeric `B_T`, angle, and wavelength columns.
  The tab pairs opposite-angle frames, subtracts optional angle-specific dark/offset
  spectra, normalizes each angle with its own near-zero-field reference, and shows
  raw paired spectra, corrected paired spectra, and raw/corrected MCD linecuts at
  a chosen B field. Drift modes include global gain, per-pair scale/offset, and an
  optional robust linear or quadratic spectral-background correction constrained
  to selected non-resonant energy intervals. **Select protected regions** opens
  a reflection plot where you directly drag across each resonance that must be
  protected. Only the active MCD(B) window and the regions you select are excluded;
  every sufficiently wide unprotected interval becomes a blue background-fit
  band and recalculates immediately as you add, edit,
  enable, or remove a region before the tool compares linear/quadratic models on
  held-out candidate points. The full B-versus-energy MCD map
  shares the selected energy window with those linecuts. MCD(B) supports signed
  mean, field-signed absolute mean, and signed integral traces for separate
  increasing/decreasing sweep branches. The compact export contains the displayed
  map PNG/DAT, MCD(B) PNG/CSV, pair diagnostics, and reproducible settings JSON.
  Manually drawn protection regions survive recalculation and are restored when
  the selector is reopened. Processing-setting changes are applied automatically
  after a short debounce; **Recalculate now** remains available as a retry. Spin
  boxes ignore ordinary mouse-wheel scrolling and require Ctrl+wheel when focused.
  New MCD sessions default to global gain plus a quadratic per-pair spectral
  baseline. MCD(B) displays only signed mean initially, and signed mean is also
  the primary PNG/settings export metric; magnitude and integral traces remain
  available as optional diagnostics, while the CSV retains all metric columns.
  **Open processed MCD Extract / Compare** catalogs saved MCD(B) results without
  reprocessing raw files. It supports tolerance-based doping and E-field filters,
  integration-energy ranges, integration-width filters, nearby-energy grouping,
  and individual include/exclude choices. Increasing-field traces remain solid
  with filled markers; decreasing-field traces remain dashed with open markers.
  Preview and PNG export place the two branches on separate axes with shared
  scaling, while the result/color legend stays outside the data panels.
  Choose automatic or explicit ordering, ascending/descending direction, and a
  color palette. Extraction writes one descriptively named Origin-ready XLSX
  with separate Increasing, Decreasing, and Slope Summary sheets; a matching PNG;
  and a compact settings JSON. The branch sheets contain X/Y trace pairs, fitted
  curves, slopes, intercepts, point counts, and R². Optional branch CSV copies are
  off by default. Missing temperature is resolved from measurement JSON or the
  original CSV before filename fallback, and its provenance is recorded.
- `Compare`: 2-4 selected files rendered in a compare grid.
- Heatmap PNG exports default to `8.0 x 6.2 in @ 150 DPI` (`1200 × 930 px`), independent of window size.
  The main plot toolbar starts at **PNG 1.0×**, preserving the original export.
  Select **Wide PNG** next to Save for an optional wider export; its multiplier
  defaults to **2.00×** and can be adjusted from **1.00×** to **3.00×**.
  **1.50×** gives `1800 × 930 px`, and **2.00×** gives
  `2400 × 930 px`. Height, font sizes, data ranges and DAT values stay unchanged.
  Widened heatmaps fit the top spacing to the title and colorbars, and wide PNGs
  reduce side margins to the space needed by labels, ticks and colorbars.
  The canvas size stays unchanged. The wide multiplier is remembered across
  restarts, while the layout starts at **PNG 1.0×**. In DRR, selecting **PNG 1.0×**
  saves only the original width. Selecting **Wide PNG** also keeps the 1.0×
  image: at **2.00×**, **Save both · PNG + DAT** creates four PNGs (two widths
  for each product) and two DAT files; at 1.0× it creates two PNGs and two DATs.
  The original-width PNG keeps its usual filename; new wide PNGs use the selected
  multiplier as a suffix, such as `_2.0x.png`. Repeat saves reuse matching images
  and repair missing selected widths without duplicating DAT files. The DRR
  toolbar PNG save follows the same rule using one frozen figure. Other wider
  multipliers, such as 1.50×, also retain the 1.0× image. The width selection
  applies to PL (including both linear/log panels), DRR (both products), Compare/VP, Power Dependence and MCD
  heatmaps, plus unified MCD Save Results. The toolbar's PNG save widens its
  current figure by the same factor; the on-screen plot keeps its current size.
  Select **PNG 1.0×** at any time for the original export layout.
- Exported `.dat` files remain tab-delimited numeric matrices for Origin. The
  app can re-open them from the PL tab and optionally restores labels and plot
  metadata from the adjacent `.metadata.json` sidecar (or
  `<file>.plotmeta.json`). Without a sidecar, safe generic labels are used.

## Smoke Check (No Test Framework Required)
Run:
```bash
python scripts/smoke_check.py
```

This performs lightweight import/path checks for the Qt app and core modules.

Use the project virtual environment for checks and launches:
```bash
.venv\Scripts\python.exe scripts\smoke_check.py
.venv\Scripts\python.exe run_qt.py
```

`Data_Plot_App.bat` launches without pulling from git by default. Set `DPTK_AUTO_UPDATE=1` before running it if you want the launcher to fetch and fast-forward pull first.

## Troubleshooting
- `No CSV files found`: confirm files are directly in the selected folder root and have `.csv` extension.
- `Folder does not exist` or `Selected path is not a folder`: re-pick the folder in the folder panel.
- `Compute failed` in DRR: verify baseline mode and selected baseline files match selected run files.
- Blank/invalid plots: confirm CSV columns are numeric and contain finite energy/gate/intensity values.
