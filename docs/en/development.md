# Development, diagnostics, and release

[Project home](../../README.md) · [简体中文](../zh-CN/development.md)

For everyday operation, use the [user guide](user-guide.md). Output formats and source-file safety are in [data and exports](data-and-exports.md); reference selection and fitting are in the [background reference](background-reference.md).

## Contents

- [Source setup and launch](#source-setup-and-launch)
- [Source shortcuts and macOS recipes](#source-shortcuts-and-macos-recipes)
- [Legacy Widgets compatibility](#legacy-widgets-compatibility)
- [Native application architecture and settings](#native-application-architecture-and-settings)
- [Rendered checks and performance probes](#rendered-checks-and-performance-probes)
- [Loading, caches, and scientific workers](#loading-caches-and-scientific-workers)
- [Rendering and export instrumentation](#rendering-and-export-instrumentation)
- [Smoke checks and CI scope](#smoke-checks-and-ci-scope)
- [Windows build and release](#windows-build-and-release)
- [Multi-computer Git synchronization](#multi-computer-git-synchronization)

## Source setup and launch

Use Python 3.13, matching the Windows launcher and CI. The source workflow needs the Python dependencies; installed release users do not need Python.

```powershell
git clone https://github.com/ylylyl98/PySide6_Data_plot.git
cd PySide6_Data_plot
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_qt.py
```

Use `.venv\Scripts\python.exe` explicitly when another Python may be on PATH. `Data_Plot_App.bat` launches without pulling Git updates by default. Set `DPTK_AUTO_UPDATE=1` before launching to request its fetch/fast-forward update path. It skips updating if changes, untracked files, missing upstream or fetch errors prevent it, and continues launching. This launcher behavior does not replace the repository synchronization rules below.

## Source shortcuts and macOS recipes

On Windows, running `Data_Plot_App.bat` creates an icon-aware `DPTK Desktop.lnk`
beside the launcher if it is missing. Use it for normal source launches and
taskbar pinning. To create the shortcut on the Desktop instead:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_windows_shortcut.ps1 -Desktop
```

Windows is the official binary and CI target. The repository also includes
`Data_Plot_App.command` and `packaging/PySide6_Data_Plot_macos.spec` as local
macOS source/build recipes; they were not executed or validated on macOS for this
documentation update. On a Mac with a suitable Python 3 installed, the launcher
uses `python3` by default (overridable with `PYTHON_BIN`), creates a project-local
`.venv`, installs `requirements.txt` if importing PySide6 fails, and runs
`run_qt.py`. It does not check every dependency or refresh already-installed
versions. Install the declared requirements explicitly when updating the source
environment. Make the launcher executable and then open it from Finder; the
first local launch may require right-click **Open**:

```bash
chmod +x Data_Plot_App.command
```

To attempt a local unsigned application build on macOS, use that Mac's project
environment and the macOS-specific spec:

```bash
.venv/bin/python -m pip install -r requirements.txt -r requirements-build.txt
.venv/bin/python -m PyInstaller packaging/PySide6_Data_Plot_macos.spec
```

The spec requires macOS and names the bundle `dist/DPTK Desktop.app`. Build on
macOS for the intended CPU architecture. Unsigned or unnotarized local builds
may trigger Gatekeeper; right-click **Open** is the documented local-build
route. Developer ID signing and notarization remain future distribution work.
The Windows BAT launcher and Windows PyInstaller spec are separate paths.

## Legacy Widgets compatibility

Normal startup now opens the native Qt Quick application described below.
The previous Widgets shell remains available for compatibility checks. Its DRR
Display page can use either the embedded QML controls or Widgets controls.

To use the Widgets display page, run:

```powershell
.venv\Scripts\python.exe run_qt.py --legacy-widgets --drr-display widgets
```

Within `--legacy-widgets`, `DPTK_DRR_DISPLAY=widgets` selects the same display fallback.
A QML loading/rendering error also returns this embedded page to standard controls
with a visible status message. Both views use the existing settings namespace,
display-history format, numeric precision, Auto/Fix behavior, and export path.

The legacy embedded QML view receives plain snapshots and commands through
`ui_qt/quick/drr_display_bridge.py`. The legacy controls are a temporary adapter
to the existing controllers; they remain the single compatibility state owner.
Direct `MainWindow()` construction defaults to Widgets for existing embedded
callers; use `MainWindow(drr_display_backend="quick")` to exercise the new view.

The source and frozen application support an isolated page smoke check:

```powershell
.venv\Scripts\python.exe run_qt.py --check-quick-display
```

This check creates the real page, verifies a rendered frame, and tears it down
using temporary settings and an empty data folder. The Quick bridge, QML input,
fallback, and resource tests run in the separate `quick-display` CI job and are
also discovered by the complete release suite. `build_windows.bat` additionally
runs the isolated smoke check against the frozen EXE with software rendering.
Native Windows visual checks are needed in addition to offscreen checks for
fonts and graphics behavior.

To compare the two views with the same synthetic DRR data, run each backend in
a fresh process:

```powershell
.venv\Scripts\python.exe scripts/profile_quick_display.py --backend widgets --output build/quick-display/widgets.json
.venv\Scripts\python.exe scripts/profile_quick_display.py --backend quick --output build/quick-display/quick.json
```

This isolated benchmark reports page startup and already-loaded gate-to-plot
latency. It excludes file loading, analysis, and export; its measurements are
specific to the selected platform and machine.

## Native application architecture and settings

The default application uses `QGuiApplication`, QML controls, and an
Agg-to-Quick plot surface without constructing `MainWindow` or hidden Widgets.
The empty shell presents before importing Matplotlib or NumPy; the first plot
loads its Agg canvas lazily, preserving the same scientific plotting code.
`python run_qt.py` and the packaged executable use this frontend by default.
The independent analysis entry points and the existing `--peak-analysis`,
`--drr-analysis`, and `--mcd-organizer` flags also open native Quick windows.

```powershell
.venv\Scripts\python.exe run_qt.py --qt-quick --demo
.venv\Scripts\python.exe run_quick.py --workflow drr --open "measurement.dat"
.venv\Scripts\python.exe run_quick.py --workflow cmp --open "sample_PL_KK.csv" --open "sample_PL_KKp.csv"
.venv\Scripts\python.exe run_quick.py --workflow power --open "power-sweep.csv"
.venv\Scripts\python.exe run_quick.py --workflow mcd --open "magnetic-field-sweep.csv"
.venv\Scripts\python.exe run_quick.py --workflow shg --open "angular-sweep.csv"
.venv\Scripts\python.exe run_quick.py --workflow slides
.venv\Scripts\python.exe run_quick.py --workflow tools
```

Normal startup retains the existing `DPTK/PySide6_Data_Plot` preferences and recent
folder list. Native settings use an application-data directory for caches;
Windows registry keys are never interpreted as filesystem paths. Analysis
windows retain their existing workspace format and default workspace location.
Direct `run_quick.py`, `--demo`, and `--check` use temporary settings by default.
Pass `--settings-file <path-to-test.ini>` to persist an explicit test profile.
Opening real sources and explicitly exporting still reads/writes those selected
data folders. Smoke checks should use synthetic files in temporary folders.
Light, dark, system-following and high-contrast appearances are supported. A
saved appearance is retained unless `--theme` explicitly overrides it.
Application windows, controls and dialogs use Arial in every appearance mode.
Plots and exported images also use Arial for titles, labels, ticks, legends and
annotations, preserving their existing sizes and bold/italic styles. Mathematical
text uses Arial where available, with symbol fallback for glyphs the font lacks.
Dropdown options and selected lists use matching foreground/background colors
for mouse and keyboard selection. File/folder pickers apply a temporary selection
palette while open and restore the window palette when accepted or cancelled.
Independent analysis and background windows follow the same appearance.

The source layout is:

| Path | Role |
|---|---|
| `core/` | UI-independent loading, processing, plotting primitives and export |
| `ui_qt/quick/` | Native QML shell, bridges, controls, workflow adapters and Agg-to-Quick surface |
| `ui_qt/main_window.py` and legacy controls | Separate Widgets compatibility shell and Matplotlib embedding |
| `run_qt.py` | Production entry point and dispatch to native, analysis or legacy modes |
| `run_quick.py` | Native entry point with isolated development/check profiles by default |

PL uses its dedicated `ui_qt/quick/qml/PlDisplay.qml` page and
`ui_qt/quick/pl_display_state.py`; DRR uses `ui_qt/quick/qml/Display.qml` and
`ui_qt/quick/drr_display_state.py`. They share control components and interaction
conventions while retaining independent scientific/display state. Native analysis
entries and the `--peak-analysis`, `--drr-analysis`, and `--mcd-organizer` flags
open separate Quick workspaces. Peak Analysis runs in a separate native process;
accepted overlays are bound to the numeric source. DRR Analysis receives the
committed raw DRR source even if a derivative is displayed, avoiding
differentiation of an already differentiated input.

PL detection/fitting lives in `core/pl_peak_analysis.py`; DRR-specific behavior
lives in `core/drr_peak_adapter.py`. Tracking, plotting and persistence services
are shared. The current Quick Peak Analysis path uses
`ui_qt/quick/peak_source_adapters.py` for source capture,
`ui_qt/quick/peak_analysis_launcher.py` for process/IPC handoff,
`ui_qt/quick/peak_analysis_bridge.py` for native state, and
`run_peak_analysis_quick.py` as the process entry point. The older
`ui_qt/peak_analysis_window.py` and `ui_qt/peak_analysis_bridge.py` retain the
Widgets window/toolbar integration. `run_peak_analysis.py` defaults to Quick
when executed directly; its `--legacy-widgets` flag selects the compatibility
window. These legacy paths do not describe the default native frontend.

Opening empty DRR, MCD and MCD Peak Shift controls defers numerical/plot imports. MCD defaults and catalogs use lightweight modules, folder selection does not initialize the scientific stack, and measurement-processing dependencies load on the existing worker. Basic background settings defer preview canvases and fitting algorithms until advanced tools are used. Appearance changes avoid repeated unchanged palette assignments and no duplicate hidden parameter pane is constructed.

## Rendered checks and performance probes

```powershell
.venv\Scripts\python.exe run_quick.py --demo --check --report build/quick-application/smoke.json
.venv\Scripts\python.exe scripts/profile_quick_startup.py --samples 3 --platform native --output build/quick-application/startup.json
.venv\Scripts\python.exe scripts/profile_quick_application.py --policy incremental --samples 12 --platform native --output build/quick-application/incremental.json
.venv\Scripts\python.exe scripts/profile_quick_application.py --policy full --samples 12 --platform native --output build/quick-application/full.json
```

The startup profiler alternates deferred imports and an emulation of the former
eager-import boundary, using fresh temporary font caches for both. It measures
the shell and first synthetic PL plot separately; it is not a legacy-app or
whole-workflow benchmark. Reports include source hashes and every sample.

The application profiler measures real QML input to the corresponding presented
Quick frame, records GUI heartbeat gaps, and compares pixels to a fresh
scientific rebuild. Its full/incremental comparison uses the same PL data and
geometry; it is not a claim about loading, export, or whole-application speed.

For a broader interaction audit, run:

```powershell
.venv\Scripts\python.exe scripts/profile_quick_main_interactions.py --output build/interaction-audit/main.json
.venv\Scripts\python.exe scripts/profile_quick_analysis_interactions.py --output build/interaction-audit/analysis.json
```

These use temporary settings and synthetic data, separating GUI slot time,
event-loop heartbeat gaps, and total completion time. Run performance probes
serially, without simultaneous test suites. The main probe also exercises a
background export holding the rendering lock: interactive redraws defer until
that lock is free, while input and timers continue to run.
Pass `--reference-full-redraw` to the main probe to emulate the former full
rebuild, QuadMesh paint, and blocking render paths with identical data and
Arial font setup. This is a controlled plotting comparison on current source,
not a complete historical application benchmark.

Native large rectilinear heatmaps paint original cell colors through a screen
image path. Their complete measured arrays and cell boundaries remain intact;
analysis, cursor values, and saved figures use the original meshes. Log axes,
custom clipping/effects, and styled mesh edges keep the ordinary renderer.
Color-only edits retain existing PL/DRR/Compare/Power axes and navigation.
The main window's bottom status bar shows the mouse coordinates over the active
plot, using that axis's formatting for linear, log and polar views. Leaving the
axes, changing plots or clearing data removes the old reading. Hovering updates
only the coordinate text, without selecting a spectrum, logging or redrawing.

Peak Analysis reuses spectrum artists when changing rows and preserves the heatmap.
The following benchmark compares the previous spectrum-rebuild/blit path and the
retained-artist/blit path in the same process, with identical synthetic data and
viewport. `--fit` includes fitted components and residuals:

```powershell
.venv\Scripts\python.exe scripts/profile_quick_peak_analysis.py --fit --output build/quick-peak-profile.json
```

`build_windows.bat` also checks the actual packaged executable across all nine
workflow pages, synthetic PL/DRR/Compare inputs, and the three independent Quick
workspaces. The checks use temporary settings and data, require rendered frames,
and inspect diagnostic logs after process exit to catch shutdown errors. Run them
separately with:

```powershell
.venv\Scripts\python.exe scripts/check_frozen_quick.py dist/PySide6_Data_Plot/PySide6_Data_Plot.exe --report build/frozen-quick-report.json
.venv\Scripts\python.exe scripts/check_frozen_quick.py --source --report build/source-quick-report.json
```

These offscreen software-renderer checks do not exercise real instruments,
external Office applications, installer downloads, or native GPU rendering.

## Loading, caches, and scientific workers

These are implementation contracts for responsiveness and scientific-data fidelity. Thresholds and cache budgets below describe current policy, not promises of total memory use or runtime. See the user guide for display-memory precedence, Fix scopes, picker gestures and cancellation behavior.
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
Legacy Widgets raw-only toolbar PNG saves retain pending Auto status for hidden derivative
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

## Rendering and export instrumentation

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

DRR PNG title fitting, viewport meshes, product markers and saved-image reuse are described in [data and exports](data-and-exports.md#png-dimensions-and-paired-exports).

The export status follows PNG build, title fitting, draw and encoding/writing.
Each newly rendered Raw/second product logs `build`, `title`, `draw`,
`encode_write` and `total` elapsed times. Title time is included in build;
these worker measurements exclude queue delay and GUI publication. Agg still
draws each PNG once. Timing collection is thread-scoped and reset on failures.

## Smoke checks and CI scope

Pull requests to `main` run **Fast PR checks**, including native Qt Quick
application and display checks. The required **quick-checks** job has a
10-minute time limit and covers Arial typography, source catalogs, P/N assignment,
representative core behavior, and the ordinary measured-BN workflow:
candidate matching, batch assignments, manual selection,
group overrides, background settings, saved recipes, and application guards.
These checks do not run the expensive fitted-BN numerical qualification. The
same workflow is reused by **Windows Build** when manually building a branch
with `test_scope: quick`; its test-module list is maintained in one place.

When changing or using fitted-BN calibration, run **Actions → BN calibration
validation → Run workflow** on the relevant branch. Its eight numerical test
modules run in separate jobs, with up to four jobs in parallel. All existing
numerical assertions and feature-protection thresholds are retained. This
manual workflow is separate from the ordinary BN assignment PR gate; the
Windows release workflow runs the full test suite in twelve parallel shards for
tag refs and manual `test_scope: full` builds, alongside packaging. Publishing
still requires a tag push, a successful build, and successful verification of
every full-suite shard report.

The smoke check needs no test framework and performs lightweight import/path
checks for the Qt application and core modules. Use the project virtual
environment for checks and launches:

```powershell
.venv\Scripts\python.exe scripts\smoke_check.py
.venv\Scripts\python.exe run_qt.py
```

`Data_Plot_App.bat` launches without pulling from git by default. Set `DPTK_AUTO_UPDATE=1` before running it if you want the launcher to fetch and fast-forward pull first.

## Windows build and release

### Local build

For a local Windows build, install the build dependency and run:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-build.txt
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
.\build_windows.bat
```

The portable application is created at
`dist\PySide6_Data_Plot\PySide6_Data_Plot.exe`. PyInstaller uses an `onedir`
layout, so distribute the entire `PySide6_Data_Plot` directory rather than the
EXE alone. The build stops automatically if the Windows PowerPoint bridge was
not included; installed users do not need to install Python packages.

`build_windows.bat` invokes `python` from PATH. The command above prepends the
project environment for the current shell so packaging uses the installed
dependencies.

To build the installer locally, install NSIS and run:

```powershell
.\scripts\build_installer.ps1 -Version 1.0.0 -OutputFile DPTK-Setup-Windows-x64.exe
```

### GitHub build and release gates

To test packaging on GitHub without publishing a release, open **Actions**,
select **Windows Build**, and choose **Run workflow**. On a branch, the default
`test_scope: quick` runs the same **Fast PR checks** workflow alongside
packaging. Choose `full` to run the complete test suite in twelve parallel
shards. Any tag ref always uses `full`, even when `quick` is
selected; missing or unrecognized scope values also use `full`.

After the selected checks and build succeed, download the `DPTK-Windows-x64`
artifact. It contains the portable ZIP, installer, and `SHA256SUMS.txt`. Packaging
can finish and expose this artifact before testing finishes, so a downloadable
artifact alone does not mean the selected checks passed. A successful quick run
confirms the representative checks, not complete regression coverage. Manual
runs never create a GitHub Release.

To publish a version, push a tag. The following version is an example; the
actual tag must match `app_version.py`:

```powershell
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions first verifies that the tag matches `app_version.py`, then runs
the complete test suite in twelve parallel shards while building the Windows
application and installer. A tag must also point to a commit contained in
`main`. Only after the build and the combined test-report verification succeed
does the workflow attach
`DPTK-v1.0.0-Windows-x64.zip`, `DPTK-Setup-v1.0.0-Windows-x64.exe`, and
`SHA256SUMS.txt` to the corresponding GitHub Release.
Generated `build/` and `dist/` content stays local and is not committed.

Full builds request twelve test runners alongside one packaging runner. Other
workflows share the account's concurrency allowance, so queued test jobs can
extend the total run time. Each discovered test still runs exactly once and all
twelve shard reports must pass verification before publishing.

### Discovery, sharding, and timing weights

The release test plan is generated from actual `unittest` discovery, so newly
added tests are included automatically. Repeated discovery of the same test ID
through imported test classes is recorded in the plan and executed once;
inherited tests belonging to different classes retain their distinct IDs.
Unsupported custom discovery or parameterized test instances stop planning
instead of silently changing the test behavior. Modules run in separate
processes to isolate Qt object lifetimes and module state; the eight expensive
fitted-BN modules are split into individual tests. Shard balancing uses measured
Windows CI execution-unit durations from `scripts/release_test_timings.json`,
including process startup. Hints are keyed by module or full test ID and count
as measurements only when the test-module source and exact test-ID set still
match; source fingerprints normalize Git's LF/CRLF checkout differences.
Changed units use the larger of the normal fallback and their previous measured
duration as an explicitly labelled historical estimate, so editing a slow test
does not suddenly schedule it as a short task. New units use normal estimates.
The file
records the source run and commit; it is optional for local fixture repositories.
These weights affect placement only, not coverage, pass/fail criteria, or a
promised runtime. Execution still checks byte-exact test sources and complete
results. Existing skip and expected-failure rules are retained.

Keep test fixtures in helpers or mixins without `test_*` methods: inheriting an
existing `TestCase` also reruns its tests under the subclass. Related shell
checks can share one window inside named `subTest` blocks; stateful scenarios
should retain fresh fixtures. When scientific checks use the same deterministic
input and only inspect the result, combine those checks to avoid recomputing an
expensive fit while retaining every numerical assertion. Refresh scheduling
weights from successful complete release reports before reducing distinct
regression coverage just to lower the test count.

Each shard uploads its test outcomes, timings and process logs as a
`release-test-shard-*` artifact, including on test failure where the runner is
still available. A failed test, abnormal process exit, timeout, or incomplete
report blocks release. The portable build artifact can finish earlier; its
presence alone does not mean the release checks passed. Manual **Run workflow**
with `test_scope: full` runs these same checks without creating a Release.

To inspect the complete assignment without executing tests:

```powershell
.venv\Scripts\python.exe scripts/run_release_tests.py plan --output build/release-tests/plan.json --shards 12
```

## Multi-computer Git synchronization

Follow [AGENTS.md](../../AGENTS.md) for each development task and before changing
computers. Check the branch, upstream, staged, unstaged and untracked files;
fetch the actual upstream remote before comparing ahead/behind. Cached remote
refs alone do not prove synchronization. If the branch is only behind,
fast-forward only when the index/worktree are clean and no Git operation,
acquisition or other task is using the directory. Otherwise preserve the work
and report the state. Do not automatically stash, reset, clean, overwrite files,
switch branches or rewrite history to synchronize.

At completion, recheck changes and upstream differences. Report the branch,
uncommitted/untracked work, unpushed commits and incoming changes, or clearly
state when the remote is unverified. Local changes and unpushed commits remain
on this computer until committed and pushed. Commit/push only within the user's
authorization; inspect the scope, run `git diff --check` and proportionate checks,
and stage task-related files or hunks without including unrelated work. Keep
credentials, machine settings, virtual environments, logs, caches and unrequested
data out of Git. After pushing, fetch and confirm the intended commit reached the
intended branch. Only a clean current branch matching its freshly verified
upstream is fully synchronized; other local branches are a separate matter.

These rules apply while an agent is working; they are not a background service.
The repository's AGENTS.md travels with this project, but does not configure
another computer's global Codex settings.
