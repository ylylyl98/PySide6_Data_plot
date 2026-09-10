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
- `Save PNG` exports with fixed Streamlit-style geometry (`8.0 x 6.2 in @ 150 DPI`) independent of window size.
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
