# DPTK Desktop — PySide6 Data Plot

**English** | [简体中文](README.zh-CN.md)

DPTK is a desktop application for exploring spectroscopy measurements, comparing
maps and spectra, and exporting figures together with numerical results and
processing metadata. It includes PL, differential reflectance (ΔR/R), polarization
comparison, power dependence, MCD, SHG, peak analysis, and PowerPoint slide building.

[Download for Windows](https://github.com/ylylyl98/PySide6_Data_plot/releases/latest)
· [User guide](docs/en/user-guide.md)
· [Report an issue](https://github.com/ylylyl98/PySide6_Data_plot/issues)

[![Fast PR checks](https://github.com/ylylyl98/PySide6_Data_plot/actions/workflows/fast-checks.yml/badge.svg)](https://github.com/ylylyl98/PySide6_Data_plot/actions/workflows/fast-checks.yml)
· [Windows build history](https://github.com/ylylyl98/PySide6_Data_plot/actions/workflows/build-windows-release.yml)

## Contents

- [Download and update](#download-and-update)
- [What you can do](#what-you-can-do)
- [Your first measurement](#your-first-measurement)
- [Display controls](#display-controls)
- [Data and exports](#data-and-exports)
- [Documentation](#documentation)
- [Run from source](#run-from-source)
- [Development and support](#development-and-support)

## Download and update

The published packages target **Windows x64**. The installer includes the
application and its Python dependencies; ordinary users do not need to install
Python or development tools.

### Installer

1. Open the [latest release](https://github.com/ylylyl98/PySide6_Data_plot/releases/latest).
2. Download `DPTK-Setup-vX.Y.Z-Windows-x64.exe`.
3. Run the installer, then open DPTK from the Start Menu.

`X.Y.Z` represents the release version. Download from **Releases**; Actions
artifacts are intended for development and testing. An unsigned installer may
show a Windows SmartScreen warning. DPTK does not disable Windows security checks.

### Portable package

1. Download `DPTK-vX.Y.Z-Windows-x64.zip` from the same release.
2. Extract the entire archive.
3. Open `PySide6_Data_Plot/PySide6_Data_Plot.exe` in the extracted folder.

Keep the executable with its `_internal` directory and other bundled files.
Moving only the EXE is insufficient. Source-code ZIP downloads are not portable
application packages.

### Updates

Choose **Help → Check for Updates…**. Automatic update checking can notify you
about newer stable versions; downloading and installing remain under your control.
The updater verifies the downloaded installer against `SHA256SUMS.txt`.
See [troubleshooting](docs/en/troubleshooting.md) if checking or downloading fails.

## What you can do

| Workflow | Typical task | Main results |
| --- | --- | --- |
| PL | Explore one measurement as a heatmap and spectrum; compare Linear/Log displays | Linear/Log PNGs, full DAT, metadata |
| DRR | Select and validate backgrounds, inspect ΔR/R and energy or mixed derivatives | Raw ΔR/R and derivative maps, DAT, metadata |
| Compare | Compare selected channels and calculate supported valley-polarization results | Channel and VP maps, numerical data |
| Power Dependence | Inspect power sweeps, assign KK/KKp channels, fit supported intensity models | Maps, comparisons, fits and export packages |
| MCD | Inspect magnetic-field sweeps, extract MCD(B), organize and compare analyses | Map data, traces, diagnostics and settings |
| SHG | Analyze angular sweeps and compare reference/sample measurements | Excel workbook with angular results and spectra, settings |
| Peak Analysis | Detect, fit and inspect spectral branches in a separate workspace | Fits, diagnostics and saved analysis results |
| Slides | Find exported PNGs and arrange them in PowerPoint | Editable layouts and captions with embedded figures |

Input requirements differ by workflow. See the [user guide](docs/en/user-guide.md)
and [data reference](docs/en/data-and-exports.md) before interpreting a new file type.
PowerPoint file generation and insertion into a running PowerPoint application
have different requirements; the user guide explains both routes.

## Your first measurement

1. **Choose an experiment folder.** Keep canonical acquisition files in
   `Initial Data/`. For the usual CSV workflow, copy the files you want to analyze
   into the experiment root and select that folder in DPTK.
2. **Choose the workflow.** Start with PL for a single PL map, or use DRR,
   Compare, Power, MCD or SHG according to the measurement type.
3. **Select compatible sources.** CSV layouts and grouping rules are workflow
   specific. Supported processed DAT files can also be reopened.
4. **Review processing.** For new DRR CSV selections without a saved background,
   the background window opens first. Choose a reference and apply it; cancelling
   leaves the selection waiting for a background.
5. **Adjust the display.** Set X/Y ranges and color limits, use Auto/Fix as needed,
   and inspect spectra. PL Linear/Log and DRR product controls are described below.
6. **Export and inspect the results.** The workflow's export action writes into
   `Processed Data/<workflow>/`. Keep metadata beside numerical files when moving
   results, and verify the export completed before removing working copies.

For a source checkout, a synthetic demo is available without selecting your own
measurements:

```powershell
.venv\Scripts\python.exe run_qt.py --qt-quick --demo
```

Detailed selection, fitting and export steps are in the
[user guide](docs/en/user-guide.md).

## Display controls

- **X/Y and color ranges:** edit Min/Max directly. Enter or moving focus commits
  a value; arrows adjust it. Editing a number does not automatically enable Fix.
- **Auto/Fix:** Fix protects an individual endpoint. Auto recalculates only
  unfixed endpoints; it does not clear the locks.
- **Scientific notation:** inputs accept values such as `1e3`, `1e-6` and
  `-2.5e4` within each field's range. Integer fields require an integer result.
  Compact display formatting preserves the field's configured storage precision.
- **PL:** Linear/Log tabs select the plot and color controls. With Side by side
  enabled, the tab selects the color controls being edited. New displays default
  to `turbo`; saved colormaps are retained. One, two or three regions share energy
  boundaries while keeping separate Linear/Log color limits.
- **DRR:** choosing ΔR/R or Derivative in the plot controls selects its matching
  Display color tab. Clicking a Display color tab edits that product's colors
  without switching the plot. Side by side retains the selected color tab.
- **More plot space:** **Ctrl+B** toggles the parameter sidebar. Drag the divider
  or use **Hide spectrum / Show spectrum** to adjust the map/spectrum layout.

PL reports invalid drafts beside their inputs. Correct them or press Escape
before switching color tabs. Log color Auto uses positive visible samples;
when no valid samples exist, it retains the previous limits and explains why.
See the [user guide](docs/en/user-guide.md) for precision rules, region boundaries
and saved display settings.

## Data and exports

A typical experiment keeps source files and generated results separate:

```text
Experiment/
├── Initial Data/          # canonical acquisition files
├── working-copy.csv       # selected working copy
└── Processed Data/
    ├── PL/
    ├── DRR/
    ├── Compare/
    ├── Power Dependence/
    ├── MCD/
    └── SHG/
```

- DAT exports are numerical matrices, suitable for re-import and Origin workflows.
  Changing the visible X/Y range does not turn the full DAT into a screenshot crop.
- PL saves independent Linear and Log PNGs plus full DAT. **Wide PNG** also keeps
  the 1.0× originals; with a multiplier greater than 1×, the export contains four PNGs.
- DRR exports ΔR/R and derivative PNG/DAT products when viewing only the derivative,
  using the derivative view's visible X/Y ranges for both figures and each
  product's own colors. Wide exports retain the original PNGs as well.
- A standard heatmap export is `1200 × 930 px` (`8.0 × 6.2 in`, 150 DPI).
  The wide option changes figure width, not the numerical data.
- Successful PL/DRR exports can retain display settings in metadata and experiment
  indexes. Keep the experiment directory together for portable file restoration;
  explicit manual locks are local preferences with their own precedence rules.
- Source cleanup is optional and verifies matching canonical copies. Failed or
  unverified exports do not authorize deleting source data. Manual archival is a
  separate action.

Read [data and exports](docs/en/data-and-exports.md) for all folder names, workbook
sheets, metadata, collision handling, source cleanup and re-import rules.

## Documentation

Each guide has a matching Chinese edition and a language switch at the top.

| Guide | Covers |
| --- | --- |
| [User guide](docs/en/user-guide.md) | Workflows, display settings, selection, analysis and slides |
| [Data and exports](docs/en/data-and-exports.md) | Input/output conventions, source lifecycle, PNG/DAT/XLSX and metadata |
| [Background and analysis reference](docs/en/background-reference.md) | DRR references, P/N construction, BN review and scientific limitations |
| [Development and release](docs/en/development.md) | Source setup, architecture, diagnostics, tests, compatibility and packaging |
| [Troubleshooting](docs/en/troubleshooting.md) | Loading, plotting, saving, updates and useful issue details |

Existing topic notes and historical work records remain under [docs/](docs/). The guides
above describe current operation; historical experiment results are not universal
processing defaults or guarantees for other measurements.

## Run from source

The Windows CI and release build use **Python 3.13**. The commands below use the
virtual environment's interpreter directly, so activating PowerShell scripts is
not required.

```powershell
git clone https://github.com/ylylyl98/PySide6_Data_plot.git
cd PySide6_Data_plot
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_qt.py
```

Normal startup uses the native Qt Quick interface. The legacy Widgets frontend
remains available for compatibility work; see [development](docs/en/development.md)
for its flags, isolated checks and explicit test settings.
Published binaries and CI validate Windows; these instructions do not promise
prebuilt packages or equivalent testing on other operating systems.

## Development and support

Start with the [development guide](docs/en/development.md) and repository
[collaboration rules](AGENTS.md). The scientific processing lives in `core/`;
`ui_qt/quick/` contains the native application bridge and QML interface.
`app_version.py` is the application version source of truth.

PR checks and release validation have different scopes. Tagged Windows releases
require the full test suite and packaged-application checks; an Actions artifact
alone is not proof that a release passed all checks.

When reporting a problem, include the app version, workflow, reproducible steps
and the visible error. Share a minimal synthetic or redacted example when possible.
See [troubleshooting](docs/en/troubleshooting.md) for what information is useful.
