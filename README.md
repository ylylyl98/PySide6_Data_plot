# PySide6 Data Plot App

Desktop data plotting app built with PySide6. It loads measurement CSV files, renders PL/DRR/Compare views with Matplotlib, and exports CSV/PNG outputs.

## Recommended Windows installation

1. Open the repository's **Releases** page.
2. Download `DPTK-Setup-vX.Y.Z-Windows-x64.exe` from the latest release.
3. Run the installer.
4. Launch DPTK from the Start Menu.

Python and other development tools are not required. After launch, the app can
be pinned to the Windows taskbar normally.

Unsigned installers may initially show a Microsoft Defender SmartScreen warning.
DPTK does not bypass or disable Windows security checks.

### Updating

Use **Help → Check for Updates...** at any time. By default DPTK also performs
one unobtrusive background check after startup and shows a small notification
only when a newer stable version exists. DPTK never downloads or installs an
update automatically, never launches an installer automatically, and never
closes or restarts an active session. Downloading and launching an installer
each require an explicit user choice, and downloaded installers are checked
against the release's `SHA256SUMS.txt` before they can be launched.

### Portable version

Download `DPTK-vX.Y.Z-Windows-x64.zip`, extract it, open the
`PySide6_Data_Plot` folder, and run `PySide6_Data_Plot.exe`. The portable ZIP
remains available for restricted systems and users who prefer no installation.

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

On Windows, run `Data_Plot_App.bat` once. It creates `DPTK Desktop.lnk` beside
the launcher with the application icon; use that shortcut for normal launches
and taskbar pinning. To create a Desktop shortcut instead, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_windows_shortcut.ps1 -Desktop
```

## Build and Release

For a local Windows build, install the build dependency and run:

```powershell
python -m pip install -r requirements.txt -r requirements-build.txt
.\build_windows.bat
```

The portable application is created at
`dist\PySide6_Data_Plot\PySide6_Data_Plot.exe`. PyInstaller uses an `onedir`
layout, so distribute the entire `PySide6_Data_Plot` directory rather than the
EXE alone.

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
When you choose a data folder, CSV files must be in the folder root (not subfolders):

```text
<user-folder>/
  sample_001.csv
  sample_002.csv
  ...
  Processed Data/                    # created by app
  Initial data after processing/     # created by app
```

## Usage Notes
- `PL`: one-file plotting workflow for heatmap + spectrum.
- `Power Dependent`: select a single header-table CSV containing `Power_uW`, optional
  `stage_pos`, and numeric wavelength/energy column names. Each row is one power
  point. Numeric wavelength headers (for example `752.5829`) are converted to
  photon energy and the rows are sorted by power. Legacy multi-file series with a
  power token such as `37.96uW` in each filename remain supported.
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
- `Compare`: 2-4 selected files rendered in a compare grid.
- `Save PNG` exports with fixed Streamlit-style geometry (`8.0 x 6.2 in @ 150 DPI`) independent of window size.

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
