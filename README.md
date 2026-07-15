# PySide6 Data Plot App

Desktop data plotting app built with PySide6. It loads measurement CSV files, renders PL/DRR/Compare views with Matplotlib, and exports CSV/PNG outputs.

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
  a per-angle local background around 515 nm, integrates a fixed peak gate, plots
  intensity versus the measured angle, and exports a sorted CSV plus JSON settings.
- `DRR`:
  - `Self (last/first frame)` baseline modes.
  - `External` baseline mode from selected baseline files.
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
