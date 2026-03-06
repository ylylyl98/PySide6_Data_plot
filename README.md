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
- `DRR`:
  - `Self (last/first frame)` baseline modes.
  - `External` baseline mode from selected baseline files.
- `Compare`: 2-4 selected files rendered in a compare grid.
- `Save PNG` exports with fixed size (`1600x1200 @ 200 DPI`) independent of window size.

## Smoke Check (No Test Framework Required)
Run:
```bash
python scripts/smoke_check.py
```

This performs lightweight import/path checks for the Qt app and core modules.

## Troubleshooting
- `No CSV files found`: confirm files are directly in the selected folder root and have `.csv` extension.
- `Folder does not exist` or `Selected path is not a folder`: re-pick the folder in the folder panel.
- `Compute failed` in DRR: verify baseline mode and selected baseline files match selected run files.
- Blank/invalid plots: confirm CSV columns are numeric and contain finite energy/gate/intensity values.
