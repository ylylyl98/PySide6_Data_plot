# Streamlit Data Plot App (Multipage)

Streamlit multipage UI for loading measurement CSV files, plotting heatmaps/spectra, and exporting PNG/DAT outputs.

## Installation
```bash
git clone https://github.com/ylylyl98/streamlit-data-plot-app.git
cd streamlit-data-plot-app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
Use either:
```bash
streamlit run app.py
```
or double-click `Data_Plot_App.bat` from the repo root.

## Expected Folder Structure
When you choose a user folder in the sidebar, CSV files must be in the folder root (not subfolders):

```text
<user-folder>/
  sample_001.csv
  sample_002.csv
  ...
  Processed Data/                    # created by app
  Initial data after processing/     # created by app
```

## Usage Notes
- `PL` page: one-file-at-a-time processing with export workflow for linear/log outputs.
- `DRR` page:
  - `Self` mode uses per-file first/last frame baselines.
  - `External` mode uses selected baseline CSV files from the same folder.
- `Compare` page: side-by-side panels for KK/KKp (or KK/KKp/KpK/KpKp) with optional VP views.

## Smoke Check (No Test Framework Required)
Run:
```bash
python scripts/smoke_check.py
```

This performs lightweight import/path checks for core modules and app entry files.

## Troubleshooting
- `No CSV files found`: confirm files are directly in the selected folder root and have `.csv` extension.
- `Folder does not exist` or `Selected path is not a folder`: re-pick the folder using sidebar `Browse...`.
- `Compute failed` in DRR: verify baseline mode and selected baseline files are compatible with the selected run files.
- Blank/invalid plots: confirm CSV columns are numeric and contain finite energy/gate/intensity values.
