# Streamlit Data Plot App (Multipage)

This is a Streamlit multipage UI.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Folder layout
- `app.py`: landing page + shared sidebar folder selector
- `pages/1_Classic.py`: PL + DR/R processing & plotting
- `pages/2_Compare.py`: 2-up / 4-up compare viewer (KK/KKp/KpK/KpKp)
- `pages/3_Log.py`: session log viewer

Processing core is in `core/processing_run.py`.

## Notes
- This Streamlit version focuses on reliable, non-GUI plotting (Agg backend).
- Interactive drag-cursors from Qt are replaced with a stable slider-based Y cursor.
