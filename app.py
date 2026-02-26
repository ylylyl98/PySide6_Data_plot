import streamlit as st
from pathlib import Path

from ui.state import init_session_state
from ui.sidebar import sidebar_folder_picker
from core.file_ops import list_root_csvs

st.set_page_config(
    page_title="DPTK - Streamlit",
    page_icon="DPTK",
    layout="wide",
)

init_session_state()
sidebar_folder_picker()

st.title("DPTK - Streamlit (Multipage)")

c1, c2 = st.columns(2)
with c1:
    if st.button("Open PL", width="stretch"):
        st.switch_page("pages/1_PL.py")
with c2:
    if st.button("Open DR/R", width="stretch"):
        st.switch_page("pages/2_DRR.py")

st.divider()

folder = st.session_state.get("user_folder", "")
if not folder:
    st.info("Choose a **User Folder** in the sidebar (click **Browse...**).")
else:
    p = Path(folder)
    if not (p.exists() and p.is_dir()):
        st.error(f"Folder does not exist: {folder}")
    else:
        files = list_root_csvs(folder)
        st.success(f"Selected folder: `{folder}`")
        st.write(f"CSV files in root: **{len(files)}**")

st.markdown(
    """
Use the page navigation (left sidebar) to open:
- **PL**: PL heatmap + spectra cursor + save PNG
- **DRR**: DR/R (self first/last, external baseline) + derivative options + save PNG
- **Compare**: KK / KKp (and optional KpK / KpKp) compare
- **Log**: session log

Tip: keep your CSVs in the *root* of the chosen folder.
"""
)
