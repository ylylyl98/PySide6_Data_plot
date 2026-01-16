import streamlit as st
from ui.state import init_session_state
from ui.sidebar import sidebar_folder_picker

st.set_page_config(
    page_title="DPTK — Streamlit",
    page_icon="📈",
    layout="wide",
)

init_session_state()

st.title("📈 DPTK — Streamlit (Multipage)")

sidebar_folder_picker()

if not st.session_state.user_folder:
    st.info("Choose a **User Folder** in the sidebar (type/paste a local path).")

st.markdown(
    """
Use the page navigation (left sidebar) to open:
- **Classic**: PL / DR\R processing + heatmap + spectra cursor
- **Compare**: KK / KKp (and optional KpK / KpKp) compare
- **Log**: session log

Tip: keep your CSVs in the *root* of the chosen folder (same as the PyQt app).
"""
)
