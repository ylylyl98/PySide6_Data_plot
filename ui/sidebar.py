import streamlit as st
from pathlib import Path
from ui.logger import log


def sidebar_folder_picker() -> None:
    with st.sidebar:
        st.header("⚙️ Settings")

        folder = st.text_input(
            "User folder (local path)",
            value=st.session_state.user_folder,
            placeholder=r"e.g. C:\\data\\YZD320\\2026-01-15",
        )

        archive_name = st.text_input(
            "Archive folder name",
            value=st.session_state.archive_name,
        )
        processed_name = st.text_input(
            "Processed folder name",
            value=st.session_state.processed_name,
        )

        if st.button("Apply"):
            st.session_state.user_folder = folder.strip()
            st.session_state.archive_name = archive_name.strip() or st.session_state.archive_name
            st.session_state.processed_name = processed_name.strip() or st.session_state.processed_name
            if st.session_state.user_folder:
                p = Path(st.session_state.user_folder)
                if p.exists() and p.is_dir():
                    log(f"User folder set: {p}")
                else:
                    log(f"Warning: folder does not exist: {p}")

        st.divider()
        st.caption("Navigation is on the left: Classic / Compare / Log")
