import streamlit as st

# Fixed (not user-editable)
DEFAULT_ARCHIVE = "initial data after processing"
DEFAULT_PROCESSED = "processed data"


def init_session_state() -> None:
    ss = st.session_state

    # Folder selection
    ss.setdefault("user_folder", "")
    ss.setdefault("user_folder_input", ss.get("user_folder", ""))

    # Fixed output folders (do not show in UI)
    ss.setdefault("archive_name", DEFAULT_ARCHIVE)
    ss.setdefault("processed_name", DEFAULT_PROCESSED)

    # Plot controls shared across pages
    ss.setdefault("vmin", None)
    ss.setdefault("vmax", None)
    ss.setdefault("xlim", (None, None))
    ss.setdefault("ylim", (None, None))
    ss.setdefault("center_zero", True)   # DR/R usually wants this
    ss.setdefault("pl_log", False)

    # Cursor
    ss.setdefault("cursor_gate", None)

    # External baseline (DR/R)
    ss.setdefault("external_baseline", None)  # dict with keys: energy, I0

    # Logging
    ss.setdefault("log", [])
