import streamlit as st

DEFAULT_ARCHIVE = "Initial data after processing"
DEFAULT_PROCESSED = "Processed Data"


def init_session_state() -> None:
    ss = st.session_state
    ss.setdefault("user_folder", "")
    ss.setdefault("archive_name", DEFAULT_ARCHIVE)
    ss.setdefault("processed_name", DEFAULT_PROCESSED)

    # Plot controls shared across pages
    ss.setdefault("vmin", None)
    ss.setdefault("vmax", None)
    ss.setdefault("xlim", (None, None))
    ss.setdefault("ylim", (None, None))
    ss.setdefault("center_zero", False)
    ss.setdefault("pl_log", False)

    # Cursor
    ss.setdefault("cursor_gate", None)  # gate value

    # External baseline (DR/R)
    ss.setdefault("external_baseline", None)  # dict with energy, I0

    # Logging
    ss.setdefault("log", [])
