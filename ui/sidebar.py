import streamlit as st
from ui.logger import log
from ui.local_dialogs import pick_directory_dialog
from core.file_ops import list_root_csvs


def _browse_pick_folder() -> None:
    """Pick a folder via native dialog and update session_state safely."""
    picked = pick_directory_dialog(initialdir=st.session_state.get("user_folder") or None)
    if picked:
        st.session_state["user_folder"] = picked
        log(f"User folder set (dialog): {picked}")


def sidebar_folder_picker() -> None:
    with st.sidebar:
        st.header("Settings")

        # Show current folder as read-only text (no manual editing)
        current = st.session_state.get("user_folder", "")
        st.text_input(
            "User folder (local path)",
            value=current,
            disabled=True,
        )

        # Browse-only
        st.button("Browse...", width="stretch", on_click=_browse_pick_folder)

        st.caption("Tip: 'Browse...' opens a native folder dialog on the machine running Streamlit.")

        st.divider()
        # Lightweight status panel
        folder = st.session_state.get("user_folder", "")
        if folder:
            files = list_root_csvs(folder)
            st.caption(f"Folder: {folder}")
            st.caption(f"Root CSVs: {len(files)}")
            st.caption(f"Archive: {st.session_state.get('archive_name', '')}")
            st.caption(f"Processed: {st.session_state.get('processed_name', '')}")

        st.divider()
        st.caption("Navigation is on the left: PL / DRR / Compare / Log")

