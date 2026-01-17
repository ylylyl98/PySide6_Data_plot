import streamlit as st

from ui.state import init_session_state
from ui.sidebar import sidebar_folder_picker

st.set_page_config(page_title="Log", page_icon="🧾", layout="wide")
init_session_state()
sidebar_folder_picker()

st.title("🧾 Log")

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Clear log", width="stretch"):
        st.session_state.log = []
with c2:
    st.download_button(
        "Download log.txt",
        data="\n".join(st.session_state.log).encode("utf-8"),
        file_name="log.txt",
        mime="text/plain",
    )

st.text_area("Session log", value="\n".join(st.session_state.log), height=600)
