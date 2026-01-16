import streamlit as st

from ui.state import init_session_state
from ui.sidebar import sidebar_folder_picker

st.set_page_config(page_title="Log", page_icon="🧾", layout="wide")
init_session_state()
sidebar_folder_picker()

st.title("🧾 Log")

if st.button("Clear log"):
    st.session_state.log = []

st.text_area("Session log", value="\n".join(st.session_state.log), height=600)
