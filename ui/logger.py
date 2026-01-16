import streamlit as st
from datetime import datetime


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log.append(f"[{ts}] {msg}")
