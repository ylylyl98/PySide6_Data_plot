import streamlit as st


def st_pyplot(fig):
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)


def btn(label: str, *, key: str, disabled: bool = False):
    try:
        return st.button(label, key=key, width="stretch", disabled=disabled)
    except TypeError:
        return st.button(label, key=key, use_container_width=True, disabled=disabled)


def btn_click(label: str, *, key: str, on_click=None, args=None, disabled: bool = False):
    try:
        return st.button(label, key=key, width="stretch", on_click=on_click, args=args, disabled=disabled)
    except TypeError:
        return st.button(label, key=key, use_container_width=True, on_click=on_click, args=args, disabled=disabled)


def dl_btn(label: str, *, data: bytes, file_name: str, mime: str, key: str):
    try:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, width="stretch")
    except TypeError:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, use_container_width=True)


def rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
