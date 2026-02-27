"""Shared helpers for Streamlit page state and export paths."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import streamlit as st


def ensure_state(key: str, default) -> None:
    """Set `st.session_state[key]` only when missing."""
    if key not in st.session_state:
        st.session_state[key] = default


def ensure_state_choice(key: str, options: Sequence[str], default: str) -> None:
    """Ensure state key exists and points to one of the current options."""
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default


def ensure_state_defaults(defaults: Mapping[str, object] | Iterable[tuple[str, object]]) -> None:
    """Apply a group of session-state defaults."""
    items = defaults.items() if hasattr(defaults, "items") else defaults
    for key, value in items:
        ensure_state(key, value)


def ensure_processed_dir(folder: str, processed_name: str) -> Path:
    """Create and return the processed-data output directory."""
    out_dir = Path(folder) / processed_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
