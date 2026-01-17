"""Local (server-side) native file dialogs.

These dialogs open on the machine running Streamlit.
- If you run Streamlit locally on your PC: this is your PC.
- If Streamlit runs on a remote server: the dialog opens on the server.

We isolate Tkinter usage here so the rest of the app stays clean.
"""

from __future__ import annotations

from typing import Optional


def pick_directory_dialog(initialdir: Optional[str] = None, title: str = "Select a folder") -> str:
    """Open a native folder picker dialog and return the selected folder path.

    Returns "" if the user cancels or if Tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return ""

    try:
        root = tk.Tk()
        root.withdraw()
        # Keep the dialog above the browser window (best effort)
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        path = filedialog.askdirectory(initialdir=initialdir, title=title)
        try:
            root.destroy()
        except Exception:
            pass

        return path or ""
    except Exception:
        return ""
