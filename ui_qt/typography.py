"""Shared Qt typography without importing Qt or plotting libraries on import."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from core.typography import FONT_FAMILY


def install_application_font(app) -> None:
    """Use Arial for Widgets and Quick while preserving font size and weight.

    The Windows offscreen plugin cannot enumerate installed system fonts. Load
    the installed Arial faces there so preview/test rendering uses the same
    family as native windows; normal startup needs no font scan or file loads.
    """
    from PySide6.QtGui import QFont, QFontDatabase

    if (sys.platform == "win32" and app.platformName() == "offscreen"
            and not getattr(app, "_dptk_arial_faces_loaded", False)):
        directory = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for filename in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
            path = directory / filename
            if path.is_file():
                QFontDatabase.addApplicationFont(str(path))
        app._dptk_arial_faces_loaded = True
    font = QFont(app.font())
    font.setFamily(FONT_FAMILY)
    app.setFont(font)
