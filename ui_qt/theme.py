"""DPTK Desktop theme bootstrap.

Loads the versioned Fluent token snapshot and Qt-facing alias maps, layers a
small set of project-specific semantic aliases on top, and installs a single
``FluentThemeManager`` that owns the application palette, generated QSS, and
system light/dark/high-contrast preference.

Project-specific aliases (``source_new_*``, ``source_processed_*``,
``source_saved_*``) describe workflow status badges that Fluent's generic
status vocabulary does not cover. They are derived values documented here, not
official Fluent tokens.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Optional

from PySide6.QtWidgets import QApplication

from ui_qt.fluent_ui.theme import FluentThemeManager, ThemeMode
from ui_qt.fluent_ui.tokens import ResolvedTheme, TokenRepository

_RESOURCE_ROOT = Path(__file__).resolve().parent / "fluent_ui" / "resources"
_QSS_TEMPLATE = Path(__file__).resolve().parent / "fluent_ui" / "app.qss.in"

PROJECT_ALIASES: dict[str, dict[str, str]] = {
    "light": {
        "source_new_foreground": "#1769AA",
        "source_new_background": "#EAF3FC",
        "source_new_border": "#86B8E3",
        "source_processed_foreground": "#237A3B",
        "source_processed_background": "#EAF6ED",
        "source_processed_border": "#8BC79A",
        "source_saved_foreground": "#6F42A5",
        "source_saved_background": "#F3EDFA",
        "source_saved_border": "#BCA1D8",
    },
    "dark": {
        "source_new_foreground": "#6CB8F0",
        "source_new_background": "#15283A",
        "source_new_border": "#3F6F9C",
        "source_processed_foreground": "#7BD88F",
        "source_processed_background": "#17331F",
        "source_processed_border": "#3F7A50",
        "source_saved_foreground": "#C9A0F0",
        "source_saved_background": "#2A2136",
        "source_saved_border": "#6F5A8F",
    },
}


class ProjectTokenRepository(TokenRepository):
    """Token repository that layers the project aliases onto resolved themes."""

    def resolve(self, theme: str, *, shell_profile: str | None = "fluent-workbench") -> ResolvedTheme:
        resolved = super().resolve(theme, shell_profile=shell_profile)
        merged = dict(resolved.aliases)
        merged.update(PROJECT_ALIASES.get(resolved.name, {}))
        return ResolvedTheme(
            name=resolved.name,
            official=resolved.official,
            aliases=MappingProxyType(merged),
            source_metadata=resolved.source_metadata,
            shell_profile=resolved.shell_profile,
        )


_manager: Optional[FluentThemeManager] = None
_default_repository: Optional[ProjectTokenRepository] = None
_default_theme: Optional[ResolvedTheme] = None


def _default_resolved(name: str) -> str:
    """Resolve an alias without a live manager (tests, offscreen tooling)."""
    global _default_repository, _default_theme
    if _default_repository is None:
        _default_repository = ProjectTokenRepository(
            _RESOURCE_ROOT / "fluent2-official-web-theme-tokens.json",
            _RESOURCE_ROOT / "qt-token-map.json",
            _RESOURCE_ROOT / "shell-token-map.json",
        )
    if _default_theme is None:
        _default_theme = _default_repository.resolve(
            "light", shell_profile="fluent-workbench"
        )
    return _default_theme.value(name)


def install_theme(
    app: QApplication,
    *,
    mode: ThemeMode | str = ThemeMode.SYSTEM,
) -> FluentThemeManager:
    """Install the single application theme manager and apply the theme."""
    global _manager
    repository = ProjectTokenRepository(
        _RESOURCE_ROOT / "fluent2-official-web-theme-tokens.json",
        _RESOURCE_ROOT / "qt-token-map.json",
        _RESOURCE_ROOT / "shell-token-map.json",
    )
    manager = FluentThemeManager(
        app,
        repository,
        qss_template=_QSS_TEMPLATE,
        mode=mode,
        shell_profile="fluent-workbench",
        prefer_fluent_font=True,
        apply_body_pixel_size=False,
    )
    manager.apply()
    _manager = manager
    return manager


def theme_manager() -> Optional[FluentThemeManager]:
    """Return the installed theme manager, or None before installation."""
    return _manager


def alias(name: str) -> str:
    """Resolve a semantic alias from the active theme (official or project)."""
    manager = _manager
    if manager is not None and manager.current_theme is not None:
        return manager.current_theme.value(name)
    return _default_resolved(name)
