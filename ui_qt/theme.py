"""DPTK Desktop theme bootstrap.

Loads the versioned Fluent token snapshot and Qt-facing alias maps, layers a
small set of project-specific semantic aliases on top, and installs a single
``FluentThemeManager`` that owns the application palette, generated QSS, and
system light/dark/high-contrast preference.

Project colors adapt Fluent's neutral and brand roles to a quiet slate workbench.
The versioned official token snapshot remains unchanged. ``source_new_*``,
``source_processed_*`` and ``source_saved_*`` describe workflow status badges;
workspace aliases below give Widgets and Quick the same surface vocabulary.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional

from PySide6.QtWidgets import QApplication

from core.typography import FONT_FAMILY
from ui_qt.fluent_ui.theme import FluentThemeManager, ThemeMode
from ui_qt.fluent_ui.tokens import ResolvedTheme, TokenRepository
from ui_qt.typography import install_application_font

_RESOURCE_ROOT = Path(__file__).resolve().parent / "fluent_ui" / "resources"
_QSS_TEMPLATE = Path(__file__).resolve().parent / "fluent_ui" / "app.qss.in"

from ui_qt.theme_tokens import PROJECT_ALIASES


WORKSPACE_ALIAS_ROLES: dict[str, str] = {
    "workspace_background": "window_background",
    "workspace_surface": "canvas_background",
    "sidebar_surface": "surface_secondary",
    "control_surface": "surface_background",
    "heading_text": "text_primary",
    "section_text": "text_secondary",
    "active_nav_background": "brand_background_subtle",
    "active_nav_foreground": "brand_foreground",
    "chrome_surface": "surface_tertiary",
    "chrome_foreground": "text_secondary",
    "chrome_border": "border_subtle",
}


def _workspace_aliases(aliases: Mapping[str, str]) -> dict[str, str]:
    return {name: aliases[role] for name, role in WORKSPACE_ALIAS_ROLES.items()}


class ProjectResolvedTheme(ResolvedTheme):
    """Keep derived project roles in step with system-palette overrides."""

    def with_alias_overrides(
        self, overrides: Mapping[str, str], *, name: str | None = None
    ) -> ResolvedTheme:
        resolved = super().with_alias_overrides(overrides, name=name)
        merged = dict(resolved.aliases)
        merged.update(_workspace_aliases(merged))
        if resolved.name == "high-contrast":
            # Domain badges retain their text labels, while contrast themes own
            # their background/foreground pair and visible boundary.
            for status in ("new", "processed", "saved"):
                merged[f"source_{status}_background"] = merged["surface_background"]
                merged[f"source_{status}_foreground"] = merged["text_primary"]
                merged[f"source_{status}_border"] = merged["border_accessible"]
        return ProjectResolvedTheme(
            name=resolved.name,
            official=resolved.official,
            aliases=MappingProxyType(merged),
            source_metadata=resolved.source_metadata,
            shell_profile=resolved.shell_profile,
        )


class ProjectTokenRepository(TokenRepository):
    """Token repository that layers the project aliases onto resolved themes."""

    def resolve(self, theme: str, *, shell_profile: str | None = "fluent-workbench") -> ResolvedTheme:
        resolved = super().resolve(theme, shell_profile=shell_profile)
        merged = dict(resolved.aliases)
        merged.update(PROJECT_ALIASES.get(resolved.name, {}))
        merged.update(_workspace_aliases(merged))
        return ProjectResolvedTheme(
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
    install_application_font(app)
    from ui_qt.wheel_policy import install_wheel_value_guard
    install_wheel_value_guard(app)
    # The manager is an application-owned event filter. Replacing only the
    # module reference leaves old managers alive and observing every Qt event.
    from shiboken6 import isValid

    manager = getattr(app, "_dptk_theme_manager", None)
    if manager is not None and isValid(manager):
        requested_mode = mode if isinstance(mode, ThemeMode) else ThemeMode(mode)
        if manager.mode != requested_mode:
            manager.set_mode(requested_mode)
        else:
            manager.apply()
        _manager = manager
        return manager
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
        prefer_fluent_font=False,
        font_family=FONT_FAMILY,
        apply_body_pixel_size=False,
    )
    manager.apply()
    app._dptk_theme_manager = manager
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
