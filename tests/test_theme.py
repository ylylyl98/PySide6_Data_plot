"""Regression tests for the DPTK Fluent theme layer."""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui_qt.fluent_ui.style import render_qss_file
from ui_qt.fluent_ui.tokens import ResolvedTheme, TokenValidationError
from ui_qt.theme import PROJECT_ALIASES, ProjectTokenRepository, alias
from tests.profile_phases import profile_phase
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox

_RESOURCES = Path(__file__).resolve().parent.parent / "ui_qt" / "fluent_ui" / "resources"
_QSS_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "ui_qt" / "fluent_ui" / "app.qss.in"
)


def _repository() -> ProjectTokenRepository:
    return ProjectTokenRepository(
        _RESOURCES / "fluent2-official-web-theme-tokens.json",
        _RESOURCES / "qt-token-map.json",
        _RESOURCES / "shell-token-map.json",
    )


class ThemeInstallationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_reapply_skips_identical_qss_but_restores_external_changes(self) -> None:
        from unittest.mock import patch
        from ui_qt.theme import install_theme

        manager = install_theme(self.app, mode="light")
        changes = []
        manager.themeChanged.connect(changes.append)
        self.addCleanup(manager.themeChanged.disconnect, changes.append)
        self.addCleanup(manager.set_mode, "light")
        expected = self.app.styleSheet()
        with patch.object(self.app, "setStyleSheet", wraps=self.app.setStyleSheet) as write:
            manager.apply()
            write.assert_not_called()
            self.assertEqual([theme.name for theme in changes], ["light"])
            self.app.setStyleSheet("QWidget { color: red; }")
            write.reset_mock()
            manager.apply()
            write.assert_called_once_with(expected)
            self.assertEqual(self.app.styleSheet(), expected)
            self.assertEqual([theme.name for theme in changes], ["light", "light"])
            write.reset_mock()
            manager.set_mode("dark")
            write.assert_called_once()
            self.assertNotEqual(self.app.styleSheet(), expected)
            self.assertEqual(changes[-1].name, "dark")

    def test_reinstallation_reuses_manager_and_changes_mode(self) -> None:
        from ui_qt.theme import install_theme
        from ui_qt.fluent_ui.theme import FluentThemeManager

        first = install_theme(self.app, mode="light")
        self.addCleanup(first.set_mode, "light")
        count = len(self.app.findChildren(FluentThemeManager))
        second = install_theme(self.app, mode="dark")
        self.assertIs(second, first)
        self.assertEqual(len(self.app.findChildren(FluentThemeManager)), count)
        self.assertEqual(second.current_theme.name, "dark")
        self.assertIs(install_theme(self.app, mode="light"), first)
        self.assertEqual(first.current_theme.name, "light")

    def test_destroyed_window_disconnects_application_theme_callbacks(self) -> None:
        import sys
        from unittest.mock import patch
        from ui_qt.main_window import MainWindow
        from ui_qt.theme import install_theme
        from tests.ui_test_helpers import dispose_owned_window

        manager = install_theme(self.app, mode="light")
        self.addCleanup(manager.set_mode, "light")
        with patch.object(MainWindow, "_restore_last_folder", autospec=True), \
             patch.object(MainWindow, "_schedule_automatic_update_check", autospec=True):
            window = MainWindow()
        self.addCleanup(dispose_owned_window, window)
        light_icon = window.load_action.icon().cacheKey()
        manager.set_mode("dark")
        self.assertNotEqual(window.load_action.icon().cacheKey(), light_icon)
        dispose_owned_window(window)
        errors = []
        with patch.object(sys, "excepthook", side_effect=lambda *args: errors.append(args)):
            install_theme(self.app, mode="light")
            self.app.processEvents()
        self.assertEqual(errors, [], [str(error[1]) for error in errors])

    def test_installation_after_application_recreation(self) -> None:
        import subprocess
        import sys
        import textwrap

        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent("""
                from PySide6.QtWidgets import QApplication
                from shiboken6 import delete, isValid
                from ui_qt.theme import install_theme
                app = QApplication([])
                first = install_theme(app, mode="light")
                delete(app)
                assert not isValid(first)
                app = QApplication([])
                second = install_theme(app, mode="dark")
                assert second is not first and second.parent() is app
                assert second.current_theme.name == "dark"
            """)],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_reinstallation_replaces_explicitly_destroyed_manager(self) -> None:
        from PySide6.QtCore import QCoreApplication, QEvent
        from ui_qt.theme import install_theme
        from shiboken6 import isValid

        first = install_theme(self.app, mode="light")
        first.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.assertFalse(isValid(first))
        second = install_theme(self.app, mode="light")
        self.assertIsNot(second, first)
        self.assertTrue(isValid(second))
        self.assertIs(second.parent(), self.app)


class ThemeLayerTests(unittest.TestCase):
    def test_repository_resolves_both_themes_with_project_aliases(self) -> None:
        repository = _repository()
        for name in ("light", "dark"):
            theme = repository.resolve(name, shell_profile="fluent-workbench")
            self.assertIsInstance(theme, ResolvedTheme)
            self.assertEqual(theme.name, name)
            self.assertIn("text_primary", theme.aliases)
            for project_alias in (
                "source_new_foreground",
                "source_processed_foreground",
                "source_saved_foreground",
            ):
                self.assertIn(project_alias, theme.aliases)

        light = repository.resolve("light", shell_profile="fluent-workbench")
        dark = repository.resolve("dark", shell_profile="fluent-workbench")
        self.assertNotEqual(light.aliases["window_background"], dark.aliases["window_background"])
        self.assertEqual(
            light.aliases["source_processed_foreground"],
            PROJECT_ALIASES["light"]["source_processed_foreground"],
        )

    def test_app_qss_renders_for_both_themes_without_leftovers(self) -> None:
        repository = _repository()
        with tempfile.TemporaryDirectory() as asset_dir:
            for name in ("light", "dark"):
                theme = repository.resolve(name, shell_profile="fluent-workbench")
                rendered = render_qss_file(_QSS_TEMPLATE, theme, asset_directory=asset_dir)
                self.assertNotIn("@{", rendered)
                self.assertIn(theme.aliases["window_background"], rendered)

    def test_workflow_separator_uses_semantic_role_in_both_themes(self) -> None:
        repository = _repository()
        with tempfile.TemporaryDirectory() as asset_dir:
            for name in ("light", "dark"):
                theme = repository.resolve(name, shell_profile="fluent-workbench")
                rendered = render_qss_file(_QSS_TEMPLATE, theme, asset_directory=asset_dir)
                self.assertIn('QFrame#workflowUtilitySeparator[fluentRole="divider"]', rendered)
                self.assertIn(theme.aliases["border_subtle"], rendered)

    def test_alias_falls_back_before_theme_install(self) -> None:
        value = alias("text_primary")
        self.assertTrue(value.startswith("#"))
        self.assertEqual(alias("source_saved_foreground"), PROJECT_ALIASES["light"]["source_saved_foreground"])

    def test_unknown_alias_raises(self) -> None:
        repository = _repository()
        theme = repository.resolve("light", shell_profile="fluent-workbench")
        with self.assertRaises(TokenValidationError):
            theme.value("definitely_not_an_alias")

    def test_spinbox_subcontrols_define_all_theme_states(self) -> None:
        repository = _repository()
        with tempfile.TemporaryDirectory() as asset_dir:
            rendered = render_qss_file(
                _QSS_TEMPLATE,
                repository.resolve("dark", shell_profile="fluent-workbench"),
                asset_directory=asset_dir,
            )
        for selector in (
            "QSpinBox::up-button",
            "QSpinBox::down-button",
            "QDoubleSpinBox::up-button",
            "QDoubleSpinBox::down-button",
            "QSpinBox::up-button:hover",
            "QSpinBox::up-button:pressed",
            "QSpinBox::up-button:disabled",
            "QSpinBox[readOnly=\"true\"]::up-arrow",
            "QSpinBox[fluentInvalid=\"true\"]::up-button",
        ):
            self.assertIn(selector, rendered)
        self.assertNotIn("QSpinBox:read-only::", rendered)
        self.assertNotIn("QDoubleSpinBox:read-only::", rendered)

    def test_read_only_spinbox_subcontrols_override_interactive_states(self) -> None:
        repository = _repository()
        with tempfile.TemporaryDirectory() as asset_dir:
            rendered = render_qss_file(
                _QSS_TEMPLATE,
                repository.resolve("dark", shell_profile="fluent-workbench"),
                asset_directory=asset_dir,
            )
        for widget in ("QSpinBox", "QDoubleSpinBox"):
            for direction in ("up", "down"):
                for state in ("hover", "pressed", "focus"):
                    self.assertIn(
                        f'{widget}[readOnly="true"]::{direction}-button:{state}',
                        rendered,
                    )
                self.assertIn(
                    f'{widget}[readOnly="true"]::{direction}-arrow:hover',
                    rendered,
                )

    def test_spinbox_arrow_assets_are_subdued_until_hover_or_focus(self) -> None:
        """Arrow imagery must distinguish quiet, interactive, and read-only states."""

        def declaration_for_selector(rendered: str, selector: str) -> str:
            for match in re.finditer(r"(?ms)(?P<header>[^{}]+)\{(?P<body>[^{}]*)\}", rendered):
                selectors = {part.strip() for part in match.group("header").split(",")}
                if selector in selectors:
                    return match.group("body")
            self.fail(f"selector not found: {selector}")

        repository = _repository()
        with tempfile.TemporaryDirectory() as asset_dir:
            for name in ("light", "dark"):
                rendered = render_qss_file(
                    _QSS_TEMPLATE,
                    repository.resolve(name, shell_profile="fluent-workbench"),
                    asset_directory=asset_dir,
                )
                for widget in ("QSpinBox", "QDoubleSpinBox"):
                    for direction in ("up", "down"):
                        quiet = declaration_for_selector(rendered, f"{widget}::{direction}-arrow")
                        self.assertIn("-text_disabled-", quiet)

                        hover = declaration_for_selector(
                            rendered, f"{widget}::{direction}-arrow:hover"
                        )
                        self.assertIn("-text_primary-", hover)

                        pressed = declaration_for_selector(
                            rendered, f"{widget}::{direction}-arrow:pressed"
                        )
                        self.assertIn("-text_primary-", pressed)

                        focus = declaration_for_selector(
                            rendered, f"{widget}::{direction}-arrow:focus"
                        )
                        self.assertIn("-text_primary-", focus)

                        disabled = declaration_for_selector(
                            rendered, f"{widget}::{direction}-arrow:disabled"
                        )
                        self.assertIn("-text_disabled-", disabled)

                        read_only = declaration_for_selector(
                            rendered, f'{widget}[readOnly="true"]::{direction}-arrow'
                        )
                        self.assertIn("-text_disabled-", read_only)
                        for state in ("hover", "pressed", "focus"):
                            override = declaration_for_selector(
                                rendered,
                                f'{widget}[readOnly="true"]::{direction}-arrow:{state}',
                            )
                            self.assertIn("-text_disabled-", override)
                        self.assertNotIn(
                            f"{widget}:focus::{direction}-arrow", rendered
                        )

    def test_spinbox_has_no_duplicate_center_arrow(self) -> None:
        """Only the two dedicated stepper arrows may render an icon."""
        with profile_phase("qapplication_acquisition"):
            app = QApplication.instance() or QApplication([])
        repository = _repository()
        previous = app.styleSheet()
        try:
            for name in ("light", "dark"):
                with tempfile.TemporaryDirectory() as asset_dir:
                    with profile_phase(f"theme_render_installation:{name}"):
                        theme = repository.resolve(name, shell_profile="fluent-workbench")
                        app.setStyleSheet(
                            render_qss_file(_QSS_TEMPLATE, theme, asset_directory=asset_dir)
                        )
                    for spin_type in (QSpinBox, QDoubleSpinBox):
                        with profile_phase(f"spinbox_render_assertion:{name}:{spin_type.__name__}"):
                            spin = spin_type()
                            spin.setRange(0, 100)
                            spin.setValue(42)
                            spin.resize(180, 40)
                            spin.show()
                            app.processEvents()
                            arrow_colors = {
                                QColor(theme.aliases["text_disabled"]).rgba(),
                                QColor(theme.aliases["text_primary"]).rgba(),
                            }
                            image = spin.grab().toImage()
                            center_matches = sum(
                                1
                                for x in range(70, 120)
                                for y in range(4, image.height() - 4)
                                if image.pixelColor(x, y).alpha() > 200
                                and image.pixelColor(x, y).rgba() in arrow_colors
                            )
                            self.assertLess(
                                center_matches,
                                5,
                                f"unexpected center arrow rendered for {name} {spin_type.__name__}",
                            )
                            spin.deleteLater()
        finally:
            with profile_phase("theme_restore_teardown"):
                app.setStyleSheet(previous)

    def test_approved_shell_icons_are_current_color_svg_assets(self) -> None:
        icon_root = _QSS_TEMPLATE.parent / "icons"
        for filename in (
            "open-folder.svg", "arrow-sync.svg", "save.svg", "home.svg",
            "arrow-left.svg", "arrow-right.svg", "cursor-move.svg", "zoom.svg",
            "layout.svg", "edit.svg", "panel-results.svg", "panel-log.svg",
        ):
            source = (icon_root / filename).read_text(encoding="utf-8")
            self.assertIn("currentColor", source)


if __name__ == "__main__":
    unittest.main()
