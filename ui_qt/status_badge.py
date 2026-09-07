"""Presentation-only status label used by workflow summaries."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel

from ui_qt.fluent_ui.style import set_fluent_property


class StatusBadge(QLabel):
    """A wrapped QLabel that owns status presentation metadata.

    Workflow controllers remain responsible for choosing the displayed text
    and opaque ``badge_state`` (for example ``new`` or ``processed``).  This
    component only presents that state; it never derives a Fluent severity
    from workflow state.
    """

    def __init__(
        self,
        text: str = "",
        parent=None,
        *,
        tooltip: str = "",
        app_role: str | None = "statusBadge",
        badge_state: str | None = None,
        fluent_severity: str | None = None,
        accessible_description: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.set_status(
            text,
            tooltip=tooltip,
            app_role=app_role,
            badge_state=badge_state,
            fluent_severity=fluent_severity,
            accessible_description=accessible_description,
        )

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        """Keep direct text changes accessible when no custom description is used."""
        super().setText(text)
        self.setAccessibleDescription(text)

    def set_status(
        self,
        text: str,
        *,
        tooltip: str = "",
        app_role: str | None = "statusBadge",
        badge_state: str | None = None,
        fluent_severity: str | None = None,
        accessible_description: str | None = None,
    ) -> None:
        """Update text and presentation-only metadata in one operation."""
        super().setText(str(text))
        self.setToolTip(str(tooltip))
        set_fluent_property(self, "appRole", app_role)
        set_fluent_property(self, "badgeState", badge_state)
        set_fluent_property(self, "fluentSeverity", fluent_severity)
        self.setAccessibleDescription(
            str(text) if accessible_description is None else str(accessible_description)
        )

