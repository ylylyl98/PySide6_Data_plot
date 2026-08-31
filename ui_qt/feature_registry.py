"""Feature-page registration for the Qt application shell.

This module owns navigation metadata so adding a feature does not require
editing the main window's layout code. Feature builders still receive the
main window as their shared application context during this first refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtWidgets import QWidget


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    description: str
    builder_name: str | None = None
    scrollable: bool = True

    def build(self, owner: Any) -> QWidget:
        if self.builder_name is None:
            return QWidget()
        return getattr(owner, self.builder_name)()


# Keep ordering stable because it is also the user-facing workflow order.
FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("pl", "PL", "Photoluminescence", "_build_pl_tab"),
    FeatureSpec("drr", "DRR", "Differential reflectance", "_build_drr_tab"),
    FeatureSpec("cmp", "Compare", "Compare measurement channels", "_build_compare_tab"),
    FeatureSpec("power", "Power", "Power Dependent", "_build_power_tab"),
    FeatureSpec("mcd", "MCD", "Magnetic circular dichroism", "_build_mcd_tab"),
    FeatureSpec("shg", "SHG", "SHG Processing", "_build_shg_tab"),
    FeatureSpec("slides", "Slides", "Build a PowerPoint from processed plot PNGs"),
    FeatureSpec("tools", "Tools", "Log / Tools", "_build_tools_tab"),
)
