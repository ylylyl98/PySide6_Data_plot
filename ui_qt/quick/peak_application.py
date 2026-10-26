"""Lazy native Peak Analysis integration for the application shell."""
from __future__ import annotations


class PeakApplicationMixin:
    def _peak_init(self):
        self._peak_launcher = None
        self._peak_source_key = None

    def _peak_available(self):
        if self._closed or self._workflow not in {"pl", "drr", "cmp", "power"}:
            return False
        owner = self._workflow_bridges.get(self._workflow, self)
        timer = getattr(owner, "_plot_timer", None)
        if timer is not None and timer.isActive():
            return False
        if owner is self:
            return not self._busy and self._cube is not None and bool(self._maps)
        state = owner.snapshot
        if state.get("busy") or state.get("analysisBusy"):
            return False
        if self._workflow == "cmp":
            return state.get("view") == "intensity" and bool(owner._maps)
        return owner._stage == "plot" and owner._values["mode"] != "vp" and bool(owner._maps)

    def _peak_snapshot(self):
        state = self._peak_launcher.snapshot if self._peak_launcher is not None else {}
        return dict(canAnalyzePeaks=self._peak_available() and not state.get("busy", False),
                    peakAnalysisBusy=state.get("busy", False),
                    hasPeakOverlay=state.get("hasOverlay", False),
                    peakOverlayVisible=state.get("overlayVisible", True))

    def _capture_peak_sources(self):
        if not self._peak_available():
            return []
        from ui_qt.quick.peak_source_adapters import capture_peak_sources
        return capture_peak_sources(self)

    @staticmethod
    def _peak_signature(sources):
        # Viewport/selected row are captured afresh when opening. They do not
        # invalidate numerical overlays. Rebuilt axes and changed arrays do.
        return tuple((item["kind"], item["channel"], item["source"],
                      id(item["cube"]), id(item["cube"].Z), id(item["cube"].energy), id(item["cube"].gate),
                      tuple(id(axis) for axis in item.get("axes", ()))) for item in sources)

    def _sync_peak_sources(self):
        if self._peak_launcher is None or self._closed:
            return
        sources = self._capture_peak_sources()
        key = self._peak_signature(sources)
        if key != self._peak_source_key:
            self._peak_source_key = key
            self._peak_launcher.setSources(sources)

    def _open_peak_analysis(self):
        sources = self._capture_peak_sources()
        if not sources:
            self._message("Plot PL, DRR, Compare intensity or Power intensity first. VP is not peak-fitted.")
            return
        if self._peak_launcher is None:
            from ui_qt.quick.peak_analysis_launcher import QuickPeakAnalysisLauncher
            self._peak_launcher = QuickPeakAnalysisLauncher(settings=self.settings, parent=self)
            self._peak_launcher.statusMessage.connect(self._message)
            self._peak_launcher.snapshotChanged.connect(self._publish)
            self._peak_launcher.overlayReady.connect(self._peak_overlay_ready)
        self._peak_source_key = self._peak_signature(sources)
        self._peak_launcher.openSources(sources)

    def _peak_overlay_ready(self, packet):
        result = packet.get("result", {})
        self._result = (f"Peak Analysis · {packet.get('name', packet.get('channel', ''))}\n"
                        f"{len(result.get('branches', []))} branches · {len(result.get('points', []))} points\n"
                        "Overlay matches the current numeric data. Open Peak Analysis for the full result table and exports.")
        self._publish()

    def _toggle_peak_overlay(self):
        if self._peak_launcher is not None:
            self._peak_launcher.setOverlayVisible(not self._peak_launcher.snapshot["overlayVisible"])

    def _close_peak(self):
        if self._peak_launcher is not None:
            self._peak_launcher.close()
