"""Explicit background actions for native DRR integration fixtures."""


def apply_self_background(bridge, wait, *, frame=1):
    """Choose Self through the same draft/apply transaction as the UI."""
    bridge.openDrrBackground()
    background = bridge.drrBackgroundBridge
    wait(lambda: background is not None and not background.snapshot["busy"])
    background.setChoice("kind", 4)
    background.setChoice("frame", frame)
    background.invoke("apply")
    wait(lambda: not background.snapshot["busy"] and not bridge._pending
         and not bridge.snapshot["busy"])
    if background.snapshot["error"]:
        raise AssertionError(background.snapshot["error"])
    if not bridge.snapshot["canExport"]:
        raise AssertionError(bridge.snapshot["status"])
    return bridge._drr_products
