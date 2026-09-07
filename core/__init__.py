"""Core APIs for data I/O, processing, plotting, and export.

Submodules are loaded on demand so lightweight tools (such as the MCD
organizer's metadata catalog) do not pay the startup cost of every plotting
and processing dependency.
"""

from importlib import import_module

__all__ = ["data_io", "processing", "plotting", "export", "mcd"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
