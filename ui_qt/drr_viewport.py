"""Cull offscreen DRR cells only while the interactive renderer is drawing."""
from contextlib import contextmanager
import numpy as np


def _cells(edges, limits):
    if not np.all(np.isfinite(limits)):
        return slice(0, len(edges) - 1)
    lo, hi = sorted(limits)
    indices = np.flatnonzero((np.maximum(edges[:-1], edges[1:]) >= lo)
                             & (np.minimum(edges[:-1], edges[1:]) <= hi))
    if not indices.size:
        return slice(0, len(edges) - 1)
    return slice(max(0, int(indices[0]) - 1), min(len(edges) - 1, int(indices[-1]) + 2))


@contextmanager
def visible_drr_meshes(canvas):
    """Keep full model arrays/coordinates and full-resolution export behavior.

    QuadMesh has no viewport-aware draw path in Matplotlib 3.10. Temporarily
    restrict its renderer input to intersecting cells plus a guard cell, then
    restore all arrays/color caches, including when drawing raises. No measured
    cell inside the view is resampled or omitted. Non-affine axes use Matplotlib's
    ordinary path. Saving bypasses this interactive-only optimization.
    """
    saved = []
    try:
        if not canvas.is_saving():
            for axis in canvas.figure.axes:
                if not axis.transData.is_affine:
                    continue
                for mesh in axis.collections:
                    if not getattr(mesh, '_dptk_drr_mesh', False):
                        continue
                    values = mesh.get_array()
                    coords = mesh.get_coordinates()
                    if values is None or values.ndim != 2 or mesh.norm.vmin is None or mesh.norm.vmax is None:
                        continue
                    rows = _cells(coords[:, 0, 1], axis.get_ylim())
                    cols = _cells(coords[0, :, 0], axis.get_xlim())
                    if (rows.stop - rows.start, cols.stop - cols.start) == values.shape:
                        continue
                    names = ('_coordinates', '_A', '_facecolors', '_edgecolors', '_mapped_colors',
                             '_face_is_mapped', '_edge_is_mapped')
                    saved.append((mesh, {name: getattr(mesh, name) for name in names}))
                    mesh._coordinates = coords[rows.start:rows.stop + 1, cols.start:cols.stop + 1]
                    mesh._A = values[rows, cols]
        yield
    finally:
        for mesh, attributes in saved:
            for name, value in attributes.items():
                setattr(mesh, name, value)
