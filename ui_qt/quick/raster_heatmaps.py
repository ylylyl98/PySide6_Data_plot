"""Fast, cell-based screen painting for large rectilinear Quick heatmaps.

The complete QuadMesh stays the source for values, cursor readout and exports.
PcolorImage maps the original cell boundaries directly to screen pixels; it
does not interpolate intensities or build a reduced scientific data array.
"""
from contextlib import contextmanager


def _eligible(mesh, axis):
    import numpy as np
    from matplotlib.collections import QuadMesh
    if (type(mesh) is not QuadMesh or "draw" in mesh.__dict__ or not mesh.get_visible()
            or mesh._shading != "flat" or not axis.transData.is_affine
            or mesh.get_transform() != axis.transData
            or not mesh.get_clip_on() or np.ndim(mesh.get_alpha()) > 0
            or mesh.get_hatch() or len(mesh.get_edgecolors())
            or mesh.get_agg_filter() is not None
            or isinstance(mesh._original_facecolor, str) and mesh._original_facecolor == "none"
            or mesh.get_path_effects() or mesh.get_clip_path() is not None
            or np.any(mesh.get_offsets())):
        return False
    values = mesh.get_array()
    coordinates = mesh.get_coordinates()
    if values is None or values.ndim != 2 or values.size < 100_000:
        return False
    x, y = coordinates[0, :, 0], coordinates[:, 0, 1]
    return (coordinates.shape == (*np.add(values.shape, 1), 2)
            and np.isfinite(x).all() and np.isfinite(y).all()
            and (np.all(np.diff(x)>0) or np.all(np.diff(x)<0))
            and (np.all(np.diff(y)>0) or np.all(np.diff(y)<0))
            and np.all(coordinates[:, :, 0] == x[None, :])
            and np.all(coordinates[:, :, 1] == y[:, None]))


def _draw_cells(mesh, fallback, renderer):
    if mesh.figure.canvas.is_saving():
        # Even a synchronous save triggered inside a screen draw context must
        # use the original full-resolution export renderer.
        return fallback(renderer)
    from matplotlib.image import PcolorImage
    coordinates = mesh.get_coordinates()
    image = PcolorImage(mesh.axes, coordinates[0, :, 0], coordinates[:, 0, 1],
                        mesh.get_array(), norm=mesh.norm, cmap=mesh.get_cmap(),
                        alpha=mesh.get_alpha())
    image.set_figure(mesh.figure)
    image.set_clip_on(mesh.get_clip_on())
    image.set_clip_box(mesh.get_clip_box())
    image.draw(renderer)


@contextmanager
def screen_heatmaps(canvas):
    """Temporarily accelerate only large, plain, linear-axis screen meshes."""
    from functools import partial
    originals = []
    try:
        if not canvas.is_saving():
            for axis in canvas.figure.axes:
                for mesh in axis.collections:
                    if _eligible(mesh, axis):
                        originals.append((mesh, mesh.__dict__.get("draw")))
                        mesh.draw = partial(_draw_cells, mesh, mesh.draw)
        yield
    finally:
        for mesh, previous in originals:
            if previous is None:
                del mesh.draw
            else:
                mesh.draw = previous
