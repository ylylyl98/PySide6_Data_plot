"""JSON-safe display settings for snapshots of the currently rendered map."""
from dataclasses import asdict

import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm, TwoSlopeNorm


def capture_display(axis):
    params=getattr(axis,'_dptk_heatmap_params',None)
    meshes=[artist for artist in axis.collections if isinstance(artist,QuadMesh)]
    if params is None or not meshes:
        return {}
    first=meshes[0]
    display=dict(cmap=first.cmap.name,vmin=float(first.norm.vmin),vmax=float(first.norm.vmax),
                 log_scale=isinstance(first.norm,LogNorm),center_zero=isinstance(first.norm,TwoSlopeNorm),
                 clip_outliers=bool(params.clip_outliers),split_scale=None)
    if params.split_scale is not None:
        split=asdict(params.split_scale)
        sides=('left','right') if len(meshes)==2 else ('left','middle','right')
        for side,mesh in zip(sides,meshes):
            split[side+'_vmin']=float(mesh.norm.vmin)
            split[side+'_vmax']=float(mesh.norm.vmax)
        edges=first.get_coordinates()[0,:,0]
        for field in ('split_x','split_x2'):
            if split.get(field) is not None:
                split[field]=float(edges[1:-1][np.argmin(abs(edges[1:-1]-split[field]))])
        if edges[0]>edges[-1]:
            for bound in ('vmin','vmax'):
                split['left_'+bound],split['right_'+bound]=split['right_'+bound],split['left_'+bound]
        display['split_scale']=split
        display['center_zero']=any(isinstance(mesh.norm,TwoSlopeNorm) for mesh in meshes)
    return display
