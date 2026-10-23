"""Measured-energy operators retained from the adopted regional recipe."""
import numpy as np
from scipy.sparse import csr_matrix
from .source_runtime import check_cancelled

def local_quadratic(e, values, width):
    """Exact V31 full-window/minimum7 measured-coordinate definition."""
    e=np.asarray(e); values=np.asarray(values); n=len(e)
    left=np.searchsorted(e,e-width/2)
    right=np.searchsorted(e,e+width/2,side="right")
    narrow=right-left<7
    left[narrow]=np.clip(np.arange(n)[narrow]-3,0,n-7)
    right[narrow]=left[narrow]+7
    offsets=np.r_[0,np.cumsum(right-left)]
    weights=np.empty(offsets[-1]); columns=np.empty(offsets[-1],dtype=np.int32)
    for i,(a,b) in enumerate(zip(left,right)):
        check_cancelled()
        x=(e[a:b]-e[i])/(width/2)
        design=np.column_stack((np.ones(len(x)),x,x*x))
        weights[offsets[i]:offsets[i+1]]=np.linalg.lstsq(design.T,[1.,0.,0.],rcond=None)[0]
        columns[offsets[i]:offsets[i+1]]=np.arange(a,b)
    projection=csr_matrix((weights,columns,offsets),shape=(n,n))
    shape=values.shape
    return (projection@values.reshape(-1,n).T).T.reshape(shape)


def edges(e):
    return np.r_[e[0], (e[:-1]+e[1:])/2, e[-1]]


def runs(mask):
    change=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8))
    return list(zip(np.flatnonzero(change==1), np.flatnonzero(change==-1)))


def smoothstep(t):
    t=np.clip(np.asarray(t,float),0,1)
    u=np.minimum(t,1-t)
    s=u**3*(10+u*(-15+6*u))
    return np.where(t<=.5,s,1-s)
