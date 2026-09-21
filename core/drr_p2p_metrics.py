"""P2P observables reuse the extrema already computed in each row."""
import numpy as np

METRICS={'amplitude':('P2P (max − min)',1,1.),'maximum':('E(max) (eV)',5,1.),
         'minimum':('E(min) (eV)',4,1.),'separation':('ΔE (meV)',None,1000.)}

def metric_values(result,metric):
    _,column,scale=METRICS[metric]
    if column is None:return abs(result[:,5]-result[:,4])*scale
    return result[:,column]*scale

def quantity_metrics(quantity):
    return ('maximum','minimum') if quantity=='positions' else (quantity,)
