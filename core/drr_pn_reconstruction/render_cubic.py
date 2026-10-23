# Migrated numerical kernel. Version names describe recipe lineage, not stored inputs.
"""Frozen V53 curvature integral with a single global cubic log anchor."""
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, PPoly

FAMILIES = ("curvature_stitched_cubic_anchor",)
CONSTRUCTION_TYPE = "twice_integrated_log_source_curvature_with_global_cubic_anchor"


def generate(e, *, baseline):
    """Pure arrays; source model unchanged, exactly four global anchor coefficients."""
    e = np.asarray(e)
    if e.ndim != 1 or len(e) < 4 or not np.isfinite(e).all() or not np.all(np.diff(e) > 0):
        raise ValueError("Expected full increasing native energy")
    np.testing.assert_array_equal(e, baseline["energy"])
    A = np.asarray(baseline["original_aligned_source_log"])
    C = np.asarray(baseline["original_corrected_source_log"])
    q = np.asarray(baseline["alignment_q_log"])
    weights = np.asarray(baseline["source_weights"])
    oldB = np.asarray(baseline["background"])
    if A.shape != (2,len(e)) or weights.shape != A.shape:
        raise ValueError("Invalid source arrays")
    np.testing.assert_array_equal(A, C+np.array([-.5,.5])[:,None]*q)
    np.testing.assert_allclose(weights.sum(0),1,rtol=0,atol=2e-15)
    if not np.isfinite(A).all() or not np.all((weights>=0)&(weights<=1)):
        raise ValueError("Invalid source/weights")
    if not np.all(np.isfinite(oldB)&(oldB>0)):
        raise ValueError("Invalid reference background")
    L0=np.log(oldB)
    np.testing.assert_allclose(L0,np.sum(weights*A,axis=0),rtol=0,atol=2e-13)
    source=CubicSpline(e,A.T,axis=0,bc_type="not-a-knot",extrapolate=False)
    W=PchipInterpolator(e,weights[0],extrapolate=False)
    # Identical V53 selected curvature and unanchored primitives.
    dd=np.stack([2*source.c[1],6*source.c[0]],axis=0)
    diff=dd[:,:,0]-dd[:,:,1]
    wa=W.c[::-1]
    ha=np.zeros((5,len(e)-1))
    for i in range(4):
        for j in range(2):
            ha[i+j]+=wa[i]*diff[j]
    ha[:2]+=dd[:,:,1]
    dx=np.diff(e)
    orders=np.arange(5)[:,None]
    first_increment=np.sum(ha*dx[None]**(orders+1)/(orders+1),axis=0)
    second_moment=np.sum(ha*dx[None]**(orders+2)/((orders+1)*(orders+2)),axis=0)
    slope=np.r_[0.,np.cumsum(first_increment)]
    log_increment=slope[:-1]*dx+second_moment
    rawlog=np.r_[0.,np.cumsum(log_increment)]
    cell=np.gradient(e)
    center=float((e[0]+e[-1])/2)
    scale=float(np.ptp(e)/2)
    u=(e-center)/scale
    X=np.polynomial.polynomial.polyvander(u,3)
    anchor,_,rank,sv=np.linalg.lstsq(np.sqrt(cell)[:,None]*X,
        np.sqrt(cell)*(L0-rawlog),rcond=None)
    if rank!=4:
        raise ArithmeticError("Cubic anchor design is not full rank")
    a0,a1,a2,a3=anchor
    ul=u[:-1]
    asc=np.zeros((7,len(dx)))
    asc[2:]=ha/((orders+1)*(orders+2))
    asc[0]=rawlog[:-1]+np.polynomial.polynomial.polyval(ul,anchor)
    asc[1]=slope[:-1]+(a1+2*a2*ul+3*a3*ul**2)/scale
    asc[2]+=(a2+3*a3*ul)/scale**2
    asc[3]+=a3/scale**3
    coeff=asc[::-1]
    poly=PPoly(coeff,e,extrapolate=False)
    L=poly(e)
    B=np.exp(L)
    if not np.all(np.isfinite(B)&(B>0)):
        raise ArithmeticError("Nonfinite/nonpositive candidate")
    anchor_curve=X@anchor
    anchor_curvature=(2*a2+6*a3*u)/scale**2
    finalha=ha.copy()
    finalha[0]+=(2*a2+6*a3*ul)/scale**2
    finalha[1]+=6*a3/scale**3
    directh=weights[0]*source(e,nu=2)[:,0]+weights[1]*source(e,nu=2)[:,1]
    end=np.polynomial.polynomial.polyval(dx,asc,tensor=False)
    end1=np.polynomial.polynomial.polyval(dx,np.arange(1,7)[:,None]*asc[1:],tensor=False)
    end2=np.polynomial.polynomial.polyval(dx,(np.arange(2,7)*np.arange(1,6))[:,None]*asc[2:],tensor=False)
    c0=end[:-1]-asc[0,1:]
    c1=end1[:-1]-asc[1,1:]
    c2=end2[:-1]-2*asc[2,1:]
    normal=X.T@(cell*(L-L0))
    tests=dict(
        global_anchor_polynomial_error=float(np.max(abs(L-(rawlog+anchor_curve)))),
        selected_h_plus_anchor_curvature_error=float(np.max(abs(poly(e,nu=2)-directh-anchor_curvature))),
        C0_join_max=float(np.max(abs(c0))),C1_join_max=float(np.max(abs(c1))),
        C2_join_max=float(np.max(abs(c2))),
        weighted_normal_equation_max=float(np.max(abs(normal))/cell.sum()))
    if max(tests["global_anchor_polynomial_error"],tests["C0_join_max"],
           tests["weighted_normal_equation_max"])>1e-10:
        raise ArithmeticError("Cubic anchor/continuity check failed")
    if tests["C1_join_max"]>1e-8 or max(tests["C2_join_max"],tests["selected_h_plus_anchor_curvature_error"])>1e-6:
        raise ArithmeticError("Derivative closure failed")
    copykeys=("original_corrected_source_log","original_aligned_source_log","alignment_q_log",
        "state","source_usable_masks","hard_lock_mask","raw_geometric_mean","source_difference_log")
    arrays={k:np.asarray(baseline[k]).copy() for k in copykeys if k in baseline}
    delta=L-L0
    arrays.update(energy=e.copy(),background=B,baseline_background=oldB.copy(),
        construction_type=np.array(CONSTRUCTION_TYPE),seams=np.array([],float),
        source_weights=weights.copy(),source_curvature_weights=weights.copy(),
        background_log=L,parent_background_log=L0,background_delta_log=delta,
        repair_log_delta=L-np.log(baseline["raw_geometric_mean"]),repair_mask=B!=oldB,
        modification_mask=B!=oldB,uncertain_mask=np.asarray(baseline["state"])==0,
        source_log_spline_coefficients=source.c.copy(),weight_pchip_coefficients=W.c.copy(),
        polynomial_knots=e.copy(),curvature_polynomial_coefficients=ha[::-1].copy(),
        final_curvature_polynomial_coefficients=finalha[::-1].copy(),
        integrated_log_polynomial_coefficients=coeff,
        first_integral_increment=first_increment,second_integral_moment=second_moment,
        interval_log_integral=log_increment,unanchored_slope_at_knots=slope,
        unanchored_log_at_knots=rawlog,cubic_anchor_coefficients=anchor,
        anchor_coordinate_center=np.array(center),anchor_coordinate_scale=np.array(scale),
        anchor_log_curve=anchor_curve,anchor_log_curvature=anchor_curvature,native_cell_weights=cell,
        stitched_log_first_derivative=poly(e,nu=1),stitched_log_second_derivative=poly(e,nu=2),
        C0_join_error=c0,C1_join_error=c1,C2_join_error=c2)
    diag=dict(family=FAMILIES[0],construction_type=CONSTRUCTION_TYPE,
        status="actual_cubic_anchor_trial_not_accepted",scientific_acceptance=False,
        BN_used=False,new_sample_read=False,no_new_source_fit=True,fitted_global_cubic_log_anchor=True,
        anchor_degree=3,cubic_anchor_coefficients=anchor.tolist(),
        anchor_coordinate_center=center,anchor_coordinate_scale=scale,
        anchor_rank=int(rank),weighted_anchor_condition=float(sv[0]/sv[-1]),
        anchor_singular_values=sv.tolist(),source_C_A_q_weights_states_frozen=True,
        selected_h_unchanged=True,pure_source_curvature_preserved=False,
        pointwise_PN_envelope_preserved=False,
        source_weights_interpretation="Selectors for the unanchored source curvature. Final curvature also includes one global linear-in-E anchor curvature.",
        verification=tests,weighted_background_delta_log_RMS=float(np.sqrt(cell@(delta*delta)/cell.sum())),
        relative_background_change_min=float(np.expm1(delta).min()),
        relative_background_change_max=float(np.expm1(delta).max()),
        added_anchor_log_curvature_range=[float(anchor_curvature.min()),float(anchor_curvature.max())],
        limits="Cubic PN gauge is not physical truth. Full dRR/SG11/SG21 still decides practical quality.")
    return arrays,diag
