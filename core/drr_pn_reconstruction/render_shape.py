# Migrated numerical kernel. Version names describe recipe lineage, not stored inputs.
"""Localized aligned-P shape with parent-preserving Hermite increments."""
from math import comb
import numpy as np
from scipy.interpolate import PPoly

FAMILIES=("local_p_region_shape_increment",)
CONSTRUCTION_TYPE="V63_plus_regional_P_shape_and_curvature_minimal_Hermite_increments"


def _runs(mask):
    d=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)-1))


def _shift(p,t0,H,n=7):
    out=np.zeros(n)
    for j,v in enumerate(p):
        for k in range(j+1):
            out[k]+=v*comb(j,k)*t0**(j-k)/H**k
    return out


def _hermite(j0,j1,H):
    z=np.zeros(6);z[:3]=[j0[0],H*j0[1],H*H*j0[2]/2]
    d0=j1[0]-z[:3].sum();d1=H*j1[1]-z[1]-2*z[2];d2=H*H*j1[2]-2*z[2]
    z[3:]=[10*d0-4*d1+d2/2,-15*d0+7*d1-d2,6*d0-3*d1+d2/2]
    return z


def _inner(p,q,dx):
    x=np.polynomial.polynomial.polymul(p,q)
    return float(np.sum(x*dx**np.arange(1,len(x)+1)/np.arange(1,len(x)+1)))


def generate(e, *, baseline, source_index=None, core=None):
    """Pure array construction; one PN-parent-derived regional level, no new source fit."""
    e=np.asarray(e)
    np.testing.assert_array_equal(e,baseline["energy"])
    if e.ndim!=1 or len(e)<4 or not np.all(np.diff(e)>0):
        raise ValueError("Expected full increasing native energy")
    A=np.asarray(baseline["original_aligned_source_log"])
    C=np.asarray(baseline["original_corrected_source_log"])
    q=np.asarray(baseline["alignment_q_log"])
    weights=np.asarray(baseline["source_weights"])
    refw=np.asarray(baseline["reference_source_weights"])
    refB=np.asarray(baseline["baseline_reference_background"])
    np.testing.assert_array_equal(A,C+np.array([-.5,.5])[:,None]*q)
    np.testing.assert_allclose(np.log(refB),np.sum(refw*A,axis=0),rtol=0,atol=2e-13)
    if A.shape!=(2,len(e)) or weights.shape!=A.shape or not np.all(np.isfinite(A)):
        raise ValueError("Invalid source dimensions/data")
    if not np.all((weights>=0)&(weights<=1)):
        raise ValueError("Weights outside bounds")
    np.testing.assert_allclose(weights.sum(0),1,rtol=0,atol=2e-15)
    w=weights[0]
    if source_index is None and core is None:
        # The historical API remains unchanged; generalized callers must
        # explicitly pass their independently selected source and core.
        pure=_runs(w==1)
        if len(pure)!=3:
            raise ValueError("Expected the three frozen pure-P runs; no selection fallback")
        left,right=pure[2]
        source_index=0
    else:
        if source_index not in (0,1) or core is None or len(core)!=2:
            raise ValueError("Explicit P/N source and native pure-core endpoints required")
        left,right=map(int,core)
        if not 0<=left<right<len(e) or not np.all(weights[source_index,left:right+1]==1):
            raise ValueError("Selected core must contain only its declared source")
    mixed=[(l-1,r+1) for l,r in _runs((w>0)&(w<1))]
    lefts=[x for x in mixed if x[1]==left];rights=[x for x in mixed if x[0]==right]
    if len(lefts)>1 or len(rights)>1 or (not lefts and left!=0) or (not rights and right!=len(e)-1):
        raise ValueError("Selected core lacks two neighboring mixed supports")
    lb=lefts[0] if lefts else None
    rb=rights[0] if rights else None
    if lb is None and rb is None:
        raise ValueError("An entire-axis pure core has no bridge-level objective")
    if (lb is not None and lb[0]<1) or (rb is not None and rb[1]>=len(e)-1):
        raise ValueError("Interior local support required")
    bridge_specs=[(side,br) for side,br in enumerate((lb,rb)) if br is not None]
    support_left=lb[0] if lb is not None else 0
    support_right=rb[1] if rb is not None else len(e)-1
    pc=np.asarray(baseline["integrated_log_polynomial_coefficients"])
    sc=np.asarray(baseline["source_log_spline_coefficients"])[:,:,source_index]
    if pc.shape!=(7,len(e)-1):
        raise ValueError("Expected native degree-six V63 parent")
    parent=PPoly(pc,e,extrapolate=False);source=PPoly(sc,e,extrapolate=False)
    np.testing.assert_allclose(source(e),A[source_index],rtol=0,atol=1e-12)
    d0=np.zeros_like(pc[::-1]);mode=np.zeros_like(d0)
    J0=[];J1=[];J2=[];bridge_records=[]
    for side,(l,r) in bridge_specs:
        H=float(e[r]-e[l])
        j0=([0.,0.,0.] if side==0 else [float(source(e[l],nu=k)-parent(e[l],nu=k)) for k in range(3)])
        j1=([float(source(e[r],nu=k)-parent(e[r],nu=k)) for k in range(3)] if side==0 else [0.,0.,0.])
        h0=_hermite(j0,j1,H)
        hm=np.array([0.,0.,0.,10.,-15.,6.]) if side==0 else np.array([1.,0.,0.,-10.,15.,-6.])
        aa=bb=cc=0.
        for i in range(l,r):
            dx=float(e[i+1]-e[i]);t0=float((e[i]-e[l])/H)
            d0[:,i]=_shift(h0,t0,H);mode[:,i]=_shift(hm,t0,H)
            d=d0[2:,i]*np.arange(2,7)*np.arange(1,6)
            m=mode[2:,i]*np.arange(2,7)*np.arange(1,6)
            aa+=_inner(d,d,dx);bb+=_inner(d,m,dx);cc+=_inner(m,m,dx)
        J0.append(aa);J1.append(bb);J2.append(cc)
        bridge_records.append(dict(side="left" if side==0 else "right",indices=[int(l),int(r)],
            energy_bounds=e[[l,r]].tolist(),width=H,increment_Hermite_at_zero_level=h0.tolist(),
            level_response_Hermite=hm.tolist()))
    Q=float(sum(J2));v=float(sum(J1))
    if not Q>0 or not np.isfinite(Q):
        raise ArithmeticError("Regional level objective not strictly convex")
    level=-v/Q
    for i in range(left,right):
        d0[:,i]=-pc[::-1,i];d0[:4,i]+=sc[::-1,i];mode[0,i]=1
    requested_delta=d0+level*mode
    coeff=pc.copy()
    active=np.zeros(len(e)-1,bool);active[support_left:support_right]=True
    coeff[:,active]+=requested_delta[::-1,active]
    # Preserve the exact selected-source polynomial within the pure core.
    coeff[:,left:right]=0
    coeff[-4:,left:right]=sc[:,left:right]
    coeff[-1,left:right]+=level
    delta_coeff=coeff-pc
    poly=PPoly(coeff,e,extrapolate=False)
    delta_poly=PPoly(delta_coeff,e,extrapolate=False)
    support=np.zeros(len(e),bool);support[support_left:support_right+1]=True
    interior=np.zeros(len(e),bool)
    interior[support_left+int(lb is not None):support_right+1-int(rb is not None)]=True
    core=np.zeros(len(e),bool);core[left:right+1]=True
    bridges=np.zeros((len(bridge_specs),len(e)),bool)
    for j,(_side,(l,r)) in enumerate(bridge_specs):bridges[j,l:r+1]=True
    L=np.asarray(baseline["background_log"]).copy();B=np.asarray(baseline["background"]).copy()
    L[interior]=poly(e[interior]);L[core]=A[source_index,core]+level;B[interior]=np.exp(L[interior])
    if not np.all(np.isfinite(B)&(B>0)):
        raise ArithmeticError("Invalid candidate values")
    deltaL=L-baseline["background_log"];cell=np.gradient(e)
    joins=[]
    for nu in range(3):
        pp=poly.derivative(nu)
        end=np.polynomial.polynomial.polyval(np.diff(e),pp.c[::-1],tensor=False)
        joins.append(float(np.max(abs(end[:-1]-pp.c[-1,1:]))))
    native_error=float(np.max(abs(poly(e)-L)))
    if native_error>1e-10 or joins[0]>1e-10 or joins[1]>1e-8 or joins[2]>1e-6:
        raise ArithmeticError("Native/C2 reconstruction failed")
    for j,br in enumerate(bridge_records):
        J=J0[j]+2*level*J1[j]+level*level*J2[j]
        br.update(integrated_squared_curvature_change=float(J),
            RMS_curvature_change=float(np.sqrt(max(0,J)/br["width"])))
    copied=("original_corrected_source_log","original_aligned_source_log","alignment_q_log","state",
        "source_usable_masks","hard_lock_mask","raw_geometric_mean","source_difference_log",
        "baseline_reference_background","reference_source_weights","source_log_spline_coefficients",
        "cubic_anchor_coefficients","anchor_coordinate_center","anchor_coordinate_scale",
        "anchor_log_curve","anchor_log_curvature")
    out={k:np.asarray(baseline[k]).copy() for k in copied if k in baseline}
    out.update(energy=e.copy(),background=B,background_log=L,construction_type=np.array(CONSTRUCTION_TYPE),
        seams=np.array([],float),baseline_background=refB.copy(),
        parent_background=np.asarray(baseline["background"]).copy(),
        parent_background_log=np.asarray(baseline["background_log"]).copy(),
        parent_source_weights=weights.copy(),source_weights=weights.copy(),effective_source_weights=weights.copy(),
        regional_log_constant=np.array(level),regional_source_index=np.array(source_index),
        level_objective_coefficients=np.array([sum(J0),2*v,Q]),
        level_objective_stationarity=np.array(Q*level+v),
        core_endpoint_indices=np.array([left,right]),support_endpoint_indices=np.array([[support_left,support_right]]),
        bridge_endpoint_indices=np.array([br for _,br in bridge_specs]),
        bridge_sides=np.array([side for side,_ in bridge_specs],dtype=np.int8),
        local_support_mask=support,local_interior_mask=interior,
        local_core_mask=core,local_bridge_masks=bridges,modified_polynomial_cell_mask=active,
        actual_change_mask=B!=baseline["background"],modification_mask=B!=baseline["background"],
        repair_mask=B!=baseline["background"],uncertain_mask=np.asarray(baseline["state"])==0,
        delta_background_log=deltaL,background_delta_log=L-np.log(refB),
        repair_log_delta=L-np.log(baseline["raw_geometric_mean"]),
        polynomial_knots=e.copy(),integrated_log_polynomial_coefficients=coeff,
        parent_integrated_log_polynomial_coefficients=pc.copy(),
        delta_integrated_log_polynomial_coefficients=delta_coeff,
        zero_level_increment_polynomial_coefficients=d0[::-1],
        level_response_polynomial_coefficients=mode[::-1],
        native_cell_weights=cell,stitched_log_first_derivative=poly(e,nu=1),
        stitched_log_second_derivative=poly(e,nu=2),delta_log_first_derivative=delta_poly(e,nu=1),
        delta_log_second_derivative=delta_poly(e,nu=2))
    diag=dict(family=FAMILIES[0],construction_type=CONSTRUCTION_TYPE,status="actual_local_P_shape_trial_not_accepted",
        scientific_acceptance=False,BN_used=False,new_sample_read=False,no_new_source_fit=True,
        fitted_one_regional_log_level=True,global_adjustment=False,inherited_global_cubic_not_refit=True,
        source_C_A_q_state_weights_frozen=True,original_V50_reference_preserved_not_reimposed=True,
        core_indices=[int(left),int(right)],core_energy_bounds=e[[left,right]].tolist(),
        bridge_records=bridge_records,optimal_regional_log_constant=float(level),
        level_objective_Q=Q,level_objective_linear_half=v,level_stationarity_residual=float(Q*level+v),
        weighted_mean_log_change=float(cell@deltaL/cell.sum()),
        delta_log_range=[float(deltaL.min()),float(deltaL.max())],native_PPoly_log_error=native_error,
        C0_C1_C2_join_max=joins,
        exterior_background_bitwise_preserved=bool(np.array_equal(B[~interior],baseline["background"][~interior])),
        exterior_coefficients_bitwise_preserved=bool(np.array_equal(coeff[:,~active],pc[:,~active])),
        selected_source="P" if source_index==0 else "N",
        boundary_core=lb is None or rb is None,
        objective_bridge_count=len(bridge_specs),
        core_source_log_shape_error=float(np.max(abs(L[core]-A[source_index,core]-level))),
        limitations=[
            "The aligned P source is an inherited conditional model, not proven clean.",
            "The scalar objective minimizes added bridge log-curvature, not actual dRR/d2 error.",
            "Quintic increments preserve parent bridge details but still add curvature and can alter weak structures.",
            "Only the declared third P region and neighboring bridges change; no global mean/level correction is applied.",
            "Frozen source_weights indicate source selection, not a pointwise value-mixture formula for this output."])
    return out,diag
