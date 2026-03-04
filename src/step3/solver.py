# src/step3/solver.py

import numpy as np
from scipy.sparse.linalg import cg
from src.solver_state import SolverState

# Global Debug Toggle
DEBUG = True

def solve_pressure(state: SolverState) -> str:
    """
    Step 3.2: Pressure Poisson Solve.
    Anchors the null space using configured pressure at the first valid fluid cell.
    Rule 5 Compliance: No hardcoded indices or values. Uses mask and config.
    """
    rho = state.density
    dt = state.dt
    
    if DEBUG:
        print(f"DEBUG [Step 3 Solver]: Starting PPE solve. rho={rho}, dt={dt}")

    # 1. Build RHS: b = (rho/dt) * Divergence(V_star)
    v_star_flat = np.concatenate([
        state.fields.U_star.flatten(order='F'), 
        state.fields.V_star.flatten(order='F'), 
        state.fields.W_star.flatten(order='F')
    ])
    
    divergence = getattr(state.operators, "divergence", getattr(state.operators, "_divergence", None))
    if divergence is None: raise RuntimeError("Access Error: divergence is uninitialized.")
    rhs = (rho / dt) * (divergence @ v_star_flat)

    # 2. DYNAMIC ANCHORING (No hardcoded index 0)
    # Find the first fluid cell index using the mask (1 = fluid)
    # We flatten the mask in 'F' order to match the operator indexing
    mask = getattr(state, "mask", getattr(state, "_mask", None))
    if mask is None: raise RuntimeError("Access Error: Mask is uninitialized.")
    mask_flat = mask.flatten(order="F")
    fluid_indices = np.where(mask_flat == 1)[0]
    
    if len(fluid_indices) == 0:
        if DEBUG: print("!!! CRITICAL: No fluid cells found in mask !!!")
        return "failed"
    
    anchor_idx = fluid_indices[0] # Use the first real fluid cell found
    ref_p = state.config.initial_pressure 
    
    ppe_matrix = getattr(state.ppe, "_A", None)
    if ppe_matrix is None: raise RuntimeError("Access Error: PPE matrix _A is uninitialized.")
    A_pinned = ppe_matrix.copy()
    
    # Identify the row in the CSR matrix for the anchor index
    r_start, r_end = A_pinned.indptr[anchor_idx], A_pinned.indptr[anchor_idx + 1]
    
    # Zero out the row and set diagonal to 1.0 (Direct Dirichlet constraint)
    A_pinned.data[r_start:r_end] = 0.0
    A_pinned[anchor_idx, anchor_idx] = 1.0
    rhs[anchor_idx] = ref_p

    if DEBUG:
        print(f"DEBUG [Step 3 Solver]: Pressure anchored at Index {anchor_idx} (Fluid) with P={ref_p}")

    # 3. Linear Solve: AP = b
    p_flat, info = cg(
        A_pinned, 
        rhs, 
        x0=state.fields.P.flatten(order='F'),
        rtol=state.config.ppe_tolerance,
        atol=state.config.ppe_atol,
        maxiter=state.config.ppe_max_iter
    )
    
    if DEBUG:
        res_norm = np.linalg.norm(rhs - A_pinned @ p_flat)
        print(f"DEBUG [Step 3 Solver]: CG status info: {info}, Res Norm: {res_norm:.6e}")

    # 4. Map Back to Grid
    state.fields.P = p_flat.reshape(state.fields.P.shape, order='F')
    
    status = "converged" if info == 0 else "failed"
    return status