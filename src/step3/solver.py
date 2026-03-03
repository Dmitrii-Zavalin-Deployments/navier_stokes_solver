# src/step3/solver.py

import numpy as np
from scipy.sparse.linalg import cg
from src.solver_state import SolverState

def solve_pressure(state: SolverState) -> str:
    """
    Step 3.2: Pressure Poisson Solve.
    Enforces 'F' order to maintain staggered grid alignment.
    """
    rho = state.density
    dt = state.dt
    
    # 1. Build RHS: b = (rho/dt) * Divergence(V_star)
    # CRITICAL: Must use order='F' to match the MMS test and grid allocation
    v_star_flat = np.concatenate([
        state.fields.U_star.flatten(order='F'), 
        state.fields.V_star.flatten(order='F'), 
        state.fields.W_star.flatten(order='F')
    ])
    
    # Calculate divergence and reshape correctly
    div_v_star = (state.operators.divergence @ v_star_flat).reshape(state.fields.P.shape, order='F')
    rhs = (rho / dt) * div_v_star

    # 2. Linear Solve: AP = b
    # Note: Using state.ppe._A which is populated in Step 2 Orchestration
    p_flat, info = cg(
        state.ppe._A, 
        rhs.flatten(order='F'), 
        x0=state.fields.P.flatten(order='F'),
        rtol=getattr(state.config, "ppe_tolerance", 1e-10),
        atol=getattr(state.config, "ppe_atol", 1e-12),
        maxiter=getattr(state.config, "ppe_max_iter", 1000)
    )
    
    # Update pressure field in-place with correct memory layout
    state.fields.P = p_flat.reshape(state.fields.P.shape, order='F')
    
    return "converged" if info == 0 else "failed"