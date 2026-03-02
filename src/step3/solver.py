# src/step3/solver.py

import numpy as np
from scipy.sparse.linalg import cg
from src.solver_state import SolverState

def solve_pressure(state: SolverState) -> str:
    """
    Step 3.2: Pressure Poisson Solve.
    Rule 1: Scale Guard. Matrix A and Divergence are strictly scipy.sparse.
    """
    # Accessing via the facade properties fixed in SolverConfig
    rho = state.density
    dt = state.dt
    
    # 1. Build RHS: b = (rho/dt) * Divergence(V_star)
    # Concatenate staggered components for the vector-multiplication
    v_star_flat = np.concatenate([
        state.fields.U_star.ravel(), 
        state.fields.V_star.ravel(), 
        state.fields.W_star.ravel()
    ])
    
    # Using the standardized divergence operator
    div_v_star = (state.operators.divergence @ v_star_flat).reshape(state.fields.P.shape)
    rhs = (rho / dt) * div_v_star

    # 2. Linear Solve: AP = b using Preconditioned Conjugate Gradient
    # state.ppe._A is the internal storage used in dummies
    p_flat, info = cg(
        state.ppe._A, 
        rhs.ravel(), 
        x0=state.fields.P.ravel(),
        tol=state.config.simulation_parameters.get("ppe_tolerance", 1e-6),
        atol=state.config.simulation_parameters.get("ppe_atol", 1e-8),
        maxiter=state.config.simulation_parameters.get("ppe_max_iter", 1000)
    )
    
    state.fields.P = p_flat.reshape(state.fields.P.shape)
    
    return "converged" if info == 0 else "failed"
