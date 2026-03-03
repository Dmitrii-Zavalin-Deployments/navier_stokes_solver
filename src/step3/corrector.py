# src/step3/corrector.py

import numpy as np
from src.solver_state import SolverState

def correct_velocity(state: SolverState) -> None:
    """
    Step 3.3: Projection/Correction.
    V_new = V* - (dt/rho) * grad(P)
    
    Rule 5 Compliance: Explicit calculation, no silent failures for missing operators.
    """
    rho = state.density
    dt = state.dt
    coeff = dt / rho

    # Ensure pressure is flattened in Fortran order for the matrix multiplication
    p_flat = state.fields.P.flatten(order='F')

    # 1. APPLY CORRECTION
    # We apply the gradient of pressure to the predicted (star) velocities.
    # We use the dot product directly to ensure the resulting vector 
    # matches the flattened dimensions of the velocity fields.
    
    # Update U
    grad_p_x = (state.operators.grad_x @ p_flat).reshape(state.fields.U.shape, order='F')
    state.fields.U = state.fields.U_star - coeff * grad_p_x

    # Update V
    grad_p_y = (state.operators.grad_y @ p_flat).reshape(state.fields.V.shape, order='F')
    state.fields.V = state.fields.V_star - coeff * grad_p_y

    # Update W
    grad_p_z = (state.operators.grad_z @ p_flat).reshape(state.fields.W.shape, order='F')
    state.fields.W = state.fields.W_star - coeff * grad_p_z

    # 2. UPDATE HEALTH VITALS
    # Reconstruct the global velocity vector in Fortran order for divergence check
    v_new_flat = np.concatenate([
        state.fields.U.flatten(order='F'), 
        state.fields.V.flatten(order='F'), 
        state.fields.W.flatten(order='F')
    ])
    
    # Calculate residual divergence
    div_new = state.operators.divergence @ v_new_flat
    
    # Sync health metrics to state
    state.health.divergence_norm = float(np.linalg.norm(div_new, np.inf))
    state.health.post_correction_divergence_norm = state.health.divergence_norm
    
    state.health.max_u = float(max(
        np.max(np.abs(state.fields.U)), 
        np.max(np.abs(state.fields.V)), 
        np.max(np.abs(state.fields.W))
    ))