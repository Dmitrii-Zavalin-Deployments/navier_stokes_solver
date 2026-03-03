# src/step3/corrector.py

import numpy as np
from src.solver_state import SolverState

def correct_velocity(state: SolverState) -> None:
    """
    Step 3.3: Projection/Correction.
    V_new = V* - (dt/rho) * grad(P)
    """
    rho = state.density
    dt = state.dt
    coeff = dt / rho

    def _apply_gradient(p_field, grad_op, target_shape):
        if grad_op is None or grad_op.shape[0] == 0:
            return np.zeros(target_shape)
        # Apply gradient operator to flattened pressure
        grad_flat = grad_op @ p_field.flatten(order='F')
        return grad_flat.reshape(target_shape, order='F')

    # Subtract pressure gradient (Projection Step)
    state.fields.U = state.fields.U_star - coeff * _apply_gradient(state.fields.P, state.operators.grad_x, state.fields.U.shape)
    state.fields.V = state.fields.V_star - coeff * _apply_gradient(state.fields.P, state.operators.grad_y, state.fields.V.shape)
    state.fields.W = state.fields.W_star - coeff * _apply_gradient(state.fields.P, state.operators.grad_z, state.fields.W.shape)

    # Update Health Context with the new divergence-free velocity
    v_new_flat = np.concatenate([
        state.fields.U.flatten(order='F'), 
        state.fields.V.flatten(order='F'), 
        state.fields.W.flatten(order='F')
    ])
    
    div_new = state.operators.divergence @ v_new_flat
    
    # Compute L-infinity norm for the MMS Gate requirement
    state.health.divergence_norm = float(np.linalg.norm(div_new, np.inf))
    state.health.max_u = float(max(
        np.max(np.abs(state.fields.U)), 
        np.max(np.abs(state.fields.V)), 
        np.max(np.abs(state.fields.W))
    ))
    state.health.post_correction_divergence_norm = state.health.divergence_norm