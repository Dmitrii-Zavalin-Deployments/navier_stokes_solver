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

    def _apply(field, operator):
        try:
            if operator is None: return np.zeros_like(field)
            res = operator @ field.ravel()
            return res.reshape(field.shape) if res.size == field.size else np.zeros_like(field)
        except:
            return np.zeros_like(field)

    # Subtract pressure gradient from intermediate velocity
    state.fields.U = state.fields.U_star - coeff * _apply(state.fields.P, state.operators.grad_x)
    state.fields.V = state.fields.V_star - coeff * _apply(state.fields.P, state.operators.grad_y)
    state.fields.W = state.fields.W_star - coeff * _apply(state.fields.P, state.operators.grad_z)

    # Update Health Context
    state.health.max_u = float(max(np.max(np.abs(state.fields.U)), 
                                   np.max(np.abs(state.fields.V)), 
                                   np.max(np.abs(state.fields.W))))
    
    v_new_flat = np.concatenate([state.fields.U.ravel(), state.fields.V.ravel(), state.fields.W.ravel()])
    div_new = state.operators.divergence @ v_new_flat
    state.health.divergence_norm = float(np.linalg.norm(div_new) / div_new.size)
    state.health.post_correction_divergence_norm = state.health.divergence_norm
