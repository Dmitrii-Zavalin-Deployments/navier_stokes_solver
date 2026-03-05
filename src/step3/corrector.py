# src/step3/corrector.py

import numpy as np
from src.solver_state import SolverState

# Global Debug Toggle
DEBUG = True

def correct_velocity(state: SolverState) -> None:
    """
    Step 3.3: Projection/Correction.
    V_new = V* - (dt/rho) * grad(P)
    """
    # These use the Facade properties in SolverState (self.config.dt / self.config.density)
    rho = state.density
    dt = state.dt
    coeff = dt / rho

    if DEBUG:
        print(f"DEBUG [Step 3 Corrector]: Applying correction with coeff (dt/rho)={coeff:.6e}")

    # Rule 5: Fortran-style flattening for matrix-vector compatibility
    p_flat = state.fields.P.flatten(order='F')

    # 1. APPLY CORRECTION
    # grad_p components reshaped back to staggered grid in 'F' order
    
    # Update U (Uses state.operators.grad_x)
    grad_p_x_flat = state.operators.grad_x @ p_flat
    state.fields.U = state.fields.U_star - coeff * grad_p_x_flat.reshape(state.fields.U.shape, order='F')

    # Update V
    grad_p_y_flat = state.operators.grad_y @ p_flat
    state.fields.V = state.fields.V_star - coeff * grad_p_y_flat.reshape(state.fields.V.shape, order='F')

    # Update W
    grad_p_z_flat = state.operators.grad_z @ p_flat
    state.fields.W = state.fields.W_star - coeff * grad_p_z_flat.reshape(state.fields.W.shape, order='F')

    if DEBUG:
        max_grad = max(np.max(np.abs(grad_p_x_flat)), 
                       np.max(np.abs(grad_p_y_flat)), 
                       np.max(np.abs(grad_p_z_flat)))
        print(f"DEBUG [Step 3 Corrector]: Max Grad P: {max_grad:.4e}")

    # 2. UPDATE HEALTH VITALS
    v_new_flat = np.concatenate([
        state.fields.U.flatten(order='F'), 
        state.fields.V.flatten(order='F'), 
        state.fields.W.flatten(order='F')
    ])
    
    div_new = state.operators.divergence @ v_new_flat
    
    # Update health metrics via the ValidatedContainer setters
    state.health.divergence_norm = float(np.linalg.norm(div_new, np.inf))
    state.health.post_correction_divergence_norm = state.health.divergence_norm
    
    state.health.max_u = float(max(
        np.max(np.abs(state.fields.U)), 
        np.max(np.abs(state.fields.V)), 
        np.max(np.abs(state.fields.W))
    ))

    if DEBUG:
        print(f"DEBUG [Step 3 Corrector]: Final Div Norm (Inf): {state.health.divergence_norm:.6e}")
        print(f"DEBUG [Step 3 Corrector]: Max Velocity: {state.health.max_u:.4f}")
        if state.health.divergence_norm > 1e-10:
             print("!!! WARNING: Divergence above logic gate threshold !!!")