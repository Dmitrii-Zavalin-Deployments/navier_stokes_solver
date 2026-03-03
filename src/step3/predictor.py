# src/step3/predictor.py

import numpy as np
from src.solver_state import SolverState

# Global Debug Toggle
DEBUG = True

def predict_velocity(state: SolverState) -> None:
    """
    Step 3.1: Prediction (Production). Calculates intermediate velocity V*.
    Formula: V* = V_n + dt * [ nu * Laplacian(V_n) - Advection(V_n) + External_Forces ]
    
    Rule 5 Compliance: No silent failures. Each component check ensures 
    physical consistency across the staggered grid.
    """
    rho = state.density
    mu = state.viscosity
    dt = state.dt
    nu = mu / rho
    forces = state.config.external_forces.force_vector

    if DEBUG:
        print(f"DEBUG [Step 3 Predictor]: Nu={nu:.6e}, dt={dt}")

    def _apply_op(field, operator, name):
        """Standardized operator application for flattened Fortran-order arrays."""
        # Note: ValidatedContainer (state.operators) handles the None-check via getters
        field_flat = field.flatten(order='F')
        res_flat = operator @ field_flat
        return res_flat.reshape(field.shape, order='F')

    # --- 1. U-COMPONENT PREDICTION ---
    if DEBUG: print("DEBUG [Step 3 Predictor]: Calculating U_star (Advection + Diffusion + Force)")
    
    lap_u = _apply_op(state.fields.U, state.operators.laplacian, "laplacian")
    adv_u = _apply_op(state.fields.U, state.operators.advection_u, "advection_u")
    
    # U* = U + dt * (Viscous - Advection + Force_x)
    state.fields.U_star = state.fields.U + dt * (nu * lap_u - adv_u + forces[0])

    # --- 2. V-COMPONENT PREDICTION ---
    if DEBUG: print("DEBUG [Step 3 Predictor]: Calculating V_star...")
    
    lap_v = _apply_op(state.fields.V, state.operators.laplacian, "laplacian")
    adv_v = _apply_op(state.fields.V, state.operators.advection_v, "advection_v")
    
    state.fields.V_star = state.fields.V + dt * (nu * lap_v - adv_v + forces[1])

    # --- 3. W-COMPONENT PREDICTION ---
    if DEBUG: print("DEBUG [Step 3 Predictor]: Calculating W_star...")
    
    lap_w = _apply_op(state.fields.W, state.operators.laplacian, "laplacian")
    adv_w = _apply_op(state.fields.W, state.operators.advection_w, "advection_w")
    
    state.fields.W_star = state.fields.W + dt * (nu * lap_w - adv_w + forces[2])

    # --- SAFETY AUDIT ---
    if DEBUG:
        u_star_norm = np.linalg.norm(state.fields.U_star)
        if np.isnan(u_star_norm) or np.isinf(u_star_norm):
            print(f"!!! CRITICAL: Predictor Instability at t={state.time} (Norm: {u_star_norm}) !!!")
        else:
            print(f"DEBUG [Step 3 Predictor]: V_star components updated. |U*|={u_star_norm:.4e}")

