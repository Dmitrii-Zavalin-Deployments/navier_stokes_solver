# src/step3/predictor.py

import numpy as np
from src.solver_state import SolverState

def predict_velocity(state: SolverState) -> None:
    """
    Step 3.1: Prediction. Calculates intermediate velocity V*.
    Rule: Use facades for rho, mu, and dt to match SolverConfig properties.
    """
    rho = state.density
    mu = state.viscosity
    dt = state.dt
    nu = mu / rho

    def _apply(field, operator):
        try:
            if operator is None: return np.zeros_like(field)
            res = operator @ field.ravel()
            return res.reshape(field.shape) if res.size == field.size else np.zeros_like(field)
        except:
            return np.zeros_like(field)

    # Note: advection_u/v/w are not in the frozen schema. 
    # We use getattr to prevent AttributeErrors if they are missing.
    state.fields.U_star = state.fields.U + dt * (
        nu * _apply(state.fields.U, state.operators.laplacian) - 
        _apply(state.fields.U, getattr(state.operators, 'advection_u', None))
    )
    
    state.fields.V_star = state.fields.V + dt * (
        nu * _apply(state.fields.V, state.operators.laplacian) - 
        _apply(state.fields.V, getattr(state.operators, 'advection_v', None))
    )
    
    state.fields.W_star = state.fields.W + dt * (
        nu * _apply(state.fields.W, state.operators.laplacian) - 
        _apply(state.fields.W, getattr(state.operators, 'advection_w', None))
    )
