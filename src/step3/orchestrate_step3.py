# src/step3/orchestrate_step3.py

import numpy as np
from src.step3.predictor import compute_predictor_step
from src.step3.ppe_solver import solve_pressure_poisson
from src.step3.corrector import apply_velocity_correction

def orchestrate_step3(state):
    """
    Step 3 Orchestrator: Complete projection method pipeline.
    """
    # 1. Hydration
    dx, dy, dz = state.grid.dx, state.grid.dy, state.grid.dz
    dt = state.config.simulation_parameters["time_step"]
    rho = state.config.fluid_properties["density"]
    mu = state.config.fluid_properties["viscosity"]
    F_vals = tuple(state.config.external_forces["force_vector"])
    
    v_n = np.stack([state.fields.U, state.fields.V, state.fields.W])
    p_n = state.fields.P
    
    # 2. PREDICT: Calculate intermediate V*
    state.fields.v_star = compute_predictor_step(v_n, p_n, dx, dy, dz, dt, rho, mu, F_vals)
    
    # 3. SOLVE: Solve PPE for p^{n+1}
    # solve_pressure_poisson updates state.fields.P internally
    solve_pressure_poisson(state)
    
    # 4. CORRECT: Project v* onto divergence-free space
    # state.fields.v_next = v^{n+1}
    state.fields.v_next = apply_velocity_correction(
        state.fields.v_star, 
        state.fields.P, 
        dx, dy, dz, dt, rho
    )
    
    return state