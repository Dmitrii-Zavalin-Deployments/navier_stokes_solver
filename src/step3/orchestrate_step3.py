# src/step3/orchestrate_step3.py

import numpy as np

from src.step3.solver.predictor import compute_predictor_step


def orchestrate_step3(state):
    """
    Step 3 Orchestrator: Direct orchestration with no unnecessary bridges.
    """
    # 1. Map SolverState object to raw numpy arrays (Hydration)
    dx = (state.grid.x_max - state.grid.x_min) / state.grid.nx
    dy = (state.grid.y_max - state.grid.y_min) / state.grid.ny
    dz = (state.grid.z_max - state.grid.z_min) / state.grid.nz
    
    dt = state.config.simulation_parameters["time_step"]
    rho = state.config.fluid_properties["density"]
    mu = state.config.fluid_properties["viscosity"]
    F_vals = tuple(state.config.external_forces["force_vector"])
    
    v_n = np.stack([state.fields.U, state.fields.V, state.fields.W])
    p_n = state.fields.P
    
    # 2. PREDICT: Calculate intermediate V*
    state.fields.v_star = compute_predictor_step(v_n, p_n, dx, dy, dz, dt, rho, mu, F_vals)
    
    # 3. SOLVE: ... (Next steps)
    # 4. CORRECT: ... (Next steps)
    
    return state