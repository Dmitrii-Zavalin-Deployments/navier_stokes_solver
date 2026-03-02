# tests/helpers/solver_step3_output_dummy.py

import numpy as np
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def make_step3_output_dummy(nx=4, ny=4, nz=4):
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)

    state.fields.U = np.zeros((nx + 1, ny, nz))
    state.fields.V = np.zeros((nx, ny + 1, nz))
    state.fields.W = np.zeros((nx, ny, nz + 1))
    state.fields.P = np.zeros((nx, ny, nz))
 
    state.fields.U_star = np.zeros((nx + 1, ny, nz))
    state.fields.V_star = np.zeros((nx, ny + 1, nz))
    state.fields.W_star = np.zeros((nx, ny, nz + 1))

    state.health.is_stable = True
    state.health.max_u = 0.5 
    state.health.divergence_norm = 1e-14
    state.health.post_correction_divergence_norm = 1e-14

    state.history.times.append(state.time)
    state.history.divergence_norms.append(1e-14)
    state.history.max_velocity_history.append(0.5)
    state.history.energy_history.append(0.0125)
    state.history.ppe_status_history.append("converged")

    state.iteration = 1
    state.time += state.config.dt
    state.ready_for_time_loop = True 

    return state