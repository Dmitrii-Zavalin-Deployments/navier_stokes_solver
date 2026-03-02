# tests/helpers/solver_step4_output_dummy.py

import numpy as np
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy

def make_step4_output_dummy(nx=4, ny=4, nz=4):
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)

    state.fields.P_ext = np.zeros((nx + 2, ny + 2, nz + 2))
    state.fields.U_ext = np.zeros((nx + 3, ny + 2, nz + 2))
    state.fields.V_ext = np.zeros((nx + 2, ny + 3, nz + 2))
    state.fields.W_ext = np.zeros((nx + 2, ny + 2, nz + 3))

    total_voxels = (nx+2)*(ny+2)*(nz+2) 
    state.diagnostics.memory_footprint_gb = (total_voxels * 8 * 4) / 1e9
    state.diagnostics.bc_verification_passed = True
    
    max_u = state.health.max_u
    state.diagnostics.initial_cfl_dt = 0.5 * state.grid.dx / max_u if max_u > 0 else 0.01

    state.ready_for_time_loop = True
    return state