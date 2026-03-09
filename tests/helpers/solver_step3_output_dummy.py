# tests/helpers/solver_step3_output_dummy.py

import numpy as np

from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def make_step3_output_dummy(nx=4, ny=4, nz=4):
    """
    Generates a valid SolverState representing the system 
    immediately after orchestrate_step3 has finished.
    """
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)

    # 1. Populate StencilBlock internal fields
    # Only physical data that Step 3 modifies
    for block in state.stencil_matrix:
        if not block.center.is_ghost:
            # Synced pressure
            block.center.p = 0.01
            block.center.p_next = 0.01
            
            # Corrected velocities (divergence-free)
            block.center.u = 0.5
            block.center.v = 0.5
            block.center.w = 0.5
            
            # Intermediate velocities (v*)
            block.center.u_star = 0.51
            block.center.v_star = 0.51
            block.center.w_star = 0.51

    # 2. Populate global SolverState fields
    # These represent the aggregated field buffers ready for downstream steps
    state.fields.U = np.ones((nx + 1, ny, nz)) * 0.5
    state.fields.V = np.ones((nx, ny + 1, nz)) * 0.5
    state.fields.W = np.ones((nx, ny, nz + 1)) * 0.5
    state.fields.P = np.ones((nx, ny, nz)) * 0.01

    return state