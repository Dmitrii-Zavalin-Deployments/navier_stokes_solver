# tests/helpers/solver_step3_output_dummy.py

from src.common.field_schema import FI
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def make_step3_output_dummy(nx=4, ny=4, nz=4):
    """
    Generates a valid SolverState representing the system 
    immediately after orchestrate_step3 has finished.
    """
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)

    # 1. Populate Field Foundation (The Hybrid Memory Bridge)
    # Orchestrate_step3 modifies buffers in-place via indices defined in FI.
    # We simulate a post-projection state (divergence-free velocities, updated pressure).
    
    # Velocity fields (FI.VX, FI.VY, FI.VZ = 0, 1, 2)
    state.fields.data[:, FI.VX] = 0.5
    state.fields.data[:, FI.VY] = 0.5
    state.fields.data[:, FI.VZ] = 0.5
    
    # Intermediate velocities (FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR = 3, 4, 5)
    # These represent the prediction before correction
    state.fields.data[:, FI.VX_STAR] = 0.51
    state.fields.data[:, FI.VY_STAR] = 0.51
    state.fields.data[:, FI.VZ_STAR] = 0.51
    
    # Pressure fields (FI.P, FI.P_NEXT = 6, 7)
    # Synchronization ensures P == P_NEXT
    state.fields.data[:, FI.P] = 0.01
    state.fields.data[:, FI.P_NEXT] = 0.01

    # 2. Synchronize StencilBlocks (The Wiring)
    # Since Cell objects are pointers/views into the Foundation, 
    # they reflect the buffer updates automatically via property getters.
    # No manual synchronization loop is required.

    return state