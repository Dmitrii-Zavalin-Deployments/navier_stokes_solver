# tests/helpers/solver_step3_output_dummy.py

"""
Archivist Testing: Snapshot-based Test Baseline (Step 3).

Compliance:
- Rule 4: SSoT (Hierarchy over convenience).
- Rule 9: Hybrid Memory Foundation (Fields initialized to verified state).
"""

from src.common.field_schema import FI
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def make_step3_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Returns a 'frozen' prototype representing a system state after Step 3.
    The Foundation (NumPy buffers) is pre-filled with verified numerical values.
    """
    # 1. Start from the Step 2 baseline
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Populate Foundation (Rule 9: Hybrid Memory Foundation)
    # We directly manipulate the contiguous buffer. Since Cell objects in
    # state.stencil_matrix are views into this buffer, they are 
    # synchronized automatically.
    data = state.fields.data
    
    # Set Primary Velocities (divergence-free state)
    data[:, FI.VX] = 0.5
    data[:, FI.VY] = 0.5
    data[:, FI.VZ] = 0.5
    
    # Set Intermediate Velocities
    data[:, FI.VX_STAR] = 0.51
    data[:, FI.VY_STAR] = 0.51
    data[:, FI.VZ_STAR] = 0.51
    
    # Set Pressure Fields
    data[:, FI.P] = 0.01
    data[:, FI.P_NEXT] = 0.01

    # 3. Validation (POST)
    # Explicitly confirm the state is ready. The ready_for_time_loop
    # flag is already set by make_step2_output_dummy.
    return state