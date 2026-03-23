# tests/helpers/solver_step3_output_dummy.py

from src.common.field_schema import FI
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def make_step3_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4, block_index: int = 0):
    """
    Step 3 Output: The 'Trial' state.
    New physics are in STAR/NEXT; Foundation (VX, VY, VZ, P) remains at Step 2 values.
    """
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    data = state.fields.data
    
    # --- TRIAL BUFFERS (The New Results) ---
    data[:, FI.VX_STAR] = 0.50 
    data[:, FI.VY_STAR] = 0.50
    data[:, FI.VZ_STAR] = 0.50
    data[:, FI.P_NEXT]  = 0.012
    
    # --- FOUNDATION BUFFERS (The Old State) ---
    # We leave these as they were in Step 2 (e.g., 0.0 or initial conditions)
    # This proves the Elasticity Manager is the only one who can 'commit' them.
    
    return state.stencil_matrix[block_index]