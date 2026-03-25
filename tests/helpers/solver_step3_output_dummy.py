# tests/helpers/solver_step3_output_dummy.py

from src.common.field_schema import FI
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
# Ensure we have the logic that wires cells to the data buffer
from src.step2.stencil_assembler import assemble_stencils 

def make_step3_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4, block_index: int = 0):
    """
    Step 3 Output: The 'Trial' state.
    Bridges the gap between monolithic SolverState and StencilBlock arrays.
    """
    # 1. Generate the foundation state from Step 2
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    data = state.fields.data
    
    # 2. Assign Trial Data (Numpy-style slicing ensures these are arrays, not scalars)
    # We use [:] to ensure we are writing into the allocated buffer
    data[:, FI.VX_STAR] = 0.50 
    data[:, FI.VY_STAR] = 0.50
    data[:, FI.VZ_STAR] = 0.50
    data[:, FI.P_NEXT]  = 0.012
    
    # 3. CRITICAL: Re-wire the stencil matrix to reflect these data changes
    # This ensures block.center.vx is a view of data[:, FI.VX], not a scalar 0.0
    state.stencil_matrix = assemble_stencils(state.grid, state.fields, state.boundary_conditions)
    
    return state.stencil_matrix[block_index]