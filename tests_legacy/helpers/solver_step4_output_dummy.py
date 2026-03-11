# tests/helpers/solver_step4_output_dummy.py

from src.common.field_schema import FI
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def make_step4_output_dummy(nx=4, ny=4, nz=4):
    """
    Generates a 'Ground Truth' SolverState.
    This does NOT call the orchestrator; it sets the expected values 
    post-boundary-enforcement for a test to verify against.
    """
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)

    # Instead of re-running dispatch logic, we manually inject the 
    # expected results for specific known locations.
    # Example: Verify that the wall (mask -1) was set to 0.0
    
    # Identify a block that we expect to be a wall (e.g., center index)
    # This is our 'Ground Truth' for the test
    for block in state.stencil_matrix:
        if block.center.mask == -1:  # Wall case
            state.fields.data[block.center.index, FI.VX] = 0.0
            state.fields.data[block.center.index, FI.VY] = 0.0
            state.fields.data[block.center.index, FI.VZ] = 0.0
        
        # Example: Verify that x_min (if inflow) matches the boundary config
        if block.center.x == 0:
             state.fields.data[block.center.index, FI.VX] = 1.0 # Inflow velocity
             
    return state