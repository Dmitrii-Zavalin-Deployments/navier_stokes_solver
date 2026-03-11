# tests/helpers/solver_step2_output_dummy.py

from src.step2.factory import get_initialization_context
from src.step2.stencil_assembler import assemble_stencil_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Generates a valid SolverState with the stencil_matrix wiring 
    assembled directly from state properties.
    """
    # 1. Start with the hydrated state from Step 1
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Extract initialization context (Factory pattern)
    ctx = get_initialization_context(state)
    
    # 3. Extract physics parameters from the state (mirroring orchestrate_step2)
    # This ensures consistency between the test dummy and the real implementation
    physics_params = {
        "dx": state.grid.dx,
        "dy": state.grid.dy,
        "dz": state.grid.dz,
        "dt": state.sim_params.time_step,
        "rho": state.fluid.density,
        "mu": state.fluid.viscosity,
        "f_vals": tuple(state.external_forces.force_vector)
    }
    
    # 4. Populate the stencil_matrix (Wiring)
    # The assembler now manages the injection of the fields_buffer (Foundation)
    # into the Cell objects.
    state.stencil_matrix = assemble_stencil_matrix(
        state, 
        state.grid.nx, 
        state.grid.ny, 
        state.grid.nz, 
        ctx, 
        physics_params
    )
    
    # 5. Commit State (Sentinel Trigger)
    # Setting this to True triggers the POST and physical sanity gates 
    # defined in SolverState.ready_for_time_loop setter.
    state.ready_for_time_loop = True
    
    return state