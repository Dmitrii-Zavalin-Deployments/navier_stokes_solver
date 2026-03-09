# src/step2/orchestrate_step2.py

from src.core.solver_state import SolverState

from .factory import get_initialization_context
from .stencil_assembler import assemble_stencil_matrix


def orchestrate_step2(state: SolverState) -> SolverState:
    """
    Orchestrates the construction of the Stencil Matrix.
    
    This function initializes the physical context and coordinates the 
    assembly of StencilBlock objects, which serve as the primary 
    units for the numerical solver in subsequent steps.
    """
    
    # 1. Hoist context and physical parameters
    # This prepares the constants needed by all stencil blocks in the grid
    ctx = get_initialization_context(state)
    
    physics_params = {
        "dx": state.grid.dx,
        "dy": state.grid.dy,
        "dz": state.grid.dz,
        "dt": state.config.simulation_parameters["time_step"],
        "rho": state.config.fluid_properties["density"],
        "mu": state.config.fluid_properties["viscosity"],
        "f_vals": tuple(state.config.external_forces["force_vector"])
    }
    
    # 2. Assemble the Stencil Matrix
    # The assembler handles grid traversal (including ghost layers) 
    # and populates the matrix with StencilBlock objects.
    state.stencil_matrix = assemble_stencil_matrix(
        state, 
        state.grid.nx, 
        state.grid.ny, 
        state.grid.nz, 
        ctx, 
        physics_params
    )
    
    # 3. Final Commit
    # Signal that topography and stencil connectivity are ready for time iteration
    state.ready_for_time_loop = True
    
    return state