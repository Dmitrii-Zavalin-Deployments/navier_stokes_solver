# src/step2/orchestrate_step2.py

from src.core.solver_state import SolverState

from .factory import get_initialization_context
from .stencil_assembler import assemble_stencil_matrix


def orchestrate_step2(state: SolverState) -> SolverState:
    """
    Orchestrates the construction of the Stencil Matrix (The Wiring).
    
    Adheres to the Hybrid Memory Foundation:
    1. Extracts the Foundation (NumPy buffer) from SolverState.
    2. Wires the Topology (StencilBlocks) via the Assembler.
    3. Commits the state to ensure the time-loop operates on existing memory.
    """
    
    # 1. Hoist context and physical parameters
    # Rule 5: Deterministic Initialization (No defaults here)
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
    
    # 2. Assemble the Stencil Matrix (Wiring)
    # The assembler now manages the injection of the fields_buffer (Foundation)
    # into the Cell objects, effectively bridging Logic and Math.
    state.stencil_matrix = assemble_stencil_matrix(
        state, 
        state.grid.nx, 
        state.grid.ny, 
        state.grid.nz, 
        ctx, 
        physics_params
    )
    
    # 3. Final Commit
    # Rule 9: Structural Persistence - The wiring is now set and persistent.
    state.ready_for_time_loop = True
    
    return state