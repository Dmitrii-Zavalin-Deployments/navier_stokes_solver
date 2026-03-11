# src/step2/orchestrate_step2.py

from src.core.solver_state import SolverState
from .stencil_assembler import assemble_stencil_matrix

# Rule 7: Granular Traceability
DEBUG = True

def orchestrate_step2(state: SolverState) -> SolverState:
    """
    Orchestrates the construction of the Stencil Matrix (The Wiring).
    
    Adheres to the SSoT Architecture Guard: 
    Data is accessed directly from state sub-containers to prevent 
    redundant context creation and state-drift.
    """
    
    if DEBUG:
        print(f"DEBUG [Step 2.0]: Orchestration Started")
        print(f"  > Grid: {state.grid.nx}x{state.grid.ny}x{state.grid.nz}")

    # Rule 4 & 5: Direct dependency access (SSoT).
    # We pass the state directly to the assembler, which extracts 
    # configuration and geometry internally. This eliminates 
    # intermediate 'ctx' or 'params' dictionaries.
    state.stencil_matrix = assemble_stencil_matrix(state)
    
    # Rule 9: Structural Persistence
    # Wiring is now set and persistent in state.stencil_matrix
    state.ready_for_time_loop = True
    
    if DEBUG:
        print(f"DEBUG [Step 2.0]: Stencil Matrix Assembly Complete")
        print(f"  > Total StencilBlocks Generated: {len(state.stencil_matrix)}")
        print(f"  > Orchestration Finalized, State Ready for Time-Loop")
    
    return state