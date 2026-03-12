# src/step2/orchestrate_step2.py

from src.common.solver_state import SolverState

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
    # The assembler consumes the state directly, maintaining architectural integrity.
    # The Foundation (state.fields) and the Wiring (stencil_matrix) are now linked.
    state.stencil_matrix = assemble_stencil_matrix(state)
    
    # Rule 9: Structural Persistence
    # The Wiring is fully materialized. 
    # Subsequent time-loop operations will operate in-place on state.fields.data.
    state.ready_for_time_loop = True
    
    if DEBUG:
        print(f"DEBUG [Step 2.0]: Stencil Matrix Assembly Complete")
        print(f"  > Total StencilBlocks Generated: {len(state.stencil_matrix)}")
        print(f"  > Orchestration Finalized, State Ready for Time-Loop")
    
    return state