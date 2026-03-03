# src/step5/orchestrate_step5.py

from src.solver_state import SolverState
from .archivist import record_snapshot
from .chronos_guard import synchronize_terminal_state

DEBUG = True

def orchestrate_step5(state: SolverState) -> SolverState:
    """
    Step 5 Orchestrator: Finalization & I/O.
    Ensures iteration data is saved and terminal conditions are evaluated.
    """
    if DEBUG and state.iteration % 10 == 0:
        print(f"DEBUG [Step 5 Orchestrator]: Syncing Iteration {state.iteration}")

    # 1. Record artifacts
    record_snapshot(state)
    
    # 2. Evaluate termination criteria (SSoT for time)
    synchronize_terminal_state(state)
    
    # 3. Dynamic Checkpoint naming
    case_prefix = state.config.case_name
    state.manifest.final_checkpoint = f"{case_prefix}_iter_{state.iteration}.npy"
    
    if not state.ready_for_time_loop and DEBUG:
        print("DEBUG [Step 5 Orchestrator]: >>> SIGNALING SIMULATION COMPLETION <<<")
        
    return state