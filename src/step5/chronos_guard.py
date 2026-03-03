# src/step5/chronos_guard.py

from src.solver_state import SolverState

DEBUG = True

def synchronize_terminal_state(state: SolverState) -> None:
    """
    Step 5.2: Chronos Guard. 
    The Single Authority for temporal loop termination.
    """
    total_time = state.total_time 
    
    # Use a small epsilon to handle floating point noise
    if float(state.time) >= (float(total_time) - 1e-9):
        state.time = total_time  # Clean up trailing decimals
        state.ready_for_time_loop = False
        if DEBUG:
            print(f"DEBUG [Chronos]: Terminal time reached. Loop Readiness -> False.")
    
    # Finalize health for the iteration
    state.health.is_stable = True
    state.health.post_correction_divergence_norm = state.health.divergence_norm