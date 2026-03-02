# src/step5/chronos_guard.py

from src.solver_state import SolverState

def synchronize_terminal_state(state: SolverState) -> None:
    """
    Step 5.2: Chronos Guard. Temporal boundary enforcement.
    """
    # Access via the facade property fixed in SolverConfig
    total_time = state.total_time 
    
    if state.time >= total_time:
        state.time = total_time
        # Signal completion
        state.ready_for_time_loop = False
    
    # Update health vitals for the final summary
    state.health.is_stable = True
    # Ensure divergence_norm exists before syncing
    div = getattr(state.health, 'divergence_norm', 0.0)
    state.health.post_correction_divergence_norm = div
