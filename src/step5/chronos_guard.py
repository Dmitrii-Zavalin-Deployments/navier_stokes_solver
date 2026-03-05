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

def test_chronos_continues_just_outside_epsilon(state_for_chronos):
    """
    Scientific check: Ensures the simulation doesn't stop prematurely 
    if time is just outside the 1e-9 epsilon.
    """
    total_limit = 1.0
    # 0.999999998 is 2e-9 away from 1.0 (Outside the 1e-9 guard)
    state_for_chronos.time = 0.999999998 
    
    with patch("src.solver_state.SolverState.total_time", new_callable=PropertyMock) as mock_total:
        mock_total.return_value = total_limit
        synchronize_terminal_state(state_for_chronos)
        
        # Should NOT have snapped to 1.0 yet
        assert state_for_chronos.time == 0.999999998
        # Should still be ready for the final step
        assert state_for_chronos.ready_for_time_loop is True