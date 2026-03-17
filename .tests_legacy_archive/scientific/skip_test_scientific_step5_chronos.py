# tests/scientific/test_scientific_step5_chronos.py

# tests/scientific/test_scientific_step5_chronos.py

from unittest.mock import PropertyMock, patch

import pytest

from src.solver_state import SolverState
from src.step5.chronos_guard import synchronize_terminal_state


@pytest.fixture
def state_for_chronos():
    """Initializes a state with clear temporal boundaries."""
    state = SolverState()
    
    # Setup baseline health and readiness
    state.health.divergence_norm = 1.23e-5
    state.ready_for_time_loop = True
    
    return state

def test_chronos_epsilon_formula_boundary(state_for_chronos):
    """
    Scientific check: Verifies the epsilon formula (1e-9).
    The simulation should stop if we are 'effectively' at total_time.
    """
    total_limit = 1.0
    # Simulate floating point noise: 0.9999999999 (within 1e-9 of 1.0)
    state_for_chronos.time = 0.9999999999 
    
    with patch("src.solver_state.SolverState.total_time", new_callable=PropertyMock) as mock_total:
        mock_total.return_value = total_limit
        
        synchronize_terminal_state(state_for_chronos)
        
        # 1. Formula Check: state.time should be snapped to exact total_time
        assert state_for_chronos.time == 1.0
        # 2. State Check: Loop should be terminated
        assert state_for_chronos.ready_for_time_loop is False

def test_chronos_continues_before_limit(state_for_chronos):
    """Scientific check: Ensures readiness remains True if limit is not reached."""
    state_for_chronos.time = 0.5
    
    with patch("src.solver_state.SolverState.total_time", new_callable=PropertyMock) as mock_total:
        mock_total.return_value = 1.0
        
        synchronize_terminal_state(state_for_chronos)
        
        assert state_for_chronos.time == 0.5
        assert state_for_chronos.ready_for_time_loop is True

def test_chronos_health_sync_formula(state_for_chronos):
    """Scientific check: Verifies mass conservation (divergence) mapping."""
    state_for_chronos.health.divergence_norm = 5.5e-6
    
    with patch("src.solver_state.SolverState.total_time", new_callable=PropertyMock) as mock_total:
        mock_total.return_value = 10.0
        
        synchronize_terminal_state(state_for_chronos)
        
        # Verification of the synchronization formula:
        # post_correction_divergence_norm must equal divergence_norm
        assert state_for_chronos.health.is_stable is True
        assert state_for_chronos.health.post_correction_divergence_norm == 5.5e-6

def test_chronos_debug_handshake(state_for_chronos, capsys):
    """Scientific check: Verifies the terminal state debug signal."""
    state_for_chronos.time = 2.0
    
    with patch("src.solver_state.SolverState.total_time", new_callable=PropertyMock) as mock_total:
        mock_total.return_value = 2.0
        
        synchronize_terminal_state(state_for_chronos)
        
        captured = capsys.readouterr().out
        assert "DEBUG [Chronos]: Terminal time reached. Loop Readiness -> False." in captured

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