# tests/scientific/test_orchestrate_step3.py

from unittest.mock import patch

import pytest

from src.solver_state import SolverState


@pytest.fixture
def state_orchestrator():
    """Fixture to set up a minimal valid state for orchestration."""
    state = SolverState()
    # Direct dictionary hydration to satisfy _get_safe validation
    fluid_props = state.fluid = {"density": 1000.0, "viscosity": 0.001}
    state.config.simulation_parameters = {
        "time_step": 0.01, 
        "total_time": 1.0, 
        "output_interval": 1
    }
    
    # Pre-populate health vitals to prevent initialization errors
    state.health.divergence_norm = 1e-12
    state.health.max_u = 0.5
    state.time = 0.123
    return state

def test_orchestrate_step3_success_flow(state_orchestrator, capsys):
    """Rule 3.5: Verify Predict -> Solve -> Correct -> Persist sequence."""
    from src.step3.orchestrate_step3 import orchestrate_step3
    
    with patch('src.step3.orchestrate_step3.predict_velocity') as mock_predict, \
         patch('src.step3.orchestrate_step3.solve_pressure', return_value="converged") as mock_solve, \
         patch('src.step3.orchestrate_step3.correct_velocity') as mock_correct:
        
        orchestrate_step3(state_orchestrator)
        
        # Verify call order (implied by execution)
        mock_predict.assert_called_once()
        mock_solve.assert_called_once()
        mock_correct.assert_called_once()
        
        # Verify History Persistence (Rule 4)
        assert state_orchestrator.history.times[-1] == 0.123
        assert state_orchestrator.history.ppe_status_history[-1] == "converged"
        assert state_orchestrator.ready_for_time_loop is True

def test_orchestrate_step3_convergence_gate(state_orchestrator):
    """Verify solver aborts on non-convergence to prevent numerical explosion."""
    from src.step3.orchestrate_step3 import orchestrate_step3
    
    with patch('src.step3.orchestrate_step3.predict_velocity'), \
         patch('src.step3.orchestrate_step3.solve_pressure', return_value="diverged"), \
         patch('src.step3.orchestrate_step3.correct_velocity') as mock_correct:
        
        with pytest.raises(RuntimeError, match="PPE Solve did not converge"):
            orchestrate_step3(state_orchestrator)
            
        # The safety gate: correction MUST NOT be applied if pressure is junk
        mock_correct.assert_not_called()

def test_orchestrate_step3_zero_debt_history(state_orchestrator):
    """Rule 4: Ensure history fails if health vitals are missing (No Defaults)."""
    from src.step3.orchestrate_step3 import orchestrate_step3
    
    # Bypassing the setter to force an invalid state for testing
    state_orchestrator.health._max_u = None 
    
    with patch('src.step3.orchestrate_step3.predict_velocity'), \
         patch('src.step3.orchestrate_step3.solve_pressure', return_value="converged"), \
         patch('src.step3.orchestrate_step3.correct_velocity'):
        
        # This will trigger the _get_safe RuntimeError in ValidatedContainer
        with pytest.raises(RuntimeError):
            orchestrate_step3(state_orchestrator)

def test_orchestrate_step3_history_matching_vitals(state_orchestrator):
    """Ensure history records EXACTLY what health reports (Referential Integrity)."""
    from src.step3.orchestrate_step3 import orchestrate_step3
    
    test_div = 5.67e-4
    test_u_max = 12.34
    state_orchestrator.health.divergence_norm = test_div
    state_orchestrator.health.max_u = test_u_max
    
    with patch('src.step3.orchestrate_step3.predict_velocity'), \
         patch('src.step3.orchestrate_step3.solve_pressure', return_value="converged"), \
         patch('src.step3.orchestrate_step3.correct_velocity'):
        
        orchestrate_step3(state_orchestrator)
        
        assert state_orchestrator.history.divergence_norms[-1] == test_div
        assert state_orchestrator.history.max_velocity_history[-1] == test_u_max