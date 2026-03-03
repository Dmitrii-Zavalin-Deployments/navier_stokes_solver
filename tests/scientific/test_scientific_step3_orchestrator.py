import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.solver_state import SolverState

@pytest.fixture
def state_orchestrator():
    """Fixture to set up a minimal valid state for orchestration."""
    state = SolverState()
    # Mocking physics/config to avoid uninitialized errors
    state.config._fluid_properties = {"density": 1000.0, "viscosity": 0.001}
    state.config._simulation_parameters = {"time_step": 0.01, "total_time": 1.0, "output_interval": 1}
    
    # Initialize health vitals (required by SSoT Rule 4)
    state.health.divergence_norm = 1e-12
    state.health.max_u = 0.5
    state.time = 0.123
    return state

## =========================================================
## RULE 3.5: ORCHESTRATION SEQUENCING
## =========================================================

def test_orchestrate_step3_success_flow(state_orchestrator, capsys):
    """
    Verify the sequence: Predict -> Solve -> Correct -> Persist.
    Formula covered: V_new = Correct(Solve(Predict(V_old)))
    """
    from src.step3.orchestrate_step3 import orchestrate_step3
    
    # Mock the internal steps so we only test the orchestration logic
    with patch('src.step3.orchestrate_step3.predict_velocity') as mock_predict, \
         patch('src.step3.orchestrate_step3.solve_pressure', return_value="converged") as mock_solve, \
         patch('src.step3.orchestrate_step3.correct_velocity') as mock_correct:
        
        result = orchestrate_step3(state_orchestrator)
        
        # Check call sequence
        mock_predict.assert_called_once()
        mock_solve.assert_called_once()
        mock_correct.assert_called_once()
        
        # Verify History Persistence (Rule 4)
        assert state_orchestrator.history.times[-1] == 0.123
        assert state_orchestrator.history.ppe_status_history[-1] == "converged"
        assert state_orchestrator.ready_for_time_loop is True
        
        # Verify Debug Output
        captured = capsys.readouterr()
        assert "Starting Step 3" in captured.out
        assert "Step 3 Complete" in captured.out

def test_orchestrate_step3_convergence_gate(state_orchestrator):
    """
    Logic Gate Check: If PPE fails, the solver must abort to prevent 'Numerical Explosion'.
    """
    from src.step3.orchestrate_step3 import orchestrate_step3
    
    with patch('src.step3.orchestrate_step3.predict_velocity'), \
         patch('src.step3.orchestrate_step3.solve_pressure', return_value="diverged"), \
         patch('src.step3.orchestrate_step3.correct_velocity') as mock_correct:
        
        # Expecting RuntimeError due to Rule 5 (No silent failures)
        with pytest.raises(RuntimeError, match="PPE Solve did not converge"):
            orchestrate_step3(state_orchestrator)
            
        # Ensure correction was NEVER applied to a broken pressure field
        mock_correct.assert_not_called()

def test_orchestrate_step3_zero_debt_history(state_orchestrator):
    """
    Rule 4 Check: Ensure history fails if health vitals are missing (No Defaults).
    """
    from src.step3.orchestrate_step3 import orchestrate_step3
    
    # Force an uninitialized health attribute
    state_orchestrator.health._max_u = None 
    
    with patch('src.step3.orchestrate_step3.predict_velocity'), \
         patch('src.step3.orchestrate_step3.solve_pressure', return_value="converged"), \
         patch('src.step3.orchestrate_step3.correct_velocity'):
        
        # Should raise RuntimeError from the ValidatedContainer 'Security Guard'
        with pytest.raises(RuntimeError, match="Access Error: 'max_u'"):
            orchestrate_step3(state_orchestrator)

## =========================================================
## ADDITIONAL LOGIC & STATE INTEGRITY CHECKS
## =========================================================

def test_orchestrate_step3_state_identity(state_orchestrator):
    """
    Verify that the orchestrator returns the same state object (Referential Integrity).
    """
    from src.step3.orchestrate_step3 import orchestrate_step3
    
    with patch('src.step3.orchestrate_step3.predict_velocity'), \
         patch('src.step3.orchestrate_step3.solve_pressure', return_value="converged"), \
         patch('src.step3.orchestrate_step3.correct_velocity'):
        
        returned_state = orchestrate_step3(state_orchestrator)
        assert returned_state is state_orchestrator

def test_orchestrate_step3_ready_flag_integrity(state_orchestrator):
    """
    Verify that the 'ready_for_time_loop' flag is ONLY flipped after success.
    """
    from src.step3.orchestrate_step3 import orchestrate_step3
    
    # Initially False
    state_orchestrator.ready_for_time_loop = False
    
    # Case 1: Failure - Flag should stay False
    with patch('src.step3.orchestrate_step3.predict_velocity'), \
         patch('src.step3.orchestrate_step3.solve_pressure', return_value="failed"):
        
        with pytest.raises(RuntimeError):
            orchestrate_step3(state_orchestrator)
        assert state_orchestrator.ready_for_time_loop is False
    
    # Case 2: Success - Flag should flip to True
    with patch('src.step3.orchestrate_step3.predict_velocity'), \
         patch('src.step3.orchestrate_step3.solve_pressure', return_value="converged"), \
         patch('src.step3.orchestrate_step3.correct_velocity'):
        
        orchestrate_step3(state_orchestrator)
        assert state_orchestrator.ready_for_time_loop is True

def test_orchestrate_step3_history_matching_vitals(state_orchestrator):
    """
    Formula Check: Ensure values in history exactly match the health values at the time of recording.
    Formula: state.history.div_norms[-1] == state.health.divergence_norm
    """
    from src.step3.orchestrate_step3 import orchestrate_step3
    
    # Set specific "vitals"
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
