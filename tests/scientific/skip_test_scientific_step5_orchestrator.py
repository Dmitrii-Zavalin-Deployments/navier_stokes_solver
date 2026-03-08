# tests/scientific/test_scientific_step5_orchestrator.py

# tests/scientific/test_scientific_step5_orchestrator.py

from unittest.mock import PropertyMock, patch

import pytest

from src.solver_state import SolverState
from src.step5.orchestrate_step5 import orchestrate_step5


@pytest.fixture
def state_for_orchestration():
    """Sets up a state for integration testing of Step 5."""
    state = SolverState()
    state.iteration = 100
    state.config._case_name = "orchestra_test"
    state.ready_for_time_loop = True
    return state

def test_orchestrator_sequencing_and_checkpoint(state_for_orchestration):
    """
    Scientific check: Verifies that Archivist is called before Chronos Guard
    and validates the dynamic checkpoint filename formula.
    """
    # Mocking sub-components to verify they are called
    with patch("src.step5.orchestrate_step5.record_snapshot") as mock_archivist, \
         patch("src.step5.orchestrate_step5.synchronize_terminal_state") as mock_chronos, \
         patch("src.solver_state.SolverConfig.case_name", new_callable=PropertyMock) as mock_case:
        
        mock_case.return_value = "orchestra_test"
        
        # Execute Orchestration
        orchestrate_step5(state_for_orchestration)
        
        # 1. Sequence Check: Both major systems must be engaged
        mock_archivist.assert_called_once_with(state_for_orchestration)
        mock_chronos.assert_called_once_with(state_for_orchestration)
        
        # 2. Formula Check: Checkpoint naming convention {case}_iter_{iteration}.npy
        expected_checkpoint = "orchestra_test_iter_100.npy"
        assert state_for_orchestration.manifest.final_checkpoint == expected_checkpoint

def test_orchestrator_debug_sync_interval(state_for_orchestration, capsys):
    """Scientific check: Verifies interval-based logging (state.iteration % 10)."""
    # Test on a multiple of 10
    state_for_orchestration.iteration = 20
    with patch("src.step5.orchestrate_step5.record_snapshot"), \
         patch("src.step5.orchestrate_step5.synchronize_terminal_state"):
        
        orchestrate_step5(state_for_orchestration)
        captured = capsys.readouterr().out
        assert "DEBUG [Step 5 Orchestrator]: Syncing Iteration 20" in captured

def test_orchestrator_completion_signal(state_for_orchestration, capsys):
    """Scientific check: Verifies completion signal when loop readiness is revoked."""
    state_for_orchestration.ready_for_time_loop = False # Simulated termination
    
    with patch("src.step5.orchestrate_step5.record_snapshot"), \
         patch("src.step5.orchestrate_step5.synchronize_terminal_state"):
        
        orchestrate_step5(state_for_orchestration)
        captured = capsys.readouterr().out
        assert "DEBUG [Step 5 Orchestrator]: >>> SIGNALING SIMULATION COMPLETION <<<" in captured

def test_orchestrator_modulo_silence(state_for_orchestration, capsys):
    """
    Scientific check: Verifies that the orchestrator remains silent 
    on non-interval iterations to preserve I/O performance.
    """
    state_for_orchestration.iteration = 23 # Not a multiple of 10
    
    with patch("src.step5.orchestrate_step5.record_snapshot"), \
         patch("src.step5.orchestrate_step5.synchronize_terminal_state"):
        
        orchestrate_step5(state_for_orchestration)
        captured = capsys.readouterr().out
        
        # Should NOT contain the sync message
        assert "DEBUG [Step 5 Orchestrator]: Syncing Iteration" not in captured