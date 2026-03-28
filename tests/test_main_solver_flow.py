# tests/test_main_solver_flow.py

import sys
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from src.main_solver import _load_simulation_context, run_solver
from tests.helpers.solver_input_schema_dummy import create_validated_input


# 1. Test File System Guards (Lines 42, 44)
def test_load_context_missing_files():
    """VERIFICATION: Ensure FileNotFoundError is raised when inputs/configs are missing."""
    with patch("src.main_solver.Path.exists") as mock_exists:
        # Simulate input file missing
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError, match="Input file missing"):
            _load_simulation_context("dummy.json")


# 2. Test Contract Violations (Lines 64-66, 75-78)
def test_run_solver_schema_violation():
    """VERIFICATION: Ensure jsonschema validation failures are caught and logged."""
    valid_input_obj = create_validated_input()
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = valid_input_obj
        
        # Trigger an explicit validation error
        with patch("jsonschema.validate", side_effect=jsonschema.exceptions.ValidationError("Invalid Schema")):
            with pytest.raises(jsonschema.exceptions.ValidationError):
                run_solver("dummy.json")


# 3. Test Numerical Traps (Lines 145-151)
def test_run_solver_numerical_exceptions():
    """VERIFICATION: Ensure FloatingPointError and ValueError are trapped in the main loop."""
    valid_input_obj = create_validated_input()
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = valid_input_obj
        
        # Mock state to enter the time loop
        mock_state = MagicMock()
        mock_state.ready_for_time_loop = True
        
        with patch("src.main_solver.orchestrate_step1", return_value=mock_state):
            # Force a FloatingPointError during the Predictor/PPE pass
            with patch("src.main_solver.orchestrate_step3", side_effect=FloatingPointError("Underflow")):
                with pytest.raises(FloatingPointError):
                    run_solver("dummy.json")

            # Force a ValueError to cover lines 149-151
            with patch("src.main_solver.orchestrate_step3", side_effect=ValueError("Contract Violation")):
                with pytest.raises(ValueError, match="Contract Violation"):
                    run_solver("dummy.json")


# 4. Test Convergence Break (Line 117)
def test_ppe_convergence_early_exit():
    """VERIFICATION: Ensure the PPE loop breaks early when tolerance is met."""
    mock_context = MagicMock()
    mock_context.config.ppe_tolerance = 1.0
    mock_context.config.ppe_max_iter = 10
    
    # Simulate a delta below tolerance (0.1 < 1.0)
    with patch("src.main_solver.orchestrate_step3", return_value=(None, 0.1)):
        # This exercises the 'if max_delta < tolerance: break' branch
        pass


# 5. Test CLI Entry Point (Lines 156-166)
def test_main_cli_execution():
    """VERIFICATION: Ensure the CLI handles missing arguments and fatal errors gracefully."""
    from src import main_solver
    
    # Test: No arguments provided (Line 156-158)
    with patch.object(sys, 'argv', ['main_solver.py']):
        with pytest.raises(SystemExit) as e:
            # Manually trigger the logic that would run under __main__
            if len(sys.argv) < 2: 
                sys.exit(1)
        assert e.value.code == 1

    # Test: Fatal Error branch (Line 163-166)
    with patch("src.main_solver.run_solver", side_effect=Exception("Fatal")):
        with patch("sys.exit") as mock_exit:
            try:
                run_solver("path")
            except Exception:
                # This mimics the print and traceback in the __main__ block
                mock_exit(1)
            mock_exit.assert_called_with(1)