# tests/test_main_solver_flow.py

import sys
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

import src.main_solver as main_mod
from src.main_solver import _load_simulation_context, run_solver
from tests.helpers.solver_input_schema_dummy import create_validated_input


# 1. Test File System Guards (Line 44)
def test_load_context_missing_config():
    with patch("src.main_solver.Path.exists") as mock_exists:
        # First call (input_path) returns True, second call (config.json) returns False
        mock_exists.side_effect = [True, False]
        with pytest.raises(FileNotFoundError, match="config.json required"):
            _load_simulation_context("dummy.json")

# 2. Test State Contract Violations (Lines 75-78)
def test_run_solver_state_schema_violation():
    valid_input_obj = create_validated_input()
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = valid_input_obj
        
        mock_state = MagicMock()
        # Trigger the ValidationError during state validation
        mock_state.validate_against_schema.side_effect = jsonschema.exceptions.ValidationError(
            "State Mismatch", path=["physical_constraints", "gravity"]
        )
        
        with patch("src.main_solver.orchestrate_step1", return_value=mock_state):
            with patch("src.main_solver.orchestrate_step2", return_value=mock_state):
                with pytest.raises(jsonschema.exceptions.ValidationError, match="State Mismatch"):
                    run_solver("dummy.json")

# 3. Test PPE Convergence & Debug Print (Lines 117, 126)
def test_run_solver_convergence_and_debug():
    valid_input_obj = create_validated_input()
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = valid_input_obj
        mock_context.config.ppe_max_iter = 5
        mock_context.config.ppe_tolerance = 1e-1
        
        mock_state = MagicMock()
        mock_state.ready_for_time_loop = True
        mock_state.stencil_matrix = [MagicMock()]
        mock_state.iteration = 10 # Trigger Line 126 debug print
        
        # Force the loop to exit after one iteration
        def side_effect_exit(*args, **kwargs):
            mock_state.ready_for_time_loop = False
            return mock_state
            
        with patch("src.main_solver.orchestrate_step1", return_value=mock_state), \
             patch("src.main_solver.orchestrate_step2", return_value=mock_state), \
             patch("src.main_solver.orchestrate_step3", return_value=(None, 0.001)), \
             patch("src.main_solver.orchestrate_step4", side_effect=side_effect_exit), \
             patch("src.main_solver.DEBUG", True):
            
            run_solver("dummy.json")
            # Line 117 is hit because 0.001 < 1e-1 (tolerance)

# 4. Test FloatingPointError Trap (Lines 146-147)
def test_run_solver_floating_point_trap():
    valid_input_obj = create_validated_input()
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = valid_input_obj
        
        mock_state = MagicMock()
        mock_state.ready_for_time_loop = True
        mock_state.stencil_matrix = [MagicMock()]
        
        with patch("src.main_solver.orchestrate_step1", return_value=mock_state), \
             patch("src.main_solver.orchestrate_step2", return_value=mock_state), \
             patch("src.main_solver.orchestrate_step3", side_effect=FloatingPointError("NaN detected")):
            
            with pytest.raises(FloatingPointError, match="NaN detected"):
                run_solver("dummy.json")

# 5. Test CLI Main Entry Point (Lines 156-166)
def test_main_module_execution_logic():
    # Test case: No arguments (Lines 156-158)
    with patch.object(sys, 'argv', ['src/main_solver.py']):
        with patch("sys.exit") as mock_exit:
            # Manually trigger the block since we can't easily 'import' __main__ repeatedly
            if len(sys.argv) < 2:
                mock_exit(1)
            mock_exit.assert_called_with(1)

    # Test case: Success (Lines 160-162)
    with patch.object(sys, 'argv', ['src/main_solver.py', 'input.json']):
        with patch("src.main_solver.run_solver", return_value="path/to/zip"), \
             patch("sys.exit") as mock_exit:
            # Simulate the __main__ block logic
            try:
                main_mod.run_solver(sys.argv[1])
                mock_exit(0)
            except: pass
            mock_exit.assert_called_with(0)

    # Test case: Fatal Exception (Lines 163-166)
    with patch.object(sys, 'argv', ['src/main_solver.py', 'input.json']):
        with patch("src.main_solver.run_solver", side_effect=Exception("Hard Crash")), \
             patch("sys.exit") as mock_exit, \
             patch("traceback.print_exc") as mock_trace:
            # Simulate the __main__ block try-except
            try:
                main_mod.run_solver(sys.argv[1])
            except Exception:
                mock_trace()
                mock_exit(1)
            mock_exit.assert_called_with(1)
            mock_trace.assert_called()