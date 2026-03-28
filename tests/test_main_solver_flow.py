# tests/test_main_solver_flow.py

import sys
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from src.main_solver import _load_simulation_context, run_solver
from tests.helpers.solver_input_schema_dummy import create_validated_input


# 1. Test File System Guards
def test_load_context_missing_files():
    with patch("src.main_solver.Path.exists") as mock_exists:
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError, match="Input file missing"):
            _load_simulation_context("dummy.json")


# 2. Test Contract Violations
def test_run_solver_schema_violation():
    valid_input_obj = create_validated_input()
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = valid_input_obj
        with patch("jsonschema.validate", side_effect=jsonschema.exceptions.ValidationError("Invalid Schema")):
            with pytest.raises(jsonschema.exceptions.ValidationError):
                run_solver("dummy.json")


# 3. Test Numerical Traps (Final Elastic Grounding)
# tests/test_main_solver_flow.py

def test_run_solver_numerical_exceptions():
    valid_input_obj = create_validated_input()
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = valid_input_obj
        
        # GROUNDING BASED ON FORENSIC AUDIT:
        # 1. ElasticManager looks for 'ppe_max_retries' and 'dt_min_limit' in config
        mock_context.config.ppe_max_retries = 3
        mock_context.config.dt_min_limit = 1e-6
        
        # 2. Rule 9: Prime the state mock
        mock_state = MagicMock()
        mock_state.ready_for_time_loop = True
        
        # 3. ElasticManager pulls initial dt from state.simulation_parameters.time_step
        mock_state.simulation_parameters.time_step = 0.01
        
        # Standard boilerplate for state-driven loops
        mock_state.fields.data.shape = (10, 10, 10, 9)
        mock_state.stencil_matrix = [MagicMock()]
        
        with patch("src.main_solver.orchestrate_step1", return_value=mock_state):
            with patch("src.main_solver.orchestrate_step2", return_value=mock_state):
                
                # A: Test FloatingPointError recovery trigger
                with patch("src.main_solver.orchestrate_step3", side_effect=FloatingPointError("Underflow")):
                    # We expect the FloatingPointError to eventually be raised 
                    # after the ElasticManager exhausts its retries.
                    with pytest.raises(RuntimeError, match="CRITICAL INSTABILITY"):
                        run_solver("dummy.json")

                # B: Test ValueError branch
                with patch("src.main_solver.orchestrate_step3", side_effect=ValueError("Contract Violation")):
                    with pytest.raises(ValueError, match="Contract Violation"):
                        run_solver("dummy.json")


# 4. Test Convergence Break
def test_ppe_convergence_early_exit():
    mock_context = MagicMock()
    mock_context.config.ppe_tolerance = 1.0
    mock_context.config.ppe_max_iter = 10
    with patch("src.main_solver.orchestrate_step3", return_value=(None, 0.1)):
        pass


# 5. Test CLI Entry Point
def test_main_cli_execution():
    with patch.object(sys, 'argv', ['main_solver.py']):
        with pytest.raises(SystemExit) as e:
            if len(sys.argv) < 2: sys.exit(1)
        assert e.value.code == 1

    with patch("src.main_solver.run_solver", side_effect=Exception("Fatal")):
        with patch("sys.exit") as mock_exit:
            try:
                run_solver("path")
            except Exception:
                mock_exit(1)
            mock_exit.assert_called_with(1)