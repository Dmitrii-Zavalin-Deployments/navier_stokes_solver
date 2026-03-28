# tests/test_main_solver_flow.py

from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from src.main_solver import _load_simulation_context, run_solver
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


# 1. Test File System Guards
def test_load_context_missing_config():
    with patch("src.main_solver.Path.exists") as mock_exists:
        mock_exists.side_effect = [True, False]
        with pytest.raises(FileNotFoundError, match="config.json required"):
            _load_simulation_context("dummy.json")

# 2. Test State Contract Violations
def test_run_solver_state_schema_violation():
    valid_input_obj = create_validated_input()
    real_state = make_step4_output_dummy() 
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = valid_input_obj
        
        # FIX: Patch the CLASS method to bypass __slots__ instance restrictions
        with patch("src.common.solver_state.SolverState.validate_against_schema") as mock_val:
            mock_val.side_effect = jsonschema.exceptions.ValidationError("State Mismatch")
            
            with patch("src.main_solver.orchestrate_step1", return_value=real_state), \
                 patch("src.main_solver.orchestrate_step2", return_value=real_state):
                with pytest.raises(jsonschema.exceptions.ValidationError, match="State Mismatch"):
                    run_solver("dummy.json")

# 3. Test Convergence
def test_run_solver_convergence_and_debug():
    real_state = make_step4_output_dummy(nx=2, ny=2, nz=2)
    real_input = create_validated_input()
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        
        # FIX: real_input is a real object; its to_dict() just works. 
        # We only need to ensure the mock_context uses our real_input.
        mock_context.input_data = real_input
        
        real_state.ready_for_time_loop = True 
        
        def side_effect_exit(state_in, context_in):
            state_in.ready_for_time_loop = False
            return state_in

        with patch("src.main_solver.orchestrate_step1", return_value=real_state), \
             patch("src.main_solver.orchestrate_step2", return_value=real_state), \
             patch("src.main_solver.orchestrate_step3", return_value=(None, 0.001)), \
             patch("src.main_solver.orchestrate_step4", side_effect=side_effect_exit), \
             patch("src.main_solver.DEBUG", True):
            
            run_solver("dummy.json")

# 4. Test Floating Point Trap
def test_run_solver_floating_point_trap():
    real_state = make_step4_output_dummy()
    real_input = create_validated_input()
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = real_input
        
        mock_context.config.ppe_max_retries = 2
        mock_context.config.dt_min_limit = 1e-6
        
        with patch("src.main_solver.orchestrate_step1", return_value=real_state), \
             patch("src.main_solver.orchestrate_step2", return_value=real_state), \
             patch("src.main_solver.orchestrate_step3", side_effect=FloatingPointError("NaN detected")):
            
            with pytest.raises(RuntimeError, match="CRITICAL INSTABILITY"):
                run_solver("dummy.json")