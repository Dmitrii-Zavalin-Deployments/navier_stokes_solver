# tests/test_main_solver_flow.py

import importlib
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from src.common.solver_config import SolverConfig  # Grounding the config
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
    
    real_config = SolverConfig(
        ppe_tolerance=1e-1,
        ppe_max_iter=5,
        dt_min_limit=1e-6,
        ppe_max_retries=5
    )
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = real_input
        mock_context.config = real_config
        
        real_state.ready_for_time_loop = True 
        
        def side_effect_exit(state_in, context_in):
            state_in.ready_for_time_loop = False
            return state_in

        # ADDED: patch("src.main_solver.archive_simulation_artifacts")
        with patch("src.main_solver.orchestrate_step1", return_value=real_state), \
             patch("src.main_solver.orchestrate_step2", return_value=real_state), \
             patch("src.main_solver.orchestrate_step3", return_value=(None, 0.001)), \
             patch("src.main_solver.orchestrate_step4", side_effect=side_effect_exit), \
             patch("src.main_solver.archive_simulation_artifacts", return_value="mock_path.zip"), \
             patch("src.main_solver.DEBUG", True):
            
            result = run_solver("dummy.json")
            assert result == "mock_path.zip"

# 4. Test Floating Point Trap
def test_run_solver_floating_point_trap():
    real_state = make_step4_output_dummy()
    real_input = create_validated_input()
    
    # GROUNDING: Same for the trap test
    real_config = SolverConfig(
        ppe_max_retries=2,
        dt_min_limit=1e-6
    )
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = real_input
        mock_context.config = real_config
        
        with patch("src.main_solver.orchestrate_step1", return_value=real_state), \
             patch("src.main_solver.orchestrate_step2", return_value=real_state), \
             patch("src.main_solver.orchestrate_step3", side_effect=FloatingPointError("NaN detected")):
            
            with pytest.raises(RuntimeError, match="CRITICAL INSTABILITY"):
                run_solver("dummy.json")

def test_cli_entrypoint_success():
    """
    Forces execution of the 'if __name__ == "__main__":' block 
    to achieve 100% coverage on src/main_solver.py.
    """
    # 1. Mock the command line arguments
    # [sys.argv[0] is the script name, sys.argv[1] is the input path]
    test_args = ["src/main_solver.py", "dummy_input.json"]
    
    with patch("sys.argv", test_args), \
         patch("src.main_solver.run_solver") as mock_run, \
         patch("builtins.print") as mock_print:
        
        # Define what the solver returns on success
        mock_run.return_value = "mock_output.zip"
        
        # 2. Re-import or Reload the module to trigger the __main__ block
        import src.main_solver
        with pytest.raises(SystemExit) as e:
            importlib.reload(src.main_solver)
        
        # 3. Assertions
        assert e.value.code == 0  # sys.exit(0)
        mock_run.assert_called_once_with("dummy_input.json")
        mock_print.assert_any_call("Pipeline complete. Artifacts archived at: mock_output.zip")

def test_cli_entrypoint_no_args():
    """Tests the 'Usage' branch when no arguments are provided."""
    with patch("sys.argv", ["src/main_solver.py"]), \
         patch("builtins.print") as mock_print:
        
        import src.main_solver
        with pytest.raises(SystemExit) as e:
            importlib.reload(src.main_solver)
            
        assert e.value.code == 1  # sys.exit(1)
        mock_print.assert_any_call("Usage: python src/main_solver.py <input_json_path>")

def test_cli_entrypoint_error():
    """Tests the 'FATAL PIPELINE ERROR' branch."""
    with patch("sys.argv", ["src/main_solver.py", "bad_input.json"]), \
         patch("src.main_solver.run_solver") as mock_run, \
         patch("traceback.print_exc"):
        
        mock_run.side_effect = Exception("System Crash")
        
        import src.main_solver
        with pytest.raises(SystemExit) as e:
            importlib.reload(src.main_solver)
            
        assert e.value.code == 1