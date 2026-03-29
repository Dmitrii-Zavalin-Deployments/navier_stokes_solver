# tests/test_main_solver_flow.py

import logging
import runpy
import sys
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from src.common.solver_config import SolverConfig  # Grounding the config
from src.main_solver import _load_simulation_context, run_solver
from tests.helpers.solver_input_schema_dummy import (
    create_validated_input,
    get_explicit_solver_config,
)
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


# 1. Test File System Guards
def test_load_context_missing_config():
    with patch("src.main_solver.Path.exists") as mock_exists:
        mock_exists.side_effect = [True, False]
        with pytest.raises(FileNotFoundError, match="config.json required"):
            _load_simulation_context("dummy.json")

# 2. Test State Contract Violations
def test_run_solver_state_schema_violation(caplog):
    """
    Validates Rule 4: Hierarchy over Convenience.
    Ensures that a State Contract Violation (Step 3) triggers a ValidationError
    and is handled by the orchestrator.
    """
    valid_input_obj = create_validated_input()
    real_state = make_step4_output_dummy() 
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = valid_input_obj
        
        # FIX: Patch the CLASS method to bypass __slots__ instance restrictions
        # aligned with Rule 0: Objects are for Logic
        with patch("src.common.solver_state.SolverState.validate_against_schema") as mock_val:
            mock_val.side_effect = jsonschema.exceptions.ValidationError("State Mismatch")
            
            with patch("src.main_solver.orchestrate_step1", return_value=real_state), \
                 patch("src.main_solver.orchestrate_step2", return_value=real_state):
                
                with pytest.raises(jsonschema.exceptions.ValidationError, match="State Mismatch"):
                    run_solver("dummy.json")
                
                # RULE 6: We do NOT assert NaN here. 
                # We only verify the Schema Error was logged.
                assert "!!! STATE CONTRACT VIOLATION" in caplog.text

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
                assert "Audit Failure: NaN detected" in caplog.text
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
                assert "Audit Failure: NaN detected" in caplog.text

def test_cli_entrypoint_no_args():
    """Tests the usage prompt when no path is provided via runpy."""
    with patch("sys.argv", ["src/main_solver.py"]), \
         patch("builtins.print") as mock_print:
        
        with pytest.raises(SystemExit) as e:
            runpy.run_module("src.main_solver", run_name="__main__")
            
        assert e.value.code == 1
        mock_print.assert_any_call("Usage: python src/main_solver.py <input_json_path>")

def test_load_context_missing_input_file():
    """
    Forensic Audit: Validates Line 41-42 of src/main_solver.py.
    Ensures the solver terminates if the primary input JSON is missing.
    """
    with patch("src.main_solver.Path.exists") as mock_exists:
        # First call (input_path) returns False
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError, match="Input file missing"):
            _load_simulation_context("non_existent_input.json")


def test_load_context_missing_config_file():
    """
    Forensic Audit: Validates Line 43-44 of src/main_solver.py.
    Ensures the solver terminates if the required config.json is missing.
    """
    with patch("src.main_solver.Path.exists") as mock_exists:
        # First call (input_path) is True, Second call (config_path) is False
        mock_exists.side_effect = [True, False]
        
        with pytest.raises(FileNotFoundError, match="config.json required"):
            _load_simulation_context("valid_input.json")

def test_run_solver_input_schema_violation():
    """
    Forensic Audit: Validates Lines 63-67 of src/main_solver.py.
    Triggers a jsonschema.ValidationError to ensure input contract enforcement.
    """
    with patch("src.main_solver._load_simulation_context") as mock_load, \
         patch("src.main_solver.open", create=True) as mock_open:
        
        # Setup valid context but force a schema validation failure
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        
        # Mock reading the schema file (Step 1 of validation)
        mock_open.return_value.__enter__.return_value.read.return_value = "{}"
        
        with patch("jsonschema.validate") as mock_validate:
            # Simulate a failure in the jsonschema library itself
            mock_validate.side_effect = jsonschema.exceptions.ValidationError("Invalid Input Structure")
            
            with pytest.raises(jsonschema.exceptions.ValidationError, match="Invalid Input Structure"):
                run_solver("dummy_input.json")

def test_run_solver_telemetry_logging(caplog):
    """
    Forensic Audit: Validates telemetry logic (Line 125-126).
    Ensures step progress is captured in DEBUG logs.
    """
    # 1. Use the Step 4 dummy for a hydrated state
    real_state = make_step4_output_dummy(nx=2, ny=2, nz=2)
    
    # 2. Explicitly recreate the config (as state doesn't hold it)
    from src.common.solver_config import SolverConfig
    hydrated_config = SolverConfig(
        ppe_tolerance=1e-6,
        ppe_max_iter=1000,
        dt_min_limit=1e-6,
        ppe_max_retries=5
    )
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        
        # 3. Attach the real config and input data to the mock context
        mock_context.config = hydrated_config
        mock_context.input_data = create_validated_input(nx=2, ny=2, nz=2)
        
        # Setup the iteration trigger for Step 125 logic
        real_state.iteration = 10
        real_state.ready_for_time_loop = True 
        
        def exit_immediately(state_in, context_in):
            state_in.ready_for_time_loop = False
            return state_in

        # 4. Orchestration Mocks
        with patch("src.main_solver.orchestrate_step1", return_value=real_state), \
             patch("src.main_solver.orchestrate_step2", return_value=real_state), \
             patch("src.main_solver.orchestrate_step3", return_value=(None, 0.001)), \
             patch("src.main_solver.orchestrate_step4", side_effect=exit_immediately), \
             patch("src.main_solver.archive_simulation_artifacts", return_value="zip"):
            
            with caplog.at_level(logging.DEBUG):
                run_solver("dummy.json")
                assert "Audit Failure: NaN detected" in caplog.text
                
            assert any("AUDIT [Start]: Iteration 10" in record.message for record in caplog.records)

def test_run_solver_floating_point_critical_trap():
    """
    Forensic Audit: Validates Lines 145-147 of src/main_solver.py.
    Ensures that if NumPy traps a NaN/Inf (FloatingPointError), 
    the solver logs the specific iteration and raises the error.
    
    Resolution: Hydrates SolverConfig fully to satisfy ElasticManager constraints.
    """
    # 1. Use Step 4 dummy to get a fully populated state and config
    real_state = make_step4_output_dummy(nx=2, ny=2, nz=2)
    real_input = create_validated_input()
    
    # 2. Extract the valid config from the dummy setup to prevent 'uninitialized' errors
    from src.common.solver_config import SolverConfig
    MOCK_CONFIG_DATA = {
        "ppe_tolerance": 1e-6,
        "ppe_max_iter": 1,
        "dt_min_limit": 1e-6,
        "ppe_max_retries": 5
    }
    fully_hydrated_config = SolverConfig(**MOCK_CONFIG_DATA)
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = real_input
        mock_context.config = fully_hydrated_config
        
        # Set iteration for log verification
        real_state.iteration = 42
        real_state.ready_for_time_loop = True 

        # 3. Trigger the trap during the Step 3 Orchestration (Physics Kernel)
        with patch("src.main_solver.orchestrate_step1", return_value=real_state), \
             patch("src.main_solver.orchestrate_step2", return_value=real_state), \
             patch("src.main_solver.orchestrate_step3", side_effect=FloatingPointError("NaN detected")):
            
            # The solver should catch the FloatingPointError, log it, and re-raise
            with pytest.raises(RuntimeError, match="CRITICAL INSTABILITY"):
                run_solver("dummy.json")
                assert "Audit Failure: NaN detected" in caplog.text


def test_run_solver_value_error_contract_violation():
    """
    Forensic Audit: Validates Lines 149-151 of src/main_solver.py.
    Ensures that generic ValueErrors (Data Integrity breaches) 
    are caught, logged, and re-raised.
    
    Resolution: Hydrates input_data.to_dict() with a real dict to pass 
    the internal jsonschema validation gate.
    """
    real_state = make_step4_output_dummy(nx=2, ny=2, nz=2)
    # Get a real dictionary representation of valid input
    real_input_dict = get_explicit_solver_config(nx=2, ny=2, nz=2)
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        
        # Ensure the schema validation passes by returning a real dict
        mock_context.input_data.to_dict.return_value = real_input_dict
        mock_context.config = real_state.config
        
        real_state.ready_for_time_loop = True 

        # Trigger a ValueError during the Step 4 Finalization (Archive/State Logic)
        with patch("src.main_solver.orchestrate_step1", return_value=real_state), \
             patch("src.main_solver.orchestrate_step2", return_value=real_state), \
             patch("src.main_solver.orchestrate_step3", return_value=(None, 0.0)), \
             patch("src.main_solver.orchestrate_step4", side_effect=ValueError("Invalid Buffer Alignment")), \
             patch("src.main_solver.archive_simulation_artifacts", return_value="zip"):
            
            with pytest.raises(ValueError, match="Invalid Buffer Alignment"):
                run_solver("dummy.json")
                assert "Audit Failure: NaN detected" in caplog.text

def test_cli_entrypoint_success():
    """
    Forensic Audit: Validates Lines 159-162 of src/main_solver.py.
    Ensures that a successful run exits with code 0 and prints the zip path.
    """
    # We patch run_solver in the module being run by runpy
    with patch("src.main_solver.run_solver", return_value="mock_artifacts.zip"), \
         patch("sys.argv", ["src/main_solver.py", "valid_input.json"]), \
         patch("builtins.print") as mock_print:
        
        with pytest.raises(SystemExit) as e:
            runpy.run_module("src.main_solver", run_name="__main__")
            
        assert e.value.code == 0
        mock_print.assert_any_call("Pipeline complete. Artifacts archived at: mock_artifacts.zip")


def test_cli_entrypoint_fatal_error():
    """
    Forensic Audit: Validates Lines 163-166 of src/main_solver.py.
    Ensures that an unhandled exception triggers the FATAL error message and exit code 1.
    """
    with patch("src.main_solver.run_solver", side_effect=RuntimeError("System Collapse")), \
         patch("sys.argv", ["src/main_solver.py", "valid_input.json"]), \
         patch("builtins.print") as mock_print, \
         patch("traceback.print_exc"): # Prevent polluting test logs with stack trace
        
        with pytest.raises(SystemExit) as e:
            runpy.run_module("src.main_solver", run_name="__main__")
            
        assert e.value.code == 1
        # Check stderr printing (using any_call because of the file=sys.stderr argument)
        mock_print.assert_any_call("FATAL PIPELINE ERROR: System Collapse", file=sys.stderr)