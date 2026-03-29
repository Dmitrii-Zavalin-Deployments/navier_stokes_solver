# tests/main_solver/test_guards.py

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

# Core imports from the solver
from src.main_solver import _load_simulation_context, main, run_solver

# --- 1. File System Presence Guards ---

def test_load_context_missing_input_file():
    """Validates the solver terminates if the primary input JSON is not found."""
    with patch("src.main_solver.Path.exists") as mock_exists:
        # First check (input_path) returns False
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError, match="Input file missing"):
            _load_simulation_context("non_existent_input.json")

def test_load_context_missing_config_file():
    """Validates the solver terminates if the required config.json is missing."""
    with patch("src.main_solver.Path.exists") as mock_exists:
        # First check (input) is True, Second check (config) is False
        mock_exists.side_effect = [True, False]
        with pytest.raises(FileNotFoundError, match="config.json required"):
            _load_simulation_context("valid_input.json")

# --- 2. CLI Entry Point & Orchestration Guards ---

def test_cli_entrypoint_no_args():
    """
    Validates the Usage prompt and Exit(1) when no path is provided.
    Uses the direct main() call to avoid runpy double-import warnings.
    """
    with patch("sys.argv", ["src/main_solver.py"]), \
         patch("builtins.print") as mock_print:
        
        with pytest.raises(SystemExit) as e:
            main()
            
        assert e.value.code == 1
        mock_print.assert_any_call("Usage: python src/main_solver.py <input_json_path>")

def test_cli_entrypoint_success():
    """
    Validates a successful end-to-end orchestration.
    Ensures SystemExit(0) is raised and the correct archive path is printed.
    """
    # Patch the orchestrator to simulate a successful physics run
    with patch("src.main_solver.run_solver", return_value="mock.zip"), \
         patch("sys.argv", ["src/main_solver.py", "valid.json"]), \
         patch("builtins.print") as mock_print:
        
        with pytest.raises(SystemExit) as e:
            main()
            
        # Verify the success contract
        assert e.value.code == 0
        mock_print.assert_any_call("Pipeline complete. Artifacts archived at: mock.zip")

def test_cli_entrypoint_fatal_error():
    """
    Validates the FATAL error handling path (Exit 1) for unhandled exceptions.
    Ensures that errors are reported to stderr as per Rule 8.
    """
    with patch("src.main_solver.run_solver", side_effect=RuntimeError("System Collapse")), \
         patch("sys.argv", ["src/main_solver.py", "valid.json"]), \
         patch("builtins.print") as mock_print, \
         patch("traceback.print_exc"): # Silent traceback for cleaner test logs
        
        with pytest.raises(SystemExit) as e:
            main()
            
        assert e.value.code == 1
        mock_print.assert_any_call("FATAL PIPELINE ERROR: System Collapse", file=sys.stderr)

def test_solver_heartbeat_logging(tmp_path, caplog):
    """
    Validates Lines 125-127: Ensures the debug heartbeat 
    triggers at iteration 10 with correct formatting.
    
    Compliance:
    - Rule 8: API Minimalism (Clean linter state, no F841).
    - Rule 5: Deterministic Initialization (Uses dummy config).
    """
    import json
    from tests.helpers.solver_input_schema_dummy import get_explicit_solver_config

    # 1. Setup mock environment with a valid schema-compliant config
    input_file = tmp_path / "input.json"
    dummy_config = get_explicit_solver_config(nx=2, ny=2, nz=2)
    input_file.write_text(json.dumps(dummy_config))

    # 2. Patch the entire pipeline to isolate the loop heartbeat logic
    # Note: 'as mock_load' removed to satisfy Ruff F841
    with patch("src.main_solver._load_simulation_context") as mock_load_call, \
         patch("src.main_solver.orchestrate_step1"), \
         patch("src.main_solver.orchestrate_step2") as mock_step2, \
         patch("src.main_solver.orchestrate_step3"), \
         patch("src.main_solver.orchestrate_step4"), \
         patch("src.main_solver.ElasticManager") as mock_elastic_cls, \
         patch("src.main_solver.archive_simulation_artifacts"):

        # Setup Context to pass the jsonschema contract validation
        mock_context = MagicMock()
        mock_context.input_data.to_dict.return_value = dummy_config
        mock_load_call.return_value = mock_context

        # Setup state to trigger the log (Iteration 10)
        mock_state = MagicMock()
        mock_state.ready_for_time_loop = True
        mock_state.iteration = 10  # This triggers the % 10 == 0 check
        mock_state.time = 0.1234
        mock_state.stencil_matrix = [MagicMock()] 
        mock_step2.return_value = mock_state

        # Mock elasticity dt for the log format check
        mock_elastic = mock_elastic_cls.return_value
        mock_elastic.dt = 0.005

        # Kill loop immediately after one pass to prevent infinite recursion
        def stop_loop(*args, **kwargs):
            mock_state.ready_for_time_loop = False
            return mock_state
        mock_state.capture_stable_state.side_effect = stop_loop

        # 3. Execute with DEBUG level enabled to catch the heartbeat
        with caplog.at_level(logging.DEBUG, logger="Solver.Main"):
            run_solver(str(input_file))

        # 4. Verification: Check the forensic string format
        # Matches f"Step {state.iteration} | Time {state.time:.4f} | dt {elasticity.dt:.2e}"
        assert "Step 10 | Time 0.1234 | dt 5.00e-03" in caplog.text