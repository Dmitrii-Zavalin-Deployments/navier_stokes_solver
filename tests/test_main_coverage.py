"""
Test suite for src/main.py execution control plane.
This narrative test module verifies parameter enforcement, file validation gates, 
and critical failure-manifest archival error handling under non-default policies.
"""

import importlib

import pytest

from src.main import run_simulation


def test_run_simulation_missing_input_output_folder():
    """
    When the input_output_folder argument is omitted or set to None, 
    the control plane must enforce the non-default policy and raise a ValueError.
    """
    # We verify that passing None for input_output_folder triggers the explicit ValueError check.
    with pytest.raises(ValueError, match="FATAL ERROR: input_output_folder must be explicitly provided"):
        run_simulation(None, "input.json", "output.json")


def test_run_simulation_missing_input_file_name():
    """
    When the input_file_name argument is omitted or set to None, 
    the control plane must reject it and raise a ValueError.
    """
    # We verify that passing None for input_file_name triggers the explicit ValueError check.
    with pytest.raises(ValueError, match="FATAL ERROR: input_file_name must be explicitly provided"):
        run_simulation("some_folder", None, "output.json")


def test_run_simulation_missing_output_file_name():
    """
    When the output_file_name argument is omitted or set to None, 
    the control plane must enforce the requirement and raise a ValueError.
    """
    # We verify that passing None for output_file_name triggers the explicit ValueError check.
    with pytest.raises(ValueError, match="FATAL ERROR: output_file_name must be explicitly provided"):
        run_simulation("some_folder", "input.json", None)


def test_run_simulation_input_file_not_found(tmp_path):
    """
    When the specified input configuration file does not exist within the designated folder,
    the execution plane must intercept the missing file and raise a FileNotFoundError.
    """
    # We define a non-existent input file name inside the temporary directory path.
    non_existent_input = "non_existent_config.json"
    
    # We verify that execution fails with FileNotFoundError pointing to the missing path.
    with pytest.raises(FileNotFoundError, match="Input configuration file not found at"):
        run_simulation(tmp_path, non_existent_input, "output.json")


def test_run_simulation_system_config_not_found(tmp_path):
    """
    When the baseline system configuration file ('config/config.json') is missing from the repository base directory,
    the execution plane must catch this state and raise a FileNotFoundError.
    """
    # We create a dummy input file so that the input check passes its initial validation gate.
    dummy_input = tmp_path / "valid_input.json"
    dummy_input.write_text("{}")
    
    # We explicitly load the src.main module via importlib to avoid package-level function shadowing.
    main_module = importlib.import_module("src.main")
    original_base_dir = main_module.BASE_DIR
    main_module.BASE_DIR = tmp_path

    try:
        # We assert that the absence of BASE_DIR/config/config.json correctly triggers FileNotFoundError.
        with pytest.raises(FileNotFoundError, match="Configuration file not found at"):
            run_simulation(tmp_path, dummy_input.name, "output.json")
    finally:
        # We restore the original base directory to maintain test isolation.
        main_module.BASE_DIR = original_base_dir


def test_run_simulation_archive_failure_manifest_error(tmp_path, monkeypatch):
    """
    When the simulation encounters an unrecoverable failure mid-execution 
    and the subsequent failure-manifest archival also fails, the critical error 
    handler successfully catches it, logs via logger.critical (lines 101-102), 
    and re-raises the original error.
    """
    # Ensure the input file exists so it passes initial validation
    dummy_input = tmp_path / "dummy.json"
    dummy_input.write_text("{}")

    main_module = importlib.import_module("src.main")

    # Mock step_simulation to fail INSIDE the active simulation try-block (line 64)
    # This guarantees 'state' is instantiated and the except block (lines 91-103) is entered.
    def mock_step_fail(*args, **kwargs):
        raise RuntimeError("Simulated mid-simulation physical instability")

    monkeypatch.setattr(main_module, "step_simulation", mock_step_fail)

    # Mock archive_simulation_results so that writing the FAILURE manifest throws an error (lines 95-100)
    def mock_archive_fail(*args, **kwargs):
        raise ValueError("Simulated archivist failure while writing error log")

    monkeypatch.setattr(main_module, "archive_simulation_results", mock_archive_fail)

    # Verifies that lines 101-102 execute (logging critical) and the original error bubbles up
    with pytest.raises(RuntimeError, match="Simulated mid-simulation physical instability"):
        run_simulation(tmp_path, dummy_input.name, "output.json")
