"""
Test suite for src/main.py execution control plane.
This narrative test module verifies parameter enforcement, file validation gates, 
and critical failure-manifest archival error handling under non-default policies.
"""

import pytest
from pathlib import Path
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
    
    # We temporarily repoint BASE_DIR in src.main to a clean temporary path that lacks a config folder.
    import src.main as main_module
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
    When the simulation encounters an unrecoverable failure and attempts to write a failure manifest via the archivist,
    if the archiving function itself raises an exception, the critical error handler must catch it,
    log it via logger.critical, and successfully re-raise the original simulation error.
    """
    # We mock load_and_validate_inputs to raise a RuntimeError, simulating a core pipeline failure.
    def mock_load_fail(*args, **kwargs):
        raise RuntimeError("Simulated core pipeline execution failure")

    monkeypatch.setattr("src.main.load_and_validate_inputs", mock_load_fail)

    # We mock archive_simulation_results so that when it attempts to write the FAILURE manifest, it also throws an error.
    def mock_archive_fail(*args, **kwargs):
        raise ValueError("Simulated archivist failure while writing error log")

    monkeypatch.setattr("src.main.archive_simulation_results", mock_archive_fail)

    # We verify that despite the secondary archiving error triggering lines 101-102, 
    # the original RuntimeError is ultimately re-raised to the caller.
    with pytest.raises(RuntimeError, match="Simulated core pipeline execution failure"):
        run_simulation(tmp_path, "dummy.json", "output.json")
