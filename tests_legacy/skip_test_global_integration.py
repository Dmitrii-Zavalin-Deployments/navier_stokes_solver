# tests/test_global_integration.py

import json
import os
import shutil
from pathlib import Path

import pytest

from src.main_solver import run_solver_from_file
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


@pytest.fixture
def setup_integration_env():
    """Sets up and cleans up the test environment."""
    test_dir = Path("data/testing-input-output")
    test_dir.mkdir(parents=True, exist_ok=True)
    input_file = test_dir / "integration_test_input.json"
    
    # 1. Generate a valid input dictionary (Phase A Contract)
    # We set total_time = 3 * dt to ensure exactly 3 iterations
    input_data = solver_input_schema_dummy()
    dt = input_data["simulation_parameters"]["time_step"]
    input_data["simulation_parameters"]["total_time"] = dt * 3
    
    with open(input_file, "w") as f:
        json.dump(input_data, f)
        
    yield str(input_file)
    
    # Cleanup after test
    if test_dir.exists():
        shutil.rmtree(test_dir)
    if os.path.exists("output"):
        shutil.rmtree("output")

def test_full_pipeline_execution(setup_integration_env):
    """
    Global Integration Audit:
    Input File -> MainSolver -> 3 Iterations -> ZIP Archive.
    """
    input_path = setup_integration_env
    
    # 1. Execute the full solver
    # This triggers Steps 1-2 once, and Steps 3-5 three times.
    zip_result_path = run_solver_from_file(input_path)
    
    # 2. Validation: Archive Existence
    assert os.path.exists(zip_result_path), "CRITICAL: Solver failed to produce a ZIP archive."
    assert zip_result_path.endswith(".zip")
    
    # 3. Validation: Internal Data Integrity (Post-Mortem)
    # We temporarily unzip to check the 'Final Brain Snapshot'
    extract_dir = Path("data/testing-input-output/temp_verify")
    shutil.unpack_archive(zip_result_path, extract_dir)
    
    snapshot_path = extract_dir / "final_state_snapshot.json"
    assert snapshot_path.exists(), "CRITICAL: Final state snapshot missing from archive."
    
    with open(snapshot_path) as f:
        final_state = json.load(f)
        
    # Verify iteration count (3 steps)
    assert final_state["iteration"] == 3
    # Verify Chronos Guard synced the time exactly
    assert final_state["time"] == pytest.approx(final_state["iteration"] * 0.01)
    # Verify SSoT: Fields exist in the final state
    assert "U" in final_state["fields"]
    assert "P_ext" in final_state["fields"]
    
    print(f"\n✅ Global Integration Test: PASSED. Archive created at {zip_result_path}")