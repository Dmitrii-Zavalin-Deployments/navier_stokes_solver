# tests/conftest.py

import json
import shutil
from pathlib import Path

import pytest

from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


@pytest.fixture(scope="session")
def test_env_root():
    """
    Rule 4: SSoT for Test Directories.
    Provides a consistent root for all test artifacts.
    """
    root = Path("data/testing-input-output")
    root.mkdir(parents=True, exist_ok=True)
    yield root
    # Zero-Debt: Cleanup after the entire test session
    if root.exists():
        shutil.rmtree(root)

@pytest.fixture
def clean_output_dir():
    """Ensures a fresh output directory for each test run."""
    output_path = Path("output")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    yield output_path
    if output_path.exists():
        shutil.rmtree(output_path)

@pytest.fixture
def sample_input_file(test_env_root):
    """
    Point 5: Deterministic Input Generator.
    Creates a standard 4x4x4 simulation contract for integration testing.
    """
    input_path = test_env_root / "contract_input.json"
    
    # Generate standard 3-step contract
    data = solver_input_schema_dummy()
    dt = data["simulation_parameters"]["time_step"]
    data["simulation_parameters"]["total_time"] = dt * 3
    
    with open(input_path, "w") as f:
        json.dump(data, f)
        
    return str(input_path)

@pytest.fixture
def mock_solver_state():
    """
    Returns a fresh SolverState for unit tests that don't need a full file-load.
    """
    from src.solver_input import SolverInput
    from src.step1.orchestrate_step1 import orchestrate_step1
    
    raw_data = solver_input_schema_dummy()
    input_obj = SolverInput.from_dict(raw_data)
    return orchestrate_step1(input_obj)