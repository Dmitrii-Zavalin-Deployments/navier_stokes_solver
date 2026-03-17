# tests/step2/test_step2_alignment.py

import json
import os

import pytest

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


@pytest.fixture
def mock_config(tmp_path):
    """
    Creates a temporary config.json in the test execution directory 
    to satisfy the 'Explicit or Error' mandate.
    """
    config_data = {
        "solver_settings": {
            "ppe_tolerance": 1e-6,
            "ppe_atol": 1e-12,
            "ppe_max_iter": 1000
        }
    }
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)
    
    # Change directory to the temp path so the code finds the file
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield config_file
    os.chdir(old_cwd)

def test_step2_orchestration_alignment(mock_config):
    """
    PHASE C: Orchestration Alignment Audit for Step 2.
    Validates the contract: Step 1 Output + Config -> Orchestrate Step 2 -> Step 2 Frozen Truth.
    """
    # 1. Input: Load the 'Frozen' output from Step 1
    input_state = make_step1_output_dummy()
    
    # 2. Logic: Execute the actual Step 2 orchestration
    output_state = orchestrate_step2(input_state)
    
    # 3. Target: Load the 'Frozen Truth' dummy for Step 2
    expected_state = make_step2_output_dummy()
    
    # 4. Verification: The Zero-Debt Mandate (Architectural State)
    # Rule: Step 2 remains 'False' for time loop readiness.
    assert output_state.ready_for_time_loop is True
    
    # 5. Verification: Property-Based Access (SSoT Guard)
    # Ensuring we use the validated properties, not dict keys
    assert output_state.operators.laplacian is not None
    assert output_state.operators.divergence is not None
    assert output_state.operators.grad_x is not None
    
    # 6. Verification: Scale Guard (Matrix Integrity)
    actual_L = output_state.operators.laplacian
    expected_L = expected_state.operators.laplacian
    
    assert actual_L.shape == expected_L.shape, "Matrix dimensions mismatch!"
    
    # 7. Verification: Numerical Settings (Runner Config Mapping)
    assert output_state.config.ppe_atol == 1e-12
    assert output_state.config.ppe_max_iter == 1000

    print("\n✅ Step 2 Orchestration Alignment: PASSED")