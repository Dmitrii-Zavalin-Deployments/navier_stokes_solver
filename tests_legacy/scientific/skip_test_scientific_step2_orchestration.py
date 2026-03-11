# tests/scientific/test_scientific_step2_orchestration.py

import json
import os

import pytest

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


@pytest.fixture
def mock_config(tmp_path):
    """Creates a temporary config.json to satisfy the orchestrator's IO requirements."""
    config_path = tmp_path / "config.json"
    data = {
        "solver_settings": {
            "ppe_atol": 1e-12,
            "ppe_tolerance": 1e-10,
            "ppe_max_iter": 500
        }
    }
    with open(config_path, "w") as f:
        json.dump(data, f)
    
    # Change working directory to temp path for the test
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield config_path
    os.chdir(old_cwd)

@pytest.fixture
def fresh_state():
    """Provides a fully hydrated SolverState using the centralized dummy factory."""
    # nx=4, ny=4, nz=4 matches the requirements of your orchestration tests
    return make_step1_output_dummy(nx=4, ny=4, nz=4)

def test_scientific_orchestration_io_success(mock_config, fresh_state):
    """Rule 2.1: Verify config.json hydration and state transition."""
    state = orchestrate_step2(fresh_state)
    
    # Verify triad initialization
    assert state.config.ppe_atol == 1e-12
    assert state.config.ppe_tolerance == 1e-10
    assert state.config.ppe_max_iter == 500
    
    # Verify worker delegation occurred
    assert state.operators.laplacian is not None
    assert state.advection.indices is not None
    assert state.ready_for_time_loop is True

def test_scientific_orchestration_health_baseline(mock_config, fresh_state):
    """Rule 2.2: Verify Health Vitals are zeroed out before time-stepping."""
    state = orchestrate_step2(fresh_state)
    
    assert state.health.max_u == 0.0
    assert state.health.divergence_norm == 0.0
    assert state.health.is_stable is True
    assert state.health.post_correction_divergence_norm == 0.0

def test_scientific_orchestration_missing_config(tmp_path, fresh_state):
    """Rule 2.3: Verify robust failure when config.json is absent."""
    os.chdir(tmp_path) 
    with pytest.raises(FileNotFoundError, match="Critical Error: 'config.json' not found"):
        orchestrate_step2(fresh_state)

def test_scientific_orchestration_malformed_config(tmp_path, fresh_state):
    """Rule 2.4: Verify JSONDecodeError handling with detailed context."""
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        f.write("NOT_JSON{!!}")
    
    os.chdir(tmp_path)
    # Match against the improved error logging in orchestrate_step2
    with pytest.raises(ValueError, match="Critical Error: 'config.json' is not a valid JSON file"):
        orchestrate_step2(fresh_state)

def test_scientific_orchestration_missing_triad_key(tmp_path, fresh_state):
    """Rule 2.5: Verify KeyError when mandatory triad settings are missing."""
    config_path = tmp_path / "config.json"
    data = {"solver_settings": {"ppe_atol": 1e-12}} # Missing max_iter and tolerance
    with open(config_path, "w") as f:
        json.dump(data, f)
    
    os.chdir(tmp_path)
    with pytest.raises(KeyError, match="Critical Error: Missing required solver setting"):
        orchestrate_step2(fresh_state)

def test_scientific_ppe_handshake(mock_config, fresh_state):
    """Rule 2.6: Verify the critical L -> PPE._A pointer assignment and _preconditioner reset."""
    state = orchestrate_step2(fresh_state)
    
    # Verify pointers are synchronized
    assert state.ppe._A is state.operators.laplacian
    assert state.ppe._preconditioner is None

def test_scientific_operator_integrity(mock_config, fresh_state):
    """Rule 2.7: Ensure matrices are not just initialized, but populated with physics."""
    state = orchestrate_step2(fresh_state)
    
    # 4x4x4 grid results in a 64x64 Laplacian
    # Ensure it is populated (non-zero elements exist)
    assert state.operators.laplacian.nnz > 0
    assert state.operators.laplacian.shape == (64, 64)