# tests/common/test_elasticity_manager.py

from types import SimpleNamespace

import numpy as np
import pytest

from src.common.elasticity import ElasticManager
from src.common.field_schema import FI

# ----------------------------------------------------------------ER
# 1. MOCK OBJECTS FOR ISOLATED TESTING
# ----------------------------------------------------------------

class MockConfig:
    def __init__(self):
        self.dt_min_limit = 0.001
        self.divergence_threshold = 1e6
        self.ppe_omega = 1.7
        self.ppe_max_iter = 50

@pytest.fixture
def config():
    return MockConfig()

@pytest.fixture
def state():
    """Creates a mock state object with a data buffer."""
    # We need at least enough columns to cover the indices in FI
    # VX_STAR, VY_STAR, VZ_STAR, P_NEXT are usually higher indices
    data = np.zeros((10, 20)) 
    fields = SimpleNamespace(data=data)
    return SimpleNamespace(fields=fields)

def trigger_instability(state):
    """Injects divergent values into the audit fields."""
    state.fields.data[:, FI.VX_STAR] = 1e12  # Above threshold

def trigger_stability(state):
    """Sets sane values in the audit fields."""
    state.fields.data[:, FI.VX_STAR] = 1.0
    state.fields.data[:, FI.VY_STAR] = 1.0
    state.fields.data[:, FI.VZ_STAR] = 1.0
    state.fields.data[:, FI.P_NEXT] = 1.0

# ----------------------------------------------------------------
# 2. THE UNIT TEST SUITE
# ----------------------------------------------------------------

def test_monotonic_decay_under_persistent_failure(config, state):
    """
    FIX FOR CURRENT ERROR: Ensures dt ONLY goes down if math stays bad.
    This test reproduces your log pattern to ensure it cannot hover or increase.
    """
    manager = ElasticManager(config, initial_dt=0.5)
    trigger_instability(state)
    
    last_dt = manager.dt
    
    # Simulate 20 failed attempts
    for i in range(20):
        try:
            success = manager.sync_state(state)
            assert not success, f"Iteration {i}: Should have failed but returned True"
            
            # THE CORE CHECK: dt must strictly decrease
            assert manager.dt < last_dt, f"Ratchet Failed! dt {manager.dt} is not less than {last_dt}"
            last_dt = manager.dt
            
        except RuntimeError:
            # This is the expected final exit when dt < dt_floor
            return 

    pytest.fail("Should have raised RuntimeError after persistent failure")

def test_runtime_error_at_floor(config, state):
    """Ensures we actually hit the floor and raise the error for pytest."""
    config.dt_min_limit = 0.1
    manager = ElasticManager(config, initial_dt=0.25)
    trigger_instability(state)
    
    # Step 1: 0.25 -> 0.125
    manager.sync_state(state)
    
    # Step 2: 0.125 -> 0.0625 (This is below 0.1) -> SHOUD RAISE
    with pytest.raises(RuntimeError, match="dropped below floor"):
        manager.sync_state(state)

def test_recovery_lockout_during_unstable_retries(config, state):
    """
    Ensures that _iteration increments ONLY on True results.
    Prevents recovery from triggering while sync_state returns False.
    """
    manager = ElasticManager(config, initial_dt=0.5)
    
    # Fail 5 times
    trigger_instability(state)
    for _ in range(5):
        manager.sync_state(state)
        assert manager._iteration == 0, "Counter should reset on every False return"
    
    # Succeed once
    trigger_stability(state)
    manager.sync_state(state)
    assert manager._iteration == 1, "Counter should be 1 after one success"

def test_omega_and_max_iter_reset(config, state):
    """Verifies PPE parameters tighten on panic and loosen on full recovery."""
    manager = ElasticManager(config, initial_dt=0.5)
    
    # Trigger Panic
    trigger_instability(state)
    manager.sync_state(state)
    
    assert manager.omega < config.ppe_omega
    assert manager.max_iter == 5000
    
    # Successful streak to recover
    trigger_stability(state)
    for _ in range(10): # Counter for recovery in your code is 10
        manager.sync_state(state)
    
    # After recovery streak, max_iter should return to config value
    assert manager.max_iter == config.ppe_max_iter

def test_inf_nan_validation(config, state):
    """Edge case: Field contains NaN or Inf."""
    manager = ElasticManager(config, initial_dt=0.5)
    
    # Test NaN
    state.fields.data[0, FI.VX_STAR] = np.nan
    assert manager.sync_state(state) is False
    
    # Test Inf
    state.fields.data[0, FI.VX_STAR] = np.inf
    assert manager.sync_state(state) is False

def test_p_next_audit(config, state):
    """Ensures pressure divergence also triggers panic."""
    manager = ElasticManager(config, initial_dt=0.5)
    trigger_stability(state)
    state.fields.data[:, FI.P_NEXT] = 1e9 # Divergent pressure
    
    assert manager.sync_state(state) is False