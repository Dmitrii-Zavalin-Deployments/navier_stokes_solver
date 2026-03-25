# tests/common/test_elasticity_manager.py

import pytest
import numpy as np
from types import SimpleNamespace
from src.common.elasticity import ElasticManager
from src.common.field_schema import FI
from tests.helpers.solver_input_schema_dummy import create_validated_input

# ----------------------------------------------------------------
# 1. MOCK OBJECTS & FIXTURES
# ----------------------------------------------------------------

class MockConfig:
    """Combines high-level solver config and minimal physics config."""
    def __init__(self):
        # Physics / Elasticity limits
        self.dt_initial = 0.5
        self.dt_min_limit = 0.001  # For sync_state logic
        self.dt_min = 1e-6         # For stabilization logic (Safety Floor)
        self.reduction_factor = 0.5
        
        # PPE parameters
        self.divergence_threshold = 1e6
        self.ppe_omega = 1.7
        self.ppe_max_iter = 50

@pytest.fixture
def config():
    return MockConfig()

@pytest.fixture
def state():
    """Creates a mock state object with a data buffer for FI indices."""
    data = np.zeros((10, 20)) 
    fields = SimpleNamespace(data=data)
    return SimpleNamespace(fields=fields)

def trigger_instability(state):
    """Injects divergent values into the audit fields."""
    state.fields.data[:, FI.VX_STAR] = 1e12

def trigger_stability(state):
    """Sets sane values in the audit fields."""
    for field in [FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR, FI.P_NEXT]:
        state.fields.data[:, field] = 1.0

# ----------------------------------------------------------------
# 2. THE INTEGRATED TEST SUITE
# ----------------------------------------------------------------

def test_manual_stabilization_reduction_logic(config):
    """
    STABILITY AUDIT: Verifies manual dt reduction (triggered by main_solver except block).
    """
    # Using the dummy helper for real-world context validation
    dummy_input = create_validated_input() 
    manager = ElasticManager(config, dummy_input)
    
    initial_dt = manager.dt
    # Simulate a manual trigger from an ArithmeticError in the solver
    manager.stabilization(is_needed=True)
    
    assert manager.dt == initial_dt * config.reduction_factor
    assert manager.dt < initial_dt

def test_manual_stabilization_safety_floor(config):
    """
    STABILITY AUDIT: Verifies manual reduction never passes the absolute floor.
    """
    config.dt_initial = 1e-5
    config.dt_min = 1e-6 # Absolute floor
    manager = ElasticManager(config, None)

    # Force excessive reductions
    for _ in range(10):
        manager.stabilization(is_needed=True)

    assert manager.dt == config.dt_min, f"dt failed to clamp at floor: {manager.dt}"

def test_sync_state_monotonic_decay(config, state):
    """
    Ensures dt strictly decreases during automated state inspection if math is bad.
    """
    manager = ElasticManager(config, initial_dt=0.5)
    trigger_instability(state)
    
    last_dt = manager.dt
    for _ in range(5):
        try:
            success = manager.sync_state(state)
            assert not success
            assert manager.dt < last_dt
            last_dt = manager.dt
        except RuntimeError:
            return # Reached floor successfully

def test_omega_and_max_iter_adaptation(config, state):
    """Verifies PPE parameters tighten on panic and recover on success."""
    manager = ElasticManager(config, initial_dt=0.5)
    
    # Trigger Panic via Automated Sync
    trigger_instability(state)
    manager.sync_state(state)
    
    assert manager.omega < config.ppe_omega, "Omega should tighten during instability"
    assert manager.max_iter > config.ppe_max_iter, "Max iterations should increase during instability"
    
    # Recover
    trigger_stability(state)
    for _ in range(11): # Trigger recovery streak
        manager.sync_state(state)
    
    assert manager.max_iter == config.ppe_max_iter, "Failed to recover base max_iter"

def test_edge_case_nan_inf_handling(config, state):
    """Ensures numerical non-finite values trigger a stabilization failure."""
    manager = ElasticManager(config, initial_dt=0.5)
    
    # Test NaN
    state.fields.data[0, FI.VX_STAR] = np.nan
    assert manager.sync_state(state) is False
    
    # Test Inf
    state.fields.data[0, FI.VX_STAR] = np.inf
    assert manager.sync_state(state) is False

def test_dt_recovery_clamping(config, state):
    """Ensures recovery logic doesn't inflate dt beyond the initial target."""
    target = 0.5
    manager = ElasticManager(config, initial_dt=target)
    
    # Drop dt
    trigger_instability(state)
    manager.sync_state(state) 
    
    # Recover fully
    trigger_stability(state)
    for _ in range(50):
        manager.sync_state(state)
        
    assert manager.dt == target, f"dt recovered to {manager.dt}, exceeding target {target}"