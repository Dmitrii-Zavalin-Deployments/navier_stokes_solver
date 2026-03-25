# tests/common/test_elasticity_manager.py

from types import SimpleNamespace

import numpy as np
import pytest

from src.common.elasticity import ElasticManager
from src.common.field_schema import FI
from tests.helpers.solver_input_schema_dummy import create_validated_input

# ----------------------------------------------------------------
# 1. MOCK OBJECTS & FIXTURES
# ----------------------------------------------------------------

class MockConfig:
    def __init__(self):
        self.dt_initial = 0.5
        self.dt_min_limit = 0.001
        self.dt_min = 1e-6
        self.reduction_factor = 0.5
        self.divergence_threshold = 1e6
        self.ppe_omega = 1.7
        self.ppe_max_iter = 50
        self.ppe_max_retries = 3 # FIX: Added missing attribute

@pytest.fixture
def config():
    return MockConfig()

@pytest.fixture
def state_mock():
    """Creates a mock state for sync_state logic."""
    data = np.zeros((10, 20)) 
    fields = SimpleNamespace(data=data)
    # We add simulation_parameters so the Manager can initialize if needed
    sim_params = SimpleNamespace(time_step=0.5)
    return SimpleNamespace(fields=fields, simulation_parameters=sim_params)

def trigger_instability(state):
    state.fields.data[:, FI.VX_STAR] = 1e12

def trigger_stability(state):
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
    manager.stabilization(is_needed=True)
    assert manager.dt == initial_dt * config.reduction_factor

def test_manual_stabilization_safety_floor(config):
    """
    STABILITY AUDIT: Verifies manual reduction never passes the absolute floor.
    """
    config.dt_min = 1e-6
    # FIX: Never pass None to ElasticManager
    dummy_input = create_validated_input()
    manager = ElasticManager(config, dummy_input)

    for _ in range(20):
        manager.stabilization(is_needed=True)

    assert manager.dt >= config.dt_min

def test_sync_state_monotonic_decay(config, state_mock):
    # FIX: Removed initial_dt keyword, using state_mock to set dt
    manager = ElasticManager(config, state_mock)
    trigger_instability(state_mock)
    
    last_dt = manager.dt
    for _ in range(3):
        try:
            manager.sync_state(state_mock)
            assert manager.dt < last_dt
            last_dt = manager.dt
        except RuntimeError:
            return

def test_omega_and_max_iter_adaptation(config, state_mock):
    manager = ElasticManager(config, state_mock)
    trigger_instability(state_mock)
    manager.sync_state(state_mock)
    
    assert manager.omega < config.ppe_omega
    assert manager.max_iter > config.ppe_max_iter

def test_edge_case_nan_inf_handling(config, state_mock):
    manager = ElasticManager(config, state_mock)
    state_mock.fields.data[0, FI.VX_STAR] = np.nan
    assert manager.sync_state(state_mock) is False

def test_dt_recovery_clamping(config, state_mock):
    manager = ElasticManager(config, state_mock)
    target = manager.dt
    
    trigger_instability(state_mock)
    manager.sync_state(state_mock) 
    
    trigger_stability(state_mock)
    for _ in range(20):
        manager.sync_state(state_mock)
        
    assert manager.dt <= target