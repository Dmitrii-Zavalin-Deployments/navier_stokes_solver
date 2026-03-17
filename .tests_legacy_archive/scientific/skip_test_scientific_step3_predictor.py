# tests/scientific/test_scientific_step3_predictor.py

import numpy as np
import pytest
from scipy import sparse

from src.solver_state import SolverState


class AttributeDict(dict):
    """A dict that allows dot notation access, satisfying both type-checks and solver logic."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

@pytest.fixture
def state_predictor():
    """Fixture to set up a valid state for full predictor physics."""
    state = SolverState()
    
    # 1. Physics Setup (rho=1000, mu=1, dt=0.1 -> nu=0.001)
    fluid_props = state.fluid = AttributeDict({"density": 1000.0, "viscosity": 1.0})
    state.config.simulation_parameters = AttributeDict({"time_step": 0.1})
    ext_forces = state.external_forces = AttributeDict({"force_vector": [1.0, 0.0, 0.0]})
    
    # 2. Field Allocation (Staggered Grid Dimensions)
    state.fields.U = np.ones((4, 3, 3), order='F') 
    state.fields.V = np.zeros((3, 4, 3), order='F')
    state.fields.W = np.zeros((3, 3, 4), order='F')
    
    # 3. Operator Mocking (Identity matrices for predictable math)
    # 4x3x3 = 36; 3x4x3 = 36; 3x3x4 = 36. All use size 36 matrices.
    eye36 = sparse.eye(36, 36)
    
    # Hydrate both private slots and public properties to satisfy the 'Security Guard'
    for op in ['_laplacian', '_advection_u', '_advection_v', '_advection_w']:
        setattr(state.operators, op, eye36)
    
    return state

def test_predict_velocity_full_physics(state_predictor):
    """Formula Check: V* = V + dt * (nu*L*V - A*V + f)"""
    from src.step3.predictor import predict_velocity
    predict_velocity(state_predictor)
    # Calculation: 1.0 + 0.1 * (0.001 * 1.0 - 1.0 + 1.0) = 1.0001
    assert np.allclose(state_predictor.fields.U_star, 1.0001)

def test_predict_velocity_instability_debug(state_predictor, capsys):
    """Verify Rule 5: Catch NaNs/Infs during prediction."""
    from src.step3.predictor import predict_velocity
    state_predictor.fields.U[0,0,0] = np.inf
    predict_velocity(state_predictor)
    captured = capsys.readouterr().out
    assert "CRITICAL: Predictor Instability" in captured

def test_predict_velocity_component_isolation(state_predictor):
    """Verify V and W components receive their respective force vectors."""
    from src.step3.predictor import predict_velocity
    state_predictor.config.external_forces.force_vector = [1.0, 2.0, 3.0]
    predict_velocity(state_predictor)
    
    # V* = 0 + 0.1 * (0 - 0 + 2.0) = 0.2
    assert np.allclose(state_predictor.fields.V_star, 0.2)
    # W* = 0 + 0.1 * (0 - 0 + 3.0) = 0.3
    assert np.allclose(state_predictor.fields.W_star, 0.3)