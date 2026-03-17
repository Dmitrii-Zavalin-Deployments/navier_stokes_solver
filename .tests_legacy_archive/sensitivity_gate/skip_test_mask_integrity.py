# tests/sensitivity_gate/test_mask_integrity.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.build_divergence_operator import build_divergence_operator
from src.step2.build_gradient_operators import build_gradient_operators
from src.step2.build_laplacian_operators import build_laplacian_operators
from src.step2.create_fluid_mask import create_fluid_mask
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


@pytest.fixture
def empty_state():
    """Provides a SolverState with 0 fluid cells to test the Physical Logic Firewall."""
    grid_data = {'nx': 2, 'ny': 2, 'nz': 2, 'dx': 1.0, 'dy': 1.0, 'dz': 1.0}
    state = SolverState(config={}, grid=grid_data, boundary_conditions=[])
    # Force an all-solid domain
    all_solid_mask = np.zeros(8, dtype=np.int32)
    state.mask = all_solid_mask
    state.is_fluid = (all_solid_mask == 1) # Result: all False
    state.operators = {}
    return state

def test_gate_2f_discrete_mask_logic():
    """Gate 2.F: Verify firewall rejects non-integer masks (Discrete Mask Rule)."""
    state = SolverState(config={}, grid={'nx': 2, 'ny': 2, 'nz': 2}, boundary_conditions=[])
    state.mask = np.ones(8, dtype=np.float64) 
    with pytest.raises(ValueError, match="Mask must be an integer array"):
        create_fluid_mask(state)

def test_gate_2b_traceability_loud_value():
    """
    Gate 2.B: Traceability Rule.
    Verify exact float propagation using a 'Loud Value'.
    """
    raw_input = solver_input_schema_dummy()
    loud_viscosity = 0.000123456789
    raw_input['fluid_properties']['viscosity'] = loud_viscosity
    
    state = orchestrate_step1(raw_input)
    
    actual = state.config['fluid_properties']['viscosity']
    assert actual == loud_viscosity, f"Traceability Failure: Expected {loud_viscosity}, got {actual}"

def test_topology_protection_laplacian(empty_state):
    """Verify Laplacian builder fails loudly on empty domains."""
    with pytest.raises(RuntimeError, match="No fluid cells detected in domain"):
        build_laplacian_operators(empty_state)

def test_topology_protection_divergence(empty_state):
    """Verify Divergence builder fails loudly on empty domains."""
    with pytest.raises(RuntimeError, match="Divergence operator"):
        build_divergence_operator(empty_state)

def test_topology_protection_gradient(empty_state):
    """Verify Gradient builder fails loudly on empty domains."""
    with pytest.raises(RuntimeError, match="Gradient operators"):
        build_gradient_operators(empty_state)

def test_gate_2f_illegal_integer_logic():
    """Gate 2.F: Verify firewall rejects integers outside {-1, 0, 1}."""
    state = SolverState(config={}, grid={'nx': 2, 'ny': 2, 'nz': 2}, boundary_conditions=[])
    # 99 is an illegal topology identifier
    state.mask = np.array([1, 0, -1, 99, 0, 1, 1, 1], dtype=np.int32) 
    
    with pytest.raises(ValueError, match="restricted to {-1, 0, 1}"):
        create_fluid_mask(state)
