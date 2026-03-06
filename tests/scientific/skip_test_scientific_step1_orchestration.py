# tests/scientific/test_scientific_step1_orchestration.py

import pytest
import numpy as np
from src.step1.orchestrate_step1 import orchestrate_step1

def test_scientific_orchestration_mapping(base_input):
    base_input.grid.nx, base_input.grid.ny, base_input.grid.nz = 4, 4, 4
    # Manually update the mask to match the new volume
    base_input.mask._data = [1] * 64
    state = orchestrate_step1(base_input)
    
    # Verify Geometry
    assert state.grid.nx == 4
    assert state.grid.x_max == 1.0
    
    # Verify Physics (Accessing attributes via state object)
    assert state.fluid.rho == base_input.fluid_properties.density
    assert state.fluid.mu == base_input.fluid_properties.viscosity

def test_scientific_field_initialization(base_input):
    """Verify staggered fields are allocated and primed with ICs."""
    base_input.grid.nx, base_input.grid.ny, base_input.grid.nz = 4, 4, 4
    base_input.mask._data = [1] * 64
    base_input.initial_conditions.pressure = 101325.0
    base_input.initial_conditions.velocity = [1.0, 0.0, 0.0]
    
    state = orchestrate_step1(base_input)
    
    # Harlow-Welch Staggering Check
    assert state.fields.U.shape == (5, 4, 4)
    assert state.fields.V.shape == (4, 5, 4)
    assert state.fields.W.shape == (4, 4, 5)
    
    # IC Priming Check
    np.testing.assert_allclose(state.fields.P, 101325.0)
    np.testing.assert_allclose(state.fields.U, 1.0)
    assert state.fields.P.dtype == np.float64

def test_scientific_audit_firewall(base_input):
    """Verify the _final_audit catches non-physical values."""
    base_input.initial_conditions.velocity = [np.nan, 0.0, 0.0] 
    
    with pytest.raises(ValueError, match="Audit Failed: Non-finite values"):
        orchestrate_step1(base_input)

def test_scientific_restart_metadata(base_input):
    """Verify that kwargs correctly override default time/iteration for restarts."""
    state = orchestrate_step1(base_input, iteration=50, time=0.123)
    
    assert state.iteration == 50
    assert state.time == 0.123

def test_scientific_mask_integrity(base_input):
    """Verify that the topology masks are correctly derived and typed."""
    base_input.grid.nx, base_input.grid.ny, base_input.grid.nz = 4, 4, 4
    base_input.mask._data = [1] * 64
    state = orchestrate_step1(base_input)
    
    # Shape check
    assert state.masks.is_fluid.shape == (4, 4, 4)
    # Type check
    assert state.masks.is_fluid.dtype == bool
    assert np.all(state.masks.is_fluid)

def test_scientific_audit_rho_guard(base_input):
    # Bypass the setter to inject a bad value for testing the auditor
    base_input.fluid_properties._density = -5.0 
    with pytest.raises(ValueError, match="Audit Failed: Non-physical density"):
        orchestrate_step1(base_input)