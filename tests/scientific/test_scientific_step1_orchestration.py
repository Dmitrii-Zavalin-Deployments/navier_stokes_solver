# tests/scientific/test_scientific_step1_orchestration.py

import pytest
import numpy as np
from src.step1.orchestrate_step1 import orchestrate_step1

def test_scientific_orchestration_mapping(base_input):
    nx, ny, nz = 4, 4, 4
    base_input.grid.nx, base_input.grid.ny, base_input.grid.nz = nx, ny, nz
    # Fix: Use dynamic product instead of hardcoded 64
    base_input.mask._data = [1] * (nx * ny * nz) 
    state = orchestrate_step1(base_input, iteration=0, time=0.0)
    
    # Verify Geometry
    assert state.grid.nx == 4
    assert state.grid.x_max == 1.0
    
    # Verify Physics (Accessing attributes via state object)
    assert state.fluid.rho == base_input.fluid_properties.density
    assert state.fluid.mu == base_input.fluid_properties.viscosity

def test_scientific_field_initialization(base_input):
    """Verify staggered fields are allocated and primed with ICs."""
    nx, ny, nz = 4, 4, 4
    base_input.grid.nx, base_input.grid.ny, base_input.grid.nz = nx, ny, nz
    base_input.mask._data = [1] * (nx * ny * nz)
    base_input.initial_conditions.pressure = 101325.0
    base_input.initial_conditions.velocity = [1.0, 0.0, 0.0]
    
    state = orchestrate_step1(base_input, iteration=0, time=0.0)
    
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
        orchestrate_step1(base_input, iteration=0, time=0.0)

def test_scientific_restart_metadata(base_input):
    """Verify that kwargs correctly override default time/iteration for restarts."""
    state = orchestrate_step1(base_input, iteration=50, time=0.123)
    
    assert state.iteration == 50
    assert state.time == 0.123

def test_scientific_mask_integrity(base_input):
    """Verify that the topology masks are correctly derived and typed."""
    nx, ny, nz = 4, 4, 4
    base_input.grid.nx, base_input.grid.ny, base_input.grid.nz = nx, ny, nz
    base_input.mask._data = [1] * (nx * ny * nz)
    state = orchestrate_step1(base_input, iteration=0, time=0.0)
    
    # Shape check
    assert state.masks.is_fluid.shape == (4, 4, 4)
    # Type check
    assert state.masks.is_fluid.dtype == bool
    assert np.all(state.masks.is_fluid)

def test_scientific_audit_rho_guard(base_input):
    # Bypass the setter to inject a bad value for testing the auditor
    base_input.fluid_properties._density = -5.0 
    with pytest.raises(ValueError, match="Audit Failed: Non-physical density"):
        orchestrate_step1(base_input, iteration=0, time=0.0)

def test_scientific_boundary_lookup_integrity(base_input):
    base_input.boundary_conditions.items = [
        DummyBC("x_min", "dirichlet", [1,2,3]),
        DummyBC("x_max", "neumann", [0,0,0]),
    ]

    state = orchestrate_step1(base_input, iteration=0, time=0.0)

    lookup = state.boundary_lookup

    assert "x_min" in lookup
    assert lookup["x_min"]["type"] == "dirichlet"
    assert np.allclose(lookup["x_min"]["values"], [1,2,3])

def test_scientific_boundary_condition_mapping(base_input):
    bc = base_input.boundary_conditions.items[0]
    state = orchestrate_step1(base_input, iteration=0, time=0.0)

    mapped = state.config.boundary_conditions[0]
    assert mapped["location"] == bc.location
    assert mapped["type"] == bc.type
    assert mapped["values"] == bc.values

def test_scientific_external_forces_mapping(base_input):
    base_input.external_forces.force_vector = [0.1, 0.0, -0.1]
    state = orchestrate_step1(base_input, iteration=0, time=0.0)

    fv = state.config.external_forces["force_vector"]
    np.testing.assert_allclose(fv, [0.1, 0.0, -0.1])

def test_scientific_boundary_mask_integrity(base_input):
    base_input.mask._data = [0] * 64  # all boundary
    state = orchestrate_step1(base_input, iteration=0, time=0.0)

    assert np.all(state.masks.is_boundary)
    assert not np.any(state.masks.is_fluid)

def test_scientific_config_hydration(base_input):
    state = orchestrate_step1(base_input, iteration=0, time=0.0)

    assert state.config._simulation_parameters is base_input.simulation_parameters
    assert state.config._fluid_properties is base_input.fluid_properties
