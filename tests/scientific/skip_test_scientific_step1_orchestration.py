# tests/scientific/test_scientific_step1_orchestration.py

import numpy as np
import pytest

from src.solver_input import BoundaryConditionItem
from src.step1.orchestrate_step1 import orchestrate_step1


def test_scientific_orchestration_mapping(base_input):
    """Rule: Verify Collocated Grid dimensions match input exactly."""
    nx, ny, nz = 4, 4, 4
    base_input.grid.nx, base_input.grid.ny, base_input.grid.nz = nx, ny, nz
    base_input.mask.data = [1] * (nx * ny * nz) 
    state = orchestrate_step1(base_input, iteration=0, time=0.0)
    
    # Verify Geometry (Collocated: No N+1 padding)
    assert state.grid.nx == 4
    assert state.grid.x_max == 1.0
    
    # Verify Physical Context
    assert state.fluid.rho == base_input.fluid_properties.density
    assert state.fluid.mu == base_input.fluid_properties.viscosity

def test_scientific_field_initialization_collocated(base_input):
    """Verify Collocated fields are initialized to (nx, ny, nz)."""
    nx, ny, nz = 4, 4, 4
    base_input.grid.nx, base_input.grid.ny, base_input.grid.nz = nx, ny, nz
    base_input.mask.data = [1] * (nx * ny * nz)
    base_input.initial_conditions.pressure = 101325.0
    base_input.initial_conditions.velocity = [1.0, 2.0, 3.0]
    
    state = orchestrate_step1(base_input, iteration=0, time=0.0)
    
    # Physics Check: All fields share the same spatial domain (Theory Section 3)
    assert state.fields.U.shape == (4, 4, 4)
    assert state.fields.V.shape == (4, 4, 4)
    assert state.fields.W.shape == (4, 4, 4)
    
    # IC Priming Check
    np.testing.assert_allclose(state.fields.P, 101325.0)
    np.testing.assert_allclose(state.fields.U, 1.0)
    np.testing.assert_allclose(state.fields.V, 2.0)
    np.testing.assert_allclose(state.fields.W, 3.0)

def test_scientific_restart_metadata(base_input):
    """Verify deterministic metadata injection."""
    state = orchestrate_step1(base_input, iteration=50, time=0.123)
    assert state.iteration == 50
    assert state.time == 0.123

def test_scientific_mask_topology_integrity(base_input):
    """Verify tri-state mask topology (Theory Section 6)."""
    nx, ny, nz = 4, 4, 4
    base_input.grid.nx, base_input.grid.ny, base_input.grid.nz = nx, ny, nz
    base_input.mask.data = [1] * (nx * ny * nz)
    state = orchestrate_step1(base_input, iteration=0, time=0.0)
    
    assert state.masks.is_fluid.shape == (4, 4, 4)
    assert np.all(state.masks.is_fluid)
    assert not np.any(state.masks.is_boundary)

def test_scientific_boundary_lookup_integrity(base_input):
    """Verify high-speed lookup table population."""
    bc1 = BoundaryConditionItem()
    bc1.location = "x_min"
    bc1.type = "inflow"
    bc1.values = {"u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0}

    base_input.boundary_conditions.items = [bc1]
    state = orchestrate_step1(base_input, iteration=0, time=0.0)

    assert "x_min" in state.boundary_lookup
    assert state.boundary_lookup["x_min"]["u"] == 1.0

def test_scientific_missing_kwargs_keyerror(base_input):
    """Rule 5: Strict failure on missing deterministic metadata."""
    with pytest.raises(KeyError):
        orchestrate_step1(base_input) # Missing iteration/time