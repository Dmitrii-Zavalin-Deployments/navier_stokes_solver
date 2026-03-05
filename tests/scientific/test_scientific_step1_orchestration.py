# # tests/scientific/test_scientific_step1_orchestration.py

import pytest
import numpy as np
from src.step1.orchestrate_step1 import orchestrate_step1
from src.solver_input import SolverInput

def create_scientific_input():
    """Uses the official from_dict factory to ensure valid hydration."""
    data = {
        "grid": {
            "nx": 4, "ny": 4, "nz": 4,
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0
        },
        "fluid_properties": {"density": 1000.0, "viscosity": 0.001},
        "initial_conditions": {"pressure": 101325.0, "velocity": [1.0, 0.0, 0.0]},
        "external_forces": {"force_vector": [0.0, -9.81, 0.0]},
        "boundary_conditions": [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0}}
        ],
        "mask": [1] * 64
    }
    return SolverInput.from_dict(data)

def test_scientific_orchestration_mapping():
    """Verify the orchestrator correctly maps Input to State."""
    inp = create_scientific_input()
    state = orchestrate_step1(inp)
    
    # Verify Geometry
    assert state.grid.nx == 4
    assert state.grid.x_max == 1.0
    
    # Verify Physics
    assert state.fluid.rho == 1000.0
    assert state.fluid.mu == 0.001

def test_scientific_field_initialization():
    """Verify staggered fields are allocated and primed with ICs."""
    inp = create_scientific_input()
    state = orchestrate_step1(inp)
    
    # Harlow-Welch Staggering Check
    assert state.fields.U.shape == (5, 4, 4)
    assert state.fields.V.shape == (4, 5, 4)
    assert state.fields.W.shape == (4, 4, 5)
    
    # IC Priming Check
    np.testing.assert_allclose(state.fields.P, 101325.0)
    np.testing.assert_allclose(state.fields.U, 1.0)
    assert state.fields.P.dtype == np.float64

def test_scientific_audit_firewall():
    """Verify the _final_audit catches non-physical values."""
    inp = create_scientific_input()
    # Reach into the hydrated object to inject a physical error
    inp.initial_conditions.velocity = [np.nan, 0.0, 0.0] 
    
    with pytest.raises(ValueError, match="Audit Failed: Non-finite values"):
        orchestrate_step1(inp)

def test_scientific_restart_metadata():
    """Verify that kwargs correctly override default time/iteration for restarts."""
    inp = create_scientific_input()
    state = orchestrate_step1(inp, iteration=50, time=0.123)
    
    assert state.iteration == 50
    assert state.time == 0.123

def test_scientific_mask_integrity():
    """Verify that the topology masks are correctly derived and typed."""
    inp = create_scientific_input()
    state = orchestrate_step1(inp)
    
    # Shape check
    assert state.masks.is_fluid.shape == (4, 4, 4)
    # Type check (Should be boolean for efficient masking)
    assert state.masks.is_fluid.dtype == bool
    # Content check (Assuming [1]*64 in create_scientific_input)
    assert np.all(state.masks.is_fluid)

def test_scientific_audit_rho_guard():
    """Verify the firewall catches non-physical fluid properties."""
    inp = create_scientific_input()
    inp.fluid_properties.density = -5.0
    
    with pytest.raises(ValueError, match="Non-physical density"):
        orchestrate_step1(inp)