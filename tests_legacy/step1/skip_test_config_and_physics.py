# tests/step1/test_config_and_physics.py


import numpy as np
import pytest

from src.step1.compute_derived_constants import compute_derived_constants
from src.step1.orchestrate_step1 import orchestrate_step1_state
from src.step1.parse_config import parse_config
from src.step1.validate_physical_constraints import validate_physical_constraints
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


@pytest.fixture
def valid_state():
    """Provides a fresh SolverState (Section 5 Compliance)."""
    json_input = solver_input_schema_dummy()
    return orchestrate_step1_state(json_input)

# --- CONFIG PARSER & DEBT TRIGGERS ---

def test_parse_config_structure_and_keys():
    """Triggers TypeErrors and KeyErrors in the config parser."""
    # 1. Section must be a dict
    with pytest.raises(TypeError, match="must be a dictionary"):
        parse_config({"grid": "not-a-dict"})

    # 2. Missing grid keys
    with pytest.raises(KeyError, match="Grid Configuration Error"):
        parse_config({"grid": {"nx": 10}})

    # 3. Missing fluid properties
    with pytest.raises(KeyError, match="requires 'density' and 'viscosity'"):
        parse_config({
            "grid": {"nx":2,"ny":2,"nz":2,"x_min":0,"x_max":1,"y_min":0,"y_max":1,"z_min":0,"z_max":1},
            "fluid_properties": {"density": 1.0}
        })

def test_parse_config_missing_time_step():
    """TARGET: parse_config.py Line 36. Ensures 'time_step' is mandated."""
    bad_input = {
        "grid": {"nx":2,"ny":2,"nz":2,"x_min":0,"x_max":1,"y_min":0,"y_max":1,"z_min":0,"z_max":1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.1},
        "simulation_parameters": {} # Missing time_step key
    }
    with pytest.raises(KeyError, match="requires 'time_step'"):
        parse_config(bad_input)

def test_parse_config_external_forces_debt():
    """Triggers force vector shape and finiteness checks."""
    base = {
        "grid": {"nx":2,"ny":2,"nz":2,"x_min":0,"x_max":1,"y_min":0,"y_max":1,"z_min":0,"z_max":1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.1},
        "simulation_parameters": {"time_step": 0.1}
    }
    
    # 1. Missing force_vector
    bad_force = {**base, "external_forces": {}}
    with pytest.raises(KeyError, match="'force_vector' is missing"):
        parse_config(bad_force)

    # 2. Invalid shape
    bad_shape = {**base, "external_forces": {"force_vector": [0, 0]}}
    with pytest.raises(ValueError, match=r"must be \[x, y, z\]"):
        parse_config(bad_shape)

    # 3. Non-finite values
    bad_math = {**base, "external_forces": {"force_vector": [0, float('nan'), 0]}}
    with pytest.raises(ValueError, match="Non-finite values detected"):
        parse_config(bad_math)

# --- PHYSICAL CONSTANTS GUARDRAILS ---

def test_derived_constants_physics_checks():
    """Triggers non-physical rho and mu checks in compute_derived_constants."""
    grid = {"dx": 0.1, "dy": 0.1, "dz": 0.1}
    params = {"time_step": 0.01}

    # Density <= 0
    with pytest.raises(ValueError, match="Non-physical constant detected: rho = 0.0"):
        compute_derived_constants(grid, {"density": 0.0, "viscosity": 0.01}, params)

    # Viscosity < 0
    with pytest.raises(ValueError, match="Non-physical viscosity detected: mu = -0.5"):
        compute_derived_constants(grid, {"density": 1.0, "viscosity": -0.5}, params)

# --- SOLVER STATE CONSTRAINT VALIDATION (CORRECTED) ---

def test_validate_constraints_internal_helpers(valid_state):
    """
    TARGET: validate_physical_constraints.py Lines 14, 18, 22, 26, 75.
    Ensures 100% coverage by hitting every helper branch.
    """
    # 1. Trigger _ensure_positive_int (Line 22)
    valid_state.grid["nx"] = 0
    with pytest.raises(ValueError, match="Topology Violation.*positive integer"):
        validate_physical_constraints(valid_state)
    valid_state.grid["nx"] = 2 # Restore

    # 2. Trigger _ensure_finite (Line 26)
    valid_state.grid["x_min"] = float('inf')
    with pytest.raises(ValueError, match="Precision Error.*must be finite"):
        validate_physical_constraints(valid_state)
    valid_state.grid["x_min"] = 0.0 # Restore

    # 3. Trigger _ensure_positive for dt (Line 14) 
    # FIX: Match Actual message "Stability Violation" and "> 0"
    valid_state.constants["dt"] = -0.1
    with pytest.raises(ValueError, match="Stability Violation.*must be finite and > 0"):
        validate_physical_constraints(valid_state)
    valid_state.constants["dt"] = 0.05 # Restore

    # 4. Trigger _ensure_non_negative (Line 18)
    # Using viscosity (mu) to trigger Physicality Violation
    valid_state.constants["mu"] = -0.001
    with pytest.raises(ValueError, match="Physicality Violation.*must be finite and >= 0"):
        validate_physical_constraints(valid_state)
    valid_state.constants["mu"] = 0.01 # Restore

    # 5. Trigger Mask Shape Mismatch (Line 75)
    valid_state.mask = [0] * 10
    with pytest.raises(ValueError, match="Mask Length Mismatch"):
        validate_physical_constraints(valid_state)

def test_validate_constraints_grid_and_time(valid_state):
    """Verifies domain inversion."""
    # 1. Domain Inversion
    valid_state.grid["x_max"] = valid_state.grid["x_min"]
    with pytest.raises(ValueError, match="Domain Inversion"):
        validate_physical_constraints(valid_state)

def test_validate_constraints_topology_and_fields(valid_state):
    """Verifies mask compliance and numerical stability (NaN/Inf)."""
    # 1. Mask Value Compliance (Forbidden Topology)
    valid_state.mask[0] = 42 
    with pytest.raises(ValueError, match="Forbidden Topology"):
        validate_physical_constraints(valid_state)
    
    valid_state.mask[0] = 1 # RESET

    # 2. Field Finiteness
    valid_state.fields["V"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="(?i)finite"):
        validate_physical_constraints(valid_state)

# --- SECTION: Derived Constants (Functional Integrity) ---

def test_step1_constants_match_dummy():
    """
    Verifies that Step 1 orchestrator correctly populates the constants.
    """
    json_input = solver_input_schema_dummy()
    
    json_input["grid"].update({
        "nx": 2, "ny": 2, "nz": 2,
        "x_min": 0.0, "x_max": 2.0,
        "y_min": 0.0, "y_max": 2.0,
        "z_min": 0.0, "z_max": 2.0
    })

    json_input["fluid_properties"] = {"density": 5.0, "viscosity": 0.2}
    json_input["simulation_parameters"].update({"time_step": 0.05})

    state = orchestrate_step1_state(json_input)

    assert state.constants["rho"] == 5.0
    assert state.constants["mu"] == 0.2
    assert state.constants["dt"] == 0.05
    assert state.constants["dx"] == 1.0

# --- SECTION: Derived Constants Math (Precision Audit) ---

def test_compute_derived_constants_mathematical_precision():
    """
    Surveyor Audit: Verifies spatial derivation from asymmetric grids.
    """
    json_input = solver_input_schema_dummy()

    nx, ny, nz = 10, 5, 2
    json_input["grid"].update({
        "nx": nx, "x_min": -5.0, "x_max": 5.0, # dx = 1.0
        "ny": ny, "y_min": 0.0,  "y_max": 5.0, # dy = 1.0
        "nz": nz, "z_min": 0.0,  "z_max": 1.0  # dz = 0.5
    })
    
    json_input["mask"] = [0] * (nx * ny * nz)
    json_input["fluid_properties"] = {"density": 997.0, "viscosity": 0.001}
    json_input["simulation_parameters"]["time_step"] = 0.005

    state = orchestrate_step1_state(json_input)

    assert state.constants["dx"] == 1.0
    assert state.constants["dy"] == 1.0
    assert state.constants["dz"] == 0.5