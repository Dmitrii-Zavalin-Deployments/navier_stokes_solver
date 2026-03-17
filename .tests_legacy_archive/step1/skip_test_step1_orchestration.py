# tests/step1/test_step1_orchestration.py

import copy

import numpy as np
import pytest

from src.step1.allocate_fields import allocate_fields
from src.step1.apply_initial_conditions import apply_initial_conditions
from src.step1.orchestrate_step1 import orchestrate_step1_state
from src.step1.parse_config import parse_config
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


@pytest.fixture
def base_input():
    """Provides the canonical JSON-safe dummy input (Section 5 Compliance)."""
    return solver_input_schema_dummy()

# --- SECTION 1: PIPELINE ORCHESTRATION & SCHEMA GATES ---

def test_step1_input_schema_failure(base_input):
    """Verifies that missing top-level JSON keys trigger a RuntimeError (North Star Rule)."""
    bad = copy.deepcopy(base_input)
    bad.pop("grid")

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step1_state(bad)
    assert "Input schema validation FAILED" in str(excinfo.value)

def test_step1_physical_constraints_failure(base_input):
    """Verifies physically impossible values (density <= 0) are caught by the JSON validator."""
    bad = copy.deepcopy(base_input)
    bad["fluid_properties"]["density"] = -1.0

    with pytest.raises(RuntimeError, match="Input schema validation FAILED"):
        orchestrate_step1_state(bad)

def test_step1_output_schema_failure(monkeypatch, base_input):
    """
    Forced failure: Verifies the second gate (Physical Validation) 
    triggers if the internal assembler produces a broken SolverState.
    """
    import src.step1.orchestrate_step1 as mod
    real_assembler = mod.assemble_simulation_state

    def broken_assembler(*args, **kwargs):
        state = real_assembler(*args, **kwargs)
        state.grid = {} # Corrupt the object state
        return state

    monkeypatch.setattr(mod, "assemble_simulation_state", broken_assembler)

    with pytest.raises(ValueError, match="Incomplete Grid Definition"):
        orchestrate_step1_state(base_input)

# --- SECTION 2: FIELD ALLOCATION & STAGGERED MATH (Arakawa C-Grid) ---



def test_step1_happy_path(base_input):
    """Final end-to-end check for Step 1 pipeline including staggered shapes."""
    state = orchestrate_step1_state(copy.deepcopy(base_input))

    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    
    # Verify Arakawa C-grid layout (N+1 for faces)
    assert state.fields["P"].shape == (nx, ny, nz)
    assert state.fields["U"].shape == (nx + 1, ny, nz)
    assert state.fields["V"].shape == (nx, ny + 1, nz)
    assert state.fields["W"].shape == (nx, ny, nz + 1)
    
    assert state.constants["dt"] > 0

def test_allocate_fields_staggered_audit(base_input):
    """Ensures 100% coverage for allocate_fields.py by verifying memory layout."""
    fields = allocate_fields(base_input["grid"])
    
    assert fields["U"].shape == (3, 2, 2)
    assert fields["V"].shape == (2, 3, 2)
    assert fields["W"].shape == (2, 2, 3)
    assert fields["P"].shape == (2, 2, 2)

# --- SECTION 3: INITIAL CONDITION (IC) EXCEPTIONS & BROADCASTING ---

def test_ic_pressure_exception_handling():
    """Triggers guardrails for non-numeric pressure ICs."""
    fields = {"P": np.zeros((2, 2, 2))}
    ic = {"pressure": "not_a_number"}
    
    with pytest.raises(ValueError, match="Invalid pressure initial condition"):
        apply_initial_conditions(fields, ic)

def test_ic_velocity_shape_validation():
    """Ensures velocity IC is a complete 3-element vector [u, v, w]."""
    fields = {"U": np.zeros((3, 2, 2)), "V": np.zeros((2, 3, 2)), "W": np.zeros((2, 2, 3))}
    ic = {"velocity": [1.0, 0.0]} # Missing 'w'
    
    with pytest.raises(ValueError, match="Initial velocity must be a 3-element list"):
        apply_initial_conditions(fields, ic)

def test_ic_velocity_cast_exception():
    """Triggers error if velocity components cannot be cast to float."""
    fields = {"U": np.zeros((3, 2, 2)), "V": np.zeros((2, 3, 2)), "W": np.zeros((2, 2, 3))}
    ic = {"velocity": [1.0, "fast", 0.0]}
    
    with pytest.raises(ValueError, match="Could not cast initial velocity components to float"):
        apply_initial_conditions(fields, ic)

def test_apply_initial_conditions_broadcasting():
    """Verifies that scalar/list ICs are correctly broadcast to 3D staggered arrays."""
    nx, ny, nz = 2, 2, 2
    fields = {
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
        "P": np.zeros((nx, ny, nz))
    }
    
    ic = {"velocity": [1.0, 0.0, 0.0], "pressure": 5.0}
    apply_initial_conditions(fields, ic)
    
    assert np.all(fields["U"] == 1.0)
    assert np.all(fields["P"] == 5.0)

# --- SECTION 4: STEP 1 DATA COVERAGE (Loud Value Traceability) ---

def test_step1_input_coverage_sensitivity(base_input):
    """
    Phase F Audit: Uses 'Loud' overrides (unique primes) to verify data 
    transport from JSON through the entire pipeline into SolverState.
    """
    input_data = copy.deepcopy(base_input)

    # Inject Loud Overrides
    input_data["grid"].update({"nx": 13, "ny": 7, "nz": 5, "x_min": -2.5, "x_max": 2.5})
    input_data["fluid_properties"].update({"density": 1234.56, "viscosity": 0.00789})
    input_data["simulation_parameters"]["time_step"] = 0.00042
    input_data["initial_conditions"].update({"pressure": 9.99, "velocity": [1.1, 2.2, 3.3]})
    input_data["mask"] = [1] * (13 * 7 * 5)

    state = orchestrate_step1_state(input_data)

    # Assert Loud Value Survival
    assert state.grid['dx'] == pytest.approx(5.0 / 13)
    assert state.constants['rho'] == 1234.56
    assert state.constants['dt'] == 0.00042
    assert state.fields["U"][0, 0, 0] == 1.1
    assert state.fields["P"][0, 0, 0] == 9.99

# --- SECTION 5: OUTPUT SCHEMA COMPLIANCE (The Final Gate) ---

def test_step1_output_matches_foundation_contract(base_input):
    """
    The Auditor: Verifies the final SolverState matches the Foundation dummy
    and maintains JSON-safe serialization compatibility.
    """
    state = orchestrate_step1_state(base_input)

    # 1. Verification of JSON Roundtrip
    json_state = state.to_json_safe()
    assert isinstance(json_state["mask"], list), "Output serialization must provide a flattened list"

    # 2. Critical Department Audit
    for dept in ["grid", "fields", "mask", "constants", "boundary_conditions"]:
        assert hasattr(state, dept), f"SolverState missing critical department: {dept}"

def test_parse_config_minimal_parameters(base_input):
    """Verifies parse_config handles stripped optional keys (e.g. output_interval)."""
    config = copy.deepcopy(base_input)
    if "output_interval" in config["simulation_parameters"]:
        del config["simulation_parameters"]["output_interval"]
    
    parsed = parse_config(config)
    assert "time_step" in parsed["simulation_parameters"]