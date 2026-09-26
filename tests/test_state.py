"""
Unit Test Module for SolverState Container (src/state.py).

This module provides comprehensive narrative and executable verification for the 
SolverState class, validating strict non-default policies, schema presence checks,
numerical stability enforcement (NaN/Inf detection), and boundary condition extraction.
"""

import numpy as np
import pytest

from src.state import SolverState


# ============================================================================
# NARRATIVE SECTION 1: Strict Non-Default Validation & Schema Presence Checks
# ============================================================================
# Under strict non-default policies, missing or null required arguments, missing 
# spatial grid sections, missing sub-schema blocks, or absent mandatory keys 
# must immediately raise explicit ValueError or KeyError exceptions.
# ============================================================================


class MockCppBoundaryCondition:
    """
    Mock object emulating non-dictionary C++ boundary condition bindings
    for testing attribute fallback extraction in get_boundary_condition_dicts().
    """
    def __init__(self, location: str, btype: str, u: float, v: float, w: float, p: float):
        self.location = location
        self.type = btype
        self.u_val = u
        self.v_val = v
        self.w_val = w
        self.scalar_p = p


def test_solver_state_none_inputs():
    """
    # We verify that providing explicit None values for input_data or config_data 
    # triggers a ValueError under strict initialization rules.
    """
    # For a null input_data dictionary, initialization must fail:
    #     SolverState(None, {}) -> ValueError
    with pytest.raises(ValueError, match="input_data must be explicitly provided"):
        SolverState(None, {})

    # For a null config_data dictionary, initialization must fail:
    #     SolverState({}, None) -> ValueError
    with pytest.raises(ValueError, match="config_data must be explicitly provided"):
        SolverState({}, None)


def test_solver_state_missing_grid():
    """
    # We verify that omitting the required 'grid' section or passing None 
    # raises a KeyError.
    """
    # When the grid section is entirely absent from input_data:
    with pytest.raises(KeyError, match="missing required 'grid' section"):
        SolverState({}, {})

    # When the grid section is explicitly set to None:
    with pytest.raises(KeyError, match="missing required 'grid' section"):
        SolverState({"grid": None}, {})


def test_solver_state_missing_grid_keys():
    """
    # We verify that missing individual required parameters within the grid section 
    # trigger a KeyError identifying the missing key.
    """
    # An incomplete grid dictionary missing required keys (e.g., nx, ny, nz):
    incomplete_grid = {"x_min": 0.0, "x_max": 1.0}
    input_data = {"grid": incomplete_grid}

    # Ingestion must raise a KeyError for the missing required grid parameters:
    with pytest.raises(KeyError, match="Non-default policy violation in 'grid'"):
        SolverState(input_data, {})


def test_solver_state_missing_sub_schemas(base_valid_input):
    """
    # We verify that omitting any required sub-schema section (e.g., fluid_properties, 
    # simulation_parameters, boundary_conditions, external_forces, physical_constraints, 
    # or mask) triggers a KeyError.
    """
    required_sections = [
        "fluid_properties",
        "simulation_parameters",
        "boundary_conditions",
        "external_forces",
        "physical_constraints",
        "mask",
    ]

    for section in required_sections:
        corrupted_input = base_valid_input.copy()
        corrupted_input[section] = None

        # Omitting or setting a required section to None must violate policy:
        with pytest.raises(KeyError, match=f"missing required section '{section}'"):
            SolverState(corrupted_input, {})


def test_solver_state_missing_simulation_parameter_keys(base_valid_input):
    """
    # We verify that omitting mandatory parameters inside simulation_parameters 
    # (time_step, total_time, output_interval) raises a KeyError.
    """
    corrupted_input = base_valid_input.copy()
    corrupted_input["simulation_parameters"] = {}  # Empty dict missing time_step

    with pytest.raises(KeyError, match="Non-default policy violation in 'simulation_parameters'"):
        SolverState(corrupted_input, {})


def test_solver_state_missing_physical_constraint_keys(base_valid_input):
    """
    # We verify that omitting mandatory parameters inside physical_constraints 
    # raises a KeyError.
    """
    corrupted_input = base_valid_input.copy()
    corrupted_input["physical_constraints"] = {}  # Empty dict missing velocity/pressure bounds

    with pytest.raises(KeyError, match="Non-default policy violation in 'physical_constraints'"):
        SolverState(corrupted_input, {})


# ============================================================================
# NARRATIVE SECTION 2: Physical Constraints & Boundary Condition Extraction
# ============================================================================
# Solver state validates numerical field stability via NaN/Inf detection and 
# normalizes mixed boundary condition representations into standard dictionaries.
# ============================================================================


def test_enforce_physical_constraints_nan_detection(base_valid_input):
    """
    # We verify that enforce_physical_constraints() detects non-finite values (NaN or Inf) 
    # in the simulation fields buffer and raises an ArithmeticError.
    """
    state = SolverState(base_valid_input, {})
    
    # We inject a NaN into the velocity field buffer:
    #     state.fields[0, 0, 0, 0] = np.nan
    state.fields[0, 0, 0, 0] = np.nan

    # Enforcing physical constraints must detect the non-finite value and raise an error:
    with pytest.raises(ArithmeticError, match="Numerical instability detected"):
        state.enforce_physical_constraints()


def test_get_boundary_condition_dicts_mixed(base_valid_input):
    """
    # We verify that get_boundary_condition_dicts() correctly normalizes both raw dictionaries 
    # and non-dictionary C++ boundary condition objects into standard configuration dictionaries.
    """
    state = SolverState(base_valid_input, {})

    # We configure mixed boundary conditions containing both standard dicts and C++ mock objects:
    mock_bc = MockCppBoundaryCondition(
        location="z_min", btype="inflow", u=1.0, v=0.0, w=0.0, p=0.0
    )
    state.boundary_conditions = [
        {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        mock_bc
    ]

    bc_dicts = state.get_boundary_condition_dicts()

    # We assert that the resulting list contains properly structured dictionaries for all entries:
    assert len(bc_dicts) == 2
    assert bc_dicts[0]["location"] == "wall"
    assert bc_dicts[1]["location"] == "z_min"
    assert bc_dicts[1]["type"] == "inflow"
    assert bc_dicts[1]["values"]["u"] == 1.0


@pytest.fixture
def base_valid_input():
    """
    # We provide a complete, schema-compliant baseline input dictionary for testing valid initializations.
    """
    return {
        "grid": {
            "nx": 4, "ny": 4, "nz": 4,
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0
        },
        "fluid_properties": {"density": 1.0, "viscosity": 0.01},
        "simulation_parameters": {"time_step": 0.01, "total_time": 0.1, "output_interval": 1},
        "boundary_conditions": [{"location": "wall", "type": "no-slip", "values": {"u": 0.0}}],
        "external_forces": {"force_vector": [0.0, 0.0, 0.0]},
        "physical_constraints": {
            "min_velocity": -10.0, "max_velocity": 10.0,
            "min_pressure": -100.0, "max_pressure": 100.0
        },
        "mask": [0] * 64
    }
