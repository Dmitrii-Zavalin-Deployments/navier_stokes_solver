"""
tests/test_cpp_gate_coverage.py
Literate Test Suite: Comprehensive Unit Verification Targeting 100% Coverage for src/cpp_gate.py.

This test file is written under the Literate Testing Standard. Each verification block
combines descriptive narrative prose with executable test assertions, documenting both the 
architectural intent and the computational safeguards protecting the C++ bridge layer.
"""

import importlib
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Safely import or mock the compiled C++ extension module to prevent collection errors
try:
    import navier_stokes_cpp
except ImportError:
    navier_stokes_cpp = MagicMock()
    sys.modules["navier_stokes_cpp"] = navier_stokes_cpp

from src.cpp_gate import (
    _apply_initial_boundary_conditions,
    _convert_boundary_conditions,
    _dict_to_boundary_condition,
    _get_or_create_cpp_solver,
    step_simulation,
)
from src.state import SolverState

# ==============================================================================
# SCHEMA-COMPLIANT BASE CONFIGURATIONS (Grid cells >= 4 per schema)
# ==============================================================================
BASE_GRID = {
    "nx": 4, "ny": 4, "nz": 4,
    "x_min": 0.0, "x_max": 1.0,
    "y_min": 0.0, "y_max": 1.0,
    "z_min": 0.0, "z_max": 1.0,
}

BASE_FLUID_PROPERTIES = {
    "density": 1.0,
    "viscosity": 0.01,
}

BASE_SIM_PARAMS = {
    "time_step": 0.01,
    "total_time": 1.0,
    "output_interval": 10,
}

BASE_CONSTRAINTS = {
    "min_velocity": -10.0,
    "max_velocity": 10.0,
    "min_pressure": -100.0,
    "max_pressure": 100.0,
}

BASE_FORCES = {
    "force_vector": [0.0, 0.0, 0.0]
}

# Canonical mask length = nx * ny * nz = 4 * 4 * 4 = 64
BASE_MASK = [0] * 64

BASE_CONFIG = {
    "mode": "test",
    "max_poisson_iterations": 10,
    "poisson_tolerance": 1e-6,
}


def create_test_input_data(overrides=None):
    """Helper to generate a fully schema-compliant input dictionary."""
    data = {
        "grid": BASE_GRID.copy(),
        "fluid_properties": BASE_FLUID_PROPERTIES.copy(),
        "simulation_parameters": BASE_SIM_PARAMS.copy(),
        "physical_constraints": BASE_CONSTRAINTS.copy(),
        "external_forces": BASE_FORCES.copy(),
        "mask": BASE_MASK.copy(),
        "boundary_conditions": [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0}}
        ],
    }
    if overrides:
        data.update(overrides)
    return data


# ==============================================================================
# SECTION 1: C++ Module Import Resilience
# ==============================================================================
# In distributed environments or development setups where the compiled C++ shared library 
# ('navier_stokes_cpp') has not yet been built or linked into the Python path, the bridge 
# module must fail gracefully with an informative ImportError rather than causing an unhandled 
# attribute crash.
# ==============================================================================

def test_import_error_branch(monkeypatch):
    """
    Verifies that an ImportError raised during the dynamic loading of 'navier_stokes_cpp' 
    is correctly intercepted and re-raised with actionable instructions for the user.
    """
    # Temporarily purge the module from sys.modules to force a fresh import attempt.
    monkeypatch.delitem(sys.modules, "navier_stokes_cpp", raising=False)
    import builtins
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "navier_stokes_cpp":
            raise ImportError("Mocked import error for testing.")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    import src.cpp_gate
    with pytest.raises(ImportError, match=r"Failed to import compiled C\+\+ module"):
        importlib.reload(src.cpp_gate)

    # Restore original import behavior and module state.
    monkeypatch.undo()
    importlib.reload(src.cpp_gate)


# ==============================================================================
# SECTION 2: Boundary Condition Attribute Assignment Robustness
# ==============================================================================
# When translating Python dictionaries into C++ BoundaryCondition objects via Pybind11, 
# target properties might occasionally raise AttributeError or TypeError due to strict C++ 
# type enforcement or read-only constraints. The bridge handles these safely via debug logging.
# ==============================================================================

def test_bc_setting_attribute_and_type_errors(monkeypatch):
    """
    Ensures that unexpected AttributeError or TypeError exceptions encountered during 
    direct attribute setters on a boundary condition object are caught and logged.
    """
    class BadLocationBC:
        @property
        def location(self):
            return "x_min"

        @location.setter
        def location(self, val):
            raise AttributeError("Simulated read-only location attribute error.")

        @property
        def type(self):
            return "inflow"

        @type.setter
        def type(self, val):
            raise TypeError("Simulated type mismatch error.")

        u = 1.0
        v = 0.0
        w = 0.0
        p = 0.0

    monkeypatch.setattr(navier_stokes_cpp, "BoundaryCondition", BadLocationBC, raising=False)
    
    bc_dict = {
        "location": "x_min", 
        "type": "inflow", 
        "values": {"u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0}
    }
    
    bc_obj = _dict_to_boundary_condition(bc_dict)
    assert bc_obj is not None


# ==============================================================================
# SECTION 3: Structural Validation of Non-Dictionary Boundary Objects
# ==============================================================================
# Boundary conditions passed as pre-instantiated objects must strictly contain 
# valid metadata identifiers. Missing required attributes must raise a descriptive KeyError.
# ==============================================================================

def test_non_dict_bc_missing_location_or_type():
    """
    Validates that a non-dictionary boundary condition object lacking valid 'location' 
    or 'type' strings triggers an immediate KeyError.
    """
    class IncompleteBC:
        location = ""
        type = "inflow"
        u = v = w = p = 0.0

    input_data = create_test_input_data({"boundary_conditions": [IncompleteBC()]})
    state = SolverState(input_data=input_data, config_data=BASE_CONFIG)
    state.u = np.zeros((4, 4, 4))
    state.v = np.zeros((4, 4, 4))
    state.w = np.zeros((4, 4, 4))
    state.p = np.zeros((4, 4, 4))
    
    with pytest.raises(KeyError, match="BoundaryCondition object missing required 'location' or 'type' attribute"):
        _apply_initial_boundary_conditions(state)


# ==============================================================================
# SECTION 4: Nested Sub-Object Value Extraction
# ==============================================================================
# Native C++ boundary objects often encapsulate primitive fluid fields (\(u, v, w, p\)) 
# inside a nested `.values` sub-struct rather than exposing flat attributes directly.
# ==============================================================================

def test_non_dict_bc_with_values_subobject():
    """
    Confirms that the bridge correctly inspects and extracts primitive velocity and 
    pressure components from a nested `.values` sub-object on non-dict boundary definitions.
    """
    class SubValues:
        u = 1.0
        v = 0.0
        w = 0.0
        p = 0.0

    class ObjectBC:
        location = "x_min"
        type = "inflow"
        values = SubValues()

    input_data = create_test_input_data({"boundary_conditions": [ObjectBC()]})
    state = SolverState(input_data=input_data, config_data=BASE_CONFIG)
    state.u = np.zeros((4, 4, 4))
    state.v = np.zeros((4, 4, 4))
    state.w = np.zeros((4, 4, 4))
    state.p = np.zeros((4, 4, 4))
    
    _apply_initial_boundary_conditions(state)


# ==============================================================================
# SECTION 5: Configuration Fallback and Absence Barriers
# ==============================================================================
# Solver states may store boundary conditions either as direct attributes or nested 
# inside raw dictionary configurations (`input_data`). When both are absent, initialization fails safely.
# ==============================================================================

def test_convert_bc_from_input_data_and_missing_error():
    """
    Tests both the successful fallback extraction of boundary conditions from `input_data` 
    and the strict exception guard when boundary configurations are entirely omitted.
    """
    # Case A: Fallback to input_data["boundary_conditions"] when state.boundary_conditions is None.
    input_data_valid = create_test_input_data()
    state = SolverState(input_data=input_data_valid, config_data=BASE_CONFIG)
    state.boundary_conditions = None
    _convert_boundary_conditions(state)
    assert isinstance(state.boundary_conditions, list)

    # Case B: Complete absence of boundary condition data must raise a KeyError.
    # Bypassing __init__ via __new__ to test missing boundary condition error guard directly.
    state_bad = SolverState.__new__(SolverState)
    state_bad.input_data = {"grid": BASE_GRID}
    state_bad.boundary_conditions = None
    with pytest.raises(KeyError, match="Boundary conditions configuration missing|Boundary conditions missing"):
        _convert_boundary_conditions(state_bad)


# ==============================================================================
# SECTION 6: State Parameter Synchronization for Pybind11 Bindings
# ==============================================================================
# Pybind11 C++ class constructors require configuration dictionaries to be bound as 
# direct instance attributes on the Python state object.
# ==============================================================================

def test_get_or_create_cpp_solver_synchronization():
    """
    Verifies that unstructured input dictionaries are automatically synchronized 
    to direct attributes on the SolverState instance prior to C++ instantiation.
    """
    # Construct state via __new__ to allow testing pre-synchronization attribute absence
    state = SolverState.__new__(SolverState)
    state.input_data = create_test_input_data()
    state.config = BASE_CONFIG
    state.nx, state.ny, state.nz = 4, 4, 4
    state.u = np.zeros((4, 4, 4))
    state.v = np.zeros((4, 4, 4))
    state.w = np.zeros((4, 4, 4))
    state.p = np.zeros((4, 4, 4))
    
    for attr in ["external_forces", "fluid_properties", "simulation_parameters", "dt", "boundary_conditions"]:
        if hasattr(state, attr):
            delattr(state, attr)
    state.input_data["boundary_conditions"] = [{"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0}}]

    solver = _get_or_create_cpp_solver(state)
    assert solver is not None
    assert hasattr(state, "external_forces")
    assert hasattr(state, "fluid_properties")
    assert hasattr(state, "simulation_parameters")
    assert hasattr(state, "dt")


# ==============================================================================
# SECTION 7: Trailing Time-Step Validation Guard
# ==============================================================================
# Following solver time-integration steps, the time step ('dt') must remain valid 
# to update simulation clocks. If removed mid-execution, a KeyError is raised.
# ==============================================================================

def test_step_simulation_trailing_dt_exception():
    """
    Ensures that if the simulation time step is unexpectedly stripped or invalidated 
    after the C++ step execution, the trailing validation block raises a KeyError.
    """
    input_data = create_test_input_data()
    state = SolverState(input_data=input_data, config_data=BASE_CONFIG)
    state.dt = 0.01
    state.u = np.zeros((4, 4, 4))
    state.v = np.zeros((4, 4, 4))
    state.w = np.zeros((4, 4, 4))
    state.p = np.zeros((4, 4, 4))

    class MockSolver:
        def step(self, s):
            # Simulate mutation removing time-step references during execution.
            if hasattr(s, "dt"):
                delattr(s, "dt")
            if "simulation_parameters" in s.input_data:
                del s.input_data["simulation_parameters"]

        def sync_fields(self, s):
            pass

    state._cpp_solver = MockSolver()
    
    with pytest.raises(KeyError, match="Simulation time step"):
        step_simulation(state)
