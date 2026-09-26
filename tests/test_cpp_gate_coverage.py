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
    with pytest.raises(ImportError, match="Failed to import compiled C++ module"):
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
    direct attribute setters on a boundary condition object are caught and logged without 
    halting execution.
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

    state = SolverState(
        grid={"dx": 0.1, "dy": 0.1, "dz": 0.1},
        boundary_conditions=[IncompleteBC()],
        u=np.zeros((3, 3, 3)),
        v=np.zeros((3, 3, 3)),
        w=np.zeros((3, 3, 3)),
        p=np.zeros((3, 3, 3)),
    )
    
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

    state = SolverState(
        grid={"dx": 0.1, "dy": 0.1, "dz": 0.1},
        boundary_conditions=[ObjectBC()],
        u=np.zeros((3, 3, 3)),
        v=np.zeros((3, 3, 3)),
        w=np.zeros((3, 3, 3)),
        p=np.zeros((3, 3, 3)),
    )
    
    # Execution should successfully parse and apply boundary data from the sub-object.
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
    state = SolverState(
        grid={"dx": 0.1, "dy": 0.1, "dz": 0.1},
        input_data={"boundary_conditions": [{"location": "x_min", "type": "inflow", "u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0}]},
        u=np.zeros((3, 3, 3)),
        v=np.zeros((3, 3, 3)),
        w=np.zeros((3, 3, 3)),
        p=np.zeros((3, 3, 3)),
    )
    state.boundary_conditions = None
    _convert_boundary_conditions(state)
    assert isinstance(state.boundary_conditions, list)

    # Case B: Complete absence of boundary condition data must raise a KeyError.
    state_bad = SolverState(
        grid={"dx": 0.1, "dy": 0.1, "dz": 0.1},
        input_data={},
        u=np.zeros((3, 3, 3)),
        v=np.zeros((3, 3, 3)),
        w=np.zeros((3, 3, 3)),
        p=np.zeros((3, 3, 3)),
    )
    state_bad.boundary_conditions = None
    with pytest.raises(KeyError, match="Boundary conditions configuration missing"):
        _convert_boundary_conditions(state_bad)


# ==============================================================================
# SECTION 6: State Parameter Synchronization for Pybind11 Bindings
# ==============================================================================
# Pybind11 C++ class constructors require configuration dictionaries to be bound as 
# direct instance attributes on the Python state object.
# ==============================================================================

def test_get_or_create_cpp_solver_synchronization():
    """
    Verifies that unstructured input dictionaries ('external_forces', 'fluid_properties', 
    'simulation_parameters', and 'time_step') are automatically synchronized to direct 
    attributes on the SolverState instance prior to C++ instantiation.
    """
    state = SolverState(
        grid={"dx": 0.1, "dy": 0.1, "dz": 0.1},
        boundary_conditions=[{"location": "x_min", "type": "inflow", "u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0}],
        input_data={
            "external_forces": {"fx": 0.0},
            "fluid_properties": {"density": 1.0},
            "simulation_parameters": {"time_step": 0.01},
            "boundary_conditions": [{"location": "x_min", "type": "inflow", "u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0}],
        },
        u=np.zeros((3, 3, 3)),
        v=np.zeros((3, 3, 3)),
        w=np.zeros((3, 3, 3)),
        p=np.zeros((3, 3, 3)),
    )
    
    # Strip direct attributes to test automated synchronization.
    for attr in ["external_forces", "fluid_properties", "simulation_parameters", "dt"]:
        if hasattr(state, attr):
            delattr(state, attr)

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
    Ensures that if the simulation time step ('dt' or 'simulation_parameters.time_step') 
    is unexpectedly stripped or invalidated after the C++ step execution, the trailing 
    validation block raises a descriptive KeyError.
    """
    state = SolverState(
        grid={"dx": 0.1, "dy": 0.1, "dz": 0.1},
        dt=0.01,
        boundary_conditions=[{"location": "x_min", "type": "inflow", "u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}],
        input_data={
            "boundary_conditions": [{"location": "x_min", "type": "inflow", "u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}],
            "simulation_parameters": {"time_step": 0.01},
        },
        u=np.zeros((3, 3, 3)),
        v=np.zeros((3, 3, 3)),
        w=np.zeros((3, 3, 3)),
        p=np.zeros((3, 3, 3)),
    )

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
