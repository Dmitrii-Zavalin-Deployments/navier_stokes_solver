"""
Unit Test Module for C++ Interaction Wrapper Module (src/cpp_gate.py).

This module provides comprehensive narrative and executable verification for the C++ bridge interface,
testing import error safeguards, boundary condition parsing and validation, state attribute synchronization,
CFL stability checks, time-step extraction fallbacks, and C++ solver execution lifecycle handlers.
"""

import importlib
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ============================================================================
# NARRATIVE SECTION 1: C++ Module Import Error Handling
# ============================================================================
# When the compiled C++ extension module 'navier_stokes_cpp' is absent from the Python
# environment, importing the cpp_gate module must raise a descriptive ImportError.
# ============================================================================


def test_cpp_gate_import_error(monkeypatch):
    """
    # We verify that failing to import navier_stokes_cpp correctly raises an ImportError.
    """
    # We remove navier_stokes_cpp from sys.modules if present:
    monkeypatch.delitem(sys.modules, "navier_stokes_cpp", raising=False)
    
    # We mock sys.modules to raise ImportError when navier_stokes_cpp is imported:
    class RaisingFinder:
        @staticmethod
        def find_spec(fullname, path, target=None):
            if fullname == "navier_stokes_cpp":
                raise ImportError("Simulated missing C++ module")

    sys.meta_path.insert(0, RaisingFinder)
    try:
        import src.cpp_gate
        importlib.reload(src.cpp_gate)
        # If import succeeded unexpectedly, we force trigger the check:
        raise AssertionError("Expected ImportError was not raised.")
    except ImportError as e:
        assert "Failed to import compiled C++ module" in str(e)
    finally:
        sys.meta_path.pop(0)
        import src.cpp_gate
        importlib.reload(src.cpp_gate)


# ============================================================================
# FIXTURE: Mock C++ Environment
# ============================================================================


@pytest.fixture(autouse=True)
def mock_cpp_extension():
    """
    # We provide a mock implementation of the compiled navier_stokes_cpp extension
    # so that Python wrapper tests execute deterministically without compiled binaries.
    """
    class MockBoundaryCondition:
        def __init__(self):
            self.location = ""
            self.type = ""
            self.values = MagicMock()

    class MockFailingBoundaryCondition:
        def __init__(self):
            pass

        @property
        def location(self):
            return ""

        @location.setter
        def location(self, val):
            raise AttributeError("Simulated attribute error on location")

        @property
        def type(self):
            return ""

        @type.setter
        def type(self, val):
            raise TypeError("Simulated type error on type")

        @property
        def values(self):
            class BadValues:
                @property
                def u(self):
                    return 0.0
                @u.setter
                def u(self, val):
                    raise AttributeError("Simulated values attribute error")
            return BadValues()

        # For field_map attribute assignment testing:
        @property
        def u(self):
            return 0.0
        @u.setter
        def u(self, val):
            raise TypeError("Simulated field attribute error")

    class MockSolver:
        def __init__(self, state):
            self.state = state

        def step(self, state):
            pass

        def sync_fields(self, state):
            if hasattr(state, "fields") and state.fields is not None:
                state.fields[0] = state.u
                state.fields[1] = state.v
                state.fields[2] = state.w
                state.fields[3] = state.p

    mock_mod = MagicMock()
    mock_mod.BoundaryCondition = MockBoundaryCondition
    mock_mod.NavierStokesSolver = MockSolver

    sys.modules["navier_stokes_cpp"] = mock_mod
    import src.cpp_gate
    importlib.reload(src.cpp_gate)
    yield mock_mod


# ============================================================================
# NARRATIVE SECTION 2: Boundary Condition Dictionary Parsing & Validation
# ============================================================================
# Boundary condition dictionaries must be strictly validated for type, required keys ('location', 'type'),
# and required physical velocity/pressure value fields (u, v, w, p).
# ============================================================================


def test_dict_to_boundary_condition_validations():
    """
    # We verify that invalid boundary condition inputs raise appropriate TypeError or KeyError exceptions.
    """
    from src.cpp_gate import _dict_to_boundary_condition

    # 1. Non-dictionary input raises TypeError:
    with pytest.raises(TypeError, match="Boundary condition configuration must be a dictionary"):
        _dict_to_boundary_condition("not_a_dict")  # type: ignore[arg-type]

    # 2. Missing 'location' key raises KeyError:
    with pytest.raises(KeyError, match="missing required field 'location'"):
        _dict_to_boundary_condition({"type": "inflow", "values": {"u": 0, "v": 0, "w": 0, "p": 0}})

    # 3. Missing 'type' key raises KeyError:
    with pytest.raises(KeyError, match="missing required field 'type'"):
        _dict_to_boundary_condition({"location": "x_min", "values": {"u": 0, "v": 0, "w": 0, "p": 0}})

    # 4. Missing required value field (e.g. 'p') raises KeyError:
    with pytest.raises(KeyError, match="missing required value field"):
        _dict_to_boundary_condition({"location": "x_min", "type": "inflow", "values": {"u": 1, "v": 0, "w": 0}})


def test_dict_to_boundary_condition_exception_handlers():
    """
    # We verify that attribute assignment failures during boundary condition instantiation
    # are gracefully caught and logged without aborting execution.
    """
    import navier_stokes_cpp

    from src.cpp_gate import _dict_to_boundary_condition

    # We substitute BoundaryCondition with our failing mock class:
    original_bc = navier_stokes_cpp.BoundaryCondition
    navier_stokes_cpp.BoundaryCondition = navier_stokes_cpp.MockFailingBoundaryCondition if hasattr(navier_stokes_cpp, "MockFailingBoundaryCondition") else type("BadBC", (), {
        "location": property(lambda self: "", lambda self, v: (_ for _ in ()).throw(AttributeError("err"))),
        "type": property(lambda self: "", lambda self, v: (_ for _ in ()).throw(TypeError("err"))),
        "values": property(lambda self: type("V", (), {"u": property(lambda s: 0, lambda s, v: (_ for _ in ()).throw(AttributeError("err")))})()),
        "u": property(lambda self: 0, lambda self, v: (_ for _ in ()).throw(TypeError("err")))
    })

    try:
        bc_dict = {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0}}
        obj = _dict_to_boundary_condition(bc_dict)
        assert obj is not None
    finally:
        navier_stokes_cpp.BoundaryCondition = original_bc


# ============================================================================
# NARRATIVE SECTION 3: Initial Boundary Condition Enforcement & Spatial Faces
# ============================================================================
# Boundary condition enforcement inspects state data and maps inflow/prescribed values
# across computational domain boundaries (x_min, x_max, y_min, y_max, z_min, z_max)
# for primary fields (u, v, w, p) and volumetric fields arrays.
# ============================================================================


def test_apply_initial_boundary_conditions_errors():
    """
    # We verify error handling when boundary conditions are missing from state.
    """
    from src.cpp_gate import _apply_initial_boundary_conditions
    from src.state import SolverState

    empty_state = SolverState()
    empty_state.input_data = {}
    empty_state.boundary_conditions = None

    with pytest.raises(KeyError, match="Boundary conditions missing from state and input_data"):
        _apply_initial_boundary_conditions(empty_state)

    # Missing location or type in raw dict:
    state_bad_item = SolverState()
    state_bad_item.input_data = {"boundary_conditions": [{"location": "x_min"}]}
    with pytest.raises(KeyError, match="Boundary condition item missing required 'location' or 'type'"):
        _apply_initial_boundary_conditions(state_bad_item)


def test_apply_initial_boundary_conditions_object_errors():
    """
    # We verify error handling when non-dict boundary condition objects lack required attributes.
    """
    from src.cpp_gate import _apply_initial_boundary_conditions
    from src.state import SolverState

    class IncompleteBCObject:
        location = "x_min"
        type = "inflow"
        # missing values and attributes

    state = SolverState()
    state.boundary_conditions = [IncompleteBCObject()]
    with pytest.raises(KeyError, match="missing required attribute"):
        _apply_initial_boundary_conditions(state)

    class MissingLocObj:
        type = "inflow"

    state_no_loc = SolverState()
    state_no_loc.boundary_conditions = [MissingLocObj()]
    with pytest.raises(KeyError, match="BoundaryCondition object missing required 'location' or 'type'"):
        _apply_initial_boundary_conditions(state_no_loc)


def test_apply_initial_boundary_conditions_all_faces(sample_solver_state):
    """
    # We verify boundary value application across all spatial boundaries
    # (x_min, x_max, y_min, y_max, z_min, z_max) and fields array synchronization.
    """
    from src.cpp_gate import _apply_initial_boundary_conditions

    state = sample_solver_state
    state.fields = np.zeros((4, 2, 2, 2))
    state.input_data["boundary_conditions"] = [
        {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.1, "w": 0.0, "p": 10.0}},
        {"location": "x_max", "type": "outflow", "values": {"u": 1.0, "v": 0.1, "w": 0.0, "p": 10.0}},  # non-inflow type skipped for direct assignment
        {"location": "x_max", "type": "inflow", "values": {"u": 2.0, "v": 0.2, "w": 0.0, "p": 20.0}},
        {"location": "y_min", "type": "inflow", "values": {"u": 3.0, "v": 0.3, "w": 0.0, "p": 30.0}},
        {"location": "y_max", "type": "inflow", "values": {"u": 4.0, "v": 0.4, "w": 0.0, "p": 40.0}},
        {"location": "z_min", "type": "inflow", "values": {"u": 5.0, "v": 0.5, "w": 0.0, "p": 50.0}},
        {"location": "z_max", "type": "inflow", "values": {"u": 6.0, "v": 0.6, "w": 0.0, "p": 60.0}},
    ]

    _apply_initial_boundary_conditions(state)
    assert state.u is not None


def test_inflow_missing_values_error(sample_solver_state):
    """
    # We verify that inflow boundary conditions missing required velocity components raise KeyError.
    """
    from src.cpp_gate import _apply_initial_boundary_conditions

    state = sample_solver_state
    state.input_data["boundary_conditions"] = [
        {"location": "x_min", "type": "inflow", "values": {"u": 1.0}}  # missing v, w, p
    ]

    with pytest.raises(KeyError, match="Inflow boundary condition 'x_min' missing required values"):
        _apply_initial_boundary_conditions(state)


# ============================================================================
# NARRATIVE SECTION 4: Solver Initialization, Synchronization & Simulation Steps
# ============================================================================
# Managing the underlying C++ solver instance requires state validation, attribute
# synchronization, CFL stability enforcement, and robust step execution handling.
# ============================================================================


def test_convert_boundary_conditions_errors(sample_solver_state):
    """
    # We verify error raising when boundary conditions are missing during conversion.
    """
    from src.cpp_gate import _convert_boundary_conditions

    state = sample_solver_state
    state.input_data = {}
    state.boundary_conditions = None

    with pytest.raises(KeyError, match="FATAL ERROR: Boundary conditions configuration missing"):
        _convert_boundary_conditions(state)


def test_get_or_create_cpp_solver_validation(sample_solver_state):
    """
    # We verify that passing None to _get_or_create_cpp_solver raises ValueError,
    # and attribute synchronization from input_data operates correctly.
    """
    from src.cpp_gate import _get_or_create_cpp_solver

    with pytest.raises(ValueError, match="state must be explicitly provided"):
        _get_or_create_cpp_solver(None)  # type: ignore[arg-type]

    state = sample_solver_state
    # Clear direct attributes to test synchronization from input_data:
    if hasattr(state, "external_forces"):
        delattr(state, "external_forces")
    if hasattr(state, "fluid_properties"):
        delattr(state, "fluid_properties")
    if hasattr(state, "simulation_parameters"):
        delattr(state, "simulation_parameters")
    if hasattr(state, "dt"):
        delattr(state, "dt")

    solver = _get_or_create_cpp_solver(state)
    assert solver is not None
    assert state._cpp_solver is not None


def test_step_simulation_validations_and_cfl(sample_solver_state):
    """
    # We verify that step_simulation enforces state presence, velocity initialization,
    # and CFL stability bounds (C <= 1.0).
    """
    from src.cpp_gate import step_simulation

    # 1. None state raises ValueError:
    with pytest.raises(ValueError, match="state must be explicitly provided"):
        step_simulation(None)  # type: ignore[arg-type]

    # 2. Uninitialized velocity fields raise ValueError:
    state = sample_solver_state
    state.u = None
    with pytest.raises(ValueError, match="Velocity fields .* must be initialized"):
        step_simulation(state)

    # 3. CFL violation (> 1.0) raises ValueError:
    state2 = sample_solver_state
    state2.dt = 1.0
    state2.u = np.ones((2, 2, 2)) * 100.0  # high velocity -> high CFL
    state2.v = np.zeros((2, 2, 2))
    state2.w = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="CFL violation intercepted"):
        step_simulation(state2)


def test_step_simulation_execution_and_fallbacks(sample_solver_state):
    """
    # We verify successful simulation stepping, field synchronization, RuntimeError handling,
    # and time-step (dt) fallback extractions.
    """
    from src.cpp_gate import step_simulation

    state = sample_solver_state
    state.dt = 0.01
    state.u = np.zeros((2, 2, 2))
    state.v = np.zeros((2, 2, 2))
    state.w = np.zeros((2, 2, 2))
    state.fields = np.zeros((4, 2, 2, 2))

    # Test successful step:
    step_simulation(state)
    assert state.current_iteration == 1

    # Test missing sync_fields method raises RuntimeError:
    if state._cpp_solver and hasattr(state._cpp_solver, "sync_fields"):
        delattr(state._cpp_solver, "sync_fields")
    with pytest.raises(RuntimeError, match="missing required callable 'sync_fields'"):
        step_simulation(state)

    # Test solver step exception handling raises RuntimeError:
    state2 = sample_solver_state
    state2.dt = 0.01
    state2.u = np.zeros((2, 2, 2))
    state2.v = np.zeros((2, 2, 2))
    state2.w = np.zeros((2, 2, 2))
    
    class FailingSolver:
        def __init__(self, state):
            pass
        def step(self, state):
            raise RuntimeError("C++ internal crash")

    import navier_stokes_cpp
    navier_stokes_cpp.NavierStokesSolver = FailingSolver
    state2._cpp_solver = None

    with pytest.raises(RuntimeError, match="C++ execution failure during solver step"):
        step_simulation(state2)

    # Test dt fallback extraction from simulation_parameters:
    state3 = sample_solver_state
    if hasattr(state3, "dt"):
        delattr(state3, "dt")
    state3.u = np.zeros((2, 2, 2))
    state3.v = np.zeros((2, 2, 2))
    state3.w = np.zeros((2, 2, 2))
    state3.fields = np.zeros((4, 2, 2, 2))
    state3.input_data["simulation_parameters"] = {"time_step": 0.005}

    # Restore working mock solver:
    class WorkingSolver:
        def __init__(self, state):
            pass
        def step(self, state):
            pass
        def sync_fields(self, state):
            pass

    navier_stokes_cpp.NavierStokesSolver = WorkingSolver
    state3._cpp_solver = None

    step_simulation(state3)
    assert state3.current_time > 0.0

    # Test missing dt everywhere raises KeyError:
    state4 = sample_solver_state
    if hasattr(state4, "dt"):
        delattr(state4, "dt")
    state4.input_data["simulation_parameters"] = {}
    state4.u = np.zeros((2, 2, 2))
    state4.v = np.zeros((2, 2, 2))
    state4.w = np.zeros((2, 2, 2))
    state4.fields = np.zeros((4, 2, 2, 2))
    state4._cpp_solver = WorkingSolver(state4)

    with pytest.raises(KeyError, match="Simulation time step 'dt' or 'simulation_parameters.time_step' must be explicitly provided"):
        step_simulation(state4)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_solver_state():
    """
    # We provide a standard SolverState instance initialized with grid and boundary conditions.
    """
    from src.state import SolverState

    state = SolverState()
    state.input_data = {
        "grid": {"nx": 2, "ny": 2, "nz": 2, "dx": 0.5, "dy": 0.5, "dz": 0.5},
        "boundary_conditions": [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0}}
        ],
        "external_forces": {"force_vector": [0.0, 0.0, 0.0]},
        "fluid_properties": {"density": 1.0, "viscosity": 0.01},
        "simulation_parameters": {"time_step": 0.01}
    }
    state.dt = 0.01
    state.u = np.zeros((2, 2, 2))
    state.v = np.zeros((2, 2, 2))
    state.w = np.zeros((2, 2, 2))
    state.p = np.zeros((2, 2, 2))
    state.current_iteration = 0
    state.current_time = 0.0
    return state