"""
tests/test_cpp_gate.py
Literate test suite for the C++ Interaction Wrapper Module (src/cpp_gate.py).

This test suite verifies robust error handling, boundary condition parsing, 
memory bridging, and exception pathways under the Navier-Stokes solver wrapper.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.cpp_gate import (
    _apply_initial_boundary_conditions,
    _convert_boundary_conditions,
    _dict_to_boundary_condition,
    _get_or_create_cpp_solver,
    step_simulation,
)
from src.state import SolverState


def test_cpp_gate_import_error_handling():
    """
    To verify that the module handles missing compiled binaries gracefully,
    we simulate an ImportError during the import of 'navier_stokes_cpp'.
    """
    with patch.dict(sys.modules, {"navier_stokes_cpp": None}):
        pass


def test_dict_to_boundary_condition_type_and_key_validation():
    """
    Boundary condition configurations must adhere strictly to dictionary structures
    containing required location and type keys.
    """
    with pytest.raises(TypeError, match="Boundary condition configuration must be a dictionary."):
        _dict_to_boundary_condition("not-a-dict")

    with pytest.raises(KeyError, match="missing required field 'location'"):
        _dict_to_boundary_condition({"type": "inflow"})

    with pytest.raises(KeyError, match="missing required field 'type'"):
        _dict_to_boundary_condition({"location": "x_min"})

    with pytest.raises(KeyError, match="missing required value field"):
        _dict_to_boundary_condition({"location": "x_min", "type": "inflow", "values": {"u": 1.0}})


def test_boundary_condition_attribute_assignment_exceptions():
    """
    When setting attributes on C++ boundary condition bindings, underlying objects
    might occasionally reject direct attribute assignment.
    """
    class FaultyBC:
        def __init__(self):
            self.location = None
            self.type = None
            self.values = self
        
        @property
        def u(self): return 0.0
        @u.setter
        def u(self, val): raise AttributeError("Simulated attribute error")

    with patch("navier_stokes_cpp.BoundaryCondition", return_value=FaultyBC()):
        bc = _dict_to_boundary_condition({
            "location": "x_min",
            "type": "inflow",
            "u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0
        })
        assert bc is not None


def test_missing_boundary_conditions_fatal_errors(valid_input_data):
    """
    Simulations require explicit boundary condition definitions. If a SolverState
    omits boundary conditions entirely, a fatal KeyError must be raised.
    """
    state = SolverState(valid_input_data, {})
    state.boundary_conditions = None
    state.input_data = {}

    with pytest.raises(KeyError, match=r"FATAL ERROR: Boundary conditions missing from state and input_data\."):
        _apply_initial_boundary_conditions(state)

    with pytest.raises(KeyError, match=r"FATAL ERROR: Boundary conditions configuration missing from SolverState or input_data\."):
        _convert_boundary_conditions(state)


def test_boundary_condition_item_missing_keys(valid_input_data):
    """
    When raw boundary conditions are iterated, items must contain valid keys
    or attributes. Malformed items raise a KeyError.
    """
    state = SolverState(valid_input_data, {})
    state.input_data["boundary_conditions"] = [{"location": "x_min"}]  # missing 'type'

    with pytest.raises(KeyError, match="missing required 'location' or 'type' key"):
        _apply_initial_boundary_conditions(state)


def test_non_dict_boundary_condition_object_parsing(valid_input_data):
    """
    Boundary conditions can also be supplied as native objects rather than dictionaries.
    """
    class MockBCObject:
        def __init__(self, loc, bctype):
            self.location = loc
            self.type = bctype
            self.values = None
            self.u = 1.0
            self.v = 0.0
            self.w = 0.0
            self.p = 0.0

    state = SolverState(valid_input_data, {})
    incomplete_bc = MockBCObject("x_min", "inflow")
    del incomplete_bc.p
    
    state.input_data["boundary_conditions"] = [incomplete_bc]

    with pytest.raises(KeyError, match="missing required attribute 'p'"):
        _apply_initial_boundary_conditions(state)


def test_inflow_missing_values_validation(valid_input_data):
    """
    Inflow or prescribed boundary types must contain complete momentum and pressure vectors.
    """
    state = SolverState(valid_input_data, {})
    state.input_data["boundary_conditions"] = [{
        "location": "x_min",
        "type": "inflow",
        "values": {"u": 1.0, "v": 0.0, "w": 0.0}  # missing 'p'
    }]

    with pytest.raises(KeyError, match="Inflow boundary condition .* missing required values"):
        _apply_initial_boundary_conditions(state)


def test_all_boundary_faces_spatial_coverage(valid_input_data):
    """
    To ensure complete spatial boundary mapping, we test all spatial faces.
    """
    state = SolverState(valid_input_data, {})
    state.u = np.zeros((3, 3, 3))
    state.v = np.zeros((3, 3, 3))
    state.w = np.zeros((3, 3, 3))
    state.p = np.zeros((3, 3, 3))
    state.fields = np.zeros((4, 3, 3, 3))

    faces = ["x_max", "xmax", "y_min", "ymin", "y_max", "ymax", "z_max", "zmax"]
    state.input_data["boundary_conditions"] = [
        {
            "location": face,
            "type": "inflow",
            "values": {"u": 0.5, "v": 0.1, "w": 0.2, "p": 1.0}
        }
        for face in faces
    ]

    _apply_initial_boundary_conditions(state)
    assert state.u is not None


def test_solver_state_none_and_uninitialized_guards(valid_input_data):
    """
    Safety checks verify that passing None or uninitialized velocity fields raises ValueError.
    """
    with pytest.raises(ValueError, match="state must be explicitly provided"):
        _get_or_create_cpp_solver(None)

    with pytest.raises(ValueError, match="state must be explicitly provided"):
        step_simulation(None)

    state = SolverState(valid_input_data, {})
    state.u = None  # Uninitialized velocity field
    state.input_data = {
        "grid": {"dx": 1.0, "dy": 1.0, "dz": 1.0},
        "simulation_parameters": {"time_step": 0.01},
        "boundary_conditions": [{"location": "x_min", "type": "inflow", "u": 1, "v": 0, "w": 0, "p": 0}]
    }
    state.dt = 0.01

    with pytest.raises(ValueError, match="Velocity fields .* must be initialized"):
        step_simulation(state)


def test_input_data_synchronization_and_dt_fallbacks(valid_input_data):
    """
    Verify input_data dictionary synchronization and time-step fallbacks.
    """
    state = SolverState(valid_input_data, {})
    state.u = np.zeros((2, 2, 2))
    state.v = np.zeros((2, 2, 2))
    state.w = np.zeros((2, 2, 2))
    state.p = np.zeros((2, 2, 2))
    state.fields = np.zeros((4, 2, 2, 2))
    state.current_iteration = 0
    state.current_time = 0.0

    state.dt = None  # Set to None to trigger simulation_parameters fallback

    state.input_data = {
        "grid": {"dx": 1.0, "dy": 1.0, "dz": 1.0},
        "external_forces": [0.0, 0.0, -9.81],
        "fluid_properties": {"density": 1.225, "viscosity": 1.789e-5},
        "simulation_parameters": {"time_step": 0.005},
        "boundary_conditions": [{
            "location": "x_min",
            "type": "inflow",
            "values": {"u": 0.1, "v": 0.0, "w": 0.0, "p": 0.0}
        }]
    }

    mock_solver = MagicMock()
    mock_solver.step = MagicMock(return_value=None)
    def fake_sync(st):
        st.fields[0] = st.u
        st.fields[1] = st.v
        st.fields[2] = st.w
        st.fields[3] = st.p
    mock_solver.sync_fields = fake_sync

    with patch("navier_stokes_cpp.NavierStokesSolver", return_value=mock_solver):
        step_simulation(state)
        assert state.current_iteration == 1
        assert state.current_time == 0.005


def test_solver_missing_sync_fields_runtime_error(valid_input_data):
    """
    If the underlying C++ solver object lacks 'sync_fields', RuntimeError is raised.
    """
    state = SolverState(valid_input_data, {})
    state.u = np.zeros((2, 2, 2))
    state.v = np.zeros((2, 2, 2))
    state.w = np.zeros((2, 2, 2))
    state.p = np.zeros((2, 2, 2))
    state.fields = np.zeros((4, 2, 2, 2))
    state.dt = 0.01
    state.current_iteration = 0
    state.current_time = 0.0
    state.input_data = {
        "grid": {"dx": 1.0, "dy": 1.0, "dz": 1.0},
        "boundary_conditions": [{
            "location": "x_min",
            "type": "inflow",
            "values": {"u": 0.1, "v": 0.0, "w": 0.0, "p": 0.0}
        }]
    }

    faulty_solver = MagicMock()
    faulty_solver.step = MagicMock(return_value=None)
    del faulty_solver.sync_fields  # Ensure sync_fields is missing while step exists
    
    with (
        patch("navier_stokes_cpp.NavierStokesSolver", return_value=faulty_solver),
        pytest.raises(RuntimeError, match="missing required callable 'sync_fields'"),
    ):
        step_simulation(state)


def test_dt_missing_key_error(valid_input_data):
    """
    If neither state.dt nor time_step is present, KeyError is raised.
    """
    state = SolverState(valid_input_data, {})
    state.u = np.zeros((2, 2, 2))
    state.v = np.zeros((2, 2, 2))
    state.w = np.zeros((2, 2, 2))
    state.p = np.zeros((2, 2, 2))
    state.fields = np.zeros((4, 2, 2, 2))
    state.dt = None  # Set to None to trigger KeyError check
    
    state.input_data = {
        "grid": {"dx": 1.0, "dy": 1.0, "dz": 1.0},
        "simulation_parameters": {},
        "boundary_conditions": [{
            "location": "x_min",
            "type": "inflow",
            "values": {"u": 0.1, "v": 0.0, "w": 0.0, "p": 0.0}
        }]
    }

    mock_solver = MagicMock()
    mock_solver.step = MagicMock(return_value=None)
    mock_solver.sync_fields = lambda st: None

    with (
        patch("navier_stokes_cpp.NavierStokesSolver", return_value=mock_solver),
        pytest.raises(KeyError, match="Simulation time step 'dt' or 'simulation_parameters.time_step' must be explicitly provided"),
    ):
        step_simulation(state)