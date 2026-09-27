"""
@file test_python_gate.py
@brief Literate Test Suite for Python Pybind11 Bindings Bridge (python_gate.cpp)
"""

import numpy as np
import pytest

try:
    import navier_stokes_cpp
except ImportError:
    navier_stokes_cpp = None


class DummySolverState:
    """Mock sovereign container matching the attributes expected by python_gate.cpp."""
    def __init__(self, nx=8, ny=8, nz=8):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.x_min = 0.0
        self.x_max = 1.0
        self.y_min = 0.0
        self.y_max = 1.0
        self.z_min = 0.0
        self.z_max = 1.0
        self.dt = 0.001

        self.fields = np.zeros((4, nx, ny, nz), dtype=np.float64)
        self.fields[0, :, :, :] = 0.1

        self.mask = np.zeros((nx, ny, nz), dtype=np.int32)
        self.mask[0, :, :] = -1
        self.mask[-1, :, :] = -1

        self.fluid_properties = {
            "density": 1000.0,
            "viscosity": 0.001
        }
        self.config = {
            "max_poisson_iterations": 50,
            "poisson_tolerance": 1e-6
        }
        self.physical_constraints = {
            "min_velocity": -10.0,
            "max_velocity": 10.0,
            "min_pressure": -100.0,
            "max_pressure": 100.0
        }
        self.external_forces = {
            "gravity_vector": [0.0, -9.81, 0.0],
            "force_vector": [10.0, 0.0, 0.0],
            "fx": np.zeros((nx, ny, nz), dtype=np.float64),
            "fy": np.zeros((nx, ny, nz), dtype=np.float64),
            "fz": np.zeros((nx, ny, nz), dtype=np.float64)
        }
        
        bc_wall = navier_stokes_cpp.BoundaryCondition() if navier_stokes_cpp else None
        bc_outflow = navier_stokes_cpp.BoundaryCondition() if navier_stokes_cpp else None
        if bc_wall and bc_outflow:
            bc_wall.location = "x_min"
            bc_wall.type = "no-slip"
            bc_outflow.location = "x_max"
            bc_outflow.type = "outflow"
            bc_outflow.scalar_p = 0.0
            self.boundary_conditions = [bc_wall, bc_outflow]
        else:
            self.boundary_conditions = []


def test_module_initialization():
    assert navier_stokes_cpp is not None, "Extension module navier_stokes_cpp must be compiled and available."
    assert isinstance(navier_stokes_cpp.__doc__, str)
    assert len(navier_stokes_cpp.__doc__) > 0
    assert hasattr(navier_stokes_cpp, "NavierStokesSolver")
    assert hasattr(navier_stokes_cpp, "BoundaryCondition")


def test_docstring_introspection():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    init_doc = str(navier_stokes_cpp.NavierStokesSolver.__init__.__doc__)
    step_doc = str(navier_stokes_cpp.NavierStokesSolver.step.__doc__)

    assert "Initialize solver instance directly from sovereign SolverState container" in init_doc
    assert "Advance the Navier-Stokes system by one time-step using state container references" in step_doc


def test_invalid_state_error_handling():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    with pytest.raises((TypeError, ValueError)):
        navier_stokes_cpp.NavierStokesSolver(None)


def test_boundary_condition_property_access():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    bc = navier_stokes_cpp.BoundaryCondition()
    bc.location = "x_min"
    bc.type = "inflow"
    bc.scalar_p = 101325.0
    bc.u_val = 1.5
    bc.v_val = 0.0
    bc.w_val = -0.5

    assert bc.location == "x_min"
    assert bc.type == "inflow"
    assert bc.scalar_p == 101325.0
    assert bc.u_val == 1.5
    assert bc.v_val == 0.0
    assert bc.w_val == -0.5


def test_navier_stokes_solver_container_execution():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)

    solver = navier_stokes_cpp.NavierStokesSolver(state)
    solver.step(state)

    assert state.nx == 8
    assert state.ny == 8
    assert state.nz == 8
    assert state.fields.shape == (4, nx, ny, nz)
    assert np.all(np.isfinite(state.fields))


def test_step_none_state_error():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    solver = navier_stokes_cpp.NavierStokesSolver(state)

    with pytest.raises((TypeError, ValueError)):
        solver.step(None)


def test_invalid_gravity_vector_size():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    state.external_forces["gravity_vector"] = [0.0, -9.81]
    solver = navier_stokes_cpp.NavierStokesSolver(state)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        solver.step(state)


def test_invalid_force_vector_size():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    state.external_forces["force_vector"] = [10.0, 0.0]
    solver = navier_stokes_cpp.NavierStokesSolver(state)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        solver.step(state)


def test_non_finite_field_simulation_failure():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    state.mask[1:-1, 1:-1, 1:-1] = 1
    
    for bc in state.boundary_conditions:
        if bc.type == "no-slip":
            bc.w_val = float('nan')
    
    solver = navier_stokes_cpp.NavierStokesSolver(state)

    with pytest.raises(RuntimeError, match="Invalid non-finite velocity encountered in boundary condition input."):
        solver.step(state)


def test_sync_fields_none_error():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    solver = navier_stokes_cpp.NavierStokesSolver(state)

    with pytest.raises((TypeError, ValueError)):
        solver.sync_fields(None)


def test_sync_fields_execution():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    solver = navier_stokes_cpp.NavierStokesSolver(state)
    
    solver.step(state)
    solver.sync_fields(state)

    assert np.all(np.isfinite(state.fields))
