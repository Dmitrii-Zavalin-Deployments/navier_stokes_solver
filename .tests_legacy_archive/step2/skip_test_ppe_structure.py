# tests/step2/test_ppe_structure.py

import numpy as np

from src.step2.create_fluid_mask import create_fluid_mask
from src.step2.prepare_ppe_structure import prepare_ppe_structure
from tests.helpers.solver_step1_output_dummy import (
    make_step1_output_dummy as make_step1_dummy_state,
)


def make_state(nx=4, ny=4, nz=4, dx=1.0, dt=0.1, rho=1.0):
    state = make_step1_dummy_state(nx=nx, ny=ny, nz=nz)

    # Use dictionary access for consistent schema
    state.grid['dx'] = dx
    state.grid['dy'] = dx
    state.grid['dz'] = dx
    state.config['dt'] = dt
    state.constants["rho"] = rho

    state.mask = np.ones((nx, ny, nz), dtype=int)
    create_fluid_mask(state)

    state.fields["P"] = np.zeros((nx, ny, nz))
    state.fields["U"] = np.zeros((nx + 1, ny, nz))
    state.fields["V"] = np.zeros((nx, ny + 1, nz))
    state.fields["W"] = np.zeros((nx, ny, nz + 1))
    state.boundary_conditions = {}

    return state

def test_ppe_singular_enclosed_box():
    state = make_state()
    prepare_ppe_structure(state)
    assert state.ppe["ppe_is_singular"] is True

def test_ppe_non_singular_with_outlet():
    state = make_state()
    state.boundary_conditions = {
        "pressure_outlet": [{"location": "x_max", "value": 0.0}]
    }
    prepare_ppe_structure(state)
    assert state.ppe["ppe_is_singular"] is False

def test_ppe_rhs_builder_correct_formula():
    nx, ny, nz = 4, 4, 4
    state = make_state(nx, ny, nz, dx=1.0, dt=0.1, rho=2.0)
    divergence = np.ones((nx, ny, nz))

    prepare_ppe_structure(state)
    rhs_builder = state.ppe["rhs_builder"]
    rhs = rhs_builder(divergence)

    # -rho/dt * divergence
    expected = -state.constants["rho"] / state.config['dt'] * divergence
    assert np.allclose(rhs, expected)

def test_ppe_rhs_zero_in_solid_cells():
    state = make_state()
    # Mask a cell as solid
    state.is_fluid[1, 1, 1] = False
    divergence = np.ones_like(state.fields["P"])

    prepare_ppe_structure(state)
    rhs_builder = state.ppe["rhs_builder"]
    rhs = rhs_builder(divergence)

    assert rhs[1, 1, 1] == 0.0