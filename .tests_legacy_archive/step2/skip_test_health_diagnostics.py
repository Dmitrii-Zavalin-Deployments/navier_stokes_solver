# tests/step2/test_health_diagnostics.py

import numpy as np
import pytest

from src.step2.build_divergence_operator import build_divergence_operator
from src.step2.compute_initial_health import compute_initial_health
from src.step2.create_fluid_mask import create_fluid_mask
from tests.helpers.solver_step1_output_dummy import (
    make_step1_output_dummy as make_step1_dummy_state,
)


def make_state(nx=4, ny=4, nz=4, dx=1.0, dt=0.1):
    """
    Helper to create a SolverState via the dummy factory and inject custom grid/config.
    """
    state = make_step1_dummy_state(nx=nx, ny=ny, nz=nz)

    # Use dictionary access for grid and config to match SolverState schema
    state.grid['dx'] = dx
    state.grid['dy'] = dx
    state.grid['dz'] = dx
    state.config['dt'] = dt

    state.mask = np.ones((nx, ny, nz), dtype=int)
    create_fluid_mask(state)

    # Ensure fields are correctly sized
    state.fields["U"] = np.zeros((nx + 1, ny, nz))
    state.fields["V"] = np.zeros((nx, ny + 1, nz))
    state.fields["W"] = np.zeros((nx, ny, nz + 1))
    state.fields["P"] = np.zeros((nx, ny, nz))
    
    state.health = {}
    return state

# ------------------------------------------------------------
# 1. Zero velocity → divergence_norm=0, max_vel=0, CFL=0
# ------------------------------------------------------------
def test_health_zero_velocity():
    state = make_state()
    build_divergence_operator(state)
    
    # Calculate divergence manually for the health check
    U, V, W = state.fields["U"], state.fields["V"], state.fields["W"]
    vel_vec = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    divergence = state.operators["divergence"] @ vel_vec
    
    # Use the specific override parameter
    compute_initial_health(state, divergence_override=divergence)

    assert state.health["divergence_norm"] == 0.0
    assert state.health["max_velocity"] == 0.0
    assert state.health["cfl"] == 0.0

# ------------------------------------------------------------
# 2. Uniform velocity → CFL = dt*(|u|/dx + |v|/dy + |w|/dz)
# ------------------------------------------------------------
def test_health_uniform_velocity_cfl():
    dx, dt = 0.5, 0.2
    state = make_state(dx=dx, dt=dt)

    u0, v0, w0 = 2.0, -1.0, 0.5
    state.fields["U"].fill(u0)
    state.fields["V"].fill(v0)
    state.fields["W"].fill(w0)

    # Production-style call: build operator first, then call with no override
    build_divergence_operator(state)
    compute_initial_health(state)

    expected_cfl = dt * (abs(u0) / dx + abs(v0) / dx + abs(w0) / dx)
    assert state.health["cfl"] == pytest.approx(expected_cfl)

# ------------------------------------------------------------
# 3. Divergent field → divergence_norm matches L2 norm
# ------------------------------------------------------------
def test_health_divergent_field_l2_norm():
    state = make_state()
    nx, ny, nz = state.grid['nx'], state.grid['ny'], state.grid['nz']

    # Create a dummy divergence vector
    divergence = np.ones(nx * ny * nz)
    expected_norm = np.sqrt(np.sum(divergence**2))

    # Test the override path specifically
    compute_initial_health(state, divergence_override=divergence)

    assert state.health["divergence_norm"] == pytest.approx(expected_norm)